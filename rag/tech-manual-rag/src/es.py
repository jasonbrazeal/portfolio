from pathlib import Path
import json

from elasticsearch import Elasticsearch
from tqdm import tqdm

from text import embed


def get_es_client(host='localhost', port=9200) -> Elasticsearch:
    """
    Get an Elasticsearch client
    """
    client = Elasticsearch(
        hosts=[
            f'http://{host}:{port}',
        ],
        request_timeout=60,
    )

    try:
        info = client.info()
        print('Cluster information:', info)
    except Exception as e:
        print(f'Error retrieving Elasticsearch cluster information: {e}')

    try:
        health = client.cluster.health()
        print('Cluster health:', health)
    except Exception as e:
        print(f'Error retrieving Elasticsearch cluster health: {e}')

    return client


def list_es_indices(client: Elasticsearch) -> list:
    """
    List all indices in the Elasticsearch cluster
    """
    try:
        indices = client.indices.get_alias(index='*')
        return list(indices.keys())
    except Exception as e:
        print(f'Error listing Elasticsearch indices: {e}')
        return []


def delete_es_index(client: Elasticsearch, index_name) -> None:
    """
    Delete an Elasticsearch index if present
    """
    client.indices.delete(
        index=index_name,
        ignore_unavailable=True,
    )


def create_es_index(client: Elasticsearch, index_name, mappings=None) -> None:
    """
    Create an Elasticsearch index if it doesn't already exist
    """
    if mappings is None:
        mappings = {}
    client.options(ignore_status=400).indices.create(
        index=index_name,
        mappings=mappings,
        settings = {
           'number_of_shards': 1,
           'number_of_replicas': 0,
        },
    )


def count_es_docs(client: Elasticsearch, index_name='*') -> int:
    """
    Get total number of documents in Elasticsearch index/indices
    """
    try:
        result = client.count(index=index_name)
        return result['count']
    except Exception as e:
        print(f'Error counting documents: {e}')
        return 0


def get_es_doc(client: Elasticsearch, index_name: str, doc_id: str) -> dict:
    """
    Get a single document from Elasticsearch by ID
    """
    try:
        result = client.get(index=index_name, id=doc_id)
        return result['_source']
    except Exception as e:
        print(f'Error getting document: {e}')
        return {}


def get_all_es_docs(client: Elasticsearch, index_name: str) -> list:
    """
    Get all documents from an Elasticsearch index
    """
    try:
        result = client.search(
            index=index_name,
            body={
                'query': {
                    'match_all': {}
                },
                'size': 10000  # default max
            }
        )
        return [hit['_source'] for hit in result['hits']['hits']]
    except Exception as e:
        print(f'Error getting documents: {e}')
        return []



def add_docs_to_es(client: Elasticsearch, filepath: Path) -> None:
    """
    Read docs from jsonl file and add them to Elasticsearch
    """
    mappings = {
        'properties': {
            'chunk_id': {'type': 'text'},
            'chunk_text': {
                'type': 'text',
                'similarity': {
                    'default': {
                        'type': 'BM25' # the default
                    }
                }
            },
            'embedding': {
                'type': 'dense_vector',
                'dims': 1024, # from intfloat/multilingual-e5-large-instruct
                'similarity': 'cosine', # the default
            },
        }
    }
    for index_name in ('chunks', 'chunks_contextualized'):
        delete_es_index(client, index_name)
        create_es_index(client, index_name, mappings)

    print('adding documents to elasticsearch...')
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc='processing docs', colour='#5dbeac'):
            doc = json.loads(line)
            chunks = doc['chunks']
            for chunk in chunks:
                client.index(
                    index='chunks',
                    document=chunk,
                )
            chunks_contextualized = doc['chunks_contextualized']
            for chunk in chunks_contextualized:
                client.index(
                    index='chunks_contextualized',
                    document=chunk,
                )
    return


def perform_semantic_search(client: Elasticsearch, index_name: str, text: str, k: int, similarity_threshold: float | None, num_candidates: int = 100):
    '''
    Embed text and search embeddings vectors for k nearest neighbors
    '''
    text_vector = embed([text])[0]

    search_args = {
        'index': index_name,
        'body': {
            'query': {
                'script_score': {
                    'query': {'match_all': {}},
                    'script': {
                        'source': 'cosineSimilarity(params.query_vector, \'embedding\') + 1.0',
                        'params': {'query_vector': text_vector}
                    }
                }
            },
            'size': k,
        },
    }
    if similarity_threshold is not None:
        search_args['body']['min_score'] = similarity_threshold + 1.0
    results = client.search(**search_args)

    print(f'{len(results['hits']['hits'])} matching documents found')
    # for doc in results['hits']['hits']:
    #     print(doc)
    return results


def perform_lexical_search(client: Elasticsearch, index_name: str, text: str, k: int):
    '''
    Search text using default BM25 keyword search
    '''
    query = {
        'match': {
            'chunk_text': {
                'query': text
            }
        }
    }
    results = client.search(index=index_name, query=query, size=k)
    print(f'{len(results['hits']['hits'])} matching documents found')
    # for doc in results['hits']['hits']:
    #     print(doc)

    return results


if __name__ == '__main__':

    # testing elasticsearch

    project_root = Path(__file__).parent.parent
    dataset = project_root / 'data/tech-manual-rag.contextualized.embedded.jsonl'

    client = get_es_client()

    # add_docs_to_es(client, dataset)
    # print(count_es_docs(client))

    # for doc in get_all_es_docs(client, 'chunks'):
    #     print(doc)

    # for doc in get_all_es_docs(client, 'chunks_contextualized'):
    #     print(doc)

    index_name = ''
    text = 'how do i enter in the x and y data for a parametric graph?'

    perform_semantic_search(client, index_name, text, k=10, similarity_threshold=0.5)
    perform_lexical_search(client, index_name, text, k=10)
