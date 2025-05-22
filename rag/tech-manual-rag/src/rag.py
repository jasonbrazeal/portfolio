from textwrap import dedent

from elasticsearch import Elasticsearch
from google import genai
from google.genai.types import GenerateContentConfig

from es import perform_semantic_search, perform_lexical_search
from goog import MODEL
from utils import retry


def generate_response_for_query(client_goog: genai.Client, query: str, retrieved_chunk_texts: list[str]) -> str:
    '''
    generate a response for the given query using the provided chunks as context
    '''
    prompt = dedent(f'''
        You have been tasked with helping us to answer the following query:

        Query: {query}

        You have access to the following chunks of text which are meant to provide context as you answer the query:

        <context>
        {'\n\n'.join(retrieved_chunk_texts)}
        </context>

        Please remain faithful to the provided context, and only deviate from it if you are 100% sure that you know the answer already.

        Answer the question now, and avoid providing preamble such as 'Here is the answer', etc.
    ''')

    def _generate_response(client: genai.Client, prompt: str) -> str:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0, # minimum creativity
            )
        )
        response_text: str | None = response.text
        return response_text or ''

    return retry(_generate_response, (client_goog, prompt))


def retrieve_advanced(client: Elasticsearch, query: str, k: int, contextual: bool, similarity_threshold: float | None):
    '''
    Execute hybrid search query using semantic similarity and lexical search (vector search + bm25)
    if contextual is True, use contextualized chunks
    '''
    index_name = 'chunks_contextualized' if contextual else 'chunks'
    results_semantic = perform_semantic_search(client, index_name, query, k, similarity_threshold)
    results_lexical = perform_lexical_search(client, index_name, query, k)

    # combine results using reciprocal rank fusion (rrf)
    semantic_hits = results_semantic['hits']['hits']
    lexical_hits = results_lexical['hits']['hits']

    # create a map of document id to weighted rrf score
    rrf_scores = {}
    for i, hit in enumerate(semantic_hits):
        doc_id = hit['_id']
        # give semantic search a weight of 0.8 and lexical search 0.2 like Anthropic
        rrf_scores[doc_id] = 0.8 * (1 / (i + 1))
        # rrf_scores[doc_id] += hit['_score']  # add original score for tiebreaking

    for i, hit in enumerate(lexical_hits):
        doc_id = hit['_id']
        # give semantic search a weight of 0.8 and lexical search 0.2 like Anthropic
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += 0.2 * (1 / (i + 1))
        else:
            rrf_scores[doc_id] = 0.2 * (1 / (i + 1))
            # rrf_scores[doc_id] += hit['_score']  # add original score for tiebreaking

    # sort by rrf score and take top k
    sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # reconstruct hits in the same format as retrieve_basic
    final_hits = []
    semantic_count = 0
    lexical_count = 0

    for doc_id, score in sorted_hits:
        # find the hit in either semantic or lexical results
        hit = None
        is_from_semantic = False
        is_from_lexical = False

        semantic_matches = [h for h in semantic_hits if h['_id'] == doc_id]
        if semantic_matches:
            hit = semantic_matches[0]
            is_from_semantic = True

        lexical_matches = [h for h in lexical_hits if h['_id'] == doc_id]
        if lexical_matches:
            hit = lexical_matches[0]
            is_from_lexical = True

        if hit is None:
            raise ValueError(f'hit is None for doc_id: {doc_id}')

        # update the score to the rrf score
        hit['_score'] = score
        hit['_source']['from_semantic'] = is_from_semantic
        hit['_source']['from_lexical'] = is_from_lexical

        # count hits
        if is_from_semantic and not is_from_lexical:
            semantic_count += 1
        elif is_from_lexical and not is_from_semantic:
            lexical_count += 1
        else:  # it's in both
            semantic_count += 0.5
            lexical_count += 0.5

        final_hits.append(hit)

    print(f'{semantic_count=}')
    print(f'{lexical_count=}')

    print(f'semantic scores for query: {query}')
    for hit in semantic_hits:
        print(hit['_score'])

    print(f'lexical scores for query: {query}')
    for hit in lexical_hits:
        print(hit['_score'])

    print(f'rrf scores (sorted) for query: {query}')
    for hit in final_hits:
        print(hit['_score'])

    return final_hits


def end_to_end_advanced(client_es: Elasticsearch, client_goog: genai.Client, query: str, k: int, contextual: bool, similarity_threshold: float | None) -> str:
    '''
    Generate end-to-end response for advanced search query (vector search + bm25)
    if contextual is True, use contextualized chunks
    '''
    retrieved_chunks = retrieve_advanced(client_es, query, k, contextual, similarity_threshold)
    retrieved_chunk_texts = [chunk['_source']['chunk_text'] for chunk in retrieved_chunks]
    return generate_response_for_query(client_goog, query, retrieved_chunk_texts)


def retrieve_basic(client: Elasticsearch, query: str, k: int, contextual: bool, similarity_threshold: float | None):
    '''
    Execute basic semantic similarity search query (vector search)
    if contextual is True, use contextualized chunks
    '''
    index_name = 'chunks_contextualized' if contextual else 'chunks'
    results = perform_semantic_search(client, index_name, query, k, similarity_threshold)
    # print(f'scores for query: {query}')
    # for hit in results['hits']['hits']:
    #     print(hit['_score'])
    print('*'*88)
    return results['hits']['hits']


def end_to_end_basic(client_es: Elasticsearch, client_goog: genai.Client, query: str, k: int, contextual: bool, similarity_threshold: float | None) -> str:
    '''
    Generate end-to-end response for basic search query (vector search)
    if contextual is True, use contextualized chunks
    '''
    retrieved_chunks = retrieve_basic(client_es, query, k, contextual, similarity_threshold)
    retrieved_chunk_texts = [chunk['_source']['chunk_text'] for chunk in retrieved_chunks]
    return generate_response_for_query(client_goog, query, retrieved_chunk_texts)
