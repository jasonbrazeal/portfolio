# %% [markdown]
# # Data Preparation
#
# The plans is to:
# * contextualize chunks - add context that explains what each chunk is about and how it fits into the document
# * embed chunks - embed chunks with their context included
# * create evaluation dataset - using the contextualized chunks, use a LLM to create a gold set of query + answer pairs to evaluate retrieval. I review each of the generated pairs for quality and delete many of the weaker ones (collected in tech-manual-rag.eval.deleted.jsonl). There are quite a few low quality data pairs to weed out, asking things like what page something is on or giving an answer that is patently incorrect. But overall, this is an effective method, yielding about 180 query + answer pairs for our gold set (tech-manual-rag.eval.jsonl).

# %%
import json
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import dedent
from tqdm import tqdm

from goog import get_google_client, MODEL
from text import contextualize_chunks, embed_all_chunks
from utils import retry

# %%
def create_eval_dataset(filepath: Path):
    """
    * choose 6 random chunks per doc (41 docs, so, 246 total, aiming for at least 150-200 good ones.)
    * prompt LLM to generate 1 question/answer pair per chunk
    * human review, not all will be good queries/answers
    """
    filepath_dst = filepath.parent.parent / filepath.name.replace('.contextualized.embedded.jsonl', '.eval.jsonl')
    with open(filepath, 'r') as f:
        total_docs = sum(1 for _ in f)
    with open(filepath, 'r') as src, open(filepath_dst, 'w') as dst:
        for line in tqdm(src, desc='creating eval dataset', unit='docs', total=total_docs, colour='blue'):
            doc = json.loads(line)
            chunks_contextualized = doc['chunks_contextualized']
            # don't use short chunks for evaluation
            non_short_chunks = [c for c in chunks_contextualized if len(c['chunk_text']) > 1000]
            print(f'filtered out {len(chunks_contextualized) - len(non_short_chunks)} short chunks for doc {doc["doc_id"]}')
            rando_chunks = random.sample(non_short_chunks, 6)

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for chunk in rando_chunks:
                    futures.append(executor.submit(generate_chunk_eval, chunk, doc['doc_id']))

                for future in tqdm(as_completed(futures), total=len(rando_chunks), desc='processing chunks', colour='#b76d4b'):
                    query, answer, doc_id, chunk_id = future.result()
                    eval_doc = {
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'query': query,
                        'answer': answer,
                    }
                    dst.write(json.dumps(eval_doc) + '\n')


def generate_chunk_eval(chunk: dict, doc_id: str) -> tuple[str, str, str, str]:
    query_prompt = dedent('''
        Here is a chunk of text from a document in a search retrieval system:
        <chunk>
        {}
        </chunk>
        Please provide a query that can be answered using the text in this chunk.
        The query should be specific, not generic, and likely to have an answer located only in this chunk, not the rest of the document.
        Give the query and nothing else.
    ''')

    answer_prompt = dedent('''
        Here is a chunk of text from a document in a search retrieval system:
        <chunk>
        {}
        </chunk>
        Here is a query about that chunk:
        <query>
        {}
        </query>
        Please provide an answer to the query based only on the text in this chunk.
        Give the answer and nothing else.
    ''')

    def _generate_chunk_eval(chunk: dict, doc_id: str) -> tuple[str, str, str, str]:
        client = get_google_client()

        response_query = client.models.generate_content(
            model=MODEL,
            contents=query_prompt.format(chunk['chunk_text']),
        )
        query: str | None = response_query.text
        if not query:
            query = ''

        response_answer = client.models.generate_content(
            model=MODEL,
            contents=answer_prompt.format(chunk['chunk_text'], query),
        )
        answer: str | None = response_answer.text
        if not answer:
            answer = ''

        return query, answer, doc_id, chunk['chunk_id']

    return retry(_generate_chunk_eval, (chunk, doc_id))

# %%
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data'
pdf_dir = data_dir / 'texas_instruments_manuals'
dataset_orig = data_dir / 'tech-manual-rag.jsonl'
dataset_contextualized = data_dir / 'tech-manual-rag.contextualized.jsonl'
dataset_embedded = data_dir / 'tech-manual-rag.contextualized.embedded.jsonl'

# %%
contextualize_chunks(dataset_orig)
embed_all_chunks(dataset_contextualized)
create_eval_dataset(dataset_embedded)
