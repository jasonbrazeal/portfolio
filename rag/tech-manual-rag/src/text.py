import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched
from pathlib import Path
from textwrap import dedent
from typing import Literal, Sequence

import pymupdf
from chonkie import RecursiveChunker, RecursiveRules, TokenChunker
from google.genai.types import GenerateContentConfig
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from goog import MODEL, get_google_client
from utils import clean_whitespace, decode_unicode_escapes, retry


def get_text_from_pdf_pymupdf(filepath: Path) -> list[str]:
    """
    Extract text from pdf and return the pages as a list of strings: [page1: str, page2: str, ...]
    Note: this did not extract every page of every pdf, so we use Google Cloud Vision API instead
    """
    pages = []
    with pymupdf.open(filepath) as pdf:
        for page in pdf:
            text = page.get_text()
            if not text:
                print(f'skipping page {page.number} because it has no text')
                continue
            pages.append(decode_unicode_escapes(clean_whitespace(text)))
    return pages


def process_chunk(chunk_id: str, chunk_text: str, doc_text: str, prompts: tuple[str, str], cache_name: str | None) -> tuple[str, str]:
    """
    Process a chunk for contextualization; call to LLM API; retry with exponential backoff if it fails
    Return the chunk text with the chunk context prepended
    """

    def _process_chunk(chunk_id: str, chunk_text: str, doc_text: str, prompts: tuple[str, str], cache_name: str | None) -> tuple[str, str]:
        document_prompt, chunk_prompt = prompts
        client = get_google_client()

        if cache_name is not None:
            response = client.models.generate_content(
                model=MODEL,
                contents=chunk_prompt.format(chunk_text),
                config=GenerateContentConfig(
                    cached_content=cache_name,
                ),
            )
        else:
            response = client.models.generate_content(
                model=MODEL,
                contents='\n'.join(
                    (document_prompt.format(doc_text), chunk_prompt.format(chunk_text)),
                ),
            )
        if response.text:
            return f'{response.text}\n{chunk_text}', chunk_id
        else:
            print('problem contextualizing chunk:')
            print(f'{chunk_text=}')
            print(f'{response.text=}')
            return '', chunk_id

    return retry(_process_chunk, (chunk_id, chunk_text, doc_text, prompts, cache_name))


def contextualize_chunks(filepath: Path) -> None:
    """
    Contextualize chunks using an LLM, add to document json, save to new jsonl file
    """
    filepath_dst = filepath.with_suffix('.contextualized.jsonl')
    with open(filepath, 'r') as f:
        total_docs = sum(1 for _ in f)

    progress = tqdm(desc='contextualizing docs', unit='docs', total=total_docs, colour='#ccc2ff')
    with open(filepath, 'r') as src, open(filepath_dst  , 'w') as dst:
        for i, line in enumerate(src):

            doc = json.loads(line)
            text = doc['text']
            chunks = doc['chunks']

            document_prompt = dedent('''
                Here is the document content:
                <document>
                {}
                </document>
            ''')

            # disable caching for now
            cache_name = None
            # cache_name, total_tokens = cache_document_prompt(document_prompt.format(text), doc['doc_id'])
            # doc['total_tokens'] = total_tokens

            chunk_prompt = dedent('''
                Here is a chunk from that document:
                <chunk>
                {}
                </chunk>
                Please give a short (no more than 90 tokens), succinct context to situate this chunk
                within the overall document for the purposes of improving search retrieval of the chunk.
                Do not describe the document, e.g. do not begin your answer "This document is a guide for the TI-84 Plus CE Python graphing calculator...".
                Do describe the chunk, e.g. "This section describes how to work with statistics on the TI-84 Plus CE Python graphing calculator..."
                Always mention the name of the calculator in your answer.
                Answer only with the succinct context and nothing else.
            ''')

            print(f'contextualizing {doc["num_chunks"]} chunks for doc {doc["doc_id"]}')
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for chunk in chunks:
                    chunk_text = chunk['chunk_text']
                    chunk_id = chunk['chunk_id']
                    futures.append(executor.submit(process_chunk, chunk_id, chunk_text, text, (document_prompt, chunk_prompt), cache_name))

                for future in tqdm(as_completed(futures), total=len(chunks), desc='processing chunks', colour='#b76d4b'):
                    chunk_contextualized, chunk_id = future.result()
                    doc['chunks_contextualized'].append({
                        'chunk_id': chunk_id,
                        'chunk_text': chunk_contextualized,
                        'embedding': []
                    })
            doc['chunks_contextualized'] = sorted(doc['chunks_contextualized'], key=lambda c: (int(c['chunk_id'].split('-')[0]), int(c['chunk_id'].split('-')[1])))
            dst.write(json.dumps(doc) + '\n')
            progress.update(1)

    print('done contextualizing chunks')
    return


def embed_all_chunks(filepath: Path) -> None:
    """
    Read docs from jsonl file and embed all chunk_text
    under "chunks" and "chunks_contextualized"
    """
    filepath_dst = filepath.with_suffix('.embedded.jsonl')
    with open(filepath, 'r') as f:
        total_docs = sum(1 for _ in f)

    with open(filepath, 'r') as src, open(filepath_dst  , 'w') as dst:
        for line in tqdm(src, desc='embedding doc chunks', unit='docs', total=total_docs, colour='blue'):
            doc = json.loads(line)
            chunks = doc['chunks']
            chunks_contextualized = doc['chunks_contextualized']

            print(f'\nembedding {len(chunks)} chunks for doc {doc["doc_id"]}')

            chunk_embeddings = []
            batches = list(batched([c['chunk_text'] for c in chunks], 32))
            for batch in tqdm(batches, desc='embedding chunk batches', unit='batches', total=len(batches), colour='#b76d4b'):
                chunk_embeddings.extend(retry(embed, (batch,)))

            print(f'\nembedding {len(chunks_contextualized)} contextualized chunks for doc {doc["doc_id"]}')
            chunk_contextualized_embeddings = []
            batches = list(batched([c['chunk_text'] for c in chunks_contextualized], 32))
            for batch in tqdm(batches, desc='embedding contextualized chunk batches', unit='batches', total=len(batches), colour='#b76d4b'):
                chunk_contextualized_embeddings.extend(retry(embed, (batch,)))

            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                chunk['embedding'] = embedding
            doc['chunks'] = chunks

            for i, (chunk, embedding) in enumerate(zip(chunks_contextualized, chunk_contextualized_embeddings)):
                chunk['embedding'] = embedding
            doc['chunks_contextualized'] = chunks_contextualized

            dst.write(json.dumps(doc) + '\n')
    return


def chunk_text(text: str, chunker_type: Literal['recursive', 'token'] = 'recursive') -> list[str]:
    if chunker_type == 'recursive':
        chunker = RecursiveChunker(
            tokenizer_or_token_counter='gpt2',
            chunk_size=420, # in tokens, embeddings model can handle 512 (caveat: it has its own tokenizer, not gpt2)
            rules=RecursiveRules(), # note: the context will be added to the chunk text, so chunk_size can't be 512 exactly
            min_characters_per_chunk=64,
            return_type='chunks',
        )
    elif chunker_type == 'token':
        chunker = TokenChunker(
            tokenizer='gpt2',
            chunk_size=420,
            chunk_overlap=32,
        )
    else:
        raise ValueError(f'Chunker must be either "recursive" or "token": {chunker_type}')
    chunks = chunker(text)
    # for chunk in chunks:
    #     print(f'chunk text: {chunk.text}')
    #     print(f'token count: {chunk.token_count}')
    # print(f'total chunks: {len(chunks)}')
    # check return type here
    return [c.text for c in chunks]


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def embed(texts: Sequence[str]) -> list[list[float]]:
    """
    Embed texts using a transformers model from Hugging Face:
     * intfloat/multilingual-e5-large-instruct - top 5 on MTEB leaderboard
     * https://huggingface.co/spaces/mteb/leaderboard
     * see MTEB: Massive Text Embedding Benchmark paper: https://arxiv.org/abs/2210.07316
    """
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

    # each query must come with a one-sentence instruction that describes the task
    # choose one at random since we're only going to use the embeddings
    task = 'Retrieve semantically similar text.'
    # from https://github.com/microsoft/unilm/blob/master/e5/utils.py#L106

    # wrap utterance in prompt instruction per model's instructions
    texts = [get_detailed_instruct(task, text) for text in texts]
    # tokenize
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    # get model outputs
    outputs: BaseModelOutputWithPoolingAndCrossAttentions = model(**batch_dict)
    # pool embeddings, combine sequence of embeddings into a sentence embedding
    raw_embeddings: Tensor = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # normalize embeddings
    embeddings_normalized: Tensor = F.normalize(raw_embeddings, p=2, dim=1)
    return embeddings_normalized.tolist()
