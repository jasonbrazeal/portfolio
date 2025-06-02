# %% [markdown]
# # Exploratory Data Analyis
#
# The plan is to:
# * ingest pdfs, extract text, create jsonl dataset
# * view some basic stats on the dataset

# %%
import json
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from goog import get_text_from_pdf_gcvision
from text import get_text_from_pdf_pymupdf, chunk_text

# %%
def preprocess_pdfs(filepath: Path) -> Generator[dict, None, None]:
    """
    Preprocess all pdfs in a directory and return a dictionary of documents
    """
    for i, file in enumerate(sorted(filepath.iterdir())):
        if file.is_file() and file.suffix == '.pdf':
            print(f'preprocessing doc #{i + 1}')
            pages = get_text_from_pdf_gcvision(file)
            # pages = get_text_from_pdf_pymupdf(file)
            if not pages or sum(len(page) for page in pages) == 0:
                print(f'skipping doc #{i + 1} because it has no text')
                continue
            doc = {}
            doc['pages'] = [{'page_text': page} for page in pages]
            doc['num_pages'] = len(pages)
            doc['text'] = '\n'.join(pages)
            doc['filename'] = file.name
            doc['doc_id'] = i + 1
            chunk_texts = chunk_text(doc['text'])
            doc['chunks'] = [
                {'chunk_text': t, 'embedding': [], 'chunk_id': f'{doc["doc_id"]}-{j + 1}'} for j, t in enumerate(chunk_texts)
            ]
            doc['num_chunks'] = len(chunk_texts)
            doc['chunks_contextualized'] = []
            print(f'doc #{i + 1} preprocessing done ({len(chunk_texts)} chunks)')
            yield doc


def ingest_pdfs_to_dataset(src: Path, dst: Path) -> None:
    """
    Ingest all pdfs in a directory, preprocess them to extract text and chunks, save to jsonl file
    """
    if not src.is_dir():
        raise ValueError(f'Directory "{src}" does not exist.')
    pdf_files = [f for f in sorted(src.iterdir()) if f.is_file() and f.suffix == '.pdf']
    if not pdf_files:
        raise ValueError(f'No PDF files found in directory "{src}"')

    # preprocess pdfs and yield a json document for each
    # write data to jsonl file
    progress = tqdm(desc='preprocessing pdfs', unit='docs', total=len(pdf_files), colour='#ccc2ff')
    for doc in preprocess_pdfs(src):
        # opening + closing the file so we can check on the progress more easily
        with open(dst, 'a') as f:
            f.write(json.dumps(doc) + '\n')
        progress.update(1)
    print(f'done preprocessing {len(pdf_files)} pdfs')
    return

# %%
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data'
pdf_dir = data_dir / 'texas_instruments_manuals'
dataset_orig = data_dir / 'tech-manual-rag.jsonl'
dataset_contextualized = data_dir / 'tech-manual-rag.contextualized.jsonl'
dataset_embedded = data_dir / 'tech-manual-rag.contextualized.embedded.jsonl'
dataset_eval = data_dir / 'tech-manual-rag.eval.jsonl'

# %%
ingest_pdfs_to_dataset(pdf_dir, dataset_orig)

# %%
def get_dataset_stats(filepath: Path) -> dict:
    """
    get a few basic statistics from the dataset
    """
    total_docs = 0
    total_pages = 0
    total_chunks = 0

    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line)
            total_docs += 1
            total_pages += doc['num_pages']
            total_chunks += doc['num_chunks']
    stats = {
        'total_documents': total_docs,
        'total_pages': total_pages,
        'total_chunks': total_chunks
    }
    print(stats)
    return stats

get_dataset_stats(dataset_orig)

# %% [markdown]
#
# ```python
# {'total_documents': 40, 'total_pages': 9349, 'total_chunks': 8875}
# ```
