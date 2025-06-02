import logging
import re
import time
import unicodedata

from io import BytesIO
from os import getenv
from pathlib import Path
from typing import Any, Callable, List
from uuid import uuid4

import matplotlib.pyplot as plt
import tiktoken
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pypdf import PdfReader

from db import Chat, DATA_DIR

logging.basicConfig()
logging.getLogger().setLevel(getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME: str = 'text-embedding-3-small'
LLM_NAME: str = 'gpt-4.1-mini-2025-04-14'
LLM_CLIENT: OpenAI = OpenAI(api_key='')
LLM_TEMPERATURE: int = 0
LLM_MAX_TOKENS: int = 150 # max tokens to respond with
VECTOR_DB_PATH: Path = DATA_DIR / 'vectorstore'
VECTOR_DB_CLIENT: ClientAPI = PersistentClient(path=str(VECTOR_DB_PATH), settings=Settings(allow_reset=True, anonymized_telemetry=False))
MAX_TOKENS_PER_CHUNK: int = 256
TOKEN_OVERLAP_PER_CHUNK: int = 64
SYSTEM_PROMPT = 'You are a helpful AI bot that a user can chat with. You answer questions for the user based on your knowledge supplemented with any context given before the question. You may ask the user clarifying questions if needed to understand the question, or simply respond "I don\'t know" if you don\'t have an accurate answer. Do not mention that a context was provided, just try to use it to inform your responses.'
PROMPT = 'Context: {}\n\n---\n\nQuestion: {}\nAnswer:'


def process_text(text: str) -> None:
    # num_tokens: int = count_tokens_in_text(text)
    text_chunked: list[str] = chunk_text(text, chunk_size=MAX_TOKENS_PER_CHUNK, chunk_overlap=TOKEN_OVERLAP_PER_CHUNK)
    logger.debug(str(len(text_chunked)) + ' chunks in text')
    text_chunked_embedded: list[list[float]] = embed_text(text_chunked)
    save_to_vectorstore(text_chunked, text_chunked_embedded)


def process_text_bytes(file_bytes: BytesIO) -> None:
    """
    Process bytes from uploaded text file
    """
    text = file_bytes.getvalue().decode('utf-8')
    process_text(text)


def process_pdf_bytes(file_bytes: BytesIO) -> None:
    """
    Process bytes from uploaded pdf file
    """
    text = get_text_from_pdf(file_bytes)
    process_text(text)


def get_text_from_pdf(file_bytes: BytesIO) -> str:
    """
    Extract text from pdf and return it as one big chunk of text
    """
    pdf: PdfReader = PdfReader(file_bytes)
    pages = []
    for page in pdf.pages:
        pages.append(clean_whitespace(page.extract_text()))
    return '\n\n'.join(pages)


def count_tokens_in_text(text: str) -> int:
    """
    Count the tokens in the text
    """
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
    return len(tokenizer.encode(text))


def chunk_text(text: str, chunk_size: int = 256, chunk_overlap: int = 64) -> list[str]:
    """
    Create chunks of text
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[str] = [doc.page_content for doc in text_splitter.create_documents([text])]
    return chunks


def embed_text(text_chunks: list[str]) -> list[list[float]]:
    """
    Embed the text in each chunk
    Return a list of embeddings
    """
    embeddings = []
    for chunk in text_chunks:
        embeddings.append(get_embedding(chunk))
    return embeddings


def get_embedding(text: str) -> List[float | int]:
    """
    Get embedding for single chunk of text and return it as a list
    """
    response: CreateEmbeddingResponse = LLM_CLIENT.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_NAME
    )
    logger.info(f'{response.usage.prompt_tokens=}')
    logger.info(f'{response.usage.total_tokens=}')
    # logger.debug(f'{dict(response)}')
    return response.data[0].embedding


def save_to_vectorstore(documents: list[str], embeddings: list[list[float]]) -> None:
    collection = VECTOR_DB_CLIENT.get_or_create_collection(name='docs-chat')
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[str(uuid4()) for _ in range(len(documents))]
    )


def get_document_context(user_message) -> List[str]:
    collection = VECTOR_DB_CLIENT.get_or_create_collection(name='docs-chat')
    results = collection.query(
        query_embeddings=get_embedding(user_message),
        n_results=5,
        include=['documents', 'distances']
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains":"search_string"}
    )
    if not results:
        return []
    # distances = [d for d in results['distances'][0]]
    # maybe check the distances for a threshold?
    documents = [d for d in results['documents'][0]]
    return documents


def get_bot_response(user_message: str, document_context: List[str], chat: Chat) -> str:
    prev_messages: List[ChatCompletionMessageParam] = []
    for m in chat.messages:
        mdict: ChatCompletionMessageParam = {'role': 'user', 'content': m.text}
        prev_messages.append(mdict)
    current_prompt = PROMPT.format(document_context, user_message)
    logger.debug('*'*88)
    logger.debug(current_prompt)
    logger.debug('*'*88)
    messages: List[ChatCompletionMessageParam] = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        *prev_messages,
        {'role': 'user', 'content': current_prompt}
    ]

    def _get_bot_response(messages: List[ChatCompletionMessageParam]) -> str:
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_NAME,
            messages = messages,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
        )
        return str(response.choices[0].message.content)

    return retry(_get_bot_response, (messages,))

def save_to_file(file_bytes: BytesIO, filename: str, filedir: Path):
    with open(filedir / filename, 'wb') as f:
        f.write(file_bytes.read())


def slugify(value: Any) -> str:
    """
    Based off Django Framework code - https://github.com/django/django/blob/main/django/utils/text.py
    """
    value = str(value)
    value = (
        unicodedata.normalize('NFKD', value)
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def clean_whitespace(text):
    return re.sub(r'\s+', r' ', text)


def check_api_key() -> None:
    models = LLM_CLIENT.models.list()
    logger.debug(models)


def set_api_key(api_key: str) -> None:
    LLM_CLIENT.api_key = api_key


def retry(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    initial_delay: float = 0.5,
    multiplier: float = 1.5,
    jitter: float | None = 0.5,
    max_delay: float = 32.0,
    max_retries: int = 10,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """
    Retry a function with exponential backoff. Defaults:
    - 0.5s initial delay between retries
    - 1.5x multiplier for exponential backoff
    - 50% jitter (so actual delay ranges from 50% to 150% of calculated delay)
    - 32s max delay
    - 10 max retries
    - retry on any exception
    """
    for attempt in range(max_retries):
        try:
            return func(*args)
        except exceptions as e:
            if attempt == max_retries - 1:
                raise
            print(e)

            delay = initial_delay * (multiplier ** attempt)

            if jitter is not None:
                # with jitter = 0.5, delay can be 50% higher or lower, so jitter_range = 1.0
                jitter_range = 2 * jitter
                # select a random value from the range
                # subtract the jitter to center it around 0
                # add 1 to center it around 1 to get final jitter multiplier
                jitter_factor = 1 + (random.random() * jitter_range - jitter)
                # adjust delay by jitter_factor
                delay = delay * jitter_factor

            delay = min(delay, max_delay)

            print(f'{func.__name__} - attempt {attempt + 1} failed, retrying in {delay:.2f} seconds...')
            time.sleep(delay)
