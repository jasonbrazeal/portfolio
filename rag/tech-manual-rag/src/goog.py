import json
import re
from pathlib import Path

from google import genai
from google.cloud import storage, vision
from google.genai.types import HttpOptions,CreateCachedContentConfig
from google.genai.errors import ClientError


MODEL = 'gemini-2.0-flash-001'


def get_google_client() -> genai.Client:
    """
    Get an Google GenerativeAI client
    """
    return genai.Client(http_options=HttpOptions(api_version='v1'))


def upload_blob(bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)


def download_blob(bucket_name, source_blob_name, destination_file_path):
    """Downloads a blob to a local file."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)


def download_blob_to_memory(bucket_name, source_blob_name):
    """Downloads a blob to memory and returns its bytes."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()


def list_blobs(bucket_name) -> list[str]:
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    names = []
    for blob in blobs:
        names.append(blob.name)
    return names


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()


def delete_all_blobs(bucket_name):
    """Deletes all blobs from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    for blob in blobs:
        blob.delete()


def async_detect_document(gcs_source_uri: str, gcs_destination_uri: str) -> None:
    """
    OCR with PDF/TIFF as source files on GCS
    https://cloud.google.com/vision/docs/pdf
    """
    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'

    # How many pages should be grouped into each json output file.
    batch_size = 100

    client = vision.ImageAnnotatorClient()

    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type=mime_type)

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size
    )

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )

    operation = client.async_batch_annotate_files(requests=[async_request])

    print('waiting for the text extraction to finish...')
    operation.result(timeout=420)
    print('done')


def get_text_from_pdf_gcvision(filepath: Path) -> list[str]:
    """
    Extract text from pdf using Google Cloud Vision API
    """
    # upload pdf to gcs bucket
    upload_blob('tech-manual-rag', filepath, filepath.name)
    # extract text to  with the same name
    cloud_dst = filepath.with_suffix('.json').name
    async_detect_document(f'gs://tech-manual-rag/{filepath.name}', f'gs://tech-manual-rag/{cloud_dst}')

    pages = []
    # get all json blobs in bucket
    output_blobs = [name for name in list_blobs('tech-manual-rag') if name.endswith('.json')]
    for blob in output_blobs:
        json_string = download_blob_to_memory('tech-manual-rag', blob)
        json_string = json_string.decode('utf-8')
        response = json.loads(json_string)

        for page_response in response['responses']:
            annotation = page_response['fullTextAnnotation']
            if annotation['text']:
                # print(f"full text:\n{annotation['text']}")
                pages.append(annotation['text'])
            else:
                print(f'skipping page because it has no text: {page_response}')

    delete_all_blobs('tech-manual-rag')
    return pages


def cache_document_prompt(text: str, doc_id: str) -> tuple[str | None, int | None]:
    """
    Cache a document prompt
    """

    client = get_google_client()

    system_instruction = """
    You are an expert researcher. You always stick to the facts in the sources provided, and never make up new facts.
    Now look at this document, and the chunk of text from it, and answer the question that follows.
    """

    print(f'caching document {doc_id}')
    try:
        content_cache = client.caches.create(
            model=MODEL,
            config=CreateCachedContentConfig(
                contents=[text],
                system_instruction=system_instruction,
                display_name=f'doc-{doc_id}-cache',
                ttl='14400s',
            ),
        )
    except ClientError as e:
        try:
            # handle specific error case where document is too short to cache
            # (Gemini requires minimum 4096 tokens as of 4/15/25)
            if (e.status is not None and
                e.status == 'INVALID_ARGUMENT' and
                e.message is not None and
                'minimum token count to start caching' in e.message):
                # extract token count from error message
                token_count = int(re.search(r'\d+', e.message).group())
                print(f'Unable to cache document {doc_id} because it is too short ({token_count} tokens)')
                return None, token_count
            # for any other ClientError, don't cache document
            else:
                print(f'Error caching document {doc_id}: {e}')
                return None, None
        # for any other error, don't cache document
        except Exception as e:
            print(f'Error caching document {doc_id}: {e}')
            return None, None

    print(content_cache.name)
    print(content_cache.usage_metadata)
    print(f'document {doc_id} cached')

    return content_cache.name, content_cache.usage_metadata.total_token_count


def delete_all_cached_content() -> None:
    """
    Delete all cached content
    """
    client = get_google_client()
    content_cache_list = client.caches.list()
    for content_cache in content_cache_list:
        if not content_cache.name:
            continue
        client.caches.delete(name=content_cache.name)
    return


def delete_cached_content(cache_name: str) -> None:
    """
    Delete cached content
    """
    client = get_google_client()
    client.caches.delete(name=cache_name)
    return


def list_cached_content() -> None:
    """
    List cached content
    """
    client = get_google_client()
    content_cache_list = client.caches.list()

    for content_cache in content_cache_list:
        print(f'Cache "{content_cache.name}" for model "{content_cache.model}"')
        print(f'Last updated at: {content_cache.update_time}')
        print(f'Expires at: {content_cache.expire_time}')
    return
