#!/usr/bin/env python

import os
import re
import shutil
import sys
from pathlib import Path
from time import sleep
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from langchain_google_community import GoogleSearchAPIWrapper
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from utils import retry, slugify

LLM_MODEL = 'gpt-4.1-2025-04-14'
LLM_TEMPERATURE: float = 0.2
client: OpenAI = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

papers_search_url = 'https://api.semanticscholar.org/graph/v1/paper/search?{}&year=1995-2025&openAccessPdf&fields=abstract,title,authors'
papers_get_url = 'https://api.semanticscholar.org/graph/v1/paper/{}?fields=paperId,openAccessPdf,externalIds'


def search_papers(query) -> list[dict]:
    '''
    Given a query, search for relevant papers using the Semantic Scholar API
    '''
    query_encoded = urlencode({'query': query})
    # print(papers_search_url.format(query_encoded))
    papers = []
    attempts = 1
    # sometimes this api doesn't return anything, but it eventually does
    # we will try 15 times before giving up
    while not papers and attempts <= 15:
        print(f'searching for papers through Semantic Scholar API, attempt #{attempts}')
        response = retry(requests.get, (papers_search_url.format(query_encoded),))
        response_json = response.json()
        papers = response_json.get('data', [])
        attempts += 1
        # limit of 1 request per second without an api key
        sleep(2)
    if not papers:
       raise Exception('Semantic Scholar API error')
    return papers


def fetch_papers(paper_search_results: list[dict], paper_pdf_dir: Path, max_papers: int = 5) -> tuple[list[str], list[Path]]:
    '''
    Given paper search results from the Semantic Scholar API, get info on
    each paper (up to min_papers), fetch a pdf of them, and save them to disk.
    '''
    citations: list[str] = []
    paper_pdf_paths: list[Path] = []
    pdfs_written = 0
    for result in paper_search_results:
        if pdfs_written >= max_papers:
            break
        try:
            response = retry(requests.get, (papers_get_url.format(result['paperId']),))
            response_json = response.json()
            # print(f'{response_json=}')
            pdf_url = response_json.get('openAccessPdf', {}).get('url')
            digital_object_identifier = response_json.get('externalIds', {}).get('DOI', '')
            # print(f'{pdf_url=}')
            # limit of 1 request per second without an api key
            sleep(2)
            if not pdf_url:
                continue
            paper_pdf_path = paper_pdf_dir / f"{slugify(result['title'])}.pdf"
            response_pdf = retry(requests.get, {'url': pdf_url, 'allow_redirects': True, 'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'}})
            if response_pdf.status_code != 200:
                continue
            with paper_pdf_path.open('wb') as f:
                f.write(response_pdf.content)
            pdfs_written += 1
            paper_pdf_paths.append(paper_pdf_path)
            if digital_object_identifier:
                doi_response = retry(requests.get, {'url': f'https://citation.doi.org/format?doi={digital_object_identifier}&style=apa&lang=en-US'})
                citations.append(doi_response.text)
            else:
                authors = ', '.join([auth['name'] for auth in result['authors']])
                citations.append(f'{result["title"]} by {authors}')
        except Exception as e:
            print(e)
            continue
    return citations, paper_pdf_paths


def search_web(query: str, num_results: int = 5, max_results: int = 10) -> str:
    '''
    Search the internet for the given query
    Return <max_results> results from search API, fetch pages and return <num_results> to user
    Requires environment variables GOOGLE_CSE_ID and GOOGLE_API_KEY
    to be set. See:
    https://python.langchain.com/v0.2/docs/integrations/tools/google_search
    https://console.cloud.google.com/apis/credentials
    https://programmablesearchengine.google.com/controlpanel/create
    '''
    search = GoogleSearchAPIWrapper()
    search_results = search.results(query=query, num_results=max_results)
    web_content = []
    # fetch text from pages returned in search results until we have <num_results> results
    for result in search_results:
        try:
            response = retry(requests.get, {'url': result['link'], 'allow_redirects': True, 'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'}}, max_retries=3)
        except Exception as e:
            print(f'skipping unretrievable page {result["link"]}:')
            print(e)
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\n+', r'\n', text)
        result['text'] = text
        del result['snippet']
        web_content.append(result)
        if len(web_content) >= num_results:
            break
    web_content_str = ''
    for item in web_content:
        web_content_str += f'# {item["title"]} ({item["link"]})\n'
        web_content_str += f'{item["text"]}\n\n'
    return web_content_str


def get_search_string(user_query: str) -> str:
    '''
    Return a search string for a given user query
    The paper search seems to work better with search strings instead of full user queries
    '''
    system_prompt = 'you are a helpful assistant'
    current_prompt = f'''
    come up with a search string that would work well with a
    web search engine for researching the user's query. only output the search string, nothing else.

    for example:

    user query: Why is homelessness such a severe problem in San Francisco?
    output: homelessness in san francisco

    user query: Does P = NP?
    output: does p = np

    user query: {user_query}
    output:
    '''

    messages: list[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': current_prompt}
        ]
    response = retry(client.chat.completions.create, {
        'model': LLM_MODEL,
        'messages': messages,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': 16,
    })
    # print(f'search string = {response.choices[0].message.content}')
    if response.choices[0].message.content:
        return response.choices[0].message.content
    else:
        return user_query


def create_vectorstore(file_paths: list[Path]) -> str:
    vector_store = client.vector_stores.create(name='research papers')

    file_streams = [path.open('rb') for path in file_paths]
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    print(f'files uploaded to OpenAI vector store ({vector_store.id}):\n{file_batch.file_counts}')
    return vector_store.id


def write_paper(web_content: str, user_query: str, vector_store_id: str, research_papers: str) -> tuple[str, str]:
    system_prompt = '''
    you are a world class researcher and an expert writer
    you have access to a file_search tool, which can help find information from the files
    always search the files for relevant information in addition to the context given
    in the user's message
    '''
    current_prompt = f'''
    write a research paper at the level of an undergraduate college student,
    approximately 1500-2500 words, about this subject or answering this question:
    {user_query}

    here is some web content about the topic:
    {web_content}

    you also have access to research papers in the form of files you can search using the file_search tool. here are their citations:
    {research_papers}

    use the context above (web content) and use the file_search tool to find
    other relevant information in the files (existing research papers) you have access to. perform at least 3 file_search calls with different keywords.
    be sure to note the file names of the file_search results; you will need to cite the paper if you use information from it
    at the bottom of the paper, include citations to the papers you searched and/or the web pages, at least 3 relevant papers and 2 relevant web pages, if available. use the citations from the papers exactly as they are formatted above, and cite web links in the format: title (url).
    do not include any citations in the body of the paper, and do not cite social media or forum sites like reddit.com
    there should be no text following the references, and everything should be plain text academic prose, no markdown, no bulleted lists
    the paper should be organized into 5-8 sections, with a paper title at the top and a title header before each section. the first and last sections are always "Introduction" and "Conclusion", and the middle sections depend on the content.
    '''
    assistant = client.beta.assistants.create(
        name='Writer',
        instructions=system_prompt,
        description='a world class researcher and an expert writer.',
        model=LLM_MODEL,
        tools=[{'type': 'file_search'}],
        tool_resources={'file_search': {'vector_store_ids': [vector_store_id]}},
    )
    print('\n######### writing paper #########\n')
    thread = client.beta.threads.create(
        messages=[
            {
            'role': 'user',
            'content': current_prompt,
            }
        ]
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        print(text := messages.data[0].content[0].text.value)
    else:
        print(run.last_error)
        raise Exception(f'an error occurred with the assistant api, status={run.status}, last_error={run.last_error}')
    return text, assistant.id


def get_feedback(paper: str, web_content: str, vector_store_id: str, research_papers: str) -> tuple[str, str]:
    system_prompt = '''
    you are an expert editor
    you have access to a file_search tool, which you can use to verify information from research paper pdf files
    always search the files to for relevant information in addition to the web content given
    in the user's message
    '''
    current_prompt = f'''
    provide editorial feedback on the following research paper, noting any problems, areas of improvement, or inconsistencies with the provided web content or research papers
    {paper}

    here is some web content about the topic, used by the writer of the paper:
    {web_content}

    you also have access to the same research papers as the writer of the paper, in the form of files you can search using the file_search tool. here are their citations:
    {research_papers}

    use the context above (web content) and use the file_search tool to find
    other relevant information in the files (existing research papers)
    avoid excessive markdown and formatting in your response
    '''
    assistant = client.beta.assistants.create(
        name='Editor',
        instructions=system_prompt,
        description='an expert editor',
        model=LLM_MODEL,
        tools=[{'type': 'file_search'}],
        tool_resources={'file_search': {'vector_store_ids': [vector_store_id]}},
    )
    print('\n######### writing feedback #########\n')
    thread = client.beta.threads.create(
        messages=[
            {
            'role': 'user',
            'content': current_prompt,
            }
        ]
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        print(text := messages.data[0].content[0].text.value)
    else:
        print(run.status)
        print(run.last_error)
        raise Exception(f'an error occurred with the assistant api, status={run.status}, last_error={run.last_error}')
    return text, assistant.id


def rewrite_paper(paper: str, feedback: str, vector_store_id: str, research_papers: str) -> tuple[str, str]:
    system_prompt = '''
    you are a world class researcher and an expert writer
    you have access to a file_search tool, which can help find information from the files
    always search the files for relevant information in addition to the context given
    in the user's message
    '''
    current_prompt = f'''
    rewrite the following paper incorporating the editorial feedback.

    paper:
    {paper}

    editorial feedback:
    {feedback}

    here is some web content about the topic:
    {web_content}

    you also have access to research papers in the form of files you can search using the file_search tool. here are their citations:
    {research_papers}

    use the context above (web content) and use the file_search tool to find/check
    other relevant information in the files (existing research papers) you have access to.

    there should be no text following the references, and everything should be plain text academic prose, no markdown, no bulleted lists
    the paper should be organized into 5-8 sections, with a paper title at the top and a title header before each section. the first and last sections are always "Introduction" and "Conclusion", and the middle sections depend on the content.
    '''
    assistant = client.beta.assistants.create(
        name='Rewriter',
        instructions=system_prompt,
        description='a world class researcher and an expert writer',
        model=LLM_MODEL,
        tools=[{'type': 'file_search'}],
        tool_resources={'file_search': {'vector_store_ids': [vector_store_id]}},
    )
    print('\n######### rewriting paper #########\n')
    thread = client.beta.threads.create(
        messages=[
            {
            'role': 'user',
            'content': current_prompt,
            }
        ]
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        print(text := messages.data[0].content[0].text.value)
    else:
        print(run.status)
        print(run.last_error)
        raise Exception(f'an error occurred with the assistant api, status={run.status}, last_error={run.last_error}')
    return text, assistant.id


def delete_artifacts(vector_store_id: str, assistant_ids: list[str]) -> None:
    '''
    Delete OpenAI artifacts created by this script: assistants, files, vector stores
    Note: does not delete threads created by assistants. See `delete_threads.py`.
    '''
    vector_store_files = client.vector_stores.files.list(vector_store_id)
    for file in vector_store_files:
        client.files.delete(file_id=file.id)
    client.vector_stores.delete(vector_store_id)
    for assistant_id in assistant_ids:
        client.beta.assistants.delete(assistant_id)


if __name__ == '__main__':

    OUTPUT_DIR = Path(__file__).parent.parent / 'output'
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    if len(sys.argv) <= 1:
        raise Exception('please include query as cli script args')

    user_query = ' '.join(sys.argv[1:])
    print(f'######### {user_query} #########')
    user_query_slug = slugify(user_query)

    current_output_dir = OUTPUT_DIR / user_query_slug
    if current_output_dir.exists():
        response = input(f'output directory {current_output_dir} already exists. press enter to delete, ctrl+c to stop...')
        shutil.rmtree(current_output_dir)
    current_output_dir.mkdir()

    # web research
    search_string = get_search_string(user_query)
    web_content = search_web(user_query)
    with (current_output_dir / 'web_content.txt').open('w') as f:
        f.write(web_content)
    papers = search_papers(search_string)
    citations, paper_paths = fetch_papers(papers, current_output_dir)
    citations_str ='\n'.join(citations)

    # create vectorstore
    vector_store_id = create_vectorstore(paper_paths)

    # Writer - write paper
    paper, writer_id  = write_paper(web_content, user_query, vector_store_id, citations_str)
    with (current_output_dir / 'paper.txt').open('w') as f:
        f.write(paper)

    # Editor - editorial feedback
    feedback, editor_id  = get_feedback(paper, web_content, vector_store_id, citations_str)
    with (current_output_dir / 'feedback.txt').open('w') as f:
        f.write(feedback)

    # Rewriter - rewrite paper
    rewrite, rewriter_id  = rewrite_paper(paper, feedback, vector_store_id, citations_str)
    with (current_output_dir / 'rewrite.txt').open('w') as f:
        f.write(rewrite)

    delete_artifacts(vector_store_id, [writer_id, editor_id, rewriter_id])
