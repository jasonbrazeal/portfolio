# Research Paper Writer

This project is an AI agent designed to automate the research and writing process. It leverages the Google Search and Semantic Scholar APIs to search the internet for web content and pdfs of relevant academic papers. Using the OpenAI Assistants API and online vector store, it generates a research paper from the content. The Research Paper Writer also includes a self-evaluation mechanism that identifies and resolves potential issues, ensuring the production of high-quality, well-structured academic papers.

## Tech

* Python
* BeautifulSoup
* OpenAI Assistants API
* Google Search API
* Semantic Scholar API

## Environment setup

### General

* install python@3.12 and uv (optional)
* create a venv anywhere and activate it
* sync it from the project root using uv or just pip:
    * `uv sync --active`
    * `python -m pip install -r requirements.txt`

### MacOS-specific

* install python@3.12 and uv through Homebrew
* create a venv anywhere and activate it
```bash
cd /path/to/project/src
uv venv /path/to/venv --python=/path/to/python
source /path/to/venv/bin/activate
uv sync --active
```
* to use brew-installed python versions, replace `/path/to/python` with `"$(brew --prefix python@3.12)/libexec/bin/python"` for whatever version you'd like

## Run script

* include paper topic as command line script arguments:
```python
python research_paper_writer.py bald eagles
```

## OpenAI online artifacts

* running the script will create artifacts in OpenAI:
    * Writer Assistant
    * Editor Assistant
    * Rewriter Assistant
    * vector store
    * uploaded files
* be sure to delete these if not needed for anything else:
    * https://platform.openai.com/assistants
    * https://platform.openai.com/storage

## Sample output

* download sample output with this script

```bash
./download_sample_output.sh
```

## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

* [dev@jasonbrazeal.com](mailto:dev@jasonbrazeal.com)
* [https://www.linkedin.com/in/jasonbrazeal](https://www.linkedin.com/in/jasonbrazeal)
* [https://jasonbrazeal.com](https://jasonbrazeal.com)
