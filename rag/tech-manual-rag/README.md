# Tech Manual RAG

Tech Manual RAG is a project that explores more advanced RAG with LLMs. The main goal is to implement Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) and evaluate how it performs with a different kind of document dataset: technical manuals for devices.

In contrast to the naive RAG chat app I created before, [Docs Chat](https://github.com/jasonbrazeal/docs-chat), Tech Manual RAG focuses less on the web application and more on data ingestion and evaluations.

## Technology

* Python
* Elasticsearch
* Google Gemini API
* Hugging Face Transformers (embeddings model)
* Pandas
* Pydantic
* Matplotlib

## Retrieval-augmented generation + contextual retrieval

Basic RAG applications perform semantic similarity searches using word embeddings to extract relevant chunks of context that are added to the LLM prompt to aid generation. Contextual retrieval adds a few more steps:
* prepend context to each chunk that explains it and situates it within the overall document before embedding
* a second search--BM25 (Best Matching 25) search that is good at matching exact terms or phrases
* rank fusion techniques to combine and deduplicate results from the BM25 search and the semantic similarity search using contextual embeddings
* optional reranking model to rerank results based on their relevance to the prompt

In their [blog project](https://www.anthropic.com/news/contextual-retrieval), Anthropic ran many tests using different combinations of these techniques on various types of datasets (e.g. peer-reviewed research papers, fiction, codebases, etc.). They measured failed retrievals for the top 20 chunks, and they were able to reduce the failure rate by 67% on average when applying all these techniques as opposed to relying on naive semantic similarity retrieval.

## Data

The dataset for Tech Manual RAG consists of 41 pdf files that are mostly guidebooks and manuals for Texas Instruments calculators. They can be freely downloaded from the [TI website](https://education.ti.com/en/product-resources/guidebooks). These guidebooks, and tech manuals in general, represent a different linguistic genre than those tested by Anthropic. They are more instructional by nature, but still technical in content. They also contain many small diagrams (e.g. what would show on a calculator screen after pressing certain keys) and other artifacts like special fonts for the calculator keys (e.g. the symbol AC inside a little box). I thought it would be interesting to see how contextual retrieval compares to naive RAG using semantic and lexical search for this type of data.

## Set up a local environment

### Start an elasticsearch instance

```bash
docker network create elastic

docker pull docker.elastic.co/elasticsearch/elasticsearch:8.17.4

docker run --name es --net elastic -p 9200:9200 -it -m 6GB -e "xpack.ml.use_auto_machine_memory_percent=true" docker.elastic.co/elasticsearch/elasticsearch:8.17.4
# prints out password

# check that elasticsearch is running
export ELASTIC_PASSWORD="$(openssl rand -base64 12)"
echo $ELASTIC_PASSWORD
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
curl --cacert http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200
```

### Or use docker compose

```bash
# set env vars
export ES_VERSION=8.17.4
export ES_PORT=9200
# increase or decrease based on the available host memory (in bytes, 6 gib)
export MEM_LIMIT=6442450944
docker compose up -d
```

### Set up python environment - General

* install python@3.12 and uv (optional)
* create a venv anywhere and activate it
* sync it from the project root using uv or just pip:
    * `uv sync --active`
    * `python -m pip install -r requirements.txt`

### Set up python environment - MacOS-specific

* install python@3.12 and uv through Homebrew
* create a venv anywhere and activate it
```bash
cd /path/to/project/src
uv venv /path/to/venv --python=/path/to/python
source /path/to/venv/bin/activate
uv sync --active
```
* to use brew-installed python versions, replace `/path/to/python` with `"$(brew --prefix python@3.12)/libexec/bin/python"` for whatever version you'd like

### Download the data

* download all data
```bash
./download_data.sh
```
* original pdfs are in data/texas_instruments_manuals/
* extracted dataset is in data/tech-manual-rag.jsonl

## Code/ideas based on:

[https://github.com/anthropics/anthropic-cookbook/tree/main/skills/retrieval_augmented_generation](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/retrieval_augmented_generation)
[https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)

## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

* [dev@jasonbrazeal.com](mailto:dev@jasonbrazeal.com)
* [https://www.linkedin.com/in/jasonbrazeal](https://www.linkedin.com/in/jasonbrazeal)
* [https://jasonbrazeal.com](https://jasonbrazeal.com)
