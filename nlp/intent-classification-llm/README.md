# Intent Classification with LLMs

This project explores how large language models (LLMs) can be used for intent classification tasks. We investigate different prompting techniques and test them with different models. As a benchmark, we use same intent dataset used in my [traditional intent classification project](https://github.com/jasonbrazeal/portfolio/tree/master/nlp/intent-classification), allowing us to directly compare LLM performance against the previously implemented Logistic Regression model.

## Tech

* Python
* OpenAI API
* Google Gemini API
* Anthropic API

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

### Data

Data from the LLM evaluations are in `data/llm`. The rest of the data is from my other [intent classification project](https://github.com/jasonbrazeal/portfolio/tree/master/nlp/intent-classification).

* download using the shell script

```bash
./download_data.sh
```

## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

* [dev@jasonbrazeal.com](mailto:dev@jasonbrazeal.com)
* [https://www.linkedin.com/in/jasonbrazeal](https://www.linkedin.com/in/jasonbrazeal)
* [https://jasonbrazeal.com](https://jasonbrazeal.com)
