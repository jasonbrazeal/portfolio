# Fiction Author Assistant

This project is a simple AI-powered writing companion tailored for fiction authors. It is a tool that transforms story outlines into rich, engaging prose using LLMs. Authors can input their story steps or outline, and the AI will generate polished, coherent text that matches their desired genre. I built this tool to explore how LLMs can serve as creative partners in the writing process.

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

## Script setup

* see examples for the generic prompt and the templated prompt in fiction_author_assistant.py
* add your own instructions (`list[str]`)
* set params for the templated prompt: genre, setting, characters

## Run script

```python
python fiction_author_assistant.py
```

## Example Output

See `fiction_author_assistant.py` for the sci-fi and rom-com instructions and params used to generate this output.

* anthropic_romcom.txt
* anthropic_scifi.txt
* google_romcom.txt
* google_scifi.txt
* openai_romcom.txt
* openai_scifi.txt

## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

* [dev@jasonbrazeal.com](mailto:dev@jasonbrazeal.com)
* [https://www.linkedin.com/in/jasonbrazeal](https://www.linkedin.com/in/jasonbrazeal)
* [https://jasonbrazeal.com](https://jasonbrazeal.com)
