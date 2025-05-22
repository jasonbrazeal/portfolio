# Intent Classification for Conversational AI - Summary

This project explores a traditional machine learning-based approach to intent classification for a simple personal assistant chatbot use case. We use the CLINC-150 dataset, a text classification dataset covering many intent classes. We explore and augment the data, train several different models using various algorithms, tune their hyperparameters, and evaluate their output. The focus is on simpler algorithms that are relatively fast and easy to train and optimize yet still perform well on the data.

## Tech

* Python
* Scikit-learn
* Pandas
* Pydantic
* Spacy
* NLTK
* OpenAI API
* Matplotlib

## Intent Classification for Conversational AI

Traditional, non-generative chatbots generally have more scripted flows and responses than newer LLM-backed systems, which is ideal for some use cases. The main components of the natural language understanding (NLU) capabilities of this type of chatbot are entity extraction and intent recognition. Relevant entities are identified and used as context for later steps in the conversation. Intent classification is how the AI understands the user's query. It allows the chatbot, at each step of the conversation, to take action (e.g. call an API to translate a word or make a new calendar appointment) and respond appropriately to advance the conversation (e.g. ask a follow-up question, move on to the next topic, end the conversation, etc.). At each step there is a predefined set of intents that the user might express, and the objective is to determine which intent class the user's query belongs to. As an example, a customer service-related chatbot for an internet service provider would likely need to handle intents such as 'user_needs_to_report_outage', 'user_needs_technical_support', 'user_wants_to_pay_bill', and several others.

This project presents a traditional, i.e. non-generative, machine learning-based approach to intent classification for a personal assistant chatbot use case. In the EDA notebook, we will explore the CLINC-150 dataset^, a text classification dataset with a wide range of intent classes over different domains. I chose this dataset because it contains intent data relevant to the personal assistant use case. In practice, data curation is often a long process involving creating or obtaining relevant data, organizing it into datasets and adding annotations, and maintaining them over time so they will continue to provide value. For this project, we will choose a subset of the intents from the CLINC-150 dataset and augment this data with a couple of new intents we want our chatbot to recognize. In the modeling notebook, we will train several intent classification models using different algorithms, tune their hyperparameters, and evaluate the results based on relevant performance metrics to select a final model for the chatbot. We will focus on simpler algorithms that are relatively fast and easy to train and optimize yet still perform well on our data.

^References:
Stefan Larson, et al. 2019. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1311–1316, Hong Kong, China. Association for Computational Linguistics.


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

### Models and Data

* either run all the notebooks to generate them or download them using the shell script

```bash
./download_data_models.sh
```

* FYIs
    * models have a timestamp/version; `newest_version` is hardcoded in notebooks 5 and 6 and should be updated after notebook 4 saves the vectorizer if you are running all the notebooks
    * notebook 6 contains code to load the models for inference; it is ready to run inference by itself with the latest model downloaded with the shell script


## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

* [dev@jasonbrazeal.com](mailto:dev@jasonbrazeal.com)
* [https://www.linkedin.com/in/jasonbrazeal](https://www.linkedin.com/in/jasonbrazeal)
* [https://jasonbrazeal.com](https://jasonbrazeal.com)
