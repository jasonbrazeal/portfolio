# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Data Preparation
#
# The plan is to:
#
# * prepare the data for modeling
#   * encode intent labels
#   * represent text as vectors
#
# We need to represent the non-numerical text data as numbers since models don't operate on
# text directly. For the intent categories, we can simply encode the intent strings as integers,
# and decode the model prediction back to an intent string. For the utterances, we have many options:
#
# * simple frequency-based representations - bag of words, TF-IDF
# * early embedding models based on word/subword co-occurrence - word2vec (Google), GloVe (Stanford), fastText (Facebook)
# * newer deep embeddings models, more context-sensitive and based on weights of neural networks - ELMo, BERT, GPT
# * latest SOTA transformers models (see HuggingFace embeddings benchmarks leaderboard - https://huggingface.co/spaces/mteb/leaderboard)
#
# For this project we will keep things simple and see what results we can achieve using a simple
# frequency-based representation, TF-IDF. Term frequency captures how often a word
# appears in an utterance. More frequent words in a document tend to be more important.
# Inverse Document Frequency considers how much information the word provides. Words
# appearing in many utterances carry less weight (an inverse relationship).
#
#

# %%
import joblib
import time

import pandas as pd

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from utils import (
    init_nb, PROCESSED_DATA_PATH, INTENT_LABELS_PATH, TFIDF_DATA_PATH,
    TFIDF_VECTORIZER_PATH, null_preprocessor, spacy_tokenizer
)

init_nb()

# %%
# categorical intent label encoding

df = pd.read_json(PROCESSED_DATA_PATH, orient='records', lines=True)

# encode intent labels as numbers
encoder = LabelEncoder()
encoder.fit(df['intent_str'])
df.loc[:, 'intent'] = encoder.transform(df['intent_str'])
# or fit/transform together:
# df.loc[:, 'intent'] = LabelEncoder().fit_transform(df['intent_str'])

# int labels
int_labels = encoder.transform(encoder.classes_).tolist()

# str labels
str_labels = encoder.classes_.tolist()

# save mapping
intent_labels: DataFrame = pd.DataFrame(
    {'label_int': int_labels,
     'label_str': str_labels},
)
intent_labels.to_json(INTENT_LABELS_PATH)

# verify number of utterances per intent
print(df['intent'].value_counts())

# %% [markdown]
# ```text
# intent
# 28    159
# 6     153
# 1     150
# 23    150
# 17    150
# 0     150
# 18    150
# 14    150
# 26    150
# 16    150
# 25    150
# 7     150
# 13    150
# 20    150
# 24    150
# 11    150
# 22    150
# 29    150
# 27    150
# 9     150
# 15    150
# 5     150
# 19    150
# 4     150
# 10    150
# 3     150
# 2     150
# 12    149
# 21    149
# 8     148
# Name: count, dtype: int64
# ```

# %%
vectorizer_kwargs = {
    # if float, these parameters represent a proportion of documents,
    # if integer they are absolute counts
    'min_df': 1,  # ignore terms that have a document frequency strictly lower than the given threshold (default 1)
    'max_df': 0.95, # ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words, default 1.0)
    'analyzer': 'word', # 'char', char_wb', 'word' (default)
    'ngram_range': (1, 2), # count unigrams and bigrams
    'preprocessor': null_preprocessor, # text is already preprocessed
    'tokenizer': spacy_tokenizer, # we already tokenized the text, but it's easiest to redo it here using a function
    'token_pattern': None # since we are using spacy's tokenizer
}

# note: TfidfVectorizer combines CountVectorizer and TfidfTransformer
tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True, **vectorizer_kwargs)
print('fitting tfidf vectorizer...')
tfidf_v = tfidf.fit_transform(df.utterance)

# save vectorizer for inference in the next notebook
timestamp = int(time.time())
tfidf_path = TFIDF_VECTORIZER_PATH.parent / f'{TFIDF_VECTORIZER_PATH.stem}_{timestamp}{TFIDF_VECTORIZER_PATH.suffix}'
print(f'saving tfidf vectorizer to {tfidf_path}')
joblib.dump(tfidf, tfidf_path)

# We could've built this vectorizer into a sklearn Pipeline with the classifier.
# That would've allowed us to vary the vectorizer parameters in the grid search.
# I checked the ngram_range manually and found using unigrams and bigrams performed best.

tfidf_data_path = TFIDF_DATA_PATH.parent / f'{TFIDF_DATA_PATH.stem}_{timestamp}{TFIDF_DATA_PATH.suffix}'
df.loc[:, 'tfidf_vector'] = DataFrame(tfidf_v.toarray()).apply(lambda row: list(row.values), axis=1)
df.to_json(tfidf_data_path, orient='records', lines=True)


# %% [markdown]
# ```text
# fitting tfidf vectorizer...
# saving tfidf vectorizer...
# ```
