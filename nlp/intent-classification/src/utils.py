import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm

RANDOM_SEED = 42
RAW_DATA_URL = 'https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json'

# not using __file__ here so this code works from notebooks and from the interpreter
DATA_DIR = Path(os.getcwd()).parent / 'data'
RAW_DATA_PATH = DATA_DIR / 'data_raw.json'
FILTERED_DATA_PATH = DATA_DIR / 'data_filtered.jsonl'
GENERATED_DATA_PATH = DATA_DIR / 'data_generated.jsonl'
ALL_DATA_PATH = DATA_DIR / 'data_all.jsonl'
PROCESSED_DATA_PATH = DATA_DIR / 'data_processed.jsonl'
TFIDF_DATA_PATH = DATA_DIR / 'data_tfidf.jsonl'
INTENT_LABELS_PATH = DATA_DIR / 'intent_labels.json'

MODEL_DIR = Path(os.getcwd()).parent  / 'models'
TFIDF_VECTORIZER_PATH = MODEL_DIR / 'tfidf_vectorizer.pkl'

nlp = spacy.load('en_core_web_trf')

def init_nb():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    pd.options.mode.copy_on_write = True

    tqdm.pandas()

    # there are LOTS of spacy-internal pytorch warning being printed, this code supresses them:
    warnings.simplefilter(action='ignore', category=FutureWarning)

    reset_matplotlib()
    setup_matplotlib()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print('notebook initialized')


@dataclass
class Timer:
    '''
        A quick timer class (context manager) to simplify profiling code
    '''
    desc: str
    start: float = 0.0
    end: float = 0.0
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *exc_info):
        self.end = time.perf_counter()
        print(f'{self.desc}: {self.end - self.start} seconds')


def format_counter(x):
    output = []
    for gram, count in x:
        if type(gram) is tuple:
            output.append(f"{' '.join(gram)} ({count})")
        else:
            output.append(f"{gram} ({count})")
    return ', '.join(output)


def null_preprocessor(text: str) -> str:
    '''
    Null preprocessor since text has already been preprocessed
    '''
    return text


def spacy_tokenizer(text: str) -> list[str]:
    '''
    Tokenize text using spacy
    '''
    doc = nlp(text)
    return [token.text for token in doc]


def reset_matplotlib():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    print('matplotlib rc params reset')


def setup_matplotlib():
    plt.style.use('dark_background')
    matplotlib.rcParams['figure.edgecolor'] = '#22252A'
    matplotlib.rcParams['figure.facecolor'] = '#22252A'
    matplotlib.rcParams['axes.facecolor'] = '#22252A'
    matplotlib.rcParams['savefig.edgecolor'] = '#22252A'
    matplotlib.rcParams['savefig.facecolor'] = '#22252A'
    matplotlib.rcParams['font.family'] = 'monospace'
    matplotlib.rcParams['font.monospace'] = [
        'Hack Nerd Font Mono',
        'DroidSansMono Nerd Font',
        'Monaco',
        'Andale Mono'
        'Consolas',
        'Lucida Console',
        'Courier New',
        'Fixed',
        'Terminal',
        'monospace'
    ]
    # matplotlib.rcParams['figure.figsize'] = [12, 7]
    matplotlib.rcParams['savefig.pad_inches'] = .75
    # padding between label and chart
    matplotlib.rcParams['axes.labelpad'] = 4 # default
    # matplotlib.rcParams['axes.labelpad'] = 20
    # padding between title and chart
    matplotlib.rcParams['axes.titlepad'] = 6 # default
    # matplotlib.rcParams['axes.titlepad'] = 20
    matplotlib.rcParams['savefig.directory'] = '.'

    print('matplotlib rc params set')
