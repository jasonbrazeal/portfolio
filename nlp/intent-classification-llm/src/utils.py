import os
import random
import time
from typing import Any, Callable
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from tqdm import tqdm


RANDOM_SEED = 42

# not using __file__ here so this code works from notebooks and from the interpreter
DATA_DIR = Path(os.getcwd()).parent / 'data'
ALL_DATA_PATH = DATA_DIR / 'data_all.jsonl'
INTENT_LABELS_PATH = DATA_DIR / 'intent_labels.json'
IMG_DIR = Path(os.getcwd()).parent / 'img'

def init_nb():
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


def strip_indent(text: str) -> str:
    return '\n'.join(line.lstrip() for line in text.splitlines())


def plot_accuracy(accuracy_data: dict[str, dict[str, float]], output_file: Path | None = None, split_charts: bool = False):
    '''
    plot accuracy as a grouped bar chart with 6 bars per provider
    if split_charts is True, create separate charts for each provider
    '''

    SILVER = '#c2c1c1'
    COLORS = [
        '#4682B4',  # zero_shot_cot - steel blue
        '#2F4F4F',  # zero_shot - dark slate gray
        '#5F7F8F',  # 10_shot - darker slate gray
        '#5F9EA0',  # 30_shot - cadet blue
        '#A9A9A9',  # 10_shot_cot - dark gray
        '#6A7FDB',  # 30_shot_cot - muted blue
        ]

    plt.rcParams['text.color'] = SILVER
    plt.rcParams['axes.labelcolor'] = SILVER
    plt.rcParams['xtick.color'] = SILVER
    plt.rcParams['ytick.color'] = SILVER
    plt.rcParams['axes.edgecolor'] = SILVER

    providers = list(accuracy_data.keys())
    prompt_types = ['zero_shot_prompt', 'zero_shot_cot_prompt', 'k_shot_prompt10', 'k_shot_cot_prompt10', 'k_shot_prompt30', 'k_shot_cot_prompt30']

    def get_bar_color(prompt_type: str) -> str:
        if 'zero_shot' in prompt_type:
            return COLORS[0 if 'cot' not in prompt_type else 1]
        elif 'k_shot' in prompt_type:
            if 'cot' not in prompt_type:
                return COLORS[2 if '10' in prompt_type else 3]
            return COLORS[4 if '10' in prompt_type else 5]
        return COLORS[0]  # fallback

    def format_label(prompt_type: str) -> str:
        label = prompt_type.replace('_', ' ').title().replace('Cot', 'COT')
        if label.startswith('K Shot'):
            k = label[-2:]
            label = label[:-2]
            label = label.replace('K Shot', f'{k}-shot')
        return label

    def create_chart(ax, data, x_positions, provider_name=None):
        width = 0.12  # reduced width to fit 6 bars

        # add bars and labels
        for i, prompt_type in enumerate(prompt_types):
            if isinstance(data, dict):  # single provider
                value = data[prompt_type] * 100
                bar_color = get_bar_color(prompt_type)
                offset = width * (i - 2.5)  # center the group of bars
                ax.bar(x_positions + offset, value, width, label=format_label(prompt_type), color=bar_color)
            else:  # multiple providers
                values = [provider_data[prompt_type] * 100 for provider_data in data]
                bar_colors = [get_bar_color(prompt_type)] * len(values)
                offset = width * (i - 2.5)
                ax.bar(x_positions + offset, values, width, label=format_label(prompt_type), color=bar_colors)

        # add a horizontal line at 95% accuracy
        ax.axhline(y=95, color='#a35046', linestyle=':', alpha=0.7)

        # set labels and configure axes
        ax.set_ylabel('Accuracy (%)')
        title = f'Intent Classification Accuracy - {provider_name}' if provider_name else 'Intent Classification Accuracy by Model and Prompt Type'
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([provider_name] if provider_name else providers)
        ax.set_ylim(75, 100)
        ax.set_yticks(np.arange(75, 101, 5))

        # add padding around legend
        legend = ax.legend(framealpha=0, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        for text in legend.get_texts():
            text.set_alpha(1)
            text.set_color(SILVER)

    def save_or_show(fig, output_path=None):
        'save figure to file or show it'
        if matplotlib.get_backend() == 'inline':
            plt.show()
        else:
            if output_path is not None:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False, format='png')
                print(f'image created: {output_path}')
            plt.close()

    if split_charts:
        # create separate figure for each provider
        for provider in providers:
            fig, ax = plt.subplots(figsize=(6, 6))
            create_chart(ax, accuracy_data[provider], np.arange(1), provider)
            plt.tight_layout(rect=(0, 0.15, 1, 1))

            if output_file is not None:
                provider_file = output_file.parent / f'{output_file.stem}_{provider.lower()}{output_file.suffix}'
                save_or_show(fig, provider_file)
            else:
                save_or_show(fig)
    else:
        # create single figure with all providers
        fig, ax = plt.subplots(figsize=(12, 6))
        create_chart(ax, [accuracy_data[p] for p in providers], np.arange(len(providers)))
        plt.tight_layout(rect=(0, 0.15, 1, 1))
        save_or_show(fig, output_file)
