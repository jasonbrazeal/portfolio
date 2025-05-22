import json
import re
import random
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame


def write_to_jsonl(results: list[dict], filepath: Path) -> None:
    with open(filepath, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


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


def clean_whitespace(text):
    return re.sub(r'\s+', r' ', text)


def decode_unicode_escapes(text: str) -> str:
    """
    Convert unicode escape sequences to their printable symbols if possible.
    For example, converts '\u2122' to '™' if it's a printable character.
    """
    # find all unicode escapes like \u2122
    pattern = r'\\u[0-9a-fA-F]{4}'
    result = text

    for match in re.finditer(pattern, text):
        escape = match.group()
        try:
            # try to decode just this escape sequence
            decoded = escape.encode('utf-8').decode('unicode-escape')
            # verify it's printable
            decoded.encode('ascii', errors='strict')
            # replace just this escape with decoded version
            result = result.replace(escape, decoded)
        except UnicodeEncodeError:
            # keep original escape sequence if not printable
            continue

    return result


def save_or_show(output_path=None):
    'save figure to file or show it'
    if matplotlib.get_backend() == 'inline':
        plt.show()
    else:
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False, format='png')
            print(f'image created: {output_path}')
        plt.close()


def create_chart(df_retrieval: DataFrame, df_end_to_end: DataFrame, output_path=None):
    '''
    Create chart to compare Basic RAG performance (non-contextualized chunks,
    vector search only) to Advanced RAG (contextualized chunks, vector search +
    BM25 search)
    '''
    # filter for basic non-contextualized and advanced contextualized rows
    retrieval_compare = df_retrieval[((df_retrieval['retrieval_type'] == 'basic') & (df_retrieval['contextualized'] == False)) |
                                     ((df_retrieval['retrieval_type'] == 'advanced') & (df_retrieval['contextualized'] == True))]

    # group by retrieval_type and contextualized, take the best k (highest recall)
    best_retrieval = retrieval_compare.sort_values('recall', ascending=False).groupby(['retrieval_type', 'contextualized']).first().reset_index()

    combined_labels = ['Precision', 'Recall', 'F1', 'MRR', 'End-to-end Accuracy']
    metrics = ['precision', 'recall', 'f1', 'mrr']

    # get retrieval metric values
    basic_row = best_retrieval[(best_retrieval['retrieval_type'] == 'basic') & (best_retrieval['contextualized'] == False)].iloc[0]
    advanced_row = best_retrieval[(best_retrieval['retrieval_type'] == 'advanced') & (best_retrieval['contextualized'] == True)].iloc[0]
    basic_values = [basic_row[m] for m in metrics]
    advanced_values = [advanced_row[m] for m in metrics]

    # get end-to-end accuracy values
    basic_acc = df_end_to_end[(df_end_to_end['retrieval_type'] == 'basic') & (df_end_to_end['contextualized'] == False)]['accuracy'].iloc[0]
    advanced_acc = df_end_to_end[(df_end_to_end['retrieval_type'] == 'advanced') & (df_end_to_end['contextualized'] == True)]['accuracy'].iloc[0]
    basic_values.append(basic_acc)
    advanced_values.append(advanced_acc)

    x = range(len(combined_labels))
    bar_width = 0.35

    # colors for dark gray background
    vibrant_orange = '#e6a96b'    # medium, earthy orange
    vibrant_teal = '#4fa3a5'      # medium teal
    bg_color = '#22252a'          # dark gray background
    fg_color = '#f0f0f0'          # light foreground
    # grid_color = '#444444'        # subtle grid

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.bar([p - bar_width/2 for p in x], basic_values, width=bar_width, label='Basic RAG', color=vibrant_orange)
    ax.bar([p + bar_width/2 for p in x], advanced_values, width=bar_width, label='Advanced RAG', color=vibrant_teal)

    ax.set_xticks(x)
    ax.set_xticklabels(combined_labels, color=fg_color)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score', color=fg_color)
    ax.set_title('Basic RAG vs. Advanced RAG with contextual embeddings', color=fg_color)
    ax.legend(facecolor=bg_color, edgecolor=fg_color, labelcolor=fg_color)
    ax.tick_params(colors=fg_color)
    ax.yaxis.label.set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    ax.title.set_color(fg_color)
    ax.spines['bottom'].set_color(fg_color)
    ax.spines['top'].set_color(fg_color)
    ax.spines['right'].set_color(fg_color)
    ax.spines['left'].set_color(fg_color)
    # ax.grid(True, axis='y', color=grid_color, alpha=0.3)
    for i, v in enumerate(basic_values):
        ax.text(i - bar_width/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=9, color=fg_color)
    for i, v in enumerate(advanced_values):
        ax.text(i + bar_width/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=9, color=fg_color)
    plt.tight_layout()

    save_or_show(output_path)
