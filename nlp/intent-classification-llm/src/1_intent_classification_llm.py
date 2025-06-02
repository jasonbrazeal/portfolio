# %% [markdown]
# ## Intent Classification with LLMs
#
# The plan is to:
#
# * explore LLM prompting strategies for intent classification
# * informally test the performance of our prompts with different flagship models
# * select a few promising prompts / models to evaluate in the next notebook
#
# The prompts are in prompts.py, see `render_template`. The strategies we'll test are:
#
# * zero-shot prompting
# * k-shot prompting (includes one-shot, few-shot, etc.)
# * chain-of-thought prompting
#
# Note: prompts are slightly different for each model. An attempt was made to follow the
# best practices for each provider according to the links below:
# [Anthropic prompting](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/classification/guide.ipynb)
# [Anthropic prompting](https://docs.anthropic.com/en/prompt-library/)
# [Google prompting](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/introduction-prompt-design)
# [Google prompting](https://ai.google.dev/gemini-api/docs/prompting-strategies)
# [OpenAI prompting](https://platform.openai.com/docs/guides/text?api-mode=responses)
# [General prompting techniques](https://www.promptingguide.ai/techniques) and references therein
#
# The current models (as of May 2025) are:
#
# * OpenAI GPT-4.1 (gpt-4.1-2025-04-14)
# * Google Gemini 2.5 Flash (gemini-2.5-flash-preview-04-17)
# * Anthropic Claude 3.7 Sonnet (claude-3-7-sonnet-20250219)

# %%
import json
import time
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from llm import OpenAIClient, AnthropicClient, GoogleClient
from utils import DATA_DIR, init_nb

init_nb()

# %%
# new examples
test_examples: list[dict[str, str]] = [
    {'text': 'how is the traffic heading into SF right now?', 'intent': 'traffic'},
    {'text': 'set an alarm for 7am tomorrow', 'intent': 'timer'},
    {'text': 'could you please help me find my phone', 'intent': 'find_phone'},
    {'text': "can you text mom and tell her I'm on my way", 'intent': 'text'},
    {'text': 'what is 18 times 4', 'intent': 'calculator'},
    {'text': 'tell me a random joke about ducks', 'intent': 'tell_joke'},
    {'text': 'are you a real person or nah', 'intent': 'are_you_a_bot'},
    {'text': 'can you add mouthwash to my shopping list', 'intent': 'shopping_list_update'},
    {'text': 'what should i call you?', 'intent': 'what_is_your_name'},
    {'text': 'hey there mister bot', 'intent': 'greeting'},
]

# from data_all.jsonl, downloadable with download_data.sh
# see this project: https://github.com/jasonbrazeal/portfolio/tree/master/nlp/intent-classification
dataset_examples: list[dict[str, str]] = [
    {'text': 'will you please tell me my to do list', 'intent': 'todo_list'},
    {'text': 'have you heard any great jokes lately', 'intent': 'tell_joke'},
    {'text': "name today's word of the day", 'intent': 'word_of_the_day'},
    {'text': 'can you tell me how to make homemade basil pesto', 'intent': 'food_beverage_recipe'},
    {'text': 'how would i say nice to meet you if i were russian', 'intent': 'translate'},
    {'text': 'las vegas weather today', 'intent': 'weather'},
    {'text': 'tell me when two minutes are up', 'intent': 'timer'},
    {'text': 'toss a coin i will take heads', 'intent': 'flip_coin'},
    {'text': 'clear out my entire todo list', 'intent': 'todo_list_update'},
    {'text': 'what is 25 percent of 6999', 'intent': 'calculator'}
]

with Path(DATA_DIR / 'intent_labels.json').open() as f:
    intent_labels = json.load(f)
intents: list[str] = list(intent_labels['label_str'].values())

# %%
# Anthropic

timestamp_anthropic = int(time.time())

anthropic = AnthropicClient(
    examples=dataset_examples,
    intents=intents,
)
responses = []
print('\nAnthropic text generating...')
for e in tqdm(test_examples):
    for prompt_name in (
        'anthropic.zero_shot_prompt', 'anthropic.k_shot_prompt',
        'anthropic.zero_shot_cot_prompt', 'anthropic.k_shot_cot_prompt'
    ):
        response = anthropic.generate_text(e['text'], prompt_name)
        responses.append({
            'provider': 'anthropic',
            'text': e['text'],
            'true_intent': e['intent'],
            'prompt_name': prompt_name,
            'response': response
        })

df_anthropic = DataFrame(responses)
# df_anthropic = pd.read_json(DATA_DIR / 'df_anthropic', orient='records', lines=True)
print(df_anthropic)
df_anthropic.to_json(DATA_DIR / 'llm' / f'df_anthropic_{timestamp_anthropic}.jsonl', orient='records', lines=True)

# %%
# Google

timestamp_google = int(time.time())

google = GoogleClient(
    examples=dataset_examples,
    intents=intents,
)
responses = []
print('\nGoogle text generating...')
for e in tqdm(test_examples):
    for prompt_name in (
        'google.zero_shot_prompt', 'google.k_shot_prompt',
        'google.zero_shot_cot_prompt', 'google.k_shot_cot_prompt'
    ):
        response = google.generate_text(e['text'], prompt_name)
        responses.append({
            'provider': 'google',
            'text': e['text'],
            'true_intent': e['intent'],
            'prompt_name': prompt_name,
            'response': response
        })

df_google = DataFrame(responses)
# df_google = pd.read_json(DATA_DIR / 'df_google', orient='records', lines=True)
print(df_google)
df_google.to_json(DATA_DIR / 'llm' / f'df_google_{timestamp_google}.jsonl', orient='records', lines=True)

# %%
# OpenAI

timestamp_openai = int(time.time())

openai = OpenAIClient(
    examples=dataset_examples,
    intents=intents,
)
responses = []
print('\nOpenAI text generating...')
for e in tqdm(test_examples):
    for prompt_name in (
        'openai.zero_shot_prompt', 'openai.k_shot_prompt',
        'openai.zero_shot_cot_prompt', 'openai.k_shot_cot_prompt'
    ):
        response = openai.generate_text(e['text'], prompt_name)
        responses.append({
            'provider': 'openai',
            'text': e['text'],
            'true_intent': e['intent'],
            'prompt_name': prompt_name,
            'response': response
        })

df_openai = DataFrame(responses)
# df_openai = pd.read_json(DATA_DIR / 'df_openai', orient='records', lines=True)
print(df_openai)
df_openai.to_json(DATA_DIR / 'llm' / f'df_openai_{timestamp_openai}.jsonl', orient='records', lines=True)

# %% [markdown]
#
# A few observations from examining the output in the the llm directory:
#   * extremely high accuracy among these few test examples
#   * very little variation in responses between the different models and prompts
#
# We'll carry out a more thorough evaluation in the next notebook.
