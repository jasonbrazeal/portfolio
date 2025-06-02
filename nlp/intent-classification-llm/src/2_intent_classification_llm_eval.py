# %% [markdown]
# ## Evaluation
#
# The plan is to:
#
# * evaluate the intent classification performance of LLMs using the test set from a [previous project](https://github.com/jasonbrazeal/portfolio/tree/master/nlp/intent-classification)
# * compare the intent classification performance of LLMs vs. a Logistic Regression model trained on human-labeled data
# * analyze the errors made by the models for the different classes and point out any relevant observations
#
# We will evaluate performance for all 3 models using these prompts, with and without a chain-of-thought block:
#
# * zero shot (no in-context examples)
# * 10-shot (10 examples chosen at random)
# * 30-shot (1 example for each intent class, chosen at random)
#
# These configurations should provide a sufficient basis for analysis. This project will not incorporate dynamic example selection (e.g., via RAG) based on user queries. For examples of RAG implementations, please see: [Document Q&A Chatbot](https://github.com/jasonbrazeal/portfolio/tree/master/rag/docs-chat) and [Tech Manual RAG](https://github.com/jasonbrazeal/portfolio/tree/master/rag/tech-manual-rag).

# %%
import time
import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from llm import OpenAIClient, AnthropicClient, GoogleClient
from utils import DATA_DIR, IMG_DIR, RANDOM_SEED, init_nb, plot_accuracy

init_nb()

# %%
# these dataset examples are from the Logistic Regression model's training set from the previous project
# they will serve as in-context learning examples for the LLMs
df_train = pd.read_json(DATA_DIR / 'df_train.jsonl', orient='records', lines=True)

# k=10 randomly selected
dataset_examples = df_train.sample(n=10, random_state=RANDOM_SEED).apply(
    lambda row: {'text': row['utterance'], 'intent': row['intent_str']},
    axis=1
).tolist()

# k=30 randomly selected but each intent represented
dataset_examples_balanced = []
for intent in sorted(df_train['intent_str'].unique()):
    # sample 1 example from each intent
    example = df_train[df_train['intent_str'] == intent].sample(n=1, random_state=RANDOM_SEED).iloc[0]
    dataset_examples_balanced.append({
        'text': example['utterance'],
        'intent': example['intent_str']
    })

df_test = pd.read_json(DATA_DIR / 'df_test.jsonl', orient='records', lines=True)
test_examples = [{'text': utterance, 'intent': intent} for intent, utterance in zip(df_test.intent_str, df_test.utterance)]

# load intent labels
with Path(DATA_DIR / 'intent_labels.json').open() as f:
    intent_labels = json.load(f)
    int_labels = intent_labels['label_int']
    str_labels = intent_labels['label_str']

intents: list[str] = list(str_labels.values())

# %%
# Anthropic

timestamp_anthropic = int(time.time())

anthropic = AnthropicClient(
    # examples=dataset_examples,
    examples=dataset_examples_balanced,
    intents=intents,
)
responses = []
for e in tqdm(test_examples):
    for prompt_name in (
        'anthropic.k_shot_prompt', 'anthropic.k_shot_cot_prompt'):
        # 'anthropic.zero_shot_prompt', 'anthropic.zero_shot_cot_prompt'):
        response = anthropic.generate_text(e['text'], prompt_name)
        if 'k_shot' in prompt_name:
            prompt_name += str(len(anthropic.examples))
        responses.append({
            'provider': 'anthropic',
            'text': e['text'],
            'true_intent': e['intent'],
            'prompt_name': prompt_name,
            'response': response
        })

df_anthropic = DataFrame(responses)
df_anthropic.to_json(DATA_DIR / 'llm' / f'df_eval_anthropic_{timestamp_anthropic}.jsonl', orient='records', lines=True)

# %%
# Google

timestamp_google = int(time.time())

google = GoogleClient(
    # examples=dataset_examples,
    examples=dataset_examples_balanced,
    intents=intents,
)
responses = []
for e in tqdm(test_examples):
    for prompt_name in (
        'google.k_shot_prompt', 'google.k_shot_cot_prompt'):
        # 'google.zero_shot_prompt', 'google.zero_shot_cot_prompt'):
        response = google.generate_text(e['text'], prompt_name)
        if 'k_shot' in prompt_name:
            prompt_name += str(len(google.examples))
        responses.append({
            'provider': 'google',
            'text': e['text'],
            'true_intent': e['intent'],
            'prompt_name': prompt_name,
            'response': response
        })

df_google = DataFrame(responses)
df_google.to_json(DATA_DIR / 'llm' / f'df_eval_google_{timestamp_google}.jsonl', orient='records', lines=True)

# %%
# OpenAI

timestamp_openai = int(time.time())

openai = OpenAIClient(
    # examples=dataset_examples,
    examples=dataset_examples_balanced,
    intents=intents,
)
responses = []
for e in tqdm(test_examples):
    for prompt_name in (
        'openai.k_shot_prompt', 'openai.k_shot_cot_prompt'):
        # 'openai.zero_shot_prompt', 'openai.zero_shot_cot_prompt'):
        response = openai.generate_text(e['text'], prompt_name)
        if 'k_shot' in prompt_name:
            prompt_name += str(len(openai.examples))
        responses.append({
            'provider': 'openai',
            'text': e['text'],
            'true_intent': e['intent'],
            'prompt_name': prompt_name,
            'response': response
        })

df_openai = DataFrame(responses)
df_openai.to_json(DATA_DIR / 'llm' / f'df_eval_openai_{timestamp_openai}.jsonl', orient='records', lines=True)

# %%
# Gather all eval data

# combine all df_eval files into one dataframe
dfs = []
eval_files = list((DATA_DIR / 'llm').glob('df_eval_*.jsonl'))
for f in eval_files:
    dfs.append(pd.read_json(f, orient='records', lines=True))
df_eval = pd.concat(dfs).reset_index(drop=True)

# convert string labels to integers for evaluation
label_to_int = {v: int(k) for k, v in str_labels.items()}
df_eval['true_intent_int'] = df_eval['true_intent'].map(label_to_int)
# map responses to integers, filling unknown values with -1 to avoid NaN
df_eval['prediction_int'] = df_eval['response'].map(lambda x: label_to_int.get(x, -1))


# %%
# Derive and plot accuracy data

PROVIDERS_FORMATTED = {
    'anthropic': 'Anthropic',
    'google': 'Google',
    'openai': 'OpenAI',
}
accuracy_data = {}
# evaluate each provider and prompt type separately
for provider in sorted(df_eval['provider'].unique()):
    for prompt_name in sorted(df_eval[df_eval['provider'] == provider]['prompt_name'].unique()):

        mask = (df_eval['provider'] == provider) & (df_eval['prompt_name'] == prompt_name)
        df_subset = df_eval[mask]

        y_test = df_subset['true_intent_int'].to_numpy()
        y_pred = df_subset['prediction_int'].to_numpy()

        report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            labels=list(int_labels.values()),
            target_names=list(str_labels.values()),
            digits=4,
            zero_division=0,
        )
        accuracy = accuracy_score(y_test, y_pred)
        provider_formatted = PROVIDERS_FORMATTED[provider]
        if provider_formatted not in accuracy_data:
            accuracy_data[provider_formatted] = {}
        accuracy_data[provider_formatted][prompt_name.split('.')[-1]] = accuracy

        # formatted for the next notebook, full results are there
        print('# %% [markdown]')
        print(f'# ### {prompt_name}')
        print(f'# accuracy = {accuracy:.4f}')
        print('# ```text')
        print()
        print(report)
        print()
        print('# ```')


def format_floats(items: dict):
    for k, v in items.items():
        if type(v) is float:
            items[k] = float(f'{v:.4f}')
        elif type(v) is dict:
            format_floats(v)
    return items


def sort_by_accuracy(items: dict):
    for k, v in items.items():
        v = dict(sorted(v.items(), key=lambda x: x[1]))
        items[k] = v
    return items


accuracy_data = format_floats(accuracy_data)
accuracy_data = sort_by_accuracy(accuracy_data)

print('*'*88)
print(json.dumps(accuracy_data, indent=2))

plot_accuracy(accuracy_data, IMG_DIR / 'accuracy_all.png')
plot_accuracy(accuracy_data, IMG_DIR / 'accuracy.png', split_charts=True)

# %% [markdown]
# ```python
# {
#   "Anthropic": {
#     "zero_shot_prompt": 0.9302,
#     "zero_shot_cot_prompt": 0.9313,
#     "k_shot_prompt10": 0.9357,
#     "k_shot_cot_prompt10": 0.9379,
#     "k_shot_cot_prompt30": 0.9812,
#     "k_shot_prompt30": 0.9845
#   },
#   "Google": {
#     "zero_shot_cot_prompt": 0.9124,
#     "zero_shot_prompt": 0.9335,
#     "k_shot_prompt10": 0.9424,
#     "k_shot_cot_prompt10": 0.9468,
#     "k_shot_cot_prompt30": 0.9845,
#     "k_shot_prompt30": 0.9889
#   },
#   "OpenAI": {
#     "zero_shot_cot_prompt": 0.9202,
#     "zero_shot_prompt": 0.9213,
#     "k_shot_prompt10": 0.9302,
#     "k_shot_cot_prompt10": 0.9346,
#     "k_shot_prompt30": 0.9845,
#     "k_shot_cot_prompt30": 0.9856
#   }
# }
# ```
#
# ![Anthropic](../img/accuracy_anthropic.png)
#
# ![Google](../img/accuracy_google.png)
#
# ![OpenAI](../img/accuracy_openai.png)
#
# %% [markdown]
# ## Discussion
#
# ### Shot count
# For all providers, increasing the number of examples in the prompt leads to a substantial improvement in accuracy. The 30-shot prompts consistently outperform the 10-shot prompts, and both outperform zero-shot prompts. The accuracy gain between zero-shot and 30-shot prompts is around 5-7% for our data. Only 30-shot prompts yielded an accuracy higher than our [Logistic Regression classifier](https://github.com/jasonbrazeal/portfolio/tree/master/nlp/intent-classification) trained on human-labeled data, which achieved an accuracy of 0.9523.
#
# ### Chain-of-thought prompting
# Adding the COT block had less of an effect than I was expecting. It sometimes adds a modest boost in perforance, but never more than a few tenths of a percent. And sometimes it lowers accuracy compared to the same prompt without the COT block. For example, Google and OpenAI perform slightly worse on zero-shot prompts with COT versus without COT. But for 10-shot prompts, Google and OpenAI see a small increase in performance when using COT.
# I suspect that the newest models we are using are not seeing as many improvements with COT as older models may have because they have the capability of "thinking" built in already. They are already trained to go through the step-by-step reasoning process that COT is meant to elicit. Some models even let you adjust the number of tokens the model can use to think.
# For Gemini 2.5 Flash, Google lets you control the amount of [thinking](https://cloud.google.com/vertex-ai/generative-ai/docs/thinking) you will permit the model to do or turn it off altogether. By default, and also our settings for this project, the model thinks up to a maximum of 8,192 tokens.
# Anthropic has an option called [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) which, like Google, allows you to set a maximum token limit for thinking or turn it off.
#
# ### Provider comparison
#
# Google achieves the highest single accuracy (98.89%) with the 30-shot prompt (no COT). But Anthropic and OpenAI were very close, with accuracies less than half a percent away from Google's. The trends across the three providers are similar:
#
# * all performed best when they saw an example of every class
# * none achieved the performance of the previously trained Logistic Regression model (red dotted line) until they had examples from every class
# * all performed decently well (>91%) without seeing ANY examples of the classes
#
# ### Intent comparison
#
# Most classes achieve near-perfect precision and recall for the higher-shot prompts, but there are some clear patterns when we consider the other prompts. Several intents were challenging in zero-shot and 10-shot settings, such as "maybe", "reminder", and "reminder_update". The "reminder_update" class is particularly problematic, with F1 scores of 0.0 for all providers except for the 30-shot prompts. Taking a look at the data, it look like the distinction between "reminder" and "reminder_update" is fine-grained:
#
# * reminder - get reminders, i.e. read my reminders to me.
# * reminder_update - set a reminder, i.e. "remind me to..."
#
# All of the models tended to classify the "reminder_update" utterances as "reminder" until they saw explicit examples of both. This is quite understandable, but points out a limitation of using LLMS in this way: they need to see the right data to make correct predictions.
#
# ## Summary
#
# In this project, we compared the intent classification performance of three leading LLM providers (Anthropic, Google, OpenAI) using a variety of prompting strategies. All three models performed well, with accuracy improving as more in-context examples were provided. 30-shot prompts, which included an example for every intent class, enabled LLMs to exceed the performance of a Logistic Regression classifier previously trained on human-labeled data. Chain-of-thought prompting provided only marginal gains.
# Overall, LLMs are highly capable for intent classification, but careful prompt engineering and sufficient in-context examples are essential.
