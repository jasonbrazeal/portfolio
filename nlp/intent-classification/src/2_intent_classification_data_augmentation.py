# %% [markdown]
# ## Data Augmentation
#
# One relatively fast way to generate new data for the 2 missing intents is to use an LLM.
#
# To ensure diversity in the generated utterances, we'll use a technique described [here](https://arxiv.org/abs/2209.11755) and [here](https://www.promptingguide.ai/applications/generating_textbooks).
#
# From promptingguide.ai:
# * Identify which parameters/entities might vary between different samples in your synthetic dataset
# * Generate or manually compile a collection of these entities to fill in the gaps
# * Produce the dataset by randomly selecting entities for insertion. It's best to set the generation temperature higher than the default but below the maximum
#
# We will instruct the LLM to use 3 different language styles to introduce some variation in the generated data. We create 3 examples for each style, so we have to come up with 9 examples for each of the 2 missing intents. We will have the LLM create 50 examples of each style and add these utterances to our dataset.
#
# The final 300 generated utterances are in `data_generated.jsonl`.


# %%
import os
import random

from pprint import pprint

import pandas as pd

from IPython.display import display
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pandas import DataFrame
from pydantic import BaseModel
from tqdm import tqdm

from utils import (
    init_nb, GENERATED_DATA_PATH, ALL_DATA_PATH, FILTERED_DATA_PATH
)
init_nb()


# %%
# LLM setup

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
# https://platform.openai.com/docs/models
# gpt-4o & gpt-4o-mini: 128K context, October 2023 knowledge cutoff, 16,384 max output tokens
# LLM_MODEL = 'gpt-4o-2024-08-06' # supports structured outputs
LLM_MODEL = 'gpt-4o-mini-2024-07-18'
# https://platform.openai.com/docs/api-reference/chat/create
# temperature ranges 0-2, default 1, we set it a bit higher to induce more variation
LLM_TEMPERATURE: float = 1.2
SYSTEM_PROMPT: str = 'you are an accomplished fiction writer, an expert at writing in different language styles'

client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)

# Note: in my testing, the gpt-4o-mini model seemed to perform better with the
# higher-than-default temperature setting. The gpt-4o model produced output
# that I found a bit too flowery, such as:
# "blessed machine of knowledge, mention today's verbiage"


# %%
# LLM helpers


# for structured output
class ResponseFormat(BaseModel):
    utterances: list[str]


def generate_text(prompt: str) -> list[str]:
    '''
    Generate LLM text from a prompt
    '''
    messages: list[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
    ]
    try:
        response = client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages = messages,
            temperature=LLM_TEMPERATURE,
            response_format=ResponseFormat,
        )
        output = response.choices[0].message.parsed
        return output.utterances
    except Exception as e:
        print(e)
        return []


def generate_utterances(intent, language_style_examples) -> list[str]:
    '''
    Generate utterances for an intent given a few examples in different styles
    '''
    utterances = []
    for language_style, examples in tqdm(language_style_examples.items()):
        # construct prompt
        prompt = f'''
        Task: Come up with 50 different ways that a user could ask an AI assistant about {intent}.
        The output should be lowercase without any final punctuation and should use {language_style}.

        Examples:
        '''
        prompt += '\n'.join([f'{i}. {example}' for i, example in enumerate(examples)])

        # generate text
        utterances += generate_text(prompt)

    return utterances


# %%
# use LLM to generate new data

# generate new utterances using data that I created for the missing intents
# we are printing out a sample of the generated utterances to make sure
# they are what we expect
language_style_examples_word_of_the_day = {
    'formal language': [
        "what is the word of the day, kind sir",
        "pardon me, madam, could you please tell me the word of the day today",
        "i will have today's word of the day now, please",
    ],
    'informal city slang': [
        "word of the day, bro, what it is",
        "yo what the word today",
        "lemme get dat word of the day",
    ],
    'standard English, not too formal or informal': [
        "what is the word of the day",
        "what's the word of the day today",
        "what's today's word of the day",
    ]
}
print('generating word_of_the_day utterances...')
generated_utterances_word_of_the_day: list[str] = generate_utterances(
    'the word of the day', language_style_examples_word_of_the_day
)
print('word_of_the_day sample:')
pprint(random.sample(generated_utterances_word_of_the_day, 10))

language_style_examples_food_beverage_recipe = {
    'formal language': [
        "please sir, do you know how to cook beef wellington",
        "pardon, could i please have the recipe for salisbury steak with mushroom gravy",
        "i would love to know how to make a fancy mojito cocktail, please",
    ],
    'informal city slang': [
        "hey, how can i make me some mac n cheese",
        "how do i cook these noodles bro",
        "recipe for some awesome tacos",
    ],
    'standard English, not too formal or informal': [
        "what is the recipe for a southern style turkey dressing",
        "could you tell me how to make a stawberry daiquiri",
        "what are the instructions to cook a perfect loaded baked potato",
    ]
}
print('generating food_beverage_recipe utterances...')
generated_utterances_food_beverage_recipe: list[str] = generate_utterances(
    'food and beverage recipes', language_style_examples_food_beverage_recipe
)
print('food_beverage_recipe sample:')
pprint(random.sample(generated_utterances_food_beverage_recipe, 10))

# %% [markdown]
# ```text
# generating word_of_the_day utterances...
# 100%|██████████| 3/3 [00:30<00:00, 10.07s/it]
#
# word_of_the_day sample:
# ['give me the scoop on the word',
#  'is there a specific word designated for today',
#  'got that daily word for me',
#  'whats today’s word highlight',
#  'that which is today’s word of the day, if you could',
#  'may i request the word of the day',
#  'what’s cookin in the word of the day',
#  "if it isn't too much trouble, what is the word of the day",
#  'can i peep the word of the day',
#  'whats the daily slang word']
#
# generating food_beverage_recipe utterances...
# 100%|██████████| 3/3 [00:28<00:00,  9.49s/it]
#
# food_beverage_recipe sample:
# ['what are the instructions for preparing ratatouille',
#  'can you provide instructions for making chili con carne',
#  'got a dope pizza recipe to share',
#  'how can i make homemade hummus',
#  "what's the best way to cook a filet mignon",
#  "show me how to make some dope s'mores",
#  'i am interested in learning how to prepare a classic coq au vin',
#  'could you elaborate on how to enjoy a perfect hot chocolate',
#  'what are the steps to whip up a fruit smoothie',
#  'can we chat about pancake toppings, my dude']
# ```

# %%
# produce complete dataset

# # aggregate the generated data into a dataframe
# all_generated_utterances: dict = {
#     'utterance': [*generated_utterances_word_of_the_day, *generated_utterances_food_beverage_recipe],
#     'intent_str': ['word_of_the_day'] * len(generated_utterances_word_of_the_day) + ['food_beverage_recipe'] * len(generated_utterances_food_beverage_recipe),
# }
# df_generated: DataFrame = DataFrame(all_generated_utterances)

# # save all generated data to a csv
# df_generated.to_json(GENERATED_DATA_PATH, orient='records', lines=True)

df_generated = pd.read_json(GENERATED_DATA_PATH, orient='records', lines=True)
# add generated data to existing (filtered) data

# read in filtered data
df = pd.read_json(FILTERED_DATA_PATH, orient='records', lines=True)

# first add new column to identify llm-generated text in both dataframes
df_generated['llm_generated'] = True
df['llm_generated'] = False

# concatenate generated utterances to existing dataframe and save to csv
df = pd.concat((df, df_generated))
df = df.reset_index(drop=True)
display(df)

# the llm-generated data and the CLINC-150 data contain both normal apostrophes (') and smart ones (’)
# it's not likely to matter, but let's standardize on normal apostrophes for more consistency
df['utterance'] = df['utterance'].apply(lambda x: x.replace("’", "'"))

df.to_json(ALL_DATA_PATH, orient='records', lines=True)

# %% [markdown]
# ```text
# utterance            intent_str  \
# 0                     how would you say fly in italian             translate
# 1                    what's the spanish word for pasta             translate
# 2                  how would they say butter in zambia             translate
# 3                       how do you say fast in spanish             translate
# 4                  what's the word for trees in norway             translate
# ...                                                ...                   ...
# 4503   how can i create an easy pasta primavera recipe  food_beverage_recipe
# 4504  can you tell me how to make homemade basil pesto  food_beverage_recipe
# 4505        what's the best way to cook a filet mignon  food_beverage_recipe
# 4506     how can i brighten up a simple vegetable soup  food_beverage_recipe
# 4507     what are the steps to make a perfect omelette  food_beverage_recipe

#       llm_generated
# 0             False
# 1             False
# 2             False
# 3             False
# 4             False
# ...             ...
# 4503           True
# 4504           True
# 4505           True
# 4506           True
# 4507           True

# [4508 rows x 3 columns]
# ```
