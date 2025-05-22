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
# ## Linguistic Analysis
#
# The plan is to:
# * calculate some basic statistics for our text data to get a sense of its scale and structure
# * examine relevant linguistic properties of the data such as most common tokens/bigrams/trigrams and stopwords
# * perform POS tagging to better understand the syntactic structure of the data
# * take a look at entities in the data using NER
# * classify the sentiment of each utterance


# %%
from collections import Counter
from math import ceil
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from IPython.display import display
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from pandas import DataFrame, Series
from utils import ALL_DATA_PATH, PROCESSED_DATA_PATH, init_nb, Timer, format_counter, nlp

init_nb()

nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

pd.options.display.max_colwidth = 100
pd.options.display.precision = 3

# %%
# tokenization and linguistic features

# since our text has already been partially processed, we don't need to worry
# about formatting or removing special characters here, just tokenizing and
# capturing other relevant linguistic properties

df = pd.read_json(ALL_DATA_PATH, orient='records', lines=True)


def get_sentiment(text):
    '''
    Vader returns sentiment scores in the format:
    {'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}
    This function returns only the highest score pos/neg/neu and ignores compound
    '''
    scores = vader.polarity_scores(text)
    del scores['compound']
    return sorted(list(scores.items()), key=lambda x: x[1])[-1][0]


def tokenize_and_preprocess(row: Series) -> Series:
    '''
    Preprocesses each row of data, calculating some basic text statistics
    and adding these new columns:
    tokens, bigrams, trigrams, lemmas, pos, ents, num_sents, num_stopwords
    '''
    doc = nlp(row.utterance)

    tokens: list[str] = []
    lemmas: list[str] = []
    pos: list[str] = []
    stopwords: list[str] = []
    num_stopwords = 0
    for token in doc:
        tokens.append(token.text)
        lemmas.append(token.lemma_)
        pos.append(token.pos_)
        if token.is_stop:
            stopwords.append(token.text)
            num_stopwords += 1

    ents: list[str] = []
    for ent in doc.ents:
        ents.append(ent.label_)

    sentiment: str = get_sentiment(row.utterance)

    if doc.has_annotation('SENT_START'):
        num_sents: int = len(list(doc.sents))
    else:
        num_sents: int = 1

    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    row['tokens'] = tokens
    row['bigrams'] = bigrams
    row['trigrams'] = trigrams
    row['lemmas'] = lemmas
    row['pos'] = pos
    row['ents'] = ents
    row['sentiment'] = sentiment
    row['stopwords'] = stopwords
    row['num_stopwords'] = num_stopwords
    row['num_sents'] = num_sents

    return row

with Timer('tokenization and preprocessing'):
    df = df.progress_apply(tokenize_and_preprocess, axis=1)
display(df)

df.to_json(PROCESSED_DATA_PATH, orient='records', lines=True)

# %% [markdown]
# ```text
# 100%|██████████| 4508/4508 [03:02<00:00, 24.71it/s]
# tokenization and preprocessing: 182.42612187500345 seconds
#
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

#       llm_generated  \
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

#                                                            tokens  \
# 0                        [how, would, you, say, fly, in, italian]
# 1                      [what, 's, the, spanish, word, for, pasta]
# 2                     [how, would, they, say, butter, in, zambia]
# 3                          [how, do, you, say, fast, in, spanish]
# 4                   [what, 's, the, word, for, trees, in, norway]
# ...                                                           ...
# 4503    [how, can, i, create, an, easy, pasta, primavera, recipe]
# 4504  [can, you, tell, me, how, to, make, homemade, basil, pesto]
# 4505       [what, 's, the, best, way, to, cook, a, filet, mignon]
# 4506      [how, can, i, brighten, up, a, simple, vegetable, soup]
# 4507      [what, are, the, steps, to, make, a, perfect, omelette]

#                                                                                                   bigrams  \
# 0                          [(how, would), (would, you), (you, say), (say, fly), (fly, in), (in, italian)]
# 1                     [(what, 's), ('s, the), (the, spanish), (spanish, word), (word, for), (for, pasta)]
# 2                   [(how, would), (would, they), (they, say), (say, butter), (butter, in), (in, zambia)]
# 3                              [(how, do), (do, you), (you, say), (say, fast), (fast, in), (in, spanish)]
# 4              [(what, 's), ('s, the), (the, word), (word, for), (for, trees), (trees, in), (in, norway)]
# ...                                                                                                   ...
# 4503  [(how, can), (can, i), (i, create), (create, an), (an, easy), (easy, pasta), (pasta, primavera),...
# 4504  [(can, you), (you, tell), (tell, me), (me, how), (how, to), (to, make), (make, homemade), (homem...
# 4505  [(what, 's), ('s, the), (the, best), (best, way), (way, to), (to, cook), (cook, a), (a, filet), ...
# 4506  [(how, can), (can, i), (i, brighten), (brighten, up), (up, a), (a, simple), (simple, vegetable),...
# 4507  [(what, are), (are, the), (the, steps), (steps, to), (to, make), (make, a), (a, perfect), (perfe...

#                                                                                                  trigrams  \
# 0             [(how, would, you), (would, you, say), (you, say, fly), (say, fly, in), (fly, in, italian)]
# 1     [(what, 's, the), ('s, the, spanish), (the, spanish, word), (spanish, word, for), (word, for, pa...
# 2     [(how, would, they), (would, they, say), (they, say, butter), (say, butter, in), (butter, in, za...
# 3                [(how, do, you), (do, you, say), (you, say, fast), (say, fast, in), (fast, in, spanish)]
# 4     [(what, 's, the), ('s, the, word), (the, word, for), (word, for, trees), (for, trees, in), (tree...
# ...                                                                                                   ...
# 4503  [(how, can, i), (can, i, create), (i, create, an), (create, an, easy), (an, easy, pasta), (easy,...
# 4504  [(can, you, tell), (you, tell, me), (tell, me, how), (me, how, to), (how, to, make), (to, make, ...
# 4505  [(what, 's, the), ('s, the, best), (the, best, way), (best, way, to), (way, to, cook), (to, cook...
# 4506  [(how, can, i), (can, i, brighten), (i, brighten, up), (brighten, up, a), (up, a, simple), (a, s...
# 4507  [(what, are, the), (are, the, steps), (the, steps, to), (steps, to, make), (to, make, a), (make,...

#                                                           lemmas  \
# 0                       [how, would, you, say, fly, in, italian]
# 1                     [what, be, the, spanish, word, for, pasta]
# 2                    [how, would, they, say, butter, in, zambia]
# 3                         [how, do, you, say, fast, in, spanish]
# 4                   [what, be, the, word, for, tree, in, norway]
# ...                                                          ...
# 4503   [how, can, I, create, an, easy, pasta, primavera, recipe]
# 4504  [can, you, tell, I, how, to, make, homemade, basil, pesto]
# 4505      [what, be, the, good, way, to, cook, a, filet, mignon]
# 4506     [how, can, I, brighten, up, a, simple, vegetable, soup]
# 4507       [what, be, the, step, to, make, a, perfect, omelette]

#                                                               pos        ents  \
# 0                      [SCONJ, AUX, PRON, VERB, VERB, ADP, PROPN]       [GPE]
# 1                          [PRON, AUX, DET, ADJ, NOUN, ADP, NOUN]      [NORP]
# 2                      [SCONJ, AUX, PRON, VERB, NOUN, ADP, PROPN]       [GPE]
# 3                       [SCONJ, AUX, PRON, VERB, ADV, ADP, PROPN]  [LANGUAGE]
# 4                   [PRON, AUX, DET, NOUN, ADP, NOUN, ADP, PROPN]       [GPE]
# ...                                                           ...         ...
# 4503            [SCONJ, AUX, PRON, VERB, DET, ADJ, NOUN, X, NOUN]          []
# 4504  [AUX, PRON, VERB, PRON, SCONJ, PART, VERB, ADJ, NOUN, NOUN]          []
# 4505     [PRON, AUX, DET, ADJ, NOUN, PART, VERB, DET, NOUN, NOUN]          []
# 4506          [SCONJ, AUX, PRON, VERB, ADP, DET, ADJ, NOUN, NOUN]          []
# 4507           [PRON, AUX, DET, NOUN, PART, VERB, DET, ADJ, NOUN]          []

#      sentiment                      stopwords  num_stopwords  num_sents
# 0          neu     [how, would, you, say, in]              5          1
# 1          neu           [what, 's, the, for]              4          1
# 2          neu    [how, would, they, say, in]              5          1
# 3          neu        [how, do, you, say, in]              5          1
# 4          neu       [what, 's, the, for, in]              5          1
# ...        ...                            ...            ...        ...
# 4503       neu              [how, can, i, an]              4          1
# 4504       neu  [can, you, me, how, to, make]              6          1
# 4505       neu         [what, 's, the, to, a]              5          1
# 4506       neu           [how, can, i, up, a]              5          1
# 4507       neu  [what, are, the, to, make, a]              6          1

# [4508 rows x 13 columns]
# ```

# %%
# corpus exploration - sentences, characters, tokens

# df = pd.read_json(PROCESSED_DATA_PATH, orient='records', lines=True)

def describe_helper(series: Series):
    '''
    inspired by: https://stackoverflow.com/questions/56029140/how-to-display-summary-statistics-next-to-a-plot-using-matplotlib-or-seaborn
    '''
    splits = series.describe()[1:].to_string().split()
    keys, values = "", ""
    for i in range(0, len(splits), 2):
        keys += "{:8}\n".format(splits[i])
        values += "{:>8}\n".format(splits[i+1])
    return keys, values


def hist(series: Series, name: str, x_label: str = '', y_label: str = '', bin_edges = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#22252A')
    if bin_edges is None:
        num_bins = 10
        bin_edges = range(0, ceil(series.max()), ceil(series.max()) // num_bins)
    series.plot(kind='hist', ax=ax, density=True, alpha=0.75, ylabel='', color='skyblue', ec='skyblue', xlim=(0, series.max()), bins=bin_edges, xticks=bin_edges)
    series.plot(kind='kde', ax=ax, color='lightcoral')
    ax.set_title(name, size=17, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for a, spine in ax.spines.items():
        spine.set_visible(False)
    plt.figtext(.69, .6, describe_helper(series)[0], {'multialignment':'left'})
    plt.figtext(.79, .6, describe_helper(series)[1], {'multialignment':'right'})
    if matplotlib.get_backend() == 'inline':
        print('showing image...')
        plt.show()
    else:
        print('saving image...')
        filename = name.lower().replace(' ', '_')
        filepath = f'../img/{filename}.png'
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, transparent=True, format='png')
        print(f'{name.lower()} image created: {filepath}')

num_rows, num_columns = df.shape

total_num_sents = df.num_sents.sum()
avg_sents_per_utterance = total_num_sents / num_rows
print(f'{total_num_sents=}')
print(f'{avg_sents_per_utterance=}')
# => len(df) is 4508, so all the utterances are single sentence except 1:
# 'do you know what calumny means  please look it up for me'

total_num_chars = df.utterance.apply(len).sum()
print(f'{total_num_chars=}')
hist(df.utterance.apply(len), 'Utterance Length', 'Length in Chars')

all_tokens = df.tokens.sum()
total_num_tokens = len(all_tokens)
print(f'{total_num_tokens=}')

unique_tokens = set(all_tokens)
total_unique_tokens = len(unique_tokens)
print(f'{total_unique_tokens=}')

hist(Series(sorted(unique_tokens)).apply(len), 'Individual Token Length', 'Length in Chars')
hist(df.tokens.apply(len), 'Tokens per Utterance', 'Num Tokens')

# -> The distribution of lengths of individual tokens and utterances (whether
# measured in chars or number of tokens) seems like what we would expect.
# Most of the lengths cluster around the mean, but the histograms have long tails
# on the right, which is common with language data. This captures the presence
# of a few elements that are quite a bit longer than most others.

# %% [markdown]
# ```text
# total_num_sents=4509
# avg_sents_per_utterance=1.0002218278615793
# total_num_chars=147744
# ```
#
# ![utterance length](../img/utterance_length.png)
#
# ```text
# total_num_tokens=32945
# total_unique_tokens=2399
# ```
#
# ![individual token length](../img/individual_token_length.png)
#
# ![tokens per utterance](../img/tokens_per_utterance.png)
#

# %%
# corpus exploration - stopwords

df = pd.read_json(PROCESSED_DATA_PATH, orient='records', lines=True)

print('top 10 most common stopwords overall:')
pprint(Counter(df.stopwords.sum()).most_common(10))

print('top 3 most common stopwords by intent:')
stops_by_intent = df.groupby('intent_str').stopwords.sum()
stops_by_intent = stops_by_intent.apply(lambda x: Counter(x).most_common(3))
print(stops_by_intent.apply(format_counter).to_string(header=False))

hist(df.num_stopwords, 'Stopwords per Utterance', 'Num Stopwords')
# -> The histogram has a long tail on the right, but the max is 18 stopwords
# in a single utterance. This seems a bit high. Most utterances
# 12 or fewer stopwords. But there are a few outliers like:
# "will i be able to get to the mall at 5:00, or will there be a lot of traffic" (14 stopwords)
# "i'm not sure if watermelon is on my shopping list, but if it isn't can you put it on there" (18 stopwords)

# Spacy's stopwords list includes many tokens, including some I wouldn't
# consider traditional stopwords, at least for this task, e.g. call, make, show, see.
# Libraries differ on what tokens to include in their English stopwords list, and
# the reality is that it is often task-specific. We will not remove stopwords from our
# data since it consists of mostly short sentences or phrases where stopwords may
# contribute to the model learning to distinguish between the intent classes.

# %% [markdown]
# ```text
# top 10 most common stopwords overall:
#
# [('the', 1406),
#  ('to', 1290),
#  ('you', 1162),
#  ('i', 1146),
#  ('what', 1144),
#  ('is', 924),
#  ('my', 849),
#  ('a', 833),
#  ('me', 825),
#  ('do', 565)]
#
# top 3 most common stopwords by intent:
#
# are_you_a_bot             you (152), a (139), are (121)
# calculator                what (108), is (103), of (43)
# date                      what (102), the (74), is (62)
# definition                what (106), the (75), of (60)
# find_phone                    my (150), i (63), me (51)
# flip_coin                     a (118), i (72), you (38)
# food_beverage_recipe           a (95), for (59), i (55)
# goodbye                     you (63), to (54), was (33)
# greeting                   how (73), you (69), are (48)
# maybe                        i (88), n't (34), not (33)
# meaning_of_life           the (101), of (94), what (82)
# no                          that (96), is (63), no (54)
# reminder                   my (105), to (75), what (68)
# reminder_update               to (135), a (98), me (73)
# shopping_list              my (113), on (91), what (65)
# shopping_list_update         my (124), on (78), to (70)
# spelling                    how (99), to (61), you (47)
# tell_joke                     me (80), a (69), you (54)
# text                            to (79), i (70), a (67)
# time                    what (116), the (104), is (102)
# timer                         a (103), for (68), i (33)
# todo_list                  my (138), to (131), do (109)
# todo_list_update            to (148), my (134), do (98)
# traffic                    the (192), to (104), on (86)
# translate                    i (99), say (98), how (97)
# weather                   the (103), what (79), is (62)
# what_is_your_name       you (111), name (94), what (86)
# who_made_you             you (108), who (100), the (39)
# word_of_the_day          the (209), of (108), what (71)
# yes                         that (74), is (45), 's (33)
# ```
#
# ![stopwords per utterance](../img/stopwords_per_utterance.png)
#

# %%
# corpus exploration - n-grams

def line(data: list[tuple], name: str, x_label: str = '', y_label: str = ''):
    def _format(x: str | tuple) -> str:
        return ' '.join(x) if type(x) is tuple else x

    df = DataFrame({'x':[_format(row[0]) for row in data], 'y':[row[1] for row in data]})
    # df = df.set_index('x')
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#22252A')
    df.plot(kind='line', ax=ax, alpha=0.75, xlabel=x_label, ylabel=y_label, xticks=df.index, rot=30, style='.-', color='lightgreen', legend=False)
    ax.set_title(name, size=17, pad=10)
    ax.set_xticklabels(df.x)
    # only show every other y label
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    if matplotlib.get_backend() == 'inline':
        plt.show()
    else:
        filename = name.lower().replace(' ', '_')
        filepath = f'../img/{filename}.png'
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, transparent=True, format='png')
        print(f'{name.lower()} image created: {filepath}')


most_common_unigrams = Counter(all_tokens).most_common(10)
print('most common unigrams:')
pprint(most_common_unigrams)
line(most_common_unigrams, name='Most Common Unigrams', y_label='Num Occurrences')

all_bigrams = df.bigrams.sum()
all_bigrams = map(tuple, all_bigrams)
most_common_bigrams = Counter(all_bigrams).most_common(10)
print('most common bigrams:')
pprint(most_common_bigrams)
line(most_common_bigrams, name='Most Common Bigrams', y_label='Num Occurrences')

all_trigrams = df.trigrams.sum()
all_trigrams = map(tuple, all_trigrams)
most_common_trigrams = Counter(all_trigrams).most_common(10)
print('most common trigrams:')
pprint(most_common_trigrams)
line(most_common_trigrams, name='Most Common Trigrams', y_label='Num Occurrences')

# -> The graphs don't show any major surprises, but they do show that several of the
# top bigrams and trigrams come from the shopping list and todo list intents.
# From the trigrams line graph, it is clear that the most common trigram,
# "my shopping list", occurs almost twice as frequently as trigrams 5-10.

print('top 3 most common unigrams by intent:')
all_tokens_by_intent = df.groupby('intent_str').tokens.sum()
all_tokens_by_intent = all_tokens_by_intent.apply(lambda x: Counter(x).most_common(3))
print(all_tokens_by_intent.apply(format_counter).to_string(header=False))

print('top 3 most common bigrams by intent:')
all_bigrams_by_intent = df.groupby('intent_str').bigrams.sum()
# sum produces a list of lists, transform to list of tuples
all_bigrams_by_intent = all_bigrams_by_intent.apply(lambda x: [tuple(b) for b in x])
all_bigrams_by_intent = all_bigrams_by_intent.apply(lambda x: Counter(x).most_common(3))
print(all_bigrams_by_intent.apply(format_counter).to_string(header=False))

print('top 3 most common trigrams by intent:')
all_trigrams_by_intent = df.groupby('intent_str').trigrams.sum()
all_trigrams_by_intent = all_trigrams_by_intent.apply(lambda x: [tuple(t) for t in x])
all_trigrams_by_intent = all_trigrams_by_intent.apply(lambda x: Counter(x).most_common(3))
print(all_trigrams_by_intent.apply(format_counter).to_string(header=False))

# -> The common trigrams list confirms what we saw in the graphs, high counts for the
# shopping and todo list intents. For each of the 2 shopping list intents,
# "my shopping list" occurs about 100 times each, out of 150 utterances. There is less
# variation around how people talk about their shopping list than most of the other
# intents. This is fine and may in fact be a representative sample. We will want to
# confirm that our final model performs well on utterances from the shopping list intent
# that don't mention "my shopping list". A similar case is "word_of_the_day". About
# 2/3 of the examples have the trigrams "of the day" and "word of the". We should
# be sure our model works on examples that have a different way of asking for the
# "word of the day".

# %% [markdown]
# ```text
# most common unigrams:
#
# [('the', 1406),
#  ('to', 1290),
#  ('you', 1162),
#  ('i', 1146),
#  ('what', 1144),
#  ('is', 924),
#  ('my', 849),
#  ('a', 833),
#  ('me', 825),
#  ('list', 665)]
# ```
#
# ![most common unigrams](../img/most_common_unigrams.png)
#
# ```text
# most common bigrams:
#
# [(('tell', 'me'), 350),
#  (('on', 'my'), 331),
#  (('what', 'is'), 323),
#  (('can', 'you'), 290),
#  (('is', 'the'), 235),
#  (('shopping', 'list'), 235),
#  (('what', "'s"), 225),
#  (('my', 'shopping'), 202),
#  (('i', 'need'), 191),
#  (('to', 'do'), 177)]
# ```
#
# ![most common bigrams](../img/most_common_bigrams.png)
#
# ```text
# most common trigrams:
#
# [(('my', 'shopping', 'list'), 200),
#  (('what', 'is', 'the'), 182),
#  (('what', "'s", 'the'), 135),
#  (('to', 'do', 'list'), 119),
#  (('you', 'tell', 'me'), 111),
#  (('i', 'need', 'to'), 108),
#  (('my', 'to', 'do'), 108),
#  (('on', 'my', 'shopping'), 105),
#  (('of', 'the', 'day'), 100),
#  (('word', 'of', 'the'), 98)]
# ```
#
# ![most common trigrams](../img/most_common_trigrams.png)
#
# ```text
# top 3 most common unigrams by intent:
#
# are_you_a_bot                  you (152), a (139), are (121)
# calculator                     what (108), is (103), of (43)
# date                        date (109), what (102), the (74)
# definition                     what (106), the (75), of (60)
# find_phone                     my (150), phone (125), i (63)
# flip_coin                    coin (137), flip (125), a (118)
# food_beverage_recipe                a (95), for (59), i (55)
# goodbye                          you (63), to (54), was (33)
# greeting                        how (73), you (69), are (48)
# maybe                             i (88), n't (34), not (33)
# meaning_of_life         life (125), meaning (102), the (101)
# no                               that (96), is (63), no (54)
# reminder                        my (105), list (92), to (75)
# reminder_update              to (135), a (98), reminder (96)
# shopping_list           list (140), shopping (130), my (113)
# shopping_list_update    list (145), shopping (126), my (124)
# spelling                      spell (101), how (99), to (61)
# tell_joke                         me (80), a (69), joke (69)
# text                             text (136), to (79), i (70)
# time                       time (155), what (116), the (104)
# timer                         timer (133), a (103), set (94)
# todo_list                     list (139), my (138), to (131)
# todo_list_update              list (148), to (148), my (134)
# traffic                   the (192), traffic (144), to (104)
# translate                         i (99), say (98), how (97)
# weather                   the (103), weather (88), what (79)
# what_is_your_name            you (111), name (94), what (86)
# who_made_you                  you (108), who (100), the (39)
# word_of_the_day              the (209), word (154), of (108)
# yes                             that (74), yes (49), is (45)
#
# top 3 most common bigrams by intent:
#
# are_you_a_bot                       are you (77), you a (40), you are (35)
# calculator                     what is (90), is the (26), square root (20)
# date                             the date (50), 's date (32), tell me (31)
# definition                what is (35), what does (31), definition of (31)
# find_phone                      my phone (115), find my (36), help me (32)
# flip_coin                        a coin (102), flip a (71), coin flip (27)
# food_beverage_recipe            recipe for (24), how to (23), what 's (20)
# goodbye                            it was (28), to you (20), with you (20)
# greeting                        are you (32), how are (28), you doing (15)
# maybe                                    i 'm (23), i do (22), do n't (20)
# meaning_of_life            of life (80), meaning of (71), the meaning (69)
# no                                   that is (45), that 's (29), no , (20)
# reminder                  on my (49), reminder list (41), my reminder (37)
# reminder_update              a reminder (86), me to (46), reminder to (43)
# shopping_list           shopping list (123), my shopping (101), on my (72)
# shopping_list_update    shopping list (112), my shopping (101), on my (55)
# spelling                        to spell (46), how to (42), you spell (32)
# tell_joke                             tell me (62), a joke (45), me a (40)
# text                               a text (53), and tell (43), send a (33)
# time                              what time (65), is it (58), time is (54)
# timer                             a timer (70), set a (58), timer for (55)
# todo_list                             on my (81), do list (70), my to (66)
# todo_list_update                      to do (94), do list (71), my to (64)
# traffic                         the traffic (73), on the (58), way to (48)
# translate                          how do (42), i say (41), how would (26)
# weather                        the weather (62), is the (34), what is (29)
# what_is_your_name               your name (46), call you (30), do you (30)
# who_made_you               who made (27), made you (22), what company (19)
# word_of_the_day                   of the (101), word of (99), the day (99)
# yes                                that 's (28), that is (26), , that (25)
#
# top 3 most common trigrams by intent:
#
# are_you_a_bot                            are you a (39), you are a (27), a real person (25)
# calculator                      what is the (26), square root of (20), the square root (19)
# date                              today 's date (21), days from now (16), tell me what (15)
# definition                    the meaning of (27), the definition of (27), what is the (25)
# find_phone                      find my phone (22), locate my phone (19), help me find (18)
# flip_coin                                 flip a coin (63), a coin flip (17), a coin , (16)
# food_beverage_recipe                    a recipe for (15), what 's the (13), how can i (12)
# goodbye                                   talk to you (12), it was nice (8), to talk to (6)
# greeting                               how are you (20), are you doing (14), , how are (11)
# maybe                                        i do n't (15), do n't know (12), i 'm not (12)
# meaning_of_life                 the meaning of (66), meaning of life (66), what is the (18)
# no                                         no , that (17), that is not (12), , that is (11)
# reminder                     my reminder list (37), my list of (27), list of reminders (23)
# reminder_update                  a reminder to (38), set a reminder (37), remind me to (30)
# shopping_list           my shopping list (101), on my shopping (62), the shopping list (16)
# shopping_list_update        my shopping list (99), on my shopping (43), to my shopping (30)
# spelling                              how to spell (40), how do you (24), do you spell (24)
# tell_joke                                 tell me a (36), me a joke (29), a joke about (23)
# text                                    send a text (30), a text to (19), and tell him (18)
# time                                      what time is (51), time is it (50), is it in (33)
# timer                                 a timer for (39), set a timer (37), timer for me (19)
# todo_list                                 my todo list (51), to do list (47), my to do (43)
# todo_list_update                              to do list (71), my to do (63), on my to (30)
# traffic                               on the way (40), the way to (38), is the traffic (35)
# translate                                would i say (22), if i were (22), how would i (21)
# weather                             what is the (26), is the weather (24), what 's the (18)
# what_is_your_name                      i call you (12), refer to you (10), what name do (9)
# who_made_you                   who made you (19), who programmed you (10), tell me who (10)
# word_of_the_day                         of the day (99), word of the (98), the word of (71)
# yes                                         yes , that (17), , that 's (14), , that is (10)
# ```

# %%
# corpus exploration - parts of speech

all_pos = df.pos.sum()
most_common_pos = Counter(all_pos).most_common(10)
print('most common parts of speech:')
pprint(most_common_pos)

print('most common parts of speech by intent:')
all_pos_by_intent = df.groupby('intent_str').pos.sum()
all_pos_by_intent = all_pos_by_intent.apply(lambda x: Counter(x).most_common(5))
print(all_pos_by_intent.apply(format_counter).to_string(header=False))

# -> This looks good for speech data for a personal assisant. If there
# were anomalies such as large amount of SYM or NUM parts of speech,
# that would warrant further investigation.

# %% [markdown]
# ```text
# most common parts of speech:
#
# [('NOUN', 6388),
#  ('PRON', 6219),
#  ('VERB', 4797),
#  ('AUX', 3518),
#  ('ADP', 2870),
#  ('DET', 2632),
#  ('PART', 1230),
#  ('ADJ', 1109),
#  ('PROPN', 726),
#  ('SCONJ', 721)]
#
# most common parts of speech by intent:
#
# are_you_a_bot               PRON (221), AUX (190), NOUN (188), DET (165), ADJ (90)
# calculator                  NUM (274), PRON (169), AUX (139), NOUN (103), ADP (94)
# date                       NOUN (275), PRON (185), AUX (162), DET (108), VERB (81)
# definition                 NOUN (188), PRON (170), VERB (140), AUX (104), DET (85)
# find_phone               PRON (325), VERB (226), NOUN (175), AUX (112), SCONJ (42)
# flip_coin                  NOUN (267), VERB (246), PRON (173), DET (139), AUX (45)
# food_beverage_recipe      NOUN (348), VERB (218), PRON (166), DET (162), AUX (121)
# goodbye                      PRON (143), VERB (116), ADP (63), INTJ (63), ADV (61)
# greeting                   PRON (125), AUX (111), VERB (80), SCONJ (76), INTJ (64)
# maybe                        PRON (166), AUX (146), VERB (81), PART (80), DET (55)
# meaning_of_life            NOUN (301), PRON (215), AUX (155), ADP (153), DET (126)
# no                          PRON (144), AUX (132), ADJ (101), PART (60), INTJ (58)
# reminder                  PRON (326), NOUN (282), VERB (222), ADP (143), AUX (110)
# reminder_update          VERB (301), NOUN (265), PRON (171), DET (159), PART (139)
# shopping_list             NOUN (357), PRON (301), VERB (172), ADP (132), AUX (119)
# shopping_list_update      NOUN (471), PRON (288), VERB (227), ADP (207), AUX (101)
# spelling                VERB (188), NOUN (186), PRON (138), AUX (112), SCONJ (101)
# tell_joke                  PRON (224), VERB (192), NOUN (179), DET (107), ADJ (88)
# text                     VERB (321), PRON (239), NOUN (178), PROPN (109), AUX (97)
# time                       NOUN (200), PRON (186), DET (153), AUX (136), ADP (107)
# timer                       NOUN (244), VERB (189), DET (128), ADP (106), NUM (98)
# todo_list                 NOUN (380), PRON (305), VERB (224), AUX (157), ADP (150)
# todo_list_update         NOUN (374), VERB (241), PRON (206), ADP (189), PART (118)
# traffic                    NOUN (369), ADP (312), DET (207), AUX (157), PRON (149)
# translate                PRON (247), VERB (195), AUX (183), ADP (149), PROPN (135)
# weather                    NOUN (213), PRON (146), AUX (145), DET (115), VERB (94)
# what_is_your_name          PRON (349), VERB (187), AUX (157), NOUN (109), ADP (45)
# who_made_you                PRON (313), VERB (178), NOUN (112), AUX (85), DET (75)
# word_of_the_day           NOUN (352), DET (232), PRON (194), ADP (165), VERB (135)
# yes                          PRON (138), AUX (104), ADJ (92), INTJ (75), VERB (45)
# ```

# %%
# corpus exploration - entities

all_ents = df.ents.sum()
most_common_ents = Counter(all_ents).most_common(10)
print('most common entities:')
pprint(most_common_ents)

print('most common entities by intent:')
all_ents_by_intent = df.groupby('intent_str').ents.sum()
all_ents_by_intent = all_ents_by_intent.apply(lambda x: Counter(x).most_common(5))
print(all_ents_by_intent.apply(format_counter).to_string())

# -> The DATE is the outlier here; it's quite a bit more common than all the
# other entities. Examining the entities by intent, this fits we what we would
# expect from this data. The bulk of the DATEs come from the "date" and
# "word_of_the_day" intents.

# %% [markdown]
# ```text
# most common entities:
#
# [('DATE', 423),
#  ('CARDINAL', 228),
#  ('GPE', 196),
#  ('TIME', 169),
#  ('PERSON', 167),
#  ('LANGUAGE', 91),
#  ('NORP', 43),
#  ('FAC', 35),
#  ('PERCENT', 19),
#  ('ORG', 18)]
#
# most common entities by intent:
#
# intent_str
# are_you_a_bot                                                               PERSON (1)
# calculator              CARDINAL (216), PERCENT (19), MONEY (8), QUANTITY (3), FAC (2)
# date                                                            DATE (134), PERSON (1)
# definition               PERSON (10), WORK_OF_ART (3), PRODUCT (2), EVENT (1), LOC (1)
# find_phone                                                                  PERSON (5)
# flip_coin                                                     CARDINAL (4), PERSON (1)
# food_beverage_recipe                                   PRODUCT (3), TIME (2), NORP (2)
# goodbye                                    PERSON (9), PRODUCT (2), TIME (2), DATE (1)
# greeting                                                DATE (12), PERSON (9), ORG (2)
# maybe                                               CARDINAL (6), DATE (1), PERSON (1)
# meaning_of_life
# no
# reminder                                                 PERSON (2), DATE (2), ORG (1)
# reminder_update                                       DATE (25), TIME (13), PERSON (4)
# shopping_list                                        DATE (2), PERSON (2), PRODUCT (1)
# shopping_list_update               PERSON (3), PRODUCT (2), QUANTITY (1), CARDINAL (1)
# spelling                       PERSON (8), GPE (3), WORK_OF_ART (1), DATE (1), ORG (1)
# tell_joke                                                                     NORP (1)
# text                                PERSON (104), DATE (8), TIME (7), ORG (1), GPE (1)
# time                                    GPE (65), LOC (8), NORP (4), ORG (2), TIME (1)
# timer                                                                        TIME (98)
# todo_list                                               DATE (18), PERSON (1), ORG (1)
# todo_list_update                          DATE (3), PERSON (2), QUANTITY (1), TIME (1)
# traffic                               GPE (62), TIME (40), FAC (31), DATE (7), ORG (3)
# translate                     LANGUAGE (89), NORP (36), GPE (24), PERSON (3), DATE (2)
# weather                           DATE (48), GPE (40), TIME (4), LOC (3), QUANTITY (1)
# what_is_your_name                                     ORG (2), PERSON (1), ORDINAL (1)
# who_made_you                                                          ORG (1), GPE (1)
# word_of_the_day                                     DATE (156), TIME (1), CARDINAL (1)
# yes
# ```

# %%
# corpus exploration - sentiment

# The sentiment column was produced by nltk's VADER, a rules-based sentiment analysis
# tool originally designed for social media text. There are much more sophisticated
# tools for sentiment analysis, but since that's not the main focus of the model
# we are building, this quick rules-based sentiment model will do.
# Rather than a detailed analysis, we calculate average sentiment (POS/NEG/NEU) across
# intents and the whole dataset. This can help reveal patterns in the data
# that may cause the model to learn patterns you don't intend.

print('sentiment summary:')
print(df.sentiment.value_counts().to_string(header=False))

# -> Mostly neutral sentiment, which is what one would expect for this intent data
# for a personal assistant chatbot. Most of the utterances with negative sentiment
# are from the intents "no" and "maybe" and are quite benign. The other couple are
# just (understandable) mistakes from the sentiment analyzer, e.g.
# i lost my phone (find_phone)
# please define institutional racism (definition)
# how i make a killer smoothie (food_beverage_recipe)

# %% [markdown]
# ```text
# sentiment summary:
#
# neu    4223
# pos     224
# neg      61
# ```
