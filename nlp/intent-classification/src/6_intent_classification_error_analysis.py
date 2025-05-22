# %% [markdown]
# ## Intent Classification Error Analysis
#
# The plan is to:
#
# * try out inference with the saved model
# * analyze the errors made by the model

# %%
import joblib
import os

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from utils import INTENT_LABELS_PATH, MODEL_DIR, DATA_DIR, init_nb

init_nb()

# %%
# quick inference class to load everything and make predictions

def load_model(model_path: Path | str) -> BaseEstimator:
    with Path(model_path).open('rb') as f:
        return joblib.load(f)


class ModelInference:
    def __init__(self, model_pkl: Path, vectorizer_pkl: Path) -> None:
        # load string label corresponding to int y[0]
        self.intent_labels = pd.read_json(INTENT_LABELS_PATH)
        self.intent_labels.set_index('label_int', inplace=True)

        # load vectorizer
        self.tfidf = joblib.load(vectorizer_pkl)
        print(f'√ vectorizer loaded from {vectorizer_pkl}')

        # load model
        self.model = load_model(model_pkl)
        print(f'√ model loaded from {model_pkl}')

    def predict(self, user_input: str) -> str:
        '''
        Given a user input, return an intent label
        '''
        x = self.tfidf.transform([user_input]).toarray()
        # X = self.tfidf.transform(df.utterance)
        # X = np.vstack(df.embedding.to_numpy())
        y = self.model.predict(x)
        return self.intent_labels.loc[y[0]]['label_str']


# newest version
timestamp = '1746644761'
vectorizer_path = Path(MODEL_DIR / f'tfidf_vectorizer_{timestamp}.pkl')
saved_model_path = Path(MODEL_DIR / f'model_{timestamp}.pkl')

ai = ModelInference(saved_model_path, vectorizer_path)
print(ai.predict('hello there, friend')) # greeting
print(ai.predict('hey can you tell me the fastest way from sf to daly city during rush hour?')) # traffic
print(ai.predict('how would i say "appreciate" in spanish')) # translate

# %% [markdown]
# ```text
# √ vectorizer loaded...
# √ model loaded...
#
# greeting
# traffic
# translate
# ```

# %%
# logistic regression model error analysis

intent_labels = pd.read_json(INTENT_LABELS_PATH)
str_labels = intent_labels.label_str.tolist()
int_labels = intent_labels.label_int.tolist()

df_train = pd.read_json(DATA_DIR / 'df_train.jsonl', orient='records', lines=True)
df_test = pd.read_json(DATA_DIR / 'df_test.jsonl', orient='records', lines=True)
df_val = pd.read_json(DATA_DIR / 'df_val.jsonl', orient='records', lines=True)
df_pred = pd.read_json(DATA_DIR / 'y_pred_lr.jsonl', orient='records', lines=True)

y_train = df_train.intent.to_numpy()
y_test = df_test.intent.to_numpy()
y_val = df_val.intent.to_numpy()
y_pred_lr = df_pred.prediction.to_numpy()

cm = confusion_matrix(
    y_true=y_test,
    y_pred=y_pred_lr,
    sample_weight=None,
    normalize=None
)

def display_confusion_matrix(cm, filepath: Path | None):
    colors = ['#333840', '#A3B18A', '#69B34C']
    nodes = [0.0, 1/len(str_labels), 1.0]
    custom_cmap = LinearSegmentedColormap.from_list('custom', list(zip(nodes, colors)))

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=custom_cmap)
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(len(str_labels))
    plt.xticks(tick_marks, str_labels, rotation=90, size=8)
    plt.yticks(tick_marks, str_labels, size=8)
    plt.tight_layout(pad=5)
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
            horizontalalignment='center',
            verticalalignment='center')
    if matplotlib.get_backend() == 'inline':
        plt.show()
    else:
        if filepath is not None:
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, transparent=True, format='png')
            print(f'image created: {filepath}')

display_confusion_matrix(cm, Path(os.getcwd()).parent / 'img' / 'cm.png')

# The confusion matrix graphic makes it easy to see where the model's predictions are
# incorrect for the test set. The errors are pretty evenly spread out, i.e. we don't see any
# clear patterns that might indicate an issue.

# %% [markdown]
# ![confusion matrix](../img/cm.png)

# %%
# Let's take a closer look at the labels that have more than 1 misclassification
# in the confusion matrix.

# add predictions to df_test
df_test.loc[:, 'prediction'] = y_pred_lr
df_test.loc[:, 'prediction_str'] = [str_labels[i] for i in y_pred_lr]

# get labels with > 1 misclassification (i.e. duplicate intents in this errors df)
df_errors = df_test[df_test.intent != df_test.prediction].sort_values('intent')
df_error_dups = df_errors[df_errors.duplicated(subset='intent', keep=False)]

# left justify column names and utterance column values
with pd.option_context('display.colheader_justify','left'):
    print(df_error_dups[['intent_str', 'prediction_str', 'utterance']].to_string(
        index=False, formatters={'utterance':'{{:<{}s}}'.format(df_error_dups.utterance.str.len().max()).format}
))

# %% [markdown]
# ```text
# intent_str           prediction_str       utterance
#           calculator    what_is_your_name what do you get if you divide 3 by 2
#           calculator food_beverage_recipe i bought 6 shirts at $499 each what was my total expenditure for them
#             greeting              goodbye it is good to see you
#             greeting                  yes heller
#             greeting              weather what's happening
#                maybe                  yes possibly
#                maybe                  yes undecided
#                   no                maybe i'll pass
#                   no                  yes absolutely not
#                   no                maybe are you sure i don't think that's correct
#                   no            translate i say negative
# shopping_list_update        shopping_list my shopping list should have carrots on it
# shopping_list_update     todo_list_update add laundry detergent to the list
# shopping_list_update        shopping_list get rid of butter on my shopping list
#                timer            translate wake me in an hour
#                timer             greeting let me know when it's been 5 minutes
#            todo_list             reminder did i create a task to clean the gutters on my list
#            todo_list        shopping_list does my errand list have goodwill on it
#            translate           calculator what is latin for i love you
#            translate    what_is_your_name what do you call a subway if you were english
#    what_is_your_name           calculator what is your names
#    what_is_your_name         who_made_you i want to know the name that was given by the person who made you
#         who_made_you             greeting how did you come to be you
#         who_made_you    what_is_your_name what's your brand
#         who_made_you    what_is_your_name tell me your brand
#         who_made_you           definition tell me the inventor of ai
#                  yes                maybe sure
#                  yes             greeting okay
#                  yes food_beverage_recipe please let's do it
#                  yes                   no do that
#                  yes           calculator can we phase
#                  yes                   no that is accurate
# ```

# %% [markdown]
# ## Error Analysis
#
# Any patterns of errors we should be aware of?
#
# > list confusion (5/32), e.g. "add laundry detergent to the list", the model predicted
# "todo_list_update" instead of "shopping_list_update".
# These intents have a lot of word overlap, e.g.
# "can you add laundry to my to do list" (todo_list_update)
# "add laundry detergent to the list" (shopping_list_update)
# This makes it hard for the model based on TF-IDF features
# to distinguish between the two.
#
# > yes/no/maybe (12/32) - nuances that our model can't pick up,
# like: "sure" vs. "are you sure", negation: "absolutely" vs. "absolutely not".
# These are some more limitations of using a simple representation like TF-IDF.
# Using more complex representations like word embeddings to better capture
# semantic meaning might result in fewer errors with this set.
#
# > other (15/32) - residual errors spread over 6 other intents
#
# How many errors came from LLM-generated data?

# %%
print(f'llm-generated data errors: {any(df_error_dups["llm_generated"])}')

# %% [markdown]
# ```text
# llm-generated data errors: False
# ```

# %% [markdown]
# # Conclusion
#
# Using the CLINC-150 dataset, we selected a subset of intents to train our model to recognize for the use case of a personal assistant chatbot. We vectorized the data using term and document frequency counts, TF-IDF. We experimented with training several different models using different algorithms, focusing on simpler model architectures. We were able to achieve an accuracy/F1 score > 95% after tuning our logistic regression model's hyperparameters. We now have a trained model saved to disk and functions to load it later and use it for inference in a chatbot.
