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
# ## Intent Classification Modeling
#
# The plan is to:
#
# * train, tune, and evaluate several different types of models
#   * k nearest neighbors
#   * logistic regression
#   * multinomial naive bayes
#   * support vector machine
# * choose a model based on the above experimentation and the below requirements
#
# Requirements:
#
# * simple models, no deep learning
# * accuracy >= 95%
# * reasonable training time
# * fast prediction time
# * low resource usage for inference
#
# We will use accuracy as our main optimizing metric. Our dataset is well-balanced,
# and there is no particular emphasis on precision or recall for our intent
# classification task. False positives and false negatives will both result in
# a misclassification, so they have a similar cost. We will look at accuracy for
# each class and the model overall to assess performance. We explore simpler
# modeling approaches, with the goal of excellent performance with some
# infrequent misclassifications allowed.

# %%
import joblib
import re
import subprocess
import sys
import time

from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils import (
    init_nb, Timer, TFIDF_DATA_PATH, INTENT_LABELS_PATH, RANDOM_SEED, MODEL_DIR, DATA_DIR
)

init_nb()

# %%

intent_labels = pd.read_json(INTENT_LABELS_PATH)
str_labels = intent_labels.label_str.tolist()
int_labels = intent_labels.label_int.tolist()

# newest version
timestamp = '1746644761'
latest_tfidf_data = TFIDF_DATA_PATH.parent / f'data_tfidf_{timestamp}.jsonl'

df = pd.read_json(latest_tfidf_data, orient='records', lines=True)

# %%

# split into train, test, validation
# use original dataset proportions: train=.66, test=.2, val=.133
# just split dataframe, ignore features vs. target for now

# split .66 train / .33 test & val
df_train, df_test_val, _, _ = train_test_split(df, df, test_size=1/3, random_state=RANDOM_SEED) # test_size=test + val (.33)

# split .133 val / .2 test
df_val, df_test, _, _ = train_test_split(df_test_val, df_test_val, test_size=3/5, random_state=RANDOM_SEED) # test_size=test (.2 = 3/5 of 1/3)

print(f'{len(df_train)=}')
print(f'{len(df_test)=}')
print(f'{len(df_val)=}')

df_train.to_json(DATA_DIR / 'df_train.jsonl', orient='records', lines=True)
df_test.to_json(DATA_DIR / 'df_test.jsonl', orient='records', lines=True)
df_val.to_json(DATA_DIR / 'df_val.jsonl', orient='records', lines=True)

# extract target
y_train = df_train.intent.to_numpy()
y_test = df_test.intent.to_numpy()
y_val = df_val.intent.to_numpy()

# extract features and create matrix required for classifiers
X_train = np.vstack(df_train.tfidf_vector.to_numpy())
X_test = np.vstack(df_test.tfidf_vector.to_numpy())
X_val = np.vstack(df_val.tfidf_vector.to_numpy())

print(f'{X_train.shape=}')
print(f'{X_test.shape=}')
print(f'{X_val.shape=}')
print(f'{y_train.shape=}')
print(f'{y_test.shape=}')
print(f'{y_val.shape=}')

# %% [markdown]
# ```text
# len(df_train)=3005
# len(df_test)=902
# len(df_val)=601
# X_train.shape=(3005, 10688)
# X_test.shape=(902, 10688)
# X_val.shape=(601, 10688)
# y_train.shape=(3005,)
# y_test.shape=(902,)
# y_val.shape=(601,)
# ```

# %%
# k nearest neighbors
with Timer('k nearest neighbors (untuned) training time'):
    knn_default = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform', # all points in each neighborhood are weighted equally.
        # weights='distance', # weigh points by the inverse of their distance, closer neighbors of a query point will have a greater influence than neighbors which are further away
        algorithm='auto', # nearest neighbors algorithm ('brute', 'kd_tree', 'ball_tree')
        leaf_size=30, # leaf size for kd_tree and ball_tree algorithms
        p=2, # power parameter for Minkowski metric
        metric='minkowski', # uses standard Euclidean distance when p=2 (can also be 'precomputed' or a Callable)
        metric_params=None, # additional params for the metric function
        n_jobs=-1, # use all processors for neighbor seach (no effect on fit)
    )
    knn_default.fit(X_train, y_train)
with Timer('k nearest neighbors (untuned) test set eval time'):
    y_pred_knn_default = knn_default.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_knn_default,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

pprint(knn_default.get_params())

knn_params = {
    'metric': ('minkowski', 'cosine'),
    'n_neighbors': list(range(1, 31)),
    'p': (1, 2),
    'weights': ('distance', 'uniform'),
}

with Timer('k nearest neighbors hyperparameter tuning'):
    knn_grid_search = GridSearchCV(
        estimator=knn_default,
        param_grid=knn_params,
        scoring='accuracy',
        n_jobs=-1, # use all cpus
        refit=True,
        cv=5,
        verbose=3,
        error_score=np.nan,
        return_train_score=False,
    )
    knn_grid_search.fit(X_train, y_train)

# parameter setting that gave the best results on the hold out data
pprint(knn_grid_search.best_params_ )
# mean cross-validated score of the best_estimator
print('best score: ', knn_grid_search.best_score_)

knn = knn_grid_search.best_estimator_

with Timer('k nearest neighbors (tuned) test set eval time'):
    y_pred_knn = knn.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_knn,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))


# %% [markdown]
# ```text
# k nearest neighbors (untuned) training time: 0.007960666989674792 seconds
# k nearest neighbors (untuned) test set eval time: 0.23888179200002924 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     0.9444    0.9444    0.9444        18
#           calculator     1.0000    0.7879    0.8814        33
#                 date     0.6364    0.9655    0.7671        29
#           definition     0.9032    0.8750    0.8889        32
#           find_phone     0.8800    1.0000    0.9362        22
#            flip_coin     1.0000    0.9355    0.9667        31
# food_beverage_recipe     0.7143    0.7143    0.7143        35
#              goodbye     0.9062    0.8788    0.8923        33
#             greeting     0.8710    0.8710    0.8710        31
#                maybe     0.7143    0.8333    0.7692        24
#      meaning_of_life     0.8250    0.8919    0.8571        37
#                   no     0.9286    0.7429    0.8254        35
#             reminder     0.8800    0.8462    0.8627        26
#      reminder_update     0.9429    0.9429    0.9429        35
#        shopping_list     0.7500    1.0000    0.8571        30
# shopping_list_update     0.9565    0.7586    0.8462        29
#             spelling     0.9630    0.8966    0.9286        29
#            tell_joke     0.9259    0.9259    0.9259        27
#                 text     1.0000    0.9630    0.9811        27
#                 time     0.8824    1.0000    0.9375        30
#                timer     0.8710    0.9000    0.8852        30
#            todo_list     0.6452    0.8000    0.7143        25
#     todo_list_update     0.8800    0.6875    0.7719        32
#              traffic     1.0000    0.8500    0.9189        40
#            translate     0.8276    0.9600    0.8889        25
#              weather     0.8333    0.7692    0.8000        26
#    what_is_your_name     0.7714    0.9643    0.8571        28
#         who_made_you     0.9697    0.8649    0.9143        37
#      word_of_the_day     0.9032    0.8485    0.8750        33
#                  yes     0.8800    0.6667    0.7586        33
#
#             accuracy                         0.8647       902
#            macro avg     0.8735    0.8695    0.8660       902
#         weighted avg     0.8774    0.8647    0.8655       902
#
# {'algorithm': 'auto',
#  'leaf_size': 30,
#  'metric': 'minkowski',
#  'metric_params': None,
#  'n_jobs': -1,
#  'n_neighbors': 5,
#  'p': 2,
#  'weights': 'uniform'}
#
# Fitting 5 folds for each of 240 candidates, totalling 1200 fits
# [CV 2/5] END metric=minkowski, n_neighbors=1, p=2, weights=distance;, score=0.852 total time=   0.6s
# [CV 1/5] END metric=minkowski, n_neighbors=1, p=2, weights=distance;, score=0.834 total time=   0.8s
# [truncated]
# [CV 4/5] END metric=cosine, n_neighbors=30, p=2, weights=uniform;, score=0.790 total time=   1.1s
# [CV 4/5] END metric=cosine, n_neighbors=29, p=1, weights=uniform;, score=0.790 total time=   0.2s
#
# k nearest neighbors hyperparameter tuning: 517.9623270409938 seconds
# {'metric': 'cosine', 'n_neighbors': 9, 'p': 1, 'weights': 'distance'}
# best score:  0.8532445923460898
#
# k nearest neighbors (tuned) test set eval time: 0.8965172089810949 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     0.9000    1.0000    0.9474        18
#           calculator     1.0000    0.8182    0.9000        33
#                 date     0.7568    0.9655    0.8485        29
#           definition     0.9630    0.8125    0.8814        32
#           find_phone     0.9167    1.0000    0.9565        22
#            flip_coin     1.0000    0.9355    0.9667        31
# food_beverage_recipe     0.7647    0.7429    0.7536        35
#              goodbye     0.9310    0.8182    0.8710        33
#             greeting     0.8333    0.8065    0.8197        31
#                maybe     0.8077    0.8750    0.8400        24
#      meaning_of_life     0.8974    0.9459    0.9211        37
#                   no     0.9655    0.8000    0.8750        35
#             reminder     0.9167    0.8462    0.8800        26
#      reminder_update     0.9118    0.8857    0.8986        35
#        shopping_list     0.7568    0.9333    0.8358        30
# shopping_list_update     0.9583    0.7931    0.8679        29
#             spelling     0.8966    0.8966    0.8966        29
#            tell_joke     0.9615    0.9259    0.9434        27
#                 text     0.9600    0.8889    0.9231        27
#                 time     0.8788    0.9667    0.9206        30
#                timer     0.8182    0.9000    0.8571        30
#            todo_list     0.6562    0.8400    0.7368        25
#     todo_list_update     0.9615    0.7812    0.8621        32
#              traffic     0.9730    0.9000    0.9351        40
#            translate     0.7742    0.9600    0.8571        25
#              weather     0.8462    0.8462    0.8462        26
#    what_is_your_name     0.7500    0.9643    0.8438        28
#         who_made_you     0.8947    0.9189    0.9067        37
#      word_of_the_day     0.8857    0.9394    0.9118        33
#                  yes     0.9200    0.6970    0.7931        33
#
#             accuracy                         0.8758       902
#            macro avg     0.8819    0.8801    0.8765       902
#         weighted avg     0.8858    0.8758    0.8764       902
# ```

# %%
# logistic regression

with Timer('logistic regression (untuned) training time'):
    lr_default = LogisticRegression(
        penalty = 'l2', # L2 regularization
        dual = False, # dual (constrained) or primal (regularized) formulation
        tol = 1e-4, # tolerance for stopping criteria
        C = 1.0, # inverse of regularization strength, smaller = stronger regularization
        fit_intercept = True, # add bias to decision function
        intercept_scaling = 1.0, # for use with liblinear solver
        class_weight = None, # dict of weights or 'balanced'
        random_state=RANDOM_SEED,
        solver='lbfgs', # lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga
    )
    lr_default.fit(X_train, y_train)
with Timer('logistic regression (untuned) test set eval time'):
    y_pred_lr_default = lr_default.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_lr_default,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

pprint(lr_default.get_params())

lr_params = {
    'C': np.logspace(-4, 4, 20)
}

with Timer('logistic regression hyperparameter tuning'):
    lr_grid_search = GridSearchCV(
        estimator=lr_default,
        param_grid=lr_params,
        scoring='accuracy',
        n_jobs=-1, # use all cpus
        refit=True,
        cv=5,
        verbose=3,
        error_score=np.nan,
        return_train_score=False,
    )
    lr_grid_search.fit(X_train, y_train)

# parameter setting that gave the best results on the hold out data
pprint(lr_grid_search.best_params_ )
# mean cross-validated score of the best_estimator
print('best score: ', lr_grid_search.best_score_)

lr = lr_grid_search.best_estimator_

with Timer('logistic regression (tuned) test set eval time'):
    y_pred_lr = lr.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_lr,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

# %% [markdown]
# ```text
# logistic regression (untuned) training time: 3.160173791984562 seconds
# logistic regression (untuned) test set eval time: 0.0327206660003867 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     0.9062    0.8788    0.8923        33
#                 date     0.9655    0.9655    0.9655        29
#           definition     1.0000    0.9062    0.9508        32
#           find_phone     0.9167    1.0000    0.9565        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.7234    0.9714    0.8293        35
#              goodbye     0.9355    0.8788    0.9062        33
#             greeting     0.8438    0.8710    0.8571        31
#                maybe     0.7692    0.8333    0.8000        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.8857    0.8857    0.8857        35
#             reminder     0.9231    0.9231    0.9231        26
#      reminder_update     0.9444    0.9714    0.9577        35
#        shopping_list     0.9032    0.9333    0.9180        30
# shopping_list_update     0.9630    0.8966    0.9286        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     0.9630    0.9630    0.9630        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    0.9667    0.9831        30
#                timer     1.0000    0.9000    0.9474        30
#            todo_list     0.8800    0.8800    0.8800        25
#     todo_list_update     0.9355    0.9062    0.9206        32
#              traffic     1.0000    0.9500    0.9744        40
#            translate     0.8462    0.8800    0.8627        25
#              weather     0.9600    0.9231    0.9412        26
#    what_is_your_name     0.8125    0.9286    0.8667        28
#         who_made_you     0.9714    0.9189    0.9444        37
#      word_of_the_day     0.8919    1.0000    0.9429        33
#                  yes     1.0000    0.6667    0.8000        33
#
#             accuracy                         0.9246       902
#            macro avg     0.9284    0.9255    0.9245       902
#         weighted avg     0.9304    0.9246    0.9249       902
#
# {'C': 1.0,
#  'class_weight': None,
#  'dual': False,
#  'fit_intercept': True,
#  'intercept_scaling': 1.0,
#  'l1_ratio': None,
#  'max_iter': 100,
#  'multi_class': 'deprecated',
#  'n_jobs': None,
#  'penalty': 'l2',
#  'random_state': 42,
#  'solver': 'lbfgs',
#  'tol': 0.0001,
#  'verbose': 0,
#  'warm_start': False}
#
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# [CV 1/5] END ...........C=0.0006951927961775605;, score=0.037 total time=   2.3s
# [CV 4/5] END ..........C=0.00026366508987303583;, score=0.037 total time=   2.5s
# [truncated]
# [CV 5/5] END .........................C=10000.0;, score=0.920 total time=   2.6s
# [CV 4/5] END .........................C=10000.0;, score=0.910 total time=   3.5s
#
# logistic regression hyperparameter tuning: 42.528201458015246 seconds
# {'C': 545.5594781168514}
# best score:  0.93477537437604
#
# logistic regression (tuned) test set eval time: 0.019191125000361353 seconds
#                       precision    recall  f1-score   support

#        are_you_a_bot     1.0000    0.9444    0.9714        18
#           calculator     0.9118    0.9394    0.9254        33
#                 date     0.9655    0.9655    0.9655        29
#           definition     0.9688    0.9688    0.9688        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.8750    1.0000    0.9333        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.8750    0.9032    0.8889        31
#                maybe     0.8462    0.9167    0.8800        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.9394    0.8857    0.9118        35
#             reminder     0.9259    0.9615    0.9434        26
#      reminder_update     0.9714    0.9714    0.9714        35
#        shopping_list     0.9091    1.0000    0.9524        30
# shopping_list_update     1.0000    0.8966    0.9455        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    0.9667    0.9831        30
#                timer     1.0000    0.9333    0.9655        30
#            todo_list     1.0000    0.9200    0.9583        25
#     todo_list_update     0.9688    0.9688    0.9688        32
#              traffic     1.0000    0.9750    0.9873        40
#            translate     0.9200    0.9200    0.9200        25
#              weather     0.9259    0.9615    0.9434        26
#    what_is_your_name     0.8387    0.9286    0.8814        28
#         who_made_you     0.9706    0.8919    0.9296        37
#      word_of_the_day     0.9706    1.0000    0.9851        33
#                  yes     0.8710    0.8182    0.8438        33

#             accuracy                         0.9523       902
#            macro avg     0.9541    0.9524    0.9525       902
#         weighted avg     0.9541    0.9523    0.9525       902
# ```

# %%
# multinomial naive bayes

with Timer('multinomial naive bayes (untuned) training time'):
    nb_default = MultinomialNB(
        alpha=1.0, # Laplace smoothing (set < 1 for Lidstone)
        force_alpha=True, # avoid numerical errors
        fit_prior=True, # calculate class prior probabilities based on the frequency of each class in the training data
    )
    nb_default.fit(X_train, y_train)
with Timer('multinomial naive bayes (untuned) test set eval time'):
    y_pred_nb_default = nb_default.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_nb_default,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

pprint(nb_default.get_params())

nb_params = {
    'alpha': (1.0, 0.6, 0.3, 0.1)
}

with Timer('multinomial naive bayes hyperparameter tuning'):
    nb_grid_search = GridSearchCV(
        estimator=nb_default,
        param_grid=nb_params,
        scoring='accuracy',
        n_jobs=-1, # use all cpus
        refit=True,
        cv=5,
        verbose=3,
        error_score=np.nan,
        return_train_score=False,
    )
    nb_grid_search.fit(X_train, y_train)

# parameter setting that gave the best results on the hold out data
pprint(nb_grid_search.best_params_ )
# mean cross-validated score of the best_estimator
print('best score: ', nb_grid_search.best_score_)

nb = nb_grid_search.best_estimator_

with Timer('multinomial naive bayes (tuned) test set eval time'):
    y_pred_nb = nb.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_nb,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

# %% [markdown]
# ```text
# multinomial naive bayes (untuned) training time: 0.06071029099985026 seconds
# multinomial naive bayes (untuned) test set eval time: 0.018574333982542157 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     0.7200    1.0000    0.8372        18
#           calculator     1.0000    0.6970    0.8214        33
#                 date     0.8485    0.9655    0.9032        29
#           definition     1.0000    0.5938    0.7451        32
#           find_phone     0.8462    1.0000    0.9167        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    0.8571    0.9231        35
#              goodbye     0.9615    0.7576    0.8475        33
#             greeting     0.9259    0.8065    0.8621        31
#                maybe     0.8750    0.8750    0.8750        24
#      meaning_of_life     0.8222    1.0000    0.9024        37
#                   no     0.9091    0.8571    0.8824        35
#             reminder     0.8929    0.9615    0.9259        26
#      reminder_update     0.9444    0.9714    0.9577        35
#        shopping_list     0.8000    0.9333    0.8615        30
# shopping_list_update     0.9259    0.8621    0.8929        29
#             spelling     0.9667    1.0000    0.9831        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     0.9000    0.9000    0.9000        30
#            todo_list     0.8214    0.9200    0.8679        25
#     todo_list_update     0.9667    0.9062    0.9355        32
#              traffic     0.9756    1.0000    0.9877        40
#            translate     0.7419    0.9200    0.8214        25
#              weather     0.9583    0.8846    0.9200        26
#    what_is_your_name     0.7500    0.9643    0.8438        28
#         who_made_you     0.9714    0.9189    0.9444        37
#      word_of_the_day     0.8250    1.0000    0.9041        33
#                  yes     0.9565    0.6667    0.7857        33
#
#             accuracy                         0.9035       902
#            macro avg     0.9078    0.9073    0.9004       902
#         weighted avg     0.9144    0.9035    0.9018       902
#
# {'alpha': 1.0, 'class_prior': None, 'fit_prior': True, 'force_alpha': True}
#
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# [CV 1/5] END .........................alpha=1.0;, score=0.872 total time=   0.2s
# [CV 5/5] END .........................alpha=1.0;, score=0.902 total time=   0.2s
# [truncated]
# [CV 3/5] END .........................alpha=0.1;, score=0.895 total time=   0.1s
# [CV 4/5] END .........................alpha=0.1;, score=0.887 total time=   0.1s
#
# multinomial naive bayes hyperparameter tuning: 0.6898890830052551 seconds
# {'alpha': 0.3}
# best score:  0.9084858569051579
#
# multinomial naive bayes (tuned) test set eval time: 0.018093249993398786 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     0.7200    1.0000    0.8372        18
#           calculator     1.0000    0.7879    0.8814        33
#                 date     0.8485    0.9655    0.9032        29
#           definition     1.0000    0.6562    0.7925        32
#           find_phone     0.9167    1.0000    0.9565        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.9375    0.8571    0.8955        35
#              goodbye     0.9630    0.7879    0.8667        33
#             greeting     0.9231    0.7742    0.8421        31
#                maybe     0.8750    0.8750    0.8750        24
#      meaning_of_life     0.8605    1.0000    0.9250        37
#                   no     0.9062    0.8286    0.8657        35
#             reminder     0.8929    0.9615    0.9259        26
#      reminder_update     0.9459    1.0000    0.9722        35
#        shopping_list     0.8485    0.9333    0.8889        30
# shopping_list_update     0.8667    0.8966    0.8814        29
#             spelling     0.9655    0.9655    0.9655        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     0.9000    0.9000    0.9000        30
#            todo_list     0.9200    0.9200    0.9200        25
#     todo_list_update     0.9677    0.9375    0.9524        32
#              traffic     0.9756    1.0000    0.9877        40
#            translate     0.7742    0.9600    0.8571        25
#              weather     0.8846    0.8846    0.8846        26
#    what_is_your_name     0.7500    0.9643    0.8438        28
#         who_made_you     0.9714    0.9189    0.9444        37
#      word_of_the_day     0.8684    1.0000    0.9296        33
#                  yes     0.9130    0.6364    0.7500        33
#
#             accuracy                         0.9102       902
#            macro avg     0.9120    0.9137    0.9075       902
#         weighted avg     0.9174    0.9102    0.9086       902
# ```

# %%
# support vector machine

with Timer('support vector machine (untuned) training time'):
    svm_default = SVC(
        kernel='linear', # 'rbf' (default), 'linear', 'poly', 'sigmoid', 'precomputed'
        C=1, # regularization
        gamma='scale', # 'scale', 'auto'
        coef0=0.0, # independent term for 'poly' and 'sigmoid'
        shrinking=True, # shrinking heuristic
        probability=False, # probability estimates
        tol=0.001, # tolerance for stopping criterion
        random_state=RANDOM_SEED,
        verbose=False,
        cache_size=1000, # use 1GB for kernel cache
    )
    svm_default.fit(X_train, y_train)

with Timer('support vector machine (untuned) test set eval time'):
    y_pred_svm_default = svm_default.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_svm_default,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

pprint(svm_default.get_params())

svm_params = {
    'C': (0.1, 1, 10, 100, 1000),
    'gamma': ('scale', 1.0, 0.1, 0.01, 0.001, 0.0001),
    'kernel': ('rbf', 'linear')
}


with Timer('support vector machine hyperparameter tuning'):
    svm_grid_search = GridSearchCV(
        estimator=svm_default,
        param_grid=svm_params,
        scoring='accuracy',
        n_jobs=-1, # use all cpus
        refit=True,
        cv=5,
        verbose=3,
        error_score=np.nan,
        return_train_score=False,
    )

    svm_grid_search.fit(X_train, y_train)


# parameter setting that gave the best results on the hold out data
pprint(svm_grid_search.best_params_ )
# mean cross-validated score of the best_estimator
print('best score: ', svm_grid_search.best_score_)

svm = svm_grid_search.best_estimator_

with Timer('support vector machine (tuned) test set eval time'):
    y_pred_svm = svm.predict(X_test)

print(classification_report(
    y_true=y_test,
    y_pred=y_pred_svm,
    labels=list(int_labels),
    target_names=list(str_labels),
    sample_weight=None,
    digits=4,
    output_dict=False,
))

# %% [markdown]
# ```text
# support vector machine (untuned) training time: 11.474093458993593 seconds
# support vector machine (untuned) test set eval time: 3.43490262501291 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     1.0000    1.0000    1.0000        18
#           calculator     0.9394    0.9394    0.9394        33
#                 date     1.0000    0.9310    0.9643        29
#           definition     1.0000    0.9688    0.9841        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.7727    0.9714    0.8608        35
#              goodbye     0.9412    0.9697    0.9552        33
#             greeting     0.8788    0.9355    0.9062        31
#                maybe     0.8462    0.9167    0.8800        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.9412    0.9143    0.9275        35
#             reminder     0.9259    0.9615    0.9434        26
#      reminder_update     0.9444    0.9714    0.9577        35
#        shopping_list     0.8824    1.0000    0.9375        30
# shopping_list_update     1.0000    0.8621    0.9259        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     1.0000    0.9630    0.9811        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    0.9333    0.9655        30
#                timer     1.0000    0.9000    0.9474        30
#            todo_list     1.0000    0.9200    0.9583        25
#     todo_list_update     0.9688    0.9688    0.9688        32
#              traffic     1.0000    0.9500    0.9744        40
#            translate     0.9583    0.9200    0.9388        25
#              weather     0.8966    1.0000    0.9455        26
#    what_is_your_name     0.8710    0.9643    0.9153        28
#         who_made_you     0.9714    0.9189    0.9444        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     0.9000    0.8182    0.8571        33
#
#             accuracy                         0.9501       902
#            macro avg     0.9546    0.9511    0.9515       902
#         weighted avg     0.9540    0.9501    0.9507       902
#
# {'C': 1,
#  'break_ties': False,
#  'cache_size': 1000,
#  'class_weight': None,
#  'coef0': 0.0,
#  'decision_function_shape': 'ovr',
#  'degree': 3,
#  'gamma': 'scale',
#  'kernel': 'linear',
#  'max_iter': -1,
#  'probability': False,
#  'random_state': 42,
#  'shrinking': True,
#  'tol': 0.001,
#  'verbose': False}
#
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# [CV 2/5] END .C=0.1, gamma=scale, kernel=linear;, score=0.642 total time= 1.8min
# [CV 5/5] END .C=0.1, gamma=scale, kernel=linear;, score=0.587 total time= 1.8min
# [truncated]
# [CV 4/5] END C=1000, gamma=0.0001, kernel=linear;, score=0.917 total time= 1.0min
# [CV 5/5] END C=1000, gamma=0.0001, kernel=linear;, score=0.933 total time=  56.5s
#
# support vector machine hyperparameter tuning: 2479.4304718750063 seconds
# {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# best score:  0.9321131447587355
#
# support vector machine (tuned) test set eval time: 8.229837332997704 seconds
#                       precision    recall  f1-score   support
#
#        are_you_a_bot     1.0000    0.9444    0.9714        18
#           calculator     0.9394    0.9394    0.9394        33
#                 date     1.0000    0.9655    0.9825        29
#           definition     1.0000    0.9688    0.9841        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.7727    0.9714    0.8608        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.8571    0.9677    0.9091        31
#                maybe     0.8462    0.9167    0.8800        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.9412    0.9143    0.9275        35
#             reminder     0.9259    0.9615    0.9434        26
#      reminder_update     0.9444    0.9714    0.9577        35
#        shopping_list     0.8824    1.0000    0.9375        30
# shopping_list_update     1.0000    0.8621    0.9259        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     1.0000    0.9630    0.9811        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    0.9333    0.9655        30
#                timer     1.0000    0.9000    0.9474        30
#            todo_list     1.0000    0.9200    0.9583        25
#     todo_list_update     0.9688    0.9688    0.9688        32
#              traffic     1.0000    0.9500    0.9744        40
#            translate     0.9583    0.9200    0.9388        25
#              weather     0.9286    1.0000    0.9630        26
#    what_is_your_name     0.8710    0.9643    0.9153        28
#         who_made_you     0.9714    0.9189    0.9444        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     0.9000    0.8182    0.8571        33
#
#             accuracy                         0.9512       902
#            macro avg     0.9559    0.9515    0.9523       902
#         weighted avg     0.9552    0.9512    0.9518       902
# ```


# %%
# save model and metadata

PICKLE_PROTOCOL = 5
PICKLE_FILE_EXTENSIONS = ('.p', '.pkl')

def save_model(model: BaseEstimator, model_path: Path | str) -> Path:
    '''
    Saves model to the given path with timestamp in filename
    Saves metadata (python and library versions) in a text file alongside model
    '''
    timestamp = int(time.time())
    model_path = Path(model_path)
    if model_path.suffix not in PICKLE_FILE_EXTENSIONS:
        raise Exception(f'Model file must end in one of these: {PICKLE_FILE_EXTENSIONS}')
    if not re.search(r'\d{10}$', model_path.stem):
        # add timestamp to path if one isn't present already
        model_path = model_path.parent / f'{model_path.stem}_{timestamp}{model_path.suffix}'
    with model_path.open('wb') as f:
        joblib.dump(model, f, protocol=PICKLE_PROTOCOL)
    print(f'saved model to {model_path}')
    # save metadata alongside model
    python = [sys.executable, '-VV']
    pip = ['uv', 'pip', 'freeze']
    results = [subprocess.run(command, capture_output=True, text=True) for command in (python, pip)]
    metadata_path = model_path.parent / f'{model_path.stem}.txt'
    with metadata_path.open('w+') as f:
        f.write('\n'.join([re.sub(r'\033\[[0-9;]*m', '', r.stdout) for r in results]))
    print(f'saved metadata to {metadata_path}')
    return model_path

# save model with highest accuracy (logistic regression) and predictions
df_pred = pd.DataFrame({'prediction': y_pred_lr})
df_pred.to_json(DATA_DIR / 'y_pred_lr.jsonl', orient='records', lines=True)

saved_model_path = save_model(lr, MODEL_DIR / f'model_{timestamp}.pkl')

# %% [markdown]
# ```text
# saved model...
# saved metadata...
# ```
