# %% [markdown]
# ## Exploratory Data Analyis
#
# The plan is to:
#
# * fetch the data from a GitHub repository and perform some initial exploration
# * extract a subset of the intents needed to train the model for our use case and take a closer look


# %%
import requests

from IPython.display import display
from pandas import DataFrame

from utils import (
    init_nb, RAW_DATA_URL, RAW_DATA_PATH, FILTERED_DATA_PATH
)

init_nb()


# %%
# fetch raw data and inspect json

response = requests.get(RAW_DATA_URL)
response.raise_for_status()
with open(RAW_DATA_PATH, 'w') as f:
    f.write(response.text)

dataset_raw = response.json()
print(type(dataset_raw))
# data is in a Python dict

print(dataset_raw.keys())
# intent data is pre-split into test, train, and validation sets with
# out-of-scope examples (i.e. data that does not fall into any of the intent classes)

print(type(dataset_raw['train']))
print(type(dataset_raw['train'][0]))
print(dataset_raw['train'][0])
# training set is a list of lists
# each data point is a list with the format [utterance, label]

# %% [markdown]
# ```text
# <class 'dict'>
# dict_keys(['oos_val', 'val', 'train', 'oos_test', 'test', 'oos_train'])
# <class 'list'>
# <class 'list'>
# ['what expression would i use to say i love you if i were an italian', 'translate']
# ```


# %%
# load and inspect data

# combine test, train, and validation sets into one dataframe, ignoring out-of-scope examples
df_orig: DataFrame = DataFrame(
    dataset_raw['test'] + dataset_raw['train'] + dataset_raw['val'],
    columns=('utterance', 'intent_str'),
)
# df will contain our "working copy" of our dataframe and will be modified throughout the notebook
df = df_orig.copy()

# explore data as a whole
display(df.head())
display(df.describe())
# 22500 rows, 2 columns (utterance and intent_str)
# no missing values, but not all rows are unique => deduplication needed
# 150 unique intents

display(df.intent_str.value_counts())
# every intent class has 150 rows (output of value_counts() is sorted descending by frequency)
# 150 rows * 150 intents = 22500 total rows √
# => the intent classes are balanced, so no need to collect more data, resample, etc.

# %% [markdown]
# ```text
#                             utterance intent_str
# 0     how would you say fly in italian  translate
# 1    what's the spanish word for pasta  translate
# 2  how would they say butter in zambia  translate
# 3       how do you say fast in spanish  translate
# 4  what's the word for trees in norway  translate
#
#                     utterance intent_str
# count                   22500      22500
# unique                  22495        150
# top     where did you grow up  translate
# freq                        2        150
#
# intent_str
# translate          150
# order_status       150
# goodbye            150
# account_blocked    150
# what_song          150
#                  ...
# reminder           150
# change_speed       150
# tire_pressure      150
# no                 150
# card_declined      150
# Name: count, Length: 150, dtype: int64
# ```


# %%
# check out all intents and choose a couple to examine more closely

print(df.intent_str.unique())
# all 150 intents, we only need a subset of these
# => we will filter out those we don't need
# there are 2 intents we need to recognize that are not in this list
# => we will generate new utterances for these missing intents to augment the dataset

# explore a subsection of the data, want to keep an eye out for things like urls,
# email addresses, hashtags, @mentions, html tags, and other patterns in the text
# that might need special treatment
display(df[df.intent_str == 'no'].sample(20))
display(df[df.intent_str == 'what_is_your_name'].sample(20))
display(df[df.intent_str == 'calculator'].sample(20))
# this text has already been partially processed; it's lower case and most special characters are gone
# => we can skip these steps in our preprocessing

# %% [markdown]
# ```text
# ['translate' 'transfer' 'timer' 'definition' 'meaning_of_life'
#  'insurance_change' 'find_phone' 'travel_alert' 'pto_request'
#  'improve_credit_score' 'fun_fact' 'change_language' 'payday'
#  'replacement_card_duration' 'time' 'application_status' 'flight_status'
#  'flip_coin' 'change_user_name' 'where_are_you_from'
#  'shopping_list_update' 'what_can_i_ask_you' 'maybe' 'oil_change_how'
#  'restaurant_reservation' 'balance' 'confirm_reservation' 'freeze_account'
#  'rollover_401k' 'who_made_you' 'distance' 'user_name' 'timezone'
#  'next_song' 'transactions' 'restaurant_suggestion' 'rewards_balance'
#  'pay_bill' 'spending_history' 'pto_request_status' 'credit_score'
#  'new_card' 'lost_luggage' 'repeat' 'mpg' 'oil_change_when' 'yes'
#  'travel_suggestion' 'insurance' 'todo_list_update' 'reminder'
#  'change_speed' 'tire_pressure' 'no' 'apr' 'nutrition_info' 'calendar'
#  'uber' 'calculator' 'date' 'carry_on' 'pto_used' 'schedule_maintenance'
#  'travel_notification' 'sync_device' 'thank_you' 'roll_dice' 'food_last'
#  'cook_time' 'reminder_update' 'report_lost_card'
#  'ingredient_substitution' 'make_call' 'alarm' 'todo_list' 'change_accent'
#  'w2' 'bill_due' 'calories' 'damaged_card' 'restaurant_reviews' 'routing'
#  'do_you_have_pets' 'schedule_meeting' 'gas_type' 'plug_type'
#  'tire_change' 'exchange_rate' 'next_holiday' 'change_volume'
#  'who_do_you_work_for' 'credit_limit' 'how_busy' 'accept_reservations'
#  'order_status' 'pin_change' 'goodbye' 'account_blocked' 'what_song'
#  'international_fees' 'last_maintenance' 'meeting_schedule'
#  'ingredients_list' 'report_fraud' 'measurement_conversion' 'smart_home'
#  'book_hotel' 'current_location' 'weather' 'taxes' 'min_payment'
#  'whisper_mode' 'cancel' 'international_visa' 'vaccines' 'pto_balance'
#  'directions' 'spelling' 'greeting' 'reset_settings' 'what_is_your_name'
#  'direct_deposit' 'interest_rate' 'credit_limit_change'
#  'what_are_your_hobbies' 'book_flight' 'shopping_list' 'text'
#  'bill_balance' 'share_location' 'redeem_rewards' 'play_music'
#  'calendar_update' 'are_you_a_bot' 'gas' 'expiration_date'
#  'update_playlist' 'cancel_reservation' 'tell_joke' 'change_ai_name'
#  'how_old_are_you' 'car_rental' 'jump_start' 'meal_suggestion' 'recipe'
#  'income' 'order' 'traffic' 'order_checks' 'card_declined']
#
#                                           utterance intent_str
# 9843                 oh hell no, that'd be terrible!         no
# 1608                            that has to be false         no
# 9888                     what you just said is wrong         no
# 9848                           that's totally wrong!         no
# 9846                         that's completely false         no
# 9801                                 that's not true         no
# 9834           i do not believe that that is correct         no
# 20571                  i do not think that is proper         no
# 9838                                             no!         no
# 9852                                  false for sure         no
# 9880                          no that isn't the case         no
# 1602                          the statement is false         no
# 9806                                             nay         no
# 1599                           no, that is incorrect         no
# 1609                                that is so false         no
# 9826   i don't believe that is possible, it is false         no
# 9874                     that's not right it's false         no
# 9839                                that's not right         no
# 9825                         no, that is my response         no
# 20562                               no, that is fake         no
#
#                                             utterance         intent_str
# 16554                    what do you like being called  what_is_your_name
# 3620                            what do i call you, ai  what_is_your_name
# 16515        should i call you something in particular  what_is_your_name
# 16517          what name would you like me to call you  what_is_your_name
# 21919                             can i have your name  what_is_your_name
# 16584    when referring to you, what name should i use  what_is_your_name
# 21914                           do you have a nickname  what_is_your_name
# 16587                          ai, what can i call you  what_is_your_name
# 3609               can you tell me what you are called  what_is_your_name
# 16560                        can you tell me your name  what_is_your_name
# 21916                        what do you call yourself  what_is_your_name
# 3614                               what can i call you  what_is_your_name
# 16503                    can you tell me the ai's name  what_is_your_name
# 21911                   what's the name you were given  what_is_your_name
# 3613                       what is your preferred name  what_is_your_name
# 16547  do people call you by a certain name what is it  what_is_your_name
# 16516                    what should i refer to you as  what_is_your_name
# 3605                          how should i address you  what_is_your_name
# 3603                                  what's your name  what_is_your_name
# 3607                           what's your name anyway  what_is_your_name
#
#                                               utterance  intent_str
# 10324                                      what is 4 x 4  calculator
# 10341                      what is 250 times 118 times 9  calculator
# 10387                    what is the square root of 2784  calculator
# 10300                                      what is 7 x 7  calculator
# 10328                                        what is 2+2  calculator
# 10351                       what is 20 times 20 times 30  calculator
# 1750                               what is 1000 plus 745  calculator
# 10334                   what is the square root of 10294  calculator
# 10321               can you help me solve a math problem  calculator
# 1768          what is the solution to sixty times thirty  calculator
# 20661  i need your assistance to solve this math problem  calculator
# 20669                     what is the square root of 888  calculator
# 10340                                what is 78 times 85  calculator
# 20670                            what is 87 divided by 4  calculator
# 10350                   what is the square root of 10500  calculator
# 10394                          what's the answer to 5-6=  calculator
# 10392                 can you calculate 18 divided by 45  calculator
# 10325                                      what is 2 + 2  calculator
# 10346                           please help with my math  calculator
# 10309                       what is the sum of 10 plus 5  calculator
# ```


# %%
# filter and dedupe intent data

# these are the intents our personal assistant chatbot should recognize
intents: set = {
    'are_you_a_bot',
    'calculator',
    'date',
    'definition',
    'find_phone',
    'flip_coin',
    'goodbye',
    'greeting',
    'maybe',
    'meaning_of_life',
    'no',
    'food_beverage_recipe',
    'reminder_update',
    'reminder',
    'shopping_list_update',
    'shopping_list',
    'spelling',
    'tell_joke',
    'text',
    'time',
    'timer',
    'todo_list_update',
    'todo_list',
    'traffic',
    'translate',
    'weather',
    'what_is_your_name',
    'who_made_you',
    'word_of_the_day',
    'yes',
}
df_filtered = df[df['intent_str'].isin(list(intents))]
print(f'{len(df_filtered_intents := df_filtered.intent_str.unique())} unique intents:')
df_filtered_intents.sort()
print(df_filtered_intents)
missing_intents = intents - set(df_filtered_intents)
print(f'{len(missing_intents)} missing intents:\n{missing_intents}')
# 28/30 of our required intents were present in the CLINC-150 dataset
# we will handle the 2 missing intents in the next section

# dedupe
print(f'before deduping: {len(df_filtered)} rows')
df = df_filtered.drop_duplicates(subset=['utterance'], keep=False, ignore_index=True)
print(f'after deduping: {len(df)} rows')
# only 2 duplicate rows, so classes are still balanced

# write out deduped and filtered data
df.to_json(FILTERED_DATA_PATH, orient='records', lines=True)

# %% [markdown]
# ```text
# 28 unique intents:
# ['are_you_a_bot' 'calculator' 'date' 'definition' 'find_phone' 'flip_coin'
#  'goodbye' 'greeting' 'maybe' 'meaning_of_life' 'no' 'reminder'
#  'reminder_update' 'shopping_list' 'shopping_list_update' 'spelling'
#  'tell_joke' 'text' 'time' 'timer' 'todo_list' 'todo_list_update'
#  'traffic' 'translate' 'weather' 'what_is_your_name' 'who_made_you' 'yes']
# 2 missing intents:
# {'food_beverage_recipe', 'word_of_the_day'}
# before deduping: 4200 rows
# after deduping: 4196 rows
# ```
