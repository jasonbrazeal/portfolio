# %% [markdown]
# ## Full results - classification reports
#
# %% [markdown]
# ### anthropic.k_shot_cot_prompt10
# accuracy = 0.9379
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9000    1.0000    0.9474        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.8889    1.0000    0.9412        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9667    0.9355    0.9508        31
#                maybe     0.7500    1.0000    0.8571        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     1.0000    0.8286    0.9062        35
#             reminder     0.4068    0.9231    0.5647        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9677    1.0000    0.9836        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9600    0.9600    0.9600        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9189    0.9577        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9091    0.9524        33
#             accuracy                         0.9379       902
#            macro avg     0.9246    0.9445    0.9299       902
#         weighted avg     0.9250    0.9379    0.9274       902
# ```
# %% [markdown]
# ### anthropic.k_shot_cot_prompt30
# accuracy = 0.9812
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     0.9706    1.0000    0.9851        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9412    1.0000    0.9697        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9677    0.9677    0.9677        31
#                maybe     0.9200    0.9583    0.9388        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.9722    1.0000    0.9859        35
#             reminder     1.0000    0.9231    0.9600        26
#      reminder_update     0.9722    1.0000    0.9859        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     0.9655    0.9655    0.9655        29
#             spelling     1.0000    0.9310    0.9643        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9231    0.9600    0.9412        25
#     todo_list_update     0.9688    0.9688    0.9688        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9394    0.9688        33
#             accuracy                         0.9812       902
#            macro avg     0.9804    0.9805    0.9801       902
#         weighted avg     0.9818    0.9812    0.9811       902
# ```
# %% [markdown]
# ### anthropic.k_shot_prompt10
# accuracy = 0.9357
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9000    1.0000    0.9474        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9143    1.0000    0.9552        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9667    0.9355    0.9508        31
#                maybe     0.7059    1.0000    0.8276        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     1.0000    0.7714    0.8710        35
#             reminder     0.4000    0.9231    0.5581        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9677    1.0000    0.9836        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    0.9667    0.9831        30
#            todo_list     0.9615    1.0000    0.9804        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9189    0.9577        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9091    0.9524        33
#             accuracy                         0.9357       902
#            macro avg     0.9238    0.9428    0.9281       902
#         weighted avg     0.9246    0.9357    0.9255       902
# ```
# %% [markdown]
# ### anthropic.k_shot_prompt30
# accuracy = 0.9845
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9412    1.0000    0.9697        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9677    0.9677    0.9677        31
#                maybe     0.9231    1.0000    0.9600        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     1.0000    1.0000    1.0000        35
#             reminder     1.0000    0.9231    0.9600        26
#      reminder_update     0.9722    1.0000    0.9859        35
#        shopping_list     1.0000    0.9667    0.9831        30
# shopping_list_update     0.9667    1.0000    0.9831        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9231    0.9600    0.9412        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9310    0.9643    0.9474        28
#         who_made_you     0.9722    0.9459    0.9589        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9394    0.9688        33
#             accuracy                         0.9845       902
#            macro avg     0.9836    0.9842    0.9836       902
#         weighted avg     0.9851    0.9845    0.9845       902
# ```
# %% [markdown]
# ### anthropic.zero_shot_cot_prompt
# accuracy = 0.9313
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9412    1.0000    0.9697        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9394    0.9688        33
#             greeting     0.9032    0.9032    0.9032        31
#                maybe     0.7273    1.0000    0.8421        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     1.0000    0.8000    0.8889        35
#             reminder     0.3898    0.8846    0.5412        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9677    1.0000    0.9836        30
# shopping_list_update     0.9667    1.0000    0.9831        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9231    0.9600    0.9412        25
#     todo_list_update     1.0000    0.9062    0.9508        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9310    0.9643    0.9474        28
#         who_made_you     0.9211    0.9459    0.9333        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9394    0.9688        33
#             accuracy                         0.9313       902
#            macro avg     0.9194    0.9376    0.9239       902
#         weighted avg     0.9192    0.9313    0.9211       902
# ```
# %% [markdown]
# ### anthropic.zero_shot_prompt
# accuracy = 0.9302
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     1.0000    1.0000    1.0000        18
#           calculator     0.9706    1.0000    0.9851        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9143    1.0000    0.9552        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     0.9688    1.0000    0.9841        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9355    0.9355    0.9355        31
#                maybe     0.7667    0.9583    0.8519        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     1.0000    0.8286    0.9062        35
#             reminder     0.3684    0.8077    0.5060        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     1.0000    1.0000    1.0000        30
# shopping_list_update     0.9667    1.0000    0.9831        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     0.9643    1.0000    0.9818        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.8276    0.9600    0.8889        25
#     todo_list_update     1.0000    0.9062    0.9508        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9310    0.9643    0.9474        28
#         who_made_you     0.9211    0.9459    0.9333        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9091    0.9524        33
#             accuracy                         0.9302       902
#            macro avg     0.9178    0.9357    0.9229       902
#         weighted avg     0.9171    0.9302    0.9202       902
# ```
# %% [markdown]
# ### google.k_shot_cot_prompt10
# accuracy = 0.9468
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.8571    1.0000    0.9231        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9697    1.0000    0.9846        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.9722    1.0000    0.9859        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.9655    0.9032    0.9333        31
#                maybe     1.0000    1.0000    1.0000        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     1.0000    1.0000    1.0000        35
#             reminder     0.3934    0.9231    0.5517        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    0.9667    0.9831        30
#            todo_list     0.9615    1.0000    0.9804        25
#     todo_list_update     1.0000    0.9375    0.9677        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     1.0000    0.9643    0.9818        28
#         who_made_you     0.9737    1.0000    0.9867        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    1.0000    1.0000        33
#             accuracy                         0.9468       902
#            macro avg     0.9343    0.9517    0.9391       902
#         weighted avg     0.9332    0.9468    0.9366       902
# ```
# %% [markdown]
# ### google.k_shot_cot_prompt30
# accuracy = 0.9845
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9697    1.0000    0.9846        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.9688    1.0000    0.9841        31
#                maybe     1.0000    0.8333    0.9091        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     0.8974    1.0000    0.9459        35
#             reminder     0.9615    0.9615    0.9615        26
#      reminder_update     0.9714    0.9714    0.9714        35
#        shopping_list     1.0000    0.9667    0.9831        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9615    1.0000    0.9804        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9600    0.9796        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9310    0.9643    0.9474        28
#         who_made_you     0.9722    0.9459    0.9589        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    1.0000    1.0000        33
#             accuracy                         0.9845       902
#            macro avg     0.9850    0.9838    0.9840       902
#         weighted avg     0.9851    0.9845    0.9844       902
# ```
# %% [markdown]
# ### google.k_shot_prompt10
# accuracy = 0.9424
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.8571    1.0000    0.9231        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9412    1.0000    0.9697        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9667    0.9355    0.9508        31
#                maybe     0.9545    0.8750    0.9130        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.9189    0.9714    0.9444        35
#             reminder     0.4032    0.9615    0.5682        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9355    0.9667    0.9508        30
# shopping_list_update     1.0000    0.9655    0.9825        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    0.9667    0.9831        30
#            todo_list     1.0000    1.0000    1.0000        25
#     todo_list_update     1.0000    0.9375    0.9677        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9600    0.9796        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     1.0000    0.9643    0.9818        28
#         who_made_you     0.9737    1.0000    0.9867        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     1.0000    0.9697    0.9846        33
#             accuracy                         0.9424       902
#            macro avg     0.9305    0.9471    0.9346       902
#         weighted avg     0.9293    0.9424    0.9322       902
# ```
# %% [markdown]
# ### google.k_shot_prompt30
# accuracy = 0.9889
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     1.0000    1.0000    1.0000        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9394    1.0000    0.9688        31
#                maybe     1.0000    0.9583    0.9787        24
#      meaning_of_life     1.0000    1.0000    1.0000        37
#                   no     0.9722    1.0000    0.9859        35
#             reminder     0.9615    0.9615    0.9615        26
#      reminder_update     0.9714    0.9714    0.9714        35
#        shopping_list     1.0000    0.9667    0.9831        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9600    0.9600    0.9600        25
#     todo_list_update     0.9688    0.9688    0.9688        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    1.0000    1.0000        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9643    0.9643    0.9643        28
#         who_made_you     0.9730    0.9730    0.9730        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9697    0.9846        33
#             accuracy                         0.9889       902
#            macro avg     0.9886    0.9888    0.9886       902
#         weighted avg     0.9891    0.9889    0.9889       902
# ```
# %% [markdown]
# ### google.zero_shot_cot_prompt
# accuracy = 0.9124
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.7826    1.0000    0.8780        18
#           calculator     1.0000    0.9697    0.9846        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9143    1.0000    0.9552        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.9722    1.0000    0.9859        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.9615    0.8065    0.8772        31
#                maybe     0.8800    0.9167    0.8980        24
#      meaning_of_life     0.9737    1.0000    0.9867        37
#                   no     0.9429    0.9429    0.9429        35
#             reminder     0.2885    0.5769    0.3846        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9310    0.9000    0.9153        30
# shopping_list_update     0.9333    0.9655    0.9492        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    0.9667    0.9831        30
#            todo_list     0.7188    0.9200    0.8070        25
#     todo_list_update     0.9231    0.7500    0.8276        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9655    1.0000    0.9825        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     1.0000    0.9697    0.9846        33
#            micro avg     0.9134    0.9124    0.9129       902
#            macro avg     0.9052    0.9163    0.9076       902
#         weighted avg     0.9073    0.9124    0.9071       902
# ```
# %% [markdown]
# ### google.zero_shot_prompt
# accuracy = 0.9335
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.8571    1.0000    0.9231        18
#           calculator     0.9706    1.0000    0.9851        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9697    1.0000    0.9846        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     0.9722    1.0000    0.9859        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9630    0.8387    0.8966        31
#                maybe     0.9231    1.0000    0.9600        24
#      meaning_of_life     0.9487    1.0000    0.9737        37
#                   no     1.0000    0.9714    0.9855        35
#             reminder     0.3077    0.6154    0.4103        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     1.0000    0.9667    0.9831        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.8571    0.9600    0.9057        25
#     todo_list_update     1.0000    0.9375    0.9677        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9600    0.9796        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     1.0000    0.9643    0.9818        28
#         who_made_you     0.9737    1.0000    0.9867        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9697    0.9846        33
#             accuracy                         0.9335       902
#            macro avg     0.9236    0.9373    0.9281       902
#         weighted avg     0.9236    0.9335    0.9265       902
# ```
# %% [markdown]
# ### openai.k_shot_cot_prompt10
# accuracy = 0.9346
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.8182    1.0000    0.9000        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.8889    1.0000    0.9412        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9600    0.7742    0.8571        31
#                maybe     0.7059    1.0000    0.8276        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     1.0000    0.9714    0.9855        35
#             reminder     0.4167    0.9615    0.5814        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     1.0000    1.0000    1.0000        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     1.0000    0.8182    0.9000        33
#             accuracy                         0.9346       902
#            macro avg     0.9218    0.9413    0.9259       902
#         weighted avg     0.9234    0.9346    0.9241       902
# ```
# %% [markdown]
# ### openai.k_shot_cot_prompt30
# accuracy = 0.9856
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     1.0000    1.0000    1.0000        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9412    1.0000    0.9697        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    0.9714    0.9855        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9394    1.0000    0.9688        31
#                maybe     0.8519    0.9583    0.9020        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     0.9722    1.0000    0.9859        35
#             reminder     1.0000    0.9615    0.9804        26
#      reminder_update     0.9722    1.0000    0.9859        35
#        shopping_list     1.0000    0.9667    0.9831        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     0.9615    1.0000    0.9804        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9600    0.9796        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9091    0.9524        33
#             accuracy                         0.9856       902
#            macro avg     0.9857    0.9861    0.9855       902
#         weighted avg     0.9866    0.9856    0.9857       902
# ```
# %% [markdown]
# ### openai.k_shot_prompt10
# accuracy = 0.9302
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.8182    1.0000    0.9000        18
#           calculator     0.9706    1.0000    0.9851        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.8889    1.0000    0.9412        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9583    0.7419    0.8364        31
#                maybe     0.6667    1.0000    0.8000        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     1.0000    0.9714    0.9855        35
#             reminder     0.4098    0.9615    0.5747        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    0.9655    0.9825        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     0.9310    1.0000    0.9643        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     1.0000    0.9600    0.9796        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     1.0000    0.7879    0.8814        33
#             accuracy                         0.9302       902
#            macro avg     0.9181    0.9367    0.9211       902
#         weighted avg     0.9200    0.9302    0.9195       902
# ```
# %% [markdown]
# ### openai.k_shot_prompt30
# accuracy = 0.9845
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     1.0000    1.0000    1.0000        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.9143    1.0000    0.9552        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    0.9714    0.9855        35
#              goodbye     1.0000    0.9697    0.9846        33
#             greeting     0.9394    1.0000    0.9688        31
#                maybe     0.8519    0.9583    0.9020        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     0.9722    1.0000    0.9859        35
#             reminder     1.0000    0.9615    0.9804        26
#      reminder_update     0.9722    1.0000    0.9859        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     1.0000    1.0000    1.0000        25
#     todo_list_update     1.0000    0.9688    0.9841        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.9200    0.9583        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9333    1.0000    0.9655        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    1.0000    1.0000        33
#                  yes     1.0000    0.9091    0.9524        33
#             accuracy                         0.9845       902
#            macro avg     0.9850    0.9848    0.9845       902
#         weighted avg     0.9856    0.9845    0.9846       902
# ```
# %% [markdown]
# ### openai.zero_shot_cot_prompt
# accuracy = 0.9202
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.8182    1.0000    0.9000        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.8649    1.0000    0.9275        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.9231    0.7742    0.8421        31
#                maybe     0.5714    1.0000    0.7273        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     1.0000    0.9429    0.9706        35
#             reminder     0.4032    0.9615    0.5682        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     0.9667    1.0000    0.9831        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     0.9643    1.0000    0.9818        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     1.0000    0.9600    0.9796        25
#     todo_list_update     1.0000    0.9375    0.9677        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.8400    0.9130        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9310    0.9643    0.9474        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     1.0000    0.6061    0.7547        33
#             accuracy                         0.9202       902
#            macro avg     0.9126    0.9270    0.9114       902
#         weighted avg     0.9150    0.9202    0.9101       902
# ```
# %% [markdown]
# ### openai.zero_shot_prompt
# accuracy = 0.9213
# ```text
#                       precision    recall  f1-score   support
#        are_you_a_bot     0.9474    1.0000    0.9730        18
#           calculator     1.0000    1.0000    1.0000        33
#                 date     1.0000    1.0000    1.0000        29
#           definition     0.8889    1.0000    0.9412        32
#           find_phone     1.0000    1.0000    1.0000        22
#            flip_coin     1.0000    1.0000    1.0000        31
# food_beverage_recipe     1.0000    1.0000    1.0000        35
#              goodbye     0.9697    0.9697    0.9697        33
#             greeting     0.9231    0.7742    0.8421        31
#                maybe     0.5106    1.0000    0.6761        24
#      meaning_of_life     1.0000    0.9730    0.9863        37
#                   no     1.0000    0.9429    0.9706        35
#             reminder     0.4032    0.9615    0.5682        26
#      reminder_update     0.0000    0.0000    0.0000        35
#        shopping_list     0.9667    0.9667    0.9667        30
# shopping_list_update     1.0000    1.0000    1.0000        29
#             spelling     1.0000    1.0000    1.0000        29
#            tell_joke     1.0000    1.0000    1.0000        27
#                 text     1.0000    1.0000    1.0000        27
#                 time     1.0000    1.0000    1.0000        30
#                timer     1.0000    1.0000    1.0000        30
#            todo_list     1.0000    1.0000    1.0000        25
#     todo_list_update     1.0000    0.9375    0.9677        32
#              traffic     1.0000    1.0000    1.0000        40
#            translate     1.0000    0.8800    0.9362        25
#              weather     1.0000    1.0000    1.0000        26
#    what_is_your_name     0.9310    0.9643    0.9474        28
#         who_made_you     1.0000    0.9459    0.9722        37
#      word_of_the_day     1.0000    0.9697    0.9846        33
#                  yes     1.0000    0.5758    0.7308        33
#             accuracy                         0.9213       902
#            macro avg     0.9180    0.9287    0.9144       902
#         weighted avg     0.9190    0.9213    0.9121       902
# ```
