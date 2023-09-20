# see of participants prefer one answer option over the other

import pandas as pd
import os
import numpy as np
import scipy.stats as stats

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
trials = pd.read_csv(os.path.join(current_dir, '../outputs/all_trials_with_Ri.csv'))

# add option_1 == "1" and option_2 == "2" for explanation and prediction
trials['option_1'] = np.where(trials['test_condition'] == 'explanation', "1", trials['option_1'])
trials['option_2'] = np.where(trials['test_condition'] == 'explanation', "2", trials['option_2'])

# add response bias column
trials['response_option2'] = np.where(trials['response'] == trials['option_2'], 1, 0)

# calculate mean response bias per participant
response_bias = trials.groupby(['participant_id', 'fsm_type', 'test_condition']).agg({'response_option2': ['mean']})
response_bias.columns = ['response_bias'] # remove multiindex
response_bias = response_bias.reset_index()

# descriptive statistics by condition and trial_type
descriptives = response_bias.groupby(['test_condition']).agg({'response_bias': ['mean', 'std', 'count']})
descriptives.columns = ['response_bias_mean', 'response_bias_sd', 'n']
descriptives = descriptives.reset_index()

# round to 2 decimals
descriptives = descriptives.round(2)
descriptives

# t-tests to see if the response bias is significantly different from 0.5
from scipy.stats import ttest_1samp

# compare response bias to 0.5
for condition in response_bias.test_condition.unique():
    print(condition)
    t_tests = ttest_1samp(response_bias.loc[(response_bias['test_condition'] == condition), 'response_bias'], 0.5)
    
    print(t_tests)
    # print t-test results t rounded to 2 decimals and p rounded to 3 decimals
    print('t = ' + str(round(t_tests[0], 2)) + ', p = ' + str(round(t_tests[1], 3)))

    # df for t-test
    df = response_bias.loc[(response_bias['test_condition'] == condition), 'response_bias'].count() - 1
    print('df = ' + str(df))