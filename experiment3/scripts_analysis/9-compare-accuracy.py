# Table with M and SD FSM (normative) accuracy by condition

import pandas as pd
import numpy as np
import os

# load data for Experiments 1 and 2
current_dir = os.path.dirname(os.path.abspath(__file__))
exp1n2_path = os.path.join(current_dir, '../../experiments1and2/data/test_PEC.csv')
trials_exp1n2 = pd.read_csv(exp1n2_path)

exp3_path = os.path.join(current_dir, '../data/trials_test.csv')
trials_exp3 = pd.read_csv(exp3_path)


# aggregate data by participant and test_condition (mean response_correct)
def aggregate_data(df):
    df = df.groupby(['participant_id', 'test_condition']).agg({'response_correct': ['mean']})
    df.columns = ['response_correct'] # remove multiindex
    df = df.reset_index() 
    return df

# aggregate data for Experiments 1 and 2
participants_exp1n2 = aggregate_data(trials_exp1n2)
participants_exp1n2['experiment'] = 'exp1and2'

# aggregate data for Experiment 3
participants_exp3 = aggregate_data(trials_exp3)
participants_exp3['experiment'] = 'exp3'

# concatenate dataframes
participants = pd.concat([participants_exp1n2, participants_exp3])

# compute mean and SD accuracy by condition (and add n per condition)
descriptives = participants.groupby(['experiment', 'test_condition']).agg({'response_correct': ['mean', 'std', 'count']})
descriptives.columns = ['accuracy_mean', 'accuracy_sd', 'n']
descriptives = descriptives.reset_index()

descriptives = descriptives.round(2)
descriptives

# t-tests to compare accuracy between Experiments 1 and 2 and Experiment 3 (by condition)
from scipy.stats import ttest_ind

# compare accuracy between Experiments 1 and 2 and Experiment 3 (by condition)
for condition in participants.test_condition.unique():
    print(condition)
    t_tests = ttest_ind(participants.loc[(participants['experiment'] == 'exp1and2') & (participants['test_condition'] == condition), 'response_correct'],
                    participants.loc[(participants['experiment'] == 'exp3') & (participants['test_condition'] == condition), 'response_correct'],
                    equal_var=True)
    
    print(t_tests)
    # print t-test results t rounded to 2 decimals and p rounded to 3 decimals
    print('t = ' + str(round(t_tests[0], 2)) + ', p = ' + str(round(t_tests[1], 3)))

    # df for t-test


