import pickle
import numpy as np
import pandas as pd
import os

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
test_trials = pd.read_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs_Ri.csv'))

# calculate correct_mm, correct_mm_an and response_correct_mm and response_correct_mm_an for each trial

# add empty columns
test_trials['correct_mm'] = np.nan
test_trials['correct_mm_an'] = np.nan
test_trials['response_correct_mm'] = np.nan
test_trials['response_correct_mm_an'] = np.nan

# define function to compute correct_mm
def compute_correct(option_1_p, option_2_p, task):
    if option_1_p != option_2_p:
        if task == 'explanation':
            option_1 = 1
            option_2 = 2
            correct_mm = option_1 if option_1_p < option_2_p else option_2
        else:
            option_1 = row['option_1']
            option_2 = row['option_2']
            correct_mm = option_1 if option_1_p > option_2_p else option_2
    else:
        correct_mm = None

    return correct_mm

# compute correct_mm for each trial
for index, row in test_trials.iterrows():
    test_trials.loc[index, 'correct_mm'] = compute_correct(row['option_1_p_mm'], row['option_2_p_mm'], row['test_condition'])
    test_trials.loc[index, 'correct_mm_an'] = compute_correct(row['option_1_p_mm_an'], row['option_2_p_mm_an'], row['test_condition']) if pd.isna(row['option_1_p_mm_an']) == False else np.nan

# define function to compute response_correct_mm
def compute_response_correct(response, correct_mm):
    if correct_mm is None:
        response_correct_mm = np.nan
    else:
        # if response is a string, convert correct_mm to string
        if isinstance(response, str):
            response_correct_mm = 1 if response == str(correct_mm) else 0 
        else:
            response_correct_mm = 1 if int(response) == int(correct_mm) else 0

    return response_correct_mm

# compute response_correct_mm for each trial
for index, row in test_trials.iterrows():
    test_trials.loc[index, 'response_correct_mm'] = compute_response_correct(row['response'], row['correct_mm'])
    test_trials.loc[index, 'response_correct_mm_an'] = compute_response_correct(row['response'], row['correct_mm_an']) if pd.isna(row['correct_mm_an']) == False else np.nan

# save data
test_trials.to_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv'), index=False)