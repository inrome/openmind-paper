import pandas as pd

import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
participants_betas = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas.csv'))
trials = pd.read_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv'))

# see if mean of response_correct_mm corresponds to max beta
participants_accuracy = trials.groupby(['participant_id', 'test_condition', 'trial_type', 'feedback_condition'])['response_correct_mm'].mean().reset_index()
participants_accuracy = participants_accuracy.rename(columns = {'test_condition': 'task', 'response_correct_mm': 'accuracy'})

# add AN data
participants_accuracy_an = trials.groupby(['participant_id', 'test_condition', 'trial_type', 'feedback_condition'])['response_correct_mm_an'].mean().reset_index()
participants_accuracy_an = participants_accuracy_an.rename(columns = {'test_condition': 'task', 'response_correct_mm_an': 'accuracy'})

# filter out nan values
participants_accuracy_an = participants_accuracy_an[~participants_accuracy_an['accuracy'].isna()]
participants_accuracy_an['trial_type'] = 'hidden_an'

# concatenate participants_accuracy and participants_accuracy_an
participants_accuracy = pd.concat([participants_accuracy, participants_accuracy_an], ignore_index = True)

# merge with participants_beta
participants = pd.merge(participants_betas, participants_accuracy, on = ['participant_id', 'task', 'trial_type', 'feedback_condition'])

# check if accurcy correlates with max beta
import scipy.stats as stats
for task in participants['task'].unique():
    for trial_type in ['visible', 'hidden', 'hidden_an']:
        print(task, trial_type)
        corr = stats.spearmanr(participants[(participants['task'] == task) & (participants['trial_type'] == trial_type)]['accuracy'],
            participants[(participants['task'] == task) & (participants['trial_type'] == trial_type)]['max_beta'], nan_policy = 'omit')
        # round to 3 decimals
        corr = [round(c, 3) for c in corr]
        print(corr)
        

# check if calculations of response_correct_mm_an for a single participant are correct
single_trials = trials[(trials['participant_id'] == 199) & (trials['test_condition'] == 'explanation') & (trials['trial_type'] == 'hidden')]
single_betas = participants[(participants['participant_id'] == 199) & (participants['task'] == "explanation") & (participants['trial_type'] == 'hidden_an')]

def compute_log_p_answer(response_correct, r_i, beta):
    ''' Returns log_p_answer for a given trial and beta value.'''
    import math
    p_correct = (math.e ** (r_i * beta)) / (math.e ** (r_i * beta) + math.e ** (-1 * r_i * beta))

    assert p_correct >= 0 and p_correct <= 1 

    if response_correct in [0, 1]:
        p_answer = p_correct if response_correct == 1 else (1 - p_correct)
        log_p_answer = math.log(p_answer)

    else: # if response_correct is nan
        log_p_answer = np.nan # return nan

    return log_p_answer

log_p_answers = {}
for beta in np.arange(-1, 0.8, 0.005):
    log_p_answers[beta] = []
    for index, row in single_trials.iterrows():
        log_p_answer = compute_log_p_answer(row['response_correct_mm_an'], row['R_i_mm_an_eps_0.1'], beta)
        log_p_answers[beta].append(log_p_answer)
    
# sum log_p_answers for each beta ignoring nan values
sum_log_p_answers = {}
for beta in log_p_answers.keys():
    sum_log_p_answers[beta] = np.nansum(log_p_answers[beta])




