# Calculate beta values for each participant for each condition, find max beta value and export to csv in wide format (one row per participant)

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
trials = pd.read_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv'))

# define function to compute log_p_answer for a given trial and beta value
def compute_log_p_answer(task, response_correct, r_i, beta):
    ''' Returns log_p_answer for a given trial and beta value.'''
    import math
    p_correct = (math.e ** (r_i * beta)) / (math.e ** (r_i * beta) + math.e ** (-1 * r_i * beta))

    assert p_correct >= 0 and p_correct <= 1 and beta >= 0

    if response_correct in [0, 1]:
        p_answer = p_correct if response_correct == 1 else (1 - p_correct)
        log_p_answer = math.log(p_answer)

    else: # if response_correct is nan
        log_p_answer = np.nan # return nan

    return log_p_answer

# function to generate log_p_answers, sum them and return max beta value for a participant
def get_max_beta_values(trials_participant, beta_max = 0.8, beta_step = 0.005):
    ''' Returns dict[task][type] with max beta values for which the sum of log_p_answers is highest.'''
    # check inputs
    assert beta_max > 0 and beta_step > 0 and beta_max > beta_step and \
        trials_participant['participant_id'].nunique() == 1
    
    log_p_answers = {}

    def calculate_log_p_answers(task, trial_type, response_correct_column, R_i_column):
        trials = trials_participant[(trials_participant['test_condition'] == task) & \
                                        (trials_participant['trial_type'] == trial_type)] # subset for task and trial type
        log_p = {}
        for beta in np.arange(0, beta_max, beta_step):
            log_p[beta] = [compute_log_p_answer(task, trial[response_correct_column], trial[R_i_column], beta)
                        for _, trial in trials.iterrows()]
        return log_p

    for task in trials['test_condition'].unique():
        log_p_answers[task] = {}
        log_p_answers[task]['visible'] = calculate_log_p_answers(task, 'visible', 'response_correct_mm', 'R_i_mm_eps_0.1')
        log_p_answers[task]['hidden'] = calculate_log_p_answers(task, 'hidden', 'response_correct_mm', 'R_i_mm_eps_0.1')
        log_p_answers[task]['hidden_an'] = calculate_log_p_answers(task, 'hidden', 'response_correct_mm_an', 'R_i_mm_an_eps_0.1')

    # sum log_p_answers for each beta value
    sum_log_p_answers = {}
    max_beta = {}
    for task in trials['test_condition'].unique():
        sum_log_p_answers[task] = {}
        max_beta[task] = {}
        for trial_type in ['visible', 'hidden', 'hidden_an']:
            sum_log_p_answers[task][trial_type] = {}
            for beta in np.arange(0, beta_max, beta_step):
                # if log_p_answers[task][trial_type][beta] contains 3 or more nans, set sum to nan
                if sum(np.isnan(log_p_answers[task][trial_type][beta])) >= 3:
                    sum_log_p_answers[task][trial_type][beta] = np.nan
                else: # if there are enough non-nan values, sum them:
                    sum_log_p_answers[task][trial_type][beta] =  np.nansum(log_p_answers[task][trial_type][beta])
            
            # find  max_beta corresponding to
            max_beta[task][trial_type] = max(sum_log_p_answers[task][trial_type], key=sum_log_p_answers[task][trial_type].get)

            if sum(np.isnan(log_p_answers[task][trial_type][0])) >= 3:
                max_beta[task][trial_type] = np.nan

    return max_beta
            

# get max beta values for each participant
max_beta_values = pd.DataFrame(columns = ['participant_id', 'fsm_type', 'task', 'test_order', 'trial_type', 'max_beta'])
for participant_id in trials['participant_id'].unique():
    trials_participant = trials[trials['participant_id'] == participant_id]
    fsm_type = trials_participant['fsm_type'].unique()[0]
    max_beta = get_max_beta_values(trials_participant)
    test_order = trials_participant['test_order'].unique()[0]

    # append to dataframe
    for task in trials['test_condition'].unique():
        for trial_type in ['visible', 'hidden', 'hidden_an']:
            new_row = {'participant_id': participant_id,'fsm_type':fsm_type, 'task': task, 'test_order': test_order, 'trial_type': trial_type, 'max_beta': max_beta[task][trial_type]}
            max_beta_values = pd.concat([max_beta_values, pd.DataFrame([new_row])])
            
# export to csv
max_beta_values.to_csv(os.path.join(current_dir, '../outputs/participants_max_betas.csv'), index = False)

# Pivot the dataframe to create a wide format
wide_format_df = max_beta_values.pivot_table(index=['participant_id', 'fsm_type', 'test_order'],
                                             columns=['task', 'trial_type'],
                                             values='max_beta',
                                             aggfunc='first').reset_index()

# Flatten the MultiIndex columns
wide_format_df.columns = ['_'.join(col).strip() for col in wide_format_df.columns.values]
wide_format_df.rename(columns={'participant_id_': 'participant_id', 'fsm_type_':'fsm_type', 'test_order_':'test_order'}, inplace=True)

# Export to CSV
wide_format_csv_path = os.path.join(current_dir, '../outputs/participants_max_betas_wide.csv')
wide_format_df.to_csv(wide_format_csv_path, index=False)
