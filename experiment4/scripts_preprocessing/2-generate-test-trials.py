# Generates test trials for all participants (test_PEC, test_prediction, test_control, test_explanation)

import pandas as pd
import os
import numpy as np


# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
trials_raw = pd.read_csv(os.path.join(current_dir, "..", "data_raw/trials_raw_anonymized.csv"), low_memory= False)
trials_learning = pd.read_csv(os.path.join(current_dir, "..", "data/trials_learning.csv"), low_memory= False)

# remove participants with no learning trials
trials_raw = trials_raw.loc[trials_raw['participant_id'].isin(trials_learning['participant_id'])]

# select only relevant columns
trials_raw = trials_raw[['participant_id', 'fsm_number', "test_condition", "transfer_order", "transfer_task_1", "transfer_task_2",
                    'attempt', 'control_hidden_counter', 'control_visible_counter', 
                    'prediction_hidden_counter', 'prediction_visible_counter', 
                    'explanation_hidden_counter', 'explanation_visible_counter',
                    'test_prediction_counter', 'test_control_counter', 'test_explanation_counter', 
                    'sender', 'duration', 'exp_type',
                    'state_1', 'response_1', 'state_2', 'response_2', 'state_3',
                    'option_1','option_2', 'option_1_p','option_2_p','correctResponse', 'response', 'correct', 'timestamp']]

# rename columns
trials_raw = trials_raw.rename(columns={"exp_type": "trial_type",
                                        "correct": "response_correct",
                                        "test_condition": "feedback_condition"})
# select columns containing counters
columns = pd.DataFrame(trials_raw.columns)
counter_columns = columns.loc[columns[0].str.contains("_counter")]

# get unique participant codes
participant_ids = trials_raw['participant_id'].unique()

# Create a function to check if any counter is not empty and response_correct is not NaN
def any_counter_not_empty(row):
    for column in counter_columns[0]:
        if row[column] >= 0 and not np.isnan(row['response_correct']):
            return column
    return np.nan

# add column with name of counter that is not empty
trials_raw['test_condition'] = trials_raw.apply(any_counter_not_empty, axis=1)

# find words "prediction", "control", "explanation" in column "test_condition" and return them instead of old values
trials_raw['test_condition'] = trials_raw['test_condition'].str.extract(r'(prediction|control|explanation)')

# subset only rows where counter is not empty
trials = trials_raw.loc[~trials_raw['test_condition'].isna()]

# if sender ends with "_timeout" or "_time" -> set duration to 15000 + duration
trials['duration'] = np.where(trials['sender'].str.endswith("_timeout") | trials['sender'].str.endswith("_time"), trials['duration'] + 15000, trials['duration'])

# add fsm_type column
trials['fsm_type'] = np.where(trials['fsm_number'] == 21, "easy", "hard")

# add column with trial number from counter_columns
trials['trial_number'] = trials[counter_columns[0]].sum(axis=1,  skipna=True).astype(int) + 1

# adjust trial number for attempts: 
trials['trial_number'] = np.where(trials['attempt'] == 2, trials['trial_number'] + 15, trials['trial_number'])
trials['trial_number'] = np.where(trials['attempt'] == 3, trials['trial_number'] + 30, trials['trial_number'])

# convert timestamp to datetime
trials['timestamp'] = pd.to_datetime(trials['timestamp'])

# add column with phase (feedback or transfer)
trials['phase'] = np.where(~trials['attempt'].isna(), "feedback", "transfer")

# sort by participant_id and timestamp
trials = trials.sort_values(by=['participant_id', 'timestamp'])

# add column with trial_order
trials['trial_order'] = trials.groupby(['participant_id']).cumcount() + 1 

# add column with transfer_order
trials['phase_order'] = trials.groupby(['participant_id', 'phase']).cumcount() + 1

# calculate n trials per participant and condition 
n_trials = trials.groupby(['participant_id', 'fsm_type', 'feedback_condition', 'transfer_order', 'transfer_task_1', 'transfer_task_2', 
                           'phase', 'test_condition', 'trial_type']).size().reset_index(name='n_trials')

# get participant_ids with n trials > 45 in feedback phase or != 20 in transfer phase
participant_ids_to_exclude = n_trials.loc[(n_trials['n_trials'] > 45) & 
                                          (n_trials['phase'] == "feedback")]['participant_id'].unique()
participant_ids_to_exclude = np.append(participant_ids_to_exclude, n_trials.loc[(n_trials['n_trials'] != 10) & 
                                                                                (n_trials['phase'] == "transfer")]['participant_id'].unique())
participant_ids_to_exclude = np.unique(participant_ids_to_exclude)

trials = trials.loc[~trials['participant_id'].isin(participant_ids_to_exclude)]

# check again
n_trials = trials.groupby(['participant_id', 'fsm_type', 'feedback_condition', 'transfer_order', 'transfer_task_1', 'transfer_task_2', 
                           'phase', 'test_condition', 'trial_type']).size().reset_index(name='n_trials')

# count n rows per participant in n_trials
n_rows = n_trials.groupby(['participant_id', 'feedback_condition', 'phase']).size().reset_index(name='n_rows')

# exclude participants with n_rows != 2 in feedback phase or != 4 in transfer phase
participant_ids_to_exclude = n_rows.loc[(n_rows['n_rows'] != 2) &
                                        (n_rows['phase'] == "feedback")]['participant_id'].unique()
participant_ids_to_exclude = np.append(participant_ids_to_exclude, n_rows.loc[(n_rows['n_rows'] != 4) &
                                                                                (n_rows['phase'] == "transfer")]['participant_id'].unique())
participant_ids_to_exclude = np.unique(participant_ids_to_exclude)

# exclude bad participants
trials = trials.loc[~trials['participant_id'].isin(participant_ids_to_exclude)]

# count n participants per condition 
n_trials = trials.groupby(['participant_id','fsm_type', 'feedback_condition', 'transfer_order', 'transfer_task_1', 'transfer_task_2']).size().reset_index(name='n_trials')
n_participants = n_trials.groupby(['fsm_type', 'feedback_condition', 'transfer_order', 'transfer_task_1', 'transfer_task_2']).size().reset_index(name='n_participants')

# save files as csv
trials.to_csv("../data/trials_test.csv", index=False)