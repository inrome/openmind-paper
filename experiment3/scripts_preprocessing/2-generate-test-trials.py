# Generates test trials for all participants (test_PEC, test_prediction, test_control, test_explanation)

import pandas as pd
import os
import numpy as np


# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
trials = pd.read_csv(os.path.join(current_dir, "..", "data_raw/trials_raw_anonymized.csv"), low_memory= False)
trials_learning = pd.read_csv(os.path.join(current_dir, "..", "data/trials_learning.csv"), low_memory= False)

# get unique participant codes (only those who have learning trials)
participant_ids = trials_learning['participant_id'].unique()

# function to get response times for test trials
def get_trials(participant_data):
    any_counter_not_empty = (participant_data['test_prediction_counter'] >= 0) | (participant_data['test_control_counter']  >= 0) | (participant_data['test_explanation_counter']  >= 0)
    test_trials = participant_data[(any_counter_not_empty) &
                            (~participant_data['response'].isna())][['participant_id','fsm_number', 'test_order', 'test1', 'test2', 'test3',
                                                                     'test_prediction_counter', 'test_control_counter', 'test_explanation_counter',
                                                                     'sender', 'duration', 'exp_type', 
                                                                     'state_1', 'response_1', 'state_2', 'response_2', 'state_3',
                                                                     'option_1','option_2', 'correctResponse', 'response', 'correct']]
    
    # add column with trial number (sum ignoring nan: 'test_prediction_counter', 'test_control_counter', 'test_explanation_counter')
    test_trials['trial_number'] = test_trials[['test_prediction_counter', 'test_control_counter', 'test_explanation_counter']].sum(axis=1,  skipna=True).astype(int) + 1

    # add test condition corresponding to the counter that is not empty
    test_trials['test_condition'] = np.where(test_trials['test_prediction_counter'] >= 0, "prediction",
                                                np.where(test_trials['test_control_counter'] >= 0, "control", "explanation"))
    
    # add column with response time (if sender ends with "screen_response", duration is response time, otherwise duration + 15000)
    test_trials['response_time'] = np.where(test_trials['sender'].str.endswith("screen_response"), test_trials['duration'], test_trials['duration'] + 15000)
    
    # add column with fsm type (easy or hard)
    test_trials['fsm_type'] = np.where(test_trials['fsm_number'] == 21, "easy", "hard")

    # rename columns (correct -> response_correct, exp_type -> trial_type)
    test_trials = test_trials.rename(columns={"correct": "response_correct", "exp_type": "trial_type"})

    # remove columns (test_prediction_counter, test_control_counter, test_explanation_counter, sender, duration)
    test_trials = test_trials.drop(columns=['test_prediction_counter', 'test_control_counter', 'test_explanation_counter', 'sender', 'duration'])

    # reorder columns
    test_trials = test_trials[['participant_id', 'fsm_number', 'fsm_type', 'test_order', 'test1', 'test2', 'test3', 
                               'trial_number', 'trial_type', 
                               'state_1', 'response_1', 'state_2', 'response_2', 'state_3', 'option_1', 'option_2', 
                               'correctResponse', 'response', 'response_correct', 'response_time']]
    return test_trials


test_PEC = pd.DataFrame()

for participant_id in participant_ids: 
    # get full trials data for participant (with column "type" == "full")
    participant_data = trials.loc[(trials['participant_id'] == participant_id) & (trials['type'] == "full")]
    if len(participant_data) == 0:
        print("Participant " + str(participant_id) + " has no full trials. Switching to incremental trials.")
        participant_data = trials.loc[(trials['participant_id'] == participant_id) & (trials['type'] == "incremental")] 

    # get test trials
    test_trials = get_trials(participant_data)

    # add test trials to test_PEC
    test_PEC = pd.concat([test_PEC, test_trials])

# save files as csv
test_PEC.to_csv(os.path.join(current_dir, "..", "data/trials_test.csv"), index=False)