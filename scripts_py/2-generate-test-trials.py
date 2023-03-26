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
                            (~participant_data['response'].isna())][['participant_id','fsm_number', 'learning_condition',
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
    test_trials = test_trials[['participant_id', 'fsm_number', 'fsm_type', 'test_condition', 'learning_condition','trial_number', 'trial_type', 
                               'state_1', 'response_1', 'state_2', 'response_2', 'state_3', 'option_1', 'option_2', 
                               'correctResponse', 'response', 'response_correct', 'response_time']]
    return test_trials


test_prediction = pd.DataFrame()
test_control = pd.DataFrame()
test_explanation = pd.DataFrame()
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

    # add test trials to test_prediction, test_control, test_explanation
    test_prediction = pd.concat([test_prediction, test_trials.loc[test_trials['test_condition'] == "prediction"]])
    test_control = pd.concat([test_control, test_trials.loc[test_trials['test_condition'] == "control"]])
    test_explanation = pd.concat([test_explanation, test_trials.loc[test_trials['test_condition'] == "explanation"]])

# save files as csv
test_PEC.to_csv(os.path.join(current_dir, "..", "data/test_PEC.csv"), index=False)
test_prediction.to_csv(os.path.join(current_dir, "..", "data/test_prediction.csv"), index=False)
test_control.to_csv(os.path.join(current_dir, "..", "data/test_control.csv"), index=False)
test_explanation.to_csv(os.path.join(current_dir, "..", "data/test_explanation.csv"), index=False)