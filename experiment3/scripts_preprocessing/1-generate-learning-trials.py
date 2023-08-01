# Description: This script generates learning trials from the raw data

import pandas as pd
import os
import numpy as np

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data_raw/trials_raw_anonymized.csv")
trials = pd.read_csv(data_path, low_memory= False)

# get unique participant codes
participant_ids = trials['participant_id'].unique()

# participant_ids = [1,96,149] # for testing purposes REMOVE THIS LINE LATER

n_trials = {}

trials_learning = pd.DataFrame() # create empty df to store learning trials

# participant_id = 1 # for testing purposes REMOVE THIS LINE LATE

for participant_id in participant_ids:
    # get full trials data for participant
    participant_data = trials.loc[(trials['participant_id'] == participant_id) & (trials['type'] == "full")]
    if len(participant_data) == 0:
        print("Participant " + str(participant_id) + " has no full trials. Switching to incremental trials.")
        participant_data = trials.loc[(trials['participant_id'] == participant_id) & (trials['type'] == "incremental")]

    # filter out trials that are not learning trials
    ts_learning_raw = participant_data[participant_data['learning_counter'] >= 0]

    # create a df with learning trials (but without responses)
    ts_learning_1 = ts_learning_raw[['participant_id', 'fsm_number',
                                     'sender', 'timestamp', 'learning_counter','state_current', 
                                     'response_current', 'state_next', 'task']].copy()

    # filter out trials with state_current or state_next less than 0
    ts_learning_1 = ts_learning_1.loc[(ts_learning_1['state_current'] >= 0) & 
                                      (ts_learning_1['state_next'] >= 0)]

    # filter out trials with response_current not in ["a", "b"]
    ts_learning_1 = ts_learning_1.loc[ts_learning_1['response_current'].isin(["a", "b"])]

    # remove duplicates (use the first attempt based on timestamp)
    if len(ts_learning_1['learning_counter']) > 59:
            ts_learning_1 = ts_learning_1.sort_values(by=['timestamp']).drop_duplicates(subset=['learning_counter'], keep='first')
            print("Participant " + str(participant_id) + " has " + str(len(ts_learning_1)) + " learning trials. Removing duplicates.")

    # get response times for interactions with the chatbot
    ts_learning_2 = ts_learning_raw[['participant_id', 'learning_counter', 'sender', 'response', 'duration', 'test_order']].copy()
    ts_learning_2 = ts_learning_2.loc[ts_learning_2['response'].notna() & 
                                      (ts_learning_2['learning_counter'] >= 0) & 
                                      ts_learning_2['sender'].isin(['response_1_screen', 'response_2_screen', 'response_1_screen_time', 
                                                                    'response_2_screen_time', 'response_further_screen', 'response_further_screen_time', 'fast_responses_screen'])]
    ts_learning_2['response_time'] = np.where(ts_learning_2['sender'].isin(['response_further_screen_time', 'response_1_screen_time', 'response_2_screen_time']),
                                                ts_learning_2['duration'] + 15000, # 15 seconds is the timeout for the previous screen
                                                ts_learning_2['duration'])
    
    # remove unnecessary columns before merging
    ts_learning_1 = ts_learning_1.drop(columns=['sender'])
    ts_learning_2 = ts_learning_2.drop(columns=['sender', 'response', 'duration'])
    
    # merge ts_learning_1 and ts_learning_2
    if len(ts_learning_1) == len(ts_learning_2['learning_counter']): 
        ts_learning = pd.merge(ts_learning_1, ts_learning_2, on=['participant_id','learning_counter'], how='left')

        ts_learning['fsm_type'] = ts_learning['fsm_number'].apply(lambda x: 'easy' if x == 21 else 'hard') 
        ts_learning['learning_counter'] = ts_learning['learning_counter'] + 1 # add 1 to start with 1 instead of 0
        ts_learning['learning_counter'] = ts_learning['learning_counter'].astype(int)
        ts_learning['response_time'] = ts_learning['response_time'].astype(int) # convert response time to int
        
        # fix negative response times
        ts_learning['response_time'] = np.where(ts_learning['response_time'] < 0, 0, ts_learning['response_time'])

    else:
        print("Number of learning trials for participant " + str(participant_id) + " does not match. Skipping participant.")
        continue

    # remove participants with less than 60 learning trials
    if len(ts_learning_1) < 59: 
         print("Participant " + str(participant_id) + " has " + str(len(ts_learning_1)) + " learning trials. Skipping participant.")
         continue
    
    else: 
        n_trials[participant_id] = len(ts_learning_1) # number of learning trials per participant

        # add participant data to df
        trials_learning = pd.concat([trials_learning, ts_learning], ignore_index=True)

# save data
save_path = os.path.join(current_dir, "..", "data/trials_learning.csv")
trials_learning.to_csv(save_path, index=False)
