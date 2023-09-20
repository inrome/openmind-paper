import pickle
import pandas as pd
import numpy as np
import os


# Load data from pickle file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../data/imported_clean_data.pickle')

participants_path = os.path.join(current_dir, '../data/participants_n210.csv')
trials_test_path = os.path.join(current_dir, '../data/trials_test_n210.csv')

participants = pd.read_csv(participants_path)
trials_test = pd.read_csv(trials_test_path)

fsm_easy = {0: {"a": [0, 0, 1, 0], "b": [0, 0, 1, 0]},
            1: {"a": [1, 0, 0, 0], "b": [1, 0, 0, 0]},
            2: {"a": [0.6, 0, 0, 0.4], "b": [0, 0, 1, 0]},
            3: {"a": [0.4, 0.6, 0, 0], "b": [0, 0, 0, 1]}
            }  
   
fsm_hard = {0: {"a": [0, 0, 0, 1], "b": [0, 0, 0, 1]},
                1: {"a": [0.4, 0.6, 0, 0], "b": [0, 0.6, 0, 0.4]},
                2: {"a": [0, 0, 1, 0], "b": [0.4, 0.6, 0, 0]},
                3: {"a": [0.4, 0, 0.6, 0], "b": [0, 0, 0.6, 0.4]}
            }    

# %%
mental_models = {}
states = [0, 1, 2, 3]
responses = ['a', 'b']

# participant_id = trials_test['participant_id'].unique()[2]
for participant_id in trials_test['participant_id'].unique():

    fsm_type = participants[participants['participant_id'] == participant_id]['fsm_type'].values[0]
    
    if fsm_type == 'easy':
        fsm = fsm_easy
    elif fsm_type == 'hard':
        fsm = fsm_hard
    else:
        raise Exception('fsm_type not recognized')
    
    mental_models[participant_id] = fsm
#%%

# check one participant
mental_models[1]
# save mental models to pickle file
save_path = os.path.join(current_dir, '../outputs/mental_models.pickle')
with open(save_path, 'wb') as f:
    pickle.dump(mental_models, f)

# %%
