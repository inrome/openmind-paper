# Adding option1_p and option2_p columns to test trials
# Visible trials are calculated using the mental models
# Hidden trials are calculated using PEC functions (normative and with Alternative Neglect (AN))

import numpy as np
import pandas as pd
import pickle
import os
from PEC_functions import predict, control, explain

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))

participants_path = os.path.join(current_dir, '../data/participants_n210.csv')
trials_test_path = os.path.join(current_dir, '../data/trials_test_n210.csv')

participants = pd.read_csv(participants_path)
trials_test = pd.read_csv(trials_test_path)

mm_path = os.path.join(current_dir, '../outputs/mental_models.pickle')

with open(mm_path, 'rb') as f:
    MM = pickle.load(f)

# Add columns
trials_test['option_1_p_mm'] = np.nan
trials_test['option_2_p_mm'] = np.nan
trials_test['option_1_p_mm_an'] = np.nan
trials_test['option_2_p_mm_an'] = np.nan

for index, row in trials_test.iterrows():
    state_1 = int(row['state_1']) if pd.isna(row['state_1']) == False  else None
    response_1 = row['response_1']
    state_2 = int(row['state_2']) if pd.isna(row['state_2']) == False else None
    response_2 = row['response_2']
    state_3 = int(row['state_3']) if pd.isna(row['state_3']) == False else None
    response_1_cf = "a" if response_1 == "b" else "b"
    response_2_cf = "a" if response_2 == "b" else "b"
    mm = MM[row['participant_id']]

    # Visble trials 
    if row['trial_type'] == "visible":
        # Prediction
        if row['test_condition'] == "prediction":
            option_1 = int(row['option_1'])
            option_2 = int(row['option_2'])
            trials_test.loc[index, 'option_1_p_mm'] = mm[state_2][response_2][option_1]
            trials_test.loc[index, 'option_2_p_mm'] = mm[state_2][response_2][option_2]
        
        # Control
        elif row['test_condition'] == "control":
            option_1 = "a"
            option_2 = "b"
            trials_test.loc[index, 'option_1_p_mm'] = mm[state_2][option_1][state_3]
            trials_test.loc[index, 'option_2_p_mm'] = mm[state_2][option_2][state_3]

        # Explanation
        elif row['test_condition'] == "explanation":
            option_1 = 1
            option_2 = 2
            trials_test.loc[index, 'option_1_p_mm'] = mm[state_1][response_1_cf][state_2] * mm[state_2][response_2][state_3]
            trials_test.loc[index, 'option_2_p_mm'] = mm[state_1][response_1][state_2] * mm[state_2][response_2_cf][state_3]

    # Hidden trials
    elif row['trial_type'] == "hidden":

        # Prediction
        if row['test_condition'] == "prediction":
            option_1 = int(row['option_1'])
            option_2 = int(row['option_2'])

            trials_test.loc[index, 'option_1_p_mm'] = predict(state_1, response_1, response_2, mm, mode="normative")[option_1][0]
            trials_test.loc[index, 'option_2_p_mm'] = predict(state_1, response_1, response_2, mm, mode="normative")[option_2][0]

            trials_test.loc[index, 'option_1_p_mm_an'] = predict(state_1, response_1, response_2, mm, mode="an")[option_1][0]
            trials_test.loc[index, 'option_2_p_mm_an'] = predict(state_1, response_1, response_2, mm, mode="an")[option_2][0]

        # Control
        elif row['test_condition'] == "control":
            option_1 = row['option_1']
            option_2 = row['option_2']

            trials_test.loc[index, 'option_1_p_mm'] = control(state_1, state_3, mm, mode="normative")[option_1]
            trials_test.loc[index, 'option_2_p_mm'] = control(state_1, state_3, mm, mode="normative")[option_2]

            trials_test.loc[index, 'option_1_p_mm_an'] = control(state_1, state_3, mm, mode="an")[option_1]
            trials_test.loc[index, 'option_2_p_mm_an'] = control(state_1, state_3, mm, mode="an")[option_2]

        # Explanation
        elif row['test_condition'] == "explanation":
            option_1 = 1
            option_2 = 2

            trials_test.loc[index, 'option_1_p_mm'] = explain(state_1, response_1, response_2, state_3, mm,
                                                             mode="normative")["1"]
            trials_test.loc[index, 'option_2_p_mm'] = explain(state_1, response_1, response_2, state_3, mm,
                                                             mode="normative")["2"]

            trials_test.loc[index, 'option_1_p_mm_an'] = explain(state_1, response_1, response_2, state_3, mm,
                                                                mode="an")["1"]
            trials_test.loc[index, 'option_2_p_mm_an'] = explain(state_1, response_1, response_2, state_3, mm, 
                                                                mode="an")["2"]

# round probabilities to 3 decimals for 'option_1_p_mm', 'option_2_p_mm', 'option_1_p_mm_an', 'option_2_p_mm_an'
for colname in ['option_1_p_mm', 'option_2_p_mm', 'option_1_p_mm_an', 'option_2_p_mm_an']:
    trials_test[colname] = trials_test[colname].apply(lambda x: round(x, 3))

# export to csv
trials_test.to_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs.csv'), index=False)
