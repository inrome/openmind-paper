import pickle
import pandas as pd
import numpy as np
import os

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
test_trials = pd.read_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs.csv'))

# define function to compute Ri
def compute_Ri(option_1_p, option_2_p, epsilon):
    import math
    # abs value of log ratio of probabilities
    r_i = abs(math.log((option_1_p + epsilon) / (option_2_p + epsilon)))

    return r_i

# compute Ri for each trial
epsilon = 0.1  # set value for epsilon
test_trials['R_i_mm_eps_' + str(epsilon)] = np.nan
test_trials['R_i_mm_an_eps_' + str(epsilon)] = np.nan

for index, row in test_trials.iterrows():
    test_trials.loc[index, 'R_i_mm_eps_' + str(epsilon)] = compute_Ri(row['option_1_p_mm'], row['option_2_p_mm'], epsilon)
    test_trials.loc[index, 'R_i_mm_an_eps_' + str(epsilon)] = compute_Ri(row['option_1_p_mm_an'], row['option_2_p_mm_an'], epsilon) if pd.isna(row['option_1_p_mm_an']) == False else np.nan

# save data
test_trials.to_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs_Ri.csv'), index=False)