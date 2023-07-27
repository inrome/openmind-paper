# Load packages
import pandas as pd
import numpy as np
import pickle
import os


# Load trials data
trials_PEC_path = "../outputs/all_trials_with_Ri.csv"
trials_PEC = pd.read_csv(trials_PEC_path)

# Load participants data
participants_path = "../data/participants.csv"
participants = pd.read_csv(participants_path)

participants.groupby(['learning_condition']).size()
participants.groupby(['test_condition']).size()
participants.groupby(['fsm_type']).size()
participants.groupby(['subject_sex']).size()
participants.groupby(['learning_condition', 'test_condition', 'fsm_type']).size()

participants['subject_age'].mean()
participants['subject_age'].std()

# count number of unique participants per learning condition
trials_PEC.groupby(['learning_condition']).agg({'participant_id': pd.Series.nunique})

# make sure that each participant has exactly 20 test trials
trials_PEC.groupby(['participant_id']).size().describe()