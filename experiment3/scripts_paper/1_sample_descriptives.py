# Load packages
import pandas as pd
import numpy as np
import pickle
import os


# Load trials data
trials_PEC_path = "../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv"
trials_PEC = pd.read_csv(trials_PEC_path)

# Load participants data
participants_path = "../data/participants.csv"
participants = pd.read_csv(participants_path)

participants.groupby(['test_order']).size()
participants.groupby(['fsm_type']).size()
participants.groupby(['subject_sex']).size()
participants.groupby(['test_order', 'fsm_type']).size()

participants['subject_age'].mean()
participants['subject_age'].std()
participants['subject_age'].min()
participants['subject_age'].max()

# count number of unique participants per condition
trials_PEC.groupby(['test_order']).agg({'participant_id': pd.Series.nunique})

# make sure that each participant has exactly 30 test trials
trials_PEC.groupby(['participant_id']).size().describe()