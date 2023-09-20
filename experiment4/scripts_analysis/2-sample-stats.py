# Load packages
import pandas as pd
import numpy as np
import pickle
import os


# Load data
trials = pd.read_csv("../data/trials_test_n210.csv")
participants = pd.read_csv("../data/participants_n210.csv")

# stats
participants.groupby(['feedback_condition']).size()
participants.groupby(['fsm_type']).size()
participants.groupby(['subject_sex']).size()
participants.groupby(['transfer_order', 'fsm_type']).size()

participants['subject_age'].mean()
participants['subject_age'].std()
participants['subject_age'].min()
participants['subject_age'].max()

# count number of unique participants per condition
trials.groupby(['feedback_condition']).agg({'participant_id': pd.Series.nunique})

