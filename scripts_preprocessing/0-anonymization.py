# Description: This script merges multiple files into two dataframes and removes data from participants that did not complete the experiment

import pandas as pd
from datetime import date
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
openlab_path = os.path.join(current_dir, "..", "data_openlab/") # these folder is not in the repo
prolific_path = os.path.join(current_dir, "..", "data_prolific/") # these folder is not in the repo

# get all files in openlab folder with .xlsx extension
files_openlab = [f for f in os.listdir(openlab_path) if f.endswith('.xlsx')]

# get all files in prolific folder with .csv extension
files_prolific = [f for f in os.listdir(prolific_path) if f.endswith('.csv')]

# merge prolific data
prolific_merged = pd.DataFrame()
for file in files_prolific:
    tmp = pd.read_csv(prolific_path + file)
    prolific_merged = pd.concat([prolific_merged, tmp], axis=0, ignore_index=True)
    # add column with filename
    prolific_merged['filename'] = file

# merge openlab data (xlsx) taking into account that some files have different column names
trials_merged = pd.DataFrame()

for file in files_openlab:
    tmp = pd.read_excel(openlab_path + file)
    trials_merged = pd.concat([trials_merged, tmp], axis=0)
    # add column with filename
    trials_merged['filename'] = file

## fix missing code: subsitute v3p22 with 63e534e8ae837950bc60d88b
trials_merged['code'] = trials_merged['code'].replace('v3p22', '63e534e8ae837950bc60d88b')

# remove REJECTED participants
prolific_merged = prolific_merged[prolific_merged['Status'] != 'REJECTED']

# get unique participant codes
codes_openlab = trials_merged['code'].unique()
codes_prolific = prolific_merged['Participant id'].unique()

# find codes that are not in prolific data
codes_openlab_not_in_prolific = np.setdiff1d(codes_openlab, codes_prolific)

# find codes that are not in openlab data
codes_prolific_not_in_openlab = np.setdiff1d(codes_prolific, codes_openlab)

# remove participants that are not in prolific data
trials_merged = trials_merged[~trials_merged['code'].isin(codes_openlab_not_in_prolific)]

# remove participants that are not in openlab data
prolific_merged = prolific_merged[~prolific_merged['Participant id'].isin(codes_prolific_not_in_openlab)]

# check if all codes are unique in prolific data
assert len(prolific_merged['Participant id'].unique()) == len(prolific_merged['Participant id'])

# create new participant_id
prolific_merged['participant_id'] = prolific_merged.index + 1

# add ids to openlab data from prolific_merged['participant_id'] 
trials_merged = trials_merged.merge(prolific_merged[['participant_id', 'Participant id']], 
                                    left_on='code', right_on='Participant id', how='left')

# check the codes
assert set(trials_merged['code'].unique()) == set(prolific_merged['Participant id'].unique())

# remove code and Prolific id columns
trials_merged = trials_merged.drop(columns=['code', 'Participant id', 'openLabId'])
prolific_merged = prolific_merged.drop(columns=['Participant id', 'Submission id', 'Status', 'Started at', 
                                                'Reviewed at', 'Archived at', 'Completion code'])

# move participant_id to first column in trials_merged and prolific_merged
trials_merged = trials_merged[['participant_id'] + [col for col in trials_merged.columns if col != 'participant_id']]
prolific_merged = prolific_merged[['participant_id'] + [col for col in prolific_merged.columns if col != 'participant_id']]

# add learning_condition == "Experiment 2 (test preview)" if timestamp > 2023-02-01 and "Experiment 1 (no preview)" otherwise
trials_merged['learning_condition'] = np.where(trials_merged['timestamp'] > '2023-02-01', 'Experiment 2 (test preview)', 'Experiment 1 (no preview)')

# save data
prolific_merged.to_csv(os.path.join(current_dir, "..", "data_raw", "prolific_raw_anonymized.csv"), index=False)
trials_merged.to_csv(os.path.join(current_dir, "..", "data_raw", "trials_raw_anonymized.csv"), index=False)