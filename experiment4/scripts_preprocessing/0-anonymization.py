# Description: This script merges multiple files into two dataframes and removes data from participants that did not complete the experiment

import pandas as pd
from datetime import date
import numpy as np
import os
import openpyxl

current_dir = os.path.dirname(os.path.abspath(__file__))
openlab_path = os.path.join(current_dir, "..", "data_openlab/") # these folder is not in the repo
prolific_path = os.path.join(current_dir, "..", "data_prolific/") # these folder is not in the repo

# read in data with user IDs 
prolific = pd.read_csv(os.path.join(prolific_path, "prolific_export_64dbb6944f4b3a5ba714c361.csv"))
prolific = prolific[prolific['Status'] == 'APPROVED'] # remove participants that were rejected or did not complete the experiment

# read openlab broken csv files
filename = "Interaction-with-a-chatbot-using-emoji-v2023-08-17-full-data_n274.csv"
openlab_full_1 = pd.read_csv(os.path.join(openlab_path, filename), nrows=982, on_bad_lines='skip', low_memory=False)
openlab_full_2 = pd.read_csv(os.path.join(openlab_path, filename), skiprows=983, nrows=2316, low_memory=False)
openlab_full_3 = pd.read_csv(os.path.join(openlab_path, filename), skiprows=3300, nrows=23228, low_memory=False)
openlab_full_4 = pd.read_csv(os.path.join(openlab_path, filename), skiprows=26529)
# merge openlab data into one dataframe by column names
openlab = pd.concat([openlab_full_1, openlab_full_2, openlab_full_3, openlab_full_4], axis=0)

# get unique participant codes
codes_openlab = openlab['code'].unique()
codes_prolific = prolific['Participant id'].unique()

# find differences between codes
codes_openlab_not_in_prolific = np.setdiff1d(codes_openlab, codes_prolific) # not in prolific
codes_prolific_not_in_openlab = np.setdiff1d(codes_prolific, codes_openlab) # not in openlab

# all mismatching codes
codes_mismatch = np.union1d(codes_openlab_not_in_prolific, codes_prolific_not_in_openlab)

# remove mismatching codes
openlab = openlab[~openlab['code'].isin(codes_openlab_not_in_prolific)]
prolific = prolific[~prolific['Participant id'].isin(codes_prolific_not_in_openlab)]

# sort prolific_merged by Completed at and update index
prolific = prolific.sort_values(by='Completed at')
prolific = prolific.reset_index(drop=True)

# create new participant_id
prolific['participant_id'] = prolific.index + 1

# add ids to openlab data from prolific_merged['participant_id'] 
openlab = openlab.merge(prolific[['participant_id', 'Participant id']], 
                                    left_on='code', right_on='Participant id', how='left')


# remove code and Prolific id columns
openlab = openlab.drop(columns=['code', 'Participant id', 'openLabId'])
prolific = prolific.drop(columns=['Participant id', 'Completion code', 'Status', 'Started at', 
                                                'Reviewed at', 'Archived at', 'Completion code'])

# move participant_id to first column in trials_merged and prolific_merged
openlab = openlab[['participant_id'] + [col for col in openlab.columns if col != 'participant_id']]
prolific = prolific[['participant_id'] + [col for col in prolific.columns if col != 'participant_id']]

# save data
prolific.to_csv(os.path.join(current_dir, "..", "data_raw", "prolific_raw_anonymized.csv"), index=False)
openlab.to_csv(os.path.join(current_dir, "..", "data_raw", "trials_raw_anonymized.csv"), index=False)
