# Description: Generate participants.csv from trials_raw_anonymized.csv and prolific_raw_anonymized.csv

import pandas as pd
import os

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
ts_raw = pd.read_csv(os.path.join(current_dir, "..", "data_raw/trials_raw_anonymized.csv"), low_memory= False)

ss_demogr = pd.read_csv(os.path.join(current_dir, "..", "data_raw/prolific_raw_anonymized.csv"), low_memory= False)

# Participants
tmp_ss = ts_raw[['participant_id', 'test_condition', 'fsm_number', 'q_estimate', 'q_stategy_scale', 'q_strategy', 'timestamp']].copy()

def first_non_null(series):
    non_null_series = series.dropna()
    if not non_null_series.empty:
        return non_null_series.iloc[0]
    return None

# Group by participant_id and aggregate
grouped = tmp_ss.groupby('participant_id', as_index=False).agg({
    'test_condition': first_non_null,
    'fsm_number': first_non_null,
    'timestamp': first_non_null,
    'q_estimate': first_non_null,
    'q_stategy_scale': first_non_null,
    'q_strategy': first_non_null,
})

# Add Prolific demographic data
ss_demogr = ss_demogr[['participant_id', 'Age', 'Sex', 'Ethnicity simplified', 'Time taken']].copy()
ss_demogr.rename(columns={'Age': 'subject_age', 'Sex': 'subject_sex', 'Ethnicity simplified': 'subject_ethnicity', 'Time taken': 'subject_timeTaken'}, inplace=True)

# Merge
ss = pd.merge(grouped, ss_demogr, on='participant_id', how='left')

# Save
ss.to_csv(os.path.join(current_dir, "..", "data/participants.csv"), index=False)