import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir, '../outputs/trials_with_max_beta.pickle')

with open(data_dir, 'rb') as f:
    sample = pickle.load(f)

# generate a DataFrame with participant ID, fsm_type, test_condition, and hidden_subset_count
subset_counts = pd.DataFrame(columns=['participant_id', 'fsm_type', 'task', 'hidden_subset_count'])

for participant_id in sample.keys():
    participant = sample[participant_id]
    participant_subset_count = pd.DataFrame(columns=['participant_id', 'fsm_type', 'task', 'hidden_subset_count'])
    participant_subset_count['participant_id'] = [participant_id]
    participant_subset_count['fsm_type'] = participant['trials_hid']['fsm_type'].unique()[0]
    participant_subset_count['task'] = participant['task']
    participant_subset_count['hidden_subset_count'] = participant['hidden_subset_count']
    subset_counts = pd.concat([subset_counts, participant_subset_count], ignore_index=True)

# make a histograms of hidden_subset_count for each fsm_type separately with bin size == 2 and hue == task
for fsm_type in subset_counts['fsm_type'].unique():
    subset_counts_fsm_type = subset_counts[subset_counts['fsm_type'] == fsm_type]
    sns.histplot(data=subset_counts_fsm_type, x='hidden_subset_count', hue='task', bins=np.arange(2, 10, 1), multiple='stack')
    plt.title(f'Number of hiddent test questiond for {fsm_type} FSM')
    plt.show()

# separate histograms for each fsm_type and task
g = sns.FacetGrid(subset_counts, col='fsm_type', hue='task', row = 'task')
g.map(sns.histplot, 'hidden_subset_count', bins=np.arange(2, 10, 1), multiple='stack')
g.set_titles('{row_name} {col_name}')
plt.show()