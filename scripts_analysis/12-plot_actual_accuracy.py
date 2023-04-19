# Description: Provide descriptive statistics for actual accuracy 
# depending on learning_condition, fsm_type, test_condition, and trial_type (withing-subjects)

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
trials_PEC_path = os.path.join(current_dir, '../data/test_PEC.csv')
trials_PEC = pd.read_csv(trials_PEC_path)

# aggregate accuracy data for each participant and each trial type

aggregated_accuracy = trials_PEC.groupby(['participant_id', 'learning_condition', 'fsm_type', 'test_condition', 'trial_type'])['response_correct'].mean().reset_index()

# rename easy to Easy and hard to Hard
aggregated_accuracy['fsm_type'] = aggregated_accuracy['fsm_type'].replace({'easy': 'Easy', 'hard': 'Hard'})

# remame test_condition 
aggregated_accuracy['test_condition'] = aggregated_accuracy['test_condition'].replace({'prediction': 'Prediction', 'control': 'Control', 'explanation': 'Explanation'})
# same for trial_type
aggregated_accuracy['trial_type'] = aggregated_accuracy['trial_type'].replace({'visible': 'Visible', 'hidden': 'Hidden'})

# convert trial_type to categorical variable with ordered levels (visible, hidden)
aggregated_accuracy['trial_type'] = pd.Categorical(aggregated_accuracy['trial_type'], categories=['Visible', 'Hidden'], ordered=True)

# convert fsm_type to categorical variable with ordered levels (easy, hard)
aggregated_accuracy['fsm_type'] = pd.Categorical(aggregated_accuracy['fsm_type'], categories=['Easy', 'Hard'], ordered=True)

# plot accuracy for each trial type
sns.set(style="whitegrid", font_scale=1.2)

# Plot mean and standard deviation for each trial type
g = sns.catplot(x="trial_type", y="response_correct", hue="learning_condition", col="fsm_type", row="test_condition", data=aggregated_accuracy, \
                kind="point", dodge=True, join=True, errorbar = ("ci", 95), palette="muted", height=3.5, aspect=1.5, legend=False)

# make line thiner
for ax in g.axes.flat:
    for line in ax.lines:
        line.set_linewidth(2)

# add dashed line at 0.5
for ax in g.axes.flat:
    ax.axhline(0.5, ls='--', color='gray')

# position legend at the top and make it in one row
g.add_legend(title="", loc='upper center', 
             bbox_to_anchor=(0.37, 1.05), ncol=2)

# Label the facets 
g.set_axis_labels("", "Accuracy")
g.set_titles("{row_name} {col_name}")