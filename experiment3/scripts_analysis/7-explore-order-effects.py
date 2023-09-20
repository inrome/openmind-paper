# Visualize and test the order effects on accuracy and beta values.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
trials = pd.read_csv(os.path.join(current_dir, '../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv'))
participants = pd.read_csv(os.path.join(current_dir, '../data/participants.csv'))

betas = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas.csv'))
betas.rename(columns = {'task': 'test_condition'}, inplace = True) # rename task to test_condition

betas_wide = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas_wide.csv'))

# add variable test_order_numeric
trials['test_order_numeric'] = np.where(trials['test1'] == trials['test_condition'], 1, np.where(trials['test2'] == trials['test_condition'], 2, 3))

# add trials['test_order_numeric'] to betas based on test_order
convert_test_order = trials[['test_order', 'test_condition', 'test_order_numeric']].drop_duplicates()
betas = pd.merge(betas, convert_test_order, on = ['test_order', 'test_condition'])

# mean accuracy per participant, test_condition, trial_type
accuracy = trials.groupby(['participant_id', 'fsm_type', 'test_condition',  'trial_type', 'test_order_numeric'])['response_correct_mm'].mean().reset_index()
accuracy_an = trials.groupby(['participant_id', 'fsm_type', 'test_condition',  'trial_type', 'test_order_numeric'])['response_correct_mm_an'].mean().reset_index().dropna(subset=['response_correct_mm_an'])
accuracy_an['trial_type'] = 'hidden_an'
accuracy = pd.concat([accuracy, accuracy_an], ignore_index=True)

# combine response_correct_mm_an and response_correct_mm to new column response_correct ignoring nans
accuracy["response_correct"] = accuracy["response_correct_mm"].fillna(accuracy["response_correct_mm_an"])

# Mean accuracy per test_condition, test_order_numeric, trial_type
accuracy['trial_type'] = accuracy['trial_type'].astype('category').cat.reorder_categories(['visible', 'hidden', 'hidden_an'])

accuracy_mean = accuracy.groupby(['test_order_numeric', 'trial_type', 'test_condition'])['response_correct'].mean().reset_index()

# plot mean accuracy with 95%CIs per test_order_numeric, test_condition, trial_type
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
g = sns.catplot(x="test_order_numeric", y="response_correct", hue="trial_type", col="test_condition", data=accuracy, kind="point", ci=95, height=4, aspect=.7, dodge=0.3)
g.set_axis_labels("", "Accuracy")
g.set(ylim=(0, 1))
g.set_titles("{col_name}")

# plot with facets (rows by fsm_type, columns by test_condition)
g = sns.catplot(x="test_order_numeric", y="response_correct", hue="trial_type", col="test_condition", row="fsm_type", data=accuracy, kind="point", ci=95, height=4, aspect=.7, dodge=0.3)
g.set_axis_labels("", "Accuracy")
# add y axis labels (Easy and Hard FSM) depending on fsm_type
g.set_titles("{row_name} {col_name}")
g.set(ylim=(0, 1))


# same plot with beta values instead of accuracy
betas['trial_type'] = betas['trial_type'].astype('category').cat.reorder_categories(['visible', 'hidden', 'hidden_an'])
betas_mean = betas.groupby(['test_order_numeric', 'trial_type', 'test_condition'])['max_beta'].mean().reset_index()

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
g = sns.catplot(x="test_order_numeric", y="max_beta", hue="trial_type", col="test_condition", data=betas, kind="point", ci=95, height=4, aspect=.7, dodge=0.3)
g.set_axis_labels("", "Beta")
g.set(ylim=(0, 0.8))
g.set_titles("{col_name}")

# plot with facets (rows by fsm_type, columns by test_condition)
g = sns.catplot(x="test_order_numeric", y="max_beta", hue="trial_type", col="test_condition", row="fsm_type", data=betas, kind="point", ci=95, height=4, aspect=.7, dodge=0.3)
g.set_axis_labels("", "Beta")
# add y axis labels (Easy and Hard FSM) depending on fsm_type
g.set_titles("{row_name} {col_name}")
g.set(ylim=(0, 1))
