import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
trials_PEC_path = '../outputs/all_trials_with_Ri.csv'
trials_PEC = pd.read_csv(trials_PEC_path)

# aggregate accuracy data 
aggregated_accuracy = trials_PEC.groupby(['participant_id', 'learning_condition', 'fsm_type', 
                                          'test_condition', 'trial_type'])['response_correct', 'response_correct_mm'].mean().reset_index()

# rename variables for better labeling
aggregated_accuracy['fsm_type'] = aggregated_accuracy['fsm_type'].replace({'easy': 'Easy', 'hard': 'Hard'})
aggregated_accuracy['test_condition'] = aggregated_accuracy['test_condition'].replace({'prediction': 'Prediction', 
                                                                                       'control': 'Control', 
                                                                                       'explanation': 'Explanation'})
aggregated_accuracy['trial_type'] = aggregated_accuracy['trial_type'].replace({'visible': 'Visible', 'hidden': 'Hidden'})

# convert trial_type and fsm_type to categorical variable with ordered levels
aggregated_accuracy['trial_type'] = pd.Categorical(aggregated_accuracy['trial_type'], categories=['Visible', 'Hidden'], ordered=True)
aggregated_accuracy['fsm_type'] = pd.Categorical(aggregated_accuracy['fsm_type'], categories=['Easy', 'Hard'], ordered=True)

aggregated_accuracy_long = pd.melt(aggregated_accuracy, id_vars=['participant_id', 'learning_condition', 'fsm_type', 'test_condition', 'trial_type'],
                            value_vars=['response_correct', 'response_correct_mm'], var_name='accuracy_type', value_name='accuracy')


# Plot actual and predicted accuracy
sns.set(style="whitegrid", font_scale=1.4) # set style and font size for all plots

g = sns.catplot(x="trial_type", y="accuracy", hue="test_condition", col="fsm_type", row="accuracy_type", data=aggregated_accuracy_long,
 kind="point", dodge=0.25, join=True, errorbar = ("ci", 95), palette=['darkorange','indigo','steelblue'],
 height=3.5, aspect=1.3, legend=False, markers=["o", "s", "^"]) 

# make line thiner
for ax in g.axes.flat:
   ax.set_ylim(0.42, 1)
   ax.axhline(0.5, ls='--', color='gray')
   for line in ax.lines:
     line.set_linewidth(2)
 
# add different y-axis labels for each row (Actual vs. Mental Model)
g.axes[0,0].set_ylabel("Actual Accuracy")
g.axes[1,0].set_ylabel("Mental Model Accuracy")

# remove x-axis label
g.set_xlabels("")

# add legend to the top center outside of the plot and in 3 rows
handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False)

# add col names to the top of each column
g.axes[0,0].set_title("Easy")
g.axes[0,1].set_title("Hard")
g.axes[1,0].set_title("")
g.axes[1,1].set_title("")

# save plot
g.savefig("../outputs/figures/fig-accuracies-both-exp.png", dpi=300, bbox_inches='tight')

# Same plot but with only the mental model accuracy
g = sns.catplot(x="trial_type", y="accuracy", hue="test_condition", col="fsm_type",
                data=aggregated_accuracy_long[aggregated_accuracy_long['accuracy_type'] == 'response_correct_mm'],
                kind="point", dodge=0.25, join=True, errorbar = ("ci", 95), palette=['darkorange','indigo','steelblue'],
                height=4, aspect=1.2, legend=False, markers=["o", "s", "^"])

# make line thiner
for ax in g.axes.flat:
    ax.set_ylim(0.42, 1)
    ax.axhline(0.5, ls='--', color='gray')
    for line in ax.lines:
      line.set_linewidth(2)

# set y-axis label 
g.set_ylabels("Accuracy")

# remove x-axis label
g.set_xlabels("")

# add legend to the top center outside of the plot and in 3 rows
handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False)

# add col names to the top of each column
g.axes[0,0].set_title("Easy")
g.axes[0,1].set_title("Hard")


# Evaluate the difference between MM and actual accuracy with a paired t-test (for participants)
from scipy import stats

actual_accuracy = trials_PEC.groupby(['participant_id', 'learning_condition', 'fsm_type', 
                                          'test_condition'])['response_correct'].mean().reset_index()
mental_model_accuracy = trials_PEC.groupby(['participant_id', 'learning_condition', 'fsm_type', 
                                          'test_condition'])['response_correct_mm'].mean().reset_index()

# make sure that the order of the rows is the same for both dataframes
actual_accuracy = actual_accuracy.sort_values(by=['participant_id', 'learning_condition', 'fsm_type', 'test_condition'])
mental_model_accuracy = mental_model_accuracy.sort_values(by=['participant_id', 'learning_condition', 'fsm_type', 'test_condition'])

# check if participant order is the same in both dataframes
np.array_equal(actual_accuracy['participant_id'], mental_model_accuracy['participant_id'])

# run paired t-test
stats.ttest_rel(actual_accuracy['response_correct'], mental_model_accuracy['response_correct_mm'], nan_policy='omit')

# round t to 2 decimal places and p to 3 decimal places
print('t = ' + str(round(stats.ttest_rel(actual_accuracy['response_correct'], mental_model_accuracy['response_correct_mm'])[0], 3)))
print('p = ' + str(round(stats.ttest_rel(actual_accuracy['response_correct'], mental_model_accuracy['response_correct_mm'])[1], 4)))
print(stats.ttest_rel(actual_accuracy['response_correct'], mental_model_accuracy['response_correct_mm']))

# means and standard deviations + number of trials and round float to 2 decimal places
actual_accuracy['response_correct'].describe().round(2)
mental_model_accuracy['response_correct_mm'].describe().round(2)


# run one-sample t-test to test if accuracy is significantly different from chance level (0.5)
for test_condition in aggregated_accuracy['test_condition'].unique():
    for fsm_type in aggregated_accuracy['fsm_type'].unique():
        for trial_type in aggregated_accuracy['trial_type'].unique():
            results = stats.ttest_1samp(aggregated_accuracy[(aggregated_accuracy['test_condition'] == test_condition) & 
                                                    (aggregated_accuracy['fsm_type'] == fsm_type) & 
                                                    (aggregated_accuracy['trial_type'] == trial_type)]['response_correct_mm'], 
                                                    popmean = 0.5, alternative='greater')
            print(test_condition + ' ' + fsm_type + ' ' + trial_type + ': t = ' + str(round(results[0], 2)) + '. p = ' + str(round(results[1], 3)))
            print(results)
            
# run pairwise t-tests to check if mental model accuracy is different between prediction, explanation and control for each FSM type and trial type (use Bonferroni correction)
import pingouin as pg

mental_model_accuracy = aggregated_accuracy_long[aggregated_accuracy_long['accuracy_type'] == 'response_correct_mm']

results = pd.DataFrame()
for fsm_type in mental_model_accuracy['fsm_type'].unique():
    subset = mental_model_accuracy[(mental_model_accuracy['fsm_type'] == fsm_type)].dropna() 
    results_current = pg.pairwise_tests(dv='accuracy', within='trial_type', subject='participant_id', between='test_condition', data=subset, padjust='fdr_bh', effsize='cohen', return_desc=True)
    results_current['fsm_type'] = fsm_type
    results = results.append(results_current)