import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
trials_PEC_path = '../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv'
trials_PEC = pd.read_csv(trials_PEC_path)

# aggregate accuracy data 
aggregated_accuracy = trials_PEC.groupby(['participant_id', 'test_order', 'fsm_type', 
                                          'test_condition', 'trial_type'])['response_correct_mm'].mean().reset_index()
                                         
# add accuracy for AN model
aggregated_accuracy = pd.merge(aggregated_accuracy, 
                                trials_PEC.groupby(['participant_id', 'test_order', 'fsm_type',
                                                    'test_condition', 'trial_type'])['response_correct_mm_an'].mean().reset_index(),
                                on=['participant_id', 'test_order', 'fsm_type', 'test_condition', 'trial_type'],
                                how='left')


# rename variables for better labeling
aggregated_accuracy['fsm_type'] = aggregated_accuracy['fsm_type'].replace({'easy': 'Easy FSM', 'hard': 'Hard FSM'})
aggregated_accuracy['test_condition'] = aggregated_accuracy['test_condition'].replace({'prediction': 'Prediction', 
                                                                                       'control': 'Control', 
                                                                                       'explanation': 'Explanation'})
aggregated_accuracy['trial_type'] = aggregated_accuracy['trial_type'].replace({'visible': 'Visible', 'hidden': 'Hidden\nNormative'})

# convert trial_type and fsm_type to categorical variable with ordered levels
aggregated_accuracy['trial_type'] = pd.Categorical(aggregated_accuracy['trial_type'], categories=['Visible', 'Hidden\nNormative'], ordered=True)
aggregated_accuracy['fsm_type'] = pd.Categorical(aggregated_accuracy['fsm_type'], categories=['Easy FSM', 'Hard FSM'], ordered=True)

aggregated_accuracy_long = pd.melt(aggregated_accuracy, id_vars=['participant_id', 'fsm_type', 'test_condition', 'trial_type'],
                            value_vars=['response_correct_mm', 'response_correct_mm_an'], var_name='accuracy_type', value_name='accuracy')

# remove NaNs from visible AN trials
aggregated_accuracy_long = aggregated_accuracy_long[~((aggregated_accuracy_long['trial_type'] == 'Visible') & 
                                                      (aggregated_accuracy_long['accuracy_type'] == 'response_correct_mm_an'))]

# create new trial_type2 variable with Hidden AN trials if accuracy_type == 'response_correct_mm_an'
aggregated_accuracy_long['trial_type2'] = np.where(aggregated_accuracy_long['accuracy_type'] == 'response_correct_mm_an', 'Hidden\nAN', aggregated_accuracy_long['trial_type'])

# check n trials per condition
aggregated_accuracy_long.groupby(['fsm_type', 'test_condition', 'trial_type2', 'accuracy_type'])['accuracy'].count()

aggregated_accuracy_long.drop(['accuracy_type'], axis=1, inplace=True)

# order trial_type2 variable for plotting
aggregated_accuracy_long['trial_type2'] = pd.Categorical(aggregated_accuracy_long['trial_type2'], categories=['Visible', 'Hidden\nNormative', 'Hidden\nAN'], ordered=True)

# order test_condition variable for plotting
aggregated_accuracy_long['test_condition'] = pd.Categorical(aggregated_accuracy_long['test_condition'], categories=['Control', 'Prediction', 'Explanation'], ordered=True)
# Plot accuracy
sns.set(style="whitegrid", font_scale=1.4) 

g = sns.catplot(x="trial_type2", y="accuracy", hue="test_condition", col="fsm_type", data=aggregated_accuracy_long, 
                kind="point", dodge=0.25, join=True, errorbar = ("ci", 95), palette=['darkorange','indigo','steelblue'],
                height=4.5, aspect=1.2, legend=False, markers=["o", "s", "^"])

# make line thiner
for ax in g.axes.flat:
    ax.set_ylim(0.42, 1)
    ax.axhline(0.5, ls='--', color='gray')
    for line in ax.lines:
      line.set_linewidth(2)

g.set_xlabels("")
g.set_ylabels("Accuracy")
g.set_titles("{col_name}")

# add legend to the top center outside of the plot
handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=3, 
             bbox_to_anchor=(0.5, 1.08), frameon=False)




