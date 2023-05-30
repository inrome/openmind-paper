import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg

# Load data
trials_PEC_path = '../outputs/all_trials_with_Ri.csv'
trials_PEC = pd.read_csv(trials_PEC_path)

# aggregate accuracy data 
aggregated_accuracy = trials_PEC.groupby(['participant_id', 'learning_condition', 'fsm_type', 
                                          'test_condition', 'trial_type'])['response_correct_mm', 'response_correct_mm_an'].mean().reset_index()

aggregated_accuracy_long = pd.melt(aggregated_accuracy, id_vars=['participant_id', 'learning_condition', 'fsm_type', 'test_condition', 'trial_type'],
                            value_vars=['response_correct_mm', 'response_correct_mm_an'], var_name='accuracy_type', value_name='accuracy')

# Evaluate the difference between MM and actual accuracy with a paired t-test (for participants)

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
            test = stats.ttest_1samp(aggregated_accuracy[(aggregated_accuracy['test_condition'] == test_condition) & 
                                                    (aggregated_accuracy['fsm_type'] == fsm_type) & 
                                                    (aggregated_accuracy['trial_type'] == trial_type)]['response_correct_mm'], 
                                                    popmean = 0.5, alternative='greater', )
            print(test_condition + ' ' + fsm_type + ' ' + trial_type + ': t = ' + str(round(test[0], 2)) + '. p = ' + str(round(test[1], 3)))
            
            # get df from ttest_1samp
            df = aggregated_accuracy[(aggregated_accuracy['test_condition'] == test_condition) &
                                                    (aggregated_accuracy['fsm_type'] == fsm_type) &     
                                                    (aggregated_accuracy['trial_type'] == trial_type)]['response_correct_mm'].count() - 1
            print('df = ' + str(df))
            
            
# run pairwise t-tests to check if mental model accuracy is different between prediction, explanation and control for each FSM type and trial type (use Bonferroni correction)
mental_model_accuracy = aggregated_accuracy_long[aggregated_accuracy_long['accuracy_type'] == 'response_correct_mm']

results = pd.DataFrame()
for fsm_type in mental_model_accuracy['fsm_type'].unique():
    subset = mental_model_accuracy[(mental_model_accuracy['fsm_type'] == fsm_type)].dropna() 
    results_current = pg.pairwise_tests(dv='accuracy', within='trial_type', subject='participant_id', between='test_condition', data=subset, padjust='fdr_bh', effsize='cohen', return_desc=True)
    results_current['fsm_type'] = fsm_type
    results = results.append(results_current)