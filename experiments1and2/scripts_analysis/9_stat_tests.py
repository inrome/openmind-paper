import pandas as pd
import numpy as np
import pingouin as pg # documentation: https://pingouin-stats.org/

# Read data 
current_dir = os.path.dirname(os.path.abspath(__file__))
predicted_accuracy_path = os.path.join(current_dir, '../outputs/predicted_accuracy.csv')
ss_betas = pd.read_csv(predicted_accuracy_path)

# function that generates a subset of the data
def subset_data(data, task=None, condition=None, fsm_type=None, learning_condition=None):
    if task:
        data = data[data['task'].isin(task)]
    if condition:
        data = data[data['condition'].isin(condition)]
    if fsm_type:
        data = data[data['fsm_type'].isin(fsm_type)]
    if learning_condition:
        data = data[data['learning_condition'].isin(learning_condition)]

    return data


# function that runs the statistical test for all factors that have 2 or more levels
def stat_test(data, task = ['control', 'prediction', 'explanation'], 
                condition = ['visible', 'hidden', 'hidden_an', 'hidden_normative_subset','hidden_an_subset'], 
                fsm_type = ['easy', 'hard'], 
                learning_condition = ['Experiment 1 (no preview)', 'Experiment 2 (test preview)']):
    # subset the data
    data = subset_data(data, task, condition, fsm_type, learning_condition)
    
    
    factors = ['task', 'condition', 'fsm_type'] # all factors
    factors = [f for f in factors if len(data[f].unique()) >= 2] # get all factors that have 2 or more levels

    
    between_factors = [f for f in factors if f != 'condition'] # get all factors except 'condition'
    
    within_factor = 'condition' if 'condition' in factors else None # add within factor if there is one
    within_id = 'participant_id' if 'condition' in factors else None # add within id if there is one
    
    # set up the parameters for the statistical test
    dv = 'predicted_accuracy' # the dependent variable

    parametric = False 
    adjust_method = "fdr_bh" # correction method for multiple comparisons

    # run the statistical test
    tests = pd.DataFrame()
    

    # get all factors except 'condition'
    between_factors = [f for f in factors if f != 'condition']

    # add within factor if there is one
    if len(between_factors) == 0:
        tests = pg.pairwise_tests(data, dv=dv, within=within_factor, subject=within_id,
                                    parametric=parametric, padjust=adjust_method, 
                                    effsize='cohen', return_desc=True)
    elif len(between_factors) == 1: 
        tests = pg.pairwise_tests(data, dv=dv, within=within_factor, subject=within_id, between = between_factors,
                                    parametric=parametric, padjust=adjust_method, 
                                    effsize='cohen', return_desc=True)
    elif len(between_factors) == 2:
        tests = pd.DataFrame()
        for fsm in data['fsm_type'].unique():
            data_fsm = data[data['fsm_type'] == fsm]
            tmp = pg.pairwise_tests(data_fsm, dv=dv, within=within_factor, subject=within_id, between = 'task',
                                    parametric=parametric, padjust=adjust_method, 
                                    effsize='cohen', return_desc=True)
            tmp['subset'] = fsm
            tests = pd.concat([tests, tmp], axis=0, ignore_index=True)
        
        for task in data['task'].unique():
            data_task = data[data['task'] == task]
            tmp = pg.pairwise_tests(data_task, dv=dv, within=within_factor, subject=within_id, between = "fsm_type",
                                    parametric=parametric, padjust=adjust_method, 
                                    effsize='cohen', return_desc=True)
            tmp['subset'] = task
            tests = pd.concat([tests, tmp], axis=0, ignore_index=True)         
    # if learning_condition has 2 levels, see if it affect the dv
    if len(data['learning_condition'].unique()) == 2:
        tmp = pg.pairwise_tests(data, dv=dv, within=within_factor, subject=within_id, between = 'learning_condition',
                                    parametric=parametric, padjust=adjust_method, 
                                    effsize='cohen', return_desc=True)
        tests = pd.concat([tests, tmp], axis=0, ignore_index=True)

    # sort the dataframe
    if tests is not None:
        tests = tests.sort_values(by='cohen', key=abs, ascending=False)
        if 'p-corr' in tests.columns:
            tests['p-value'] = tests['p-corr'].fillna(tests['p-unc'])
            tests = tests.drop(columns=['Paired', 'Parametric', 'alternative', 'p-unc', 'p-corr', 'p-adjust'])

        # round all float values to 3 decimals
        tests = tests.round(3)

        # remove columns "Paired", "Parametric", "alternative", "p-unc", "p-corr", "p-adjust"
    
        # if there is subset column, move it to the front
        if 'subset' in tests.columns:
            cols = tests.pop('subset')
            tests.insert(0, cols.name, cols)
        
        # move Contrast column to the end
        cols = tests.pop('Contrast')
        tests.insert(len(tests.columns), cols.name, cols)
    return tests


# Are there differences between PEC
tests = stat_test(ss_betas, task = ['explanation'],
                condition = ['visible', 'hidden'],
                fsm_type = ['hard', 'easy'], 
                learning_condition = ['Experiment 1 (no preview)', 'Experiment 2 (test preview)'])

