# Description: mixed-effects logistic regression analysis for Experiment 1 in CogSci paper 
# (response_correct ~ test_condition * trial_type + (1 | participant_id)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from patsy import dmatrices
import statsmodels.genmod.bayes_mixed_glm as smgb

# load data
df = pd.read_csv('../outputs/all_trials_with_Ri.csv')

# remove Experiment 2 data
df = df[df['learning_condition'] == 'Experiment 1 (no preview)']

# create subsets
df_easy = df[df['fsm_type'] == 'easy']
df_hard = df[df['fsm_type'] == 'hard']
df_p = df[df['test_condition'] == 'prediction']
df_c = df[df['test_condition'] == 'control']
df_e = df[df['test_condition'] == 'explanation']

# means and standard deviations + number of trials and round float to 2 decimal places
df.groupby(['test_condition', 'trial_type', 'fsm_type'])['response_correct'].agg(['mean', 'std', 'count']).round(2)
df.groupby(['test_condition', 'trial_type', 'fsm_type'])['response_correct_mm'].agg(['mean', 'std', 'count']).round(2)


# mean accuracy for each participant
df.groupby(['participant_id'])['response_correct'].agg(['mean']).round(2)

'''model = smgb.BinomialBayesMixedGLM.from_formula('response_correct ~ test_condition * trial_type', 
                                                {'participant_id': '1 + participant_id'}, df)

result = model.fit_vb()
print(result.summary())'''