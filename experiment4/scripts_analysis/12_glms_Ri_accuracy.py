import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Load data
trials_PEC_path = '../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv'
trials_PEC = pd.read_csv(trials_PEC_path)

df = trials_PEC.copy()

# rename R_i_mm_eps_0.1 column to R_i_mm_eps_0_1
df = df.rename(columns={'R_i_mm_eps_0.1': 'R_i_mm_eps_0_1'})
df = df.rename(columns={'R_i_mm_an_eps_0.1': 'R_i_mm_an_eps_0_1'})

# convert variables to categorical
df['participant_id'] = df['participant_id'].astype('category')
df['trial_type'] = pd.Categorical(df['trial_type'], categories=["visible", "hidden"], ordered=False)

# Generalized (Binomial) Linear Models ####

glms = {} # dictionary to store all glms
for fsm_type in ["easy", "hard"]:
    for feedback_condition in ["prediction", "control", "explanation"]:
        for test_condition in ["prediction", "control", "explanation"]:
            for trial_type in ["visible", "hidden"]:
                dataset = df[(df['fsm_type'] == fsm_type) & 
                            (df['test_condition'] == test_condition) & 
                            (df['trial_type'] == trial_type) & 
                            (df['feedback_condition'] == feedback_condition)
                            ]
                md = smf.glm('response_correct_mm ~ 0 + R_i_mm_eps_0_1', 
                    data = dataset, family = sm.families.Binomial()).fit()
                print(md.summary())
                glms[(fsm_type,feedback_condition, test_condition, trial_type)] = md

# see results
glms[("easy", "prediction", "prediction", "visible")].summary()

# save each model results in a dataframe
glms_df = pd.DataFrame(columns=['fsm_type', 'feedback_condition', 'test_condition', 'trial_type', 'coef', '0.025', '0.975', 'SE', 'z', 'p', 'LL'])
for key, value in glms.items():
    value.summary()
    fsm_type,feedback_condition, test_condition, trial_type = key
    coef = value.params[0]
    conf_int = value.conf_int(alpha=0.05, cols=None)
    SE_value = value.bse[0]
    z_value = value.tvalues[0]
    p_value = value.pvalues[0]
    LL = value.llf

    glm_df = pd.DataFrame([[fsm_type, feedback_condition, test_condition, trial_type, coef, conf_int[0][0], conf_int[1][0], SE_value, z_value, p_value, LL]],
                            columns=['fsm_type', 'feedback_condition','test_condition', 'trial_type', 'coef', '0.025', '0.975', 'SE', 'z', 'p', 'LL'])
    
    glms_df = pd.concat([glms_df, glm_df], ignore_index=True)

# round all values to 2 decimal places except p values (3 decimal places)
glms_df = glms_df.round({'coef': 2, '0.025': 2, '0.975': 2, 'SE': 2, 'z': 2, 'LL': 2})
glms_df['p'] = glms_df['p'].round(3)
    

# Same glm but only hidden case with Alternative Neglect ####
df_hidden = df[df['trial_type'] == 'hidden']

glms_an = {} 

for fsm_type in ["easy", "hard"]:
    for feedback_condition in ["prediction", "control", "explanation"]:
        for test_condition in ["prediction", "control", "explanation"]:
            dataset = df_hidden[(df_hidden['fsm_type'] == fsm_type) & 
                            (df_hidden['test_condition'] == test_condition) & 
                                (df_hidden['feedback_condition'] == feedback_condition)]
            md = smf.glm('response_correct_mm_an ~ 0 + R_i_mm_an_eps_0_1', 
                    data = dataset, family = sm.families.Binomial()).fit()
            print(md.summary())
            glms_an[(fsm_type,feedback_condition, test_condition)] = md

# see results
glms_an[("easy", "prediction", "prediction")].summary()

# save each model results in a dataframe
glms_an_df = pd.DataFrame(columns=['fsm_type', 'feedback_condition','test_condition', 'coef', '0.025', '0.975', 'SE', 'z', 'p', 'LL'])

for key, value in glms_an.items():
    value.summary()
    fsm_type, feedback_condition, test_condition = key
    coef = value.params[0]
    conf_int = value.conf_int(alpha=0.05, cols=None)
    SE_value = value.bse[0]
    z_value = value.tvalues[0]
    p_value = value.pvalues[0]
    LL = value.llf

    glm_df = pd.DataFrame([[fsm_type,feedback_condition, test_condition, coef, conf_int[0][0], conf_int[1][0], SE_value, z_value, p_value, LL]],
                            columns=['fsm_type', 'feedback_condition','test_condition', 'coef', '0.025', '0.975', 'SE', 'z', 'p', 'LL'])
    
    glms_an_df = pd.concat([glms_an_df, glm_df], ignore_index=True)

# round all values to 2 decimal places except p values (3 decimal places)
glms_an_df = glms_an_df.round({'coef': 2, '0.025': 2, '0.975': 2, 'SE': 2, 'z': 2, 'LL': 2})
glms_an_df['p'] = glms_an_df['p'].round(3)

# add LL_an to glms_df dataframe (merge on fsm_type, test_condition and trial_type)
glms_an_df['trial_type'] = "hidden"
glms_an_df = glms_an_df.rename(columns={'LL': 'LL_an'}) # rename LL column to LL_an

# remove coef, 0.025, 0.975, SE, z, p columns
tmp = glms_an_df.drop(columns=['coef', '0.025', '0.975', 'SE', 'z', 'p'])

glms_df = pd.merge(glms_df, tmp, how='left', on=['fsm_type','feedback_condition', 'test_condition', 'trial_type'])

# add LL_comparison column with "AN" if LL_an > LL, "Normative" if LL_an < LL and "" if LL_an == LL or NaN
glms_df['LL_comparison'] = np.where(glms_df['LL_an'] > glms_df['LL'], "AN", 
                                    np.where(glms_df['LL_an'] < glms_df['LL'], "Normative", ""))

glms_df['LL_diff'] = glms_df['LL_an'] - glms_df['LL'] # add LL_diff column with LL_an - LL

glms_df = glms_df.sort_values(by=['fsm_type', 'trial_type', 'feedback_condition','test_condition'])

# cleate long version of glms_df
glms_an_df['trial_type'] = "Hidden\nAN"
glms_df['trial_type'] = glms_df['trial_type'].replace(['visible', 'hidden'], ['Visible', 'Hidden\nNormative'])
glms_df_long = pd.concat([glms_df, glms_an_df], ignore_index=True)

# export glms_df_long to csv
glms_df_long.to_csv('../outputs/glms_Ri_accuracy_long.csv', index=False)
