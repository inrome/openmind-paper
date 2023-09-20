import pickle
import pandas as pd
import numpy as np
import os

# Load data
trials = pd.read_csv("../outputs/trials_test_with_PEC_mm_probs_Ri_withAccuracy.csv")

trials_feedback = trials[trials['phase'] == 'feedback']

# mean response_correct per participant groupped by feedback_condition, fsm_type, and trial_type
accuracy = trials_feedback.groupby(['participant_id','feedback_condition', 'fsm_type', 'trial_type', 'attempt'])['response_correct'].mean().reset_index()

# new var "passed_threshold" that is True if response_correct >= 0.65
accuracy['passed_threshold'] = accuracy['response_correct'] >= 0.65

# see if participant passed the threshold in trial_type == 'visible'
participants_visible = accuracy[accuracy['trial_type'] == 'visible']
participants_visible = participants_visible.rename(columns={'passed_threshold': 'passed_threshold_visible'})
participants_visible = participants_visible.drop(columns=['response_correct', 'trial_type', 'attempt'])

# same for trial_type == 'hidden'
participants_hidden = accuracy[accuracy['trial_type'] == 'hidden']
participants_hidden = participants_hidden.rename(columns={'passed_threshold': 'passed_threshold_hidden'})

# drop other columns
participants_hidden = participants_hidden.drop(columns=['response_correct', 'trial_type', 'attempt'])

# merge the two dataframes
participants = pd.merge(participants_visible, participants_hidden, on=['participant_id','feedback_condition', 'fsm_type'])

# add handy columns
participants["passed_both"] = participants["passed_threshold_visible"] & participants["passed_threshold_hidden"]
participants["passed_visible_not_hidden"] = participants["passed_threshold_visible"] & ~participants["passed_threshold_hidden"]
participants["passed_hidden_not_visible"] = ~participants["passed_threshold_visible"] & participants["passed_threshold_hidden"]


participants["passed_both"].sum() # number of participants that passed the threshold in both trial types
participants["passed_visible_not_hidden"].sum() # number of participants that passed the threshold in visible but not hidden
participants["passed_hidden_not_visible"].sum() # number of participants that passed the threshold in hidden but not visible


# The vast majority of participants passed the threshold in both trial types 

# mean and sd attempts and mean and sd response_correct per participant groupped by feedback_condition, fsm_type, and trial_type
mean_acc = trials_feedback.groupby(['fsm_type', 'feedback_condition', 'trial_type'])['response_correct'].agg(['mean', 'std']).round(2).reset_index()
mean_attempts = trials_feedback.groupby(['fsm_type', 'feedback_condition', 'trial_type'])['attempt'].agg(['mean', 'std']).round(2).reset_index()

# use t-tests to compare hard and easy conditions in the number of attempts

def my_ttest(subset1, subset2, paired = False, **kwargs):
    from scipy.stats import ttest_ind
    from scipy.stats import ttest_rel
    """ Wrapper around scipy.stats.ttest_ind, which returns APA style output """
    if paired == True:
        ttest = ttest_rel(subset1, subset2, **kwargs)
        df = len(subset1) - 1
    else: 
        ttest = ttest_ind(subset1, subset2, **kwargs)
        df = len(subset1) + len(subset2) - 2
    M1 = subset1.mean().round(2)
    SD1 = subset1.std().round(2)
    M2 = subset2.mean().round(2)
    SD2 = subset2.std().round(2)

    t = ttest.statistic.round(2)
    p = ttest.pvalue.round(3)
    p = "<.001" if p < .001 else p

    
    print("t(%s) = %s, p = %s" % (df, t, p))
    print("M1 = %s, SD1 = %s" % (M1, SD1))
    print("M2 = %s, SD2 = %s" % (M2, SD2))

    return ttest

# compare hard and easy conditions
subset1 = accuracy[accuracy['fsm_type'] == 'hard']['attempt']
subset2 = accuracy[accuracy['fsm_type'] == 'easy']['attempt']

ttest_fsm = my_ttest(subset1, subset2)

# same for explanation and control + prediction
subset1 = accuracy[accuracy['feedback_condition'] == 'explanation']['attempt']
subset2 = accuracy[accuracy['feedback_condition'] != 'explanation']['attempt']

ttest_forms_of_q = my_ttest(subset1, subset2)

# same for visible and hidden
subset1 = accuracy[accuracy['trial_type'] == 'hidden']['attempt']
subset2 = accuracy[accuracy['trial_type'] == 'visible']['attempt']

ttest_trial_type = my_ttest(subset1, subset2, paired = True)

accuracy['attempt'].mean() # mean number of attempts
accuracy['attempt'].std() # sd number of attempts