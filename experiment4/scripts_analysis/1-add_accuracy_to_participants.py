# Add mean accuracy in feedback and transfer phases to participants dataframe

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

participants = pd.read_csv("../data/participants.csv")
trials = pd.read_csv("../data/trials_test.csv")

# calculate mean accuracy in feedback phase for each participant, test_condition, trial_type, and attempt
accuracy_feedback = trials.groupby(['participant_id','fsm_type','phase','feedback_condition','test_condition', 'trial_type', 'attempt'])['response_correct'].mean().reset_index(name='accuracy')

# add 'attempts_max' column 
accuracy_feedback['attempts_max'] = accuracy_feedback.groupby(['participant_id','fsm_type','phase','test_condition', 'trial_type'])['attempt'].transform('max')

# select only rows with attempt == attempts_max
accuracy_feedback = accuracy_feedback.loc[accuracy_feedback['attempt'] == accuracy_feedback['attempts_max']]

# same for trials dataframe 
trials['attempts_max'] = trials.groupby(['participant_id','fsm_type','phase','test_condition', 'trial_type'])['attempt'].transform('max')
trials = trials.loc[(trials['attempt'].isnull()) | (trials['attempt'] == trials['attempts_max'])]
trials = trials.drop(['attempts_max'], axis=1)

# drop 'attempts_max' columns
accuracy_feedback = accuracy_feedback.drop(['attempts_max'], axis=1)

# means for transfer phase
accuracy_transfer = trials.groupby(['participant_id','fsm_type','feedback_condition','phase','test_condition', 'trial_type'])['response_correct'].mean().reset_index(name='accuracy')

# remove rows with phase == 'feedback'
accuracy_transfer = accuracy_transfer.loc[accuracy_transfer['phase'] == 'transfer']

# merge accuracy_feedback and accuracy_transfer
accuracy = pd.concat([accuracy_feedback, accuracy_transfer]).sort_values(by=['participant_id', 'phase', 'test_condition', 'trial_type'])

# trial_type to categorical (easy, hard)
accuracy['trial_type'] = pd.Categorical(accuracy['trial_type'], categories=['visible', 'hidden'], ordered=True)
accuracy['fsm_type'] = pd.Categorical(accuracy['fsm_type'], categories=['easy', 'hard'], ordered=True)
accuracy['phase'] = pd.Categorical(accuracy['phase'], categories=['feedback', 'transfer'], ordered=True)

# new column "phase_condition" that combines phase (as striing) and test_condition 
accuracy['phase_condition'] = accuracy['phase'].astype(str) + '_' + accuracy['test_condition'].astype(str)

# mark participants with no feedback hidden data 
participants['bug_no_feedback_hidden'] = participants['participant_id'].isin(accuracy.loc[(accuracy['phase'] == 'feedback') &
                                                                                            (accuracy['trial_type'] == 'hidden')]['participant_id']) == False

# count bug_no_feedback_hidden by feedback_condition and fsm_type
participants_bug_no_feedback_hidden = participants.groupby(['feedback_condition', 'fsm_type'])['bug_no_feedback_hidden'].sum().reset_index(name='bug_no_feedback_hidden_count')

participants_with_bugs = participants # save participants dataframe with bug_no_feedback_hidden column before removing them

# remove participants with no feedback hidden data
participants = participants.loc[participants['bug_no_feedback_hidden'] == False]
trials = trials.loc[trials['participant_id'].isin(participants['participant_id'])]

# mark participants with no transfer phase data
participants['passed_threshold'] = participants['participant_id'].isin(accuracy.loc[accuracy['phase'] == 'transfer']['participant_id'])

# count participants who passed threshold by feedback_condition and fsm_type
participants_passed_threshold = participants.groupby(['feedback_condition', 'fsm_type'])['passed_threshold'].sum().reset_index(name='passed_threshold_count')

participants_passed_threshold['not_passed_threshold_count'] = participants.groupby(['feedback_condition', 'fsm_type'])['passed_threshold'].count().reset_index(name='not_passed_threshold_count')['not_passed_threshold_count'] - participants_passed_threshold['passed_threshold_count']

participants_passed_threshold['mean_attempts'] = accuracy.groupby(['feedback_condition', 'fsm_type'])['attempt'].mean().reset_index(name='mean_attempt')['mean_attempt'].round(2)

participants_passed_threshold['n_participants'] = participants_passed_threshold['passed_threshold_count'] + participants_passed_threshold['not_passed_threshold_count']

# n participants with no transfer phase data
participants.loc[participants['passed_threshold'] == False]['participant_id'].unique().shape[0]

# remove participants with bug_no_feedback_hidden data from accuracy dataframe
accuracy = accuracy.loc[accuracy['participant_id'].isin(participants['participant_id'])]

# export data to csv
n = accuracy['participant_id'].unique().shape[0] 
accuracy.to_csv("../data/participants_accuracy_" + "n" + str(n) + ".csv", index=False)

n = participants['participant_id'].unique().shape[0]
participants.to_csv("../data/participants_" + "n" + str(n) + ".csv", index=False)

n = trials['participant_id'].unique().shape[0]
trials.to_csv("../data/trials_test_" + "n" + str(n) + ".csv", index=False)

# remove participants with no transfer phase data

participants = participants.loc[participants['passed_threshold'] == True]
trials = trials.loc[trials['participant_id'].isin(participants['participant_id'])]
accuracy = accuracy.loc[accuracy['participant_id'].isin(participants['participant_id'])]

# export data to csv again
n = accuracy['participant_id'].unique().shape[0] 
accuracy.to_csv("../data/participants_accuracy_" + "n" + str(n) + ".csv", index=False)

n = participants['participant_id'].unique().shape[0]
participants.to_csv("../data/participants_" + "n" + str(n) + ".csv", index=False)

n = trials['participant_id'].unique().shape[0]
trials.to_csv("../data/trials_test_" + "n" + str(n) + ".csv", index=False)
