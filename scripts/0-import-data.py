import pandas as pd
import pickle

def load_experiment_data(exp_num, participants_path, trials_learning_path, trials_prediction_path, trials_control_path, trials_explanation_path):
    exp_data = {}
    exp_data['participants'] = pd.read_csv(participants_path, sep=",")
    exp_data['trials_learning'] = pd.read_csv(trials_learning_path, sep=",")
    exp_data['trials_prediction'] = pd.read_csv(trials_prediction_path, sep=",")
    exp_data['trials_control'] = pd.read_csv(trials_control_path, sep=",")
    exp_data['trials_explanation'] = pd.read_csv(trials_explanation_path, sep=",")

    for key, df in exp_data.items(): # add experiment number to each dataframe
        df['experiment'] = exp_num

    return exp_data
#%%
exp1_data = load_experiment_data(1,
                                 '../exp-1/data_clean/2023-01-28_participants_clean_an-exp1_n-91.csv',
                                 '../exp-1/data_clean/2023-01-28_learning_clean_an-exp1_n-91.csv',
                                 '../exp-1/data_clean/2023-01-28_test_prediction_clean_an-exp1_n-91.csv',
                                 '../exp-1/data_clean/2023-01-28_test_control_clean_an-exp1_n-91.csv',
                                 '../exp-1/data_clean/2023-01-28_test_explanation_clean_an-exp1_n-91.csv')
#%%
exp2_data = load_experiment_data(2,
                                 '../exp-2/data_preprocessed/2023-02-16_participants__an-exp2_n-24.csv',
                                 '../exp-2/data_preprocessed/2023-02-17_learning_an-exp2_n-24.csv',
                                 '../exp-2/data_preprocessed/2023-02-17_test_prediction__an-exp2_n-8.csv',
                                 '../exp-2/data_preprocessed/2023-02-17_test_control__an-exp2_n-8.csv',
                                 '../exp-2/data_preprocessed/2023-02-17_test_explanation__an-exp2_n-8.csv')

#%%
def compare_columns(exp1_data, exp2_data):
    for key in exp1_data.keys():
        exp1_columns = set(exp1_data[key].columns)
        exp2_columns = set(exp2_data[key].columns)

        missing_in_exp1 = exp2_columns - exp1_columns
        missing_in_exp2 = exp1_columns - exp2_columns

        if missing_in_exp1 or missing_in_exp2:
            print(f"Columns mismatch for {key}:")

            if missing_in_exp1:
                print(f"  Columns missing in exp1_data: {', '.join(missing_in_exp1)}")

            if missing_in_exp2:
                print(f"  Columns missing in exp2_data: {', '.join(missing_in_exp2)}")
        else:
            print(f"No column mismatch for {key}")

compare_columns(exp1_data, exp2_data)

#%%
# resolve column mismatch in trials_learning
# rename learning_counter to trial_number
exp2_data['trials_learning'] = exp2_data['trials_learning'].rename(columns={'learning_counter': 'trial_number'})

# add fsm_type column in exp1_data with values "easy" and "hard" for fsm_number 21 and 22 respectively
exp1_data['trials_learning']['fsm_type'] = exp1_data['trials_learning']['fsm_number'].apply(lambda x: 'easy' if x == 21 else 'hard')

# remove fsm_number column from exp1_data
exp1_data['trials_learning'] = exp1_data['trials_learning'].drop(columns=['fsm_number'])

# remove sender and test_condition columns from exp2_data
exp2_data['trials_learning'] = exp2_data['trials_learning'].drop(columns=['test_condition', 'sender'])
#%%
# resolve column mismatch in trials_prediction
# rename correctResponse in exp2_data to prediction_correct
exp2_data['trials_prediction'] = exp2_data['trials_prediction'].rename(columns={'correctResponse': 'prediction_correct'})

# remove response_time and test_condition columns from exp2_data
exp2_data['trials_prediction'] = exp2_data['trials_prediction'].drop(columns=['test_condition', 'response_time'])

#  trials_control
# rename correctResponse in exp2_data to control_correct
exp2_data['trials_control'] = exp2_data['trials_control'].rename(columns={'correctResponse': 'control_correct'})

# remove response_time, duration, and test_condition columns from exp2_data
exp2_data['trials_control'] = exp2_data['trials_control'].drop(columns=['test_condition', 'response_time', 'duration'])

# trials_explanation
# rename correctResponse in exp2_data to explanation_correct
exp2_data['trials_explanation'] = exp2_data['trials_explanation'].rename(columns={'correctResponse': 'explanation_correct'})

# remove response_time and test_condition columns from exp2_data
exp2_data['trials_explanation'] = exp2_data['trials_explanation'].drop(columns=['test_condition', 'response_time'])

#%%
merged_data = {}
for key in exp1_data.keys():
    merged_data[key] = pd.concat([exp1_data[key], exp2_data[key]], ignore_index=True)

with open('../data_merged/imported_clean_data.pickle', 'wb') as f:
    pickle.dump([merged_data['participants'], merged_data['trials_learning'], merged_data['trials_prediction'],
                 merged_data['trials_control'], merged_data['trials_explanation']], f)
