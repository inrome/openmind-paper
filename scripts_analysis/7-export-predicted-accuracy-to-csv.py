import os
import pandas as pd
import numpy as np
import pickle

# Read data
current_dir = os.path.dirname(os.path.abspath(__file__))
imported_data_path = os.path.join(current_dir, '../data/imported_clean_data.pickle')

with open(imported_data_path, 'rb') as f:
    imported_data = pickle.load(f)

imported_participants = imported_data[0] # select only participants dataframe

sample_path = os.path.join(current_dir, '../outputs/trials_with_max_beta.pickle')
with open(sample_path, 'rb') as f:
    sample = pickle.load(f)

# export 'predicted_accuracy' to csv file:
results_all = pd.DataFrame()
for participant_id in sample.keys():
    results = pd.DataFrame() 
    tmp_acc = pd.DataFrame(sample[participant_id]['predicted_accuracy'], index=[0])
    results = pd.concat([results, tmp_acc], axis=1)

    # add participant_id and task to results
    results['participant_id'] = participant_id
    results['task'] = sample[participant_id]['task']

    # reorder columns
    cols = results.columns.tolist() # get columns as a list
    cols = cols[-2:] + cols[:-2] # move last two columns to the front of the list
    results = results[cols] # reorder columns

    # reformat results condition columns (visible, hidden, hidden_an) into long format
    id_vars = ['participant_id', 'task']
    value_vars=cols[2:] # everything except participant_id and task

    results = pd.melt(results, id_vars=id_vars, value_vars=value_vars,
                      var_name='condition', value_name='predicted_accuracy')

    # add fsm to results
    # if fsm_number == 21, then fsm_type = 'easy', else fsm_type = 'hard'
    results['fsm_number'] = imported_participants[imported_participants['participant_id'] == participant_id]['fsm_number'].values[0]
    results['fsm_type'] = 'easy' if results['fsm_number'].values[0] == 21 else 'hard'

    # add mm_accuracy to results
    results_accuracy = pd.DataFrame(sample[participant_id]['mm_accuracy'], index=[0]) 
    results_accuracy = pd.melt(results_accuracy, value_vars=['visible', 'hidden', 'hidden_an'],
                                 var_name='condition', value_name='mm_accuracy') # reformat into long format
    results = pd.merge(results, results_accuracy, on='condition', how='left') 
    

    #add 'count_nan'
    results_count_nan = pd.DataFrame(sample[participant_id]['count_nan'], index=[0])
    results_count_nan = pd.melt(results_count_nan, value_vars=['visible', 'hidden', 'hidden_an'],
                                var_name='condition', value_name='count_nan') # reformat into long format
    results = pd.merge(results, results_count_nan, on='condition', how='left')

    results['learning_condition'] = sample[participant_id]['trials_vis']['learning_condition'].values[0]

    results_all = pd.concat([results_all, results], ignore_index=True)

save_path_acc = os.path.join(current_dir, '../outputs/predicted_accuracy.csv')

results_all.to_csv(save_path_acc, index=False)