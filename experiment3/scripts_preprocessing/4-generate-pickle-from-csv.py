# Create a pickle containing all data from the csv files

import pandas as pd
import pickle
import os

def load_experiment_data(participants_path, trials_learning_path, trials_test_path):
    exp_data = {}
    exp_data['participants'] = pd.read_csv(participants_path, sep=",")
    exp_data['trials_learning'] = pd.read_csv(trials_learning_path, sep=",")
    exp_data['trials_test'] = pd.read_csv(trials_test_path, sep=",")

    return exp_data

current_dir = os.path.dirname(os.path.abspath(__file__))
participants_path = os.path.join(current_dir, '../data/participants.csv')
trials_learning_path = os.path.join(current_dir, '../data/trials_learning.csv')
trials_test_path = os.path.join(current_dir, '../data/trials_test.csv')

exp_data = load_experiment_data(participants_path, trials_learning_path, trials_test_path)

with open(os.path.join(current_dir, '../data/imported_clean_data.pickle'), 'wb') as f:
    pickle.dump([exp_data['participants'], 
                 exp_data['trials_learning'], 
                 exp_data['trials_test']], f)
    
