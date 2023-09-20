# Create dataframe with spearman correlations and p-values


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
betas = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas.csv'))

# add column with phase
betas["phase"] = np.where(betas["task"] == betas["feedback_condition"], "feedback", "transfer")

correlations = pd.DataFrame(columns = ["fsm_type", "feedback_condition", "trial_type", "correlation_type", "label", "corr", "p", "n"])

for fsm in betas["fsm_type"].unique():
    for feedback_condition in betas["feedback_condition"].unique():
        for trial_type in betas['trial_type'].unique():
            
            # feedback_condition-specific subset
            subset = betas[(betas["feedback_condition"] == feedback_condition) & 
                        (betas["trial_type"] == trial_type) &
                        (betas["fsm_type"] == fsm)]
            
            # phase-specific subsets
            subset_feedback = subset[subset["phase"] == "feedback"] 
            subset_transfer = subset[subset["phase"] == "transfer"]

            transfer_tasks = subset_transfer["task"].unique()

            results = {} # "learned-transfer" or "transfer-transfer" ("correlation_type")

            for transfer_task in transfer_tasks:
                subset_transfer_task = subset_transfer[subset_transfer["task"] == transfer_task]
                            
                # assert that data comes from the same participant ids in the same order
                assert np.array_equal(subset_feedback["participant_id"], subset_transfer_task["participant_id"])
                
                # correlation between subset_feedback and subset_transfer_task
                corr = stats.spearmanr(subset_feedback["max_beta"], subset_transfer_task["max_beta"], nan_policy = 'omit')[0]
                corr = round(corr, 2)
                
                p = stats.spearmanr(subset_feedback["max_beta"], subset_transfer_task["max_beta"], nan_policy = 'omit')[1]
                p = round(p, 3)

                label = feedback_condition + "-" + transfer_task
                results[label] = {"correlation_type": "learned-transfer", "corr": corr, "p": p}

            # correlation between two transfer tasks
            corr = stats.spearmanr(subset_transfer[subset_transfer["task"] == transfer_tasks[0]]["max_beta"], 
                                subset_transfer[subset_transfer["task"] == transfer_tasks[1]]["max_beta"], 
                                nan_policy = 'omit')[0]
            corr = round(corr, 2)

            p = stats.spearmanr(subset_transfer[subset_transfer["task"] == transfer_tasks[0]]["max_beta"], 
                                subset_transfer[subset_transfer["task"] == transfer_tasks[1]]["max_beta"], 
                                nan_policy = 'omit')[1]
            p = round(p, 3)

            label = transfer_tasks[0] + "-" + transfer_tasks[1]
            results[label] = {"correlation_type": "transfer-transfer", "corr": corr, "p": p}

            # create dataframe
            for label in results.keys():
                new_row = {"fsm_type": fsm, 
                    "feedback_condition": feedback_condition, 
                        "trial_type": trial_type, "label": label, 
                            "correlation_type": results[label]["correlation_type"],
                            "corr": results[label]["corr"],
                            "p": results[label]["p"], 
                            "n": len(subset_feedback)}
                correlations = pd.concat([correlations, pd.DataFrame([new_row])])
        
