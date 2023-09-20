import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import math

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
betas_wide = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas_wide.csv'))

# correlations between all three tasks
correlations = pd.DataFrame(columns=["fsm_type", "trial_type", "label", "pearson", "p","n"])
for trial_type in ["visible", "hidden", "hidden_an"]:
    for pair in [("control", "explanation"), ("control", "prediction"), ("explanation", "prediction")]:
        varname_1 = pair[0] + "_" + trial_type
        varname_2 = pair[1] + "_" + trial_type

        for fsm_type in ["easy", "hard"]:
            # condition-specisic subset 
            subset = betas_wide[(betas_wide["fsm_type"] == fsm_type) & 
                                (~pd.isna(betas_wide[varname_1])) & 
                                (~pd.isna(betas_wide[varname_2]))]
            corr = stats.pearsonr(subset[varname_1], subset[varname_2])
            stat = round(corr[0],2)
            p = round(corr[1], 3)

            new_row = {"fsm_type": fsm_type, 
                        "trial_type": trial_type, "label": pair[0] + "~" + pair[1], 
                            "pearson": stat,
                            "p": p,
                            "n": len(subset)}
            
            correlations = pd.concat([correlations, pd.DataFrame([new_row])],ignore_index=True)
            
