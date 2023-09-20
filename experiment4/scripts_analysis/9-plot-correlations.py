# Scatterplots with correlations between the different test conditions

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
betas_long = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas.csv'))

# remove trial_type == 'hidden'
betas_long = betas_long[betas_long['trial_type'] != 'hidden']

# Styling
sns.set_style("white")
sns.set_context("paper", font_scale=1.5)

# Initialize Figure
fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)

pec = ['control', 'prediction', 'explanation']

# pallettte green and pink colorblind safe 
palette = ['#009E73', '#D41159']

for i, feedback_condition in enumerate(pec):
    # remove tasks that do not have the current feedback condition
    two_other_tasks = [task for task in pec if task != feedback_condition]
    tasks = [(two_other_tasks[0], feedback_condition), (two_other_tasks[1], feedback_condition)]

    for j, (task_y, task_x) in enumerate(tasks):
        ax = axes[i, j]
        
        # Filter data for plotting
        data_y = betas_long[(betas_long['task'] == task_y) & (betas_long['feedback_condition'] == feedback_condition)]
        data_x = betas_long[(betas_long['task'] == task_x) & (betas_long['feedback_condition'] == feedback_condition)]
        
        merged_data = pd.merge(data_x, data_y, on=['participant_id', 'trial_type'], suffixes=('_x', '_y'))
        
        y_position = 0.85  # Initialize the y-position for the text

        for idx, trial_type in enumerate(merged_data['trial_type'].unique()):
            subset = merged_data[merged_data['trial_type'] == trial_type]
            sns.scatterplot(data=subset, x='max_beta_x', y='max_beta_y', ax=ax, color=palette[idx], label=trial_type, s=15, alpha=0.6)
            sns.regplot(data=subset, x='max_beta_x', y='max_beta_y', ax=ax, color=palette[idx], scatter=False)
            
            # Calculate Pearson correlation coefficient
            r, p = pearsonr(subset['max_beta_x'], subset['max_beta_y'])

            # Format r and p based on their values
            r_text = f"r = {r:.3f}"
            p_text = f"p = {p:.3f}" if p > 0.001 else "p < 0.001"

            # Add text to the plot
            ax.text(0.95, y_position, f'{r_text}, {p_text}', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12, color=palette[idx], fontweight='bold')

            y_position -= 0.05  # Adjust the y-position for the next text

        # Set labels and title
        ax.set_xlabel(f"{task_x.capitalize()}")
        ax.set_ylabel(f"{task_y.capitalize()}")
        #ax.axhline(y=0, color='gray', linestyle='--')
        #ax.axvline(x=0, color='gray', linestyle='--')
        #ax.set_ylim(-0.01, 1.02)
        #ax.set_xlim(-0.01, 1.02)

        # Set tick labels
        #ax.set_yticks([-1,  -0.5, 0, 0.5, 1])
        #ax.set_xticks([0, 0.5, 1])

        ax.get_legend().remove()

# Add row titles "Trained on control/prediction/explanation"
fig.text(-0.01, 0.82, 'Trained on control', ha='center', va='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(-0.01, 0.50, 'Trained on prediction', ha='center', va='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(-0.01, 0.18, 'Trained on explanation', ha='center', va='center', rotation='vertical', fontsize=18, fontweight='bold')
        
# Add a custom legend at the top
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Visible form', markerfacecolor=palette[0], markersize=10),
                     plt.Line2D([0], [0], marker='o', color='w', label='Hidden (AN) form ', markerfacecolor=palette[1], markersize=10)]
leg = fig.legend(handles=legend_elements, loc='upper center', ncol=2, title="", frameon=False)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.94)
plt.show()

# Save figure
fig.savefig(os.path.join(current_dir, '../outputs/fig-exp4-correlations.png'), dpi=300, bbox_inches='tight')