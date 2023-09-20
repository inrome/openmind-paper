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
betas_long = betas_long.replace({'easy': 'Easy FSM', 'hard': 'Hard FSM'})

# filter out nan
betas_long = betas_long[~np.isnan(betas_long['max_beta'])]

# Styling
sns.set_style("white")
sns.set_context("paper", font_scale=1.8)

# Initialize Figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=False)

tasks = [("explanation", "prediction"), ("control", "prediction"), ("explanation", "control")]

# Loop over rows ('visible', 'hidden')
palette = sns.color_palette('colorblind')

for i, trial_type in enumerate(['visible', 'hidden_an']):
    for j, (task_y, task_x) in enumerate(tasks):
        ax = axes[i, j]
        
        # Filter data for plotting
        data_y = betas_long[(betas_long['task'] == task_y) & (betas_long['trial_type'] == trial_type)]
        data_x = betas_long[(betas_long['task'] == task_x) & (betas_long['trial_type'] == trial_type)]
        
        merged_data = pd.merge(data_x, data_y, on=['participant_id', 'fsm_type'], suffixes=('_x', '_y'))
        
        y_position = 0.90  # Initialize the y-position for the text

        for idx, fsm_type in enumerate(merged_data['fsm_type'].unique()):
            subset = merged_data[merged_data['fsm_type'] == fsm_type]
            sns.scatterplot(data=subset, x='max_beta_x', y='max_beta_y', ax=ax, color=palette[idx], label=fsm_type, s=20, alpha=0.6)
            sns.regplot(data=subset, x='max_beta_x', y='max_beta_y', ax=ax, color=palette[idx], scatter=False)
            
            # Calculate Pearson correlation coefficient
            r, p = pearsonr(subset['max_beta_x'], subset['max_beta_y'])
            
            # Format r and p based on their values
            r_text = f"r = {r:.3f}"
            p_text = f"p = {p:.3f}" if p > 0.001 else "p < 0.001"
            
            # Add text to the plot
            ax.text(0.05, y_position, f'{r_text}, {p_text}', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=12, color=palette[idx], fontweight='bold')

            y_position -= 0.05  # Adjust the y-position for the next text

        # Set labels and title
        ax.set_xlabel(f"{task_x.capitalize()}")
        ax.set_ylabel(f"{task_y.capitalize()}")
        ax.axhline(y=0, color='gray', linestyle='--')
        #ax.set_ylim(0, 1.02)
        #ax.set_xlim(0, 1.02)

        # Set tick labels
        #ax.set_yticks([-1,  -0.5, 0, 0.5, 1])
        #ax.set_xticks([-1,  -0.5, 0, 0.5, 1])

        ax.get_legend().remove()
        
# Add row titles
fig.text(-0.01, 0.75, 'Visible', ha='center', va='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(-0.01, 0.28, 'Hidden AN', ha='center', va='center', rotation='vertical', fontsize=18, fontweight='bold')

# Add a custom legend at the top
labels = merged_data['fsm_type'].unique()
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in range(len(labels))]
leg = fig.legend(legend_labels, labels, loc='upper center', ncol=3, title="", frameon=False) 

# Set legend text color
for i, text in enumerate(leg.get_texts()):
    text.set_color(palette[i])

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.94)
plt.show()

# Save figure
fig.savefig(os.path.join(current_dir, '../outputs/figure_correlations.png'), dpi=300, bbox_inches='tight')