import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
glms_df_long = pd.read_csv('../outputs/glms_Ri_accuracy_long.csv')
glms_df_long = glms_df_long[glms_df_long['feedback_condition'] == 'explanation']
# plot coef and 95% CI for each FSM type, test condition and trial type
glms_df_long = glms_df_long.rename(columns={'0.025': 'lower', 
                                            '0.975': 'upper'})

df_p_easy = glms_df_long.loc[(glms_df_long['fsm_type'] == 'easy') & (glms_df_long['test_condition'] == 'prediction')]
df_c_easy = glms_df_long.loc[(glms_df_long['fsm_type'] == 'easy') & (glms_df_long['test_condition'] == 'control')]
df_e_easy = glms_df_long.loc[(glms_df_long['fsm_type'] == 'easy') & (glms_df_long['test_condition'] == 'explanation')]

df_p_hard = glms_df_long.loc[(glms_df_long['fsm_type'] == 'hard') & (glms_df_long['test_condition'] == 'prediction')]
df_c_hard = glms_df_long.loc[(glms_df_long['fsm_type'] == 'hard') & (glms_df_long['test_condition'] == 'control')]
df_e_hard = glms_df_long.loc[(glms_df_long['fsm_type'] == 'hard') & (glms_df_long['test_condition'] == 'explanation')]


trial_type_order = ['Visible', 'Hidden\nNormative', 'Hidden\nAN']

# Helper function to plot scatter and error bars
def plot_scatter_and_error(df, color, axis, shift, marker):
    # Map trial types to their positions in trial_type_order
    x_values = [trial_type_order.index(ttype) + shift for ttype in df['trial_type']]

    # Make a scatter plot with mapped x-values and markers
    axis.scatter(x_values, df['coef'], marker=marker, color=color)

    # Draw error bars with mapped x-values
    axis.errorbar(x_values, df['coef'], yerr=[df['coef'] - df['lower'], df['upper'] - df['coef']],
                  fmt='None', color=color)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

conditions = [("Easy", ax1, df_p_easy, df_c_easy, df_e_easy),
              ("Hard", ax2, df_p_hard, df_c_hard, df_e_hard)]

colors = ['indigo', 'darkorange', 'steelblue']

for condition_name, axis, df_p, df_c, df_e in conditions:
    plot_scatter_and_error(df_p, colors[0], axis, 0.1, "s") # prediction
    plot_scatter_and_error(df_c, colors[1], axis, 0, "o") # control
    plot_scatter_and_error(df_e, colors[2], axis, 0.2, "^") # explanation

    axis.set_title(f'{condition_name} FSM')  # Set the title for subplot


    # Set the x-axis tick labels
    axis.set_xticks(range(len(trial_type_order)))
    axis.set_xticklabels(trial_type_order)

    axis.set_ylim(-0.15, 0.8)   # Set the y-axis limits

    axis.axhline(y=0, color='gray', linestyle='--', linewidth=2) # Add dashed line at y=0

fig.tight_layout() # Adjust the spacing between subplots

fig.text(0, 0.5, 'Beta', va='center', rotation='vertical') # Add y axis label

plt.show()

