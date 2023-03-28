import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os 

# Read data 
current_dir = os.path.dirname(os.path.abspath(__file__))
predicted_accuracy_path = os.path.join(current_dir, '../outputs/predicted_accuracy.csv')
ss_betas = pd.read_csv(predicted_accuracy_path)

# subset data
# ss_betas = ss_betas[ss_betas["learning_condition"] == "Experiment 1 (no preview)"]
# ss_betas = ss_betas[ss_betas["learning_condition"] == "Experiment 2 (test preview)"]


# Rename columns and labels
ss_betas = ss_betas.rename(columns={"fsm_type": "difficulty", "condition": "visibility"})
ss_betas["task"] = ss_betas["task"].replace({"prediction": "Prediction",
                                             "control": "Control",
                                             "explanation": "Explanation"})
ss_betas["difficulty"] = ss_betas["difficulty"].replace({"easy": "Easy", "hard": "Hard"})
ss_betas["difficulty"] = pd.Categorical(ss_betas["difficulty"], categories=["Easy", "Hard"], ordered=True)
ss_betas['task'] = pd.Categorical(ss_betas['task'], categories=['Control', 'Prediction', 'Explanation'], ordered=True)

# subset data for AN-sensitive only (visibility in ['hidden_normative_subset', 'hidden_an_subset'])
ss_betas_subset = ss_betas[ss_betas["visibility"].isin(['hidden_normative_subset', 'hidden_an_subset'])]
ss_betas = ss_betas[~ss_betas["visibility"].isin(['hidden_normative_subset', 'hidden_an_subset'])] # remove AN-sensitive only

ss_betas["visibility"] = ss_betas["visibility"].replace({"visible": "Visible",
                                                         "hidden": "Hidden",
                                                         "hidden_an": "Hidden AN"})

# Create the violinplot
sns.set(style="whitegrid", font_scale=1.2)
g = sns.catplot(data=ss_betas, x="task", y="predicted_accuracy", hue="difficulty", col="visibility",
                kind="violin", dodge=True, width=0.7, cut=0, saturation=0.75, split=True, inner='stick',
                scale="count",
                scale_hue=True,
                palette={"Easy": "#1b9e77", "Hard": "#d95f02"},
                legend=False)
    
# Customize plot
g.set_axis_labels("", "Predicted Accuracy")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
# make text of row titles bold
for ax, title in zip(g.axes.flat, ["Visible", "Hidden (Normative)", "Hidden (Alternative Neglect)"]):
    ax.set_title(title, fontweight="bold")

g.despine(left=True)

plt.legend(loc='upper left', bbox_to_anchor=(-2.1, 1.3), ncol=2, title="Finite-State Machine:")
save_path = os.path.join(current_dir, '../outputs/predicted_accuracy_alpha0_violin.png')
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# same plot but with boxplots #### 

# Create the boxplot
sns.set(style="whitegrid", font_scale=1.2)
g = sns.catplot(data=ss_betas, x="task", y="predicted_accuracy", hue="difficulty", col="visibility",
                kind="box", dodge=4, width=0.5, saturation=0.75, 
                palette={"Easy": "#1b9e77", "Hard": "#d95f02"},
                legend=False)

# # Add jittered individual data points
# for i, visibility in enumerate(["Visible", "Hidden", "Hidden AN"]):
#     sns.stripplot(data=ss_betas[ss_betas["visibility"] == visibility],
#                   x="task", y="predicted_accuracy", hue="difficulty",
#                   dodge=True, edgecolor="black", alpha=0.4, jitter=0.2, ax=g.axes[0][i], size=3, linewidth=0.5,
#                   palette={"Easy": "#1b9e77", "Hard": "#d95f02"}, legend=False)

# Customize plot
g.set_axis_labels("", "Predicted Accuracy")
g.set_titles(col_template="{col_name}", row_template="{row_name}")

for ax, title in zip(g.axes.flat, ["Visible", "Hidden (Normative)", "Hidden (Alternative Neglect)"]):
    ax.set_title(title, fontweight="bold")

g.despine(left=True)

plt.legend(loc='upper left', bbox_to_anchor=(-2.1, 1.3), ncol=2, title="Finite-State Machine:")
save_path = os.path.join(current_dir, '../outputs/predicted_accuracy_alpha0_boxplot.png')
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# violinplot for AN-sensitive only
ss_betas_subset['visibility'] = ss_betas_subset['visibility'].replace({'hidden_normative_subset': 'Normative',
                                                                        'hidden_an_subset': 'Alternative Neglect'})
sns.set(style="whitegrid", font_scale=1.2)
g = sns.catplot(data=ss_betas_subset, x="task", y="predicted_accuracy", hue="visibility", col="difficulty",
                kind="violin", dodge=True, width=0.7, cut=0, saturation=0.75, split=True, inner='stick',
                scale="count",
                scale_hue=True,
                palette={"Normative": "white", "Alternative Neglect": "gray"},
                legend=False)
    
g.set_axis_labels("", "Predicted Accuracy")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
# make text of row titles bold
for ax, title in zip(g.axes.flat, ["Easy", "Hard"]):
    ax.set_title(title, fontweight="bold")

g.despine(left=True)

plt.legend(loc='upper left', bbox_to_anchor=(-1.1, 1.3), ncol=2, title="Finite-State Machine:")
save_path = os.path.join(current_dir, '../outputs/predicted_accuracy_alpha0_violin_an-sensitive-only.png')
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# boxplot for AN-sensitive only
sns.set(style="whitegrid", font_scale=1.2)
g = sns.catplot(data=ss_betas_subset, x="task", y="predicted_accuracy", hue="visibility", col="difficulty",
                kind="box", dodge=4, width=0.5, saturation=0.75, 
                palette={"Normative": "white", "Alternative Neglect": "gray"},
                legend=False)

# Customize plot
g.set_axis_labels("", "Predicted Accuracy")
g.set_titles(col_template="{col_name}", row_template="{row_name}")

for ax, title in zip(g.axes.flat, ["Easy", "Hard"]):
    ax.set_title(title, fontweight="bold")

g.despine(left=True)

plt.legend(loc='upper left', bbox_to_anchor=(-1.1, 1.3), ncol=2, title="Model for hidden tasks:")
save_path = os.path.join(current_dir, '../outputs/predicted_accuracy_alpha0_boxplot-ansensitive-subset.png')
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
# calculate number of observations per group (task, difficulty, visibility)
ss_betas.groupby(["visibility", "task", "difficulty"]).size()

##
# %%

# Means for subset of AN-sensitive tasks
sns.set(style="whitegrid", font_scale=1.2)

g = sns.catplot(data=ss_betas_subset, x="task", y="predicted_accuracy", hue="visibility", col="difficulty",
                kind="point", dodge=0.2, estimator='mean', errorbar=('ci', 95), capsize=0.05, join=False, 
                palette={"Normative": "gray", "Alternative Neglect": "black"},
                legend=False)

# Customize plot
g.set_axis_labels("", "Predicted Accuracy")
g.set_titles(col_template="{col_name}", row_template="{row_name}")

for ax, title in zip(g.axes.flat, ["Easy", "Hard"]):
    ax.set_title(title, fontweight="bold")

g.despine(left=True)

plt.legend(loc='upper left', bbox_to_anchor=(-1.1, 1.3), ncol=2, title="Model for hidden tasks:")
save_path = os.path.join(current_dir, '../outputs/predicted_accuracy_alpha0_pointplot-ansensitive-subset.png')
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
 