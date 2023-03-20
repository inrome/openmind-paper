import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read and preprocess the data
ss_betas = pd.read_csv("../outputs/predicted_accuracy.csv")
ss_betas = ss_betas.rename(columns={"fsm": "difficulty", "condition": "visibility"})
ss_betas["task"] = ss_betas["task"].replace({"prediction": "Prediction",
                                             "control": "Control",
                                             "explanation": "Explanation"})
ss_betas["difficulty"] = ss_betas["difficulty"].replace({"easy": "Easy", "hard": "Hard"})
ss_betas["visibility"] = ss_betas["visibility"].replace({"visible": "Visible",
                                                         "hidden": "Hidden",
                                                         "hidden_an": "Hidden AN"})
ss_betas["difficulty"] = pd.Categorical(ss_betas["difficulty"], categories=["Easy", "Hard"], ordered=True)

# Create the boxplot
sns.set(style="whitegrid", font_scale=1.2)
g = sns.catplot(data=ss_betas, x="task", y="predicted_accuracy", hue="difficulty", col="visibility",
                kind="violin", dodge=True, width=0.6, cut=0, saturation=0.75, split=True, inner="stick", scale="width",
                scale_hue=True,
                palette={"Easy": "#1b9e77", "Hard": "#d95f02"},
                legend=False)

# Add jittered individual data points
for i, visibility in enumerate(["Visible", "Hidden", "Hidden AN"]):
    sns.stripplot(data=ss_betas[ss_betas["visibility"] == visibility],
                  x="task", y="predicted_accuracy", hue="difficulty",
                  dodge=True, edgecolor="black", alpha=0.8, jitter=0.2, ax=g.axes[0][i], size=3, linewidth=0.5,
                  palette={"Easy": "#1b9e77", "Hard": "#d95f02"}, legend=False)

# Customize plot
g.set_axis_labels("", "Predicted Accuracy")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
# make text of row titles bold
for ax, title in zip(g.axes.flat, ["Visible", "Hidden (Normative)", "Hidden (Alternative Neglect)"]):
    ax.set_title(title, fontweight="bold")

g.despine(left=True)

plt.legend(loc='upper left', bbox_to_anchor=(-2.1, 1.3), ncol=2, title="Finite-State Machine:")
plt.savefig("../outputs/predicted_accuracy_alpha0.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
# calculate number of observations per group (task, difficulty, visibility)
ss_betas.groupby(["task", "difficulty", "visibility"]).size()