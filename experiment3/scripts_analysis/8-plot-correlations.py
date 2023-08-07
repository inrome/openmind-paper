# Scatterplots with correlations between the different test conditions

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load data
current_dir = os.path.dirname(os.path.abspath(__file__))
betas_wide = pd.read_csv(os.path.join(current_dir, '../outputs/participants_max_betas_wide.csv'))

# scatterplots with correlations between control_hidden and explanation_hidden (max_beta)
sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
g = sns.lmplot(x="control_hidden", y="explanation_hidden", hue="fsm_type", data=betas_wide, height=4, aspect=1.2, scatter_kws={"s": 10})
g.set_axis_labels("Control", "Explanation")
g.set(ylim=(0, 1))
g.set(xlim=(0, 1))
g.set_titles("{col_name}")

sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
g = sns.lmplot(x="prediction_hidden", y="explanation_hidden", hue="fsm_type", data=betas_wide, height=4, aspect=1.2, scatter_kws={"s": 10})
g.set_axis_labels("Prediction", "Explanation")
g.set(ylim=(0, 1))
g.set(xlim=(0, 1))
g.set_titles("{col_name}")

sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
g = sns.lmplot(x="prediction_hidden", y="control_hidden", hue="fsm_type", data=betas_wide, height=4, aspect=1.2, scatter_kws={"s": 10})
g.set_axis_labels("Prediction", "Control")
g.set(ylim=(0, 1))
g.set(xlim=(0, 1))
g.set_titles("{col_name}")

# same for visible trials
sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
g = sns.lmplot(x="control_visible", y="explanation_visible", hue="fsm_type", data=betas_wide, height=4, aspect=1.2, scatter_kws={"s": 10})
g.set_axis_labels("Control", "Explanation")
g.set(ylim=(0, 0.9))
g.set(xlim=(0, 0.9))
g.set_titles("{col_name}")

sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
g = sns.lmplot(x="prediction_visible", y="explanation_visible", hue="fsm_type", data=betas_wide, height=4, aspect=1.2, scatter_kws={"s": 10})
g.set_axis_labels("Prediction", "Explanation")
g.set(ylim=(0, 0.9))
g.set(xlim=(0, 0.9))
g.set_titles("{col_name}")

sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
g = sns.lmplot(x="prediction_visible", y="control_visible", hue="fsm_type", data=betas_wide, height=4, aspect=1.2, scatter_kws={"s": 10})
g.set_axis_labels("Prediction", "Control")
g.set(ylim=(0, 0.9))
g.set(xlim=(0, 0.9))
g.set_titles("{col_name}")
