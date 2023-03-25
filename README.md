# Data Analysis for Prediction, Explanation, and Control

## Prerequisites

* Python 3.7

## Components

* `exp-1` — a git submodule with data and scripts for [Experiment 1](https://github.com/inrome/cogsci-2023)

* `exp-2` — a git submodule with data and scripts for [Experiment 2](https://github.com/inrome/pec-preview)
* `data_merged` — contains merged data from Experiment 1 and Experiment 2 (see `scripts/0-import-data.py`)
* `scripts` — contains scripts for data analysis
* `outputs` — data analysis outputs (csv and pickle files + figures)

## Data Analysis

* `scripts/0-import-data.py` — merges data from Experiment 1 and Experiment 2, adds new participant IDs, and saves
the merged data to `data_merged/imported_clean_data.pickle`
* All other scripts — see [README.md](https://github.com/inrome/cogsci-2023/blob/7ffe6d109eb12fa75ac6859269258421612119e2/README.md) for Experiment 1

## Related Projects

* [Prediction, Explanation, and Control Under Free Exploration Learning: Experiment 1](https://github.com/inrome/cogsci-2023)
* [Prediction, Explanation, and Control Under Free Exploration Learning: Experiment 2](https://github.com/inrome/pec-preview)

## Authors

* Roman Tikhonov, [Google Scholar](https://scholar.google.ru/citations?user=4ag4R48AAAAJ&hl=ru)

* Simon DeDeo, [Google Scholar](https://scholar.google.com/citations?user=UW3tRn8AAAAJ&hl=en)
