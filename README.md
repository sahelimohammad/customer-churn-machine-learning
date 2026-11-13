# Iranian Customer Churn Analysis

**Exploratory Data Analysis · Classification · Clustering**

[نسخه فارسی](README.fa.md)

Customer churn is an important problem for subscription-based services: understanding which customers leave and what patterns are associated with churn can support retention decisions. This project explores a telecom customer dataset and applies classification and clustering methods to it.

## Dataset

The [Iranian Churn dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) contains 3,150 records, 13 input features and a binary target (`Churn`). A value of `1` indicates a customer who churned.

| Item | Value |
| --- | ---: |
| Records | 3,150 |
| Input features | 13 |
| Churn rate | 15.7% |
| Missing values | 0 |
| Exact duplicate rows | 300 |

The CSV file is available in [`data/`](data/). For model evaluation, duplicate rows are removed before the train/test split, leaving 2,850 observations.

## Methods

### 1. Exploratory data analysis

The EDA section covers descriptive statistics, class distribution, feature distributions and correlations. Plots and summary tables are saved in [`figures/eda/`](figures/eda/) and [`results/`](results/).

![Churn distribution](figures/eda/churn_distribution.png)

### 2. Classification

Two algorithms are compared:

- **K-Nearest Neighbors:** `k = 5, 7, 13`
- **Decision Tree:** `max_depth = 3, 5, 8`

The data is split into 70% training and 30% testing with class stratification. Candidate models are compared using 10-fold stratified cross-validation on the training set. The final model is selected by F1-score for the churn class and evaluated on the test set. Preprocessing is fitted on training data only.

| Test-set metric | Result |
| --- | ---: |
| Selected model | KNN (`k=5`) |
| Accuracy | 93.68% |
| Precision (churn) | 78.99% |
| Recall (churn) | 81.34% |
| F1-score (churn) | 80.15% |
| ROC AUC | 0.955 |

![Classification confusion matrix](figures/classification/confusion_matrix.png)

The detailed results are in [`results/classification_test_metrics.json`](results/classification_test_metrics.json).

### 3. Clustering

K-Means is applied with cluster counts from 2 to 12. Silhouette Score and Davies–Bouldin Index are used to compare the alternatives. The largest sampled Silhouette Score occurs at **10 clusters** (`0.306`, calculated on a reproducible sample of 1,000 observations).

`Churn` is excluded from clustering features. Churn rates are examined afterward to describe the resulting groups; cluster IDs are not class predictions.

![Cluster selection metrics](figures/clustering/k_selection_metrics.png)

## Project structure

```text
customer-churn-machine-learning/
├── data/                   # Dataset and source information
├── eda/                    # Exploratory analysis
├── classification/         # KNN and Decision Tree
├── clustering/             # K-Means
├── figures/                # Saved charts
├── results/                # Evaluation metrics and tables
├── models/                 # Trained classifier
├── reports/                # Course reports (PDF)
├── original_code/          # Earlier course scripts
├── docs/                   # Methodology
├── tests/                  # Data integrity checks
├── run_all.py
├── requirements.txt
├── README.md
└── README.fa.md
```

## Installation and usage

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

To run an individual stage:

```bash
python eda/eda.py
python classification/classification.py
python clustering/clustering.py
```

To run the data integrity tests:

```bash
python -m unittest discover -s tests -v
```

The scripts read the CSV included in this repository. On completion, charts and result files are saved under `figures/` and `results/`.

## Notes

- Classification metrics refer to the implementation in `classification/` and its deduplicated train/test split. The PDF reports document the course exercises and may use different calculations.
- Exact duplicate removal is a preprocessing decision; the dataset does not provide a unique customer identifier.
- Clustering metrics describe separation in feature space, not predictive accuracy.
- This is an academic dataset and does not establish real-world deployment performance.

**Dataset citation:** Iranian Churn. (2020). UCI Machine Learning Repository. [DOI: 10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z). Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
