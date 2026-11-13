# Methodology and corrections / روش‌شناسی و اصلاحات

## What came from the supplied archives?

- `Dataset.zip`: original EDA script, original CSV, EDA coursework PDF.
- `Classification.zip`: original KNN/Decision Tree script, original CSV, classification coursework PDF.
- `Clustering.zip`: original K-Means script, original CSV, clustering coursework PDF.
- The included CSV file is byte-identical in all three original archives.
- Original scripts are preserved unedited under `original_code/`.
- Reports contain course exercises as well as project discussion. Only the cover-page identifying text was redacted from the public copies; they are not rewritten to match the upgraded code.

## Corrected implementation (not a recreation of the original report numbers)

1. **Offline dataset loading:** the original scripts fetch UCI via the network despite also reading CSV; the upgraded scripts read only the included CSV.
2. **Missing value handling:** no missing values were found. Scikit-learn imputation still lives *inside* the classification pipeline in case future versions contain missing numeric cells.
3. **Potentially invalid zeros:** the old code tries to change zero to missing in `Age Group`, `Tariff Plan`, `Status`, `Age`, although the current CSV has zero occurrences in those columns. New scripts leave valid data unchanged.
4. **Exact duplicates:** 300 of 3,150 records are duplicates under all 14 columns. These are removed before the classification holdout split (2,850 distinct rows). No unique customer ID is available, so this choice is explicitly disclosed.
5. **Scaling leakage:** the original code fit a different `StandardScaler` to each of the train and test sets. This is fixed using a training-only Pipeline; CV folds also have their own training-only fitted preprocessing.
6. **Class imbalance:** `Churn` is the positive class and about 15.7% of the raw dataset. Candidates are selected by churn-class F1 and the holdout receives a precision/recall and PR/ROC report, not accuracy alone.
7. **Validation:** 10-fold `StratifiedKFold` on training data, selection by mean F1, then a single test-set evaluation. The original script evaluated multiple candidates on test data; the upgraded script avoids choosing a model based on test scores.
8. **Clustering target usage:** `Churn` is excluded from scaling and fitting. The original confusion matrix compared arbitrary K-Means cluster IDs to the churn target; this is not a classification metric. New clustering evaluation uses sampled Silhouette and full Davies–Bouldin; churn is used only for post-hoc cluster profiles.
9. **Reproducibility:** random seeds are fixed, output tables and charts are saved, and the best supervised classifier is serialized.

## Interpretation caveats

- K-Means cluster IDs are arbitrary labels; a given cluster is not inherently a churn/no-churn prediction.
- Silhouette is calculated on a stable random sample of 1,000 rows for speed, so results can vary slightly with library versions and numerical settings.
- K-Means has multiple alternative solutions; the chosen number is based on maximum *sampled* silhouette, not necessarily a business-optimal segmentation.
- The extracted PDF reports are educational assignments and should not be used as formal evidence for new figures; refer to `results/` for actual computed results.
- Data deduplication removes exact clones but does not prove the absence of near duplicates, sampling bias, or target-correlated business leakage.

## Reproduce

```bash
python -m pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests -v
```

Outputs: EDA tables, classifier training CV scores, holdout metrics and predictions summary, clustering quality table, cluster-profile table, PNG figures, and a fitted classifier.
