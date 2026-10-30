# Beta Bank Customer Churn Prediction

## Problem
Beta Bank is losing customers a little at a time, and keeping a customer is cheaper than finding a new one. The goal is to predict whether a customer will leave the bank soon, with an F1 score of at least 0.59 on the test set, and to report AUC-ROC alongside it.

## Data
`data/Churn.csv`: 10,000 customers with credit score, country, gender, age, tenure, balance, number of products, credit card and activity flags, and estimated salary. The target is `Exited` (1 = left the bank).

- Only 20.4% of customers left, so the classes are imbalanced (about 4 to 1).
- `Tenure` had 909 missing values, filled with the median.

The dataset was provided by the TripleTen data science program and is included here so the notebook runs as is.

## Approach
1. **Preprocessing:** filled missing `Tenure`, one-hot encoded `Geography` and `Gender` (`drop_first=True`), and dropped ID columns (`RowNumber`, `CustomerId`, `Surname`).
2. **Split:** 60/20/20 train/validation/test, stratified on the target.
3. **Scaling:** `StandardScaler` on `Balance`, `EstimatedSalary` and `CreditScore`, fitted on the training set only.
4. **Models and imbalance handling,** compared on the validation set:
   - logistic regression: baseline, then scaled, then `class_weight='balanced'`, then upsampling the minority class 3x
   - decision tree: with and without `class_weight='balanced'`
   - random forest: `class_weight='balanced'`, a grid search over `max_depth` (1 to 10) and `n_estimators` (10 to 100) on the original data, and the same grid on upsampled data
5. **Final model:** the random forest trained on upsampled data with `max_depth=5` and `n_estimators=40`, evaluated once on the test set.

## Results
Validation F1 scores:

| Model | F1 |
|---|---|
| Logistic regression, baseline | 0.090 |
| Logistic regression, scaled | 0.321 (AUC-ROC 0.788) |
| Logistic regression, balanced class weights | 0.511 |
| Logistic regression, upsampled | 0.535 |
| Decision tree | 0.498 |
| Decision tree, balanced class weights | 0.496 |
| Random forest, balanced class weights (default settings) | 0.609 |
| Random forest, tuned on original data (depth 10, 100 trees) | 0.571 |
| **Random forest, tuned on upsampled data (depth 5, 40 trees)** | **0.611** (AUC-ROC 0.804) |

Final model on the held-out test set:

| Metric | Value |
|---|---|
| **F1** | **0.621** |
| **AUC-ROC** | **0.844** |

The final model beats the 0.59 F1 target on unseen data. An AUC-ROC of 0.844 means it ranks customers who leave above those who stay most of the time.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook beta-bank-churn-prediction.ipynb
```
Run all cells from the top. The notebook reads `data/Churn.csv` relative to this folder.

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter.
