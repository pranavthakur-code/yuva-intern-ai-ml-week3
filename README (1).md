# Data Science Tool Mastery Project

A portfolio notebook demonstrating hands-on proficiency with the core AI/ML toolchain, each tool applied to a concrete problem on real public data.

| Tool | Exercise |
|---|---|
| NumPy | Vectorization benchmark (~70x faster than a Python loop), broadcasting, least squares checked against scikit-learn |
| Pandas | Method chaining, `transform`, validated merges, pivot/melt, memory tuning (86% smaller), time-series resampling |
| SQL (SQLite) | GROUP BY, CTEs, window functions, JOIN, index; every result cross-checked against Pandas |
| Matplotlib / Seaborn | Multi-panel analytical figures with annotations |
| Scikit-learn | Leak-free pipelines, 5-fold CV, hyperparameter tuning, metrics, threshold selection, permutation importance, model saving |
| Git | Branch and merge workflow executed from the notebook |

## Datasets
- IBM **Telco Customer Churn** sample dataset (7,043 customers): https://github.com/IBM/telco-customer-churn-on-icp4d
- **Airline passengers** 1949-1960 (monthly, 144 rows): https://github.com/mwaskom/seaborn-data

## Headline results
- Churn model (tuned gradient boosting): test ROC AUC **0.847**, average precision 0.659 (random guessing: 0.265).
- Selected on cross-validation only; the test set was used once. Differences between models were within fold-to-fold noise.
- Contract type and tenure are the dominant drivers of predicted churn, consistent with the earlier descriptive analysis.

## Files
- `Tool_Mastery_Notebook.ipynb`: fully executed notebook
- `requirements.txt`: dependencies
- `Telco-Customer-Churn.csv`, `flights.csv`: input data
- `churn_model.joblib`: trained scikit-learn pipeline

## Run
```bash
pip install -r requirements.txt
jupyter notebook Tool_Mastery_Notebook.ipynb
```
Results come from one train/test split on one dataset; they illustrate technique, not production performance.
