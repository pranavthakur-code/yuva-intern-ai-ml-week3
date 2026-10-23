# Telecom Customer Churn: Real-World Dataset Analysis

Exploratory data analysis of the public IBM **Telco Customer Churn** sample dataset (7,043 customers, 21 columns).

**Question:** who churns, how much revenue does it represent, and what should the company do about it?

## Contents
- `Telco_Churn_Analysis.ipynb`: full notebook (loading, exploration, cleaning, EDA, visualizations, findings, recommendations)
- `Telco-Customer-Churn.csv`: raw data (source: https://github.com/IBM/telco-customer-churn-on-icp4d)
- `telco_churn_clean.csv`: cleaned data with engineered features

## Key findings
- 26.5% of customers churned, representing 30.5% of monthly recurring revenue.
- Contract type is the strongest factor: churn is 42.7% (month-to-month), 11.3% (one-year), 2.8% (two-year).
- Churn falls from 47.4% in the first year of tenure to 9.5% after four years.
- Fiber optic and electronic-check customers churn about twice as often as others.
- Month-to-month + fiber + electronic check customers are 18.6% of customers but 42.2% of churners.

## Run it
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
jupyter notebook Telco_Churn_Analysis.ipynb
```
Findings are associations from a single snapshot of a sample dataset, not proven causes.
