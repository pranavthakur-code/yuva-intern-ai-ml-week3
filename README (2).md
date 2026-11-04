# Retail Profitability Review: Analytics Report & Insights

A business analytics project: where does a retailer make and lose money, why, and what should it do first?

**Data:** the *Superstore* sample dataset published with Tableau (a fictional US retailer, about 10,000 order lines, 2015-2018): https://github.com/leonism/sample-superstore

## Headline findings
- Sales grew 20.4% in 2018 but profit only 14.2%; margin fell from 13.4% to 12.7%.
- Order lines discounted more than 20% are 13.9% of lines but lost $135k, equal to 47% of total profit.
- Binders, Tables and Machines are profitable at normal discounts and lose heavily above 20% off: it is a discount problem, not a product problem.
- 10 states, all averaging more than 25% discount, lose $98k. Central region has the lowest margin (7.9%).
- 155 customers (20%) have negative lifetime profit; West returns 11.7% of orders versus about 3% elsewhere.

## Files
| File | Purpose |
|---|---|
| `Analytics_Insights_Notebook.ipynb` | Executed analysis: business questions, cleaning, findings, scenario, recommendations |
| `superstore_lib.py` | All calculations and charts (the notebook and the report use the same code) |
| `superstore.csv` | Raw data (contains order lines plus appended returns and managers tables) |
| `superstore_clean.csv` | Cleaned order lines with derived fields |
| `performance_dashboard.png` | One-page KPI dashboard |
| `requirements.txt` | Dependencies |

## Run
```bash
pip install -r requirements.txt
jupyter notebook Analytics_Insights_Notebook.ipynb
```

## Limitations
Sample data from a fictional company; no cost data; findings are associations that a controlled pilot should confirm.
