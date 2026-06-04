# Drug Shortage Early Warning System

Most people don't think about drug shortages until they're the patient 
who can't get their medication. I built this to make that problem visible.

## What this is

A full end-to-end data science project that pulls real FDA drug shortage 
data, scores each shortage by severity, and surfaces the results in an 
interactive dashboard that a hospital procurement team could actually use.

Not a tutorial. Not a Kaggle notebook. Real messy FDA data, real cleaning 
decisions, real insights.

## What I built

- **ETL pipeline** — raw FDA CSV → cleaned pandas dataframe → SQLite database
- **Severity scoring model** — weighted feature engineering across duration, 
  category criticality, availability status, and manufacturer data
- **Interactive dashboard** — built in Streamlit with Plotly charts, 
  filters, and a per-drug deep dive with a severity gauge
- **Deployed live** — not just a notebook, an actual running application

## The interesting finding

Anesthesia/Pediatric drugs have been in shortage for an average of 
**13 years**. Oncology and Cardiovascular categories are not far behind. 
These aren't temporary supply hiccups — they are systemic failures that 
have been quietly ongoing for over a decade.

## Tech stack

- Python (pandas, scikit-learn, XGBoost, SHAP)
- SQL (SQLite)
- Streamlit + Plotly
- FDA public dataset (1,673 shortage records)

## How to run it locally

```bash
git clone https://github.com/neehh001/drug_shortage_dashboard.git
cd drug_shortage_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

[https://drugshortagedashboard-neha.streamlit.app/](#) 

## What I learned

Writing SQL against messy real-world data is a completely different 
experience from Kaggle CSVs. Column names have spaces, dates are 
inconsistent, and half the fields you expect are missing. Building 
something that handles that gracefully without hardcoding fixes is 
the actual skill.

---
Built by Neha Khadeeja
