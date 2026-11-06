# StockMind: Project Summary

StockMind is an end-to-end demand forecasting and ordering-decision project built on a real retail transaction dataset. It goes from raw transactions to forecasts, and from forecasts to inventory decisions that are tested with a backtest.

## Problem

A retailer must decide how much of each product to order. Ordering too little loses sales (service level drops). Ordering too much ties up capital. This project forecasts weekly demand per product and evaluates how forecast quality turns into ordering decisions.

## Data

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Check the license terms on the dataset page before reuse. Raw and generated data are not committed to this repository.

## Pipeline

| Stage | Script | Purpose |
|---|---|---|
| Load and check | `src/load_data.py`, `src/check_data.py` | Read the raw data and run sanity checks |
| Cleaning | `src/clean_data.py`, `src/inspect_outliers.py` | Clean transactions and inspect outliers (decisions logged in `reports/cleaning_log.csv`) |
| Weekly table | `src/build_weekly.py` | Aggregate transactions into weekly units per product |
| Exploration | `src/eda.py` | Exploratory figures (`reports/figures/01` to `04`) |
| Segmentation | `src/segment_products.py` | ABC (Pareto) classes and demand-pattern classes (`reports/figures/05`, `06`) |
| Baselines | `src/baselines.py` | Simple benchmark forecasts (`reports/baseline_results.csv`, figure `07`) |
| Model | `src/forecast_model.py` | Forecast model compared with baselines (`reports/model_results.csv`, figure `08`) |
| Decision backtest | `src/decision_backtest.py` | Turns forecasts into ordering decisions and measures service level vs. capital (`reports/decision_results.csv`, `reports/decision_curves.csv`, figure `09`) |
| Out-of-sample check | `src/out_of_sample_check.py` | Validation on held-out data (`reports/out_of_sample_results.csv`) |
| Dashboard | `app/dashboard.py` | Streamlit dashboard to explore results |

## Where to find the results

All result tables are in `reports/` and all figures are in `reports/figures/`. Every number quoted in the README or in an interview should be read directly from these files.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The full list of pipeline commands is in `README.md`.

## Design choices worth explaining

- **Baselines first.** The model is only meaningful relative to simple benchmarks.
- **Decisions, not only accuracy.** Forecast error is translated into service level and capital tied up in stock.
- **Out-of-sample validation.** Results are re-checked on data the model did not see.
- **Reproducibility.** Every step is a script, every cleaning decision is logged, and `PROJECT_LOG.md` records the reasoning step by step.
