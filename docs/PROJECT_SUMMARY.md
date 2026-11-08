# StockMind: Project Summary

StockMind forecasts weekly demand for each product, with a prediction interval, and uses the forecast to decide how much to order. It is tested on about one million real retail transactions with a strict time-based backtest: models learn from the past only and are scored on the last 26 weeks.

## Problem

A retailer must decide how much of each product to order. Ordering too little loses sales. Ordering too much ties up capital. This project measures how forecast quality turns into ordering decisions.

## Data

Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Check the license terms on the dataset page before reuse. Raw and generated data are not committed to this repository.

- Products in the dataset: 4,838
- Products modelled: 2,042 (81% of all units)

## Key results (last 26 weeks, WAPE, lower is better)

Every method is scored on exactly the same product-weeks.

| Group | Model (GBM, p50) | Best baseline |
|---|---|---|
| All products | 0.631 | 0.650 (13-week moving median) |
| ABC class A | 0.583 | 0.602 |
| ABC class B | 0.759 | 0.781 |
| ABC class C | 0.911 | 0.912 |
| Demand: smooth | 0.434 | 0.433 |
| Demand: erratic | 0.640 | 0.651 |
| Demand: intermittent | 0.526 | 0.521 |
| Demand: lumpy | 0.794 | 0.850 |

Other baselines on all products: last week 0.814, 13-week moving average 0.680, 4-week moving average 0.696, same week last year 1.099.

Prediction interval (p10 to p90) coverage: 89.7% against an 80% nominal target. The intervals are somewhat conservative.

## What the results say

- The model beats the best simple baseline by 2.9% overall. The gain is modest, and it is clearest for high-volume (class A) and lumpy products.
- For smooth and intermittent products the model is about equal to a simple moving average, so a simple method is a fair choice there.
- The prediction intervals cover more than the nominal 80%. This is safe for ordering but means stock buffers may be slightly larger than needed.

## Cost assumptions

Cost results use assumptions that are not in the data: unit cost is 60% of the selling price and holding stock costs 20% of its value per year. Money figures illustrate the method. They are not a forecast of savings.

## Pipeline

| Stage | Script |
|---|---|
| Load and check | src/load_data.py, src/check_data.py |
| Cleaning | src/clean_data.py, src/inspect_outliers.py |
| Weekly table | src/build_weekly.py |
| Exploration | src/eda.py |
| Segmentation (ABC and demand pattern) | src/segment_products.py |
| Baselines | src/baselines.py |
| Forecast model | src/forecast_model.py |
| Decision backtest | src/decision_backtest.py |
| Out-of-sample check | src/out_of_sample_check.py |
| Dashboard | app/dashboard.py |

Result tables are in reports/ and figures are in reports/figures/. The dashboard has six tabs: Overview, Data, Forecast accuracy, Product explorer, Ordering simulation, Method and limits.

## Run

    source .venv/bin/activate
    streamlit run app/dashboard.py

The full list of pipeline commands is in README.md.
