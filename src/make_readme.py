# FILE: src/make_readme.py
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_POLICIES = ["repeat last week x k", "moving average 13 x k", "moving average 13 + z * std"]
CLASSICAL_BEST = "moving average 13 + z * std"
MODEL_POLICY = "gbm median + z * (p90 - p50)"
FILL_TARGETS = [0.90, 0.95]


def md_table(frame):
    """Render a DataFrame as a GitHub Markdown table."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def frontier(points):
    front = points.sort_values("capital").drop_duplicates("capital").copy()
    front["fill_rate"] = front["fill_rate"].cummax()
    return front.drop_duplicates("fill_rate", keep="first")


def capital_for_fill(front, target):
    return float(np.interp(target, front["fill_rate"], front["capital"])) if target <= front["fill_rate"].max() else np.nan


def pct(value, digits=1):
    return f"{value:.{digits}%}"


def image(name, alt):
    """Markdown for a figure, or nothing if the figure does not exist."""
    return f"![{alt}](reports/figures/{name})\n\n" if (ROOT / "reports/figures" / name).exists() else ""


def data_section():
    clean_rows = len(pd.read_parquet(ROOT / "data/retail_clean.parquet", columns=["invoice"]))
    segments = pd.read_parquet(ROOT / "data/product_segments.parquet")
    modelled = int(segments["forecastable"].sum())
    share = segments.loc[segments["forecastable"], "total_units"].sum() / segments["total_units"].sum()
    log_path = ROOT / "reports/cleaning_log.csv"
    text = ""
    if log_path.exists():
        log = pd.read_csv(log_path)
        raw_rows = clean_rows + int(log["rows_removed"].sum())
        text += (f"- Raw rows: {raw_rows:,}. After cleaning: {clean_rows:,} ({clean_rows / raw_rows:.1%} kept).\n")
        removed = log[log["rows_removed"] > 0].rename(columns={"rule": "cleaning rule", "rows_removed": "rows removed"})
        removed["rows removed"] = removed["rows removed"].map(lambda v: f"{v:,}")
        text += "\n" + md_table(removed) + "\n\n"
    text += (f"- Products with enough history to model: {modelled:,} of {len(segments):,}, "
             f"covering {share:.0%} of all units sold.\n")
    return text


def accuracy_section():
    results = pd.read_csv(ROOT / "reports/model_results.csv")
    point = results[(results["group"] == "all products") & results["wape"].notna()].set_index("method")
    table = point[["wape", "bias"]].sort_values("wape")
    table.index.name = "method"
    view = table.reset_index()
    view["wape"] = view["wape"].map(lambda v: f"{v:.3f}")
    view["bias"] = view["bias"].map(lambda v: f"{v:+.3f}")
    view = view.rename(columns={"wape": "WAPE (lower is better)", "bias": "bias (positive = over-forecast)"})
    baselines = point["wape"].drop("gbm_p50")
    best_name, best = baselines.idxmin(), baselines.min()
    model = point.loc["gbm_p50", "wape"]
    last_week = point.loc["last_week", "wape"]
    coverage = results[(results["group"] == "all products") & (results["method"] == "gbm_interval_p10_p90")]["coverage"].iloc[0]
    text = md_table(view) + "\n\n"
    text += (f"- The median forecast of the model has WAPE {model:.3f}, which is {1 - model / best:.1%} lower than the best simple baseline "
             f"(`{best_name}`, {best:.3f}) and {1 - model / last_week:.1%} lower than repeating last week ({last_week:.3f}).\n")
    text += (f"- The p10 to p90 interval covers {coverage:.1%} of actual values (target 80%), so the intervals are wider than needed.\n")
    text += "- The median under-forecasts total units (negative bias). This is expected for skewed demand, so ordering uses the upper part of the interval.\n"
    return text


def decision_section():
    curves = pd.read_csv(ROOT / "reports/decision_curves.csv")
    fronts = {name: frontier(part) for name, part in curves.groupby("policy")}
    rows = []
    for name, front in fronts.items():
        row = {"policy": name}
        for target in FILL_TARGETS:
            row[f"capital for {target:.0%} fill"] = capital_for_fill(front, target)
        rows.append(row)
    table = pd.DataFrame(rows)
    reference = table.set_index("policy").loc[CLASSICAL_BEST]
    model = table.set_index("policy").loc[MODEL_POLICY]
    view = table.copy()
    for column in view.columns[1:]:
        view[column] = view[column].map(lambda v: "not reached" if pd.isna(v) else f"{v:,.0f}")
    text = ("Each ordering policy has one dial: a multiplier k or a safety factor z. In-sample sweep (the dial was swept on the evaluated weeks, so this view is optimistic for every policy):\n\n"
            + md_table(view) + "\n\n")
    parts = [f"{1 - model[c] / reference[c]:.1%} less capital at {c.split()[2]} fill" for c in table.columns[1:]
             if not (pd.isna(model[c]) or pd.isna(reference[c]))]
    if parts:
        text += f"- Against the best tuned classical baseline, the quantile model needs " + " and ".join(parts) + ".\n"
    return text


def out_of_sample_section():
    path = ROOT / "reports/out_of_sample_results.csv"
    if not path.exists():
        return ""
    data = pd.read_csv(path)
    text = ("The dial (k or z) of each policy is chosen on the first 13 test weeks and then fixed and evaluated on the last 13 weeks "
            "(the busy autumn season).\n\n")
    for mode, part in data.groupby("mode", sort=False):
        view = pd.DataFrame({"policy": part["policy"], "dial": part["tuned_parameter"].round(1),
                             "fill rate": part["eval_fill_rate"].map(pct), "capital": part["eval_capital"].map(lambda v: f"{v:,.0f}"),
                             "total cost": part["eval_total_cost"].map(lambda v: f"{v:,.0f}")})
        text += f"Dial chosen by: **{mode}**\n\n" + md_table(view) + "\n\n"
    first = data[data["mode"] == data["mode"].iloc[0]].set_index("policy")["eval_total_cost"]
    best_baseline = first[BASELINE_POLICIES].min()
    text += (f"- With the dial chosen by lowest cost, the quantile model's total cost is {1 - first[MODEL_POLICY] / best_baseline:.1%} lower "
             f"than the best tuned baseline and {1 - first[MODEL_POLICY] / first['repeat last week x k']:.1%} lower than repeat last week x k.\n")
    return text


STATIC_TOP = """# StockMind: demand forecasting and ordering decisions

StockMind forecasts weekly demand for each product with a prediction interval and tests how the forecasts change ordering decisions.
It is built on about one million real retail transactions (UCI Online Retail II) and evaluated with a strict time-based backtest:
models learn only from the past and are scored on the last 26 weeks.

**Honest summary:** the quantile model beats simple rules clearly, and beats a well-tuned classical safety-stock baseline by a small margin.
Cost figures depend on assumptions that are not in the data (see Limitations).
"""

STATIC_BOTTOM = """## Limitations

- One dataset from one UK wholesaler, two years of history, one test window of 26 weeks.
- Only one autumn is available for training, so seasonality is learned from a single cycle.
- Cost figures use assumptions that are not in the data: unit cost is 60% of the selling price and holding stock costs 20% of its value per year.
- The ordering simulation uses zero lead time, lost sales, and ignores capital limits and shelf space.
- The model is trained once; weekly retraining was not tested.
- Prediction intervals are conservative (coverage above the 80% target).
- The "current practice" (repeat last week) is an assumption, not something observed in the data.

## Method

1. **Clean:** remove cancellations, non-positive quantities and prices, fees and vouchers, duplicates, and sales cancelled within 24 hours.
2. **Weekly table:** units per product and week. Zeros count only between a product's first and last sale; the incomplete first and last weeks are dropped.
3. **Segment:** ABC by revenue, Syntetos-Boylan demand classes (smooth, erratic, intermittent, lumpy) and k-means clusters.
4. **Baselines:** repeat last week, 4-week and 13-week averages, 13-week median, same week last year.
5. **Model:** histogram-based gradient boosting with quantile loss (p10, p50, p90), trained once on weeks before the test period, with features built only from earlier weeks.
6. **Decide:** simulate weekly order-up-to policies and compare service level against capital tied up, first in-sample and then out-of-sample.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put online_retail_II.xlsx (UCI Machine Learning Repository) in data/
python src/load_data.py
python src/clean_data.py
python src/eda.py
python src/build_weekly.py
python src/segment_products.py
python src/forecast_model.py
python src/decision_backtest.py
python src/out_of_sample_check.py
python src/make_readme.py
streamlit run app/dashboard.py
```

## Project structure

```
src/        pipeline scripts (cleaning, weekly table, segmentation, baselines, model, backtests)
app/        Streamlit dashboard
reports/    result tables and figures
data/       generated data (not committed)
PROJECT_LOG.md   step-by-step log of decisions and findings
```

## Data

Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
Check the license terms on the dataset page before reuse.
"""


def main():
    parts = [STATIC_TOP,
             "## Results\n\n### Data\n\n" + data_section() + "\n" + image("01_weekly_units.png", "Weekly units sold")
             + image("05_abc_pareto.png", "About 1,000 products make 80% of revenue"),
             "### Forecast accuracy on the last 26 weeks (same product-weeks for every method)\n\n" + accuracy_section()
             + "\n" + image("08_model_vs_baselines.png", "WAPE of the baselines and the model"),
             "### Ordering decisions\n\n" + decision_section() + "\n" + image("09_service_vs_capital.png", "Service level versus capital tied up"),
             "### Out-of-sample check of the ordering policies\n\n" + out_of_sample_section(),
             STATIC_BOTTOM]
    (ROOT / "README.md").write_text("\n".join(parts))
    print("Saved:", ROOT / "README.md")


if __name__ == "__main__":
    main()
