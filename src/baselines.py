# FILE: src/baselines.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from pathlib import Path

WEEKLY_PATH = Path("data/weekly_demand.parquet")
SEGMENTS_PATH = Path("data/product_segments.parquet")
RESULTS_PATH = Path("reports/baseline_results.csv")
FIG_DIR = Path("reports/figures")

TEST_WEEKS = 26   # the last 26 complete weeks are the test period (rolling one-week-ahead forecasts)
SEASON = 52       # weeks in a year, used by the seasonal naive method


def build_forecasts(wide):
    """One-week-ahead forecasts. The forecast for week t only uses weeks before t."""
    return {
        "last_week": wide.shift(1),
        "moving_avg_4": wide.shift(1).rolling(4, min_periods=4).mean(),
        "moving_avg_13": wide.shift(1).rolling(13, min_periods=13).mean(),
        "moving_median_13": wide.shift(1).rolling(13, min_periods=13).median(),
        "same_week_last_year": wide.shift(SEASON),
    }


def score(actual, forecast):
    """WAPE = total absolute error / total actual units. Bias = total forecast error / total actual units."""
    total = actual.sum().sum()
    return ((forecast - actual).abs().sum().sum() / total,
            (forecast - actual).sum().sum() / total)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(WEEKLY_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH).set_index("stock_code")

    model_codes = segments.index[segments["forecastable"]]
    wide = (weekly[weekly["stock_code"].isin(model_codes)]
            .pivot(index="week", columns="stock_code", values="units")
            .sort_index())
    # Make sure every week is present, so that shifting by one row always means shifting by one week.
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="7D"))
    forecasts = build_forecasts(wide)

    # Fair comparison: score every method on exactly the same product-weeks.
    test = wide.index[-TEST_WEEKS:]
    mask = wide.loc[test].notna()
    for forecast in forecasts.values():
        mask &= forecast.loc[test].notna()
    actual = wide.loc[test].where(mask)
    print(f"test period: {test[0].date()} to {test[-1].date()} ({TEST_WEEKS} weeks)")
    print(f"products: {wide.shape[1]} | scored product-weeks: {int(mask.sum().sum())} "
          f"| share of actual units covered: {actual.sum().sum() / wide.loc[test].sum().sum():.1%}")

    groups = {"all products": list(wide.columns)}
    for abc, part in segments.loc[model_codes].groupby("abc"):
        groups[f"ABC class {abc}"] = list(part.index)
    for cls, part in segments.loc[model_codes].groupby("demand_class"):
        groups[f"demand: {cls}"] = list(part.index)

    rows = []
    for group_name, cols in groups.items():
        for method, forecast in forecasts.items():
            f = forecast.loc[test].where(mask)
            w, b = score(actual[cols], f[cols])
            rows.append({"group": group_name, "method": method, "wape": w, "bias": b,
                         "units": actual[cols].sum().sum()})
    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("\nWAPE (lower is better):")
    print(results.pivot(index="group", columns="method", values="wape").round(3).to_string())
    print("\nBias (positive = over-forecast):")
    print(results.pivot(index="group", columns="method", values="bias").round(3).to_string())

    overall = results[results["group"] == "all products"].set_index("method")["wape"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(overall.index, overall.values)
    ax.set_title("Baseline forecasts: WAPE on the test period (lower is better)")
    ax.set_ylabel("WAPE")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_baseline_wape.png", dpi=150)
    plt.close(fig)
    print("\nSaved:", RESULTS_PATH, "and", FIG_DIR / "07_baseline_wape.png")


if __name__ == "__main__":
    main()
