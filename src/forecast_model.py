# FILE: src/forecast_model.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

from baselines import build_forecasts, score, TEST_WEEKS

WEEKLY_PATH = Path("data/weekly_demand.parquet")
SEGMENTS_PATH = Path("data/product_segments.parquet")
PRODUCTS_PATH = Path("data/products.parquet")
PREDICTIONS_PATH = Path("data/model_predictions.parquet")
RESULTS_PATH = Path("reports/model_results.csv")
FIG_DIR = Path("reports/figures")

QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
MODEL_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                    min_samples_leaf=50, l2_regularization=1.0,
                    early_stopping=False, random_state=42)
FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag8", "lag13", "mean4", "mean13", "std4", "std13",
            "nonzero13", "week_of_year", "month", "train_mean", "train_nonzero", "mean_price"]


def build_feature_table(wide, test_start, prices):
    """One row per product and week. Every feature of week t uses only weeks before t."""
    past = wide.shift(1)
    was_sale = (past > 0).astype(float).where(past.notna())
    feats = {
        "lag1": wide.shift(1), "lag2": wide.shift(2), "lag3": wide.shift(3), "lag4": wide.shift(4),
        "lag8": wide.shift(8), "lag13": wide.shift(13),
        "mean4": past.rolling(4, min_periods=4).mean(),
        "mean13": past.rolling(13, min_periods=13).mean(),
        "std4": past.rolling(4, min_periods=4).std(),
        "std13": past.rolling(13, min_periods=13).std(),
        "nonzero13": was_sale.rolling(13, min_periods=13).mean(),
    }
    n_weeks, n_products = wide.shape
    table = pd.DataFrame({name: frame.to_numpy().ravel() for name, frame in feats.items()})
    table.insert(0, "week", np.repeat(wide.index.values, n_products))
    table.insert(1, "stock_code", np.tile(wide.columns.values, n_weeks))
    table.insert(2, "units", wide.to_numpy().ravel())
    table["week_of_year"] = pd.DatetimeIndex(table["week"]).isocalendar().week.to_numpy().astype(int)
    table["month"] = pd.DatetimeIndex(table["week"]).month.to_numpy()

    # Product-level features come from the training period only, so nothing from the test period leaks in.
    train = wide.loc[wide.index < test_start]
    table["train_mean"] = table["stock_code"].map(train.mean())
    table["train_nonzero"] = table["stock_code"].map((train > 0).astype(float).where(train.notna()).mean())
    table["mean_price"] = table["stock_code"].map(prices)
    return table


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(WEEKLY_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH).set_index("stock_code")
    prices = pd.read_parquet(PRODUCTS_PATH).set_index("stock_code")["mean_price"]

    model_codes = segments.index[segments["forecastable"]]
    wide = (weekly[weekly["stock_code"].isin(model_codes)]
            .pivot(index="week", columns="stock_code", values="units").sort_index())
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="7D"))
    test = wide.index[-TEST_WEEKS:]

    table = build_feature_table(wide, test[0], prices)
    usable = table["units"].notna() & table[FEATURES].notna().all(axis=1)
    train_rows = table[usable & (table["week"] < test[0])]
    test_rows = table[usable & table["week"].isin(test)].copy()
    print(f"training rows: {len(train_rows)} | test rows: {len(test_rows)}")

    # Quantile models are trained on log(1 + units). Quantiles are unchanged by this transform,
    # and the log makes the skewed demand much easier to learn.
    target = np.log1p(train_rows["units"])
    predictions = {}
    for name, alpha in QUANTILES.items():
        model = HistGradientBoostingRegressor(loss="quantile", quantile=alpha, **MODEL_PARAMS)
        model.fit(train_rows[FEATURES], target)
        predictions[name] = np.expm1(model.predict(test_rows[FEATURES])).clip(min=0)
        print(f"trained model {name} (quantile {alpha})")
    ordered = np.sort(np.column_stack(list(predictions.values())), axis=1)   # no crossing quantiles
    for i, name in enumerate(QUANTILES):
        test_rows[name] = ordered[:, i]

    def to_wide(column):
        return (test_rows.pivot(index="week", columns="stock_code", values=column)
                .reindex(index=test, columns=wide.columns))

    forecasts = {name: frame for name, frame in build_forecasts(wide).items()}
    forecasts["gbm_p50"] = to_wide("p50")
    p10, p90 = to_wide("p10"), to_wide("p90")

    # Fair comparison: every method is scored on exactly the same product-weeks.
    mask = wide.loc[test].notna()
    for forecast in forecasts.values():
        mask &= forecast.loc[test].notna()
    actual = wide.loc[test].where(mask)
    print(f"scored product-weeks: {int(mask.sum().sum())} "
          f"| share of actual units covered: {actual.sum().sum() / wide.loc[test].sum().sum():.1%}")

    groups = {"all products": list(wide.columns)}
    for abc, part in segments.loc[model_codes].groupby("abc"):
        groups[f"ABC class {abc}"] = list(part.index)
    for cls, part in segments.loc[model_codes].groupby("demand_class"):
        groups[f"demand: {cls}"] = list(part.index)

    rows = []
    for group_name, cols in groups.items():
        for method, forecast in forecasts.items():
            wape_value, bias_value = score(actual[cols], forecast.loc[test].where(mask)[cols])
            rows.append({"group": group_name, "method": method, "wape": wape_value, "bias": bias_value})
        inside = ((actual[cols] >= p10.where(mask)[cols]) & (actual[cols] <= p90.where(mask)[cols]))
        n_cells = int(mask[cols].sum().sum())
        width = (p90.where(mask)[cols] - p10.where(mask)[cols]).sum().sum() / actual[cols].sum().sum()
        rows.append({"group": group_name, "method": "gbm_interval_p10_p90",
                     "coverage": inside.sum().sum() / n_cells, "relative_width": width})
    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    point = results[results["method"] != "gbm_interval_p10_p90"]
    print("\nWAPE (lower is better):")
    print(point.pivot(index="group", columns="method", values="wape").round(3).to_string())
    print("\nBias (positive = over-forecast):")
    print(point.pivot(index="group", columns="method", values="bias").round(3).to_string())
    interval = results[results["method"] == "gbm_interval_p10_p90"].set_index("group")
    print("\nPrediction interval p10 to p90 (target coverage is 80%):")
    print(interval[["coverage", "relative_width"]].round(3).to_string())

    test_rows["in_scored_set"] = [bool(mask.at[w, s]) for w, s in zip(test_rows["week"], test_rows["stock_code"])]
    test_rows[["week", "stock_code", "units", "p10", "p50", "p90", "in_scored_set"]].to_parquet(PREDICTIONS_PATH)

    overall = point[point["group"] == "all products"].set_index("method")["wape"]
    colors = ["tab:orange" if name == "gbm_p50" else "tab:blue" for name in overall.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(overall.index, overall.values, color=colors)
    ax.set_title("WAPE on the test period: baselines versus gradient boosting (lower is better)")
    ax.set_ylabel("WAPE")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_model_vs_baselines.png", dpi=150)
    plt.close(fig)
    print("\nSaved:", PREDICTIONS_PATH, RESULTS_PATH, "and", FIG_DIR / "08_model_vs_baselines.png")


if __name__ == "__main__":
    main()
