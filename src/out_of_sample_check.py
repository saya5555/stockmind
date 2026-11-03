# FILE: src/out_of_sample_check.py
import numpy as np
import pandas as pd
from pathlib import Path

from baselines import TEST_WEEKS
from decision_backtest import simulate, COST_RATIO

WEEKLY_PATH = Path("data/weekly_demand.parquet")
SEGMENTS_PATH = Path("data/product_segments.parquet")
PRODUCTS_PATH = Path("data/products.parquet")
PREDICTIONS_PATH = Path("data/model_predictions.parquet")
RESULTS_PATH = Path("reports/out_of_sample_results.csv")

TUNE_WEEKS = 13                                        # first half of the test period: choose k or z here
PARAMETERS = np.round(np.arange(0.0, 12.01, 0.1), 2)   # wider grid than before, so the best value is not cut off
TARGET_FILL = 0.95


def load_inputs():
    weekly = pd.read_parquet(WEEKLY_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH).set_index("stock_code")
    price_by_code = pd.read_parquet(PRODUCTS_PATH).set_index("stock_code")["mean_price"]
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    model_codes = segments.index[segments["forecastable"]]
    wide = (weekly[weekly["stock_code"].isin(model_codes)]
            .pivot(index="week", columns="stock_code", values="units").sort_index())
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="7D"))
    test = wide.index[-TEST_WEEKS:]

    def from_predictions(column):
        return predictions.pivot(index="week", columns="stock_code", values=column).reindex(index=test, columns=wide.columns)

    past = wide.shift(1)
    arrays = {
        "active": from_predictions("in_scored_set").eq(True).to_numpy(),
        "demand": wide.loc[test].fillna(0).to_numpy(),
        "last_week": past.loc[test], "ma13": past.rolling(13, min_periods=13).mean().loc[test],
        "std13": past.rolling(13, min_periods=13).std().loc[test],
        "p50": from_predictions("p50"), "p90": from_predictions("p90"),
    }
    for name in ["last_week", "ma13", "std13", "p50", "p90"]:
        arrays[name] = np.nan_to_num(arrays[name].to_numpy(), nan=0.0)
    price = price_by_code.reindex(wide.columns).to_numpy()
    return arrays, price


def run(target, arrays, price, weeks):
    return simulate(target[weeks], arrays["demand"][weeks], arrays["active"][weeks], COST_RATIO * price, price)


def main():
    arrays, price = load_inputs()
    tune, evaluate = slice(0, TUNE_WEEKS), slice(TUNE_WEEKS, TEST_WEEKS)
    print(f"tuning weeks: {TUNE_WEEKS} | evaluation weeks: {TEST_WEEKS - TUNE_WEEKS} "
          f"| product-weeks in evaluation half: {int(arrays['active'][evaluate].sum())}")

    policies = {
        "repeat last week x k": lambda a: a * arrays["last_week"],
        "moving average 13 x k": lambda a: a * arrays["ma13"],
        "moving average 13 + z * std": lambda a: arrays["ma13"] + a * arrays["std13"],
        "gbm median x k": lambda a: a * arrays["p50"],
        "gbm median + z * (p90 - p50)": lambda a: arrays["p50"] + a * (arrays["p90"] - arrays["p50"]),
    }
    current = run(1.0 * arrays["last_week"], arrays, price, evaluate)
    print(f"\nCurrent practice on the evaluation half (repeat last week, k = 1): fill {current['fill_rate']:.1%} | "
          f"capital {current['capital']:,.0f} | total cost {current['total_cost']:,.0f}")

    rows = []
    for mode in ["lowest total cost", f"smallest parameter reaching {TARGET_FILL:.0%} fill"]:
        print(f"\nParameter chosen on the tuning half by: {mode}")
        print(f"{'policy':32s} {'param':>6s} {'eval fill':>10s} {'eval capital':>13s} {'eval total cost':>16s} {'cost vs current':>16s}")
        for name, build in policies.items():
            tuning = [run(build(p), arrays, price, tune) for p in PARAMETERS]
            if mode == "lowest total cost":
                index = int(np.argmin([t["total_cost"] for t in tuning]))
            else:
                reaching = [i for i, t in enumerate(tuning) if t["fill_rate"] >= TARGET_FILL]
                index = reaching[0] if reaching else len(PARAMETERS) - 1
            parameter = PARAMETERS[index]
            result = run(build(parameter), arrays, price, evaluate)
            edge = " (edge)" if index == len(PARAMETERS) - 1 else ""
            change = result["total_cost"] / current["total_cost"] - 1
            print(f"{name:32s} {parameter:6.1f} {result['fill_rate']:10.1%} {result['capital']:13,.0f} "
                  f"{result['total_cost']:16,.0f} {change:+16.1%}{edge}")
            rows.append({"mode": mode, "policy": name, "tuned_parameter": parameter,
                         "eval_fill_rate": result["fill_rate"], "eval_capital": result["capital"],
                         "eval_total_cost": result["total_cost"], "cost_vs_current": change})
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print("\nSaved:", RESULTS_PATH)


if __name__ == "__main__":
    main()
