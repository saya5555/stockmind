# FILE: src/decision_backtest.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from pathlib import Path

from baselines import TEST_WEEKS

WEEKLY_PATH = Path("data/weekly_demand.parquet")
SEGMENTS_PATH = Path("data/product_segments.parquet")
PRODUCTS_PATH = Path("data/products.parquet")
PREDICTIONS_PATH = Path("data/model_predictions.parquet")
CURVES_PATH = Path("reports/decision_curves.csv")
RESULTS_PATH = Path("reports/decision_results.csv")
FIG_DIR = Path("reports/figures")

# Cost assumptions. The dataset has no cost data, so these are explicit assumptions, not facts.
COST_RATIO = 0.60               # unit cost = 60% of the selling price (so the margin is 40%)
HOLDING_RATE_PER_YEAR = 0.20    # holding stock for a year costs 20% of its value
PARAMETERS = np.round(np.arange(0.0, 8.01, 0.1), 2)   # multiplier k or safety factor z
FILL_TARGETS = [0.90, 0.95, 0.98]


def simulate(target, demand, active, unit_cost, price):
    """Weekly order-up-to policy with carry-over stock.

    At the start of each week we order so that the stock reaches the target (never a negative order).
    Unsold stock is carried to the next week. Demand that cannot be served is lost.
    """
    leftover = np.zeros(demand.shape[1])
    sales_total = demand_total = lost_margin = capital_total = 0.0
    stockout_cells = cells = 0
    for week in range(demand.shape[0]):
        act = active[week]
        available = np.maximum(leftover, np.where(act, target[week], 0.0))
        sales = np.minimum(demand[week], available)
        left = available - sales
        lost = demand[week] - sales
        on_hand = (available + left) / 2            # average stock held during the week
        sales_total += sales[act].sum()
        demand_total += demand[week][act].sum()
        lost_margin += (lost * price * (1 - COST_RATIO))[act].sum()
        capital_total += (on_hand * unit_cost)[act].sum()
        stockout_cells += int(((lost > 1e-9) & act).sum())
        cells += int(act.sum())
        leftover = np.where(act, left, leftover)
    capital = capital_total / demand.shape[0]        # average capital tied up per week
    holding_cost = capital * HOLDING_RATE_PER_YEAR * demand.shape[0] / 52
    return {"fill_rate": sales_total / demand_total, "stockout_rate": stockout_cells / cells,
            "capital": capital, "holding_cost": holding_cost, "lost_margin": lost_margin,
            "total_cost": holding_cost + lost_margin}


def frontier(points):
    """Sort by capital and keep only the best achievable service at each capital level."""
    front = points.sort_values("capital").drop_duplicates("capital").copy()
    front["fill_rate"] = front["fill_rate"].cummax()
    front["stockout_rate"] = front["stockout_rate"].cummin()
    return front.drop_duplicates("fill_rate", keep="first")


def capital_for_fill(front, target):
    return float(np.interp(target, front["fill_rate"], front["capital"])) if target <= front["fill_rate"].max() else np.nan


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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

    active = from_predictions("in_scored_set").eq(True).to_numpy()     # same product-weeks as the forecast comparison
    past = wide.shift(1)
    inputs = {
        "last_week": past.loc[test],
        "ma13": past.rolling(13, min_periods=13).mean().loc[test],
        "std13": past.rolling(13, min_periods=13).std().loc[test],
        "p50": from_predictions("p50"),
        "p90": from_predictions("p90"),
    }
    last_week, ma13, std13, p50, p90 = [np.nan_to_num(inputs[n].to_numpy(), nan=0.0)
                                        for n in ["last_week", "ma13", "std13", "p50", "p90"]]
    demand = wide.loc[test].fillna(0).to_numpy()
    price = price_by_code.reindex(wide.columns).to_numpy()
    unit_cost = COST_RATIO * price
    print(f"test weeks: {len(test)} | products: {demand.shape[1]} | simulated product-weeks: {int(active.sum())}")

    policies = {
        "repeat last week x k": lambda a: a * last_week,
        "moving average 13 x k": lambda a: a * ma13,
        "moving average 13 + z * std": lambda a: ma13 + a * std13,
        "gbm median x k": lambda a: a * p50,
        "gbm median + z * (p90 - p50)": lambda a: p50 + a * (p90 - p50),
    }
    rows = []
    for name, build in policies.items():
        for parameter in PARAMETERS:
            result = simulate(build(parameter), demand, active, unit_cost, price)
            rows.append({"policy": name, "parameter": parameter, **result})
    curves = pd.DataFrame(rows)
    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(CURVES_PATH, index=False)
    fronts = {name: frontier(part) for name, part in curves.groupby("policy")}

    # 1. Current practice: order what was sold last week (k = 1)
    now = curves[(curves["policy"] == "repeat last week x k") & (curves["parameter"] == 1.0)].iloc[0]
    print("\nCurrent practice (repeat last week, k = 1):")
    print(f"  fill rate {now['fill_rate']:.1%} | weeks with a stockout {now['stockout_rate']:.1%} | "
          f"average capital tied up {now['capital']:,.0f} | total cost {now['total_cost']:,.0f}")

    # 2. Same service level: how much capital does each policy need to match the current fill rate?
    print(f"\nCapital needed to reach the current fill rate ({now['fill_rate']:.1%}):")
    summary = []
    for name, front in fronts.items():
        needed = capital_for_fill(front, now["fill_rate"])
        change = needed / now["capital"] - 1
        summary.append({"comparison": "same fill rate", "policy": name, "value": needed, "change_vs_current": change})
        print(f"  {name:32s} capital {needed:12,.0f}   change vs current {change:+.1%}")

    # 3. Same capital: how much service does each policy deliver with the current capital?
    print(f"\nService with the current capital ({now['capital']:,.0f}):")
    for name, front in fronts.items():
        fill = float(np.interp(now["capital"], front["capital"], front["fill_rate"]))
        stockout = float(np.interp(now["capital"], front["capital"], front["stockout_rate"]))
        summary.append({"comparison": "same capital", "policy": name, "value": fill, "change_vs_current": stockout})
        print(f"  {name:32s} fill rate {fill:.1%}   weeks with a stockout {stockout:.1%}")

    # 4. Capital needed for fixed service targets
    print("\nCapital needed for a target fill rate (blank = not reachable):")
    table = pd.DataFrame({name: [capital_for_fill(front, t) for t in FILL_TARGETS] for name, front in fronts.items()},
                         index=[f"fill {t:.0%}" for t in FILL_TARGETS])
    print(table.round(0).to_string())

    # 5. Cheapest total cost under the stated cost assumptions
    print(f"\nLowest total cost over the parameter grid (unit cost = {COST_RATIO:.0%} of price, holding = {HOLDING_RATE_PER_YEAR:.0%} per year):")
    for name, part in curves.groupby("policy"):
        best = part.loc[part["total_cost"].idxmin()]
        summary.append({"comparison": "lowest total cost", "policy": name, "value": best["total_cost"],
                        "change_vs_current": best["total_cost"] / now["total_cost"] - 1})
        print(f"  {name:32s} total cost {best['total_cost']:12,.0f}  (parameter {best['parameter']:.1f}, "
              f"fill {best['fill_rate']:.1%}, capital {best['capital']:,.0f})  vs current {best['total_cost'] / now['total_cost'] - 1:+.1%}")
    pd.DataFrame(summary).to_csv(RESULTS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, front in fronts.items():
        ax.plot(front["capital"] / 1000, front["fill_rate"] * 100, label=name)
    ax.scatter([now["capital"] / 1000], [now["fill_rate"] * 100], color="black", zorder=5, label="current practice")
    ax.set_xlabel("Average capital tied up per week (thousands, at cost)")
    ax.set_ylabel("Fill rate (% of demand served)")
    ax.set_title("Service level versus capital tied up (up and to the left is better)")
    ax.set_ylim(40, 101)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_service_vs_capital.png", dpi=150)
    plt.close(fig)
    print("\nSaved:", CURVES_PATH, RESULTS_PATH, "and", FIG_DIR / "09_service_vs_capital.png")


if __name__ == "__main__":
    main()
