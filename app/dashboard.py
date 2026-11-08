# FILE: app/dashboard.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
REFERENCE_POLICY = "moving average 13 + z * std"    # best classical baseline (safety stock from recent variability)
MODEL_POLICY = "gbm median + z * (p90 - p50)"
COST_RATIO, HOLDING_RATE = 0.60, 0.20               # explicit assumptions, not facts from the data

st.set_page_config(page_title="StockMind", layout="wide")
from ui_style import inject_css
inject_css()


# ---------- helpers ----------
def read_csv(relative_path):
    path = ROOT / relative_path
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def read_parquet(relative_path):
    return pd.read_parquet(ROOT / relative_path)


def show_image(name, caption):
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Missing figure: {name}")


def frontier(points):
    """Best service level achievable at each capital level."""
    front = points.sort_values("capital").drop_duplicates("capital").copy()
    front["fill_rate"] = front["fill_rate"].cummax()
    front["stockout_rate"] = front["stockout_rate"].cummin()
    return front.drop_duplicates("fill_rate", keep="first")


def capital_for_fill(front, target):
    if target > front["fill_rate"].max():
        return np.nan
    return float(np.interp(target, front["fill_rate"], front["capital"]))


def percent(value):
    return "" if pd.isna(value) else f"{value:.1%}"


# ---------- tabs ----------
def overview_tab(model_results, segments):
    st.subheader("What this project does")
    st.markdown(
        "StockMind forecasts weekly demand for each product, with a prediction interval, and uses the forecast to decide "
        "how much to order. It is tested on about one million real retail transactions (UCI Online Retail II) with a "
        "strict time-based backtest: models learn from the past only and are scored on the last 26 weeks.")
    modelled = int(segments["forecastable"].sum())
    total_units = segments["total_units"].sum()
    share = segments.loc[segments["forecastable"], "total_units"].sum() / total_units
    columns = st.columns(4)
    columns[0].metric("Products in the dataset", f"{len(segments):,}")
    columns[1].metric("Products modelled", f"{modelled:,}", f"{share:.0%} of all units")
    if model_results is not None:
        overall = model_results[(model_results["group"] == "all products") & model_results["wape"].notna()]
        wape = overall.set_index("method")["wape"]
        baselines = wape.drop("gbm_p50", errors="ignore")
        columns[2].metric("Forecast error (WAPE), model", f"{wape['gbm_p50']:.3f}",
                          f"{wape['gbm_p50'] / baselines.min() - 1:+.1%} vs best baseline", delta_color="inverse")
        interval = model_results[(model_results["group"] == "all products") & (model_results["method"] == "gbm_interval_p10_p90")]
        if len(interval):
            columns[3].metric("p10 to p90 interval coverage", f"{interval['coverage'].iloc[0]:.1%}", "target 80%", delta_color="off")
    st.info("Cost results use assumptions that are not in the data: unit cost is 60% of the selling price and holding "
            "stock costs 20% of its value per year. Treat money figures as illustrations of the method, not as a forecast of savings.")


def data_tab(cleaning_log):
    st.subheader("Cleaning")
    if cleaning_log is not None:
        st.dataframe(cleaning_log)
    st.markdown("Order lines that were cancelled within 24 hours (same customer, product and quantity) are removed, "
                "because they are order mistakes and not real demand. Two of them alone were 155,210 units.")
    left, right = st.columns(2)
    with left:
        show_image("01_weekly_units.png", "Weekly units sold")
        show_image("05_abc_pareto.png", "About 1,000 products make 80% of revenue")
    with right:
        show_image("02_top_products.png", "Top 15 products by units")
        show_image("06_demand_classes.png", "Most products have erratic or lumpy demand")


def accuracy_tab(model_results):
    st.subheader("Forecast accuracy on the last 26 weeks")
    if model_results is None:
        st.warning("reports/model_results.csv not found. Run src/forecast_model.py first.")
        return
    point = model_results[model_results["wape"].notna()]
    wape = point.pivot(index="group", columns="method", values="wape")
    st.markdown("**WAPE** (total absolute error divided by total actual units, lower is better). "
                "Every method is scored on exactly the same product-weeks.")
    st.dataframe(wape.round(3))
    overall = wape.loc["all products"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(overall.index, overall.values, color=["tab:orange" if m == "gbm_p50" else "tab:blue" for m in overall.index])
    ax.set_ylabel("WAPE")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Bias** (positive means over-forecast). The median forecast under-forecasts total units, which is expected "
                "for skewed demand, so ordering decisions use the upper part of the interval.")
    st.dataframe(point.pivot(index="group", columns="method", values="bias").round(3))
    interval = model_results[model_results["method"] == "gbm_interval_p10_p90"].set_index("group")[["coverage", "relative_width"]]
    st.markdown("**Prediction interval p10 to p90.** Coverage should be 80%. Higher means the intervals are wider than needed.")
    st.dataframe(interval.round(3))


def product_tab(segments):
    st.subheader("Product explorer")
    modelled = segments[segments["forecastable"]].sort_values("total_revenue", ascending=False)
    labels = (modelled["stock_code"] + " | " + modelled["description"].fillna("").astype(str).str.slice(0, 40)
              + " | ABC " + modelled["abc"] + " | " + modelled["demand_class"]).tolist()
    choice = st.selectbox("Choose a product (sorted by revenue)", labels)
    code = choice.split(" | ")[0]
    row = modelled[modelled["stock_code"] == code].iloc[0]
    weekly = read_parquet("data/weekly_demand.parquet")
    predictions = read_parquet("data/model_predictions.parquet")
    history = weekly[weekly["stock_code"] == code].sort_values("week")
    forecast = predictions[predictions["stock_code"] == code].sort_values("week")

    columns = st.columns(4)
    columns[0].metric("ABC class", row["abc"])
    columns[1].metric("Demand class", row["demand_class"])
    columns[2].metric("Average units per week", f"{row['mean_units']:.1f}")
    columns[3].metric("Weeks with sales", f"{int(row['selling_weeks'])} of {int(row['active_weeks'])}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["week"], history["units"], color="tab:blue", label="actual units")
    if len(forecast):
        ax.fill_between(forecast["week"], forecast["p10"], forecast["p90"], color="tab:orange", alpha=0.25, label="p10 to p90")
        ax.plot(forecast["week"], forecast["p50"], color="tab:orange", label="forecast (median)")
    ax.set_ylabel("Units per week")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Orange: one-week-ahead forecasts for the 26 test weeks, made with information from earlier weeks only.")


def decision_tab(curves, out_of_sample):
    st.subheader("Ordering simulation")
    if curves is None:
        st.warning("reports/decision_curves.csv not found. Run src/decision_backtest.py first.")
        return
    st.markdown("Each week, stock is ordered up to a target. Unsold stock is carried over, unserved demand is lost, lead time is zero. "
                "Each policy has one dial (a multiplier k or a safety factor z). Sweeping the dial gives a curve of service level against "
                "the capital tied up in stock.")
    target = st.slider("Target fill rate (share of demand served)", 80, 98, 95, format="%d%%") / 100
    fronts = {name: frontier(part) for name, part in curves.groupby("policy")}
    table = pd.DataFrame({"policy": list(fronts), "capital needed": [capital_for_fill(f, target) for f in fronts.values()]})
    reference = table.loc[table["policy"] == REFERENCE_POLICY, "capital needed"]
    reference = float(reference.iloc[0]) if len(reference) else np.nan
    table["vs best classical baseline"] = table["capital needed"] / reference - 1
    shown = table.copy()
    shown["capital needed"] = shown["capital needed"].map(lambda v: "not reached" if pd.isna(v) else f"{v:,.0f}")
    shown["vs best classical baseline"] = shown["vs best classical baseline"].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    st.dataframe(shown)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, front in fronts.items():
        ax.plot(front["capital"] / 1000, front["fill_rate"] * 100, label=name)
    ax.axhline(target * 100, color="gray", linestyle="--")
    ax.set_xlabel("Average capital tied up per week (thousands, at cost)")
    ax.set_ylabel("Fill rate (%)")
    ax.set_ylim(40, 101)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("In-sample view: the dial was swept on the same weeks that are evaluated, so every policy looks a little better than it would in practice.")

    st.subheader("Out-of-sample check")
    if out_of_sample is None:
        st.warning("reports/out_of_sample_results.csv not found. Run src/out_of_sample_check.py first.")
        return
    st.markdown("The dial is chosen on the first 13 test weeks and then fixed and evaluated on the next 13 weeks (September to November, the busy season).")
    for mode, part in out_of_sample.groupby("mode", sort=False):
        view = pd.DataFrame({
            "policy": part["policy"], "dial": part["tuned_parameter"].round(1),
            "fill rate": part["eval_fill_rate"].map(percent), "capital": part["eval_capital"].map(lambda v: f"{v:,.0f}"),
            "total cost": part["eval_total_cost"].map(lambda v: f"{v:,.0f}"),
        })
        st.markdown(f"**Dial chosen by: {mode}**")
        st.dataframe(view)


def method_tab():
    st.subheader("Method")
    st.markdown("""
1. **Clean:** remove cancellations, non-positive quantities and prices, fees and vouchers, duplicates, and sales that were cancelled within 24 hours.
2. **Weekly table:** units per product and week, with zeros counted only between a product's first and last sale. The first and last week are incomplete and dropped.
3. **Segment:** ABC by revenue, Syntetos-Boylan demand classes, and k-means clusters. Only the 2,042 products with enough history are modelled (81% of units).
4. **Baselines:** repeat last week, 4-week and 13-week averages, 13-week median, same week last year.
5. **Model:** histogram-based gradient boosting with quantile loss (p10, p50, p90), trained once on the weeks before the test period.
6. **Decide:** simulate weekly ordering policies and compare service level against capital tied up.
""")
    st.subheader("Limitations")
    st.markdown("""
- One dataset from one UK wholesaler, two years of history, one test window of 26 weeks.
- Only one autumn is available for training, so seasonality is learned from a single cycle.
- Cost figures rely on assumptions (unit cost 60% of price, holding 20% per year) that are not in the data.
- Zero lead time and lost sales are simplifications. Capital limits and shelf space are ignored.
- The model is trained once; weekly retraining was not tested.
- Prediction intervals are conservative (coverage above the 80% target).
- Gains over a well-tuned classical baseline are modest; gains over repeating last week are large.
""")


def main():
    st.title("StockMind: demand forecasting and ordering decisions")
    segments = read_parquet("data/product_segments.parquet")
    model_results = read_csv("reports/model_results.csv")
    tabs = st.tabs(["Overview", "Data", "Forecast accuracy", "Product explorer", "Ordering simulation", "Method and limits"])
    with tabs[0]:
        overview_tab(model_results, segments)
    with tabs[1]:
        data_tab(read_csv("reports/cleaning_log.csv"))
    with tabs[2]:
        accuracy_tab(model_results)
    with tabs[3]:
        product_tab(segments)
    with tabs[4]:
        decision_tab(read_csv("reports/decision_curves.csv"), read_csv("reports/out_of_sample_results.csv"))
    with tabs[5]:
        method_tab()


main()
