# FILE: src/segment_products.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

WEEKLY_PATH = Path("data/weekly_demand.parquet")
PRODUCTS_PATH = Path("data/products.parquet")
OUT_PATH = Path("data/product_segments.parquet")
FIG_DIR = Path("reports/figures")

MIN_ACTIVE_WEEKS_CLUSTER = 26      # products with less history are too noisy to cluster
MIN_ACTIVE_WEEKS_MODEL = 52        # forecastable: at least one year on offer ...
MIN_SELLING_WEEKS_MODEL = 40       # ... and sold in at least 40 of those weeks


def demand_class(adi, cv2):
    """Syntetos-Boylan demand classes: smooth, erratic, intermittent, lumpy."""
    return np.select(
        [cv2.isna(), (adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
        ["insufficient", "smooth", "erratic", "intermittent"],
        default="lumpy",
    )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(WEEKLY_PATH)
    products = pd.read_parquet(PRODUCTS_PATH).set_index("stock_code")

    # --- demand statistics per product ---
    grouped = weekly.groupby("stock_code")["units"]
    stats = pd.DataFrame({"mean_units": grouped.mean(), "std_units": grouped.std(ddof=0)})
    stats["cv"] = stats["std_units"] / stats["mean_units"]

    nonzero = weekly[weekly["units"] > 0].groupby("stock_code")["units"]
    size_mean, size_std = nonzero.mean(), nonzero.std(ddof=0)
    stats = stats.join(products[["active_weeks", "selling_weeks", "total_units", "total_revenue",
                                 "mean_price", "description"]])
    stats["adi"] = stats["active_weeks"] / stats["selling_weeks"]            # average weeks between sales
    stats["cv2"] = ((size_std / size_mean) ** 2).where(stats["selling_weeks"] >= 2)

    q4_units = weekly[weekly["week"].dt.month >= 10].groupby("stock_code")["units"].sum()
    stats["q4_share"] = (q4_units / stats["total_units"]).reindex(stats.index).fillna(0)
    stats["demand_class"] = demand_class(stats["adi"], stats["cv2"])

    # --- ABC analysis by revenue ---
    stats = stats.sort_values("total_revenue", ascending=False)
    share = stats["total_revenue"] / stats["total_revenue"].sum()
    before = share.cumsum() - share
    stats["abc"] = np.select([before < 0.80, before < 0.95], ["A", "B"], default="C")

    # --- which products have enough history to model? ---
    stats["forecastable"] = ((stats["active_weeks"] >= MIN_ACTIVE_WEEKS_MODEL)
                             & (stats["selling_weeks"] >= MIN_SELLING_WEEKS_MODEL))

    # --- clustering (unsupervised learning) ---
    eligible = stats[stats["active_weeks"] >= MIN_ACTIVE_WEEKS_CLUSTER].copy()
    features = pd.DataFrame({
        "log_mean_units": np.log1p(eligible["mean_units"]),
        "cv": eligible["cv"],
        "log_adi": np.log(eligible["adi"]),
        "log_price": np.log(eligible["mean_price"]),
        "q4_share": eligible["q4_share"],
    }).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(features)

    best_k, best_score = None, -1
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
        score = silhouette_score(scaled, labels, sample_size=min(3000, len(scaled)), random_state=42)
        print(f"k={k}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    eligible["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(scaled)
    stats["cluster"] = eligible["cluster"].reindex(stats.index).fillna(-1).astype(int)

    stats.reset_index().to_parquet(OUT_PATH)

    # --- report ---
    total_units = stats["total_units"].sum()
    print(f"\nproducts: {len(stats)} | chosen k = {best_k} (silhouette {best_score:.3f})")
    print("\nABC (share of products / share of revenue):")
    abc = stats.groupby("abc").agg(products=("abc", "size"), revenue=("total_revenue", "sum"))
    abc["product_share"] = (abc["products"] / len(stats)).round(3)
    abc["revenue_share"] = (abc["revenue"] / abc["revenue"].sum()).round(3)
    print(abc[["products", "product_share", "revenue_share"]].to_string())
    print("\ndemand classes:")
    print(stats["demand_class"].value_counts().to_string())
    print("\nclusters (medians):")
    summary = stats[stats["cluster"] >= 0].groupby("cluster").agg(
        products=("cluster", "size"), mean_units=("mean_units", "median"), cv=("cv", "median"),
        adi=("adi", "median"), price=("mean_price", "median"), q4_share=("q4_share", "median"))
    print(summary.round(2).to_string())
    model_set = stats[stats["forecastable"]]
    print(f"\nforecastable products: {len(model_set)} of {len(stats)} "
          f"({model_set['total_units'].sum() / total_units:.1%} of all units)")
    print(model_set["demand_class"].value_counts().to_string())

    # --- figures ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(stats) + 1), share.cumsum().values * 100)
    ax.axhline(80, linestyle="--", color="gray")
    ax.axhline(95, linestyle="--", color="gray")
    ax.set_title("Pareto curve: cumulative revenue share by product rank")
    ax.set_xlabel("Product rank by revenue")
    ax.set_ylabel("Cumulative revenue share (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_abc_pareto.png", dpi=150)
    plt.close(fig)

    plot_df = stats[stats["cv2"].notna() & (stats["cv2"] > 0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, part in plot_df.groupby("demand_class"):
        ax.scatter(part["adi"], part["cv2"], s=6, alpha=0.5, label=f"{name} ({len(part)})")
    ax.axvline(1.32, linestyle="--", color="gray")
    ax.axhline(0.49, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Demand classes: ADI versus CV squared")
    ax.set_xlabel("ADI (average weeks between sales)")
    ax.set_ylabel("CV squared of non-zero demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_demand_classes.png", dpi=150)
    plt.close(fig)
    print("\nSaved:", OUT_PATH, "and 2 figures in", FIG_DIR)


if __name__ == "__main__":
    main()
