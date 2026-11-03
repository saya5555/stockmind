# FILE: src/eda.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from pathlib import Path

IN_PATH = Path("data/retail_clean.parquet")
FIG_DIR = Path("reports/figures")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PATH)

    # 1. Weekly units sold
    weekly = df.set_index("invoice_date")["quantity"].resample("W").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weekly.index, weekly.values)
    ax.set_title("Weekly units sold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    save(fig, "01_weekly_units.png")

    # 2. Top 15 products by units sold
    top = (df.groupby("stock_code")
             .agg(units=("quantity", "sum"), description=("description", "first"))
             .nlargest(15, "units")
             .sort_values("units"))
    labels = top.index + " " + top["description"].fillna("").astype(str).str.slice(0, 25)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top["units"])
    ax.set_title("Top 15 products by units sold")
    ax.set_xlabel("Units")
    save(fig, "02_top_products.png")

    # 3. Distribution of quantity per order line (log scale shows the long tail)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["quantity"], bins=60, log=True)
    ax.set_title("Quantity per order line (log scale)")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Number of lines")
    save(fig, "03_quantity_distribution.png")

    # 4. Units by country (top 8)
    by_country = df.groupby("country")["quantity"].sum().nlargest(8).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(by_country.index, by_country.values)
    ax.set_title("Units sold by country (top 8)")
    ax.set_xlabel("Units")
    save(fig, "04_units_by_country.png")

    print("\nweeks in data:", len(weekly))
    print("share of units from the top 15 products:", f"{top['units'].sum() / df['quantity'].sum():.1%}")


if __name__ == "__main__":
    main()
