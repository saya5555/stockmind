# FILE: src/build_weekly.py
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/retail_clean.parquet")
WEEKLY_PATH = Path("data/weekly_demand.parquet")
PRODUCTS_PATH = Path("data/products.parquet")


def main():
    df = pd.read_parquet(IN_PATH)

    # Weeks run Monday to Sunday and are labelled by their Monday.
    df["week"] = df["invoice_date"].dt.to_period("W-SUN").dt.start_time

    # The first week (data starts on a Tuesday) and the last week (data ends on a Friday) are incomplete.
    # Incomplete weeks would look like a sudden drop in demand, so we drop them.
    first_week, last_week = df["week"].min(), df["week"].max()
    df = df[(df["week"] > first_week) & (df["week"] < last_week)].copy()
    all_weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="7D")

    weekly = df.groupby(["stock_code", "week"])["quantity"].sum().rename("units").reset_index()
    wide = weekly.pivot(index="week", columns="stock_code", values="units").reindex(all_weeks)
    wide.columns.name = "stock_code"

    # A week without sales is a real zero only between a product's first and last sale.
    # Before launch and after withdrawal the product was not on offer, so there is no demand to forecast.
    has_sale = wide.notna()
    first_sale = has_sale.idxmax()
    last_sale = has_sale.iloc[::-1].idxmax()
    weeks = wide.index.values[:, None]
    active = (weeks >= first_sale.values[None, :]) & (weeks <= last_sale.values[None, :])
    units = wide.fillna(0).where(active)

    long = (units.reset_index(names="week")
                 .melt(id_vars="week", var_name="stock_code", value_name="units")
                 .dropna(subset=["units"]))
    long["units"] = long["units"].astype(int)
    long = long.sort_values(["stock_code", "week"]).reset_index(drop=True)

    products = pd.DataFrame({
        "first_week": first_sale,
        "last_week": last_sale,
        "active_weeks": units.notna().sum(),
        "selling_weeks": (units > 0).sum(),
        "total_units": units.sum(),
        "total_revenue": df.groupby("stock_code")["revenue"].sum(),
        "mean_price": df.groupby("stock_code")["price"].mean(),
        "description": df.groupby("stock_code")["description"].agg(
            lambda s: s.mode().iloc[0] if s.notna().any() else ""),
    }).reset_index(names="stock_code")

    long.to_parquet(WEEKLY_PATH)
    products.to_parquet(PRODUCTS_PATH)

    print("complete weeks used:", len(all_weeks), "| from", all_weeks[0].date(), "to", all_weeks[-1].date())
    print("products:", len(products))
    print("rows in weekly table:", len(long))
    print(f"share of product-weeks with zero sales: {(long['units'] == 0).mean():.1%}")
    print("Saved:", WEEKLY_PATH, "and", PRODUCTS_PATH)


if __name__ == "__main__":
    main()
