# FILE: src/clean_data.py
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/retail.parquet")
OUT_PATH = Path("data/retail_clean.parquet")
LOG_PATH = Path("reports/cleaning_log.csv")

# A real product code is 5 digits, optionally followed by letters (e.g. 85123A, 15056BL).
# Codes such as POST, DOT, M, BANK CHARGES or gift_0001_20 are fees or vouchers, not products.
PRODUCT_CODE_PATTERN = r"^\d{5}[A-Za-z]*$"

# A sale that is cancelled within this many hours is treated as an order mistake, not real demand.
QUICK_CANCEL_HOURS = 24


def apply_rule(frame, remove_mask, rule_name, log):
    """Drop the rows selected by remove_mask and record how many were removed."""
    removed = int(remove_mask.sum())
    log.append({"rule": rule_name, "rows_removed": removed})
    return frame.loc[~remove_mask]


def find_quickly_cancelled_sales(df, max_hours):
    """Return the index labels of sale lines that were fully cancelled within max_hours.

    A sale matches a cancellation when customer, product and quantity are the same
    (the cancellation has the opposite sign) and the cancellation comes later but within max_hours.
    Lines without a customer_id cannot be matched.
    """
    has_customer = df["customer_id"].notna()
    is_cancel = df["invoice"].str.startswith("C")
    columns = ["customer_id", "stock_code", "quantity", "invoice_date"]

    sales = df.loc[has_customer & ~is_cancel & (df["quantity"] > 0), columns].copy()
    sales["sale_index"] = sales.index
    cancels = df.loc[has_customer & is_cancel & (df["quantity"] < 0), columns].copy()
    cancels["quantity"] = -cancels["quantity"]

    matched = pd.merge_asof(
        cancels.sort_values("invoice_date"),
        sales.sort_values("invoice_date"),
        on="invoice_date",
        by=["customer_id", "stock_code", "quantity"],
        direction="backward",
        tolerance=pd.Timedelta(hours=max_hours),
    )
    return pd.Index(matched["sale_index"].dropna().astype(int).unique())


def main():
    df = pd.read_parquet(IN_PATH)
    log = [{"rule": "start (raw rows)", "rows_removed": 0}]
    rows_raw = len(df)

    # This rule must run first, because it needs the cancellation rows that the next rule removes.
    quick = find_quickly_cancelled_sales(df, QUICK_CANCEL_HOURS)
    quick_mask = df.index.isin(quick)
    units_quick = int(df.loc[quick_mask, "quantity"].sum())
    df = apply_rule(df, quick_mask, f"sales fully cancelled within {QUICK_CANCEL_HOURS} hours", log)

    df = apply_rule(df, df["invoice"].str.startswith("C"), "cancelled invoices (start with C)", log)
    df = apply_rule(df, df["quantity"] <= 0, "quantity <= 0", log)
    df = apply_rule(df, df["price"] <= 0, "price <= 0", log)
    df = apply_rule(df, ~df["stock_code"].str.match(PRODUCT_CODE_PATTERN), "non-product stock codes", log)
    df = apply_rule(df, df.duplicated(), "exact duplicate rows", log)

    df = df.copy()
    df["revenue"] = df["quantity"] * df["price"]
    df = df.reset_index(drop=True)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log)
    log_df.to_csv(LOG_PATH, index=False)
    df.to_parquet(OUT_PATH)

    print(log_df.to_string(index=False))
    print(f"\nunits removed by the quick-cancellation rule: {units_quick}")
    print(f"rows: {rows_raw} -> {len(df)} ({len(df) / rows_raw:.1%} kept)")
    print("date range:", df["invoice_date"].min(), "to", df["invoice_date"].max())
    print("unique products:", df["stock_code"].nunique())
    print("rows with missing customer_id (kept on purpose):", int(df["customer_id"].isna().sum()))
    print("\nlargest 5 quantities in one line (outlier check):")
    print(df.nlargest(5, "quantity")[["invoice", "stock_code", "quantity", "price"]].to_string(index=False))
    print("\nSaved:", OUT_PATH, "and", LOG_PATH)


if __name__ == "__main__":
    main()
