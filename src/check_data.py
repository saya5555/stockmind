# FILE: src/check_data.py
import pandas as pd

df = pd.read_parquet("data/retail.parquet")

print("rows:", len(df))
print("date range:", df["invoice_date"].min(), "to", df["invoice_date"].max())
print("\nmissing share per column:")
print(df.isna().mean().round(3))
print("\ncancelled invoices (start with C):", df["invoice"].str.startswith("C").sum())
print("negative quantity rows:", (df["quantity"] < 0).sum())
print("zero or negative price rows:", (df["price"] <= 0).sum())
print("unique products:", df["stock_code"].nunique())
print("countries:", df["country"].nunique())
