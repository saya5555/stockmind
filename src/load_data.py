# FILE: src/load_data.py
import pandas as pd
from pathlib import Path

RAW = Path("data/online_retail_II.xlsx")
OUT = Path("data/retail.parquet")

# Excel columns often mix numbers and text (for example, a number typed into a description).
# Parquet needs every value in a text column to be text, so we convert them all.
TEXT_COLUMNS = ["invoice", "stock_code", "description", "country"]


def to_clean_text(series):
    """Turn every value into text; missing values stay missing."""
    return series.map(lambda value: None if pd.isna(value) else str(value))


def main():
    # The Excel file has two sheets (2009-10 and 2010-11). Reading takes a few minutes, only once.
    sheets = pd.read_excel(RAW, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    df.columns = ["invoice", "stock_code", "description", "quantity",
                  "invoice_date", "price", "customer_id", "country"]

    for col in TEXT_COLUMNS:
        df[col] = to_clean_text(df[col])

    # Safety check: no non-text value may remain in a text column
    for col in TEXT_COLUMNS:
        bad = int(df[col].dropna().map(lambda value: not isinstance(value, str)).sum())
        print(f"non-text values left in '{col}': {bad}")

    df.to_parquet(OUT)
    print("Saved:", OUT, "| shape:", df.shape)


if __name__ == "__main__":
    main()
