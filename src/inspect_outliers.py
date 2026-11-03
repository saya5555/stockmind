# FILE: src/inspect_outliers.py
import pandas as pd

RAW_PATH = "data/retail.parquet"
TOP_N = 10


def main():
    raw = pd.read_parquet(RAW_PATH)
    is_cancel = raw["invoice"].str.startswith("C")
    sales = raw[~is_cancel & (raw["quantity"] > 0)]
    cancels = raw[is_cancel]

    results = []
    for _, row in sales.nlargest(TOP_N, "quantity").iterrows():
        # A cancellation "matches" when it has the same customer, product and exact opposite quantity, later in time.
        # Rows without a customer_id can never match (missing values are not equal to each other).
        match = cancels[
            (cancels["stock_code"] == row["stock_code"])
            & (cancels["customer_id"] == row["customer_id"])
            & (cancels["quantity"] == -row["quantity"])
            & (cancels["invoice_date"] >= row["invoice_date"])
        ].sort_values("invoice_date")
        found = len(match) > 0
        results.append({
            "invoice": row["invoice"],
            "stock_code": row["stock_code"],
            "customer_id": row["customer_id"],
            "date": row["invoice_date"],
            "quantity": row["quantity"],
            "fully_cancelled": found,
            "cancel_invoice": match["invoice"].iloc[0] if found else "",
            "hours_to_cancel": round((match["invoice_date"].iloc[0] - row["invoice_date"]).total_seconds() / 3600, 1) if found else None,
        })

    out = pd.DataFrame(results)
    print(out.to_string(index=False))
    print(f"\n{int(out['fully_cancelled'].sum())} of the top {TOP_N} largest sale lines were fully cancelled later.")


if __name__ == "__main__":
    main()
