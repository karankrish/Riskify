import pandas as pd
import numpy as np

def prepare_all_metrics(df, year_cols):
    """
    Returns a dictionary where keys are financial metrics and values are DataFrames with companies as index.
    """
    metrics = df["Financial_Metric"].unique()
    metric_data = {}

    for metric in metrics:
        metric_df = df[df["Financial_Metric"] == metric].copy()
        metric_df = metric_df[["Company"] + year_cols].drop_duplicates(subset="Company")
        metric_data[metric] = metric_df.set_index("Company")

    return metric_data

def calculate_derived_metrics(metric_data, year_cols):
    equity = metric_data.get("Equity Capital")
    reserves = metric_data.get("Reserves")
    borrowings = metric_data.get("Borrowings")

    # Check presence
    if equity is None or reserves is None or borrowings is None:
        raise ValueError("Missing Equity/Reserves/Borrowings metrics.")

    # Align indices
    common_companies = equity.index.intersection(reserves.index).intersection(borrowings.index)
    equity = equity.loc[common_companies]
    reserves = reserves.loc[common_companies]
    borrowings = borrowings.loc[common_companies]

    # Calculate net worth and debt-equity ratio
    net_worth = equity + reserves
    debt_equity = borrowings / net_worth
    debt_equity = debt_equity.replace([np.inf, -np.inf], np.nan)

    # Debug print
    print("✅ Checking Debt/Equity presence:")
    print(debt_equity.head())

    # High leverage: D/E > 2
    high_leverage_flags = (debt_equity > 2).astype(int)
    high_leverage_years = {
        company: ', '.join([str(year) for year in year_cols if high_leverage_flags.loc[company, year] == 1]) or 'None'
        for company in high_leverage_flags.index
    }

    # Low Equity flag
    low_equity_flags = (equity < 50).astype(int)
    low_equity_years = {
        company: ', '.join([str(year) for year in year_cols if low_equity_flags.loc[company, year] == 1]) or 'None'
        for company in equity.index
    }

    return net_worth, debt_equity, high_leverage_years, low_equity_years

def main():
    file_path = input("📁 Enter path to Balance Sheet Excel file: ").strip('"')
    sheet_name = input("📄 Enter sheet name (e.g., Combined_Balance_Sheet): ").strip()
    output_path = input("💾 Enter output file path (.csv or .xlsx): ").strip('"')

    print("📥 Loading data...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]
    df['Company'] = df['Company'].str.strip()
    year_cols = [col for col in df.columns if col.isdigit()]

    # Prepare all original metrics
    metric_data = prepare_all_metrics(df, year_cols)

    # Derived metrics
    net_worth, debt_equity, high_leverage_years, low_equity_years = calculate_derived_metrics(metric_data, year_cols)
    metric_data["Net Worth"] = net_worth
    metric_data["Debt/Equity Ratio"] = debt_equity

    # Combine all into output rows
    output_rows = []

    # Prioritize Net Worth and Debt/Equity Ratio first
    ordered_metric_names = ["Net Worth", "Debt/Equity Ratio"] + [
        m for m in metric_data if m not in {"Net Worth", "Debt/Equity Ratio"}
    ]

    for metric_name in ordered_metric_names:
        data = metric_data[metric_name]
        for company in data.index:
            row = {
                "Company": company,
                "Financial_Metric": metric_name,
                **{year: data.loc[company, year] for year in year_cols},
                "High_Leverage_Years": high_leverage_years.get(company, 'None') if metric_name == "Debt/Equity Ratio" else '',
                "Low_Equity_Years": low_equity_years.get(company, 'None') if metric_name == "Equity Capital" else ''
            }
            output_rows.append(row)

    final_df = pd.DataFrame(output_rows)

    # Save the final output
    print("💾 Saving to file...")
    if output_path.endswith('.csv'):
        final_df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        final_df.to_excel(output_path, index=False)
    else:
        print("❌ Unsupported file format. Use .csv or .xlsx only.")
        return

    print("✅ Done! File saved at:")
    print(output_path)

if __name__ == "__main__":
    main()
