import pandas as pd
import numpy as np
import os

def load_data(bs_path, pnl_path):
    bs_df = pd.read_excel(bs_path)
    pl_df = pd.read_excel(pnl_path)

    bs_df = bs_df[~bs_df['Company'].str.contains("Average|Median", case=False, na=False)]
    pl_df = pl_df[~pl_df['Company'].str.contains("Average|Median", case=False, na=False)]

    year_cols = [col for col in bs_df.columns if str(col).isdigit()]
    bs_df[year_cols] = bs_df[year_cols].apply(pd.to_numeric, errors='coerce')
    pl_df[year_cols] = pl_df[year_cols].apply(pd.to_numeric, errors='coerce')

    return bs_df, pl_df, year_cols

def interpolate_zero_as_nan(df, year_cols):
    def interpolate_series(x):
        x_cleaned = x.replace(0, np.nan)
        x_interp = x_cleaned.interpolate(method='polynomial', order=2, axis=0, limit_direction='both')
        return x_interp.fillna(0)

    return df.groupby(['Company', 'Financial_Metric'])[year_cols].transform(interpolate_series)

def get_metric(df, metric, company, year_cols):
    row = df[(df['Company'] == company) & (df['Financial_Metric'] == metric)]
    return row[year_cols].values.flatten() if not row.empty else np.full(len(year_cols), np.nan)

def safe_div(a, b):
    return np.where((b == 0) & (a != 0), np.inf,
                    np.where(np.isnan(b), np.nan, a / b))

def calculate_ratios(bs_df, pl_df, year_cols):
    new_rows = []
    companies = bs_df['Company'].unique()

    for company in companies:
        code = company[:3].upper()
        pl_company = company.replace("Balance_Sheet", "Profit_&_Loss")

        equity_capital = get_metric(bs_df, 'Equity Capital', company, year_cols)
        reserves = get_metric(bs_df, 'Reserves', company, year_cols)
        equity = equity_capital + reserves
        total_liabilities = get_metric(bs_df, 'Total Liabilities', company, year_cols)
        total_assets = get_metric(bs_df, 'Total Assets', company, year_cols)
        borrowings = get_metric(bs_df, 'Borrowings', company, year_cols)
        other_liab = get_metric(bs_df, 'Other Liabilities', company, year_cols)
        current_liab = total_liabilities - borrowings - other_liab
        capital_employed = total_assets - current_liab

        sales = get_metric(pl_df, 'Sales', pl_company, year_cols)
        op_profit = get_metric(pl_df, 'Operating Profit', pl_company, year_cols)
        interest = get_metric(pl_df, 'Interest', pl_company, year_cols)

        de_ratio = safe_div(total_liabilities, equity)
        icr = safe_div(op_profit, interest)
        atr = safe_div(sales, total_assets)
        lt_borrow_pct = safe_div(borrowings, borrowings + other_liab) * 100
        st_borrow_pct = safe_div(other_liab, borrowings + other_liab) * 100

        def add_row(metric_name, values):
            new_rows.append(
                pd.Series([f"{metric_name} - {code}", company, *values], index=['Financial_Metric', 'Company'] + year_cols)
            )

        add_row("Debt/Equity Ratio", de_ratio)
        add_row("Interest Coverage Ratio", icr)
        add_row("Asset Turnover Ratio", atr)
        add_row("Capital Employed", capital_employed)
        add_row("LT Borrowing %", lt_borrow_pct)
        add_row("ST Borrowing %", st_borrow_pct)

    return pd.DataFrame(new_rows)

def main():
    bs_path = r"C:\\Users\\ASUS\\Desktop\\College_Last_sem_pro\\git\\Riskify\\Data\\processed\\Combined_Balance_With_Average_Median_Yearwise.xlsx"
    pnl_path = r"C:\\Users\\ASUS\\Desktop\\College_Last_sem_pro\\git\\Riskify\\Data\\processed\\Combined_PnL_Yearwise.xlsx"
    output_path = r"C:\\Users\\ASUS\\Downloads\\risk\\output.csv"

    bs_df, pl_df, year_cols = load_data(bs_path, pnl_path)
    bs_df[year_cols] = interpolate_zero_as_nan(bs_df, year_cols)
    pl_df[year_cols] = interpolate_zero_as_nan(pl_df, year_cols)

    metrics_df = calculate_ratios(bs_df, pl_df, year_cols)
    final_df = pd.concat([bs_df, metrics_df], ignore_index=True)

    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Final dataset with ratios added and saved to: {output_path}")

if __name__ == "__main__":
    main()
