import pandas as pd
import numpy as np
import os
import re

def combine_company_balance_sheets():
    folder_path = input("📁 Enter the folder path containing industry folders: ").strip()
    output_file = input("💾 Enter the full path to save combined Excel (e.g., C:\\...\\Combined_Balance_Sheet.xlsx): ").strip()

    if not os.path.isdir(folder_path):
        print("❌ The folder path is invalid.")
        return
    if not output_file.endswith('.xlsx'):
        print("❌ The output file must have a .xlsx extension.")
        return

    combined_data = []

    for industry_folder in os.listdir(folder_path):
        industry_path = os.path.join(folder_path, industry_folder)
        if os.path.isdir(industry_path):
            for filename in os.listdir(industry_path):
                if filename.lower().endswith(".csv"):
                    file_path = os.path.join(industry_path, filename)
                    company_name = os.path.splitext(filename)[0]
                    clean_company = re.sub(r'[_\s]?balance[_\s]?sheet$', '', company_name, flags=re.IGNORECASE).strip()

                    try:
                        df = pd.read_csv(file_path)
                        df = df.loc[:, ~df.columns.duplicated()]

                        first_col = df.columns[0]
                        if df[first_col].dtype == object:
                            df[first_col] = df[first_col].astype(str).str.replace('+', '', regex=False)
                        df.rename(columns={first_col: 'Financial_Metric'}, inplace=True)

                        df.insert(loc=0, column='Industry', value=industry_folder)
                        df.insert(loc=1, column='Company', value=clean_company)

                        col_map = {}
                        for col in df.columns:
                            if col not in ['Financial_Metric', 'Company', 'Industry']:
                                try:
                                    year = pd.to_datetime(col, format='%b-%y', errors='coerce').year
                                    if pd.notna(year) and year >= 2013:
                                        col_map[col] = str(year)
                                    else:
                                        year = pd.to_datetime(col, errors='coerce').year
                                        if pd.notna(year) and year >= 2013:
                                            col_map[col] = str(year)
                                except:
                                    pass
                        df.rename(columns=col_map, inplace=True)

                        keep_cols = ['Industry', 'Company', 'Financial_Metric']
                        keep_cols += [col for col in df.columns if col not in keep_cols and col.isdigit() and int(col) >= 2013]
                        df = df[keep_cols]

                        fixed_cols = ['Industry', 'Company', 'Financial_Metric']
                        year_cols = [col for col in df.columns if col not in fixed_cols]
                        if year_cols:
                            df_years = df[year_cols].groupby(axis=1, level=0).mean(numeric_only=True)
                            df = pd.concat([df[fixed_cols], df_years], axis=1)

                        combined_data.append(df)
                        print(f"✅ Processed: {industry_folder}/{clean_company}")

                    except Exception as e:
                        print(f"⚠ Error processing {filename}: {e}")

    if not combined_data:
        print("❌ No valid CSV files found in the folder.")
        return

    final_df = pd.concat(combined_data, ignore_index=True)

    fixed_columns = ['Industry', 'Company', 'Financial_Metric']
    year_columns = [col for col in final_df.columns if col not in fixed_columns and col.isdigit()]
    year_columns_int = [int(col) for col in year_columns]
    year_data = final_df[year_columns].copy()
    year_data.columns = year_columns_int

    year_data_interp = year_data.T.interpolate(method='linear', limit_direction='both').T
    year_data_interp.columns = year_columns
    final_df[year_columns] = year_data_interp
    final_df[year_columns] = final_df[year_columns].fillna(0)

    # Summary rows
    main_categories = ['Total Assets', 'Total Liabilities', 'Equity Capital', 'Reserves',
                       'Borrowings', 'Other Liabilities', 'Fixed Assets', 'CWIP', 'Investments', 'Other Assets']
    avg_rows, med_rows = [], []

    for industry in final_df['Industry'].unique():
        industry_df = final_df[final_df['Industry'] == industry]
        for category in main_categories:
            cat_df = industry_df[industry_df['Financial_Metric'].str.contains(category, case=False, na=False)]
            if not cat_df.empty:
                avg_row = cat_df[year_columns].mean()
                med_row = cat_df[year_columns].median()

                avg_series = pd.Series({col: pd.NA for col in final_df.columns})
                avg_series['Industry'] = industry
                avg_series['Company'] = 'avg'
                avg_series['Financial_Metric'] = category
                avg_series.update(avg_row)
                avg_rows.append(avg_series)

                med_series = pd.Series({col: pd.NA for col in final_df.columns})
                med_series['Industry'] = industry
                med_series['Company'] = 'median'
                med_series['Financial_Metric'] = category
                med_series.update(med_row)
                med_rows.append(med_series)

    summary_df = pd.DataFrame(avg_rows + med_rows)
    final_df = pd.concat([final_df, summary_df], ignore_index=True)

    # Derived Metrics
    company_derived_rows = []

    for industry in final_df['Industry'].unique():
        industry_df = final_df[final_df['Industry'] == industry]
        for company in industry_df['Company'].unique():
            if company in ['avg', 'median']:
                continue
            company_df = industry_df[industry_df['Company'] == company]
            equity = company_df[company_df['Financial_Metric'].str.fullmatch('Equity Capital', case=False, na=False)]
            reserves = company_df[company_df['Financial_Metric'].str.fullmatch('Reserves', case=False, na=False)]
            borrowings = company_df[company_df['Financial_Metric'].str.fullmatch('Borrowings', case=False, na=False)]

            if not equity.empty and not reserves.empty and not borrowings.empty:
                equity_series = equity[year_columns].iloc[0]
                reserves_series = reserves[year_columns].iloc[0]
                borrowings_series = borrowings[year_columns].iloc[0]

                networth_series = equity_series + reserves_series
                debt_equity_series = borrowings_series / networth_series.replace(0, np.nan)

                for label, values in {
                    'Net Worth': networth_series,
                    'Deabt/Equity Ratio': debt_equity_series
                }.items():
                    row = {col: pd.NA for col in final_df.columns}
                    row['Industry'] = industry
                    row['Company'] = company
                    row['Financial_Metric'] = label
                    for col in year_columns:
                        row[col] = values.get(col, pd.NA)
                    company_derived_rows.append(row)

    company_metric_df = pd.DataFrame(company_derived_rows)
    final_df = pd.concat([final_df, company_metric_df], ignore_index=True)

    # High Leverage & Low Equity Years
    networth_df = final_df[final_df['Financial_Metric'].str.fullmatch('Net Worth', case=False)].copy()
    de_ratio_df = final_df[final_df['Financial_Metric'].str.fullmatch('Deabt/Equity Ratio', case=False)]

    high_leverage_threshold = 2.0
    low_equity_threshold = 9000
    high_leverage_years = []
    low_equity_years = []

    for _, row in networth_df.iterrows():
        company = row['Company']
        industry = row['Industry']
        de_row = de_ratio_df[(de_ratio_df['Company'] == company) & (de_ratio_df['Industry'] == industry)]
        high_years = []
        low_years = []

        for year in year_columns:
            try:
                networth_val = float(row[year])
                de_val = float(de_row[year].values[0]) if not de_row.empty else np.nan
                if networth_val < low_equity_threshold:
                    low_years.append(year)
                if de_val > high_leverage_threshold:
                    high_years.append(year)
            except:
                continue

        high_leverage_years.append(','.join(high_years))
        low_equity_years.append(','.join(low_years))

    networth_df['High_Leverage_Years'] = high_leverage_years
    networth_df['Low_Equity_Years'] = low_equity_years

    final_df = final_df[~final_df['Financial_Metric'].str.fullmatch('Net Worth', case=False)]
    final_df = pd.concat([final_df, networth_df], ignore_index=True)

    # Save final output
    final_df = final_df.sort_values(by=['Industry', 'Company', 'Financial_Metric'])
    final_df.to_excel(output_file, index=False)
    print(f"📁 Combined data saved to: {output_file}")

# Run
if __name__ == "__main__":
    combine_company_balance_sheets()
