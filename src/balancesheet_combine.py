import pandas as pd
import os

def combine_company_balance_sheets():
    folder_path = input("📁 Enter the folder path containing company CSV files: ").strip()
    output_file = input("💾 Enter the full path to save combined Excel (e.g., C:\\...\\Combined_Balance_Sheet.xlsx): ").strip()

    if not os.path.isdir(folder_path):
        print("❌ The folder path is invalid.")
        return
    if not output_file.endswith('.xlsx'):
        print("❌ The output file must have a .xlsx extension.")
        return

    combined_data = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            company_name = os.path.splitext(filename)[0]

            try:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns

                # ➕ Clean and rename first column
                first_col = df.columns[0]
                if df[first_col].dtype == object:
                    df[first_col] = df[first_col].astype(str).str.replace('+', '', regex=False)
                df.rename(columns={first_col: 'Financial_Metric'}, inplace=True)

                # 🧱 Add Company name column
                df.insert(loc=1, column='Company', value=company_name)

                # 🧠 Normalize columns like 'Mar-24' or '2022' to 'YYYY'
                col_map = {}
                for col in df.columns:
                    if col not in ['Financial_Metric', 'Company']:
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

                # 🧹 Keep only relevant columns (2013 onwards)
                keep_cols = ['Financial_Metric', 'Company']
                keep_cols += [col for col in df.columns if col not in keep_cols and col.isdigit() and int(col) >= 2013]
                df = df[keep_cols]

                # 🎯 Combine duplicate year columns by averaging
                fixed_cols = ['Financial_Metric', 'Company']
                year_cols = [col for col in df.columns if col not in fixed_cols]
                if year_cols:
                    df_years = df[year_cols].groupby(axis=1, level=0).mean(numeric_only=True)
                    df = pd.concat([df[fixed_cols], df_years], axis=1)

                combined_data.append(df)
                print(f"✅ Processed: {company_name}")

            except Exception as e:
                print(f"⚠ Error processing {filename}: {e}")

    if not combined_data:
        print("❌ No valid CSV files found in the folder.")
        return

    # 🧱 Combine all data
    final_df = pd.concat(combined_data, ignore_index=True)

    # 🧮 Interpolate missing values
    fixed_columns = ['Financial_Metric', 'Company']
    year_columns = [col for col in final_df.columns if col not in fixed_columns and col.isdigit()]
    year_columns_int = [int(col) for col in year_columns]
    year_data = final_df[year_columns].copy()
    year_data.columns = year_columns_int

    year_data_interp = (
        year_data.T
        .interpolate(method='linear', limit_direction='both')
        .T
    )
    year_data_interp.columns = year_columns
    final_df[year_columns] = year_data_interp

    # 📊 Add summary (Average/Median) rows
    main_categories = ['Total Assets', 'Total Liabilities', 'Equity Capital', 'Reserves', 'Borrowings', 'Other Liabilities', 'Fixed Assets', 'CWIP', 'Investments', 'Other Assets']

    average_rows, median_rows = [], []

    for category in main_categories:
        cat_df = final_df[final_df['Financial_Metric'].str.contains(category, case=False, na=False)]
        if not cat_df.empty:
            numeric_data = cat_df[year_columns]

            avg_row = numeric_data.mean()
            avg_series = pd.Series({col: pd.NA for col in final_df.columns})
            avg_series['Financial_Metric'] = f"Average - {category}"
            avg_series['Company'] = 'avg'
            for col in avg_row.index:
                avg_series[col] = avg_row[col]
            average_rows.append(avg_series)

            med_row = numeric_data.median()
            med_series = pd.Series({col: pd.NA for col in final_df.columns})
            med_series['Financial_Metric'] = f"Median - {category}"
            med_series['Company'] = 'median'
            for col in med_row.index:
                med_series[col] = med_row[col]
            median_rows.append(med_series)

    summary_df = pd.DataFrame(average_rows + median_rows)
    final_df = pd.concat([final_df, summary_df], ignore_index=True)

    # 🔢 Sort year columns numerically
    final_df = final_df[fixed_columns + sorted(year_columns, key=int)]

    # 💾 Save to Excel
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name="Combined_Balance_Sheet", index=False)

    print(f"\n✅ All Balance Sheet data from 2013 onwards combined and saved to:\n{output_file}")

# Run
if __name__ == "__main__":
    combine_company_balance_sheets()
