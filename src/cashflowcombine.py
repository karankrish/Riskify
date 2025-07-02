import pandas as pd
import os

def combine_company_cashflows():
    # 🔧 User inputs
    folder_path = input("📁 Enter the folder path containing company CSV files: ").strip()
    output_file = input("💾 Enter the full path and filename to save the combined Excel (e.g., C:\\...\\Combined_Cashflow.xlsx): ").strip()

    if not os.path.isdir(folder_path):
        print("❌ The folder path is invalid.")
        return
    if not output_file.endswith('.xlsx'):
        print("❌ The output file must have a .xlsx extension.")
        return

    combined_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            company_name = filename.replace(".csv", "")
            try:
                df = pd.read_csv(file_path)

                # ➕ Clean first column
                first_col = df.columns[0]
                if df[first_col].dtype == object:
                    df[first_col] = df[first_col].astype(str).str.replace('+', '', regex=False)

                df.rename(columns={first_col: 'Financial_Metric'}, inplace=True)
                df.insert(loc=1, column='Company', value=company_name)

                # 🔁 Normalize year-like columns
                new_columns = []
                for col in df.columns:
                    if col not in ['Financial_Metric', 'Company']:
                        try:
                            year = pd.to_datetime(col, errors='coerce').year
                            new_columns.append(str(int(year)) if pd.notna(year) else col)
                        except:
                            new_columns.append(col)
                    else:
                        new_columns.append(col)
                df.columns = new_columns

                # 🔁 Handle duplicate year columns by averaging
                if len(set(df.columns)) != len(df.columns):
                    numeric_cols = df.columns.difference(['Financial_Metric', 'Company'])
                    df = df.groupby(['Financial_Metric', 'Company'], as_index=False).agg(
                        lambda x: x.mean() if x.name in numeric_cols else x.iloc[0]
                    )

                combined_data.append(df)
                print(f"✅ Processed: {filename}")

            except Exception as e:
                print(f"⚠ Error processing {filename}: {e}")

    if not combined_data:
        print("❌ No valid CSV files found in the folder.")
        return

    # 🧱 Combine all
    final_df = pd.concat(combined_data, ignore_index=True)

    # 📊 Metrics for summary
    main_categories = [
        'Cash from Operating Activity',
        'Cash from Investing Activity',
        'Cash from Financing Activity',
        'Net Cash Flow'
    ]

    average_rows = []
    median_rows = []

    for category in main_categories:
        cat_df = final_df[final_df['Financial_Metric'].str.contains(category, case=False, na=False)]
        if not cat_df.empty:
            numeric_data = cat_df.select_dtypes(include='number')

            # ➕ Average
            avg_row = numeric_data.mean()
            avg_series = pd.Series({col: pd.NA for col in final_df.columns})
            avg_series['Financial_Metric'] = f"Average - {category}"
            avg_series['Company'] = 'avg'
            for col in avg_row.index:
                avg_series[col] = avg_row[col]
            average_rows.append(avg_series)

            # ➕ Median
            med_row = numeric_data.median()
            med_series = pd.Series({col: pd.NA for col in final_df.columns})
            med_series['Financial_Metric'] = f"Median - {category}"
            med_series['Company'] = 'median'
            for col in med_row.index:
                med_series[col] = med_row[col]
            median_rows.append(med_series)

    # 🔚 Append summary rows
    summary_df = pd.DataFrame(average_rows + median_rows)
    final_df = pd.concat([final_df, summary_df], ignore_index=True)

    # 🔢 Sort year columns numerically
    fixed_columns = ['Financial_Metric', 'Company']
    year_columns = sorted([col for col in final_df.columns if col not in fixed_columns], key=lambda x: int(x) if x.isdigit() else x)
    final_df = final_df[fixed_columns + year_columns]

    # 💾 Save to Excel
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name="Combined_Cashflow", index=False)

    print(f"\n✅ All data combined and saved to:\n{output_file}")

# Run the script
if __name__ == "__main__":
    combine_company_cashflows()
