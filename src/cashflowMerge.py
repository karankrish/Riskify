import pandas as pd
import os

def combine_industry_cashflow_files():
    # 🔧 User inputs
    folder_path = input("📁 Enter path to the folder containing industry CSV files: ").strip()
    output_file = input("💾 Enter name for the final combined CSV (with .csv extension): ").strip()

    # 🛡 Validate inputs
    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path. Please check and try again.")
        return

    if not output_file.endswith('.csv'):
        print("❌ Output file name must end with .csv")
        return

    # 📆 Define expected years
    year_cols_ordered = [str(y) for y in range(2012, 2026)]

    # 📦 List for all files
    all_dataframes = []

    print("\n🔄 Processing files...\n")

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)

                if not {'Company', 'Financial_Metric'}.issubset(df.columns):
                    print(f"❌ Skipping {filename}: Missing 'Company' or 'Financial_Metric'")
                    continue

                # ➕ Add Industry from filename
                df['Industry'] = filename.split('_')[0]

                # 🎯 Extract year columns that are present
                available_years = [y for y in year_cols_ordered if y in df.columns]
                has_heavy = 'Heavy_Investment_Years' in df.columns

                # 📐 Column order
                final_cols = (
                    ['Industry', 'Company', 'Financial_Metric'] +
                    available_years +
                    (['Heavy_Investment_Years'] if has_heavy else [])
                )

                df = df[final_cols]
                all_dataframes.append(df)

                print(f"✅ Processed: {filename}")

            except Exception as e:
                print(f"⚠ Error reading {filename}: {e}")

    # 📚 Combine and Save
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # 🔁 Add any missing year columns
        for year in year_cols_ordered:
            if year not in combined_df.columns:
                combined_df[year] = pd.NA

        final_order = (
            ['Industry', 'Company', 'Financial_Metric'] +
            year_cols_ordered +
            (['Heavy_Investment_Years'] if 'Heavy_Investment_Years' in combined_df.columns else [])
        )
        combined_df = combined_df[final_order]

        output_path = os.path.join(folder_path, output_file)
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Done! File saved as: {output_path}")

    else:
        print("❌ No valid CSV files found in the folder.")

if __name__ == "__main__":
    combine_industry_cashflow_files()
