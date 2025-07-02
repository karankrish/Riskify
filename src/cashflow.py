import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

def interpolate_missing_values(df, year_cols):
    for idx, row in df.iterrows():
        for year in year_cols:
            if year not in row.index:
                continue
            if pd.isna(row.loc[year]):
                known_years = [y for y in year_cols if y != year and y in row.index and not pd.isna(row.loc[y])]
                known_values = [row.loc[y] for y in known_years]
                if len(known_years) >= 3:
                    X = np.array(known_years).astype(int).reshape(-1, 1)
                    y_known = np.array(known_values).reshape(-1, 1)
                    poly = PolynomialFeatures(degree=2)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(X_poly, y_known)
                    pred = model.predict(poly.transform(np.array([[int(year)]])))[0, 0]
                    df.at[idx, year] = pred
    return df

def prepare_df(df, metric_name, year_cols):
    sub = df[df['Financial_Metric'] == metric_name].copy()
    sub = sub[['Company'] + year_cols]
    sub[year_cols] = sub[year_cols].apply(pd.to_numeric, errors='coerce')
    sub.set_index('Company', inplace=True)
    return sub

def add_avg_median_rows(df, metric_name, year_cols):
    avg = df[year_cols].mean().to_frame().T
    med = df[year_cols].median().to_frame().T

    avg.index = [metric_name]
    med.index = [metric_name]

    df.index.name = 'Company'
    avg.index.name = 'Company'
    med.index.name = 'Company'

    avg['Financial_Metric'] = f"{metric_name}-Average"
    med['Financial_Metric'] = f"{metric_name}-Median"
    df['Financial_Metric'] = metric_name

    return pd.concat([df, avg, med])

def format_metric(df, year_cols):
    df = df.copy().reset_index()
    return df[['Company', 'Financial_Metric'] + year_cols]

def main():
    # 📥 Dynamic input from user
    file_path = input("Enter the full path to the input Excel file (.xlsx): ").strip('"')
    sheet_name = input("Enter the Excel sheet name (e.g., Combined_Cashflow_new_1): ").strip()
    output_path = input("Enter the full path for output file (with .csv or .xlsx): ").strip('"')

    print("📥 Loading data...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]
    df['Company'] = df['Company'].str.strip()

    # ✅ Remove '_Cash_Flow', '__Cash_Flow', ' Cash Flow' etc.
    df['Company'] = df['Company'].str.replace(r'_?Cash[_ ]?Flow$', '', regex=True)

    year_cols = [col for col in df.columns if col.isdigit()]

    # 📊 Prepare metrics
    ocf = prepare_df(df, "Cash from Operating Activity", year_cols)
    capex = prepare_df(df, "Cash from Investing Activity", year_cols)
    financing = prepare_df(df, "Cash from Financing Activity", year_cols)
    net_cf = prepare_df(df, "Net Cash Flow", year_cols)

    for metric_df in [ocf, capex, financing, net_cf]:
        interpolate_missing_values(metric_df, year_cols)

    # ➕ Derived metrics
    fcf = ocf - capex
    fcf_rank = fcf.rank(axis=1, method='min', ascending=False)
    heavy_flag = (capex.lt(capex.quantile(0.25))).astype(int)

    heavy_years_dict = {
        company: ", ".join([year for year in year_cols if heavy_flag.loc[company, year] == 1]) or "None"
        for company in heavy_flag.index
    }

    # 📐 Metrics to export
    metrics = {
        "OCF": ocf,
        "CAPEX": capex,
        "FCF": fcf,
        "FCF_Rank": fcf_rank,
        "Financing_CF": financing,
        "Net_Cash_Flow": net_cf,
        "Heavy_Investment_Flag": heavy_flag,
        "Cash from Operating Activity": ocf,
        "Cash from Investing Activity": capex,
        "Cash from Financing Activity": financing,
        "Net Cash Flow": net_cf
    }

    final_df = []
    for metric_name, data in metrics.items():
        data_with_stats = add_avg_median_rows(data, metric_name, year_cols)
        formatted = format_metric(data_with_stats, year_cols)
        final_df.append(formatted)

    combined_df = pd.concat(final_df, ignore_index=True)
    combined_df['Heavy_Investment_Years'] = combined_df['Company'].map(heavy_years_dict)
    combined_df['Heavy_Investment_Years'].fillna('', inplace=True)

    print("💾 Saving output...")

    # Save based on extension
    if output_path.endswith(".csv"):
        combined_df.to_csv(output_path, index=False)
    elif output_path.endswith(".xlsx"):
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            combined_df.to_excel(writer, sheet_name="Cashflow_Analysis", index=False)
    else:
        print("❌ Unsupported file format. Use .csv or .xlsx")
        return

    print("✅ Done! File saved to:")
    print(output_path)

if __name__ == "__main__":
    main()
