import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

def interpolate_missing_values(df, year_cols):
    """Interpolate missing values using polynomial regression"""
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
    """Prepare dataframe for a specific financial metric"""
    sub = df[df['Financial_Metric'] == metric_name].copy()
    sub = sub[['Company'] + year_cols]
    sub[year_cols] = sub[year_cols].apply(pd.to_numeric, errors='coerce')
    sub.set_index('Company', inplace=True)
    return sub

def add_avg_median_rows(df, metric_name, year_cols):
    """Add average and median rows for each metric"""
    avg = df[year_cols].mean().to_frame().T
    med = df[year_cols].median().to_frame().T
    avg.index = [f"{metric_name}-Average"]
    med.index = [f"{metric_name}-Median"]
    df.index.name = 'Company'
    avg.index.name = 'Company'
    med.index.name = 'Company'
    avg['Financial_Metric'] = f"{metric_name}-Average"
    med['Financial_Metric'] = f"{metric_name}-Median"
    df['Financial_Metric'] = metric_name
    return pd.concat([df, avg, med])

def format_metric(df, year_cols):
    """Format the metric dataframe for final output"""
    df = df.copy().reset_index()
    return df[['Company', 'Financial_Metric'] + year_cols]

def calculate_growth_rates(df, year_cols):
    """Calculate year-over-year growth rates"""
    growth_df = df.copy()
    for i in range(1, len(year_cols)):
        prev_year = year_cols[i-1]
        curr_year = year_cols[i]
        growth_col = f"Growth_{prev_year}_{curr_year}"
        growth_df[growth_col] = ((df[curr_year] - df[prev_year]) / df[prev_year] * 100).round(2)
    return growth_df

def calculate_margins(sales_df, expense_df, operating_df, net_profit_df, year_cols):
    """Calculate various profit margins"""
    margins = {}
    
    # Gross Margin = (Sales - Expenses) / Sales * 100
    gross_margin = ((sales_df - expense_df) / sales_df * 100).round(2)
    margins['Gross_Margin_%'] = gross_margin
    
    # Operating Margin = Operating Profit / Sales * 100
    operating_margin = (operating_df / sales_df * 100).round(2)
    margins['Operating_Margin_%'] = operating_margin
    
    # Net Margin = Net Profit / Sales * 100
    net_margin = (net_profit_df / sales_df * 100).round(2)
    margins['Net_Margin_%'] = net_margin
    
    return margins

def identify_performance_flags(df, year_cols, threshold_percentile=25):
    """Identify companies with poor performance (bottom quartile)"""
    poor_performance = (df.lt(df.quantile(threshold_percentile/100))).astype(int)
    
    poor_years_dict = {
        company: ", ".join([year for year in year_cols if poor_performance.loc[company, year] == 1]) or "None"
        for company in poor_performance.index
    }
    
    return poor_performance, poor_years_dict

def main():
    # 📥 Dynamic input from user
    file_path = input("Enter the full path to the input Excel file (.xlsx): ").strip('"')
    sheet_name = input("Enter the Excel sheet name (e.g., PL_Analysis): ").strip()
    output_path = input("Enter the full path for output file (with .csv or .xlsx): ").strip('"')

    print("📥 Loading data...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]
    df['Company'] = df['Company'].str.strip()
    
    # Remove Average and Median rows from company data
    print("🧹 Removing Average and Median rows...")
    before_count = len(df)
    df = df[~df['Company'].str.contains('Average|Median', case=False, na=False)]
    after_count = len(df)
    removed_count = before_count - after_count
    if removed_count > 0:
        print(f"   Removed {removed_count} Average/Median rows")
    
    year_cols = [col for col in df.columns if col.isdigit()]

    print("📊 Preparing P&L metrics...")
    
    # Core P&L metrics
    sales = prepare_df(df, "Sales", year_cols)
    expenses = prepare_df(df, "Expenses", year_cols)
    operating = prepare_df(df, "Operating", year_cols)
    net_profit = prepare_df(df, "Net Profit", year_cols)
    
    # Additional metrics that might be present
    opm = prepare_df(df, "OPM", year_cols) if "OPM" in df['Financial_Metric'].values else None
    eps = prepare_df(df, "EPS in Rs", year_cols) if "EPS in Rs" in df['Financial_Metric'].values else None
    dividend = prepare_df(df, "Dividend P", year_cols) if "Dividend P" in df['Financial_Metric'].values else None

    # Interpolate missing values for core metrics
    core_metrics = [sales, expenses, operating, net_profit]
    if opm is not None:
        core_metrics.append(opm)
    if eps is not None:
        core_metrics.append(eps)
    if dividend is not None:
        core_metrics.append(dividend)
    
    for metric_df in core_metrics:
        interpolate_missing_values(metric_df, year_cols)

    print("🧮 Calculating derived metrics...")
    
    # Calculate margins
    margins = calculate_margins(sales, expenses, operating, net_profit, year_cols)
    
    # Calculate rankings
    sales_rank = sales.rank(axis=1, method='min', ascending=False)
    profit_rank = net_profit.rank(axis=1, method='min', ascending=False)
    
    # Calculate growth rates
    sales_growth = calculate_growth_rates(sales, year_cols)
    profit_growth = calculate_growth_rates(net_profit, year_cols)
    
    # Performance flags (companies with poor performance)
    poor_profit_flag, poor_profit_years = identify_performance_flags(net_profit, year_cols)
    poor_sales_flag, poor_sales_years = identify_performance_flags(sales, year_cols)

    print("📐 Organizing metrics for export...")
    
    # Metrics to export
    metrics = {
        "Sales": sales,
        "Expenses": expenses,
        "Operating_Profit": operating,
        "Net_Profit": net_profit,
        "Sales_Rank": sales_rank,
        "Profit_Rank": profit_rank,
        "Gross_Margin_%": margins['Gross_Margin_%'],
        "Operating_Margin_%": margins['Operating_Margin_%'],
        "Net_Margin_%": margins['Net_Margin_%'],
        "Poor_Profit_Performance_Flag": poor_profit_flag,
        "Poor_Sales_Performance_Flag": poor_sales_flag
    }
    
    # Add optional metrics if they exist
    if opm is not None:
        metrics["OPM"] = opm
    if eps is not None:
        metrics["EPS_in_Rs"] = eps
    if dividend is not None:
        metrics["Dividend_P"] = dividend

    # Prepare final dataframe
    final_df = []
    for metric_name, data in metrics.items():
        data_with_stats = add_avg_median_rows(data, metric_name, year_cols)
        formatted = format_metric(data_with_stats, year_cols)
        final_df.append(formatted)

    combined_df = pd.concat(final_df, ignore_index=True)
    
    # Add performance flags
    combined_df['Poor_Profit_Years'] = combined_df['Company'].map(poor_profit_years)
    combined_df['Poor_Sales_Years'] = combined_df['Company'].map(poor_sales_years)
    combined_df['Poor_Profit_Years'] = combined_df['Poor_Profit_Years'].fillna('')
    combined_df['Poor_Sales_Years'] = combined_df['Poor_Sales_Years'].fillna('')

    print("💾 Saving output...")

    # Save based on extension
    if output_path.endswith(".csv"):
        combined_df.to_csv(output_path, index=False)
        print("✅ CSV file saved!")
    elif output_path.endswith(".xlsx"):
        try:
            # Try xlsxwriter first (preferred)
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                # Main analysis sheet
                combined_df.to_excel(writer, sheet_name="PL_Analysis", index=False)
                
                # Growth analysis sheet
                sales_growth_formatted = format_metric(
                    add_avg_median_rows(sales_growth, "Sales_Growth_%", year_cols + [col for col in sales_growth.columns if col.startswith('Growth_')]), 
                    year_cols + [col for col in sales_growth.columns if col.startswith('Growth_')]
                )
                profit_growth_formatted = format_metric(
                    add_avg_median_rows(profit_growth, "Profit_Growth_%", year_cols + [col for col in profit_growth.columns if col.startswith('Growth_')]), 
                    year_cols + [col for col in profit_growth.columns if col.startswith('Growth_')]
                )
                
                growth_combined = pd.concat([sales_growth_formatted, profit_growth_formatted], ignore_index=True)
                growth_combined.to_excel(writer, sheet_name="Growth_Analysis", index=False)
            print("✅ Excel file with multiple sheets saved!")
            
        except ImportError:
            print("⚠️  xlsxwriter not found. Installing openpyxl as alternative...")
            try:
                # Fallback to openpyxl
                with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                    # Main analysis sheet
                    combined_df.to_excel(writer, sheet_name="PL_Analysis", index=False)
                    
                    # Growth analysis sheet
                    sales_growth_formatted = format_metric(
                        add_avg_median_rows(sales_growth, "Sales_Growth_%", year_cols + [col for col in sales_growth.columns if col.startswith('Growth_')]), 
                        year_cols + [col for col in sales_growth.columns if col.startswith('Growth_')]
                    )
                    profit_growth_formatted = format_metric(
                        add_avg_median_rows(profit_growth, "Profit_Growth_%", year_cols + [col for col in profit_growth.columns if col.startswith('Growth_')]), 
                        year_cols + [col for col in profit_growth.columns if col.startswith('Growth_')]
                    )
                    
                    growth_combined = pd.concat([sales_growth_formatted, profit_growth_formatted], ignore_index=True)
                    growth_combined.to_excel(writer, sheet_name="Growth_Analysis", index=False)
                print("✅ Excel file saved using openpyxl!")
                
            except ImportError:
                print("❌ No Excel engine available. Please install xlsxwriter or openpyxl:")
                print("   pip install xlsxwriter")
                print("   or")
                print("   pip install openpyxl")
                print("Saving as CSV instead...")
                csv_path = output_path.replace('.xlsx', '.csv')
                combined_df.to_csv(csv_path, index=False)
                print(f"✅ CSV file saved to: {csv_path}")
    else:
        print("❌ Unsupported file format. Use .csv or .xlsx")
        return

    print(f"✅ Analysis complete! File saved to: {output_path}")
    print("\n📊 Summary of analysis:")
    print(f"• Companies analyzed: {len(sales.index)}")
    print(f"• Years covered: {len(year_cols)} ({min(year_cols)} - {max(year_cols)})")
    print(f"• Metrics calculated: {len(metrics)}")
    print("• Included margin analysis, rankings, and performance flags")

if __name__ == "__main__":
    main()