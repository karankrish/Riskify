import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def extract_year_from_column(col_name):
    """Extract year from column names like 'Mar 2014', 'Dec 2012', 'Jun 2012' etc."""
    if isinstance(col_name, str):
        # Look for 4-digit year in the column name
        year_match = re.search(r'\b(20\d{2})\b', col_name)
        if year_match:
            return int(year_match.group(1))
    return None

def polynomial_interpolation(series, degree=2):
    """Fill missing values using polynomial interpolation"""
    if series.isna().all():
        return series
    
    # Get non-null values and their indices
    non_null_mask = ~series.isna()
    if non_null_mask.sum() < degree + 1:
        # Not enough points for polynomial interpolation, use linear
        return series.interpolate(method='linear')
    
    x = np.arange(len(series))[non_null_mask]
    y = series[non_null_mask].values
    
    try:
        # Fit polynomial
        poly_coeffs = np.polyfit(x, y, degree)
        poly_func = np.poly1d(poly_coeffs)
        
        # Fill missing values
        filled_series = series.copy()
        missing_indices = np.arange(len(series))[series.isna()]
        filled_series.iloc[missing_indices] = poly_func(missing_indices)
        
        return filled_series
    except:
        # Fallback to linear interpolation if polynomial fails
        return series.interpolate(method='linear')

def clean_metric_name(metric_name):
    """Clean financial metric names by removing special characters"""
    if not isinstance(metric_name, str):
        return str(metric_name)
    
    # Remove special characters like Â, +, and other unwanted characters
    cleaned = re.sub(r'[^\w\s%&-]', '', metric_name)
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    # Remove trailing/leading spaces
    cleaned = cleaned.strip()
    
    return cleaned

def read_excel_file(file_path):
    """Read Excel file and extract P&L data"""
    try:
        # Try reading different sheets that might contain P&L data
        possible_sheets = ['profit_&_loss', 'P&L', 'PL', 'Profit and Loss', 'Sheet1', 0]
        
        df = None
        for sheet in possible_sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet, header=0)
                break
            except:
                continue
        
        if df is None:
            # Try reading the first sheet
            df = pd.read_excel(file_path, sheet_name=0, header=0)
        
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def process_company_file(file_path, company_name):
    """Process individual company Excel file and extract P&L data"""
    df = read_excel_file(file_path)
    if df is None:
        return None
    
    # Find columns that contain years
    year_columns = []
    year_mapping = {}
    
    for col in df.columns:
        year = extract_year_from_column(str(col))
        if year:
            year_columns.append(col)
            year_mapping[col] = year
    
    if not year_columns:
        print(f"No year columns found in {file_path}")
        return None
    
    # Create a mapping of financial metrics
    pl_metrics = [
        'Sales', 'Revenue', 'Total Revenue', 'Net Sales',
        'Expenses', 'Total Expenses', 'Operating Expenses',
        'Operating', 'Operating Income', 'Operating Profit',
        'EBITDA', 'EBIT',
        'OPM', 'Operating Margin',
        'Other Income', 'Other Inco',
        'Interest', 'Interest Expense',
        'Depreciation', 'Depreciati',
        'Profit befo', 'Profit before Tax', 'PBT',
        'Tax', 'Income Tax',
        'Net Profit', 'PAT', 'Profit After Tax',
        'EPS', 'Earnings Per Share',
        'Dividend', 'Dividend P'
    ]
    
    # Extract P&L data
    pl_data = {}
    
    # Look for the financial metrics in the first column
    for idx, row in df.iterrows():
        metric_name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        
        # Check if this row contains a P&L metric
        for metric in pl_metrics:
            if metric.lower() in metric_name.lower():
                metric_values = {}
                
                # Extract values for each year
                for col in year_columns:
                    year = year_mapping[col]
                    try:
                        value = pd.to_numeric(row[col], errors='coerce')
                        metric_values[year] = value
                    except:
                        metric_values[year] = np.nan
                
                # Clean the metric name and use it
                cleaned_metric_name = clean_metric_name(metric_name)
                pl_data[cleaned_metric_name] = metric_values
                break
    
    if not pl_data:
        print(f"No P&L metrics found in {file_path}")
        return None
    
    # Convert to DataFrame
    result_df = pd.DataFrame(pl_data).T
    
    # Reset index to make Financial_Metric a column
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'Financial_Metric'}, inplace=True)
    
    # Add company column after Financial_Metric
    result_df.insert(1, 'Company', company_name)
    
    # Sort year columns in ascending order
    year_cols = [col for col in result_df.columns if isinstance(col, (int, float)) and col > 2000]
    other_cols = [col for col in result_df.columns if col not in year_cols]
    result_df = result_df[other_cols + sorted(year_cols)]
    
    return result_df

def process_sector(sector_path, sector_name):
    """Process all companies in a sector and combine their data"""
    print(f"Processing sector: {sector_name}")
    
    sector_data = []
    company_files = [f for f in os.listdir(sector_path) if f.endswith(('.xlsx', '.xls'))]
    
    for file_name in company_files:
        file_path = os.path.join(sector_path, file_name)
        company_name = os.path.splitext(file_name)[0]
        
        print(f"  Processing: {company_name}")
        company_data = process_company_file(file_path, company_name)
        
        if company_data is not None:
            sector_data.append(company_data)
    
    if not sector_data:
        print(f"No valid data found for sector: {sector_name}")
        return None
    
    # Combine all company data
    combined_df = pd.concat(sector_data, ignore_index=True)
    
    # Ensure Financial_Metric column exists and is properly named
    if 'Financial_Metric' not in combined_df.columns:
        # If somehow the column is missing, try to find it by other names
        possible_names = ['Financial_Metric', 'Metric', 'index', 'Financial Metric']
        for name in possible_names:
            if name in combined_df.columns:
                combined_df.rename(columns={name: 'Financial_Metric'}, inplace=True)
                break
        
        # If still not found, create it from index if it exists
        if 'Financial_Metric' not in combined_df.columns and combined_df.index.name:
            combined_df.reset_index(inplace=True)
            combined_df.rename(columns={combined_df.columns[0]: 'Financial_Metric'}, inplace=True)
    
    # Clean Financial_Metric names if they exist
    if 'Financial_Metric' in combined_df.columns:
        combined_df['Financial_Metric'] = combined_df['Financial_Metric'].apply(clean_metric_name)
    
    # Ensure proper column order: Financial_Metric, Company, then years in ascending order
    year_columns = [col for col in combined_df.columns if isinstance(col, (int, float)) and col > 2000]
    other_cols = ['Financial_Metric', 'Company']
    
    # Make sure we have all required columns
    final_columns = []
    for col in other_cols:
        if col in combined_df.columns:
            final_columns.append(col)
    
    # Add year columns in ascending order
    final_columns.extend(sorted(year_columns))
    
    # Reorder columns
    combined_df = combined_df[final_columns]
    
    # Apply polynomial interpolation to fill missing values
    print(f"  Applying polynomial interpolation for missing data...")
    
    # Group by financial metric and company, then interpolate
    year_columns = [col for col in combined_df.columns if isinstance(col, (int, float)) and col > 2000]
    
    for idx, row in combined_df.iterrows():
        # Extract numerical data for this row
        values = row[year_columns]
        
        # Apply polynomial interpolation
        filled_values = polynomial_interpolation(values, degree=2)
        
        # Update the dataframe
        combined_df.loc[idx, year_columns] = filled_values
    
    return combined_df

def main(base_path):
    """Main function to process all sectors"""
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return
    
    # Get all sector folders
    sector_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print(f"Found {len(sector_folders)} sectors to process")
    
    for sector_name in sector_folders:
        sector_path = base_path / sector_name
        
        try:
            # Process the sector
            combined_data = process_sector(sector_path, sector_name)
            
            if combined_data is not None:
                # Save to CSV
                output_file = sector_path / f"{sector_name}_combined.csv"
                combined_data.to_csv(output_file, index=False)
                print(f"  Saved: {output_file}")
                print(f"  Shape: {combined_data.shape}")
                print()
            else:
                print(f"  No data to save for {sector_name}")
                print()
                
        except Exception as e:
            print(f"Error processing sector {sector_name}: {str(e)}")
            print()
    
    print("Processing complete!")

# Example usage
if __name__ == "__main__":
    # Replace this with your actual path
    kennedys_path = r"C:\Users\Padma\OneDrive\Desktop\Internship\Kennedys\Kennedys"
    
    print("Financial Data Consolidation Script")
    print("=" * 50)
    
    main(kennedys_path)
    
    print("\nScript completed!")
    print("\nThe script has:")
    print("1. Scanned all sector folders")
    print("2. Extracted P&L data from each company's Excel files")
    print("3. Combined data by sector into CSV files")
    print("4. Applied 2nd degree polynomial interpolation for missing values")
    print("5. Added Company column for identification")
    print("6. Saved results as 'sector_name_combined.csv' in each sector folder")