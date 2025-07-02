import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')

class EnhancedFinancialAnalyzer:
    def __init__(self, base_folder_path):
        """
        Initialize the Enhanced Financial Analyzer
        
        Args:
            base_folder_path (str): Path to the folder containing sector subfolders
        """
        self.base_folder_path = Path(base_folder_path)
        self.master_data = []
        
        # Standard financial metrics mapping for cleaning
        self.standard_metrics = {
            'revenue': ['revenue', 'sales', 'total income', 'turnover', 'income from operations'],
            'cost_of_goods_sold': ['cost of goods sold', 'cogs', 'cost of sales', 'direct cost'],
            'gross_profit': ['gross profit', 'gross income'],
            'operating_expenses': ['operating expenses', 'opex', 'administrative expenses', 'admin expenses'],
            'ebitda': ['ebitda', 'earnings before interest tax depreciation amortization'],
            'depreciation': ['depreciation', 'depreciation and amortization'],
            'ebit': ['ebit', 'earnings before interest and tax', 'operating profit'],
            'interest_expense': ['interest expense', 'interest cost', 'finance cost'],
            'pbt': ['pbt', 'profit before tax', 'earnings before tax'],
            'tax': ['tax', 'income tax', 'provision for tax'],
            'pat': ['pat', 'profit after tax', 'net profit', 'net income'],
            'other_income': ['other income', 'non operating income']
        }

    def clean_financial_metric_name(self, metric_name):
        """
        Clean financial metric names by removing special characters and standardizing

        Args:
            metric_name (str): Original metric name

        Returns:
            str: Cleaned metric name
        """
        if pd.isna(metric_name):
            return ''

        # Convert to string and lowercase
        metric = str(metric_name).lower().strip()

        # Remove special characters except spaces, hyphens, and parentheses
        metric = re.sub(r'[^\w\s\-\(\)]', '', metric)

        # Remove extra spaces
        metric = re.sub(r'\s+', ' ', metric).strip()

        # Standardize common metric names
        for standard_name, variations in self.standard_metrics.items():
            for variation in variations:
                if variation in metric:
                    return standard_name.replace('_', ' ').title()

        # If not found in standard metrics, clean and return
        # Capitalize first letter of each word
        return ' '.join(word.capitalize() for word in metric.split())

    def extract_single_year_from_column(self, col_name):
        """
        Extract a 4-digit year (e.g., 2015) from a column name string.
        Handles formats like '2015', 'Mar 2015', '2015_Mar'.

        Args:
            col_name (str): The column name string.

        Returns:
            str or None: The extracted 4-digit year as a string, or None if not found.
        """
        if pd.isna(col_name):
            return None
        col_str = str(col_name)
        year_matches = re.findall(r'20\d{2}', col_str) # Look for 4 digits starting with '20'
        if year_matches:
            return year_matches[0] # Return the first found 4-digit year
        return None

    def read_excel_files(self, folder_path):
        """
        Read all Excel files from a folder and return list of DataFrames

        Args:
            folder_path (Path): Path to folder containing Excel files

        Returns:
            list: List of tuples (company_name, DataFrame)
        """
        excel_files = []
        supported_extensions = ['.xlsx', '.xls', '.csv']

        for file_path in folder_path.glob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    company_name = file_path.stem

                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path)
                    else:
                        # Try to read the profit & loss sheet first
                        xl_file = pd.ExcelFile(file_path)
                        pnl_sheets = [sheet for sheet in xl_file.sheet_names
                                      if any(keyword in sheet.lower()
                                             for keyword in ['profit', 'loss', 'p&l', 'pnl', 'pl'])]

                        if pnl_sheets:
                            df = pd.read_excel(file_path, sheet_name=pnl_sheets[0])
                        else:
                            # If no P&L sheet found, read first sheet
                            df = pd.read_excel(file_path, sheet_name=0)

                    excel_files.append((company_name, df))
                    print(f" ✅  Successfully loaded: {company_name}")

                except Exception as e:
                    print(f" ❌  Error reading {file_path}: {str(e)}")
                    continue

        return excel_files

    def clean_and_standardize_pnl(self, df, company_name):
        """
        Clean and standardize P&L data with enhanced cleaning,
        and pivot to a wide format where each row is a unique
        (Company, Financial_Metric) pair and columns are years.

        Args:
            df (DataFrame): Raw P&L data
            company_name (str): Company name

        Returns:
            DataFrame: Cleaned and standardized P&L data in wide format
            list: Sorted list of unique year strings found in the data
        """
        df = df.copy()

        # Reset index if needed (e.g., if first column was part of index)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Find the first column (should contain financial metrics)
        first_col = df.columns[0]

        # Rename first column to 'Financial_Metric'
        df = df.rename(columns={first_col: 'Financial_Metric'})

        # Clean the Financial_Metric column
        df['Financial_Metric'] = df['Financial_Metric'].apply(self.clean_financial_metric_name)

        # Remove empty rows based on Financial_Metric (important before melting)
        df = df[df['Financial_Metric'].notna() & (df['Financial_Metric'] != '') & (df['Financial_Metric'] != 'Nan')]

        # Add company name column
        df['Company'] = company_name

        # Identify columns that are not 'Financial_Metric' or 'Company'
        # These are the columns containing year-specific values
        value_vars_candidates = [col for col in df.columns if col not in ['Financial_Metric', 'Company']]

        # Melt the DataFrame to a long format (unpivot year columns into rows)
        # This creates 'Original_Year_Col' (which holds the original column name like '2015_Mar')
        # and 'Value' (which holds the corresponding financial data).
        melted_df = pd.melt(df,
                            id_vars=['Company', 'Financial_Metric'],
                            value_vars=value_vars_candidates,
                            var_name='Original_Year_Col',
                            value_name='Value')

        # Clean the 'Original_Year_Col' to extract standardized 4-digit years (e.g., '2015')
        # This new 'Year' column will become the column headers in the final wide DataFrame.
        melted_df['Year'] = melted_df['Original_Year_Col'].apply(lambda x: self.extract_single_year_from_column(x))

        # Convert 'Value' column to numeric, coercing errors to NaN
        melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce')

        # Drop rows where 'Value' is NaN or 'Year' could not be extracted.
        # This is crucial for consolidation as we don't want rows with no meaningful data.
        melted_df = melted_df.dropna(subset=['Value', 'Year'])

        # Get all unique and sorted years from the cleaned 'Year' column
        all_years = sorted(melted_df['Year'].unique().tolist())

        # Pivot the DataFrame to the desired wide format:
        # - index: Unique combinations of 'Company' and 'Financial_Metric' will form rows.
        # - columns: The cleaned 'Year' values will become new columns.
        # - values: The 'Value' column (financial data) will populate these new year columns.
        # - aggfunc='first': In case of duplicate (Company, Financial_Metric, Year) combinations
        #   (though with proper cleaning, this should be rare for original data),
        #   take the first encountered value.
        pivoted_df = melted_df.pivot_table(
            index=['Company', 'Financial_Metric'],
            columns='Year',
            values='Value',
            aggfunc='first'
        ).reset_index() # Reset index to make 'Company' and 'Financial_Metric' regular columns

        # Ensure all years from 'all_years' are present as columns in the pivoted_df.
        # This handles cases where some companies might not have data for all years
        # that exist in the overall dataset. Missing years will be filled with NaN.
        for year_col in all_years:
            if year_col not in pivoted_df.columns:
                pivoted_df[year_col] = np.nan

        # Reorder columns to have 'Company', 'Financial_Metric' first, then sorted years.
        # This ensures a consistent and readable output structure.
        base_cols = ['Company', 'Financial_Metric']
        # Filter final_cols to include only columns actually present in pivoted_df
        # This prevents errors if a base_col was somehow missing (though unlikely here)
        final_cols = [col for col in base_cols if col in pivoted_df.columns] + \
                     [col for col in all_years if col in pivoted_df.columns]
        pivoted_df = pivoted_df[final_cols]

        return pivoted_df, all_years

    def calculate_comprehensive_metrics(self, sector_df):
        """
        Calculate comprehensive financial metrics (margins, ratios, CAGR)
        and add them to the sector DataFrame in wide format.

        Args:
            sector_df (DataFrame): Sector data in wide format.

        Returns:
            DataFrame: Data with comprehensive calculated metrics.
        """
        if sector_df.empty:
            return sector_df

        # Get year columns in ascending order from the wide-format DataFrame
        year_columns = sorted([col for col in sector_df.columns if col.isdigit()])
        if not year_columns:
            return sector_df # No years to calculate metrics for

        new_calculated_rows_list = [] # To store calculated metrics for new rows

        # Group by company to calculate metrics for each company individually
        for company in sector_df['Company'].unique():
            company_data = sector_df[sector_df['Company'] == company].copy()
            
            # Ensure 'Industry' column exists before trying to access iloc[0]
            industry_name = company_data['Industry'].iloc[0] if 'Industry' in company_data.columns and not company_data.empty else None

            # Create a dictionary for easy lookup of metrics:
            # {cleaned_metric_name: pandas Series (row data for that metric)}
            metrics_dict = {
                row['Financial_Metric'].lower().strip(): row
                for _, row in company_data.iterrows()
            }

            # Dictionary to store calculated metrics for the current company across years
            # Structure: { 'Metric Name': { 'Year': Value, ... }, ... }
            company_calculated_metrics_values = {}

            # Iterate through each year to calculate year-specific metrics
            for i, year_col in enumerate(year_columns):
                # Retrieve base values using the helper function
                revenue = self.get_metric_value(metrics_dict, ['revenue', 'sales'], year_col)
                cogs = self.get_metric_value(metrics_dict, ['cost of goods sold', 'cogs'], year_col)
                gross_profit = self.get_metric_value(metrics_dict, ['gross profit'], year_col)
                ebitda = self.get_metric_value(metrics_dict, ['ebitda'], year_col)
                ebit = self.get_metric_value(metrics_dict, ['ebit', 'operating profit'], year_col)
                pat = self.get_metric_value(metrics_dict, ['pat', 'profit after tax', 'net profit'], year_col)
                interest = self.get_metric_value(metrics_dict, ['interest expense'], year_col)

                # Calculate derived gross profit if not explicitly present
                if pd.notna(revenue) and pd.notna(cogs) and (pd.isna(gross_profit) or gross_profit == 0):
                    gross_profit = revenue - cogs

                # Calculate margins (Gross, EBITDA, Net) if revenue is available and non-zero
                if pd.notna(revenue) and revenue != 0:
                    if pd.notna(gross_profit):
                        margin = self.safe_divide(gross_profit, revenue)
                        if margin is not None:
                            company_calculated_metrics_values.setdefault('Gross Margin %', {})[year_col] = round(margin * 100, 2)
                    if pd.notna(ebitda):
                        margin = self.safe_divide(ebitda, revenue)
                        if margin is not None:
                            company_calculated_metrics_values.setdefault('EBITDA Margin %', {})[year_col] = round(margin * 100, 2)
                    if pd.notna(pat):
                        margin = self.safe_divide(pat, revenue)
                        if margin is not None:
                            company_calculated_metrics_values.setdefault('Net Margin %', {})[year_col] = round(margin * 100, 2)

                # Calculate Interest Coverage Ratio
                if pd.notna(ebit) and pd.notna(interest) and interest != 0:
                    ratio = self.safe_divide(ebit, interest)
                    if ratio is not None:
                        company_calculated_metrics_values.setdefault('Interest Coverage Ratio', {})[year_col] = round(ratio, 2)

                # Calculate CAGR (Compound Annual Growth Rate) only for the last year, if enough years exist
                if i == len(year_columns) - 1 and len(year_columns) >= 2:
                    first_year = year_columns[0]
                    last_year = year_columns[-1]
                    years_diff = int(last_year) - int(first_year)

                    if years_diff > 0:
                        # Revenue CAGR
                        first_revenue = self.get_metric_value(metrics_dict, ['revenue', 'sales'], first_year)
                        last_revenue = self.get_metric_value(metrics_dict, ['revenue', 'sales'], last_year)
                        revenue_cagr = self.calculate_cagr(first_revenue, last_revenue, years_diff)
                        if revenue_cagr is not None:
                            company_calculated_metrics_values.setdefault('Revenue CAGR %', {})[last_year] = revenue_cagr

                        # PAT CAGR
                        first_pat = self.get_metric_value(metrics_dict, ['pat', 'profit after tax'], first_year)
                        last_pat = self.get_metric_value(metrics_dict, ['pat', 'profit after tax'], last_year)
                        pat_cagr = self.calculate_cagr(first_pat, last_pat, years_diff)
                        if pat_cagr is not None:
                            company_calculated_metrics_values.setdefault('PAT CAGR %', {})[last_year] = pat_cagr

                        # EBITDA CAGR
                        first_ebitda = self.get_metric_value(metrics_dict, ['ebitda'], first_year)
                        last_ebitda = self.get_metric_value(metrics_dict, ['ebitda'], last_year)
                        ebitda_cagr = self.calculate_cagr(first_ebitda, last_ebitda, years_diff)
                        if ebitda_cagr is not None:
                            company_calculated_metrics_values.setdefault('EBITDA CAGR %', {})[last_year] = ebitda_cagr
            
            # Flatten the collected calculated metrics for the current company into a list of rows.
            # Each calculated metric (e.g., 'Gross Margin %') will become a new row.
            for metric_name, year_values_dict in company_calculated_metrics_values.items():
                row = {
                    'Industry': industry_name,
                    'Company': company,
                    'Financial_Metric': metric_name
                }
                # Populate all year columns, filling with NaN if no value for a specific year
                for year_col in year_columns:
                    row[year_col] = year_values_dict.get(year_col, np.nan)
                new_calculated_rows_list.append(row)

        # Combine the original sector_df with the newly calculated metrics
        if new_calculated_rows_list:
            calculated_df = pd.DataFrame(new_calculated_rows_list)
            # Ensure the calculated_df has the same columns as sector_df to avoid issues during concat
            # This is important for year columns alignment
            missing_cols_in_calculated = [col for col in sector_df.columns if col not in calculated_df.columns]
            for col in missing_cols_in_calculated:
                calculated_df[col] = np.nan
            
            # Ensure column order is the same before concatenating
            calculated_df = calculated_df[sector_df.columns]

            combined_df = pd.concat([sector_df, calculated_df], ignore_index=True)
            
            # Drop duplicates based on Company and Financial_Metric, keeping the last (which should be the calculated one
            # if a base metric had the same name, or simply ensuring uniqueness).
            # This is a safety measure; ideally calculated metrics have unique names.
            combined_df.drop_duplicates(subset=['Company', 'Financial_Metric'], keep='last', inplace=True)
            return combined_df
        
        return sector_df
    
    def calculate_cagr(self, start_value, end_value, years):
        """
        Calculate CAGR with proper error handling for edge cases

        Args:
            start_value: Starting value
            end_value: Ending value    
            years: Number of years

        Returns:
            float or None: CAGR percentage or None if calculation not possible
        """
        try:
            # Check if values are valid
            if pd.isna(start_value) or pd.isna(end_value) or years <= 0:
                return None
            
            # Handle negative values by taking absolute values for the calculation
            # and then applying sign correction. This allows CAGR calculation even with losses.
            start_abs = abs(start_value)
            end_abs = abs(end_value)
            
            if start_abs == 0 or end_abs == 0:
                return None # Cannot calculate CAGR if start or end value is zero
            
            # Calculate CAGR
            cagr = ((end_abs / start_abs) ** (1/years) - 1) * 100
            
            # Check if result is complex (shouldn't happen with abs values, but safety check)
            if isinstance(cagr, complex):
                return None
                
            # Apply sign correction:
            # If start and end values have different signs, the growth is effectively negative/decay.
            # If start_value was positive and decreased to negative, or vice versa.
            if (start_value > 0 and end_value < 0) or (start_value < 0 and end_value > 0):
                cagr = -abs(cagr)
            
            return round(float(cagr), 2)
            
        except Exception as e:
            print(f" ⚠️  CAGR calculation error: {e}")
            return None

    def safe_divide(self, numerator, denominator):
        """
        Safely divide two numbers with error handling

        Args:
            numerator: Numerator value
            denominator: Denominator value

        Returns:
            float or None: Result or None if division not possible
        """
        try:
            if pd.isna(denominator) or denominator == 0:
                return None
            if pd.isna(numerator):
                return None
            
            result = numerator / denominator
            
            # Check for complex numbers or infinity (though pd.to_numeric usually handles inf)
            if isinstance(result, complex) or np.isinf(result):
                return None
                    
            return float(result)
            
        except Exception:
            return None

    def get_metric_value(self, metrics_dict, search_terms, year_col):
        """
        Find a metric value by searching for keywords in the metrics_dict (wide format data).

        Args:
            metrics_dict (dict): Dictionary mapping cleaned metric names (str)
                                 to their corresponding pandas Series (row) from the wide DataFrame.
            search_terms (list): List of terms (strings) to search for in metric names.
            year_col (str): The specific year column (e.g., '2015') to retrieve the value from.

        Returns:
            float or None: The found numeric value for the specified metric and year,
                           or None if the metric is not found or value is invalid.
        """
        for term in search_terms:
            # Iterate through the dictionary items to find a matching metric
            for metric_key, row_series in metrics_dict.items():
                if term in metric_key:
                    # Check if the year_col exists in the row_series (which it should in wide format)
                    if year_col in row_series.index:
                        value = row_series[year_col]
                        # Ensure the value is not NaN and is not zero (unless zero is a valid value to return)
                        # Here, we return value even if zero, as it's a valid data point.
                        if pd.notna(value):
                            return float(value)
        return None

    def calculate_sector_margins(self, sector_df):
        """
        Calculate sector-wise margins (Gross, EBITDA, Net) for each year
        on a wide-format DataFrame.

        Args:
            sector_df (DataFrame): Sector data in wide format.

        Returns:
            DataFrame: Dataframe with sector margin calculations added.
        """
        if sector_df.empty:
            return sector_df

        # Get year columns from the wide-format DataFrame, sorted numerically
        year_columns = sorted([col for col in sector_df.columns if col.isdigit()])
        if not year_columns:
            return sector_df # No years to calculate for

        industry = sector_df['Industry'].iloc[0]  # Get the industry name (assuming one industry per sector_df)

        # Filter out existing summary/sector rows before aggregating
        # We only want to sum up data from actual companies, not previously calculated averages/sector totals.
        data_for_aggregation = sector_df[
            ~sector_df['Company'].str.contains('MEAN_|MEDIAN_|SECTOR_', na=False)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning if modified

        # Dictionary to store calculated sector margins (in a temporary wide-like structure)
        # Structure: { 'Metric Name': { 'Year': Value, ... } }
        sector_margins_calculated_values = {}

        # Calculate sector margins for each year
        for year_col in year_columns:
            # Sum up aggregated values for the sector for this specific year
            # Using .sum(min_count=1) ensures that if all values are NaN, the sum is NaN, not 0.
            sector_revenue = data_for_aggregation[
                data_for_aggregation['Financial_Metric'].str.lower().str.contains('revenue|sales', na=False, regex=True)
            ][year_col].sum(min_count=1)

            sector_gross_profit = data_for_aggregation[
                data_for_aggregation['Financial_Metric'].str.lower().str.contains('gross profit', na=False, regex=True)
            ][year_col].sum(min_count=1)

            sector_ebitda = data_for_aggregation[
                data_for_aggregation['Financial_Metric'].str.lower().str.contains('ebitda', na=False, regex=True)
            ][year_col].sum(min_count=1)

            sector_pat = data_for_aggregation[
                data_for_aggregation['Financial_Metric'].str.lower().str.contains('pat|profit after tax|net profit', na=False, regex=True)
            ][year_col].sum(min_count=1)

            # Calculate margins if sector_revenue is available and non-zero
            if pd.notna(sector_revenue) and sector_revenue != 0:
                if pd.notna(sector_gross_profit):
                    gross_margin = self.safe_divide(sector_gross_profit, sector_revenue)
                    if gross_margin is not None:
                        sector_margins_calculated_values.setdefault('Sector Gross Margin %', {})[year_col] = round(gross_margin * 100, 2)
                
                if pd.notna(sector_ebitda):
                    ebitda_margin = self.safe_divide(sector_ebitda, sector_revenue)
                    if ebitda_margin is not None:
                        sector_margins_calculated_values.setdefault('Sector EBITDA Margin %', {})[year_col] = round(ebitda_margin * 100, 2)
                
                if pd.notna(sector_pat):
                    net_margin = self.safe_divide(sector_pat, sector_revenue)
                    if net_margin is not None:
                        sector_margins_calculated_values.setdefault('Sector Net Margin %', {})[year_col] = round(net_margin * 100, 2)

        # Flatten the collected sector margin data into a list of rows for concatenation.
        # Each calculated sector margin (e.g., 'Sector Gross Margin %') will become a new row.
        final_sector_margin_rows = []
        if sector_margins_calculated_values:
            for metric_name, year_values_dict in sector_margins_calculated_values.items():
                row = {
                    'Industry': industry,
                    'Company': f'SECTOR_{industry}', # Special company name for sector aggregates
                    'Financial_Metric': metric_name
                }
                # Populate all year columns, filling with NaN if no value for a specific year
                for year_col in year_columns:
                    row[year_col] = year_values_dict.get(year_col, np.nan)
                final_sector_margin_rows.append(row)

        # Add sector margin rows to the dataframe
        if final_sector_margin_rows:
            sector_margins_df = pd.DataFrame(final_sector_margin_rows)
            
            # Ensure the calculated sector_margins_df has the same columns as sector_df
            # This is critical for successful concatenation and maintaining consistent schema.
            missing_cols_in_sector_margins = [col for col in sector_df.columns if col not in sector_margins_df.columns]
            for col in missing_cols_in_sector_margins:
                sector_margins_df[col] = np.nan
            
            # Reorder columns of sector_margins_df to match sector_df
            sector_margins_df = sector_margins_df[sector_df.columns]

            # Concatenate the original sector_df with the new sector margin rows
            combined_df = pd.concat([sector_df, sector_margins_df], ignore_index=True)
            return combined_df
        
        return sector_df # Return original if no new margins were calculated

    def create_summary_statistics(self, master_df):
        """
        Create summary statistics (mean and median) for each Industry and Financial_Metric
        across all companies (excluding already aggregated rows).
        Args:
            master_df (DataFrame): Master dataframe in wide format.
        Returns:
            DataFrame: Summary statistics dataframe in wide format.
        """
        if master_df.empty:
            return pd.DataFrame()

        # Get year columns from the wide-format DataFrame, sorted numerically
        year_columns = sorted([col for col in master_df.columns if col.isdigit()])
        if not year_columns:
            return pd.DataFrame() # No years to calculate statistics for

        summary_rows = []

        # Filter out existing summary/sector rows from the master_df before calculating new statistics
        # This ensures that mean/median are calculated only on raw company data.
        data_for_stats = master_df[
            ~master_df['Company'].str.contains('MEAN_|MEDIAN_|SECTOR_', na=False)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning if modified

        # Group by Industry and Financial Metric
        for industry in data_for_stats['Industry'].unique():
            industry_data = data_for_stats[data_for_stats['Industry'] == industry]
            
            for metric in industry_data['Financial_Metric'].unique():
                metric_data_for_stats = industry_data[industry_data['Financial_Metric'] == metric]
                
                # Calculate Mean
                mean_row = {'Industry': industry, 'Company': 'Mean', 'Financial_Metric': metric}
                for year_col in year_columns:
                    mean_val = metric_data_for_stats[year_col].mean()
                    mean_row[year_col] = round(mean_val, 2) if pd.notna(mean_val) else np.nan
                summary_rows.append(mean_row)

                # Calculate Median
                median_row = {'Industry': industry, 'Company': 'Median', 'Financial_Metric': metric}
                for year_col in year_columns:
                    median_val = metric_data_for_stats[year_col].median()
                    median_row[year_col] = round(median_val, 2) if pd.notna(median_val) else np.nan
                summary_rows.append(median_row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            # Ensure the summary_df has all original columns, filling missing year columns with NaN
            missing_cols_in_summary = [col for col in master_df.columns if col not in summary_df.columns]
            for col in missing_cols_in_summary:
                summary_df[col] = np.nan
            
            # Reorder columns to match the master_df structure (Industry, Company, Financial_Metric, then years)
            final_cols = ['Industry', 'Company', 'Financial_Metric'] + year_columns
            summary_df = summary_df[final_cols]
            return summary_df
        
        return pd.DataFrame()

    def process_sector(self, sector_name, sector_path):
        """
        Process all companies in a sector with enhanced analysis.
        Reads company files, cleans, standardizes, pivots to wide format,
        and calculates comprehensive metrics including sector aggregates.

        Args:
            sector_name (str): Name of the sector
            sector_path (Path): Path to sector folder

        Returns:
            DataFrame: Processed sector data with comprehensive metrics and sector margins in wide format.
        """
        print(f"\n 🔄  Processing sector: {sector_name}")

        # Read all company files
        company_dataframes = self.read_excel_files(sector_path)

        if not company_dataframes:
            print(f" ❌  No valid files found in {sector_name}")
            return pd.DataFrame()

        print(f" 📊  Found {len(company_dataframes)} companies in {sector_name}")

        # Process each company and collect pivoted DataFrames and all years encountered
        all_years_across_companies = set()
        processed_companies_dfs = []

        for company_name, df_raw in company_dataframes:
            # clean_and_standardize_pnl now returns a pivoted (wide) DataFrame
            pivoted_company_df, years_in_company_df = self.clean_and_standardize_pnl(df_raw, company_name)
            
            if not pivoted_company_df.empty:
                processed_companies_dfs.append(pivoted_company_df)
                all_years_across_companies.update(years_in_company_df)

        if not processed_companies_dfs:
            print(f" ❌  Could not process any companies in {sector_name}")
            return pd.DataFrame()

        # Sort all years encountered across all companies in this sector
        all_years = sorted(list(all_years_across_companies))

        # Combine all company dataframes into a single DataFrame for the sector.
        # Ensure all DataFrames have a consistent set of year columns before concatenation.
        # This prevents pandas from creating new columns with different names during concat.
        combined_sector_df_list = []
        for df_comp in processed_companies_dfs:
            temp_df = df_comp.copy()
            for year in all_years:
                if year not in temp_df.columns:
                    temp_df[year] = np.nan
            # Reorder columns to ensure consistency before concat
            # Ensure base columns are always present.
            base_cols_for_concat = ['Company', 'Financial_Metric'] + all_years
            # Filter to only include columns actually present in temp_df and in our desired list
            cols_to_keep = [col for col in base_cols_for_concat if col in temp_df.columns]
            combined_sector_df_list.append(temp_df[cols_to_keep])

        # Concatenate all company DataFrames for the current sector
        combined_df = pd.concat(combined_sector_df_list, ignore_index=True, sort=False)
        
        # Add industry column (if not already added by clean_and_standardize_pnl, though it's here now)
        combined_df['Industry'] = sector_name
        
        # Reorder columns for final sector DataFrame (Industry, Company, Financial_Metric, then sorted years)
        base_columns_final_order = ['Industry', 'Company', 'Financial_Metric']
        final_sector_columns = base_columns_final_order + all_years
        
        # Filter to only include columns actually present
        final_sector_columns = [col for col in final_sector_columns if col in combined_df.columns]
        combined_df = combined_df[final_sector_columns]

        # Calculate comprehensive metrics (margins, ratios, CAGR)
        print(f" 🧮  Calculating comprehensive metrics for {sector_name}...")
        final_data_with_company_metrics = self.calculate_comprehensive_metrics(combined_df)
        
        # Calculate sector-wise margins
        print(f" 🏭  Calculating sector-wise margins for {sector_name}...")
        final_data_with_margins = self.calculate_sector_margins(final_data_with_company_metrics)
        
        # Save sector-specific file
        sector_output_path = self.base_folder_path / f"{sector_name}_Enhanced_Analysis.csv" # Changed to CSV
        final_data_with_margins.to_csv(sector_output_path, index=False, encoding='utf-8')
        print(f" 💾  Saved enhanced sector analysis with margins: {sector_output_path}")
        
        return final_data_with_margins

    def run_analysis(self):
        """
        Main function to run the enhanced financial analysis
        """
        print(" 💼  ENHANCED FINANCIAL DATA ANALYZER")
        print("=" * 60)
        print(" 🔧  Features:")
        print("   • Cleaned financial metric names")
        print("   • Unique year columns in ascending order")
        print("   • Comprehensive financial ratios and margins")
        print("   • CAGR calculations")
        print("   • Sector-wise margin calculations")
        print("   • Mean and median statistics")
        print("   • Multi-file CSV output") # Updated feature list
        print("=" * 60)
        
        # Get folder path from user
        folder_path = input(" 📁  Enter the path to your main folder (containing sector subfolders): ").strip()
        
        # Remove quotes if present (common when path is copied with quotes)
        folder_path = folder_path.strip('"\'')
        
        if not os.path.exists(folder_path):
            print(" ❌  Folder path not found! Please check the path and try again.")
            return
        
        self.base_folder_path = Path(folder_path)
        
        # Find all sector folders
        sector_folders = [folder for folder in self.base_folder_path.iterdir()
                          if folder.is_dir()]

        if not sector_folders:
            print(" ❌  No sector folders found! Please ensure your main folder contains subfolders for each sector.")
            return

        print(f" 📂  Found {len(sector_folders)} sector folders.")

        all_sector_data = []

        # Process each sector
        for sector_folder in sector_folders:
            sector_name = sector_folder.name
            sector_data = self.process_sector(sector_name, sector_folder)
            
            if not sector_data.empty:
                all_sector_data.append(sector_data)
                print(f" ✅  Successfully processed {sector_name}.")
            else:
                print(f" ⚠️  No data processed for {sector_name}.")

        # Create comprehensive master file
        if all_sector_data:
            print(f"\n 🔗  Combining data from {len(all_sector_data)} sectors into master file...")
            
            # Combine all sector data. Ensure consistent columns for concatenation.
            # First, gather all unique years across all sector dataframes
            all_master_years = set()
            for df_sector in all_sector_data:
                all_master_years.update(sorted([col for col in df_sector.columns if col.isdigit()]))
            all_master_years = sorted(list(all_master_years))

            # Prepare each sector dataframe to have a consistent set of columns (all_master_years)
            harmonized_sector_dfs = []
            for df_sector in all_sector_data:
                temp_df = df_sector.copy()
                for year in all_master_years:
                    if year not in temp_df.columns:
                        temp_df[year] = np.nan
                # Reorder columns to ensure consistency before final concat
                base_cols_master = ['Industry', 'Company', 'Financial_Metric'] + all_master_years
                # Filter to only include columns actually present in temp_df and in our desired list
                cols_to_keep_master = [col for col in base_cols_master if col in temp_df.columns]
                harmonized_sector_dfs.append(temp_df[cols_to_keep_master])


            master_df = pd.concat(harmonized_sector_dfs, ignore_index=True, sort=False)
            
            # Create summary statistics
            print(" 📊  Calculating summary statistics (Mean & Median) for the master dataset...")
            summary_df = self.create_summary_statistics(master_df)
            
            # Combine master data with summary statistics
            final_master_df = pd.concat([master_df, summary_df], ignore_index=True, sort=False)
            
            # Save comprehensive master files as CSV
            master_output_path = self.base_folder_path / "Enhanced_Master_Financial_Analysis.csv"
            summary_output_path = self.base_folder_path / "Enhanced_Master_Summary_Statistics.csv"
            company_summary_output_path = self.base_folder_path / "Enhanced_Master_Company_Summary.csv"

            print(f" 💾  Saving master analysis to {master_output_path}...")
            
            # Save main data
            final_master_df.to_csv(master_output_path, index=False, encoding='utf-8')
            
            # Save summary statistics (if not empty)
            if not summary_df.empty:
                summary_df.to_csv(summary_output_path, index=False, encoding='utf-8')
                print(f" 💾  Saved summary statistics to {summary_output_path}")
            else:
                print(" ⚠️  No summary statistics generated (summary_df was empty), skipping save.")
            
            # Save company-wise summary
            company_summary_data = master_df[
                ~master_df['Company'].str.contains('MEAN_|MEDIAN_|SECTOR_', na=False)
            ]
            company_summary = company_summary_data.groupby(['Industry', 'Company']).size().reset_index(name='Metrics_Count')
            company_summary.to_csv(company_summary_output_path, index=False, encoding='utf-8')
            print(f" 💾  Saved company summary to {company_summary_output_path}")

            print(f"\n 🎉  SUCCESS! Enhanced Master files created in CSV format in: {self.base_folder_path}")
            
            # Display comprehensive summary
            print(f"\n 📈  COMPREHENSIVE ANALYSIS SUMMARY:")
            # Recalculate unique companies to exclude aggregate rows from count
            actual_companies_count = master_df[~master_df['Company'].str.contains('MEAN_|MEDIAN_|SECTOR_', na=False)]['Company'].nunique()
            print(f"    📊  Total Companies Analyzed: {actual_companies_count}")
            print(f"    🏭  Total Industries: {master_df['Industry'].nunique()}")
            print(f"    📋  Total Records in Complete Analysis: {len(final_master_df)}")
            print(f"    📊  Summary Statistics Records: {len(summary_df)}")
            
            # Show calculated metrics
            calculated_metrics = master_df[master_df['Financial_Metric'].str.contains('CAGR|Margin|Ratio|Sector', na=False)]['Financial_Metric'].unique()
            if len(calculated_metrics) > 0:
                print(f"\n 🧮  Enhanced Metrics Calculated:")
                for metric in sorted(calculated_metrics):
                    print(f"   • {metric}")
            
            # Show year range
            year_columns_in_final = sorted([col for col in final_master_df.columns if col.isdigit()])
            if year_columns_in_final:
                print(f"\n 📅  Year Range Covered: {year_columns_in_final[0]} to {year_columns_in_final[-1]}")
            
        else:
            print(" ❌  No data could be processed from any sector to create a master file!")

if __name__ == "__main__":
    analyzer = EnhancedFinancialAnalyzer(base_folder_path=".") # Placeholder path, will be prompted from user
    analyzer.run_analysis()
