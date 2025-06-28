import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import warnings
warnings.filterwarnings('ignore')

class IndustryFileCollectorAndConsolidator:
    def __init__(self):
        self.base_path = None
        self.progress_callback = None
        self.year_columns = list(range(2012, 2026))  # 2012-2025
        
    def create_industry_pnl_folder(self, base_path):
        """Create Industry_PnL folder if it doesn't exist"""
        industry_pnl_path = Path(base_path) / "Industry_PnL"
        industry_pnl_path.mkdir(exist_ok=True)
        return industry_pnl_path
    
    def find_combined_files(self, base_path):
        """Find all *_combined.csv files in sector subfolders"""
        base_path = Path(base_path)
        combined_files = []
        
        # Look for folders in the base path
        for item in base_path.iterdir():
            if item.is_dir():
                sector_name = item.name
                
                # Look for *_combined.csv files in this sector folder
                for file in item.glob("*_combined.csv"):
                    combined_files.append({
                        'sector_folder': sector_name,
                        'file_path': file,
                        'file_name': file.name,
                        'new_name': f"{sector_name}_PL.csv"  # Standardized naming
                    })
        
        return combined_files
    
    def collect_industry_files(self, base_path, progress_callback=None):
        """Step 1: Collect all *_combined.csv files into Industry_PnL folder"""
        self.progress_callback = progress_callback
        
        if self.progress_callback:
            self.progress_callback("🔍 STEP 1: Collecting Industry P&L Files")
            self.progress_callback("="*50)
        
        base_path = Path(base_path)
        if not base_path.exists():
            error_msg = f"Error: Path {base_path} does not exist"
            if self.progress_callback:
                self.progress_callback(error_msg)
            return False, error_msg, None
        
        # Create Industry_PnL folder
        industry_pnl_path = self.create_industry_pnl_folder(base_path)
        if self.progress_callback:
            self.progress_callback(f"📁 Created/Found Industry_PnL folder: {industry_pnl_path}")
        
        # Find all combined files
        combined_files = self.find_combined_files(base_path)
        
        if not combined_files:
            error_msg = "No *_combined.csv files found in sector subfolders"
            if self.progress_callback:
                self.progress_callback(f"❌ {error_msg}")
            return False, error_msg, None
        
        if self.progress_callback:
            self.progress_callback(f"\n🔍 Found {len(combined_files)} combined files:")
            for file_info in combined_files:
                self.progress_callback(f"    📊 {file_info['sector_folder']}/{file_info['file_name']}")
        
        # Copy files to Industry_PnL folder
        copied_files = []
        failed_files = []
        
        if self.progress_callback:
            self.progress_callback(f"\n📋 Copying files to Industry_PnL folder...")
        
        for i, file_info in enumerate(combined_files):
            try:
                source_path = file_info['file_path']
                destination_path = industry_pnl_path / file_info['new_name']
                
                # Copy the file
                shutil.copy2(source_path, destination_path)
                copied_files.append(file_info)
                
                if self.progress_callback:
                    self.progress_callback(f"    ✅ ({i+1}/{len(combined_files)}) {file_info['sector_folder']} → {file_info['new_name']}")
                    
            except Exception as e:
                failed_files.append({'file_info': file_info, 'error': str(e)})
                if self.progress_callback:
                    self.progress_callback(f"    ❌ ({i+1}/{len(combined_files)}) Failed: {file_info['sector_folder']} - {str(e)}")
        
        # Summary of Step 1
        if self.progress_callback:
            self.progress_callback(f"\n📊 STEP 1 SUMMARY:")
            self.progress_callback(f"    ✅ Successfully copied: {len(copied_files)} files")
            self.progress_callback(f"    ❌ Failed to copy: {len(failed_files)} files")
            self.progress_callback(f"    📁 Destination: {industry_pnl_path}")
        
        if len(copied_files) == 0:
            error_msg = "No files were successfully copied"
            return False, error_msg, None
        
        return True, f"Successfully collected {len(copied_files)} files", industry_pnl_path
    
    def extract_industry_name(self, filename):
        """Extract industry name from filename (everything before _PL.csv or _combined.csv)"""
        # Remove file extension
        name = os.path.splitext(filename)[0]
        
        # Remove _PL or _combined suffix
        name = re.sub(r'_(?:PL|combined)$', '', name, flags=re.IGNORECASE)
        
        # Clean and standardize
        industry = name.strip().title()
        return industry
    
    def clean_duplicate_columns(self, df):
        """Remove duplicate columns and fix pandas auto-renamed duplicates"""
        original_cols = list(df.columns)
        if self.progress_callback:
            self.progress_callback(f"          - Original columns: {original_cols}")
        
        # Dictionary to track year columns and their values
        year_data_indices = {} # Maps year (int) to a list of original column indices
        
        # Non-year columns to keep
        non_year_cols_to_keep = []
        
        for i, col in enumerate(original_cols):
            col_str = str(col).strip()
            
            # Handle year columns (int, float, or string representations like '2019', '2019.1')
            year_match = re.match(r'^(\d{4})(?:\.\d+)?$', col_str)
            if year_match:
                year = int(year_match.group(1))
                if 2000 <= year <= 2030: # Assuming years are within this range
                    if year not in year_data_indices:
                        year_data_indices[year] = []
                    year_data_indices[year].append(i)
                    continue
            
            # If not a year-like column, add to non-year columns
            non_year_cols_to_keep.append(col)
        
        # Create a new DataFrame with the non-year columns
        df_cleaned = df[non_year_cols_to_keep].copy()
        
        # Process and add year columns
        for year in sorted(year_data_indices.keys()):
            indices = year_data_indices[year]
            
            if len(indices) == 1:
                # Only one column for this year, just rename it
                df_cleaned[year] = df.iloc[:, indices[0]].astype(float, errors='ignore')
            else:
                # Multiple columns for this year - need to merge
                if self.progress_callback:
                    self.progress_callback(f"          - Found {len(indices)} columns for year {year}, merging...")
                
                # Create a temporary DataFrame with just the duplicate year columns
                temp_df_for_merge = df.iloc[:, indices]
                
                # Convert to numeric, coercing errors to NaN
                for col_idx in indices:
                    temp_df_for_merge[original_cols[col_idx]] = pd.to_numeric(temp_df_for_merge[original_cols[col_idx]], errors='coerce')

                # Combine by taking the first non-null value across rows for these year columns
                # This ensures we prioritize valid data and merge duplicate entries for the same year
                merged_series = temp_df_for_merge.bfill(axis=1).iloc[:, 0]
                df_cleaned[year] = merged_series.astype(float, errors='ignore')
        
        # Ensure final column names are unique (e.g., if a non-year column also was '2012')
        # This shouldn't be an issue if year columns are properly identified and removed from non-year_cols_to_keep logic,
        # but as a safeguard:
        final_cols_unique = []
        for col in df_cleaned.columns:
            if col not in final_cols_unique:
                final_cols_unique.append(col)
        df_cleaned = df_cleaned[final_cols_unique]

        if self.progress_callback:
            self.progress_callback(f"          - Cleaned columns: {list(df_cleaned.columns)}")
            
        return df_cleaned
    
    def standardize_columns(self, df, industry_name):
        """Standardize dataframe columns and ensure all required years are present"""
        # First clean duplicate columns (like '2019' and '2019.1')
        df = self.clean_duplicate_columns(df)
        
        # Required base columns
        required_columns = ['Industry', 'Company', 'Financial_Metric']
        
        # Add Industry column if not present
        if 'Industry' not in df.columns:
            df.insert(0, 'Industry', industry_name)
        else:
            df['Industry'] = industry_name # Ensure it's correctly set to the current industry
        
        # Ensure Company and Financial_Metric columns exist
        if 'Company' not in df.columns:
            raise ValueError(f"'Company' column not found in {industry_name} data")
        if 'Financial_Metric' not in df.columns:
            raise ValueError(f"'Financial_Metric' column not found in {industry_name} data")
        
        # Ensure all year columns exist (add missing ones with NaN)
        # Convert year columns to numeric type for consistency if they are not already
        for year in self.year_columns:
            if year not in df.columns:
                df[year] = np.nan
            else:
                # Ensure year column is numeric
                df[year] = pd.to_numeric(df[year], errors='coerce')
        
        # Get other columns (not in required or year columns)
        existing_columns = set(df.columns)
        required_set = set(required_columns)
        year_set = set(self.year_columns)
        other_columns = list(existing_columns - required_set - year_set)
        
        # Reorder columns: Industry | Company | Financial_Metric | Years (sorted) | Other columns (sorted)
        final_columns_order = required_columns + sorted(self.year_columns) + sorted(other_columns)
        
        # Only include columns that actually exist in the dataframe after all operations
        final_columns_order = [col for col in final_columns_order if col in df.columns]
        
        return df[final_columns_order]
    
    def validate_csv_file(self, file_path):
        """Validate if CSV file has required structure"""
        try:
            df = pd.read_csv(file_path)
            
            # Check if file has minimum required columns
            required_columns = ['Company', 'Financial_Metric']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            if len(df) == 0:
                return False, "File is empty"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def process_csv_file(self, file_path, industry_name):
        """
        Process individual CSV file, standardize columns, and deduplicate rows.
        Ensures uniqueness for (Industry, Company, Financial_Metric, Year) combinations.
        """
        try:
            df = pd.read_csv(file_path)
            
            if self.progress_callback:
                self.progress_callback(f"       📊 Processing: {os.path.basename(file_path)}")
                self.progress_callback(f"         - Original shape: {df.shape}")
            
            # Step 1: Standardize columns (add Industry, Company, Financial_Metric, handle year columns)
            df_standardized = self.standardize_columns(df, industry_name)
            
            if df_standardized is None:
                # Error should have been logged by standardize_columns if it returns None
                return None 

            # Step 2: Handle potential duplicate rows within the standardized dataframe
            # Define the key columns that should uniquely identify a row (before year values)
            key_columns = ['Industry', 'Company', 'Financial_Metric']
            
            # Identify current year columns in the standardized dataframe
            current_year_columns = [col for col in df_standardized.columns if col in self.year_columns]

            if not current_year_columns:
                if self.progress_callback:
                    self.progress_callback(f"         ⚠️ Warning: No valid year columns found in {os.path.basename(file_path)} after standardization. Skipping row deduplication.")
                # If no year columns, just return the standardized df as is.
                # However, this might indicate an issue with the input files if years are expected.
                return df_standardized


            # Melt the year columns into rows to easily identify duplicates across years
            # This creates rows like [Industry, Company, Metric, Year, Value]
            df_melted = df_standardized.melt(id_vars=key_columns, 
                                            value_vars=current_year_columns,
                                            var_name='Year', 
                                            value_name='Value')
            
            # Convert 'Value' to numeric, coercing errors to NaN. This is crucial for aggregation.
            df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

            # Drop rows where Value is NaN before grouping if they are considered "empty" entries
            # If you want to preserve rows even if all year values are NaN, remove this line
            df_melted.dropna(subset=['Value'], inplace=True)
            
            if df_melted.empty:
                if self.progress_callback:
                    self.progress_callback(f"         ⚠️ No valid data found in {os.path.basename(file_path)} after melting and dropping NaNs. Skipping.")
                return None


            # Aggregate by key columns and year, taking the first non-null value
            # This is where duplicates for (Industry, Company, Metric, Year) are resolved.
            # Use .first() if you trust the first occurrence, or .mean(), .sum() if values should be combined.
            # For P&L, often `first()` is sufficient if data should be unique per cell.
            df_deduplicated = df_melted.groupby(key_columns + ['Year'])['Value'].first().reset_index()

            # Pivot back to wide format, with years as columns
            df_final = df_deduplicated.pivot_table(index=key_columns, 
                                                  columns='Year', 
                                                  values='Value').reset_index()
            
            # Ensure all required year columns are present after pivoting, filling missing with NaN
            for year in self.year_columns:
                if year not in df_final.columns:
                    df_final[year] = np.nan
            
            # Reorder columns to match the desired final structure (Industry, Company, Financial_Metric, Years)
            # Ensure the order of other columns is consistent if they exist
            other_columns_after_pivot = [col for col in df_final.columns if col not in key_columns and col not in self.year_columns]
            final_col_order = key_columns + sorted(self.year_columns) + sorted(other_columns_after_pivot)
            
            # Filter for columns that actually exist in df_final
            final_col_order_existing = [col for col in final_col_order if col in df_final.columns]
            
            df_final = df_final[final_col_order_existing]

            if self.progress_callback:
                self.progress_callback(f"         - Deduped and finalized shape: {df_final.shape}")
                self.progress_callback(f"         - Companies: {df_final['Company'].nunique()}")
                self.progress_callback(f"         - Metrics: {df_final['Financial_Metric'].nunique()}")
            
            return df_final
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"       ❌ Error processing {file_path}: {str(e)}")
            return None
    
    def consolidate_collected_files(self, industry_pnl_path, progress_callback=None):
        """Step 2: Consolidate all collected files into one master file"""
        self.progress_callback = progress_callback
        
        if self.progress_callback:
            self.progress_callback("\n🔗 STEP 2: Consolidating All Industry Files")
            self.progress_callback("="*50)
        
        # Find all CSV files in Industry_PnL folder
        csv_files = list(Path(industry_pnl_path).glob("*.csv"))
        
        # Filter out already consolidated files
        csv_files = [f for f in csv_files if 'All_Industries' not in f.name]
        
        if not csv_files:
            error_msg = "No CSV files found in Industry_PnL folder for consolidation."
            if self.progress_callback:
                self.progress_callback(f"❌ {error_msg}")
            return False, error_msg
        
        if self.progress_callback:
            self.progress_callback(f"📋 Found {len(csv_files)} files to consolidate:")
            for file in csv_files:
                industry_name = self.extract_industry_name(file.name)
                self.progress_callback(f"    • {file.name} → Industry: {industry_name}")
        
        # Process each file
        all_dataframes = []
        processed_count = 0
        
        if self.progress_callback:
            self.progress_callback(f"\n🔄 Processing files...")
        
        for file_path in csv_files:
            try:
                industry_name = self.extract_industry_name(file_path.name)
                
                # Validate file structure
                is_valid, message = self.validate_csv_file(file_path)
                
                if not is_valid:
                    if self.progress_callback:
                        self.progress_callback(f"    ⚠️ Skipping {file_path.name}: {message}")
                    continue
                
                # Process and deduplicate rows within each file
                df = self.process_csv_file(file_path, industry_name)
                
                if df is not None and not df.empty:
                    all_dataframes.append(df)
                    processed_count += 1
                    if self.progress_callback:
                        self.progress_callback(f"          ✅ Successfully processed and prepared {industry_name}")
                elif df is not None and df.empty:
                    if self.progress_callback:
                        self.progress_callback(f"          ⚠️ Processed {industry_name} but resulting DataFrame is empty.")
                else: # df is None, meaning an error occurred during processing
                    if self.progress_callback:
                        self.progress_callback(f"          ❌ Failed to process {industry_name} (details above).")
                        
            except Exception as e:
                if self.progress_callback:
                    self.progress_callback(f"    ❌ Critical Error with {file_path.name}: {str(e)}")
        
        if not all_dataframes:
            error_msg = "No files were successfully processed into valid DataFrames for consolidation."
            if self.progress_callback:
                self.progress_callback(f"❌ {error_msg}")
            return False, error_msg
        
        # Combine all dataframes
        if self.progress_callback:
            self.progress_callback(f"\n🔗 Combining all industry data...")
        
        try:
            # Concatenate all dataframes.
            # ignore_index=True ensures a fresh, non-problematic index for the final combined DataFrame.
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Ensure final combined_df has unique (Industry, Company, Financial_Metric) before final sort
            # This step handles cases where different files might have the same (Company, Metric) for the same Industry,
            # which *shouldn't* happen if files are truly industry-specific and well-formed.
            # But as a final safeguard:
            
            # Group by the identifying columns and take the first valid value for each year.
            # This implicitly merges rows that might be duplicates across different source files for the same Industry/Company/Metric
            # (though the earlier logic in process_csv_file should largely prevent this for same-industry duplicates)
            
            # Re-convert year columns to numeric just in case there was mixed type after concat
            for year_col in self.year_columns:
                if year_col in combined_df.columns:
                    combined_df[year_col] = pd.to_numeric(combined_df[year_col], errors='coerce')

            # Define the columns that form the unique identifier for a row in the final output
            final_unique_id_cols = ['Industry', 'Company', 'Financial_Metric']
            
            # Identify columns that are not part of the unique ID or year columns
            other_non_year_cols = [col for col in combined_df.columns if col not in final_unique_id_cols and col not in self.year_columns]

            # Melt to handle potential row duplicates *after* concatenation for years
            combined_df_melted = combined_df.melt(id_vars=final_unique_id_cols + other_non_year_cols,
                                                  value_vars=self.year_columns,
                                                  var_name='Year',
                                                  value_name='Value')
            
            # Group by all identifying columns and Year, taking the first valid value
            # This is the ultimate deduplication for the final master file
            combined_df_final_deduplicated = combined_df_melted.groupby(final_unique_id_cols + other_non_year_cols + ['Year'])['Value'].first().reset_index()

            # Pivot back to wide format
            combined_df_final = combined_df_final_deduplicated.pivot_table(index=final_unique_id_cols + other_non_year_cols,
                                                                           columns='Year',
                                                                           values='Value').reset_index()

            # Ensure all year columns are present after the final pivot
            for year in self.year_columns:
                if year not in combined_df_final.columns:
                    combined_df_final[year] = np.nan

            # Reorder columns for the final output file
            final_output_columns_order = final_unique_id_cols + sorted(self.year_columns) + sorted(other_non_year_cols)
            final_output_columns_order = [col for col in final_output_columns_order if col in combined_df_final.columns]

            combined_df_final = combined_df_final[final_output_columns_order]
            
            # Sort the final DataFrame
            combined_df_final = combined_df_final.sort_values(final_unique_id_cols)
            combined_df_final.reset_index(drop=True, inplace=True)
            
            # Generate output filename
            output_file = Path(industry_pnl_path) / "All_Industries_ProfitLoss.csv"
            
            # Save the combined file
            combined_df_final.to_csv(output_file, index=False)
            
            # Generate summary statistics
            total_industries = combined_df_final['Industry'].nunique()
            total_companies = combined_df_final['Company'].nunique()
            total_metrics = combined_df_final['Financial_Metric'].nunique()
            total_rows = len(combined_df_final)
            
            success_msg = f"""
📊 CONSOLIDATION COMPLETE! 

📁 Output File: {output_file}
📈 Summary Statistics:
    • Total Industries: {total_industries}
    • Total Companies: {total_companies}
    • Total Financial Metrics: {total_metrics}
    • Total Data Rows: {total_rows:,}
    • Final Shape: {combined_df_final.shape}

🏭 Industries Included:
{chr(10).join([f'    • {industry}' for industry in sorted(combined_df_final['Industry'].unique())])}

📅 Year Coverage: {min(self.year_columns)} - {max(self.year_columns)}
            """
            
            if self.progress_callback:
                self.progress_callback(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Error combining data: {str(e)}"
            if self.progress_callback:
                self.progress_callback(f"❌ {error_msg}")
            return False, error_msg
    
    def run_complete_process(self, base_path, progress_callback=None):
        """Run the complete 2-step process"""
        self.progress_callback = progress_callback
        
        # Step 1: Collect files
        success1, message1, industry_pnl_path = self.collect_industry_files(base_path, progress_callback)
        
        if not success1:
            return False, message1
        
        # Step 2: Consolidate files
        success2, message2 = self.consolidate_collected_files(industry_pnl_path, progress_callback)
        
        if success2:
            final_message = f"🎉 COMPLETE SUCCESS!\n\nStep 1: {message1}\nStep 2: Consolidation completed successfully!"
            return True, final_message
        else:
            return False, f"Step 1 succeeded, but Step 2 failed: {message2}"

class IndustryCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Industry P&L File Collector & Consolidator")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        self.processor = IndustryFileCollectorAndConsolidator()
        self.selected_path = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏭 Industry P&L File Collector & Consolidator", 
                                 font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(main_frame, text="Step 1: Collect *_combined.csv files | Step 2: Consolidate into master file", 
                                     font=('Helvetica', 11))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Path selection
        ttk.Label(main_frame, text="Select Root Folder (containing sector subfolders):").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        path_entry = ttk.Entry(main_frame, textvariable=self.selected_path, width=80)
        path_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        
        browse_button = ttk.Button(main_frame, text="📁 Browse", command=self.browse_folder)
        browse_button.grid(row=2, column=2, padx=(5, 0), pady=5)
        
        # Instructions
        instructions = """
📋 TWO-STEP PROCESS:

🔸 STEP 1: FILE COLLECTION
    • Scans each sector subfolder (Automobiles, Cement, etc.)
    • Finds *_combined.csv files (like Automobiles_combined.csv)
    • Copies them to new Industry_PnL folder with standardized names
    • Example: Automobiles/Automobiles_combined.csv → Industry_PnL/Automobiles_PL.csv

🔸 STEP 2: CONSOLIDATION  
    • Processes all files in Industry_PnL folder
    • Adds Industry column to each dataset
    • Removes duplicate year columns and merges data
    • Ensures all years 2012-2025 are included
    • Creates master file: All_Industries_ProfitLoss.csv

📁 EXPECTED FOLDER STRUCTURE:
    Root_Folder/
    ├── Automobiles/
    │   └── Automobiles_combined.csv    ← Will be collected
    ├── Cement/
    │   └── Cement_combined.csv         ← Will be collected  
    ├── Banking/
    │   └── Banking_combined.csv        ← Will be collected
    └── Industry_PnL/                   ← Will be created
        ├── Automobiles_PL.csv          ← Step 1 output
        ├── Cement_PL.csv               ← Step 1 output
        └── All_Industries_ProfitLoss.csv ← Step 2 output

✅ WHAT THE TOOL DOES:
    • Automatically finds and collects all *_combined.csv files
    • Creates Industry_PnL folder structure
    • Handles duplicate year columns by merging them
    • Standardizes file naming and data structure
    • Consolidates duplicate rows based on unique identifiers (Industry, Company, Financial_Metric, Year)
    • Combines everything into one master dataset
        """
        
        instructions_label = ttk.Label(main_frame, text=instructions, 
                                         font=('Consolas', 9), justify=tk.LEFT,
                                         background='#f0f0f0', relief='sunken', padding=10)
        instructions_label.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=15)
        
        # Process buttons
        self.collect_button = ttk.Button(button_frame, text="📋 Step 1: Collect Files Only", 
                                         command=self.collect_files_only, state='disabled')
        self.collect_button.grid(row=0, column=0, padx=5)
        
        self.full_process_button = ttk.Button(button_frame, text="🚀 Run Complete Process (Steps 1+2)", 
                                                 command=self.run_full_process, state='disabled')
        self.full_process_button.grid(row=0, column=1, padx=5)
        
        # Progress area
        progress_frame = ttk.LabelFrame(main_frame, text="📊 Progress & Results", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Progress text
        self.progress_text = tk.Text(progress_frame, height=20, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=self.progress_text.yview)
        self.progress_text.configure(yscrollcommand=scrollbar.set)
        
        self.progress_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # Bind path change to enable/disable buttons
        self.selected_path.trace_add('write', self.on_path_change)
    
    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="Select Root Folder Containing Sector Subfolders")
        if folder_path:
            self.selected_path.set(folder_path)
    
    def on_path_change(self, *args):
        if self.selected_path.get().strip():
            self.collect_button.config(state='normal')
            self.full_process_button.config(state='normal')
        else:
            self.collect_button.config(state='disabled')
            self.full_process_button.config(state='disabled')
    
    def update_progress(self, message):
        self.progress_text.insert(tk.END, message + '\n')
        self.progress_text.see(tk.END)
        self.root.update_idletasks()
    
    def collect_files_only(self):
        if not self.selected_path.get().strip():
            messagebox.showerror("Error", "Please select a folder first!")
            return
        
        # Clear progress text
        self.progress_text.delete(1.0, tk.END)
        
        # Disable buttons and start progress bar
        self.collect_button.config(state='disabled')
        self.full_process_button.config(state='disabled')
        self.progress_bar.start()
        
        # Start processing in a separate thread
        threading.Thread(target=self.run_collect_only, daemon=True).start()
    
    def run_full_process(self):
        if not self.selected_path.get().strip():
            messagebox.showerror("Error", "Please select a folder first!")
            return
        
        # Clear progress text
        self.progress_text.delete(1.0, tk.END)
        
        # Disable buttons and start progress bar
        self.collect_button.config(state='disabled')
        self.full_process_button.config(state='disabled')
        self.progress_bar.start()
        
        # Start processing in a separate thread
        threading.Thread(target=self.run_complete_process, daemon=True).start()
    
    def run_collect_only(self):
        try:
            success, message, _ = self.processor.collect_industry_files(
                self.selected_path.get(), 
                self.update_progress
            )
            
            # Stop progress bar and re-enable buttons
            self.progress_bar.stop()
            self.collect_button.config(state='normal')
            self.full_process_button.config(state='normal')
            
            if success:
                messagebox.showinfo("✅ Step 1 Complete!", "Files have been successfully collected in Industry_PnL folder!")
            else:
                messagebox.showerror("❌ Error", message)
                
        except Exception as e:
            self.progress_bar.stop()
            self.collect_button.config(state='normal')
            self.full_process_button.config(state='normal')
            error_msg = f"An unexpected error occurred: {str(e)}"
            self.update_progress(error_msg)
            messagebox.showerror("❌ Error", error_msg)
    
    def run_complete_process(self):
        try:
            success, message = self.processor.run_complete_process(
                self.selected_path.get(), 
                self.update_progress
            )
            
            # Stop progress bar and re-enable buttons
            self.progress_bar.stop()
            self.collect_button.config(state='normal')
            self.full_process_button.config(state='normal')
            
            if success:
                messagebox.showinfo("🎉 Complete Success!", "All files collected and consolidated successfully!")
            else:
                messagebox.showerror("❌ Error", message)
                
        except Exception as e:
            self.progress_bar.stop()
            self.collect_button.config(state='normal')
            self.full_process_button.config(state='normal')
            error_msg = f"An unexpected error occurred: {str(e)}"
            self.update_progress(error_msg)
            messagebox.showerror("❌ Error", error_msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustryCollectorGUI(root)
    root.mainloop()