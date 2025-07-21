import pandas as pd
import sys
import warnings

# Ignore specific Feather Dtype warning if it occurs
warnings.simplefilter(action='ignore', category=FutureWarning)

def inspect_feather(file_path):
    """Reads a feather file and prints its date range and column info."""
    try:
        print(f"--- Analyzing File: {file_path} ---")
        df = pd.read_feather(file_path)

        if 'date' not in df.columns and 'datetime' not in df.columns:
            print(f"Error: Neither 'date' nor 'datetime' column found in {file_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return

        date_col = 'date' if 'date' in df.columns else 'datetime'
        
        # Ensure the date column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
             df[date_col] = pd.to_datetime(df[date_col])

        print(f"Total rows: {len(df):,}")
        print(f"Start date: {df[date_col].min()}")
        print(f"End date:   {df[date_col].max()}")
        print(f"Columns: {df.columns.tolist()}")
        print("---------------------------------")

    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_feather.py <path_to_feather_file>")
    else:
        inspect_feather(sys.argv[1]) 