import pandas as pd
import sys
import os

def fix_feather_date_column(file_path):
    """Reads a feather file, renames 'datetime' to 'date', and saves it back."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

        print(f"--- Fixing Date Column in: {file_path} ---")
        df = pd.read_feather(file_path)

        if 'date' in df.columns:
            print("Column 'date' already exists. No changes needed.")
            return

        if 'datetime' not in df.columns:
            print("Error: Neither 'date' nor 'datetime' column found.")
            return

        print("Renaming 'datetime' column to 'date'...")
        df.rename(columns={'datetime': 'date'}, inplace=True)

        print("Saving the updated dataframe back to the file...")
        df.to_feather(file_path)

        print(f"--- File successfully updated. ---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_date_column.py <path_to_feather_file>")
    else:
        fix_feather_date_column(sys.argv[1]) 