import sys
from pathlib import Path
import pandas as pd


def check_file(path: Path):
    print(f"FILE: {path}")
    df = pd.read_feather(path)
    total_rows, total_cols = df.shape
    print(f"rows={total_rows:,}, cols={total_cols}")

    nan_counts = df.isna().sum()
    any_nan_rows = int(df.isna().any(axis=1).sum())
    n_cols_with_nan = int((nan_counts > 0).sum())
    print(f"rows_with_any_nan={any_nan_rows:,}")
    print(f"columns_with_nan={n_cols_with_nan}")
    if n_cols_with_nan:
        # Show top 20 columns by NaN count
        top = nan_counts.sort_values(ascending=False).head(20)
        print("top_nan_columns:")
        for k, v in top.items():
            if v > 0:
                print(f"  {k}: {v:,}")
    print("")


def main(argv):
    if len(argv) < 2:
        print("Usage: python -u analysis/check_nans.py <feather_file1> [<feather_file2> ...]")
        sys.exit(1)
    for p in argv[1:]:
        check_file(Path(p))


if __name__ == '__main__':
    main(sys.argv)

