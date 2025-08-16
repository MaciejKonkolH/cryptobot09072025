import sys
from pathlib import Path
import pandas as pd
import numpy as np


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=False, errors='coerce')
        # Normalize to minute precision, timezone-naive
        ts = ts.dt.tz_localize(None)
        ts = ts.dt.floor('T')
        df = df.assign(timestamp=ts).dropna(subset=['timestamp']).set_index('timestamp')
    return df


def main():
    project = Path(__file__).resolve().parent.parent
    l3_path = project / 'labeler3' / 'output' / 'ohlc_orderbook_labeled_3class_fw120m_15levels.feather'
    l5_path = project / 'labeler5' / 'output' / 'labeled_BTCUSDT.feather'

    if not l3_path.exists() or not l5_path.exists():
        print('Missing input files:')
        print(f'  labeler3: {l3_path.exists()} ({l3_path})')
        print(f'  labeler5: {l5_path.exists()} ({l5_path})')
        sys.exit(1)

    df3 = load_df(l3_path)
    df5 = load_df(l5_path)

    # Debug ranges
    if not df3.empty:
        print(f"L3 range: {df3.index.min()} -> {df3.index.max()} ({len(df3):,} rows)")
    if not df5.empty:
        print(f"L5 range: {df5.index.min()} -> {df5.index.max()} ({len(df5):,} rows)")

    # Align on common timestamps
    # Try multiple small offsets to account for 1-minute alignment differences
    offsets = [0, 1, -1]
    best_common = None
    best_k = 0
    for k in offsets:
        if k == 0:
            idx5 = df5.index
        else:
            idx5 = (df5.index + pd.Timedelta(minutes=k))
        common = df3.index.intersection(idx5)
        if len(common) > best_k:
            best_k = len(common)
            best_common = (k, common)

    if best_common is None or best_k == 0:
        print('Common timestamps: 0 (after minute-normalization and +/-1m offsets)')
        print('Hint: Ranges may not overlap. Verify date ranges and future_window trimming.')
        out_dir = project / 'analysis'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'labeler3_vs_labeler5.txt').write_text('No overlap after normalization.', encoding='utf-8')
        sys.exit(0)

    k_shift, common_idx = best_common
    if k_shift != 0:
        print(f"Using shift of L5 by {k_shift} minute(s) for alignment.")
        df5 = df5.copy()
        df5.index = df5.index + pd.Timedelta(minutes=k_shift)

    df3 = df3.loc[common_idx]
    df5 = df5.loc[common_idx]

    # Derive label columns present in both
    label_cols3 = [c for c in df3.columns if c.startswith('label_tp')]
    label_cols5 = [c for c in df5.columns if c.startswith('label_tp')]
    label_cols = sorted(list(set(label_cols3).intersection(label_cols5)))
    if not label_cols:
        print('No common label columns found.')
        sys.exit(1)

    print(f'Common timestamps: {len(common_idx):,}')
    print(f'Label levels compared: {len(label_cols)}')

    out_lines = []
    for col in label_cols:
        a = df3[col].astype('int8').values
        b = df5[col].astype('int8').values
        agree = (a == b)
        agree_pct = 100.0 * agree.mean()

        # 3x3 confusion: rows=labeler3 actual, cols=labeler5 actual
        cm = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                cm[i, j] = int(np.sum((a == i) & (b == j)))

        dist3 = {k: int(np.sum(a == k)) for k in [0, 1, 2]}
        dist5 = {k: int(np.sum(b == k)) for k in [0, 1, 2]}

        out_lines.append(f'Level: {col}')
        out_lines.append(f'  Agreement: {agree_pct:.2f}%')
        out_lines.append(f'  Dist L3: LONG={dist3.get(0,0):,}, SHORT={dist3.get(1,0):,}, NEUTRAL={dist3.get(2,0):,}')
        out_lines.append(f'  Dist L5: LONG={dist5.get(0,0):,}, SHORT={dist5.get(1,0):,}, NEUTRAL={dist5.get(2,0):,}')
        out_lines.append('  Confusion L3 vs L5 (rows=L3, cols=L5):')
        out_lines.append(f'           LONG    SHORT   NEUTRAL')
        out_lines.append(f'  LONG   {cm[0,0]:7d} {cm[0,1]:7d} {cm[0,2]:9d}')
        out_lines.append(f'  SHORT  {cm[1,0]:7d} {cm[1,1]:7d} {cm[1,2]:9d}')
        out_lines.append(f'  NEUTRL {cm[2,0]:7d} {cm[2,1]:7d} {cm[2,2]:9d}')
        out_lines.append('')

    out_text = '\n'.join(out_lines)
    out_dir = project / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'labeler3_vs_labeler5.txt'
    out_file.write_text(out_text, encoding='utf-8')
    print(f'Report saved: {out_file}')


if __name__ == '__main__':
    main()

