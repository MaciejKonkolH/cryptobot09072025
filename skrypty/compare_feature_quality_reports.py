from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_latest_report(symbol: str, diag_dir: Path) -> Path | None:
    candidates = sorted(diag_dir.glob(f"feature_quality_{symbol}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_reports(symbols: list[str]) -> dict[str, pd.DataFrame]:
    diag_dir = PROJECT_ROOT / "training5" / "output" / "reports" / "diagnostics"
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        fp = find_latest_report(sym, diag_dir)
        if not fp:
            print(f"No report for {sym}")
            continue
        df = pd.read_csv(fp)
        out[sym] = df.set_index("feature")
    return out


def score_feature(row_ref: pd.Series, row_cmp: pd.Series) -> float:
    score = 0.0
    # Zero share high
    zero_cmp = row_cmp.get("zero_share", 0.0)
    if pd.notna(zero_cmp) and zero_cmp > 0.5:
        score += 2.0
    # NaN/Inf presence
    if (row_cmp.get("nan_count", 0) or 0) > 0:
        score += 2.0
    if (row_cmp.get("inf_count", 0) or 0) > 0:
        score += 2.0
    # Near-constancy: very small spread
    p95 = row_cmp.get("p95", np.nan)
    p5 = row_cmp.get("p5", np.nan)
    std = row_cmp.get("std", np.nan)
    if pd.notna(p95) and pd.notna(p5) and (p95 - p5) == 0:
        score += 2.0
    if pd.notna(std) and std != 0 and std < 1e-6:
        score += 1.5
    # Outlier vs reference (BTC) on upper tail and std
    if row_ref is not None and len(row_ref) > 0:
        ref_p99_5 = abs(row_ref.get("p99_5", np.nan))
        cmp_p99_5 = abs(row_cmp.get("p99_5", np.nan))
        if pd.notna(ref_p99_5) and pd.notna(cmp_p99_5) and ref_p99_5 > 0:
            ratio = cmp_p99_5 / ref_p99_5
            if ratio > 5:
                score += min(3.0, 0.5 * ratio)
        ref_std = abs(row_ref.get("std", np.nan))
        if pd.notna(ref_std) and pd.notna(std) and ref_std > 0 and std / ref_std > 5:
            score += min(2.0, std / ref_std / 2)
    return score


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    dfs = load_reports(symbols)
    if "BTCUSDT" not in dfs:
        print("BTCUSDT reference report missing")
        sys.exit(1)

    ref = dfs["BTCUSDT"]
    results = {}
    for sym in (s for s in symbols if s != "BTCUSDT" and s in dfs):
        cmp = dfs[sym]
        # Align features by union
        feats = sorted(set(ref.index).union(set(cmp.index)))
        scored = []
        for feat in feats:
            row_ref = ref.loc[feat] if feat in ref.index else pd.Series(dtype=float)
            row_cmp = cmp.loc[feat] if feat in cmp.index else pd.Series(dtype=float)
            sc = score_feature(row_ref, row_cmp)
            scored.append((feat, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:8]
        # Build summary rows with key stats
        summary_rows = []
        for feat, sc in top:
            rr = ref.loc[feat] if feat in ref.index else pd.Series(dtype=float)
            rc = cmp.loc[feat] if feat in cmp.index else pd.Series(dtype=float)
            summary_rows.append({
                "feature": feat,
                "score": round(sc, 3),
                "BTC_std": rr.get("std", np.nan),
                f"{sym}_std": rc.get("std", np.nan),
                "BTC_p99_5": rr.get("p99_5", np.nan),
                f"{sym}_p99_5": rc.get("p99_5", np.nan),
                f"{sym}_zero_share": rc.get("zero_share", np.nan),
                f"{sym}_nan": rc.get("nan_count", np.nan),
                f"{sym}_inf": rc.get("inf_count", np.nan),
            })
        results[sym] = pd.DataFrame(summary_rows)

    # Print concise summaries
    for sym, df in results.items():
        print(f"\n=== Top issues: {sym} vs BTCUSDT ===")
        if df.empty:
            print("No data")
            continue
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()

