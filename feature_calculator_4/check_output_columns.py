import sys
from pathlib import Path
import pandas as pd


def expected_columns() -> list[str]:
    chan = []
    for w in [240, 180, 120]:
        chan += [
            f"pos_in_channel_{w}",
            f"width_over_ATR_{w}",
            f"slope_over_ATR_window_{w}",
            f"channel_fit_score_{w}",
            f"pos_in_channel_{w}_x_imbalance_1pct",
            f"slope_over_ATR_window_{w}_x_imbalance_1pct",
            f"width_over_ATR_{w}_x_imbalance_1pct",
        ]
    ohlc = [
        "body_ratio", "wick_up_ratio", "wick_down_ratio",
        "r_1", "r_5", "r_15", "slope_return_120",
        "vol_regime_120", "vol_of_vol_120", "r2_trend_120",
        "RSI_14", "RSI_30", "StochK_14_3", "StochD_14_3",
        "MACD_hist_over_ATR", "ADX_14", "di_diff_14",
        "CCI_20_over_ATR", "bb_pos_20", "bb_width_over_ATR_20",
        "donch_pos_60", "donch_width_over_ATR_60",
        "close_vs_ema_60", "close_vs_ema_120", "slope_ema_60_over_ATR",
        "MFI_14", "OBV_slope_over_ATR",
    ]
    ob = [
        "imbalance_1pct_notional", "log_ratio_ask_2_over_1", "log_ratio_bid_2_over_1",
        "ask_near_ratio", "bid_near_ratio", "concentration_near_mkt",
        "ask_com", "bid_com", "com_diff", "pressure_12", "pressure_12_norm",
        "side_skew", "dA1", "dB1", "dImb1",
        "ema_dImb1_5", "ema_dImb1_10",
        "persistence_imbalance_1pct_ema5", "persistence_imbalance_1pct_ema10",
    ]
    # Optional per-model: reach_TP_notional, reach_SL_notional, resistance_vs_support
    return chan + ohlc + ob


def main(symbol: str):
    feat_path = Path(__file__).resolve().parent / "output" / f"features_{symbol}.feather"
    df = pd.read_feather(feat_path)
    cols = set(df.columns.tolist())
    exp = expected_columns()
    missing = [c for c in exp if c not in cols]
    extra = [c for c in cols if c not in exp and c != "timestamp"]
    print(f"Total columns: {len(cols)}")
    print(f"Expected listed: {len(exp)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("MISSING:")
        for c in missing:
            print(c)
    print(f"Extra (not in expected list): {len(extra)}")
    if extra:
        print("EXTRA:")
        for c in sorted(extra):
            print(c)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    main(sym)

