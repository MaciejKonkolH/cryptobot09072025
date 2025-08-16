import sys
from pathlib import Path
import pandas as pd


def main(symbol: str = "BTCUSDT"):
    p = Path(__file__).resolve().parent / "output" / f"features_{symbol}.feather"
    df = pd.read_feather(p)
    required = [
        'price_trend_30m','price_trend_2h','price_trend_6h','price_strength','price_consistency_score',
        'price_vs_ma_60','price_vs_ma_240','ma_trend','price_volatility_rolling',
        'volume_trend_1h','volume_intensity','volume_volatility_rolling','volume_price_correlation','volume_momentum',
        'spread_tightness','depth_ratio_s1','depth_ratio_s2','depth_momentum',
        'market_trend_strength','market_trend_direction','market_choppiness','bollinger_band_width','market_regime',
        'volatility_regime','volatility_percentile','volatility_persistence','volatility_momentum','volatility_of_volatility','volatility_term_structure',
        'volume_imbalance','weighted_volume_imbalance','volume_imbalance_trend','price_pressure','weighted_price_pressure','price_pressure_momentum','order_flow_imbalance','order_flow_trend'
    ]
    cols = set(df.columns)
    present = [c for c in required if c in cols]
    missing = [c for c in required if c not in cols]
    print(f"present {len(present)}")
    print(present)
    print(f"missing {len(missing)}")
    print(missing)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    main(sym)

