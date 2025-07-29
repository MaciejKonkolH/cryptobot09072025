import pandas as pd
import numpy as np

def compare_candles():
    """Porównuje świece z tych samych godzin z obu plików"""
    
    # Wczytaj dane
    print("Wczytywanie danych...")
    ft_df = pd.read_feather('user_data/data/binanceusdm/futures/BTC_USDT_USDT-1m-futures.feather')
    feat_df = pd.read_feather('../feature_calculator_ohlc_snapshot/output/ohlc_orderbook_features.feather')
    
    # Przygotuj timestampy
    feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp'])
    feat_df.set_index('timestamp', inplace=True)
    feat_df.index = feat_df.index.tz_localize('UTC')
    
    # Zakres backtestingu
    start_date = pd.to_datetime('2025-02-18 00:00:00+00:00')
    end_date = pd.to_datetime('2025-06-30 23:59:00+00:00')
    
    # Filtruj dane
    ft_filtered = ft_df[(ft_df.date >= start_date) & (ft_df.date <= end_date)]
    feat_filtered = feat_df[(feat_df.index >= start_date) & (feat_df.index <= end_date)]
    
    print(f"FreqTrade: {len(ft_filtered)} wierszy")
    print(f"Features: {len(feat_filtered)} wierszy")
    
    # Znajdź wspólne timestampy
    ft_timestamps = set(ft_filtered['date'])
    feat_timestamps = set(feat_filtered.index)
    common_timestamps = ft_timestamps.intersection(feat_timestamps)
    
    print(f"Wspólne timestampy: {len(common_timestamps)}")
    
    # Porównaj pierwsze 10 wspólnych świec
    common_list = sorted(list(common_timestamps))[:10]
    
    print("\nPorównanie 10 świec z tych samych godzin:")
    print("=" * 80)
    
    for i, timestamp in enumerate(common_list):
        ft_row = ft_filtered[ft_filtered['date'] == timestamp].iloc[0]
        feat_row = feat_filtered.loc[timestamp]
        
        print(f"{i+1}. {timestamp}")
        print(f"   FreqTrade: O={ft_row.open:.2f}, H={ft_row.high:.2f}, L={ft_row.low:.2f}, C={ft_row.close:.2f}, V={ft_row.volume:.2f}")
        print(f"   Features:  O={feat_row.open:.2f}, H={feat_row.high:.2f}, L={feat_row.low:.2f}, C={feat_row.close:.2f}, V={feat_row.volume:.2f}")
        
        # Oblicz różnice
        diff_o = abs(ft_row.open - feat_row.open)
        diff_h = abs(ft_row.high - feat_row.high)
        diff_l = abs(ft_row.low - feat_row.low)
        diff_c = abs(ft_row.close - feat_row.close)
        diff_v = abs(ft_row.volume - feat_row.volume)
        
        print(f"   Różnica: O={diff_o:.6f}, H={diff_h:.6f}, L={diff_l:.6f}, C={diff_c:.6f}, V={diff_v:.6f}")
        
        if diff_o > 0.01 or diff_h > 0.01 or diff_l > 0.01 or diff_c > 0.01 or diff_v > 0.01:
            print(f"   ⚠️  RÓŻNICA WYKRYTA!")
        
        print()

if __name__ == "__main__":
    compare_candles() 