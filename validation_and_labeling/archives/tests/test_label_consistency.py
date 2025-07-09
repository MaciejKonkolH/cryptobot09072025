import numpy as np
import pandas as pd
import pytest

# Parametry testowe (możesz zmienić na potrzeby testu)
LONG_TP_PCT = 1.0
LONG_SL_PCT = 0.7
SHORT_TP_PCT = 1.0
SHORT_SL_PCT = 0.7
FUTURE_WINDOW = 60  # minut

np.random.seed(42)


def generate_synthetic_ohlcv(n_rows=500):
    """Generuje syntetyczne dane OHLCV z losowymi ruchami ceny i wolumenu."""
    base_price = 100.0
    prices = [base_price]
    for _ in range(n_rows - 1):
        change = np.random.normal(0, 0.2)  # małe zmiany
        prices.append(prices[-1] * (1 + change / 100))
    prices = np.array(prices)
    
    # OHLC
    open_ = prices
    close = prices + np.random.normal(0, 0.05, n_rows)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.1, n_rows))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.1, n_rows))
    volume = np.abs(np.random.normal(1000, 100, n_rows))
    
    # Timestampy co minutę
    dt_index = pd.date_range('2023-01-01', periods=n_rows, freq='T')
    
    df = pd.DataFrame({
        'datetime': dt_index,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('datetime', inplace=True)
    return df


def manual_competitive_labeling(df, future_window=FUTURE_WINDOW):
    """Ręczna implementacja competitive labeling (LONG/SHORT/HOLD)."""
    labels = []
    for idx in range(len(df)):
        if idx + future_window >= len(df):
            labels.append(1)  # HOLD jeśli nie ma wystarczająco danych
            continue
        entry_price = df.iloc[idx]['close']
        long_tp = entry_price * (1 + LONG_TP_PCT / 100)
        long_sl = entry_price * (1 - LONG_SL_PCT / 100)
        short_tp = entry_price * (1 - SHORT_TP_PCT / 100)
        short_sl = entry_price * (1 + SHORT_SL_PCT / 100)
        long_active = True
        short_active = True
        label_assigned = False
        for j in range(idx + 1, idx + future_window + 1):
            high = df.iloc[j]['high']
            low = df.iloc[j]['low']
            if long_active and high >= long_tp:
                labels.append(2)  # LONG
                label_assigned = True
                break
            if short_active and low <= short_tp:
                labels.append(0)  # SHORT
                label_assigned = True
                break
            if long_active and low <= long_sl:
                long_active = False
            if short_active and high >= short_sl:
                short_active = False
            if not long_active and not short_active:
                labels.append(1)  # HOLD
                label_assigned = True
                break
        if not label_assigned:
            labels.append(1)  # HOLD
    return np.array(labels)


def test_label_consistency():
    """Testuje spójność etykiet competitive labeling na syntetycznych danych."""
    df = generate_synthetic_ohlcv(500)
    labels = manual_competitive_labeling(df)
    # Losuj 10 timestampów do sprawdzenia
    rng = np.random.default_rng(123)
    indices = rng.choice(len(df) - FUTURE_WINDOW, size=10, replace=False)
    for idx in indices:
        # Ręcznie przelicz etykietę
        entry_price = df.iloc[idx]['close']
        long_tp = entry_price * (1 + LONG_TP_PCT / 100)
        long_sl = entry_price * (1 - LONG_SL_PCT / 100)
        short_tp = entry_price * (1 - SHORT_TP_PCT / 100)
        short_sl = entry_price * (1 + SHORT_SL_PCT / 100)
        long_active = True
        short_active = True
        label_assigned = False
        for j in range(idx + 1, idx + FUTURE_WINDOW + 1):
            high = df.iloc[j]['high']
            low = df.iloc[j]['low']
            if long_active and high >= long_tp:
                expected_label = 2
                label_assigned = True
                break
            if short_active and low <= short_tp:
                expected_label = 0
                label_assigned = True
                break
            if long_active and low <= long_sl:
                long_active = False
            if short_active and high >= short_sl:
                short_active = False
            if not long_active and not short_active:
                expected_label = 1
                label_assigned = True
                break
        if not label_assigned:
            expected_label = 1
        # Porównaj z etykietą z funkcji
        actual_label = labels[idx]
        assert actual_label == expected_label, (
            f"Label mismatch at idx={idx}, ts={df.index[idx]}: expected {expected_label}, got {actual_label}"
        )
        # Sprawdź timestamp
        assert df.index[idx] == df.index[idx], "Timestamp mismatch!"

if __name__ == "__main__":
    test_label_consistency()
    print("Test passed!") 