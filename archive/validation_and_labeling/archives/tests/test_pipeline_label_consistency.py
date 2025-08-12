import numpy as np
import pandas as pd
import pytest

# ======================
# KONFIGURACJA TESTU
# ======================
RAW_DATA_PATH = "../input/BTCUSDT_1m_raw.feather"  # Surowe dane OHLCV
LABELS_PATH = "../output/BTCUSDT_TF-1m__FW-120__SL-050__TP-100__single_label.feather"  # Dane z etykietami pipeline
LABEL_COLUMN = "label"  # Nazwa kolumny z etykietą (np. 'label', 'label_0', ...)
FUTURE_WINDOW = 120  # Okno predykcji (minuty)
LONG_TP_PCT = 1.0
LONG_SL_PCT = 0.5
SHORT_TP_PCT = 1.0
SHORT_SL_PCT = 0.5
N_SAMPLES = 1000  # Liczba losowych timestampów do sprawdzenia

# Wymagane kolumny na podstawie analizy kodu modułu
REQUIRED_RAW_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']
REQUIRED_FEATURE_COLUMNS = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200', 
    'volume_to_ma1440', 'volume_to_ma43200'
]
VALID_LABEL_VALUES = [0, 1, 2]  # SHORT, HOLD, LONG
LABEL_DTYPE = 'uint8'

# ======================
# FUNKCJE WALIDACJI
# ======================

def validate_raw_data_structure(df_raw):
    """Waliduje strukturę surowych danych OHLCV."""
    errors = []
    
    # Sprawdź wymagane kolumny
    if 'datetime' in df_raw.columns:
        # datetime jako kolumna
        missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]
    else:
        # datetime jako indeks
        required_cols_no_dt = [col for col in REQUIRED_RAW_COLUMNS if col != 'datetime']
        missing_cols = [col for col in required_cols_no_dt if col not in df_raw.columns]
        if not pd.api.types.is_datetime64_any_dtype(df_raw.index):
            errors.append("Indeks nie jest typu datetime")
    
    if missing_cols:
        errors.append(f"Brakuje wymaganych kolumn w surowych danych: {missing_cols}")
    
    # Sprawdź typy danych numerycznych
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df_raw.columns:
            if not pd.api.types.is_numeric_dtype(df_raw[col]):
                errors.append(f"Kolumna {col} nie jest typu numerycznego")
    
    # Sprawdź logikę OHLC
    if all(col in df_raw.columns for col in ['open', 'high', 'low', 'close']):
        high_violations = (df_raw['high'] < df_raw[['open', 'close']].max(axis=1)).sum()
        if high_violations > 0:
            errors.append(f"Naruszenie logiki OHLC: {high_violations} przypadków gdzie high < max(open,close)")
        
        low_violations = (df_raw['low'] > df_raw[['open', 'close']].min(axis=1)).sum()
        if low_violations > 0:
            errors.append(f"Naruszenie logiki OHLC: {low_violations} przypadków gdzie low > min(open,close)")
        
        negative_prices = (df_raw[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if negative_prices > 0:
            errors.append(f"Nieprawidłowe ceny: {negative_prices} przypadków z cenami <= 0")
    
    # Sprawdź volume
    if 'volume' in df_raw.columns:
        negative_volume = (df_raw['volume'] < 0).sum()
        if negative_volume > 0:
            errors.append(f"Nieprawidłowy volume: {negative_volume} przypadków z volume < 0")
    
    if errors:
        raise ValueError("Błędy w strukturze surowych danych:\n" + "\n".join(f"  - {error}" for error in errors))

def validate_labels_data_structure(df_labels):
    """Waliduje strukturę danych z etykietami."""
    errors = []
    
    # Sprawdź indeks datetime
    if 'datetime' in df_labels.columns:
        # datetime jako kolumna
        if not pd.api.types.is_datetime64_any_dtype(df_labels['datetime']):
            errors.append("Kolumna datetime nie jest typu datetime")
    else:
        # datetime jako indeks
        if not pd.api.types.is_datetime64_any_dtype(df_labels.index):
            errors.append("Indeks nie jest typu datetime")
    
    # Sprawdź features
    missing_features = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df_labels.columns]
    if missing_features:
        errors.append(f"Brakuje wymaganych features: {missing_features}")
    
    # Sprawdź kolumnę z etykietą
    if LABEL_COLUMN not in df_labels.columns:
        errors.append(f"Brak kolumny z etykietą: {LABEL_COLUMN}")
    else:
        # Sprawdź typ danych etykiety
        if df_labels[LABEL_COLUMN].dtype != LABEL_DTYPE:
            errors.append(f"Nieprawidłowy typ etykiety: {df_labels[LABEL_COLUMN].dtype}, oczekiwano {LABEL_DTYPE}")
        
        # Sprawdź wartości etykiet
        unique_labels = df_labels[LABEL_COLUMN].unique()
        invalid_labels = [label for label in unique_labels if label not in VALID_LABEL_VALUES]
        if invalid_labels:
            errors.append(f"Nieprawidłowe wartości etykiet: {invalid_labels}, dozwolone: {VALID_LABEL_VALUES}")
        
        # Sprawdź czy są NaN w etykietach
        nan_labels = df_labels[LABEL_COLUMN].isna().sum()
        if nan_labels > 0:
            errors.append(f"Etykiety zawierają {nan_labels} wartości NaN")
    
    # Sprawdź features pod kątem NaN/inf
    for feature in REQUIRED_FEATURE_COLUMNS:
        if feature in df_labels.columns:
            nan_count = df_labels[feature].isna().sum()
            if nan_count > 0:
                errors.append(f"Feature {feature} zawiera {nan_count} wartości NaN")
            
            inf_count = np.isinf(df_labels[feature]).sum()
            if inf_count > 0:
                errors.append(f"Feature {feature} zawiera {inf_count} wartości inf")
    
    if errors:
        raise ValueError("Błędy w strukturze danych z etykietami:\n" + "\n".join(f"  - {error}" for error in errors))

def validate_time_synchronization(df_raw, df_labels):
    """Sprawdza synchronizację czasową między plikami."""
    errors = []
    
    # Pobierz indeksy datetime
    raw_index = df_raw.index if pd.api.types.is_datetime64_any_dtype(df_raw.index) else pd.to_datetime(df_raw['datetime'])
    labels_index = df_labels.index if pd.api.types.is_datetime64_any_dtype(df_labels.index) else pd.to_datetime(df_labels['datetime'])
    
    # Sprawdź czy timestampy są posortowane
    if not raw_index.is_monotonic_increasing:
        errors.append("Timestampy w surowych danych nie są posortowane chronologicznie")
    
    if not labels_index.is_monotonic_increasing:
        errors.append("Timestampy w danych z etykietami nie są posortowane chronologicznie")
    
    # Sprawdź częstotliwość (co 1 minutę)
    if len(raw_index) > 1:
        time_diffs = raw_index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=1)
        non_minute_diffs = (time_diffs != expected_diff).sum()
        if non_minute_diffs > 0:
            errors.append(f"Surowe dane: {non_minute_diffs} timestampów nie co minutę")
    
    if len(labels_index) > 1:
        time_diffs = labels_index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=1)
        non_minute_diffs = (time_diffs != expected_diff).sum()
        if non_minute_diffs > 0:
            errors.append(f"Dane z etykietami: {non_minute_diffs} timestampów nie co minutę")
    
    # Sprawdź zakresy czasowe (etykiety mogą być krótsze)
    raw_start, raw_end = raw_index.min(), raw_index.max()
    labels_start, labels_end = labels_index.min(), labels_index.max()
    
    if labels_start < raw_start:
        errors.append(f"Etykiety zaczynają się wcześniej niż surowe dane: {labels_start} < {raw_start}")
    
    if labels_end > raw_end:
        errors.append(f"Etykiety kończą się później niż surowe dane: {labels_end} > {raw_end}")
    
    # Sprawdź czy wszystkie timestampy z etykiet istnieją w surowych danych
    missing_in_raw = labels_index.difference(raw_index)
    if len(missing_in_raw) > 0:
        errors.append(f"Brak {len(missing_in_raw)} timestampów z etykiet w surowych danych")
    
    if errors:
        raise ValueError("Błędy synchronizacji czasowej:\n" + "\n".join(f"  - {error}" for error in errors))

# ======================
# FUNKCJE POMOCNICZE
# ======================

def manual_competitive_labeling(df, idx, future_window, long_tp, long_sl, short_tp, short_sl):
    """Przelicza etykietę competitive labeling dla pojedynczego indeksu."""
    if idx + future_window >= len(df):
        return 1  # HOLD jeśli nie ma wystarczająco danych
    entry_price = df.iloc[idx]['close']
    long_tp_val = entry_price * (1 + long_tp / 100)
    long_sl_val = entry_price * (1 - long_sl / 100)
    short_tp_val = entry_price * (1 - short_tp / 100)
    short_sl_val = entry_price * (1 + short_sl / 100)
    long_active = True
    short_active = True
    for j in range(idx + 1, idx + future_window + 1):
        high = df.iloc[j]['high']
        low = df.iloc[j]['low']
        if long_active and high >= long_tp_val:
            return 2  # LONG
        if short_active and low <= short_tp_val:
            return 0  # SHORT
        if long_active and low <= long_sl_val:
            long_active = False
        if short_active and high >= short_sl_val:
            short_active = False
        if not long_active and not short_active:
            return 1  # HOLD
    return 1  # HOLD jeśli nie było TP/SL

def prepare_dataframes(df_raw, df_labels):
    """Przygotowuje DataFrames z prawidłowymi indeksami datetime."""
    # Przygotuj surowe dane
    if 'datetime' in df_raw.columns:
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw = df_raw.set_index('datetime')
    
    # Przygotuj dane z etykietami
    if 'datetime' in df_labels.columns:
        df_labels['datetime'] = pd.to_datetime(df_labels['datetime'])
        df_labels = df_labels.set_index('datetime')
    
    return df_raw, df_labels

# ======================
# TEST WŁAŚCIWY
# ======================

def test_pipeline_label_consistency():
    """Testuje spójność etykiet pipeline z ręcznym przeliczeniem + pełna walidacja struktury."""
    print("🔍 Rozpoczynam test spójności etykiet pipeline...")
    
    # KROK 1: Wczytaj dane
    print("📂 Wczytuję dane...")
    try:
        df_raw = pd.read_feather(RAW_DATA_PATH)
        df_labels = pd.read_feather(LABELS_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Nie można wczytać pliku: {e}")
    
    print(f"   Surowe dane: {len(df_raw):,} wierszy")
    print(f"   Dane z etykietami: {len(df_labels):,} wierszy")
    
    # KROK 2: Walidacja struktury danych
    print("🔍 Walidacja struktury surowych danych...")
    validate_raw_data_structure(df_raw)
    print("   ✅ Struktura surowych danych OK")
    
    print("🔍 Walidacja struktury danych z etykietami...")
    validate_labels_data_structure(df_labels)
    print("   ✅ Struktura danych z etykietami OK")
    
    # KROK 3: Przygotuj DataFrames
    df_raw, df_labels = prepare_dataframes(df_raw, df_labels)
    
    # KROK 4: Walidacja synchronizacji czasowej
    print("🔍 Walidacja synchronizacji czasowej...")
    validate_time_synchronization(df_raw, df_labels)
    print("   ✅ Synchronizacja czasowa OK")
    
    # KROK 5: Test spójności etykiet
    print("🔍 Test spójności etykiet...")
    min_len = min(len(df_raw), len(df_labels))
    
    # Losuj N_SAMPLES indeksów z zakresu, gdzie future_window się mieści
    rng = np.random.default_rng(123)
    valid_indices = np.arange(0, min_len - FUTURE_WINDOW)
    sample_indices = rng.choice(valid_indices, size=min(N_SAMPLES, len(valid_indices)), replace=False)
    
    errors = []
    for i, idx in enumerate(sample_indices):
        if (i + 1) % 10 == 0:
            print(f"   Sprawdzam próbkę {i + 1}/{len(sample_indices)}...")
        
        ts = df_raw.index[idx]
        
        # Sprawdź, czy timestampy się zgadzają
        if ts not in df_labels.index:
            errors.append(f"Timestamp {ts} z surowych danych nie występuje w pliku z etykietami!")
            continue
        
        label_pipeline = df_labels.loc[ts][LABEL_COLUMN]
        label_manual = manual_competitive_labeling(
            df_raw, idx, FUTURE_WINDOW, LONG_TP_PCT, LONG_SL_PCT, SHORT_TP_PCT, SHORT_SL_PCT
        )
        
        if label_pipeline != label_manual:
            errors.append(
                f"Label mismatch at idx={idx}, ts={ts}: pipeline={label_pipeline}, manual={label_manual}"
            )
    
    # KROK 6: Raportowanie wyników
    if errors:
        print(f"\n❌ BŁĘDY ({len(errors)}):")
        for err in errors:
            print(f"   - {err}")
        raise AssertionError(f"Niektóre etykiety się nie zgadzają! ({len(errors)} błędów)")
    else:
        print(f"\n✅ TEST ZAKOŃCZONY POMYŚLNIE!")
        print(f"   Sprawdzono {len(sample_indices)} losowych etykiet")
        print(f"   Wszystkie etykiety są zgodne między pipeline a ręcznym przeliczeniem")
        print(f"   Struktura danych i synchronizacja czasowa są prawidłowe")

if __name__ == "__main__":
    test_pipeline_label_consistency() 