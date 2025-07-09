"""
Moduł walidacji i wypełniania luk w danych OHLCV
Implementuje algorytmy walidacji podstawowej i BRIDGE strategy
"""
import logging
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta
import random

# Obsługa importów
try:
    from . import config
    from .utils import setup_logging, validate_dataframe_columns, ProgressReporter
    from .data_interpolator import DataInterpolator
except ImportError:
    # Standalone script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from utils import setup_logging, validate_dataframe_columns, ProgressReporter
    from data_interpolator import DataInterpolator

class DataValidator:
    """Klasa odpowiedzialna za walidację i wypełnianie luk w danych OHLCV"""
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.DataValidator")
    
    def validate_and_clean(self, df: pd.DataFrame, pair_name: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Wykonuje pełną walidację i czyszczenie danych zgodnie z algorytmem z planu
        🆕 ENHANCED: Dodana integracja z systemem obcinania wartości ekstremalnych
        
        Args:
            df: Surowe dane OHLCV
            pair_name: Nazwa pary (do logowania)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (oczyszczone_dane, raport_walidacji)
        """
        self.logger.info(f"Rozpoczynam walidację danych dla {pair_name}")
        
        validation_report = {
            "input_rows": len(df),
            "duplicates_removed": 0,
            "gaps_detected": 0,
            "gaps_filled": 0,
            "largest_gap_minutes": 0,
            "data_quality_passed": True,
            "validation_warnings": [],
            "interpolation_report": {"enabled": False},
            "feature_clipping_report": {"enabled": False},
            "ohlcv_clipping_report": {"enabled": False}
        }
        
        try:
            # KROK 1: SPRAWDŹ KOLUMNY
            df_clean = self._validate_columns(df.copy(), validation_report)
            
            # KROK 1.5: INTERPOLACJA ZEPSUTYCH DANYCH (ZMODYFIKOWANY)
            if getattr(config, 'INTERPOLATION_ENABLED', True):
                self.logger.info("Rozpoczynam interpolację zepsutych danych")
                
                try:
                    interpolator = DataInterpolator()
                    cleaned_data, interpolation_report = interpolator.interpolate_corrupted_data(df_clean, pair_name)
                    validation_report["interpolation_report"] = interpolation_report
                    
                    # Sprawdź czy interpolacja się udała
                    if interpolation_report.get("success", False):
                        self.logger.info(f"✅ Interpolacja: {interpolation_report['total_interpolated']} świec naprawionych")
                        df_clean = cleaned_data
                    else:
                        # FALLBACK: Jeśli interpolacja failed
                        if getattr(config, 'INTERPOLATION_FALLBACK_ON_FAILURE', True):
                            self.logger.warning("⚠️ Interpolacja nie powiodła się - stosowanie fallback sanitization")
                            df_clean = self._apply_fallback_sanitization(df_clean)
                            validation_report["interpolation_report"]["fallback_used"] = True
                        else:
                            self.logger.error("❌ Interpolacja failed i fallback wyłączony!")
                            
                except Exception as e:
                    self.logger.error(f"❌ Błąd interpolacji: {str(e)}")
                    
                    # FALLBACK na exception
                    if getattr(config, 'INTERPOLATION_FALLBACK_ON_FAILURE', True):
                        self.logger.warning("⚠️ Exception w interpolacji - stosowanie fallback")
                        df_clean = self._apply_fallback_sanitization(df_clean)
                        validation_report["interpolation_report"] = {"enabled": True, "failed": True, "fallback_used": True, "error": str(e)}
                    else:
                        raise  # Re-raise exception jeśli fallback wyłączony
            else:
                self.logger.info("Interpolacja wyłączona w konfiguracji")
            
            # KROK 2: POPRAW INDEKS DATETIME
            df_clean = self._fix_datetime_index(df_clean, validation_report)
            
            # KROK 3: POSORTUJ DANE CHRONOLOGICZNIE
            df_clean = self._sort_chronologically(df_clean, validation_report)
            
            # KROK 4: USUŃ CAŁKOWITE DUPLIKATY WIERSZY
            df_clean = self._remove_duplicates(df_clean, validation_report)
            
            # KROK 5: WALIDACJA LOGICZNA OHLC
            self._validate_ohlc_logic(df_clean, validation_report)
            
            # KROK 6: NOWY - ANALIZA I OBCINANIE EKSTREMALNYCH ZMIAN OHLCV
            if config.OHLCV_VALIDATION_CONFIG.get('enabled', False):
                df_clean, validation_report['ohlcv_clipping_report'] = self._analyze_and_clip_ohlcv_extremes(
                    df_clean, validation_report
                )
            
            # KROK 7: WYKRYJ I WYPEŁNIJ LUKI CZASOWE
            df_clean = self._detect_and_fill_gaps(df_clean, validation_report, pair_name)
            
            validation_report["output_rows"] = len(df_clean)
            
            self.logger.info(
                f"Walidacja {pair_name} zakończona: "
                f"{validation_report['input_rows']:,} → {validation_report['output_rows']:,} wierszy, "
                f"duplikaty: {validation_report['duplicates_removed']}, "
                f"luki wypełnione: {validation_report['gaps_filled']}"
            )
            
            return df_clean, validation_report
            
        except Exception as e:
            self.logger.error(f"Błąd podczas walidacji {pair_name}: {str(e)}")
            validation_report["data_quality_passed"] = False
            raise
    
    def _validate_columns(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM WALIDACJI DANYCH - KROK 1: SPRAWDŹ KOLUMNY
        """
        self.logger.debug("Sprawdzam kolumny DataFrame")
        
        # Sprawdź czy wszystkie wymagane kolumny istnieją
        missing_columns = validate_dataframe_columns(df, config.REQUIRED_COLUMNS)
        
        if missing_columns:
            error_msg = f"Brakuje wymaganych kolumn: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sprawdź typy danych dla kolumn numerycznych
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(f"Kolumna {col} nie jest typu numerycznego, konwertuję")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    raise ValueError(f"Nie można skonwertować kolumny {col} na typ numeryczny")
        
        return df
    
    def _fix_datetime_index(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM WALIDACJI DANYCH - KROK 2: POPRAW INDEKS DATETIME
        ✅ TRAINING COMPATIBILITY: Standaryzuj format timestamp
        """
        self.logger.debug("Naprawiam indeks datetime")
        
        # Konwertuj kolumnę datetime na pd.DatetimeIndex
        if 'datetime' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Jeśli indeks to już datetime, użyj go
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index()
                df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
            else:
                raise ValueError("Brak kolumny datetime lub datetime index")
        
        # ✅ TRAINING COMPATIBILITY: Standaryzuj format timestamp
        if config.STANDARDIZE_TIMESTAMP_FORMAT and config.TRAINING_COMPATIBILITY_MODE:
            # Konwertuj na pandas datetime64[ns] UTC-naive format (standardowy dla ML)
            if df['datetime'].dt.tz is not None:
                # Usuń timezone info dla consistency
                df['datetime'] = df['datetime'].dt.tz_localize(None)
                self.logger.debug("Usunięto timezone info dla training compatibility")
            
            # Upewnij się że format to datetime64[ns]
            df['datetime'] = pd.to_datetime(df['datetime'])
            self.logger.debug("Timestamp standaryzowany do pandas datetime64[ns] UTC-naive")
        
        # Ustaw jako indeks DataFrame
        df.set_index('datetime', inplace=True)
        
        # Usuń duplikaty timestamp (keep='first')
        duplicated_timestamps = df.index.duplicated()
        if duplicated_timestamps.any():
            duplicate_count = duplicated_timestamps.sum()
            self.logger.warning(f"Znaleziono {duplicate_count} duplikatów timestamp, usuwam")
            df = df[~duplicated_timestamps]
            report["duplicates_removed"] += duplicate_count
        
        # Sprawdź czy timeframe = 1m (opcjonalne sprawdzenie)
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            most_common_diff = time_diffs.mode()
            if len(most_common_diff) > 0:
                expected_diff = pd.Timedelta(minutes=1)
                actual_diff = most_common_diff.iloc[0]
                if actual_diff != expected_diff:
                    self.logger.warning(
                        f"Timeframe może nie być 1m: najczęstszy odstęp = {actual_diff}"
                    )
        
        # ✅ TRAINING COMPATIBILITY: Zaloguj format timestamp
        if config.TRAINING_COMPATIBILITY_MODE:
            self.logger.debug(f"Timestamp format: {df.index.dtype}, tz-aware: {df.index.tz is not None}")
        
        return df
    
    def _sort_chronologically(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM WALIDACJI DANYCH - KROK 3: POSORTUJ DANE CHRONOLOGICZNIE
        """
        self.logger.debug("Sortuję dane chronologicznie")
        
        # Sortuj po indeksie datetime
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            self.logger.debug("Dane zostały posortowane chronologicznie")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM WALIDACJI DANYCH - KROK 4: USUŃ CAŁKOWITE DUPLIKATY WIERSZY
        """
        self.logger.debug("Usuwam całkowite duplikaty wierszy")
        
        # Sprawdź duplikaty wszystkich kolumn (nie tylko indeksu)
        initial_length = len(df)
        df = df.drop_duplicates(keep='first')
        final_length = len(df)
        
        duplicates_removed = initial_length - final_length
        if duplicates_removed > 0:
            self.logger.warning(f"Usunięto {duplicates_removed} całkowicie zduplikowanych wierszy")
            report["duplicates_removed"] += duplicates_removed
        
        return df
    
    def _validate_ohlc_logic(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """
        ALGORYTM WALIDACJI DANYCH - KROK 5: WALIDACJA LOGICZNA OHLC
        """
        self.logger.debug("Sprawdzam logiczność danych OHLC")
        
        warnings = []
        
        # Sprawdź czy high >= max(open, close)
        high_violations = (df['high'] < df[['open', 'close']].max(axis=1))
        if high_violations.any():
            violation_count = high_violations.sum()
            warnings.append(f"High < max(open,close): {violation_count} przypadków")
        
        # Sprawdź czy low <= min(open, close)
        low_violations = (df['low'] > df[['open', 'close']].min(axis=1))
        if low_violations.any():
            violation_count = low_violations.sum()
            warnings.append(f"Low > min(open,close): {violation_count} przypadków")
        
        # Sprawdź czy wszystkie ceny > 0
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if negative_prices.any():
            violation_count = negative_prices.sum()
            warnings.append(f"Ceny <= 0: {violation_count} przypadków")
        
        # Sprawdź czy volume >= 0
        negative_volume = (df['volume'] < 0)
        if negative_volume.any():
            violation_count = negative_volume.sum()
            warnings.append(f"Volume < 0: {violation_count} przypadków")
        
        # Zaloguj anomalie (bez przerywania)
        for warning in warnings:
            self.logger.warning(f"Anomalia OHLC: {warning}")
            report["validation_warnings"].append(warning)
    
        if (df['low'] > df[['open', 'high', 'close']].min(axis=1)).any() or \
           (df['high'] < df[['open', 'low', 'close']].max(axis=1)).any():
            report['validation_warnings'].append("Wykryto niespójności w logice OHLC (np. low > high)")
    
    def _analyze_and_clip_ohlcv_extremes(self, df: pd.DataFrame, report: Dict[str, Any]) -> Tuple[pd.DataFrame, dict]:
        """
        Hierarchiczny algorytm analizy i obcinania ekstremalnych zmian w danych OHLCV.
        ETAP I: Stabilizacja międzyświecowa przez interpolację i obcinanie wolumenu.
        ETAP II: Iteracyjna stabilizacja wewnątrzświecowa (high/low/close vs open).
        """
        self.logger.info("Rozpoczynam hierarchiczną analizę i obcinanie ekstremalnych zmian OHLCV")

        clipping_config = config.OHLCV_VALIDATION_CONFIG
        thresholds = clipping_config['clipping_thresholds']
        df_original = df.copy()
        df_processed = df.copy()
        
        clipping_report = {
            "enabled": True,
            "candles_interpolated_for_open_spike": 0,
            "volume_spikes_clipped": 0,
            "intra_candle_clipping_iterations": 0,
            "final_summary": {}
        }

        # --- ETAP I: STABILIZACJA MIĘDZYŚWIECOWA ---
        self.logger.info("ETAP I: Stabilizacja międzyświecowa (open_vs_prev_close, volume)")

        df_processed['prev_close'] = df_processed['close'].shift(1)
        change_open = (df_processed['open'] / df_processed['prev_close'] - 1).abs() * 100
        
        open_spike_indices = df_processed.index[change_open > thresholds['open_vs_prev_close']].tolist()
        
        if open_spike_indices:
            self.logger.warning(f"Wykryto {len(open_spike_indices)} świec z ekstremalnym skokiem 'open_vs_prev_close' > {thresholds['open_vs_prev_close']}%. Zostaną zinterpolowane.")
            interpolator = DataInterpolator()
            iloc_indices = [df_processed.index.get_loc(i) for i in open_spike_indices]
            df_processed, interp_report = interpolator.interpolate_specific_indices(
                df_processed,
                iloc_indices,
                reason=f"open_vs_prev_close spike > {thresholds['open_vs_prev_close']}%"
            )
            clipping_report['candles_interpolated_for_open_spike'] = interp_report.get('total_interpolated', 0)
            df_processed['prev_close'] = df_processed['close'].shift(1)
        
        df_processed['prev_volume'] = df_processed['volume'].shift(1).replace(0, 0.0001)
        change_volume = (df_processed['volume'] / df_processed['prev_volume']) * 100
        
        volume_spike_mask = change_volume > thresholds['volume_vs_prev_volume']
        volume_spike_count = volume_spike_mask.sum()

        if volume_spike_count > 0:
            self.logger.warning(f"Wykryto {volume_spike_count} anomalii wolumenu. Zostaną obcięte.")
            max_volume = df_processed['prev_volume'] * (thresholds['volume_vs_prev_volume'] / 100)
            df_processed.loc[volume_spike_mask, 'volume'] = max_volume[volume_spike_mask]
            clipping_report['volume_spikes_clipped'] = volume_spike_count

        self._perform_ohlcv_distribution_analysis(df_original, "Statystyki rozkładu zmian OHLCV PRZED całą operacją", clipping_config['statistics_thresholds'])

        # --- ETAP II: ITERACYJNA STABILIZACJA WEWNĄTRZŚWIECOWA ---
        self.logger.info("ETAP II: Iteracyjna stabilizacja wewnątrzświecowa (high/low/close vs open)")
        max_iterations = 5
        
        for i in range(max_iterations):
            self.logger.info(f"  Iteracja stabilizacji wewnątrzświecowej [{i+1}/{max_iterations}]...")
            
            changes = pd.DataFrame(index=df_processed.index)
            changes['high_vs_open'] = (df_processed['high'] / df_processed['open'] - 1) * 100
            changes['low_vs_open'] = (df_processed['low'] / df_processed['open'] - 1).abs() * 100
            changes['close_vs_open'] = (df_processed['close'] / df_processed['open'] - 1).abs() * 100

            high_mask = changes['high_vs_open'] > thresholds['high_vs_open']
            low_mask = changes['low_vs_open'] > thresholds['low_vs_open']
            close_mask = changes['close_vs_open'] > thresholds['close_vs_open']
            
            anomalies_mask = high_mask | low_mask | close_mask
            anomalies_count = anomalies_mask.sum()

            if anomalies_count == 0:
                self.logger.info(f"  Brak anomalii wewnątrzświecowych. Stabilność osiągnięta w iteracji {i+1}.")
                clipping_report["intra_candle_clipping_iterations"] = i + 1
                break

            self.logger.info(f"    W iteracji {i+1} znaleziono {anomalies_count} świec do wewnętrznej korekty.")

            max_high = df_processed['open'] * (1 + thresholds['high_vs_open'] / 100)
            df_processed.loc[high_mask, 'high'] = max_high[high_mask]

            min_low = df_processed['open'] * (1 - thresholds['low_vs_open'] / 100)
            df_processed.loc[low_mask, 'low'] = min_low[low_mask]

            close_change_direction = np.sign(df_processed['close'] - df_processed['open'])
            max_close_change = df_processed['open'] * (thresholds['close_vs_open'] / 100)
            df_processed.loc[close_mask, 'close'] = df_processed['open'] + (max_close_change * close_change_direction)

            self.logger.info(f"    Korygowanie logiki OHLC dla {anomalies_count} świec...")
            df_processed.loc[anomalies_mask, 'high'] = df_processed.loc[anomalies_mask, ['open', 'high', 'low', 'close']].max(axis=1)
            df_processed.loc[anomalies_mask, 'low'] = df_processed.loc[anomalies_mask, ['open', 'high', 'low', 'close']].min(axis=1)
            
            if i == max_iterations - 1:
                self.logger.warning(f"  Osiągnięto maksymalną liczbę iteracji ({max_iterations}). Proces stabilizacji przerwany.")
                clipping_report["intra_candle_clipping_iterations"] = max_iterations

        df_processed_final = df_processed.drop(columns=['prev_close', 'prev_volume'], errors='ignore')

        self.logger.info("Podsumowanie operacji obcinania OHLCV:")
        summary_report = self._create_clipping_summary(df_original, df_processed_final, list(thresholds.keys()))
        clipping_report["final_summary"] = summary_report
        for key, value in summary_report.items():
            self.logger.info(f"  - {key}: {value}")

        self._perform_ohlcv_distribution_analysis(df_processed_final, "Statystyki rozkładu zmian OHLCV PO całej operacji", clipping_config['statistics_thresholds'])
        
        return df_processed_final, clipping_report

    def _create_clipping_summary(self, df_before: pd.DataFrame, df_after: pd.DataFrame, metrics: list) -> dict:
        """Tworzy podsumowanie porównujące dane przed i po obcinaniu."""
        summary = {}
        # Upewnij się, że obie ramki danych mają te same kolumny do porównania
        common_columns = df_before.columns.intersection(df_after.columns).tolist()
        df_before_common = df_before[common_columns]
        df_after_common = df_after[common_columns]

        # Znajdź wszystkie zmodyfikowane wiersze
        diff_mask = (df_before_common != df_after_common).any(axis=1)
        summary['total_unique_candles_modified'] = diff_mask.sum()

        # Liczba zmian dla każdej metryki (kolumny)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in common_columns:
                summary[f'modified_{col}'] = (df_before_common[col] != df_after_common[col]).sum()
        
        return summary
        
    def _perform_ohlcv_distribution_analysis(self, df: pd.DataFrame, analysis_title: str, thresholds: dict):
        """
        Wykonuje i loguje analizę dystrybucji zmian procentowych OHLCV.
        """
        self.logger.info(analysis_title)
        df_analysis = df.copy()

        # Obliczanie zmian procentowych
        changes = {
            'open_vs_prev_close': (df_analysis['open'] / df_analysis['close'].shift(1) - 1).abs() * 100,
            'high_vs_open': (df_analysis['high'] / df_analysis['open'] - 1).abs() * 100,
            'low_vs_open': (df_analysis['low'] / df_analysis['open'] - 1).abs() * 100,
            'close_vs_open': (df_analysis['close'] / df_analysis['open'] - 1).abs() * 100,
            'volume_vs_prev_volume': (df_analysis['volume'] / df_analysis['volume'].shift(1).replace(0, np.nan) - 1).abs() * 100,
        }
        
        for name, series in changes.items():
            bins = thresholds.get(name, [])
            if not bins:
                continue

            self.logger.info(f"  {name}:")
            
            counts = []
            lower_bound = 0
            for upper_bound in sorted(bins):
                count = series[(series > lower_bound) & (series <= upper_bound)].count()
                if count > 0:
                    self.logger.info(f"    {lower_bound}% - {upper_bound}%: {count:,} świec")
                lower_bound = upper_bound
            
            # Ostatnia grupa "powyżej"
            last_bound = max(bins)
            count_above = series[series > last_bound].count()
            if count_above > 0:
                self.logger.info(f"    > {last_bound}%: {count_above:,} świec")

    def _detect_and_fill_gaps(self, df: pd.DataFrame, report: Dict[str, Any], 
                             pair_name: str = "") -> pd.DataFrame:
        """
        ALGORYTM WALIDACJI DANYCH - KROK 7: WYKRYJ I WYPEŁNIJ LUKI CZASOWE
        """
        self.logger.debug("Wykrywam i wypełniam luki czasowe")
        
        # Upewnij się, że indeks ma częstotliwość 1 minuty
        df_resampled = df.asfreq('1min')
        
        # Znajdź brakujące timestampy
        missing_timestamps = df_resampled.index[df_resampled['open'].isnull()]
        
        if not missing_timestamps.empty:
            report["gaps_detected"] = len(missing_timestamps)
            self.logger.warning(
                f"Wykryto {len(missing_timestamps)} brakujących świec, rozpoczynam wypełnianie"
            )
            
            # Grupuj brakujące timestampy w ciągłe luki
            gaps = self._group_missing_timestamps_into_gaps(missing_timestamps)
            
            # Wypełnij każdą lukę
            filled_rows = []
            for gap in gaps:
                gap_duration = (gap[-1] - gap[0]).total_seconds() / 60 + 1
                if gap_duration > report.get("largest_gap_minutes", 0):
                    report["largest_gap_minutes"] = gap_duration
                
                filled_rows.extend(self._fill_gap_with_bridge_strategy(df, gap))
            
            if filled_rows:
                df_filled = pd.DataFrame(filled_rows).set_index('datetime')
                df = pd.concat([df, df_filled]).sort_index()
                report["gaps_filled"] = len(filled_rows)
        
        return df
    
    def _group_missing_timestamps_into_gaps(self, missing_timestamps: pd.DatetimeIndex) -> List[List[pd.Timestamp]]:
        """
        Grupuje brakujące timestamp w ciągłe luki
        """
        if len(missing_timestamps) == 0:
            return []
        
        gaps = []
        current_gap = [missing_timestamps[0]]
        
        for i in range(1, len(missing_timestamps)):
            current_ts = missing_timestamps[i]
            previous_ts = missing_timestamps[i-1]
            
            # Jeśli odstęp to 1 minuta, dodaj do aktualnej luki
            if current_ts - previous_ts == pd.Timedelta(minutes=1):
                current_gap.append(current_ts)
            else:
                # Nowa luka
                gaps.append(current_gap)
                current_gap = [current_ts]
        
        # Dodaj ostatnią lukę
        gaps.append(current_gap)
        
        return gaps
    
    def _fill_gap_with_bridge_strategy(self, df: pd.DataFrame, gap: List[pd.Timestamp]) -> List[Dict]:
        """
        Wypełnia lukę używając algorytmu BRIDGE
        """
        if len(gap) == 0:
            return []
        
        gap_start = gap[0]
        gap_end = gap[-1]
        
        # Pobierz punkty graniczne
        last_point = self._get_point_before_gap(df, gap_start)
        next_point = self._get_point_after_gap(df, gap_end)
        
        if last_point is None or next_point is None:
            self.logger.warning(f"Nie można wypełnić luki {gap_start} - {gap_end}: brak punktów granicznych")
            return []
        
        gap_length = len(gap)
        interpolated_rows = []
        
        for i, timestamp in enumerate(gap):
            # Interpoluj liniowo OHLC
            progress = (i + 1) / (gap_length + 1)  # 0 < progress < 1
            
            interpolated_close = self._linear_interpolate(
                last_point['close'], next_point['open'], progress
            )
            
            # Dla uproszczenia: open = close poprzedniej świecy (lub interpolowana wartość)
            if i == 0:
                interpolated_open = last_point['close']
            else:
                interpolated_open = interpolated_rows[i-1]['close']
            
            # High/Low z szumem
            base_price = interpolated_close
            noise_factor = config.BRIDGE_NOISE_PCT / 100
            
            high_noise = random.uniform(0, noise_factor) * base_price
            low_noise = random.uniform(0, noise_factor) * base_price
            
            interpolated_high = max(interpolated_open, interpolated_close) + high_noise
            interpolated_low = min(interpolated_open, interpolated_close) - low_noise
            
            # Volume z losowym faktorem
            base_volume = (last_point['volume'] + next_point['volume']) / 2
            volume_factor = random.uniform(*config.BRIDGE_VOLUME_RANDOM_FACTOR)
            interpolated_volume = base_volume * volume_factor
            
            interpolated_row = {
                'datetime': timestamp,
                'open': interpolated_open,
                'high': interpolated_high,
                'low': interpolated_low,
                'close': interpolated_close,
                'volume': max(0, interpolated_volume)  # Volume nie może być ujemne
            }
            
            interpolated_rows.append(interpolated_row)
        
        return interpolated_rows
    
    def _get_point_before_gap(self, df: pd.DataFrame, gap_start: pd.Timestamp) -> Dict:
        """Znajdź ostatni punkt przed luką"""
        before_gap = df[df.index < gap_start]
        if len(before_gap) == 0:
            return None
        
        last_row = before_gap.iloc[-1]
        return {
            'open': last_row['open'],
            'high': last_row['high'],
            'low': last_row['low'],
            'close': last_row['close'],
            'volume': last_row['volume']
        }
    
    def _get_point_after_gap(self, df: pd.DataFrame, gap_end: pd.Timestamp) -> Dict:
        """Znajdź pierwszy punkt po luce"""
        after_gap = df[df.index > gap_end]
        if len(after_gap) == 0:
            return None
        
        next_row = after_gap.iloc[0]
        return {
            'open': next_row['open'],
            'high': next_row['high'],
            'low': next_row['low'],
            'close': next_row['close'],
            'volume': next_row['volume']
        }
    
    def _linear_interpolate(self, start_value: float, end_value: float, progress: float) -> float:
        """Interpolacja liniowa między dwoma wartościami"""
        return start_value + (end_value - start_value) * progress 

    def _apply_fallback_sanitization(self, df: pd.DataFrame) -> pd.DataFrame:
        """FALLBACK SANITIZATION - stare metody jako backup"""
        self.logger.info("Stosowanie fallback sanitization...")
        
        # Zastąp inf/NaN bezpiecznymi wartościami
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in df.columns:
                # Zastąp inf maksymalną sensowną wartością
                if col == 'volume':
                    max_value = df[col][np.isfinite(df[col])].quantile(0.99) * 2  # 99th percentile * 2
                else:
                    max_value = df[col][np.isfinite(df[col])].quantile(0.99) * 1.1  # 99th percentile * 1.1
                
                df[col].replace([np.inf, -np.inf], max_value, inplace=True)
                
                # Zastąp NaN median lub forward fill
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)
                
                # Zastąp zera minimalną wartością
                if col == 'volume':
                    df[col] = df[col].replace(0, 0.0001)
                else:
                    min_price = df[col][df[col] > 0].min() * 0.99
                    df[col] = df[col].replace(0, min_price)
        
        self.logger.info("✅ Fallback sanitization zastosowana")
        return df
    