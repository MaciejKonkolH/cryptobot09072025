"""
Data Interpolator - Algorytm interpolacji zepsutych danych
Eliminuje wartości 0, inf, NaN które powodują błędy treningowe
Bazuje na: memory-bank/Algorytmy/Algorytm_uzupelniania_danaych_historycznych.md
"""
import logging
import pandas as pd
import numpy as np
import random
import time
from typing import Dict, Any, List, Tuple, Optional

# Obsługa importów
try:
    from . import config
    from .utils import setup_logging
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from utils import setup_logging


class DataInterpolator:
    """Algorytm interpolacji zepsutych danych"""
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.DataInterpolator")
        self.noise_pct = getattr(config, 'NOISE_PERCENTAGE', 2.0)
        self.max_iterations = getattr(config, 'MAX_INTERPOLATION_ITERATIONS', 3)
        self.min_valid_volume = getattr(config, 'MIN_VALID_VOLUME', 0.0001)
        
        self.logger.info("DataInterpolator initialized")
        
    def interpolate_corrupted_data(self, df: pd.DataFrame, pair_name: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """GŁÓWNA METODA - Interpolacja zepsutych danych"""
        start_time = time.time()  # START TIMER
        timeout = getattr(config, 'INTERPOLATION_MAX_PROCESSING_TIME', 300)  # 5 minut default
        
        self.logger.info(f"🔧 Rozpoczynam interpolację dla {pair_name} ({len(df):,} świec)")
        
        # Przygotuj raport
        report = {
            "pair": pair_name,
            "input_rows": len(df),
            "corrupted_candles_detected": 0,
            "total_interpolated": 0,
            "iterations_performed": 0,
            "success": False,
            "warnings": []
        }
        
        # Sprawdź czy dataset nie jest za duży
        corrupted_percentage_limit = getattr(config, 'INTERPOLATION_MAX_CORRUPTED_PERCENTAGE', 50)
        
        # Kopia DataFrame do modyfikacji
        df_fixed = df.copy()
        
        # Iteracyjna naprawa (maksimum 3 iteracje)
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            self.logger.info(f"🔄 Iteracja {iteration}/{self.max_iterations}")
            
            # Sprawdź timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                warning_msg = f"⏰ Timeout ({timeout}s) - przerywam interpolację"
                self.logger.warning(warning_msg)
                report["warnings"].append(warning_msg)
                break
            
            # IDENTYFIKUJ zepsute świece
            self.logger.info(f"🔍 Skanowanie świec - iteracja {iteration}")
            corrupted_indices = self._identify_corrupted_candles(df_fixed)
            
            if not corrupted_indices:
                self.logger.info(f"✅ Brak zepsutych świec w iteracji {iteration}")
                break
                
            # Sprawdź czy nie za dużo zepsutych danych
            corrupted_percentage = (len(corrupted_indices) / len(df_fixed)) * 100
            if corrupted_percentage > corrupted_percentage_limit:
                warning_msg = f"⚠️ Zbyt dużo zepsutych danych: {corrupted_percentage:.1f}% > {corrupted_percentage_limit}%"
                self.logger.warning(warning_msg)
                report["warnings"].append(warning_msg)
                break
                
            self.logger.info(f"🚨 Iteracja {iteration}: Znaleziono {len(corrupted_indices)} zepsutych świec ({corrupted_percentage:.1f}%)")
            
            if iteration == 1:
                report["corrupted_candles_detected"] = len(corrupted_indices)
            
            # GRUPUJ w bloki ciągłe
            corrupted_blocks = self._group_corrupted_into_blocks(corrupted_indices)
            self.logger.info(f"📦 Pogrupowano w {len(corrupted_blocks)} bloków")
            
            # NAPRAW każdy blok
            fixes_in_iteration = 0
            for block_idx, block in enumerate(corrupted_blocks):
                if block_idx % 100 == 0 and block_idx > 0:
                    self.logger.info(f"🔧 Naprawiam blok {block_idx}/{len(corrupted_blocks)}")
                
                fixed_count = self._fix_corrupted_block(df_fixed, block)
                fixes_in_iteration += fixed_count
            
            report["total_interpolated"] += fixes_in_iteration
            report["iterations_performed"] = iteration
            
            iteration_time = time.time() - iteration_start
            self.logger.info(f"✅ Iteracja {iteration}: Naprawiono {fixes_in_iteration} świec w {iteration_time:.2f}s")
        
        # Finalne sprawdzenie - TYLKO jeśli wykonano wszystkie iteracje
        if report["iterations_performed"] >= self.max_iterations:
            final_corrupted = self._identify_corrupted_candles(df_fixed)
            if final_corrupted:
                warning_msg = f"⚠️ Po {self.max_iterations} iteracjach nadal {len(final_corrupted)} zepsutych świec"
                self.logger.warning(warning_msg)
                report["warnings"].append(warning_msg)
            else:
                self.logger.info(f"✅ Interpolacja zakończona pomyślnie")
                report["success"] = True
        else:
            # Iteracje zakończone wcześniej z powodu braku problemów
            self.logger.info(f"✅ Interpolacja zakończona pomyślnie - brak problemów po {report['iterations_performed']} iteracjach")
            report["success"] = True
        
        self.logger.info(f"📊 {pair_name}: {report['total_interpolated']} świec interpolowanych w {report['iterations_performed']} iteracjach")
        
        # WALIDACJA KOŃCOWA (NOWE)
        validation_report = self._validate_interpolation_success(df_fixed, pair_name)
        report["final_validation"] = validation_report
        
        if not validation_report["success"]:
            # Jeśli nadal są problemy, oznacz jako partial success
            report["success"] = False
            report["warnings"].append("Interpolacja nie usunęła wszystkich problemów!")
        
        end_time = time.time()
        processing_time = end_time - start_time
        report["processing_time_seconds"] = processing_time
        
        # Log performance
        if processing_time > 10.0:  # Warning jeśli > 10 sekund
            self.logger.warning(f"⏱️ Interpolacja {pair_name}: {processing_time:.2f}s (SLOW!)")
        else:
            self.logger.info(f"⏱️ Interpolacja {pair_name}: {processing_time:.2f}s")
        
        # Throughput
        rows_per_second = len(df) / processing_time if processing_time > 0 else 0
        self.logger.info(f"📈 Throughput: {rows_per_second:.0f} świec/sekunda")
        
        return df_fixed, report
    
    def interpolate_specific_indices(self, df: pd.DataFrame, indices_to_fix: List[int], reason: str = "External request") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Interpoluje świece na podstawie jawnie podanej listy indeksów.
        Używane do naprawy anomalii wykrytych przez zewnętrzne moduły (np. DataValidator).
        """
        start_time = time.time()
        self.logger.info(f"🔧 Rozpoczynam interpolację {len(indices_to_fix)} świec na żądanie (powód: {reason})")
        
        df_fixed = df.copy()
        report = {
            "reason": reason,
            "indices_provided": len(indices_to_fix),
            "total_interpolated": 0,
            "success": False
        }

        if not indices_to_fix:
            self.logger.info("Brak indeksów do interpolacji. Zwracam oryginalne dane.")
            report["success"] = True
            return df_fixed, report

        # GRUPUJ w bloki ciągłe
        corrupted_blocks = self._group_corrupted_into_blocks(indices_to_fix)
        self.logger.info(f"📦 Pogrupowano w {len(corrupted_blocks)} bloków do interpolacji.")

        # NAPRAW każdy blok
        fixes_in_run = 0
        for block_idx, block in enumerate(corrupted_blocks):
            if block_idx % 100 == 0 and block_idx > 0:
                self.logger.info(f"🔧 Interpoluję blok {block_idx}/{len(corrupted_blocks)}")
            
            fixed_count = self._fix_corrupted_block(df_fixed, block)
            fixes_in_run += fixed_count
        
        report["total_interpolated"] = fixes_in_run
        report["success"] = True  # Zakładamy sukces, jeśli proces przeszedł
        
        end_time = time.time()
        processing_time = end_time - start_time
        report["processing_time_seconds"] = processing_time

        self.logger.info(f"✅ Interpolacja na żądanie zakończona w {processing_time:.2f}s. Zmieniono {fixes_in_run} świec.")

        return df_fixed, report

    def _is_ohlc_valid(self, candle: pd.Series) -> bool:
        """Sprawdza, czy ceny OHLC w świecy są poprawne i logiczne."""
        try:
            # Sprawdź, czy ceny są dodatnie i skończone
            for col in ['open', 'high', 'low', 'close']:
                price = candle[col]
                if price <= 0 or np.isinf(price) or np.isnan(price):
                    return False
            
            # Sprawdź logikę OHLC
            if candle['high'] < max(candle['open'], candle['close']):
                return False
            if candle['low'] > min(candle['open'], candle['close']):
                return False
                
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def _identify_corrupted_candles(self, df: pd.DataFrame) -> List[int]:
        """IDENTYFIKACJA ZEPSUTYCH ŚWIEC - OPTIMIZED VERSION"""
        total_rows = len(df)
        self.logger.info(f"Skanowanie {total_rows:,} świec w poszukiwaniu problemów...")
        
        # --- Diagnoza wektorowa ---
        volume_invalid = (df['volume'] <= self.min_valid_volume) | np.isinf(df['volume']) | np.isnan(df['volume'])
        
        price_invalid_flags = pd.DataFrame(index=df.index)
        for col in ['open', 'high', 'low', 'close']:
            price_invalid_flags[col] = (df[col] <= 0) | np.isinf(df[col]) | np.isnan(df[col])
        any_price_invalid = price_invalid_flags.any(axis=1)

        ohlc_logic_invalid = (df['high'] < np.maximum(df['open'], df['close'])) | \
                             (df['low'] > np.minimum(df['open'], df['close']))
        
        # Połącz wszystkie warunki
        all_invalid = volume_invalid | any_price_invalid | ohlc_logic_invalid
        
        corrupted_indices = df.index[all_invalid].tolist()
        
        # --- Logowanie i raportowanie postępu ---
        if total_rows > 100000:
            progress_step = max(1, total_rows // 20)  # Raportuj co 5%
            for i in range(0, total_rows, progress_step):
                if i > 0:
                    progress_pct = (i / total_rows) * 100
                    self.logger.info(f"📊 Postęp skanowania: {progress_pct:.1f}% ({i:,}/{total_rows:,} świec)")
        
        if corrupted_indices:
            self.logger.info(f"🚨 Znaleziono {len(corrupted_indices)} zepsutych świec z {total_rows:,} total ({len(corrupted_indices)/total_rows*100:.2f}%)")
        else:
            self.logger.info("✅ Wszystkie świece są prawidłowe")
            
        return corrupted_indices

    def _group_corrupted_into_blocks(self, corrupted_indices: List[int]) -> List[List[int]]:
        """Grupuje ciągłe indeksy zepsutych świec w bloki"""
        if not corrupted_indices:
            return []
        
        blocks = []
        current_block = [corrupted_indices[0]]
        
        for i in range(1, len(corrupted_indices)):
            if corrupted_indices[i] == corrupted_indices[i-1] + 1:
                current_block.append(corrupted_indices[i])
            else:
                blocks.append(current_block)
                current_block = [corrupted_indices[i]]
        
        blocks.append(current_block)
        return blocks

    def _fix_corrupted_block(self, df: pd.DataFrame, block: List[int]) -> int:
        """
        "CHIRURGICZNA" NAPRAWA BLOKU: Iteruje przez każdą świecę w bloku
        i naprawia tylko te wartości, które są uszkodzone.
        """
        left_index, right_index = self._find_nearest_valid_candles(df, block)
        fixed_count = 0

        for candle_index in block:
            candle = df.iloc[candle_index]
            
            volume_is_invalid = candle['volume'] <= self.min_valid_volume
            ohlc_is_valid = self._is_ohlc_valid(candle)

            if not ohlc_is_valid:
                # Scenariusz A: OHLC jest zepsute. Napraw całą świecę.
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    self._interpolate_value(df, candle_index, col, left_index, right_index)
                fixed_count += 1
                
            elif volume_is_invalid:
                # Scenariusz B: OHLC jest OK, ale wolumen jest zły. Napraw tylko wolumen.
                self._interpolate_value(df, candle_index, 'volume', left_index, right_index)
                fixed_count += 1
        
        # Po naprawie wartości, upewnij się, że logika OHLC jest spójna
        for candle_index in block:
            self._fix_ohlc_logic(df, candle_index)
            
        return fixed_count

    def _find_nearest_valid_candles(self, df: pd.DataFrame, block: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """Znajduje indeksy najbliższych prawidłowych świec otaczających blok"""
        start_block_index = block[0]
        end_block_index = block[-1]
        
        # Szukaj na lewo
        left_index = None
        for i in range(start_block_index - 1, -1, -1):
            if i not in block and self._is_ohlc_valid(df.iloc[i]) and df.iloc[i]['volume'] > self.min_valid_volume:
                left_index = i
                break
        
        # Szukaj na prawo
        right_index = None
        for i in range(end_block_index + 1, len(df)):
             if i not in block and self._is_ohlc_valid(df.iloc[i]) and df.iloc[i]['volume'] > self.min_valid_volume:
                right_index = i
                break
                
        return left_index, right_index

    def _interpolate_value(self, df: pd.DataFrame, idx: int, col: str, left_idx: Optional[int], right_idx: Optional[int]):
        """
        Interpoluje, forward-filluje lub backward-filluje pojedynczą wartość
        w danej kolumnie i indeksie.
        """
        # Scenariusz 1: Pełna interpolacja liniowa
        if left_idx is not None and right_idx is not None:
            left_val = df.at[df.index[left_idx], col]
            right_val = df.at[df.index[right_idx], col]
            
            total_distance = right_idx - left_idx
            current_distance = idx - left_idx
            
            if total_distance > 0:
                step = (right_val - left_val) / total_distance
                interpolated_val = left_val + (step * current_distance)
                
                if col != 'volume':
                    interpolated_val = self._add_realistic_noise(interpolated_val)
                df.at[df.index[idx], col] = interpolated_val
            else:
                 df.at[df.index[idx], col] = left_val

        # Scenariusz 2: Backward fill (tylko prawa granica)
        elif right_idx is not None:
            val = df.at[df.index[right_idx], col]
            df.at[df.index[idx], col] = val

        # Scenariusz 3: Forward fill (tylko lewa granica)
        elif left_idx is not None:
            val = df.at[df.index[left_idx], col]
            df.at[df.index[idx], col] = val
        
        else:
            self.logger.warning(f"Brak granic do interpolacji wartości dla {col} w indeksie {idx}. Wartość pozostaje bez zmian.")

    def _add_realistic_noise(self, value: float) -> float:
        """Dodaje niewielki, realistyczny szum do interpolowanych cen"""
        if value > 0:
            noise = value * (self.noise_pct / 100) * (random.uniform(-0.5, 0.5))
            return max(0, value + noise)
        return value

    def _fix_ohlc_logic(self, df: pd.DataFrame, candle_index: int) -> None:
        """
        Naprawia logikę OHLC dla pojedynczej świecy po interpolacji,
        upewniając się, że high jest max, a low jest min.
        """
        candle = df.iloc[candle_index]
        open_val, high_val, low_val, close_val = candle['open'], candle['high'], candle['low'], candle['close']
        
        # Upewnij się, że high jest najwyższą, a low najniższą wartością
        correct_high = max(open_val, high_val, low_val, close_val)
        correct_low = min(open_val, high_val, low_val, close_val)
        
        df.at[df.index[candle_index], 'high'] = correct_high
        df.at[df.index[candle_index], 'low'] = correct_low
        
        # Upewnij się, że wolumen nie jest ujemny
        if df.at[df.index[candle_index], 'volume'] < 0:
            df.at[df.index[candle_index], 'volume'] = 0
            
    def _validate_interpolation_success(self, df: pd.DataFrame, pair_name: str = "") -> Dict[str, Any]:
        """FINALNA WALIDACJA - sprawdza, czy po wszystkich operacjach nie ma już zepsutych świec"""
        self.logger.info(f"✅ Walidacja {pair_name}: Sprawdzam ostateczny wynik interpolacji...")
        
        remaining_corrupted = self._identify_corrupted_candles(df)
        
        report = {"success": not remaining_corrupted, "remaining_corrupted": len(remaining_corrupted)}
        
        if not report["success"]:
            self.logger.warning(f"🚨 Walidacja {pair_name}: Nadal istnieje {len(remaining_corrupted)} zepsutych świec!")
        else:
            self.logger.info(f"✅ Walidacja {pair_name}: Wszystkie problemy usunięte!")
            
        return report 