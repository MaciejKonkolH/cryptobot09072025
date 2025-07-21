"""
DataFrameExtender - integracja z Freqtrade populate_indicators

🆕 NOWY ALGORYTM - Wykorzystuje raw_validated data z modułu walidacji

GŁÓWNY PUNKT INTEGRACJI:
Ta klasa zawiera funkcję extend_dataframe_for_ma43200() która będzie 
wywoływana na początku populate_indicators() w strategii.

NOWY Workflow:
1. Strategy wywołuje extend_dataframe_for_ma43200(dataframe, pair)
2. DataFrameExtender sprawdza czy istnieje {pair}_raw_validated.feather
3. Jeśli TAK → ładuje pełne dane historyczne 
4. Gap fill do teraz jeśli potrzeba (SimpleGapFiller)
5. Zwraca extended dataframe z latami danych
6. Strategy oblicza MA43200 na extended dataframe ✅

NOWY vs STARY:
STARY: ExternalDataCollector + 45-dniowa rotacja → kompleksowość
NOWY: Raw validated files + gap filling → prostota + lata danych
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import os

# SimpleGapFiller - prosty system gap filling
from .simple_gap_filler import SimpleGapFiller

import pandas as pd
from pandas import DataFrame
import numpy as np

from freqtrade.strategy import IStrategy
from freqtrade.data.converter import clean_ohlcv_dataframe
# Nowe importy
from functools import lru_cache
from freqtrade.exchange import timeframe_to_resample_freq
from freqtrade.exceptions import DependencyException


logger = logging.getLogger(__name__)

class DataFrameExtender:
    """
    🎯 NOWA ARCHITEKTURA V5 - ZGODNA Z OSTATECZNYM PLANEM 🎯
    
    Ta klasa implementuje dwumodalną logikę przygotowywania cech:
    1.  **Tryb Backtest:** Operuje na pojedynczych timestampach, symulując przepływ "świeca po świecy".
    2.  **Tryb Live:** Działa na paczkach 60 świec, z logiką synchronizacji na starcie i okresowym zapisem na dysk.
    """
    
    _instance = None
    _initialized = False
    _resolved_datadir: Optional[Path] = None

    FEATURE_COLUMNS = [
        'high_change', 'low_change', 'close_change', 'volume_change',
        'price_to_ma1440', 'price_to_ma43200', 
        'volume_to_ma1440', 'volume_to_ma43200'
    ]
    
    def __new__(cls, config: Dict = None):
        """Singleton pattern - jedna instancja na cały system"""
        if cls._instance is None:
            cls._instance = super(DataFrameExtender, cls).__new__(cls)
            cls._initialized = False
        return cls._instance
    
    def __init__(self, config: Dict = None):
        if self._initialized:
            return
            
        logger.info("🆕 DataFrameExtender V5 Initializing - Dual-Mode Architecture")
        
        self.config = config or {}
        self.ml_config = self.config.get('ml_config', {})
        self.enabled = self.ml_config.get('enabled', False)
        
        # Zmienne stanu dla trybu LIVE
        self.live_data_buffer: Dict[str, pd.DataFrame] = {}
        self.last_disk_sync_time: Dict[str, pd.Timestamp] = {}

        self.total_required_candles = self.ml_config.get('total_required_candles', 43321) # 43200 + 121
        
        # Dodaję brakujące inicjalizacje
        self.features_cache = {}
        self.initialization_status = {}
        self.startup_complete = False
        self.min_candles_for_ma43200 = 43200
        self.simple_gap_filler = SimpleGapFiller()  # Poprawiam inicjalizację
        
        self._initialized = True
        logger.info("✅ DataFrameExtender V5 Initialized")

    def set_resolved_datadir(self, datadir: Path):
        """Metoda pozwalająca strategii na wstrzyknięcie poprawnie rozwiązanej ścieżki."""
        logger.info(f"Otrzymano rozwiązaną ścieżkę do danych: {datadir}")
        self._resolved_datadir = datadir
    
    def is_enabled(self) -> bool:
        return self.enabled

    # ==============================================================================
    # GŁÓWNA METODA DLA TRYBU BACKTESTINGU
    # ==============================================================================
    def get_features_for_backtest(self, pair: str, timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """
        Generuje tabelę cech (120, 8) dla pojedynczego punktu w czasie.
        Realizuje logikę "Analityka Historycznego" z naszego planu.
        """
        # Krok 1: Wczytanie Historii
        raw_data = self._load_raw_data_cached(pair)
        if raw_data is None:
            logger.error(f"❌ {pair}: Brak danych historycznych dla timestamp {timestamp}.")
            return None
            
        # Określenie wymaganego okna
        end_date = timestamp
        start_date = end_date - pd.Timedelta(minutes=self.total_required_candles)
        
        work_df = raw_data.loc[start_date:end_date].copy()

        if len(work_df) < self.total_required_candles:
            logger.warning(f"⚠️ {pair} at {timestamp}: Niepełne dane historyczne ({len(work_df)} < {self.total_required_candles}). Pomijanie.")
            return None

        # Krok 2: Obliczenie Długich Średnich
        work_df = self._calculate_long_term_indicators(work_df, pair)

        # Krok 3: Przycięcie do Okna Pracy
        work_df = work_df.iloc[-121:]

        # Krok 4: Obliczenie 8 Cech
        work_df = self._calculate_short_term_features(work_df, pair)

        # Krok 5: Finalne Cięcie
        work_df = work_df.iloc[-120:]
        
        # Sprawdzenie ostatecznego rozmiaru
        if len(work_df) != 120:
            logger.warning(f"⚠️ {pair} at {timestamp}: Ostateczny rozmiar ramki to {len(work_df)}, a nie 120. Pomijanie.")
            return None

        # Krok 6: Zwrócenie wyniku jako tablica NumPy
        final_features = work_df[self.FEATURE_COLUMNS].values
        
        # Sprawdzenie, czy nie ma wartości NaN/inf
        if not np.isfinite(final_features).all():
            logger.warning(f"⚠️ {pair} at {timestamp}: Wykryto wartości NaN/inf w finalnych cechach. Pomijanie.")
            return None
            
        return final_features

    # ==============================================================================
    # GŁÓWNA METODA DLA TRYBU BACKTESTINGU - WERSJA OPTYMALIZOWANA
    # ==============================================================================
    
    def get_features_for_backtest_batch(self, pair: str, timestamps: List[pd.Timestamp]) -> Dict[pd.Timestamp, Optional[np.ndarray]]:
        """
        🚀 VECTORIZED WERSJA - oblicza wszystkie cechy naraz dla całego datasetu
        Dramatycznie szybsza niż poprzednia wersja batch.
        
        Strategia:
        1. Wczytaj pełne dane historyczne
        2. Oblicz MA43200 i MA1440 RAZ dla całego datasetu
        3. Oblicz WSZYSTKIE 8 cech RAZ dla całego datasetu (vectorized)
        4. Dla każdego timestamp tylko wytnij gotowe okno
        """
        logger.info(f"🚀 {pair}: Rozpoczynanie VECTORIZED processing dla {len(timestamps)} timestampów...")
        
        # Krok 1: Wczytanie pełnych danych historycznych
        raw_data = self._load_raw_data_cached(pair)
        if raw_data is None:
            logger.error(f"❌ {pair}: Brak danych historycznych.")
            return {}
        
        # Krok 2: Obliczenie MA43200 i MA1440 RAZ dla całego datasetu
        logger.info(f"💾 {pair}: Obliczanie MA43200 i MA1440 dla {len(raw_data)} świec...")
        start_time = pd.Timestamp.now()
        
        # Oblicz długie średnie dla całego datasetu
        raw_data['ma43200'] = raw_data['close'].rolling(window=43200, min_periods=43200).mean()
        raw_data['volume_ma43200'] = raw_data['volume'].rolling(window=43200, min_periods=43200).mean()
        raw_data['ma1440'] = raw_data['close'].rolling(window=1440, min_periods=1440).mean()
        raw_data['volume_ma1440'] = raw_data['volume'].rolling(window=1440, min_periods=1440).mean()
        
        ma_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"✅ {pair}: MA obliczone w {ma_time:.1f}s.")
        
        # Krok 3: Obliczenie WSZYSTKICH 8 cech RAZ dla całego datasetu (VECTORIZED!)
        logger.info(f"⚡ {pair}: Obliczanie wszystkich cech vectorized dla {len(raw_data)} świec...")
        start_time = pd.Timestamp.now()
        
        # Stosunki (vectorized)
        raw_data['price_to_ma1440'] = raw_data['close'] / raw_data['ma1440']
        raw_data['price_to_ma43200'] = raw_data['close'] / raw_data['ma43200']
        raw_data['volume_to_ma1440'] = raw_data['volume'] / raw_data['volume_ma1440']
        raw_data['volume_to_ma43200'] = raw_data['volume'] / raw_data['volume_ma43200']

        # Zmiany procentowe (vectorized)
        close_prev = raw_data['close'].shift(1)
        raw_data['high_change'] = ((raw_data['high'] - close_prev) / close_prev * 100)
        raw_data['low_change'] = ((raw_data['low'] - close_prev) / close_prev * 100)
        raw_data['close_change'] = raw_data['close'].pct_change() * 100
        raw_data['volume_change'] = raw_data['volume'].pct_change() * 100
        
        # Obsługa NaN/inf (vectorized)
        raw_data[self.FEATURE_COLUMNS] = raw_data[self.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        raw_data[self.FEATURE_COLUMNS] = raw_data[self.FEATURE_COLUMNS].fillna(0)
        
        features_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"✅ {pair}: Wszystkie cechy obliczone w {features_time:.1f}s. Rozpoczynanie wycinania okien...")
        
        # Krok 4: Szybkie wycinanie okien z pre-obliczonymi cechami
        all_features = {}
        processed_count = 0
        failed_count = 0
        total_count = len(timestamps)
        start_time_total = pd.Timestamp.now()
        last_log_time = start_time_total
        
        for i, timestamp in enumerate(timestamps):
            try:
                # CORRECTED LOGIC: Shift the window back by one minute to prevent lookahead bias.
                # For a decision at timestamp T, we use the feature window ending at T-1.
                target_timestamp = timestamp - pd.Timedelta(minutes=1)
                
                # Find the position of the target timestamp directly in the index.
                end_idx_pos = raw_data.index.get_loc(target_timestamp)
                
                # Determine the starting position of the 120-candle window.
                start_idx_pos = end_idx_pos - 120 + 1
                
                if start_idx_pos < 0:
                    all_features[timestamp] = None
                    failed_count += 1
                    continue
                
                # Wytnij okno za pomocą iloc - jest to znacznie szybsze
                window_data = raw_data.iloc[start_idx_pos:end_idx_pos + 1]
                
                if len(window_data) != 120:
                    all_features[timestamp] = None
                    failed_count += 1
                    continue
                    
                # Konwersja do NumPy
                final_features = window_data[self.FEATURE_COLUMNS].values
                
                if not np.isfinite(final_features).all():
                    all_features[timestamp] = None
                    failed_count += 1
                else:
                    all_features[timestamp] = final_features
                    processed_count += 1

            except KeyError:
                all_features[timestamp] = None
                failed_count += 1
            
            # Logowanie postępu co określony czas lub co X iteracji
            now = pd.Timestamp.now()
            if (now - last_log_time).total_seconds() > 20 or (i + 1) % 50000 == 0:
                elapsed = (now - start_time_total).total_seconds()
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"📊 {pair}: Przetworzono {i + 1}/{total_count} ({(i + 1) / total_count:.1%}) - {speed:.0f} timestamps/s")
                last_log_time = now

        total_time = (pd.Timestamp.now() - start_time_total).total_seconds()
        avg_speed = total_count / total_time if total_time > 0 else 0

        logger.info(f"✅ {pair}: VECTORIZED processing zakończony w {total_time:.1f}s ({avg_speed:.0f} timestamps/s)")
        logger.info(f"📊 {pair}: Sukces: {processed_count}/{total_count} ({processed_count / total_count:.1%}), Błędy: {failed_count}")
        
        return all_features

    # ==============================================================================
    # METODY DLA TRYBU LIVE (WERSJE ROBOCZE)
    # ==============================================================================
    def initialize_live_mode_for_pair(self, pair: str):
        """
        Etap 1: Inicjalizacja i Synchronizacja (Jednorazowo na starcie).
        Ta metoda musi zostać zaimplementowana.
        """
        logger.info(f"🚀 {pair}: Rozpoczynanie synchronizacji w trybie LIVE...")
        # TODO:
        # 1. Wczytaj historię z pliku do self.live_data_buffer[pair]
        # 2. Wykryj "lukę"
        # 3. Pobierz brakujące dane z giełdy
        # 4. Zaktualizuj bufor w pamięci i zapisz na dysk
        # 5. Ustaw self.last_disk_sync_time[pair]
        logger.warning(f"⚠️ {pair}: Funkcjonalność trybu LIVE nie jest jeszcze zaimplementowana.")
        pass

    def process_live_data(self, pair: str, new_candles_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Etap 2: Normalna Praca (Pętla co minutę).
        Przetwarza paczkę 60 świec.
        """
        # TODO: Zaimplementować pełną logikę z naszego planu.
        logger.warning(f"⚠️ {pair}: Funkcjonalność trybu LIVE nie jest jeszcze zaimplementowana.")
        return None

    # ==============================================================================
    # FUNKCJE POMOCNICZE (WSPÓLNE DLA OBU TRYBÓW)
    # ==============================================================================
    def _calculate_long_term_indicators(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Oblicza tylko wskaźniki wymagające długiej historii."""
        try:
            dataframe['ma1440'] = dataframe['close'].rolling(window=1440, min_periods=1440).mean()
            dataframe['volume_ma1440'] = dataframe['volume'].rolling(window=1440, min_periods=1440).mean()
            dataframe['ma43200'] = dataframe['close'].rolling(window=43200, min_periods=43200).mean()
            dataframe['volume_ma43200'] = dataframe['volume'].rolling(window=43200, min_periods=43200).mean()
            return dataframe
        except Exception as e:
            logger.error(f"❌ {pair}: Error calculating long-term indicators: {e}")
            return dataframe # Zwróć oryginalną, aby uniknąć crashu

    def _calculate_short_term_features(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Oblicza cechy, które zależą od wskaźników długoterminowych i/lub krótkiej historii."""
        try:
            # Stosunki
            dataframe['price_to_ma1440'] = dataframe['close'] / dataframe['ma1440']
            dataframe['price_to_ma43200'] = dataframe['close'] / dataframe['ma43200']
            dataframe['volume_to_ma1440'] = dataframe['volume'] / dataframe['volume_ma1440']
            dataframe['volume_to_ma43200'] = dataframe['volume'] / dataframe['volume_ma43200']

            # Zmiany procentowe
            close_prev = dataframe['close'].shift(1)
            dataframe['high_change'] = ((dataframe['high'] - close_prev) / close_prev * 100)
            dataframe['low_change'] = ((dataframe['low'] - close_prev) / close_prev * 100)
            dataframe['close_change'] = dataframe['close'].pct_change() * 100
            dataframe['volume_change'] = dataframe['volume'].pct_change() * 100
            return dataframe
        except Exception as e:
            logger.error(f"❌ {pair}: Error calculating short-term features: {e}")
            return dataframe # Zwróć oryginalną, aby uniknąć crashu

    @lru_cache(maxsize=10)
    def _load_raw_data_cached(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Wczytuje i cache'uje plik _raw_validated dla danej pary.
        """
        logger.info(f"💾 {pair}: Loading and caching raw_validated file...")
        filepath = self._get_raw_validated_path(pair)
        if not filepath.exists():
            logger.error(f"❌ KRYTYCZNY BŁĄD: Nie znaleziono pliku danych historycznych: {filepath}")
            logger.error("Upewnij się, że plik `_raw_validated.feather` istnieje i został wygenerowany przez moduł `validation_and_labeling`.")
            return None
        
        try:
            data = pd.read_feather(filepath)
            data['date'] = pd.to_datetime(data['date'], utc=True)
            data.set_index('date', inplace=True)
            
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()

            logger.info(f"✅ {pair}: Raw data cached successfully ({len(data)} candles).")
            return data
        except Exception as e:
            logger.error(f"❌ Błąd podczas wczytywania pliku {filepath}: {e}")
            return None

    def _get_raw_validated_path(self, pair: str) -> Path:
        """Zwraca ścieżkę do pliku raw_validated dla danej pary."""
        if self._resolved_datadir is None:
            raise DependencyException(
                "Ścieżka do danych (datadir) nie została zainicjalizowana. "
                "Upewnij się, że strategia wywołuje `set_resolved_datadir()`."
            )

        base_path = self._resolved_datadir
        # KOREKTA: Zamieniamy ':' na '_', a nie usuwamy go.
        normalized_pair = pair.replace('/', '_').replace(':', '_')
        filename = f"{normalized_pair}-1m-futures.feather"
        full_path = base_path / 'futures' / filename
        
        return full_path

    async def initialize_for_pairs(self, pairs: List[str]) -> Dict[str, bool]:
        """
        Centralna metoda inicjalizacji dla wielu par.
        """
        logger.info(f"🚀 Inicjalizacja DataFrameExtendera dla {len(pairs)} par...")
        results = {}
        for pair in pairs:
            # W tym miejscu możemy dodać logikę wczytywania/cache'owania
            # danych per para, jeśli to konieczne.
            # Na razie, po prostu oznaczamy jako gotowe.
            self.initialization_status[pair] = True
            results[pair] = True
            logger.info(f"✅ Para {pair} gotowa do przetwarzania.")
        
        self.startup_complete = True
        logger.info("✅ DataFrameExtender pomyślnie zainicjalizowany dla wszystkich par.")
        return results

    def get_initialization_status(self) -> Dict[str, bool]:
        """Zwraca status inicjalizacji dla wszystkich par."""
        return self.initialization_status

    def start_realtime_sync(self, pairs: List[str]):
        """Real-time sync nie jest potrzebny - gap filling on demand"""
        if not self.is_enabled():
            return
        
        logger.info(f"🆕 NEW: Real-time sync nie jest potrzebny w nowym systemie")
        logger.info("Gap filling dzieje się on-demand podczas każdego wywołania extend_dataframe_for_ma43200")
    
    def stop_realtime_sync(self):
        """Real-time sync nie jest potrzebny - gap filling on demand"""
        if not self.is_enabled():
            return
            
        logger.info("🆕 NEW: Real-time sync nie jest używany w nowym systemie")
    
    def get_system_status(self) -> Dict:
        """Zwraca status nowego systemu raw_validated + gap filling"""
        if not self.is_enabled():
            return {'enabled': False}
        
        return {
            'enabled': True,
            'system_type': 'raw_validated_plus_gap_filling',
            'startup_complete': self.startup_complete,
            'initialization_status': self.initialization_status,
            'components': {
                'simple_gap_filler': 'active',
                'raw_validated_files': 'file_based',
                'real_time_sync': 'on_demand_gap_filling'
            }
        }
    
    async def manual_refresh_pair(self, pair: str) -> bool:
        """Manualny gap fill dla pary"""
        if not self.is_enabled():
            return False
        
        logger.info(f"🆕 Manual gap fill for {pair}")
        
        try:
            # Sprawdź czy istnieje plik raw_validated
            raw_validated_path = self._get_raw_validated_path(pair)
            if not raw_validated_path.exists():
                logger.error(f"❌ {pair}: Brak pliku raw_validated")
                return False
            
            # Załaduj dane i sprawdź ostatnią datę
            historical_data = pd.read_feather(raw_validated_path)
            if 'datetime' in historical_data.columns:
                historical_data.rename(columns={'datetime': 'date'}, inplace=True)
            
            last_timestamp = historical_data['date'].iloc[-1]
            
            # Force gap fill
            gap_data = self.simple_gap_filler.fill_gap_to_now(pair, last_timestamp)
            
            if not gap_data.empty:
                # Combine i zapisz
                combined_data = pd.concat([historical_data, gap_data])
                # Sortuj według kolumny 'date' zamiast index
                combined_data = combined_data.sort_values('date')
                # Zapisz bez reset_index
                combined_data.to_feather(raw_validated_path)
                
                logger.info(f"✅ Manual gap fill successful for {pair}: dodano {len(gap_data)} świec")
                return True
            else:
                logger.info(f"ℹ️ {pair}: Brak gap-a do wypełnienia")
                return True
            
        except Exception as e:
            logger.error(f"❌ Error in manual gap fill for {pair}: {e}")
            return False

# Globalny dostęp do instancji
_extender_instance = None

def get_dataframe_extender(config: Dict = None) -> 'DataFrameExtender':
    """Zwraca instancję singletona DataFrameExtender."""
    global _extender_instance
    if _extender_instance is None:
        _extender_instance = DataFrameExtender(config)
    return _extender_instance 