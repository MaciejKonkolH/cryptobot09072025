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
        🚀 NOWA WERSJA - używa danych z raw_validated.feather z 37 cechami
        Zamiast generować własne cechy, używa pre-obliczonych cech z feature_calculator_ohlc_snapshot
        """
        logger.info(f"🚀 {pair}: Rozpoczynanie VECTORIZED processing dla {len(timestamps)} timestampów...")
        
        # Krok 1: Wczytanie danych z raw_validated.feather
        raw_data = self._load_raw_data_cached(pair)
        if raw_data is None:
            logger.error(f"❌ {pair}: Brak danych historycznych.")
            return {}
        
        # Krok 2: Sprawdzenie czy dane zawierają wymagane cechy
        required_features = [
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
            'market_trend_strength', 'market_trend_direction', 'market_choppiness', 'bollinger_band_width', 'market_regime',
            'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend', 'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend'
        ]
        
        missing_features = [f for f in required_features if f not in raw_data.columns]
        if missing_features:
            logger.error(f"❌ {pair}: Brakujące cechy: {missing_features}")
            return {}
        
        logger.info(f"✅ {pair}: Wszystkie 37 cech dostępne. Rozpoczynanie przetwarzania...")
        
        # Krok 3: Przetwarzanie timestampów
        all_features = {}
        processed_count = 0
        failed_count = 0
        total_count = len(timestamps)
        start_time_total = pd.Timestamp.now()
        last_log_time = start_time_total
        
        for i, timestamp in enumerate(timestamps):
            try:
                # Znajdź wiersz dla tego timestampu
                if timestamp in raw_data.index:
                    row_data = raw_data.loc[timestamp]
                    
                    # Wybierz tylko wymagane cechy
                    features = row_data[required_features].values
                    
                    # Konwertuj na float64 jeśli to nie jest już float
                    features = features.astype(np.float64)
                    
                    # Sprawdź czy wszystkie cechy są skończone
                    if np.isfinite(features).all():
                        all_features[timestamp] = features
                        processed_count += 1
                    else:
                        all_features[timestamp] = None
                        failed_count += 1
                else:
                    all_features[timestamp] = None
                    failed_count += 1
                    if i < 5:  # Debug tylko pierwszych 5 błędów
                        logger.warning(f"🔍 DEBUG {pair}: Timestamp {timestamp} nie znaleziony w raw_data.index")
                        logger.warning(f"🔍 DEBUG {pair}: raw_data.index range: {raw_data.index[0]} do {raw_data.index[-1]}")
                        logger.warning(f"🔍 DEBUG {pair}: raw_data.index type: {type(raw_data.index[0])}")
                        logger.warning(f"🔍 DEBUG {pair}: timestamp type: {type(timestamp)}")
                    
            except Exception as e:
                all_features[timestamp] = None
                failed_count += 1
                if i < 5:  # Debug tylko pierwszych 5 błędów
                    logger.warning(f"❌ {pair}: Błąd dla timestamp {timestamp}: {e}")
            
            # Logowanie postępu
            current_time = pd.Timestamp.now()
            if (current_time - last_log_time).total_seconds() > 5 or i % max(1, total_count // 10) == 0:
                progress_pct = (i / total_count) * 100
                elapsed_time = (current_time - start_time_total).total_seconds()
                estimated_total = elapsed_time / (i + 1) * total_count if i > 0 else 0
                remaining_time = estimated_total - elapsed_time
                
                logger.info(f"📊 {pair}: Przetworzono {processed_count}/{total_count} ({progress_pct:.1f}%) - "
                           f"Czas: {elapsed_time:.0f}s, Pozostało: {remaining_time:.0f}s")
                last_log_time = current_time
        
        total_time = (pd.Timestamp.now() - start_time_total).total_seconds()
        logger.info(f"✅ {pair}: VECTORIZED processing zakończony w {total_time:.1f}s ({total_count/total_time:.0f} timestamps/s)")
        logger.info(f"📊 {pair}: Sukces: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%), Błędy: {failed_count}")
        
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
        Wczytuje i cache'uje plik z cechami dla danej pary.
        """
        logger.info(f"💾 {pair}: Loading and caching raw_validated file...")
        filepath = self._get_raw_validated_path(pair)
        if not filepath.exists():
            logger.error(f"❌ KRYTYCZNY BŁĄD: Nie znaleziono pliku z cechami: {filepath}")
            logger.error("Upewnij się, że plik `_raw_validated.feather` istnieje i został skopiowany z feature_calculator_ohlc_snapshot.")
            return None
        
        try:
            data = pd.read_feather(filepath)
            
            # Sprawdź czy kolumna timestamp istnieje
            if 'timestamp' in data.columns:
                # Konwertuj timestamp na UTC timezone
                data['date'] = pd.to_datetime(data['timestamp'], utc=True)
                data.set_index('date', inplace=True)
            elif 'date' in data.columns:
                # Konwertuj date na UTC timezone
                data['date'] = pd.to_datetime(data['date'], utc=True)
                data.set_index('date', inplace=True)
            else:
                logger.error(f"❌ Brak kolumny timestamp lub date w pliku {filepath}")
                return None
            
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

        # NOWA LOGIKA: Używamy pliku z cechami z feature_calculator_ohlc_snapshot
        base_path = Path(self.config.get('user_data_dir', 'user_data')) / 'strategies' / 'inputs'
        normalized_pair = pair.replace('/', '_').replace(':', '_')
        filename = f"{normalized_pair}_raw_validated.feather"
        full_path = base_path / filename
        
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