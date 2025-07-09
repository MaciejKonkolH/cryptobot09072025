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


logger = logging.getLogger(__name__)

class DataFrameExtender:
    """
    🎯 NOWA ARCHITEKTURA V4 - Feature Service 🎯
    
    Zamiast zwracać gigantyczne DataFrame, ta klasa działa jako serwis,
    który na żądanie dostarcza gotowe, obliczone cechy dla konkretnego punktu w czasie.
    """
    
    _instance = None
    _initialized = False

    # 🚀 NOWOŚĆ: Definicja kolumn z cechami w jednym miejscu
    FEATURE_COLUMNS = [
        'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200',
        'high_change', 'low_change', 'close_change', 'volume_change'
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
            
        logger.info("🆕 NEW DataFrameExtender V4 Initializing - Feature Service Mode")
        
        self.config = config or {}
        self.ml_config = self.config.get('ml_config', {})
        self.enabled = self.ml_config.get('enabled', False)
        
        # 🔧 NOWE: Cache dla obliczonych cech
        # Klucz: (pair, timestamp), Wartość: dict z 8 cechami
        self.features_cache = {}

        # Ustawienia bufora
        self.total_required_candles = self.ml_config.get('total_required_candles', 43300)
        self.min_candles_for_ma43200 = self.total_required_candles - 120 # Dopuszczalny margines
        
        self.initialization_status = {}
        self._initialized = True
        logger.info("✅ DataFrameExtender V4 Initialized")
    
    def is_enabled(self) -> bool:
        return self.enabled

    def extend_and_calculate_features(self, df_backtest: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        🚀 NOWA WERSJA ZGODNA Z PRECYZYJNYM, DWUETAPOWYM PLANEM 🚀
        Pełna kontrola nad każdym krokiem, aby zagwarantować poprawność wyniku.
        """
        logger.info(f"🚀 {pair}: Starting PRECISE, user-defined feature calculation workflow...")

        # Krok 1: Otrzymanie Czystych Danych (zgodnie z planem, startup_candle_count=0)
        # Krok 2: Dodanie Pełnej Historii
        raw_data = self._load_raw_data_cached(pair)
        if raw_data is None:
            logger.error(f"❌ {pair}: Cannot perform backtest calculation without raw_data.")
            return df_backtest.assign(**{col: 0.0 for col in self.FEATURE_COLUMNS})

        first_candle_date = df_backtest['date'].iloc[0]
        history_start_date = first_candle_date - pd.Timedelta(minutes=self.total_required_candles)
        
        logger.info(f"Slicing historical data from {history_start_date} to {first_candle_date}...")
        historical_slice = raw_data.loc[history_start_date:first_candle_date - pd.Timedelta(minutes=1)]
        
        if len(historical_slice) < self.min_candles_for_ma43200:
            logger.warning(f"⚠️ {pair}: Insufficient historical data ({len(historical_slice)} < {self.min_candles_for_ma43200}).")
            return df_backtest.assign(**{col: 0.0 for col in self.FEATURE_COLUMNS})
            
        logger.info(f"Found {len(historical_slice)} historical candles. Creating combined dataframe...")
        work_df = pd.concat([historical_slice, df_backtest], ignore_index=True)
        
        # Krok 3: Obliczenie TYLKO Wskaźników Długoterminowych
        logger.info(f"Calculating long-term indicators on a dataframe of size {len(work_df)}...")
        work_df = self._calculate_long_term_indicators(work_df, pair)

        # Krok 4: Pierwsze, Precyzyjne Przycięcie
        # Zostawiamy 121 świec bufora (120 dla modelu + 1 dla pct_change)
        num_to_cut = self.total_required_candles - 121
        logger.info(f"Performing first cut: removing {num_to_cut} rows...")
        work_df = work_df.iloc[num_to_cut:].reset_index(drop=True)
        logger.info(f"Intermediate dataframe size: {len(work_df)}")

        # Krok 5: Obliczenie Pozostałych Cech
        logger.info("Calculating short-term features...")
        work_df = self._calculate_short_term_features(work_df, pair)

        # Krok 6: Drugie, Finałowe Przycięcie
        logger.info("Performing final cut: removing 1 row to get final 120-candle buffer...")
        final_df = work_df.iloc[1:].reset_index(drop=True)
        logger.info(f"Final dataframe size for strategy: {len(final_df)}")

        # Oczyszczenie danych
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_df.fillna(0, inplace=True)

        return final_df

    def _calculate_long_term_indicators(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Oblicza tylko wskaźniki wymagające długiej historii."""
        try:
            dataframe['ma1440'] = dataframe['close'].rolling(window=1440, min_periods=1).mean()
            dataframe['volume_ma1440'] = dataframe['volume'].rolling(window=1440, min_periods=1).mean()
            dataframe['ma43200'] = dataframe['close'].rolling(window=43200, min_periods=1).mean()
            dataframe['volume_ma43200'] = dataframe['volume'].rolling(window=43200, min_periods=1).mean()
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

    @lru_cache(maxsize=128)
    def _load_raw_data_cached(self, pair: str) -> Optional[pd.DataFrame]:
        """Wczytuje i cache'uje cały plik raw_validated dla danej pary."""
        raw_validated_path = self._get_raw_validated_path(pair)
        if not raw_validated_path.exists():
            logger.error(f"❌ {pair}: Brak pliku raw_validated, nie można obliczyć cech.")
            return None
        
        try:
            logger.info(f"💾 {pair}: Loading and caching raw_validated file...")
            data = pd.read_feather(raw_validated_path)
            data['date'] = pd.to_datetime(data['date'], utc=True)
            data = data.set_index('date', drop=False)
            logger.info(f"✅ {pair}: Raw data cached successfully ({len(data)} candles).")
            return data
        except Exception as e:
            logger.error(f"❌ {pair}: Failed to load raw_validated file: {e}")
            return None

    def get_features_for_timestamp(self, pair: str, timestamp: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        🎯 GŁÓWNA METODA API V4 - DLA TRYBU LIVE 🎯
        Ta metoda będzie wymagała refaktoryzacji, aby była zgodna z nową logiką.
        NA RAZIE POZOSTAJE BEZ ZMIAN - SKUPIAMY SIĘ NA BACKTESTINGU.
        """
        cache_key = (pair, timestamp)
        if cache_key in self.features_cache:
            return self.features_cache[cache_key]
            
        # 1. Wczytaj dane z cache'u pliku
        raw_data = self._load_raw_data_cached(pair)
        if raw_data is None:
            return None

        # 2. Określ zakres danych historycznych potrzebny do obliczeń
        historical_end = timestamp
        historical_start = timestamp - pd.Timedelta(minutes=self.total_required_candles)
        
        # 3. Wytnij potrzebny fragment z wczytanych danych
        historical_slice = raw_data.loc[historical_start:historical_end]
        
        if len(historical_slice) < self.min_candles_for_ma43200:
            logger.warning(f"⚠️ {pair} at {timestamp}: Insufficient historical data ({len(historical_slice)} < {self.min_candles_for_ma43200}). Features may be inaccurate.")
            return None
            
        # 4. Oblicz cechy na tym fragmencie
        # TODO: Użyć nowych, rozdzielonych funkcji obliczających
        features_df = self._calculate_all_features(historical_slice.copy(), pair)
        
        # 5. Pobierz ostatni wiersz z obliczonymi cechami
        if features_df is None or features_df.empty:
            return None
            
        last_features = features_df.iloc[-1]
        
        # 6. Wyodrębnij tylko kluczowe cechy
        result = last_features[self.FEATURE_COLUMNS].to_dict()
        
        # 7. Zapisz do cache'u i zwróć
        self.features_cache[cache_key] = result
        return result

    def _calculate_all_features(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Helper do obliczania wszystkich 8 cech, identyczny z logiką V3.
        TA METODA JEST TERAZ PRZESTARZAŁA - ZOSTANIE ZASTĄPIONA.
        """
        try:
            # MA
            dataframe['ma1440'] = dataframe['close'].rolling(window=1440, min_periods=1).mean()
            dataframe['volume_ma1440'] = dataframe['volume'].rolling(window=1440, min_periods=1).mean()
            dataframe['ma43200'] = dataframe['close'].rolling(window=43200, min_periods=1).mean()
            dataframe['volume_ma43200'] = dataframe['volume'].rolling(window=43200, min_periods=1).mean()
            
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
            
            # Obsługa NaN
            dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
            dataframe.fillna(0, inplace=True)
            
            return dataframe
        except Exception as e:
            logger.error(f"❌ {pair}: Error calculating features in V4: {e}")
            return None
    
    def _get_raw_validated_path(self, pair: str) -> Path:
        """Buduje ścieżkę do pliku raw_validated."""
        clean_pair = pair.replace('/', '_').replace(':', '_')
        freqtrade_filename = f"{clean_pair}-1m-futures.feather"
        freqtrade_data_dir = Path(self.config['datadir'], "futures")
        return freqtrade_data_dir / freqtrade_filename
    
    async def initialize_for_pairs(self, pairs: List[str]) -> Dict[str, bool]:
        """
        🆕 NOWY SYSTEM: Sprawdza dostępność plików raw_validated dla par
        
        Ta funkcja może być wywoływana podczas startu bota
        """
        if not self.is_enabled():
            logger.info("🆕 NEW Buffer system disabled - skipping initialization")
            return {}
        
        logger.info(f"🆕 NEW: Sprawdzam dostępność raw_validated files dla {len(pairs)} par...")
        
        results = {}
        successful_pairs = 0
        failed_pairs = 0
        
        for pair in pairs:
            try:
                # 🆕 SPRAWDŹ czy istnieje plik raw_validated
                raw_validated_path = self._get_raw_validated_path(pair)
                success = raw_validated_path.exists()
                
                results[pair] = success
                
                if success:
                    logger.info(f"✅ {pair}: raw_validated file available ({raw_validated_path.name})")
                    successful_pairs += 1
                else:
                    logger.error(f"❌ {pair}: raw_validated file missing ({raw_validated_path.name})")
                    failed_pairs += 1
                
                self.initialization_status[pair] = {
                    'success': success,
                    'raw_validated_path': str(raw_validated_path),
                    'file_exists': success,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"❌ Error checking {pair}: {e}")
                results[pair] = False
                failed_pairs += 1
                
                self.initialization_status[pair] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Summary
        logger.info(f"📊 Raw validated files check complete: {successful_pairs}/{len(pairs)} pairs available ({successful_pairs/len(pairs)*100:.1f}%)")
        
        if failed_pairs > 0:
            logger.warning(f"⚠️ {failed_pairs} par bez plików raw_validated - zostaną pominięte w strategii")
            logger.info("💡 Uruchom moduł walidacji aby wygenerować brakujące pliki raw_validated")
        
        self.startup_complete = True
        return results
    
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
            'raw_validated_path': str(self.raw_validated_path),
            'gap_fill_only': self.gap_fill_only,
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

def get_dataframe_extender(config: Dict = None) -> DataFrameExtender:
    """Zwraca globalną instancję DataFrameExtendera."""
    global _extender_instance
    if _extender_instance is None:
        _extender_instance = DataFrameExtender(config)
    return _extender_instance 