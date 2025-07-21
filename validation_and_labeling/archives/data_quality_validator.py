"""
MODUŁ WALIDACJI JAKOŚCI DANYCH

Kompleksowy system walidacji jakości danych finansowych OHLCV.
Sprawdza ciągłość danych, logikę świec, anomalie cenowe i wzorce statystyczne.

FUNKCJONALNOŚCI:
1. Sprawdzenie ciągłości timestamps (1-minutowe interwały)
2. Wykrycie brakujących świec i uzupełnianie luk strategią BRIDGE
3. Walidacja logiki świec OHLCV
4. Wykrywanie anomalii cenowych
5. Analiza wzorców statystycznych
6. Kompleksowy system scoring i raportowanie

STRATEGIA BRIDGE (uniwersalna dla wszystkich luk):
- Interpolacja cenowa: płynne przejście od before_candle.close → after_candle.open
- Realistyczny szum proporcjonalny do volatility i trendu
- Zachowanie ciągłości OHLC między świecami
- Specjalne logowanie dla luk >1 godzina (ale nadal uzupełniane)
- Brak przerywania działania dla żadnych luk

KORZYŚCI:
- Eliminuje skoki cenowe między wygenerowanymi a rzeczywistymi danymi
- ML-friendly: brak artefaktów w danych treningowych
- Uniwersalność: jedna strategia dla wszystkich rozmiarów luk
- Robustność: nie przerywa działania, tylko informuje o dużych lukach
- Kompleksowa walidacja jakości danych
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import json
from pathlib import Path


class DataQualityValidator:
    """
    Kompleksowy walidator jakości danych finansowych OHLCV.
    
    FUNKCJONALNOŚCI:
    1. Sprawdzenie ciągłości timestamps
    2. Wykrycie i uzupełnienie luk strategią BRIDGE
    3. Walidacja logiki świec OHLCV
    4. Wykrywanie anomalii cenowych
    5. Analiza wzorców statystycznych
    6. Kompleksowy system scoring i raportowanie
    
    STRATEGIA BRIDGE (uniwersalna dla wszystkich luk):
    - Interpolacja cenowa: płynne przejście od before_candle.close → after_candle.open
    - Realistyczny szum proporcjonalny do volatility i trendu
    - Zachowanie ciągłości OHLC między świecami
    - Specjalne logowanie dla luk >1 godzina (ale nadal uzupełniane)
    - Brak przerywania działania
    
    KORZYŚCI STRATEGII BRIDGE:
    - Eliminuje skoki cenowe między wygenerowanymi a rzeczywistymi danymi
    - ML-friendly: brak artefaktów w danych treningowych
    - Uniwersalność: jedna strategia dla wszystkich rozmiarów luk
    - Robustność: nie przerywa działania, tylko informuje o dużych lukach
    """
    
    def __init__(self, 
                 timeframe: str = '1m', 
                 tolerance_seconds: int = 60,
                 # PARAMETRY WALIDACJI JAKOŚCI:
                 enable_quality_validation: bool = True,  # Domyślnie włączone - pełna walidacja
                 quality_checks: List[str] = None,
                 anomaly_thresholds: Dict = None,
                 statistical_thresholds: Dict = None,
                 # PARAMETRY COMPETITIVE LABELING:
                 enable_competitive_labeling: bool = False,  # Competitive labeling
                 competitive_config: Dict = None):
        """
        Inicjalizacja walidatora jakości danych z opcjonalnym competitive labeling.
        
        Args:
            timeframe: Interwał czasowy ('1m', '5m', '15m', '1h')
            tolerance_seconds: Tolerancja dla wykrywania luk (sekundy)
            enable_quality_validation: Włącz walidację jakości danych (domyślnie: True)
            quality_checks: Lista walidacji do wykonania ['ohlcv_logic', 'price_anomalies', 'statistical_patterns']
            anomaly_thresholds: Progi dla wykrywania anomalii cenowych
            statistical_thresholds: Progi dla analizy statystycznej
            enable_competitive_labeling: Włącz competitive labeling (domyślnie: False)
            competitive_config: Konfiguracja competitive labeling (TP/SL/WINDOW)
        """
        self.timeframe = timeframe
        self.tolerance_seconds = tolerance_seconds
        
        # Mapowanie timeframe na sekundy
        self.timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        if timeframe not in self.timeframe_seconds:
            raise ValueError(f"Nieobsługiwany timeframe: {timeframe}")
            
        self.interval_seconds = self.timeframe_seconds[timeframe]
        
        # Statystyki
        self.gaps_detected = []
        self.candles_added = 0
        self.original_length = 0
        
        # NOWE: Konfiguracja walidacji jakości
        self.enable_quality_validation = enable_quality_validation
        
        # Domyślne walidacje jeśli nie podano
        if quality_checks is None:
            self.quality_checks = ['ohlcv_logic', 'price_anomalies', 'statistical_patterns']
        else:
            self.quality_checks = quality_checks
        
        # Domyślne progi anomalii
        self.DEFAULT_ANOMALY_THRESHOLDS = {
            'price_jump': 0.05,      # 5% skok cenowy
            'outlier_sigma': 3,      # 3 sigma outliers
            'extreme_spread': 0.2,   # 20% spread
            'flash_crash': -0.1      # 10% spadek
        }
        
        if anomaly_thresholds is None:
            self.anomaly_thresholds = self.DEFAULT_ANOMALY_THRESHOLDS.copy()
        else:
            # Merge z domyślnymi (użytkownik może podać tylko część)
            self.anomaly_thresholds = {**self.DEFAULT_ANOMALY_THRESHOLDS, **anomaly_thresholds}
        
        # Domyślne progi statystyczne
        self.DEFAULT_STATISTICAL_THRESHOLDS = {
            'uniqueness_min': 0.1,   # Min 10% unikalnych wartości
            'volatility_max': 0.1    # Max 10% volatility
        }
        
        if statistical_thresholds is None:
            self.statistical_thresholds = self.DEFAULT_STATISTICAL_THRESHOLDS.copy()
        else:
            # Merge z domyślnymi
            self.statistical_thresholds = {**self.DEFAULT_STATISTICAL_THRESHOLDS, **statistical_thresholds}
        
        # NOWE: Statystyki walidacji jakości
        self.validation_results = {}
        self.quality_issues = []
        
        # NOWE: Konfiguracja competitive labeling
        self.enable_competitive_labeling = enable_competitive_labeling
        
        # Domyślna konfiguracja competitive labeling
        self.DEFAULT_COMPETITIVE_CONFIG = {
            'LONG_TP_PCT': 0.010,    # 1.0% - Take Profit dla pozycji długiej
            'LONG_SL_PCT': 0.005,    # 0.5% - Stop Loss dla pozycji długiej
            'SHORT_TP_PCT': 0.010,   # 1.0% - Take Profit dla pozycji krótkiej
            'SHORT_SL_PCT': 0.005,   # 0.5% - Stop Loss dla pozycji krótkiej
            'FUTURE_WINDOW': 120     # 120 minut - okno obserwacji przyszłości
        }
        
        if competitive_config is None:
            self.competitive_config = self.DEFAULT_COMPETITIVE_CONFIG.copy()
        else:
            # Merge z domyślnymi
            self.competitive_config = {**self.DEFAULT_COMPETITIVE_CONFIG, **competitive_config}
        
        # Statystyki competitive labeling
        self.labels_generated = 0
        self.label_distribution = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        
        # Ustawienia reprodukowalności
        np.random.seed(42)
        
    def check_and_fill_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Główna funkcja sprawdzania i uzupełniania luk z opcjonalną walidacją jakości.
        
        Args:
            df: DataFrame z danymi OHLCV (index = timestamp)
            
        Returns:
            Tuple[DataFrame, Dict]: (uzupełnione_dane, raport)
        """
        print(f"🔍 Sprawdzanie ciągłości danych ({self.timeframe})...")
        
        # Reset statystyk
        self.gaps_detected = []
        self.candles_added = 0
        self.original_length = len(df)
        self.validation_results = {}  # Reset wyników walidacji
        self.quality_issues = []      # Reset listy problemów
        
        # 1. Istniejąca walidacja podstawowa (bez zmian)
        df_validated = self._validate_input_data(df)
        
        # 2. NOWA: Rozszerzona walidacja jakości (opcjonalna)
        if self.enable_quality_validation:
            print(f"🔍 Walidacja jakości danych (sprawdzenia: {', '.join(self.quality_checks)})...")
            self._perform_quality_validation(df_validated)
        
        # 3. NOWA: Competitive labeling (opcjonalne)
        if self.enable_competitive_labeling:
            print(f"🎯 Competitive labeling danych (okno: {self.competitive_config['FUTURE_WINDOW']}min)...")
            df_validated = self._add_competitive_labels(df_validated)
        
        # 4. Istniejące wykrywanie i uzupełnianie luk (bez zmian)
        gaps = self._detect_gaps(df_validated)
        
        if not gaps:
            print("   ✅ Brak luk w danych")
            # Generuj rozszerzony raport (może zawierać wyniki walidacji jakości)
            report = self._generate_comprehensive_report(df_validated)
            return df_validated, report
        
        print(f"   🔍 Wykryto {len(gaps)} luk")
        
        # 5. Uzupełnienie luk
        df_filled = self._fill_gaps(df_validated, gaps)
        
        # 6. Rozszerzony raport
        report = self._generate_comprehensive_report(df_filled)
        
        print(f"   ✅ Uzupełniono {self.candles_added} świec")
        
        # Wyświetl odpowiedni score (comprehensive lub bazowy)
        if self.enable_quality_validation and 'comprehensive_quality_score' in report:
            print(f"   📊 Kompleksowa jakość danych: {report['comprehensive_quality_score']:.1f}%")
        else:
            print(f"   📊 Jakość danych: {report['quality_score']:.1f}%")
        
        return df_filled, report
    
    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Walidacja i przygotowanie danych wejściowych."""
        
        # Sprawdź wymagane kolumny
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Brakujące kolumny: {missing_cols}")
        
        # Sprawdź czy index to timestamp
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index)
            else:
                raise ValueError("Brak kolumny timestamp lub DatetimeIndex")
        
        # Sortuj chronologicznie
        df = df.sort_index()
        
        # Usuń duplikaty
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _add_competitive_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 COMPETITIVE LABELING - IMPLEMENTACJA ALGORYTMU SYMULACJI GIEŁDY
        
        Implementuje algorytm competitive labeling zgodnie z wymaganiami użytkownika:
        - Dwie przeciwstawne pozycje (LONG i SHORT) konkurują
        - Ta która pierwsza osiągnie Take Profit wygrywa
        - Chronologiczna symulacja przez przyszłe świece
        - Priorytet TP > SL, eliminacja przez SL
        
        Args:
            df: DataFrame z danymi OHLCV (index = timestamp)
            
        Returns:
            DataFrame: Dane z dodaną kolumną 'label' (0=SHORT, 1=HOLD, 2=LONG)
        """
        print(f"   🎯 Rozpoczynam competitive labeling...")
        print(f"      Parametry: LONG TP={self.competitive_config['LONG_TP_PCT']*100:.1f}%, SL={self.competitive_config['LONG_SL_PCT']*100:.1f}%")
        print(f"      Parametry: SHORT TP={self.competitive_config['SHORT_TP_PCT']*100:.1f}%, SL={self.competitive_config['SHORT_SL_PCT']*100:.1f}%")
        print(f"      Okno przyszłości: {self.competitive_config['FUTURE_WINDOW']} minut")
        
        # Reset statystyk
        self.labels_generated = 0
        self.label_distribution = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        
        # Inicjalizacja kolumny label (domyślnie HOLD=1)
        df = df.copy()
        df['label'] = 1  # HOLD
        
        # Oblicz zakres indeksów do przetworzenia
        # Potrzebujemy FUTURE_WINDOW świec w przyszłości dla każdego punktu predykcji
        future_window = self.competitive_config['FUTURE_WINDOW']
        valid_start = 0  # Można zacząć od pierwszej świecy
        valid_end = len(df) - future_window  # Potrzeba przyszłości dla algorytmu
        
        if valid_end <= valid_start:
            print(f"   ⚠️  Za mało danych dla competitive labeling (potrzeba min {future_window} świec)")
            return df
        
        total_points = valid_end - valid_start
        print(f"   📊 Przetwarzanie {total_points:,} punktów predykcji...")
        
        # Reset indeksów numerycznych dla łatwiejszego dostępu
        df_reset = df.reset_index()
        
        # Przetwarzanie każdego punktu predykcji
        processed_count = 0
        for idx in range(valid_start, valid_end):
            try:
                label = self._execute_competitive_algorithm(df_reset, idx)
                df.iloc[idx, df.columns.get_loc('label')] = label
                
                # Aktualizuj statystyki
                self.labels_generated += 1
                if label == 0:
                    self.label_distribution['SHORT'] += 1
                elif label == 1:
                    self.label_distribution['HOLD'] += 1
                elif label == 2:
                    self.label_distribution['LONG'] += 1
                
                processed_count += 1
                
                # Progress indicator co 10000 punktów
                if processed_count % 10000 == 0:
                    progress = processed_count / total_points * 100
                    print(f"      Progress: {progress:.1f}% ({processed_count:,}/{total_points:,})")
                    
            except Exception as e:
                # W przypadku błędu pozostaw HOLD
                df.iloc[idx, df.columns.get_loc('label')] = 1
                self.label_distribution['HOLD'] += 1
                continue
        
        # Wyświetl statystyki końcowe
        total_labels = sum(self.label_distribution.values())
        print(f"   ✅ Competitive labeling zakończony:")
        print(f"      Wygenerowano {self.labels_generated:,} etykiet")
        for label_name, count in self.label_distribution.items():
            if total_labels > 0:
                percentage = (count / total_labels) * 100
                print(f"      {label_name}: {count:,} ({percentage:.1f}%)")
        
        return df
    
    def _execute_competitive_algorithm(self, df: pd.DataFrame, current_idx: int) -> int:
        """
        🏆 GŁÓWNY ALGORYTM COMPETITIVE LABELING
        
        Wykonuje symulację giełdy dla punktu current_idx zgodnie z algorytmem:
        KROK 1: Setup pozycji (entry price, TP/SL levels)
        KROK 2: Symulacja chronologiczna (iteracja przez przyszłe świece)
        KROK 3: Logika priorytetów (TP > SL, eliminacja pozycji)
        KROK 4: Warunki końcowe (timeout = HOLD)
        
        Args:
            df: DataFrame z danymi OHLCV (reset_index)
            current_idx: Indeks punktu predykcji
            
        Returns:
            int: Etykieta (0=SHORT, 1=HOLD, 2=LONG)
        """
        # === KROK 1: SETUP POZYCJI ===
        entry_price = df.iloc[current_idx]['close']
        
        # Oblicz poziomy TP/SL
        long_tp = entry_price * (1 + self.competitive_config['LONG_TP_PCT'])
        long_sl = entry_price * (1 - self.competitive_config['LONG_SL_PCT'])
        short_tp = entry_price * (1 - self.competitive_config['SHORT_TP_PCT'])
        short_sl = entry_price * (1 + self.competitive_config['SHORT_SL_PCT'])
        
        # Status pozycji - obie startują jako aktywne
        long_active = True
        short_active = True
        
        # === KROK 2: SYMULACJA CHRONOLOGICZNA ===
        future_start = current_idx + 1
        future_end = min(current_idx + self.competitive_config['FUTURE_WINDOW'] + 1, len(df))
        
        for check_idx in range(future_start, future_end):
            # Pobierz dane świecy
            candle = df.iloc[check_idx]
            high = candle['high']
            low = candle['low']
            
            # === KROK 3: LOGIKA PRIORYTETÓW ===
            
            # Sprawdź wydarzenia TP i SL w tej świecy
            tp_events = []
            sl_events = []
            
            # Sprawdź LONG pozycję
            if long_active:
                if high > long_tp:  # LONG TP osiągnięty (strict inequality)
                    tp_events.append('LONG_TP')
                if low < long_sl:   # LONG SL osiągnięty (strict inequality)
                    sl_events.append('LONG_SL')
            
            # Sprawdź SHORT pozycję  
            if short_active:
                if low < short_tp:   # SHORT TP osiągnięty (strict inequality)
                    tp_events.append('SHORT_TP')
                if high > short_sl:  # SHORT SL osiągnięty (strict inequality)
                    sl_events.append('SHORT_SL')
            
            # === PRIORYTET 1: TAKE PROFIT EVENTS ===
            if len(tp_events) == 1:
                # Tylko jedna pozycja osiągnęła TP - jasny zwycięzca
                if tp_events[0] == 'LONG_TP':
                    return 2  # LONG wygrywa
                else:  # SHORT_TP
                    return 0  # SHORT wygrywa
                    
            elif len(tp_events) == 2:
                # Oba TP w tej samej świecy - losowy wybór (symuluje spread execution)
                winner = np.random.choice(['LONG', 'SHORT'])
                return 2 if winner == 'LONG' else 0
            
            # === PRIORYTET 2: STOP LOSS EVENTS (ELIMINACJA) ===
            if 'LONG_SL' in sl_events:
                long_active = False
            if 'SHORT_SL' in sl_events:
                short_active = False
            
            # Sprawdź czy pozostały aktywne pozycje
            if not long_active and not short_active:
                return 1  # Obie pozycje wyeliminowane - HOLD
        
        # === KROK 4: WARUNKI KOŃCOWE ===
        # Timeout - żadna pozycja nie osiągnęła TP w czasie FUTURE_WINDOW
        return 1  # HOLD
    
    def _detect_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Wykrywa luki w danych.
        
        Returns:
            List[Dict]: Lista luk z informacjami o rozmiarze (bez klasyfikacji typu)
        """
        gaps = []
        
        for i in range(1, len(df)):
            time_diff = (df.index[i] - df.index[i-1]).total_seconds()
            
            # Sprawdź czy różnica przekracza tolerancję
            if time_diff > (self.interval_seconds + self.tolerance_seconds):
                missing_minutes = int((time_diff - self.interval_seconds) / 60)
                
                gap_info = {
                    'start_time': df.index[i-1],
                    'end_time': df.index[i],
                    'duration_minutes': missing_minutes,
                    'before_candle': df.iloc[i-1].to_dict(),
                    'after_candle': df.iloc[i].to_dict()
                }
                
                gaps.append(gap_info)
        
        self.gaps_detected = gaps
        return gaps
    
    def _fill_gaps(self, df: pd.DataFrame, gaps: List[Dict]) -> pd.DataFrame:
        """
        Uzupełnia luki w danych strategią BRIDGE.
        
        NOWA STRATEGIA BRIDGE:
        - Dla WSZYSTKICH luk: interpolacja z trendem (bridge)
        - Ciągłość cenowa: płynne przejście od before_candle.close → after_candle.open
        - Luki >1h: specjalne logowanie ostrzeżeń (ale nadal uzupełniane)
        - Brak przerywania działania
        
        Args:
            df: Oryginalne dane
            gaps: Lista wykrytych luk
            
        Returns:
            DataFrame: Dane z uzupełnionymi lukami
        """
        df_filled = df.copy()
        
        for gap in gaps:
            # Specjalne logowanie dla luk >1 godzina
            if gap['duration_minutes'] > 60:
                print(f"   ⚠️  DUŻA LUKA ({gap['duration_minutes']}min = {gap['duration_minutes']/60:.1f}h): {gap['start_time']} → {gap['end_time']}")
                print(f"       Stosowanie strategii BRIDGE dla zachowania ciągłości cenowej")
            
            # Zastosuj strategię bridge dla WSZYSTKICH luk
            filled_candles = self._fill_gap_with_bridge(gap)
            
            # Dodaj uzupełnione świece do DataFrame
            if not filled_candles.empty:
                df_filled = pd.concat([df_filled, filled_candles]).sort_index()
                self.candles_added += len(filled_candles)
        
        return df_filled
    
    def _fill_gap_with_bridge(self, gap: Dict) -> pd.DataFrame:
        """
        Uzupełnia lukę strategią BRIDGE - uniwersalna metoda dla wszystkich rozmiarów luk.
        
        ALGORYTM BRIDGE:
        1. Oblicz trend cenowy: (after_candle.open - before_candle.close) / duration_minutes
        2. Generuj świece z płynną interpolacją ceny
        3. Dodaj realistyczny szum proporcjonalny do trendu
        4. Zapewnij ciągłość OHLC między świecami
        
        Args:
            gap: Informacje o luce
            
        Returns:
            DataFrame: Uzupełnione świece z ciągłością cenową
        """
        start_time = gap['start_time']
        end_time = gap['end_time']
        before_candle = gap['before_candle']
        after_candle = gap['after_candle']
        duration_minutes = gap['duration_minutes']
        
        # Generuj timestamps dla brakujących świec
        timestamps = pd.date_range(
            start=start_time + timedelta(minutes=1),
            end=end_time - timedelta(minutes=1),
            freq='1min'
        )
        
        if len(timestamps) == 0:
            return pd.DataFrame()
        
        # KLUCZOWE: Oblicz trend cenowy dla ciągłości
        start_price = before_candle['close']
        end_price = after_candle['open']
        total_price_change = end_price - start_price
        trend_per_minute = total_price_change / (duration_minutes + 1)  # +1 bo liczymy z before i after
        
        print(f"       Bridge: {start_price:.2f} → {end_price:.2f} (trend: {trend_per_minute:.4f}/min)")
        
        filled_data = []
        
        for i, ts in enumerate(timestamps):
            # Oblicz docelową cenę z trendem (interpolacja liniowa)
            progress = (i + 1) / (len(timestamps) + 1)  # 0 to 1
            target_price = start_price + (total_price_change * progress)
            
            # Dodaj realistyczny szum proporcjonalny do trendu i volatility
            base_volatility = abs(before_candle['high'] - before_candle['low']) / before_candle['close']
            trend_magnitude = abs(trend_per_minute) / start_price
            
            # Szum proporcjonalny do volatility i trendu
            noise_factor = max(base_volatility, trend_magnitude) * 0.5
            price_noise = np.random.normal(0, noise_factor * target_price)
            
            # Finalna cena z szumem
            actual_price = target_price + price_noise
            
            # Generuj realistyczne OHLC z ciągłością
            candle = self._create_realistic_candle(
                target_price=actual_price,
                base_volatility=base_volatility,
                reference_volume=before_candle['volume'],
                is_bridge=True
            )
            
            filled_data.append(candle)
        
        return pd.DataFrame(filled_data, index=timestamps)
    
    def _create_realistic_candle(self, target_price: float, base_volatility: float, 
                               reference_volume: float, is_bridge: bool = False) -> Dict:
        """
        Tworzy realistyczną świecę OHLCV z zachowaniem ciągłości.
        
        Args:
            target_price: Docelowa cena (close)
            base_volatility: Bazowa volatility z danych historycznych
            reference_volume: Referencyjny volume
            is_bridge: Czy to świeca bridge (mniejsza volatility)
            
        Returns:
            Dict: Świeca OHLCV
        """
        # Dla bridge świec używamy mniejszej volatility
        volatility_factor = 0.5 if is_bridge else 1.0
        price_range = target_price * base_volatility * volatility_factor
        
        # Generuj OHLC wokół target_price
        open_price = target_price + np.random.uniform(-price_range*0.3, price_range*0.3)
        close_price = target_price + np.random.uniform(-price_range*0.3, price_range*0.3)
        
        # High i Low z większym zakresem
        high_price = max(open_price, close_price) + np.random.uniform(0, price_range*0.7)
        low_price = min(open_price, close_price) - np.random.uniform(0, price_range*0.7)
        
        # Volume z większą zmiennością dla bridge
        volume_factor = 2.0 if is_bridge else 1.5
        volume = np.random.uniform(
            reference_volume * 0.3, 
            reference_volume * volume_factor
        )
        
        candle = {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': max(volume, 1.0)  # Minimum volume = 1
        }
        
        # Zapewnij poprawność OHLC
        candle['high'] = max(candle['open'], candle['high'], candle['close'])
        candle['low'] = min(candle['open'], candle['low'], candle['close'])
        
        return candle
    
    def _calculate_quality_score(self) -> float:
        """
        Oblicza ocenę jakości danych (0-100%) - NOWA LOGIKA dla strategii BRIDGE.
        
        Ponieważ wszystkie luki są teraz uzupełniane strategią bridge,
        jakość zależy od:
        1. Procentu oryginalnych danych vs uzupełnionych
        2. Liczby i rozmiaru luk (większe luki = mniejsza pewność)
        3. Specjalna kara za bardzo duże luki (>1h)
        """
        
        if self.original_length == 0:
            return 0.0
        
        # 1. Bazowa kompletność (% oryginalnych danych)
        total_candles = self.original_length + self.candles_added
        original_data_ratio = (self.original_length / total_candles) * 100
        
        # 2. Kara za liczbę luk (każda luka obniża pewność)
        gap_count_penalty = min(len(self.gaps_detected) * 2, 15)  # Max 15% kary
        
        # 3. Kara za rozmiar luk
        size_penalty = 0
        very_large_gaps = 0  # >1h
        
        for gap in self.gaps_detected:
            duration = gap['duration_minutes']
            
            if duration > 60:  # >1h
                very_large_gaps += 1
                size_penalty += min(duration / 60 * 3, 20)  # 3% za każdą godzinę, max 20%
            elif duration > 30:  # 30min-1h
                size_penalty += min(duration / 30 * 2, 10)  # 2% za każde 30min, max 10%
            elif duration > 10:  # 10-30min
                size_penalty += min(duration / 10 * 1, 5)   # 1% za każde 10min, max 5%
        
        # Ogranicz całkowitą karę za rozmiar
        size_penalty = min(size_penalty, 30)
        
        # 4. Dodatkowa kara za bardzo duże luki (>1h)
        very_large_penalty = min(very_large_gaps * 10, 25)  # 10% za każdą lukę >1h, max 25%
        
        # 5. Finalna ocena
        quality_score = original_data_ratio - gap_count_penalty - size_penalty - very_large_penalty
        quality_score = max(0, min(100, quality_score))  # Ogranicz do 0-100
        
        return quality_score
    
    def _generate_report(self, df_final: pd.DataFrame) -> Dict:
        """Generuje szczegółowy raport analizy ciągłości - ZAKTUALIZOWANY dla strategii BRIDGE."""
        
        # Klasyfikacja luk według rozmiaru (nowa logika)
        gap_stats = {
            'small': sum(1 for gap in self.gaps_detected if gap['duration_minutes'] <= 10),
            'medium': sum(1 for gap in self.gaps_detected if 10 < gap['duration_minutes'] <= 60),
            'large': sum(1 for gap in self.gaps_detected if gap['duration_minutes'] > 60)
        }
        
        # Analiza dużych luk (>1h)
        large_gaps_details = []
        for gap in self.gaps_detected:
            if gap['duration_minutes'] > 60:
                large_gaps_details.append({
                    'start': gap['start_time'].isoformat(),
                    'end': gap['end_time'].isoformat(),
                    'duration_minutes': gap['duration_minutes'],
                    'duration_hours': round(gap['duration_minutes'] / 60, 2),
                    'price_change': gap['after_candle']['open'] - gap['before_candle']['close'],
                    'price_change_percent': round(
                        ((gap['after_candle']['open'] - gap['before_candle']['close']) / gap['before_candle']['close']) * 100, 4
                    )
                })
        
        # Oblicz jakość
        quality_score = self._calculate_quality_score()
        
        # Statystyki bridge
        bridge_stats = {
            'strategy_used': 'BRIDGE',
            'all_gaps_filled': True,
            'total_candles_added': self.candles_added,
            'original_data_ratio': round((self.original_length / (self.original_length + self.candles_added)) * 100, 2) if self.candles_added > 0 else 100.0,
            'synthetic_data_ratio': round((self.candles_added / (self.original_length + self.candles_added)) * 100, 2) if self.candles_added > 0 else 0.0
        }
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'timeframe': self.timeframe,
            'strategy': 'BRIDGE - Universal gap filling with price continuity',
            'original_candles': self.original_length,
            'final_candles': len(df_final),
            'candles_added': self.candles_added,
            'gaps_detected': len(self.gaps_detected),
            'gap_size_distribution': gap_stats,
            'bridge_statistics': bridge_stats,
            'quality_score': quality_score,
            'quality_status': 'EXCELLENT' if quality_score >= 95 else 'GOOD' if quality_score >= 85 else 'ACCEPTABLE' if quality_score >= 70 else 'PROBLEMATIC',
            'data_range': {
                'start': df_final.index[0].isoformat() if len(df_final) > 0 else None,
                'end': df_final.index[-1].isoformat() if len(df_final) > 0 else None,
                'duration_hours': round((df_final.index[-1] - df_final.index[0]).total_seconds() / 3600, 2) if len(df_final) > 0 else 0
            },
            'large_gaps_analysis': {
                'count': len(large_gaps_details),
                'total_duration_hours': round(sum(gap['duration_hours'] for gap in large_gaps_details), 2),
                'details': large_gaps_details
            },
            'gap_details': [
                {
                    'start': gap['start_time'].isoformat(),
                    'end': gap['end_time'].isoformat(),
                    'duration_minutes': gap['duration_minutes'],
                    'duration_hours': round(gap['duration_minutes'] / 60, 2),
                    'size_category': 'large' if gap['duration_minutes'] > 60 else 'medium' if gap['duration_minutes'] > 10 else 'small',
                    'filled_with_bridge': True,
                    'price_continuity': {
                        'before_close': gap['before_candle']['close'],
                        'after_open': gap['after_candle']['open'],
                        'price_change': gap['after_candle']['open'] - gap['before_candle']['close'],
                        'price_change_percent': round(
                            ((gap['after_candle']['open'] - gap['before_candle']['close']) / gap['before_candle']['close']) * 100, 4
                        )
                    }
                }
                for gap in self.gaps_detected
            ]
        }
        
        return report
    
    def save_report(self, report: Dict, filepath: str):
        """Zapisuje raport do pliku JSON."""
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   📋 Raport zapisany: {filepath}")

    # ============================================================================
    # NOWE METODY WALIDACJI JAKOŚCI DANYCH
    # ============================================================================
    
    def _perform_quality_validation(self, df: pd.DataFrame) -> None:
        """
        Wykonuje kompleksową walidację jakości danych.
        
        Args:
            df: DataFrame z danymi OHLCV do walidacji
        """
        # Reset wyników walidacji
        self.validation_results = {}
        self.quality_issues = []
        
        # Wykonaj wybrane walidacje
        for check in self.quality_checks:
            if check == 'ohlcv_logic':
                print("   🔍 Walidacja logiki OHLCV...")
                self.validation_results['ohlcv_validation'] = self._validate_ohlcv_logic(df)
                
            elif check == 'price_anomalies':
                print("   🔍 Wykrywanie anomalii cenowych...")
                self.validation_results['anomaly_detection'] = self._validate_price_anomalies(df)
                
            elif check == 'statistical_patterns':
                print("   🔍 Analiza wzorców statystycznych...")
                self.validation_results['statistical_analysis'] = self._validate_statistical_patterns(df)
                
            else:
                print(f"   ⚠️  Nieznany typ walidacji: {check}")
        
        # Podsumowanie walidacji
        total_issues = sum(
            result.get('issues_found', 0) if isinstance(result, dict) else 0
            for result in self.validation_results.values()
        )
        
        if total_issues > 0:
            print(f"   ⚠️  Wykryto {total_issues} problemów z jakością danych")
        else:
            print("   ✅ Walidacja jakości: brak problemów")
    
    def _validate_ohlcv_logic(self, df: pd.DataFrame) -> Dict:
        """
        Walidacja logiki świec OHLCV.
        
        Sprawdza:
        - High >= max(open, close)
        - Low <= min(open, close)  
        - High >= Low
        - Volume >= 0
        - Ceny > 0
        
        Args:
            df: DataFrame z danymi OHLCV
            
        Returns:
            Dict: Wyniki walidacji OHLCV
        """
        issues = []
        
        # Test 1: High >= max(open, close)
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        if invalid_high.any():
            count = invalid_high.sum()
            issues.append({
                'type': 'invalid_high',
                'description': f'High < max(open, close) w {count} świecach',
                'count': count,
                'severity': 'critical'
            })
        
        # Test 2: Low <= min(open, close)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        if invalid_low.any():
            count = invalid_low.sum()
            issues.append({
                'type': 'invalid_low',
                'description': f'Low > min(open, close) w {count} świecach',
                'count': count,
                'severity': 'critical'
            })
        
        # Test 3: High >= Low
        invalid_range = df['high'] < df['low']
        if invalid_range.any():
            count = invalid_range.sum()
            issues.append({
                'type': 'invalid_range',
                'description': f'High < Low w {count} świecach',
                'count': count,
                'severity': 'critical'
            })
        
        # Test 4: Volume >= 0
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            issues.append({
                'type': 'negative_volume',
                'description': f'Ujemny volume w {count} świecach',
                'count': count,
                'severity': 'critical'
            })
        
        # Test 5: Ceny > 0
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if zero_prices.any():
            count = zero_prices.sum()
            issues.append({
                'type': 'zero_prices',
                'description': f'Zerowe lub ujemne ceny w {count} świecach',
                'count': count,
                'severity': 'critical'
            })
        
        # Dodaj do globalnej listy problemów
        self.quality_issues.extend(issues)
        
        return {
            'ohlcv_logic_valid': len(issues) == 0,
            'issues_found': len(issues),
            'issues': issues,
            'tests_performed': 5,
            'invalid_candles_count': sum(issue['count'] for issue in issues)
        }
    
    def _validate_price_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Wykrywanie anomalii cenowych.
        
        Sprawdza:
        - Duże skoki cenowe (>threshold%)
        - Outliers statystyczne (>N sigma)
        - Ekstremalne spready (high-low vs close)
        - Flash crashes (>threshold% spadek)
        
        Args:
            df: DataFrame z danymi OHLCV
            
        Returns:
            Dict: Wyniki wykrywania anomalii
        """
        anomalies = []
        
        # Anomalia 1: Duże skoki cenowe
        price_changes = df['close'].pct_change().abs()
        large_jumps = price_changes > self.anomaly_thresholds['price_jump']
        if large_jumps.any():
            count = int(large_jumps.sum())  # Konwersja na int
            max_jump = float(price_changes.max())  # Konwersja na float
            anomalies.append({
                'type': 'large_price_jumps',
                'description': f'Duże skoki cenowe (>{self.anomaly_thresholds["price_jump"]*100:.1f}%) w {count} świecach',
                'count': count,
                'max_value': max_jump,
                'severity': 'warning' if max_jump < 0.2 else 'critical'
            })
        
        # Anomalia 2: Outliers statystyczne dla każdej kolumny cenowej
        for col in ['open', 'high', 'low', 'close']:
            if df[col].std() > 0:  # Unikaj dzielenia przez zero
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.anomaly_thresholds['outlier_sigma']
                if outliers.any():
                    count = int(outliers.sum())  # Konwersja na int
                    max_z = float(z_scores.max())  # Konwersja na float
                    anomalies.append({
                        'type': f'{col}_outliers',
                        'description': f'Outliers w {col} (>{self.anomaly_thresholds["outlier_sigma"]} sigma) w {count} świecach',
                        'count': count,
                        'max_z_score': max_z,
                        'severity': 'warning'
                    })
        
        # Anomalia 3: Ekstremalne spready
        spreads = (df['high'] - df['low']) / df['close']
        extreme_spreads = spreads > self.anomaly_thresholds['extreme_spread']
        if extreme_spreads.any():
            count = int(extreme_spreads.sum())  # Konwersja na int
            max_spread = float(spreads.max())  # Konwersja na float
            anomalies.append({
                'type': 'extreme_spreads',
                'description': f'Ekstremalne spready (>{self.anomaly_thresholds["extreme_spread"]*100:.1f}%) w {count} świecach',
                'count': count,
                'max_spread': max_spread,
                'severity': 'warning'
            })
        
        # Anomalia 4: Flash crashes
        flash_crashes = df['close'].pct_change() < self.anomaly_thresholds['flash_crash']
        if flash_crashes.any():
            count = int(flash_crashes.sum())  # Konwersja na int
            min_change = float(df['close'].pct_change().min())  # Konwersja na float
            anomalies.append({
                'type': 'flash_crashes',
                'description': f'Flash crashes (<{self.anomaly_thresholds["flash_crash"]*100:.1f}%) w {count} świecach',
                'count': count,
                'min_change': min_change,
                'severity': 'critical'
            })
        
        # Dodaj do globalnej listy problemów
        self.quality_issues.extend(anomalies)
        
        return {
            'price_anomalies_detected': len(anomalies),
            'anomalies': anomalies,
            'anomaly_types': ['large_jumps', 'outliers', 'extreme_spreads', 'flash_crashes'],
            'total_anomalous_candles': sum(anomaly['count'] for anomaly in anomalies)
        }
    
    def _validate_statistical_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Walidacja wzorców statystycznych.
        
        Sprawdza:
        - Mock data detection (niska unikalność)
        - Volatility analysis
        - Volume patterns
        - Correlation analysis
        
        Args:
            df: DataFrame z danymi OHLCV
            
        Returns:
            Dict: Wyniki analizy statystycznej
        """
        patterns = {}
        statistical_issues = []
        
        # Pattern 1: Mock data detection (niska unikalność)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            unique_ratio = df[col].nunique() / len(df)
            patterns[f'{col}_uniqueness'] = float(unique_ratio)  # Konwersja na float
            
            if unique_ratio < self.statistical_thresholds['uniqueness_min']:
                statistical_issues.append({
                    'type': f'{col}_low_uniqueness',
                    'description': f'Niska unikalność w {col}: {unique_ratio:.3f} (próg: {self.statistical_thresholds["uniqueness_min"]})',
                    'value': float(unique_ratio),  # Konwersja na float
                    'severity': 'warning'
                })
        
        # Pattern 2: Volatility analysis
        volatility = (df['high'] - df['low']) / df['close']
        patterns['volatility_stats'] = {
            'mean': float(volatility.mean()),  # Konwersja na float
            'std': float(volatility.std()),    # Konwersja na float
            'zero_volatility_count': int((volatility == 0).sum()),  # Konwersja na int
            'extreme_volatility_count': int((volatility > self.statistical_thresholds['volatility_max']).sum())  # Konwersja na int
        }
        
        # Sprawdź ekstremalne volatility
        if patterns['volatility_stats']['extreme_volatility_count'] > 0:
            count = patterns['volatility_stats']['extreme_volatility_count']
            statistical_issues.append({
                'type': 'extreme_volatility',
                'description': f'Ekstremalna volatility (>{self.statistical_thresholds["volatility_max"]*100:.1f}%) w {count} świecach',
                'count': count,
                'severity': 'warning'
            })
        
        # Pattern 3: Volume patterns
        patterns['volume_stats'] = {
            'mean': float(df['volume'].mean()),  # Konwersja na float
            'std': float(df['volume'].std()),    # Konwersja na float
            'zero_volume_count': int((df['volume'] == 0).sum()),  # Konwersja na int
            'constant_volume_sequences': self._detect_constant_sequences(df['volume'])
        }
        
        # Sprawdź zerowy volume
        if patterns['volume_stats']['zero_volume_count'] > 0:
            count = patterns['volume_stats']['zero_volume_count']
            statistical_issues.append({
                'type': 'zero_volume',
                'description': f'Zerowy volume w {count} świecach',
                'count': count,
                'severity': 'warning'
            })
        
        # Pattern 4: Correlation analysis
        price_cols = ['open', 'high', 'low', 'close']
        correlation_matrix = df[price_cols].corr()
        # Konwertuj correlation matrix na zwykłe floaty
        patterns['price_correlations'] = {
            col: {inner_col: float(correlation_matrix.loc[col, inner_col]) 
                  for inner_col in price_cols}
            for col in price_cols
        }
        
        # Dodaj do globalnej listy problemów
        self.quality_issues.extend(statistical_issues)
        
        return {
            'statistical_patterns': patterns,
            'statistical_issues': statistical_issues,
            'issues_found': len(statistical_issues)
        }
    
    def _detect_constant_sequences(self, series: pd.Series, min_length: int = 10) -> List[Dict]:
        """
        Wykrywa sekwencje identycznych wartości.
        
        Args:
            series: Seria danych do analizy
            min_length: Minimalna długość sekwencji do wykrycia
            
        Returns:
            List[Dict]: Lista wykrytych sekwencji
        """
        sequences = []
        current_value = None
        current_start = None
        current_length = 0
        
        for i, value in enumerate(series):
            if value == current_value:
                current_length += 1
            else:
                # Zakończ poprzednią sekwencję jeśli była wystarczająco długa
                if current_length >= min_length:
                    sequences.append({
                        'value': current_value,
                        'start_index': current_start,
                        'length': current_length,
                        'end_index': i - 1
                    })
                
                # Rozpocznij nową sekwencję
                current_value = value
                current_start = i
                current_length = 1
        
        # Sprawdź ostatnią sekwencję
        if current_length >= min_length:
            sequences.append({
                'value': current_value,
                'start_index': current_start,
                'length': current_length,
                'end_index': len(series) - 1
            })
        
        return sequences

    # ============================================================================
    # NOWE METODY COMPREHENSIVE SCORING I REPORTING
    # ============================================================================
    
    def _generate_comprehensive_report(self, df_final: pd.DataFrame) -> Dict:
        """
        Generuje komprehensywny raport walidacji (bazowy + jakość).
        
        Args:
            df_final: Finalne dane po przetworzeniu
            
        Returns:
            Dict: Komprehensywny raport
        """
        # Bazowy raport z istniejącej metody
        base_report = self._generate_report(df_final)
        
        # Rozszerzony raport comprehensive
        comprehensive_report = base_report.copy()
        
        # Dodaj wyniki walidacji jakości (jeśli włączona)
        if self.enable_quality_validation and self.validation_results:
            comprehensive_report['quality_validation'] = self.validation_results
            comprehensive_report['comprehensive_quality_score'] = self._calculate_comprehensive_quality_score()
            comprehensive_report['quality_breakdown'] = self._calculate_quality_breakdown()
            comprehensive_report['comprehensive_quality_status'] = self._determine_comprehensive_status()
            comprehensive_report['validation_summary'] = self._generate_validation_summary()
        
        # NOWE: Dodaj wyniki competitive labeling (jeśli włączony)
        if self.enable_competitive_labeling and self.labels_generated > 0:
            comprehensive_report['competitive_labeling'] = {
                'enabled': True,
                'labels_generated': self.labels_generated,
                'label_distribution': self.label_distribution.copy(),
                'labeling_parameters': self.competitive_config.copy(),
                'label_percentages': {
                    label: (count / self.labels_generated * 100) if self.labels_generated > 0 else 0
                    for label, count in self.label_distribution.items()
                },
                'algorithm_summary': {
                    'long_tp_pct': self.competitive_config['LONG_TP_PCT'] * 100,
                    'long_sl_pct': self.competitive_config['LONG_SL_PCT'] * 100,
                    'short_tp_pct': self.competitive_config['SHORT_TP_PCT'] * 100,
                    'short_sl_pct': self.competitive_config['SHORT_SL_PCT'] * 100,
                    'future_window_minutes': self.competitive_config['FUTURE_WINDOW'],
                    'total_prediction_points': self.labels_generated
                }
            }
        elif self.enable_competitive_labeling:
            comprehensive_report['competitive_labeling'] = {
                'enabled': True,
                'labels_generated': 0,
                'error': 'Competitive labeling był włączony ale nie wygenerowano etykiet (prawdopodobnie za mało danych)'
            }
        
        return comprehensive_report
    
    def _calculate_comprehensive_quality_score(self) -> float:
        """
        Oblicza kompleksową ocenę jakości danych.
        
        Returns:
            float: Score 0-100
        """
        # Bazowy score z ciągłości (istniejący)
        base_score = self._calculate_quality_score()  # 0-100
        
        # Kary za problemy OHLCV logic
        ohlcv_penalty = 0
        if 'ohlcv_validation' in self.validation_results:
            issues = self.validation_results['ohlcv_validation']['issues_found']
            ohlcv_penalty = min(issues * 5, 20)  # Max 20% kary
        
        # Kary za anomalie cenowe
        anomaly_penalty = 0
        if 'anomaly_detection' in self.validation_results:
            anomalies = self.validation_results['anomaly_detection']['price_anomalies_detected']
            anomaly_penalty = min(anomalies * 3, 15)  # Max 15% kary
        
        # Kary za problemy statystyczne
        statistical_penalty = 0
        if 'statistical_analysis' in self.validation_results:
            # Analiza wzorców i przyznanie kar
            statistical_issues = self.validation_results['statistical_analysis']['issues_found']
            statistical_penalty = min(statistical_issues * 2, 10)  # Max 10% kary
        
        # Finalna ocena
        comprehensive_score = base_score - ohlcv_penalty - anomaly_penalty - statistical_penalty
        return max(0, min(100, comprehensive_score))
    
    def _calculate_quality_breakdown(self) -> Dict:
        """
        Oblicza szczegółowy breakdown oceny jakości.
        
        Returns:
            Dict: Breakdown scoring
        """
        breakdown = {
            'continuity_score': self._calculate_quality_score()
        }
        
        # OHLCV Logic Score
        if 'ohlcv_validation' in self.validation_results:
            ohlcv_result = self.validation_results['ohlcv_validation']
            if ohlcv_result['ohlcv_logic_valid']:
                breakdown['ohlcv_logic_score'] = 100.0
            else:
                # Kara proporcjonalna do liczby problemów
                penalty = min(ohlcv_result['issues_found'] * 10, 100)
                breakdown['ohlcv_logic_score'] = max(0, 100 - penalty)
        else:
            breakdown['ohlcv_logic_score'] = None  # Nie sprawdzane
        
        # Anomaly Score
        if 'anomaly_detection' in self.validation_results:
            anomaly_result = self.validation_results['anomaly_detection']
            anomaly_count = anomaly_result['price_anomalies_detected']
            if anomaly_count == 0:
                breakdown['anomaly_score'] = 100.0
            else:
                # Kara proporcjonalna do liczby anomalii
                penalty = min(anomaly_count * 5, 100)
                breakdown['anomaly_score'] = max(0, 100 - penalty)
        else:
            breakdown['anomaly_score'] = None  # Nie sprawdzane
        
        # Statistical Score
        if 'statistical_analysis' in self.validation_results:
            statistical_result = self.validation_results['statistical_analysis']
            issues_count = statistical_result['issues_found']
            if issues_count == 0:
                breakdown['statistical_score'] = 100.0
            else:
                # Kara proporcjonalna do liczby problemów
                penalty = min(issues_count * 8, 100)
                breakdown['statistical_score'] = max(0, 100 - penalty)
        else:
            breakdown['statistical_score'] = None  # Nie sprawdzane
        
        return breakdown
    
    def _determine_comprehensive_status(self) -> str:
        """
        Określa status jakości na podstawie comprehensive score.
        
        Returns:
            str: Status jakości
        """
        score = self._calculate_comprehensive_quality_score()
        
        if score >= 95:
            return 'EXCELLENT'
        elif score >= 85:
            return 'GOOD'
        elif score >= 70:
            return 'ACCEPTABLE'
        elif score >= 50:
            return 'PROBLEMATIC'
        else:
            return 'POOR'
    
    def _generate_validation_summary(self) -> Dict:
        """
        Generuje podsumowanie walidacji.
        
        Returns:
            Dict: Podsumowanie walidacji
        """
        # Zlicz problemy według severity
        critical_issues = []
        warnings = []
        
        for issue in self.quality_issues:
            if issue.get('severity') == 'critical':
                critical_issues.append(issue)
            else:
                warnings.append(issue)
        
        # Generuj rekomendacje
        recommendations = []
        
        if critical_issues:
            recommendations.append("Krytyczne problemy z danymi wymagają natychmiastowej uwagi")
        
        if len(warnings) > 10:
            recommendations.append("Duża liczba ostrzeżeń może wskazywać na problemy z jakością źródła danych")
        
        # Sprawdź czy walidacja była włączona
        if not self.enable_quality_validation:
            recommendations.append("Rozważ włączenie walidacji jakości dla lepszej kontroli danych")
        
        # Rekomendacje specyficzne dla typów problemów
        ohlcv_issues = [issue for issue in self.quality_issues if 'invalid' in issue.get('type', '')]
        if ohlcv_issues:
            recommendations.append("Problemy z logiką OHLCV mogą wskazywać na błędy w źródle danych")
        
        anomaly_issues = [issue for issue in self.quality_issues if 'anomal' in issue.get('type', '') or 'jump' in issue.get('type', '')]
        if anomaly_issues:
            recommendations.append("Anomalie cenowe mogą wymagać dodatkowej analizy lub filtrowania")
        
        return {
            'total_issues': len(self.quality_issues),
            'critical_issues': len(critical_issues),
            'warnings': len(warnings),
            'validation_enabled': self.enable_quality_validation,
            'checks_performed': self.quality_checks if self.enable_quality_validation else [],
            'recommendations': recommendations,
            'issue_breakdown': {
                'critical': [issue['type'] for issue in critical_issues],
                'warnings': [issue['type'] for issue in warnings]
            }
        }
    
    # ============================================================================
    # KONIEC COMPREHENSIVE SCORING I REPORTING
    # ============================================================================


# Funkcja pomocnicza dla batch processing
def preprocess_and_save_data(
    input_path: str,
    output_path: str,
    report_path: str,
    min_quality: float = 80.0
) -> dict:
    """
    Funkcja batch processing - zgodnie z nowym workflow.
    
    Args:
        input_path: Ścieżka do surowych danych
        output_path: Ścieżka zapisu czystych danych
        report_path: Ścieżka zapisu raportu
        min_quality: Minimalny próg jakości
        
    Returns:
        dict: Statystyki preprocessing
    """
    # Ta funkcja jest teraz w scripts/preprocess_data.py
    # Pozostawiam jako placeholder dla kompatybilności
    raise NotImplementedError("Użyj scripts/preprocess_data.py dla batch processing")


# Funkcja pomocnicza dla kompatybilności wstecznej
def check_data_continuity(df: pd.DataFrame, timeframe: str = '1m', report_path: str = None) -> Dict:
    """
    Funkcja pomocnicza dla kompatybilności wstecznej.
    
    Args:
        df: DataFrame z danymi OHLCV
        timeframe: Interwał czasowy
        report_path: Ścieżka zapisu raportu (opcjonalne)
        
    Returns:
        Dict: Wyniki sprawdzenia ciągłości
    """
    validator = DataQualityValidator(timeframe=timeframe)
    df_clean, report = validator.check_and_fill_gaps(df)
    
    # Zapisz raport jeśli podano ścieżkę
    if report_path:
        validator.save_report(report, report_path)
    
    # Format kompatybilny ze starą funkcją
    return {
        'status': report['quality_status'].lower(),
        'quality_score': report['quality_score'],
        'gaps': validator.gaps_detected,
        'fixed_df': df_clean,
        'original_df': df,
        'stats': {
            'interpolated': validator.candles_added,
            'forward_filled': 0,  # BRIDGE nie używa forward fill
            'short_gaps': len([g for g in validator.gaps_detected if g.get('gap_type') == 'short']),
            'medium_gaps': len([g for g in validator.gaps_detected if g.get('gap_type') == 'medium']),
            'long_gaps': len([g for g in validator.gaps_detected if g.get('gap_type') == 'long'])
        }
    }


if __name__ == "__main__":
    # Test modułu
    print("🧪 TEST MODUŁU WALIDACJI JAKOŚCI DANYCH")
    print("=" * 50)
    
    # Utwórz testowe dane z lukami
    dates = pd.date_range('2024-01-01 12:00:00', periods=200, freq='1min')
    
    # Usuń niektóre świece aby symulować luki
    gaps_to_create = [
        slice(50, 52),   # 2-minutowa luka (short)
        slice(100, 105), # 5-minutowa luka (short) 
        slice(150, 165), # 15-minutowa luka (medium)
        slice(180, 220)  # 40-minutowa luka (long)
    ]
    
    # Utwórz indeks z lukami
    mask = np.ones(len(dates), dtype=bool)
    for gap in gaps_to_create:
        mask[gap] = False
    
    gapped_dates = dates[mask]
    
    # Utwórz testowe dane OHLCV
    test_data = {
        'open': [50000 + i*10 + np.random.normal(0, 50) for i in range(len(gapped_dates))],
        'high': [50100 + i*10 + np.random.normal(0, 50) for i in range(len(gapped_dates))],
        'low': [49900 + i*10 + np.random.normal(0, 50) for i in range(len(gapped_dates))],
        'close': [50000 + i*10 + np.random.normal(0, 50) for i in range(len(gapped_dates))],
        'volume': [500 + np.random.normal(0, 100) for _ in range(len(gapped_dates))]
    }
    
    df_test = pd.DataFrame(test_data, index=gapped_dates)
    
    # Zapewnij poprawność OHLC
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        df_test.iloc[i, df_test.columns.get_loc('high')] = max(row['open'], row['high'], row['close'])
        df_test.iloc[i, df_test.columns.get_loc('low')] = min(row['open'], row['low'], row['close'])
    
    print(f"📊 Dane testowe:")
    print(f"   Oryginalne świece: {len(dates)}")
    print(f"   Dane z lukami: {len(df_test)}")
    print(f"   Brakujące świece: {len(dates) - len(df_test)}")
    print(f"   Symulowane luki: {len(gaps_to_create)}")
    
    # Test modułu
    validator = DataQualityValidator(timeframe='1m', tolerance_seconds=60)
    df_clean, report = validator.check_and_fill_gaps(df_test)
    
    print(f"\n📈 WYNIKI TESTU:")
    print(f"   Wykryte luki: {report['gaps_detected']}")
    print(f"   Dodane świece: {report['candles_added']}")
    print(f"   Finalne świece: {report['final_candles']}")
    print(f"   Jakość danych: {report['quality_score']:.1f}%")
    print(f"   Status: {report['quality_status']}")
    
    print(f"\n✅ Test zakończony pomyślnie!") 