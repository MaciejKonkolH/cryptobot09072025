"""
Moduł obliczania features technicznych z danych OHLCV
Implementuje algorytm obliczania 8 features: zmiany procentowe, średnie kroczące, stosunki do MA, volume features
"""
import logging
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, List

# Obsługa importów
try:
    from . import config
    from .utils import setup_logging, ProgressReporter
except ImportError:
    # Standalone script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from utils import setup_logging, ProgressReporter

class FeatureCalculator:
    """Klasa odpowiedzialna za obliczanie features technicznych"""
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.FeatureCalculator")
    
    def calculate_features(self, df: pd.DataFrame, pair_name: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Oblicza 8 features technicznych zgodnie z algorytmem z planu
        
        Args:
            df: Dane OHLCV po walidacji i wypełnieniu luk
            pair_name: Nazwa pary (do logowania)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (dane_z_features, raport_obliczen)
        """
        self.logger.info(f"Rozpoczynam obliczanie features dla {pair_name}")
        
        features_report = {
            "input_rows": len(df),
            "features_calculated": [],
            "ma_calculation_method": "expanding_window",
            "features_anomalies": {}
        }
        
        if len(df) == 0:
            self.logger.warning("Brak danych do obliczenia features")
            return df, features_report
        
        # Kopia DataFrame żeby nie modyfikować oryginału
        df_features = df.copy()
        
        try:
            # Progress reporter dla długich operacji
            progress = ProgressReporter(len(df), self.logger)
            
            # KROK 1: OBLICZ ZMIANY PROCENTOWE (3 features)
            df_features = self._calculate_percentage_changes(df_features, features_report)
            progress.update(len(df) // 4, pair_name)
            
            # KROK 2: OBLICZ ŚREDNIE KROCZĄCE (na dostępnych danych)
            df_features = self._calculate_moving_averages(df_features, features_report, pair_name)
            progress.update(len(df) // 2, pair_name)
            
            # KROK 3: OBLICZ STOSUNKI DO MA (2 features)
            df_features = self._calculate_ma_ratios(df_features, features_report)
            progress.update(3 * len(df) // 4, pair_name)
            
            # KROK 4: OBLICZ VOLUME FEATURES (3 features)
            df_features = self._calculate_volume_features(df_features, features_report)
            progress.update(len(df), pair_name)
            
            # KROK 4: ZBIERZ WSZYSTKIE OBLICZONE FEATURES
            self.logger.debug("Finalizuję DataFrame z features")

            feature_columns = [
                'high_change', 'low_change', 'close_change', 'volume_change',
                'price_to_ma1440', 'price_to_ma43200',
                'volume_to_ma1440', 'volume_to_ma43200'
            ]

            # ✅ ZACHOWAJ DATETIME INDEX - nie usuwaj informacji czasowej!
            df_final = df_features[feature_columns].copy()
            # Datetime index zostaje zachowany automatycznie

            features_report["output_rows"] = len(df_final)
            features_report["features_calculated"] = feature_columns

            progress.finish(pair_name)

            self.logger.info(
                f"Features dla {pair_name} obliczone: {len(feature_columns)} kolumn, "
                f"{len(df_final):,} wierszy"
            )

            # 🔥 KLUCZOWA POPRAWKA: Odrzuć pierwsze 30 dni (43200 świec), aby zapewnić
            # że wszystkie wartości MA są obliczone na pełnym oknie historycznym.
            # To gwarantuje 100% zgodność z logiką bufora strategii.
            warmup_period = config.MA_LONG_WINDOW
            if len(df_final) > warmup_period:
                self.logger.info(f"Odrzucam okres rozgrzewkowy {warmup_period} świec dla zapewnienia poprawności MA...")
                df_final = df_final.iloc[warmup_period:].copy()
                self.logger.info(f"✅ Finalna liczba wierszy po odrzuceniu okresu rozgrzewkowego: {len(df_final):,}")
            else:
                self.logger.warning(
                    f"Za mało danych ({len(df_final)}) do odrzucenia pełnego okresu rozgrzewkowego ({warmup_period}). "
                    f"Zwracam pusty DataFrame, aby uniknąć błędów."
                )
                # Zwróć pusty dataframe z tymi samymi kolumnami
                return pd.DataFrame(columns=df_final.columns), features_report
            
            return df_final, features_report
            
        except Exception as e:
            self.logger.error(f"Błąd podczas obliczania features {pair_name}: {str(e)}")
            raise
    
    def _calculate_percentage_changes(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM OBLICZANIA FEATURES - KROK 1: OBLICZ ZMIANY PROCENTOWE (3 features)
        """
        self.logger.debug("Obliczam zmiany procentowe")
        
        # high_change = (high[t] - close[t-1]) / close[t-1] * 100
        df['close_prev'] = df['close'].shift(1)
        df['high_change'] = ((df['high'] - df['close_prev']) / df['close_prev'] * 100)
        
        # low_change = (low[t] - close[t-1]) / close[t-1] * 100  
        df['low_change'] = ((df['low'] - df['close_prev']) / df['close_prev'] * 100)
        
        # close_change = (close[t] - close[t-1]) / close[t-1] * 100
        df['close_change'] = ((df['close'] - df['close_prev']) / df['close_prev'] * 100)
        
        # Usuń pomocniczą kolumnę
        df.drop(['close_prev'], axis=1, inplace=True)
        
        # Pierwsza świeca będzie miała NaN. Wypełnij metodą back-fill,
        # aby zreplikować zachowanie Freqtrade i zapewnić spójność danych.
        for col in ['high_change', 'low_change', 'close_change']:
            df[col] = df[col].bfill()
        
        # Po bfill() pierwszy wiersz może nadal być NaN, jeśli cały zbiór jest krótki.
        # Wypełnij zerem jako ostateczny fallback.
        df.fillna(0, inplace=True)
        
        self.logger.debug("Zmiany procentowe obliczone")
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame, report: Dict[str, Any], 
                                 pair_name: str = "") -> pd.DataFrame:
        """
        ALGORYTM OBLICZANIA FEATURES - KROK 2: OBLICZ ŚREDNIE KROCZĄCE (na dostępnych danych)
        """
        self.logger.debug("Obliczam średnie kroczące na dostępnych danych")
        
        # MA_1440 (1 dzień = 1440 minut) - średnia na dostępnych danych
        df['ma_close_1440'] = self._calculate_expanding_ma(df['close'], config.MA_SHORT_WINDOW)
        
        # MA_43200 (30 dni = 43200 minut) - średnia na dostępnych danych  
        df['ma_close_43200'] = self._calculate_expanding_ma(df['close'], config.MA_LONG_WINDOW)
        
        # Średnie kroczące dla volume (potrzebne do volume features)
        df['ma_volume_1440'] = self._calculate_expanding_ma(df['volume'], config.MA_SHORT_WINDOW)
        df['ma_volume_43200'] = self._calculate_expanding_ma(df['volume'], config.MA_LONG_WINDOW)
        
        self.logger.debug(
            f"Średnie kroczące obliczone: okna {config.MA_SHORT_WINDOW} i {config.MA_LONG_WINDOW}"
        )
        
        return df
    
    def _calculate_expanding_ma(self, series: pd.Series, max_window: int) -> pd.Series:
        """
        Oblicza średnią kroczącą na dostępnych danych zgodnie ze specyfikacją:
        - Świeca 1: MA = current_value
        - Świeca 2: MA = (value1 + value2) / 2
        - ...
        - Świeca max_window+: MA = mean(value[t-max_window+1:t+1])
        
        OPTIMIZED VERSION: O(n) complexity using pandas built-in methods
        """
        # FAZA 1: Expanding window (rosnące okno do max_window)
        expanding_ma = series.expanding().mean()
        
        # FAZA 2: Rolling window (stałe okno max_window) 
        rolling_ma = series.rolling(window=max_window, min_periods=1).mean()
        
        # POŁĄCZ: expanding do max_window, potem rolling dla reszty
        result = expanding_ma.copy()
        
        # Jeśli mamy więcej danych niż max_window, użyj rolling dla dalszych świec
        if len(series) > max_window:
            result.iloc[max_window-1:] = rolling_ma.iloc[max_window-1:]
        
        return result
    
    def _calculate_ma_ratios(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM OBLICZANIA FEATURES - KROK 3: OBLICZ STOSUNKI DO MA (2 features)
        """
        self.logger.debug("Obliczam stosunki ceny do średnich kroczących")
        
        # price_to_ma1440 = close[t] / MA_1440[t]
        df['price_to_ma1440'] = df['close'] / df['ma_close_1440']
        
        # price_to_ma43200 = close[t] / MA_43200[t]
        df['price_to_ma43200'] = df['close'] / df['ma_close_43200']
        
        # Zabezpieczenie przed dzieleniem przez zero (nie powinno się zdarzyć, ale...)
        df.loc[:, 'price_to_ma1440'] = df['price_to_ma1440'].replace([np.inf, -np.inf], 1.0)
        df.loc[:, 'price_to_ma43200'] = df['price_to_ma43200'].replace([np.inf, -np.inf], 1.0)
        
        self.logger.debug("Stosunki do MA obliczone")
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """
        ALGORYTM OBLICZANIA FEATURES - KROK 4: OBLICZ VOLUME FEATURES (3 features)
        """
        self.logger.debug("Obliczam volume features")
        
        # 🔥 Usunięto epsilon, aby zapewnić 100% zgodność z logiką Freqtrade.
        # Dodano obsługę dzielenia przez zero przez zastąpienie `inf` wartościami.
        
        # volume_to_ma1440 = volume[t] / MA_volume_1440[t]
        df['volume_to_ma1440'] = df['volume'] / df['ma_volume_1440']
        
        # volume_to_ma43200 = volume[t] / MA_volume_43200[t]
        df['volume_to_ma43200'] = df['volume'] / df['ma_volume_43200']
        
        # volume_change = (volume[t] - volume[t-1]) / volume[t-1] * 100
        df['volume_prev'] = df['volume'].shift(1)
        df['volume_change'] = ((df['volume'] - df['volume_prev']) / df['volume_prev'] * 100)
        
        # Usuń pomocniczą kolumnę
        df.drop(['volume_prev'], axis=1, inplace=True)
        
        # Zastąp potencjalne NaN/Inf, które mogły powstać, bezpiecznymi wartościami
        # Użyj bfill() dla volume_change, aby zachować spójność z Freqtrade
        df['volume_change'] = df['volume_change'].bfill()
        
        # 🔥 Zabezpieczenie przed dzieleniem przez zero: zastąp nieskończoności zerami/jedynkami
        df['volume_change'] = df['volume_change'].replace([np.inf, -np.inf], 0)
        df['volume_to_ma1440'] = df['volume_to_ma1440'].replace([np.inf, -np.inf], 1)
        df['volume_to_ma43200'] = df['volume_to_ma43200'].replace([np.inf, -np.inf], 1)
        
        # Fallback na zero/jeden, jeśli pierwszy wiersz nadal jest NaN
        df.loc[:, 'volume_change'] = df['volume_change'].fillna(0)
        df.loc[:, 'volume_to_ma1440'] = df['volume_to_ma1440'].fillna(1.0)
        df.loc[:, 'volume_to_ma43200'] = df['volume_to_ma43200'].fillna(1.0)
        
        # Usuń pomocnicze kolumny MA (nie są potrzebne w finalnych danych)
        df.drop(['ma_close_1440', 'ma_close_43200', 'ma_volume_1440', 'ma_volume_43200'], 
                axis=1, inplace=True)
        
        self.logger.debug("Volume features obliczone")
        
        return df

class FeatureQualityValidator:
    """Klasa do walidacji jakości wygenerowanych features"""
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.FeatureQualityValidator")
    
    def validate_features_quality(self, df: pd.DataFrame, pair_name: str = "") -> Dict[str, Any]:
        """
        ALGORYTM WALIDACJI JAKOŚCI FEATURES
        
        Args:
            df: DataFrame z obliczonymi features
            pair_name: Nazwa pary (do logowania)
            
        Returns:
            Dict: Raport anomalii w features
        """
        self.logger.debug(f"Waliduje jakość features dla {pair_name}")
        
        anomalies_report = {
            "total_anomalies": 0,
            "nan_values": {},
            "inf_values": {},
            "extreme_values": {},
            "warnings": []
        }
        
        # SPRAWDŹ WARTOŚCI NaN/Inf
        self._check_nan_inf_values(df, anomalies_report)
        
        # WYKRYJ EKSTREMALNE WARTOŚCI
        self._detect_extreme_values(df, anomalies_report)
        
        # WYGENERUJ KOMUNIKATY OSTRZEGAWCZE
        self._generate_warning_messages(anomalies_report, pair_name)
        
        return anomalies_report
    
    def _check_nan_inf_values(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Sprawdza wartości NaN/Inf w każdej kolumnie features"""
        
        feature_columns = ['high_change', 'low_change', 'close_change', 'volume_change',
                          'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200']
        
        for col in feature_columns:
            if col in df.columns:
                # Policz wartości NaN
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    report["nan_values"][col] = nan_count
                    report["total_anomalies"] += nan_count
                    self.logger.warning(f"Feature validation: {nan_count} wartości NaN w {col}")
                
                # Policz wartości Inf/-Inf
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    report["inf_values"][col] = inf_count
                    report["total_anomalies"] += inf_count
                    self.logger.warning(f"Feature validation: {inf_count} wartości Inf w {col}")
    
    def _detect_extreme_values(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Wykrywa ekstremalne wartości w features"""
        
        # ZMIANY PROCENTOWE (high_change, low_change, close_change)
        change_columns = ['high_change', 'low_change', 'close_change']
        for col in change_columns:
            if col in df.columns:
                extreme_changes = (df[col].abs() > config.MAX_CHANGE_THRESHOLD)
                if extreme_changes.any():
                    extreme_count = extreme_changes.sum()
                    report["extreme_values"][f"{col}_extreme"] = extreme_count
                    report["total_anomalies"] += extreme_count
                    self.logger.warning(
                        f"Feature validation: {extreme_count} ekstremalnych zmian w {col} "
                        f"(>{config.MAX_CHANGE_THRESHOLD}%)"
                    )
        
        # STOSUNKI DO MA (price_to_ma1440, price_to_ma43200)
        ma_ratio_columns = ['price_to_ma1440', 'price_to_ma43200']
        for col in ma_ratio_columns:
            if col in df.columns:
                # Sprawdź czy stosunek <= 0
                zero_or_negative = (df[col] <= 0)
                if zero_or_negative.any():
                    negative_count = zero_or_negative.sum()
                    report["extreme_values"][f"{col}_negative"] = negative_count
                    report["total_anomalies"] += negative_count
                    self.logger.warning(f"Feature validation: {negative_count} wartości <= 0 w {col}")
                
                # Sprawdź czy stosunek > MAX_MA_RATIO (300% od MA)
                extreme_ratios = (df[col] > config.MAX_MA_RATIO)
                if extreme_ratios.any():
                    extreme_count = extreme_ratios.sum()
                    report["extreme_values"][f"{col}_extreme"] = extreme_count
                    report["total_anomalies"] += extreme_count
                    self.logger.warning(
                        f"Feature validation: {extreme_count} ekstremalnych stosunków w {col} "
                        f"(>{config.MAX_MA_RATIO})"
                    )
        
        # VOLUME FEATURES
        # volume_change
        if 'volume_change' in df.columns:
            extreme_volume_changes = (df['volume_change'].abs() > config.MAX_VOLUME_CHANGE)
            if extreme_volume_changes.any():
                extreme_count = extreme_volume_changes.sum()
                report["extreme_values"]["volume_change_extreme"] = extreme_count
                report["total_anomalies"] += extreme_count
                self.logger.warning(
                    f"Feature validation: {extreme_count} ekstremalnych zmian volume "
                    f"(>{config.MAX_VOLUME_CHANGE}%)"
                )
        
        # volume_to_ma ratio
        volume_ratio_columns = ['volume_to_ma1440', 'volume_to_ma43200']
        for col in volume_ratio_columns:
            if col in df.columns:
                zero_or_negative = (df[col] <= 0)
                if zero_or_negative.any():
                    negative_count = zero_or_negative.sum()
                    report["extreme_values"][f"{col}_negative"] = negative_count
                    report["total_anomalies"] += negative_count
                    self.logger.warning(f"Feature validation: {negative_count} wartości <= 0 w {col}")
    
    def _generate_warning_messages(self, report: Dict[str, Any], pair_name: str) -> None:
        """Generuje komunikaty ostrzegawcze"""
        
        if report["total_anomalies"] > 0:
            warning_msg = f"Feature validation: {report['total_anomalies']} anomalii wykrytych w {pair_name}"
            report["warnings"].append(warning_msg)
            self.logger.warning(warning_msg)
            
            # Szczegółowe komunikaty
            if report["nan_values"]:
                for col, count in report["nan_values"].items():
                    msg = f"NaN values: {count} w {col}"
                    report["warnings"].append(msg)
            
            if report["inf_values"]:
                for col, count in report["inf_values"].items():
                    msg = f"Inf values: {count} w {col}"
                    report["warnings"].append(msg)
            
            if report["extreme_values"]:
                for anomaly_type, count in report["extreme_values"].items():
                    msg = f"Extreme values: {count} przypadków {anomaly_type}"
                    report["warnings"].append(msg)
        else:
            self.logger.info(f"Feature validation: Brak anomalii w {pair_name}")
            report["warnings"].append(f"Brak anomalii w features dla {pair_name}") 

class FeatureDistributionAnalyzer:
    """Klasa do analizy rozkładu wartości features"""
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.FeatureDistributionAnalyzer")
    
    def analyze_feature_distributions(self, df: pd.DataFrame, pair_name: str = "") -> Dict[str, Any]:
        """
        Analizuje rozkład wartości dla wszystkich features
        
        Args:
            df: DataFrame z obliczonymi features
            pair_name: Nazwa pary (do logowania)
            
        Returns:
            Dict: Szczegółowy raport rozkładu wartości
        """
        self.logger.info(f"Rozpoczynam analizę rozkładu wartości features dla {pair_name}")
        
        distribution_report = {
            "pair": pair_name,
            "total_rows": len(df),
            "features_analyzed": [],
            "distributions": {}
        }
        
        if len(df) == 0:
            self.logger.warning("Brak danych do analizy rozkładu")
            return distribution_report
        
        # Lista wszystkich features do analizy
        feature_columns = [
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200', 
            'volume_to_ma1440', 'volume_to_ma43200'
        ]
        
        self.logger.info("=" * 80)
        self.logger.info(f"📊 ANALIZA ROZKŁADU WARTOŚCI FEATURES - {pair_name}")
        self.logger.info("=" * 80)
        
        for feature_name in feature_columns:
            if feature_name in df.columns:
                feature_distribution = self._analyze_single_feature_distribution(
                    df[feature_name], feature_name, len(df)
                )
                distribution_report["features_analyzed"].append(feature_name)
                distribution_report["distributions"][feature_name] = feature_distribution
                
                # Loguj szczegóły dla tej feature
                self._log_feature_distribution(feature_distribution, feature_name)
        
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Analiza rozkładu zakończona: {len(distribution_report['features_analyzed'])} features przeanalizowanych")
        self.logger.info("=" * 80)
        
        return distribution_report
    
    def _analyze_single_feature_distribution(self, series: pd.Series, feature_name: str, 
                                           total_rows: int) -> Dict[str, Any]:
        """
        Analizuje rozkład wartości dla pojedynczej feature
        
        Args:
            series: Seria z wartościami feature
            feature_name: Nazwa feature
            total_rows: Łączna liczba wierszy
            
        Returns:
            Dict: Szczegółowy raport rozkładu dla tej feature
        """
        # Podstawowe statystyki
        min_val = float(series.min())
        max_val = float(series.max())
        mean_val = float(series.mean())
        std_val = float(series.std())
        median_val = float(series.median())
        
        # Podziel zakres na 10 równych przedziałów
        bins = np.linspace(min_val, max_val, 11)  # 11 punktów = 10 przedziałów
        
        # Oblicz histogram
        counts, bin_edges = np.histogram(series, bins=bins)
        
        # Przygotuj szczegóły przedziałów
        bin_details = []
        for i in range(len(counts)):
            bin_start = float(bin_edges[i])
            bin_end = float(bin_edges[i + 1])
            count = int(counts[i])
            percentage = (count / total_rows) * 100 if total_rows > 0 else 0
            
            bin_details.append({
                "bin_number": i + 1,
                "range_start": bin_start,
                "range_end": bin_end,
                "count": count,
                "percentage": percentage
            })
        
        return {
            "feature_name": feature_name,
            "basic_stats": {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "range": max_val - min_val
            },
            "distribution_bins": bin_details,
            "total_values": total_rows
        }
    
    def _log_feature_distribution(self, distribution: Dict[str, Any], feature_name: str) -> None:
        """Loguje szczegóły rozkładu dla pojedynczej feature"""
        
        stats = distribution["basic_stats"]
        bins = distribution["distribution_bins"]
        
        self.logger.info(f"\n📈 ROZKŁAD WARTOŚCI - {feature_name}:")
        self.logger.info(f"   Zakres: [{stats['min']:.3f}, {stats['max']:.3f}] (rozpiętość: {stats['range']:.3f})")
        self.logger.info(f"   Średnia: {stats['mean']:.3f}, Mediana: {stats['median']:.3f}, Odch.std: {stats['std']:.3f}")
        
        # Loguj przedziały z największą liczbą wartości
        sorted_bins = sorted(bins, key=lambda x: x['count'], reverse=True)
        
        self.logger.info(f"   📊 Rozkład w 10 przedziałach:")
        for bin_info in bins:
            bin_num = bin_info['bin_number']
            start = bin_info['range_start']
            end = bin_info['range_end']
            count = bin_info['count']
            pct = bin_info['percentage']
            
            # Wizualizacja słupka
            bar_length = int(pct / 2)  # Maksymalnie 50 znaków dla 100%
            bar = "█" * bar_length
            
            self.logger.info(f"      Przedział {bin_num:2d}: [{start:8.3f}, {end:8.3f}] → {count:8,} wartości ({pct:5.1f}%) {bar}")
        
        # Pokaż top 3 przedziały
        self.logger.info(f"   🏆 Top 3 przedziały:")
        for i, bin_info in enumerate(sorted_bins[:3]):
            self.logger.info(f"      {i+1}. Przedział {bin_info['bin_number']}: {bin_info['count']:,} wartości ({bin_info['percentage']:.1f}%)") 