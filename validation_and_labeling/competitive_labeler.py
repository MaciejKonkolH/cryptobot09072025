"""
Moduł competitive labeling - symulacja jednoczesnych pozycji LONG i SHORT

⚠️ UWAGA: TEN ALGORYTM JEST OSTATECZNY I ZAKAZUJE SIĘ GO ZMIENIAĆ ⚠️

Implementuje rzeczywistą symulację jednoczesnych pozycji zgodnie z algorytmem z planu.
Teraz obsługuje różne formaty etykiet dla training compatibility.
"""
import logging
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Any, Tuple, List, Optional

# Obsługa importów
try:
    from . import config
    from .utils import setup_logging, ProgressReporter
except ImportError:
    # Standalone script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from utils import setup_logging, ProgressReporter

class CompetitiveLabeler:
    """
    Klasa odpowiedzialna za competitive labeling
    

    ✅ DODANA OBSŁUGA TRAINING COMPATIBILITY
    """
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.CompetitiveLabeler")
        self.logger.info(f"CompetitiveLabeler initialized with training compatibility mode: {config.TRAINING_COMPATIBILITY_MODE}")
        if config.TRAINING_COMPATIBILITY_MODE:
            self.logger.info(f"Label format: {config.LABEL_OUTPUT_FORMAT}, dtype: {config.LABEL_DTYPE}")
    
    def generate_labels(self, df: pd.DataFrame, pair_name: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generuje etykiety competitive labeling zgodnie z algorytmem z planu
        
        Args:
            df: DataFrame z features (8 kolumn + datetime index)
            pair_name: Nazwa pary (do logowania)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (dane_z_etykietami, raport_labelowania)
        """
        self.logger.info(f"Rozpoczynam competitive labeling dla {pair_name}")
        
        labeling_report = {
            "input_rows": len(df),
            "labels_generated": 0,
            "hold_due_to_insufficient_data": 0,
            "label_distribution": {
                "SHORT": 0,
                "HOLD": 0,
                "LONG": 0
            },
            "algorithm_version": "competitive_simultaneous_positions_v1.0"
        }
        
        if len(df) == 0:
            self.logger.warning("Brak danych do generowania etykiet")
            return df, labeling_report
        
        # Potrzebujemy OHLC do symulacji - musimy je przywrócić z oryginalnych danych
        # Ten moduł otrzymuje tylko features, więc trzeba przekazać też OHLC oddzielnie
        # Dla uproszczenia implementacji zakładam że otrzymujemy kompletne dane
        
        try:
            # Progress reporter dla długich operacji (co ~2%)
            progress_interval = max(10000, len(df) // 50)  # Co 2% ale minimum 10k
            progress = ProgressReporter(len(df), self.logger, report_every_n_rows=progress_interval)
            
            # Generuj etykiety dla każdego punktu czasowego
            labels = self._generate_labels_for_all_points(df, labeling_report, progress, pair_name)
            
            # ✅ TRAINING COMPATIBILITY: Formatuj etykiety zgodnie z konfiguracją
            df_with_labels = self._format_labels_for_training(df, labels, labeling_report)
            
            labeling_report["output_rows"] = len(df_with_labels)
            labeling_report["labels_generated"] = len(labels)
            
            # Policz dystrybucję etykiet (z raw labels - zawsze int)
            label_counts = pd.Series(labels).value_counts()
            labeling_report["label_distribution"]["SHORT"] = int(label_counts.get(0, 0))
            labeling_report["label_distribution"]["HOLD"] = int(label_counts.get(1, 0))
            labeling_report["label_distribution"]["LONG"] = int(label_counts.get(2, 0))
            
            progress.finish(pair_name)
            
            self.logger.info(
                f"Competitive labeling dla {pair_name} zakończone: "
                f"SHORT={labeling_report['label_distribution']['SHORT']:,}, "
                f"HOLD={labeling_report['label_distribution']['HOLD']:,}, "
                f"LONG={labeling_report['label_distribution']['LONG']:,}"
            )
            
            return df_with_labels, labeling_report
            
        except Exception as e:
            self.logger.error(f"Błąd podczas competitive labeling {pair_name}: {str(e)}")
            raise
    
    def generate_labels_with_ohlc(self, df_features: pd.DataFrame, df_ohlc: pd.DataFrame, 
                                 pair_name: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generuje etykiety competitive labeling z dostępem do danych OHLC
        
        Args:
            df_features: DataFrame z features (8 kolumn)
            df_ohlc: DataFrame z danymi OHLC (potrzebne do symulacji pozycji)
            pair_name: Nazwa pary (do logowania)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (features_z_etykietami, raport_labelowania)
        """
        self.logger.info(f"Rozpoczynam competitive labeling z OHLC dla {pair_name}")
        
        labeling_report = {
            "input_rows": len(df_features),
            "labels_generated": 0,
            "hold_due_to_insufficient_data": 0,
            "label_distribution": {
                "SHORT": 0,
                "HOLD": 0,
                "LONG": 0
            },
            "algorithm_version": "competitive_simultaneous_positions_v1.0_with_ohlc"
        }
        
        if len(df_features) == 0 or len(df_ohlc) == 0:
            self.logger.warning("Brak danych do generowania etykiet")
            return df_features, labeling_report
        
        try:
            # Progress reporter (co ~2%)
            progress_interval = max(5000, len(df_features) // 50)  # Co 2% ale minimum 5k
            progress = ProgressReporter(len(df_features), self.logger, report_every_n_rows=progress_interval)
            
            # ⚠️ GŁÓWNY ALGORYTM COMPETITIVE LABELING - ZAKAZANY DO MODYFIKACJI ⚠️
            labels = self._execute_competitive_labeling_algorithm(
                df_features, df_ohlc, labeling_report, progress, pair_name
            )
            
            # ✅ TRAINING COMPATIBILITY: Formatuj etykiety zgodnie z konfiguracją
            df_with_labels = self._format_labels_for_training(df_features, labels, labeling_report)
            
            # Finalne statystyki
            labeling_report["output_rows"] = len(df_with_labels)
            labeling_report["labels_generated"] = len(labels)
            
            label_counts = pd.Series(labels).value_counts()
            labeling_report["label_distribution"]["SHORT"] = int(label_counts.get(0, 0))
            labeling_report["label_distribution"]["HOLD"] = int(label_counts.get(1, 0))
            labeling_report["label_distribution"]["LONG"] = int(label_counts.get(2, 0))
            
            progress.finish(pair_name)
            
            self.logger.info(
                f"Competitive labeling zakończone: SHORT={labeling_report['label_distribution']['SHORT']:,}, "
                f"HOLD={labeling_report['label_distribution']['HOLD']:,}, "
                f"LONG={labeling_report['label_distribution']['LONG']:,}"
            )
            
            # ✅ TRAINING COMPATIBILITY: Dodaj metadata do raportu
            if config.INCLUDE_TRAINING_METADATA:
                labeling_report["training_metadata"] = config.get_training_metadata()
                labeling_report["training_metadata"]["actual_label_format"] = config.LABEL_OUTPUT_FORMAT
                if config.LABEL_OUTPUT_FORMAT == "onehot":
                    labeling_report["training_metadata"]["label_columns"] = ['label_0', 'label_1', 'label_2']
                else:
                    labeling_report["training_metadata"]["label_columns"] = ['label']
                labeling_report["training_metadata"]["label_mapping"] = {
                    0: "SHORT", 
                    1: "HOLD", 
                    2: "LONG"
                }
            
            return df_with_labels, labeling_report
            
        except Exception as e:
            self.logger.error(f"Błąd podczas competitive labeling: {str(e)}")
            raise
    
    def _format_labels_for_training(self, df_features: pd.DataFrame, raw_labels: List[int], 
                                   report: Dict[str, Any]) -> pd.DataFrame:
        """
        ✅ TRAINING COMPATIBILITY: Formatuj etykiety zgodnie z konfiguracją
        ✅ ZACHOWAJ DATETIME INDEX - nie trać informacji czasowej!
        
        Args:
            df_features: DataFrame z features (z datetime index)
            raw_labels: Lista raw labels [0, 1, 2]
            report: Raport labelowania
            
        Returns:
            DataFrame z features + sformatowane labels + datetime index
        """
        # ✅ ZACHOWAJ DATETIME INDEX przy kopiowaniu
        df_result = df_features.copy()
        labels_array = np.array(raw_labels)

        if config.LABEL_OUTPUT_FORMAT == "onehot":
            # One-hot encoding: [0,1,2] → [[1,0,0],[0,1,0],[0,0,1]]
            self.logger.debug("Konwertuję etykiety do formatu one-hot")
            
            # Utwórz one-hot matrix
            onehot_labels = np.zeros((len(labels_array), 3), dtype=config.LABEL_DTYPE)
            onehot_labels[np.arange(len(labels_array)), labels_array] = 1.0
            
            # Dodaj jako oddzielne kolumny
            df_result['label_0'] = onehot_labels[:, 0]  # SHORT
            df_result['label_1'] = onehot_labels[:, 1]  # HOLD  
            df_result['label_2'] = onehot_labels[:, 2]  # LONG
            
            self.logger.info(f"Etykiety skonwertowane do one-hot: 3 kolumny (label_0, label_1, label_2)")
            
        elif config.LABEL_OUTPUT_FORMAT == "sparse_categorical":
            # Sparse categorical: [0,1,2] jako int32
            self.logger.debug("Konwertuję etykiety do formatu sparse categorical")
            df_result['label'] = labels_array.astype(config.LABEL_DTYPE)
            
        elif config.LABEL_OUTPUT_FORMAT == "int8":
            # Compact int8: [0,1,2] jako int8 (original format)
            self.logger.debug("Zachowuję etykiety w formacie int8")
            df_result['label'] = labels_array.astype(config.LABEL_DTYPE)
            
        else:
            raise ValueError(f"Unsupported LABEL_OUTPUT_FORMAT: {config.LABEL_OUTPUT_FORMAT}")
        
        return df_result
    
    def _execute_competitive_labeling_algorithm(self, df_features: pd.DataFrame, df_ohlc: pd.DataFrame,
                                              report: Dict[str, Any], progress: ProgressReporter,
                                              pair_name: str = "") -> List[int]:
        """
        ⚠️ GŁÓWNY ALGORYTM COMPETITIVE LABELING - ZAKAZ MODYFIKACJI ⚠️
        
        COMPETITIVE LABELING - RZECZYWISTA SYMULACJA JEDNOCZESNYCH POZYCJI
        OPTIMIZED VERSION: Pre-indexing for O(n) complexity instead of O(n²)
        """
        
        # ⚡ PERFORMANCE OPTIMIZATION: Pre-calculate data indices and availability
        timestamps_map, ohlc_data_array, sufficient_data_mask = self._prepare_optimized_data_access(
            df_features, df_ohlc, config.FUTURE_WINDOW
        )
        
        labels = []
        total_rows = len(df_features)
        
        for t in range(total_rows):
            current_timestamp = df_features.index[t]
            
            # ⚡ OPTIMIZED: O(1) lookup instead of O(n) filtering
            if not sufficient_data_mask[t]:
                # JEŚLI NIE → PRZYPISZ label = 1 (HOLD)
                labels.append(1)
                report["hold_due_to_insufficient_data"] += 1
                continue
            
            # JEŚLI TAK → KONTYNUUJ symulację
            
            # ⚡ OPTIMIZED: Direct array access instead of DataFrame lookup
            current_pos = timestamps_map.get(current_timestamp)
            if current_pos is None or current_pos >= len(ohlc_data_array):
                labels.append(1)  # HOLD jeśli brak pozycji
                continue
                
            entry_price = ohlc_data_array[current_pos]['close']
            
            long_tp = entry_price * (1 + config.LONG_TP_PCT / 100)
            long_sl = entry_price * (1 - config.LONG_SL_PCT / 100)
            short_tp = entry_price * (1 - config.SHORT_TP_PCT / 100)
            short_sl = entry_price * (1 + config.SHORT_SL_PCT / 100)
            
            long_active = True
            short_active = True
            
            # ⚡ OPTIMIZED: Sequential array access instead of DataFrame filtering
            future_window_end = min(current_pos + config.FUTURE_WINDOW, len(ohlc_data_array) - 1)
            
            label_assigned = False
            
            # PĘTLA PRZEZ 120 ŚWIEC [t+1, t+120] - OPTIMIZED VERSION
            for future_pos in range(current_pos + 1, future_window_end + 1):
                if future_pos >= len(ohlc_data_array):
                    break
                    
                high_price = ohlc_data_array[future_pos]['high']
                low_price = ohlc_data_array[future_pos]['low']
                
                # SPRAWDŹ WSZYSTKIE ZDARZENIA W KOLEJNOŚCI:
                
                # JEŚLI long_active AND high[i] >= LONG_TP:
                if long_active and high_price >= long_tp:
                    labels.append(2)  # LONG wygrywa
                    label_assigned = True
                    break  # PRZERWIJ całą pętlę
                
                # JEŚLI short_active AND low[i] <= SHORT_TP:
                if short_active and low_price <= short_tp:
                    labels.append(0)  # SHORT wygrywa
                    label_assigned = True
                    break  # PRZERWIJ całą pętlę
                
                # JEŚLI long_active AND low[i] <= LONG_SL:
                if long_active and low_price <= long_sl:
                    long_active = False  # zamykamy pozycję LONG
                    # KONTYNUUJ obserwację SHORT
                
                # JEŚLI short_active AND high[i] >= SHORT_SL:
                if short_active and high_price >= short_sl:
                    short_active = False  # zamykamy pozycję SHORT
                    # KONTYNUUJ obserwację LONG
                
                # SPRAWDŹ status pozycji:
                if not long_active and not short_active:
                    # JEŚLI long_active == False AND short_active == False:
                    labels.append(1)  # HOLD - obie na SL
                    label_assigned = True
                    break  # PRZERWIJ pętlę
                
                # KONTYNUUJ do następnej świecy
            
            # JEŚLI koniec pętli bez TP:
            if not label_assigned:
                labels.append(1)  # HOLD - brak zdarzenia lub tylko SL
            
            # ⚡ OPTIMIZED: Progress reporting aligned with ProgressReporter frequency
            if t % 5000 == 0:
                progress.update(t, pair_name)
        
        return labels
    
    def _prepare_optimized_data_access(self, df_features: pd.DataFrame, df_ohlc: pd.DataFrame, 
                                     future_window: int) -> Tuple[Dict, np.ndarray, List[bool]]:
        """
        ⚡ PERFORMANCE OPTIMIZATION: Pre-calculate data indices and availability
        
        Prepares optimized data structures for O(1) access instead of O(n) filtering
        
        Returns:
            Tuple[Dict, np.ndarray, List[bool]]: (timestamps_map, ohlc_array, sufficient_data_mask)
        """
        self.logger.debug("Preparing optimized data access structures...")
        
        # Create timestamp to position mapping for O(1) lookups
        timestamps_map = {timestamp: pos for pos, timestamp in enumerate(df_ohlc.index)}
        
        # Convert DataFrame to numpy array for faster access
        ohlc_data_array = df_ohlc[['open', 'high', 'low', 'close', 'volume']].to_records()
        
        # Pre-calculate data availability for each timestamp
        sufficient_data_mask = []
        
        for timestamp in df_features.index:
            current_pos = timestamps_map.get(timestamp, -1)
            if current_pos == -1:
                sufficient_data_mask.append(False)
                continue
                
            # Check if we have at least future_window minutes of data ahead
            required_end_pos = current_pos + future_window
            has_sufficient_data = required_end_pos < len(ohlc_data_array)
            sufficient_data_mask.append(has_sufficient_data)
        
        self.logger.debug(f"Optimized structures prepared: {len(timestamps_map)} timestamps mapped, "
                         f"{len(sufficient_data_mask)} availability checks pre-calculated")
        
        return timestamps_map, ohlc_data_array, sufficient_data_mask
    
    def _has_sufficient_future_data(self, df_ohlc: pd.DataFrame, current_timestamp: pd.Timestamp, 
                                   future_minutes: int) -> bool:
        """
        ⚡ LEGACY METHOD: Kept for compatibility, but optimized version used in main algorithm
        Sprawdza czy istnieje wystarczająco danych w przyszłość
        """
        try:
            # Quick position-based check instead of filtering
            current_pos = df_ohlc.index.get_loc(current_timestamp)
            required_end_pos = current_pos + future_minutes
            return required_end_pos < len(df_ohlc)
        except KeyError:
            return False
    
    def _get_entry_price(self, df_ohlc: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[float]:
        """
        ⚡ OPTIMIZED: Direct index access instead of conditional lookup
        Pobiera cenę entry (close) dla danego timestamp
        """
        try:
            return float(df_ohlc.loc[timestamp, 'close'])
        except KeyError:
            return None
    
    def _get_future_window_data(self, df_ohlc: pd.DataFrame, current_timestamp: pd.Timestamp,
                              future_minutes: int) -> pd.DataFrame:
        """
        ⚡ LEGACY METHOD: Kept for compatibility, but optimized version used in main algorithm
        Pobiera dane OHLC dla okna przyszłości [t+1, t+future_minutes]
        """
        try:
            current_pos = df_ohlc.index.get_loc(current_timestamp)
            start_pos = current_pos + 1
            end_pos = min(current_pos + future_minutes, len(df_ohlc) - 1)
            
            if start_pos >= len(df_ohlc):
                return pd.DataFrame()
                
            return df_ohlc.iloc[start_pos:end_pos + 1]
        except (KeyError, IndexError):
            return pd.DataFrame()
    
    def _generate_labels_for_all_points(self, df: pd.DataFrame, report: Dict[str, Any], 
                                       progress: ProgressReporter, pair_name: str = "") -> List[int]:
        """
        Fallback method gdy nie ma dostępu do OHLC
        Generuje etykiety HOLD dla wszystkich punktów
        """
        self.logger.warning(
            "Brak danych OHLC dla competitive labeling, wszystkie etykiety = HOLD. "
            "Użyj generate_labels_with_ohlc() dla pełnej funkcjonalności."
        )
        
        labels = [1] * len(df)  # Wszystkie HOLD
        report["hold_due_to_insufficient_data"] = len(df)
        
        for i in range(0, len(df), 10000):
            progress.update(i, pair_name)
        
        return labels

# ⚠️ DODATKOWE KLASY POMOCNICZE (dozwolone modyfikacje tylko tutaj) ⚠️

class LabelingStatistics:
    """Klasa do analizy statystyk etykietowania"""
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.LabelingStatistics")
    
    def analyze_label_distribution(self, labels: pd.Series) -> Dict[str, Any]:
        """
        Analizuje rozkład etykiet i zwraca statystyki
        ✅ TRAINING COMPATIBILITY: Obsługuje różne formaty etykiet
        """
        
        # ✅ TRAINING COMPATIBILITY: Detect label format
        if config.LABEL_OUTPUT_FORMAT == "onehot":
            # Labels są w kolumnach label_0, label_1, label_2
            if isinstance(labels, pd.DataFrame):
                # Konwertuj one-hot z powrotem do int dla statystyk
                labels_int = labels[['label_0', 'label_1', 'label_2']].values.argmax(axis=1)
                labels = pd.Series(labels_int)
            else:
                self.logger.warning("Expected DataFrame for onehot labels, got Series. Assuming fallback.")
        
        label_counts = labels.value_counts()
        total_labels = len(labels)
        
        stats = {
            "total_labels": total_labels,
            "short_count": int(label_counts.get(0, 0)),
            "hold_count": int(label_counts.get(1, 0)),
            "long_count": int(label_counts.get(2, 0)),
            "short_percentage": (label_counts.get(0, 0) / total_labels * 100) if total_labels > 0 else 0,
            "hold_percentage": (label_counts.get(1, 0) / total_labels * 100) if total_labels > 0 else 0,
            "long_percentage": (label_counts.get(2, 0) / total_labels * 100) if total_labels > 0 else 0,
            "balance_score": self._calculate_balance_score(label_counts, total_labels)
        }
        
        return stats
    
    def _calculate_balance_score(self, label_counts: pd.Series, total_labels: int) -> float:
        """
        Oblicza wynik balansu etykiet (0-100, gdzie 100 = idealny balans 33/33/33)
        """
        if total_labels == 0:
            return 0.0
        
        ideal_percentage = 33.33
        short_pct = (label_counts.get(0, 0) / total_labels * 100)
        hold_pct = (label_counts.get(1, 0) / total_labels * 100)
        long_pct = (label_counts.get(2, 0) / total_labels * 100)
        
        # Oblicz odchylenie od idealnego rozkładu
        deviation = (abs(short_pct - ideal_percentage) + 
                    abs(hold_pct - ideal_percentage) + 
                    abs(long_pct - ideal_percentage)) / 3
        
        # Konwertuj na wynik 0-100 (im mniejsze odchylenie, tym wyższy wynik)
        balance_score = max(0, 100 - deviation * 3)
        
        return round(balance_score, 2) 

def main(input_file: Optional[str] = None):
    """Główna funkcja uruchamiająca proces etykietowania."""
    main_logger = setup_logging(__name__)
    
    input_path = Path(input_file) if input_file else config.INPUT_FILE_PATH
    main_logger.info(f"Rozpoczynam proces etykietowania dla pliku: {input_path}")
    
    if not input_path.exists():
        main_logger.error(f"Plik wejściowy nie istnieje: {input_path}")
        sys.exit(1)
        
    try:
        df_with_features = pd.read_feather(input_path)
        main_logger.info(f"Wczytano {len(df_with_features):,} wierszy danych z cechami.")
        
        # Ustawienie indeksu czasowego, jeśli nie jest ustawiony
        if 'date' in df_with_features.columns and df_with_features.index.name != 'date':
             df_with_features['date'] = pd.to_datetime(df_with_features['date'])
             df_with_features.set_index('date', inplace=True)
        
        pair_name = Path(input_path).stem.split('_')[0]

        labeler = CompetitiveLabeler()
        df_labeled, report = labeler.generate_labels(df_with_features, pair_name)
        
        # --- Tworzenie nazwy pliku wyjściowego ---
        tp_str = str(config.TAKE_PROFIT_PCT * 100).replace('.', 'p')
        sl_str = str(config.STOP_LOSS_PCT * 100).replace('.', 'p')
        base_name = input_path.stem
        
        output_filename = config.OUTPUT_FILENAME_TEMPLATE.format(
            base_name=base_name,
            tp=tp_str,
            sl=sl_str,
            fw=config.FUTURE_WINDOW
        )
        output_path = config.OUTPUT_DIR / output_filename
        
        # Zapisywanie danych
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.reset_index().to_feather(output_path)
        main_logger.info(f"Zapisano dane z etykietami do: {output_path}")

        # Zapisywanie raportu
        report_filename = output_path.with_suffix('.json')
        with open(report_filename, 'w') as f:
            import json
            json.dump(report, f, indent=4, default=str)
        main_logger.info(f"Zapisano raport z etykietowania do: {report_filename}")

        main_logger.info("Proces etykietowania zakończony pomyślnie.")

    except Exception as e:
        main_logger.error(f"Główny proces etykietowania napotkał błąd: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 