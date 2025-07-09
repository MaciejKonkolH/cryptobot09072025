"""
Główny moduł przetwarzania danych
Orchestrator dla całego pipeline'u validation_and_labeling
✅ TRAINING COMPATIBILITY: Obsługuje generowanie training-ready output
"""
import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np

# Obsługa importów - zarówno jako moduł jak i standalone script
try:
    from . import config
    from .utils import (
        setup_logging, find_input_files, load_data_file, save_data_file,
        cleanup_partial_file, get_memory_usage_mb, PerformanceTimer
    )
    from .data_validator import DataValidator
    from .feature_calculator import FeatureCalculator, FeatureQualityValidator, FeatureDistributionAnalyzer
    from .competitive_labeler import CompetitiveLabeler, LabelingStatistics
except ImportError:
    # Standalone script - dodaj bieżący katalog do sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    import config
    from utils import (
        setup_logging, find_input_files, load_data_file, save_data_file,
        cleanup_partial_file, get_memory_usage_mb, PerformanceTimer
    )
    from data_validator import DataValidator
    from feature_calculator import FeatureCalculator, FeatureQualityValidator, FeatureDistributionAnalyzer
    from competitive_labeler import CompetitiveLabeler, LabelingStatistics

class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder który obsługuje numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (bool, np.bool)):
            return bool(obj)
        return super().default(obj)

class ValidationAndLabelingPipeline:
    """
    Główna klasa Pipeline dla przetwarzania danych
    ✅ TRAINING COMPATIBILITY: Generuje training-ready output
    """
    
    def __init__(self):
        self.logger = setup_logging(f"{__name__}.ValidationAndLabelingPipeline")
        
        # Inicjalizuj komponenty
        self.data_validator = DataValidator()
        self.feature_calculator = FeatureCalculator()
        self.feature_quality_validator = FeatureQualityValidator()
        self.feature_distribution_analyzer = FeatureDistributionAnalyzer()
        self.competitive_labeler = CompetitiveLabeler()
        self.labeling_statistics = LabelingStatistics()
        
        self.logger.info("Pipeline validation_and_labeling initialized")
        
        # ✅ TRAINING COMPATIBILITY: Zaloguj konfigurację
        if config.TRAINING_COMPATIBILITY_MODE:
            self.logger.info(f"🎯 TRAINING COMPATIBILITY MODE: ENABLED")
            self.logger.info(f"Label format: {config.LABEL_OUTPUT_FORMAT}, dtype: {config.LABEL_DTYPE}")
            self.logger.info(f"Output files: *_training_ready.feather")
        else:
            self.logger.info("Training compatibility mode: DISABLED")
        
        self.logger.info(f"Konfiguracja: TP={config.LONG_TP_PCT}%, SL={config.LONG_SL_PCT}%, FW={config.FUTURE_WINDOW}min")
    
    def process_single_pair(self, pair_name: str, input_file_path: Path) -> Dict[str, Any]:
        """
        Przetwarza pojedynczą parę walutową
        ✅ TRAINING COMPATIBILITY: Generuje training-ready output z metadata
        
        Args:
            pair_name: Nazwa pary (np. BTCUSDT)
            input_file_path: Ścieżka do pliku wejściowego
            
        Returns:
            dict: Raport przetwarzania z training compatibility metadata
        """
        self.logger.info(f"Rozpoczynam przetwarzanie pary {pair_name}")
        
        timer = PerformanceTimer()
        timer.start()
        
        # Przygotuj raport
        processing_report = {
            "pair": pair_name,
            "input_file": str(input_file_path),
            "config_snapshot": self._get_config_snapshot(),
            "success": False
        }
        
        # ✅ TRAINING COMPATIBILITY: Dodaj metadata do raportu
        if config.TRAINING_COMPATIBILITY_MODE:
            processing_report["training_compatibility"] = {
                "enabled": True,
                "metadata": config.get_training_metadata(),
                "output_filename_suffix": "training_ready"
            }
        
        output_file_path = None
        
        try:
            # KROK 1: Załaduj surowe dane
            self.logger.info(f"Ładuję dane z {input_file_path}")
            raw_data = load_data_file(input_file_path)
            self.logger.info(f"Załadowano {len(raw_data):,} wierszy surowych danych")
            processing_report["input_rows"] = len(raw_data)
            
            # KROK 2: Walidacja i czyszczenie danych
            self.logger.info("Rozpoczynam walidację i czyszczenie danych")
            validated_data, validation_report = self.data_validator.validate_and_clean(raw_data, pair_name)
            processing_report["validation_report"] = validation_report
            
            # 🆕 KROK 2.5: Zapisz raw validated data dla strategii (jeśli włączone)
            if config.SAVE_RAW_VALIDATED_DATA:
                self.logger.info("Zapisuję raw validated data w formacie FreqTrade")
                
                # Przygotuj nazwę pliku w formacie FreqTrade
                # BTCUSDT -> BTC_USDT_USDT-1m-futures.feather
                freqtrade_pair_name = pair_name.replace('/', '_').replace(':', '_')
                raw_validated_filename = f"{freqtrade_pair_name}-1m-futures.feather"
                raw_validated_path = config.RAW_VALIDATED_OUTPUT_PATH / raw_validated_filename
                
                # Przygotuj dane w formacie FreqTrade
                freqtrade_data = validated_data.reset_index()
                
                # Zmień nazwę kolumny datetime -> date (wymagane przez FreqTrade)
                freqtrade_data.rename(columns={'datetime': 'date'}, inplace=True)
                
                # Dodaj timezone UTC (wymagane przez FreqTrade)
                freqtrade_data['date'] = pd.to_datetime(freqtrade_data['date'], utc=True)
                
                # Zapisz w formacie kompatybilnym z FreqTrade
                freqtrade_data.to_feather(raw_validated_path)
                
                self.logger.info(f"💾 Zapisano raw validated data (FreqTrade format): {raw_validated_path}")
                self.logger.info(f"📊 Format: date={freqtrade_data['date'].dtype}, shape={freqtrade_data.shape}")
                processing_report["raw_validated_file"] = str(raw_validated_path)
            
            # KROK 3: Obliczanie features technicznych
            self.logger.info("Rozpoczynam obliczanie features technicznych")
            features_data, features_report = self.feature_calculator.calculate_features(validated_data, pair_name)
            processing_report["features_report"] = features_report
            
            # KROK 4: Walidacja jakości features
            self.logger.info("Rozpoczynam walidację jakości features")
            features_quality_report = self.feature_quality_validator.validate_features_quality(
                features_data, pair_name
            )
            processing_report["features_report"]["quality_validation"] = features_quality_report
            
            # KROK 4.5: Analiza rozkładu wartości features
            self.logger.info("Rozpoczynam analizę rozkładu wartości features")
            distribution_report = self.feature_distribution_analyzer.analyze_feature_distributions(
                features_data, pair_name
            )
            processing_report["features_report"]["distribution_analysis"] = distribution_report
            
            # KROK 5: Competitive labeling (z dostępem do OHLC)
            self.logger.info("Rozpoczynam competitive labeling")
            labeled_data, labeling_report = self.competitive_labeler.generate_labels_with_ohlc(
                features_data, validated_data, pair_name
            )
            processing_report["labeling_report"] = labeling_report
            
            # KROK 6: Statystyki końcowe
            # ✅ TRAINING COMPATIBILITY: Handle different label formats for statistics
            if config.LABEL_OUTPUT_FORMAT == "onehot":
                # For onehot format, pass the DataFrame to handle properly
                label_stats = self.labeling_statistics.analyze_label_distribution(
                    labeled_data[['label_0', 'label_1', 'label_2']]
                )
            else:
                # For int8/sparse_categorical, pass the label column
                label_stats = self.labeling_statistics.analyze_label_distribution(labeled_data['label'])
            
            processing_report["labeling_report"]["detailed_statistics"] = label_stats
            
            # KROK 7: Zapisz finalne dane
            output_filename = config.get_output_filename(pair_name, config.TIMEFRAME)
            output_file_path = config.OUTPUT_DATA_PATH / output_filename
            
            self.logger.info(f"Zapisuję finalne dane do {output_file_path}")
            
            # ✅ TRAINING COMPATIBILITY: Log format info
            if config.TRAINING_COMPATIBILITY_MODE:
                if config.LABEL_OUTPUT_FORMAT == "onehot":
                    label_cols = ['label_0', 'label_1', 'label_2']
                    self.logger.info(f"🎯 Training-ready format: Features + One-hot labels ({label_cols})")
                else:
                    self.logger.info(f"🎯 Training-ready format: Features + {config.LABEL_OUTPUT_FORMAT} labels")
            
            # Sprawdź czy plik istnieje i loguj nadpisanie
            if output_file_path.exists():
                self.logger.warning(f"Nadpisuję istniejący plik: {output_file_path}")
            
            # ✅ ZACHOWAJ TIMESTAMP - konwertuj datetime index na kolumnę przed zapisem
            labeled_data_with_timestamp = labeled_data.reset_index()
            if 'datetime' not in labeled_data_with_timestamp.columns and labeled_data_with_timestamp.columns[0] != 'datetime':
                # Jeśli pierwsza kolumna to datetime index, zmień nazwę
                labeled_data_with_timestamp.rename(columns={labeled_data_with_timestamp.columns[0]: 'datetime'}, inplace=True)
            
            # 🎯 KLUCZOWA POPRAWKA: Upewnij się, że dane treningowe mają strefę czasową UTC
            if 'datetime' in labeled_data_with_timestamp.columns:
                self.logger.info("Konwertuję kolumnę 'datetime' na świadomą strefy czasowej UTC.")
                labeled_data_with_timestamp['datetime'] = pd.to_datetime(labeled_data_with_timestamp['datetime'], utc=True)

            # ✅ KOMPATYBILNOŚĆ Z MODUŁEM TRENUJĄCYM - zmień 'datetime' na 'timestamp'
            if 'datetime' in labeled_data_with_timestamp.columns:
                labeled_data_with_timestamp.rename(columns={'datetime': 'timestamp'}, inplace=True)
                self.logger.info("🔄 Renamed 'datetime' column to 'timestamp' for training module compatibility")
            
            self.logger.info(f"💾 Zapisuję dane z timestamp: {labeled_data_with_timestamp.shape} (kolumny: {list(labeled_data_with_timestamp.columns)})")
            
            save_data_file(labeled_data_with_timestamp, output_file_path, overwrite=config.OVERWRITE_FILES)
            processing_report["output_file"] = str(output_file_path)
            
            # KROK 8: Finalne statystyki
            timer.stop()
            processing_report["processing_time_seconds"] = timer.elapsed_seconds()
            processing_report["memory_usage_mb"] = get_memory_usage_mb()
            processing_report["success"] = True
            
            # Przygotuj statystyki do raportu
            processing_report["statistics"] = {
                "total_rows": len(labeled_data),
                "input_rows": len(raw_data),
                "gaps_filled": validation_report.get("gaps_filled", 0),
                "duplicates_removed": validation_report.get("duplicates_removed", 0),
                "features_anomalies": features_quality_report.get("extreme_values", {}),
                "label_distribution": labeling_report["label_distribution"],
                "processing_time_seconds": processing_report["processing_time_seconds"],
                "rows_per_second": timer.rows_per_second(len(labeled_data)),
                "memory_usage_mb": processing_report["memory_usage_mb"]
            }
            
            # ✅ TRAINING COMPATIBILITY: Add training-specific statistics
            if config.TRAINING_COMPATIBILITY_MODE:
                processing_report["statistics"]["training_compatibility"] = {
                    "label_format": config.LABEL_OUTPUT_FORMAT,
                    "features_count": len(config.get_training_metadata()["features_list"]),
                    "data_shape": list(labeled_data.shape),
                    "memory_efficient": config.LABEL_OUTPUT_FORMAT != "onehot",
                    "ready_for_keras": True
                }
            
            success_msg = f"Przetwarzanie {pair_name} zakończone pomyślnie: {len(labeled_data):,} wierszy w {timer.elapsed_seconds():.2f}s (Speed: {timer.rows_per_second(len(labeled_data)):.0f} rows/s)"
            
            if config.TRAINING_COMPATIBILITY_MODE:
                success_msg += f" → Training-ready output: {output_filename}"
            
            self.logger.info(success_msg)
            
            return processing_report
            
        except Exception as e:
            timer.stop()
            error_msg = f"Błąd podczas przetwarzania {pair_name}: {str(e)}"
            self.logger.error(error_msg)
            
            # Cleanup w przypadku błędu (strategia all-or-nothing)
            if output_file_path and output_file_path.exists():
                cleanup_partial_file(output_file_path, self.logger)
            
            processing_report["success"] = False
            processing_report["error_message"] = str(e)
            processing_report["processing_time_seconds"] = timer.elapsed_seconds()
            processing_report["memory_usage_mb"] = get_memory_usage_mb()
            
            return processing_report
    
    def process_all_pairs(self, input_dir: Optional[Path] = None, 
                         output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Przetwarza wszystkie pary z katalogu input
        
        Args:
            input_dir: Katalog z surowymi danymi (domyślnie z config)
            output_dir: Katalog docelowy (domyślnie z config)
            
        Returns:
            list: Lista raportów dla każdej pary
        """
        if input_dir is None:
            input_dir = config.INPUT_DATA_PATH
        if output_dir is None:
            output_dir = config.OUTPUT_DATA_PATH
        
        self.logger.info(f"Rozpoczynam przetwarzanie wszystkich par z {input_dir}")
        
        # Znajdź wszystkie pliki wejściowe
        input_files = find_input_files(input_dir)
        
        if not input_files:
            self.logger.warning(f"Nie znaleziono plików wejściowych w {input_dir}")
            return []
        
        self.logger.info(f"Znaleziono {len(input_files)} plików do przetworzenia")
        
        all_reports = []
        successful_pairs = 0
        failed_pairs = 0
        
        for pair_name, file_path in input_files:
            try:
                self.logger.info(f"Przetwarzam parę {pair_name} ({successful_pairs + failed_pairs + 1}/{len(input_files)})")
                
                report = self.process_single_pair(pair_name, file_path)
                all_reports.append(report)
                
                if report["success"]:
                    successful_pairs += 1
                    # Zapisz raport dla tej pary
                    self._save_pair_report(report)
                else:
                    failed_pairs += 1
                    self.logger.error(f"Przetwarzanie {pair_name} nieudane: {report.get('error_message', 'Unknown error')}")
                
            except Exception as e:
                failed_pairs += 1
                error_report = {
                    "pair": pair_name,
                    "success": False,
                    "error_message": str(e),
                    "input_file": str(file_path)
                }
                all_reports.append(error_report)
                self.logger.error(f"Krytyczny błąd dla {pair_name}: {str(e)}")
        
        # Statystyki końcowe
        final_msg = f"Przetwarzanie zakończone: {successful_pairs} sukces, {failed_pairs} błędów, łącznie {len(input_files)} par"
        
        if config.TRAINING_COMPATIBILITY_MODE and successful_pairs > 0:
            final_msg += f"\n🎯 Training-ready files generated in: {config.OUTPUT_DATA_PATH}"
        
        self.logger.info(final_msg)
        
        return all_reports
    
    def _get_config_snapshot(self) -> Dict[str, Any]:
        """
        Zwraca snapshot aktualnej konfiguracji
        ✅ TRAINING COMPATIBILITY: Dodaje training-specific config
        """
        base_config = {
            "LONG_TP_PCT": config.LONG_TP_PCT,
            "LONG_SL_PCT": config.LONG_SL_PCT,
            "SHORT_TP_PCT": config.SHORT_TP_PCT,
            "SHORT_SL_PCT": config.SHORT_SL_PCT,
            "FUTURE_WINDOW": config.FUTURE_WINDOW,
            "MA_SHORT_WINDOW": config.MA_SHORT_WINDOW,
            "MA_LONG_WINDOW": config.MA_LONG_WINDOW,
            "MAX_CHANGE_THRESHOLD": config.MAX_CHANGE_THRESHOLD,
            "MAX_VOLUME_CHANGE": config.MAX_VOLUME_CHANGE,
            "TIMEFRAME": config.TIMEFRAME
        }
        
        # ✅ TRAINING COMPATIBILITY: Add training-specific config
        if config.TRAINING_COMPATIBILITY_MODE:
            base_config.update({
                "TRAINING_COMPATIBILITY_MODE": config.TRAINING_COMPATIBILITY_MODE,
                "LABEL_OUTPUT_FORMAT": config.LABEL_OUTPUT_FORMAT,
                "LABEL_DTYPE": config.LABEL_DTYPE,
                "STANDARDIZE_TIMESTAMP_FORMAT": config.STANDARDIZE_TIMESTAMP_FORMAT,
                "INCLUDE_TRAINING_METADATA": config.INCLUDE_TRAINING_METADATA
            })
        
        return base_config
    
    def _save_pair_report(self, report: Dict[str, Any]) -> None:
        """
        Zapisuje raport JSON dla pary
        ✅ TRAINING COMPATIBILITY: Enhanced report with training metadata
        """
        try:
            pair_name = report["pair"]
            report_filename = config.get_report_filename(pair_name, config.TIMEFRAME)
            report_path = config.REPORTS_PATH / report_filename
            
            # Upewnij się że katalog reports istnieje
            config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            self.logger.debug(f"Raport zapisany: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania raportu dla {report.get('pair', 'unknown')}: {e}")

def main():
    """
    Główna funkcja uruchamiająca przetwarzanie
    Uruchomienie: python main.py
    """
    # Ustawienie logowania na poziomie głównym
    logger = setup_logging("main")
    
    logger.info("=" * 80)
    logger.info("URUCHOMIENIE MODUŁU VALIDATION_AND_LABELING")
    # ✅ TRAINING COMPATIBILITY: Enhanced startup message
    if config.TRAINING_COMPATIBILITY_MODE:
        logger.info(f"🎯 TRAINING COMPATIBILITY MODE: {config.LABEL_OUTPUT_FORMAT} labels")
    logger.info("=" * 80)
    
    try:
        # Sprawdź czy katalogi istnieją
        if not config.INPUT_DATA_PATH.exists():
            logger.error(f"Katalog input nie istnieje: {config.INPUT_DATA_PATH}")
            logger.info("Utwórz katalog i umieść w nim pliki .feather lub .csv z danymi OHLCV")
            return
        
        # Inicjalizuj pipeline
        pipeline = ValidationAndLabelingPipeline()
        
        # Przetwórz wszystkie pary
        reports = pipeline.process_all_pairs()
        
        # Podsumowanie
        if reports:
            successful = sum(1 for r in reports if r.get("success", False))
            total = len(reports)
            
            logger.info("=" * 80)
            logger.info(f"PODSUMOWANIE: {successful}/{total} par przetworzonych pomyślnie")
            
            # ✅ TRAINING COMPATIBILITY: Enhanced summary
            if config.TRAINING_COMPATIBILITY_MODE and successful > 0:
                logger.info(f"🎯 TRAINING-READY OUTPUT:")
                logger.info(f"   Format: {config.LABEL_OUTPUT_FORMAT} labels, {config.LABEL_DTYPE} dtype")
                logger.info(f"   Location: {config.OUTPUT_DATA_PATH}")
                logger.info(f"   Files: *_training_ready.feather")
                logger.info(f"   Ready for: Keras/TensorFlow training pipeline")
            
            logger.info("=" * 80)
            
            if successful < total:
                failed_pairs = [r["pair"] for r in reports if not r.get("success", False)]
                logger.warning(f"Nieudane pary: {failed_pairs}")
        else:
            logger.warning("Brak raportów - prawdopodobnie nie znaleziono plików do przetworzenia")
    
    except Exception as e:
        logger.error(f"Krytyczny błąd w main(): {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 