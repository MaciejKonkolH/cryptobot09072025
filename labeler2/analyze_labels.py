"""
Skrypt do analizy rozkładu etykiet w pliku wynikowym z modułu etykietowania.
Analizuje rozkład etykiet dla każdego poziomu TP/SL i generuje raport.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Dodaj ścieżkę do głównego katalogu
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import labeler2.config as config
except ImportError:
    import config

def setup_logging():
    """Konfiguruje system logowania."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class LabelAnalyzer:
    """Klasa do analizy rozkładu etykiet."""
    
    # Mapowanie etykiet na nazwy
    LABEL_NAMES = {
        0: "PROFIT_SHORT",
        1: "TIMEOUT_HOLD", 
        2: "PROFIT_LONG",
        3: "LOSS_SHORT",
        4: "LOSS_LONG",
        5: "CHAOS_HOLD"
    }
    
    def __init__(self):
        """Inicjalizuje analizator."""
        self.label_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Wczytuje dane z pliku feather."""
        logger.info(f"Wczytywanie danych z: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Plik nie istnieje: {file_path}")
            return None
            
        try:
            df = pd.read_feather(file_path)
            logger.info(f"Wczytano {len(df):,} wierszy danych")
            
            # Znajdź kolumny z etykietami
            self.label_columns = [col for col in df.columns if col.startswith('label_')]
            logger.info(f"Znaleziono {len(self.label_columns)} kolumn z etykietami: {self.label_columns}")
            
            return df
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania danych: {e}")
            return None
    
    def analyze_label_distribution(self, df: pd.DataFrame) -> dict:
        """Analizuje rozkład etykiet dla każdego poziomu."""
        results = {}
        
        for label_col in self.label_columns:
            logger.info(f"\n--- Analiza kolumny: {label_col} ---")
            
            # Podstawowe statystyki
            value_counts = df[label_col].value_counts().sort_index()
            total_rows = len(df)
            
            # Przygotuj wyniki
            level_results = {
                'total_rows': total_rows,
                'distribution': {},
                'percentages': {},
                'summary': {}
            }
            
            print(f"\n📊 Rozkład etykiet dla {label_col}:")
            print("=" * 60)
            
            for label_value, count in value_counts.items():
                percentage = (count / total_rows) * 100
                label_name = self.LABEL_NAMES.get(label_value, f"UNKNOWN_{label_value}")
                
                level_results['distribution'][label_value] = count
                level_results['percentages'][label_value] = percentage
                
                print(f"  {label_name:15} | {count:8,} | {percentage:6.2f}%")
            
            # Dodatkowe statystyki
            profit_labels = value_counts.get(0, 0) + value_counts.get(2, 0)  # PROFIT_SHORT + PROFIT_LONG
            loss_labels = value_counts.get(3, 0) + value_counts.get(4, 0)     # LOSS_SHORT + LOSS_LONG
            hold_labels = value_counts.get(1, 0) + value_counts.get(5, 0)     # TIMEOUT_HOLD + CHAOS_HOLD
            
            profit_pct = (profit_labels / total_rows) * 100
            loss_pct = (loss_labels / total_rows) * 100
            hold_pct = (hold_labels / total_rows) * 100
            
            level_results['summary'] = {
                'profit_labels': profit_labels,
                'loss_labels': loss_labels,
                'hold_labels': hold_labels,
                'profit_percentage': profit_pct,
                'loss_percentage': loss_pct,
                'hold_percentage': hold_pct
            }
            
            print("=" * 60)
            print(f"📈 Podsumowanie:")
            print(f"  Zyski (PROFIT):     {profit_labels:8,} | {profit_pct:6.2f}%")
            print(f"  Straty (LOSS):      {loss_labels:8,} | {loss_pct:6.2f}%")
            print(f"  Bez zmian (HOLD):    {hold_labels:8,} | {hold_pct:6.2f}%")
            
            results[label_col] = level_results
        
        return results
    
    def generate_report(self, results: dict, output_file: str = None):
        """Generuje raport z analizy."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"label_analysis_report_{timestamp}.txt"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAPORT ANALIZY ROZKŁADU ETYKIET")
        report_lines.append("=" * 80)
        report_lines.append(f"Data generowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for label_col, data in results.items():
            report_lines.append(f"📊 ANALIZA KOLUMNY: {label_col}")
            report_lines.append("-" * 60)
            
            # Rozkład szczegółowy
            report_lines.append("Rozkład szczegółowy:")
            for label_value, count in data['distribution'].items():
                percentage = data['percentages'][label_value]
                label_name = self.LABEL_NAMES.get(label_value, f"UNKNOWN_{label_value}")
                report_lines.append(f"  {label_name:15} | {count:8,} | {percentage:6.2f}%")
            
            report_lines.append("")
            
            # Podsumowanie
            summary = data['summary']
            report_lines.append("Podsumowanie:")
            report_lines.append(f"  Zyski (PROFIT):     {summary['profit_labels']:8,} | {summary['profit_percentage']:6.2f}%")
            report_lines.append(f"  Straty (LOSS):      {summary['loss_labels']:8,} | {summary['loss_percentage']:6.2f}%")
            report_lines.append(f"  Bez zmian (HOLD):    {summary['hold_labels']:8,} | {summary['hold_percentage']:6.2f}%")
            report_lines.append("")
            report_lines.append("")
        
        # Zapisz raport
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Raport zapisany do: {output_file}")
        return output_file

def main():
    """Główna funkcja programu."""
    parser = argparse.ArgumentParser(description="Analizator rozkładu etykiet")
    parser.add_argument(
        '--input',
        type=str,
        help="Ścieżka do pliku z etykietami (feather)"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Ścieżka do pliku raportu (opcjonalnie)"
    )
    parser.add_argument(
        '--auto-find',
        action='store_true',
        help="Automatycznie znajdź najnowszy plik z etykietami"
    )
    
    args = parser.parse_args()
    
    # Określ plik wejściowy
    if args.auto_find:
        # Znajdź najnowszy plik z etykietami
        output_dir = Path(config.OUTPUT_DIR)
        if output_dir.exists():
            label_files = list(output_dir.glob("*labels*.feather"))
            if label_files:
                input_file = max(label_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Znaleziono najnowszy plik: {input_file}")
            else:
                logger.error("Nie znaleziono plików z etykietami w katalogu output")
                return
        else:
            logger.error(f"Katalog output nie istnieje: {output_dir}")
            return
    elif args.input:
        input_file = args.input
    else:
        # Domyślny plik
        input_file = str(Path(config.OUTPUT_DIR) / "orderbook_ohlc_labels_FW-120_levels-3.feather")
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(input_file):
        logger.error(f"Plik nie istnieje: {input_file}")
        return
    
    # Wykonaj analizę
    analyzer = LabelAnalyzer()
    df = analyzer.load_data(input_file)
    
    if df is not None:
        results = analyzer.analyze_label_distribution(df)
        analyzer.generate_report(results, args.output)
        
        logger.info("--- Analiza zakończona pomyślnie ---")
    else:
        logger.error("Nie udało się wczytać danych")

if __name__ == "__main__":
    main() 