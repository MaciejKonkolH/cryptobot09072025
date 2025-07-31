#!/usr/bin/env python3
"""
Skrypt sprawdzający ciągłość danych orderbook
Sprawdza czy w plikach orderbook_filled_*.feather nie ma luk czasowych
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
from pathlib import Path
import logging
import sys
from typing import List, Dict, Optional, Tuple

# Import konfiguracji
from config import PAIRS, LOGGING_CONFIG

def setup_logging():
    """Konfiguruje system logowania"""
    log_file = Path("continuity_check.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_orderbook_data(symbol: str, logger) -> Optional[pd.DataFrame]:
    """Wczytuje dane order book dla jednej pary"""
    feather_file = Path("orderbook_completed") / f"orderbook_filled_{symbol}.feather"
    
    logger.info(f"Wczytuję dane z {feather_file}...")
    
    if not os.path.exists(feather_file):
        logger.error(f"Plik {feather_file} nie istnieje!")
        return None
    
    try:
        df = pd.read_feather(feather_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Wczytano {len(df):,} wierszy")
        logger.info(f"Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
        
        return df
    except Exception as e:
        logger.error(f"Błąd wczytania {feather_file}: {e}")
        return None

def check_continuity(df: pd.DataFrame, symbol: str, logger) -> Dict:
    """Sprawdza ciągłość danych orderbook"""
    logger.info(f"Sprawdzam ciągłość danych dla {symbol}...")
    
    # KROK 1: Sprawdź czy timestamps są co minutę
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Oblicz różnice czasowe
    df_sorted['time_diff'] = df_sorted['timestamp'].diff()
    
    # Znajdź luki (różnica > 1 minuta)
    gaps = df_sorted[df_sorted['time_diff'] > timedelta(minutes=1)].copy()
    
    # KROK 2: Sprawdź czy wszystkie minuty są obecne
    expected_start = df_sorted['timestamp'].min().replace(second=0, microsecond=0)
    expected_end = df_sorted['timestamp'].max().replace(second=0, microsecond=0)
    
    # Generuj oczekiwane minuty
    expected_minutes = pd.date_range(start=expected_start, end=expected_end, freq='1min')
    actual_minutes = df_sorted['timestamp'].dt.floor('min').unique()
    
    missing_minutes = set(expected_minutes) - set(actual_minutes)
    extra_minutes = set(actual_minutes) - set(expected_minutes)
    
    # KROK 3: Sprawdź duplikaty
    duplicates = df_sorted[df_sorted.duplicated(subset=['timestamp'], keep=False)]
    
    # KROK 4: Sprawdź czy są 2 snapshoty na minutę
    minute_counts = df_sorted.groupby(df_sorted['timestamp'].dt.floor('min')).size()
    minutes_with_one_snapshot = minute_counts[minute_counts == 1]
    minutes_with_more_than_two = minute_counts[minute_counts > 2]
    
    # KROK 5: Sprawdź czy dane orderbook są kompletne
    orderbook_cols = [col for col in df_sorted.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    missing_data = df_sorted[orderbook_cols].isnull().sum()
    
    # KROK 6: Sprawdź czy kolumny fill_method są obecne (jeśli były wypełniane luki)
    has_fill_method = 'fill_method' in df_sorted.columns
    
    # Przygotuj raport
    report = {
        'symbol': symbol,
        'total_rows': len(df_sorted),
        'date_range': {
            'start': df_sorted['timestamp'].min(),
            'end': df_sorted['timestamp'].max(),
            'duration_days': (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days
        },
        'continuity': {
            'gaps_found': len(gaps),
            'missing_minutes': len(missing_minutes),
            'extra_minutes': len(extra_minutes),
            'duplicates': len(duplicates),
            'minutes_with_one_snapshot': len(minutes_with_one_snapshot),
            'minutes_with_more_than_two': len(minutes_with_more_than_two),
            'is_continuous': len(gaps) == 0 and len(missing_minutes) == 0
        },
        'data_quality': {
            'missing_orderbook_data': missing_data.to_dict(),
            'has_fill_method': has_fill_method,
            'fill_method_stats': df_sorted['fill_method'].value_counts().to_dict() if has_fill_method else {}
        },
        'details': {
            'gaps': gaps[['timestamp', 'time_diff']].to_dict('records') if len(gaps) > 0 else [],
            'missing_minutes_list': list(missing_minutes)[:10] if len(missing_minutes) > 0 else [],  # Pierwsze 10
            'duplicate_timestamps': duplicates['timestamp'].unique().tolist() if len(duplicates) > 0 else []
        }
    }
    
    return report

def print_continuity_report(report: Dict, logger):
    """Wyświetla raport ciągłości danych"""
    symbol = report['symbol']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RAPORT CIĄGŁOŚCI DANYCH: {symbol}")
    logger.info(f"{'='*60}")
    
    # Podstawowe informacje
    logger.info(f"📊 PODSTAWOWE INFORMACJE:")
    logger.info(f"  Wierszy: {report['total_rows']:,}")
    logger.info(f"  Zakres: {report['date_range']['start']} do {report['date_range']['end']}")
    logger.info(f"  Czas trwania: {report['date_range']['duration_days']} dni")
    
    # Ciągłość
    continuity = report['continuity']
    logger.info(f"\n🔍 CIĄGŁOŚĆ DANYCH:")
    
    if continuity['is_continuous']:
        logger.info(f"  ✅ DANE SĄ CIĄGŁE - brak luk!")
    else:
        logger.info(f"  ❌ ZNALEZIONO PROBLEMY Z CIĄGŁOŚCIĄ:")
        
        if continuity['gaps_found'] > 0:
            logger.info(f"    - Luki czasowe: {continuity['gaps_found']}")
            
        if continuity['missing_minutes'] > 0:
            logger.info(f"    - Brakujące minuty: {continuity['missing_minutes']}")
            
        if continuity['extra_minutes'] > 0:
            logger.info(f"    - Dodatkowe minuty: {continuity['extra_minutes']}")
            
        if continuity['duplicates'] > 0:
            logger.info(f"    - Duplikaty: {continuity['duplicates']}")
    
    # Jakość danych
    data_quality = report['data_quality']
    logger.info(f"\n📈 JAKOŚĆ DANYCH:")
    
    if data_quality['has_fill_method']:
        logger.info(f"  Metody wypełniania luk:")
        for method, count in data_quality['fill_method_stats'].items():
            logger.info(f"    - {method}: {count:,} wierszy")
    else:
        logger.info(f"  Brak informacji o metodach wypełniania")
    
    # Sprawdź brakujące dane orderbook
    missing_data = data_quality['missing_orderbook_data']
    if any(count > 0 for count in missing_data.values()):
        logger.info(f"  ⚠️ Brakujące dane orderbook:")
        for col, count in missing_data.items():
            if count > 0:
                logger.info(f"    - {col}: {count:,} brakujących wartości")
    else:
        logger.info(f"  ✅ Wszystkie dane orderbook są kompletne")
    
    # Szczegóły problemów
    details = report['details']
    if details['gaps']:
        logger.info(f"\n⚠️ SZCZEGÓŁY LUK:")
        for i, gap in enumerate(details['gaps'][:5]):  # Pierwsze 5 luk
            logger.info(f"  {i+1}. {gap['timestamp']} - luka {gap['time_diff']}")
        if len(details['gaps']) > 5:
            logger.info(f"  ... i {len(details['gaps']) - 5} więcej")
    
    if details['missing_minutes_list']:
        logger.info(f"\n⚠️ PRZYKŁADY BRAKUJĄCYCH MINUT:")
        for minute in details['missing_minutes_list']:
            logger.info(f"  - {minute}")
    
    # Podsumowanie
    logger.info(f"\n🎯 PODSUMOWANIE:")
    if continuity['is_continuous']:
        logger.info(f"  ✅ {symbol}: DANE SĄ CIĄGŁE I GOTOWE DO UŻYCIA")
    else:
        logger.info(f"  ❌ {symbol}: WYMAGA POPRAWEK - znaleziono problemy z ciągłością")

def check_orderbook_continuity_for_symbol(symbol: str, logger) -> Optional[Dict]:
    """Sprawdza ciągłość danych orderbook dla jednej pary"""
    logger.info(f"Rozpoczynam sprawdzanie ciągłości dla {symbol}")
    
    # KROK 1: Wczytaj dane
    df = load_orderbook_data(symbol, logger)
    if df is None:
        return None
    
    # KROK 2: Sprawdź ciągłość
    report = check_continuity(df, symbol, logger)
    
    # KROK 3: Wyświetl raport
    print_continuity_report(report, logger)
    
    return report

def check_all_pairs_continuity():
    """Główna funkcja sprawdzania ciągłości dla wszystkich par"""
    logger = setup_logging()
    
    logger.info("Rozpoczynam sprawdzanie ciągłości dla wszystkich par")
    logger.info(f"Pary: {', '.join(PAIRS)}")
    
    start_time = datetime.now()
    successful_pairs = []
    failed_pairs = []
    all_reports = []
    
    for i, symbol in enumerate(PAIRS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sprawdzam parę {i}/{len(PAIRS)}: {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            report = check_orderbook_continuity_for_symbol(symbol, logger)
            if report is not None:
                successful_pairs.append(symbol)
                all_reports.append(report)
                
                if report['continuity']['is_continuous']:
                    logger.info(f"[OK] {symbol} - dane są ciągłe")
                else:
                    logger.warning(f"[WARN] {symbol} - znaleziono problemy z ciągłością")
            else:
                failed_pairs.append(symbol)
                logger.error(f"[ERROR] {symbol} - błąd sprawdzania")
        except Exception as e:
            failed_pairs.append(symbol)
            logger.error(f"[ERROR] {symbol} - błąd krytyczny: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Podsumowanie
    logger.info(f"\n{'='*60}")
    logger.info(f"SPRAWDZANIE CIĄGŁOŚCI ZAKOŃCZONE!")
    logger.info(f"{'='*60}")
    
    continuous_pairs = [r['symbol'] for r in all_reports if r['continuity']['is_continuous']]
    problematic_pairs = [r['symbol'] for r in all_reports if not r['continuity']['is_continuous']]
    
    logger.info(f"Sprawdzono: {len(successful_pairs)}/{len(PAIRS)} par")
    logger.info(f"Czas: {duration}")
    logger.info(f"✅ Ciągłe pary: {len(continuous_pairs)}")
    logger.info(f"❌ Problematyczne pary: {len(problematic_pairs)}")
    
    if continuous_pairs:
        logger.info(f"Pary z ciągłymi danymi: {', '.join(continuous_pairs)}")
    
    if problematic_pairs:
        logger.warning(f"Pary z problemami: {', '.join(problematic_pairs)}")
    
    if failed_pairs:
        logger.error(f"Nieudane pary: {', '.join(failed_pairs)}")
    
    # Zapisz metadane
    metadata = {
        'check_date': datetime.now().isoformat(),
        'pairs': PAIRS,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'continuous_pairs': continuous_pairs,
        'problematic_pairs': problematic_pairs,
        'duration_seconds': duration.total_seconds(),
        'success_rate': f"{len(successful_pairs)/len(PAIRS)*100:.1f}%",
        'continuity_rate': f"{len(continuous_pairs)/len(successful_pairs)*100:.1f}%" if successful_pairs else "0%"
    }
    
    metadata_file = Path("continuity_check_metadata.json")
    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadane zapisane: {metadata_file}")
    except Exception as e:
        logger.error(f"Błąd zapisywania metadanych: {e}")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Sprawdza ciągłość danych orderbook')
    parser.add_argument('--symbol', help='Sprawdź tylko jedną parę (opcjonalnie)')
    
    args = parser.parse_args()
    
    if args.symbol:
        # Sprawdź tylko jedną parę
        logger = setup_logging()
        report = check_orderbook_continuity_for_symbol(args.symbol, logger)
        if report is not None:
            if report['continuity']['is_continuous']:
                print(f"\n[OK] Para {args.symbol} ma ciągłe dane!")
            else:
                print(f"\n[WARN] Para {args.symbol} ma problemy z ciągłością!")
        else:
            print(f"\n[ERROR] Błąd podczas sprawdzania {args.symbol}!")
    else:
        # Sprawdź wszystkie pary
        check_all_pairs_continuity()

if __name__ == "__main__":
    main() 