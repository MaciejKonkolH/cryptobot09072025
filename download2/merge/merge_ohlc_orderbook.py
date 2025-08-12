#!/usr/bin/env python3
"""
Skrypt do łączenia danych OHLC z orderbook
Dostosowany do modułu download2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Dodaj ścieżki do modułów
sys.path.append(str(Path(__file__).parent / '../orderbook'))
sys.path.append(str(Path(__file__).parent / '../OHLC'))

# Import konfiguracji
try:
    from orderbook.config import PAIRS, LOGGING_CONFIG
except ImportError:
    # Fallback - zdefiniuj podstawową konfigurację
    PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'BCHUSDT', 'LTCUSDT', 
        'LINKUSDT', 'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'XMRUSDT', 'DASHUSDT', 
        'ZECUSDT', 'XTZUSDT', 'ATOMUSDT', 'BATUSDT', 'IOTAUSDT', 'NEOUSDT', 
        'VETUSDT', 'ONTUSDT'
    ]
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }

def setup_logging():
    """Konfiguruje system logowania"""
    log_file = Path("merge_ohlc_orderbook.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_ohlc_data_for_symbol(symbol: str, logger) -> Optional[pd.DataFrame]:
    """Wczytuje dane OHLC dla jednej pary z pliku CSV"""
    ohlc_file = Path("../OHLC/ohlc_raw") / f"{symbol}_1m.csv"
    
    logger.info(f"📊 Wczytuję dane OHLC dla {symbol} z {ohlc_file}...")
    
    if not ohlc_file.exists():
        logger.error(f"❌ Plik {ohlc_file} nie istnieje!")
        return None
    
    try:
        # Wczytaj dane CSV
        df = pd.read_csv(ohlc_file)
        
        # Konwertuj timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sortuj po czasie
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Wczytano {len(df):,} wierszy OHLC dla {symbol}")
        logger.info(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Błąd wczytania {ohlc_file}: {e}")
        return None

def load_orderbook_data_for_symbol(symbol: str, logger) -> Optional[pd.DataFrame]:
    """Wczytuje dane orderbook dla jednej pary z pliku feather"""
    orderbook_file = Path("../orderbook/orderbook_completed") / f"orderbook_filled_{symbol}.feather"
    
    logger.info(f"📊 Wczytuję dane orderbook dla {symbol} z {orderbook_file}...")
    
    if not orderbook_file.exists():
        logger.error(f"❌ Plik {orderbook_file} nie istnieje!")
        return None
    
    try:
        # Wczytaj dane feather
        df = pd.read_feather(orderbook_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sortuj po czasie
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Wczytano {len(df):,} wierszy orderbook dla {symbol}")
        logger.info(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Błąd wczytania {orderbook_file}: {e}")
        return None

def align_timestamps(ohlc_df, orderbook_df, symbol: str, logger):
    """Wyrównuje timestampy między danymi OHLC i orderbook"""
    logger.info(f"🔧 Wyrównuję timestampy dla {symbol}...")
    
    # Znajdź wspólny zakres czasowy
    ohlc_start = ohlc_df['timestamp'].min()
    ohlc_end = ohlc_df['timestamp'].max()
    ob_start = orderbook_df['timestamp'].min()
    ob_end = orderbook_df['timestamp'].max()
    
    logger.info(f"📅 OHLC zakres: {ohlc_start} - {ohlc_end}")
    logger.info(f"📅 Orderbook zakres: {ob_start} - {ob_end}")
    
    # Znajdź wspólny zakres
    common_start = max(ohlc_start, ob_start)
    common_end = min(ohlc_end, ob_end)
    
    logger.info(f"📅 Wspólny zakres: {common_start} - {common_end}")
    
    # Filtruj dane do wspólnego zakresu
    ohlc_filtered = ohlc_df[
        (ohlc_df['timestamp'] >= common_start) & 
        (ohlc_df['timestamp'] <= common_end)
    ].copy()
    
    orderbook_filtered = orderbook_df[
        (orderbook_df['timestamp'] >= common_start) & 
        (orderbook_df['timestamp'] <= common_end)
    ].copy()
    
    logger.info(f"📊 OHLC po filtrowaniu: {len(ohlc_filtered):,} wierszy")
    logger.info(f"📊 Orderbook po filtrowaniu: {len(orderbook_filtered):,} wierszy")
    
    return ohlc_filtered, orderbook_filtered

def merge_data(ohlc_df, orderbook_df, symbol: str, logger):
    """Łączy dane OHLC z orderbook"""
    logger.info(f"🔗 Łączę dane OHLC z orderbook dla {symbol}...")
    
    # Usuń kolumny które mogą powodować konflikty
    columns_to_drop = ['time_diff']
    for col in columns_to_drop:
        if col in ohlc_df.columns:
            ohlc_df = ohlc_df.drop(columns=[col])
        if col in orderbook_df.columns:
            orderbook_df = orderbook_df.drop(columns=[col])
    
    # Ustaw timestamp jako indeks dla szybszego merge
    ohlc_df = ohlc_df.set_index('timestamp')
    orderbook_df = orderbook_df.set_index('timestamp')
    
    # Wykonaj left join (zachowaj wszystkie wiersze OHLC)
    merged_df = ohlc_df.join(orderbook_df, how='left')
    
    # Resetuj indeks
    merged_df = merged_df.reset_index()
    
    logger.info(f"✅ Połączono dane dla {symbol}: {len(merged_df):,} wierszy")
    
    # Sprawdź pokrycie - znajdź pierwszą kolumnę orderbook
    orderbook_columns = [col for col in merged_df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    if orderbook_columns:
        first_ob_col = orderbook_columns[0]
        missing_orderbook = merged_df[merged_df[first_ob_col].isna()]
        missing_percent = len(missing_orderbook) / len(merged_df) * 100
        logger.info(f"⚠️  Wiersze bez danych orderbook: {len(missing_orderbook):,} ({missing_percent:.2f}%)")
    else:
        logger.warning("⚠️  Nie znaleziono kolumn orderbook!")
    
    return merged_df

def analyze_merged_data(df, symbol: str, logger):
    """Analizuje połączone dane"""
    logger.info(f"📊 ANALIZA POŁĄCZONYCH DANYCH DLA {symbol}:")
    logger.info("-" * 60)
    
    # Podstawowe statystyki
    logger.info(f"📈 Łączna liczba wierszy: {len(df):,}")
    logger.info(f"⏰ Zakres czasowy: {df['timestamp'].min()} - {df['timestamp'].max()}")
    logger.info(f"📋 Liczba kolumn: {len(df.columns)}")
    
    # Sprawdź brakujące dane
    logger.info(f"🔍 ANALIZA BRAKUJĄCYCH DANYCH:")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        logger.info("Kolumny z brakującymi danymi:")
        for col, count in missing_data.head(10).items():
            percentage = count / len(df) * 100
            logger.info(f"  {col}: {count:,} ({percentage:.2f}%)")
    else:
        logger.info("✅ Brak brakujących danych!")
    
    # Sprawdź kolumny OHLC
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    logger.info(f"📈 KOLUMNY OHLC:")
    for col in ohlc_columns:
        if col in df.columns:
            logger.info(f"  {col}: {df[col].dtype}")
    
    # Sprawdź kolumny orderbook
    ob_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    logger.info(f"📊 KOLUMNY ORDERBOOK (pierwsze 10):")
    for col in ob_columns[:10]:
        logger.info(f"  {col}: {df[col].dtype}")
    
    if len(ob_columns) > 10:
        logger.info(f"  ... i {len(ob_columns) - 10} więcej kolumn orderbook")
    
    # Podsumowanie
    logger.info(f"📋 PODSUMOWANIE KOLUMN:")
    logger.info(f"  OHLC: {len(ohlc_columns)} kolumn")
    logger.info(f"  Orderbook: {len(ob_columns)} kolumn")
    logger.info(f"  Inne: {len(df.columns) - len(ohlc_columns) - len(ob_columns)} kolumn")

def save_merged_data(df, symbol: str, logger, output_dir: Path = Path("merged_data")):
    """Zapisuje połączone dane"""
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"merged_{symbol}.feather"
    
    logger.info(f"💾 Zapisuję połączone dane dla {symbol} do {output_file}...")
    
    # Usuń kolumny z oryginalnymi timestampami
    columns_to_drop = ['open_time', 'close_time']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 🧹 USUŃ METADANE PO WYPEŁNIENIU LUK
    metadata_columns = ['fill_method', 'gap_duration_minutes', 'price_change_percent']
    removed_count = 0
    for col in metadata_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_count += 1
            logger.info(f"  🧹 Usunięto metadane: {col}")
    
    if removed_count > 0:
        logger.info(f"  📊 Usunięto {removed_count} kolumn metadanych")
    
    # Zapisz do feather
    df.to_feather(output_file)
    
    # Sprawdź rozmiar pliku
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✅ Zapisano: {output_file}")
    logger.info(f"📁 Rozmiar pliku: {file_size:.2f} MB")
    logger.info(f"📋 Kolumn w pliku: {len(df.columns)}")
    
    return output_file

def process_single_symbol(symbol: str, logger, output_dir: Path):
    """Przetwarza jedną parę walut"""
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 PRZETWARZAM PARĘ: {symbol}")
    logger.info(f"{'='*80}")
    
    # Wczytaj dane
    ohlc_df = load_ohlc_data_for_symbol(symbol, logger)
    if ohlc_df is None:
        return None
    
    orderbook_df = load_orderbook_data_for_symbol(symbol, logger)
    if orderbook_df is None:
        return None
    
    # Wyrównaj timestampy
    ohlc_aligned, orderbook_aligned = align_timestamps(ohlc_df, orderbook_df, symbol, logger)
    
    # Połącz dane
    merged_df = merge_data(ohlc_aligned, orderbook_aligned, symbol, logger)
    
    # Przeanalizuj wyniki
    analyze_merged_data(merged_df, symbol, logger)
    
    # Zapisz dane
    output_file = save_merged_data(merged_df, symbol, logger, output_dir)
    
    logger.info(f"🎉 PRZETWARZANIE {symbol} ZAKOŃCZONE POMYŚLNIE!")
    logger.info(f"📁 Plik wynikowy: {output_file}")
    logger.info(f"📊 Wierszy: {len(merged_df):,}")
    logger.info(f"📋 Kolumn: {len(merged_df.columns)}")
    
    return output_file

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Łączy dane OHLC z orderbook dla wszystkich par')
    parser.add_argument('--symbol', help='Przetwórz tylko jedną parę (opcjonalnie)')
    parser.add_argument('--output-dir', default='merged_data', help='Katalog wyjściowy (domyślnie: merged_data)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("🚀 ROZPOCZYNAM ŁĄCZENIE DANYCH OHLC Z ORDERBOOK")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    successful_pairs = []
    failed_pairs = []
    
    if args.symbol:
        # Przetwórz tylko jedną parę
        if args.symbol in PAIRS:
            output_file = process_single_symbol(args.symbol, logger, output_dir)
            if output_file:
                successful_pairs.append(args.symbol)
            else:
                failed_pairs.append(args.symbol)
        else:
            logger.error(f"❌ Para {args.symbol} nie jest w konfiguracji!")
            return
    else:
        # Przetwórz wszystkie pary
        for i, symbol in enumerate(PAIRS, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"PARA {i}/{len(PAIRS)}: {symbol}")
            logger.info(f"{'='*80}")
            
            try:
                output_file = process_single_symbol(symbol, logger, output_dir)
                if output_file:
                    successful_pairs.append(symbol)
                else:
                    failed_pairs.append(symbol)
            except Exception as e:
                logger.error(f"❌ Błąd przetwarzania {symbol}: {e}")
                failed_pairs.append(symbol)
    
    # Podsumowanie
    logger.info(f"\n{'='*80}")
    logger.info(f"🎉 ŁĄCZENIE ZAKOŃCZONE!")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Udało się: {len(successful_pairs)}/{len(PAIRS)} par")
    
    if successful_pairs:
        logger.info(f"✅ Pomyślnie przetworzone pary: {', '.join(successful_pairs)}")
    
    if failed_pairs:
        logger.warning(f"❌ Nieudane pary: {', '.join(failed_pairs)}")
    
    logger.info(f"📁 Katalog wyjściowy: {output_dir.absolute()}")

if __name__ == "__main__":
    main() 