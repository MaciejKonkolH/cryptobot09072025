#!/usr/bin/env python3
"""
Skrypt równoległego łączenia danych orderbook w format feather
Przetwarza wiele par jednocześnie w osobnych procesach
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging
import sys
import subprocess
import time
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import konfiguracji
from config import PAIRS, FILE_CONFIG, LOGGING_CONFIG

def setup_logging():
    """Konfiguruje system logowania"""
    log_file = Path("merge_orderbook_parallel.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def process_single_pair(symbol: str) -> Dict:
    """Przetwarza jedną parę w osobnym procesie"""
    start_time = datetime.now()
    
    try:
        # Uruchom skrypt dla jednej pary
        cmd = [sys.executable, "merge_orderbook_to_feather.py", "--symbol", symbol]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2h timeout
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if result.returncode == 0:
            return {
                'symbol': symbol,
                'status': 'success',
                'output': result.stdout,
                'error': None,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            }
        else:
            return {
                'symbol': symbol,
                'status': 'error',
                'output': result.stdout,
                'error': result.stderr,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            }
    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = end_time - start_time
        return {
            'symbol': symbol,
            'status': 'timeout',
            'output': None,
            'error': f"Przekroczono limit czasu (2h) dla {symbol}",
            'duration': duration,
            'start_time': start_time,
            'end_time': end_time
        }
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        return {
            'symbol': symbol,
            'status': 'exception',
            'output': None,
            'error': str(e),
            'duration': duration,
            'start_time': start_time,
            'end_time': end_time
        }

def merge_orderbook_parallel(max_workers: int = None):
    """Główna funkcja równoległego łączenia order book"""
    logger = setup_logging()
    
    if max_workers is None:
        max_workers = min(4, len(PAIRS), multiprocessing.cpu_count())
    
    logger.info("Rozpoczynam równoległe łączenie order book")
    logger.info(f"Pary: {', '.join(PAIRS)}")
    logger.info(f"Liczba procesów równoległych: {max_workers}")
    
    start_time = datetime.now()
    successful_pairs = []
    failed_pairs = []
    
    # KROK 1: Uruchom przetwarzanie równoległe
    logger.info(f"Uruchamiam {max_workers} procesów równoległych...")
    logger.info(f"Rozpoczynam przetwarzanie {len(PAIRS)} par...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submituj wszystkie zadania
        future_to_symbol = {executor.submit(process_single_pair, symbol): symbol for symbol in PAIRS}
        logger.info(f"Wszystkie zadania zostały uruchomione. Oczekuję na wyniki...")
        
        # Zbierz wyniki z lepszym logowaniem
        completed = 0
        pair_durations = []
        start_waiting = datetime.now()
        last_progress_time = start_waiting
        
        # Dodaj timer dla sprawdzania postępu
        logger.info(f"⏰ Pierwsze wyniki powinny pojawić się w ciągu 5-15 minut...")
        logger.info(f"📊 Logowanie postępu co 30 sekund...")
        
        # Prostsze rozwiązanie - sprawdzaj co 30 sekund
        last_log_time = datetime.now()
        
        for future in as_completed(future_to_symbol):
            # Loguj postęp co 30 sekund
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() >= 30:
                elapsed = current_time - start_waiting
                logger.info(f"⏳ Czekam na wyniki... (upłynęło: {elapsed:.0f}, ukończono: {completed}/{len(PAIRS)})")
                last_log_time = current_time
            
            symbol = future_to_symbol[future]
            completed += 1
            
            # Dodaj informację o rozpoczęciu przetwarzania wyniku
            waiting_time = datetime.now() - start_waiting
            logger.info(f"🎯 Otrzymano wynik dla {symbol} (po {waiting_time:.0f} oczekiwania)")
            
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    successful_pairs.append(symbol)
                    duration = result['duration']
                    pair_durations.append(duration.total_seconds())
                    
                    # Oblicz średni czas i szacowany czas pozostały
                    avg_duration = sum(pair_durations) / len(pair_durations)
                    remaining_pairs = len(PAIRS) - completed
                    estimated_remaining = timedelta(seconds=avg_duration * remaining_pairs / max_workers)
                    
                    logger.info(f"[OK] {symbol} - pomyślnie przetworzono ({completed}/{len(PAIRS)})")
                    logger.info(f"  Czas: {duration}")
                    logger.info(f"  Średni czas na parę: {timedelta(seconds=avg_duration):.0f}")
                    logger.info(f"  Szacowany czas pozostały: {estimated_remaining:.0f}")
                    
                else:
                    failed_pairs.append(symbol)
                    duration = result['duration']
                    logger.error(f"[ERROR] {symbol} - {result['status']}: {result['error']}")
                    logger.error(f"  Czas: {duration}")
                
                # Progress update z szacowaniem
                progress = (completed / len(PAIRS)) * 100
                elapsed_time = datetime.now() - start_time
                
                if completed > 0:
                    # Szacuj całkowity czas na podstawie dotychczasowego postępu
                    estimated_total = elapsed_time * len(PAIRS) / completed
                    estimated_remaining = estimated_total - elapsed_time
                    
                    logger.info(f"📊 POSTĘP: {completed}/{len(PAIRS)} par ({progress:.1f}%)")
                    logger.info(f"  Czas upłyniony: {elapsed_time:.0f}")
                    logger.info(f"  Szacowany czas całkowity: {estimated_total:.0f}")
                    logger.info(f"  Szacowany czas pozostały: {estimated_remaining:.0f}")
                    logger.info(f"  Prędkość: {completed/elapsed_time.total_seconds()*3600:.1f} par/godzinę")
                    logger.info("-" * 50)
                
            except Exception as e:
                failed_pairs.append(symbol)
                logger.error(f"[ERROR] {symbol} - błąd krytyczny: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # KROK 2: Podsumowanie
    logger.info(f"\n{'='*60}")
    logger.info(f"RÓWNOLEGŁE ŁĄCZENIE ORDERBOOK ZAKOŃCZONE!")
    logger.info(f"{'='*60}")
    logger.info(f"Udało się: {len(successful_pairs)}/{len(PAIRS)} par")
    logger.info(f"Czas całkowity: {duration}")
    
    if pair_durations:
        avg_duration = sum(pair_durations) / len(pair_durations)
        min_duration = min(pair_durations)
        max_duration = max(pair_durations)
        
        logger.info(f"Statystyki czasów:")
        logger.info(f"  Średni czas na parę: {timedelta(seconds=avg_duration):.0f}")
        logger.info(f"  Najszybsza para: {timedelta(seconds=min_duration):.0f}")
        logger.info(f"  Najwolniejsza para: {timedelta(seconds=max_duration):.0f}")
        logger.info(f"  Prędkość: {len(successful_pairs)/duration.total_seconds()*3600:.1f} par/godzinę")
    
    logger.info(f"Liczba procesów równoległych: {max_workers}")
    
    if successful_pairs:
        logger.info(f"Pomyślnie przetworzone pary: {', '.join(successful_pairs)}")
    
    if failed_pairs:
        logger.warning(f"Nieudane pary: {', '.join(failed_pairs)}")
    
    # KROK 3: Zapisz metadane
    metadata = {
        'merge_date': datetime.now().isoformat(),
        'pairs': PAIRS,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'duration_seconds': duration.total_seconds(),
        'success_rate': f"{len(successful_pairs)/len(PAIRS)*100:.1f}%",
        'max_workers': max_workers,
        'parallel_processing': True
    }
    
    metadata_file = Path("merge_orderbook_parallel_metadata.json")
    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadane zapisane: {metadata_file}")
    except Exception as e:
        logger.error(f"Błąd zapisywania metadanych: {e}")
    
    return successful_pairs, failed_pairs

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Równoległe łączenie danych order book')
    parser.add_argument('--max-workers', type=int, help='Maksymalna liczba procesów równoległych')
    
    args = parser.parse_args()
    
    merge_orderbook_parallel(max_workers=args.max_workers)

if __name__ == "__main__":
    main() 