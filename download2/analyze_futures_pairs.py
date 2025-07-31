#!/usr/bin/env python3
"""
Analizator par futures na Binance
Sprawdza wszystkie dostępne pary i ich zakresy dat
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import time
from pathlib import Path

class BinanceFuturesAnalyzer:
    """Analizator par futures na Binance"""
    
    def __init__(self):
        self.exchange = ccxt.binanceusdm({
            'timeout': 30000,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        
        self.pairs_data = []
        
    def get_all_futures_pairs(self) -> List[str]:
        """Pobiera listę wszystkich par futures"""
        try:
            # Lista popularnych par futures na Binance
            # To są pary które mają najdłuższą historię i największy wolumen
            popular_pairs = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
                "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
                "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT",
                "FILUSDT", "NEARUSDT", "ALGOUSDT", "ICPUSDT", "VETUSDT",
                "FLOWUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT",
                "CHZUSDT", "HOTUSDT", "ENJUSDT", "BATUSDT", "ZILUSDT",
                "ONEUSDT", "IOTAUSDT", "XTZUSDT", "NEOUSDT", "QTUMUSDT",
                "ZECUSDT", "DASHUSDT", "WAVESUSDT", "HBARUSDT", "THETAUSDT",
                "EOSUSDT", "TRXUSDT", "XLMUSDT", "BCHUSDT", "XMRUSDT",
                "AAVEUSDT", "SUSHIUSDT", "COMPUSDT", "MKRUSDT", "YFIUSDT",
                "SNXUSDT", "CRVUSDT", "BALUSDT", "RENUSDT", "ZRXUSDT",
                "KNCUSDT", "BANDUSDT", "OCEANUSDT", "ALPHAUSDT", "RSRUSDT",
                "STORJUSDT", "ANKRUSDT", "CTSIUSDT", "SKLUSDT", "GRTUSDT",
                "LRCUSDT", "OMGUSDT", "ZENUSDT", "IOTXUSDT", "RVNUSDT",
                "COTIUSDT", "CHRUSDT", "DENTUSDT", "HIVEUSDT", "STMXUSDT",
                "DUSKUSDT", "WRXUSDT", "BTSUSDT", "FTMUSDT", "ROSEUSDT",
                "CELOUSDT", "OGNUSDT", "NKNUSDT", "DGBUSDT", "SCUSDT",
                "ICXUSDT", "ONTUSDT", "ZRXUSDT", "IOSTUSDT", "NULSUSDT",
                "WANUSDT", "WTCUSDT", "POAUSDT", "VITEUSDT", "FETUSDT",
                "CELRUSDT", "CTXCUSDT", "ARPAUSDT", "ARDRUSDT", "ARKUSDT",
                "REPUSDT", "RLCUSDT", "PIVXUSDT", "NEBLUSDT", "VIBUSDT",
                "MITHUSDT", "BCNUSDT", "XVGUSDT", "SYSUSDT", "STEEMUSDT",
                "STRATUSDT", "WAVESUSDT", "WTCUSDT", "POAUSDT", "VITEUSDT"
            ]
            
            # Usuń duplikaty i posortuj
            unique_pairs = list(set(popular_pairs))
            unique_pairs.sort()
            
            print(f"Przygotowano {len(unique_pairs)} popularnych par futures do analizy")
            return unique_pairs
            
        except Exception as e:
            print(f"Błąd pobierania par: {e}")
            return []
    
    def get_date_range_for_pair(self, symbol: str) -> Tuple[datetime, datetime, int]:
        """Sprawdza zakres dat dla jednej pary"""
        try:
            print(f"Sprawdzam {symbol}...")
            
            # Pobierz najnowsze dane
            latest_ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                '1m', 
                limit=1000
            )
            
            if not latest_ohlcv:
                return None, None, 0
            
            # Najnowszy timestamp
            latest_timestamp = latest_ohlcv[-1][0]
            latest_date = datetime.fromtimestamp(latest_timestamp / 1000)
            
            # Sprawdź starsze dane - używaj dłuższych okresów
            test_dates = [
                datetime.now() - timedelta(days=365),   # 1 rok
                datetime.now() - timedelta(days=730),   # 2 lata
                datetime.now() - timedelta(days=1095),  # 3 lata
                datetime.now() - timedelta(days=1460),  # 4 lata
                datetime.now() - timedelta(days=1825),  # 5 lat
                datetime.now() - timedelta(days=2190),  # 6 lat
                datetime.now() - timedelta(days=2555),  # 7 lat
                datetime.now() - timedelta(days=2920),  # 8 lat
                datetime.now() - timedelta(days=3285),  # 9 lat
                datetime.now() - timedelta(days=3650),  # 10 lat
            ]
            
            oldest_date = latest_date
            
            for test_date in test_dates:
                try:
                    test_timestamp = int(test_date.timestamp() * 1000)
                    historical_ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        '1m',
                        since=test_timestamp,
                        limit=1000
                    )
                    
                    if historical_ohlcv and len(historical_ohlcv) > 0:
                        actual_oldest_timestamp = historical_ohlcv[0][0]
                        actual_oldest_date = datetime.fromtimestamp(actual_oldest_timestamp / 1000)
                        
                        # Sprawdź czy to rzeczywiście starsze dane
                        if actual_oldest_date < oldest_date:
                            oldest_date = actual_oldest_date
                            print(f"    Znaleziono starsze dane: {actual_oldest_date.strftime('%Y-%m-%d')}")
                        
                except Exception as e:
                    # Jeśli błąd, spróbuj następną datę
                    continue
            
            # Oblicz długość zakresu w dniach
            date_range_days = (latest_date - oldest_date).days
            
            print(f"  {symbol}: {oldest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')} ({date_range_days} dni)")
            
            return oldest_date, latest_date, date_range_days
            
        except Exception as e:
            print(f"  Błąd sprawdzania {symbol}: {e}")
            return None, None, 0
    
    def analyze_all_pairs(self):
        """Analizuje wszystkie pary futures"""
        print("Rozpoczynam analizę par futures na Binance...")
        
        # Pobierz wszystkie pary
        pairs = self.get_all_futures_pairs()
        
        if not pairs:
            print("Nie udało się pobrać listy par")
            return
        
        print(f"\nAnalizuję {len(pairs)} par...")
        
        # Analizuj każdą parę
        for i, symbol in enumerate(pairs, 1):
            oldest_date, latest_date, range_days = self.get_date_range_for_pair(symbol)
            
            if oldest_date and latest_date:
                self.pairs_data.append({
                    'symbol': symbol,
                    'oldest_date': oldest_date,
                    'latest_date': latest_date,
                    'range_days': range_days,
                    'oldest_date_str': oldest_date.strftime('%Y-%m-%d'),
                    'latest_date_str': latest_date.strftime('%Y-%m-%d')
                })
            
            # Rate limiting
            time.sleep(0.1)
            
            # Progress
            if i % 10 == 0:
                print(f"Postęp: {i}/{len(pairs)} par")
        
        # Sortuj według długości zakresu (malejąco)
        self.pairs_data.sort(key=lambda x: x['range_days'], reverse=True)
        
        print(f"\nAnaliza zakończona! Przeanalizowano {len(self.pairs_data)} par.")
    
    def save_results(self):
        """Zapisuje wyniki do plików"""
        if not self.pairs_data:
            print("Brak danych do zapisania")
            return
        
        # Zapisz jako JSON
        json_file = Path("futures_pairs_analysis.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.pairs_data, f, indent=2, default=str)
        
        # Zapisz jako CSV
        df = pd.DataFrame(self.pairs_data)
        csv_file = Path("futures_pairs_analysis.csv")
        df.to_csv(csv_file, index=False)
        
        # Zapisz jako tekst
        txt_file = Path("futures_pairs_analysis.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("ANALIZA PAR FUTURES NA BINANCE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Liczba par: {len(self.pairs_data)}\n\n")
            
            f.write("PARY POSORTOWANE WEDŁUG DŁUGOŚCI HISTORII:\n")
            f.write("-" * 50 + "\n")
            
            for i, pair in enumerate(self.pairs_data, 1):
                f.write(f"{i:3d}. {pair['symbol']:<12} | {pair['oldest_date_str']} - {pair['latest_date_str']} | {pair['range_days']:5d} dni\n")
        
        print(f"\nWyniki zapisane:")
        print(f"  JSON: {json_file}")
        print(f"  CSV:  {csv_file}")
        print(f"  TXT:  {txt_file}")
    
    def print_summary(self):
        """Wyświetla podsumowanie"""
        if not self.pairs_data:
            return
        
        print(f"\n{'='*60}")
        print(f"PODSUMOWANIE ANALIZY")
        print(f"{'='*60}")
        print(f"Liczba par: {len(self.pairs_data)}")
        print(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Statystyki
        ranges = [p['range_days'] for p in self.pairs_data]
        print(f"\nStatystyki zakresów dat:")
        print(f"  Najdłuższy: {max(ranges)} dni")
        print(f"  Najkrótszy: {min(ranges)} dni")
        print(f"  Średni: {sum(ranges)/len(ranges):.1f} dni")
        
        # Top 10 najdłuższych
        print(f"\nTOP 10 PAR Z NAJDŁUŻSZĄ HISTORIĄ:")
        print(f"{'='*60}")
        for i, pair in enumerate(self.pairs_data[:10], 1):
            print(f"{i:2d}. {pair['symbol']:<12} | {pair['oldest_date_str']} - {pair['latest_date_str']} | {pair['range_days']:5d} dni")
        
        # Pary z historią > 1000 dni
        long_history = [p for p in self.pairs_data if p['range_days'] > 1000]
        print(f"\nPary z historią > 1000 dni: {len(long_history)}")
        
        # Pary z historią > 500 dni
        medium_history = [p for p in self.pairs_data if p['range_days'] > 500]
        print(f"Pary z historią > 500 dni: {len(medium_history)}")
        
        # Pary z historią < 100 dni
        short_history = [p for p in self.pairs_data if p['range_days'] < 100]
        print(f"Pary z historią < 100 dni: {len(short_history)}")

def main():
    """Główna funkcja"""
    try:
        analyzer = BinanceFuturesAnalyzer()
        analyzer.analyze_all_pairs()
        analyzer.save_results()
        analyzer.print_summary()
        
    except KeyboardInterrupt:
        print("\nAnaliza przerwana przez użytkownika")
    except Exception as e:
        print(f"Błąd krytyczny: {e}")

if __name__ == "__main__":
    main() 