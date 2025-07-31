#!/usr/bin/env python3
"""
Analizator par orderbook na Binance
Sprawdza wszystkie dostępne pary i ich zakresy dat dla danych orderbook
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import time
from pathlib import Path

class BinanceOrderbookAnalyzer:
    """Analizator par orderbook na Binance"""
    
    def __init__(self):
        self.base_url = "https://data.binance.vision/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BinanceOrderbookAnalyzer/1.0'
        })
        
        self.pairs_data = []
        
    def get_all_orderbook_pairs(self) -> List[str]:
        """Pobiera listę wszystkich par orderbook"""
        try:
            # Lista popularnych par futures na Binance (te same co w OHLC)
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
            
            print(f"Przygotowano {len(unique_pairs)} popularnych par orderbook do analizy")
            return unique_pairs
            
        except Exception as e:
            print(f"Błąd pobierania par: {e}")
            return []
    
    def check_orderbook_available(self, symbol: str, date_str: str) -> bool:
        """Sprawdza czy dane orderbook są dostępne dla pary i daty"""
        url = f"{self.base_url}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
        
        try:
            response = self.session.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_date_range_for_pair(self, symbol: str) -> Tuple[datetime, datetime, int]:
        """Sprawdza zakres dat dla jednej pary orderbook"""
        try:
            print(f"Sprawdzam orderbook {symbol}...")
            
            # Sprawdź najnowsze dane - ostatnie 7 dni
            latest_date = None
            for i in range(7):
                test_date = datetime.now() - timedelta(days=i)
                date_str = test_date.strftime('%Y-%m-%d')
                
                if self.check_orderbook_available(symbol, date_str):
                    latest_date = test_date
                    print(f"    [OK] Znaleziono najnowsze dane: {date_str}")
                    break
            
            if latest_date is None:
                print(f"  [ERROR] {symbol}: Brak najnowszych danych orderbook")
                return None, None, 0
            
            # Sprawdź starsze dane - używaj dłuższych okresów
            test_dates = [
                datetime.now() - timedelta(days=30),   # 1 miesiąc
                datetime.now() - timedelta(days=90),   # 3 miesiące
                datetime.now() - timedelta(days=180),  # 6 miesięcy
                datetime.now() - timedelta(days=365),  # 1 rok
                datetime.now() - timedelta(days=730),  # 2 lata
                datetime.now() - timedelta(days=1095), # 3 lata
                datetime.now() - timedelta(days=1460), # 4 lata
                datetime.now() - timedelta(days=1825), # 5 lat
            ]
            
            oldest_date = latest_date
            
            for test_date in test_dates:
                try:
                    date_str = test_date.strftime('%Y-%m-%d')
                    
                    if self.check_orderbook_available(symbol, date_str):
                        oldest_date = test_date
                        print(f"    [OK] Znaleziono starsze dane: {date_str}")
                        # NIE PRZERYWAJ - sprawdź wszystkie daty aby znaleźć najstarszą
                        
                except Exception as e:
                    # Jeśli błąd, spróbuj następną datę
                    continue
            
            # Oblicz długość zakresu w dniach
            date_range_days = (latest_date - oldest_date).days
            
            print(f"  {symbol} orderbook: {oldest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')} ({date_range_days} dni)")
            
            return oldest_date, latest_date, date_range_days
            
        except Exception as e:
            print(f"  [ERROR] Błąd sprawdzania orderbook {symbol}: {e}")
            return None, None, 0
    
    def analyze_all_pairs(self):
        """Analizuje wszystkie pary orderbook"""
        print("Rozpoczynam analizę par orderbook na Binance...")
        
        # Pobierz wszystkie pary
        pairs = self.get_all_orderbook_pairs()
        
        if not pairs:
            print("Nie udało się pobrać listy par")
            return
        
        print(f"\nAnalizuję {len(pairs)} par orderbook...")
        
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
        
        print(f"\nAnaliza zakończona! Przeanalizowano {len(self.pairs_data)} par orderbook.")
    
    def save_results(self):
        """Zapisuje wyniki do plików"""
        if not self.pairs_data:
            print("Brak danych do zapisania")
            return
        
        # Zapisz jako JSON
        json_file = Path("orderbook_pairs_analysis.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.pairs_data, f, indent=2, default=str)
        
        # Zapisz jako CSV
        df = pd.DataFrame(self.pairs_data)
        csv_file = Path("orderbook_pairs_analysis.csv")
        df.to_csv(csv_file, index=False)
        
        # Zapisz jako tekst
        txt_file = Path("orderbook_pairs_analysis.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("ANALIZA PAR ORDERBOOK NA BINANCE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Liczba par: {len(self.pairs_data)}\n\n")
            
            f.write("PARY POSORTOWANE WEDŁUG DŁUGOŚCI HISTORII ORDERBOOK:\n")
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
        print(f"PODSUMOWANIE ANALIZY ORDERBOOK")
        print(f"{'='*60}")
        print(f"Liczba par: {len(self.pairs_data)}")
        print(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Statystyki
        ranges = [p['range_days'] for p in self.pairs_data]
        print(f"\nStatystyki zakresów dat orderbook:")
        print(f"  Najdłuższy: {max(ranges)} dni")
        print(f"  Najkrótszy: {min(ranges)} dni")
        print(f"  Średni: {sum(ranges)/len(ranges):.1f} dni")
        
        # Top 10 najdłuższych
        print(f"\nTOP 10 PAR Z NAJDŁUŻSZĄ HISTORIĄ ORDERBOOK:")
        print(f"{'='*60}")
        for i, pair in enumerate(self.pairs_data[:10], 1):
            print(f"{i:2d}. {pair['symbol']:<12} | {pair['oldest_date_str']} - {pair['latest_date_str']} | {pair['range_days']:5d} dni")
        
        # Pary z historią > 1000 dni
        long_history = [p for p in self.pairs_data if p['range_days'] > 1000]
        print(f"\nPary z historią orderbook > 1000 dni: {len(long_history)}")
        
        # Pary z historią > 500 dni
        medium_history = [p for p in self.pairs_data if p['range_days'] > 500]
        print(f"Pary z historią orderbook > 500 dni: {len(medium_history)}")
        
        # Pary z historią < 100 dni
        short_history = [p for p in self.pairs_data if p['range_days'] < 100]
        print(f"Pary z historią orderbook < 100 dni: {len(short_history)}")
        
        # Porównanie z OHLC
        print(f"\nUWAGA: Dane orderbook mogą mieć krótszą historię niż OHLC!")
        print(f"Orderbook to bardziej zaawansowane dane i mogą być dostępne od późniejszej daty.")

def main():
    """Główna funkcja"""
    try:
        analyzer = BinanceOrderbookAnalyzer()
        analyzer.analyze_all_pairs()
        analyzer.save_results()
        analyzer.print_summary()
        
    except KeyboardInterrupt:
        print("\nAnaliza orderbook przerwana przez użytkownika")
    except Exception as e:
        print(f"Błąd krytyczny: {e}")

if __name__ == "__main__":
    main() 