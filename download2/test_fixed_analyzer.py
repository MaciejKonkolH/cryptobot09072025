#!/usr/bin/env python3
"""
Test poprawionej logiki analizatora par futures
"""

import ccxt
from datetime import datetime, timedelta

def test_date_range():
    exchange = ccxt.binanceusdm({
        'timeout': 30000,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })
    
    test_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    for symbol in test_pairs:
        print(f"\n{'='*50}")
        print(f"Testuję {symbol}")
        print(f"{'='*50}")
        
        try:
            # Pobierz najnowsze dane
            latest_ohlcv = exchange.fetch_ohlcv(
                symbol, 
                '1m', 
                limit=1000
            )
            
            if not latest_ohlcv:
                print(f"Brak danych dla {symbol}")
                continue
            
            # Najnowszy timestamp
            latest_timestamp = latest_ohlcv[-1][0]
            latest_date = datetime.fromtimestamp(latest_timestamp / 1000)
            
            print(f"Najnowsze dane: {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
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
                    print(f"  Sprawdzam od {test_date.strftime('%Y-%m-%d')}...")
                    
                    historical_ohlcv = exchange.fetch_ohlcv(
                        symbol,
                        '1m',
                        since=test_timestamp,
                        limit=1000
                    )
                    
                    if historical_ohlcv and len(historical_ohlcv) > 0:
                        actual_oldest_timestamp = historical_ohlcv[0][0]
                        actual_oldest_date = datetime.fromtimestamp(actual_oldest_timestamp / 1000)
                        
                        print(f"    Pobrano {len(historical_ohlcv)} świec od {actual_oldest_date.strftime('%Y-%m-%d')}")
                        
                        # Sprawdź czy to rzeczywiście starsze dane
                        if actual_oldest_date < oldest_date:
                            oldest_date = actual_oldest_date
                            print(f"    ✅ Znaleziono starsze dane: {actual_oldest_date.strftime('%Y-%m-%d')}")
                        else:
                            print(f"    ❌ Nie znaleziono starszych danych")
                            break
                    else:
                        print(f"    ❌ Brak danych dla tego okresu")
                        break
                        
                except Exception as e:
                    print(f"    ❌ Błąd: {e}")
                    break
            
            # Oblicz długość zakresu w dniach
            date_range_days = (latest_date - oldest_date).days
            
            print(f"\nWYNIK: {symbol}: {oldest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')} ({date_range_days} dni)")
            
        except Exception as e:
            print(f"Błąd dla {symbol}: {e}")

if __name__ == "__main__":
    test_date_range() 