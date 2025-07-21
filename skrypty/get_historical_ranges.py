import requests
import time
from datetime import datetime
import sys
import csv
import os

# ==============================================================================
# KONFIGURACJA
# ==============================================================================
# Oficjalny endpoint API dla Binance USD-S Futures
FUTURES_API_BASE_URL = "https://fapi.binance.com"
# Waluta, względem której kwotowane są pary (np. USDT, BUSD)
QUOTE_ASSET = "USDT"
# Mała pauza między zapytaniami, aby nie przeciążać API (w sekundach)
REQUEST_DELAY = 0.2
# Nazwa pliku wyjściowego, który zostanie stworzony w tym samym folderze co skrypt
OUTPUT_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_ranges.csv")
# ==============================================================================

def get_all_futures_usdt_pairs():
    """Pobiera listę wszystkich aktywnych par futures kwotowanych w USDT."""
    print("1. Pobieranie listy wszystkich dostępnych par z rynku futures...")
    url = f"{FUTURES_API_BASE_URL}/fapi/v1/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Rzuć wyjątkiem dla złych statusów (4xx, 5xx)
        data = response.json()
        
        symbols = [
            s['symbol'] for s in data['symbols'] 
            if s['quoteAsset'] == QUOTE_ASSET and s['status'] == 'TRADING'
        ]
        
        print(f"   -> Znaleziono {len(symbols)} aktywnych par kwotowanych w {QUOTE_ASSET}.")
        return sorted(symbols)

    except requests.exceptions.RequestException as e:
        print(f"   -> BŁĄD: Nie można połączyć się z Binance API. {e}")
        return []

def get_first_kline_date(symbol: str) -> str:
    """Znajduje datę pierwszej dostępnej świeczki dziennej dla danej pary."""
    url = f"{FUTURES_API_BASE_URL}/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': '1d',  # Interwał dzienny jest wystarczający do znalezienia daty startowej
        'startTime': 0,    # Od samego początku
        'limit': 1         # Potrzebujemy tylko pierwszej świeczki
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            # Pierwszy element to timestamp otwarcia w milisekundach
            first_timestamp_ms = data[0][0]
            # Konwertuj na czytelną datę
            first_date = datetime.fromtimestamp(first_timestamp_ms / 1000).strftime('%Y-%m-%d')
            return first_date
        else:
            return "Brak danych"
            
    except requests.exceptions.RequestException:
        # Czasami API może zwrócić błąd dla niektórych par, np. tych świeżo wycofanych
        return "Błąd zapytania"

def save_results_to_csv(results: list, filename: str):
    """Zapisuje wyniki do pliku CSV."""
    print(f"\nZapisywanie wyników do pliku CSV: {os.path.basename(filename)}")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Zapisz nagłówek
            writer.writerow(['pair', 'start_date'])
            # Zapisz dane
            writer.writerows(results)
        print(f"   -> Pomyślnie zapisano {len(results)} wierszy do pliku.")
    except IOError as e:
        print(f"   -> BŁĄD: Nie można zapisać pliku. {e}")

def main():
    """Główna funkcja skryptu."""
    print("=" * 60)
    print("🚀 Skrypt sprawdzający zakres historyczny par na Binance Futures 🚀")
    print("=" * 60)

    try:
        pairs = get_all_futures_usdt_pairs()
        
        if not pairs:
            print("\nZakończono pracę z powodu braku par do sprawdzenia.")
            return

        print("\n2. Sprawdzanie zakresu danych dla każdej pary (może to potrwać kilka minut)...")
        
        results = []
        
        for i, pair in enumerate(pairs):
            # Wyświetl postęp, nadpisując poprzednią linię
            progress_msg = f"   -> Sprawdzam: {pair} ({i + 1}/{len(pairs)})"
            # Użyj sys.stdout.write i .flush() dla lepszej kompatybilności
            sys.stdout.write(f"{progress_msg}\r")
            sys.stdout.flush()
            
            start_date = get_first_kline_date(pair)
            results.append((pair, start_date))
            
            time.sleep(REQUEST_DELAY) # Bądź miły dla API

        # Wyczyść ostatnią linię postępu
        print(" " * (len(progress_msg) + 5), end='\r')
        
        print("\n3. Sortowanie wyników...")
        def sort_key(item):
            date_str = item[1]
            try:
                # Konwertuj string na obiekt daty do poprawnego sortowania
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                # Elementy z błędami (np. "Brak danych") umieść na końcu listy
                return datetime.max
        
        results.sort(key=sort_key)

        print("\n4. Zakończono sprawdzanie. Oto wyniki:")
        print("-" * 40)

        # Znajdź najdłuższą nazwę pary dla ładnego formatowania
        max_len = max(len(p) for p, _ in results) if results else 0

        for pair, date in results:
            # Wyrównaj tekst dla lepszej czytelności
            print(f"{pair:<{max_len}} - Dostępne od: {date}")
            
        print("-" * 40)
        
        # Zapisz wyniki do pliku CSV
        save_results_to_csv(results, OUTPUT_FILENAME)
        
        print("✅ Gotowe!")

    except KeyboardInterrupt:
        print("\n\n❌ Przerwano pracę skryptu.")
        sys.exit(0)


if __name__ == "__main__":
    main() 