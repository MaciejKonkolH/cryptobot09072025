"""
Skrypt do pobierania 10 wierszy z określonej daty z pliku z cechami.
"""
import argparse
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Dodajemy ścieżkę do głównego katalogu
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import feature_calculator_ohlc_snapshot.config as config
except ImportError:
    import config

def load_features_file(file_path: str) -> pd.DataFrame:
    """Wczytuje plik z cechami."""
    if not os.path.exists(file_path):
        print(f"BŁĄD: Plik nie istnieje: {file_path}")
        return None
    
    try:
        print(f"Wczytywanie pliku: {file_path}")
        df = pd.read_feather(file_path)
        
        # Konwertuj timestamp na datetime jeśli nie jest indeksem
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name != 'timestamp':
            print("OSTRZEŻENIE: Nie znaleziono kolumny 'timestamp'")
        
        print(f"Wczytano {len(df):,} wierszy")
        print(f"Zakres czasowy: {df.index.min()} do {df.index.max()}")
        print(f"Liczba kolumn: {len(df.columns)}")
        
        return df
    except Exception as e:
        print(f"BŁĄD podczas wczytywania pliku: {e}")
        return None

def get_rows_for_date(df: pd.DataFrame, target_date: str, num_rows: int = 10) -> pd.DataFrame:
    """Pobiera określoną liczbę wierszy z danej daty."""
    try:
        # Konwertuj datę na datetime
        target_dt = pd.to_datetime(target_date)
        start_of_day = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        print(f"Szukam wierszy z daty: {target_date}")
        print(f"Zakres: {start_of_day} do {end_of_day}")
        
        # Filtruj wiersze z danej daty
        mask = (df.index >= start_of_day) & (df.index < end_of_day)
        daily_data = df[mask]
        
        print(f"Znaleziono {len(daily_data)} wierszy z tej daty")
        
        if len(daily_data) == 0:
            print("Brak danych z podanej daty!")
            return None
        
        # Pobierz pierwsze N wierszy
        if len(daily_data) <= num_rows:
            print(f"Pobieram wszystkie {len(daily_data)} wierszy (mniej niż {num_rows})")
            return daily_data
        else:
            print(f"Pobieram pierwsze {num_rows} wierszy z {len(daily_data)} dostępnych")
            return daily_data.head(num_rows)
            
    except Exception as e:
        print(f"BŁĄD podczas filtrowania danych: {e}")
        return None

def display_sample_info(df: pd.DataFrame):
    """Wyświetla informacje o pobranych danych."""
    if df is None or len(df) == 0:
        print("Brak danych do wyświetlenia")
        return
    
    print("\n" + "="*80)
    print("INFORMACJE O POBRANYCH DANYCH:")
    print("="*80)
    print(f"Liczba wierszy: {len(df)}")
    print(f"Zakres czasowy: {df.index.min()} do {df.index.max()}")
    print(f"Liczba kolumn: {len(df.columns)}")
    
    # Podział kolumn na kategorie
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in ohlc_cols]
    
    print(f"\nKolumny OHLC: {len(ohlc_cols)}")
    print(f"Kolumny cech: {len(feature_cols)}")
    
    # Wyświetl pierwsze kilka cech
    if feature_cols:
        print(f"\nPrzykładowe cechy (pierwsze 10):")
        for i, col in enumerate(feature_cols[:10]):
            print(f"  {i+1:2d}. {col}")
        if len(feature_cols) > 10:
            print(f"  ... i {len(feature_cols) - 10} więcej")

def display_sample_data(df: pd.DataFrame, show_features: bool = False):
    """Wyświetla pobrane dane."""
    if df is None or len(df) == 0:
        print("Brak danych do wyświetlenia")
        return
    
    print("\n" + "="*80)
    print("POBRANE DANE:")
    print("="*80)
    
    # Wyświetl podstawowe kolumny OHLC
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    available_ohlc = [col for col in ohlc_cols if col in df.columns]
    
    if available_ohlc:
        print("\nDane OHLC:")
        print(df[available_ohlc].to_string())
    
    # Wyświetl kilka przykładowych cech
    if show_features:
        feature_cols = [col for col in df.columns if col not in ohlc_cols]
        if feature_cols:
            print(f"\nPrzykładowe cechy (pierwsze 5):")
            sample_features = feature_cols[:5]
            print(df[sample_features].to_string())
    
    # Wyświetl pełne dane w formacie tabelarycznym
    print(f"\nPełne dane ({len(df)} wierszy x {len(df.columns)} kolumn):")
    print(df.to_string())

def save_sample_to_csv(df: pd.DataFrame, output_path: str):
    """Zapisuje pobrane dane do pliku CSV."""
    if df is None or len(df) == 0:
        print("Brak danych do zapisania")
        return
    
    try:
        # Resetuj indeks przed zapisem
        df_to_save = df.reset_index()
        df_to_save.to_csv(output_path, index=False)
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"\nZapisano do: {output_path}")
        print(f"Rozmiar pliku: {file_size:.2f} KB")
        
    except Exception as e:
        print(f"BŁĄD podczas zapisywania: {e}")

def main():
    """Główna funkcja."""
    parser = argparse.ArgumentParser(description='Pobiera 10 wierszy z określonej daty z pliku z cechami')
    parser.add_argument('--date', required=True, 
                       help='Data w formacie YYYY-MM-DD (np. 2023-02-01)')
    parser.add_argument('--input', default=str(config.OUTPUT_DIR / config.DEFAULT_OUTPUT_FILENAME),
                       help='Ścieżka do pliku wejściowego z cechami')
    parser.add_argument('--output', 
                       help='Ścieżka do pliku wyjściowego CSV (opcjonalne)')
    parser.add_argument('--rows', type=int, default=10,
                       help='Liczba wierszy do pobrania (domyślnie 10)')
    parser.add_argument('--show-features', action='store_true',
                       help='Pokaż przykładowe cechy w wyświetlaniu')
    parser.add_argument('--info-only', action='store_true',
                       help='Pokaż tylko informacje, bez wyświetlania danych')
    
    args = parser.parse_args()
    
    print("POBIERANIE PRÓBKI DANYCH Z OKREŚLONEJ DATY")
    print("="*60)
    
    # Wczytaj plik z cechami
    df = load_features_file(args.input)
    if df is None:
        return
    
    # Pobierz wiersze z określonej daty
    sample_df = get_rows_for_date(df, args.date, args.rows)
    if sample_df is None:
        return
    
    # Wyświetl informacje
    display_sample_info(sample_df)
    
    # Wyświetl dane (jeśli nie tylko info)
    if not args.info_only:
        display_sample_data(sample_df, args.show_features)
    
    # Zapisz do CSV jeśli podano ścieżkę
    if args.output:
        save_sample_to_csv(sample_df, args.output)
    
    print("\n" + "="*60)
    print("GOTOWE!")

if __name__ == "__main__":
    main() 