import pandas as pd
import json
from pathlib import Path
import logging
import zipfile
import shutil
import tempfile

# --- Konfiguracja ---
# Ustawienie loggera
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ustalanie ścieżek względem lokalizacji skryptu, aby działał z każdego miejsca
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
except NameError:
    # Fallback dla interaktywnego uruchomienia
    SCRIPT_DIR = Path.cwd()
    PROJECT_ROOT = SCRIPT_DIR

BACKTEST_RESULTS_DIR = PROJECT_ROOT / "ft_bot_clean" / "user_data" / "backtest_results"
OUTPUT_CSV_PATH = BACKTEST_RESULTS_DIR / "trades_report.csv"

def find_latest_zip_file(directory: Path) -> Path | None:
    """Znajduje najnowszy plik ZIP z wynikami backtestu."""
    logging.info(f"Przeszukuję folder w poszukiwaniu plików .zip: {directory}")
    if not directory.exists():
        logging.error(f"Folder docelowy nie istnieje: {directory}")
        return None
        
    zip_files = list(directory.glob('backtest-result-*.zip'))

    if not zip_files:
        logging.warning("Nie znaleziono plików .zip z wynikami backtestu (np. 'backtest-result-....zip').")
        return None

    logging.info(f"Znaleziono {len(zip_files)} plików .zip. Wybieram najnowszy.")
    latest_file = max(zip_files, key=lambda f: f.stat().st_mtime)
    logging.info(f"Wybrano najnowszy plik .zip: {latest_file.name}")
    return latest_file

def extract_zip_and_find_json(zip_path: Path) -> tuple[Path | None, Path | None]:
    """Wypakowuje plik ZIP i znajduje w nim plik JSON z wynikami."""
    temp_dir = Path(tempfile.mkdtemp(prefix="freqtrade_"))
    logging.info(f"Rozpakowuję archiwum do folderu tymczasowego: {temp_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Szukaj pliku JSON w rozpakowanym folderze
        json_files = list(temp_dir.glob('backtest-result-*.json'))
        non_meta_files = [f for f in json_files if not f.name.endswith('.meta.json')]

        if not non_meta_files:
            logging.error("Nie znaleziono pliku JSON z wynikami w archiwum.")
            shutil.rmtree(temp_dir)
            return None, None

        # Zwracamy ścieżkę do pliku i do folderu tymczasowego, żeby go później usunąć
        return non_meta_files[0], temp_dir

    except Exception as e:
        logging.error(f"Błąd podczas rozpakowywania pliku ZIP: {e}")
        shutil.rmtree(temp_dir) # Posprzątaj w razie błędu
        return None, None


def extract_trades_from_json(file_path: Path) -> list:
    """Wyciąga listę transakcji ze wszystkich strategii w pliku JSON."""
    all_trades = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'strategy' not in data:
        logging.error("Plik JSON nie zawiera klucza 'strategy'.")
        return []

    for strategy_name, strategy_data in data['strategy'].items():
        if 'trades' in strategy_data:
            logging.info(f"Znaleziono {len(strategy_data['trades'])} transakcji dla strategii '{strategy_name}'.")
            # Dodajemy informację o strategii do każdej transakcji
            for trade in strategy_data['trades']:
                trade['strategy'] = strategy_name
            all_trades.extend(strategy_data['trades'])
        else:
            logging.warning(f"Brak klucza 'trades' dla strategii '{strategy_name}'.")

    return all_trades

def save_trades_to_csv(trades: list, output_path: Path):
    """Zapisuje listę transakcji do ładnie sformatowanego pliku CSV."""
    if not trades:
        logging.warning("Brak transakcji do zapisania.")
        return

    df = pd.DataFrame(trades)

    # 1. Stwórz nową kolumnę 'direction' na podstawie 'is_short'
    if 'is_short' in df.columns:
        df['direction'] = df['is_short'].apply(lambda x: 'SHORT' if x else 'LONG')
    else:
        df['direction'] = 'N/A' # Fallback

    # 2. Zaokrąglij wartości w wybranych kolumnach
    df['open_rate'] = pd.to_numeric(df['open_rate'], errors='coerce').round(2)
    df['close_rate'] = pd.to_numeric(df['close_rate'], errors='coerce').round(2)
    df['stake_amount'] = pd.to_numeric(df['stake_amount'], errors='coerce').round(4)
    df['profit_abs'] = pd.to_numeric(df['profit_abs'], errors='coerce').round(6)

    # 3. Zdefiniuj ostateczną listę i kolejność kolumn
    final_columns_order = [
        'open_date', 'close_date', 'direction', 'exit_reason',
        'open_rate', 'close_rate', 'stake_amount',
        'profit_ratio', 'profit_abs'
    ]

    # Użyj tylko tych kolumn z `final_columns_order`, które faktycznie istnieją w DataFrame
    existing_columns = [col for col in final_columns_order if col in df.columns]
    df_final = df[existing_columns]

    logging.info(f"Zapisuję {len(df_final)} transakcji do pliku: {output_path}")

    # Konwertujemy dataframe do stringa CSV
    csv_data = df_final.to_csv(
        index=False,
        sep=';',
        decimal='.',
    )

    # Ręcznie zapisujemy string do pliku, aby zapewnić kompatybilność
    with open(output_path, 'w', encoding='utf-8-sig', newline='\r\n') as f:
        f.write(csv_data)

    logging.info(f"✅ Pomyślnie zapisano raport transakcji: {output_path}")


def main():
    """Główna funkcja skryptu."""
    latest_zip = find_latest_zip_file(BACKTEST_RESULTS_DIR)
    if not latest_zip:
        return

    json_file_path, temp_dir = extract_zip_and_find_json(latest_zip)
    
    if json_file_path and temp_dir:
        try:
            trades = extract_trades_from_json(json_file_path)
            save_trades_to_csv(trades, OUTPUT_CSV_PATH)
        finally:
            # Posprzątaj folder tymczasowy
            logging.info(f"Usuwam folder tymczasowy: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main() 