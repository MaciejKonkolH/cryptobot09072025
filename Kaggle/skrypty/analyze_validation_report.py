import pandas as pd
import argparse
from pathlib import Path

def analyze_report(file_path: Path):
    """
    Analizuje plik z wynikami walidacji, oblicza i wyświetla statystyki
    poprawności predykcji.

    Args:
        file_path (Path): Ścieżka do pliku validation_analysis_...csv.
    """
    if not file_path.exists():
        print(f"❌ BŁĄD: Plik nie istnieje: {file_path}")
        return

    print(f"🔍 Analizowanie pliku: {file_path.name}")

    try:
        df = pd.read_csv(file_path)

        if 'is_correct' not in df.columns:
            print(f"❌ BŁĄD: W pliku brakuje wymaganej kolumny 'is_correct'.")
            return
            
        if df.empty:
            print("⚠️ Plik jest pusty. Brak predykcji do analizy.")
            return

        total_predictions = len(df)
        correct_predictions = df['is_correct'].sum()
        incorrect_predictions = total_predictions - correct_predictions

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        print("\n" + "="*50)
        print("📊 WYNIKI ANALIZY POPRAWNOŚCI PREDYKCJI")
        print("="*50)
        print(f"Całkowita liczba predykcji (SHORT/LONG): {total_predictions:>8,}")
        print("-" * 50)
        print(f"✅ Poprawne predykcje: {'':<18} {correct_predictions:>8,} ({accuracy:.2f}%)")
        print(f"❌ Błędne predykcje: {'':<20} {incorrect_predictions:>8,} ({100-accuracy:.2f}%)")
        print("="*50)

    except Exception as e:
        print(f"❌ BŁĄD: Wystąpił nieoczekiwany problem podczas przetwarzania pliku: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analizuje raport z walidacji modelu i podsumowuje jego skuteczność."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Ścieżka do pliku CSV z analizą walidacji (np. 'Kaggle/output/validation_analysis_...csv')."
    )

    args = parser.parse_args()
    analyze_report(Path(args.file_path)) 