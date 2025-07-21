import pandas as pd
import os

def analyze_label_distribution(file_path: str):
    """
    Analizuje i wyświetla rozkład etykiet w danym pliku Feather.

    Args:
        file_path (str): Ścieżka do pliku .feather.
    """
    if not os.path.exists(file_path):
        print(f"\n--- Błąd ---")
        print(f"Plik nie został znaleziony pod ścieżką zdefiniowaną w skrypcie:")
        print(f"  -> {file_path}")
        print("-" * 12)
        return

    try:
        # Optymalizacja: Wczytujemy tylko kolumnę 'label'
        df = pd.read_feather(file_path, columns=['label'])
    except Exception as e:
        print(f"\n--- Błąd w pliku: {os.path.basename(file_path)} ---")
        print(f"Nie można wczytać pliku: {e}")
        print("-" * (len(file_path) + 4))
        return

    if 'label' not in df.columns:
        print(f"\n--- Błąd w pliku: {os.path.basename(file_path)} ---")
        print("W pliku brakuje kolumny 'label'.")
        print("-" * (len(file_path) + 4))
        return

    total_rows = len(df)
    if total_rows == 0:
        print(f"\n--- Plik: {os.path.basename(file_path)} ---")
        print("Plik jest pusty.")
        print("-" * (len(file_path) + 4))
        return

    label_counts = df['label'].value_counts()
    label_percentages = df['label'].value_counts(normalize=True) * 100

    # Mapowanie etykiet numerycznych na nazwy
    label_map = {
        0: 'PROFIT_SHORT',
        1: 'TIMEOUT_HOLD',
        2: 'PROFIT_LONG',
        3: 'LOSS_SHORT',
        4: 'LOSS_LONG',
        5: 'CHAOS_HOLD',
    }
    
    # Ocena balansu na podstawie sumy "pasywnych" etykiet
    hold_percentage = label_percentages.get(1, 0) + label_percentages.get(5, 0)
    
    if hold_percentage > 75:
        balance_assessment = "🔴 Ekstremalnie niezbalansowany (dużo HOLD)"
    elif hold_percentage > 60:
        balance_assessment = "🟠 Niezbalansowany (sporo HOLD)"
    else:
        balance_assessment = "🟢 Dobrze zbalansowany"

    print(f"\n--- Analiza Pliku: {os.path.basename(file_path)} ---")
    print(f"Całkowita liczba wierszy: {total_rows:,}")
    print("-" * 30)
    
    for label_id, label_name in sorted(label_map.items()):
        count = label_counts.get(label_id, 0)
        percent = label_percentages.get(label_id, 0)
        print(f"  - {label_name:<15}: {percent:6.2f}% ({count:10,d} wierszy)")
        
    print("-" * 30)
    print(f"Ocena balansu: {balance_assessment}")
    print("-" * (len(os.path.basename(file_path)) + 19))


if __name__ == '__main__':
    # =================================================================
    #  👇 TUTAJ WPISZ NAZWĘ PLIKU, KTÓRY CHCESZ PRZEANALIZOWAĆ 👇
    # =================================================================
    
    # Ścieżka jest budowana od lokalizacji tego skryptu
    # Nazwa pliku wygenerowana dla parametrów: FW=240, SL=0.3, TP=0.6
    FILENAME = "BTCUSDT-1m-futures_features_and_labels_FW-020_SL-010_TP-030.feather"
    
    # =================================================================
    
    # Budowanie pełnej ścieżki do pliku
    # Zakładamy, że plik znajduje się w katalogu ../labeler/output/
    try:
        script_dir = os.path.dirname(__file__)
        file_to_analyze = os.path.normpath(os.path.join(
            script_dir, '..', 'labeler', 'output', FILENAME
        ))
        
        analyze_label_distribution(file_to_analyze)
        
    except NameError:
        print("Błąd: Wygląda na to, że skrypt jest uruchamiany w środowisku, gdzie `__file__` nie jest zdefiniowane.")
        print("Proszę uruchomić skrypt bezpośrednio przez `python skrypty/check_label_distribution.py`") 