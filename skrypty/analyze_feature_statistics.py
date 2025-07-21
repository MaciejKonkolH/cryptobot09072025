import pandas as pd
import argparse
import os

def analyze_feature_statistics(file_path: str):
    """
    Analizuje i wyświetla statystyki (średnia, odch. std.) cech pogrupowanych
    według etykiet (SHORT, HOLD, LONG).

    Args:
        file_path (str): Ścieżka do pliku .feather z danymi.
    """
    if not os.path.exists(file_path):
        print(f"Błąd: Plik nie został znaleziony pod ścieżką: {file_path}")
        return

    print(f"Wczytywanie danych z pliku: {os.path.basename(file_path)}...")
    try:
        df = pd.read_feather(file_path)
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return

    print("Dane wczytane pomyślnie.")

    # Definicja cech do analizy
    features_to_analyze = [
        'bb_width', 'bb_position', 'rsi_14', 'macd_hist_norm', 'adx_14',
        'price_to_ma_60', 'price_to_ma_240', 'ma_60_to_ma_240',
        'volume_change_norm', 'price_to_ma_1440', 'price_to_ma_43200',
        'volume_to_ma_1440', 'volume_to_ma_43200',
        'open', 'high', 'low', 'close'
    ]
    
    # Sprawdzenie, czy wszystkie cechy istnieją w DataFrame
    missing_features = [f for f in features_to_analyze if f not in df.columns]
    if missing_features:
        print(f"Błąd: W pliku brakuje następujących cech: {missing_features}")
        return

    # Mapowanie etykiet numerycznych na nazwy dla czytelności
    label_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
    df['label_name'] = df['label'].map(label_map)

    print("\nAnalizowanie statystyk cech...\n")
    
    # Grupowanie po nazwie etykiety i obliczanie statystyk
    grouped_stats = df.groupby('label_name')[features_to_analyze].agg(['mean', 'std'])

    # Formatowanie wyświetlania
    pd.set_option('display.float_format', '{:,.4f}'.format)
    
    print("======================================================================")
    print("  ŚREDNIE WARTOŚCI CECH DLA KAŻDEJ ETYKIETY (SHORT / HOLD / LONG)  ")
    print("======================================================================")
    # POPRAWKA: Użycie metody .xs() do poprawnego wybrania danych z MultiIndex
    print(grouped_stats.xs('mean', level=1, axis=1).T)
    
    print("\n" + "="*70)
    print("\n======================================================================")
    print("  ODCHYLENIE STANDARDOWE CECH DLA KAŻDEJ ETYKIETY (ZMIENNOŚĆ) ")
    print("======================================================================")
    # POPRAWKA: Użycie metody .xs() do poprawnego wybrania danych z MultiIndex
    print(grouped_stats.xs('std', level=1, axis=1).T)
    print("\n" + "="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analizuje statystyki cech (średnia, odch. std.) w podziale na etykiety SHORT, HOLD, LONG.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Ustalanie domyślnej ścieżki do pliku, który dał najlepsze wyniki
    default_file_path = os.path.join(
        os.path.dirname(__file__), '..', 'labeler', 'output', 
        'BTCUSDT-1m-futures_features_and_labels_FW-480_SL-050_TP-100.feather'
    )
    default_file_path = os.path.normpath(default_file_path)

    parser.add_argument(
        '--file',
        type=str,
        default=default_file_path,
        help=(
            "Ścieżka do pliku .feather z cechami i etykietami.\n"
            f"Domyślnie: {default_file_path}"
        )
    )

    args = parser.parse_args()
    analyze_feature_statistics(args.file) 