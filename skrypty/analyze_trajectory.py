import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict

# --- Konfiguracja Wykresów ---
plt.style.use('seaborn-v0_8-darkgrid')

def analyze_and_present_results(
    mean_trajectories: Dict[int, np.ndarray],
    feature_names: List[str],
    sequence_length: int,
    output_dir: str,
    save_plots: bool = False
):
    """
    Analizuje średnie trajektorie, drukuje podsumowanie statystyczne w konsoli
    i opcjonalnie zapisuje wykresy na dysku.
    """
    print("\n" + "="*80)
    print(" " * 20 + "WYNIKI ANALIZY TRAJEKTORII CECH")
    print("="*80)

    for i, feature_name in enumerate(feature_names):
        print(f"\n--- Analiza cechy: {feature_name} ---")

        headers = ["Etykieta", "Start (t-120)", "Koniec (t-1)", "Trend", "Min", "Max"]
        col_widths = [10, 15, 15, 15, 15, 15]
        print(" | ".join(header.ljust(width) for header, width in zip(headers, col_widths)))
        print("-" * sum(col_widths + [len(col_widths) * 3 - 1]))

        stats = {}
        for label, label_name in [(2, "LONG"), (0, "SHORT"), (1, "HOLD")]:
            if label in mean_trajectories:
                trajectory = mean_trajectories[label][:, i]
                start_val = trajectory[0]
                end_val = trajectory[-1]
                trend = end_val - start_val
                min_val = np.min(trajectory)
                max_val = np.max(trajectory)
                stats[label_name] = {'trend': trend}

                row = [
                    label_name,
                    f"{start_val:.4f}",
                    f"{end_val:.4f}",
                    f"{trend:+.4f}",
                    f"{min_val:.4f}",
                    f"{max_val:.4f}"
                ]
                print(" | ".join(item.ljust(width) for item, width in zip(row, col_widths)))
        
        # --- Automatyczna interpretacja ---
        print("\nInterpretacja:")
        if 'LONG' in stats and 'SHORT' in stats:
            long_trend = stats['LONG']['trend']
            short_trend = stats['SHORT']['trend']
            
            if np.sign(long_trend) != np.sign(short_trend) and abs(long_trend) > 1e-9 and abs(short_trend) > 1e-9:
                 print("  > WNIOSKI: SILNY SYGNAŁ. Trendy dla LONG i SHORT są przeciwstawne.")
            elif abs(long_trend) > abs(short_trend) * 2 or abs(short_trend) > abs(long_trend) * 2:
                 print("  > WNIOSKI: UMIARKOWANY SYGNAŁ. Jeden z trendów jest znacznie silniejszy od drugiego.")
            else:
                 print("  > WNIOSKI: SŁABY SYGNAŁ. Trendy nie wykazują wyraźnego, różnicującego wzorca.")
        else:
            print("  > Brak wystarczających danych do wyciągnięcia wniosków (brak LONG lub SHORT).")

        # --- Opcjonalne zapisywanie wykresów ---
        if save_plots:
            plt.figure(figsize=(12, 7))
            
            if 2 in mean_trajectories:
                plt.plot(range(-sequence_length, 0), mean_trajectories[2][:, i], label='LONG (do sygnału)', color='green', alpha=0.8)
            if 0 in mean_trajectories:
                plt.plot(range(-sequence_length, 0), mean_trajectories[0][:, i], label='SHORT (do sygnału)', color='red', alpha=0.8)
            if 1 in mean_trajectories:
                plt.plot(range(-sequence_length, 0), mean_trajectories[1][:, i], label='HOLD (próbka)', color='gray', linestyle='--', alpha=0.6)
                
            plt.title(f'Średnia trajektoria cechy: "{feature_name}"', fontsize=16)
            plt.xlabel(f'Kroki czasowe (świece 1m) przed wystąpieniem etykiety (t=0)')
            plt.ylabel('Średnia wartość cechy')
            plt.legend()
            
            plot_filename = os.path.join(output_dir, f'{feature_name.replace(" ", "_")}_trajectory.png')
            plt.savefig(plot_filename)
            plt.close()

    print("\n" + "="*80)
    if save_plots:
        print(f"Wykresy zostały zapisane w katalogu: {output_dir}")
    print("Analiza zakończona.")


def calculate_trajectories(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    feature_names: List[str],
    sequence_length: int
) -> Dict[int, np.ndarray]:
    """
    Optymalna funkcja, która oblicza średnie trajektorie "w locie"
    i zwraca je w postaci słownika.
    """
    print("Rozpoczynam analizę trajektorii (metoda zoptymalizowana pod kątem pamięci)...")
    
    data = df_features.join(df_labels, how='inner')
    if data.empty:
        print("Błąd krytyczny: Połączenie cech i etykiet (inner join) dało pusty zbiór danych.")
        return {}

    feature_data_np = data[feature_names].to_numpy()
    label_data_np = data['label'].to_numpy()
    
    num_features = len(feature_names)
    
    trajectory_sums = {
        0: np.zeros((sequence_length, num_features)),
        1: np.zeros((sequence_length, num_features)),
        2: np.zeros((sequence_length, num_features))
    }
    sequence_counts = {0: 0, 1: 0, 2: 0}
    
    total_iterations = len(data) - sequence_length
    if total_iterations <= 0:
        print(f"Błąd: Za mało danych ({len(data)} wierszy) do stworzenia sekwencji o dł. {sequence_length}.")
        return {}

    progress_interval = max(1, total_iterations // 20)
    print(f"Przetwarzanie {total_iterations:,} świec...")

    for i in range(sequence_length, len(data)):
        if (i - sequence_length) % progress_interval == 0:
            progress = (i - sequence_length) / total_iterations
            print(f"  ...Postęp: [{int(progress*20)*'='}>{int((1-progress)*20)*' '}] {progress:.0%}", end='\\r', flush=True)

        label = int(label_data_np[i])
        if label in [0, 1, 2]:
            sequence_counts[label] += 1
            sequence = feature_data_np[i - sequence_length : i]
            trajectory_sums[label] += sequence
            
    print("  ...Postęp: [====================] 100%          ")
    print("Zakończono uśrednianie.")

    mean_trajectories = {}
    for label, count in sequence_counts.items():
        if count > 0:
            mean_trajectories[label] = trajectory_sums[label] / count
        else:
            print(f"Ostrzeżenie: Nie znaleziono żadnych sekwencji dla etykiety {label}.")
            
    return mean_trajectories

def main():
    # --- Konfiguracja ---
    SAVE_PLOTS = False  # Zmień na True, jeśli chcesz dodatkowo zapisać wykresy
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(
        base_dir,
        'labeler',
        'output',
        'BTCUSDT-1m-futures_features_and_labels_FW-480_SL-050_TP-100.feather'
    )
    window_size = 120

    # --- Wczytywanie i Przygotowanie Danych ---
    if not os.path.exists(data_file):
        print(f"Błąd krytyczny: Plik danych nie znaleziony: {data_file}")
        return
        
    print(f"Wczytywanie danych z pliku: {data_file}")
    df = pd.read_feather(data_file)

    if 'date' in df.columns:
        df = df.set_index('date')

    if 'label' not in df.columns:
        print(f"Błąd krytyczny: Brak kolumny 'label' w pliku: {data_file}")
        return

    df_labels = df[['label']].copy()
    df_features = df.drop(columns=['label'])

    print("Rozkład etykiet w pliku:")
    print(df_labels['label'].value_counts(normalize=True).map("{:.2%}".format))
    print("-" * 30)

    # --- Analiza i Prezentacja Wyników ---
    feature_names = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'label']]
    
    output_directory = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'wykresy', 'analiza_trajektorii'
    ))
    os.makedirs(output_directory, exist_ok=True)
    
    mean_trajectories = calculate_trajectories(df_features, df_labels, feature_names, window_size)
    
    if mean_trajectories:
        analyze_and_present_results(mean_trajectories, feature_names, window_size, output_directory, save_plots=SAVE_PLOTS)

if __name__ == '__main__':
    main() 