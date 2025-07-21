"""
Skrypt do naukowej analizy szeregu czasowego cen w celu weryfikacji
hipotezy błądzenia losowego oraz oceny siły predykcyjnej cech.

FAZA 1: Analiza "Pamięci Rynku" za pomocą Wykładnika Hursta (H).
    - H == 0.5 -> Błądzenie losowe (brak pamięci).
    - H > 0.5 -> Rynek trendujący (pamięć pozytywna).
    - H < 0.5 -> Rynek powracający do średniej (pamięć negatywna).

FAZA 2: Analiza Siły Predykcyjnej Cech.
    - Oblicza korelację każdej z cech z przyszłymi zwrotami cen.
    - Tworzy ranking cech, od najbardziej do najmniej predykcyjnych.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# --- Dynamiczne dodawanie ścieżki projektu ---
def _find_project_root(start_path: Path, marker_file: str = ".project_root") -> Path:
    path = start_path
    while path.parent != path:
        if (path / marker_file).is_file():
            return path
        path = path.parent
    raise FileNotFoundError(f"Nie znaleziono pliku znacznika '{marker_file}' w żadnym z nadrzędnych katalogów.")

try:
    project_root = _find_project_root(Path(__file__).resolve().parent)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except FileNotFoundError as e:
    print(f"BŁĄD KRYTYCZNY: {e}", file=sys.stderr)
    sys.exit(1)
# --- Koniec dynamicznego dodawania ścieżki ---

# Import konfiguracji po ustawieniu ścieżki
from training import config as cfg

def calculate_hurst(series: pd.Series, max_chunk_size: int = 20000) -> float:
    """
    Oblicza Wykładnik Hursta dla szeregu czasowego za pomocą analizy R/S (Rescaled Range).
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Usuwamy wartości NaN, które mogły powstać przy liczeniu zwrotów
    series = series.dropna()
    
    print(f"   Analizuję szereg czasowy o długości: {len(series):,} punktów")
    
    if len(series) < 100:
        print("Ostrzeżenie: Szereg czasowy jest zbyt krótki do wiarygodnej analizy Hursta (<100 punktów).")
        return np.nan

    min_chunk_size = 10
    max_chunk_size = min(len(series) // 2, max_chunk_size)
    
    print(f"   Używam rozmiarów okien od {min_chunk_size} do {max_chunk_size}")
    
    # Używamy skali logarytmicznej do wyboru rozmiarów okien
    n_sizes = [int(s) for s in np.geomspace(min_chunk_size, max_chunk_size, 20)]
    n_sizes = sorted(list(set(n_sizes))) # Unikalne i posortowane
    
    print(f"   Analizuję {len(n_sizes)} różnych rozmiarów okien: {n_sizes[:5]}...{n_sizes[-5:]}")
    
    rs_values = []
    valid_n_sizes = []
    
    for i, n in enumerate(n_sizes):
        print(f"   Okno {i+1}/{len(n_sizes)}: rozmiar {n} ({(i+1)/len(n_sizes)*100:.1f}%)")
        
        # Podział na nienakładające się na siebie fragmenty o rozmiarze n
        chunks = [series.iloc[i:i+n] for i in range(0, len(series) - n + 1, n)]
        
        if not chunks:
            continue

        rs_per_chunk = []
        for chunk in chunks:
            # 1. Obliczamy serię odchyleń od średniej
            mean = np.mean(chunk)
            y = chunk - mean
            
            # 2. Tworzymy skumulowaną sumę odchyleń (cumulative deviate series)
            z = np.cumsum(y)
            
            # 3. Obliczamy rozpiętość (range)
            r = np.max(z) - np.min(z)
            
            # 4. Obliczamy odchylenie standardowe
            s = np.std(chunk)
            
            if s > 0:
                rs_per_chunk.append(r / s)

        if rs_per_chunk:
            mean_rs = np.mean(rs_per_chunk)
            rs_values.append(mean_rs)
            valid_n_sizes.append(n)
            print(f"      Średnia wartość R/S dla tego okna: {mean_rs:.4f}")

    if not rs_values:
        print("   BŁĄD: Nie udało się obliczyć żadnych wartości R/S")
        return np.nan

    print(f"   Obliczono {len(rs_values)} wartości R/S. Dopasowuję linię prostą...")

    # POPRAWKA: Używamy logarytmów do obliczenia wykładnika Hursta
    # Wykładnik Hursta: log(R/S) = H * log(n) + C
    log_n = np.log(valid_n_sizes)
    log_rs = np.log(rs_values)

    if len(log_n) < 2:
        print("   BŁĄD: Za mało punktów do dopasowania linii prostej")
        return np.nan

    hurst_exponent, _ = np.polyfit(log_n, log_rs, 1)
    
    print(f"   Wykładnik Hursta obliczony pomyślnie")
    return hurst_exponent

def analyze_feature_predictive_power(df: pd.DataFrame, features: list, horizons: list) -> pd.DataFrame:
    """
    Oblicza korelację pomiędzy wartością cechy w danym momencie a przyszłym zwrotem ceny.
    """
    print("\nFAZA 2: Analiza Siły Predykcyjnej Cech")
    print("-" * 60)
    
    print(f"   Analizuję {len(features)} cech dla {len(horizons)} horyzontów czasowych")
    print(f"   Horyzonty: {horizons} minut")
    
    results = {}
    
    for horizon in horizons:
        print(f"   Obliczam przyszłe zwroty dla horyzontu {horizon} minut...")
        # Obliczamy przyszły logarytmiczny zwrot ceny
        # Używamy shift(-horizon), aby "przesunąć przyszłość" do obecnego wiersza
        df[f'future_return_{horizon}m'] = np.log(df['close'].shift(-horizon) / df['close'])
    
    # Usuwamy ostatnie wiersze, dla których nie można obliczyć przyszłych zwrotów
    df.dropna(inplace=True)
    print(f"   Po usunięciu NaN pozostało {len(df):,} wierszy do analizy")
    
    for i, feature in enumerate(features):
        print(f"   Analizuję cechę {i+1}/{len(features)}: {feature}")
        correlations = []
        for horizon in horizons:
            corr = df[feature].corr(df[f'future_return_{horizon}m'])
            correlations.append(corr)
        results[feature] = correlations
        
    print("   Tworzę raport rankingowy...")
    
    # Tworzymy czytelną ramkę danych z wynikami
    corr_df = pd.DataFrame.from_dict(
        results, 
        orient='index', 
        columns=[f'corr_in_{h}m' for h in horizons]
    )
    
    # Dodajemy kolumnę z absolutną średnią korelacją, aby ułatwić ranking
    corr_df['abs_mean_corr'] = corr_df.abs().mean(axis=1)
    corr_df.sort_values(by='abs_mean_corr', ascending=False, inplace=True)
    
    return corr_df

def main():
    """Główna funkcja orkiestrująca analizę."""
    print("=" * 60)
    print("ANALIZA PAMIĘCI RYNKU I SIŁY PREDYKCYJNEJ CECH")
    print("=" * 60)

    # --- Wczytywanie danych ---
    print(f"Wczytywanie danych z: {cfg.INPUT_FILE_PATH}\n")
    if not os.path.exists(cfg.INPUT_FILE_PATH):
        print(f"BŁĄD: Plik danych nie istnieje. Sprawdź ścieżkę w training/config.py")
        return
        
    df = pd.read_feather(cfg.INPUT_FILE_PATH)
    print(f"Wczytano {len(df):,} wierszy danych")
    print(f"Zakres dat: od {df['date'].min()} do {df['date'].max()}")
    
    # --- FAZA 1: Obliczenie Wykładnika Hursta ---
    print("\nFAZA 1: Analiza Pamięci Rynku (Wykładnik Hursta)")
    print("-" * 60)
    
    # Obliczamy logarytmiczne zwroty, ponieważ są one bardziej stacjonarne
    print("Obliczam logarytmiczne zwroty cen...")
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    print("Rozpoczynam obliczanie Wykładnika Hursta...")
    hurst_exponent = calculate_hurst(log_returns)
    
    print(f"\nObliczony Wykładnik Hursta (H) = {hurst_exponent:.4f}\n")
    
    if np.isnan(hurst_exponent):
        print("Nie udało się obliczyć Wykładnika Hursta.")
        return

    print("Interpretacja:")
    if 0.55 > hurst_exponent > 0.45:
        print(" -> Wartość H jest bardzo bliska 0.5. Oznacza to, że rynek")
        print("    zachowuje się w sposób BARDZO ZBLIŻONY DO LOSOWEGO (błądzenie losowe).")
        print("    Przewidywanie trendów na podstawie historycznych cen jest niezwykle trudne.")
    elif hurst_exponent >= 0.55:
        print(" -> Wartość H > 0.5. Oznacza to, że rynek wykazuje cechy TRENDUJĄCE.")
        print("    Istnieje statystyczna tendencja do kontynuacji ruchów (pamięć pozytywna).")
        print("    Strategie podążania za trendem mają teoretyczne podstawy.")
    else: # hurst_exponent <= 0.45
        print(" -> Wartość H < 0.5. Oznacza to, że rynek wykazuje cechy POWROTU DO ŚREDNIEJ.")
        print("    Istnieje statystyczna tendencja do odwracania ruchów (pamięć negatywna).")
        print("    Strategie oscylacyjne (kupowanie dołków, sprzedawanie szczytów) mają podstawy.")

    # --- FAZA 2: Analiza korelacji cech ---
    # Definiujemy horyzonty czasowe dla analizy (w minutach)
    horizons = [5, 15, 30, 60, 120]
    
    # Używamy tylko cech zdefiniowanych w konfiguracji treningowej
    features_to_analyze = cfg.FEATURES
    
    correlation_report = analyze_feature_predictive_power(df.copy(), features_to_analyze, horizons)
    
    print("\nRanking cech na podstawie ich średniej korelacji z przyszłymi zwrotami:")
    print(correlation_report.to_string())
    
    print("\nInterpretacja tabeli:")
    print(" - Dodatnia korelacja: gdy cecha rośnie, cena w przyszłości również ma tendencję do wzrostu.")
    print(" - Ujemna korelacja: gdy cecha rośnie, cena w przyszłości ma tendencję do spadku.")
    print(" - Wartości bliskie zera oznaczają brak liniowej zależności (cecha jest słabym predyktorem).")
    print(" - `abs_mean_corr` to miara ogólnej siły predykcyjnej cechy, im wyższa, tym lepsza.")

    print("\n" + "="*60)
    print("Analiza zakończona.")
    print("="*60)

if __name__ == "__main__":
    main() 