"""
Wektoryzowana symulacja strategii tradingowej:
- Otwiera pozycje LONG co 5 minut
- TP: 1%, SL: 0.5%
- Śledzi najdłuższy streak przegranych pozycji
- Używa wektoryzacji dla maksymalnej wydajności
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

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

# Import konfiguracji po ustawieniu ścieżki
from training import config as cfg

class VectorizedTradingSimulator:
    def __init__(self, tp_percent=1.0, sl_percent=0.5, position_interval_minutes=5):
        """
        Inicjalizacja wektoryzowanego symulatora tradingowego.
        
        Args:
            tp_percent: Take Profit w procentach (domyślnie 1.0%)
            sl_percent: Stop Loss w procentach (domyślnie 0.5%)
            position_interval_minutes: Interwał między otwarciem pozycji w minutach (domyślnie 5)
        """
        self.tp_percent = tp_percent / 100.0
        self.sl_percent = sl_percent / 100.0
        self.position_interval_minutes = position_interval_minutes
        
    def check_all_positions_vectorized(self, df, positions_df):
        """
        Sprawdza wyniki wszystkich pozycji używając pełnej wektoryzacji.
        
        Args:
            df: DataFrame z danymi
            positions_df: DataFrame z pozycjami do sprawdzenia
            
        Returns:
            DataFrame z wynikami
        """
        print("    - Przygotowuję dane do wektoryzacji...")
        
        # Przygotuj dane
        df_sorted = df.sort_values('date').reset_index(drop=True)
        df_dates = df_sorted['date'].values
        df_highs = df_sorted['high'].values
        df_lows = df_sorted['low'].values
        df_closes = df_sorted['close'].values
        
        results = []
        
        print("    - Sprawdzam pozycje w pętli (zoptymalizowane)...")
        for i, (_, position) in enumerate(positions_df.iterrows()):
            if i % 100 == 0:  # Logi co 100 pozycji
                print(f"      Pozycja {i:,}/{len(positions_df):,} ({(i+1)/len(positions_df)*100:.1f}%)")
            
            open_time = position['open_time']
            open_price = position['open_price']
            
            # Znajdź indeks otwarcia pozycji
            try:
                start_idx = np.searchsorted(df_dates.astype(np.int64), open_time.value, side='left')
            except:
                start_idx = df_sorted['date'].searchsorted(open_time, side='left')
            
            if start_idx >= len(df_sorted):
                results.append(('LOSS', open_time, open_price, 'TIMEOUT'))
                continue
            
            # Wektoryzowane sprawdzenie TP/SL dla tej pozycji
            tp_target = open_price * (1 + self.tp_percent)
            sl_target = open_price * (1 - self.sl_percent)
            
            # Sprawdź tylko świece po otwarciu pozycji
            future_highs = df_highs[start_idx:]
            future_lows = df_lows[start_idx:]
            future_dates = df_dates[start_idx:]
            
            # Wektoryzowane sprawdzenie TP/SL
            tp_hit = future_highs >= tp_target
            sl_hit = future_lows <= sl_target
            
            # Znajdź pierwszy hit
            tp_idx = np.argmax(tp_hit) if tp_hit.any() else -1
            sl_idx = np.argmax(sl_hit) if sl_hit.any() else -1
            
            # Konwertuj na indeksy względem całego DataFrame
            if tp_idx >= 0:
                tp_idx += start_idx
            if sl_idx >= 0:
                sl_idx += start_idx
            
            # Zastosuj logikę TP/SL (identyczna jak wcześniej)
            if tp_idx >= 0 and (sl_idx < 0 or tp_idx <= sl_idx):
                # TP wygrywa
                results.append(('WIN', df_dates[tp_idx], tp_target, 'TP'))
            elif sl_idx >= 0:
                # SL przegrywa
                results.append(('LOSS', df_dates[sl_idx], sl_target, 'SL'))
            else:
                # Timeout
                results.append(('LOSS', df_dates[-1], df_closes[-1], 'TIMEOUT'))
        
        return results
    
    def simulate_strategy_vectorized(self, df, max_positions=None):
        """
        Symuluje strategię tradingową używając wektoryzacji.
        
        Args:
            df: DataFrame z kolumnami 'date', 'close', 'high', 'low'
            max_positions: Maksymalna liczba pozycji do symulacji (dla testów)
        """
        print(f"Rozpoczynam wektoryzowaną symulację strategii:")
        print(f"  - TP: {self.tp_percent*100:.1f}%")
        print(f"  - SL: {self.sl_percent*100:.1f}%")
        print(f"  - Interwał pozycji: {self.position_interval_minutes} minut")
        print(f"  - Okres danych: {df['date'].min()} do {df['date'].max()}")
        print(f"  - Liczba świec: {len(df):,}")
        
        # Konwertuj date na datetime jeśli to string
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])
        
        # Sortuj po dacie
        df = df.sort_values('date').reset_index(drop=True)
        
        # Znajdź momenty otwarcia pozycji (co 5 minut od początku)
        start_time = df['date'].iloc[0]
        position_times = []
        current_time = start_time
        
        while current_time <= df['date'].iloc[-1]:
            position_times.append(current_time)
            current_time += timedelta(minutes=self.position_interval_minutes)
        
        # Ogranicz liczbę pozycji jeśli podano max_positions
        if max_positions:
            position_times = position_times[:max_positions]
            print(f"  - TEST: Symuluję tylko pierwsze {max_positions} pozycji")
        
        print(f"  - Znaleziono {len(position_times):,} momentów otwarcia pozycji")
        
        # Utwórz DataFrame z pozycjami
        positions_df = pd.DataFrame({
            'position_id': range(1, len(position_times) + 1),
            'open_time': position_times
        })
        
        # Dodaj ceny otwarcia z postępem (zoptymalizowane)
        print("  - Obliczam ceny otwarcia pozycji (zoptymalizowane)...")
        start_time = time.time()
        
        # Konwertuj daty na numeryczne dla szybszego wyszukiwania
        df_sorted = df.sort_values('date').reset_index(drop=True)
        df_dates = df_sorted['date'].values
        df_closes = df_sorted['close'].values
        
        open_prices = []
        for i, open_time in enumerate(positions_df['open_time']):
            if i % 50 == 0:  # Bardzo częste logi
                elapsed = time.time() - start_time
                print(f"    Postęp cen: {i:,}/{len(positions_df):,} ({(i+1)/len(positions_df)*100:.1f}%) - Czas: {elapsed:.1f}s")
            
            # Szybsze wyszukiwanie używając searchsorted z konwersją typów
            try:
                idx = np.searchsorted(df_dates.astype(np.int64), open_time.value, side='left')
            except:
                # Fallback: użyj pandas searchsorted
                idx = df_sorted['date'].searchsorted(open_time, side='left')
            
            if idx >= len(df_closes):
                idx = len(df_closes) - 1
            price = df_closes[idx]
            open_prices.append(price)
        
        positions_df['open_price'] = open_prices
        
        # Wektoryzowane sprawdzenie wyników pozycji (zoptymalizowane)
        print("  - Sprawdzam wyniki pozycji (w pełni zoptymalizowane)...")
        start_time = time.time()
        
        results = self.check_all_positions_vectorized(df_sorted, positions_df)
        
        elapsed = time.time() - start_time
        print(f"  - Sprawdzanie zakończone w {elapsed:.2f} sekund")
        
        # Rozpakuj wyniki
        positions_df[['result', 'close_time', 'close_price', 'reason']] = pd.DataFrame(
            results, index=positions_df.index
        )
        
        # Oblicz statystyki
        self.calculate_statistics(positions_df)
        
        return positions_df
    
    def calculate_statistics(self, positions_df):
        """Oblicza statystyki z wyników symulacji."""
        print("  - Obliczam statystyki...")
        
        # Podstawowe statystyki
        total_positions = len(positions_df)
        winning_positions = (positions_df['result'] == 'WIN').sum()
        losing_positions = (positions_df['result'] == 'LOSS').sum()
        
        # Wektoryzowane obliczanie streaków
        positions_df['is_loss'] = positions_df['result'] == 'LOSS'
        positions_df['streak_group'] = (positions_df['is_loss'] != positions_df['is_loss'].shift()).cumsum()
        positions_df['current_streak'] = positions_df.groupby('streak_group')['is_loss'].cumsum()
        
        max_losing_streak = positions_df[positions_df['is_loss']]['current_streak'].max()
        if pd.isna(max_losing_streak):
            max_losing_streak = 0
        
        # Znajdź najdłuższy streak z datami
        max_streak_mask = positions_df['current_streak'] == max_losing_streak
        max_streak_positions = positions_df[max_streak_mask & positions_df['is_loss']]
        
        if len(max_streak_positions) > 0:
            max_streak_start = max_streak_positions.iloc[0]['open_time']
            max_streak_end = max_streak_positions.iloc[-1]['close_time']
        else:
            max_streak_start = None
            max_streak_end = None
        
        # Zapisz statystyki
        self.stats = {
            'total_positions': total_positions,
            'winning_positions': winning_positions,
            'losing_positions': losing_positions,
            'max_losing_streak': max_losing_streak,
            'max_streak_start': max_streak_start,
            'max_streak_end': max_streak_end,
            'positions_df': positions_df
        }
        
        print("  - Obliczenia zakończone!")
    
    def print_results(self):
        """Wyświetla wyniki symulacji."""
        if not hasattr(self, 'stats'):
            print("BŁĄD: Brak wyników do wyświetlenia.")
            return
            
        print("\n" + "="*60)
        print("WYNIKI WEKTORYZOWANEJ SYMULACJI STRATEGII TRADINGOWEJ")
        print("="*60)
        
        print(f"Parametry strategii:")
        print(f"  - Take Profit: {self.tp_percent*100:.1f}%")
        print(f"  - Stop Loss: {self.sl_percent*100:.1f}%")
        print(f"  - Interwał pozycji: {self.position_interval_minutes} minut")
        
        print(f"\nStatystyki ogólne:")
        print(f"  - Całkowita liczba pozycji: {self.stats['total_positions']:,}")
        print(f"  - Pozycje wygrywające: {self.stats['winning_positions']:,} ({self.stats['winning_positions']/self.stats['total_positions']*100:.1f}%)")
        print(f"  - Pozycje przegrywające: {self.stats['losing_positions']:,} ({self.stats['losing_positions']/self.stats['total_positions']*100:.1f}%)")
        
        print(f"\nAnaliza streaków:")
        print(f"  - Najdłuższy streak przegranych: {self.stats['max_losing_streak']}")
        
        if self.stats['max_streak_start'] and self.stats['max_streak_end']:
            print(f"\nNajdłuższy streak przegranych:")
            print(f"  - Od: {self.stats['max_streak_start']}")
            print(f"  - Do: {self.stats['max_streak_end']}")
            print(f"  - Długość: {self.stats['max_losing_streak']} pozycji")
        
        # Dodatkowe statystyki
        positions_df = self.stats['positions_df']
        
        # Średni czas trwania pozycji (naprawione timezone)
        try:
            # Upewnij się, że oba kolumny mają ten sam format timezone
            open_times = positions_df['open_time'].dt.tz_localize(None) if positions_df['open_time'].dt.tz is not None else positions_df['open_time']
            close_times = positions_df['close_time'].dt.tz_localize(None) if positions_df['close_time'].dt.tz is not None else positions_df['close_time']
            
            positions_df['duration_minutes'] = (close_times - open_times).dt.total_seconds() / 60
            avg_duration = positions_df['duration_minutes'].mean()
            print(f"  - Średni czas trwania pozycji: {avg_duration:.1f} minut")
        except Exception as e:
            print(f"  - Błąd obliczania czasu trwania: {e}")
            print(f"  - Pomijam obliczanie średniego czasu trwania")
        
        # Analiza przyczyn zamknięcia
        reason_counts = positions_df['reason'].value_counts()
        print(f"\nPrzyczyny zamknięcia pozycji:")
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count:,} ({count/len(positions_df)*100:.1f}%)")

def main():
    """Główna funkcja."""
    print("="*60)
    print("WEKTORYZOWANA SYMULACJA STRATEGII TRADINGOWEJ")
    print("="*60)
    
    # Wczytaj dane
    print(f"Wczytywanie danych z: {cfg.INPUT_FILE_PATH}")
    if not os.path.exists(cfg.INPUT_FILE_PATH):
        print(f"BŁĄD: Plik danych nie istnieje.")
        return
    
    df = pd.read_feather(cfg.INPUT_FILE_PATH)
    print(f"Wczytano {len(df):,} wierszy danych")
    
    # Sprawdź wymagane kolumny
    required_columns = ['date', 'close', 'high', 'low']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"BŁĄD: Brakujące kolumny: {missing_columns}")
        return
    
    # Utwórz symulator
    simulator = VectorizedTradingSimulator(
        tp_percent=1.0,      # 1% TP
        sl_percent=0.5,       # 0.5% SL
        position_interval_minutes=5  # Co 5 minut
    )
    
    # Uruchom symulację (możesz dodać max_positions=1000 dla testów)
    start_time = time.time()
    
    # Dla testów - użyj mniejszego zbioru danych
    test_mode = True  # Zmień na False dla pełnej symulacji
    if test_mode:
        print("TEST MODE: Symuluję tylko pierwsze 1000 pozycji")
        positions_df = simulator.simulate_strategy_vectorized(df, max_positions=1000)
    else:
        positions_df = simulator.simulate_strategy_vectorized(df)
    
    total_time = time.time() - start_time
    
    print(f"\nCałkowity czas symulacji: {total_time:.2f} sekund")
    
    # Wyświetl wyniki
    simulator.print_results()
    
    print("\n" + "="*60)
    print("Wektoryzowana symulacja zakończona.")
    print("="*60)

if __name__ == "__main__":
    main() 