"""
Skrypt do symulacji strategii tradingowej:
- Otwiera pozycje LONG co 5 minut
- TP: 1%, SL: 0.5%
- Śledzi najdłuższy streak przegranych pozycji
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

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

class TradingSimulator:
    def __init__(self, tp_percent=1.0, sl_percent=0.5, position_interval_minutes=5):
        """
        Inicjalizacja symulatora tradingowego.
        
        Args:
            tp_percent: Take Profit w procentach (domyślnie 1.0%)
            sl_percent: Stop Loss w procentach (domyślnie 0.5%)
            position_interval_minutes: Interwał między otwarciem pozycji w minutach (domyślnie 5)
        """
        self.tp_percent = tp_percent / 100.0  # Konwersja na ułamek dziesiętny
        self.sl_percent = sl_percent / 100.0
        self.position_interval_minutes = position_interval_minutes
        
        # Statystyki
        self.total_positions = 0
        self.winning_positions = 0
        self.losing_positions = 0
        self.current_streak = 0
        self.max_losing_streak = 0
        
        # Lista wszystkich pozycji do analizy
        self.positions = []
        
    def simulate_strategy(self, df):
        """
        Symuluje strategię tradingową na danych.
        
        Args:
            df: DataFrame z kolumnami 'date', 'close', 'high', 'low'
        """
        print(f"Rozpoczynam symulację strategii:")
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
        
        print(f"  - Znaleziono {len(position_times)} momentów otwarcia pozycji")
        
        # Symuluj każdą pozycję
        for i, open_time in enumerate(position_times):
            if i % 1000 == 0:
                print(f"    Symuluję pozycję {i+1:,}/{len(position_times):,} ({(i+1)/len(position_times)*100:.1f}%)...")
            elif i % 100 == 0:
                print(f"      Postęp: {i+1:,}/{len(position_times):,} ({(i+1)/len(position_times)*100:.1f}%)")
            
            # Znajdź indeks otwarcia pozycji
            open_idx = df[df['date'] >= open_time].index[0] if len(df[df['date'] >= open_time]) > 0 else None
            
            if open_idx is None:
                continue
                
            # Otwórz pozycję
            open_price = df.loc[open_idx, 'close']
            position = {
                'position_id': i + 1,
                'open_time': open_time,
                'open_price': open_price,
                'open_idx': open_idx,
                'result': None,
                'close_time': None,
                'close_price': None,
                'close_idx': None,
                'reason': None
            }
            
            # Sprawdź czy pozycja zostanie zamknięta w kolejnych świecach
            for j in range(open_idx + 1, len(df)):
                current_high = df.loc[j, 'high']
                current_low = df.loc[j, 'low']
                current_time = df.loc[j, 'date']
                
                # Sprawdź TP
                if current_high >= open_price * (1 + self.tp_percent):
                    position['result'] = 'WIN'
                    position['close_time'] = current_time
                    position['close_price'] = open_price * (1 + self.tp_percent)
                    position['close_idx'] = j
                    position['reason'] = 'TP'
                    break
                
                # Sprawdź SL
                if current_low <= open_price * (1 - self.sl_percent):
                    position['result'] = 'LOSS'
                    position['close_time'] = current_time
                    position['close_price'] = open_price * (1 - self.sl_percent)
                    position['close_idx'] = j
                    position['reason'] = 'SL'
                    break
            
            # Jeśli pozycja nie została zamknięta, oznacz jako timeout
            if position['result'] is None:
                position['result'] = 'LOSS'
                position['close_time'] = df.loc[len(df)-1, 'date']
                position['close_price'] = df.loc[len(df)-1, 'close']
                position['close_idx'] = len(df)-1
                position['reason'] = 'TIMEOUT'
            
            # Aktualizuj statystyki
            self.positions.append(position)
            self.total_positions += 1
            
            if position['result'] == 'WIN':
                self.winning_positions += 1
                self.current_streak = 0
            else:
                self.losing_positions += 1
                self.current_streak += 1
                self.max_losing_streak = max(self.max_losing_streak, self.current_streak)
        
        print(f"    Symulacja zakończona!")
    
    def print_results(self):
        """Wyświetla wyniki symulacji."""
        print("\n" + "="*60)
        print("WYNIKI SYMULACJI STRATEGII TRADINGOWEJ")
        print("="*60)
        
        print(f"Parametry strategii:")
        print(f"  - Take Profit: {self.tp_percent*100:.1f}%")
        print(f"  - Stop Loss: {self.sl_percent*100:.1f}%")
        print(f"  - Interwał pozycji: {self.position_interval_minutes} minut")
        
        print(f"\nStatystyki ogólne:")
        print(f"  - Całkowita liczba pozycji: {self.total_positions:,}")
        print(f"  - Pozycje wygrywające: {self.winning_positions:,} ({self.winning_positions/self.total_positions*100:.1f}%)")
        print(f"  - Pozycje przegrywające: {self.losing_positions:,} ({self.losing_positions/self.total_positions*100:.1f}%)")
        
        print(f"\nAnaliza streaków:")
        print(f"  - Najdłuższy streak przegranych: {self.max_losing_streak}")
        
        # Oblicz dodatkowe statystyki
        if self.positions:
            positions_df = pd.DataFrame(self.positions)
            
            # Średni czas trwania pozycji
            positions_df['duration_minutes'] = (positions_df['close_time'] - positions_df['open_time']).dt.total_seconds() / 60
            avg_duration = positions_df['duration_minutes'].mean()
            print(f"  - Średni czas trwania pozycji: {avg_duration:.1f} minut")
            
            # Analiza przyczyn zamknięcia
            reason_counts = positions_df['reason'].value_counts()
            print(f"\nPrzyczyny zamknięcia pozycji:")
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count:,} ({count/len(positions_df)*100:.1f}%)")
            
            # Znajdź najdłuższy streak przegranych z datami
            current_streak = 0
            max_streak_start = None
            max_streak_end = None
            
            for i, pos in enumerate(self.positions):
                if pos['result'] == 'LOSS':
                    current_streak += 1
                    if current_streak == 1:  # Początek streaku
                        streak_start = pos['open_time']
                else:
                    if current_streak > self.max_losing_streak:
                        max_streak_start = streak_start
                        max_streak_end = self.positions[i-1]['close_time']
                    current_streak = 0
            
            if max_streak_start and max_streak_end:
                print(f"\nNajdłuższy streak przegranych:")
                print(f"  - Od: {max_streak_start}")
                print(f"  - Do: {max_streak_end}")
                print(f"  - Długość: {self.max_losing_streak} pozycji")

def main():
    """Główna funkcja."""
    print("="*60)
    print("SYMULACJA STRATEGII TRADINGOWEJ")
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
    simulator = TradingSimulator(
        tp_percent=1.0,      # 1% TP
        sl_percent=0.5,       # 0.5% SL
        position_interval_minutes=5  # Co 5 minut
    )
    
    # Uruchom symulację
    simulator.simulate_strategy(df)
    
    # Wyświetl wyniki
    simulator.print_results()
    
    print("\n" + "="*60)
    print("Symulacja zakończona.")
    print("="*60)

if __name__ == "__main__":
    main() 