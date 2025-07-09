#!/usr/bin/env python3
"""
Export FreqTrade Backtest Trades to CSV
=======================================
Eksportuje wszystkie transakcje z backtestingu FreqTrade do pliku CSV.

Użycie: python export_trades.py
"""

import json
import zipfile
import pandas as pd
from pathlib import Path
from datetime import datetime

def export_trades_to_csv():
    print("🚀 FreqTrade Trades Exporter")
    print("=" * 35)
    
    try:
        # Znajdź najnowszy plik ZIP z backtestingu
        backtest_dir = Path('user_data/backtest_results')
        
        if not backtest_dir.exists():
            print("❌ Katalog user_data/backtest_results nie istnieje!")
            return
        
        zip_files = list(backtest_dir.glob('*.zip'))
        
        if not zip_files:
            print("❌ Brak plików ZIP w katalogu backtest_results!")
            return
        
        # Wybierz najnowszy plik
        latest_zip = max(zip_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 Przetwarzam: {latest_zip.name}")
        
        # Otwórz plik ZIP i znajdź JSON z wynikami
        with zipfile.ZipFile(latest_zip, 'r') as zip_file:
            files = zip_file.namelist()
            
            # Znajdź główny plik JSON (nie config)
            json_file = None
            for file in files:
                if file.endswith('.json') and 'config' not in file:
                    json_file = file
                    break
            
            if not json_file:
                print("❌ Nie znaleziono pliku JSON w archiwum!")
                return
            
            print(f"📊 Ładuję dane z: {json_file}")
            
            # Załaduj dane JSON
            with zip_file.open(json_file) as jf:
                data = json.load(jf)
            
            # Sprawdź strukturę danych - rzeczywiste transakcje są w strategy
            if 'strategy' not in data:
                print("❌ Brak danych strategy w pliku JSON!")
                return
            
            # Znajdź pierwszą strategię
            strategy_names = list(data['strategy'].keys())
            if not strategy_names:
                print("❌ Brak strategii w danych!")
                return
            
            strategy_name = strategy_names[0]
            print(f"📋 Strategia: {strategy_name}")
            
            # Pobierz transakcje z strategy
            strategy_data = data['strategy'][strategy_name]
            
            if 'trades' not in strategy_data:
                print("❌ Brak transakcji w danych strategii!")
                return
            
            trades = strategy_data['trades']
            
            if not trades or len(trades) == 0:
                print("❌ Lista transakcji jest pusta!")
                return
            
            print(f"✅ Znaleziono {len(trades)} transakcji")
            
            # Konwertuj do DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Wygeneruj nazwę pliku CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"backtest_trades_{timestamp}.csv"
            
            # Zapisz do CSV
            trades_df.to_csv(csv_filename, index=False)
            
            print(f"💾 Transakcje zapisane do: {csv_filename}")
            print(f"📋 Kolumny: {len(trades_df.columns)}")
            print(f"📊 Wiersze: {len(trades_df)}")
            
            # Pokaż podstawowe statystyki
            if 'profit_abs' in trades_df.columns:
                total_profit = trades_df['profit_abs'].sum()
                winning_trades = (trades_df['profit_abs'] > 0).sum()
                losing_trades = (trades_df['profit_abs'] < 0).sum()
                win_rate = (winning_trades / len(trades_df)) * 100
                
                print(f"\n📈 STATYSTYKI:")
                print(f"   💰 Całkowity zysk: {total_profit:.2f} USDT")
                print(f"   ✅ Zyskowne transakcje: {winning_trades}")
                print(f"   ❌ Stratne transakcje: {losing_trades}")
                print(f"   📊 Wskaźnik wygranych: {win_rate:.1f}%")
                
                if winning_trades > 0:
                    avg_win = trades_df[trades_df['profit_abs'] > 0]['profit_abs'].mean()
                    print(f"   📈 Średni zysk: {avg_win:.2f} USDT")
                
                if losing_trades > 0:
                    avg_loss = trades_df[trades_df['profit_abs'] < 0]['profit_abs'].mean()
                    print(f"   📉 Średnia strata: {avg_loss:.2f} USDT")
            
            # Pokaż nazwy wszystkich kolumn
            print(f"\n📋 DOSTĘPNE KOLUMNY:")
            for i, col in enumerate(trades_df.columns, 1):
                print(f"   {i:2d}. {col}")
            
            # Pokaż przykładowe dane
            print(f"\n📊 PRZYKŁADOWE TRANSAKCJE (5 pierwszych):")
            display_cols = ['pair', 'profit_abs', 'profit_ratio', 'trade_duration']
            available_cols = [col for col in display_cols if col in trades_df.columns]
            
            if available_cols:
                print(trades_df[available_cols].head().to_string())
            else:
                print(trades_df.head())
            
            print(f"\n✅ SUKCES! Plik {csv_filename} został utworzony.")
            
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_trades_to_csv() 