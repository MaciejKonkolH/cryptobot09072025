import pandas as pd
from datetime import datetime, timedelta

def analyze_concurrent_positions(csv_file):
    """Analizuje maksymalną liczbę jednoczesnych pozycji na podstawie pliku CSV z predykcjami"""
    
    # Wczytaj dane
    df = pd.read_csv(csv_file)
    
    # Konwertuj timestamp na datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filtruj tylko sygnały long i short (które generują transakcje)
    trades = df[df['signal'].isin(['long', 'short'])].copy()
    
    print(f"📊 ANALIZA JEDNOCZESNYCH POZYCJI")
    print(f"=" * 50)
    print(f"Całkowita liczba sygnałów: {len(df):,}")
    print(f"Sygnały LONG: {len(trades[trades['signal'] == 'long']):,}")
    print(f"Sygnały SHORT: {len(trades[trades['signal'] == 'short']):,}")
    print(f"Sygnały NEUTRAL: {len(df[df['signal'] == 'neutral']):,}")
    print(f"Łączne sygnały transakcji: {len(trades):,}")
    print()
    
    if len(trades) == 0:
        print("❌ Brak sygnałów transakcji do analizy")
        return
    
    # Sortuj według czasu
    trades = trades.sort_values('timestamp')
    
    # Załóżmy średni czas trwania pozycji na podstawie raportu: 1:32:00 = 92 minuty
    avg_duration_minutes = 92
    
    # Dla każdego sygnału, oblicz czas zamknięcia pozycji
    trades['close_time'] = trades['timestamp'] + timedelta(minutes=avg_duration_minutes)
    
    # Znajdź wszystkie unikalne momenty czasowe (otwarcie i zamknięcie pozycji)
    events = []
    
    for _, trade in trades.iterrows():
        events.append({'time': trade['timestamp'], 'type': 'open', 'signal': trade['signal']})
        events.append({'time': trade['close_time'], 'type': 'close', 'signal': trade['signal']})
    
    # Sortuj wydarzenia według czasu
    events_df = pd.DataFrame(events).sort_values('time')
    
    # Oblicz liczbę otwartych pozycji w każdym momencie
    open_positions = 0
    max_positions = 0
    max_time = None
    position_history = []
    
    for _, event in events_df.iterrows():
        if event['type'] == 'open':
            open_positions += 1
        else:
            open_positions -= 1
        
        if open_positions > max_positions:
            max_positions = open_positions
            max_time = event['time']
        
        position_history.append({
            'time': event['time'],
            'open_positions': open_positions,
            'event': f"{event['type']} {event['signal']}"
        })
    
    print(f"🎯 WYNIKI ANALIZY:")
    print(f"Maksymalna liczba jednoczesnych pozycji: {max_positions}")
    print(f"Moment maksymalnej liczby pozycji: {max_time}")
    print(f"Założony średni czas trwania pozycji: {avg_duration_minutes} minut")
    print()
    
    # Pokaż próbkę historii pozycji
    print(f"📈 PRÓBKA HISTORII POZYCJI (pierwsze 10 wydarzeń):")
    for i, pos in enumerate(position_history[:10]):
        print(f"{pos['time'].strftime('%Y-%m-%d %H:%M')} | {pos['open_positions']:2d} pozycji | {pos['event']}")
    
    if len(position_history) > 10:
        print("...")
        print(f"Pokazano 10 z {len(position_history)} wydarzeń")
    
    print()
    print(f"🔍 PORÓWNANIE Z RAPORTEM FREQTRADE:")
    print(f"Raport FreqTrade: Max open trades = 1")
    print(f"Rzeczywista analiza: Max open trades = {max_positions}")
    print(f"Różnica: {max_positions - 1} pozycji więcej niż pokazuje raport")
    
    return max_positions

if __name__ == "__main__":
    csv_file = "user_data/backtest_results/predictions_BTCUSDTUSDT_20250730_113943.csv"
    analyze_concurrent_positions(csv_file) 