import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
from typing import Optional

def visualize_backtest(
    trades_file: Path, 
    candles_file: Path, 
    output_file: Path,
    validation_trades_file: Optional[Path] = None,
    start_date_str: Optional[str] = None, 
    end_date_str: Optional[str] = None
):
    """
    Tworzy interaktywny wykres świecowy z nałożonymi transakcjami z backtestu.

    Args:
        trades_file (Path): Ścieżka do pliku JSON z wynikami backtestu Freqtrade.
        candles_file (Path): Ścieżka do pliku Feather z danymi OHLCV.
        output_file (Path): Ścieżka do zapisu wyjściowego pliku HTML z wykresem.
        validation_trades_file (Optional[Path], optional): Ścieżka do pliku CSV z sygnałami walidacji. Defaults to None.
        start_date_str (Optional[str], optional): Data początkowa filtrowania ('RRRR-MM-DD'). Defaults to None.
        end_date_str (Optional[str], optional): Data końcowa filtrowania ('RRRR-MM-DD'). Defaults to None.
    """
    # --- Krok 1: Wczytanie i przygotowanie danych ---

    print(f"Wczytywanie danych o świecach z: {candles_file}")
    df = pd.read_feather(candles_file)
    
    if 'date' not in df.columns:
        df.reset_index(inplace=True)

    if 'date' not in df.columns or df['date'].isnull().all():
        raise KeyError("Brak kolumny 'date' lub wszystkie wartości są puste po resecie indeksu.")
        
    df['date'] = pd.to_datetime(df['date'], unit='ms', errors='coerce')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print(f"Zakres dat w pliku: od {df.index.min()} do {df.index.max()}")
    
    if start_date_str:
        print(f"Filtrowanie danych od: {start_date_str}")
        start_date = pd.to_datetime(start_date_str, utc=True)
        df = df[df.index >= start_date]

    if end_date_str:
        print(f"Filtrowanie danych do: {end_date_str}")
        end_date = pd.to_datetime(end_date_str, utc=True)
        df = df[df.index <= end_date]

    if df.empty:
        raise ValueError("Po zastosowaniu filtrów dat DataFrame jest pusty.")

    print(f"Wczytywanie danych o transakcjach z: {trades_file}")
    with open(trades_file, 'r') as f:
        backtest_data = json.load(f)
    
    strategy_name = list(backtest_data['strategy'].keys())[0]
    trades = backtest_data['strategy'][strategy_name]['trades']
    
    if not trades:
        print("Nie znaleziono żadnych transakcji w pliku. Zamykanie.")
        return

    # Filtruj transakcje, aby pasowały do zakresu czasowego świec
    trades = [
        t for t in trades 
        if pd.to_datetime(t['open_date']) >= df.index.min() and pd.to_datetime(t['close_date']) <= df.index.max()
    ]

    if not trades:
        print("Nie znaleziono żadnych transakcji w podanym zakresie czasowym. Zamykanie.")
        return

    # Zmiana: Dzielimy wejścia na zyskowne i stratne, aby nadać im odpowiednie kolory
    entry_long_profit = []
    entry_long_loss = []
    entry_short_profit = []
    entry_short_loss = []
    exit_points = []
    trade_lines = []  # Lista do przechowywania linii łączących transakcje

    for trade in trades:
        open_dt = pd.to_datetime(trade['open_date'])
        close_dt = pd.to_datetime(trade['close_date'])
        
        is_profit = False
        # Sprawdzenie wyniku transakcji i przypisanie do odpowiedniej listy
        if trade.get('is_short', False):
            is_profit = trade['open_rate'] > trade['close_rate']
            (entry_short_profit if is_profit else entry_short_loss).append({'date': open_dt, 'rate': trade['open_rate']})
        else:
            is_profit = trade['close_rate'] > trade['open_rate']
            (entry_long_profit if is_profit else entry_long_loss).append({'date': open_dt, 'rate': trade['open_rate']})

        line_color = 'green' if is_profit else 'red'
        
        # Wzbogacamy dane o zamknięciu o informacje z otwarcia
        exit_points.append({
            'date': close_dt, 
            'rate': trade['close_rate'],
            'open_date': trade['open_date'],
            'open_rate': trade['open_rate']
        })
        
        # Dodawanie obiektu linii do listy
        trade_lines.append(dict(
            type='line',
            x0=trade['open_date'], y0=trade['open_rate'],
            x1=trade['close_date'], y1=trade['close_rate'],
            line=dict(color=line_color, width=1, dash='dot')
        ))

    # Konwersja list na DataFrame
    df_entry_long_profit = pd.DataFrame(entry_long_profit)
    df_entry_long_loss = pd.DataFrame(entry_long_loss)
    df_entry_short_profit = pd.DataFrame(entry_short_profit)
    df_entry_short_loss = pd.DataFrame(entry_short_loss)
    df_exit = pd.DataFrame(exit_points)

    # --- Krok 2: Tworzenie wykresu ---

    print("Tworzenie wykresu...")
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Świece'
        )
    )

    # Zmiana: Rysowanie czterech rodzajów znaczników wejścia zamiast dwóch
    # Wejścia LONG z zyskiem (zielony trójkąt w górę)
    if not df_entry_long_profit.empty:
        fig.add_trace(go.Scatter(
            x=df_entry_long_profit['date'], y=df_entry_long_profit['rate'], mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12, line=dict(width=1, color='DarkSlateGrey')),
            name='Wejście LONG (Zysk)'
        ))

    # Wejścia LONG ze stratą (czerwony trójkąt w górę)
    if not df_entry_long_loss.empty:
        fig.add_trace(go.Scatter(
            x=df_entry_long_loss['date'], y=df_entry_long_loss['rate'], mode='markers',
            marker=dict(symbol='triangle-up', color='red', size=12, line=dict(width=1, color='DarkSlateGrey')),
            name='Wejście LONG (Strata)'
        ))

    # Wejścia SHORT z zyskiem (zielony trójkąt w dół)
    if not df_entry_short_profit.empty:
        fig.add_trace(go.Scatter(
            x=df_entry_short_profit['date'], y=df_entry_short_profit['rate'], mode='markers',
            marker=dict(symbol='triangle-down', color='green', size=12, line=dict(width=1, color='DarkSlateGrey')),
            name='Wejście SHORT (Zysk)'
        ))

    # Wejścia SHORT ze stratą (czerwony trójkąt w dół)
    if not df_entry_short_loss.empty:
        fig.add_trace(go.Scatter(
            x=df_entry_short_loss['date'], y=df_entry_short_loss['rate'], mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='DarkSlateGrey')),
            name='Wejście SHORT (Strata)'
        ))

    if not df_exit.empty:
        # Przygotowanie niestandardowego tekstu do podpowiedzi
        hover_texts = [
            f"<b>Exit</b><br>" +
            f"Close: {row['rate']}<br>" +
            f"Open: {pd.to_datetime(row['open_date']).strftime('%Y-%m-%d %H:%M')} @ {row['open_rate']}"
            for index, row in df_exit.iterrows()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=df_exit['date'],
                y=df_exit['rate'],
                mode='markers',
                marker=dict(
                    symbol='x',
                    color='black',
                    size=10
                ),
                name='Wyjście z pozycji',
                hoverinfo='text',
                hovertext=hover_texts
            )
        )
        
    # --- NOWA, UPROSZCZONA LOGIKA: Nanoszenie sygnałów z walidacji ---
    if validation_trades_file and validation_trades_file.exists():
        print(f"Wczytywanie sygnałów z pliku analizy: {validation_trades_file}")
        validation_df = pd.read_csv(validation_trades_file)
        # Zmiana nazwy kolumny dla spójności
        if 'predicted_signal' in validation_df.columns:
            validation_df.rename(columns={'predicted_signal': 'final_signal'}, inplace=True)
        
        # 🎯 KLUCZOWA POPRAWKA: Uproszczona i poprawna obsługa dat
        # Po prostu konwertujemy kolumnę 'timestamp' na obiekty datetime.
        # Nie wykonujemy żadnych operacji na strefach czasowych, ponieważ ufamy,
        # że dane wejściowe są już spójne (wszystkie w UTC).
        validation_df['timestamp'] = pd.to_datetime(validation_df['timestamp'])
        validation_df.set_index('timestamp', inplace=True)

        # Filtrowanie, aby pasowały do zakresu wykresu
        validation_df = validation_df[validation_df.index.isin(df.index)]

        # --- Podział sygnałów na poprawne i błędne ---
        correct_signals = validation_df[validation_df['is_correct'] == True]
        incorrect_signals = validation_df[validation_df['is_correct'] == False]

        # --- Rysowanie sygnałów POPRAWNYCH (Lazurowy) ---
        # Correct SHORT
        correct_short = correct_signals[correct_signals['final_signal'] == 0]
        if not correct_short.empty:
            y_values = correct_short.index.map(df['high'])
            fig.add_trace(go.Scatter(
                x=correct_short.index, y=y_values, mode='markers',
                marker=dict(symbol='triangle-down', color='turquoise', size=9, line=dict(width=1, color='DarkSlateGrey')),
                name='Correct SHORT'
            ))
        # Correct LONG
        correct_long = correct_signals[correct_signals['final_signal'] == 2]
        if not correct_long.empty:
            y_values = correct_long.index.map(df['low'])
            fig.add_trace(go.Scatter(
                x=correct_long.index, y=y_values, mode='markers',
                marker=dict(symbol='triangle-up', color='turquoise', size=9, line=dict(width=1, color='DarkSlateGrey')),
                name='Correct LONG'
            ))

        # --- Rysowanie sygnałów BŁĘDNYCH (Żółty) ---
        # Incorrect SHORT
        incorrect_short = incorrect_signals[incorrect_signals['final_signal'] == 0]
        if not incorrect_short.empty:
            y_values = incorrect_short.index.map(df['high'])
            fig.add_trace(go.Scatter(
                x=incorrect_short.index, y=y_values, mode='markers',
                marker=dict(symbol='triangle-down', color='yellow', size=7, line=dict(width=1, color='DarkSlateGrey')),
                name='Incorrect SHORT'
            ))
        # Incorrect LONG
        incorrect_long = incorrect_signals[incorrect_signals['final_signal'] == 2]
        if not incorrect_long.empty:
            y_values = incorrect_long.index.map(df['low'])
            fig.add_trace(go.Scatter(
                x=incorrect_long.index, y=y_values, mode='markers',
                marker=dict(symbol='triangle-up', color='yellow', size=7, line=dict(width=1, color='DarkSlateGrey')),
                name='Incorrect LONG'
            ))

        print(f"Dodano {len(correct_short) + len(correct_long)} poprawnych i {len(incorrect_short) + len(incorrect_long)} błędnych sygnałów z walidacji.")

    # --- Krok 3: Konfiguracja wyglądu i zapis ---

    fig.update_layout(
        title=f'Wizualizacja Backtestu: {strategy_name} ({start_date_str} - {end_date_str})',
        xaxis_title='Data',
        yaxis_title='Cena',
        xaxis_rangeslider_visible=False,
        # Konfiguracja siatki osi Y
        yaxis=dict(
            dtick=250,  # Odstęp co 250 jednostek
            gridcolor='LightGray',  # Kolor linii
            gridwidth=1  # Grubość linii
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Dodanie linii łączących transakcje
        shapes=trade_lines
    )

    print(f"Zapisywanie wykresu do pliku: {output_file}")
    fig.write_html(output_file)
    print("Zakończono pomyślnie!")


if __name__ == '__main__':
    # --- Konfiguracja Ścieżek ---
    CANDLES_FEATHER_FILE = Path('ft_bot_clean/user_data/strategies/inputs/BTC_USDT_USDT/BTCUSDT-1m-futures.feather')
    OUTPUT_HTML_FILE = Path(__file__).parent / 'backtest_plot_with_validation.html'

    # --- Automatyczne Wyszukiwanie Najnowszego Pliku z Wynikami Backtestu ---
    backtest_results_dir = Path('ft_bot_clean/user_data/backtest_results')
    TRADES_JSON_FILE = None
    
    # 🎯 NOWA, ODPORNA NA BŁĘDY LOGIKA WYSZUKIWANIA PLIKU
    # Krok 1: Zbierz wszystkie pasujące katalogi do listy
    all_backtest_dirs = [
        p for p in backtest_results_dir.iterdir() 
        if p.is_dir() and p.name.startswith('backtest-result-')
    ]

    # Krok 2: Jeśli lista nie jest pusta, znajdź najnowszy katalog
    if all_backtest_dirs:
        latest_backtest_dir = max(all_backtest_dirs, key=lambda p: p.stat().st_mtime)
        json_file_name = f"{latest_backtest_dir.name}.json"
        TRADES_JSON_FILE = latest_backtest_dir / json_file_name
        print(f"✅ Znaleziono i wybrano najnowszy plik backtestu: {TRADES_JSON_FILE}")
    else:
        # Ten komunikat pojawi się tylko wtedy, gdy folder jest FAKTYCZNIE pusty
        print(f"⚠️ OSTRZEŻENIE: Nie znaleziono żadnych katalogów 'backtest-result-*' w '{backtest_results_dir}'.")

    # --- Automatyczne Wyszukiwanie Najnowszego Pliku Analizy Walidacji ---
    validation_dir = Path('Kaggle/output')
    VALIDATION_TRADES_CSV = None
    try:
        # Znajdź najnowszy plik na podstawie czasu modyfikacji
        latest_validation_file = max(
            validation_dir.glob('validation_analysis_BTCUSDT_*.csv'), 
            key=lambda p: p.stat().st_mtime
        )
        VALIDATION_TRADES_CSV = latest_validation_file
        print(f"✅ Znaleziono i wybrano najnowszy plik walidacji: {VALIDATION_TRADES_CSV.name}")
    except ValueError:
        # Ten błąd wystąpi, jeśli glob nie znajdzie żadnych pasujących plików
        print(f"⚠️ OSTRZEŻENIE: Nie znaleziono plików 'validation_analysis...' w katalogu '{validation_dir}'.")
        print("Wykres zostanie wygenerowany bez sygnałów z walidacji.")

    # --- Uruchomienie Wizualizacji ---
    if not TRADES_JSON_FILE or not TRADES_JSON_FILE.exists():
        print(f"BŁĄD: Plik transakcji nie istnieje lub nie został znaleziony w '{backtest_results_dir}'.")
    elif not CANDLES_FEATHER_FILE.exists():
        print(f"BŁĄD: Plik danych historycznych nie istnieje: {CANDLES_FEATHER_FILE}")
    else:
        # --- Konfiguracja Zakresu Czasowego ---
        # Ustaw daty, aby filtrować dane. Użyj formatu 'RRRR-MM-DD'.
        # Ustaw na None, aby wyłączyć filtrowanie dla danej daty.
        START_DATE = '2024-12-20'
        END_DATE = '2024-12-21'

        visualize_backtest(
            trades_file=TRADES_JSON_FILE, 
            candles_file=CANDLES_FEATHER_FILE, 
            output_file=OUTPUT_HTML_FILE,
            validation_trades_file=VALIDATION_TRADES_CSV, # Może być None, funkcja to obsłuży
            start_date_str=START_DATE,
            end_date_str=END_DATE
        ) 