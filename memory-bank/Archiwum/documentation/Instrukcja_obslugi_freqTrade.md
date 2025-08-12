# Instrukcja uruchamiania bota Freqtrade z ML

## Wymagania wstępne
- Docker i docker-compose zainstalowane na systemie
- Skonfigurowany plik `config.json` w `Freqtrade/ft_bot_docker_compose/user_data/`
- Wytrenowany model ML i skaler w `Freqtrade/ft_bot_docker_compose/user_data/ml_artifacts/`
- Strategia ML: `BinanceBotSignalStrategy.py` w `Freqtrade/ft_bot_docker_compose/user_data/strategies/`

## Uruchomienie bota
1. Otwórz terminal w katalogu:
   ```
   C:\Users\macie\OneDrive\Python\Binance\Freqtrade
   ```
2. Uruchom bota poleceniem:
   ```
   docker-compose run --rm --entrypoint= freqtrade python3 -m freqtrade trade --strategy BinanceBotSignalStrategy --config /freqtrade/user_data/config.json --logfile /freqtrade/user_data/logs/freqtrade_run.log --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite --strategy-path /freqtrade/user_data/strategies/ --verbose
   ```

- Bot uruchomi się w trybie dry-run (symulacja) jeśli tak ustawiono w `config.json`.
- Logi działania znajdziesz w pliku:
  - `Freqtrade/ft_bot_docker_compose/user_data/logs/freqtrade_run.log`

## Najczęstsze problemy
- **Błąd: `NotADirectoryError: ...BinanceBotSignalStrategy.py`**
  - Upewnij się, że parametr `--strategy-path` wskazuje na katalog, a nie plik.
- **Błąd: `X has 8 features, but MinMaxScaler is expecting 10 features as input.`**
  - Usuń stare pliki modelu i skalera z `ml_artifacts/` i wytrenuj model ponownie, aby liczba cech była zgodna (8 cech).

## Zatrzymanie bota
- Wciśnij `Ctrl+C` w terminalu, aby zatrzymać bota.

---
W razie problemów sprawdź logi lub skontaktuj się z autorem projektu.
