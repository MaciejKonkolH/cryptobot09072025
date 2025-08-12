# Procedura Testowania Bota Freqtrade z Modelem ML

Data aktualizacji: 2025-05-21

## Cel

Celem tej procedury jest weryfikacja poprawnego działania bota Freqtrade, zintegrowanego ze strategią opartą o model uczenia maszynowego, ze szczególnym uwzględnieniem ładowania artefaktów ML, generowania predykcji i obsługi błędów.

## Wymagane Skrypty Pomocnicze

Przed przystąpieniem do testów upewnij się, że w katalogu `Freqtrade/scripts/` znajdują się następujące skrypty:

1.  `clear_log_file.py`: Skrypt do usuwania pliku logów Freqtrade przed każdym testem.
    *   Argumenty: `--file <ścieżka_do_pliku_logów>`
    *   Przykład: `python scripts/clear_log_file.py --file ft_bot_docker_compose/user_data/logs/freqtrade.log`
2.  `search_log.py`: Skrypt do przeszukiwania pliku logów Freqtrade w poszukiwaniu określonych wzorców (np. błędów, komunikatów ze strategii).
    *   Argumenty: `--file <ścieżka_do_pliku_logów> --query "<wzorzec_regex>" [--context <liczba_linii_kontekstu>]`
    *   Przykład: `python scripts/search_log.py --file ft_bot_docker_compose/user_data/logs/freqtrade.log --query "ERROR|No module named|BinanceBotSignalStrategy.*(Prediction|Loaded model)" --context 3`
3.  `run_bot_timed.py`: Skrypt do uruchamiania kontenera Freqtrade na określony czas, a następnie jego automatycznego zatrzymywania.
    *   Argumenty: `--duration <czas_w_sekundach>`
    *   Przykład: `python scripts/run_bot_timed.py --duration 120` (uruchamia bota na 2 minuty)

## Kroki Testowe

### 1. Przygotowanie Środowiska Docker

*   Upewnij się, że wszystkie niezbędne obrazy Docker są zbudowane (`docker compose build freqtrade`). Jeśli wprowadzono zmiany w `Dockerfile.custom`, użyj opcji `--no-cache`.
*   Upewnij się, że żadne poprzednie instancje kontenera `freqtrade` nie są uruchomione (`docker compose down -v` może być pomocne do "wyczyszczenia" środowiska).

### 2. Czyszczenie Logów

Przed każdym uruchomieniem bota, wyczyść (usuń) istniejący plik logów, aby analiza dotyczyła tylko bieżącej sesji.

```bash
python scripts/clear_log_file.py --file ft_bot_docker_compose/user_data/logs/freqtrade.log
```

### 3. Uruchomienie Bota Freqtrade na Określony Czas

Użyj skryptu `run_bot_timed.py` do uruchomienia bota. Pozwoli to na zebranie wystarczającej ilości logów bez potrzeby ręcznego zatrzymywania. Zalecany czas to co najmniej 2-5 minut, aby strategia miała szansę przetworzyć dane dla kilku świec.

```bash
python scripts/run_bot_timed.py --duration 180 # Przykład: 3 minuty
```
Skrypt automatycznie uruchomi `docker compose up -d freqtrade`, odczeka określony czas, a następnie wykona `docker compose down`.

### 4. Analiza Logów

Po zakończeniu działania bota (skrypt `run_bot_timed.py` zakończy swoje działanie), przeanalizuj logi za pomocą skryptu `search_log.py`.

**Przykładowe zapytania do analizy:**

*   **Wyszukiwanie krytycznych błędów i problemów z modułami:**
    ```bash
    python scripts/search_log.py --file ft_bot_docker_compose/user_data/logs/freqtrade.log --query "ERROR|CRITICAL|Traceback|No module named|failed to load" --context 5
    ```
*   **Weryfikacja działania strategii `BinanceBotSignalStrategy` (ładowanie modelu, skalera, generowanie predykcji):**
    ```bash
    python scripts/search_log.py --file ft_bot_docker_compose/user_data/logs/freqtrade.log --query "BinanceBotSignalStrategy.*(Loaded model|Loaded scaler|Prediction made|Signal generated|Error loading|BTC/USDT)" --context 3
    ```
*   **Sprawdzanie, czy bot przetwarza dane dla konkretnej pary (np. BTC/USDT):**
    ```bash
    python scripts/search_log.py --file ft_bot_docker_compose/user_data/logs/freqtrade.log --query "BTC/USDT" --context 2
    ```

### 5. Weryfikacja Wyników

*   **Brak krytycznych błędów:** Logi nie powinny zawierać `ERROR` (poza tymi, które są oczekiwane, np. `No module named 'sklearn'` jeśli testujemy przed jego dodaniem) ani `CRITICAL` związanych z działaniem bota czy strategii.
*   **Poprawne ładowanie artefaktów ML dla `BTC/USDT:USDT`:** Powinny pojawić się komunikaty o pomyślnym załadowaniu modelu i skalera dla tej pary.
*   **Generowanie predykcji dla `BTC/USDT:USDT`:** Strategia powinna logować informacje o wygenerowanych predykcjach.
*   **Ostrzeżenia o braku artefaktów dla innych par:** Dla par innych niż `BTC/USDT:USDT` oczekiwane są ostrzeżenia `Scaler file not found` lub `Model file not found`, ponieważ artefakty są przygotowane tylko dla BTC. To *nie* jest błąd krytyczny.
*   **Ciągłość działania:** Bot powinien regularnie logować `Bot heartbeat`.

### 6. Dokumentacja Wyników

Zanotuj wszelkie napotkane błędy, ostrzeżenia (poza oczekiwanymi) oraz obserwacje dotyczące działania strategii.

## Iteracja

W przypadku wykrycia błędów, wprowadź odpowiednie poprawki (np. w `Dockerfile.custom`, strategii, konfiguracji), a następnie powtórz procedurę testową od kroku 1 (lub 2, jeśli nie było zmian w Dockerfile).
