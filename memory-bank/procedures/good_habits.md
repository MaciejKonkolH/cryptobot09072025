# Dobre Praktyki i Oczekiwania Wobec Asystenta AI

Data utworzenia: 2025-05-21

Niniejszy dokument zawiera zbiór oczekiwanych zachowań i dobrych praktyk, którymi powinien kierować się asystent AI podczas współpracy nad projektem.

## Komunikacja i Autonomia

1.  **Samodzielność w Działaniu:** Asystent powinien dążyć do samodzielnego rozwiązywania problemów i wykonywania zadań. Jeśli ma dostęp do niezbędnych narzędzi i informacji, powinien podejmować działania bez oczekiwania na bezpośrednie polecenie wykonania każdego kroku.
2.  **Proaktywność:** W przypadku napotkania problemów lub niejasności, których nie może rozwiązać samodzielnie, asystent powinien jasno komunikować problem i proponować możliwe rozwiązania lub pytać o niezbędne informacje.
3.  **Komunikacja w Języku Polskim:** Wszystkie odpowiedzi i komunikaty powinny być formułowane w języku polskim, chyba że użytkownik wyraźnie zaznaczy inaczej.

## Wykonywanie Poleceń i Zarządzanie Zadaniami

1.  **Zadania w Tle:** Jeśli użytkownik przeniesie wykonanie polecenia (np. długotrwałego skryptu, budowania obrazu Docker) do działania w tle, asystent powinien **wstrzymać się z podejmowaniem kolejnych kroków i czekać na dalsze instrukcje od użytkownika**. Nie należy zakładać, że zadanie w tle zakończyło się pomyślnie ani kontynuować pracy nad kolejnymi etapami bez wyraźnego polecenia.

## Praca z Plikami i Kodem

1.  **Dostęp do Plików:** Asystent powinien pamiętać, że ma dostęp do plików w przestrzeni roboczej i wykorzystywać tę możliwość do odczytu konfiguracji, analizy kodu źródłowego czy modyfikacji plików zgodnie z poleceniami, zamiast prosić użytkownika o manualne wykonanie tych czynności.
2.  **Testowanie Zmian:** Po wprowadzeniu zmian w konfiguracji, kodzie strategii, Dockerfile itp., asystent powinien przeprowadzić odpowiednie kroki testowe (np. przebudowanie obrazu, restart kontenera, analiza logów) w celu weryfikacji poprawności wprowadzonych zmian.

## Organizacja Plików i Struktura Projektu

1.  **Plany:** Wszystkie pliki z planami należy tworzyć w lokalizacji `memory-bank/plans/`
2.  **Analizy:** Analizy zapisywać w lokalizacji `memory-bank/analyses/`
3.  **Notatki:** Notatki umieszczać w lokalizacji `memory-bank/notes/`
4.  **Zakaz Tworzenia Plików w Katalogu Głównym:** Nie należy tworzyć plików w katalogu głównym projektu, chyba że są ku temu wyraźne przesłanki.
5.  **Skrypty Testowe i Robocze:**
    - Standardowe skrypty: `scripts/`
    - Skrypty widoczne z poziomu Dockera: `ft_bot_docker_compose/user_data/scripts/`
6.  **Testy:**
    - Standardowe testy: `tests/`
    - Testy widoczne z poziomu Dockera: `ft_bot_docker_compose/user_data/tests/`

## Analiza Logów i Diagnoza Problemów

1.  **Dokładna Analiza:** Przy analizie logów, asystent powinien dokładnie czytać komunikaty, zwracając uwagę na błędy (`ERROR`, `CRITICAL`, `Traceback`) oraz istotne informacje (`INFO`, `WARNING`) pochodzące ze strategii lub kluczowych komponentów systemu.
2.  **Korzystanie z Narzędzi:** Do analizy dużych plików logów należy wykorzystywać dostępne narzędzia, takie jak dedykowane skrypty przeszukujące (np. `search_log.py`).

## Ogólne Zasady

1.  **Zapamiętywanie Kontekstu:** Asystent powinien starać się pamiętać wcześniejsze ustalenia, problemy i ich rozwiązania, aby unikać powtarzania tych samych błędów lub zadawania tych samych pytań.
2.  **Adaptacja do Procedur:** Należy stosować się do zdefiniowanych procedur (np. `bot_testing.md`) i aktualizować je w razie potrzeby.

*(Dokument będzie rozwijany w miarę postępów prac i pojawiania się nowych ustaleń.)*
