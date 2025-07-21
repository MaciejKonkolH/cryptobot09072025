Poprawiona Propozycja Algorytmu (zgodna z Twoją logiką)
Oto ostateczna wersja, która uwzględnia Twoją kluczową uwagę. Jest prostsza i bardziej wydajna.
Dla każdej świecy T:
Krok 1: Inicjalizacja
Ustaw dwie zmienne na puste:
zdarzenie_long = None
zdarzenie_short = None
Krok 2: Pętla przez Przyszłość (w poszukiwaniu pierwszych zdarzeń)
Rozpocznij pętlę po 120 przyszłych świecach. W każdej iteracji:
Sprawdź pozycję LONG (tylko jeśli zdarzenie_long jest wciąż puste):
Czy cena uderzyła w LONG TP? Jeśli tak, zapisz zdarzenie_long = ('TP', czas)
Czy cena uderzyła w LONG SL? Jeśli tak, zapisz zdarzenie_long = ('SL', czas)
Sprawdź pozycję SHORT (tylko jeśli zdarzenie_short jest wciąż puste):
Czy cena uderzyła w SHORT TP? Jeśli tak, zapisz zdarzenie_short = ('TP', czas)
Czy cena uderzyła w SHORT SL? Jeśli tak, zapisz zdarzenie_short = ('SL', czas)
Warunek końca pętli: Jeśli oba, zdarzenie_long i zdarzenie_short, nie są już puste, przerwij pętlę.
Krok 3: Podejmij Decyzję na podstawie znalezionych zdarzeń
Scenariusz 1: Nie znaleziono żadnego zdarzenia
Jeśli zdarzenie_long i zdarzenie_short są oba puste -> Etykieta: HOLD.
Scenariusz 2: Znaleziono zdarzenie tylko dla jednej pozycji
Jeśli istnieje tylko zdarzenie_long i jest to TP -> Etykieta: LONG. W przeciwnym razie -> HOLD.
Jeśli istnieje tylko zdarzenie_short i jest to TP -> Etykieta: SHORT. W przeciwnym razie -> HOLD.
Scenariusz 3: Znaleziono zdarzenia dla obu pozycji
Priorytet Take Profit:
Jeśli jedna pozycja ma TP, a druga SL -> Wygrywa ta z TP (np. zdarzenie_long=('TP', czas1), zdarzenie_short=('SL', czas2) -> etykieta LONG).
Wyścig Take Profit:
Jeśli obie mają TP, porównaj ich czasy. Ta z wcześniejszym czasem wygrywa.
Podwójny Stop Loss:
Jeśli obie mają SL -> Etykieta: HOLD.