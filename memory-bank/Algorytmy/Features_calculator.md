WCZYTANIE: Algorytm startuje od wczytania pliku z danymi OHLCV i wcześniej przygotowanymi etykietami.

OBLICZENIA: Na całym, pełnym zbiorze danych, wykonuje wszystkie obliczenia: najpierw tworzy kolumny z długoterminowymi średnimi kroczącymi (dla ceny i wolumenu), a następnie na ich podstawie oblicza osiem docelowych cech, które porównują aktualne wartości do poprzedniej świecy i do tych średnich.

CZYSZCZENIE: Po obliczeniu wszystkich kolumn następuje kluczowy krok – z całego zbioru danych usuwane są wszystkie wiersze, które zawierają jakąkolwiek brakującą wartość (NaN). To automatycznie eliminuje początkowy okres, dla którego nie dało się obliczyć długich średnich, gwarantując 100% kompletność pozostałych danych.

SELEKCJA I ZAPIS: Na koniec z idealnie oczyszczonych danych algorytm wybiera wyłącznie osiem kolumn z cechami oraz jedną kolumnę z etykietami. Wszystkie inne dane (surowe ceny, pośrednie średnie) są odrzucane. Wynik jest zapisywany jako finalny plik gotowy do treningu.