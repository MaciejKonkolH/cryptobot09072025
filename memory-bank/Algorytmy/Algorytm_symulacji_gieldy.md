COMPETITIVE LABELING - RZECZYWISTA SYMULACJA JEDNOCZESNYCH POZYCJI
├── DLA każdego punktu czasowego t:
│   ├── SPRAWDZENIE DOSTĘPNOŚCI DANYCH
│   │   ├── Czy istnieje 120 minut danych do przodu?
│   │   ├── JEŚLI NIE → PRZYPISZ label = 1 (HOLD)
│   │   └── JEŚLI TAK → KONTYNUUJ symulację
│   │
│   ├── OTWARCIE POZYCJI (JEDNOCZEŚNIE)
│   │   ├── Entry_price = close[t]
│   │   ├── LONG_TP = Entry_price × (1 + LONG_TP_PCT/100)
│   │   ├── LONG_SL = Entry_price × (1 - LONG_SL_PCT/100)  
│   │   ├── SHORT_TP = Entry_price × (1 - SHORT_TP_PCT/100)
│   │   ├── SHORT_SL = Entry_price × (1 + SHORT_SL_PCT/100)
│   │   ├── long_active = True
│   │   └── short_active = True
│   │
│   ├── PĘTLA PRZEZ 120 ŚWIEC [t+1, t+120]
│   │   ├── DLA każdej świecy i:
│   │   │   ├── SPRAWDŹ WSZYSTKIE ZDARZENIA W KOLEJNOŚCI:
│   │   │   │
│   │   │   ├── JEŚLI long_active AND high[i] >= LONG_TP:
│   │   │   │   ├── label[t] = 2 (LONG)
│   │   │   │   └── PRZERWIJ całą pętlę (LONG wygrywa)
│   │   │   │
│   │   │   ├── JEŚLI short_active AND low[i] <= SHORT_TP:
│   │   │   │   ├── label[t] = 0 (SHORT)  
│   │   │   │   └── PRZERWIJ całą pętlę (SHORT wygrywa)
│   │   │   │
│   │   │   ├── JEŚLI long_active AND low[i] <= LONG_SL:
│   │   │   │   ├── long_active = False (zamykamy pozycję LONG)
│   │   │   │   └── KONTYNUUJ obserwację SHORT
│   │   │   │
│   │   │   ├── JEŚLI short_active AND high[i] >= SHORT_SL:
│   │   │   │   ├── short_active = False (zamykamy pozycję SHORT)
│   │   │   │   └── KONTYNUUJ obserwację LONG
│   │   │   │
│   │   │   └── SPRAWDŹ status pozycji:
│   │   │       ├── JEŚLI long_active == False AND short_active == False:
│   │   │       │   ├── label[t] = 1 (HOLD) - obie na SL
│   │   │       │   └── PRZERWIJ pętlę
│   │   │       └── KONTYNUUJ do następnej świecy
│   │   │
│   │   └── JEŚLI koniec pętli bez TP:
│   │       └── label[t] = 1 (HOLD) - brak zdarzenia lub tylko SL
│   │
│   └── KONTYNUUJ do następnego punktu t+1
│
└── ZWRÓĆ kompletną kolumnę labels[0...n]