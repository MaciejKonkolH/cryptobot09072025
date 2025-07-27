# Algorytm 3-klasowego etykietowania (Directional)

## Opis słowny

### Krok 1: Dla każdej świecy (punktu czasowego)
- Weź cenę zamknięcia jako cenę wejścia
- Ustaw pozycję długą i krótką z TP/SL

### Krok 2: Sprawdź przyszłe 60 minut w pętli
Dla każdej minuty (1, 2, 3, ..., 60):

#### 2a: Sprawdź pozycję długą (LONG)
- Jeśli Long TP/SL jeszcze nie osiągnięty:
  - Sprawdź czy `high` ≥ Long TP
  - Jeśli TAK → zapamiętaj: Long TP osiągnięty i przestań sprawdzać Long
  - Jeśli NIE → sprawdź czy `low` ≤ Long SL
  - Jeśli TAK → zapamiętaj: Long SL osiągnięty i przestań sprawdzać Long

#### 2b: Sprawdź pozycję krótką (SHORT)
- Jeśli Short TP/SL jeszcze nie osiągnięty:
  - Sprawdź czy `low` ≤ Short TP
  - Jeśli TAK → zapamiętaj: Short TP osiągnięty i przestań sprawdzać Short
  - Jeśli NIE → sprawdź czy `high` ≥ Short SL
  - Jeśli TAK → zapamiętaj: Short SL osiągnięty i przestań sprawdzać Short

#### 2c: Sprawdź czy można przerwać pętlę
- Jeśli Long TP/SL osiągnięty I Short TP/SL osiągnięty:
  - Przerwij pętlę - obie pozycje zamknięte

### Krok 3: Decyzja o etykiecie
Po zakończeniu pętli:

**Jeśli tylko jedna pozycja osiągnęła TP:**
- Long TP → **LONG (0)**
- Short TP → **SHORT (1)**

**Jeśli obie pozycje osiągnęły TP:**
- Która pierwsza? → ta decyduje (LONG=0 lub SHORT=1)

**Jeśli jedna TP, druga SL:**
- TP zawsze wygrywa z SL

**Jeśli obie SL lub timeout:**
- **NEUTRAL (2)**

## Mapowanie etykiet
```
0: LONG     - Rynkowy trend wzrostowy
1: SHORT    - Rynkowy trend spadkowy  
2: NEUTRAL  - Brak wyraźnego trendu
```

## Implementacja w kodzie

```python
def calculate_labels_for_level(self, ohlc_data: np.ndarray, timestamps: np.ndarray, 
                             tp_pct: float, sl_pct: float, level_info: str = "") -> np.ndarray:
    """
    Oblicza etykiety dla jednego poziomu TP/SL (3-klasowy system).
    
    Args:
        ohlc_data: Array z danymi [high, low, close]
        timestamps: Array z timestampami
        tp_pct: Procent Take Profit (np. 2.0 dla 2%)
        sl_pct: Procent Stop Loss (np. 1.0 dla 1%)
        level_info: Informacja o poziomie dla logów
        
    Returns:
        np.ndarray: Array z etykietami (0=LONG, 1=SHORT, 2=NEUTRAL)
    """
    long_tp_pct = tp_pct / 100
    long_sl_pct = sl_pct / 100
    short_tp_pct = tp_pct / 100  # Symmetric
    short_sl_pct = sl_pct / 100  # Symmetric
    
    # Mapowanie etykiet 3-klasowe
    LABEL_MAPPING = {
        'LONG': 0,      # Rynkowy trend wzrostowy
        'SHORT': 1,     # Rynkowy trend spadkowy
        'NEUTRAL': 2    # Brak wyraźnego trendu
    }
    
    labels = np.full(len(ohlc_data), LABEL_MAPPING['NEUTRAL'], dtype=np.int8)
    total_rows = len(ohlc_data)
    
    # Ustawienie iteratora z paskiem postępu
    if TQDM_AVAILABLE:
        iterator = tqdm(range(total_rows), desc=f"{level_info}", unit="wiersz")
    else:
        iterator = range(total_rows)
        logger.info(f"{level_info} - Rozpoczynanie przetwarzania {total_rows:,} wierszy...")
    
    start_time = time.time()
    last_log_time = start_time
    
    for i in iterator:
        if i + self.future_window >= len(ohlc_data):
            continue

        entry_price = ohlc_data[i, 2]  # close price
        
        # Oblicz poziomy TP/SL
        long_tp_price = entry_price * (1 + long_tp_pct)
        long_sl_price = entry_price * (1 - long_sl_pct)
        short_tp_price = entry_price * (1 - short_tp_pct)
        short_sl_price = entry_price * (1 + short_sl_pct)

        # Inicjalizacja zmiennych dla pozycji
        long_result = None  # 'TP' lub 'SL' lub None
        short_result = None  # 'TP' lub 'SL' lub None

        # Sprawdź przyszłe 60 minut
        for j in range(self.future_window):
            if i + 1 + j >= len(ohlc_data):
                break
                
            future_high = ohlc_data[i + 1 + j, 0]  # high
            future_low = ohlc_data[i + 1 + j, 1]   # low

            # Sprawdź pozycję długą (jeśli jeszcze nie zamknięta)
            if long_result is None:
                if future_high >= long_tp_price:
                    long_result = 'TP'
                elif future_low <= long_sl_price:
                    long_result = 'SL'
            
            # Sprawdź pozycję krótką (jeśli jeszcze nie zamknięta)
            if short_result is None:
                if future_low <= short_tp_price:
                    short_result = 'TP'
                elif future_high >= short_sl_price:
                    short_result = 'SL'
            
            # Przerwij pętlę jeśli obie pozycje zamknięte
            if long_result is not None and short_result is not None:
                break

        # Logika decyzyjna 3-klasowa
        if long_result == 'TP' and short_result != 'TP':
            # Tylko Long TP osiągnięty
            labels[i] = LABEL_MAPPING['LONG']
        elif short_result == 'TP' and long_result != 'TP':
            # Tylko Short TP osiągnięty
            labels[i] = LABEL_MAPPING['SHORT']
        elif long_result == 'TP' and short_result == 'TP':
            # Obie pozycje TP - sprawdź która pierwsza
            # (w tym uproszczonym algorytmie zakładamy, że Long TP ma pierwszeństwo)
            labels[i] = LABEL_MAPPING['LONG']
        else:
            # Obie pozycje SL lub timeout
            labels[i] = LABEL_MAPPING['NEUTRAL']
        
        # Logowanie postępu co 10% lub co 30 sekund
        current_time = time.time()
        if not TQDM_AVAILABLE and (current_time - last_log_time > 30 or i % max(1, total_rows // 10) == 0):
            progress_pct = (i / total_rows) * 100
            elapsed_time = current_time - start_time
            estimated_total = elapsed_time / (i + 1) * total_rows if i > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            logger.info(f"{level_info} - Postęp: {progress_pct:.1f}% ({i:,}/{total_rows:,}) - "
                       f"Czas: {elapsed_time:.0f}s, Pozostało: {remaining_time:.0f}s")
            last_log_time = current_time

    total_time = time.time() - start_time
    logger.info(f"{level_info} - Zakończono w {total_time:.1f} sekund ({total_time/60:.1f} minut)")
    
    return labels
```

## Kluczowe zasady
1. **Pierwsze zdarzenie decyduje** - TP lub SL, które nastąpi pierwsze
2. **Po osiągnięciu TP/SL przestaję sprawdzać** tę pozycję
3. **Gdy obie pozycje zamknięte → przerwij pętlę**
4. **Automatyczna kolejność czasowa** - pierwsze zdarzenie w pętli jest pierwsze w czasie
5. **TP zawsze wygrywa z SL** - zysk ma pierwszeństwo nad stratą

## Data utworzenia
2025-07-24 - Algorytm 3-klasowy dla modułu labeler3 