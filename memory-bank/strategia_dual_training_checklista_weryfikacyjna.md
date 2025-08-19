### Cel
- **Potwierdzić**, że strategia (dual_labeler + dual_training) generuje dodatnią i stabilną przewagę handlową netto (po kosztach) i nie wynika z losowości lub przecieku danych.
- **Zakres**: ETHUSDT + rozszerzenie na inne pary; weryfikacja na wielu oknach czasowych i z różnymi progami pewności.

### Metryki kluczowe
- **Średni zysk na transakcję (netto)** i 95% CI
- **Coverage**: odsetek sygnałów vs. całość
- **Max DD, Sharpe, MAR** na krzywej kapitału
- **Stabilność w czasie**: tygodniowe/miesięczne PnL, equity
- **Permutation p-value** dla wyników netto na testach

### Dane i ustawienia bazowe
- Etykiety: `dual_labeler` (TP=SL, do-until, konserwatywny tie-break, wejście na kolejnej świecy)
- Poziomy: 0.6/0.8/1.0/1.2/1.4/1.7/2.0/2.5/3.0
- Progi (bazowo): 0.50 i 0.55 (osobno dla LONG/SHORT po kalibracji)
- Koszty: prowizja + poślizg (ustaw min. 0.06–0.10% round-trip; uwzględnij funding dla SHORT)

### Krok 0 — Sanity check i brak przecieków
- [ ] Sprawdź, że wszystkie cechy użyte w t na wejściu są liczone maksymalnie do t-1 (w razie potrzeby `shift(1)`).
- [ ] Etykieta startuje od t+1; wejście w pozycję następuje na otwarciu świecy t+1.
- [ ] Tie-break intrabar ustawiony na wariant konserwatywny (nie faworyzuje strategii).

### Krok 1 — Kalibracja prawdopodobieństw
- [ ] Na zbiorze walidacyjnym przeprowadź kalibrację (isotonic/Platt) dla `p_long` i `p_short` per poziom TP/SL.
- [ ] Zweryfikuj ECE/Brier i krzywe kalibracji; dopiero po tym ustal progi decyzyjne.

Przykład (ETHUSDT):
```bash
python dual_training/main.py --symbol ETHUSDT --calibrate true --save-calibration true
```

### Krok 2 — Wybór progów na walidacji (bez dotykania testu)
- [ ] Siatka progów: np. LONG ∈ {0.48, 0.50, 0.52}, SHORT ∈ {0.50, 0.52, 0.55}
- [ ] Kryterium wyboru: najwyższy średni zysk netto/trade przy minimalnym coverage (np. ≥2%) i stabilnym tygodniowym PnL (mała wariancja).
- [ ] Zamroź wybrane progi do ewaluacji na teście.

### Krok 3 — Walk-forward / Purged CV
- [ ] Podziel test na 8–12 sekwencyjnych foldów (np. tygodniowych/miesięcznych).
- [ ] Na każdym foldzie: stałe parametry z Krok 2, policz netto expectancy, DD, Sharpe.
- [ ] Uznaj stabilność, jeśli >70% foldów jest dodatnich netto i DD/Sharpe mieszczą się w akceptowalnych widełkach.

Uruchomienie (przykład):
```bash
python dual_training/analiza/walk_forward.py --symbol ETHUSDT --levels 2p0 2p5 3p0 --thr-long 0.50 --thr-short 0.55 --fees 0.10
```

### Krok 4 — Koszty transakcyjne i funding
- [ ] Dodaj prowizje i poślizg do obliczeń (zależnie od pary/segmentu; min. 0.06–0.10% RT).
- [ ] Dla SHORT dodaj estymację funding (średni dla okresu testowego; wrażliwość ±50%).
- [ ] Raportuj również wynik netto po kosztach i funding.

### Krok 5 — Permutation / Target shuffle (istotność)
- [ ] Dla konfiguracji wybranych w Krok 2 (np. 2.5% i 3.0%, thr 50–55%) wykonaj 100 permutacji z blokami (np. tygodnie) i policz p‑value dla wyników netto.
- [ ] Uznaj istotność, jeśli p ≤ 0.05 na większości poziomów.

Przykład:
```bash
python dual_training/analiza/target_shuffle_test.py --symbol ETHUSDT --levels 2p5 3p0 --thr-long 0.50 --thr-short 0.55 --n-iters 100 --block-freq W-MON --fees 0.10
```

### Krok 6 — Maksymalny czas trwania pozycji
- [ ] Wprowadź limit (np. 5–7 dni) i porównaj wyniki vs. bez limitu.
- [ ] Zaakceptuj limit, jeśli zmniejsza DD/ogony przy minimalnej utracie EV.

### Krok 7 — Test wieloparowy (robustness cross-asset)
- [ ] Wybierz min. 6–8 par z płynnością: BTCUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, MATICUSDT.
- [ ] Powtórz Kroki 1–6 dla każdej pary. Progi z Krok 2 ustal per para (nie przenoś w ciemno z ETH).
- [ ] Uznaj „generalizację”, jeśli ≥60% par spełnia kryteria akceptacji (poniżej).

Uruchomienie (batch):
```bash
python dual_training/main.py --symbol-list BTCUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT MATICUSDT --calibrate true
```

### Krok 8 — Testy wrażliwości
- [ ] Wrażliwość na progi (±0.02) i koszty (±0.05% RT).
- [ ] Asymetryczne progi: np. `thr_long=0.48`, `thr_short=0.52`.
- [ ] Zmiana tie‑break intrabar (konserwatywny vs. alternatywny) – różnica nie powinna istotnie zmieniać EV.

### Krok 9 — Raportowanie artefaktów
- [ ] Raporty MD/JSON per para/fold (metryki, koszty, CI, wykresy equity tygodniami).
- [ ] CSV z transakcjami netto; agregaty tygodniowe/miesięczne; histogramy `p_long`/`p_short` po kalibracji.
- [ ] Zestawienie zbiorcze: tabela par × poziomy × progi z EV, CI, DD, Sharpe, p‑value.

### Kryteria akceptacji (Go/No‑Go)
- [ ] Netto EV/trade > 0 na teście i w ≥70% foldów (walk‑forward) dla min. jednego poziomu (prefer. 2.5% lub 3.0%).
- [ ] Max DD akceptowalny (np. < 2× średni zysk miesięczny) i Sharpe dodatni (>0.5) na teście.
- [ ] p‑value ≤ 0.05 w permutacjach dla kluczowych konfiguracji.
- [ ] Stabilność tygodniowa: brak pojedynczych outlierów, które „robią” cały wynik.
- [ ] Multi‑pair: ≥60% par spełnia powyższe z własnymi progami.

### Skrócona checklista „do odhaczenia”
- [ ] Brak przecieków danych i poprawna chronologia (wejście na t+1)
- [ ] Kalibracja i weryfikacja ECE/Brier
- [ ] Wybór progów na walidacji (nie na teście)
- [ ] Walk‑forward z kosztami (EV, DD, Sharpe, CI)
- [ ] Permutation test (p‑value)
- [ ] Limit czasu pozycji – analiza wpływu
- [ ] Multi‑pair – konsolidacja wyników
- [ ] Raport zbiorczy z wnioskami Go/No‑Go

### Notatki
- Ekstremalne progi (np. 60%) mają bardzo mało transakcji – raportuj z szerokim CI i traktuj jako sygnały pomocnicze.
- Progi decyzyjne powinny być ustawiane po kalibracji; bez kalibracji wartości 0.50/0.55 mogą nie odpowiadać realnym prawdopodobieństwom.