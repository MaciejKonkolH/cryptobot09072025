## Trening – jakość danych, cech i modeli (playbook + dziennik)

Cel: skondensowane, praktyczne notatki oraz procedury, które pomagają budować i utrzymywać modele o wysokiej precyzji i dodatnim oczekiwanym zysku. Dokument służy jako centralny dziennik obserwacji, hipotez i eksperymentów dla różnych par walutowych.

### 1) Metryki i wskaźniki, które śledzimy
- Accuracy (3 klasy) oraz Accuracy SHORT+LONG (ważona) – globalnie i dla progów pewności 30/40/45/50%.
- Best validation mlogloss (per poziom TP/SL) i best_iteration.
- Pokrycie sygnałami o wysokiej pewności (% próbek spełniających próg).
- Precision LONG/SHORT (per próg) oraz przewidywany dochód netto per-trade: p_win*TP − (1−p_win)*SL.
- Rozkład etykiet na zbiorze testowym (udział LONG/SHORT/NEUTRAL).

### 2) Checklista jakości danych/cech (per para)
- [ ] Ciągłość minut po merge (brak urwanych zakresów).  
- [ ] Udział NaN/inf i liczba wartości zastąpionych 0.0 w cechach po finalnym fill/replace.  
- [ ] Wariancja i zakres (p0.5–p99.5) dla kluczowych cech: `OBV_slope_over_ATR`, `bb_pos_20`, `close_vs_ema_120`, `pos_in_channel_*`, `*_x_imbalance_1pct`.  
- [ ] Korelacje/Spearman cech z etykietami (czy istnieje choć słaba monotonia wobec LONG/SHORT vs NEUTRAL).  
- [ ] Dystrybucja `max_proba` na teście (czy model jest zawsze blisko 0.33–0.5 czy ma wyspy wysokiej pewności).  
- [ ] Spójność poziomów TP/SL i przyszłego okna z labelerem (sanity check).  

### 3) Obserwacje dotychczasowe (BTC vs ETH vs XRP)
- Rozkład etykiet (TP 0.8% / SL 0.2%):  
  - BTC: LONG 8%, SHORT 8%, NEUTRAL 84% (silna dominacja NEUTRAL).  
  - ETH: LONG ~18%, SHORT ~17%, NEUTRAL ~65%.  
  - XRP: LONG ~16%, SHORT ~17%, NEUTRAL ~66%.  
- Best mlogloss (wybrane przykłady):  
  - BTC wiele poziomów: ~0.75–0.87 (uczciwe dopasowanie).  
  - ETH/XRP – część poziomów zbliżona do ~0.98–1.05; dla XRP `TP=0.6, SL=0.3` zdarza się `~1.0986 @ iter 0` (≈ ln(3), brak poprawy względem baseline), co sugeruje brak użytecznego sygnału dla tego poziomu.  
- Wniosek: pipeline jest spójny, problem dotyczy jakości/skalowania wybranych cech i różnic mikrostruktury między parami (inne relacje orderbook↔cena, inna skala zmienności), co skutkuje słabszą separacją klas dla ETH/XRP na części poziomów.

### 4) Hipotezy do weryfikacji
- H1: Outliery (zwłaszcza `OBV_slope_over_ATR`) zaburzają rozdzielczość drzew mimo RobustScaler.  
- H2: Interakcje kanał×imbalance są słabsze na ETH/XRP (mniejsza płynność/inna struktura notional), przez co „szumią”.  
- H3: Część cech ma near-constancy lub niską wariancję na ETH/XRP, więc nie oferuje splitów.  
- H4: Parametry kanałów (120/180/240) są zbyt krótkie dla ETH/XRP – dłuższe okna (np. 360/480) mogą poprawić sygnał strukturalny.  
- H5: Konieczność delikatnych różnic w konfiguracji per para (lista cech, kanały, klasy-wagi, progi decyzyjne) bez „prze-tuningu”.

### 5) Plan eksperymentów (priorytety)
1. Diagnostyka cech (ETH i XRP vs BTC):
   - Raport wariancji i kwantyli (p0.5/p99.5) dla cech wskazanych wyżej.
   - Odsetek wartości 0.0 po fill/replace oraz liczba NaN/inf przed i po.  
2. Test A/B winsoryzacji: `OBV_slope_over_ATR` (np. p0.5–p99.5 per para), ew. inne skrajne cechy – ponowny trening ETH/XRP tylko z tą zmianą.  
3. Post-decisions (dla precyzji): reguły po-treningowe:  
   - `p_class ≥ t`,  
   - margin `p_class − p_NEUTRAL ≥ Δ`,  
   - ratio `p_top1/p_top2 ≥ m`.  
   Ocena wpływu na precyzję i coverage.  
4. Wagi klas: lekkie podbicie LONG/SHORT (np. 2.0→2.5) i analiza skutków (czy zwiększa się ilość sygnałów bez dużej utraty precyzji).  
5. Kanały dłuższe (360/480) dla ETH/XRP – wariant cech kanałowych i interakcji, porównanie mlogloss i metryk.  
6. Per-para lista cech (`custom_strict`) – odchudzenie o najsłabsze cechy wg ważności/diagnostyki (najpierw ocenić FI i monotonię vs etykiety).  
7. Dobór poziomów TP/SL per para – identyfikacja poziomów, na których model „coś widzi” (niższy mlogloss/wyższa precyzja przy sensownym coverage).

### 6) Protokół zapisu eksperymentów
- Id: data_czas, para, zmiana (np. winsoryzacja `OBV_slope_over_ATR` p0.5–p99.5), setup (cechy/parametry),
  wyniki główne: best mlogloss (per poziom), Accuracy SHORT+LONG (ważona) @ 40/45/50%, coverage, precision LONG/SHORT, netto per-trade.
- Wnioski i decyzja: włączamy/odrzucamy/ponawiamy z korektą.

### 7) Backlog zadań (do uporządkowania)
- [ ] Diagnostyka cech ETH/XRP vs BTC: wariancje/kwantyle/NaN-inf/zero-share (raport).  
- [ ] Winsoryzacja `OBV_slope_over_ATR` (globalnie) – test A/B (ETH, XRP).  
- [ ] Reguły po-treningowe (t, Δ, m) – wersja 1.  
- [ ] Dłuższe kanały (360/480) – wariant cech i porównanie.  
- [ ] Wagi klas LONG/SHORT – lekkie podniesienie i ocena.  
- [ ] Analiza FI i per-para selekcja cech (`custom_strict` różne listy).  
- [ ] Per-para poziomy TP/SL – wybór stabilnych poziomów.  

---

Notujmy tu kolejne obserwacje i rezultaty eksperymentów. Celem jest stabilna, wysoka precyzja przy rozsądnym pokryciu i dodatnim oczekiwanym zysku.


### 8) Zmiany wdrożone w kodzie i raportowaniu (2025-08-17)
- Ujednolicone progi pewności w raportach (Markdown/JSON): 0.30, 0.40, 0.45, 0.50.
- Naprawiona metryka "Accuracy SHORT+LONG (ważona)" – ignoruje klasy z 0 predykcji.
- Raporty wzbogacone o: listę użytych cech na początku, oraz najlepsze wyniki validation mlogloss (per poziom) wraz z iteracją.
- Dodany export tylko sygnałów LONG/SHORT (bez NEUTRAL) z flagą poprawności do osobnego CSV (`predictions_trades_*.csv`).
- Wymuszona ścisła lista cech (`FEATURE_SELECTION_MODE = 'custom_strict'`) – trening przerywa się, jeśli którejś z cech brakuje.
- Zmieniona lista cech:
  - Dodane wcześniej cechy kanałowe: `pos_in_channel_{240,180,120}`, `width_over_ATR_{240,180,120}`, `slope_over_ATR_window_{240,180,120}`, `channel_fit_score_{240,180,120}`.
  - Tymczasowo wyłączona `OBV_slope_over_ATR` (łatwy A/B zamiast natychmiastowej winsoryzacji).
  - Usunięte interakcje kanał×imbalance (`*_x_imbalance_1pct`) – redukcja szumu, zwłaszcza na ETH/XRP.

### 9) Wykonane eksperymenty i diagnostyka (2025-08-17)
- Pełny pipeline dla ETHUSDT i XRPUSDT:
  - Orderbook: brakujące dni dosztukowane (ETH: m.in. 2023-02-08/09), następnie normalizacja per-dzień → merge normalized → merge z OHLC.
  - Cechy (feature_calculator_4): 109 kolumn wejściowych; labeling (labeler5): 15 poziomów.
  - Trening (training5): zapis raportów (MD/JSON), best mlogloss per poziom, FI oraz CSV z sygnałami LONG/SHORT.
- Diagnostyka jakości cech na oknie testowym (skrypty):
  - `skrypty/analyze_feature_quality_test_window.py` – generuje per-parę raport p0.5/p99.5/STD/NaN/zero-share dla kluczowych cech.
  - `skrypty/compare_feature_quality_reports.py` – szybkie porównanie BTC vs ETH/XRP i wskazanie top-problemów.
- Wynik diagnostyki: `OBV_slope_over_ATR` na ETH/XRP ma ekstremalne ogony (p99.5, STD ≫ BTC) – potencjalnie szkodzi uczeniu (wczesne stopowanie, gorsza kalibracja). Inne kluczowe cechy (BB/EMA/kanały) wyglądają stabilnie.
- A/B 1: wyłączenie `OBV_slope_over_ATR` oraz interakcji kanał×imbalance → uzyskana odchudzona lista ~51 cech, trening uruchomiony dla BTC/ETH/XRP.

### 10) Obserwacje i wnioski operacyjne (2025-08-17)
- BTC: spójnie najniższe mlogloss na wielu poziomach (np. ~0.75–0.87), mimo silnej dominacji NEUTRAL; rośnie dokładność przy wyższych progach kosztem coverage.
- ETH: mlogloss umiarkowany; część poziomów poprawia się przy redukcji cech (potrzeba dalszych testów po winsoryzacji i z dłuższymi kanałami).
- XRP: na niektórych poziomach (np. TP=0.6/SL=0.3) brak postępu względem baseline (best_mlogloss≈ln(3) przy iter=0) – brak separacji sygnału w obecnych cechach i/lub oknach.
- Hipoteza potwierdzona częściowo: ogony `OBV_slope_over_ATR` szkodzą bardziej ETH/XRP niż BTC. Dalszy krok: winsoryzacja zamiast całkowitego wyłączenia.

### 11) Decyzje i następne kroki taktyczne
- Kontynuować A/B: (a) `OBV_slope_over_ATR` winsoryzowana p0.5–p99.5 per para vs (b) całkowicie wyłączona – porównać mlogloss i precyzję sygnałów.
- Rozszerzyć kanały dla ETH/XRP (360/480) i sprawdzić wpływ na mlogloss/coverage.
- Wdrożyć post-decisions: progi `t`, margines `Δ = p_class − p_NEUTRAL`, oraz ratio `p_top1/p_top2 ≥ m` – skupić się na wysokiej precyzji.
- Rozważyć lekkie podbicie wag klas LONG/SHORT dla ETH/XRP, jeśli recall sygnałów będzie zbyt niski przy zachowaniu precyzji.
- Ewentualna per-para lista cech w trybie `custom_strict` (najpierw analiza FI i monotoniczności względem etykiet).

### 12) Backlog – status (2025-08-17)
- [x] Diagnostyka cech ETH/XRP vs BTC: wariancje/kwantyle/NaN-inf/zero-share (raporty + porównanie).
- [ ] Winsoryzacja `OBV_slope_over_ATR` (globalnie) – test A/B (ETH, XRP).  
      Stan: tymczasowo cecha wyłączona; eksperyment winsoryzacji do wykonania.
- [ ] Reguły po-treningowe (t, Δ, m) – wersja 1 (ocena precyzji vs coverage).
- [ ] Dłuższe kanały (360/480) – wariant cech dla ETH/XRP i porównanie.
- [ ] Wagi klas LONG/SHORT – lekkie podniesienie i ocena.
- [ ] Analiza FI i per-para selekcja cech (`custom_strict` różne listy).
- [ ] Per-para poziomy TP/SL – wybór stabilnych poziomów.
