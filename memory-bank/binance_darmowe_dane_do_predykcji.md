# Binance – darmowe dane przydatne do predykcji cen

## Świece i ceny
- **klines (OHLCV)**: świeczki (open, high, low, close, volume). Podstawa do cech trendu i zmienności.
- **indexPriceKlines**: „cena indeksu” (średnia rynkowa, bez premii). Dobra do porównań z mark/spot.
- **markPriceKlines**: „cena rozliczeniowa” kontraktów perpetual (używana do likwidacji). Pokazuje ryzyko squeeze.
- **premiumIndexKlines**: różnica mark − index (premium) oraz stawki fundingu w formie świec. Mierzy „przegrzanie” rynku.

## Order book i transakcje
- **bookTicker**: najlepszy bid/ask i ilości (poziom L1). Do liczenia spreadu i prostego imbalance.
- **bookDepth**: głębokość arkusza w koszykach co 1% (L2, mocno zgrubne). Obraz ścian płynności.
- **trades**: każdy pojedynczy trade (tick). Pozwala mierzyć przepływ takerów i tempo transakcji.
- **aggTrades**: transakcje zgrupowane. Łatwiejsze do agregacji wolumenu i kierunku kupna/sprzedaży.
- (na żywo) **WebSocket depth 100ms**: pełny L2 w czasie rzeczywistym (brak pełnej historii za darmo — warto samemu logować).

## Futures – sentyment i dźwignia
- **Funding rate**: okresowa opłata co ~8h między longami a shortami. Skrajnie dodatni/ujemny funding bywa ostrzeżeniem przed ruchem przeciwnym.
- **Premium (mark − index)**: wysoka dodatnia/ujemna premia sugeruje przegrzanie rynku i większe ryzyko odwrócenia.
- **Open Interest (OI)**: liczba otwartych kontraktów. Wzrost = dopływ nowego kapitału („paliwo” dla trendu).
- **Taker Buy/Sell Volume**: kto agresywnie kupuje/sprzedaje. Miernik momentum i przepływu zleceń.
- **Taker Long/Short Ratio**: przewaga takerów long vs short.
- **Global Long/Short Account Ratio**: ile kont jest long/short (pozycjonowanie tłumu). Często działa kontrariańsko.
- **Top Traders Long/Short Position Ratio**: pozycje „top” kont (większe środki). Bardziej informacyjne niż średnia.
- **Liquidations**:
  - COIN‑M: zrzuty dzienne likwidacji (sumy po stronach) — snapshot.
  - USDT‑M: likwidacje przez API (strumień zdarzeń w czasie).

## Gdzie to znaleźć (darmowe)
- **Binance Vision (zrzuty dzienne)**: katalogi `futures/um` (USDT‑M) i `futures/cm` (COIN‑M):
  - `klines/`, `indexPriceKlines/`, `markPriceKlines/`, `premiumIndexKlines/`
  - `bookDepth/`, `bookTicker/`, `trades/`, `aggTrades/`
  - `metrics/` (m.in. OI, taker), oraz w `cm` także `liquidationSnapshot/`
- **Binance API (ciągłe/bieżące)**:
  - Funding: `/fapi/v1/fundingRate`, bieżące: `/fapi/v1/premiumIndex`
  - OI: `/futures/data/openInterestHist` (historia), `/fapi/v1/openInterest` (stan)
  - Taker: `/futures/data/takerBuySellVol`, `takerlongshortRatio`
  - Long/Short (global/top): `/futures/data/globalLongShortAccountRatio`, `topLongShortPositionRatio`
  - Likwidacje: `/fapi/v1/forceOrders`
  - WebSocket: strumienie `depth`, `aggTrade`, itp.

## Co dodać najpierw (najlepszy efekt vs wysiłek)
- **OI zmiana (Δ15m/Δ60m)** i z‑score — sygnał dopływu kapitału.
- **Taker buy ratio (5–15m)** — potwierdzenie kierunku (momentum).
- **Funding z‑score** + flagi „skrajny” — filtr unikania zagrań w tłumie.
- Opcjonalnie: **sumy likwidacji (60m)** i **premium (mark − index)** jako wskaźniki przegrzania.

## Uwaga praktyczna
- Serie z różnymi interwałami zresampluj do swojej siatki czasu (np. 1m), zrób „forward‑fill” między odczytami (bez backfillu), a następnie przesuń o jedną świecę, aby uniknąć wycieku informacji.