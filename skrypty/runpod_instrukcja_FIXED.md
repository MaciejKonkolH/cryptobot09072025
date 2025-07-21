# 🎯 RUNPOD BACKTEST - INSTRUKCJA NAPRAWIONA

## Problem
Freqtrade na RunPod nie znajdowało danych historycznych dla BTC/USDT:USDT mimo że pliki istnieją.

## Rozwiązanie
Dodanie flagi `--datadir` do komendy backtestingu.

## Poprawiona komenda backtestingu

```bash
freqtrade backtesting \
  --config user_data/config.json \
  --strategy Enhanced_ML_MA43200_Buffer_Strategy \
  --timerange 20240101-20240102 \
  --dry-run-wallet 1000 \
  --datadir user_data/strategies/inputs
```

## Kluczowa różnica

**PRZED (nie działało):**
```bash
freqtrade backtesting --config user_data/config.json --strategy Enhanced_ML_MA43200_Buffer_Strategy --timerange 20240101-20240102 --dry-run-wallet 1000
```

**PO (powinno działać):**
```bash
freqtrade backtesting --config user_data/config.json --strategy Enhanced_ML_MA43200_Buffer_Strategy --timerange 20240101-20240102 --dry-run-wallet 1000 --datadir user_data/strategies/inputs
```

## Wyjaśnienie

- Flaga `--datadir` mówi Freqtrade gdzie szukać plików z danymi historycznymi
- Bez tej flagi Freqtrade szuka w domyślnej lokalizacji `user_data/data/`
- Nasze dane są w `user_data/strategies/inputs/BTC_USDT_USDT/raw_validated.feather`
- Dzięki `--datadir user_data/strategies/inputs` Freqtrade znajdzie nasze dane

## Struktura plików na RunPod

```
/workspace/crypto/
├── user_data/
│   ├── config.json
│   └── strategies/
│       ├── Enhanced_ML_MA43200_Buffer_Strategy.py
│       └── inputs/
│           └── BTC_USDT_USDT/
│               └── raw_validated.feather  # <-- TUTAJ SĄ NASZE DANE
└── freqtrade backtesting --datadir user_data/strategies/inputs
```

## Testowy skrypt

Użyj `runpod_backtest_fixed.py` aby przetestować rozwiązanie.

## Dlaczego to działa lokalnie a nie na RunPod?

Lokalnie prawdopodobnie mamy dane w kilku miejscach lub różną konfigurację ścieżek. RunPod ma "czystą" instalację więc potrzebuje explicite wskazać gdzie są dane. 