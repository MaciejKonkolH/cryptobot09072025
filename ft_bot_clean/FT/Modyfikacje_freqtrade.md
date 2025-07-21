# Modyfikacje FreqTrade dla Position Stacking

## Cel modyfikacji
Umożliwienie wielu pozycji na jednej parze (position stacking) zamiast standardowego ograniczenia "jedna pozycja per para".

## Status implementacji
⚠️ **UWAGA**: Modyfikacje zostały wykonane w katalogu `FT/freqtrade-develop/` ale **NIE SĄ AKTYWNE** ponieważ FreqTrade uruchamia kod z `venv_new/Lib/site-packages/freqtrade/`.

## Wymagane modyfikacje

### 1. freqtradebot.py (linie 612-628)
**Lokalizacja**: `freqtrade/freqtradebot.py`
**Funkcja**: `enter_positions()`

**Oryginalny kod**:
```python
# Original logic - remove pairs for currently opened trades from the whitelist
for trade in Trade.get_open_trades():
    if trade.pair in whitelist:
        whitelist.remove(trade.pair)
        logger.debug("Ignoring %s in pair whitelist", trade.pair)
```

**Zmodyfikowany kod**:
```python
# Handle position stacking logic
if self.config.get("position_stacking", False):
    max_positions_per_pair = self.config.get("max_positions_per_pair", 1)
    # Count positions per pair
    pair_position_count = {}
    for trade in Trade.get_open_trades():
        pair_position_count[trade.pair] = pair_position_count.get(trade.pair, 0) + 1
    
    # Remove pairs that have reached max positions per pair
    pairs_to_remove = []
    for pair in whitelist:
        if pair_position_count.get(pair, 0) >= max_positions_per_pair:
            pairs_to_remove.append(pair)
            logger.debug("Ignoring %s in pair whitelist - max positions (%d) reached", 
                       pair, max_positions_per_pair)
    
    for pair in pairs_to_remove:
        whitelist.remove(pair)
else:
    # Original logic - remove pairs for currently opened trades from the whitelist
    for trade in Trade.get_open_trades():
        if trade.pair in whitelist:
            whitelist.remove(trade.pair)
            logger.debug("Ignoring %s in pair whitelist", trade.pair)
```

### 2. wallets.py (linie 383-393)
**Lokalizacja**: `freqtrade/wallets.py`
**Funkcja**: `get_trade_stake_amount()`

**Oryginalny kod**:
```python
if stake_amount == UNLIMITED_STAKE_AMOUNT:
    stake_amount = self._calculate_unlimited_stake_amount(
        available_amount, val_tied_up, max_open_trades
    )
```

**Zmodyfikowany kod**:
```python
if stake_amount == UNLIMITED_STAKE_AMOUNT:
    # For position stacking, adjust max_open_trades calculation
    if self._config.get("position_stacking", False):
        max_positions_per_pair = self._config.get("max_positions_per_pair", 1)
        pair_count = len(self._config.get("exchange", {}).get("pair_whitelist", [pair]))
        effective_max_trades = max_open_trades * max_positions_per_pair / pair_count if pair_count > 0 else max_open_trades
        stake_amount = self._calculate_unlimited_stake_amount(
            available_amount, val_tied_up, effective_max_trades
        )
    else:
        stake_amount = self._calculate_unlimited_stake_amount(
            available_amount, val_tied_up, max_open_trades
        )
```

### 3. optimize_reports.py (linie 463-467)
**Lokalizacja**: `freqtrade/optimize/optimize_reports/optimize_reports.py`
**Funkcja**: `generate_strategy_stats()`

**Oryginalny kod**:
```python
max_open_trades = min(config["max_open_trades"], len(pairlist))
```

**Zmodyfikowany kod**:
```python
# For position stacking, adjust max_open_trades calculation
if config.get("position_stacking", False):
    max_positions_per_pair = config.get("max_positions_per_pair", 1)
    max_open_trades = min(config["max_open_trades"], len(pairlist) * max_positions_per_pair)
else:
    max_open_trades = min(config["max_open_trades"], len(pairlist))
```

## Konfiguracja

### Wymagane parametry w config.json:
```json
{
    "position_stacking": true,
    "max_positions_per_pair": 5,
    "max_open_trades": 100,
    "exchange": {
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT", 
            "ADA/USDT:USDT",
            "SOL/USDT:USDT",
            "MATIC/USDT:USDT"
        ]
    }
}
```

### Wyjaśnienie parametrów:
- `position_stacking`: Włącza/wyłącza funkcjonalność
- `max_positions_per_pair`: Maksymalna liczba pozycji na jednej parze
- `max_open_trades`: Ogólny limit otwartych pozycji
- `pair_whitelist`: Lista par (musi być wystarczająco długa)

## Logika działania

### Bez position stacking (oryginalny FreqTrade):
- 1 para = maksymalnie 1 pozycja
- Para jest usuwana z whitelist gdy ma otwartą pozycję

### Z position stacking:
- 1 para = maksymalnie `max_positions_per_pair` pozycji
- Para jest usuwana z whitelist dopiero gdy osiągnie maksymalną liczbę pozycji

### Obliczanie max_open_trades:
- **Bez stacking**: `min(config_max_trades, liczba_par)`
- **Ze stackingiem**: `min(config_max_trades, liczba_par × max_positions_per_pair)`

## Przykład:
- 5 par w whitelist
- `max_positions_per_pair = 5`
- `max_open_trades = 100`
- **Wynik**: Maksymalnie 25 pozycji (5 par × 5 pozycji per para)

## Problem z implementacją
❌ **Modyfikacje nie są aktywne** ponieważ:
1. Kod zmodyfikowano w `FT/freqtrade-develop/`
2. FreqTrade uruchamia kod z `venv_new/Lib/site-packages/freqtrade/`
3. Wymagana ponowna instalacja lub kopiowanie modyfikacji

## Rozwiązanie
```bash
# Opcja 1: Ponowna instalacja editable
cd FT/freqtrade-develop
pip uninstall freqtrade -y
pip install -e .

# Opcja 2: Kopiowanie modyfikacji do site-packages
# (skopiować zmodyfikowane pliki do venv_new/Lib/site-packages/freqtrade/)
```

## Testowanie
Po poprawnej instalacji modyfikacji:
```bash
freqtrade backtesting -c config_backtest.json -s Strategy --timerange 20250617-20250618
```

Oczekiwany wynik w logach:
```
Max open trades : 25  # (zamiast 1)
```
