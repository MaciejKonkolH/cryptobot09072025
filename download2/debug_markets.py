#!/usr/bin/env python3
"""
Debug script - sprawdza dostępne pary na Binance Futures
"""

import ccxt

def debug_markets():
    exchange = ccxt.binanceusdm({
        'timeout': 30000,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })
    
    try:
        markets = exchange.load_markets()
        
        print(f"Łącznie par: {len(markets)}")
        
        # Kategorie par
        futures_pairs = []
        spot_pairs = []
        other_pairs = []
        
        for symbol, market in markets.items():
            if market['type'] == 'future':
                futures_pairs.append(symbol)
            elif market['type'] == 'spot':
                spot_pairs.append(symbol)
            else:
                other_pairs.append(symbol)
        
        print(f"\nFutures pairs: {len(futures_pairs)}")
        print(f"Spot pairs: {len(spot_pairs)}")
        print(f"Other pairs: {len(other_pairs)}")
        
        print(f"\nPrzykłady futures pairs:")
        for i, pair in enumerate(futures_pairs[:20], 1):
            print(f"  {i:2d}. {pair}")
        
        if len(futures_pairs) > 20:
            print(f"  ... i {len(futures_pairs) - 20} więcej")
        
        # Sprawdź strukturę przykładowej pary
        if futures_pairs:
            example_pair = futures_pairs[0]
            example_market = markets[example_pair]
            print(f"\nPrzykład struktury pary '{example_pair}':")
            for key, value in example_market.items():
                if key in ['symbol', 'type', 'active', 'base', 'quote', 'contract', 'contractType']:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    debug_markets() 