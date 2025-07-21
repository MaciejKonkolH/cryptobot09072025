import pandas as pd

# Mapowanie klas
CLASS_LABELS = {
    0: 'PROFIT_SHORT',
    1: 'TIMEOUT_HOLD', 
    2: 'PROFIT_LONG',
    3: 'LOSS_SHORT',
    4: 'LOSS_LONG',
    5: 'CHAOS_HOLD'
}

df = pd.read_feather('output/orderbook_ohlc_labels_FW-120_levels-3.feather')

print("=== ROZKŁAD ETYKIET PO KOREKCIE TP/SL ===\n")

for col in ['label_tp02_sl01', 'label_tp015_sl0075', 'label_tp01_sl005']:
    print(f"{col}:")
    counts = df[col].value_counts().sort_index()
    total = len(df)
    
    for idx, count in counts.items():
        class_name = CLASS_LABELS.get(idx, f'UNKNOWN_{idx}')
        percentage = count/total*100
        print(f"  {idx} ({class_name}): {count} ({percentage:.1f}%)")
    print()

print("=== PODSUMOWANIE ===")
print(f"Całkowita liczba próbek: {len(df)}")
print("Czy mamy różnorodność klas? TAK!" if len(df['label_tp02_sl01'].unique()) > 1 else "Czy mamy różnorodność klas? NIE!") 