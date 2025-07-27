import pandas as pd
import numpy as np

print("🧪 TEST POPRAWKI VOLUME_CHANGE_NORM")
print("=" * 50)

# Symuluj dane z problemem
test_data = pd.DataFrame({
    'volume': [0, 100, 50, 0, 200, 150, 0, 75]
})

print("📊 DANE TESTOWE:")
print(test_data)

print(f"\n🔍 TEST OBECNEGO KODU (pct_change):")
# Obecny kod (problemowy)
old_result = test_data['volume'].pct_change().fillna(0)
print("Wynik:")
for i, val in enumerate(old_result):
    print(f"  Wiersz {i}: {val}")

print(f"\n🔍 TEST NOWEGO KODU (logarytm z replace):")
# Nowy kod (naprawiony)
volume_ratio = test_data['volume'] / (test_data['volume'].shift(1) + 1e-8)
new_result = np.log(volume_ratio).replace([np.inf, -np.inf], 0).fillna(0)
print("Wynik:")
for i, val in enumerate(new_result):
    print(f"  Wiersz {i}: {val}")

print(f"\n📋 PORÓWNANIE:")
print("  Wiersz | Volume | Stary kod | Nowy kod | Komentarz")
print("  -------|--------|-----------|----------|----------")
for i in range(len(test_data)):
    old_val = old_result.iloc[i]
    new_val = new_result.iloc[i]
    
    if np.isinf(old_val):
        comment = "INFINITY → naprawione"
    elif old_val == new_val:
        comment = "bez zmian"
    else:
        comment = "różne wartości"
    
    print(f"  {i:6d} | {test_data['volume'].iloc[i]:6.0f} | {old_val:9.2f} | {new_val:8.2f} | {comment}")

print(f"\n✅ WNIOSEK:")
infinity_count_old = np.isinf(old_result).sum()
infinity_count_new = np.isinf(new_result).sum()

if infinity_count_old > 0 and infinity_count_new == 0:
    print(f"  POPRAWKA DZIAŁA! Usunięto {infinity_count_old} wartości infinity")
else:
    print(f"  PROBLEM: Stary kod ma {infinity_count_old} infinity, nowy kod ma {infinity_count_new} infinity")

print("\n" + "=" * 50)
print("TEST ZAKOŃCZONY") 