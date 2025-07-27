PS C:\Users\macie\OneDrive\Python\Binance\crypto\training3> python main.py
2025-07-25 18:29:17,277 - training3.utils - INFO - ============================================================
2025-07-25 18:29:17,277 - training3.utils - INFO - PODSUMOWANIE KONFIGURACJI TRENINGU (Multi-Output XGBoost)
2025-07-25 18:29:17,277 - training3.utils - INFO - ============================================================
2025-07-25 18:29:17,277 - training3.utils - INFO - Dane Wejściowe:
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Plik: ohlc_orderbook_labeled_3class_fw60m_5levels.feather
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Cechy: 73 cech (z rzeczywiście dostępnych)
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Poziomy TP/SL: 5
2025-07-25 18:29:17,277 - training3.utils - INFO - ------------------------------------------------------------
2025-07-25 18:29:17,277 - training3.utils - INFO - Parametry Podziału Danych:
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Podział Walidacyjny: 15%
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Podział Testowy: 15%
2025-07-25 18:29:17,277 - training3.utils - INFO - ------------------------------------------------------------
2025-07-25 18:29:17,277 - training3.utils - INFO - Parametry Modelu:
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Liczba drzew: 500
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Learning rate: 0.05
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Max depth: 5
2025-07-25 18:29:17,277 - training3.utils - INFO -   - Balansowanie klas: WŁĄCZONE
2025-07-25 18:29:17,277 - training3.utils - INFO - ============================================================
2025-07-25 18:29:17,277 - training3.utils - INFO - >>> KROK 1: Wczytywanie i Przygotowanie Danych <<<
2025-07-25 18:29:17,277 - training3.data_loader - INFO - Wczytywanie danych z: C:\Users\macie\OneDrive\Python\Binance\crypto\labeler3\output\ohlc_orderbook_labeled_3class_fw60m_5levels.feather
2025-07-25 18:29:17,731 - training3.data_loader - INFO - Wczytano: 1,270,074 wierszy, 87 kolumn
2025-07-25 18:29:17,784 - training3.data_loader - INFO - Ustawiono timestamp jako indeks
2025-07-25 18:29:17,784 - training3.data_loader - INFO - Rozkład etykiet:
2025-07-25 18:29:17,790 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.2% (label_tp0p8_sl0p2): {0: 81827, 1: 86039, 2: 1102208}      
2025-07-25 18:29:17,801 - training3.data_loader - INFO -   TP: 0.6%, SL: 0.3% (label_tp0p6_sl0p3): {0: 164063, 1: 166408, 2: 939603}
2025-07-25 18:29:17,807 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.4% (label_tp0p8_sl0p4): {0: 102954, 1: 108158, 2: 1058962}    
2025-07-25 18:29:17,814 - training3.data_loader - INFO -   TP: 1.0%, SL: 0.5% (label_tp1_sl0p5): {0: 68383, 1: 73207, 2: 1128484}
2025-07-25 18:29:17,822 - training3.data_loader - INFO -   TP: 1.2%, SL: 0.6% (label_tp1p2_sl0p6): {0: 46590, 1: 51235, 2: 1172249}      
2025-07-25 18:29:17,823 - training3.utils - INFO - >>> KROK 2: Przygotowanie Danych do Treningu <<<
2025-07-25 18:29:17,823 - training3.data_loader - INFO - Przygotowywanie danych do treningu...
2025-07-25 18:29:17,932 - training3.data_loader - INFO - Liczba cech: 73
2025-07-25 18:29:17,932 - training3.data_loader - INFO - Liczba wyjść (poziomów TP/SL): 5
2025-07-25 18:29:18,572 - training3.data_loader - INFO - Usunięto 0 wierszy z brakującymi danymi
2025-07-25 18:29:18,572 - training3.data_loader - INFO - Podział danych (chronologiczny):
2025-07-25 18:29:18,572 - training3.data_loader - INFO -   Trening: 889,051 próbek
2025-07-25 18:29:18,572 - training3.data_loader - INFO -   Walidacja: 190,511 próbek
2025-07-25 18:29:18,572 - training3.data_loader - INFO -   Test: 190,512 próbek
2025-07-25 18:29:18,572 - training3.data_loader - INFO - Zakresy czasowe:
2025-07-25 18:29:18,572 - training3.data_loader - INFO -   Train: 2023-01-31 00:06:00 - 2024-10-09 09:36:00
2025-07-25 18:29:18,572 - training3.data_loader - INFO -   Val:   2024-10-09 09:37:00 - 2025-02-18 16:47:00
2025-07-25 18:29:18,572 - training3.data_loader - INFO -   Test:  2025-02-18 16:48:00 - 2025-06-30 23:59:00
2025-07-25 18:29:18,572 - training3.data_loader - INFO - Skalowanie cech...
2025-07-25 18:29:20,083 - training3.data_loader - INFO - Cechy przeskalowane za pomocą RobustScaler.
2025-07-25 18:29:20,083 - training3.data_loader - INFO - Balansowanie klas...
2025-07-25 18:29:20,083 - training3.data_loader - INFO - Balansowanie na podstawie poziomu: TP: 0.8%, SL: 0.2%
2025-07-25 18:29:23,124 - training3.data_loader - INFO - Po balansowaniu: 177556 próbek treningowych
2025-07-25 18:29:23,124 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.2%: {0: 57989, 1: 60382, 2: 59185}
2025-07-25 18:29:23,124 - training3.data_loader - INFO -   TP: 0.6%, SL: 0.3%: {0: 62272, 1: 64488, 2: 50796}
2025-07-25 18:29:23,129 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.4%: {0: 59044, 1: 61470, 2: 57042}
2025-07-25 18:29:23,129 - training3.data_loader - INFO -   TP: 1.0%, SL: 0.5%: {0: 38220, 1: 40341, 2: 98995}
2025-07-25 18:29:23,129 - training3.data_loader - INFO -   TP: 1.2%, SL: 0.6%: {0: 25687, 1: 27604, 2: 124265}
2025-07-25 18:29:23,167 - training3.data_loader - INFO - Dane przygotowane pomyślnie.
2025-07-25 18:29:23,224 - training3.utils - INFO - >>> KROK 3: Trening Modelu Multi-Output XGBoost <<<
2025-07-25 18:29:23,224 - training3.model_builder - INFO - Rozpoczynanie treningu modelu Multi-Output XGBoost...
2025-07-25 18:29:23,227 - training3.model_builder - INFO - Budowanie modelu XGBoost Multi-Output...
2025-07-25 18:29:23,227 - training3.model_builder - INFO - Parametry XGBoost: {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'random_state': 42, 'n_jobs': 1, 'verbosity': 0, 'objective': 'multi:softprob', 'num_class': 3}
2025-07-25 18:29:23,227 - training3.model_builder - INFO - Model XGBoost Multi-Output zbudowany pomyślnie.
2025-07-25 18:29:23,227 - training3.model_builder - INFO - Trening w toku...
2025-07-25 18:31:15,054 - training3.model_builder - INFO - Metryki walidacyjne:
2025-07-25 18:31:15,231 - training3.model_builder - INFO -   TP: 0.8%, SL: 0.2%:
2025-07-25 18:31:15,231 - training3.model_builder - INFO -     Accuracy: 0.4094
2025-07-25 18:31:15,231 - training3.model_builder - INFO -     LONG: P=0.095, R=0.226, F1=0.134
2025-07-25 18:31:15,231 - training3.model_builder - INFO -     SHORT: P=0.109, R=0.667, F1=0.188
2025-07-25 18:31:15,231 - training3.model_builder - INFO -     NEUTRAL: P=0.929, R=0.402, F1=0.561
2025-07-25 18:31:15,231 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 18:31:15,231 - training3.model_builder - INFO -                 Predicted
2025-07-25 18:31:15,231 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 18:31:15,235 - training3.model_builder - INFO -     LONG     3037   7730     2684
2025-07-25 18:31:15,235 - training3.model_builder - INFO -     SHORT    2480   9625     2333
2025-07-25 18:31:15,235 - training3.model_builder - INFO -     NEUTRAL  26461  70832    65329
2025-07-25 18:31:15,411 - training3.model_builder - INFO -   TP: 0.6%, SL: 0.3%:
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     Accuracy: 0.3507
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     LONG: P=0.163, R=0.243, F1=0.195
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     SHORT: P=0.194, R=0.689, F1=0.303
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     NEUTRAL: P=0.862, R=0.300, F1=0.446
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 18:31:15,411 - training3.model_builder - INFO -                 Predicted
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     LONG     6808  17582     3591
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     SHORT    6084  19818     2864
2025-07-25 18:31:15,411 - training3.model_builder - INFO -     NEUTRAL  28817  64754    40193
2025-07-25 18:31:15,581 - training3.model_builder - INFO -   TP: 0.8%, SL: 0.4%:
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     Accuracy: 0.4312
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     LONG: P=0.124, R=0.170, F1=0.143
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     SHORT: P=0.141, R=0.726, F1=0.236
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     NEUTRAL: P=0.919, R=0.425, F1=0.582
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 18:31:15,581 - training3.model_builder - INFO -                 Predicted
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     LONG     2972  11390     3102
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     SHORT    2373  13415     2698
2025-07-25 18:31:15,581 - training3.model_builder - INFO -     NEUTRAL  18630  70173    65758
2025-07-25 18:31:15,748 - training3.model_builder - INFO -   TP: 1.0%, SL: 0.5%:
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     Accuracy: 0.7066
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     LONG: P=0.177, R=0.042, F1=0.068
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     SHORT: P=0.131, R=0.520, F1=0.209
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     NEUTRAL: P=0.920, R=0.766, F1=0.836
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 18:31:15,748 - training3.model_builder - INFO -                 Predicted
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     LONG      480   5309     5649
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     SHORT     520   6410     5396
2025-07-25 18:31:15,748 - training3.model_builder - INFO -     NEUTRAL  1713  37315   127719
2025-07-25 18:31:15,923 - training3.model_builder - INFO -   TP: 1.2%, SL: 0.6%:
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     Accuracy: 0.8458
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     LONG: P=0.175, R=0.020, F1=0.035
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     SHORT: P=0.127, R=0.306, F1=0.179
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     NEUTRAL: P=0.935, R=0.908, F1=0.921
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 18:31:15,923 - training3.model_builder - INFO -                 Predicted
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     LONG      149   2069     5325
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     SHORT     236   2578     5610
2025-07-25 18:31:15,923 - training3.model_builder - INFO -     NEUTRAL   468  15664   158412
2025-07-25 18:31:15,923 - training3.model_builder - INFO - Trening modelu zakończony pomyślnie.
2025-07-25 18:31:15,923 - training3.utils - INFO - >>> KROK 4: Ewaluacja Modelu <<<
2025-07-25 18:31:15,923 - training3.utils - INFO - Ewaluacja na zbiorze testowym...
2025-07-25 18:31:23,468 - training3.utils - INFO - 
--- Ewaluacja dla poziomu: TP: 0.8%, SL: 0.2% ---
2025-07-25 18:31:23,644 - training3.utils - INFO - Accuracy: 0.5152
2025-07-25 18:31:23,644 - training3.utils - INFO - LONG: P=0.127, R=0.073, F1=0.092
2025-07-25 18:31:23,646 - training3.utils - INFO - SHORT: P=0.091, R=0.738, F1=0.163
2025-07-25 18:31:23,646 - training3.utils - INFO - NEUTRAL: P=0.949, R=0.528, F1=0.678
2025-07-25 18:31:23,646 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 0.6%, SL: 0.3% ---
2025-07-25 18:31:23,832 - training3.utils - INFO - Accuracy: 0.4366
2025-07-25 18:31:23,832 - training3.utils - INFO - LONG: P=0.210, R=0.085, F1=0.121
2025-07-25 18:31:23,832 - training3.utils - INFO - SHORT: P=0.160, R=0.772, F1=0.265
2025-07-25 18:31:23,832 - training3.utils - INFO - NEUTRAL: P=0.895, R=0.437, F1=0.587
2025-07-25 18:31:23,832 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 0.8%, SL: 0.4% ---
2025-07-25 18:31:23,997 - training3.utils - INFO - Accuracy: 0.4765
2025-07-25 18:31:23,997 - training3.utils - INFO - LONG: P=0.183, R=0.064, F1=0.095
2025-07-25 18:31:23,997 - training3.utils - INFO - SHORT: P=0.113, R=0.803, F1=0.199
2025-07-25 18:31:23,997 - training3.utils - INFO - NEUTRAL: P=0.946, R=0.482, F1=0.639
2025-07-25 18:31:23,997 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 1.0%, SL: 0.5% ---
2025-07-25 18:31:24,164 - training3.utils - INFO - Accuracy: 0.7714
2025-07-25 18:31:24,164 - training3.utils - INFO - LONG: P=0.326, R=0.029, F1=0.053
2025-07-25 18:31:24,164 - training3.utils - INFO - SHORT: P=0.143, R=0.592, F1=0.230
2025-07-25 18:31:24,164 - training3.utils - INFO - NEUTRAL: P=0.945, R=0.820, F1=0.878
2025-07-25 18:31:24,164 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 1.2%, SL: 0.6% ---
2025-07-25 18:31:24,335 - training3.utils - INFO - Accuracy: 0.8627
2025-07-25 18:31:24,336 - training3.utils - INFO - LONG: P=0.308, R=0.032, F1=0.058
2025-07-25 18:31:24,336 - training3.utils - INFO - SHORT: P=0.133, R=0.387, F1=0.198
2025-07-25 18:31:24,336 - training3.utils - INFO - NEUTRAL: P=0.952, R=0.910, F1=0.931
2025-07-25 18:31:24,336 - training3.utils - INFO - >>> KROK 5: Zapisywanie Artifaktów <<<
2025-07-25 18:31:24,523 - training3.model_builder - INFO - Model zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\model_multioutput.pkl
2025-07-25 18:31:24,523 - training3.utils - INFO - Model zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\model_multioutput.pkl
2025-07-25 18:31:24,523 - training3.utils - INFO - Scaler zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\scaler.pkl
2025-07-25 18:31:24,523 - training3.utils - INFO - Wyniki ewaluacji zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\evaluation_results.json
2025-07-25 18:31:24,523 - training3.utils - INFO - >>> KROK 6: Generowanie Raportów <<<
2025-07-25 18:31:24,531 - training3.utils - INFO - Generowanie raportów...
2025-07-25 18:31:24,554 - training3.utils - INFO - Ważności cech dla TP_0.8%,_SL_0.2% zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance_TP_0.8%,_SL_0.2%.csv
2025-07-25 18:31:24,555 - training3.utils - INFO - Ważności cech dla TP_0.6%,_SL_0.3% zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance_TP_0.6%,_SL_0.3%.csv
2025-07-25 18:31:24,555 - training3.utils - INFO - Ważności cech dla TP_0.8%,_SL_0.4% zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance_TP_0.8%,_SL_0.4%.csv
2025-07-25 18:31:24,555 - training3.utils - INFO - Ważności cech dla TP_1.0%,_SL_0.5% zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance_TP_1.0%,_SL_0.5%.csv
2025-07-25 18:31:24,559 - training3.utils - INFO - Ważności cech dla TP_1.2%,_SL_0.6% zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance_TP_1.2%,_SL_0.6%.csv
2025-07-25 18:31:25,936 - training3.utils - INFO - Wykres ważności cech zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance.png
2025-07-25 18:31:28,524 - training3.utils - INFO - Wykresy confusion matrix zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\confusion_matrices.png
2025-07-25 18:31:28,532 - training3.utils - INFO - Generowanie raportu porównawczego...
2025-07-25 18:31:28,532 - training3.utils - INFO - Raport porównawczy zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\level_comparison.csv
2025-07-25 18:31:28,532 - training3.utils - INFO - NAJLEPSZE WYNIKI:
2025-07-25 18:31:28,532 - training3.utils - INFO - Ogólna dokładność: TP: 1.2%, SL: 0.6% (Accuracy=0.863)
2025-07-25 18:31:28,532 - training3.utils - INFO - LONG: TP: 0.6%, SL: 0.3% (F1=0.121)
2025-07-25 18:31:28,532 - training3.utils - INFO - SHORT: TP: 0.6%, SL: 0.3% (F1=0.265)
2025-07-25 18:31:28,532 - training3.utils - INFO - --- Proces treningowy zakończony pomyślnie! ---
2025-07-25 18:31:28,532 - training3.utils - INFO - Czas trwania: 131.25 sekund.