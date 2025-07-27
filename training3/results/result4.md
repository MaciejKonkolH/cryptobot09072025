PS C:\Users\macie\OneDrive\Python\Binance\crypto\training3> python main.py
2025-07-25 22:59:51,337 - training3.utils - INFO - ============================================================
2025-07-25 22:59:51,337 - training3.utils - INFO - PODSUMOWANIE KONFIGURACJI TRENINGU (Multi-Output XGBoost)
2025-07-25 22:59:51,337 - training3.utils - INFO - ============================================================
2025-07-25 22:59:51,337 - training3.utils - INFO - Dane Wejściowe:
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Plik: ohlc_orderbook_labeled_3class_fw60m_5levels.feather
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Cechy: 73 cech (z rzeczywiście dostępnych)
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Poziomy TP/SL: 5
2025-07-25 22:59:51,337 - training3.utils - INFO - ------------------------------------------------------------
2025-07-25 22:59:51,337 - training3.utils - INFO - Parametry Podziału Danych:
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Podział Walidacyjny: 15%
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Podział Testowy: 15%
2025-07-25 22:59:51,337 - training3.utils - INFO - ------------------------------------------------------------
2025-07-25 22:59:51,337 - training3.utils - INFO - Parametry Modelu:
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Liczba drzew: 500
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Learning rate: 0.05
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Max depth: 5
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Balansowanie klas: WYŁĄCZONE
2025-07-25 22:59:51,337 - training3.utils - INFO -   - Weighted Loss: WYŁĄCZONE
2025-07-25 22:59:51,337 - training3.utils - INFO - ============================================================
2025-07-25 22:59:51,337 - training3.utils - INFO - >>> KROK 1: Wczytywanie i Przygotowanie Danych <<<
2025-07-25 22:59:51,337 - training3.data_loader - INFO - Wczytywanie danych z: C:\Users\macie\OneDrive\Python\Binance\crypto\labeler3\output\ohlc_orderbook_labeled_3class_fw60m_5levels.feather
2025-07-25 22:59:51,820 - training3.data_loader - INFO - Wczytano: 1,270,074 wierszy, 87 kolumn
2025-07-25 22:59:51,876 - training3.data_loader - INFO - Ustawiono timestamp jako indeks
2025-07-25 22:59:51,877 - training3.data_loader - INFO - Rozkład etykiet:
2025-07-25 22:59:51,887 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.2% (label_tp0p8_sl0p2): {0: 81827, 1: 86039, 2: 1102208}
2025-07-25 22:59:51,897 - training3.data_loader - INFO -   TP: 0.6%, SL: 0.3% (label_tp0p6_sl0p3): {0: 164063, 1: 166408, 2: 939603}
2025-07-25 22:59:51,903 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.4% (label_tp0p8_sl0p4): {0: 102954, 1: 108158, 2: 1058962}
2025-07-25 22:59:51,911 - training3.data_loader - INFO -   TP: 1.0%, SL: 0.5% (label_tp1_sl0p5): {0: 68383, 1: 73207, 2: 1128484}
2025-07-25 22:59:51,919 - training3.data_loader - INFO -   TP: 1.2%, SL: 0.6% (label_tp1p2_sl0p6): {0: 46590, 1: 51235, 2: 1172249}
2025-07-25 22:59:51,919 - training3.utils - INFO - >>> KROK 2: Przygotowanie Danych do Treningu <<<
2025-07-25 22:59:51,919 - training3.data_loader - INFO - Przygotowywanie danych do treningu...
2025-07-25 22:59:52,047 - training3.data_loader - INFO - Liczba cech: 73
2025-07-25 22:59:52,047 - training3.data_loader - INFO - Liczba wyjść (poziomów TP/SL): 5
2025-07-25 22:59:52,814 - training3.data_loader - INFO - Usunięto 0 wierszy z brakującymi danymi
2025-07-25 22:59:52,814 - training3.data_loader - INFO - Podział danych (chronologiczny):
2025-07-25 22:59:52,814 - training3.data_loader - INFO -   Trening: 889,051 próbek
2025-07-25 22:59:52,814 - training3.data_loader - INFO -   Walidacja: 190,511 próbek
2025-07-25 22:59:52,814 - training3.data_loader - INFO -   Test: 190,512 próbek
2025-07-25 22:59:52,814 - training3.data_loader - INFO - Zakresy czasowe:
2025-07-25 22:59:52,814 - training3.data_loader - INFO -   Train: 2023-01-31 00:06:00 - 2024-10-09 09:36:00
2025-07-25 22:59:52,814 - training3.data_loader - INFO -   Val:   2024-10-09 09:37:00 - 2025-02-18 16:47:00
2025-07-25 22:59:52,814 - training3.data_loader - INFO -   Test:  2025-02-18 16:48:00 - 2025-06-30 23:59:00
2025-07-25 22:59:52,814 - training3.data_loader - INFO - Skalowanie cech...
2025-07-25 22:59:54,385 - training3.data_loader - INFO - Cechy przeskalowane za pomocą RobustScaler.
2025-07-25 22:59:54,385 - training3.data_loader - INFO - Balansowanie klas wyłączone - bez balansowania
2025-07-25 22:59:54,385 - training3.data_loader - INFO - Dane przygotowane pomyślnie.
2025-07-25 22:59:54,438 - training3.utils - INFO - >>> KROK 3: Trening Modelu Multi-Output XGBoost <<<
2025-07-25 22:59:54,438 - training3.model_builder - INFO - Rozpoczynanie treningu osobnych modeli XGBoost dla każdego poziomu TP/SL...
2025-07-25 22:59:54,438 - training3.model_builder - INFO - === TRENING POZIOMU 1/5 (TP: 0.8%, SL: 0.2%) ===
2025-07-25 22:59:54,438 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 22:59:54,438 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 1...
2025-07-25 23:01:20,987 - training3.model_builder - INFO -   Model 1 wytrenowany.
2025-07-25 23:01:21,000 - training3.model_builder - INFO - === TRENING POZIOMU 2/5 (TP: 0.6%, SL: 0.3%) ===
2025-07-25 23:01:21,000 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 23:01:21,000 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 2...
2025-07-25 23:02:31,455 - training3.model_builder - INFO -   Model 2 wytrenowany.
2025-07-25 23:02:31,455 - training3.model_builder - INFO - === TRENING POZIOMU 3/5 (TP: 0.8%, SL: 0.4%) ===
2025-07-25 23:02:31,455 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 23:02:31,455 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 3...
2025-07-25 23:03:40,466 - training3.model_builder - INFO -   Model 3 wytrenowany.
2025-07-25 23:03:40,474 - training3.model_builder - INFO - === TRENING POZIOMU 4/5 (TP: 1.0%, SL: 0.5%) ===
2025-07-25 23:03:40,474 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 23:03:40,474 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 4...
2025-07-25 23:04:49,718 - training3.model_builder - INFO -   Model 4 wytrenowany.
2025-07-25 23:04:49,718 - training3.model_builder - INFO - === TRENING POZIOMU 5/5 (TP: 1.2%, SL: 0.6%) ===
2025-07-25 23:04:49,718 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 23:04:49,718 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 5...
2025-07-25 23:05:58,480 - training3.model_builder - INFO -   Model 5 wytrenowany.
2025-07-25 23:05:58,480 - training3.model_builder - INFO - Całkowity czas treningu: 364.0s (6.1 min)
2025-07-25 23:06:02,946 - training3.model_builder - INFO - Metryki walidacyjne:
2025-07-25 23:06:03,119 - training3.model_builder - INFO -   TP: 0.8%, SL: 0.2%:
2025-07-25 23:06:03,119 - training3.model_builder - INFO -     Accuracy: 0.7958
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     LONG: P=0.075, R=0.005, F1=0.009
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     SHORT: P=0.135, R=0.151, F1=0.143
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     NEUTRAL: P=0.861, R=0.918, F1=0.889
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 23:06:03,127 - training3.model_builder - INFO -                 Predicted
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     LONG       65   1521    11865
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     SHORT      43   2186    12209
2025-07-25 23:06:03,127 - training3.model_builder - INFO -     NEUTRAL   753  12504   149365
2025-07-25 23:06:03,286 - training3.model_builder - INFO -   TP: 0.6%, SL: 0.3%:
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     Accuracy: 0.6576
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     LONG: P=0.204, R=0.022, F1=0.040
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     SHORT: P=0.269, R=0.312, F1=0.289
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     NEUTRAL: P=0.751, R=0.865, F1=0.804
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 23:06:03,286 - training3.model_builder - INFO -                 Predicted
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     LONG      614   8244    19123
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     SHORT     539   8986    19241
2025-07-25 23:06:03,286 - training3.model_builder - INFO -     NEUTRAL  1862  16220   115682
2025-07-25 23:06:03,456 - training3.model_builder - INFO -   TP: 0.8%, SL: 0.4%:
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     Accuracy: 0.7681
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     LONG: P=0.199, R=0.009, F1=0.018
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     SHORT: P=0.225, R=0.272, F1=0.246
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     NEUTRAL: P=0.843, R=0.913, F1=0.877
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 23:06:03,456 - training3.model_builder - INFO -                 Predicted
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     LONG      160   4407    12897
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     SHORT     153   5021    13312
2025-07-25 23:06:03,456 - training3.model_builder - INFO -     NEUTRAL   491  12913   141157
2025-07-25 23:06:03,612 - training3.model_builder - INFO -   TP: 1.0%, SL: 0.5%:
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     Accuracy: 0.8291
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     LONG: P=0.345, R=0.012, F1=0.024
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     SHORT: P=0.186, R=0.273, F1=0.221
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     NEUTRAL: P=0.898, R=0.926, F1=0.912
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 23:06:03,612 - training3.model_builder - INFO -                 Predicted
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     LONG      141   2587     8710
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     SHORT     102   3364     8860
2025-07-25 23:06:03,612 - training3.model_builder - INFO -     NEUTRAL   166  12137   154444
2025-07-25 23:06:03,765 - training3.model_builder - INFO -   TP: 1.2%, SL: 0.6%:
2025-07-25 23:06:03,765 - training3.model_builder - INFO -     Accuracy: 0.8667
2025-07-25 23:06:03,765 - training3.model_builder - INFO -     LONG: P=0.321, R=0.007, F1=0.014
2025-07-25 23:06:03,765 - training3.model_builder - INFO -     SHORT: P=0.149, R=0.285, F1=0.196
2025-07-25 23:06:03,773 - training3.model_builder - INFO -     NEUTRAL: P=0.933, R=0.932, F1=0.933
2025-07-25 23:06:03,773 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 23:06:03,773 - training3.model_builder - INFO -                 Predicted
2025-07-25 23:06:03,773 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 23:06:03,773 - training3.model_builder - INFO -     LONG       53   1859     5631
2025-07-25 23:06:03,773 - training3.model_builder - INFO -     SHORT      49   2405     5970
2025-07-25 23:06:03,773 - training3.model_builder - INFO -     NEUTRAL    63  11831   162650
2025-07-25 23:06:03,773 - training3.model_builder - INFO - Trening wszystkich modeli zakończony pomyślnie.
2025-07-25 23:06:03,773 - training3.utils - INFO - >>> KROK 4: Ewaluacja Modelu <<<
2025-07-25 23:06:03,773 - training3.utils - INFO - Ewaluacja na zbiorze testowym...
2025-07-25 23:06:08,221 - training3.utils - INFO - 
--- Ewaluacja dla poziomu: TP: 0.8%, SL: 0.2% ---
2025-07-25 23:06:08,395 - training3.utils - INFO - Accuracy: 0.8361
2025-07-25 23:06:08,395 - training3.utils - INFO - LONG: P=0.137, R=0.003, F1=0.005
2025-07-25 23:06:08,395 - training3.utils - INFO - SHORT: P=0.123, R=0.162, F1=0.140
2025-07-25 23:06:08,395 - training3.utils - INFO - NEUTRAL: P=0.897, R=0.932, F1=0.914
2025-07-25 23:06:08,395 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 0.6%, SL: 0.3% ---
2025-07-25 23:06:08,544 - training3.utils - INFO - Accuracy: 0.6994
2025-07-25 23:06:08,544 - training3.utils - INFO - LONG: P=0.275, R=0.013, F1=0.024
2025-07-25 23:06:08,544 - training3.utils - INFO - SHORT: P=0.240, R=0.415, F1=0.304
2025-07-25 23:06:08,544 - training3.utils - INFO - NEUTRAL: P=0.823, R=0.848, F1=0.835
2025-07-25 23:06:08,559 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 0.8%, SL: 0.4% ---
2025-07-25 23:06:08,710 - training3.utils - INFO - Accuracy: 0.8037
2025-07-25 23:06:08,710 - training3.utils - INFO - LONG: P=0.206, R=0.006, F1=0.011
2025-07-25 23:06:08,710 - training3.utils - INFO - SHORT: P=0.214, R=0.345, F1=0.264
2025-07-25 23:06:08,710 - training3.utils - INFO - NEUTRAL: P=0.888, R=0.912, F1=0.900
2025-07-25 23:06:08,710 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 1.0%, SL: 0.5% ---
2025-07-25 23:06:08,871 - training3.utils - INFO - Accuracy: 0.8634
2025-07-25 23:06:08,871 - training3.utils - INFO - LONG: P=0.369, R=0.024, F1=0.045
2025-07-25 23:06:08,871 - training3.utils - INFO - SHORT: P=0.193, R=0.311, F1=0.238
2025-07-25 23:06:08,871 - training3.utils - INFO - NEUTRAL: P=0.926, R=0.938, F1=0.932
2025-07-25 23:06:08,871 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 1.2%, SL: 0.6% ---
2025-07-25 23:06:09,027 - training3.utils - INFO - Accuracy: 0.8723
2025-07-25 23:06:09,028 - training3.utils - INFO - LONG: P=0.262, R=0.013, F1=0.024
2025-07-25 23:06:09,028 - training3.utils - INFO - SHORT: P=0.135, R=0.348, F1=0.195
2025-07-25 23:06:09,028 - training3.utils - INFO - NEUTRAL: P=0.950, R=0.922, F1=0.936
2025-07-25 23:06:09,028 - training3.utils - INFO - >>> KROK 5: Zapisywanie Artifaktów <<<
2025-07-25 23:06:09,215 - training3.model_builder - INFO - Modele zapisane do C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\model_multioutput.pkl_level*.joblib
2025-07-25 23:06:09,215 - training3.utils - INFO - Model zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\model_multioutput.pkl
2025-07-25 23:06:09,215 - training3.utils - INFO - Scaler zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\scaler.pkl
2025-07-25 23:06:09,220 - training3.utils - INFO - Wyniki ewaluacji zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\evaluation_results.json
2025-07-25 23:06:09,220 - training3.utils - INFO - >>> KROK 6: Generowanie Raportów <<<
2025-07-25 23:06:09,220 - training3.utils - INFO - Generowanie raportów...
2025-07-25 23:06:09,918 - training3.utils - INFO - Wykres ważności cech zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\feature_importance.png
2025-07-25 23:06:12,072 - training3.utils - INFO - Wykresy confusion matrix zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\confusion_matrices.png
2025-07-25 23:06:12,079 - training3.utils - INFO - --- Proces treningowy zakończony pomyślnie! ---
2025-07-25 23:06:12,079 - training3.utils - INFO - Czas trwania: 380.74 sekund.
PS C:\Users\macie\OneDrive\Python\Binance\crypto\training3> 