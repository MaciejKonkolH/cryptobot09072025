PS C:\Users\macie\OneDrive\Python\Binance\crypto\training3> python main.py
2025-07-25 21:03:27,165 - training3.utils - INFO - ============================================================
2025-07-25 21:03:27,165 - training3.utils - INFO - PODSUMOWANIE KONFIGURACJI TRENINGU (Multi-Output XGBoost)
2025-07-25 21:03:27,165 - training3.utils - INFO - ============================================================
2025-07-25 21:03:27,165 - training3.utils - INFO - Dane Wejściowe:
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Plik: ohlc_orderbook_labeled_3class_fw60m_5levels.feather
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Cechy: 73 cech (z rzeczywiście dostępnych)
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Poziomy TP/SL: 5
2025-07-25 21:03:27,165 - training3.utils - INFO - ------------------------------------------------------------
2025-07-25 21:03:27,165 - training3.utils - INFO - Parametry Podziału Danych:
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Podział Walidacyjny: 15%
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Podział Testowy: 15%
2025-07-25 21:03:27,165 - training3.utils - INFO - ------------------------------------------------------------
2025-07-25 21:03:27,165 - training3.utils - INFO - Parametry Modelu:
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Liczba drzew: 500
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Learning rate: 0.05
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Max depth: 5
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Balansowanie klas: WYŁĄCZONE
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Weighted Loss: WŁĄCZONE
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Mnożnik wag LONG/SHORT: 2.0
2025-07-25 21:03:27,165 - training3.utils - INFO -   - Redukcja wagi NEUTRAL: 0.5
2025-07-25 21:03:27,165 - training3.utils - INFO - ============================================================
2025-07-25 21:03:27,165 - training3.utils - INFO - >>> KROK 1: Wczytywanie i Przygotowanie Danych <<<
2025-07-25 21:03:27,165 - training3.data_loader - INFO - Wczytywanie danych z: C:\Users\macie\OneDrive\Python\Binance\crypto\labeler3\output\ohlc_orderbook_labeled_3class_fw60m_5levels.feather
2025-07-25 21:03:27,636 - training3.data_loader - INFO - Wczytano: 1,270,074 wierszy, 87 kolumn
2025-07-25 21:03:27,683 - training3.data_loader - INFO - Ustawiono timestamp jako indeks
2025-07-25 21:03:27,683 - training3.data_loader - INFO - Rozkład etykiet:
2025-07-25 21:03:27,691 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.2% (label_tp0p8_sl0p2): {0: 81827, 1: 86039, 2: 1102208}
2025-07-25 21:03:27,700 - training3.data_loader - INFO -   TP: 0.6%, SL: 0.3% (label_tp0p6_sl0p3): {0: 164063, 1: 166408, 2: 939603}
2025-07-25 21:03:27,700 - training3.data_loader - INFO -   TP: 0.8%, SL: 0.4% (label_tp0p8_sl0p4): {0: 102954, 1: 108158, 2: 1058962}
2025-07-25 21:03:27,714 - training3.data_loader - INFO -   TP: 1.0%, SL: 0.5% (label_tp1_sl0p5): {0: 68383, 1: 73207, 2: 1128484}
2025-07-25 21:03:27,721 - training3.data_loader - INFO -   TP: 1.2%, SL: 0.6% (label_tp1p2_sl0p6): {0: 46590, 1: 51235, 2: 1172249}
2025-07-25 21:03:27,722 - training3.utils - INFO - >>> KROK 2: Przygotowanie Danych do Treningu <<<
2025-07-25 21:03:27,722 - training3.data_loader - INFO - Przygotowywanie danych do treningu...
2025-07-25 21:03:27,839 - training3.data_loader - INFO - Liczba cech: 73
2025-07-25 21:03:27,839 - training3.data_loader - INFO - Liczba wyjść (poziomów TP/SL): 5
2025-07-25 21:03:28,352 - training3.data_loader - INFO - Usunięto 0 wierszy z brakującymi danymi
2025-07-25 21:03:28,352 - training3.data_loader - INFO - Podział danych (chronologiczny):
2025-07-25 21:03:28,352 - training3.data_loader - INFO -   Trening: 889,051 próbek
2025-07-25 21:03:28,352 - training3.data_loader - INFO -   Walidacja: 190,511 próbek
2025-07-25 21:03:28,357 - training3.data_loader - INFO -   Test: 190,512 próbek
2025-07-25 21:03:28,357 - training3.data_loader - INFO - Zakresy czasowe:
2025-07-25 21:03:28,357 - training3.data_loader - INFO -   Train: 2023-01-31 00:06:00 - 2024-10-09 09:36:00
2025-07-25 21:03:28,361 - training3.data_loader - INFO -   Val:   2024-10-09 09:37:00 - 2025-02-18 16:47:00
2025-07-25 21:03:28,361 - training3.data_loader - INFO -   Test:  2025-02-18 16:48:00 - 2025-06-30 23:59:00
2025-07-25 21:03:28,361 - training3.data_loader - INFO - Skalowanie cech...
2025-07-25 21:03:29,879 - training3.data_loader - INFO - Cechy przeskalowane za pomocą RobustScaler.
2025-07-25 21:03:29,879 - training3.data_loader - INFO - Balansowanie klas wyłączone - używamy Weighted Loss
2025-07-25 21:03:29,879 - training3.data_loader - INFO - Dane przygotowane pomyślnie.
2025-07-25 21:03:29,906 - training3.utils - INFO - >>> KROK 3: Trening Modelu Multi-Output XGBoost <<<
2025-07-25 21:03:29,906 - training3.model_builder - INFO - Rozpoczynanie treningu osobnych modeli XGBoost dla każdego poziomu TP/SL...
2025-07-25 21:03:29,906 - training3.model_builder - INFO - === TRENING POZIOMU 1/5 (TP: 0.8%, SL: 0.2%) ===
2025-07-25 21:03:29,906 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 21:03:30,103 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 1...
2025-07-25 21:04:55,510 - training3.model_builder - INFO -   Model 1 wytrenowany.
2025-07-25 21:04:55,519 - training3.model_builder - INFO - === TRENING POZIOMU 2/5 (TP: 0.6%, SL: 0.3%) ===
2025-07-25 21:04:55,519 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 21:04:55,730 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 2...
2025-07-25 21:05:59,919 - training3.model_builder - INFO -   Model 2 wytrenowany.
2025-07-25 21:05:59,919 - training3.model_builder - INFO - === TRENING POZIOMU 3/5 (TP: 0.8%, SL: 0.4%) ===
2025-07-25 21:05:59,919 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 21:06:00,135 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 3...
2025-07-25 21:07:04,912 - training3.model_builder - INFO -   Model 3 wytrenowany.
2025-07-25 21:07:04,913 - training3.model_builder - INFO - === TRENING POZIOMU 4/5 (TP: 1.0%, SL: 0.5%) ===
2025-07-25 21:07:04,913 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 21:07:05,122 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 4...
2025-07-25 21:08:08,769 - training3.model_builder - INFO -   Model 4 wytrenowany.
2025-07-25 21:08:08,769 - training3.model_builder - INFO - === TRENING POZIOMU 5/5 (TP: 1.2%, SL: 0.6%) ===
2025-07-25 21:08:08,769 - training3.model_builder - INFO -   Bez balansowania: 889051 próbek
2025-07-25 21:08:08,979 - training3.model_builder - INFO -   Rozpoczynanie treningu modelu 5...
2025-07-25 21:09:14,216 - training3.model_builder - INFO -   Model 5 wytrenowany.
2025-07-25 21:09:14,216 - training3.model_builder - INFO - Całkowity czas treningu: 344.3s (5.7 min)
2025-07-25 21:09:18,479 - training3.model_builder - INFO - Metryki walidacyjne:
2025-07-25 21:09:18,657 - training3.model_builder - INFO -   TP: 0.8%, SL: 0.2%:
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     Accuracy: 0.5304
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     LONG: P=0.104, R=0.118, F1=0.111
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     SHORT: P=0.115, R=0.589, F1=0.192
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     NEUTRAL: P=0.900, R=0.559, F1=0.690
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 21:09:18,657 - training3.model_builder - INFO -                 Predicted
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     LONG     1591   6683     5177
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     SHORT    1033   8507     4898
2025-07-25 21:09:18,657 - training3.model_builder - INFO -     NEUTRAL  12701  58969    90952
2025-07-25 21:09:18,825 - training3.model_builder - INFO -   TP: 0.6%, SL: 0.3%:
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     Accuracy: 0.4218
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     LONG: P=0.175, R=0.164, F1=0.169
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     SHORT: P=0.207, R=0.705, F1=0.320
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     NEUTRAL: P=0.836, R=0.415, F1=0.555
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 21:09:18,825 - training3.model_builder - INFO -                 Predicted
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     LONG     4595  17262     6124
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     SHORT    3714  20266     4786
2025-07-25 21:09:18,825 - training3.model_builder - INFO -     NEUTRAL  17954  60305    55505
2025-07-25 21:09:18,990 - training3.model_builder - INFO -   TP: 0.8%, SL: 0.4%:
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     Accuracy: 0.4774
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     LONG: P=0.134, R=0.177, F1=0.152
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     SHORT: P=0.146, R=0.670, F1=0.240
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     NEUTRAL: P=0.913, R=0.488, F1=0.636
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 21:09:18,990 - training3.model_builder - INFO -                 Predicted
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     LONG     3094  10625     3745
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     SHORT    2639  12383     3464
2025-07-25 21:09:18,990 - training3.model_builder - INFO -     NEUTRAL  17430  61650    75481
2025-07-25 21:09:19,149 - training3.model_builder - INFO -   TP: 1.0%, SL: 0.5%:
2025-07-25 21:09:19,149 - training3.model_builder - INFO -     Accuracy: 0.6043
2025-07-25 21:09:19,149 - training3.model_builder - INFO -     LONG: P=0.116, R=0.100, F1=0.107
2025-07-25 21:09:19,149 - training3.model_builder - INFO -     SHORT: P=0.118, R=0.654, F1=0.201
2025-07-25 21:09:19,149 - training3.model_builder - INFO -     NEUTRAL: P=0.940, R=0.635, F1=0.758
2025-07-25 21:09:19,150 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 21:09:19,150 - training3.model_builder - INFO -                 Predicted
2025-07-25 21:09:19,150 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 21:09:19,150 - training3.model_builder - INFO -     LONG     1139   6978     3321
2025-07-25 21:09:19,150 - training3.model_builder - INFO -     SHORT     859   8063     3404
2025-07-25 21:09:19,150 - training3.model_builder - INFO -     NEUTRAL  7820  53007   105920
2025-07-25 21:09:19,306 - training3.model_builder - INFO -   TP: 1.2%, SL: 0.6%:
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     Accuracy: 0.6890
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     LONG: P=0.070, R=0.061, F1=0.065
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     SHORT: P=0.089, R=0.535, F1=0.152
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     NEUTRAL: P=0.948, R=0.724, F1=0.821
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     Confusion Matrix:
2025-07-25 21:09:19,306 - training3.model_builder - INFO -                 Predicted
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     Actual    LONG  SHORT  NEUTRAL
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     LONG      457   3756     3330
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     SHORT     390   4506     3528
2025-07-25 21:09:19,306 - training3.model_builder - INFO -     NEUTRAL  5660  42578   126306
2025-07-25 21:09:19,306 - training3.model_builder - INFO - Trening wszystkich modeli zakończony pomyślnie.
2025-07-25 21:09:19,306 - training3.utils - INFO - >>> KROK 4: Ewaluacja Modelu <<<
2025-07-25 21:09:19,306 - training3.utils - INFO - Ewaluacja na zbiorze testowym...
2025-07-25 21:09:23,575 - training3.utils - INFO - 
--- Ewaluacja dla poziomu: TP: 0.8%, SL: 0.2% ---
2025-07-25 21:09:23,750 - training3.utils - INFO - Accuracy: 0.5834
2025-07-25 21:09:23,750 - training3.utils - INFO - LONG: P=0.140, R=0.050, F1=0.074
2025-07-25 21:09:23,750 - training3.utils - INFO - SHORT: P=0.095, R=0.647, F1=0.166
2025-07-25 21:09:23,750 - training3.utils - INFO - NEUTRAL: P=0.934, R=0.612, F1=0.739
2025-07-25 21:09:23,750 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 0.6%, SL: 0.3% ---
2025-07-25 21:09:23,908 - training3.utils - INFO - Accuracy: 0.5084
2025-07-25 21:09:23,908 - training3.utils - INFO - LONG: P=0.171, R=0.093, F1=0.120
2025-07-25 21:09:23,908 - training3.utils - INFO - SHORT: P=0.178, R=0.696, F1=0.284
2025-07-25 21:09:23,908 - training3.utils - INFO - NEUTRAL: P=0.880, R=0.542, F1=0.671
2025-07-25 21:09:23,909 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 0.8%, SL: 0.4% ---
2025-07-25 21:09:24,061 - training3.utils - INFO - Accuracy: 0.5396
2025-07-25 21:09:24,061 - training3.utils - INFO - LONG: P=0.190, R=0.086, F1=0.118
2025-07-25 21:09:24,061 - training3.utils - INFO - SHORT: P=0.122, R=0.734, F1=0.210
2025-07-25 21:09:24,061 - training3.utils - INFO - NEUTRAL: P=0.939, R=0.560, F1=0.702
2025-07-25 21:09:24,061 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 1.0%, SL: 0.5% ---
2025-07-25 21:09:24,211 - training3.utils - INFO - Accuracy: 0.7041
2025-07-25 21:09:24,211 - training3.utils - INFO - LONG: P=0.223, R=0.062, F1=0.097
2025-07-25 21:09:24,211 - training3.utils - INFO - SHORT: P=0.123, R=0.705, F1=0.210
2025-07-25 21:09:24,211 - training3.utils - INFO - NEUTRAL: P=0.960, R=0.737, F1=0.834
2025-07-25 21:09:24,211 - training3.utils - INFO -
--- Ewaluacja dla poziomu: TP: 1.2%, SL: 0.6% ---
2025-07-25 21:09:24,367 - training3.utils - INFO - Accuracy: 0.7429
2025-07-25 21:09:24,367 - training3.utils - INFO - LONG: P=0.257, R=0.063, F1=0.101
2025-07-25 21:09:24,367 - training3.utils - INFO - SHORT: P=0.098, R=0.675, F1=0.171
2025-07-25 21:09:24,367 - training3.utils - INFO - NEUTRAL: P=0.968, R=0.769, F1=0.857
2025-07-25 21:09:24,367 - training3.utils - INFO - >>> KROK 5: Zapisywanie Artifaktów <<<
2025-07-25 21:09:24,543 - training3.model_builder - INFO - Modele zapisane do C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\model_multioutput.pkl_level*.joblib
2025-07-25 21:09:24,544 - training3.utils - INFO - Model zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\model_multioutput.pkl
2025-07-25 21:09:24,545 - training3.utils - INFO - Scaler zapisany: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\models\scaler.pkl
2025-07-25 21:09:24,546 - training3.utils - INFO - Wyniki ewaluacji zapisane: C:\Users\macie\OneDrive\Python\Binance\crypto\training3\output\reports\evaluation_results.json
2025-07-25 21:09:24,546 - training3.utils - INFO - >>> KROK 6: Generowanie Raportów <<<
2025-07-25 21:09:24,546 - training3.utils - INFO - Generowanie raportów...
2025-07-25 21:09:24,560 - training3.utils - ERROR - Błąd podczas treningu: 'numpy.float32' object has no attribute 'to_csv'
Traceback (most recent call last):
  File "C:\Users\macie\OneDrive\Python\Binance\crypto\training3\main.py", line 98, in run
    self._generate_reports()
  File "C:\Users\macie\OneDrive\Python\Binance\crypto\training3\main.py", line 197, in _generate_reports
    feature_importance.to_csv(importance_path, index=False)
AttributeError: 'numpy.float32' object has no attribute 'to_csv'
Traceback (most recent call last):
  File "C:\Users\macie\OneDrive\Python\Binance\crypto\training3\main.py", line 317, in <module>
    main()
  File "C:\Users\macie\OneDrive\Python\Binance\crypto\training3\main.py", line 314, in main
    trainer.run()
  File "C:\Users\macie\OneDrive\Python\Binance\crypto\training3\main.py", line 98, in run
    self._generate_reports()
  File "C:\Users\macie\OneDrive\Python\Binance\crypto\training3\main.py", line 197, in _generate_reports
    feature_importance.to_csv(importance_path, index=False)
AttributeError: 'numpy.float32' object has no attribute 'to_csv'
PS C:\Users\macie\OneDrive\Python\Binance\crypto\training3> 