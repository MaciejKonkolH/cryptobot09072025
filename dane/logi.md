(venv) root@6208ee5b0954:/workspace/training# python main.py
2025-07-15 13:18:39.385366: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-07-15 13:18:39.385423: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-07-15 13:18:39.386195: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-07-15 13:18:39.390395: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-07-15 13:18:40,645 - training.utils - INFO - ============================================================
2025-07-15 13:18:40,645 - training.utils - INFO - 🎯 PODSUMOWANIE KONFIGURACJI TRENINGU 🎯
2025-07-15 13:18:40,645 - training.utils - INFO - ============================================================
2025-07-15 13:18:40,645 - training.utils - INFO - Dane Wejściowe:
2025-07-15 13:18:40,645 - training.utils - INFO -   - Plik: BTCUSDT-1m-futures_features_and_labels_FW-480_SL-050_TP-100.feather
2025-07-15 13:18:40,645 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:18:40,645 - training.utils - INFO - Parametry Modelu i Treningu:
2025-07-15 13:18:40,645 - training.utils - INFO -   - Długość Sekwencji: 120
2025-07-15 13:18:40,645 - training.utils - INFO -   - Podział Walidacyjny: 10%
2025-07-15 13:18:40,645 - training.utils - INFO -   - Podział Testowy: 10%
2025-07-15 13:18:40,645 - training.utils - INFO -   - Epoki: 5
2025-07-15 13:18:40,645 - training.utils - INFO -   - Rozmiar Batcha: 4096
2025-07-15 13:18:40,645 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:18:40,645 - training.utils - INFO - Parametry Predykcji:
2025-07-15 13:18:40,645 - training.utils - INFO -   - Progi Pewności: S=0.44, H=0.3, L=0.44
2025-07-15 13:18:40,645 - training.utils - INFO - ============================================================
2025-07-15 13:18:40,645 - training.utils - INFO - >>> KROK 1: Wczytywanie i Wstępne Przetworzenie Danych <<<
2025-07-15 13:18:40,958 - training.utils - INFO - Liczba wierszy po wczytaniu: 2,812,195
2025-07-15 13:18:40,967 - training.utils - INFO - Rozkład etykiet: 0 (SHORT): 27.49%, 1 (HOLD): 44.49%, 2 (LONG): 28.02%
2025-07-15 13:18:40,967 - training.utils - INFO - >>> KROK 2: Chronologiczny Podział na Zbiory <<<
2025-07-15 13:18:40,967 - training.utils - INFO - Podział danych: Trening: 2,249,756, Walidacja: 281,219, Test: 281,220
2025-07-15 13:18:40,967 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:18:40,967 - training.utils - INFO - Zakresy dat dla zbiorów:
2025-07-15 13:18:40,970 - training.utils - INFO -   - Treningowy:  od 2020-01-29 23:59 do 2024-05-10 09:04
2025-07-15 13:18:40,971 - training.utils - INFO -   - Walidacyjny: od 2024-05-10 09:05 do 2024-11-21 16:59
2025-07-15 13:18:40,971 - training.utils - INFO -   - Testowy:     od 2024-11-21 17:00 do 2025-06-04 23:59
2025-07-15 13:18:40,971 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:18:40,971 - training.utils - INFO - >>> KROK 3: Skalowanie Cech Statycznych <<<
2025-07-15 13:18:40,991 - training.utils - INFO - Dopasowywanie skalera 'robust' na 5 cechach statycznych...
2025-07-15 13:18:41,211 - training.utils - INFO - Zapisano skaler do pliku: /workspace/training/output/models/scaler.pkl
2025-07-15 13:18:41,211 - training.utils - INFO - Transformacja zbiorów treningowego, walidacyjnego i testowego dla cech statycznych.
2025-07-15 13:18:41,270 - training.utils - INFO - >>> KROK 4: Przygotowanie Generatorów Danych (Architektura Hybrydowa) <<<

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
2025-07-15 13:18:41,345 - training.utils - INFO - Generatory danych (hybrydowe) zostały utworzone.
2025-07-15 13:18:41,345 - training.utils - INFO - >>> KROK 5: Budowanie i Kompilacja Modelu <<<
INFO: Budowanie modelu o architekturze: conv_lstm
2025-07-15 13:18:41.495849: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 46866 MB memory:  -> device: 0, name: NVIDIA RTX A6000, pci bus id: 0000:46:00.0, compute capability: 8.6
INFO: Architektura modelu została zbudowana pomyślnie.
INFO: Używanie funkcji straty: categorical_crossentropy
2025-07-15 13:18:42,179 - training.utils - INFO - Model: "model"
2025-07-15 13:18:42,179 - training.utils - INFO - _________________________________________________________________
2025-07-15 13:18:42,179 - training.utils - INFO -  Layer (type)                Output Shape              Param #   
2025-07-15 13:18:42,179 - training.utils - INFO - =================================================================
2025-07-15 13:18:42,179 - training.utils - INFO -  input_1 (InputLayer)        [(None, 120, 9)]          0         
2025-07-15 13:18:42,179 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,179 - training.utils - INFO -  conv1d (Conv1D)             (None, 120, 64)           2944      
2025-07-15 13:18:42,179 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,179 - training.utils - INFO -  conv1d_1 (Conv1D)           (None, 120, 128)          41088     
2025-07-15 13:18:42,180 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,180 - training.utils - INFO -  max_pooling1d (MaxPooling1  (None, 60, 128)           0         
2025-07-15 13:18:42,180 - training.utils - INFO -  D)                                                              
2025-07-15 13:18:42,180 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,180 - training.utils - INFO -  dropout (Dropout)           (None, 60, 128)           0         
2025-07-15 13:18:42,180 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,180 - training.utils - INFO -  lstm (LSTM)                 (None, 64)                49408     
2025-07-15 13:18:42,180 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,180 - training.utils - INFO -  dropout_1 (Dropout)         (None, 64)                0         
2025-07-15 13:18:42,180 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,180 - training.utils - INFO -  dense (Dense)               (None, 3)                 195       
2025-07-15 13:18:42,180 - training.utils - INFO -                                                                  
2025-07-15 13:18:42,180 - training.utils - INFO - =================================================================
2025-07-15 13:18:42,180 - training.utils - INFO - Total params: 93635 (365.76 KB)
2025-07-15 13:18:42,181 - training.utils - INFO - Trainable params: 93635 (365.76 KB)
2025-07-15 13:18:42,181 - training.utils - INFO - Non-trainable params: 0 (0.00 Byte)
2025-07-15 13:18:42,181 - training.utils - INFO - _________________________________________________________________
2025-07-15 13:18:42,181 - training.utils - INFO - >>> KROK 6: Konfiguracja i Trening Modelu <<<
2025-07-15 13:18:42,181 - training.utils - INFO - >>> KROK 6: Konfiguracja i Trening Modelu <<<
2025-07-15 13:18:42,181 - training.utils - INFO - Utworzono podstawowe callbacki: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
2025-07-15 13:18:42,181 - training.utils - INFO - Nie zastosowano żadnych wag klas.
2025-07-15 13:18:42,181 - training.utils - INFO - Rozpoczynanie treningu modelu...
Epoch 1/5
2025-07-15 13:18:44.027114: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8907
2025-07-15 13:18:45.137697: I external/local_xla/xla/service/service.cc:168] XLA service 0x7f54ac784cc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2025-07-15 13:18:45.137801: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA RTX A6000, Compute Capability 8.6
2025-07-15 13:18:45.144978: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1752585525.249258   60580 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
462/463 [============================>.] - ETA: 0s - loss: 1.0693 - accuracy: 0.4096   
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)

Epoch 1: val_loss improved from inf to 1.07153, saving model to /workspace/training/output/models/model.h5
/usr/local/lib/python3.11/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
463/463 [==============================] - 26s 48ms/step - loss: 1.0692 - accuracy: 0.4096 - val_loss: 1.0715 - val_accuracy: 0.4061 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 2/5
462/463 [============================>.] - ETA: 0s - loss: 1.0592 - accuracy: 0.4246  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)

Epoch 2: val_loss did not improve from 1.07153
463/463 [==============================] - 26s 55ms/step - loss: 1.0592 - accuracy: 0.4246 - val_loss: 1.0812 - val_accuracy: 0.3750 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 3/5
462/463 [============================>.] - ETA: 0s - loss: 1.0497 - accuracy: 0.4369  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
463/463 [==============================] - ETA: 0s - loss: 1.0497 - accuracy: 0.4369
Epoch 3: val_loss did not improve from 1.07153
463/463 [==============================] - 22s 46ms/step - loss: 1.0497 - accuracy: 0.4369 - val_loss: 1.0863 - val_accuracy: 0.3716 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 4/5
463/463 [==============================] - ETA: 0s - loss: 1.0302 - accuracy: 0.4578  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)

Epoch 4: val_loss did not improve from 1.07153
463/463 [==============================] - 26s 54ms/step - loss: 1.0302 - accuracy: 0.4578 - val_loss: 1.0863 - val_accuracy: 0.3807 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 5/5
463/463 [==============================] - ETA: 0s - loss: 1.0035 - accuracy: 0.4833  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)

Epoch 5: val_loss did not improve from 1.07153
463/463 [==============================] - 27s 55ms/step - loss: 1.0035 - accuracy: 0.4833 - val_loss: 1.1231 - val_accuracy: 0.3612 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
2025-07-15 13:20:50,580 - training.utils - INFO - Trening zakończony.
2025-07-15 13:20:50,581 - training.utils - INFO - >>> KROK 7: Ostateczny Egzamin i Analiza Predykcji <<<
2025-07-15 13:20:50,581 - training.utils - INFO - Ocenianie modelu na zbiorze testowym...
69/69 [==============================] - 7s 101ms/step - loss: 1.1713 - accuracy: 0.3033
2025-07-15 13:20:57,761 - training.utils - INFO - Wyniki na zbiorze testowym - Strata: 1.1713, Dokładność: 0.3033
2025-07-15 13:20:57,761 - training.utils - INFO - Generowanie predykcji dla zbioru testowego...
69/69 [==============================] - 7s 93ms/step
2025-07-15 13:21:05,222 - training.utils - INFO - Predykcje zapisane w: /workspace/training/output/reports/test_predictions.csv
2025-07-15 13:21:05,223 - training.utils - INFO - Generowanie raportu końcowego (placeholder).
2025-07-15 13:21:05,223 - training.utils - INFO - Metadane zapisane w: /workspace/training/output/models/metadata.json
2025-07-15 13:21:05,304 - training.utils - INFO - Raport z ewaluacji zapisany w: /workspace/training/output/reports/evaluation_report.txt
2025-07-15 13:21:05,304 - training.utils - INFO - --- Proces treningowy zakończony pomyślnie! ---

================================================================================
🎯                           TRENING - RAPORT KOŃCOWY                           🎯
================================================================================
📊 DANE:
   Plik źródłowy: BTCUSDT-1m-futures_features_and_labels_FW-480_SL-050_TP-100.feather
   Całkowity czas: 0:02:24.659901
   Rozmiary zbiorów:
     - Treningowy: (2249756, 19)
     - Walidacyjny: (281219, 19)
     - Testowy: (281220, 19)
🧠 MODEL:
   Architektura: InputLayer -> Conv1D -> MaxPooling1D -> Dropout -> LSTM -> Dense
   Liczba parametrów: 93,635
   Zapisany w: model.h5

🏋️ WYNIKI TRENINGU:
   Liczba epok: 5 / 5
   Najlepsza epoka (wg val_loss): 1
   Końcowa strata walidacyjna: 1.1231
   Najlepsza strata walidacyjna: 1.0715
   Końcowa dokładność walidacyjna: 0.3612
   Najlepsza dokładność walidacyjna: 0.4061

⚖️ BALANSOWANIE I WAGI KLAS:
   - Metoda balansowania: Dynamiczny Undersampling (Callback)
   - Metoda wag: Wyłączona w konfiguracji

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #1)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.4, HOLD=0.3, LONG=0.4
   Zaakceptowano: 219,458 / 281,101 (78.07%)
   Odrzucono: 61,643 / 281,101 (21.93%)

WYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:
               precision      recall    f1-score     support
--------------------------------------------------------------
SHORT (0)         0.2591      0.6108      0.3639    55,077.0
HOLD (1)          0.6401      0.1503      0.2434   105,510.0
LONG (2)          0.2829      0.3117      0.2966    58,871.0
--------------------------------------------------------------
accuracy                                  0.3092   219,458.0
macro avg         0.3940      0.3576      0.3013   219,458.0
weighted avg      0.4487      0.3092      0.2879   219,458.0

Macierz pomyłek (dla zaakceptowanych predykcji):
               SHORT (0)    HOLD (1)    LONG (2)
   SHORT (0)      33,639       4,619      16,819
    HOLD (1)      59,963      15,858      29,689
    LONG (2)      36,221       4,298      18,352

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #2)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.43, HOLD=0.3, LONG=0.43
   Zaakceptowano: 171,649 / 281,101 (61.06%)
   Odrzucono: 109,452 / 281,101 (38.94%)

WYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:
               precision      recall    f1-score     support
--------------------------------------------------------------
SHORT (0)         0.2522      0.6287      0.3600    41,518.0
HOLD (1)          0.6401      0.1866      0.2889    84,988.0
LONG (2)          0.2806      0.2697      0.2750    45,143.0
--------------------------------------------------------------
accuracy                                  0.3154   171,649.0
macro avg         0.3910      0.3616      0.3080   171,649.0
weighted avg      0.4517      0.3154      0.3025   171,649.0

Macierz pomyłek (dla zaakceptowanych predykcji):
               SHORT (0)    HOLD (1)    LONG (2)
   SHORT (0)      26,102       4,619      10,797
    HOLD (1)      48,721      15,858      20,409
    LONG (2)      28,672       4,298      12,173

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #3)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.46, HOLD=0.3, LONG=0.46
   Zaakceptowano: 134,882 / 281,101 (47.98%)
   Odrzucono: 146,219 / 281,101 (52.02%)

WYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:
               precision      recall    f1-score     support
--------------------------------------------------------------
SHORT (0)         0.2467      0.6374      0.3558    31,467.0
HOLD (1)          0.6401      0.2303      0.3388    68,844.0
LONG (2)          0.2791      0.2327      0.2538    34,571.0
--------------------------------------------------------------
accuracy                                  0.3259   134,882.0
macro avg         0.3887      0.3668      0.3161   134,882.0
weighted avg      0.4558      0.3259      0.3210   134,882.0

Macierz pomyłek (dla zaakceptowanych predykcji):
               SHORT (0)    HOLD (1)    LONG (2)
   SHORT (0)      20,058       4,619       6,790
    HOLD (1)      39,006      15,858      13,980
    LONG (2)      22,230       4,298       8,043

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #4)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.49, HOLD=0.3, LONG=0.49
   Zaakceptowano: 107,373 / 281,101 (38.20%)
   Odrzucono: 173,728 / 281,101 (61.80%)

WYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:
               precision      recall    f1-score     support
--------------------------------------------------------------
SHORT (0)         0.2426      0.6384      0.3515    24,245.0
HOLD (1)          0.6401      0.2824      0.3919    56,164.0
LONG (2)          0.2872      0.2001      0.2359    26,964.0
--------------------------------------------------------------
accuracy                                  0.3421   107,373.0
macro avg         0.3900      0.3736      0.3264   107,373.0
weighted avg      0.4617      0.3421      0.3436   107,373.0

Macierz pomyłek (dla zaakceptowanych predykcji):
               SHORT (0)    HOLD (1)    LONG (2)
   SHORT (0)      15,478       4,619       4,148
    HOLD (1)      31,063      15,858       9,243
    LONG (2)      17,270       4,298       5,396

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #5)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.52, HOLD=0.3, LONG=0.52
   Zaakceptowano: 86,742 / 281,101 (30.86%)
   Odrzucono: 194,359 / 281,101 (69.14%)

WYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:
               precision      recall    f1-score     support
--------------------------------------------------------------
SHORT (0)         0.2408      0.6285      0.3481    19,142.0
HOLD (1)          0.6401      0.3421      0.4459    46,361.0
LONG (2)          0.2937      0.1659      0.2120    21,239.0
--------------------------------------------------------------
accuracy                                  0.3621    86,742.0
macro avg         0.3915      0.3788      0.3353    86,742.0
weighted avg      0.4671      0.3621      0.3670    86,742.0

Macierz pomyłek (dla zaakceptowanych predykcji):
               SHORT (0)    HOLD (1)    LONG (2)
   SHORT (0)      12,030       4,619       2,493
    HOLD (1)      24,520      15,858       5,983
    LONG (2)      13,417       4,298       3,524

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #6)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.55, HOLD=0.3, LONG=0.55
   Zaakceptowano: 71,066 / 281,101 (25.28%)
   Odrzucono: 210,035 / 281,101 (74.72%)

WYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:
               precision      recall    f1-score     support
--------------------------------------------------------------
SHORT (0)         0.2392      0.6021      0.3424    15,392.0
HOLD (1)          0.6401      0.4070      0.4976    38,961.0
LONG (2)          0.2995      0.1352      0.1863    16,713.0
--------------------------------------------------------------
accuracy                                  0.3853    71,066.0
macro avg         0.3929      0.3814      0.3421    71,066.0
weighted avg      0.4732      0.3853      0.3908    71,066.0

Macierz pomyłek (dla zaakceptowanych predykcji):
               SHORT (0)    HOLD (1)    LONG (2)
   SHORT (0)       9,268       4,619       1,505
    HOLD (1)      19,324      15,858       3,779
    LONG (2)      10,156       4,298       2,259
================================================================================
(venv) root@6208ee5b0954:/workspace/training# python main.py
2025-07-15 13:21:43.671018: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-07-15 13:21:43.671071: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-07-15 13:21:43.671853: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-07-15 13:21:43.676187: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-07-15 13:21:44,859 - training.utils - INFO - ============================================================
2025-07-15 13:21:44,860 - training.utils - INFO - 🎯 PODSUMOWANIE KONFIGURACJI TRENINGU 🎯
2025-07-15 13:21:44,860 - training.utils - INFO - ============================================================
2025-07-15 13:21:44,860 - training.utils - INFO - Dane Wejściowe:
2025-07-15 13:21:44,860 - training.utils - INFO -   - Plik: BTCUSDT-1m-futures_features_and_labels_FW-480_SL-050_TP-100.feather
2025-07-15 13:21:44,860 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:21:44,860 - training.utils - INFO - Parametry Modelu i Treningu:
2025-07-15 13:21:44,860 - training.utils - INFO -   - Długość Sekwencji: 120
2025-07-15 13:21:44,860 - training.utils - INFO -   - Podział Walidacyjny: 10%
2025-07-15 13:21:44,860 - training.utils - INFO -   - Podział Testowy: 10%
2025-07-15 13:21:44,860 - training.utils - INFO -   - Epoki: 5
2025-07-15 13:21:44,860 - training.utils - INFO -   - Rozmiar Batcha: 4096
2025-07-15 13:21:44,860 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:21:44,860 - training.utils - INFO - Parametry Predykcji:
2025-07-15 13:21:44,860 - training.utils - INFO -   - Progi Pewności: S=0.44, H=0.3, L=0.44
2025-07-15 13:21:44,860 - training.utils - INFO - ============================================================
2025-07-15 13:21:44,860 - training.utils - INFO - >>> KROK 1: Wczytywanie i Wstępne Przetworzenie Danych <<<
2025-07-15 13:21:45,080 - training.utils - INFO - Liczba wierszy po wczytaniu: 2,812,195
2025-07-15 13:21:45,089 - training.utils - INFO - Rozkład etykiet: 0 (SHORT): 27.49%, 1 (HOLD): 44.49%, 2 (LONG): 28.02%
2025-07-15 13:21:45,089 - training.utils - INFO - >>> KROK 2: Chronologiczny Podział na Zbiory <<<
2025-07-15 13:21:45,089 - training.utils - INFO - Podział danych: Trening: 2,249,756, Walidacja: 281,219, Test: 281,220
2025-07-15 13:21:45,089 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:21:45,089 - training.utils - INFO - Zakresy dat dla zbiorów:
2025-07-15 13:21:45,092 - training.utils - INFO -   - Treningowy:  od 2020-01-29 23:59 do 2024-05-10 09:04
2025-07-15 13:21:45,092 - training.utils - INFO -   - Walidacyjny: od 2024-05-10 09:05 do 2024-11-21 16:59
2025-07-15 13:21:45,093 - training.utils - INFO -   - Testowy:     od 2024-11-21 17:00 do 2025-06-04 23:59
2025-07-15 13:21:45,093 - training.utils - INFO - ------------------------------------------------------------
2025-07-15 13:21:45,093 - training.utils - INFO - >>> KROK 3: Skalowanie Cech Statycznych <<<
2025-07-15 13:21:45,111 - training.utils - INFO - Dopasowywanie skalera 'robust' na 5 cechach statycznych...
2025-07-15 13:21:45,327 - training.utils - INFO - Zapisano skaler do pliku: /workspace/training/output/models/scaler.pkl
2025-07-15 13:21:45,328 - training.utils - INFO - Transformacja zbiorów treningowego, walidacyjnego i testowego dla cech statycznych.
2025-07-15 13:21:45,391 - training.utils - INFO - >>> KROK 4: Przygotowanie Generatorów Danych (Architektura Hybrydowa) <<<

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
2025-07-15 13:21:45,476 - training.utils - INFO - Generatory danych (hybrydowe) zostały utworzone.
2025-07-15 13:21:45,477 - training.utils - INFO - >>> KROK 5: Budowanie i Kompilacja Modelu <<<
INFO: Budowanie modelu o architekturze: conv_lstm
2025-07-15 13:21:45.636116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 46866 MB memory:  -> device: 0, name: NVIDIA RTX A6000, pci bus id: 0000:46:00.0, compute capability: 8.6
INFO: Architektura modelu została zbudowana pomyślnie.
INFO: Używanie funkcji straty: categorical_crossentropy
2025-07-15 13:21:46,316 - training.utils - INFO - Model: "model"
2025-07-15 13:21:46,317 - training.utils - INFO - _________________________________________________________________
2025-07-15 13:21:46,317 - training.utils - INFO -  Layer (type)                Output Shape              Param #   
2025-07-15 13:21:46,317 - training.utils - INFO - =================================================================
2025-07-15 13:21:46,317 - training.utils - INFO -  input_1 (InputLayer)        [(None, 120, 9)]          0         
2025-07-15 13:21:46,317 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,317 - training.utils - INFO -  conv1d (Conv1D)             (None, 120, 64)           2944      
2025-07-15 13:21:46,317 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,317 - training.utils - INFO -  conv1d_1 (Conv1D)           (None, 120, 128)          41088     
2025-07-15 13:21:46,317 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,317 - training.utils - INFO -  max_pooling1d (MaxPooling1  (None, 60, 128)           0         
2025-07-15 13:21:46,317 - training.utils - INFO -  D)                                                              
2025-07-15 13:21:46,317 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,317 - training.utils - INFO -  dropout (Dropout)           (None, 60, 128)           0         
2025-07-15 13:21:46,317 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,317 - training.utils - INFO -  lstm (LSTM)                 (None, 64)                49408     
2025-07-15 13:21:46,318 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,318 - training.utils - INFO -  dropout_1 (Dropout)         (None, 64)                0         
2025-07-15 13:21:46,318 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,318 - training.utils - INFO -  dense (Dense)               (None, 3)                 195       
2025-07-15 13:21:46,318 - training.utils - INFO -                                                                  
2025-07-15 13:21:46,318 - training.utils - INFO - =================================================================
2025-07-15 13:21:46,318 - training.utils - INFO - Total params: 93635 (365.76 KB)
2025-07-15 13:21:46,318 - training.utils - INFO - Trainable params: 93635 (365.76 KB)
2025-07-15 13:21:46,318 - training.utils - INFO - Non-trainable params: 0 (0.00 Byte)
2025-07-15 13:21:46,318 - training.utils - INFO - _________________________________________________________________
2025-07-15 13:21:46,318 - training.utils - INFO - >>> KROK 6: Konfiguracja i Trening Modelu <<<
2025-07-15 13:21:46,318 - training.utils - INFO - >>> KROK 6: Konfiguracja i Trening Modelu <<<
2025-07-15 13:21:46,318 - training.utils - INFO - Utworzono podstawowe callbacki: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
2025-07-15 13:21:46,318 - training.utils - INFO - Nie zastosowano żadnych wag klas.
2025-07-15 13:21:46,318 - training.utils - INFO - Rozpoczynanie treningu modelu...
Epoch 1/5
2025-07-15 13:21:48.189437: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8907
2025-07-15 13:21:49.303752: I external/local_xla/xla/service/service.cc:168] XLA service 0x7f2bd0779330 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2025-07-15 13:21:49.303863: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA RTX A6000, Compute Capability 8.6
2025-07-15 13:21:49.313515: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1752585709.412753   66242 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
461/463 [============================>.] - ETA: 0s - loss: 1.0988 - accuracy: 0.3335   
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
463/463 [==============================] - ETA: 0s - loss: 1.0988 - accuracy: 0.3335
Epoch 1: val_loss improved from inf to 1.09736, saving model to /workspace/training/output/models/model.h5
/usr/local/lib/python3.11/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
463/463 [==============================] - 27s 50ms/step - loss: 1.0988 - accuracy: 0.3335 - val_loss: 1.0974 - val_accuracy: 0.4715 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 2/5
462/463 [============================>.] - ETA: 0s - loss: 1.0986 - accuracy: 0.3340  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
463/463 [==============================] - ETA: 0s - loss: 1.0986 - accuracy: 0.3340
Epoch 2: val_loss did not improve from 1.09736
463/463 [==============================] - 26s 54ms/step - loss: 1.0986 - accuracy: 0.3340 - val_loss: 1.0981 - val_accuracy: 0.2701 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 3/5
463/463 [==============================] - ETA: 0s - loss: 1.0986 - accuracy: 0.3344  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)

Epoch 3: val_loss did not improve from 1.09736
463/463 [==============================] - 26s 54ms/step - loss: 1.0986 - accuracy: 0.3344 - val_loss: 1.1003 - val_accuracy: 0.2698 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 4/5
462/463 [============================>.] - ETA: 0s - loss: 1.0986 - accuracy: 0.3343  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)

Epoch 4: val_loss did not improve from 1.09736
463/463 [==============================] - 25s 52ms/step - loss: 1.0986 - accuracy: 0.3343 - val_loss: 1.0995 - val_accuracy: 0.2698 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
Epoch 5/5
461/463 [============================>.] - ETA: 0s - loss: 1.0986 - accuracy: 0.3344  
Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
463/463 [==============================] - ETA: 0s - loss: 1.0986 - accuracy: 0.3344
Epoch 5: val_loss did not improve from 1.09736
463/463 [==============================] - 26s 54ms/step - loss: 1.0986 - accuracy: 0.3344 - val_loss: 1.0990 - val_accuracy: 0.2698 - lr: 0.0010

Rebalancing complete. New training samples for next epoch: 1896067. (635580 LONG, 628465 SHORT, 632022 HOLD)
2025-07-15 13:23:57,703 - training.utils - INFO - Trening zakończony.
2025-07-15 13:23:57,703 - training.utils - INFO - >>> KROK 7: Ostateczny Egzamin i Analiza Predykcji <<<
2025-07-15 13:23:57,704 - training.utils - INFO - Ocenianie modelu na zbiorze testowym...
69/69 [==============================] - 3s 41ms/step - loss: 1.0990 - accuracy: 0.2718
2025-07-15 13:24:00,675 - training.utils - INFO - Wyniki na zbiorze testowym - Strata: 1.0990, Dokładność: 0.2718
2025-07-15 13:24:00,676 - training.utils - INFO - Generowanie predykcji dla zbioru testowego...
69/69 [==============================] - 3s 37ms/step
2025-07-15 13:24:04,182 - training.utils - INFO - Predykcje zapisane w: /workspace/training/output/reports/test_predictions.csv
2025-07-15 13:24:04,182 - training.utils - INFO - Generowanie raportu końcowego (placeholder).
2025-07-15 13:24:04,183 - training.utils - INFO - Metadane zapisane w: /workspace/training/output/models/metadata.json
2025-07-15 13:24:04,264 - training.utils - INFO - Raport z ewaluacji zapisany w: /workspace/training/output/reports/evaluation_report.txt
2025-07-15 13:24:04,264 - training.utils - INFO - --- Proces treningowy zakończony pomyślnie! ---

================================================================================
🎯                           TRENING - RAPORT KOŃCOWY                           🎯
================================================================================
📊 DANE:
   Plik źródłowy: BTCUSDT-1m-futures_features_and_labels_FW-480_SL-050_TP-100.feather
   Całkowity czas: 0:02:19.404990
   Rozmiary zbiorów:
     - Treningowy: (2249756, 19)
     - Walidacyjny: (281219, 19)
     - Testowy: (281220, 19)
🧠 MODEL:
   Architektura: InputLayer -> Conv1D -> MaxPooling1D -> Dropout -> LSTM -> Dense
   Liczba parametrów: 93,635
   Zapisany w: model.h5

🏋️ WYNIKI TRENINGU:
   Liczba epok: 5 / 5
   Najlepsza epoka (wg val_loss): 1
   Końcowa strata walidacyjna: 1.0990
   Najlepsza strata walidacyjna: 1.0974
   Końcowa dokładność walidacyjna: 0.2698
   Najlepsza dokładność walidacyjna: 0.4715

⚖️ BALANSOWANIE I WAGI KLAS:
   - Metoda balansowania: Dynamiczny Undersampling (Callback)
   - Metoda wag: Wyłączona w konfiguracji

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #1)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.4, HOLD=0.3, LONG=0.4
   Zaakceptowano: 0 / 281,101 (0.00%)
   Odrzucono: 281,101 / 281,101 (100.00%)

Żadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #2)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.43, HOLD=0.3, LONG=0.43
   Zaakceptowano: 0 / 281,101 (0.00%)
   Odrzucono: 281,101 / 281,101 (100.00%)

Żadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #3)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.46, HOLD=0.3, LONG=0.46
   Zaakceptowano: 0 / 281,101 (0.00%)
   Odrzucono: 281,101 / 281,101 (100.00%)

Żadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #4)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.49, HOLD=0.3, LONG=0.49
   Zaakceptowano: 0 / 281,101 (0.00%)
   Odrzucono: 281,101 / 281,101 (100.00%)

Żadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #5)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.52, HOLD=0.3, LONG=0.52
   Zaakceptowano: 0 / 281,101 (0.00%)
   Odrzucono: 281,101 / 281,101 (100.00%)

Żadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.

--------------------------------------------------------------------------------
🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #6)
--------------------------------------------------------------------------------
THRESHOLDING - FILTROWANIE PREDYKCJI:
   Próg pewności: SHORT=0.55, HOLD=0.3, LONG=0.55
   Zaakceptowano: 0 / 281,101 (0.00%)
   Odrzucono: 281,101 / 281,101 (100.00%)

Żadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.
================================================================================