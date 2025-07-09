"""
Signal Generator - Generowanie sygnałów ML per para

OPTIMIZED VERSION V2.0:
- Batch predictions (1000x faster)
- Memory cleanup system
- GPU memory management
- Progress tracking z ETA
- Hybrid mode: batch dla backtest, single dla live

Odpowiedzialny za:
- Generowanie sygnałów ML per para z różnymi window_size
- Przygotowanie sekwencji dla modeli
- Normalizacja danych przez scaler
- Error handling w przypadku problemów z predykcją
"""

import logging
import numpy as np
import pandas as pd
import gc
import time
from typing import Dict, Optional, Tuple, Any

# GPU Memory Management
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Smart progress tracker z ETA calculation"""
    
    def __init__(self, total_items: int, task_name: str = "Processing"):
        self.total = total_items
        self.task_name = task_name
        self.start_time = time.time()
        self.processed = 0
        self.last_update = 0
    
    def update(self, processed_count: int):
        """Update progress z smart ETA calculation"""
        self.processed = processed_count
        current_time = time.time()
        
        # Update co 1000 items lub co 10 sekund
        if (self.processed - self.last_update >= 1000) or (current_time - self.last_update >= 10):
            elapsed = current_time - self.start_time
            
            if self.processed > 0 and elapsed > 0:
                items_per_second = self.processed / elapsed
                remaining_items = self.total - self.processed
                eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
                eta_minutes = eta_seconds / 60
                
                progress_pct = (self.processed / self.total) * 100
                
                logger.info(f"📊 {self.task_name}: {self.processed:,}/{self.total:,} ({progress_pct:.1f}%)")
                logger.info(f"⏱️ Speed: {items_per_second:.1f}/sec | ETA: {eta_minutes:.1f} min")
                
                self.last_update = self.processed


def setup_gpu_memory():
    """Configure GPU memory growth to prevent OOM"""
    if not TF_AVAILABLE:
        return False
        
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optional: Set memory limit (4GB safe for 6GB GPU)
                # tf.config.experimental.set_memory_limit(gpu, 4096)
                
            logger.info(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            return True
        else:
            logger.info("ℹ️ No GPU devices found")
            return False
    except Exception as e:
        logger.warning(f"⚠️ GPU setup failed: {e}")
        return False


def cleanup_memory(step_name: str = "batch"):
    """Comprehensive memory cleanup"""
    try:
        # Force garbage collection
        collected = gc.collect()
        
        # TensorFlow cleanup if available
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
        
        logger.debug(f"🧹 Memory cleanup after {step_name}: {collected} objects collected")
        
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup failed: {e}")


class SignalGenerator:
    """Generuje sygnały ML dla strategii multi-pair - OPTIMIZED VERSION"""
    
    def __init__(self):
        """Inicjalizacja Signal Generator z optimizations"""
        self.logger = logging.getLogger(__name__)
        
        # 🎯 CONFIGURABLE THRESHOLDS - domyślne wartości
        self.short_threshold = 0.42
        self.long_threshold = 0.42
        self.hold_threshold = 0.30
        
        # Optimization parameters
        self.batch_size = 1000  # Configurable batch size
        self.cleanup_frequency = 10  # Cleanup co 10 batches
        self.gpu_configured = False
        
        # 📊 CSV PREDICTIONS LOGGER SETUP
        self.csv_file_path = None
        self.csv_logging_enabled = True  # Can be disabled for performance
        
        # Feature columns
        self.feature_columns = [
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200',
            'volume_to_ma1440', 'volume_to_ma43200'
        ]
        
        # Setup GPU and CSV logger
        self._setup_gpu()
        self._setup_csv_logger()
    
    def set_thresholds(self, short_threshold: float = None, long_threshold: float = None, hold_threshold: float = None):
        """
        🎯 Ustawia thresholdy confidence dla sygnałów ML
        
        Args:
            short_threshold: Próg pewności dla SHORT (np. 0.42)
            long_threshold: Próg pewności dla LONG (np. 0.42) 
            hold_threshold: Próg pewności dla HOLD (np. 0.30)
        """
        if short_threshold is not None:
            self.short_threshold = short_threshold
        if long_threshold is not None:
            self.long_threshold = long_threshold
        if hold_threshold is not None:
            self.hold_threshold = hold_threshold
            
        self.logger.info(f"✅ Updated ML thresholds: SHORT={self.short_threshold}, LONG={self.long_threshold}, HOLD={self.hold_threshold}")
    
    def _is_backtest_mode(self, dataframe: pd.DataFrame) -> bool:
        """
        Prosta heurystyka do wykrywania trybu backtest.
        Ramki danych w trybie live są zazwyczaj znacznie krótsze.
        """
        # Uznajemy za backtest, jeśli ramka ma więcej niż 500 świec.
        # To bezpieczny próg, który odróżnia pełny backtest od danych live.
        return len(dataframe) > 500

    def _add_default_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Dodaje puste/domyślne kolumny sygnałów w przypadku błędu."""
        dataframe['ml_signal'] = 1  # HOLD
        dataframe['ml_confidence'] = 0.0
        dataframe['ml_short_prob'] = 0.0
        dataframe['ml_hold_prob'] = 0.0
        dataframe['ml_long_prob'] = 0.0
        dataframe['ml_buy_prob'] = 0.0
        dataframe['ml_sell_prob'] = 0.0
        return dataframe

    def _setup_gpu(self):
        """Initialize GPU settings once"""
        if not self.gpu_configured:
            self.gpu_configured = setup_gpu_memory()
    
    def _setup_csv_logger(self):
        """Setup CSV logger for predictions"""
        if not self.csv_logging_enabled:
            return
            
        import os
        import csv
        from datetime import datetime
        
        try:
            # Utwórz folder logs
            logs_dir = "user_data/logs"
            os.makedirs(logs_dir, exist_ok=True)
            
            # Nazwa pliku z timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file_path = f"{logs_dir}/ml_predictions_{timestamp}.csv"
            
            # Utwórz plik z headerem
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'pair', 'chunk_id', 'pred_idx', 'short_prob', 'hold_prob', 
                    'long_prob', 'best_class', 'confidence', 'final_signal', 
                    'threshold_short', 'threshold_long', 'timestamp'
                ])
            
            self.logger.info(f"📊 CSV predictions logging enabled: {self.csv_file_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup CSV logging: {e}")
            self.csv_logging_enabled = False
    
    def generate_ml_signals(self, 
                           dataframe: pd.DataFrame, 
                           pair: str, 
                           model: Any, 
                           scaler: Any, 
                           window_size: int) -> pd.DataFrame:
        """
        Główna funkcja generowania sygnałów.
        NOWA LOGIKA V4: Otrzymuje dataframe z gotowymi cechami.
        """
        try:
            # 🔥 KRYTYCZNY LOG DODANY NA ŻĄDANIE UŻYTKOWNIKA 🔥
            if not dataframe.empty and 'date' in dataframe.columns:
                first_date = dataframe['date'].iloc[0]
                logger.info(f"🔥 KRYTYCZNA DIAGNOSTYKA DATY: Data pierwszej świecy w ramce danych tuż przed predykcją: {first_date}")
            else:
                logger.warning("🔥 KRYTYCZNA DIAGNOSTYKA DATY: Ramka danych jest pusta lub nie zawiera kolumny 'date'.")
                
            # 1. Wyodrębnij gotowe cechy z dataframe
            feature_data = dataframe[self.feature_columns].values
            
            if not self._validate_feature_data(feature_data, pair):
                return self._add_default_signals(dataframe)

            # 2. Wygeneruj predykcje (logika batch/live jest w środku)
            predictions = self._generate_predictions(
                feature_data=feature_data,
                model=model,
                scaler=scaler,
                window_size=window_size,
                pair=pair,
                is_backtest=self._is_backtest_mode(dataframe),
                debug_timestamps=dataframe['date']
            )

            if predictions is None:
                return self._add_default_signals(dataframe)

            # 3. Dodaj predykcje i sygnały do dataframe
            dataframe = self._add_predictions_to_dataframe(dataframe, predictions, window_size, pair)
            
            # 4. Konwertuj prawdopodobieństwa na finalny sygnał (0, 1, 2)
            dataframe['ml_signal'] = self._convert_probabilities_to_signals(
                dataframe[['ml_short_prob', 'ml_hold_prob', 'ml_long_prob']].values,
                pair,
                timestamps=dataframe['date']
            )

        except Exception as e:
            logger.error(f"❌ Critical error in SignalGenerator for {pair}: {e}", exc_info=True)
            return self._add_default_signals(dataframe)

        return dataframe

    def _generate_predictions(self, feature_data, model, scaler, window_size, pair, is_backtest, debug_timestamps):
        """Nowa funkcja łącząca logikę live i backtest."""
        if is_backtest:
            return self._generate_predictions_batch(feature_data, model, scaler, window_size, pair, debug_timestamps)
        else:
            return self._generate_predictions_live(feature_data, model, scaler, window_size, pair)

    def _generate_predictions_batch(self, 
                                  feature_data: np.ndarray, 
                                  model: Any, 
                                  scaler: Any, 
                                  window_size: int, 
                                  pair: str,
                                  debug_timestamps=None) -> Optional[np.ndarray]:
        """OPTIMIZED V3: Scaled, sequenced, and chunked batch predictions."""
        try:
            # 1. Sprawdź, czy mamy wystarczająco danych
            if len(feature_data) < window_size:
                logger.warning(f"⚠️ {pair}: Not enough data for prediction ({len(feature_data)} < {window_size})")
                return None

            logger.debug(f"🚀 {pair}: Starting BATCH ML predictions...")
            
            # 2. Skaluj cały zbiór danych JEDEN RAZ
            logger.debug(f"   🔬 Scaling {len(feature_data)} rows of feature data...")
            scaled_feature_data = scaler.transform(feature_data)
            logger.debug(f"   ✅ Scaling complete. Shape: {scaled_feature_data.shape}")
            
            # 3. Stwórz wszystkie sekwencje JEDEN RAZ
            # 🔧 KLUCZOWA POPRAWKA: Używaj danych z przeszłości [t-120:t] zamiast [t:t+120]
            logger.debug(f"   🧠 Preparing sequences...")
            
            # Sprawdź czy mamy wystarczająco danych
            if len(scaled_feature_data) < window_size:
                logger.warning(f"⚠️ {pair}: Not enough data for sequences ({len(scaled_feature_data)} < {window_size})")
                return None
            
            # Tworzymy sekwencje poprawnie: dla każdego punktu t używamy danych [t-window_size:t]
            sequences = []
            for i in range(window_size, len(scaled_feature_data) + 1):
                # Sekwencja dla punktu i używa danych z przeszłości [i-window_size:i]
                sequence = scaled_feature_data[i-window_size:i]
                sequences.append(sequence)
            
            sequences = np.array(sequences)
            
            total_sequences = len(sequences)
            if total_sequences == 0:
                logger.warning(f"⚠️ {pair}: No sequences generated, cannot predict.")
                return None
            
            logger.debug(f"   ✅ Prepared {total_sequences:,} sequences for prediction.")

            # 4. Przetwarzaj sekwencje w chunkach, aby oszczędzać pamięć
            chunk_size = self._calculate_optimal_chunk_size(total_sequences)
            all_predictions = []
            
            # Użyj nowej nazwy, żeby było jasne co jest śledzone
            progress_tracker = ProgressTracker(total_items=total_sequences, task_name=f"{pair} BATCH")

            for i in range(0, total_sequences, chunk_size):
                chunk_sequences = sequences[i:i + chunk_size]
                
                batch_predictions = model.predict(chunk_sequences, batch_size=self.batch_size, verbose=0)
                all_predictions.append(batch_predictions)
                
                # Aktualizuj tracker na podstawie przetworzonych sekwencji
                progress_tracker.update(i + len(chunk_sequences))

            if not all_predictions:
                logger.warning(f"⚠️ {pair}: No predictions generated after chunking.")
                return None
            
            # Połącz wszystkie predykcje
            combined_predictions = np.concatenate(all_predictions, axis=0)
            logger.debug(f"✅ {pair}: All chunks processed. Combined predictions shape: {combined_predictions.shape}")
            
            # Weryfikacja
            if len(combined_predictions) != total_sequences:
                logger.error(f"❌ {pair}: Final prediction count ({len(combined_predictions)}) mismatch with expected sequences ({total_sequences})!")

            return combined_predictions

        except Exception as e:
            logger.error(f"❌ {pair}: Error in batch predictions: {e}", exc_info=True)
            cleanup_memory("batch_error")
            return None
    
    def _calculate_optimal_chunk_size(self, total_predictions: int) -> int:
        """Calculate optimal chunk size based on dataset size and available memory"""
        # Conservative chunk sizes to prevent OOM
        if total_predictions < 50000:
            return 50000 # Zawsze używaj co najmniej tego chunka
        elif total_predictions < 500000:
            return 50000  # 50k chunks - safe for 4GB GPU
        elif total_predictions < 1000000:
            return 75000  # 75k chunks for medium datasets
        else:
            return 100000  # 100k chunks for large datasets (5+ year backtests)
    
    def _generate_predictions_live(self, 
                                 feature_data: np.ndarray, 
                                 model: Any, 
                                 scaler: Any, 
                                 window_size: int, 
                                 pair: str) -> Optional[np.ndarray]:
        """Generate prediction for the latest candle (live mode)"""
        try:
            # We need the last `window_size` candles
            if len(feature_data) < window_size:
                logger.warning(f"⚠️ {pair}: Not enough data for live prediction ({len(feature_data)} < {window_size})")
                return None

            # Get the last sequence
            last_sequence_raw = feature_data[-window_size:]
            
            # 🎯 KLUCZOWA POPRAWKA: Skalowanie sekwencji
            logger.debug(f"   🔬 Scaling live sequence (shape: {last_sequence_raw.shape})...")
            last_sequence_scaled = scaler.transform(last_sequence_raw)
            logger.debug(f"   ✅ Live sequence scaled.")
            
            # Reshape for prediction
            sequence_input = last_sequence_scaled.reshape(1, window_size, len(self.feature_columns))
            
            # Predict
            prediction = model.predict(sequence_input, verbose=0)
            return prediction

        except Exception as e:
            logger.error(f"❌ {pair}: Error during live prediction: {e}")
            return None
    
    def _validate_feature_data(self, feature_data: np.ndarray, pair: str) -> bool:
        """Validate feature data for NaN or Inf values"""
        try:
            # Sprawdź shape
            if feature_data.shape[1] != len(self.feature_columns):
                logger.error(f"❌ {pair}: Wrong feature count: {feature_data.shape[1]} != {len(self.feature_columns)}")
                return False
            
            # Sprawdź NaN values
            if np.any(np.isnan(feature_data)):
                logger.warning(f"⚠️ {pair}: NaN values detected in features")
                return False
            
            # Sprawdź infinite values
            if np.any(np.isinf(feature_data)):
                logger.warning(f"⚠️ {pair}: Infinite values detected in features")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error validating feature data: {e}")
            return False
    
    def _add_predictions_to_dataframe(self, 
                                    df: pd.DataFrame, 
                                    predictions: np.ndarray, 
                                    window_size: int, 
                                    pair: str) -> pd.DataFrame:
        """Dodaje predykcje do dataframe"""
        try:
            # Sprawdź czy predictions mają odpowiedni format
            if predictions.shape[1] != 3:  # [SHORT, HOLD, LONG]
                logger.error(f"❌ {pair}: Invalid prediction shape: {predictions.shape}")
                return df
            
            # Indeksy gdzie dodajemy predykcje (pomijamy pierwsze window_size punktów)
            start_idx = window_size - 1
            end_idx = start_idx + len(predictions)
            
            if end_idx > len(df):
                logger.warning(f"⚠️ {pair}: Prediction count mismatch, truncating")
                end_idx = len(df)
                predictions = predictions[:end_idx-start_idx]
            
            # Dodaj probability scores - NOWY FORMAT 3 kolumny [SHORT, HOLD, LONG]
            df.loc[start_idx:end_idx-1, 'ml_short_prob'] = predictions[:, 0]  # SHORT
            df.loc[start_idx:end_idx-1, 'ml_hold_prob'] = predictions[:, 1]   # HOLD
            df.loc[start_idx:end_idx-1, 'ml_long_prob'] = predictions[:, 2]   # LONG
            
            # Zachowaj backward compatibility
            df.loc[start_idx:end_idx-1, 'ml_buy_prob'] = predictions[:, 2]    # LONG = BUY
            df.loc[start_idx:end_idx-1, 'ml_sell_prob'] = predictions[:, 0]   # SHORT = SELL
            
            # Oblicz confidence jako max probability
            df.loc[start_idx:end_idx-1, 'ml_confidence'] = np.max(predictions, axis=1)
            
            # 🎯 NOWE: Pobierz faktyczne timestamps świec dla CSV
            timestamps = None
            if 'date' in df.columns:
                timestamps = df.loc[start_idx:end_idx-1, 'date'].values
            
            # 🎯 FIXED: skip_csv_logging nie powinno blokować finalnego CSV logging
            # Chunked batches już nie logują podczas przetwarzania (wyłączone w _process_chunked_batches)
            # Tutaj logujemy finalne, połączone rezultaty
            df.loc[start_idx:end_idx-1, 'ml_signal'] = self._convert_probabilities_to_signals(
                predictions, pair, chunk_id=0, timestamps=timestamps  # Works for both single batch and combined chunks
            )
            
            logger.debug(f"✅ {pair}: Predictions added to dataframe ({start_idx}:{end_idx})")
            logger.debug(f"📊 {pair}: Added 3-column probabilities: SHORT, HOLD, LONG + backward compatibility")

            # --------------------------------------------------------------------
            # --- ⚠️ KRYTYCZNA MODYFIKACJA: TWARDE PRZESUNIĘCIE PREdykcji ⚠️ ---
            # --------------------------------------------------------------------
            # Na wyraźne polecenie użytkownika, wszystkie predykcje są przesuwane
            # o 119 minut do tyłu (shift(-119)), aby obejść uporczywy problem
            # niezgodności czasowej.
            # TO JEST ROZWIĄZANIE TYMCZASOWE, NIE STANDARDOWE.
            # --------------------------------------------------------------------
            shift_value = -119
            self.logger.critical(f"🔥🔥🔥 UWAGA: Aktywna jest twarda modyfikacja przesunięcia predykcji o {shift_value} minut! 🔥🔥🔥")
            
            prediction_cols = [
                'ml_signal', 'ml_confidence', 
                'ml_short_prob', 'ml_hold_prob', 'ml_long_prob',
                'ml_buy_prob', 'ml_sell_prob'
            ]
            
            for col in prediction_cols:
                if col in df.columns:
                    df[col] = df[col].shift(shift_value)

            # Wypełnienie luk po przesunięciu
            # Sygnał ustawiamy na HOLD, prawdopodobieństwa na 0
            df['ml_signal'].fillna(1, inplace=True) # 1 = HOLD
            df.fillna({col: 0.0 for col in prediction_cols if col != 'ml_signal'}, inplace=True)

            return df
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error adding predictions to dataframe: {e}")
            return df
    
    def _convert_probabilities_to_signals(self, predictions: np.ndarray, pair: str, chunk_id: int = 0, timestamps=None) -> np.ndarray:
        """
        🎯 OPTIMIZED V2: Konwertuje prawdopodobieństwa na sygnały i przygotowuje dane do logowania.
        """
        try:
            short_threshold = self.short_threshold
            long_threshold = self.long_threshold

            signals = []
            log_entries = []  # Lista do zbierania logów

            for i, pred in enumerate(predictions):
                best_class = np.argmax(pred)
                confidence = pred[best_class]
                final_signal = 1  # Domyślnie HOLD

                if best_class == 0 and confidence > short_threshold:
                    final_signal = 0  # SHORT
                elif best_class == 2 and confidence > long_threshold:
                    final_signal = 2  # LONG
                
                signals.append(final_signal)

                # Zbierz dane do logu
                log_entry = {
                    'pair': pair,
                    'chunk_id': chunk_id,
                    'pred_idx': i,
                    'short_prob': pred[0],
                    'hold_prob': pred[1],
                    'long_prob': pred[2],
                    'best_class': best_class,
                    'confidence': confidence,
                    'final_signal': final_signal,
                    'threshold_short': short_threshold,
                    'threshold_long': long_threshold,
                    'timestamp': timestamps[i] if timestamps is not None and i < len(timestamps) else None
                }
                log_entries.append(log_entry)
            
            # Zapisz wszystkie logi do CSV za jednym razem
            self._log_predictions_to_csv_bulk(log_entries)

            signals_array = np.array(signals, dtype=int)
            
            unique, counts = np.unique(signals_array, return_counts=True)
            signal_stats = dict(zip(unique, counts))
            
            logger.info(f"📊 {pair} ML SIGNALS CONVERSION:")
            logger.info(f"   📈 Input predictions: {len(predictions)}")
            logger.info(f"   🎯 Output signals: SHORT={signal_stats.get(0, 0)}, HOLD={signal_stats.get(1, 0)}, LONG={signal_stats.get(2, 0)}")
            logger.info(f"   ✅ Using thresholds from config: SHORT > {short_threshold}, LONG > {long_threshold}")
            
            return signals_array
            
        except Exception as e:
            logger.error(f"❌ Error converting probabilities to signals for {pair}: {e}", exc_info=True)
            return np.ones(len(predictions), dtype=int)

    def _log_predictions_to_csv_bulk(self, log_entries: list):
        """Zapisuje listę logów predykcji do pliku CSV za jednym razem."""
        if not self.csv_logging_enabled or not log_entries:
            return
        
        try:
            # Konwertuj listę słowników na DataFrame
            log_df = pd.DataFrame(log_entries)
            
            logger.info(f"📊 Saving {len(log_entries)} predictions to {self.csv_file_path}...")
            # Zapisz do CSV w trybie 'append' i bez headera
            log_df.to_csv(self.csv_file_path, mode='a', header=False, index=False, encoding='utf-8')

        except Exception as e:
            logger.warning(f"⚠️ Failed to write predictions to CSV: {e}")

    def get_latest_signal(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        🎯 POPRAWIONE: Pobiera najnowszy sygnał ML (0=SHORT, 1=HOLD, 2=LONG)
        
        Args:
            df: DataFrame z sygnałami
            pair: Nazwa pary
            
        Returns:
            Dict: Sygnał z interpretacją klas
        """
        try:
            if len(df) == 0:
                return {
                    'signal': 1,  # HOLD jako default
                    'confidence': 0.5,
                    'short_prob': 0.33, 'hold_prob': 0.34, 'long_prob': 0.33,
                    'action': 'HOLD'
                }
            
            # Pobierz najnowszy sygnał
            latest_signal = df['ml_signal'].iloc[-1]
            latest_confidence = df['ml_confidence'].iloc[-1]
            
            # Pobierz raw probabilities jeśli dostępne
            short_prob = df['ml_short_prob'].iloc[-1] if 'ml_short_prob' in df.columns else 0.33
            hold_prob = df['ml_hold_prob'].iloc[-1] if 'ml_hold_prob' in df.columns else 0.34  
            long_prob = df['ml_long_prob'].iloc[-1] if 'ml_long_prob' in df.columns else 0.33
            
            # Interpretuj sygnał
            if latest_signal == 0:
                action = 'SHORT'
            elif latest_signal == 2:
                action = 'LONG'
            else:  # latest_signal == 1
                action = 'HOLD'
            
            return {
                'signal': int(latest_signal),  # 0, 1, lub 2
                'confidence': float(latest_confidence),
                'short_prob': float(short_prob),
                'hold_prob': float(hold_prob), 
                'long_prob': float(long_prob),
                'action': action
            }
                
        except Exception as e:
            self.logger.error(f"❌ {pair}: Error getting latest signal: {e}")
            return {
                'signal': 1,  # HOLD jako fallback
                'confidence': 0.5,
                'short_prob': 0.33, 'hold_prob': 0.34, 'long_prob': 0.33,
                'action': 'HOLD'
            }
    
    def get_optimization_stats(self) -> Dict:
        """Zwraca statystyki optymalizacji (jeśli istnieją)"""
        return {
            "prediction_cache_hits": self.prediction_cache_hits,
            "total_predictions": self.total_predictions
        } 