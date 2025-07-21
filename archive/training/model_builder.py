"""
Moduł odpowiedzialny za budowanie i kompilowanie modelu Keras.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from training import config as trainer_config

# Import naszej nowej funkcji straty
from losses import categorical_focal_loss

def build_model(input_shape, cfg):
    """
    Buduje model Keras na podstawie konfiguracji.
    Obsługuje różne architektury zdefiniowane w pliku konfiguracyjnym.
    """
    model_type = cfg.ARCHITECTURE.lower()
    print(f"INFO: Budowanie modelu o architekturze: {model_type}")

    inputs = Input(shape=input_shape)
    x = inputs

    if model_type == 'conv_lstm':
        # Architektura hybrydowa Conv1D -> LSTM
        # 1. Blok konwolucyjny do ekstrakcji cech
        for filters in cfg.CONV1D_FILTERS:
            x = Conv1D(filters=filters,
                       kernel_size=cfg.CONV1D_KERNEL_SIZE,
                       activation='relu',
                       padding='causal')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(cfg.DROPOUT_RATE)(x)

        # 2. Blok rekurencyjny do analizy sekwencji cech
        x = LSTM(units=cfg.LSTM_UNITS, return_sequences=False)(x)
        x = Dropout(cfg.DROPOUT_RATE)(x)

    elif model_type == 'dense':
        # Architektura prostej sieci gęstej (Dense)
        x = Flatten()(x)
        for units in cfg.DENSE_UNITS:
            x = Dense(units, activation='relu', kernel_regularizer=l2(cfg.L2_REG_STRENGTH))(x)
            x = Dropout(cfg.DROPOUT_RATE)(x)
            
    else:
        raise ValueError(f"Nieznana architektura w config.py: '{cfg.ARCHITECTURE}'")

    # 3. Warstwa wyjściowa (wspólna dla obu architektur)
    outputs = Dense(cfg.NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    print("INFO: Architektura modelu została zbudowana pomyślnie.")
    return model

def compile_model(model, cfg_obj):
    """Kompiluje model z zadaną funkcją straty i optymalizatorem."""
    
    # --- Logika wyboru funkcji straty ---
    if cfg_obj.LOSS_FUNCTION == 'focal_loss':
        print(f"INFO: Używanie funkcji straty: Focal Loss (gamma={cfg_obj.FOCAL_LOSS_GAMMA}, alpha={cfg_obj.FOCAL_LOSS_ALPHA})")
        loss_func = categorical_focal_loss(alpha=cfg_obj.FOCAL_LOSS_ALPHA, gamma=cfg_obj.FOCAL_LOSS_GAMMA)
    else:
        print("INFO: Używanie funkcji straty: categorical_crossentropy")
        loss_func = 'categorical_crossentropy'
        
    model.compile(
        optimizer=Adam(learning_rate=cfg_obj.LEARNING_RATE),
        loss=loss_func,
        metrics=['accuracy']
    )
    return model 