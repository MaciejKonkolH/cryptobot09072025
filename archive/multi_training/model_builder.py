"""
Moduł odpowiedzialny za budowanie i kompilowanie modelu Keras.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from training import config as trainer_config

def build_model(input_shape: tuple) -> Sequential:
    """
    Buduje i kompiluje model LSTM na podstawie parametrów z pliku konfiguracyjnego.

    Args:
        input_shape (tuple): Kształt danych wejściowych (sequence_length, num_features).

    Returns:
        Sequential: Skompilowany model Keras, gotowy do treningu.
    """
    model = Sequential()

    # Pierwsza warstwa LSTM
    # return_sequences=True jest potrzebne, gdy kolejna warstwa to też LSTM
    model.add(LSTM(
        units=trainer_config.LSTM_UNITS[0], 
        return_sequences=True, 
        input_shape=input_shape
    ))
    model.add(Dropout(trainer_config.DROPOUT_RATE))

    # Ukryte warstwy LSTM
    # Ostatnia warstwa LSTM nie potrzebuje return_sequences=True
    for i, units in enumerate(trainer_config.LSTM_UNITS[1:]):
        is_last_lstm = (i == len(trainer_config.LSTM_UNITS) - 2)
        model.add(LSTM(units, return_sequences=not is_last_lstm))
        model.add(Dropout(trainer_config.DROPOUT_RATE))

    # Gęste warstwy (Dense)
    for units in trainer_config.DENSE_UNITS:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(trainer_config.DROPOUT_RATE))

    # Warstwa wyjściowa
    # 3 neurony (dla SHORT, HOLD, LONG) i aktywacja softmax
    model.add(Dense(3, activation='softmax'))

    # Kompilacja modelu
    optimizer = Adam(learning_rate=trainer_config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model 