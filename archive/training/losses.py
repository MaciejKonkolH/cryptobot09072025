"""
Moduł zawierający niestandardowe funkcje straty dla modeli Keras.
"""
import tensorflow as tf
import keras.backend as K

def categorical_focal_loss(alpha, gamma=2.0):
    """
    Wersja 'softmax' funkcji Focal Loss.

    Ta funkcja straty została zaprojektowana, aby rozwiązać problem niezbalansowania klas
    podczas treningu. Obniża ona wagę straty dla dobrze sklasyfikowanych przykładów (z dużą
    pewnością), pozwalając modelowi skupić się na trudnych, błędnie klasyfikowanych przykładach.

    Argumenty:
        alpha (list lub tensor): Współczynniki wagowe dla każdej z klas. Powinien to być
                                 tensor o kształcie (num_classes, 1) lub lista.
                                 Np. [[0.25], [0.5], [0.25]]
        gamma (float): Parametr skupienia. Wartości > 0 zmniejszają względną stratę
                       dla dobrze sklasyfikowanych przykładów. Zalecana wartość to 2.0.

    Zwraca:
        Funkcję straty, która może być użyta w Keras.
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        """
        Właściwa funkcja obliczająca stratę.

        :param y_true: Rzeczywiste etykiety (one-hot encoded).
        :param y_pred: Predykcje modelu (wynik z warstwy softmax).
        :return: Tensor ze stratą.
        """
        # Konwertujemy etykiety y_true na float32, aby pasowały do typu y_pred.
        y_true = tf.cast(y_true, tf.float32)

        # Zabezpieczenie przed wartościami NaN i Inf przez obcięcie predykcji
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Oblicz p_t - prawdopodobieństwo dla prawdziwej klasy
        p_t = tf.reduce_sum(y_true * y_pred, axis=1)

        # Oblicz standardową cross-entropię, ale tylko dla prawdziwej klasy
        cross_entropy = -K.log(p_t)

        # Oblicz czynnik modulujący Focal Loss
        loss_modulator = K.pow(1 - p_t, gamma)

        # Oblicz końcową stratę
        # Wyodrębnij odpowiednią wagę alpha dla prawdziwej klasy
        alpha_t = tf.reduce_sum(y_true * alpha, axis=1)
        loss = alpha_t * loss_modulator * cross_entropy

        # Zwracamy średnią stratę w batchu, a nie sumę
        return tf.reduce_mean(loss)

    return focal_loss_fixed 