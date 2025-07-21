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
        # Zabezpieczenie przed wartościami NaN i Inf przez obcięcie predykcji
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Obliczenie standardowej straty cross-entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Obliczenie Focal Loss
        # Waga (alpha) jest mnożona przez (1 - p_t)^gamma, co redukuje stratę
        # dla przykładów, gdzie p_t (prawdopodobieństwo) jest wysokie.
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sumowanie strat dla batcha
        return K.sum(loss, axis=1)

    return focal_loss_fixed 