# Analiza Porównawcza Logiki: Trening vs. Backtesting

Data rozpoczęcia: 2025-06-28

## Cel Główny

Zidentyfikować fundamentalną różnicę w logice przetwarzania danych między pipeline'em treningowym/walidacyjnym a pipeline'em predykcji na żywo w strategii Freqtrade, która powoduje rozbieżności w predykcjach na tych samych danych wejściowych.

---

## Hipoteza Robocza

Mimo że pliki modelu i skalera są te same, a surowe cechy wejściowe są identyczne, musi istnieć subtelna różnica w sposobie przygotowania danych *bezpośrednio przed* wywołaniem `scaler.transform()` lub `model.predict_proba()`. Potencjalne przyczyny:
- Różna kolejność kolumn w DataFrame.
- Różnice w typach danych (`float32` vs `float64`).
- Niejawna modyfikacja DataFrame przez jedną z bibliotek (pandas, numpy) w jednym ze środowisk.
- Inny sposób obsługi "okna" danych (`window_size`) przekazywanego do modelu.

---

## Faza 1: Analiza Ścieżki Treningowej/Walidacyjnej

W tej fazie zbadamy kod odpowiedzialny za generowanie predykcji w procesie walidacji.

### Pliki do analizy:
- `validation_and_labeling/model_predictor.py` (lub podobny)
- `validation_and_labeling/main.py` (lub skrypt główny)
- `validation_and_labeling/feature_calculator.py` (już przeanalizowany, ale warty ponownego przejrzenia w kontekście)

### Punkty do zbadania:
1.  **Ładowanie modelu i skalera:**
    - [ ] Jakie ścieżki są używane?
    - [ ] W którym momencie procesu są ładowane?
2.  **Przygotowanie danych do predykcji:**
    - [ ] Jak tworzone są "okna" danych (`window_size`)?
    - [ ] Czy kolejność kolumn jest jawnie ustawiana/sprawdzana?
    - [ ] Czy jest wykonywana jakaś konwersja typów danych?
3.  **Wywołanie predykcji:**
    - [ ] Dokładna linia kodu wywołująca `scaler.transform()`.
    - [ ] Dokładna linia kodu wywołująca `model.predict_proba()`.
    - [ ] Jakie są kształty (`shape`) danych wejściowych do obu metod?

### Notatki z Analizy:
*   ...

---

## Faza 2: Analiza Ścieżki Backtestingu (Freqtrade)

W tej fazie zbadamy kod strategii Freqtrade.

### Pliki do analizy:
- `ft_bot_clean/user_data/strategies/Enhanced_ML_MA43200_Buffer_Strategy.py`

### Punkty do zbadania:
1.  **Odbiór danych od Freqtrade:**
    - [ ] W jakiej formie `dataframe` jest dostępny w metodzie `populate_indicators` lub podobnej?
2.  **Ładowanie modelu i skalera:**
    - [ ] Jakie ścieżki są używane?
    - [ ] Czy model/skaler są ładowane raz, czy przy każdej świecy?
3.  **Przygotowanie danych do predykcji:**
    - [ ] Gdzie i jak są brane cechy z `dataframe`?
    - [ ] Jak tworzone są "okna" danych?
    - [ ] Czy kolejność kolumn jest jawnie ustawiana/sprawdzana?
4.  **Wywołanie predykcji:**
    - [ ] Dokładna linia kodu wywołująca `scaler.transform()`.
    - [ ] Dokładna linia kodu wywołująca `model.predict_proba()`.
    - [ ] Jakie są kształty (`shape`) danych wejściowych?

### Notatki z Analizy:
*   ...

---

## Faza 3: Porównanie i Wnioski

| Krok procesu                  | Ścieżka Treningu/Walidacji | Ścieżka Backtestingu (Freqtrade) | Zgodność? (✅/❌) |
| ----------------------------- | -------------------------- | -------------------------------- | ---------------- |
| **Kolejność kolumn wejściowych** | ???                        | ???                              | ???              |
| **Typy danych (dtype)**       | ???                        | ???                              | ???              |
| **Kształt danych dla skalera**  | ???                        | ???                              | ???              |
| **Wynik `scaler.transform()`**  | (do testu)                 | (do testu)                       | ???              |
| **Kształt danych dla modelu**   | ???                        | ???                              | ???              |
| **Wynik `predict_proba()`**     | Różny                      | Różny                            | ❌               |

### Ostateczna diagnoza:
*   ... 