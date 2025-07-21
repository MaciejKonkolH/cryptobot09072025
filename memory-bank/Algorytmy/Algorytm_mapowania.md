# 🎯 Algorytm Mapowania Sekwencji na Etykiety

## Opis Algorytmu (Prostym Językiem)

### **Krok 1: Dane wejściowe**
Mamy **ciągłe dane czasowe** (minutowe świeczki):
- Rząd 0: minuty 00:00 - features + label
- Rząd 1: minuty 00:01 - features + label  
- Rząd 2: minuty 00:02 - features + label
- ...
- Rząd 2,855,520: ostatnia minuta - features + label

### **Krok 2: Class Balancing (Systematic Undersampling)**
Żeby zbalansować klasy, **wybieramy co 5. próbkę**:
- Zamiast wszystkich 2.8M próbek
- Bierzemy tylko: rząd 120, 125, 130, 135, 140, 145...
- To daje nam **selected_target_indices** = [120, 125, 130, 135, ...]

### **Krok 3: Tworzenie sekwencji LSTM** 
Dla każdego wybranego momentu (target_idx) tworzymy **sekwencję 120 kroków**:

**PRZYKŁAD:**
- **Target moment**: rząd 125 (minuta 00:125)
- **Sekwencja**: rzędy 5 do 124 (minuty 00:05 do 00:124) 
- **Etykieta**: label z rzędu 125

### **Krok 4: Mapowanie**
Model dostaje:
- **X (input)**: sekwencję features z minut 00:05-00:124 
- **Y (target)**: label z minuty 00:125
- Model ma przewidzieć co się stanie w 00:125 na podstawie historii 00:05-00:124

---

## 🔍 Identyfikacja Potencjalnego Problemu

**Problem może być w tym, że:**

1. **Systematic undersampling** wybiera sparse momenty (co 5. próbka)
2. **Ale sequence generator** używa tych indices w **ciągłych danych**
3. **Rezultat**: sekwencja i label mogą pochodzić z **różnych kontekstów czasowych**

### **Kluczowe Pytania do Weryfikacji:**
- Czy **selected_target_indices** to rzeczywiście indeksy w oryginalnych danych?
- Czy **sekwencja** bierze się z **ciągłych** oryginalnych danych?  
- Czy **etykieta** bierze się z **tego samego momentu** co koniec sekwencji?

### **Symptomy Błędu:**
- Accuracy 11.2% (znacznie gorsza niż losowa 33.3%)
- Duże feature jumps między końcem sekwencji a momentem target
- Model uczy się fałszywych wzorców z błędnych par (sekwencja, etykieta)

---

## ✅ Status Weryfikacji
- [x] Algorytm opisany i zrozumiany
- [ ] Testy implementowane  
- [ ] Błędy zidentyfikowane
- [ ] Poprawki wprowadzone
- [ ] Weryfikacja końcowa

*Dokument utworzony: 2025-01-08*
*Status: Potwierdzone rozumienie algorytmu* 