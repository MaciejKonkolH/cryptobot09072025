# 🎯 SEQUENCE-AWARE UNDERSAMPLING ALGORITHM

## 📖 **OPIS DLA CZŁOWIEKA - INTUICYJNY**

### **🔍 Problem do rozwiązania:**
W danych finansowych czasowych mamy problem podwójny:
1. **Class imbalance** - za dużo etykiet HOLD (70%), za mało SHORT/LONG (15% każda)
2. **Temporal continuity** - LSTM potrzebuje ciągłych sekwencji czasowych żeby się uczyć

### **❌ Dlaczego tradycyjny undersampling nie działa:**
Klasyczny undersampling usuwa co N-tą próbkę (np. co 5-tą). To niszczy ciągłość czasową:
- Oryginał: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
- Po undersampling: [t0, t5] 
- LSTM dostaje sekwencję [t0, t5] - ale to nie są kolejne minuty!

### **✅ Nasz algorytm:**
**Kluczowa idea:** Rozdzielamy problem balansowania od problemu sekwencji.

**Jak to działa:**
1. **Oryginalne dane zostają nietknięte** - pełna historia czasowa [t0, t1, t2, ..., tN]
2. **Undersampling wybiera tylko KTÓRE momenty czasowe użyć do treningu** 
3. **Dla każdego wybranego momentu, sekwencja input jest zawsze z oryginalnych danych**

**Przykład:**
- Undersampling wybrał moment t500 do treningu
- Target (y) = etykieta z t500
- Sequence (X) = poprzednie 120 minut z ORYGINALNYCH danych [t380, t381, t382, ..., t499]
- Model dostaje naturalną, ciągłą sekwencję czasową!

### **🎯 Efekt:**
- ✅ Eliminujemy nadmiar nudnych momentów HOLD (balansowanie klas)
- ✅ Zachowujemy naturalne wzorce czasowe (LSTM może się uczyć)
- ✅ Model widzi rzeczywiste przejścia między stanami rynku

---

## 🤖 **OPIS DLA AGENTÓW AI - TECHNICZNY**

### **ALGORITHM: Sequence-Aware Temporal Undersampling**

**INPUT:**
- `time_series_data`: Array[N, features] - complete temporal dataset
- `labels`: Array[N] - labels for each timestamp
- `window_size`: int - LSTM sequence length (e.g., 120)
- `target_balance`: Dict[class] -> float - desired class distribution

**CORE PRINCIPLE:**
Decouple sampling strategy from sequence generation. Sample target indices while preserving temporal continuity for feature sequences.

**ALGORITHM STEPS:**

1. **TARGET SELECTION PHASE:**
   ```
   eligible_indices = range(window_size, N)  # indices that can have full history
   class_counts = count_labels_per_class(labels[eligible_indices])
   minority_class_size = min(class_counts.values())
   
   selected_indices = []
   for class_id in classes:
       class_indices = find_indices_for_class(eligible_indices, class_id)
       target_count = calculate_target_count(minority_class_size, target_balance[class_id])
       sampled_indices = systematic_sample(class_indices, target_count)
       selected_indices.extend(sampled_indices)
   
   selected_indices.sort()  # maintain temporal order of training examples
   ```

2. **SEQUENCE GENERATION PHASE:**
   ```
   for target_idx in selected_indices:
       sequence_start = target_idx - window_size
       sequence_end = target_idx
       
       X_sequence = time_series_data[sequence_start:sequence_end]  # ORIGINAL data
       y_target = labels[target_idx]  # SAMPLED target
       
       training_batch.append((X_sequence, y_target))
   ```

**KEY PROPERTIES:**
- `X_sequences` are always temporally continuous from original dataset
- `y_targets` are class-balanced through selective sampling
- No data interpolation or synthetic generation
- Maintains causal relationships in temporal patterns
- Preserves LSTM's ability to learn temporal dependencies

**MATHEMATICAL GUARANTEE:**
```
∀ sequence X_i: X_i = time_series_data[j:j+window_size] for some j
∀ consecutive elements in X_i: temporal_gap = 1 time_unit
Class_distribution(sampled_targets) ≈ target_balance
```

**COMPLEXITY:**
- Time: O(N) for sampling + O(M * window_size) for sequence generation, where M = selected samples
- Space: O(M * window_size) for training data
- Original dataset memory: O(N * features) - unchanged

**ADVANTAGES:**
1. Resolves class imbalance without destroying temporal patterns
2. LSTM receives natural progression of market states
3. Model can learn transition patterns between classes
4. No artificial data generation or time-series interpolation
5. Maintains causality: past → present relationships intact

**USE CASE:**
Optimal for time-series classification where:
- Class imbalance exists in temporal data
- Sequential patterns are crucial for prediction
- Model architecture expects continuous temporal input (LSTM/GRU/Transformer)
- Temporal gaps would destroy learning signal
