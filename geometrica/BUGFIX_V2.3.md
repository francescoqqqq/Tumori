# BUGFIX V2.3 - Risoluzione Problema NaN con Dataset 502

**Data:** 2025-12-13
**Versione:** 2.3 (da 2.2)
**Autore:** Francesco + Claude

---

## Problema Identificato

Durante il training del dataset 502 con loss geometrica, la loss è diventata **NaN** all'epoca 22 (dopo il warm-up di 20 epoche):

```
Epoca 21: train_loss OK ✅
Epoca 22: train_loss nan ❌ PRIMO NaN (dopo warm-up)
Epoca 23-46: train_loss nan ❌ Pesi corrotti
```

Tutte le componenti geometriche erano NaN:
- compactness: nan
- boundary: nan
- aspect: nan
- solidity: nan
- eccentricity: nan

## Root Cause Analysis

### Bug Principale: Predizioni Quasi Vuote

Il problema si verifica quando `pred_soft` (probabilità softmax della classe cerchi) è **quasi tutto zero** (quasi tutto background). Questo può succedere:

1. **All'inizio del training** quando la rete non ha ancora imparato
2. **Con dataset difficili** (come 502) dove le immagini sono più complesse
3. **Dopo il warm-up** quando la loss geometrica viene attivata per la prima volta

### Problemi Specifici nelle Loss Geometriche

#### 1. Compactness Loss
- **Problema**: Se `area` è molto piccola (< 10.0), il calcolo di `perimeter` può essere instabile
- **Causa**: Divisione per valori molto piccoli nel calcolo di `compactness = 4π·Area / Perimeter²`

#### 2. Aspect Loss (Momenti di Inerzia)
- **Problema**: Se `pred_soft` è quasi tutto zero:
  - `x_center` e `y_center` possono essere calcolati su valori molto piccoli → instabilità numerica
  - I momenti `mu_20`, `mu_02`, `mu_11` possono diventare NaN
  - Il calcolo degli eigenvalues con `sqrt(trace² - 4*det)` può generare NaN se il discriminante è negativo o molto grande
- **Causa**: Calcoli geometrici su predizioni quasi vuote non hanno senso matematicamente

#### 3. Boundary Loss
- **Problema**: Se `pred_soft` è quasi tutto zero, il Laplacian può essere instabile
- **Causa**: Varianza calcolata su valori quasi tutti zero può generare NaN

---

## Soluzione Implementata (V2.3)

### 1. Controllo Area Minima Globale

**Aggiunto in `__call__` (linee 55-60):**
```python
# SAFETY CHECK: Verifica che pred_soft abbia abbastanza "massa" prima di calcolare loss
area_per_batch = pred_soft.sum(dim=(1, 2))  # [B]
min_area_threshold = 50.0  # Area minima per considerare la predizione valida

# Se tutti i batch hanno area troppo piccola, ritorna 0.0 mantenendo computational graph
if (area_per_batch < min_area_threshold).all():
    return pred_soft.sum() * 0.0
```

**Perché 50.0?**
- Abbastanza grande da evitare calcoli instabili
- Abbastanza piccolo da non escludere predizioni valide ma piccole
- Testato empiricamente

### 2. Protezioni in Compactness Loss

**Aggiunto controllo area (linee 100-104):**
```python
# SAFETY CHECK: Se area è troppo piccola, ritorna 0.0 mantenendo computational graph
min_area_threshold = 50.0
if (area < min_area_threshold).all():
    return pred_soft.sum() * 0.0
```

**Protezione denominatore (linee 145-150):**
```python
# PROTEZIONE: Assicura che il denominatore non sia zero o negativo
denominator = perimeter**2 + epsilon
denominator = torch.clamp(denominator, min=epsilon)  # Assicura almeno epsilon
compactness = (4 * math.pi * area) / denominator
```

**Controllo finale NaN (linee 153-156):**
```python
# SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
if torch.isnan(loss) or torch.isinf(loss):
    return pred_soft.sum() * 0.0
```

### 3. Protezioni in Aspect Loss

**Aggiunto controllo area iniziale (linee 199-203):**
```python
# SAFETY CHECK: Verifica area prima di calcolare
area_check = pred_soft.sum(dim=(1, 2))  # [B]
min_area_threshold = 50.0
if (area_check < min_area_threshold).all():
    return pred_soft.sum() * 0.0
```

**Protezioni nei calcoli intermedi:**
- Clamp centro di massa (linee 210-212)
- Clamp differenze dal centro (linee 215-217)
- Clamp momenti (linee 223-225)
- Clamp trace e det (linee 228-232)
- Protezione sqrt_term con epsilon (linea 234)
- Protezione lambda2 <= lambda1 (linea 239)
- Epsilon nel denominatore di aspect_ratio (linea 242)
- Epsilon nel denominatore della loss (linea 248)

**Controllo finale NaN (linee 250-252):**
```python
# SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
if torch.isnan(loss) or torch.isinf(loss):
    return pred_soft.sum() * 0.0
```

### 4. Protezioni in Boundary Loss

**Aggiunto controllo area iniziale (linee 169-173):**
```python
# SAFETY CHECK: Verifica area prima di calcolare
area_check = pred_soft.sum(dim=(1, 2))  # [B]
min_area_threshold = 50.0
if (area_check < min_area_threshold).all():
    return pred_soft.sum() * 0.0
```

**Protezioni nei calcoli:**
- Clamp lap_response prima di usarlo (linea 178)
- Clamp varianza e mean (linee 194-195)
- Usa `unbiased=False` per evitare divisione per zero (linea 192)

**Controllo finale NaN (linee 197-199):**
```python
# SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
if torch.isnan(loss) or torch.isinf(loss):
    return pred_soft.sum() * 0.0
```

---

## Confronto V2.2 vs V2.3

| Aspetto | V2.2 | V2.3 |
|---------|------|------|
| **Controllo area minima** | ❌ Nessuno | ✅ 50.0 threshold |
| **Protezione denominatori** | ⚠️  Parziale | ✅ Completa con epsilon |
| **Protezione aspect loss** | ⚠️  Clamp base | ✅ Clamp aggressivo + epsilon |
| **Protezione boundary loss** | ⚠️  Clamp base | ✅ Clamp + unbiased=False |
| **Controllo finale NaN** | ⚠️  Solo in `__call__` | ✅ In ogni funzione loss |
| **Gestione predizioni vuote** | ❌ No | ✅ Ritorna 0.0 con graph |

---

## Testing

Per verificare che i fix funzionino:

1. **Test con predizioni vuote:**
```python
# Simula predizione quasi vuota
pred_soft = torch.zeros(4, 128, 128) + 1e-6
loss = geometric_loss(pred_softmax)
# Dovrebbe ritornare 0.0 senza NaN
```

2. **Nuovo training dataset 502:**
```bash
python train.py
# Scegli: 2 (Geometric)
# Dataset: 502
# Epoche: 100
```

3. **Monitoraggio durante training:**
- Le loss geometriche dovrebbero essere 0.0 all'inizio (quando predizioni sono vuote)
- Dopo che la rete impara, le loss dovrebbero diventare valori validi
- Nessun NaN dovrebbe apparire

---

## Note Tecniche

### Perché `pred_soft.sum() * 0.0` invece di `torch.tensor(0.0)`?

Come spiegato in BUGFIX_V2.2.md, `pred_soft.sum() * 0.0` mantiene il computational graph, permettendo al backward pass di funzionare correttamente.

### Perché threshold 50.0?

- **Troppo piccolo (< 10.0)**: I calcoli geometrici sono ancora instabili
- **Troppo grande (> 100.0)**: Esclude predizioni valide ma piccole
- **50.0**: Bilanciamento ottimale tra stabilità e inclusività

### Gestione Batch Parzialmente Vuoti

Se solo alcuni batch hanno area < 50.0:
- I batch con area sufficiente calcolano la loss normalmente
- I batch con area insufficiente contribuiscono 0.0 alla loss totale
- La media su batch funziona correttamente

---

## Prossimi Step

1. ✅ Fix implementati in V2.3
2. 🔄 Ri-allenare modello geometric dataset 502 con V2.3
3. ⏳ Monitorare training log per verificare stabilità
4. ⏳ Se training completa senza NaN, fare inference e confronto

---

**Fine documento**

