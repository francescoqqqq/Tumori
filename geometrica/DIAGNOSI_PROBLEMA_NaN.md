# DIAGNOSI PROBLEMA NaN - Dataset 502

**Data:** 2025-12-13
**Training analizzato:** Dataset502_Shapes_One con nnUNetTrainerGeometric
**Versione codice:** V2.3 (con protezioni anti-NaN)

---

## Sintomi

Training log mostra:
```
Epoca 0-4: train_loss OK (da -0.38 a -0.89) ✅
Epoca 5: train_loss NaN ❌ PRIMO NaN
Epoca 6-99: train_loss NaN ❌ Pesi corrotti permanentemente
```

---

## Root Cause Analysis

### 1. Timing del Problema

Il NaN compare **esattamente all'epoca 5**, che è quando:
- Termina il warm-up (epoche 0-4: solo Dice+CE)
- Si attiva la loss geometrica per la prima volta

**PROBLEMA CRITICO:** Il file nel venv aveva `warmup=5` invece di `warmup=20`!
- File locale: `/workspace/geometrica/nnUNetTrainerGeometric.py` → `warmup=20` ✅
- File venv: `/venv/.../nnUNetTrainerGeometric.py` → `warmup=5` ❌

### 2. Perché le Protezioni V2.3 Non Funzionano

La versione V2.3 include protezioni aggressive:
- Controllo area minima (50.0 pixel)
- Clamp aggressivi su tutti i valori intermedi
- Epsilon grandi nei denominatori (1e-2, 1e-4)
- Controlli finali NaN in ogni funzione loss
- Safety checks nel trainer per gradienti NaN

**MA:** Il NaN persiste per 95 epoche consecutive, il che significa che:

1. **I safety checks nel trainer NON stanno funzionando correttamente**
   - Dovrebbero skippare l'optimizer step quando rilevano NaN
   - Ma i pesi continuano a essere corrotti

2. **Il problema è nei gradienti esplosivi, non nei valori delle loss**
   - La sqrt(x) ha derivata d/dx = 1/(2*sqrt(x))
   - Quando x → 0, il gradiente → ∞
   - Anche con clamp(x, min=1e-4), se il batch contiene predizioni quasi vuote,
     i gradienti possono esplodere durante il backward

3. **Dataset 502 è più difficile del 501**
   - Più batch hanno predizioni quasi vuote (specialmente all'inizio)
   - Batch misti: alcuni con area > 50, altri con area < 50
   - I batch validi calcolano loss, ma i gradienti dei batch quasi vuoti esplodono

### 3. Perché Dataset 502 è Diverso

Differenze osservate:
- Dataset 501: Immagini più facili, rete impara rapidamente
- Dataset 502: Immagini più complesse, la rete fa fatica all'inizio

Quando la geometric loss si attiva all'epoca 5 (troppo presto!):
- La rete sta ancora imparando la segmentazione base
- Molte predizioni sono ancora quasi vuote o molto incerte
- La loss geometrica cerca di calcolare proprietà su "blob" quasi vuoti
- I gradienti esplodono anche se la loss stessa è protetta

---

## Soluzioni Proposte (Ordine di Priorità)

### IMMEDIATO - Fix Critico

1. **Copia il file corretto nel venv**
   ```bash
   cp /workspace/geometrica/nnUNetTrainerGeometric.py \
      /venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/
   ```
   Questo porta il warmup da 5 a 20 epoche.

2. **Copia anche geometric_losses.py aggiornato**
   ```bash
   cp /workspace/geometrica/geometric_losses.py \
      /venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/
   ```

3. **Ri-allena con warmup corretto (20 epoche)**

### SHORT TERM - Protezioni Aggiuntive (V2.4)

Se anche con warmup=20 il problema persiste, implementare **V2.4**:

#### A. Protezione Gradienti più Aggressiva in sqrt

**Problema attuale:**
```python
grad_mag_squared = grad_x**2 + grad_y**2
grad_mag_squared = torch.clamp(grad_mag_squared, min=1e-4)
grad_mag = torch.sqrt(grad_mag_squared)
```

Il gradiente di sqrt(1e-4) è ancora 1/(2*sqrt(1e-4)) = 50, troppo grande!

**Soluzione V2.4:**
```python
# PROTEZIONE PIÙ AGGRESSIVA: min=1e-2 invece di 1e-4
# Gradiente sqrt(1e-2) = 1/(2*0.1) = 5 (molto più stabile)
grad_mag_squared = torch.clamp(grad_x**2 + grad_y**2, min=1e-2)
grad_mag = torch.sqrt(grad_mag_squared)
```

Lo stesso vale per `aspect_loss`:
```python
# V2.3: discriminant clampato a min=1e-4
discriminant = torch.clamp(trace**2 - 4*det, min=1e-4)
sqrt_term = torch.sqrt(discriminant)

# V2.4: min=1e-2 per gradienti più stabili
discriminant = torch.clamp(trace**2 - 4*det, min=1e-2)
sqrt_term = torch.sqrt(discriminant)
```

#### B. Gestione Batch Misti

**Problema:** Se un batch ha 4 campioni con area=200 e 4 con area=10:
- I 4 campioni validi calcolano loss normalmente
- I 4 campioni quasi vuoti hanno gradienti instabili
- Il backward esplode a causa dei campioni problematici

**Soluzione V2.4:**
```python
def _vectorized_compactness_loss(self, pred_soft):
    area = pred_soft.sum(dim=(1, 2))  # [B]

    # NUOVO: Crea mask per batch validi
    valid_mask = area >= 50.0  # [B] boolean

    # NUOVO: Se NESSUN batch è valido, ritorna 0.0
    if not valid_mask.any():
        return pred_soft.sum() * 0.0

    # Calcola loss SOLO sui batch validi
    pred_soft_valid = pred_soft[valid_mask]  # [B_valid, H, W]

    # ... calcola compactness solo su pred_soft_valid ...

    # Media solo sui batch validi
    return loss_valid.mean()
```

Questo evita completamente i calcoli su batch problematici.

#### C. Controlli Più Robusti nel Trainer

**Problema attuale:** I safety checks rilevano NaN ma non fermano correttamente il training.

**Soluzione V2.4:**
```python
# Nel train_step, PRIMA del backward:
if torch.isnan(total_loss) or torch.isinf(total_loss):
    print("WARNING: total_loss è NaN, skipping step")
    self.optimizer.zero_grad(set_to_none=True)
    return {'loss': torch.tensor(0.0).cpu().numpy()}

# Aggiungi TRY-CATCH attorno al backward
try:
    total_loss.backward()
except RuntimeError as e:
    if 'nan' in str(e).lower():
        print("WARNING: backward failed with NaN")
        self.optimizer.zero_grad(set_to_none=True)
        return {'loss': torch.tensor(0.0).cpu().numpy()}
    raise

# Dopo backward, verifica gradienti PRIMA di clip_grad_norm
has_nan_grad = False
for param in self.network.parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        has_nan_grad = True
        break

if has_nan_grad:
    print("WARNING: NaN in gradienti, skipping optimizer step")
    self.optimizer.zero_grad(set_to_none=True)
    return {'loss': torch.tensor(0.0).cpu().numpy()}

# Solo ora chiama clip_grad_norm e optimizer.step
torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
self.optimizer.step()
```

### LONG TERM - Approccio Alternativo

Se il problema persiste anche con V2.4, considerare:

1. **Aumentare ulteriormente il warmup**
   - warmup=50 epoche invece di 20
   - La rete deve imparare bene la segmentazione base prima di aggiungere vincoli geometrici

2. **Ridurre ulteriormente i pesi delle loss geometriche**
   - weight_compactness=0.005 (invece di 0.01)
   - weight_boundary=0.0025 (invece di 0.005)
   - weight_aspect=0.0025 (invece di 0.005)

3. **Attivazione graduale della loss geometrica**
   ```python
   # Invece di on/off netto all'epoca 20, usa ramp-up graduale
   if self.current_epoch < warmup_epochs:
       geometric_weight = 0.0
   elif self.current_epoch < warmup_epochs + 20:
       # Ramp-up da 0.0 a 1.0 in 20 epoche
       geometric_weight = (self.current_epoch - warmup_epochs) / 20.0
   else:
       geometric_weight = 1.0

   loss_geometric = geometric_weight * self.geometric_loss(output_softmax)
   ```

---

## Piano d'Azione

### STEP 1: Fix Immediato (5 minuti)
1. Copia i file corretti nel venv
2. Verifica che warmup=20
3. Ri-allena Dataset 502

### STEP 2: Monitoraggio (durante training)
- Osserva se il NaN appare ancora
- Se sì, a quale epoca?
- Leggi i log per capire se i safety checks funzionano

### STEP 3: Se Problema Persiste
Implementa V2.4 con:
- Protezione sqrt più aggressiva (min=1e-2)
- Gestione batch misti (esclusione batch problematici)
- Try-catch nel backward

### STEP 4: Se Ancora Problemi
- Aumenta warmup a 50
- Riduci pesi geometric di 2x
- Implementa ramp-up graduale

---

## Note Tecniche

### Perché sqrt è Problematico

Il gradiente di sqrt esplode quando x → 0:
```
f(x) = sqrt(x)
f'(x) = 1 / (2*sqrt(x))

x = 1e-4 → f'(x) = 50
x = 1e-6 → f'(x) = 500
x = 1e-8 → f'(x) = 5000
```

Anche con clamp, se il backward propaga gradienti attraverso sqrt(1e-4),
il gradiente viene moltiplicato per 50, causando esplosione.

### Soluzione: Clamp Più Alto

Con clamp(x, min=1e-2):
```
x = 1e-2 → f'(x) = 5  (molto più stabile!)
```

Questo è il fix chiave della V2.4.

---

**Fine documento**
