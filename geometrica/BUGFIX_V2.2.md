# BUGFIX V2.2 - Risoluzione Problema NaN

**Data:** 2025-12-10
**Versione:** 2.2 (da 2.1)
**Autore:** Francesco + Claude

---

## Problema Identificato

Durante il training dell'epoca **76**, la loss è improvvisamente diventata **NaN**, corrompendo tutti i pesi della rete:

```
Epoca 75: train_loss -0.8433  ✅ OK - Best Dice: 0.9622
Epoca 76: train_loss nan       ❌ PRIMO NaN
Epoca 77-99: train_loss nan    ❌ Pesi corrotti
```

## Root Cause Analysis

### Bug 1: Computational Graph Interrotto

**Codice problematico (V2.1):**
```python
# CHECK NaN: Se una loss è NaN, la setta a 0
if torch.isnan(loss_compact) or torch.isinf(loss_compact):
    loss_compact = torch.tensor(0.0, device=pred_soft.device, dtype=pred_soft.dtype)
```

**Problema:**
`torch.tensor(0.0)` crea un **leaf tensor senza computational graph**. Quando la loss diventa NaN e viene sostituita con questo tensor:
1. Il backward pass **non propaga gradienti** attraverso quel branch
2. I gradienti delle altre loss diventano instabili
3. NaN si propaga comunque nella rete

**Soluzione (V2.2):**
```python
# IMPORTANTE: Mantiene computational graph per gradient flow
if torch.isnan(loss_compact) or torch.isinf(loss_compact):
    loss_compact = pred_soft.sum() * 0.0  # Mantiene computational graph!
```

`pred_soft.sum() * 0.0`:
- ✅ Risultato = 0.0
- ✅ Mantiene il computational graph
- ✅ Permette gradient flow corretto
- ✅ Non interrompe il backward pass

### Bug 2: Nessun Controllo Pre-Backward

Il trainer non controllava se la loss totale era NaN **prima** di chiamare `.backward()`:

```python
# V2.1 - PROBLEMA
total_loss = loss_dice_ce + loss_geometric
total_loss.backward()  # Se NaN, corrompe tutti i gradienti!
```

**Soluzione (V2.2):**
```python
# V2.2 - FIX
if torch.isnan(total_loss) or torch.isinf(total_loss):
    print(f"⚠️  WARNING: Loss totale è NaN/Inf!")
    print(f"   Dice+CE: {loss_dice_ce.item()}")
    print(f"   Geometric: {loss_geometric.item()}")
    total_loss = loss_dice_ce  # Fallback su Dice+CE
```

### Bug 3: Nessun Controllo Post-Backward

Anche con gradient clipping (12), se i gradienti sono NaN, l'optimizer li propaga nei pesi:

```python
# V2.1 - PROBLEMA
total_loss.backward()
torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
self.optimizer.step()  # Propaga NaN nei pesi!
```

**Soluzione (V2.2):**
```python
total_loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)

# SAFETY CHECK: Verifica gradienti per NaN
has_nan_grad = False
for name, param in self.network.named_parameters():
    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
        has_nan_grad = True
        print(f"⚠️  WARNING: Gradient NaN/Inf in {name}")
        break

if has_nan_grad:
    print(f"⚠️  WARNING: Skipping optimizer step due to NaN gradients.")
    self.optimizer.zero_grad(set_to_none=True)  # Azzera gradienti corrotti
else:
    self.optimizer.step()  # Procedi solo se gradienti validi
```

---

## Modifiche Implementate

### 1. Fix Computational Graph in `geometric_losses.py`

**File modificati:**
- `/workspace/geometrica/geometric_losses.py`
- `/venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/geometric_losses.py`

**Linee modificate:**
```python
# PRIMA (V2.1) - linee 61-66, 81-82
if torch.isnan(loss_compact) or torch.isinf(loss_compact):
    loss_compact = torch.tensor(0.0, device=pred_soft.device, dtype=pred_soft.dtype)

# DOPO (V2.2)
if torch.isnan(loss_compact) or torch.isinf(loss_compact):
    loss_compact = pred_soft.sum() * 0.0  # Mantiene computational graph!
```

Applicato a:
- `loss_compact`
- `loss_bound`
- `loss_aspect`
- `total` loss

### 2. Safety Checks in `nnUNetTrainerGeometric.py`

**Controllo Pre-Backward (linee 203-217):**
```python
if torch.isnan(total_loss) or torch.isinf(total_loss):
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING [Epoch {self.current_epoch}]: Loss totale è NaN/Inf!")
    print(f"   Dice+CE loss: {loss_dice_ce.item()}")
    print(f"   Geometric loss: {loss_geometric.item()}")
    print(f"   Componenti geometric: {geom_components}")
    print(f"{'='*70}\n")
    total_loss = loss_dice_ce  # Fallback
```

**Controllo Post-Backward (linee 228-240):**
```python
has_nan_grad = False
for name, param in self.network.named_parameters():
    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
        has_nan_grad = True
        print(f"\n⚠️  WARNING: Gradient NaN/Inf in {name}")
        break

if has_nan_grad:
    print(f"⚠️  WARNING: Skipping optimizer step due to NaN gradients.")
    self.optimizer.zero_grad(set_to_none=True)
else:
    self.optimizer.step()
```

### 3. Logging Migliorato

**Logging periodico ogni 10 epoche (linee 265-276):**
```python
if self.current_epoch % 10 == 0 and self.use_geometric_loss:
    print(f"\n{'='*70}")
    print(f"[EPOCH {self.current_epoch}] Loss Breakdown:")
    print(f"  Dice+CE:    {avg_dice_ce:.6f}")
    print(f"  Geometric:  {avg_geometric:.6f}")
    print(f"  Total:      {avg_total:.6f}")
    print(f"  Components: compactness={...}, boundary={...}, aspect={...}")
    print(f"{'='*70}\n")
```

**Logging exception geometric loss (linea 197):**
```python
except Exception as e:
    print(f"⚠️  WARNING [Epoch {self.current_epoch}]: Geometric loss failed: {e}")
    loss_geometric = torch.tensor(0.0, device=self.device)
```

---

## Testing

Per verificare che i fix funzionino:

1. **Test gradient flow:**
```bash
python /workspace/geometrica/geometric_losses.py
```
Output atteso:
```
✅✅✅ GRADIENT FLOW OK!
I gradienti sono presenti e non-zero - la rete può imparare!
```

2. **Nuovo training:**
```bash
python train.py
# Scegli: 2 (Geometric)
# Epoche: 100
```

3. **Monitoraggio durante training:**
- Ogni 10 epoche: Stampa loss breakdown
- Se NaN appare: Warning immediato con dettagli
- Se gradienti NaN: Skip optimizer step invece di corrompere pesi

---

## Confronto V2.1 vs V2.2

| Aspetto | V2.1 | V2.2 |
|---------|------|------|
| **NaN handling** | `torch.tensor(0.0)` | `pred_soft.sum() * 0.0` |
| **Computational graph** | ❌ Interrotto | ✅ Mantenuto |
| **Gradient flow** | ❌ Bloccato se NaN | ✅ Sempre attivo |
| **Check pre-backward** | ❌ Nessuno | ✅ Fallback su Dice+CE |
| **Check post-backward** | ❌ Nessuno | ✅ Skip step se NaN |
| **Logging** | ⚠️  Minimo | ✅ Dettagliato ogni 10 epoche |
| **Crash con NaN** | ❌ Sì (epoca 76) | ✅ No (gestito) |

---

## Prossimi Step

1. ✅ Fix implementati in V2.2
2. 🔄 Ri-allenare modello geometric con V2.2
3. ⏳ Monitorare training log per verificare stabilità
4. ⏳ Se training completa senza NaN, fare inference e confronto

---

## Note Tecniche

### Perché `pred_soft.sum() * 0.0` funziona?

```python
# Esempio computational graph:

# SBAGLIATO:
loss = torch.tensor(0.0)  # Leaf node, no parents
loss.backward()  # Non propaga nulla

# CORRETTO:
loss = pred_soft.sum() * 0.0
# Graph: pred_soft -> sum() -> mul(0.0) -> loss
loss.backward()  # Propaga gradienti correttamente (anche se zero)
```

Il punto chiave è che **il computational graph rimane intatto**, permettendo al backward pass di funzionare correttamente su tutte le altre loss.

### Gradient Clipping

Il gradient clipping (`clip_grad_norm_(params, 12)`) **non protegge da NaN**:
- Clampa la **norma** dei gradienti tra [-12, 12]
- Ma se gradiente = NaN, clamp(NaN) = NaN ancora
- Serve solo per gradienti molto grandi ma validi

---

**Fine documento**
