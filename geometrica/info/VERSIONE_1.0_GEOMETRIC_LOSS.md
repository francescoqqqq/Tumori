# Geometric Loss - Versione 1.0

**Progetto:** Segmentazione cerchi con nnU-Net + Loss Geometrica
**Autori:** Francesco + Claude
**Data:** Dicembre 2025
**Versione:** 1.0 (stabile)

---

## Panoramica

Questo progetto estende nnU-Net con **loss geometriche differenziabili** per migliorare la qualità della segmentazione di forme circolari. La loss geometrica guida la rete a produrre cerchi più regolari, compatti e con bordi più lisci.

### Datasets

- **Dataset 501**: Multi-cerchi (2-5 cerchi per immagine)
- **Dataset 502**: Singolo cerchio (1 cerchio per immagine)

Entrambi i dataset funzionano con la stessa configurazione di loss geometrica.

---

## Architettura della Soluzione

### File Principali

```
geometrica/
├── geometric_losses.py           # Loss geometriche V2.4 (differenziabili)
├── nnUNetTrainerGeometric.py     # Trainer custom nnU-Net
├── data_geom.py                  # Generazione dataset sintetici
├── convert_to_nnunet.py          # Conversione formato nnU-Net
├── train.py                      # Script training interattivo
├── run_inference.py              # Inference sui test set
└── test.py                       # Analisi risultati e confronti
```

### Loss Geometriche Implementate

La classe `DifferentiableGeometricLossesV2` implementa 3 loss differenziabili:

1. **Compactness Loss** (peso: 0.01)
   - Metrica: `4π·Area / Perimetro²`
   - Valore ideale cerchio: 1.0
   - Implementazione: Soft area + gradiente Sobel per perimetro

2. **Boundary Smoothness Loss** (peso: 0.005)
   - Metrica: Varianza del Laplaciano
   - Penalizza bordi frastagliati e irregolari
   - Implementazione: Convoluzione Laplaciana

3. **Aspect Ratio Loss** (peso: 0.015)
   - Metrica: Rapporto assi principali (λ₁/λ₂)
   - Valore ideale cerchio: 1.0
   - Implementazione: Momenti di inerzia + eigenvalues

**Loss totale:**
```python
L_total = L_dice_ce + 0.01·L_compact + 0.005·L_boundary + 0.015·L_aspect
```

---

## Caratteristiche Chiave V2.4

### 1. Differenziabilità Completa

Tutte le operazioni mantengono il computational graph di PyTorch:
- ✅ **NO NumPy** - solo operazioni PyTorch
- ✅ **NO thresholding hard** - usa probabilità soft
- ✅ **NO OpenCV** - approssimazioni differenziabili
- ✅ **Fully vectorized** - processa batch completo in parallelo

### 2. Protezioni Anti-NaN

La V2.4 include protezioni multiple contro instabilità numerica:

#### A. Clamp Aggressivo prima di sqrt
Il problema critico risolto nella V2.4:

```python
# Il gradiente di sqrt(x) è: d/dx sqrt(x) = 1/(2*sqrt(x))
# Quando x → 0, il gradiente → ∞ (ESPLOSIONE!)

# Soluzione V2.4: Clamp AGGRESSIVO
grad_mag_squared = torch.clamp(grad_x**2 + grad_y**2, min=1e-2)
grad_mag = torch.sqrt(grad_mag_squared)

# Confronto:
# - sqrt(1e-4) → gradiente = 50  ❌ Instabile
# - sqrt(1e-2) → gradiente = 5   ✅ Stabile (10× meglio)
```

Questo fix risolve il problema di **NaN durante training** causato da gradienti esplosivi quando le predizioni sono incerte (epoche iniziali).

#### B. Controllo Area Minima

```python
area_per_batch = pred_soft.sum(dim=(1, 2))  # [B]
min_area_threshold = 50.0

if (area_per_batch < min_area_threshold).all():
    return pred_soft.sum() * 0.0  # Mantiene computational graph
```

Se tutte le predizioni nel batch sono quasi vuote, ritorna 0.0 senza calcolare metriche geometriche instabili.

#### C. Gestione Batch Misti

Usa maschere per escludere batch problematici dai calcoli:

```python
valid_mask = area >= min_area_threshold  # [B] boolean

if not valid_mask.any():
    return pred_soft.sum() * 0.0

# Calcola loss solo sui batch validi
loss = loss_per_batch[valid_mask].mean()
```

#### D. Controlli Finali NaN

Ogni funzione loss ha un controllo finale:

```python
if torch.isnan(loss) or torch.isinf(loss):
    return pred_soft.sum() * 0.0  # Fallback sicuro
```

### 3. Safety Checks nel Trainer

Il `nnUNetTrainerGeometric` include controlli a più livelli:

**Pre-backward:**
```python
if torch.isnan(total_loss) or torch.isinf(total_loss):
    print("WARNING: Loss NaN, usando solo Dice+CE")
    total_loss = loss_dice_ce  # Fallback
```

**Durante backward:**
```python
try:
    total_loss.backward()
except RuntimeError as e:
    if 'nan' in str(e).lower():
        print("WARNING: Backward failed, skipping step")
        self.optimizer.zero_grad(set_to_none=True)
        return {'loss': 0.0}
```

**Post-backward:**
```python
# Verifica gradienti per NaN
has_nan_grad = False
for name, param in self.network.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        has_nan_grad = True
        break

if has_nan_grad:
    print("WARNING: NaN nei gradienti, skipping optimizer step")
    self.optimizer.zero_grad(set_to_none=True)
else:
    self.optimizer.step()  # Procedi solo se gradienti validi
```

---

## Configurazione Training

### Parametri Principali

```python
# Numero epoche
num_epochs = 100

# Warm-up: loss geometrica attivata dopo N epoche
geometric_loss_warmup_epochs = 20

# Batch size
batch_size = 8  # Ridotto per memoria GPU

# Pesi loss geometriche
weight_compactness = 0.01   # Forma circolare
weight_boundary = 0.005     # Bordi lisci
weight_aspect = 0.015       # Rapporto assi (combina solidity + eccentricity)
```

### Schedule Training

```
Epoche 0-19:  Solo Dice + CrossEntropy (warm-up)
              ↓
              La rete impara segmentazione base

Epoche 20-99: Dice + CE + Geometric Loss
              ↓
              La rete affina la forma dei cerchi
```

### Perché Warm-up di 20 Epoche?

- **Epoche 0-19**: La rete impara a segmentare i cerchi (task base)
- **Epoca 20+**: La rete affina la geometria (task avanzato)

Attivare la loss geometrica troppo presto (es. epoca 5) causa instabilità perché:
- Le predizioni sono ancora molto incerte
- Molti pixel hanno probabilità basse → gradienti grandi
- I calcoli geometrici su "blob" informi non hanno senso

---

## Workflow Completo

### 1. Generazione Dataset

```bash
python data_geom.py
# Scegli: 1 (Multi-circle) o 2 (Single-circle)
# Output: 500 immagini 512×512
```

### 2. Conversione nnU-Net

```bash
python convert_to_nnunet.py
# Scegli il dataset generato
# Output: Dataset501_Shapes o Dataset502_Shapes_One
```

### 3. Planning e Preprocessing

```bash
# Dataset 501 (multi-cerchi)
nnUNetv2_plan_and_preprocess -d 501

# Dataset 502 (singolo cerchio)
nnUNetv2_plan_and_preprocess -d 502
```

### 4. Training

```bash
python train.py
```

Opzioni interattive:
- **Trainer**:
  - `1` = Baseline (nnU-Net standard, 250 epoche)
  - `2` = Geometric (nnU-Net + loss geometrica, 100 epoche)
- **Dataset**: Seleziona tra quelli disponibili
- **Epoche**: Conferma o modifica (default: 100 per geometric, 250 per baseline)

### 5. Inference

```bash
python run_inference.py --both
# Genera predizioni per baseline E geometric
```

Output:
```
baseline_results/predictions/      # Predizioni baseline
geometric_results/predictions/     # Predizioni geometric
```

### 6. Analisi Risultati

```bash
python test.py
```

Opzioni:
- `1` = Analizza solo baseline
- `2` = Analizza solo geometric
- `3` = Confronto baseline vs geometric

Output generato:
```
baseline_results/
├── visualizations/           # PNG confronti (originale, GT, pred, overlap)
├── metrics_summary.json      # Metriche aggregate
└── metrics_summary.txt       # Report leggibile

geometric_results/
├── visualizations/
├── metrics_summary.json
└── metrics_summary.txt

confronto_results/
├── visualizations/           # Confronti side-by-side
├── metrics_comparison_chart.png  # Grafico barre
├── metrics_comparison.json
└── metrics_comparison.txt
```

---

## Metriche Valutate

### Metriche Standard

- **Dice Score**: Sovrapposizione predizione-GT
- **IoU (Intersection over Union)**: Jaccard index
- **Hausdorff Distance**: Distanza massima tra contorni
- **Boundary IoU**: Sovrapposizione bordi (dilatati 2px)

### Metriche Geometriche

- **Compactness**: `4π·Area/Perimetro²` (ideale: 1.0)
- **Solidity**: `Area/ConvexHull` (ideale: ~0.9-1.0)
- **Eccentricity**: Allungamento ellisse (ideale: <0.5)

---

## Risultati Attesi

### Target Metriche

| Metrica | Baseline | Target Geometric | Miglioramento |
|---------|----------|------------------|---------------|
| **Dice Score** | ~0.987 | ≥0.985 | Mantenere alto |
| **Compactness** | ~0.31 | 0.65-0.80 | +110-160% |
| **Solidity** | ~0.34 | 0.80-0.95 | +135-180% |
| **Eccentricity** | ~0.87 | 0.30-0.50 | -42-65% ↓ |
| **Boundary IoU** | ~0.87 | 0.92-0.95 | +6-9% |
| **Hausdorff Dist** | ~7.88 | 3.00-5.00 | -36-62% ↓ |

### Interpretazione

- ✅ **Dice elevato**: La rete segmenta correttamente i cerchi
- ✅ **Compactness alta**: Le forme sono circolari, non irregolari
- ✅ **Solidity alta**: Poche concavità, forme convesse
- ✅ **Eccentricity bassa**: Cerchi, non ellissi allungate
- ✅ **Hausdorff basso**: Bordi precisi, pochi outlier

---

## Troubleshooting

### Problema: NaN durante Training

**Sintomo:** Loss diventa NaN durante training

**Verifica versione:**
```bash
grep "V2.4" /venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/geometric_losses.py
grep "min=1e-2" /venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/geometric_losses.py
```

Dovresti vedere:
- Header con "V2.4"
- Almeno 2 occorrenze di `min=1e-2` (compactness + aspect loss)

**Soluzione:**
```bash
# Ri-copia i file corretti
cp /workspace/geometrica/geometric_losses.py \
   /venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/

cp /workspace/geometrica/nnUNetTrainerGeometric.py \
   /venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/
```

### Problema: Loss Geometrica sempre 0.0

**Sintomo:** La loss geometrica è sempre 0.0 durante training

**Cause possibili:**
1. **Warm-up attivo**: Normale nelle epoche 0-19
2. **Area troppo piccola**: Tutte le predizioni hanno area < 50 pixel
3. **Problema rete base**: La rete non impara a segmentare (problema Dice+CE)

**Verifica:**
```bash
# Controlla a che epoca sei
grep "Epoch [0-9]" <training_log> | tail -5

# Se epoca < 20: è normale (warm-up)
# Se epoca >= 20 e geometric=0.0: problema
```

**Soluzione epoca ≥ 20:**
- Verifica che il baseline trainer funzioni (training senza geometric loss)
- Se baseline non funziona → problema dataset/preprocessing
- Se baseline funziona → contatta per debug

### Problema: Out of Memory (OOM)

**Sintomo:** CUDA out of memory durante training

**Soluzione:**
```python
# In nnUNetTrainerGeometric.py, riduci:
self.geometric_loss_samples = 2  # Era 4, ora calcola loss su 2 campioni
```

Oppure:
```python
# Riduci batch size (già 8 di default, ma puoi scendere)
# In __init__, cambia:
batch_size = 4  # Era 8
```

### Problema: Dice Score Scende

**Sintomo:** Con geometric loss, il Dice score peggiora rispetto al baseline

**Cause possibili:**
1. **Pesi geometric troppo alti**: Dominano su Dice+CE
2. **Warm-up troppo corto**: Loss geometric attivata troppo presto

**Soluzione:**
```python
# Riduci pesi geometric (in nnUNetTrainerGeometric.py):
self.geometric_loss = GeometricLosses(
    weight_compactness=0.005,    # Era 0.01, dimezza
    weight_solidity=0.005,       # Era 0.01, dimezza
    weight_eccentricity=0.0025,  # Era 0.005, dimezza
    weight_boundary=0.0025       # Era 0.005, dimezza
)
```

O aumenta warm-up:
```python
self.geometric_loss_warmup_epochs = 50  # Era 20
```

---

## Note Tecniche

### Computational Graph

**Problema risolto:** Mantenere il computational graph intatto per backpropagation.

**Soluzione:**
```python
# ❌ SBAGLIATO: Rompe computational graph
loss = torch.tensor(0.0, device=device)

# ✅ CORRETTO: Mantiene computational graph
loss = pred_soft.sum() * 0.0
```

Il secondo approccio crea un nodo nel graph connesso a `pred_soft`, permettendo ai gradienti di propagarsi correttamente.

### Gradienti di sqrt()

**Problema:** La derivata di `sqrt(x)` è `1/(2*sqrt(x))`, che esplode quando `x → 0`.

**Soluzione V2.4:**
```python
# Clamp PRIMA di sqrt con valore abbastanza grande
x = torch.clamp(x, min=1e-2)  # NON 1e-4 o meno!
y = torch.sqrt(x)

# Risultato:
# - sqrt(1e-2) → gradiente = 5 (STABILE)
# - sqrt(1e-4) → gradiente = 50 (INSTABILE)
```

**Regola generale:** Per ogni `sqrt()` in deep learning, usa `clamp(min ≥ 1e-2)`.

### Approssimazioni Differenziabili

| Operazione Classica | Approssimazione Differenziabile |
|---------------------|--------------------------------|
| `cv2.arcLength()` | Magnitudine Sobel gradient |
| `cv2.convexHull()` | Max pooling iterato (non usato in V2.4) |
| `cv2.moments()` | Coordinate grids pesate |
| `cv2.findContours()` | Laplacian convoluzione |
| Threshold binario `>0.5` | Probabilità soft (no threshold) |

---

## Estensioni Future

### Versione 1.1 (Possibili Aggiunte)

- [ ] **Ramp-up graduale**: Invece di on/off netto all'epoca 20, aumenta peso geometric gradualmente (epoche 20-40)
- [ ] **Learning rate adaptive**: Riduce LR quando loss geometrica si attiva
- [ ] **Multi-scala**: Calcola geometric loss su più risoluzioni (pyramid)
- [ ] **Attention mechanism**: Pesa geometric loss per difficoltà batch

### Versione 2.0 (Architetture Diverse)

- [ ] **Geometric constraints come regularization**: L₁/L₂ invece di loss diretta
- [ ] **Adversarial training**: GAN per discriminare cerchi "veri" da predetti
- [ ] **Shape priors**: Prior Bayesiani sulla distribuzione forme
- [ ] **Active contours**: Snake differenziabili per boundary smoothing

---

## Citazioni

Se usi questo codice, per favore cita:

**nnU-Net:**
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature methods, 18(2), 203-211.
```

**Questo progetto:**
```
Geometric Loss for Circle Segmentation (2025)
Francesco + Claude
https://github.com/[your-repo]
```

---

## Changelog

### V1.0 (Dicembre 2025) - Versione Stabile

**Caratteristiche:**
- ✅ Loss geometriche differenziabili (compactness, boundary, aspect)
- ✅ Protezioni anti-NaN V2.4 (clamp aggressivo sqrt)
- ✅ Warm-up 20 epoche
- ✅ Safety checks multipli nel trainer
- ✅ Support per Dataset 501 (multi-cerchi) e 502 (singolo cerchio)
- ✅ Pipeline completa: generazione → training → inference → analisi
- ✅ Documentazione completa

**Testing:**
- ✅ Gradient flow verificato
- ✅ Training stabile su Dataset 501
- ✅ Training stabile su Dataset 502
- ✅ Metriche geometriche migliorate vs baseline

**Note:**
- Versione pronta per uso in produzione
- Codice testato e stabile
- Documentazione completa

---

**Versione:** 1.0
**Status:** ✅ Stabile
**Last Update:** Dicembre 2025
