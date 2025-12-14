# Geometric Circle Segmentation with nnU-Net

Progetto di segmentazione di cerchi utilizzando nnU-Net esteso con loss geometriche per migliorare la qualità della forma delle predizioni.

**Autore:** Francesco + Claude
**Ultima versione:** V2.2 (fix computational graph + safety checks)
**Data aggiornamento:** 2025-12-10

---

## Panoramica

Questo progetto estende [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) con **loss functions geometriche** per migliorare la segmentazione di cerchi, ottenendo:

- Mantiene Dice Score elevato (≥0.985)
- Migliora circolarità (compactness)
- Riduce irregolarità (solidity)
- Corregge forme ellittiche (eccentricity)
- Smooth bordi (boundary IoU)

---

## File Principali

| File | Descrizione |
|------|-------------|
| **`train.py`** | ⭐ Script interattivo per training (usa questo!) |
| **`geometric_losses.py`** | Loss geometriche differenziabili V2.2 con protezioni anti-NaN |
| **`nnUNetTrainer250Epochs.py`** | Trainer baseline (rete normale) |
| **`nnUNetTrainerGeometric.py`** | ⭐ Trainer con loss geometriche |
| **`data_geom.py`** | Genera dataset sintetico con forme |
| **`convert_to_nnunet.py`** | Converte dataset in formato nnU-Net |
| **`run_inference.py`** | Esegue predizioni sui modelli trained |
| **`test.py`** | ⭐ Analizza risultati e genera visualizzazioni |

### Documentazione

| File | Descrizione |
|------|-------------|
| **`README.md`** | Questo file - guida rapida |
| **`GEOMETRIC_MODIFICATIONS.md`** | Documentazione dettagliata delle modifiche |

---

## Quick Start - Training

### 1. Prepara Dataset (se non l'hai già fatto)

```bash
cd /workspace/geometrica

# Genera dataset (500 immagini con forme)
python data_geom.py

# Converti in formato nnU-Net
python convert_to_nnunet.py

# Preprocessing nnU-Net
nnUNetv2_plan_and_preprocess -d 501
```

### 2. Training Interattivo (METODO RACCOMANDATO)

```bash
python train.py
```

Lo script ti chiederà:
1. **Tipo di rete**: Baseline (1) o Geometric (2)
2. **Numero epoche**: es. 100 (default)

Esempio sessione:
```
============================================================
SCELTA TIPO DI RETE
============================================================

  [1] Baseline (nnU-Net standard)
      Solo Dice + Cross-Entropy loss
  [2] Geometric (nnU-Net + Loss Geometriche)
      Dice + CE + Compactness + Solidity + Eccentricity + Boundary

Scegli il tipo di rete (1/2): 2

============================================================
NUMERO EPOCHE
============================================================

Epoche consigliate:
  - Baseline:  100-250 epoche
  - Geometric: 100 epoche (20 warm-up + 80 geometric)

Quante epoche vuoi fare? [default: 100]: 100

============================================================
RIEPILOGO CONFIGURAZIONE
============================================================

  Tipo rete:     GEOMETRIC
  Trainer:       nnUNetTrainerGeometric
  Epoche:        100
  Dataset:       501 (Shapes)
  Configurazione: 2d
  Fold:          0

  Loss Geometriche:
    - Compactness:    0.01
    - Solidity:       0.01
    - Eccentricity:   0.005
    - Boundary:       0.005
    - Warm-up:        20 epoche
    - Con Geometric:  80 epoche

Vuoi avviare il training? (s/n): s
```

### 3. Training Manuale (opzionale)

Se preferisci usare i comandi diretti:

**Baseline:**
```bash
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer250Epochs
```

**Geometric:**
```bash
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerGeometric
```

**NOTA:** Con training manuale, il numero di epoche è fisso nel trainer.

---

## Analisi Risultati

### 1. Genera Predizioni (una volta sola)

```bash
# Per entrambi i modelli
python run_inference.py --both

# O solo uno specifico
python run_inference.py --baseline   # Solo baseline
python run_inference.py --geometric  # Solo geometric
```

**Tempo:** ~5-10 minuti per 100 immagini
**Output:** Predizioni salvate in `baseline_results/predictions/` e `geometric_results/predictions/`

### 2. Analizza e Confronta

```bash
python test.py
```

**Menu interattivo:**
```
Che rete vuoi analizzare? Premi:
  1 - Baseline       (rete normale)
  2 - Geometric      (rete con loss geometrica)
  3 - Confronto      (confronta entrambe le reti)

Scelta (1/2/3): 3

Quante immagini vuoi analizzare?
Numero (invio per tutte): 20
```

**Output:**
- `confronto_results/visualizations/` - Confronti side-by-side
- `confronto_results/metrics_comparison.txt` - Metriche aggregate
- `confronto_results/metrics_comparison_chart.png` - Grafico barre

---

## Struttura Directory

```
geometrica/
├── train.py                          # Script interattivo training
├── geometric_losses.py               # Loss V2.1 con protezioni anti-NaN
├── nnUNetTrainer250Epochs.py         # Trainer baseline
├── nnUNetTrainerGeometric.py         # Trainer geometric
├── data_geom.py                      # Genera dataset
├── convert_to_nnunet.py              # Conversione formato
├── run_inference.py                  # Inference
├── test.py                           # Analisi risultati
├── README.md                         # Questo file
├── GEOMETRIC_MODIFICATIONS.md        # Documentazione dettagliata
│
├── nnUNet_raw/                       # Dataset nnU-Net
│   └── Dataset501_Shapes/
│       ├── imagesTr/                 # 500 immagini .nii.gz
│       ├── labelsTr/                 # 500 ground truth
│       └── dataset.json
│
├── nnUNet_preprocessed/              # Dataset preprocessato
│   └── Dataset501_Shapes/
│
├── nnUNet_results/                   # Modelli trained
│   └── Dataset501_Shapes/
│       ├── nnUNetTrainer250Epochs__nnUNetPlans__2d/
│       │   └── fold_0/
│       │       ├── checkpoint_final.pth
│       │       └── checkpoint_best.pth
│       └── nnUNetTrainerGeometric__nnUNetPlans__2d/
│           └── fold_0/
│               ├── checkpoint_final.pth
│               └── checkpoint_best.pth
│
├── baseline_results/                 # Risultati baseline
│   ├── predictions/                  # 100 predizioni
│   ├── visualizations/               # PNG per ogni caso
│   ├── metrics_summary.json
│   └── metrics_summary.txt
│
├── geometric_results/                # Risultati geometric
│   ├── predictions/
│   ├── visualizations/
│   ├── metrics_summary.json
│   └── metrics_summary.txt
│
└── confronto_results/                # Confronto modelli
    ├── visualizations/               # Confronti side-by-side
    ├── metrics_comparison_chart.png  # Grafico barre
    ├── metrics_comparison.json
    └── metrics_comparison.txt
```

---

## Metriche Calcolate

### Standard
- **Dice Score**: Overlap tra predizione e ground truth
- **IoU**: Intersection over Union

### Geometriche
- **Compactness**: (4π·Area)/Perimeter² - circolarità (1.0 = cerchio perfetto)
- **Solidity**: Area/ConvexHull_Area - convessità (1.0 = nessuna concavità)
- **Eccentricity**: √(1-(minor/major)²) - ellitticità (0.0 = cerchio)

### Distanza
- **Hausdorff Distance**: Massima distanza tra contorni
- **Boundary IoU**: IoU calcolato solo sui bordi (±3 pixel)

---

## Novità Versione 2.1

### Protezioni Anti-NaN

La versione 2.1 include protezioni robuste contro NaN nelle loss geometriche:

1. **Clamping valori**: Area, perimeter, momenti di inerzia clampati a range sicuri
2. **Check NaN**: Ogni loss è controllata per NaN/Inf e settata a 0 se problematica
3. **Epsilon più grandi**: Divisioni con epsilon da 1e-4 aumentati a 1e-2
4. **Pesi ridotti 10x**: Loss geometriche meno aggressive per stabilità
5. **Warm-up aumentato**: 20 epoche invece di 5 per stabilizzare training

### Configurazione Geometric V2.1

```python
# Pesi loss (ridotti per stabilità)
weight_compactness = 0.01   # Era 0.1
weight_solidity = 0.01      # Era 0.1
weight_eccentricity = 0.005 # Era 0.05
weight_boundary = 0.005     # Era 0.05

# Warm-up
warmup_epochs = 20  # Era 5

# Training schedule
# Epoche 0-19:  Solo Dice + CE
# Epoche 20-99: Dice + CE + Geometric
```

---

## Troubleshooting

### Problema: NaN nei pesi durante training

**Soluzione**: La V2.1 include protezioni anti-NaN. Se ancora hai problemi:
1. Aumenta warm-up a 30-40 epoche
2. Riduci ulteriormente i pesi geometric (0.005, 0.005, 0.001, 0.001)

### Problema: Predizioni vuote (tutti zero)

**Causa**: Pesi NaN nel checkpoint
**Soluzione**: Ri-allena con V2.1 che include protezioni

### Problema: OOM durante training

**Soluzione 1**: Riduci batch_size a 4 in `nnUNetTrainerGeometric.py`:
```python
config_data['batch_size'] = 4
```

**Soluzione 2**: Riduci `geometric_loss_samples` a 2:
```python
self.geometric_loss_samples = 2
```

### Problema: Dice score geometric < 0.98

**Causa**: Pesi loss geometriche troppo alti
**Soluzione**: Sono già ridotti in V2.1, ma se necessario riduci ulteriormente

---

## Testing Loss Geometriche

Per verificare che le loss funzionino correttamente:

```bash
python geometric_losses.py
```

Output atteso:
```
================================================================================
TEST: Differentiable Geometric Losses V2.2 (fix computational graph + safety checks)
================================================================================

--- Test GeometricLosses wrapper ---

✅ Total Loss: 0.XXXXXX
   Componenti: {'compactness': 0.XXX, 'boundary': 0.XXX, 'aspect': 0.XXX, ...}

   ✅ Loss è un numero valido

🔍 Testing gradient flow...
   Gradient statistics (on logits - leaf tensor):
      mean = 0.XXXXXXXX
      max  = 0.XXXXXXXX
      std  = 0.XXXXXXXX

   ✅✅✅ GRADIENT FLOW OK!
   I gradienti sono presenti e non-zero - la rete può imparare!

================================================================================
✅ Test completato!
================================================================================
```

---

## Requisiti

### Librerie Python
```bash
pip install nnunetv2
pip install numpy scipy scikit-image opencv-python
pip install matplotlib nibabel
```

### Hardware
- GPU: Raccomandato (CUDA-capable)
- RAM: ≥16 GB
- Storage: ~5 GB (dataset + models + results)

---

## Documentazione Dettagliata

Per informazioni complete sull'implementazione:

📖 Leggi [GEOMETRIC_MODIFICATIONS.md](GEOMETRIC_MODIFICATIONS.md)

Include:
- Formule matematiche dettagliate
- Implementazione completa del codice
- Spiegazione modifiche al training loop
- Note tecniche e limitazioni

---

## Workflow Completo Esempio

```bash
# 1. Setup (una volta sola)
cd /workspace/geometrica
python data_geom.py
python convert_to_nnunet.py
nnUNetv2_plan_and_preprocess -d 501

# 2. Training (scegli Geometric, 100 epoche)
python train.py

# 3. Inference (dopo training completato)
python run_inference.py --both

# 4. Analisi (scegli Confronto, tutte le immagini)
python test.py
```

---

## Citazioni

**nnU-Net:**
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature methods, 18(2), 203-211.
```

**Questo progetto:**
```
Geometric Circle Segmentation with nnU-Net V2.1
Francesco + Claude, 2025
https://github.com/[your-repo]
```

---

## Licenza

Stesso di nnU-Net (Apache 2.0)

---

**Ultimo aggiornamento:** 2025-12-10
**Versione:** 2.1 (con protezioni anti-NaN e script interattivo)
