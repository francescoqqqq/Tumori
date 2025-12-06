# Geometric Circle Segmentation with nnU-Net

Progetto di segmentazione di cerchi utilizzando nnU-Net esteso con loss geometriche per migliorare la qualità della forma delle predizioni.

**Autore:** Francesco + Claude
**Data:** 2025-12-04
**Ultimo aggiornamento:** 2025-12-06

---

## Panoramica

Questo progetto estende [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) con **loss functions geometriche** per migliorare la segmentazione di cerchi, ottenendo:

- ✅ Mantiene Dice Score elevato (≥0.985)
- ✅ Migliora circolarità (compactness: 0.31 → 0.65+)
- ✅ Riduce irregolarità (solidity: 0.34 → 0.80+)
- ✅ Corregge forme ellittiche (eccentricity: 0.87 → <0.50)
- ✅ Smooth bordi (boundary IoU: 0.87 → 0.92+)

---

## Struttura File

### File Python Essenziali

| File | Descrizione |
|------|-------------|
| **`data_geom.py`** | Genera dataset sintetico con cerchi, triangoli, quadrati + rumore |
| **`convert_to_nnunet.py`** | Converte dataset in formato nnU-Net (Dataset501_Shapes) |
| **`geometric_losses.py`** | ⭐ Implementa le 4 loss geometriche (compactness, solidity, eccentricity, boundary) |
| **`nnUNetTrainer250Epochs.py`** | Trainer baseline nnU-Net (250 epoche) |
| **`nnUNetTrainerGeometric.py`** | ⭐ Trainer custom con loss geometriche (100 epoche) |
| **`run_inference.py`** | Esegue inference sui modelli e salva predizioni (run once) |
| **`test.py`** | ⭐ Analizza risultati, genera visualizzazioni e metriche (run many times) |

### File Documentazione

| File | Descrizione |
|------|-------------|
| **`README.md`** | Questo file - panoramica e guida file |
| **`GEOMETRIC_MODIFICATIONS.md`** | ⭐ Documentazione dettagliata delle modifiche geometriche |

### Directory Risultati

| Directory | Contenuto |
|-----------|-----------|
| **`baseline_results/`** | Predizioni e metriche del modello baseline |
| **`geometric_results/`** | Predizioni e metriche del modello geometric |
| **`confronto_results/`** | Confronto tra i due modelli |
| **`nnUNet_raw/`** | Dataset in formato nnU-Net (generato da convert) |
| **`nnUNet_preprocessed/`** | Dataset preprocessato (generato da nnU-Net) |
| **`nnUNet_results/`** | Modelli trained e checkpoint |

---

## Workflow Completo

### 1. Setup Iniziale

```bash
# Clona/naviga nella directory geometrica
cd /workspace/geometrica

# Setup environment variables (opzionale, script lo fanno automaticamente)
export nnUNet_raw="/workspace/geometrica/nnUNet_raw"
export nnUNet_preprocessed="/workspace/geometrica/nnUNet_preprocessed"
export nnUNet_results="/workspace/geometrica/nnUNet_results"
```

---

### 2. Generazione Dataset

```bash
# Genera 500 immagini 512x512 con forme geometriche
python data_geom.py
```

**Output:**
- `dataset_shapes/imagesTr/`: 500 immagini PNG (cerchi + triangoli + quadrati + noise)
- `dataset_shapes/labelsTr/`: 500 label PNG (solo cerchi)

---

### 3. Conversione formato nnU-Net

```bash
# Converte PNG → NIfTI e crea struttura nnU-Net
python convert_to_nnunet.py
```

**Output:**
- `nnUNet_raw/Dataset501_Shapes/`
  - `imagesTr/`: 500 file .nii.gz (immagini)
  - `labelsTr/`: 500 file .nii.gz (ground truth)
  - `dataset.json`: Metadata dataset

---

### 4. Planning e Preprocessing

```bash
# nnU-Net analizza dataset e crea plans
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

**Output:**
- `nnUNet_preprocessed/Dataset501_Shapes/`
  - Plans configurazione (batch size, patch size, etc.)
  - Dataset preprocessato
  - Cross-validation splits (5 folds)

---

### 5. Training

#### Opzione A: Training Baseline (rete normale)
```bash
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer250Epochs
```
- **Epoche**: 250 (poi ridotte a 100)
- **Loss**: Dice + Cross-Entropy standard
- **Tempo**: ~2-4 ore (dipende da GPU)

#### Opzione B: Training Geometric (rete con loss geometriche)
```bash
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerGeometric
```
- **Epoche**: 100 (5 warm-up + 95 geometric)
- **Loss**: Dice + CE + Compactness + Solidity + Eccentricity + Boundary
- **Tempo**: ~3-5 ore (+20% per loss geometriche)

**Output:**
- `nnUNet_results/Dataset501_Shapes/nnUNetTrainer__nnUNetPlans__2d/fold_0/`
  - `checkpoint_final.pth`: Modello finale
  - `checkpoint_best.pth`: Best checkpoint (validation Dice)
  - `progress.png`: Grafico loss/Dice durante training
  - Logs e validation metrics

---

### 6. Inference

```bash
# Genera predizioni per entrambi i modelli (run ONCE)
python run_inference.py --both

# Oppure solo uno:
python run_inference.py --baseline   # Solo baseline
python run_inference.py --geometric  # Solo geometric
```

**Cosa fa:**
1. Carica validation split da `splits_final.json` (100 immagini)
2. Copia immagini in folder temporaneo
3. Esegue `nnUNetv2_predict` per generare predizioni
4. Salva predizioni in `baseline_results/predictions/` e `geometric_results/predictions/`
5. Pulisce folder temporanei

**Tempo**: ~5-10 min per 100 immagini

**IMPORTANTE**: Questo step si fa **UNA VOLTA SOLA**. Le predizioni vengono salvate e riusate da test.py.

---

### 7. Analisi Risultati

```bash
# Analizza risultati (run MANY TIMES)
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

**Output (mode 1 o 2 - single model):**
```
baseline_results/  (o geometric_results/)
├── visualizations/
│   ├── shape_0009_visualization.png  # 4 pannelli
│   ├── shape_0010_visualization.png
│   └── ...
├── metrics_summary.json              # Metriche aggregate
└── metrics_summary.txt               # Report leggibile
```

**Visualizzazione 4 pannelli:**
1. **Originale**: Immagine input
2. **Ground Truth**: Solo cerchi (GT)
3. **Predizione**: Output modello
4. **Overlap Analysis**:
   - 🟢 Verde = Corretto (True Positive)
   - 🔴 Rosso = Mancato (False Negative)
   - 🟡 Giallo = Errato (False Positive)

**Output (mode 3 - confronto):**
```
confronto_results/
├── visualizations/
│   ├── shape_0009_comparison.png     # 5 pannelli
│   ├── shape_0010_comparison.png
│   └── ...
├── metrics_comparison_chart.png      # ⭐ Grafico barre verde/rosso
├── metrics_comparison.json
└── metrics_comparison.txt
```

**Visualizzazione 5 pannelli:**
1. **Originale**
2. **Ground Truth**
3. **Baseline**
4. **Geometric**
5. **Confronto Prestazioni**:
   - 🟢 Verde = Entrambi corretti
   - 🔴 Rosso = Solo baseline corretto
   - 🔵 Blu = Solo geometric corretto

**Metrics Comparison Chart:**
- Grafico a barre per ogni metrica (Dice, IoU, Compactness, Solidity, Eccentricity, Boundary IoU, Hausdorff)
- Barra VERDE = modello migliore per quella metrica
- Barra ROSSA = modello peggiore
- Percentuale miglioramento sopra ogni barra

---

## Metriche Calcolate

### Metriche Standard
- **Dice Score**: Overlap tra predizione e ground truth
- **IoU**: Intersection over Union
- **Precision/Recall**: (calcolabili da Dice)

### Metriche Geometriche ⭐
- **Compactness**: (4π·Area)/Perimeter² - misura circolarità (1.0 = cerchio perfetto)
- **Solidity**: Area/ConvexHull_Area - misura convessità (1.0 = nessuna concavità)
- **Eccentricity**: √(1-(minor/major)²) - misura ellitticità (0.0 = cerchio, 1.0 = linea)

### Metriche di Distanza
- **Hausdorff Distance**: Massima distanza tra contorni
- **Boundary IoU**: IoU calcolato solo sui bordi (±3 pixel)

---

## Esempi d'Uso

### Scenario 1: Test rapido su 5 immagini
```bash
python test.py
# Scelta: 3 (Confronto)
# Immagini: 5
```
→ Genera confronto veloce per vedere se le modifiche funzionano

### Scenario 2: Analisi completa validation set
```bash
python test.py
# Scelta: 3 (Confronto)
# Immagini: [invio] (tutte)
```
→ Analisi completa su 100 immagini validation

### Scenario 3: Analisi solo baseline
```bash
python test.py
# Scelta: 1 (Baseline)
# Immagini: [invio]
```
→ Genera report e visualizzazioni solo per baseline

---

## File di Configurazione

### Dataset (data_geom.py)
```python
NUM_TRAIN = 500          # Numero immagini totali
IMAGE_SIZE = (512, 512)  # Dimensione immagini
VALIDATION_SPLIT = 0.2   # 20% validation (100 immagini)

# Forme generate
SHAPES = ['circle', 'triangle', 'square']
NUM_SHAPES_RANGE = (3, 8)  # 3-8 forme per immagine
TARGET_SHAPES = ['circle']  # Solo cerchi nel GT
```

### Training Geometric (nnUNetTrainerGeometric.py)
```python
# Loss weights
weight_compactness = 0.1
weight_solidity = 0.1
weight_eccentricity = 0.05
weight_boundary = 0.05

# Training config
num_epochs = 100
geometric_loss_warmup_epochs = 5
batch_size = 8
geometric_loss_samples = 4  # Solo 4 campioni per loss
```

---

## Troubleshooting

### Problema: OOM durante training geometric
**Soluzione 1**: Riduci batch size a 4 in `nnUNetTrainerGeometric.py`:
```python
config_data['batch_size'] = 4
```

**Soluzione 2**: Riduci `geometric_loss_samples` a 2:
```python
self.geometric_loss_samples = 2
```

### Problema: Inference lenta
**Normale**: Inference su 100 immagini richiede 5-10 min. È un one-time operation.

### Problema: test.py non trova predizioni
**Causa**: Non hai eseguito `run_inference.py`
**Soluzione**:
```bash
python run_inference.py --both
```

### Problema: Dice score geometric < 0.98
**Causa**: Pesi loss geometriche troppo alti
**Soluzione**: Riduci pesi in `nnUNetTrainerGeometric.py`:
```python
weight_compactness = 0.05
weight_solidity = 0.05
weight_eccentricity = 0.02
weight_boundary = 0.02
```

---

## Documentazione Dettagliata

Per dettagli completi sull'implementazione delle loss geometriche e modifiche al trainer:

📖 **Leggi [`GEOMETRIC_MODIFICATIONS.md`](GEOMETRIC_MODIFICATIONS.md)**

Questo documento include:
- Formule matematiche dettagliate per ogni loss
- Implementazione completa del codice
- Spiegazione modifiche al training loop
- Note tecniche e limitazioni
- Troubleshooting avanzato

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

## Citazioni

Se usi questo codice, cita:

**nnU-Net:**
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature methods, 18(2), 203-211.
```

**Questo progetto:**
```
Geometric Circle Segmentation with nnU-Net
Francesco + Claude, 2025
https://github.com/[your-repo]
```

---

## Licenza

Stesso di nnU-Net (Apache 2.0)

---

## Contatti

Per domande o problemi, apri una issue su GitHub.

**Ultimo aggiornamento:** 2025-12-06
