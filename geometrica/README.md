# Geometric Circle Segmentation with nnU-Net

Progetto di segmentazione di cerchi con [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), esteso con **loss functions geometriche differenziabili** per migliorare la qualità della forma delle predizioni.

**Autore:** Francesco + Claude — **Versione:** 3.0

---

## Panoramica

Il progetto confronta due architetture identiche:

- **Baseline** — ottimizza solo Dice + Cross-Entropy
- **Geometrica** — aggiunge penalità differenziabili su compactness, eccentricità e smoothness del bordo

L'obiettivo è verificare se le loss geometriche producono predizioni più circolari mantenendo un Dice Score elevato.

---

## File Principali

| File | Descrizione |
|------|-------------|
| `config.py` | ⭐ **Unico file da modificare** — parametri del run |
| `run_experiment.py` | Pipeline completa automatica (Step 1–5) — non modificare |
| `data_geom.py` | Genera dataset sintetico con cerchi e distrattori |
| `geometric_losses.py` | Loss geometriche differenziabili V2.5 |
| `nnUNetTrainerBaseline.py` | Trainer baseline (solo Dice + CE) |
| `nnUNetTrainerGeometric.py` | Trainer con loss geometriche |
| `metrics_utils.py` | Funzioni metriche e visualizzazioni (usato da pipeline e standalone) |
| `test.py` | Analisi standalone risultati di un esperimento |

### Documentazione tecnica (cartella `info/`)

| File | Contenuto |
|------|-----------|
| `GEOMETRIC_MODIFICATIONS.md` | Dettagli implementativi e formule |
| `TEORIA_DIFFERENZIABILITA.md` | Spiegazione matematica della differenziabilità |
| `run_exp_info.md` | Note sulla pipeline MLOps |

---

## Quick Start

### 1. Modifica `config.py`

```python
FOLDER_NAME      = "esp01_identico_100img"  # cartella dentro experiments/
AUTOMATIC        = "si"   # "si" = pipeline automatica | "no" = conferma step-by-step

# --- Dataset ---
IMG_SIZE         = 512
NUM_IMAGES       = 100
SPLIT_TEST_SIZE  = 0.2          # 20% riservato al test
TARGET_MODE      = 1            # 1 = single-circle  |  2 = multi-circle
COLOR_STYLE      = "identico"   # vedi tabella sotto
CIRCLE_ALONE     = "no"         # "si" = distrattori non sovrapposti al cerchio

# --- nnU-Net e Training ---
DATASET_ID       = 501
RETI_DA_ALLENARE = "entrambe"   # "baseline" | "geometrica" | "entrambe"
EPOCHS           = 100
BATCH_SIZE       = 8
WARMUP_EPOCHS    = 15

# Pesi loss geometrica
WEIGHT_COMPACTNESS  = 0.01
WEIGHT_ECCENTRICITY = 0.03
WEIGHT_BOUNDARY     = 0.005
```

### 2. Avvia la pipeline

```bash
cd /workspace/geometrica
python run_experiment.py
```

La pipeline esegue automaticamente cinque step:

| Step | Azione |
|------|--------|
| **1** | Genera dataset PNG, split train/test, salva `config_riassunto.yaml` |
| **2** | Converte PNG → NIfTI, lancia `nnUNetv2_plan_and_preprocess` |
| **3** | Installa i trainer custom, allena le reti scelte |
| **4** | Inference sul test set, converte predizioni NIfTI → PNG |
| **5** | Calcola metriche, genera visualizzazioni e grafico comparativo |

---

## Parametro `COLOR_STYLE` — Shortcut Learning

| Valore | Cerchi | Distrattori | Effetto |
|--------|--------|-------------|---------|
| `"differente"` | Gradiente radiale (più luminoso al centro) | Piatti, range `[180,240]` | Cerchi visivamente distinti → la rete può usare la texture come scorciatoia |
| `"uguale"` | Piatti, range casuale `[180,240]` | Piatti, range `[180,240]` | Stessa gamma di colori → la rete **non può** distinguere per colore |
| `"identico"` | Stesso colore esatto dei distrattori nell'immagine | Stesso colore del cerchio | Test massimo: la rete è costretta a ragionare **solo sulla forma** |

`CIRCLE_ALONE = "si"` (attivo solo con `COLOR_STYLE = "identico"`) impedisce la sovrapposizione spaziale tra distrattori e cerchio, rendendo il task ancora più puro dal punto di vista della forma.

---

## Struttura Output

Ogni esperimento è completamente isolato in `experiments/<FOLDER_NAME>/`:

```
experiments/
└── esp01_identico_100img/
    ├── config_riassunto.yaml       ← tutta la config dell'esperimento
    ├── logs/                       ← log dettagliati per ogni step
    ├── 1_dataset/
    │   ├── train/  images/ + labels/
    │   └── test/   images/ + labels/
    ├── 2_nnunet_engine/
    │   ├── nnUNet_raw/             ← NIfTI training
    │   ├── nnUNet_preprocessed/
    │   └── nnUNet_results/         ← checkpoint modelli
    ├── 3_predizioni/
    │   ├── baseline/               ← predizioni PNG rete baseline
    │   ├── geometrica/             ← predizioni PNG rete geometrica
    │   ├── baseline_nifti/
    │   └── geometrica_nifti/
    └── 4_confronto_finale/
        ├── visualizations/         ← immagini confronto per ogni test case
        ├── metrics_comparison.json
        ├── metrics_comparison.txt
        └── metrics_comparison_chart.png
```

> Le variabili d'ambiente `nnUNet_*` vengono impostate all'inizio dello script,
> prima di qualsiasi import di nnunetv2, puntando alle cartelle interne
> dell'esperimento. Esperimenti multipli non si sovrascrivono mai.

---

## Analisi Standalone

Per riesaminare o rigenerare i risultati di un esperimento già completato:

```bash
python test.py
```

Il menu interattivo elenca gli esperimenti disponibili in `experiments/`, chiede quale rete analizzare (baseline, geometrica o entrambe) e rigenera metriche e visualizzazioni in `4_confronto_finale/`.

---

## Metriche

### Overlap pred/GT
| Metrica | Formula | Direzione |
|---------|---------|-----------|
| **Dice Score** | 2·\|P∩G\| / (\|P\|+\|G\|) | ↑ |
| **IoU** | \|P∩G\| / \|P∪G\| | ↑ |
| **Hausdorff Distance** | max distanza tra contorni | ↓ |
| **Boundary IoU** | IoU calcolato solo sui bordi (±3 px) | ↑ |

### Qualità della forma (sulla predizione)
| Metrica | Formula | Direzione |
|---------|---------|-----------|
| **Compactness** | (4π·Area) / Perimetro² | ↑ (1.0 = cerchio perfetto) |
| **Eccentricity** | √(1 − (b/a)²) via fitting ellisse | ↓ (0.0 = cerchio perfetto) |

---

## Pesi Loss Geometrica

I pesi vengono scritti automaticamente in `geometric_config.py` dai valori definiti in `config.py`.

```python
# Valori di default
WEIGHT_COMPACTNESS  = 0.01   # Area vs Perimetro²
WEIGHT_ECCENTRICITY = 0.03   # Rapporto assi (momenti di inerzia)
WEIGHT_BOUNDARY     = 0.005  # Smoothness bordi
WARMUP_EPOCHS       = 15     # Epoche prima di attivare la loss geometrica
```

**Regola empirica:**
- Aumenta `WEIGHT_ECCENTRICITY` se le predizioni rimangono ellittiche
- Se il Dice Score scende sotto 0.94, riduci i pesi (es. dimezza tutti)
- Se compare NaN durante il training, aumenta `WARMUP_EPOCHS` o dimezza i pesi

---

## Troubleshooting

**NaN nei pesi durante il training**
Aumenta `WARMUP_EPOCHS` a 30–40, oppure riduci i pesi geometrici della metà.

**Predizioni vuote (tutto zero)**
Il checkpoint è corrotto (training fallito per gradient explosion).
Verifica `experiments/<FOLDER_NAME>/logs/verify.log` e ri-allena con pesi più bassi.

**OOM durante il training**
Riduci `BATCH_SIZE = 4` oppure `GEOMETRIC_LOSS_SAMPLES = 2` in `config.py`.

**Eccentricity non migliora**
Aumenta `WEIGHT_ECCENTRICITY = 0.05` (il default è 0.03).

---

## Requisiti

```bash
pip install nnunetv2 numpy scipy scikit-image opencv-python matplotlib nibabel
```

Hardware consigliato: GPU CUDA, ≥16 GB RAM, ~5 GB storage per esperimento.

---

## Citazioni

```
Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based
biomedical image segmentation. Nature Methods, 18(2), 203–211.
```

Licenza: Apache 2.0 (stessa di nnU-Net)
