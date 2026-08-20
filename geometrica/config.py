# ==============================================================================
#  config.py  –  Parametri esperimento  
# ==============================================================================

FOLDER_NAME      = "test_40_6"   # Nome sottocartella dentro experiments/
AUTOMATIC        = "si"          # "si" = pipeline automatica  |  "no" = conferma interattiva step-by-step

# --- Dataset ---
IMG_SIZE         = 512           # Lato immagine quadrata (px)
NUM_IMAGES       = 50           # Totale immagini generate
SPLIT_TEST_SIZE  = 0.80          # Frazione test set  (0.2 → 20 %)
TARGET_MODE      = 1             # 1 = single-circle  |  2 = multi-circle
CIRCLE_ALONE     = "no"          # "si" = distrattori non sovrapposti ai cerchi (solo style=identico)
COLOR_STYLE      = "uguale"
# COLOR_STYLE:
#   "differente" → cerchi sfumati (gradiente), visivamente distinti dai distrattori piatti
#   "uguale"     → cerchi piatti stesso range colori [180,240] dei distrattori → no shortcut
#   "identico"   → cerchi e distrattori hanno lo stesso identico colore nella stessa immagine

# --- nnU-Net e Training ---
DATASET_ID       = 501           # ID numerico dataset (DatasetXXX_Shapes)
RETI_DA_ALLENARE = "entrambe"   # "baseline" | "geometrica" | "entrambe"
EPOCHS           = 6
BATCH_SIZE       = 8             # batch size per GPU
WARMUP_EPOCHS    = 2            # Epoche warm-up prima di attivare loss geometrica

# Device di training/inference: "auto" rileva automaticamente cuda/mps/cpu
# (usa GPU se disponibile, altrimenti CPU) → nessuna modifica manuale serve
# quando si sposta il progetto su una macchina diversa. Valori espliciti
# supportati da nnU-Net: "cuda", "cpu", "mps".
DEVICE           = "auto"

# Pesi loss geometrica (solo per rete geometrica)
WEIGHT_COMPACTNESS      = 0.03
WEIGHT_ECCENTRICITY     = 0.05
WEIGHT_BOUNDARY         = 0.01
GEOMETRIC_LOSS_SAMPLES  = 4      # Campioni del batch su cui calcolare la loss geometrica
# per motivi di tempi di allenamento conviene lasciarlo a 4

# Post-processing inference: rimuove componenti connesse più piccole di questa soglia.
# Converte "blob piccolo e sbagliato" → predizione vuota (più onesta metricamente).
# 0 = disabilitato  |  valore consigliato: ~0.2% dell'area immagine
# es. per IMG_SIZE=512: 512*512*0.002 ≈ 524 → usa 500
MIN_COMPONENT_PX        = 500
