#!/usr/bin/env python3
"""
run_experiment.py  –  MLOps pipeline COMPLETA per segmentazione geometrica con nnU-Net.

⚙  Per cambiare i parametri di un run modifica SOLO config.py, non questo file.

PIPELINE (tutto in un unico run):
    STEP 1  – Generazione dataset PNG + split Train/Test
    STEP 2  – Conversione PNG → NIfTI + preprocessing nnU-Net
    STEP 3  – Training (baseline e/o geometrica)
    STEP 4  – Inference sul Test Set + conversione NIfTI → PNG
    STEP 5  – Metriche, visualizzazioni e confronto finale

Struttura output:
    experiments/<FOLDER_NAME>/
    ├── config_riassunto.yaml
    ├── 1_dataset/         train/ e test/  (images/ + labels/)
    ├── 2_nnunet_engine/   nnUNet_raw/ preprocessed/ results/
    ├── 3_predizioni/      baseline/  geometrica/  (PNG)
    └── 4_confronto_finale/ visualizations/ metriche JSON/TXT/chart

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  AVVISO CRITICO – ORDINE IMPORT
   Non riorganizzare mai questo file con isort o autoflake.
   os.environ['nnUNet_*'] DEVE essere impostato PRIMA di qualsiasi
   `import nnunetv2`, altrimenti nnU-Net legge i path globali del
   devcontainer invece di quelli dell'esperimento, rompendo l'isolamento.
   Ordine obbligatorio:
       1. from config import *  ← solo variabili, nessun nnunetv2
       2. import stdlib  (os, sys, shutil, …)
       3. calcolo percorsi
       4. os.environ['nnUNet_*'] = …  ← QUI
       5. import nnunetv2 / import locale  ← DOPO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

# ==============================================================================
#  CONFIGURAZIONE  –  modifica config.py per cambiare i parametri del run
# ==============================================================================
from config import *  # noqa: F401, F403

# ==============================================================================
#  IMPORT LIBRERIE STANDARD  –  safe, non nnunetv2
# ==============================================================================
import os
import sys
import json
import shutil
import random
import subprocess
import tempfile
import time
from datetime import datetime
import re

# ==============================================================================
#  CALCOLO PERCORSI ESPERIMENTO
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXP_DIR     = os.path.join(SCRIPT_DIR, "experiments", FOLDER_NAME)
DATASET_DIR = os.path.join(EXP_DIR, "1_dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

NNUNET_ENGINE_DIR       = os.path.join(EXP_DIR, "2_nnunet_engine")
NNUNET_RAW_DIR          = os.path.join(NNUNET_ENGINE_DIR, "nnUNet_raw")
NNUNET_PREPROCESSED_DIR = os.path.join(NNUNET_ENGINE_DIR, "nnUNet_preprocessed")
NNUNET_RESULTS_DIR      = os.path.join(NNUNET_ENGINE_DIR, "nnUNet_results")

PRED_DIR      = os.path.join(EXP_DIR, "3_predizioni")
CONFRONTO_DIR = os.path.join(EXP_DIR, "4_confronto_finale")

DATASET_NAME = f"Dataset{DATASET_ID:03d}_Shapes"

TRAINERS = {
    "baseline":   "nnUNetTrainerBaseline",
    "geometrica": "nnUNetTrainerGeometric",
}

# ==============================================================================
#  VARIABILI D'AMBIENTE nnU-Net  –  DEVONO STARE QUI, PRIMA DI import nnunetv2
#
#  ⚠  NON spostare. Sovrascrivono i path globali del devcontainer
#     (/workspace/nnUNet_*) garantendo l'isolamento per esperimento.
# ==============================================================================
os.environ['nnUNet_raw']          = NNUNET_RAW_DIR
os.environ['nnUNet_preprocessed'] = NNUNET_PREPROCESSED_DIR
os.environ['nnUNet_results']      = NNUNET_RESULTS_DIR

# NOTA: nnUNet_n_proc_DA=0 disabilita i worker multiprocessing (fix /dev/shm),
# ma non va impostato globalmente perché rompe il preprocessing (torch.set_num_threads).
# Viene passato solo ai subprocess di training e inference (vedi _run_subprocess).

# ==============================================================================
#  IMPORT nnunetv2  –  SOLO QUI, DOPO os.environ
# ==============================================================================
# import nnunetv2   # decommentare solo se necessario a livello modulo

# ==============================================================================
#  IMPORT LOCALI
# ==============================================================================
sys.path.insert(0, SCRIPT_DIR)
from data_geom import generate_single_circle_dataset, generate_multi_circle_dataset  # noqa: E402
from metrics_utils import (  # noqa: E402
    calculate_all_metrics,
    create_visualization,
    create_comparison_visualization,
    create_metrics_comparison_chart,
    create_vis_bad,
)


# ==============================================================================
#  UTILITY
# ==============================================================================

_resolved_device_cache = None


def _resolve_device():
    """
    Determina il device da passare a nnU-Net CLI (-device cuda/cpu/mps).

    DEVICE in config.py puo' essere "auto" (autodetect cuda -> mps -> cpu)
    oppure un valore esplicito ("cuda", "cpu", "mps") per forzare un device.
    Il risultato viene calcolato una sola volta e messo in cache, cosi' le
    macchine senza GPU non richiedono alcuna modifica manuale al progetto.
    """
    global _resolved_device_cache
    if _resolved_device_cache is not None:
        return _resolved_device_cache

    if DEVICE != "auto":
        _resolved_device_cache = DEVICE
        return _resolved_device_cache

    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        _resolved_device_cache = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        _resolved_device_cache = "mps"
    else:
        _resolved_device_cache = "cpu"
    return _resolved_device_cache


def _separator(title=""):
    width = 65
    print(f"\n{'='*width}")
    if title:
        print(f"  {title}")
        print(f"{'='*width}")


def _format_duration(seconds):
    """Formatta secondi in Hh Mm Ss per stampa umana."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m:02d}m {s:05.2f}s"


def _get_nets_to_run():
    """Ritorna lista ordinata di net_type da RETI_DA_ALLENARE."""
    mapping = {
        "baseline":   ["baseline"],
        "geometrica": ["geometrica"],
        "entrambe":   ["baseline", "geometrica"],
    }
    if RETI_DA_ALLENARE not in mapping:
        raise ValueError(
            f"RETI_DA_ALLENARE='{RETI_DA_ALLENARE}' non valido. "
            f"Usa: 'baseline', 'geometrica', 'entrambe'."
        )
    return mapping[RETI_DA_ALLENARE]


def _run_subprocess(cmd, step_label, extra_env=None):
    """
    Esegue un subprocess con os.environ isolato.
    extra_env: dict di variabili aggiuntive da sovrascrivere (es. per training/inference).
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Forza Python a non bufferizzare stdout/stderr nel subprocess,
    # altrimenti le righe arrivano in blocchi e non in tempo reale.
    env["PYTHONUNBUFFERED"] = "1"
    print(f"  Comando: {' '.join(cmd)}")
    logs_dir = os.path.join(EXP_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    safe_step = re.sub(r"[^a-zA-Z0-9_.-]+", "_", step_label.strip().lower())
    log_path = os.path.join(logs_dir, f"{safe_step}.log")
    print(f"  Log dettagliato: {log_path}")

    useful_patterns = [
        r": Epoch \d+$",
        r"Current learning rate:",
        r"train_loss ",
        r"val_loss ",
        r"Pseudo dice ",
        r"Epoch time:",
        r"New best EMA pseudo Dice:",
        r"There are \d+ cases in the source folder",
        r"There are \d+ cases that I would like to predict",
        r"I am processing \d+ out of \d+",
        r"Training done\.",
    ]

    def _is_useful_line(line: str) -> bool:
        return any(re.search(pat, line) for pat in useful_patterns)

    def _translate_line(line: str) -> str:
        """Traduce in italiano le righe verbose di nnUNet prima di stamparle."""
        m = re.search(r"There are (\d+) cases in the source folder", line)
        if m:
            return f"  Immagini trovate nella cartella sorgente: {m.group(1)}"
        m = re.search(r"There are (\d+) cases that I would like to predict", line)
        if m:
            return f"  Immagini da predire: {m.group(1)}"
        m = re.search(r"I am processing (\d+) out of (\d+)", line)
        if m:
            return f"  Predizione in corso: processo {int(m.group(1))+1} di {m.group(2)}"
        # Rimuove il prefisso timestamp "YYYY-MM-DD HH:MM:SS.ffffff: " se presente
        clean = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+: ", "", line)
        return f"  {clean}"

    # Sovrascrive il log ad ogni run (non accumula run precedenti)
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write(f"STEP: {step_label}\n")
        log_f.write(f"CMD : {' '.join(cmd)}\n")
        log_f.write("=" * 80 + "\n")
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            log_f.write(raw_line)
            line = raw_line.rstrip()
            if not line or not _is_useful_line(line):
                continue
            # Separatore prima di ogni nuova epoca
            if re.search(r": Epoch \d+$", line):
                print(f"  {'─' * 55}")
            # Messaggio esplicito quando parte la validazione interna di nnUNet
            if re.search(r"Training done\.", line):
                print(f"  {'─' * 55}")
                print(f"  [nnUNet] Training terminato – avvio validazione interna automatica...")
                print(f"  {'─' * 55}")
                continue
            print(_translate_line(line))
        process.wait()
        result = subprocess.CompletedProcess(cmd, process.returncode)
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_label} fallito (exit code {result.returncode})\n"
            f"Comando: {' '.join(cmd)}"
        )


# ==============================================================================
#  INTERACTIVE MODE  –  attivo solo quando AUTOMATIC = "no"
# ==============================================================================

def _ask_input(prompt, default=None):
    """Chiede un valore testuale. Se default è fornito, Enter lo conferma."""
    if default is not None:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else str(default)
    while True:
        val = input(f"  {prompt}: ").strip()
        if val:
            return val
        print("  → Valore non può essere vuoto.")


def _ask_yn(prompt):
    """Chiede si/no. Ritorna True se 'si'."""
    while True:
        val = input(f"  {prompt} [si/no]: ").strip().lower()
        if val in ("si", "s"):
            return True
        if val in ("no", "n"):
            return False
        print("  → Scrivi 'si' o 'no'.")


def _print_box(lines):
    """Stampa un riquadro ASCII attorno alle righe date."""
    width = max(len(l) for l in lines) + 2
    print(f"  ┌{'─' * width}┐")
    for l in lines:
        print(f"  │ {l:<{width - 1}}│")
    print(f"  └{'─' * width}┘")


def _recalc_paths():
    """
    Ricalcola tutti i path globali dopo una modifica di FOLDER_NAME o DATASET_ID.
    Aggiorna anche os.environ in modo che i subprocess usino i path corretti.
    """
    global EXP_DIR, DATASET_DIR, TRAIN_DIR, TEST_DIR
    global NNUNET_ENGINE_DIR, NNUNET_RAW_DIR, NNUNET_PREPROCESSED_DIR, NNUNET_RESULTS_DIR
    global PRED_DIR, CONFRONTO_DIR, DATASET_NAME

    EXP_DIR     = os.path.join(SCRIPT_DIR, "experiments", FOLDER_NAME)
    DATASET_DIR = os.path.join(EXP_DIR, "1_dataset")
    TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
    TEST_DIR    = os.path.join(DATASET_DIR, "test")

    NNUNET_ENGINE_DIR       = os.path.join(EXP_DIR, "2_nnunet_engine")
    NNUNET_RAW_DIR          = os.path.join(NNUNET_ENGINE_DIR, "nnUNet_raw")
    NNUNET_PREPROCESSED_DIR = os.path.join(NNUNET_ENGINE_DIR, "nnUNet_preprocessed")
    NNUNET_RESULTS_DIR      = os.path.join(NNUNET_ENGINE_DIR, "nnUNet_results")

    PRED_DIR      = os.path.join(EXP_DIR, "3_predizioni")
    CONFRONTO_DIR = os.path.join(EXP_DIR, "4_confronto_finale")
    DATASET_NAME  = f"Dataset{DATASET_ID:03d}_Shapes"

    os.environ['nnUNet_raw']          = NNUNET_RAW_DIR
    os.environ['nnUNet_preprocessed'] = NNUNET_PREPROCESSED_DIR
    os.environ['nnUNet_results']      = NNUNET_RESULTS_DIR


def _interactive_setup():
    """
    Fase 0 – chiede il nome della cartella esperimento e conferma i parametri
    dataset. Modifica le variabili globali se l'utente sceglie di cambiarle.
    Chiamata all'inizio di main() quando AUTOMATIC = "no".
    """
    global FOLDER_NAME
    global IMG_SIZE, NUM_IMAGES, SPLIT_TEST_SIZE, TARGET_MODE, COLOR_STYLE, CIRCLE_ALONE

    # ── Nome cartella esperimento ──────────────────────────────────────────────
    print()
    print("  Come vuoi chiamare la cartella per questo esperimento?")
    FOLDER_NAME = _ask_input("Nome cartella", default=FOLDER_NAME)
    _recalc_paths()

    # ── Conferma parametri Dataset ─────────────────────────────────────────────
    while True:
        print()
        print("  Parametri DATASET:")
        _print_box([
            f"IMG_SIZE        = {IMG_SIZE}",
            f"NUM_IMAGES      = {NUM_IMAGES}",
            f"SPLIT_TEST_SIZE = {SPLIT_TEST_SIZE}",
            f"TARGET_MODE     = {TARGET_MODE}  "
            f"({'single-circle' if TARGET_MODE == 1 else 'multi-circle'})",
            f"COLOR_STYLE     = \"{COLOR_STYLE}\"",
            f"CIRCLE_ALONE    = \"{CIRCLE_ALONE}\"",
        ])
        print()
        if _ask_yn("Procedere alla creazione del dataset con questi valori?"):
            break
        print()
        print("  Modifica parametri (invio = mantieni valore attuale):")
        IMG_SIZE        = int(_ask_input("IMG_SIZE",        default=IMG_SIZE))
        NUM_IMAGES      = int(_ask_input("NUM_IMAGES",      default=NUM_IMAGES))
        SPLIT_TEST_SIZE = float(_ask_input("SPLIT_TEST_SIZE",default=SPLIT_TEST_SIZE))
        TARGET_MODE     = int(_ask_input(
            "TARGET_MODE (1=single / 2=multi)", default=TARGET_MODE))
        COLOR_STYLE     = _ask_input(
            "COLOR_STYLE (uguale / differente / identico)", default=COLOR_STYLE)
        CIRCLE_ALONE    = _ask_input(
            "CIRCLE_ALONE (si / no)", default=CIRCLE_ALONE)


def _interactive_confirm_training():
    """
    Fase 2 – dopo la creazione del dataset, mostra i parametri di training e
    chiede conferma separatamente per baseline (linee 53-57) e geometrica
    (linee 60-64). Modifica i globali se l'utente sceglie di cambiarli.
    """
    global DATASET_ID, RETI_DA_ALLENARE, EPOCHS, BATCH_SIZE, WARMUP_EPOCHS
    global WEIGHT_COMPACTNESS, WEIGHT_ECCENTRICITY
    global WEIGHT_BOUNDARY, GEOMETRIC_LOSS_SAMPLES, DATASET_NAME

    nets = _get_nets_to_run()

    # ── Parametri training generale (mostrati per baseline o geometrica) ───────
    # Etichetta: "BASELINE" se alleniamo la baseline, altrimenti "TRAINING"
    label_generale = "BASELINE" if "baseline" in nets else "TRAINING"

    while True:
        print()
        print(f"  Parametri TRAINING {label_generale}:")
        _print_box([
            f"DATASET_ID       = {DATASET_ID}",
            f"RETI_DA_ALLENARE = {RETI_DA_ALLENARE}",
            f"EPOCHS           = {EPOCHS}",
            f"BATCH_SIZE       = {BATCH_SIZE}",
            f"WARMUP_EPOCHS    = {WARMUP_EPOCHS}",
        ])
        print()
        domanda = (
            "Procedere con il training BASELINE?"
            if "baseline" in nets
            else "Procedere con questi parametri di training?"
        )
        if _ask_yn(domanda):
            break
        print()
        print("  Modifica parametri (invio = mantieni valore attuale):")
        DATASET_ID       = int(_ask_input("DATASET_ID", default=DATASET_ID))
        DATASET_NAME     = f"Dataset{DATASET_ID:03d}_Shapes"
        RETI_DA_ALLENARE = _ask_input(
            "RETI_DA_ALLENARE (baseline / geometrica / entrambe)",
            default=RETI_DA_ALLENARE,
        )
        EPOCHS           = int(_ask_input("EPOCHS",        default=EPOCHS))
        BATCH_SIZE       = int(_ask_input("BATCH_SIZE",    default=BATCH_SIZE))
        WARMUP_EPOCHS    = int(_ask_input("WARMUP_EPOCHS", default=WARMUP_EPOCHS))

    # Ricalcola nets dopo eventuale modifica di RETI_DA_ALLENARE
    nets = _get_nets_to_run()

    # ── Parametri loss geometrica (solo se geometrica è inclusa) ──────────────
    if "geometrica" in nets:
        while True:
            print()
            print("  Parametri LOSS GEOMETRICA:")
            _print_box([
                f"WEIGHT_COMPACTNESS     = {WEIGHT_COMPACTNESS}",
                f"WEIGHT_ECCENTRICITY    = {WEIGHT_ECCENTRICITY}",
                f"WEIGHT_BOUNDARY        = {WEIGHT_BOUNDARY}",
                f"GEOMETRIC_LOSS_SAMPLES = {GEOMETRIC_LOSS_SAMPLES}",
            ])
            print()
            if _ask_yn("Procedere con il training GEOMETRICO?"):
                break
            print()
            print("  Modifica parametri (invio = mantieni valore attuale):")
            WEIGHT_COMPACTNESS     = float(_ask_input(
                "WEIGHT_COMPACTNESS",     default=WEIGHT_COMPACTNESS))
            WEIGHT_ECCENTRICITY    = float(_ask_input(
                "WEIGHT_ECCENTRICITY",    default=WEIGHT_ECCENTRICITY))
            WEIGHT_BOUNDARY        = float(_ask_input(
                "WEIGHT_BOUNDARY",        default=WEIGHT_BOUNDARY))
            GEOMETRIC_LOSS_SAMPLES = int(_ask_input(
                "GEOMETRIC_LOSS_SAMPLES", default=GEOMETRIC_LOSS_SAMPLES))


def _interactive_confirm_confronto():
    """
    Fase 4 – dopo il training, chiede se procedere con il confronto finale.
    Ritorna True per procedere, False per fermarsi.
    """
    print()
    print("  Training completato.")
    print()
    print("  Vuoi procedere con il confronto finale?")
    print("  (calcolo metriche, visualizzazioni, grafico comparativo)")
    print()
    return _ask_yn("Procedere con il confronto finale?")


# ==============================================================================
#  STEP 1  –  Dataset: generazione, split, organizzazione
# ==============================================================================

def create_experiment_structure():
    """Crea l'albero di cartelle dell'esperimento."""
    _separator("STEP 1a: STRUTTURA ESPERIMENTO")

    for d in [
        os.path.join(TRAIN_DIR, "images"), os.path.join(TRAIN_DIR, "labels"),
        os.path.join(TEST_DIR,  "images"), os.path.join(TEST_DIR,  "labels"),
        NNUNET_RAW_DIR, NNUNET_PREPROCESSED_DIR, NNUNET_RESULTS_DIR,
        PRED_DIR, CONFRONTO_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    print(f"  {EXP_DIR}")
    print(f"  ├── 1_dataset/  train/ test/")
    print(f"  ├── 2_nnunet_engine/  raw/ preprocessed/ results/")
    print(f"  ├── 3_predizioni/")
    print(f"  └── 4_confronto_finale/")
    print()
    print(f"  nnUNet_raw          = {os.environ['nnUNet_raw']}")
    print(f"  nnUNet_preprocessed = {os.environ['nnUNet_preprocessed']}")
    print(f"  nnUNet_results      = {os.environ['nnUNet_results']}")


def generate_dataset():
    """Genera il dataset raw in _raw_generated/ e ritorna il path."""
    _separator("STEP 1b: GENERAZIONE DATASET")
    print(f"  Modalità   : {'SINGLE-circle' if TARGET_MODE == 1 else 'MULTI-circle'}")
    print(f"  Immagini   : {NUM_IMAGES}")
    print(f"  Dimensione : {IMG_SIZE}×{IMG_SIZE} px")
    print(f"  Color style: {COLOR_STYLE}")
    print(f"  Circle alone: {CIRCLE_ALONE}")

    raw_dir = os.path.join(EXP_DIR, "_raw_generated")
    kw = dict(output_dir=raw_dir, num_images=NUM_IMAGES,
              img_size=(IMG_SIZE, IMG_SIZE), color_style=COLOR_STYLE, circle_alone=CIRCLE_ALONE)

    if TARGET_MODE == 1:
        generate_single_circle_dataset(**kw)
    elif TARGET_MODE == 2:
        generate_multi_circle_dataset(**kw)
    else:
        raise ValueError(f"TARGET_MODE deve essere 1 o 2, ricevuto: {TARGET_MODE}")
    return raw_dir


def split_and_organize(raw_dir):
    """Divide il dataset in train/test con seed fisso. Ritorna (n_train, n_test)."""
    _separator("STEP 1c: SPLIT TRAIN / TEST")

    images_src = os.path.join(raw_dir, "imagesTr")
    labels_src = os.path.join(raw_dir, "labelsTr")
    all_files  = sorted(f for f in os.listdir(images_src) if f.lower().endswith(".png"))
    total      = len(all_files)

    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    n_test  = max(1, int(total * SPLIT_TEST_SIZE))
    n_train = total - n_test
    test_files, train_files = shuffled[:n_test], shuffled[n_test:]

    print(f"  Totale  : {total}")
    print(f"  Train   : {n_train}  ({100*n_train/total:.1f} %)")
    print(f"  Test    : {n_test}   ({100*n_test/total:.1f} %)")
    print(f"  Seed    : 42")

    def _copy(flist, dest):
        for f in flist:
            shutil.copy2(os.path.join(images_src, f), os.path.join(dest, "images", f))
            shutil.copy2(os.path.join(labels_src, f), os.path.join(dest, "labels", f))

    _copy(train_files, TRAIN_DIR)
    _copy(test_files,  TEST_DIR)

    meta_src = os.path.join(raw_dir, "metadata.json")
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(DATASET_DIR, "metadata_full.json"))

    return n_train, n_test


def cleanup_raw(raw_dir):
    shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"\n  Cartella temporanea rimossa: {os.path.basename(raw_dir)}/")


def save_config_yaml(n_train, n_test):
    """Scrive config_riassunto.yaml nella root dell'esperimento."""
    lines = [
        "# Configurazione esperimento – auto-generato da run_experiment.py",
        f"# Data: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "esperimento:",
        f"  folder_name:      {FOLDER_NAME}",
        f"  cartella:         {EXP_DIR}",
        "",
        "dataset:",
        f"  img_size:         {IMG_SIZE}",
        f"  num_images:       {NUM_IMAGES}",
        f"  split_test_size:  {SPLIT_TEST_SIZE}",
        f"  n_train:          {n_train}",
        f"  n_test:           {n_test}",
        f"  target_mode:      {TARGET_MODE}   # 1=single  2=multi",
        f"  color_style:      {COLOR_STYLE}",
        f"  circle_alone:     {CIRCLE_ALONE}",
        "",
        "training:",
        f"  dataset_id:       {DATASET_ID}",
        f"  dataset_name:     {DATASET_NAME}",
        f"  reti_da_allenare: {RETI_DA_ALLENARE}",
        f"  epochs:           {EPOCHS}",
        f"  batch_size:       {BATCH_SIZE}",
        f"  warmup_epochs:    {WARMUP_EPOCHS}",
        f"  weight_compactness:  {WEIGHT_COMPACTNESS}",
        f"  weight_eccentricity: {WEIGHT_ECCENTRICITY}",
        f"  weight_boundary:     {WEIGHT_BOUNDARY}",
        f"  min_component_px:    {MIN_COMPONENT_PX}",
        "",
        "nnunet_env:",
        f"  nnUNet_raw:          {NNUNET_RAW_DIR}",
        f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED_DIR}",
        f"  nnUNet_results:      {NNUNET_RESULTS_DIR}",
    ]
    yaml_path = os.path.join(EXP_DIR, "config_riassunto.yaml")
    with open(yaml_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Config salvata: {yaml_path}")


# ==============================================================================
#  STEP 2  –  Conversione PNG→NIfTI + Preprocessing nnU-Net
# ==============================================================================

def convert_train_to_nnunet():
    """Converte 1_dataset/train/ nel formato NIfTI atteso da nnU-Net."""
    _separator("STEP 2a: CONVERSIONE PNG → NIfTI (train set)")

    import cv2        # noqa: PLC0415
    import numpy as np # noqa: PLC0415
    import nibabel as nib  # noqa: PLC0415

    src_images   = os.path.join(TRAIN_DIR, "images")
    src_labels   = os.path.join(TRAIN_DIR, "labels")
    dest_dataset = os.path.join(NNUNET_RAW_DIR, DATASET_NAME)
    dest_images  = os.path.join(dest_dataset, "imagesTr")
    dest_labels  = os.path.join(dest_dataset, "labelsTr")

    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    png_files = sorted(f for f in os.listdir(src_images) if f.endswith(".png"))
    n = len(png_files)
    print(f"  Conversione {n} coppie immagine/maschera → {dest_dataset}")

    affine = np.eye(4)
    for i, fname in enumerate(png_files):
        case_id = os.path.splitext(fname)[0]

        # Immagine: float32 grayscale NIfTI
        img   = cv2.imread(os.path.join(src_images, fname), cv2.IMREAD_GRAYSCALE)
        img3d = np.expand_dims(img.astype(np.float32), axis=2)
        nib.save(nib.Nifti1Image(img3d, affine),
                 os.path.join(dest_images, f"{case_id}_0000.nii.gz"))

        # Maschera: binaria uint8 (255→1, 0→0)
        mask    = cv2.imread(os.path.join(src_labels, fname), cv2.IMREAD_GRAYSCALE)
        mask_b  = (mask > 127).astype(np.uint8)
        mask3d  = np.expand_dims(mask_b, axis=2)
        nib.save(nib.Nifti1Image(mask3d, affine),
                 os.path.join(dest_labels, f"{case_id}.nii.gz"))

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n} ...")

    # dataset.json richiesto da nnU-Net
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels":        {"background": 0, "circle": 1},
        "numTraining":   n,
        "file_ending":   ".nii.gz",
        "name":          "Shapes",
        "description":   f"Geometric shapes – esperimento {FOLDER_NAME}",
    }
    with open(os.path.join(dest_dataset, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"  ✓ {n} coppie convertite")


def run_preprocessing():
    """Lancia nnUNetv2_plan_and_preprocess con env isolato."""
    _separator("STEP 2b: PREPROCESSING nnU-Net")

    preprocessed_dataset = os.path.join(NNUNET_PREPROCESSED_DIR, DATASET_NAME)
    if os.path.exists(preprocessed_dataset):
        print(f"  Preprocessing già presente, skip: {preprocessed_dataset}")
        return

    _run_subprocess(
        ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "--verify_dataset_integrity"],
        "Preprocessing",
    )
    print("  ✓ Preprocessing completato")


# ==============================================================================
#  STEP 3  –  Training
# ==============================================================================

def _find_nnunetv2_trainer_dir():
    """Trova la directory trainer nel package nnunetv2 installato."""
    try:
        import nnunetv2 as _nn  # noqa: PLC0415
        nnunetv2_root = os.path.dirname(_nn.__file__)
        trainer_dir   = os.path.join(nnunetv2_root, "training", "nnUNetTrainer")
        if not os.path.isdir(trainer_dir):
            raise FileNotFoundError(trainer_dir)
        return trainer_dir
    except (ImportError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Package nnunetv2 non trovato o struttura inattesa: {e}\n"
            f"Assicurati che nnunetv2 sia installato nell'ambiente corrente."
        ) from e


def _write_geometric_config(trainer_dir):
    """
    Genera geometric_config.py con i valori dell'esperimento corrente e lo scrive in:
      1. trainer_dir  (package nnunetv2 installato)
      2. SCRIPT_DIR   (cwd dei subprocess nnUNetv2_train / nnUNetv2_predict)

    Il punto (2) è critico: quando il trainer gira come subprocess con
    cwd=geometrica/, Python trova PRIMA il geometric_config.py nella cwd rispetto
    a quello nel package. Scrivendo in entrambe le posizioni garantiamo coerenza.
    """
    content = f'''"""
geometric_config.py – auto-generato da run_experiment.py
Esperimento : {FOLDER_NAME}
Generato il : {datetime.now().isoformat(timespec='seconds')}
NON modificare manualmente: viene sovrascritto da ogni run di run_experiment.py.
"""

WEIGHT_COMPACTNESS  = {WEIGHT_COMPACTNESS}
WEIGHT_ECCENTRICITY = {WEIGHT_ECCENTRICITY}
WEIGHT_BOUNDARY     = {WEIGHT_BOUNDARY}

WARMUP_EPOCHS          = {WARMUP_EPOCHS}
NUM_EPOCHS             = {EPOCHS}
BATCH_SIZE             = {BATCH_SIZE}
GEOMETRIC_LOSS_SAMPLES = {GEOMETRIC_LOSS_SAMPLES}

MIN_AREA_THRESHOLD = 50.0
SQRT_CLAMP_MIN     = 1e-2
'''
    for dest in [
        os.path.join(trainer_dir, "geometric_config.py"),
        os.path.join(SCRIPT_DIR,  "geometric_config.py"),
    ]:
        with open(dest, "w") as f:
            f.write(content)
        print(f"    ✓ geometric_config.py → {dest}")


def _patch_epochs_in_trainer(trainer_dir):
    """
    Patcha i valori di epoche direttamente nei file trainer installati,
    sostituendo le variabili/costanti hardcoded con i valori del blocco config.

    Perché è necessario:
    - nnUNetTrainerBaseline ha self.num_epochs = 100 hardcoded nel sorgente.
    - nnUNetTrainerGeometric: NUM_EPOCHS, WARMUP_EPOCHS e BATCH_SIZE vengono
      patchati direttamente per sicurezza (anche se _write_geometric_config
      già scrive il config nella cwd del subprocess).
    Il patch diretto garantisce coerenza anche in casi edge (cache di moduli, ecc.).
    """
    import re  # noqa: PLC0415

    # ── Baseline: self.num_epochs e BATCH_SIZE ────────────────────────────
    target_b = os.path.join(trainer_dir, "nnUNetTrainerBaseline.py")
    if os.path.exists(target_b):
        with open(target_b) as f:
            content = f.read()

        # Patcha epoche
        new_content, n_ep = re.subn(
            r'self\.num_epochs\s*=\s*\d+',
            f'self.num_epochs = {EPOCHS}',
            content,
        )
        # Patcha BATCH_SIZE (placeholder nel sorgente: "BATCH_SIZE = 8")
        new_content, n_bs = re.subn(
            r'^BATCH_SIZE\s*=\s*\d+',
            f'BATCH_SIZE = {BATCH_SIZE}',
            new_content,
            flags=re.MULTILINE,
        )
        if n_ep > 0 or n_bs > 0:
            with open(target_b, "w") as f:
                f.write(new_content)
            print(f"    ✓ nnUNetTrainerBaseline    → epochs={EPOCHS}, batch={BATCH_SIZE}")
        else:
            print(f"    ⚠  pattern non trovati in nnUNetTrainerBaseline.py")
    else:
        print(f"    ⚠  nnUNetTrainerBaseline.py non trovato, skip")

    # ── Geometric: epoche, warmup, batch_size ────────────────────────────────
    target_g = os.path.join(trainer_dir, "nnUNetTrainerGeometric.py")
    if os.path.exists(target_g):
        with open(target_g) as f:
            content = f.read()

        new_content = re.sub(
            r'self\.num_epochs\s*=\s*NUM_EPOCHS',
            f'self.num_epochs = {EPOCHS}',
            content,
        )
        new_content = re.sub(
            r'self\.geometric_loss_warmup_epochs\s*=\s*WARMUP_EPOCHS',
            f'self.geometric_loss_warmup_epochs = {WARMUP_EPOCHS}',
            new_content,
        )
        # batch_size: il sorgente usa BATCH_SIZE (variabile importata).
        # Aggiungiamo una riga di override esplicita dopo super().__init__ come
        # ulteriore safety net, nel caso l'import di geometric_config fallisse.
        if f'self._patched_batch_size = {BATCH_SIZE}' not in new_content:
            new_content = new_content.replace(
                'super().__init__(plans, configuration, fold, dataset_json, device)',
                f'super().__init__(plans, configuration, fold, dataset_json, device)\n'
                f'        self._patched_batch_size = {BATCH_SIZE}  # sentinel patch',
            )
        with open(target_g, "w") as f:
            f.write(new_content)
        print(f"    ✓ nnUNetTrainerGeometric      → epochs={EPOCHS}, "
              f"warmup={WARMUP_EPOCHS}, batch={BATCH_SIZE}")
    else:
        print(f"    ⚠  nnUNetTrainerGeometric.py non trovato, skip")


def install_trainer_files():
    """
    Copia nnUNetTrainerBaseline, nnUNetTrainerGeometric, geometric_losses
    nella directory trainer del package nnunetv2 installato.
    Patcha le epoche e scrive geometric_config.py con i valori del blocco config.
    """
    _separator("STEP 3a: INSTALLAZIONE TRAINER FILES")

    trainer_dir = _find_nnunetv2_trainer_dir()
    print(f"  Target: {trainer_dir}")

    for fname in ["nnUNetTrainerBaseline.py", "nnUNetTrainerGeometric.py", "geometric_losses.py"]:
        src = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(trainer_dir, fname))
            print(f"    ✓ {fname}")
        else:
            print(f"    ⚠  Non trovato (skip): {fname}")

    # Patch epoche nel trainer baseline (hardcoded nel sorgente, va sovrascritto)
    _patch_epochs_in_trainer(trainer_dir)

    # Geometric config con pesi e NUM_EPOCHS dell'esperimento corrente
    _write_geometric_config(trainer_dir)
    print(f"  ✓ Tutti i trainer installati  (epoche → {EPOCHS})")


def run_training_single(net_type):
    """Lancia nnUNetv2_train per un singolo tipo di rete."""
    trainer_name = TRAINERS[net_type]
    t0 = time.perf_counter()
    print(f"\n  {'─'*55}")
    print(f"  Training → {net_type.upper()}  ({trainer_name})")
    print(f"  {'─'*55}")
    # nnUNet_n_proc_DA=0: disabilita worker multiprocessing per evitare
    # "No space left on device" su /dev/shm (63 MB nei container Docker).
    # NON impostare globalmente: rompe il preprocessing (torch.set_num_threads).
    #
    # PYTHONPATH=SCRIPT_DIR: quando nnUNetv2_train gira come entry-point script,
    # Python imposta sys.path[0]='/venv/bin/' invece di '' (CWD). Senza PYTHONPATH,
    # 'from geometric_config import ...' non trova geometric_config.py nella CWD
    # e scatta sempre il fallback hardcoded, ignorando i pesi impostati dall'utente.
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = f"{SCRIPT_DIR}:{existing_pp}" if existing_pp else SCRIPT_DIR
    device = _resolve_device()
    print(f"  Device     : {device}")
    _run_subprocess(
        ["nnUNetv2_train", str(DATASET_ID), "2d", "0", "-tr", trainer_name, "-device", device],
        f"Training {net_type}",
        extra_env={"nnUNet_n_proc_DA": "0", "PYTHONPATH": pythonpath},
    )
    elapsed = time.perf_counter() - t0
    print(f"  ✓ Training {net_type} completato ({_format_duration(elapsed)})")
    return elapsed


def run_all_training():
    """Lancia il training per le reti configurate in RETI_DA_ALLENARE."""
    _separator("STEP 3b: TRAINING")
    nets = _get_nets_to_run()
    print(f"  Reti     : {nets}")
    print(f"  Dataset  : {DATASET_ID}  ({DATASET_NAME})")
    print(f"  Epoche   : {EPOCHS}")
    training_times = {"baseline": 0.0, "geometrica": 0.0}
    for net_type in nets:
        training_times[net_type] += run_training_single(net_type)
    print(f"\n  ✓ Training completato per: {nets}")
    return training_times


def copy_geometric_verify_log_if_present():
    """
    Copia verify.log della rete geometrica in experiments/<run>/logs/verify.log.
    Se non presente, stampa un warning non bloccante.
    """
    logs_dir = os.path.join(EXP_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    src = os.path.join(
        NNUNET_RESULTS_DIR,
        DATASET_NAME,
        f"{TRAINERS['geometrica']}__nnUNetPlans__2d",
        "fold_0",
        "verify.log",
    )
    dst = os.path.join(logs_dir, "verify.log")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✓ Verify log copiato in: {dst}")
    else:
        print(f"  ⚠  Verify log non trovato (skip): {src}")


def _verify_geometric_config_in_log():
    """
    Legge il training log della rete geometrica, estrae i pesi effettivamente
    usati dal trainer e li confronta con quelli configurati in run_experiment.py.

    Appende una sezione POST-TRAINING WEIGHT VERIFICATION al verify.log
    dell'esperimento (già copiato da copy_geometric_verify_log_if_present).
    Stampa anche un riepilogo a console.

    Serve a rilevare il bug 'PYTHONPATH mancante': quando geometric_config.py
    non viene trovato nel sys.path del subprocess, il trainer usa valori
    hardcoded (fallback) senza alcun errore esplicito.
    """
    log_path    = os.path.join(EXP_DIR, "logs", "training_geometrica.log")
    verify_path = os.path.join(EXP_DIR, "logs", "verify.log")

    if not os.path.exists(log_path):
        return

    # Estrai la riga "Pesi:" dal log del trainer
    pesi_line = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("Pesi:"):
                pesi_line = line.strip()
                break

    if pesi_line is None:
        return

    # Parsing: "Pesi: compactness=0.01  eccentricity=0.03  boundary=0.005"
    actual = {}
    for key in ("compactness", "eccentricity", "boundary"):
        m = re.search(rf"{key}=([\d.e+\-]+)", pesi_line)
        if m:
            actual[key] = float(m.group(1))

    expected = {
        "compactness":  WEIGHT_COMPACTNESS,
        "eccentricity": WEIGHT_ECCENTRICITY,
        "boundary":     WEIGHT_BOUNDARY,
    }

    mismatches = {
        k for k in expected
        if abs(actual.get(k, float("nan")) - expected[k]) > 1e-9
    }
    all_ok = len(mismatches) == 0

    # Appende al verify.log dell'esperimento
    with open(verify_path, "a", encoding="utf-8") as f:
        f.write("\n\nPOST-TRAINING WEIGHT VERIFICATION (run_experiment.py)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Pesi configurati in run_experiment.py:\n")
        for k in ("compactness", "eccentricity", "boundary"):
            f.write(f"  {k:<15} {expected[k]}\n")
        f.write("\nPesi usati dal trainer (da training_geometrica.log):\n")
        for k in ("compactness", "eccentricity", "boundary"):
            used = actual.get(k, "N/A")
            status = "OK" if k not in mismatches else "⚠  MISMATCH"
            f.write(f"  {k:<15} {used}  [{status}]\n")
        f.write("\n")
        if all_ok:
            f.write("OK: Tutti i pesi usati corrispondono alla configurazione.\n")
        else:
            f.write("WARNING: Pesi usati DIVERSI da run_experiment.py!\n")
            f.write("  Causa: geometric_config.py non trovato nel sys.path del subprocess.\n")
            f.write("  Fix applicato in questa versione: PYTHONPATH viene passato al subprocess.\n")

    # Stampa riepilogo a console
    print(f"\n  {'─'*55}")
    print(f"  Weight verification:")
    for k in ("compactness", "eccentricity", "boundary"):
        used = actual.get(k, "N/A")
        icon = "✅" if k not in mismatches else "⚠️ "
        print(f"    {icon} {k:<15} atteso={expected[k]}  usato={used}")
    if all_ok:
        print(f"  ✅ OK: pesi corrispondono alla configurazione")
    else:
        print(f"  ⚠️  WARNING: pesi NON corrispondono — vedi verify.log")
    print(f"  {'─'*55}\n")


# ==============================================================================
#  STEP 4  –  Inference + conversione NIfTI → PNG
# ==============================================================================

def _convert_test_to_nifti(temp_dir):
    """
    Converte 1_dataset/test/images/*.png in NIfTI dentro temp_dir.
    Necessario per nnUNetv2_predict che accetta solo file .nii.gz.
    """
    import cv2        # noqa: PLC0415
    import numpy as np # noqa: PLC0415
    import nibabel as nib  # noqa: PLC0415

    src   = os.path.join(TEST_DIR, "images")
    files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    affine = np.eye(4)

    for fname in files:
        case_id = os.path.splitext(fname)[0]
        img   = cv2.imread(os.path.join(src, fname), cv2.IMREAD_GRAYSCALE)
        img3d = np.expand_dims(img.astype(np.float32), axis=2)
        nib.save(nib.Nifti1Image(img3d, affine),
                 os.path.join(temp_dir, f"{case_id}_0000.nii.gz"))

    print(f"  {len(files)} immagini test → NIfTI (dir temporanea)")
    return files


def _nifti_preds_to_png(nifti_dir, png_out_dir, min_component_px=0):
    """
    Converte predizioni NIfTI di nnU-Net (valori 0/1) in PNG (0/255).

    ATTENZIONE: nnU-Net salva le predizioni con valori 0 (background) e 1
    (foreground). Senza la moltiplicazione ×255 il PNG risulterebbe tutto
    nero poiché il valore 1 è invisibile come intensità uint8.

    Parameters
    ----------
    min_component_px : se > 0, rimuove componenti connesse più piccole di questo
                       numero di pixel (elimina blob spurii che non sono il cerchio).
                       Converte "blob piccolo e sbagliato" → predizione vuota,
                       più onesta metricamente (dice=0, ecc=NaN esclusa dalla media).
    """
    import nibabel as nib  # noqa: PLC0415
    import numpy as np     # noqa: PLC0415
    import cv2             # noqa: PLC0415

    os.makedirs(png_out_dir, exist_ok=True)
    nii_files     = sorted(f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz"))
    n_removed     = 0
    n_cases_clean = 0

    for fname in nii_files:
        case_id  = fname.replace(".nii.gz", "")
        data     = nib.load(os.path.join(nifti_dir, fname)).get_fdata()
        if data.ndim == 3:
            data = data[:, :, 0]
        pred_png = (data * 255).astype(np.uint8)

        if min_component_px > 0 and pred_png.any():
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                pred_png, connectivity=8
            )
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_component_px:
                    pred_png[labels == i] = 0
                    n_removed += 1
                    n_cases_clean += 1

        cv2.imwrite(os.path.join(png_out_dir, f"{case_id}.png"), pred_png)

    msg = f"  {len(nii_files)} predizioni NIfTI → PNG  ({png_out_dir})"
    if min_component_px > 0 and n_removed > 0:
        msg += f"  [{n_cases_clean} blob spurii rimossi (<{min_component_px}px)]"
    elif min_component_px > 0:
        msg += f"  [nessun blob spurio rimosso]"
    print(msg)


def _check_checkpoint_weights(net_type):
    """
    Verifica che i pesi del checkpoint non contengano NaN/Inf prima dell'inference.
    Ritorna True se il checkpoint è integro, False altrimenti.
    Stampa un warning dettagliato in caso di corruzione (tipicamente causata da
    gradient explosion durante il training geometrico).
    """
    import torch  # noqa: PLC0415

    trainer_name   = TRAINERS[net_type]
    checkpoint_dir = os.path.join(
        NNUNET_RESULTS_DIR,
        DATASET_NAME,
        f"{trainer_name}__nnUNetPlans__2d",
        "fold_0",
    )
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pth")

    if not os.path.exists(checkpoint_path):
        print(f"  ⚠  Checkpoint non trovato: {checkpoint_path}")
        return False

    print(f"  Verifica checkpoint {net_type}...")
    try:
        import warnings  # noqa: PLC0415
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = ckpt.get("network_weights", {})
        for key, tensor in weights.items():
            if torch.isnan(tensor).any():
                print(f"  ✗ NaN nei pesi: {key}")
                print(f"    Il modello è corrotto — ri-allena con pesi geometrici più bassi.")
                return False
            if torch.isinf(tensor).any():
                print(f"  ✗ Inf nei pesi: {key}")
                print(f"    Il modello è corrotto — ri-allena con pesi geometrici più bassi.")
                return False
        first_key    = next(iter(weights))
        first_tensor = weights[first_key]
        print(f"  ✓ Checkpoint OK  "
              f"(mean={first_tensor.mean().item():.4f}, "
              f"std={first_tensor.std().item():.4f})")
        return True
    except Exception as e:
        print(f"  ⚠  Errore durante verifica checkpoint: {e}")
        return False


def run_inference_single(net_type):
    """Esegue inference per un tipo di rete sul test set, ritorna path cartella PNG."""
    trainer_name  = TRAINERS[net_type]
    nifti_out_dir = os.path.join(PRED_DIR, f"{net_type}_nifti")
    png_out_dir   = os.path.join(PRED_DIR, net_type)

    os.makedirs(nifti_out_dir, exist_ok=True)

    print(f"\n  {'─'*55}")
    print(f"  Predizione → {net_type.upper()}  ({trainer_name})")
    print(f"  {'─'*55}")

    if not _check_checkpoint_weights(net_type):
        raise RuntimeError(
            f"Checkpoint corrotto per '{net_type}'. Predizione annullata.\n"
            f"Ri-allena il modello prima di procedere."
        )

    with tempfile.TemporaryDirectory(prefix="nnunet_inf_") as tmp_input:
        _convert_test_to_nifti(tmp_input)
        device = _resolve_device()
        _run_subprocess([
            "nnUNetv2_predict",
            "-i", tmp_input,
            "-o", nifti_out_dir,
            "-d", str(DATASET_ID),
            "-c", "2d",
            "-f", "0",
            "-tr", trainer_name,
            "-device", device,
            "--disable_tta",
        ], f"Inferenza {net_type}",
        extra_env={"nnUNet_n_proc_DA": "0"})

    _nifti_preds_to_png(nifti_out_dir, png_out_dir, min_component_px=MIN_COMPONENT_PX)
    print(f"  ✓ Maschere salvate in: {png_out_dir}")
    return png_out_dir


def run_all_inference():
    """Predizione per tutte le reti allenate."""
    _separator("STEP 4: PREDIZIONE SUL TEST SET")
    nets = _get_nets_to_run()
    for net_type in nets:
        run_inference_single(net_type)
    print(f"\n  ✓ Predizioni completate → {PRED_DIR}")


# ==============================================================================
#  STEP 5  –  Confronto finale: metriche + visualizzazioni
# ==============================================================================

def _aggregate_metrics(metrics_list):
    """
    Calcola statistiche aggregate da una lista di dict metriche per-immagine.

    - NaN (compactness/eccentricity su pred vuote) vengono esclusi dal calcolo:
      una pred vuota non contribuisce alle metriche di forma, altrimenti
      i fallimenti (eccentricity=0.0 = "cerchio perfetto"!) distorcono la media.
    - Viene calcolata anche la mediana, più robusta agli outlier estremi
      di Hausdorff (es. il valore 724px per pred vuote).
    - {key}_n conta solo i campioni validi (non-NaN).
    """
    import numpy as np  # noqa: PLC0415
    agg = {}
    for key in metrics_list[0]:
        all_vals = [m[key] for m in metrics_list]
        vals = [v for v in all_vals if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            vals = [0.0]
        agg[f"{key}_mean"]   = float(np.mean(vals))
        agg[f"{key}_median"] = float(np.median(vals))
        agg[f"{key}_p95"]    = float(np.percentile(vals, 95))
        agg[f"{key}_std"]    = float(np.std(vals))
        agg[f"{key}_min"]    = float(np.min(vals))
        agg[f"{key}_max"]    = float(np.max(vals))
        agg[f"{key}_n"]      = len(vals)
    return agg


def run_comparison():
    """
    Carica predizioni PNG (3_predizioni/) e GT PNG (1_dataset/test/labels/),
    calcola tutte le metriche e salva risultati in 4_confronto_finale/.
    """
    _separator("STEP 5: CONFRONTO FINALE")

    import cv2         # noqa: PLC0415
    import numpy as np # noqa: PLC0415

    nets = _get_nets_to_run()
    viz_dir = os.path.join(CONFRONTO_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    test_img_dir = os.path.join(TEST_DIR, "images")
    test_lbl_dir = os.path.join(TEST_DIR, "labels")
    test_files   = sorted(f for f in os.listdir(test_img_dir) if f.endswith(".png"))

    print(f"  Immagini test : {len(test_files)}")
    print(f"  Reti          : {nets}")

    metrics_per_net = {net: [] for net in nets}

    for idx, fname in enumerate(test_files):
        case_id = os.path.splitext(fname)[0]

        img_orig = cv2.imread(os.path.join(test_img_dir, fname), cv2.IMREAD_GRAYSCALE)
        gt_raw   = cv2.imread(os.path.join(test_lbl_dir, fname), cv2.IMREAD_GRAYSCALE)
        # Normalizza in [0,1] float per le funzioni metriche (usano soglia 0.5)
        gt_f = gt_raw.astype(np.float32) / 255.0

        preds = {}
        for net_type in nets:
            pred_path = os.path.join(PRED_DIR, net_type, f"{case_id}.png")
            if not os.path.exists(pred_path):
                print(f"    ⚠  Predizione non trovata: {pred_path} — skip")
                continue
            pred_raw = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred_f   = pred_raw.astype(np.float32) / 255.0
            preds[net_type] = pred_f
            metrics_per_net[net_type].append(calculate_all_metrics(pred_f, gt_f))

        # Visualizzazione
        if len(nets) == 2 and "baseline" in preds and "geometrica" in preds:
            create_comparison_visualization(
                img_orig, preds["baseline"], preds["geometrica"], gt_f, case_id, viz_dir
            )
        elif len(nets) == 1:
            net_type = nets[0]
            if net_type in preds and metrics_per_net[net_type]:
                create_visualization(
                    img_orig, preds[net_type], gt_f, case_id, viz_dir,
                    metrics_per_net[net_type][-1],
                    title_prefix=f"{net_type.capitalize()} – ",
                )

        if (idx + 1) % 10 == 0:
            print(f"    {idx+1}/{len(test_files)} immagini processate...")

    # Aggregazione metriche
    aggregate = {
        net: _aggregate_metrics(ml)
        for net, ml in metrics_per_net.items()
        if ml
    }

    # Grafico a barre (solo con entrambe le reti)
    if "baseline" in aggregate and "geometrica" in aggregate:
        create_metrics_comparison_chart(aggregate["baseline"], aggregate["geometrica"], CONFRONTO_DIR)

    # JSON
    comparison_data = {**aggregate, "num_images": len(test_files)}
    with open(os.path.join(CONFRONTO_DIR, "metrics_comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)

    # TXT leggibile
    _METRIC_NAMES   = ["dice", "iou", "compactness", "eccentricity",
                       "hausdorff_distance", "boundary_iou"]
    _SHAPE_METRICS  = {"compactness", "eccentricity"}   # solo su pred non-vuote
    _LOWER_BETTER   = {"hausdorff_distance", "eccentricity"}

    txt_path = os.path.join(CONFRONTO_DIR, "metrics_comparison.txt")
    with open(txt_path, "w") as f:
        f.write(f"CONFRONTO METRICHE AGGREGATE – {FOLDER_NAME}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Immagini test analizzate: {len(test_files)}\n")
        f.write(f"  nota: compactness/eccentricity escluse quando pred è vuota\n")
        f.write(f"  nota: hausdorff_distance — mediana usata nel confronto (robusta agli outlier)\n")

        for net_type, agg in aggregate.items():
            f.write(f"\n{'─'*40}\n{net_type.upper()}\n{'─'*40}\n")
            for m in _METRIC_NAMES:
                mean   = agg.get(f"{m}_mean",   0.0)
                std    = agg.get(f"{m}_std",    0.0)
                p95    = agg.get(f"{m}_p95",    0.0)
                n      = agg.get(f"{m}_n",      len(test_files))
                if m in _SHAPE_METRICS:
                    f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}  (n={n}/{len(test_files)} pred non-vuote)\n")
                elif m == "hausdorff_distance":
                    f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}  [p95: {p95:.4f}]\n")
                else:
                    f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}\n")

        if "baseline" in aggregate and "geometrica" in aggregate:
            f.write(f"\n\n{'─'*40}\nMIGLIORAMENTI GEOMETRIC vs BASELINE\n{'─'*40}\n")
            f.write(f"  (hausdorff_distance: confronto su p95; altri: su media)\n")
            for m in _METRIC_NAMES:
                key = "p95" if m == "hausdorff_distance" else "mean"
                b = aggregate["baseline"].get(f"{m}_{key}",   0.0)
                g = aggregate["geometrica"].get(f"{m}_{key}", 0.0)
                if m in _LOWER_BETTER:
                    imp = ((b - g) / b * 100) if b != 0 else 0.0
                else:
                    imp = ((g - b) / b * 100) if b != 0 else 0.0
                f.write(f"  {m:25s}: {imp:+.2f}%\n")

    # vis_bad: peggiori predizioni per rete + chart metriche filtrato
    if "baseline" in aggregate and "geometrica" in aggregate:
        nets_pred_map_local = {nt: os.path.join(PRED_DIR, nt) for nt in nets}
        create_vis_bad(test_img_dir, test_lbl_dir, nets_pred_map_local, CONFRONTO_DIR)

    print(f"  ✓ Risultati in: {CONFRONTO_DIR}")
    print(f"    visualizations/              ({len(test_files)} immagini)")
    print(f"    vis_bad/                     (10 peggiori per rete + chart filtrato)")
    print(f"    metrics_comparison.json")
    print(f"    metrics_comparison.txt")
    if "baseline" in aggregate and "geometrica" in aggregate:
        print(f"    metrics_comparison_chart.png")


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main():
    t_pipeline_start = time.perf_counter()
    _separator("MASTER SCRIPT  –  PIPELINE COMPLETA (Step 1-5)")
    print(f"  Cartella : experiments/{FOLDER_NAME}")
    print(f"  Dataset  : {DATASET_ID} ({DATASET_NAME})")
    print(f"  Reti     : {RETI_DA_ALLENARE}")
    print(f"  Modalità : {'AUTOMATICA' if AUTOMATIC == 'si' else 'INTERATTIVA'}")
    print(f"  Device   : {_resolve_device()}  (config DEVICE='{DEVICE}')")

    # ── Setup interattivo: nome cartella + conferma dataset ────────────────
    if AUTOMATIC == "no":
        _interactive_setup()

    # ── STEP 1: Dataset ────────────────────────────────────────────────────
    create_experiment_structure()
    raw_dir = generate_dataset()
    n_train, n_test = split_and_organize(raw_dir)
    cleanup_raw(raw_dir)
    save_config_yaml(n_train, n_test)

    # ── Breve riepilogo dataset + conferma training ────────────────────────
    if AUTOMATIC == "no":
        print()
        print(f"  ✓ Dataset creato: {n_train} immagini train  |  {n_test} immagini test")
        _interactive_confirm_training()

    # ── STEP 2: Conversione + Preprocessing ────────────────────────────────
    convert_train_to_nnunet()
    run_preprocessing()

    # ── STEP 3: Training ───────────────────────────────────────────────────
    install_trainer_files()
    training_times = run_all_training()
    if "geometrica" in _get_nets_to_run():
        copy_geometric_verify_log_if_present()
        _verify_geometric_config_in_log()

    # ── Conferma confronto finale ──────────────────────────────────────────
    if AUTOMATIC == "no":
        if not _interactive_confirm_confronto():
            _separator("PIPELINE INTERROTTA")
            print(f"  Training completato. Confronto finale non eseguito.")
            print(f"  I modelli si trovano in: {NNUNET_RESULTS_DIR}")
            print(f"{'='*65}\n")
            return

    # ── STEP 4: Inference ──────────────────────────────────────────────────
    run_all_inference()

    # ── STEP 5: Confronto ──────────────────────────────────────────────────
    run_comparison()

    # ── Riepilogo finale ───────────────────────────────────────────────────
    _separator("PIPELINE COMPLETATA")
    print(f"  Esperimento : {FOLDER_NAME}")
    print(f"  Output      : {EXP_DIR}")
    print()
    print(f"  1_dataset/           – immagini train ({n_train}) + test ({n_test})")
    print(f"  2_nnunet_engine/     – checkpoint modelli trained")
    print(f"  3_predizioni/        – predizioni PNG per rete")
    print(f"  4_confronto_finale/  – metriche, chart, visualizzazioni")
    print()
    print(f"  Tempo training BASELINE   : {_format_duration(training_times.get('baseline', 0.0))}")
    print(f"  Tempo training GEOMETRICA : {_format_duration(training_times.get('geometrica', 0.0))}")
    print(f"  Tempo totale pipeline     : {_format_duration(time.perf_counter() - t_pipeline_start)}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
