"""
test.py – Analisi standalone dei risultati di un esperimento geometrica.

Scansiona la cartella experiments/, chiede all'utente quale esperimento analizzare
e in quale modalità:

  [1] Analizza le predizioni già presenti (3_predizioni/) di questo esperimento
  [2] Usa un modello allenato in un altro esperimento senza riallenare

La modalità 2 esegue l'inference sul dataset dell'esperimento scelto usando il
checkpoint di un altro esperimento, e salva i risultati in:
  4_analisi_ext_{source_experiment}_{net_type}/

Uso:
    cd /workspace/geometrica
    python test.py

Author: Francesco + Claude
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import cv2  # pyright: ignore[reportMissingImports]

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "experiments")

TRAINERS = {
    "baseline":   "nnUNetTrainerBaseline",
    "geometrica": "nnUNetTrainerGeometric",
}

sys.path.insert(0, SCRIPT_DIR)
from metrics_utils import (  # noqa: E402
    calculate_all_metrics,
    create_visualization,
    create_comparison_visualization,
    create_metrics_comparison_chart,
    create_vis_bad,
)

# ==============================================================================
#  BOOTSTRAP TRAINER CUSTOM  –  nessun file copiato nel package nnunetv2
# ==============================================================================
# Stessa soluzione usata da run_experiment.py (vedi commento li' per il
# dettaglio): i trainer custom (nnUNetTrainerBaseline, nnUNetTrainerGeometric,
# geometric_losses) restano in geometrica/ e vengono trovati a runtime senza
# richiedere permessi di scrittura su site-packages.
_NNUNET_BOOTSTRAP_TEMPLATE = '''\
import sys
script_dir = {script_dir!r}
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import nnunetv2.training.nnUNetTrainer as _trainer_pkg
if script_dir not in _trainer_pkg.__path__:
    _trainer_pkg.__path__.insert(0, script_dir)

import nnunetv2.utilities.find_class_by_name as _fcbn
_original_rfpc = _fcbn.recursive_find_python_class


def _patched_rfpc(folder, class_name, current_module):
    result = _original_rfpc(folder, class_name, current_module)
    if result is None:
        result = _original_rfpc(script_dir, class_name, "nnunetv2.training.nnUNetTrainer")
    return result


_fcbn.recursive_find_python_class = _patched_rfpc

from {entry_module} import {entry_func}
{entry_func}()
'''


def _run_nnunet_entry(entry_module, entry_func, cli_args, env=None):
    """Lancia un entry point nnU-Net (es. predict_entry_point) risolvendo i
    trainer custom direttamente da geometrica/, senza CLI nnUNetv2_predict."""
    bootstrap_code = _NNUNET_BOOTSTRAP_TEMPLATE.format(
        script_dir=SCRIPT_DIR, entry_module=entry_module, entry_func=entry_func,
    )
    cmd = [sys.executable, "-c", bootstrap_code] + list(cli_args)
    return subprocess.run(cmd, env=env)


_METRIC_NAMES = ["dice", "iou", "compactness", "eccentricity",
                 "hausdorff_distance", "boundary_iou"]
_SHAPE_METRICS = {"compactness", "eccentricity"}
_LOWER_BETTER  = {"hausdorff_distance", "eccentricity"}


# ==============================================================================
#  MENU INTERATTIVO
# ==============================================================================

def _list_experiments():
    if not os.path.exists(EXPERIMENTS_DIR):
        return []
    return sorted(
        d for d in os.listdir(EXPERIMENTS_DIR)
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))
    )


def _choose_experiment():
    experiments = _list_experiments()
    if not experiments:
        print(f"\n  Nessun esperimento trovato in:\n  {EXPERIMENTS_DIR}")
        return None

    print("\n  Esperimenti disponibili:")
    for i, e in enumerate(experiments, 1):
        print(f"    [{i}] {e}")

    while True:
        try:
            choice = int(input(f"\n  Scegli esperimento (1-{len(experiments)}): ")) - 1
            if 0 <= choice < len(experiments):
                return experiments[choice]
        except ValueError:
            pass
        print("  → Scelta non valida.")


def _choose_mode():
    """Chiede come procedere: predizioni esistenti o modello esterno."""
    print("\n  Modalità di analisi:")
    print("    [1] Analizza le predizioni esistenti di questo esperimento")
    print("    [2] Usa un modello già allenato in un altro esperimento (no nuovo training)")
    while True:
        choice = input("\n  Scelta (1/2): ").strip()
        if choice in ("1", "2"):
            return choice
        print("  → Scelta non valida.")


def _choose_nets(exp_dir):
    """Determina le reti disponibili in 3_predizioni/ e chiede all'utente."""
    pred_dir  = os.path.join(exp_dir, "3_predizioni")
    available = [
        n for n in ("baseline", "geometrica")
        if os.path.isdir(os.path.join(pred_dir, n))
    ]
    if not available:
        print(f"\n  Nessuna predizione trovata in:\n  {pred_dir}")
        return None

    print("\n  Reti disponibili:")
    for i, n in enumerate(available, 1):
        print(f"    [{i}] {n}")
    if len(available) == 2:
        print(f"    [3] entrambe (confronto)")

    while True:
        choice = input("\n  Scelta: ").strip()
        if choice == "1":
            return [available[0]]
        if choice == "2" and len(available) >= 2:
            return [available[1]]
        if choice == "3" and len(available) == 2:
            return available
        print("  → Scelta non valida.")


def _find_trained_models():
    """
    Scansiona tutti gli esperimenti e ritorna quelli con almeno un checkpoint.
    Ritorna lista di tuple: (exp_name, net_type, dataset_id, checkpoint_path)
    """
    found = []
    for exp_name in _list_experiments():
        results_dir = os.path.join(EXPERIMENTS_DIR, exp_name,
                                   "2_nnunet_engine", "nnUNet_results")
        if not os.path.isdir(results_dir):
            continue
        for dset_folder in sorted(os.listdir(results_dir)):
            if not (dset_folder.startswith("Dataset") and dset_folder.endswith("_Shapes")):
                continue
            try:
                dataset_id = int(dset_folder[7:10])
            except ValueError:
                continue
            dset_path = os.path.join(results_dir, dset_folder)
            for net_type, trainer_name in TRAINERS.items():
                ckpt = os.path.join(dset_path,
                                    f"{trainer_name}__nnUNetPlans__2d",
                                    "fold_0", "checkpoint_final.pth")
                if os.path.exists(ckpt):
                    found.append((exp_name, net_type, dataset_id, ckpt))
    return found


def _choose_external_model():
    """
    Chiede prima quale tipo di rete (baseline / geometrica / entrambe),
    poi mostra solo gli esperimenti che hanno quel tipo allenato.
    Ritorna lista di tuple (exp_name, net_type, dataset_id) — una o due.
    """
    all_models = _find_trained_models()
    if not all_models:
        print("\n  Nessun esperimento con checkpoint allenati trovato.")
        return None

    # Step 1: scegli tipo di rete
    available_types = sorted({net for _, net, _, _ in all_models})
    print("\n  Quale tipo di rete vuoi usare?")
    opts = available_types[:]
    if len(available_types) == 2:
        opts.append("entrambe")
    for i, o in enumerate(opts, 1):
        print(f"    [{i}] {o}")

    while True:
        try:
            t = int(input(f"\n  Scelta (1-{len(opts)}): ")) - 1
            if 0 <= t < len(opts):
                chosen_types = available_types if opts[t] == "entrambe" else [opts[t]]
                break
        except ValueError:
            pass
        print("  → Scelta non valida.")

    # Step 2: filtra esperimenti che hanno TUTTI i tipi scelti
    # raggruppa per esperimento: exp -> {net_type: (dataset_id, ckpt)}
    exp_map = {}
    for exp, net, did, ckpt in all_models:
        exp_map.setdefault(exp, {})[net] = (did, ckpt)

    valid_exps = sorted(
        exp for exp, nets in exp_map.items()
        if all(t in nets for t in chosen_types)
    )
    if not valid_exps:
        print(f"\n  Nessun esperimento ha entrambi i tipi: {chosen_types}")
        return None

    print(f"\n  Esperimenti disponibili ({', '.join(chosen_types)}):")
    for i, exp in enumerate(valid_exps, 1):
        dids = " / ".join(str(exp_map[exp][t][0]) for t in chosen_types)
        print(f"    [{i}] {exp}  (Dataset {dids})")

    while True:
        try:
            c = int(input(f"\n  Scegli esperimento (1-{len(valid_exps)}): ")) - 1
            if 0 <= c < len(valid_exps):
                chosen_exp = valid_exps[c]
                break
        except ValueError:
            pass
        print("  → Scelta non valida.")

    # Ritorna lista di (exp_name, net_type, dataset_id)
    return [
        (chosen_exp, t, exp_map[chosen_exp][t][0])
        for t in chosen_types
    ]


# ==============================================================================
#  INFERENCE ESTERNA
# ==============================================================================

def _convert_to_nifti(src_dir, tmp_dir):
    """Converte PNG di src_dir in NIfTI dentro tmp_dir."""
    import nibabel as nib  # noqa: PLC0415

    files  = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
    affine = np.eye(4)
    for fname in files:
        case_id = os.path.splitext(fname)[0]
        img     = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
        img3d   = np.expand_dims(img.astype(np.float32), axis=2)
        nib.save(nib.Nifti1Image(img3d, affine),
                 os.path.join(tmp_dir, f"{case_id}_0000.nii.gz"))
    print(f"  {len(files)} immagini → NIfTI")
    return files


def _nifti_to_png(nifti_dir, png_dir):
    """Converte predizioni NIfTI (valori 0/1) in PNG (0/255)."""
    import nibabel as nib  # noqa: PLC0415

    os.makedirs(png_dir, exist_ok=True)
    nii_files = sorted(f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz"))
    for fname in nii_files:
        case_id = fname.replace(".nii.gz", "")
        data    = nib.load(os.path.join(nifti_dir, fname)).get_fdata()
        if data.ndim == 3:
            data = data[:, :, 0]
        cv2.imwrite(os.path.join(png_dir, f"{case_id}.png"),
                    (data * 255).astype(np.uint8))
    print(f"  {len(nii_files)} predizioni → PNG  ({png_dir})")


def _run_external_inference(target_exp_dir, source_exp_name, net_type, dataset_id):
    """
    Esegue nnUNetv2_predict usando il modello di source_exp_name sulle immagini
    di target_exp_dir. Le predizioni PNG vengono salvate in:
      target_exp_dir/3_predizioni_ext/{source_exp_name}_{net_type}/
    Ritorna il path di quella cartella, oppure None in caso di errore.
    """
    source_engine = os.path.join(EXPERIMENTS_DIR, source_exp_name, "2_nnunet_engine")
    trainer_name  = TRAINERS[net_type]
    test_img_dir  = os.path.join(target_exp_dir, "1_dataset", "test", "images")
    out_label     = f"{source_exp_name}_{net_type}"
    png_out_dir   = os.path.join(target_exp_dir, "3_predizioni_ext", out_label)

    print(f"\n  {'─'*55}")
    print(f"  Inference esterna: {source_exp_name} / {net_type}")
    print(f"  Dataset ID:        {dataset_id:03d}")
    print(f"  Trainer:           {trainer_name}")
    print(f"  {'─'*55}")

    env = os.environ.copy()
    env["nnUNet_raw"]          = os.path.join(source_engine, "nnUNet_raw")
    env["nnUNet_preprocessed"] = os.path.join(source_engine, "nnUNet_preprocessed")
    env["nnUNet_results"]      = os.path.join(source_engine, "nnUNet_results")
    env["nnUNet_n_proc_DA"]    = "0"

    with tempfile.TemporaryDirectory(prefix="nnunet_in_")  as tmp_in, \
         tempfile.TemporaryDirectory(prefix="nnunet_out_") as tmp_out:

        _convert_to_nifti(test_img_dir, tmp_in)

        cli_args = [
            "-i", tmp_in,
            "-o", tmp_out,
            "-d", str(dataset_id),
            "-c", "2d",
            "-f", "0",
            "-tr", trainer_name,
            "--disable_tta",
        ]
        print(f"\n  Avvio predict_entry_point...")
        result = _run_nnunet_entry(
            "nnunetv2.inference.predict_from_raw_data", "predict_entry_point",
            cli_args, env=env,
        )
        if result.returncode != 0:
            print(f"\n  ✗ Inference fallita (exit code {result.returncode})")
            return None

        _nifti_to_png(tmp_out, png_out_dir)

    return png_out_dir


# ==============================================================================
#  AGGREGAZIONE METRICHE
# ==============================================================================

def _aggregate(metrics_list):
    """
    Statistiche aggregate con NaN filtering.
    - NaN (compactness/eccentricity su pred vuote) esclusi dal calcolo.
    - p95 incluso per Hausdorff (robusto agli outlier, ma sensibile alla coda).
    - {key}_n conta i campioni validi (non-NaN).
    """
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


# ==============================================================================
#  CORE ANALISI  (riusabile per entrambe le modalità)
# ==============================================================================

def _run_analysis_core(folder_name, test_img_dir, test_lbl_dir,
                       nets_pred_map, output_dir, header_note=""):
    """
    Calcola metriche e genera visualizzazioni.

    Parameters
    ----------
    folder_name   : nome esperimento (usato solo nel titolo del TXT)
    test_img_dir  : cartella con le immagini originali (.png)
    test_lbl_dir  : cartella con le ground truth (.png)
    nets_pred_map : {net_label: png_pred_folder}
    output_dir    : dove salvare risultati
    header_note   : riga aggiuntiva nel TXT (es. "modello da: altro_esp")
    """
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    test_files = sorted(f for f in os.listdir(test_img_dir) if f.endswith(".png"))
    print(f"\n  Immagini test disponibili: {len(test_files)}")

    num_str = input("  Quante immagini analizzare? (Invio = tutte): ").strip()
    if num_str.isdigit():
        test_files = test_files[:int(num_str)]

    nets = list(nets_pred_map.keys())
    print(f"\n  Analisi di {len(test_files)} immagini per: {nets}")
    print(f"  {'─'*50}")

    metrics_per_net = {n: [] for n in nets}

    for idx, fname in enumerate(test_files):
        case_id  = os.path.splitext(fname)[0]
        img_orig = cv2.imread(os.path.join(test_img_dir, fname), cv2.IMREAD_GRAYSCALE)
        gt_raw   = cv2.imread(os.path.join(test_lbl_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img_orig is None or gt_raw is None:
            print(f"  ⚠  File mancante per {case_id}, skip")
            continue
        gt_f = gt_raw.astype(np.float32) / 255.0

        preds = {}
        for net, pred_dir in nets_pred_map.items():
            p_path = os.path.join(pred_dir, f"{case_id}.png")
            p_raw  = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
            if p_raw is not None:
                preds[net] = p_raw.astype(np.float32) / 255.0
                metrics_per_net[net].append(calculate_all_metrics(preds[net], gt_f))
            else:
                print(f"  ⚠  Predizione non trovata: {p_path}")

        # Visualizzazione
        if len(nets) == 2 and "baseline" in preds and "geometrica" in preds:
            create_comparison_visualization(
                img_orig, preds["baseline"], preds["geometrica"], gt_f, case_id, viz_dir
            )
        elif len(nets) == 1 and nets[0] in preds and metrics_per_net[nets[0]]:
            create_visualization(
                img_orig, preds[nets[0]], gt_f, case_id, viz_dir,
                metrics_per_net[nets[0]][-1],
                title_prefix=f"{nets[0].capitalize()} – ",
            )

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(test_files)} processate...")

    # Aggregazione
    aggregate = {n: _aggregate(ml) for n, ml in metrics_per_net.items() if ml}
    if not aggregate:
        print("  Nessuna metrica calcolata (predizioni mancanti?).")
        return

    # Grafico (solo se ci sono entrambe baseline e geometrica)
    if "baseline" in aggregate and "geometrica" in aggregate:
        create_metrics_comparison_chart(aggregate["baseline"], aggregate["geometrica"], output_dir)

    # JSON
    with open(os.path.join(output_dir, "metrics_comparison.json"), "w") as f:
        json.dump({**aggregate, "num_images": len(test_files)}, f, indent=2)

    # TXT leggibile
    txt_path = os.path.join(output_dir, "metrics_comparison.txt")
    with open(txt_path, "w") as f:
        f.write(f"CONFRONTO METRICHE – {folder_name}\n{'=' * 70}\n")
        f.write(f"Immagini analizzate: {len(test_files)}\n")
        if header_note:
            f.write(f"  {header_note}\n")
        f.write(f"  nota: compactness/eccentricity escluse quando pred è vuota\n")
        f.write(f"  nota: hausdorff_distance — mediana usata nel confronto (robusta agli outlier)\n")
        for net, agg in aggregate.items():
            f.write(f"\n{'─' * 40}\n{net.upper()}\n{'─' * 40}\n")
            for m in _METRIC_NAMES:
                mean = agg.get(f"{m}_mean", 0.0)
                std  = agg.get(f"{m}_std",  0.0)
                median = agg.get(f"{m}_median", 0.0)
                n    = agg.get(f"{m}_n",    len(test_files))
                if m in _SHAPE_METRICS:
                    f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}"
                            f"  (n={n}/{len(test_files)} pred non-vuote)\n")
                elif m == "hausdorff_distance":
                    f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}  [mediana: {median:.4f}]\n")
                else:
                    f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}\n")
        if "baseline" in aggregate and "geometrica" in aggregate:
            f.write(f"\n{'─' * 40}\nMIGLIORAMENTI GEOMETRIC vs BASELINE\n{'─' * 40}\n")
            f.write(f"  (hausdorff_distance: confronto su mediana; altri: su media)\n")
            for m in _METRIC_NAMES:
                key = "median" if m == "hausdorff_distance" else "mean"
                b   = aggregate["baseline"].get(f"{m}_{key}",   0.0)
                g   = aggregate["geometrica"].get(f"{m}_{key}", 0.0)
                if m in _LOWER_BETTER:
                    imp = (b - g) / b * 100 if b != 0 else 0.0
                else:
                    imp = (g - b) / b * 100 if b != 0 else 0.0
                f.write(f"  {m:25s}: {imp:+.2f}%\n")

    # vis_bad: peggiori predizioni per rete + chart metriche filtrato
    if "baseline" in nets_pred_map and "geometrica" in nets_pred_map:
        create_vis_bad(test_img_dir, test_lbl_dir, nets_pred_map, output_dir)

    print(f"\n  ✓ Risultati salvati in: {output_dir}")
    print(f"    visualizations/              ({len(test_files)} immagini)")
    print(f"    vis_bad/                     (10 peggiori per rete + chart filtrato)")
    print(f"    metrics_comparison.json")
    print(f"    metrics_comparison.txt")


# ==============================================================================
#  MODALITÀ 1  –  predizioni esistenti
# ==============================================================================

def run_analysis(folder_name):
    exp_dir      = os.path.join(EXPERIMENTS_DIR, folder_name)
    test_img_dir = os.path.join(exp_dir, "1_dataset", "test", "images")
    test_lbl_dir = os.path.join(exp_dir, "1_dataset", "test", "labels")
    output_dir   = os.path.join(exp_dir, "4_confronto_finale")

    if not os.path.isdir(test_img_dir):
        print(f"\n  Cartella test non trovata:\n  {test_img_dir}")
        return

    nets = _choose_nets(exp_dir)
    if nets is None:
        return

    pred_dir      = os.path.join(exp_dir, "3_predizioni")
    nets_pred_map = {n: os.path.join(pred_dir, n) for n in nets}

    _run_analysis_core(folder_name, test_img_dir, test_lbl_dir,
                       nets_pred_map, output_dir)


# ==============================================================================
#  MODALITÀ 2  –  modello esterno
# ==============================================================================

def run_analysis_external(target_folder_name):
    target_exp_dir = os.path.join(EXPERIMENTS_DIR, target_folder_name)
    test_img_dir   = os.path.join(target_exp_dir, "1_dataset", "test", "images")
    test_lbl_dir   = os.path.join(target_exp_dir, "1_dataset", "test", "labels")

    if not os.path.isdir(test_img_dir):
        print(f"\n  Cartella test non trovata:\n  {test_img_dir}")
        return

    selections = _choose_external_model()
    if selections is None:
        return

    # Esegui inference per ogni rete scelta e costruisci nets_pred_map
    nets_pred_map = {}
    source_notes  = []
    for source_exp_name, net_type, dataset_id in selections:
        png_dir = _run_external_inference(
            target_exp_dir, source_exp_name, net_type, dataset_id
        )
        if png_dir is None:
            print(f"  ✗ Inference fallita per {source_exp_name}/{net_type}, skip.")
            continue
        # Usa "baseline"/"geometrica" come label se è il tipo nativo,
        # così create_comparison_visualization funziona correttamente.
        nets_pred_map[net_type] = png_dir
        source_notes.append(
            f"{net_type}: {source_exp_name} (Dataset {dataset_id:03d})"
        )

    if not nets_pred_map:
        print("  Nessuna predizione disponibile, analisi annullata.")
        return

    # Nome cartella output: 4_analisi_ext_{source_exp}_{net} oppure
    # 4_analisi_ext_{source_exp} se entrambe le reti vengono dallo stesso esperimento
    source_exps = list({s for s, _, _ in selections})
    if len(source_exps) == 1:
        out_label = source_exps[0]
    else:
        out_label = "_".join(source_exps)
    output_dir  = os.path.join(target_exp_dir, f"4_analisi_ext_{out_label}")
    header_note = "modello esterno – " + "  |  ".join(source_notes)

    _run_analysis_core(
        target_folder_name,
        test_img_dir, test_lbl_dir,
        nets_pred_map,
        output_dir,
        header_note=header_note,
    )


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("  TEST STANDALONE – Analisi esperimenti geometrica")
    print("=" * 60)

    folder = _choose_experiment()
    if folder is None:
        return 1

    mode = _choose_mode()

    if mode == "1":
        run_analysis(folder)
    else:
        run_analysis_external(folder)

    print("\n  Analisi completata!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
