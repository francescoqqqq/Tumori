"""
metrics_utils.py – Funzioni metriche e visualizzazioni per segmentazione circolare.

Importato da run_experiment.py (pipeline automatica) e test.py (uso standalone).
NON contiene logica interattiva né I/O di file: solo funzioni pure.

Metriche calcolate:
  - dice               : overlap predizione / GT  (↑ meglio)
  - iou                : intersection over union  (↑ meglio)
  - compactness        : (4π·Area)/P²  sulla predizione  (↑ meglio, 1.0 = cerchio perfetto)
  - eccentricity       : √(1-(b/a)²)  via fitting ellisse  (↓ meglio, 0.0 = cerchio perfetto)
  - hausdorff_distance : max distanza tra contorni pred/GT  (↓ meglio)
  - boundary_iou       : IoU calcolato solo sui bordi  (↑ meglio)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch        # pyright: ignore[reportMissingImports]
from scipy.spatial.distance import directed_hausdorff  # pyright: ignore[reportMissingImports]


# ==============================================================================
#  METRICHE
# ==============================================================================

def calculate_dice(pred, gt):
    """Dice score: 2·|P∩G| / (|P|+|G|). [0,1], 1=perfetto."""
    pred_b = (pred > 0.5).astype(np.float32)
    gt_b   = (gt   > 0.5).astype(np.float32)
    intersection = np.sum(pred_b * gt_b)
    union        = np.sum(pred_b) + np.sum(gt_b)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / union)


def calculate_iou(pred, gt):
    """IoU: |P∩G| / |P∪G|. [0,1], 1=perfetto."""
    pred_b = (pred > 0.5).astype(np.float32)
    gt_b   = (gt   > 0.5).astype(np.float32)
    intersection = np.sum(pred_b * gt_b)
    union        = np.sum(pred_b + gt_b > 0)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def calculate_compactness(mask):
    """
    Compactness: (4π·Area) / Perimetro²  calcolata sulla predizione.
    [0,1], 1.0 = cerchio perfetto.
    Ritorna nan se la predizione è vuota (nessun contorno trovato), così
    l'aggregazione può escludere i casi di fallimento totale invece di
    trattarli come "forma perfetta con area zero".
    """
    mask_b = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return float("nan")

    # Contorno principale: maggiore area, stesso criterio di eccentricity e Hausdorff
    main_c = max(contours, key=cv2.contourArea)
    area   = cv2.contourArea(main_c)
    perim  = cv2.arcLength(main_c, True)
    if perim > 0 and area > 10:
        return float(min((4 * np.pi * area) / (perim ** 2), 1.0))
    return float("nan")


def calculate_eccentricity(mask):
    """
    Eccentricity via fitting ellisse sul contorno principale: √(1 - (b/a)²).
    [0,1], 0.0 = cerchio perfetto, 1.0 = segmento.

    Usa solo il contorno con area maggiore (come Hausdorff) per evitare che
    piccoli artefatti di bordo (<50px) — che la rete geometrica può produrre
    più facilmente per via della loss composita — alzino artificialmente la
    media. Ritorna nan se la predizione è vuota.
    """
    mask_b = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return float("nan")

    # Contorno principale: maggiore area, ignora artefatti piccoli
    main_c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_c) < 10 or len(main_c) < 5:
        return float("nan")

    try:
        ellipse = cv2.fitEllipse(main_c)
        major = max(ellipse[1])
        minor = min(ellipse[1])
        if major > 0:
            return float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))
    except Exception:
        pass
    return float("nan")


def calculate_hausdorff_distance(pred, gt):
    """
    Hausdorff distance bidirezionale tra contorni pred e GT.

    Usa cv2.findContours con CHAIN_APPROX_NONE (tutti i punti del contorno)
    e seleziona il contorno principale per AREA (cv2.contourArea), non per
    numero di punti: un blob irregolare lungo può avere più punti di un
    cerchio compatto, portando alla selezione del contorno sbagliato.

    Se pred o GT sono vuoti ritorna la diagonale dell'immagine (caso peggiore).
    La media di questa metrica è distorta da outlier estremi: preferire la
    mediana per confronti aggregati.
    """
    pred_b = (pred > 0.5).astype(np.uint8)
    gt_b   = (gt   > 0.5).astype(np.uint8)

    pred_contours, _ = cv2.findContours(pred_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_contours,   _ = cv2.findContours(gt_b,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not pred_contours or not gt_contours:
        h, w = pred_b.shape
        return float(np.sqrt(h ** 2 + w ** 2))

    # Selezione per area: il contorno più grande è la predizione principale
    pred_pts = max(pred_contours, key=cv2.contourArea)[:, 0, :]  # (N, 2) x,y
    gt_pts   = max(gt_contours,   key=cv2.contourArea)[:, 0, :]

    hd1, _, _ = directed_hausdorff(pred_pts, gt_pts)
    hd2, _, _ = directed_hausdorff(gt_pts,   pred_pts)
    return float(max(hd1, hd2))


def calculate_boundary_iou(pred, gt, thickness=3):
    """
    IoU calcolato solo sui bordi delle maschere.
    Il bordo è estratto via sottrazione morfologica (mask - erosione(mask)),
    più stabile di cv2.Canny su maschere binarie pulite.
    """
    pred_b = (pred > 0.5).astype(np.uint8)
    gt_b   = (gt   > 0.5).astype(np.uint8)

    kernel       = np.ones((thickness, thickness), np.uint8)
    pred_boundary = pred_b - cv2.erode(pred_b, kernel, iterations=1)
    gt_boundary   = gt_b   - cv2.erode(gt_b,   kernel, iterations=1)

    intersection = np.sum((pred_boundary > 0) & (gt_boundary > 0))
    union        = np.sum((pred_boundary > 0) | (gt_boundary > 0))

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def calculate_all_metrics(pred, gt):
    """Calcola e restituisce tutte le metriche in un dict."""
    return {
        "dice":               calculate_dice(pred, gt),
        "iou":                calculate_iou(pred, gt),
        "compactness":        calculate_compactness(pred),
        "eccentricity":       calculate_eccentricity(pred),
        "hausdorff_distance": calculate_hausdorff_distance(pred, gt),
        "boundary_iou":       calculate_boundary_iou(pred, gt),
    }


# ==============================================================================
#  VISUALIZZAZIONI
# ==============================================================================

def create_visualization(img_original, pred, gt, case_id, output_dir, metrics,
                         title_prefix=""):
    """
    Crea immagine di analisi con 4 pannelli:
    Originale | Ground Truth | Predizione | Overlap (TP/FN/FP)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    pred_bin = pred > 0.5
    gt_bin   = gt   > 0.5

    # 1. Originale
    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Immagine Originale", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    gt_display = np.zeros_like(img_original)
    gt_display[gt_bin] = 255
    axes[1].imshow(gt_display, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # 3. Predizione
    pred_display = np.zeros_like(img_original)
    pred_display[pred_bin] = 255
    axes[2].imshow(pred_display, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"{title_prefix}Predizione", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # 4. Overlap TP/FN/FP
    overlay = np.zeros((*img_original.shape, 4), dtype=np.float32)
    overlay[gt_bin & pred_bin]   = [0, 1, 0, 0.6]   # Verde  – TP
    overlay[gt_bin & ~pred_bin]  = [1, 0, 0, 0.6]   # Rosso  – FN
    overlay[~gt_bin & pred_bin]  = [1, 1, 0, 0.6]   # Giallo – FP

    axes[3].imshow(img_original, cmap="gray", alpha=0.3, vmin=0, vmax=255)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlap Analysis", fontsize=14, fontweight="bold")
    axes[3].axis("off")
    axes[3].legend(handles=[
        Patch(facecolor="green",  alpha=0.6, label="Corretto (TP)"),
        Patch(facecolor="red",    alpha=0.6, label="Mancato (FN)"),
        Patch(facecolor="yellow", alpha=0.6, label="Errato (FP)"),
    ], loc="upper right", fontsize=10)

    metrics_text = (
        f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | "
        f"Compactness: {metrics['compactness']:.4f} | "
        f"Eccentricity: {metrics['eccentricity']:.4f}"
    )
    fig.suptitle(f"{title_prefix}{case_id} – {metrics_text}",
                 fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, f"{case_id}_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def create_comparison_visualization(img_original, pred_baseline, pred_geometric,
                                    gt, case_id, output_dir):
    """
    Crea immagine di confronto 2 righe × 4 colonne:
    Riga 1 (Baseline):  Originale | GT | Pred | Overlap
    Riga 2 (Geometric): Originale | GT | Pred | Overlap
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    pred_b_bin = pred_baseline  > 0.5
    pred_g_bin = pred_geometric > 0.5
    gt_bin     = gt             > 0.5

    gt_display = np.zeros_like(img_original)
    gt_display[gt_bin] = 255

    def _make_overlay(pred_bin):
        ov = np.zeros((*img_original.shape, 4), dtype=np.float32)
        ov[gt_bin &  pred_bin] = [0, 1, 0, 0.7]
        ov[gt_bin & ~pred_bin] = [1, 0, 0, 0.7]
        ov[~gt_bin & pred_bin] = [1, 1, 0, 0.7]
        return ov

    legend_patches = [
        Patch(facecolor="green",  alpha=0.7, label="TP"),
        Patch(facecolor="red",    alpha=0.7, label="FN"),
        Patch(facecolor="yellow", alpha=0.7, label="FP"),
    ]

    for row, (pred_bin, label) in enumerate([(pred_b_bin, "Baseline"), (pred_g_bin, "Geometric")]):
        pred_display = np.zeros_like(img_original)
        pred_display[pred_bin] = 255

        axes[row, 0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("Originale", fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_display, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_display, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title(f"{label} Prediction", fontsize=12, fontweight="bold")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(img_original, cmap="gray", alpha=0.2, vmin=0, vmax=255)
        axes[row, 3].imshow(_make_overlay(pred_bin))
        axes[row, 3].set_title(f"{label} Overlap", fontsize=12, fontweight="bold")
        axes[row, 3].axis("off")
        axes[row, 3].legend(handles=legend_patches, loc="upper right", fontsize=9)

    dice_b  = calculate_dice(pred_baseline,  gt)
    dice_g  = calculate_dice(pred_geometric, gt)
    comp_b  = calculate_compactness(pred_baseline)
    comp_g  = calculate_compactness(pred_geometric)

    fig.suptitle(f"{case_id} – Confronto Baseline vs Geometric",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.02, 0.75, "BASELINE",  va="center", rotation="vertical",
             fontsize=14, fontweight="bold", color="darkblue")
    fig.text(0.02, 0.25, "GEOMETRIC", va="center", rotation="vertical",
             fontsize=14, fontweight="bold", color="darkgreen")
    fig.text(0.5, 0.02,
             f"Baseline: Dice={dice_b:.3f}, Compactness={comp_b:.3f}  |  "
             f"Geometric: Dice={dice_g:.3f}, Compactness={comp_g:.3f}",
             ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    output_path = os.path.join(output_dir, f"{case_id}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def create_metrics_comparison_chart(baseline_metrics, geometric_metrics, output_dir,
                                    title_note=None):
    """
    Grafico a barre: confronto metriche aggregate baseline vs geometric.
    Verde = rete migliore per quella metrica, Rosso = rete peggiore.

    Parameters
    ----------
    title_note : str opzionale aggiunto come riga extra al titolo del grafico
                 (es. "escluse top-10 peggiori per rete")
    """
    metrics_config = [
        ("Dice Score",     "dice_mean",               "higher"),
        ("IoU",            "iou_mean",                "higher"),
        ("Compactness",    "compactness_mean",         "higher"),
        ("Eccentricity",   "eccentricity_mean",        "lower"),
        ("Boundary IoU",   "boundary_iou_mean",        "higher"),
        ("Hausdorff Dist.","hausdorff_distance_p95",    "lower"),
    ]

    n = len(metrics_config)
    fig, axes = plt.subplots(1, n, figsize=(18, 5))

    for idx, (name, key, direction) in enumerate(metrics_config):
        ax = axes[idx]
        bval = baseline_metrics.get(key, 0)
        gval = geometric_metrics.get(key, 0)

        baseline_better = (bval > gval) if direction == "higher" else (bval < gval)
        c_b = "green" if baseline_better else "red"
        c_g = "red"   if baseline_better else "green"

        bars = ax.bar(["Baseline", "Geometric"], [bval, gval],
                      color=[c_b, c_g], alpha=0.7, edgecolor="black", linewidth=1.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

        if bval != 0:
            imp = ((bval - gval) / bval * 100) if direction == "lower" else ((gval - bval) / bval * 100)
        else:
            imp = 0.0
        sign = "+" if imp > 0 else ""
        ax.set_title(f"{name}\n({sign}{imp:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Valore", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    base_title = "Confronto Metriche Aggregate – Baseline vs Geometric\n(Verde = Migliore, Rosso = Peggiore)"
    full_title  = f"{base_title}\n{title_note}" if title_note else base_title
    plt.suptitle(full_title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = os.path.join(output_dir, "metrics_comparison_chart.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafico confronto metriche: {output_path}")
    return output_path


# ==============================================================================
#  VIS_BAD  –  peggiori predizioni + chart filtrato
# ==============================================================================

def _aggregate_for_vis_bad(metrics_list):
    """
    Aggregazione statistica da lista di dict metriche per-immagine.
    Identica a _aggregate_metrics in run_experiment.py e _aggregate in test.py
    (duplicata qui per rendere metrics_utils.py autosufficiente).
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


def create_vis_bad(test_img_dir, test_lbl_dir, nets_pred_map, output_dir, n_worst=10):
    """
    Crea vis_bad/ con le N predizioni peggiori (per Dice) e un chart filtrato.

    Struttura output in output_dir/vis_bad/:
      worstB_<case>_comparison.png   – peggiori per la sola baseline
      worstG_<case>_comparison.png   – peggiori per la sola geometrica
      worstBG_<case>_comparison.png  – peggiori per entrambe
      metrics_comparison_chart.png   – metriche aggregate senza i casi peggiori

    Parameters
    ----------
    test_img_dir  : cartella immagini originali (.png)
    test_lbl_dir  : cartella ground truth (.png)
    nets_pred_map : {"baseline": pred_dir, "geometrica": pred_dir}
                    (funziona anche con una sola rete)
    output_dir    : cartella padre (es. 4_confronto_finale/)
    n_worst       : numero di predizioni peggiori per rete (default 10)
    """
    vis_bad_dir = os.path.join(output_dir, "vis_bad")
    os.makedirs(vis_bad_dir, exist_ok=True)

    nets       = list(nets_pred_map.keys())
    test_files = sorted(f for f in os.listdir(test_img_dir) if f.endswith(".png"))

    # ── Step 1: calcola Dice + tutte le metriche per ogni immagine/rete ──────
    dice_per_net = {net: {} for net in nets}
    all_metrics  = {net: {} for net in nets}

    for fname in test_files:
        case_id = os.path.splitext(fname)[0]
        gt_raw  = cv2.imread(os.path.join(test_lbl_dir, fname), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            continue
        gt_f = gt_raw.astype(np.float32) / 255.0

        for net, pred_dir in nets_pred_map.items():
            p_raw = cv2.imread(
                os.path.join(pred_dir, f"{case_id}.png"), cv2.IMREAD_GRAYSCALE
            )
            if p_raw is not None:
                pred_f = p_raw.astype(np.float32) / 255.0
                dice_per_net[net][case_id] = calculate_dice(pred_f, gt_f)
                all_metrics[net][case_id]  = calculate_all_metrics(pred_f, gt_f)

    # ── Step 2: identifica i N peggiori per ogni rete ─────────────────────────
    worst_per_net = {}
    for net in nets:
        sorted_cases = sorted(dice_per_net[net].items(), key=lambda x: x[1])
        worst_per_net[net] = [cid for cid, _ in sorted_cases[:n_worst]]

    all_worst_set = set()
    for net in nets:
        all_worst_set.update(worst_per_net[net])

    # ── Step 3: visualizzazioni per i casi peggiori ───────────────────────────
    # Prefisso del filename: B = worst baseline, G = worst geometrica, BG = entrambe
    for case_id in sorted(all_worst_set):
        fname    = f"{case_id}.png"
        img_orig = cv2.imread(os.path.join(test_img_dir, fname), cv2.IMREAD_GRAYSCALE)
        gt_raw   = cv2.imread(os.path.join(test_lbl_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img_orig is None or gt_raw is None:
            continue
        gt_f = gt_raw.astype(np.float32) / 255.0

        preds = {}
        for net, pred_dir in nets_pred_map.items():
            p_raw = cv2.imread(os.path.join(pred_dir, fname), cv2.IMREAD_GRAYSCALE)
            if p_raw is not None:
                preds[net] = p_raw.astype(np.float32) / 255.0

        # Tag: quali reti lo hanno tra i peggiori
        tag = "".join(
            letter for letter, net in [("B", "baseline"), ("G", "geometrica")]
            if case_id in worst_per_net.get(net, [])
        )
        viz_case_id = f"worst{tag}_{case_id}"

        if len(nets) == 2 and "baseline" in preds and "geometrica" in preds:
            create_comparison_visualization(
                img_orig, preds["baseline"], preds["geometrica"],
                gt_f, viz_case_id, vis_bad_dir,
            )
        elif len(nets) == 1 and nets[0] in preds:
            net_type = nets[0]
            m = all_metrics[net_type].get(case_id, calculate_all_metrics(preds[net_type], gt_f))
            create_visualization(
                img_orig, preds[net_type], gt_f, viz_case_id, vis_bad_dir, m,
                title_prefix=f"{net_type.capitalize()} WORST – ",
            )

    # ── Step 4: metriche filtrate, chart e TXT ────────────────────────────────
    _METRIC_NAMES  = ["dice", "iou", "compactness", "eccentricity",
                      "hausdorff_distance", "boundary_iou"]
    _SHAPE_METRICS = {"compactness", "eccentricity"}
    _LOWER_BETTER  = {"hausdorff_distance", "eccentricity"}

    if "baseline" in nets_pred_map and "geometrica" in nets_pred_map:
        baseline_filt = [
            all_metrics["baseline"][c]
            for c in all_metrics["baseline"]
            if c not in all_worst_set
        ]
        geom_filt = [
            all_metrics["geometrica"][c]
            for c in all_metrics["geometrica"]
            if c not in all_worst_set
        ]
        if baseline_filt and geom_filt:
            agg_b  = _aggregate_for_vis_bad(baseline_filt)
            agg_g  = _aggregate_for_vis_bad(geom_filt)
            n_excl = len(all_worst_set)
            n_kept = len(baseline_filt)

            # Chart
            note = (f"Filtrato: escluse top-{n_worst} peggiori per rete "
                    f"({n_excl} immagini escluse, {n_kept} usate)")
            create_metrics_comparison_chart(agg_b, agg_g, vis_bad_dir, title_note=note)

            # TXT
            exp_name = os.path.basename(os.path.dirname(output_dir))
            txt_path = os.path.join(vis_bad_dir, "metrics_comparison.txt")
            with open(txt_path, "w") as f:
                f.write(f"CONFRONTO METRICHE FILTRATE – {exp_name}\n")
                f.write("=" * 70 + "\n")
                f.write(f"Immagini usate:   {n_kept}  (escluse {n_excl} peggiori per rete)\n")
                f.write(f"Immagini escluse: {sorted(all_worst_set)}\n")
                f.write(f"  nota: compactness/eccentricity escluse quando pred è vuota\n")
                f.write(f"  nota: hausdorff_distance — mediana usata nel confronto\n")

                for net_label, agg in [("baseline", agg_b), ("geometrica", agg_g)]:
                    n_tot = n_kept
                    f.write(f"\n{'─'*40}\n{net_label.upper()}\n{'─'*40}\n")
                    for m in _METRIC_NAMES:
                        mean = agg.get(f"{m}_mean", 0.0)
                        std  = agg.get(f"{m}_std",  0.0)
                        p95  = agg.get(f"{m}_p95",  0.0)
                        n    = agg.get(f"{m}_n",    n_tot)
                        if m in _SHAPE_METRICS:
                            f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}"
                                    f"  (n={n}/{n_tot} pred non-vuote)\n")
                        elif m == "hausdorff_distance":
                            f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}"
                                    f"  [p95: {p95:.4f}]\n")
                        else:
                            f.write(f"  {m:25s}: {mean:.4f} ± {std:.4f}\n")

                f.write(f"\n{'─'*40}\nMIGLIORAMENTI GEOMETRIC vs BASELINE\n{'─'*40}\n")
                f.write(f"  (hausdorff_distance: confronto su p95; altri: su media)\n")
                for m in _METRIC_NAMES:
                    key = "p95" if m == "hausdorff_distance" else "mean"
                    b   = agg_b.get(f"{m}_{key}", 0.0)
                    g   = agg_g.get(f"{m}_{key}", 0.0)
                    if m in _LOWER_BETTER:
                        imp = (b - g) / b * 100 if b != 0 else 0.0
                    else:
                        imp = (g - b) / b * 100 if b != 0 else 0.0
                    f.write(f"  {m:25s}: {imp:+.2f}%\n")

            print(f"  ✓ metrics_comparison.txt filtrato: {txt_path}")

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    n_total_worst = len(all_worst_set)
    print(f"  ✓ vis_bad/ → {n_total_worst} casi totali")
    for net in nets:
        worst_list = worst_per_net.get(net, [])
        dices = [f"{dice_per_net[net].get(c, 0):.3f}" for c in worst_list[:4]]
        print(f"    {net:12s}: {worst_list[:4]}  dice={dices}")

    return vis_bad_dir
