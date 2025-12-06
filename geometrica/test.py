"""
Script per testare modelli nnU-Net e generare visualizzazioni/metriche.

PREREQUISITO: Esegui prima run_inference.py per generare le predizioni!

Modalità:
1. Baseline       - Analizza solo modello baseline
2. Geometric      - Analizza solo modello geometric
3. Confronto      - Confronta entrambi i modelli

Author: Francesco + Claude
Date: 2025-12-05
"""
import os
import sys
import json
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage import measure

# Constants
DATASET_ID = 501
OUTPUT_DIRS = {
    'baseline': 'baseline_results',
    'geometric': 'geometric_results',
    'confronto': 'confronto_results'
}


def load_nifti(path):
    """Carica file NIfTI e ritorna i dati."""
    nii = nib.load(path)
    return nii.get_fdata()


def calculate_dice(pred, gt):
    """Calcola Dice score tra predizione e ground truth."""
    pred_binary = (pred > 0.5).astype(np.float32)
    gt_binary = (gt > 0.5).astype(np.float32)

    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = 2.0 * intersection / union
    return dice


def calculate_iou(pred, gt):
    """Calcola IoU (Intersection over Union)."""
    pred_binary = (pred > 0.5).astype(np.float32)
    gt_binary = (gt > 0.5).astype(np.float32)

    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary + gt_binary > 0)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou


def calculate_compactness(mask):
    """Calcola compactness: (4π·Area) / (Perimeter²)."""
    mask_binary = (mask > 0.5).astype(np.uint8)

    # Trova tutti i contorni
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    # Calcola compactness media di tutti i contorni
    compactness_values = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0 and area > 10:  # Filtro per rumore
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            compactness_values.append(min(compactness, 1.0))  # Cap a 1.0

    return np.mean(compactness_values) if compactness_values else 0.0


def calculate_solidity(mask):
    """Calcola solidity: Area / ConvexHull_Area."""
    mask_binary = (mask > 0.5).astype(np.uint8)

    # Trova tutti i contorni
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    # Calcola solidity media
    solidity_values = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Filtro per rumore
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                solidity_values.append(min(solidity, 1.0))

    return np.mean(solidity_values) if solidity_values else 0.0


def calculate_eccentricity(mask):
    """Calcola eccentricity tramite fitting ellisse."""
    mask_binary = (mask > 0.5).astype(np.uint8)

    # Trova tutti i contorni
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    eccentricity_values = []
    for contour in contours:
        if len(contour) >= 5:  # Serve almeno 5 punti per fitting ellisse
            try:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])

                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    eccentricity_values.append(eccentricity)
            except:
                pass

    return np.mean(eccentricity_values) if eccentricity_values else 0.0


def calculate_hausdorff_distance(pred, gt):
    """Calcola Hausdorff distance tra contorni."""
    pred_binary = (pred > 0.5).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)

    # Trova contorni
    pred_contours = measure.find_contours(pred_binary, 0.5)
    gt_contours = measure.find_contours(gt_binary, 0.5)

    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return 0.0

    # Usa il contorno più grande
    pred_contour = max(pred_contours, key=len)
    gt_contour = max(gt_contours, key=len)

    # Calcola Hausdorff distance bidirezionale
    hd1, _, _ = directed_hausdorff(pred_contour, gt_contour)
    hd2, _, _ = directed_hausdorff(gt_contour, pred_contour)

    return max(hd1, hd2)


def calculate_boundary_iou(pred, gt, thickness=3):
    """Calcola IoU solo sui bordi."""
    pred_binary = (pred > 0.5).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)

    # Estrai bordi
    pred_boundary = cv2.Canny(pred_binary * 255, 50, 150)
    gt_boundary = cv2.Canny(gt_binary * 255, 50, 150)

    # Dilata leggermente per tolleranza
    kernel = np.ones((thickness, thickness), np.uint8)
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=1)
    gt_boundary = cv2.dilate(gt_boundary, kernel, iterations=1)

    # Calcola IoU sui bordi
    intersection = np.sum((pred_boundary > 0) & (gt_boundary > 0))
    union = np.sum((pred_boundary > 0) | (gt_boundary > 0))

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def calculate_all_metrics(pred, gt):
    """Calcola tutte le metriche per una predizione."""
    metrics = {
        'dice': calculate_dice(pred, gt),
        'iou': calculate_iou(pred, gt),
        'compactness': calculate_compactness(pred),
        'solidity': calculate_solidity(pred),
        'eccentricity': calculate_eccentricity(pred),
        'hausdorff_distance': calculate_hausdorff_distance(pred, gt),
        'boundary_iou': calculate_boundary_iou(pred, gt),
    }
    return metrics


def create_visualization(img_original, pred, gt, case_id, output_dir, metrics, title_prefix=""):
    """
    Crea visualizzazione con 4 pannelli:
    1. Immagine originale
    2. Ground Truth (solo cerchi in bianco su nero)
    3. Predizione (solo predizione in bianco su nero)
    4. Overlap con codice colore
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Converti mask in binarie
    pred_binary = pred > 0.5
    gt_binary = gt > 0.5

    # 1. Immagine originale
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Immagine Originale', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Ground Truth (solo cerchi in bianco su nero)
    gt_display = np.zeros_like(img_original)
    gt_display[gt_binary] = 255  # GT in bianco
    axes[1].imshow(gt_display, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Predizione (solo predizione in bianco su nero)
    pred_display = np.zeros_like(img_original)
    pred_display[pred_binary] = 255  # Predizione in bianco
    axes[2].imshow(pred_display, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'{title_prefix}Predizione', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 4. Overlap con codice colore chiaro
    # Crea immagine RGB per overlay
    overlay = np.zeros((img_original.shape[0], img_original.shape[1], 4), dtype=np.float32)

    # Definisci i colori
    intersection = gt_binary & pred_binary  # Veri positivi - VERDE
    gt_only = gt_binary & ~pred_binary  # Falsi negativi - ROSSO (GT mancato)
    pred_only = pred_binary & ~gt_binary  # Falsi positivi - GIALLO (predizione errata)

    # Assegna colori con trasparenza
    overlay[intersection] = [0, 1, 0, 0.6]  # Verde
    overlay[gt_only] = [1, 0, 0, 0.6]  # Rosso
    overlay[pred_only] = [1, 1, 0, 0.6]  # Giallo

    # Mostra immagine originale in grigio come background
    axes[3].imshow(img_original, cmap='gray', alpha=0.3, vmin=0, vmax=255)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlap Analysis', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    # Aggiungi legenda per overlap
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Corretto (TP)'),
        Patch(facecolor='red', alpha=0.6, label='Mancato (FN)'),
        Patch(facecolor='yellow', alpha=0.6, label='Errato (FP)')
    ]
    axes[3].legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Aggiungi metriche come testo sopra la figura
    metrics_text = f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | Compactness: {metrics['compactness']:.4f} | Solidity: {metrics['solidity']:.4f} | Eccentricity: {metrics['eccentricity']:.4f}"
    fig.suptitle(f'{title_prefix}{case_id} - {metrics_text}', fontsize=12, fontweight='bold', y=0.98)

    # Salva
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, f'{case_id}_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_comparison_visualization(img_original, pred_baseline, pred_geometric, gt, case_id, output_dir):
    """
    Crea visualizzazione confronto con 5 pannelli:
    1. Originale
    2. GT
    3. Baseline
    4. Geometric
    5. Confronto overlap
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Binarizza
    pred_baseline_bin = pred_baseline > 0.5
    pred_geometric_bin = pred_geometric > 0.5
    gt_binary = gt > 0.5

    # 1. Originale
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Originale', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. GT
    gt_display = np.zeros_like(img_original)
    gt_display[gt_binary] = 255
    axes[1].imshow(gt_display, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. Baseline
    baseline_display = np.zeros_like(img_original)
    baseline_display[pred_baseline_bin] = 255
    axes[2].imshow(baseline_display, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Baseline', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 4. Geometric
    geometric_display = np.zeros_like(img_original)
    geometric_display[pred_geometric_bin] = 255
    axes[3].imshow(geometric_display, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('Geometric', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    # 5. Confronto - mostra differenze
    # Verde: entrambi corretti
    # Rosso: solo baseline corretto
    # Blu: solo geometric corretto
    comparison = np.zeros((img_original.shape[0], img_original.shape[1], 4), dtype=np.float32)

    baseline_correct = pred_baseline_bin & gt_binary
    geometric_correct = pred_geometric_bin & gt_binary

    both_correct = baseline_correct & geometric_correct
    only_baseline = baseline_correct & ~geometric_correct
    only_geometric = ~baseline_correct & geometric_correct

    comparison[both_correct] = [0, 1, 0, 0.7]  # Verde - entrambi corretti
    comparison[only_baseline] = [1, 0, 0, 0.7]  # Rosso - solo baseline
    comparison[only_geometric] = [0, 0, 1, 0.7]  # Blu - solo geometric

    axes[4].imshow(img_original, cmap='gray', alpha=0.2, vmin=0, vmax=255)
    axes[4].imshow(comparison)
    axes[4].set_title('Confronto Prestazioni', fontsize=12, fontweight='bold')
    axes[4].axis('off')

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Entrambi corretti'),
        Patch(facecolor='red', alpha=0.7, label='Solo Baseline'),
        Patch(facecolor='blue', alpha=0.7, label='Solo Geometric')
    ]
    axes[4].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Calcola metriche per confronto
    dice_b = calculate_dice(pred_baseline, gt)
    dice_g = calculate_dice(pred_geometric, gt)
    comp_b = calculate_compactness(pred_baseline)
    comp_g = calculate_compactness(pred_geometric)

    metrics_text = f"Dice: B={dice_b:.3f} G={dice_g:.3f} | Compactness: B={comp_b:.3f} G={comp_g:.3f}"
    fig.suptitle(f'{case_id} - {metrics_text}', fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, f'{case_id}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_metrics_comparison_chart(baseline_metrics, geometric_metrics, output_dir):
    """
    Crea grafico a barre confronto metriche aggregate.
    Verde = migliore, Rosso = peggiore per ogni metrica.
    """
    # Metriche da confrontare (nome visualizzato, chiave JSON, direzione)
    metrics_config = [
        ('Dice Score', 'dice_mean', 'higher'),
        ('IoU', 'iou_mean', 'higher'),
        ('Compactness', 'compactness_mean', 'higher'),
        ('Solidity', 'solidity_mean', 'higher'),
        ('Eccentricity', 'eccentricity_mean', 'lower'),
        ('Boundary IoU', 'boundary_iou_mean', 'higher'),
        ('Hausdorff Dist.', 'hausdorff_distance_mean', 'lower'),
    ]

    n_metrics = len(metrics_config)
    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 5))

    for idx, (metric_name, metric_key, direction) in enumerate(metrics_config):
        ax = axes[idx]

        baseline_val = baseline_metrics.get(metric_key, 0)
        geometric_val = geometric_metrics.get(metric_key, 0)

        # Determina chi è migliore
        if direction == 'higher':
            baseline_better = baseline_val > geometric_val
        else:  # lower è migliore
            baseline_better = baseline_val < geometric_val

        # Colori
        color_baseline = 'green' if baseline_better else 'red'
        color_geometric = 'red' if baseline_better else 'green'

        # Barre
        bars = ax.bar(['Baseline', 'Geometric'], [baseline_val, geometric_val],
                      color=[color_baseline, color_geometric], alpha=0.7, edgecolor='black', linewidth=1.5)

        # Valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Calcola miglioramento percentuale
        if baseline_val != 0:
            if direction == 'higher':
                improvement = ((geometric_val - baseline_val) / baseline_val) * 100
            else:
                improvement = ((baseline_val - geometric_val) / baseline_val) * 100
        else:
            improvement = 0

        # Titolo con delta
        delta_sign = '+' if improvement > 0 else ''
        ax.set_title(f'{metric_name}\n({delta_sign}{improvement:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Valore', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Ruota etichette x
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    plt.suptitle('Confronto Metriche Aggregate - Baseline vs Geometric\n(Verde = Migliore, Rosso = Peggiore)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = os.path.join(output_dir, 'metrics_comparison_chart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Grafico confronto metriche salvato: {output_path}")
    return output_path


def process_single_model(model_name, output_subdir, num_images=None):
    """Processa risultati per un singolo modello."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_subdir)
    predictions_dir = os.path.join(output_dir, 'predictions')
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    # Verifica che esistano predizioni
    if not os.path.exists(predictions_dir):
        print(f"❌ Errore: Directory predizioni non trovata: {predictions_dir}")
        print(f"   Esegui prima: python run_inference.py --{model_name}")
        return False

    # Setup environment
    os.environ['nnUNet_raw'] = os.path.join(base_dir, 'nnUNet_raw')
    raw_dir = os.environ['nnUNet_raw']

    # Lista predizioni
    pred_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith('.nii.gz')])

    if len(pred_files) == 0:
        print(f"❌ Errore: Nessuna predizione trovata in {predictions_dir}")
        print(f"   Esegui prima: python run_inference.py --{model_name}")
        return False

    if num_images:
        pred_files = pred_files[:num_images]

    print(f"\n📊 Processando {len(pred_files)} immagini per {model_name}...")

    all_metrics = []

    for idx, pred_file in enumerate(pred_files):
        case_id = pred_file.replace('.nii.gz', '')

        # Carica predizione
        pred_path = os.path.join(predictions_dir, pred_file)
        pred = load_nifti(pred_path)

        # Carica GT
        gt_path = os.path.join(raw_dir, f'Dataset{DATASET_ID:03d}_Shapes', 'labelsTr', f'{case_id}.nii.gz')
        if not os.path.exists(gt_path):
            print(f"⚠️  GT non trovato per {case_id}, skip")
            continue

        gt = load_nifti(gt_path)

        # Carica immagine originale
        img_path = os.path.join(raw_dir, f'Dataset{DATASET_ID:03d}_Shapes', 'imagesTr', f'{case_id}_0000.nii.gz')
        if not os.path.exists(img_path):
            print(f"⚠️  Immagine originale non trovata per {case_id}, skip")
            continue

        img_original = load_nifti(img_path)

        # Assicurati che siano 2D
        if len(pred.shape) == 3:
            pred = pred[:, :, 0]
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        if len(img_original.shape) == 3:
            img_original = img_original[:, :, 0]

        # Calcola metriche
        metrics = calculate_all_metrics(pred, gt)
        all_metrics.append(metrics)

        # Crea visualizzazione
        title_prefix = f"{model_name.capitalize()} - "
        create_visualization(img_original, pred, gt, case_id, visualizations_dir, metrics, title_prefix)

        # Progress
        if (idx + 1) % 10 == 0:
            print(f"   Processate {idx + 1}/{len(pred_files)} immagini...")

    # Calcola statistiche aggregate
    aggregate_metrics = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        aggregate_metrics[f'{metric_name}_mean'] = float(np.mean(values))
        aggregate_metrics[f'{metric_name}_std'] = float(np.std(values))
        aggregate_metrics[f'{metric_name}_min'] = float(np.min(values))
        aggregate_metrics[f'{metric_name}_max'] = float(np.max(values))

    # Salva metriche
    metrics_file = os.path.join(output_dir, 'metrics_summary.json')
    with open(metrics_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)

    # Salva metriche in formato testo
    metrics_txt = os.path.join(output_dir, 'metrics_summary.txt')
    with open(metrics_txt, 'w') as f:
        f.write(f"METRICHE AGGREGATE - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Numero immagini analizzate: {len(all_metrics)}\n\n")

        for metric_name in ['dice', 'iou', 'compactness', 'solidity', 'eccentricity', 'hausdorff_distance', 'boundary_iou']:
            f.write(f"{metric_name.upper()}\n")
            f.write(f"  Media:    {aggregate_metrics[f'{metric_name}_mean']:.4f}\n")
            f.write(f"  Std Dev:  {aggregate_metrics[f'{metric_name}_std']:.4f}\n")
            f.write(f"  Min:      {aggregate_metrics[f'{metric_name}_min']:.4f}\n")
            f.write(f"  Max:      {aggregate_metrics[f'{metric_name}_max']:.4f}\n\n")

    print(f"\n✅ Analisi {model_name} completata!")
    print(f"   Visualizzazioni: {visualizations_dir}")
    print(f"   Metriche JSON:   {metrics_file}")
    print(f"   Metriche TXT:    {metrics_txt}")

    return aggregate_metrics


def process_comparison(num_images=None):
    """Processa confronto tra baseline e geometric."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Directory input
    baseline_pred_dir = os.path.join(base_dir, OUTPUT_DIRS['baseline'], 'predictions')
    geometric_pred_dir = os.path.join(base_dir, OUTPUT_DIRS['geometric'], 'predictions')

    # Directory output
    output_dir = os.path.join(base_dir, OUTPUT_DIRS['confronto'])
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    # Verifica che esistano predizioni per entrambi
    if not os.path.exists(baseline_pred_dir):
        print(f"❌ Errore: Predizioni baseline non trovate!")
        print(f"   Esegui prima: python run_inference.py --baseline")
        return False

    if not os.path.exists(geometric_pred_dir):
        print(f"❌ Errore: Predizioni geometric non trovate!")
        print(f"   Esegui prima: python run_inference.py --geometric")
        return False

    # Setup environment
    os.environ['nnUNet_raw'] = os.path.join(base_dir, 'nnUNet_raw')
    raw_dir = os.environ['nnUNet_raw']

    # Lista predizioni (usa baseline come riferimento)
    pred_files = sorted([f for f in os.listdir(baseline_pred_dir) if f.endswith('.nii.gz')])

    if num_images:
        pred_files = pred_files[:num_images]

    print(f"\n📊 Processando confronto per {len(pred_files)} immagini...")

    baseline_metrics = []
    geometric_metrics = []

    for idx, pred_file in enumerate(pred_files):
        case_id = pred_file.replace('.nii.gz', '')

        # Carica predizioni
        pred_baseline_path = os.path.join(baseline_pred_dir, pred_file)
        pred_geometric_path = os.path.join(geometric_pred_dir, pred_file)

        if not os.path.exists(pred_geometric_path):
            print(f"⚠️  Predizione geometric non trovata per {case_id}, skip")
            continue

        pred_baseline = load_nifti(pred_baseline_path)
        pred_geometric = load_nifti(pred_geometric_path)

        # Carica GT
        gt_path = os.path.join(raw_dir, f'Dataset{DATASET_ID:03d}_Shapes', 'labelsTr', f'{case_id}.nii.gz')
        if not os.path.exists(gt_path):
            print(f"⚠️  GT non trovato per {case_id}, skip")
            continue

        gt = load_nifti(gt_path)

        # Carica immagine originale
        img_path = os.path.join(raw_dir, f'Dataset{DATASET_ID:03d}_Shapes', 'imagesTr', f'{case_id}_0000.nii.gz')
        if not os.path.exists(img_path):
            print(f"⚠️  Immagine originale non trovata per {case_id}, skip")
            continue

        img_original = load_nifti(img_path)

        # Assicurati che siano 2D
        if len(pred_baseline.shape) == 3:
            pred_baseline = pred_baseline[:, :, 0]
        if len(pred_geometric.shape) == 3:
            pred_geometric = pred_geometric[:, :, 0]
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        if len(img_original.shape) == 3:
            img_original = img_original[:, :, 0]

        # Calcola metriche per entrambi
        metrics_b = calculate_all_metrics(pred_baseline, gt)
        metrics_g = calculate_all_metrics(pred_geometric, gt)

        baseline_metrics.append(metrics_b)
        geometric_metrics.append(metrics_g)

        # Crea visualizzazione confronto
        create_comparison_visualization(img_original, pred_baseline, pred_geometric, gt, case_id, visualizations_dir)

        # Progress
        if (idx + 1) % 10 == 0:
            print(f"   Processate {idx + 1}/{len(pred_files)} immagini...")

    # Calcola statistiche aggregate per entrambi
    aggregate_baseline = {}
    aggregate_geometric = {}

    for metric_name in baseline_metrics[0].keys():
        # Baseline
        values_b = [m[metric_name] for m in baseline_metrics]
        aggregate_baseline[f'{metric_name}_mean'] = float(np.mean(values_b))
        aggregate_baseline[f'{metric_name}_std'] = float(np.std(values_b))

        # Geometric
        values_g = [m[metric_name] for m in geometric_metrics]
        aggregate_geometric[f'{metric_name}_mean'] = float(np.mean(values_g))
        aggregate_geometric[f'{metric_name}_std'] = float(np.std(values_g))

    # Crea grafico confronto metriche
    create_metrics_comparison_chart(aggregate_baseline, aggregate_geometric, output_dir)

    # Salva confronto metriche in JSON
    comparison_data = {
        'baseline': aggregate_baseline,
        'geometric': aggregate_geometric,
        'num_images': len(baseline_metrics)
    }

    comparison_file = os.path.join(output_dir, 'metrics_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    # Salva confronto in formato testo
    comparison_txt = os.path.join(output_dir, 'metrics_comparison.txt')
    with open(comparison_txt, 'w') as f:
        f.write("CONFRONTO METRICHE - BASELINE vs GEOMETRIC\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Numero immagini analizzate: {len(baseline_metrics)}\n\n")

        for metric_name in ['dice', 'iou', 'compactness', 'solidity', 'eccentricity', 'hausdorff_distance', 'boundary_iou']:
            baseline_mean = aggregate_baseline[f'{metric_name}_mean']
            geometric_mean = aggregate_geometric[f'{metric_name}_mean']

            # Determina miglioramento
            if metric_name in ['hausdorff_distance', 'eccentricity']:
                improvement = ((baseline_mean - geometric_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                better = 'Geometric' if geometric_mean < baseline_mean else 'Baseline'
            else:
                improvement = ((geometric_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                better = 'Geometric' if geometric_mean > baseline_mean else 'Baseline'

            f.write(f"{metric_name.upper()}\n")
            f.write(f"  Baseline:  {baseline_mean:.4f} ± {aggregate_baseline[f'{metric_name}_std']:.4f}\n")
            f.write(f"  Geometric: {geometric_mean:.4f} ± {aggregate_geometric[f'{metric_name}_std']:.4f}\n")
            f.write(f"  Migliore:  {better} ({improvement:+.1f}%)\n\n")

    print(f"\n✅ Confronto completato!")
    print(f"   Visualizzazioni:     {visualizations_dir}")
    print(f"   Grafico confronto:   {os.path.join(output_dir, 'metrics_comparison_chart.png')}")
    print(f"   Metriche JSON:       {comparison_file}")
    print(f"   Metriche TXT:        {comparison_txt}")

    return True


def main():
    """Main con menu interattivo."""
    print("=" * 80)
    print("TEST MODELLI nnU-Net - Baseline vs Geometric")
    print("=" * 80)
    print("\nPREREQUISITO: Esegui prima run_inference.py per generare le predizioni!\n")
    print("Che rete vuoi analizzare? Premi:")
    print("  1 - Baseline       (rete normale)")
    print("  2 - Geometric      (rete con loss geometrica)")
    print("  3 - Confronto      (confronta entrambe le reti)")
    print()

    mode_input = input("Scelta (1/2/3): ").strip()

    if mode_input not in ['1', '2', '3']:
        print("❌ Scelta non valida!")
        return 1

    # Chiedi numero immagini
    print(f"\nQuante immagini vuoi analizzare?")
    num_images_input = input("Numero (invio per tutte): ").strip()
    num_images = int(num_images_input) if num_images_input else None

    print("\n" + "=" * 80)

    # Processa in base alla scelta
    if mode_input == '1':
        print("ANALISI BASELINE")
        print("=" * 80)
        process_single_model('baseline', OUTPUT_DIRS['baseline'], num_images)

    elif mode_input == '2':
        print("ANALISI GEOMETRIC")
        print("=" * 80)
        process_single_model('geometric', OUTPUT_DIRS['geometric'], num_images)

    elif mode_input == '3':
        print("ANALISI CONFRONTO")
        print("=" * 80)
        process_comparison(num_images)

    print("\n" + "=" * 80)
    print("✅ ANALISI COMPLETATA!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
