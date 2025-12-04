"""
Script per testare nnU-Net sul dataset geometrico e calcolare metriche.
"""
import os
import numpy as np
import nibabel as nib  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from pathlib import Path
import json
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
import subprocess
import shutil

def load_nifti(path):
    """Load NIfTI file and return data + affine."""
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def calculate_dice(pred, gt):
    """Calculate Dice score. Works on 2D or 3D volumes."""
    # Flatten to handle both 2D and 3D
    pred_binary = (pred > 0).astype(bool)
    gt_binary = (gt > 0).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
    
    return dice

def calculate_iou(pred, gt):
    """Calculate IoU (Jaccard Index). Works on 2D or 3D volumes."""
    # Flatten to handle both 2D and 3D
    pred_binary = (pred > 0).astype(bool)
    gt_binary = (gt > 0).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / (union + 1e-8)
    
    return iou

def visualize_prediction(input_img, pred, gt, case_id, output_dir):
    """
    Create visualization comparing input, prediction, and ground truth.
    Handles 2D images saved as 3D NIfTI with shape (H, W, 1).
    """
    # Handle different shapes
    if len(input_img.shape) == 3:
        # Check if it's (H, W, 1) format (2D image saved as 3D)
        if input_img.shape[2] == 1:
            # Take the single slice: (H, W, 1) -> (H, W)
            input_slice = input_img[:, :, 0]
            pred_slice = pred[:, :, 0] if len(pred.shape) == 3 and pred.shape[2] == 1 else (pred[0, :, :] if len(pred.shape) == 3 else pred)
            gt_slice = gt[:, :, 0] if len(gt.shape) == 3 and gt.shape[2] == 1 else (gt[0, :, :] if len(gt.shape) == 3 else gt)
        else:
            # True 3D volume: take middle slice along first dimension
            slice_idx = input_img.shape[0] // 2
            input_slice = input_img[slice_idx, :, :]
            pred_slice = pred[slice_idx, :, :] if len(pred.shape) == 3 else pred
            gt_slice = gt[slice_idx, :, :] if len(gt.shape) == 3 else gt
    elif len(input_img.shape) == 2:
        # Already 2D
        input_slice = input_img
        pred_slice = pred if len(pred.shape) == 2 else (pred[:, :, 0] if len(pred.shape) == 3 and pred.shape[2] == 1 else pred[0, :, :])
        gt_slice = gt if len(gt.shape) == 2 else (gt[:, :, 0] if len(gt.shape) == 3 and gt.shape[2] == 1 else gt[0, :, :])
    else:
        input_slice = input_img
        pred_slice = pred
        gt_slice = gt
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Input image
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].set_title('Input Image', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(input_slice, cmap='gray', alpha=0.7)
    axes[1].imshow(gt_slice, cmap='Reds', alpha=0.6, vmin=0, vmax=gt_slice.max())
    axes[1].set_title('Ground Truth (Circles)', fontsize=12)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(input_slice, cmap='gray', alpha=0.7)
    axes[2].imshow(pred_slice, cmap='Blues', alpha=0.6, vmin=0, vmax=pred_slice.max())
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')
    
    # Overlay (GT in red, Pred in blue, overlap in purple)
    axes[3].imshow(input_slice, cmap='gray', alpha=0.5)
    overlay = np.zeros((*input_slice.shape, 3))
    overlay[:, :, 0] = (gt_slice > 0) * 0.8  # Red for GT
    overlay[:, :, 2] = (pred_slice > 0) * 0.8  # Blue for Pred
    axes[3].imshow(overlay, alpha=0.6)
    axes[3].set_title('Overlay (Red=GT, Blue=Pred)', fontsize=12)
    axes[3].axis('off')
    
    plt.suptitle(f'Case: {case_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{case_id}_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def run_inference(input_dir, output_dir, dataset_id, config, fold):
    """Run nnU-Net inference."""
    print(f"Running nnU-Net inference...")
    print(f"Using checkpoint_final (link to checkpoint_best)")
    
    cmd = [
        'nnUNetv2_predict',
        '-i', input_dir,
        '-o', output_dir,
        '-d', str(dataset_id),
        '-c', config,
        '-f', str(fold)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during inference:")
        print(result.stderr)
        raise RuntimeError("nnU-Net inference failed")
    
    print("Inference completed successfully!")

def test_geometric_model(dataset_id=501, config='2d', fold=0, results_dir='risultati',
                         use_existing_predictions=False, prediction_dir=None):
    """
    Test nnU-Net model on geometric dataset.
    """
    print("=" * 60)
    print("Testing nnU-Net on Geometric Dataset")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print(f"Configuration: {config}")
    print(f"Fold: {fold}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nnunet_raw_env = os.environ.get('nnUNet_raw')
    if nnunet_raw_env and os.path.exists(nnunet_raw_env):
        nnunet_raw = nnunet_raw_env
    else:
        nnunet_raw = os.path.join(base_dir, 'nnUNet_raw')
    
    nnunet_preprocessed_env = os.environ.get('nnUNet_preprocessed')
    if nnunet_preprocessed_env and os.path.exists(nnunet_preprocessed_env):
        nnunet_preprocessed = nnunet_preprocessed_env
    else:
        nnunet_preprocessed = os.path.join(base_dir, 'nnUNet_preprocessed')
    
    dataset_name = f"Dataset{dataset_id:03d}_Shapes"
    dataset_path = os.path.join(nnunet_raw, dataset_name)
    
    # Fallback: try geometrica directory if not found
    if not os.path.exists(dataset_path):
        geometrica_raw = os.path.join(base_dir, 'nnUNet_raw')
        geometrica_path = os.path.join(geometrica_raw, dataset_name)
        if os.path.exists(geometrica_path):
            dataset_path = geometrica_path
            nnunet_raw = geometrica_raw
    
    # Load dataset.json
    dataset_json_path = os.path.join(dataset_path, 'dataset.json')
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_path}")
    
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
    
    print(f"Classes: {dataset_info.get('labels', {})}\n")
    
    # Get test cases (use all cases for now, or split train/test)
    labels_dir = os.path.join(dataset_path, 'labelsTr')
    test_cases = sorted([f.replace('.nii.gz', '') for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    
    # For testing, use a subset (e.g., last 20%)
    num_test = max(1, len(test_cases) // 5)
    test_cases = test_cases[-num_test:]
    
    print(f"Using {len(test_cases)} test cases\n")
    
    # Create output directories
    results_path = os.path.join(base_dir, results_dir)
    os.makedirs(results_path, exist_ok=True)
    vis_dir = os.path.join(results_path, 'visualizations')
    pred_dir = os.path.join(results_path, 'predictions')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Step 1: Run inference if needed
    if not use_existing_predictions:
        print("Step 1: Running nnU-Net inference...")
        # Use system temp directory instead of results directory for temp files
        import tempfile
        temp_input = tempfile.mkdtemp(prefix='nnunet_test_')
        
        images_dir = os.path.join(dataset_path, 'imagesTr')
        for case_id in test_cases:
            src = os.path.join(images_dir, f'{case_id}_0000.nii.gz')
            if os.path.exists(src):
                dst = os.path.join(temp_input, f'{case_id}_0000.nii.gz')
                shutil.copy2(src, dst)
        
        temp_output = tempfile.mkdtemp(prefix='nnunet_pred_')
        run_inference(temp_input, temp_output, dataset_id, config, fold)
        prediction_dir = temp_output
    else:
        if prediction_dir is None:
            raise ValueError("prediction_dir required when use_existing_predictions=True")
        print(f"Using existing predictions from: {prediction_dir}\n")
    
    # Step 2: Evaluate predictions
    print("\nStep 2: Evaluating predictions...")
    
    all_metrics = []
    labels_dir = os.path.join(dataset_path, 'labelsTr')
    images_dir = os.path.join(dataset_path, 'imagesTr')
    
    for case_id in tqdm(test_cases, desc="Processing cases"):
        try:
            # Load prediction
            pred_path = os.path.join(prediction_dir, f'{case_id}.nii.gz')
            if not os.path.exists(pred_path):
                print(f"Warning: Prediction not found for {case_id}")
                continue
            
            pred_data, _ = load_nifti(pred_path)
            
            # Load ground truth
            gt_path = os.path.join(labels_dir, f'{case_id}.nii.gz')
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth not found for {case_id}")
                continue
            
            gt_data, _ = load_nifti(gt_path)
            
            # Load input image
            input_path = os.path.join(images_dir, f'{case_id}_0000.nii.gz')
            if not os.path.exists(input_path):
                continue
            
            input_data, _ = load_nifti(input_path)
            
            # Calculate metrics
            dice = calculate_dice(pred_data, gt_data)
            iou = calculate_iou(pred_data, gt_data)
            
            all_metrics.append({
                'case_id': case_id,
                'dice': dice,
                'iou': iou
            })
            
            # Visualize
            visualize_prediction(input_data, pred_data, gt_data, case_id, vis_dir)
            
            # Save prediction copy
            shutil.copy2(pred_path, os.path.join(pred_dir, f'{case_id}.nii.gz'))
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Generate summary
    print("\nStep 3: Generating summary...")
    
    if len(all_metrics) > 0:
        dices = [m['dice'] for m in all_metrics]
        ious = [m['iou'] for m in all_metrics]
        
        summary_path = os.path.join(results_path, 'metrics_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Geometric Dataset Test Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset: {dataset_id} ({dataset_name})\n")
            f.write(f"Configuration: {config}\n")
            f.write(f"Fold: {fold}\n")
            f.write(f"Test Cases: {len(all_metrics)}\n\n")
            f.write("-" * 80 + "\n")
            f.write("Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dice Score:     {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")
            f.write(f"IoU:            {np.mean(ious):.4f} ± {np.std(ious):.4f}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("Per-Case Metrics:\n")
            f.write("=" * 80 + "\n")
            for m in all_metrics:
                f.write(f"{m['case_id']}: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}\n")
        
        json_path = os.path.join(results_path, 'metrics_summary.json')
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Testing Complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {results_path}")
        print(f"  - Visualizations: {vis_dir} ({len(all_metrics)} images)")
        print(f"  - Predictions: {pred_dir}")
        print(f"  - Summary (TXT): {summary_path}")
        print(f"  - Summary (JSON): {json_path}")
        print(f"\nAverage Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"Average IoU:  {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    else:
        print("No metrics calculated!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test nnU-Net on geometric dataset')
    parser.add_argument('-d', '--dataset', type=int, default=501, help='Dataset ID')
    parser.add_argument('-c', '--config', type=str, default='2d', help='Configuration (2d, 3d_fullres, etc.)')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Fold number')
    parser.add_argument('-o', '--output', type=str, default='risultati', help='Output directory')
    parser.add_argument('--use-existing', action='store_true',
                       help='Use existing predictions (skip inference)')
    parser.add_argument('--pred-dir', type=str, default=None,
                       help='Directory with existing predictions (required if --use-existing)')
    
    args = parser.parse_args()
    
    test_geometric_model(args.dataset, args.config, args.fold, args.output,
                        args.use_existing, args.pred_dir)

