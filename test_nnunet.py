import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import subprocess
import shutil

def load_nifti(path):
    """Load NIfTI file and return data + affine."""
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def calculate_metrics(pred, gt, class_labels=None):
    """
    Calculate segmentation metrics for multi-class segmentation.
    pred, gt: segmentation masks (can be multi-class)
    class_labels: list of class IDs to evaluate (None = all classes)
    """
    if class_labels is None:
        # Get all unique classes (excluding background 0)
        class_labels = sorted(set(np.unique(gt)) | set(np.unique(pred)))
        class_labels = [c for c in class_labels if c > 0]
    
    metrics_per_class = {}
    
    for class_id in class_labels:
        pred_binary = (pred == class_id).astype(bool)
        gt_binary = (gt == class_id).astype(bool)
        
        # Dice Score
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
        
        # IoU (Jaccard Index)
        union = np.logical_or(pred_binary, gt_binary).sum()
        iou = intersection / (union + 1e-8)
        
        # Sensitivity (Recall)
        tp = intersection
        fn = np.logical_and(~pred_binary, gt_binary).sum()
        sensitivity = tp / (tp + fn + 1e-8)
        
        # Specificity
        tn = np.logical_and(~pred_binary, ~gt_binary).sum()
        fp = np.logical_and(pred_binary, ~gt_binary).sum()
        specificity = tn / (tn + fp + 1e-8)
        
        # Volume (voxel count)
        pred_volume = pred_binary.sum()
        gt_volume = gt_binary.sum()
        
        metrics_per_class[class_id] = {
            'dice': dice,
            'iou': iou,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'pred_volume': pred_volume,
            'gt_volume': gt_volume
        }
    
    # Overall metrics (treating all classes as one)
    pred_any = (pred > 0).astype(bool)
    gt_any = (gt > 0).astype(bool)
    
    intersection = np.logical_and(pred_any, gt_any).sum()
    dice_overall = (2.0 * intersection) / (pred_any.sum() + gt_any.sum() + 1e-8)
    iou_overall = intersection / (np.logical_or(pred_any, gt_any).sum() + 1e-8)
    
    return {
        'per_class': metrics_per_class,
        'overall': {
            'dice': dice_overall,
            'iou': iou_overall
        }
    }

def visualize_prediction(input_img, pred, gt, case_id, output_dir, slice_idx=None):
    """
    Create visualization comparing input, prediction, and ground truth.
    For 3D volumes, shows middle slice or specified slice.
    """
    # Handle 3D volumes
    if len(input_img.shape) == 3:
        if slice_idx is None:
            # Find slice with most tumor
            tumor_slices = (gt > 0).sum(axis=(1, 2))
            if tumor_slices.sum() > 0:
                slice_idx = np.argmax(tumor_slices)
            else:
                slice_idx = input_img.shape[0] // 2
        
        input_slice = input_img[slice_idx, :, :]
        pred_slice = pred[slice_idx, :, :] if len(pred.shape) == 3 else pred
        gt_slice = gt[slice_idx, :, :] if len(gt.shape) == 3 else gt
    else:
        input_slice = input_img
        pred_slice = pred
        gt_slice = gt
    
    # Create figure with 4 subplots: Input, GT, Pred, Overlay
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Input image
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].set_title('Input (FLAIR)', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(input_slice, cmap='gray', alpha=0.7)
    axes[1].imshow(gt_slice, cmap='Reds', alpha=0.6, vmin=0, vmax=gt_slice.max())
    axes[1].set_title('Ground Truth', fontsize=12)
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
    
    plt.suptitle(f'Case: {case_id} | Slice: {slice_idx if len(input_img.shape) == 3 else "2D"}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{case_id}_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def run_nnunet_inference(input_dir, output_dir, dataset_id, config, fold):
    """
    Run nnU-Net inference using CLI tool.
    """
    print(f"Running nnU-Net inference...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    cmd = [
        'nnUNetv2_predict',
        '-i', input_dir,
        '-o', output_dir,
        '-d', str(dataset_id),
        '-c', config,
        '-f', str(fold),
        '--save_probabilities'  # Save probability maps too
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during inference:")
        print(result.stderr)
        raise RuntimeError("nnU-Net inference failed")
    
    print("Inference completed successfully!")
    return output_dir

def test_nnunet_model(dataset_id=500, config='2d', fold=0, results_dir='risultati', 
                      use_existing_predictions=False, prediction_dir=None):
    """
    Test nnU-Net model on test set and generate visualizations + metrics.
    """
    print("=" * 60)
    print("nnU-Net Model Testing & Evaluation")
    print("=" * 60)
    print(f"Dataset: {dataset_id}")
    print(f"Config: {config}")
    print(f"Fold: {fold}\n")
    
    # Setup paths
    nnunet_raw = os.environ.get('nnUNet_raw', '/workspace/nnUNet_raw')
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed', '/workspace/nnUNet_preprocessed')
    nnunet_results = os.environ.get('nnUNet_results', '/workspace/nnUNet_results')
    
    # Find dataset
    dataset_name = None
    for d in os.listdir(nnunet_raw):
        if d.startswith(f"Dataset{dataset_id:03d}_"):
            dataset_name = d
            break
    
    if dataset_name is None:
        raise ValueError(f"Dataset {dataset_id} not found in {nnunet_raw}")
    
    dataset_path = os.path.join(nnunet_raw, dataset_name)
    
    # Load dataset.json for class labels
    dataset_json_path = os.path.join(dataset_path, 'dataset.json')
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
    
    class_names = dataset_info.get('labels', {})
    print(f"Classes: {class_names}\n")
    
    # Load splits to get test cases
    # First, try to use our custom hybrid_splits.json (if it exists)
    hybrid_splits_file = os.path.join('/workspace', 'hybrid_splits.json')
    if os.path.exists(hybrid_splits_file):
        print("Using custom hybrid_splits.json for test set...")
        with open(hybrid_splits_file, 'r') as f:
            hybrid_splits = json.load(f)
        test_cases = hybrid_splits.get('test', [])
    else:
        # Fallback to nnU-Net splits (which only have train/val, no test)
        splits_file = os.path.join(nnunet_preprocessed, dataset_name, 'splits_final.json')
        if not os.path.exists(splits_file):
            splits_file = os.path.join(dataset_path, 'splits_final.json')
        
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            # nnU-Net splits have 'train' and 'val', but no 'test'
            # Use 'val' as test set if 'test' doesn't exist
            if isinstance(splits, list):
                if 'test' in splits[fold]:
                    test_cases = splits[fold]['test']
                elif 'val' in splits[fold]:
                    print("Note: Using validation set as test set (no separate test set in splits)")
                    test_cases = splits[fold]['val']
                else:
                    test_cases = []
            else:
                test_cases = splits.get('test', splits.get('val', []))
        else:
            print("Warning: No splits file found. Using all cases in labelsTr.")
            labels_dir = os.path.join(dataset_path, 'labelsTr')
            test_cases = [f.replace('.nii.gz', '') for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]
    
    print(f"Found {len(test_cases)} test cases\n")
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, 'visualizations')
    pred_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Step 1: Run inference if needed
    if not use_existing_predictions:
        print("Step 1: Running nnU-Net inference...")
        # Create temporary input directory with test cases
        temp_input = os.path.join(results_dir, 'temp_input')
        os.makedirs(temp_input, exist_ok=True)
        
        images_dir = os.path.join(dataset_path, 'imagesTr')
        for case_id in test_cases:
            # Copy all 4 modalities
            for mod_idx in range(4):
                src = os.path.join(images_dir, f'{case_id}_{mod_idx:04d}.nii.gz')
                if os.path.exists(src):
                    dst = os.path.join(temp_input, f'{case_id}_{mod_idx:04d}.nii.gz')
                    shutil.copy2(src, dst)
        
        # Run inference
        temp_output = os.path.join(results_dir, 'temp_output')
        run_nnunet_inference(temp_input, temp_output, dataset_id, config, fold)
        prediction_dir = temp_output
    else:
        if prediction_dir is None:
            raise ValueError("prediction_dir required when use_existing_predictions=True")
        print(f"Using existing predictions from: {prediction_dir}\n")
    
    # Step 2: Evaluate predictions
    print("\nStep 2: Evaluating predictions and generating visualizations...")
    
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
            
            # Load input image (FLAIR for visualization)
            input_path = os.path.join(images_dir, f'{case_id}_0003.nii.gz')  # FLAIR is usually 0003
            if not os.path.exists(input_path):
                input_path = os.path.join(images_dir, f'{case_id}_0000.nii.gz')  # Fallback to T1
            
            input_data, _ = load_nifti(input_path)
            
            # Calculate metrics
            metrics = calculate_metrics(pred_data, gt_data, class_labels=list(class_names.keys()))
            metrics['case_id'] = case_id
            all_metrics.append(metrics)
            
            # Visualize
            visualize_prediction(input_data, pred_data, gt_data, case_id, vis_dir)
            
            # Save prediction copy
            shutil.copy2(pred_path, os.path.join(pred_dir, f'{case_id}.nii.gz'))
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Generate summary report
    print("\nStep 3: Generating summary report...")
    
    summary_path = os.path.join(results_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("nnU-Net Test Results Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset_id} ({dataset_name})\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Test Cases: {len(all_metrics)}\n\n")
        
        # Per-class averages
        if len(all_metrics) > 0 and 'per_class' in all_metrics[0]:
            f.write("-" * 80 + "\n")
            f.write("Per-Class Average Metrics:\n")
            f.write("-" * 80 + "\n")
            
            class_ids = sorted(set().union(*[m['per_class'].keys() for m in all_metrics]))
            for class_id in class_ids:
                class_name = class_names.get(str(class_id), f"Class_{class_id}")
                f.write(f"\n{class_name} (ID: {class_id}):\n")
                
                dices = [m['per_class'][class_id]['dice'] for m in all_metrics if class_id in m['per_class']]
                ious = [m['per_class'][class_id]['iou'] for m in all_metrics if class_id in m['per_class']]
                sens = [m['per_class'][class_id]['sensitivity'] for m in all_metrics if class_id in m['per_class']]
                spec = [m['per_class'][class_id]['specificity'] for m in all_metrics if class_id in m['per_class']]
                
                f.write(f"  Dice Score:     {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")
                f.write(f"  IoU:            {np.mean(ious):.4f} ± {np.std(ious):.4f}\n")
                f.write(f"  Sensitivity:    {np.mean(sens):.4f} ± {np.std(sens):.4f}\n")
                f.write(f"  Specificity:    {np.mean(spec):.4f} ± {np.std(spec):.4f}\n")
        
        # Overall metrics
        f.write("\n" + "-" * 80 + "\n")
        f.write("Overall Metrics (All Classes Combined):\n")
        f.write("-" * 80 + "\n")
        
        overall_dice = [m['overall']['dice'] for m in all_metrics]
        overall_iou = [m['overall']['iou'] for m in all_metrics]
        
        f.write(f"Dice Score:     {np.mean(overall_dice):.4f} ± {np.std(overall_dice):.4f}\n")
        f.write(f"IoU:            {np.mean(overall_iou):.4f} ± {np.std(overall_iou):.4f}\n")
        
        # Per-case breakdown
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Case Detailed Metrics:\n")
        f.write("=" * 80 + "\n")
        
        for m in all_metrics:
            f.write(f"\n{m['case_id']}:\n")
            f.write(f"  Overall Dice: {m['overall']['dice']:.4f}, IoU: {m['overall']['iou']:.4f}\n")
            for class_id, class_metrics in m['per_class'].items():
                class_name = class_names.get(str(class_id), f"Class_{class_id}")
                f.write(f"    {class_name}: Dice={class_metrics['dice']:.4f}, "
                       f"IoU={class_metrics['iou']:.4f}, "
                       f"Sens={class_metrics['sensitivity']:.4f}\n")
    
    # Save JSON version too (convert numpy types to Python native types)
    def convert_to_native(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    json_path = os.path.join(results_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(convert_to_native(all_metrics), f, indent=2)
    
    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"  - Visualizations: {vis_dir} ({len(all_metrics)} images)")
    print(f"  - Predictions: {pred_dir}")
    print(f"  - Summary (TXT): {summary_path}")
    print(f"  - Summary (JSON): {json_path}")
    print(f"\nAverage Overall Dice: {np.mean(overall_dice):.4f} ± {np.std(overall_dice):.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test nnU-Net model and generate visualizations')
    parser.add_argument('-d', '--dataset', type=int, default=500, help='Dataset ID')
    parser.add_argument('-c', '--config', type=str, default='2d', help='Configuration (2d, 3d_fullres, etc.)')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Fold number')
    parser.add_argument('-o', '--output', type=str, default='risultati', help='Output directory')
    parser.add_argument('--use-existing', action='store_true', 
                       help='Use existing predictions (skip inference)')
    parser.add_argument('--pred-dir', type=str, default=None,
                       help='Directory with existing predictions (required if --use-existing)')
    
    args = parser.parse_args()
    
    test_nnunet_model(args.dataset, args.config, args.fold, args.output, 
                     args.use_existing, args.pred_dir)
