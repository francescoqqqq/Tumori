"""
Script per training nnU-Net sul dataset geometrico.
"""
import os
import subprocess
import sys

def train_geometric(dataset_id=501, config='2d', fold=0):
    """
    Train nnU-Net on geometric dataset.
    """
    print("=" * 60)
    print("Training nnU-Net on Geometric Dataset")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print(f"Configuration: {config}")
    print(f"Fold: {fold}\n")
    
    # Check environment variables
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nnunet_raw = os.environ.get('nnUNet_raw', os.path.join(base_dir, 'nnUNet_raw'))
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed', os.path.join(base_dir, 'nnUNet_preprocessed'))
    nnunet_results = os.environ.get('nnUNet_results', os.path.join(base_dir, 'nnUNet_results'))
    
    print(f"nnUNet_raw: {nnunet_raw}")
    print(f"nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"nnUNet_results: {nnunet_results}\n")
    
    # Check if preprocessing was done
    preprocessed_dir = os.path.join(nnunet_preprocessed, f"Dataset{dataset_id:03d}_Shapes")
    if not os.path.exists(preprocessed_dir):
        print("⚠️  Warning: Preprocessed dataset not found!")
        print(f"   Expected: {preprocessed_dir}")
        print("\nPlease run preprocessing first:")
        print(f"   nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
        return False
    
    # Run training
    cmd = [
        'nnUNetv2_train',
        str(dataset_id),
        config,
        str(fold)
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        return True
    else:
        print("\n❌ Training failed!")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train nnU-Net on geometric dataset')
    parser.add_argument('-d', '--dataset', type=int, default=501, help='Dataset ID')
    parser.add_argument('-c', '--config', type=str, default='2d', help='Configuration (2d, 3d_fullres, etc.)')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Fold number')
    
    args = parser.parse_args()
    
    success = train_geometric(args.dataset, args.config, args.fold)
    sys.exit(0 if success else 1)

