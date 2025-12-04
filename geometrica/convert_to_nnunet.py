"""
Convert geometric dataset (PNG images) to nnU-Net format (NIfTI).
This script converts the PNG images and masks to NIfTI format required by nnU-Net.
"""
import os
import cv2
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import shutil
from tqdm import tqdm

# Configuration
SOURCE_DIR = "dataset_shapes"
TARGET_DATASET_ID = 501  # Different from clinical dataset (500)
TARGET_DATASET_NAME = f"Dataset{TARGET_DATASET_ID:03d}_Shapes"
NNUNET_RAW_BASE = "nnUNet_raw"

def convert_png_to_nifti(png_path, nifti_path):
    """
    Convert PNG image to NIfTI format.
    For nnU-Net 2D, images should be 2D saved as 3D NIfTI with a single slice in Z.
    Format: (H, W) -> (H, W, 1) for NIfTI
    """
    # Read PNG image
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    
    # Add Z dimension: (H, W) -> (H, W, 1)
    # NIfTI expects (x, y, z) format
    img_3d = np.expand_dims(img, axis=2)  # Shape: (H, W, 1)
    
    # Create NIfTI image with identity affine
    affine = np.eye(4)
    nii = nib.Nifti1Image(img_3d.astype(np.float32), affine)
    
    # Save
    os.makedirs(os.path.dirname(nifti_path), exist_ok=True)
    nib.save(nii, nifti_path)

def convert_mask_to_nnunet_format(mask_path, nifti_path):
    """
    Convert mask PNG to nnU-Net format NIfTI.
    Converts 255 -> 1 (circle class), 0 -> 0 (background)
    Format: (H, W) -> (H, W, 1) for NIfTI
    """
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert: 255 -> 1, 0 -> 0
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Add Z dimension: (H, W) -> (H, W, 1)
    mask_3d = np.expand_dims(mask_binary, axis=2)  # Shape: (H, W, 1)
    
    # Create NIfTI
    affine = np.eye(4)
    nii = nib.Nifti1Image(mask_3d.astype(np.uint8), affine)
    
    # Save
    os.makedirs(os.path.dirname(nifti_path), exist_ok=True)
    nib.save(nii, nifti_path)

def create_dataset_json(output_dir, num_training):
    """
    Create dataset.json for nnU-Net.
    """
    dataset_json = {
        "channel_names": {
            "0": "grayscale"
        },
        "labels": {
            "background": 0,
            "circle": 1
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "Shapes",
        "description": "Geometric shapes dataset for testing nnU-Net on simple segmentation tasks. Target: circles among other geometric shapes."
    }
    
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created dataset.json: {json_path}")
    return dataset_json

def convert_dataset():
    """
    Main function to convert the geometric dataset to nnU-Net format.
    """
    print("=" * 60)
    print("Converting Geometric Dataset to nnU-Net Format")
    print("=" * 60)
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_images_dir = os.path.join(base_dir, SOURCE_DIR, "imagesTr")
    source_labels_dir = os.path.join(base_dir, SOURCE_DIR, "labelsTr")
    
    target_base = os.path.join(base_dir, NNUNET_RAW_BASE, TARGET_DATASET_NAME)
    target_images_dir = os.path.join(target_base, "imagesTr")
    target_labels_dir = os.path.join(target_base, "labelsTr")
    
    # Check source exists
    if not os.path.exists(source_images_dir):
        raise FileNotFoundError(f"Source images directory not found: {source_images_dir}")
    if not os.path.exists(source_labels_dir):
        raise FileNotFoundError(f"Source labels directory not found: {source_labels_dir}")
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(source_images_dir) if f.endswith('.png')])
    num_images = len(image_files)
    
    if num_images == 0:
        raise ValueError(f"No PNG images found in {source_images_dir}")
    
    print(f"Found {num_images} images to convert")
    print(f"Target dataset: {TARGET_DATASET_NAME}")
    print(f"Output directory: {target_base}\n")
    
    # Create directories
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    # Convert images
    print("Converting images and masks...")
    for img_file in tqdm(image_files, desc="Converting"):
        # Get base name (without extension)
        case_id = os.path.splitext(img_file)[0]  # e.g., "shape_000"
        
        # Source paths
        src_img = os.path.join(source_images_dir, img_file)
        src_mask = os.path.join(source_labels_dir, img_file)
        
        # Target paths (nnU-Net format: case_id_0000.nii.gz for image, case_id.nii.gz for label)
        target_img = os.path.join(target_images_dir, f"{case_id}_0000.nii.gz")
        target_mask = os.path.join(target_labels_dir, f"{case_id}.nii.gz")
        
        # Convert image
        convert_png_to_nifti(src_img, target_img)
        
        # Convert mask
        convert_mask_to_nnunet_format(src_mask, target_mask)
    
    # Create dataset.json
    print("\nCreating dataset.json...")
    create_dataset_json(target_base, num_images)
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Dataset ID: {TARGET_DATASET_ID}")
    print(f"Dataset location: {target_base}")
    print(f"Number of training cases: {num_images}")
    print("\nNext steps:")
    print(f"1. Set environment variables:")
    print(f"   export nnUNet_raw=\"{os.path.abspath(os.path.join(base_dir, NNUNET_RAW_BASE))}\"")
    print(f"   export nnUNet_preprocessed=\"{os.path.abspath(os.path.join(base_dir, 'nnUNet_preprocessed'))}\"")
    print(f"   export nnUNet_results=\"{os.path.abspath(os.path.join(base_dir, 'nnUNet_results'))}\"")
    print(f"2. Run preprocessing:")
    print(f"   nnUNetv2_plan_and_preprocess -d {TARGET_DATASET_ID} --verify_dataset_integrity")
    print(f"3. Train:")
    print(f"   nnUNetv2_train {TARGET_DATASET_ID} 2d 0")

if __name__ == "__main__":
    convert_dataset()

