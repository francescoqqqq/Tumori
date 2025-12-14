"""
Converti il dataset PNG in formato nnU-Net e fai preprocessing.
Questo script rende il dataset pronto per il training.
"""
import os
import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import nibabel as nib  # pyright: ignore[reportMissingImports]
import subprocess
import json
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

TARGET_DIR = "nnUNet_raw"

def convert_png_to_nifti(png_path, nifti_path):
    """Converti PNG in NIfTI 3D (con 1 slice)."""
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    img_3d = np.expand_dims(img, axis=2)  # (H, W) -> (H, W, 1)
    affine = np.eye(4)
    nii = nib.Nifti1Image(img_3d.astype(np.float32), affine)
    os.makedirs(os.path.dirname(nifti_path), exist_ok=True)
    nib.save(nii, nifti_path)

def convert_mask_to_nifti(mask_path, nifti_path):
    """Converti maschera PNG in NIfTI (255 -> 1, 0 -> 0)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_binary = (mask > 127).astype(np.uint8)
    mask_3d = np.expand_dims(mask_binary, axis=2)
    affine = np.eye(4)
    nii = nib.Nifti1Image(mask_3d.astype(np.uint8), affine)
    os.makedirs(os.path.dirname(nifti_path), exist_ok=True)
    nib.save(nii, nifti_path)

def create_dataset_json(output_dir, num_training):
    """Crea dataset.json per nnU-Net."""
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "circle": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "Shapes",
        "description": "Geometric shapes dataset - segment circles"
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"✅ Creato: {json_path}")

def convert_dataset(dataset_id, dataset_name, source_dir):
    """
    Converti un dataset PNG in formato nnU-Net.

    Args:
        dataset_id: ID del dataset (es. 501, 502)
        dataset_name: Nome del dataset (es. "Dataset501_Shapes")
        source_dir: Directory sorgente (es. "dataset_shapes", "dataset_shapes_one")
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_images = os.path.join(base_dir, source_dir, "imagesTr")
    source_labels = os.path.join(base_dir, source_dir, "labelsTr")

    target_base = os.path.join(base_dir, TARGET_DIR, dataset_name)
    target_images = os.path.join(target_base, "imagesTr")
    target_labels = os.path.join(target_base, "labelsTr")
    
    if not os.path.exists(source_images):
        raise FileNotFoundError(f"Directory non trovata: {source_images}")
    
    image_files = sorted([f for f in os.listdir(source_images) if f.endswith('.png')])
    num_images = len(image_files)
    
    print(f"Trovate {num_images} immagini da convertire")
    print(f"Output: {target_base}\n")
    
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    for img_file in tqdm(image_files, desc="Conversione"):
        case_id = os.path.splitext(img_file)[0]
        src_img = os.path.join(source_images, img_file)
        src_mask = os.path.join(source_labels, img_file)
        
        target_img = os.path.join(target_images, f"{case_id}_0000.nii.gz")
        target_mask = os.path.join(target_labels, f"{case_id}.nii.gz")
        
        convert_png_to_nifti(src_img, target_img)
        convert_mask_to_nifti(src_mask, target_mask)
    
    create_dataset_json(target_base, num_images)

    print(f"\n✅ Conversione completata!")
    print(f"Dataset ID: {dataset_id}")

    # Imposta variabili d'ambiente per preprocessing
    os.environ['nnUNet_raw'] = os.path.join(base_dir, "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = os.path.join(base_dir, "nnUNet_results")
    env = os.environ.copy()

    # Verifica se preprocessing è già fatto
    preprocessed_dir = os.path.join(env['nnUNet_preprocessed'], dataset_name)
    if os.path.exists(preprocessed_dir):
        print(f"\n✅ Preprocessing già fatto, salto")
        return True

    # Esegui preprocessing
    print(f"\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    cmd = [
        'nnUNetv2_plan_and_preprocess',
        '-d', str(dataset_id),
        '--verify_dataset_integrity'
    ]

    print(f"Comando: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env, cwd=base_dir)

    if result.returncode == 0:
        print("\n✅ Preprocessing completato!")
        print(f"\n✅ Dataset pronto per il training!")
        print(f"   Prossimo passo: python train.py")
        return True
    else:
        print("\n❌ Preprocessing fallito!")
        return False


def main():
    """Menu interattivo per scegliere quale dataset convertire."""
    print("\n" + "="*60)
    print("CONVERSIONE DATASET → nnU-Net Format")
    print("="*60)
    print("\nScegli dataset da convertire:")
    print("  [1] dataset_shapes (multi-circle: 2-5 cerchi) → Dataset501")
    print("  [2] dataset_shapes_one (single-circle: 1 cerchio) → Dataset502")
    print()

    while True:
        choice = input("Inserisci scelta (1 o 2): ").strip()

        if choice == "1":
            # Multi-circle → Dataset501
            print("\n🔄 Conversione dataset_shapes → Dataset501_Shapes...")
            return convert_dataset(
                dataset_id=501,
                dataset_name="Dataset501_Shapes",
                source_dir="dataset_shapes"
            )
        elif choice == "2":
            # Single-circle → Dataset502
            print("\n🔄 Conversione dataset_shapes_one → Dataset502_Shapes_One...")
            return convert_dataset(
                dataset_id=502,
                dataset_name="Dataset502_Shapes_One",
                source_dir="dataset_shapes_one"
            )
        else:
            print("❌ Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()

