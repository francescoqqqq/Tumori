"""
Script per verificare rapidamente i risultati del modello anche durante il training.
Controlla se ci sono problemi nel calcolo delle metriche o nel dataset.
"""
import os
import numpy as np
import nibabel as nib  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from pathlib import Path
import json

def load_nifti(path):
    """Load NIfTI file and return data."""
    nii = nib.load(path)
    return nii.get_fdata()

def check_dataset_integrity(dataset_id=501):
    """
    Verifica l'integrità del dataset e controlla se ci sono problemi.
    """
    print("=" * 60)
    print("Verifica Integrità Dataset Geometrico")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Use environment variable or default to geometrica directory
    nnunet_raw_env = os.environ.get('nnUNet_raw')
    if nnunet_raw_env and os.path.exists(nnunet_raw_env):
        nnunet_raw = nnunet_raw_env
    else:
        # Default to geometrica directory
        nnunet_raw = os.path.join(base_dir, 'nnUNet_raw')
    
    dataset_name = f"Dataset{dataset_id:03d}_Shapes"
    dataset_path = os.path.join(nnunet_raw, dataset_name)
    
    # Fallback: try geometrica directory if not found
    if not os.path.exists(dataset_path):
        geometrica_raw = os.path.join(base_dir, 'nnUNet_raw')
        geometrica_path = os.path.join(geometrica_raw, dataset_name)
        if os.path.exists(geometrica_path):
            dataset_path = geometrica_path
            nnunet_raw = geometrica_raw
    
    labels_dir = os.path.join(dataset_path, 'labelsTr')
    images_dir = os.path.join(dataset_path, 'imagesTr')
    
    # Conta casi
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    
    print(f"\nCasi trovati:")
    print(f"  Labels: {len(label_files)}")
    print(f"  Images: {len(image_files)}")
    
    # Verifica alcuni casi
    print(f"\nVerifica primi 5 casi...")
    issues = []
    
    for i, label_file in enumerate(label_files[:5]):
        case_id = label_file.replace('.nii.gz', '')
        label_path = os.path.join(labels_dir, label_file)
        image_path = os.path.join(images_dir, f'{case_id}_0000.nii.gz')
        
        if not os.path.exists(image_path):
            issues.append(f"❌ {case_id}: Immagine mancante")
            continue
        
        # Carica dati
        label_data = load_nifti(label_path)
        image_data = load_nifti(image_path)
        
        # Verifica dimensioni
        label_shape = label_data.shape
        image_shape = image_data.shape
        
        # Verifica valori label
        unique_labels = np.unique(label_data)
        label_sum = label_data.sum()
        
        # Verifica valori immagine
        image_min = image_data.min()
        image_max = image_data.max()
        image_mean = image_data.mean()
        
        print(f"\n  {case_id}:")
        print(f"    Label shape: {label_shape}, Unique values: {unique_labels}, Sum: {label_sum}")
        print(f"    Image shape: {image_shape}, Range: [{image_min:.1f}, {image_max:.1f}], Mean: {image_mean:.1f}")
        
        # Controlli
        if len(unique_labels) > 2:
            issues.append(f"⚠️  {case_id}: Label ha più di 2 valori unici: {unique_labels}")
        if label_sum == 0:
            issues.append(f"⚠️  {case_id}: Label completamente vuota (nessun cerchio)")
        if image_shape != label_shape:
            issues.append(f"❌ {case_id}: Dimensioni non corrispondono: image={image_shape}, label={label_shape}")
    
    if issues:
        print(f"\n⚠️  Problemi trovati:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ Nessun problema evidente nei primi 5 casi")
    
    return len(issues) == 0

def check_predictions(dataset_id=501, config='2d', fold=0):
    """
    Verifica se ci sono predizioni già salvate e le analizza.
    """
    print("\n" + "=" * 60)
    print("Verifica Predizioni")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nnunet_results = os.environ.get('nnUNet_results', os.path.join(base_dir, 'nnUNet_results'))
    
    results_dir = os.path.join(nnunet_results, f"Dataset{dataset_id:03d}_Shapes", 
                               f"nnUNetTrainer__nnUNetPlans__{config}", f"fold_{fold}")
    
    validation_dir = os.path.join(results_dir, 'validation')
    
    if os.path.exists(validation_dir):
        pred_files = [f for f in os.listdir(validation_dir) if f.endswith('.nii.gz')]
        print(f"\nPredizioni di validazione trovate: {len(pred_files)}")
        
        if len(pred_files) > 0:
            print(f"\nAnalisi prima predizione: {pred_files[0]}")
            pred_path = os.path.join(validation_dir, pred_files[0])
            pred_data = load_nifti(pred_path)
            
            print(f"  Shape: {pred_data.shape}")
            print(f"  Unique values: {np.unique(pred_data)}")
            print(f"  Sum: {pred_data.sum()}")
            print(f"  Min: {pred_data.min()}, Max: {pred_data.max()}")
            
            # Verifica se è tutto zero o tutto uno
            if pred_data.sum() == 0:
                print(f"  ⚠️  Predizione completamente vuota!")
            elif (pred_data > 0).sum() == pred_data.size:
                print(f"  ⚠️  Predizione completamente piena!")
            else:
                print(f"  ✅ Predizione sembra ragionevole")
    else:
        print(f"\nNessuna predizione trovata in: {validation_dir}")
        print(f"Esegui prima il test con: python test_geometric.py -d {dataset_id} -c {config} -f {fold}")

def check_training_log(dataset_id=501, config='2d', fold=0):
    """
    Controlla il log di training per vedere i Dice scores.
    """
    print("\n" + "=" * 60)
    print("Verifica Log di Training")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nnunet_results = os.environ.get('nnUNet_results', os.path.join(base_dir, 'nnUNet_results'))
    
    results_dir = os.path.join(nnunet_results, f"Dataset{dataset_id:03d}_Shapes", 
                               f"nnUNetTrainer__nnUNetPlans__{config}", f"fold_{fold}")
    
    # Cerca log file
    import glob
    log_files = glob.glob(os.path.join(results_dir, 'training_log_*.txt'))
    
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"\nLeggendo log: {os.path.basename(latest_log)}")
        
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        # Cerca Dice scores
        dice_scores = []
        epochs = []
        
        for line in lines[-100:]:  # Ultime 100 righe
            if 'Pseudo dice' in line or 'dice' in line.lower():
                # Cerca numeri nella riga
                import re
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    try:
                        dice = float(numbers[0])
                        if 0 <= dice <= 1:
                            dice_scores.append(dice)
                    except:
                        pass
        
        if dice_scores:
            print(f"\nDice scores trovati (ultimi {len(dice_scores)}):")
            print(f"  Min: {min(dice_scores):.4f}")
            print(f"  Max: {max(dice_scores):.4f}")
            print(f"  Mean: {np.mean(dice_scores):.4f}")
            print(f"  Last: {dice_scores[-1]:.4f}")
            
            if max(dice_scores) > 0.99:
                print(f"\n⚠️  ATTENZIONE: Dice score molto alto (>0.99)")
                print(f"   Questo potrebbe indicare:")
                print(f"   1. Il problema è troppo semplice (normale per dataset geometrico)")
                print(f"   2. Data leakage (stesso caso in train e validation)")
                print(f"   3. Problema nel calcolo delle metriche")
        else:
            print(f"\nNessun Dice score trovato nel log")
    else:
        print(f"\nNessun log file trovato")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verifica integrità dataset e risultati')
    parser.add_argument('-d', '--dataset', type=int, default=501, help='Dataset ID')
    parser.add_argument('-c', '--config', type=str, default='2d', help='Configuration')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Fold number')
    
    args = parser.parse_args()
    
    # Verifica dataset
    dataset_ok = check_dataset_integrity(args.dataset)
    
    # Verifica predizioni
    check_predictions(args.dataset, args.config, args.fold)
    
    # Verifica log
    check_training_log(args.dataset, args.config, args.fold)
    
    print("\n" + "=" * 60)
    print("Verifica Completata")
    print("=" * 60)
    print("\nPer testare il modello attuale:")
    print(f"  python test_geometric.py -d {args.dataset} -c {args.config} -f {args.fold}")

