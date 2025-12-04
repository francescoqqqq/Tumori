"""
Script per analizzare i risultati e verificare se ci sono problemi.
"""
import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_nifti(path):
    """Load NIfTI file."""
    return nib.load(path).get_fdata()

def check_data_leakage(dataset_id=501):
    """
    Verifica se ci sono casi che appaiono sia nel train che nel test.
    """
    print("=" * 60)
    print("Verifica Data Leakage")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed', os.path.join(base_dir, 'nnUNet_preprocessed'))
    
    dataset_name = f"Dataset{dataset_id:03d}_Shapes"
    splits_file = os.path.join(nnunet_preprocessed, dataset_name, 'splits_final.json')
    
    if not os.path.exists(splits_file):
        print("⚠️  File splits_final.json non trovato")
        return
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # nnU-Net usa una lista di fold
    if isinstance(splits, list) and len(splits) > 0:
        fold_0 = splits[0]
        train_cases = set(fold_0.get('train', []))
        val_cases = set(fold_0.get('val', []))
        
        print(f"\nTrain cases: {len(train_cases)}")
        print(f"Val cases: {len(val_cases)}")
        
        overlap = train_cases & val_cases
        if overlap:
            print(f"\n❌ PROBLEMA: {len(overlap)} casi in comune tra train e val!")
            print(f"   Casi: {sorted(overlap)}")
        else:
            print(f"\n✅ Nessun overlap tra train e validation")
        
        # Verifica test cases (se esistono)
        test_cases = set(fold_0.get('test', []))
        if test_cases:
            print(f"\nTest cases: {len(test_cases)}")
            train_test_overlap = train_cases & test_cases
            val_test_overlap = val_cases & test_cases
            
            if train_test_overlap:
                print(f"❌ PROBLEMA: {len(train_test_overlap)} casi in comune tra train e test!")
            if val_test_overlap:
                print(f"⚠️  {len(val_test_overlap)} casi in comune tra val e test (normale se usi val come test)")

def analyze_predictions(dataset_id=501):
    """
    Analizza le predizioni per vedere se sono realistiche.
    """
    print("\n" + "=" * 60)
    print("Analisi Predizioni")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'risultati')
    
    metrics_file = os.path.join(results_dir, 'metrics_summary.json')
    if not os.path.exists(metrics_file):
        print("⚠️  File metrics_summary.json non trovato")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    dices = [m['dice'] for m in metrics]
    ious = [m['iou'] for m in metrics]
    
    print(f"\nStatistiche Dice Score:")
    print(f"  Min: {min(dices):.6f}")
    print(f"  Max: {max(dices):.6f}")
    print(f"  Mean: {np.mean(dices):.6f}")
    print(f"  Std: {np.std(dices):.6f}")
    print(f"  Median: {np.median(dices):.6f}")
    
    # Conta casi perfetti
    perfect = sum(1 for d in dices if d >= 0.9999)
    very_high = sum(1 for d in dices if d >= 0.99)
    
    print(f"\nCasi con Dice >= 0.9999: {perfect}/{len(dices)} ({100*perfect/len(dices):.1f}%)")
    print(f"Casi con Dice >= 0.99: {very_high}/{len(dices)} ({100*very_high/len(dices):.1f}%)")
    
    if perfect == len(dices):
        print(f"\n⚠️  ATTENZIONE: Tutti i casi hanno Dice quasi perfetto!")
        print(f"   Questo potrebbe indicare:")
        print(f"   1. Il problema è troppo semplice (normale per dataset geometrico)")
        print(f"   2. Data leakage")
        print(f"   3. Il modello ha già imparato perfettamente")
    
    # Verifica alcuni casi specifici
    print(f"\nAnalisi casi specifici:")
    worst = min(metrics, key=lambda x: x['dice'])
    best = max(metrics, key=lambda x: x['dice'])
    
    print(f"  Caso peggiore: {worst['case_id']} - Dice: {worst['dice']:.6f}")
    print(f"  Caso migliore: {best['case_id']} - Dice: {best['dice']:.6f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizza risultati per problemi')
    parser.add_argument('-d', '--dataset', type=int, default=501, help='Dataset ID')
    
    args = parser.parse_args()
    
    check_data_leakage(args.dataset)
    analyze_predictions(args.dataset)
    
    print("\n" + "=" * 60)
    print("Raccomandazioni:")
    print("=" * 60)
    print("1. Controlla le visualizzazioni in risultati/visualizations/")
    print("2. Verifica che le predizioni siano realistiche")
    print("3. Se il problema è davvero semplice, un Dice alto è normale")
    print("4. Per test più difficili, prova ad aumentare la complessità del dataset")

