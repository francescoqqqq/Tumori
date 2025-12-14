"""
Script per eseguire SOLO inference nnU-Net (una tantum).

Esegue inference per baseline e/o geometric, salvando predizioni.
Poi test.py può usare le predizioni già pronte.

Usage:
    python run_inference.py --baseline    # Solo baseline
    python run_inference.py --geometric   # Solo geometric
    python run_inference.py --both        # Entrambi (default)

Author: Francesco + Claude
Date: 2025-12-05
"""
import os
import sys
import subprocess
import argparse
import re
import torch  # pyright: ignore[reportMissingImports]

CONFIG = '2d'
FOLD = 0

TRAINERS = {
    'baseline': 'nnUNetTrainer250Epochs',
    'geometric': 'nnUNetTrainerGeometric'
}


def print_header(text):
    """Stampa header formattato."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_option(number, text, description=""):
    """Stampa opzione numerata."""
    print(f"  [{number}] {text}")
    if description:
        print(f"      {description}")


def get_available_datasets():
    """Scansiona nnUNet_preprocessed per trovare i dataset disponibili."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.join(base_dir, 'nnUNet_preprocessed')
    
    if not os.path.exists(preprocessed_dir):
        return []
    
    datasets = []
    for item in os.listdir(preprocessed_dir):
        item_path = os.path.join(preprocessed_dir, item)
        if os.path.isdir(item_path) and item.startswith('Dataset'):
            # Estrai l'ID numerico dal nome (es. "Dataset501_Shapes" -> 501)
            match = re.match(r'Dataset(\d+)_', item)
            if match:
                dataset_id = int(match.group(1))
                datasets.append((dataset_id, item))
    
    # Ordina per ID
    datasets.sort(key=lambda x: x[0])
    return datasets


def get_dataset_choice():
    """Chiede all'utente quale dataset usare per l'inference."""
    print_header("SCELTA DATASET")
    
    datasets = get_available_datasets()
    
    if not datasets:
        print("❌ ERRORE: Nessun dataset trovato in nnUNet_preprocessed!")
        print(f"   Directory: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nnUNet_preprocessed')}")
        print("\n   Assicurati di aver eseguito il preprocessing dei dataset.")
        return None
    
    print(f"Trovati {len(datasets)} dataset(s) preprocessati:\n")
    
    for idx, (dataset_id, dataset_name) in enumerate(datasets, 1):
        print_option(idx, f"Dataset{dataset_id}", f"{dataset_name}")
    
    while True:
        try:
            choice = input(f"\nScegli il dataset (1-{len(datasets)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(datasets):
                selected_id, selected_name = datasets[choice_num - 1]
                print(f"\n✅ Dataset selezionato: Dataset{selected_id} ({selected_name})")
                return selected_id, selected_name
            else:
                print(f"Scelta non valida! Inserisci un numero tra 1 e {len(datasets)}.")
        except ValueError:
            print("Input non valido! Inserisci un numero.")


def get_model_choice():
    """Chiede all'utente quale modello usare per l'inference."""
    print_header("SCELTA MODELLO")
    
    print_option(1, "Baseline",
                 "Modello baseline (nnU-Net standard)")
    print_option(2, "Geometric",
                 "Modello geometric (nnU-Net + Loss Geometriche)")
    print_option(3, "Entrambi",
                 "Esegui inference per baseline e geometric")
    
    while True:
        choice = input("\nScegli il modello (1/2/3): ").strip()
        if choice == "1":
            return ['baseline']
        elif choice == "2":
            return ['geometric']
        elif choice == "3":
            return ['baseline', 'geometric']
        else:
            print("Scelta non valida! Riprova.")


def check_checkpoint_weights(trainer_name, base_dir, dataset_id):
    """
    Verifica che i pesi del checkpoint non siano NaN/Inf.

    Returns:
        True se i pesi sono OK, False se ci sono NaN/Inf
    """
    print(f"\n🔍 Verifica integrità checkpoint per {trainer_name}...")

    # Costruisci il path del checkpoint
    # Trova il nome completo del dataset
    results_dir = os.path.join(base_dir, 'nnUNet_results')
    if not os.path.exists(results_dir):
        print(f"❌ Directory nnUNet_results non trovata: {results_dir}")
        print(f"   Esegui prima il training con: python train.py")
        return False
    
    dataset_name = None
    for item in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, item)) and item.startswith(f'Dataset{dataset_id:03d}_'):
            dataset_name = item
            break
    
    if not dataset_name:
        print(f"❌ Dataset {dataset_id} non trovato in nnUNet_results")
        print(f"   Directory: {results_dir}")
        print(f"   Esegui prima il training per questo dataset con: python train.py")
        return False
    
    checkpoint_dir = os.path.join(
        base_dir, 'nnUNet_results',
        dataset_name,
        f'{trainer_name}__nnUNetPlans__2d',
        f'fold_{FOLD}'
    )

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')

    # Verifica che il checkpoint esista
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint non trovato: {checkpoint_path}")
        return False

    print(f"   Path: {checkpoint_path}")

    # Carica il checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'network_weights' not in checkpoint:
            print(f"❌ Checkpoint non contiene 'network_weights'")
            return False

        weights = checkpoint['network_weights']

        # Controlla il primo tensore per NaN/Inf
        has_nan = False
        has_inf = False

        for key, tensor in weights.items():
            if torch.isnan(tensor).any():
                has_nan = True
                print(f"   ❌ NaN trovato in: {key}")
                break
            if torch.isinf(tensor).any():
                has_inf = True
                print(f"   ❌ Inf trovato in: {key}")
                break

        if has_nan or has_inf:
            print(f"\n❌ ATTENZIONE: Il modello {trainer_name} ha pesi corrotti!")
            print(f"   Il modello è stato allenato con problemi numerici.")
            print(f"   Le predizioni saranno tutte zero/invalide.")
            print(f"\n   Soluzione: Ri-allena il modello con:")
            print(f"     python train.py")
            return False

        # Se arriviamo qui, i pesi sono OK
        print(f"   ✅ Checkpoint OK - Nessun NaN/Inf rilevato")

        # Opzionale: mostra alcune statistiche
        first_key = list(weights.keys())[0]
        first_tensor = weights[first_key]
        print(f"   📊 Stats primo layer ({first_key}):")
        print(f"      Mean: {first_tensor.mean().item():.6f}")
        print(f"      Std:  {first_tensor.std().item():.6f}")
        print(f"      Min:  {first_tensor.min().item():.6f}")
        print(f"      Max:  {first_tensor.max().item():.6f}")

        return True

    except Exception as e:
        print(f"❌ Errore durante verifica checkpoint: {e}")
        return False


def run_inference(trainer_name, model_type, dataset_id, dataset_name):
    """Esegue inference per un trainer specifico."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup environment
    os.environ['nnUNet_raw'] = os.path.join(base_dir, 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = os.path.join(base_dir, 'nnUNet_results')

    # VERIFICA CHECKPOINT PRIMA DI PROCEDERE
    if not check_checkpoint_weights(trainer_name, base_dir, dataset_id):
        print(f"\n❌ Checkpoint non valido per {trainer_name}. Inference annullata.")
        return False

    # Output directory: predictions/{dataset_id}{model_type}/
    predictions_base = os.path.join(base_dir, 'predictions')
    os.makedirs(predictions_base, exist_ok=True)
    predictions_dir = os.path.join(predictions_base, f'{dataset_id}{model_type}')
    os.makedirs(predictions_dir, exist_ok=True)

    # Input directory (validation set)
    # nnU-Net inference si aspetta le immagini in un folder specifico
    # Usiamo il validation set dal preprocessed
    preprocessed_base = os.path.join(os.environ['nnUNet_preprocessed'], dataset_name)

    # Verifica che esista il preprocessed
    if not os.path.exists(preprocessed_base):
        print(f"❌ Dataset preprocessato non trovato: {preprocessed_base}")
        return False

    # Per nnUNet_predict, dobbiamo usare le immagini raw del validation set
    # Creiamo un folder temporaneo con le immagini del validation set
    import tempfile
    import shutil
    import json

    temp_input_dir = tempfile.mkdtemp(prefix='nnunet_inference_')

    try:
        # Carica splits per sapere quali sono le immagini di validazione
        splits_file = os.path.join(preprocessed_base, 'splits_final.json')
        with open(splits_file, 'r') as f:
            splits = json.load(f)

        val_cases = splits[FOLD]['val']

        print(f"\n📋 Validation set: {len(val_cases)} casi")
        print(f"   Esempi: {val_cases[:5]}")

        # Copia immagini validation nel temp folder
        # Trova il nome completo del dataset in nnUNet_raw
        raw_base = os.environ['nnUNet_raw']
        if not os.path.exists(raw_base):
            print(f"❌ Directory nnUNet_raw non trovata: {raw_base}")
            return False
        
        raw_dataset_name = None
        for item in os.listdir(raw_base):
            if os.path.isdir(os.path.join(raw_base, item)) and item.startswith(f'Dataset{dataset_id:03d}_'):
                raw_dataset_name = item
                break
        
        if not raw_dataset_name:
            print(f"❌ Dataset {dataset_id} non trovato in nnUNet_raw")
            print(f"   Directory: {raw_base}")
            return False
        
        raw_images_dir = os.path.join(raw_base, raw_dataset_name, 'imagesTr')  # Tutti i case sono in Tr

        print(f"\n📁 Copiando immagini validation in folder temporaneo...")
        for case_id in val_cases:
            src = os.path.join(raw_images_dir, f'{case_id}_0000.nii.gz')
            if os.path.exists(src):
                shutil.copy(src, temp_input_dir)
            else:
                print(f"⚠️  Immagine non trovata: {src}")

        print(f"✅ Copiate {len(os.listdir(temp_input_dir))} immagini")

        # Comando inference
        cmd = [
            'nnUNetv2_predict',
            '-i', temp_input_dir,
            '-o', predictions_dir,
            '-d', str(dataset_id),
            '-c', CONFIG,
            '-f', str(FOLD),
            '-tr', trainer_name,
            '--disable_tta'  # Disabilita test-time augmentation per velocità
        ]

        print(f"\n🚀 Esecuzione inference per {trainer_name}...")
        print(f"   Dataset: {dataset_id} ({dataset_name})")
        print(f"   Input: {temp_input_dir}")
        print(f"   Output: {predictions_dir}")
        print(f"   Comando: {' '.join(cmd)}\n")
        print("=" * 80)

        result = subprocess.run(cmd, cwd=base_dir)

        if result.returncode != 0:
            print(f"\n❌ Errore durante inference di {trainer_name}")
            return False

        print("\n" + "=" * 80)
        print(f"✅ Inference {trainer_name} completata!")
        print(f"   Dataset: {dataset_id} ({dataset_name})")
        print(f"   Predizioni salvate in: {predictions_dir}")
        num_files = len([f for f in os.listdir(predictions_dir) if f.endswith('.nii.gz')])
        print(f"   Numero files: {num_files}")

        return True

    finally:
        # Pulizia folder temporaneo
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
            print(f"\n🧹 Folder temporaneo rimosso: {temp_input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Esegui inference nnU-Net')
    parser.add_argument('--baseline', action='store_true',
                       help='Esegui inference solo per baseline (non interattivo)')
    parser.add_argument('--geometric', action='store_true',
                       help='Esegui inference solo per geometric (non interattivo)')
    parser.add_argument('--both', action='store_true',
                       help='Esegui inference per entrambi (non interattivo)')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Usa argomenti da linea di comando invece di menu interattivo')

    args = parser.parse_args()

    print_header("INFERENCE nnU-Net - Baseline vs Geometric")

    # Setup environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['nnUNet_raw'] = os.path.join(base_dir, 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = os.path.join(base_dir, 'nnUNet_results')

    # Scelta dataset
    dataset_choice = get_dataset_choice()
    if dataset_choice is None:
        print("\n❌ Impossibile continuare senza un dataset valido.")
        return 1
    dataset_id, dataset_name = dataset_choice

    # Scelta modello (interattivo o da argomenti)
    if args.non_interactive or args.baseline or args.geometric or args.both:
        # Modalità non interattiva: usa argomenti da linea di comando
        models_to_run = []
        if args.baseline:
            models_to_run.append('baseline')
        if args.geometric:
            models_to_run.append('geometric')
        if args.both or (not args.baseline and not args.geometric):
            models_to_run = ['baseline', 'geometric']
    else:
        # Modalità interattiva: chiedi all'utente
        models_to_run = get_model_choice()

    # Mostra riepilogo
    print_header("RIEPILOGO CONFIGURAZIONE")
    print(f"  Dataset:       {dataset_id} ({dataset_name})")
    print(f"  Configurazione: {CONFIG}")
    print(f"  Fold:          {FOLD}")
    print(f"  Modelli da eseguire:")
    for model in models_to_run:
        print(f"    - {model.capitalize()}")
    print(f"\n  Output:")
    for model in models_to_run:
        print(f"    - predictions/{dataset_id}{model}/")
    print()

    success = True

    if 'baseline' in models_to_run:
        print("\n" + "🔵 " * 40)
        print("BASELINE INFERENCE")
        print("🔵 " * 40)
        if not run_inference(TRAINERS['baseline'], 'baseline', dataset_id, dataset_name):
            success = False

    if 'geometric' in models_to_run:
        print("\n" + "🟢 " * 40)
        print("GEOMETRIC INFERENCE")
        print("🟢 " * 40)
        if not run_inference(TRAINERS['geometric'], 'geometric', dataset_id, dataset_name):
            success = False

    print("\n" + "=" * 80)
    if success:
        print("✅ INFERENCE COMPLETATA CON SUCCESSO!")
        print("=" * 80)
        print(f"\nPredizioni salvate in:")
        predictions_base = os.path.join(base_dir, 'predictions')
        for model in models_to_run:
            print(f"  - {os.path.join(predictions_base, f'{dataset_id}{model}')}")
        print("\nProssimo step:")
        print("  python test.py")
        print("\nQuesto script caricherà le predizioni già pronte e genererà:")
        print("  - Visualizzazioni")
        print("  - Metriche")
        print("  - Confronti")
    else:
        print("❌ INFERENCE FALLITA")
        print("=" * 80)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
