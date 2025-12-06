"""
Script per eseguire SOLO inference nnU-Net (una tantum).

Esegue inference per baseline e/o geometric, salvando predizioni.
Poi test_interactive.py può usare le predizioni già pronte.

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

DATASET_ID = 501
CONFIG = '2d'
FOLD = 0

TRAINERS = {
    'baseline': 'nnUNetTrainer250Epochs',
    'geometric': 'nnUNetTrainerGeometric'
}


def run_inference(trainer_name, output_dir):
    """Esegue inference per un trainer specifico."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup environment
    os.environ['nnUNet_raw'] = os.path.join(base_dir, 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = os.path.join(base_dir, 'nnUNet_results')

    # Output directory
    predictions_dir = os.path.join(base_dir, output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Input directory (validation set)
    # nnU-Net inference si aspetta le immagini in un folder specifico
    # Usiamo il validation set dal preprocessed
    preprocessed_base = os.path.join(os.environ['nnUNet_preprocessed'],
                                     f'Dataset{DATASET_ID:03d}_Shapes')

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
        raw_images_dir = os.path.join(os.environ['nnUNet_raw'],
                                     f'Dataset{DATASET_ID:03d}_Shapes',
                                     'imagesTr')  # Tutti i case sono in Tr

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
            '-d', str(DATASET_ID),
            '-c', CONFIG,
            '-f', str(FOLD),
            '-tr', trainer_name,
            '--disable_tta'  # Disabilita test-time augmentation per velocità
        ]

        print(f"\n🚀 Esecuzione inference per {trainer_name}...")
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
        print(f"   Predizioni salvate in: {predictions_dir}")
        print(f"   Numero files: {len([f for f in os.listdir(predictions_dir) if f.endswith('.nii.gz')])}")

        return True

    finally:
        # Pulizia folder temporaneo
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
            print(f"\n🧹 Folder temporaneo rimosso: {temp_input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Esegui inference nnU-Net')
    parser.add_argument('--baseline', action='store_true',
                       help='Esegui inference solo per baseline')
    parser.add_argument('--geometric', action='store_true',
                       help='Esegui inference solo per geometric')
    parser.add_argument('--both', action='store_true',
                       help='Esegui inference per entrambi (default)')

    args = parser.parse_args()

    # Default: entrambi
    if not (args.baseline or args.geometric):
        args.both = True

    print("=" * 80)
    print("INFERENCE nnU-Net - Baseline vs Geometric")
    print("=" * 80)

    success = True

    if args.baseline or args.both:
        print("\n" + "🔵 " * 40)
        print("BASELINE INFERENCE")
        print("🔵 " * 40)
        if not run_inference(TRAINERS['baseline'], 'baseline_results'):
            success = False

    if args.geometric or args.both:
        print("\n" + "🟢 " * 40)
        print("GEOMETRIC INFERENCE")
        print("🟢 " * 40)
        if not run_inference(TRAINERS['geometric'], 'geometric_results'):
            success = False

    print("\n" + "=" * 80)
    if success:
        print("✅ INFERENCE COMPLETATA CON SUCCESSO!")
        print("=" * 80)
        print("\nProssimo step:")
        print("  python test_interactive.py")
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
