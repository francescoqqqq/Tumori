#!/usr/bin/env python3
"""
Script interattivo per training nnU-Net.

Permette di scegliere:
1. Tipo di rete (Baseline o Geometric)
2. Numero di epoche

Author: Francesco + Claude
Date: 2025-12-10
"""

import os
import sys
import subprocess
import re


def print_header(text):
    """Stampa header colorato."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_option(number, text, description=""):
    """Stampa opzione numerata."""
    print(f"  [{number}] {text}")
    if description:
        print(f"      {description}")


def setup_environment():
    """Configura variabili d'ambiente nnU-Net."""
    base_path = "/workspace/geometrica"
    os.environ['nnUNet_raw'] = f"{base_path}/nnUNet_raw"
    os.environ['nnUNet_preprocessed'] = f"{base_path}/nnUNet_preprocessed"
    os.environ['nnUNet_results'] = f"{base_path}/nnUNet_results"

    print("\nVariabili d'ambiente configurate:")
    print(f"  nnUNet_raw          = {os.environ['nnUNet_raw']}")
    print(f"  nnUNet_preprocessed = {os.environ['nnUNet_preprocessed']}")
    print(f"  nnUNet_results      = {os.environ['nnUNet_results']}")


def get_available_datasets():
    """Scansiona nnUNet_preprocessed per trovare i dataset disponibili."""
    preprocessed_dir = os.environ.get('nnUNet_preprocessed', '')
    
    if not preprocessed_dir or not os.path.exists(preprocessed_dir):
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
    """Chiede all'utente quale dataset usare per il training."""
    print_header("SCELTA DATASET")
    
    datasets = get_available_datasets()
    
    if not datasets:
        print("❌ ERRORE: Nessun dataset trovato in nnUNet_preprocessed!")
        print(f"   Directory: {os.environ.get('nnUNet_preprocessed', 'NON CONFIGURATA')}")
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


def get_trainer_choice():
    """Chiede all'utente quale tipo di rete allenare."""
    print_header("SCELTA TIPO DI RETE")

    print_option(1, "Baseline (nnU-Net standard)",
                 "Solo Dice + Cross-Entropy loss")
    print_option(2, "Geometric (nnU-Net + Loss Geometriche)",
                 "Dice + CE + Compactness + Solidity + Eccentricity + Boundary")

    while True:
        choice = input("\nScegli il tipo di rete (1/2): ").strip()
        if choice == "1":
            return "baseline", "nnUNetTrainer250Epochs"
        elif choice == "2":
            return "geometric", "nnUNetTrainerGeometric"
        else:
            print("Scelta non valida! Riprova.")


def get_epochs():
    """Chiede all'utente quante epoche vuole fare."""
    print_header("NUMERO EPOCHE")

    print("Epoche consigliate:")
    print("  - Baseline:  100-250 epoche")
    print("  - Geometric: 100 epoche (20 warm-up + 80 geometric)")

    while True:
        try:
            epochs_input = input("\nQuante epoche vuoi fare? [default: 100]: ").strip()
            if not epochs_input:
                return 100
            epochs = int(epochs_input)
            if epochs < 1:
                print("Numero di epoche deve essere positivo!")
                continue
            if epochs > 1000:
                confirm = input(f"{epochs} epoche sono tante! Sei sicuro? (s/n): ").strip().lower()
                if confirm != 's':
                    continue
            return epochs
        except ValueError:
            print("Input non valido! Inserisci un numero intero.")


def show_summary(network_type, trainer_name, epochs, dataset_id, dataset_name):
    """Mostra riepilogo delle scelte."""
    print_header("RIEPILOGO CONFIGURAZIONE")

    print(f"  Tipo rete:     {network_type.upper()}")
    print(f"  Trainer:       {trainer_name}")
    print(f"  Epoche:        {epochs}")
    print(f"  Dataset:       {dataset_id} ({dataset_name})")
    print(f"  Configurazione: 2d")
    print(f"  Fold:          0")

    if network_type == "geometric":
        print(f"\n  Loss Geometriche:")
        print(f"    - Compactness:    0.01")
        print(f"    - Solidity:       0.01")
        print(f"    - Eccentricity:   0.005")
        print(f"    - Boundary:       0.005")
        print(f"    - Warm-up:        20 epoche")
        print(f"    - Con Geometric:  {epochs-20} epoche")

    print()


def confirm_training():
    """Chiede conferma prima di avviare training."""
    while True:
        confirm = input("Vuoi avviare il training? (s/n): ").strip().lower()
        if confirm == 's':
            return True
        elif confirm == 'n':
            return False
        else:
            print("Risposta non valida! Usa 's' per sì o 'n' per no.")


def update_trainer_epochs(trainer_name, epochs):
    """Aggiorna il numero di epoche nel file trainer."""
    # Trova il file trainer
    trainer_files = {
        "nnUNetTrainer250Epochs": "/workspace/geometrica/nnUNetTrainer250Epochs.py",
        "nnUNetTrainerGeometric": "/workspace/geometrica/nnUNetTrainerGeometric.py"
    }

    trainer_path = trainer_files.get(trainer_name)
    if not trainer_path or not os.path.exists(trainer_path):
        print(f"\n⚠️  WARNING: File trainer non trovato: {trainer_path}")
        print(f"   Il training userà il numero di epoche di default del trainer.")
        return False

    # Leggi il file
    with open(trainer_path, 'r') as f:
        content = f.read()

    # Aggiorna self.num_epochs
    pattern = r'self\.num_epochs\s*=\s*\d+'
    replacement = f'self.num_epochs = {epochs}'

    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        with open(trainer_path, 'w') as f:
            f.write(new_content)
        print(f"\n✅ Trainer aggiornato: {epochs} epoche")
        return True
    else:
        print(f"\n⚠️  WARNING: Non ho trovato 'self.num_epochs' nel trainer")
        print(f"   Il training userà il numero di epoche di default.")
        return False


def run_training(trainer_name, dataset_id):
    """Esegue il comando di training nnU-Net."""
    print_header("AVVIO TRAINING")

    # Comando training
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),  # dataset
        "2d",   # configuration
        "0",    # fold
        "-tr", trainer_name  # trainer
    ]

    print(f"Comando: {' '.join(cmd)}\n")
    print(f"Il training è iniziato...")
    print(f"Premi Ctrl+C per interrompere\n")
    print("="*70)

    try:
        # Esegui comando
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("✅ Training completato con successo!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print(f"❌ Training fallito con errore: {e}")
        return False
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("⚠️  Training interrotto dall'utente")
        return False


def main():
    """Main function."""
    print_header("TRAINING nnU-Net - Script Interattivo")

    print("Questo script ti permette di allenare una rete nnU-Net")
    print("scegliendo il dataset, il tipo di rete e il numero di epoche.")

    # Setup ambiente
    setup_environment()

    # Scelta dataset
    dataset_choice = get_dataset_choice()
    if dataset_choice is None:
        print("\n❌ Impossibile continuare senza un dataset valido.")
        return 1
    dataset_id, dataset_name = dataset_choice

    # Scelta tipo rete
    network_type, trainer_name = get_trainer_choice()

    # Scelta epoche
    epochs = get_epochs()

    # Aggiorna trainer con epoche scelte
    update_trainer_epochs(trainer_name, epochs)

    # Mostra riepilogo
    show_summary(network_type, trainer_name, epochs, dataset_id, dataset_name)

    # Conferma
    if not confirm_training():
        print("\n❌ Training annullato dall'utente.")
        return 1

    # Esegui training
    success = run_training(trainer_name, dataset_id)

    if success:
        print_header("PROSSIMI PASSI")
        print("1. Esegui inference con: python run_inference.py --both")
        print("2. Analizza risultati con: python test.py")
        print()
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
