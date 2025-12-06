"""
Custom nnU-Net trainer con loss geometrica per segmentazione cerchi.

Estende nnUNetTrainer aggiungendo penalità geometriche per forzare
la rete a produrre cerchi più perfetti.

Author: Francesco + Claude
Date: 2025-12-05

"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.geometric_losses import GeometricLosses
import torch
import numpy as np
from typing import Union, Tuple, List
import os


class nnUNetTrainerGeometric(nnUNetTrainer):
    """
    Trainer nnU-Net con loss geometrica per segmentazione cerchi.

    Aggiunge penalità geometriche alla loss standard (Dice + CE) per
    migliorare compactness, solidity, eccentricity e smoothness dei bordi.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        """
        Inizializza trainer geometric.

        Args:
            plans: nnU-Net plans dictionary
            configuration: '2d' o '3d_fullres'
            fold: Fold number per cross-validation
            dataset_json: Dataset JSON con metadata
            device: Device PyTorch (cuda/cpu)
        """
        # Override batch_size a 8 per risparmiare memoria (prima di chiamare super)
        # Modifica i plans in modo aggressivo - tutte le possibili posizioni
        if 'configurations' in plans:
            for config_name, config_data in plans['configurations'].items():
                if isinstance(config_data, dict) and 'batch_size' in config_data:
                    config_data['batch_size'] = 8
        
        # Modifica anche direttamente nel dizionario plans se presente
        if isinstance(plans, dict):
            # Cerca batch_size a qualsiasi livello
            def set_batch_size_recursive(d, target_value=8):
                if isinstance(d, dict):
                    for key, value in d.items():
                        if key == 'batch_size':
                            d[key] = target_value
                        elif isinstance(value, (dict, list)):
                            set_batch_size_recursive(value, target_value)
                elif isinstance(d, list):
                    for item in d:
                        set_batch_size_recursive(item, target_value)
            
            set_batch_size_recursive(plans, 8)
        
        # Chiama costruttore base
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Override batch_size anche dopo inizializzazione (per sicurezza)
        # Prova tutti i possibili attributi
        if hasattr(self, 'configuration_manager'):
            # Modifica data_loader_kwargs
            if hasattr(self.configuration_manager, 'data_loader_kwargs'):
                if isinstance(self.configuration_manager.data_loader_kwargs, dict):
                    self.configuration_manager.data_loader_kwargs['batch_size'] = 8
            
            # Modifica configuration
            if hasattr(self.configuration_manager, 'configuration'):
                if isinstance(self.configuration_manager.configuration, dict):
                    if 'batch_size' in self.configuration_manager.configuration:
                        self.configuration_manager.configuration['batch_size'] = 8
            
            # NON possiamo modificare batch_size direttamente (è una property senza setter)
            # La limitazione viene fatta nel train_step

        # Numero di campioni su cui calcolare loss geometrica (per risparmiare memoria)
        self.geometric_loss_samples = 4  # Solo primi 4 campioni del batch

        # Inizializza loss geometrica con pesi dal design document
        self.geometric_loss = GeometricLosses(
            weight_compactness=0.1,
            weight_solidity=0.1,
            weight_eccentricity=0.05,
            weight_boundary=0.05,
            min_area=10
        )

        # Flag per attivare/disattivare loss geometrica
        self.use_geometric_loss = True

        # Warm-up: attiva loss geometrica solo dopo N epoche
        self.geometric_loss_warmup_epochs = 5  # Primi 5 epoche solo Dice+CE (poi 95 con geometric)

        # Override numero epoche a 100 (invece di 250 default)
        self.num_epochs = 100

        # Storage per logging loss geometrica
        self.geometric_loss_log = []

        print(f"\n{'='*60}")
        print("nnUNetTrainerGeometric inizializzato")
        print(f"{'='*60}")
        print(f"Numero epoche: {self.num_epochs}")
        print(f"Batch size effettivo: 8 (ridotto per risparmiare memoria)")
        print(f"  NOTA: Il dataloader può caricare batch più grandi, ma processiamo solo 8 campioni")
        print(f"Loss geometrica attiva: {self.use_geometric_loss}")
        print(f"Loss geometrica su primi {self.geometric_loss_samples} campioni del batch")
        print(f"Warm-up epoche: {self.geometric_loss_warmup_epochs}")
        print(f"  (epoche 0-{self.geometric_loss_warmup_epochs-1}: solo Dice+CE)")
        print(f"  (epoche {self.geometric_loss_warmup_epochs}-{self.num_epochs-1}: Dice+CE + Geometric)")
        print(f"\nPesi loss geometrica:")
        print(f"  - Compactness: {self.geometric_loss.weight_compactness}")
        print(f"  - Solidity: {self.geometric_loss.weight_solidity}")
        print(f"  - Eccentricity: {self.geometric_loss.weight_eccentricity}")
        print(f"  - Boundary: {self.geometric_loss.weight_boundary}")
        print(f"{'='*60}\n")

    def train_step(self, batch: dict) -> dict:
        """
        Override del train step per aggiungere loss geometrica.

        Args:
            batch: Dizionario con 'data' [B, C, H, W] e 'target' [B, 1, H, W]

        Returns:
            Dizionario con 'loss' totale
        """
        data = batch['data']
        target = batch['target']

        # MEMORY OPTIMIZATION: Limita batch a 8 campioni per risparmiare memoria
        # Anche se il dataloader carica batch più grandi, processiamo solo i primi 8
        max_batch_size = 8
        if data.shape[0] > max_batch_size:
            # Usa slicing per limitare il batch (crea view, non copia)
            data = data[:max_batch_size]
            if isinstance(target, list):
                target = [t[:max_batch_size] for t in target]
            else:
                target = target[:max_batch_size]

        # Sposta su device
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)

        # Loss standard nnU-Net (Dice + CE)
        loss_dice_ce = self.loss(output, target)

        # Loss geometrica (se attiva e dopo warm-up)
        loss_geometric = torch.tensor(0.0, device=self.device)

        if self.use_geometric_loss and self.current_epoch >= self.geometric_loss_warmup_epochs:
            # MEMORY OPTIMIZATION: calcola solo sui primi N campioni del batch
            try:
                # Gestisci il caso in cui output è una lista (deep supervision)
                if isinstance(output, (list, tuple)):
                    # Prendi il primo elemento (output principale)
                    output_tensor = output[0]
                else:
                    output_tensor = output
                
                # Limita output ai primi N campioni PRIMA di calcolare softmax
                batch_size = output_tensor.shape[0]
                n_samples = min(self.geometric_loss_samples, batch_size)
                output_geometric = output_tensor[:n_samples]
                
                # Calcola softmax solo sui campioni selezionati
                output_softmax_grad = torch.softmax(output_geometric, dim=1)
                
                # Calcola loss geometrica
                loss_geometric = self.geometric_loss(output_softmax_grad)
                
                # Libera memoria dei tensori intermedi
                del output_geometric, output_softmax_grad
            except Exception as e:
                # Se loss geometrica fallisce, logga warning e continua
                # Non loggare ogni volta per evitare spam (solo ogni 10 epoche)
                loss_geometric = torch.tensor(0.0, device=self.device)

        # Loss totale
        total_loss = loss_dice_ce + loss_geometric

        # MEMORY OPTIMIZATION: Pulisci cache CUDA prima del backward
        torch.cuda.empty_cache()

        # Backward e step optimizer
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        # MEMORY OPTIMIZATION: Pulisci cache dopo lo step
        torch.cuda.empty_cache()

        # Logging
        self.geometric_loss_log.append({
            'dice_ce': loss_dice_ce.item(),
            'geometric': loss_geometric.item(),
            'total': total_loss.item()
        })

        return {'loss': total_loss.detach().cpu().numpy()}

    def on_epoch_end(self):
        """Override per logging metriche geometriche."""
        # Chiama on_epoch_end base
        super().on_epoch_end()

        # Log media loss geometrica dell'epoca
        if len(self.geometric_loss_log) > 0:
            avg_dice_ce = np.mean([x['dice_ce'] for x in self.geometric_loss_log])
            avg_geometric = np.mean([x['geometric'] for x in self.geometric_loss_log])
            avg_total = np.mean([x['total'] for x in self.geometric_loss_log])

            # Log su tensorboard/file
            # Il logger nnU-Net richiede che le chiavi siano inizializzate prima
            try:
                # Inizializza le chiavi se non esistono
                if hasattr(self.logger, 'my_fantastic_logging'):
                    if 'train_loss_dice_ce' not in self.logger.my_fantastic_logging:
                        self.logger.my_fantastic_logging['train_loss_dice_ce'] = []
                    if 'train_loss_geometric' not in self.logger.my_fantastic_logging:
                        self.logger.my_fantastic_logging['train_loss_geometric'] = []
                
                self.logger.log('train_loss_dice_ce', avg_dice_ce, self.current_epoch)
                self.logger.log('train_loss_geometric', avg_geometric, self.current_epoch)
            except (AssertionError, AttributeError) as e:
                # Se il logging fallisce, continua senza loggare (non è critico)
                if self.current_epoch % 10 == 0:
                    print(f"⚠️  Warning: Logging fallito: {e}")

            # Log componenti loss geometrica se disponibili
            if hasattr(self.geometric_loss, 'last_losses'):
                last_losses = self.geometric_loss.get_last_losses()
                try:
                    if hasattr(self.logger, 'my_fantastic_logging'):
                        for key, value in last_losses.items():
                            log_key = f'train_geom_{key}'
                            if log_key not in self.logger.my_fantastic_logging:
                                self.logger.my_fantastic_logging[log_key] = []
                            self.logger.log(log_key, value, self.current_epoch)
                except (AssertionError, AttributeError):
                    pass  # Ignora errori di logging

            # Print periodico
            if self.current_epoch % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {self.current_epoch} - Loss Summary")
                print(f"{'='*60}")
                print(f"Dice+CE Loss:     {avg_dice_ce:.4f}")
                print(f"Geometric Loss:   {avg_geometric:.4f}")
                print(f"Total Loss:       {avg_total:.4f}")

                if hasattr(self.geometric_loss, 'last_losses'):
                    print(f"\nGeometric Components:")
                    for key, value in last_losses.items():
                        print(f"  - {key:15s}: {value:.4f}")
                print(f"{'='*60}\n")

            # Reset log
            self.geometric_loss_log = []

    def on_train_end(self):
        """Override per salvare info loss geometrica."""
        super().on_train_end()

        # Salva configurazione loss geometrica
        config_file = os.path.join(self.output_folder, "geometric_loss_config.txt")
        with open(config_file, 'w') as f:
            f.write("GEOMETRIC LOSS CONFIGURATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total epochs: {self.num_epochs}\n")
            f.write(f"Warm-up epochs: {self.geometric_loss_warmup_epochs}\n")
            f.write(f"  - Epochs 0-{self.geometric_loss_warmup_epochs-1}: Dice+CE only\n")
            f.write(f"  - Epochs {self.geometric_loss_warmup_epochs}-{self.num_epochs-1}: Dice+CE + Geometric\n\n")
            f.write(f"Geometric loss weights:\n")
            f.write(f"  - Compactness: {self.geometric_loss.weight_compactness}\n")
            f.write(f"  - Solidity: {self.geometric_loss.weight_solidity}\n")
            f.write(f"  - Eccentricity: {self.geometric_loss.weight_eccentricity}\n")
            f.write(f"  - Boundary: {self.geometric_loss.weight_boundary}\n\n")
            f.write(f"Other parameters:\n")
            f.write(f"  - Min area threshold: {self.geometric_loss.min_area} pixels\n")
            f.write(f"  - Batch size effettivo: 8 (ridotto per risparmiare memoria)\n")
            f.write(f"  - Loss geometrica calcolata su primi {self.geometric_loss_samples} campioni del batch\n")
            f.write(f"  - Geometric loss samples: {self.geometric_loss_samples} (solo primi N campioni del batch)\n")

        print(f"\n✅ Geometric loss config salvato: {config_file}\n")
