"""
Trainer baseline personalizzato per nnU-Net (Dice + CE, nessuna loss geometrica).

Risiede in geometrica/ e viene trovato a runtime da run_experiment.py tramite
un bootstrap che estende la ricerca dei trainer di nnunetv2 a questa cartella
(vedi NNUNET_BOOTSTRAP_TEMPLATE in run_experiment.py): non viene copiato ne'
installato dentro il package nnunetv2, quindi non serve alcun permesso di
scrittura su site-packages.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import os

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

# Configurazione centralizzata (stesso file usato da nnUNetTrainerGeometric):
# generata da run_experiment.py con i valori dell'esperimento corrente.
_TRAINER_CONFIG_LOADED = False
try:
    from geometric_config import NUM_EPOCHS, BATCH_SIZE
    _TRAINER_CONFIG_LOADED = True
except ImportError:
    # Fallback se geometric_config non e' disponibile
    NUM_EPOCHS = 100
    BATCH_SIZE = 8


_DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class nnUNetTrainerBaseline(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = _DEFAULT_DEVICE):
        # Override batch_size nei plans prima di chiamare super(), identico
        # al trainer geometrico, per garantire lo stesso batch a entrambe le reti
        # e rendere la comparazione scientificamente valida.
        if 'configurations' in plans:
            for config_data in plans['configurations'].values():
                if isinstance(config_data, dict) and 'batch_size' in config_data:
                    config_data['batch_size'] = BATCH_SIZE

        if isinstance(plans, dict):
            def _set_bs(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == 'batch_size':
                            d[k] = BATCH_SIZE
                        elif isinstance(v, (dict, list)):
                            _set_bs(v)
                elif isinstance(d, list):
                    for item in d:
                        _set_bs(item)
            _set_bs(plans)

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = NUM_EPOCHS

        # Override post-init (per sicurezza, come nel trainer geometrico)
        if hasattr(self, 'configuration_manager'):
            if hasattr(self.configuration_manager, 'data_loader_kwargs'):
                if isinstance(self.configuration_manager.data_loader_kwargs, dict):
                    self.configuration_manager.data_loader_kwargs['batch_size'] = BATCH_SIZE
            if hasattr(self.configuration_manager, 'configuration'):
                if isinstance(self.configuration_manager.configuration, dict):
                    if 'batch_size' in self.configuration_manager.configuration:
                        self.configuration_manager.configuration['batch_size'] = BATCH_SIZE

        print(f"\n{'='*60}")
        print("nnUNetTrainerBaseline")
        print(f"{'='*60}")
        print(f"Epoche:     {self.num_epochs}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"{'='*60}\n")
