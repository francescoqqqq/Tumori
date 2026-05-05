"""
Trainer baseline personalizzato per nnU-Net.
Deve essere copiato in: nnunetv2/training/nnUNetTrainer/

NOTA: num_epochs e BATCH_SIZE vengono patchati dinamicamente da
run_experiment.py prima del training, quindi i valori hardcoded qui
sotto sono solo placeholder di default.
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

# Placeholder patchati da run_experiment.py prima del training
BATCH_SIZE = 8   # -> sostituito con il valore di BATCH_SIZE del blocco config


class nnUNetTrainerBaseline(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
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
        self.num_epochs = 100  # -> patchato da run_experiment.py

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
