"""
Trainer personalizzato per nnU-Net con 100 epoche.
Deve essere copiato in: nnunetv2/training/nnUNetTrainer/
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import os

# Disabilita torch.compile per evitare problemi con librerie CUDA
os.environ['TORCH_COMPILE_DEBUG'] = '0'
# NON impostare TORCH_LOGS qui perché può causare problemi con il logging di PyTorch
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    # Disabilita completamente la compilazione
    torch._dynamo.config.disable = True
except:
    pass


class nnUNetTrainer250Epochs(nnUNetTrainer):
    """Trainer con 100 epoche invece del default."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        # Disabilita torch.compile prima di chiamare super()
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
        except:
            pass
        
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        
        print(f"\n{'='*60}")
        print(f"Trainer: {self.__class__.__name__}")
        print(f"Epoche: {self.num_epochs}")
        print(f"{'='*60}\n")
