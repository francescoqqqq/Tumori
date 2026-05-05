"""
Custom nnU-Net trainer con loss geometrica per segmentazione cerchi.

Estende nnUNetTrainer aggiungendo penalità geometriche per forzare
la rete a produrre cerchi più perfetti.

Author: Francesco + Claude
Date: 2025-12-05

"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # pyright: ignore[reportMissingImports]
from nnunetv2.training.nnUNetTrainer.geometric_losses import GeometricLosses, _GEOMETRIC_CONFIG_LOADED as _LOSSES_CONFIG_LOADED  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import numpy as np
import os

# Importa configurazione centralizzata
# _TRAINER_CONFIG_LOADED: True se il file è stato trovato, False se si usano valori fallback
_TRAINER_CONFIG_LOADED = False
try:
    from geometric_config import (
        WEIGHT_COMPACTNESS, WEIGHT_ECCENTRICITY,
        WEIGHT_BOUNDARY, WARMUP_EPOCHS, NUM_EPOCHS, BATCH_SIZE,
        GEOMETRIC_LOSS_SAMPLES
    )
    _TRAINER_CONFIG_LOADED = True
except ImportError:
    # Fallback se geometric_config non è disponibile
    WEIGHT_COMPACTNESS = 0.01
    WEIGHT_ECCENTRICITY = 0.03
    WEIGHT_BOUNDARY = 0.005
    WARMUP_EPOCHS = 20
    NUM_EPOCHS = 100
    BATCH_SIZE = 8
    GEOMETRIC_LOSS_SAMPLES = 4


class nnUNetTrainerGeometric(nnUNetTrainer):
    """
    Trainer nnU-Net con loss geometrica per segmentazione cerchi.

    Aggiunge penalità geometriche alla loss standard (Dice + CE) per
    migliorare compactness, eccentricity e smoothness dei bordi.
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
        # Override batch_size nei plans prima di chiamare super()
        if 'configurations' in plans:
            for config_name, config_data in plans['configurations'].items():
                if isinstance(config_data, dict) and 'batch_size' in config_data:
                    config_data['batch_size'] = BATCH_SIZE

        if isinstance(plans, dict):
            def set_batch_size_recursive(d, target_value=BATCH_SIZE):
                if isinstance(d, dict):
                    for key, value in d.items():
                        if key == 'batch_size':
                            d[key] = target_value
                        elif isinstance(value, (dict, list)):
                            set_batch_size_recursive(value, target_value)
                elif isinstance(d, list):
                    for item in d:
                        set_batch_size_recursive(item, target_value)

            set_batch_size_recursive(plans, BATCH_SIZE)

        # Chiama costruttore base
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Override batch_size anche dopo inizializzazione (per sicurezza)
        if hasattr(self, 'configuration_manager'):
            if hasattr(self.configuration_manager, 'data_loader_kwargs'):
                if isinstance(self.configuration_manager.data_loader_kwargs, dict):
                    self.configuration_manager.data_loader_kwargs['batch_size'] = BATCH_SIZE

            if hasattr(self.configuration_manager, 'configuration'):
                if isinstance(self.configuration_manager.configuration, dict):
                    if 'batch_size' in self.configuration_manager.configuration:
                        self.configuration_manager.configuration['batch_size'] = BATCH_SIZE

        # Numero di campioni su cui calcolare loss geometrica (per risparmiare memoria)
        # Valori da geometric_config.py
        self.geometric_loss_samples = GEOMETRIC_LOSS_SAMPLES

        # Loss geometrica completa (usata solo per logging in on_epoch_end)
        self.geometric_loss = GeometricLosses(
            weight_compactness=WEIGHT_COMPACTNESS,
            weight_eccentricity=WEIGHT_ECCENTRICITY,
            weight_boundary=WEIGHT_BOUNDARY,
            min_area=10,
        )

        # ── Split gating ────────────────────────────────────────────────────
        # PROBLEMA: quando la rete fallisce (Dice basso), il Dice gate azzera
        # tutta la loss geometrica → i fallimenti producono blob elongati invece
        # di forme circolari, peggiorando l'eccentricità media aggregata.
        #
        # SOLUZIONE: separare compactness da eccentricity+boundary.
        #
        # geometric_loss_shape  (eccentricity + boundary, weight_compactness=0)
        #   → gate SOFT basato su Dice grezzo: attiva anche su fallimenti parziali
        #   → spinge i blob sbagliati verso forme circolari invece che elongate
        #
        # geometric_loss_compact  (solo compactness, ecc/boundary=0)
        #   → gate HARD (Dice gate originale): attivo solo quando la rete ha già
        #     trovato il cerchio → non rischia di rimpicciolire predizioni sbagliate
        self.geometric_loss_shape = GeometricLosses(
            weight_compactness=0.0,
            weight_eccentricity=WEIGHT_ECCENTRICITY,
            weight_boundary=WEIGHT_BOUNDARY,
            min_area=10,
        )
        self.geometric_loss_compact = GeometricLosses(
            weight_compactness=WEIGHT_COMPACTNESS,
            weight_eccentricity=0.0,
            weight_boundary=0.0,
            min_area=10,
        )

        # Flag per attivare/disattivare loss geometrica
        self.use_geometric_loss = True

        # Warm-up e num epoche da geometric_config.py
        self.geometric_loss_warmup_epochs = WARMUP_EPOCHS
        self.num_epochs = NUM_EPOCHS

        # Ramp progressiva: dopo il warmup la geometric loss scala linearmente
        # da 0 a 1 per WARMUP_EPOCHS epoche, evitando il salto brusco che
        # destabilizza i casi borderline.
        self.geometric_loss_warmup_ramp = WARMUP_EPOCHS

        # Storage per logging loss geometrica e dice gate
        self.geometric_loss_log = []
        self.dice_gate_log = []  # traccia l'efficacia del Dice gate per epoch
        # Verify counters: misurano quanto la loss geometrica viene davvero usata
        self.verify_stats = {
            'total_steps': 0,
            'geom_attempted_steps': 0,
            'geom_applied_steps': 0,
            'geom_bypassed_warmup_steps': 0,
            'geom_bypassed_exception_steps': 0,
            'geom_bypassed_naninf_steps': 0,
            'optimizer_skipped_steps': 0,
        }
        
        # Gradient debugging: verifica gradient flow ogni N epoche
        self.gradient_check_interval = 10  # Ogni 10 epoche

        print(f"\n{'='*60}")
        print("nnUNetTrainerGeometric inizializzato")
        print(f"{'='*60}")
        print(f"Numero epoche:   {self.num_epochs}")
        _ramp_end = self.geometric_loss_warmup_epochs + self.geometric_loss_warmup_ramp
        print(f"Warm-up epoche:  {self.geometric_loss_warmup_epochs}")
        print(f"  (epoche 0-{self.geometric_loss_warmup_epochs-1}: solo Dice+CE)")
        print(f"  (epoche {self.geometric_loss_warmup_epochs}-{_ramp_end-1}: "
              f"Dice+CE + Geometric ramp 0→1)")
        print(f"  (epoche {_ramp_end}-{self.num_epochs-1}: "
              f"Dice+CE + Geometric peso pieno + Dice gate)")
        print(f"Loss geometrica: {self.use_geometric_loss}  "
              f"(su primi {self.geometric_loss_samples} campioni del batch)")
        print(f"Pesi: compactness={self.geometric_loss.weight_compactness}  "
              f"eccentricity={self.geometric_loss.weight_eccentricity}  "
              f"boundary={self.geometric_loss.weight_boundary}")
        print(f"{'='*60}")

        # Verifica che geometric_config sia stato caricato dal file e non dal fallback
        self._config_load_ok = _TRAINER_CONFIG_LOADED and _LOSSES_CONFIG_LOADED
        if not self._config_load_ok:
            print(f"\n{'!'*60}")
            print("⚠️  CONFIG WARNING: geometric_config.py NON trovato nel sys.path!")
            print("   I pesi impostati in run_experiment.py NON vengono usati.")
            print("   Vengono usati valori di FALLBACK hardcoded:")
            print(f"     compactness={self.geometric_loss.weight_compactness}  "
                  f"eccentricity={self.geometric_loss.weight_eccentricity}  "
                  f"boundary={self.geometric_loss.weight_boundary}")
            print("   FIX: impostare PYTHONPATH nella variabile d'ambiente del subprocess.")
            print(f"{'!'*60}\n")
        else:
            print(f"Config caricata da file: OK\n")

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
        self.verify_stats['total_steps'] += 1

        max_batch_size = BATCH_SIZE
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
        geometric_applied_this_step = False

        # Warmup progressivo: rampa lineare da 0 a 1 invece di step secco.
        # Epoche [0, warmup): warmup_scale = 0.0  (solo Dice+CE)
        # Epoche [warmup, warmup+ramp): warmup_scale sale 0→1
        # Epoche [warmup+ramp, fine): warmup_scale = 1.0
        ramp_epochs = max(1, self.geometric_loss_warmup_ramp)
        warmup_scale = max(0.0, min(1.0,
            (self.current_epoch - self.geometric_loss_warmup_epochs) / ramp_epochs
        ))

        if self.use_geometric_loss and warmup_scale > 0.0:
            self.verify_stats['geom_attempted_steps'] += 1
            try:
                if isinstance(output, (list, tuple)):
                    output_tensor = output[0]
                else:
                    output_tensor = output

                batch_size = output_tensor.shape[0]
                n_samples = min(self.geometric_loss_samples, batch_size)
                output_geometric = output_tensor[:n_samples]

                output_softmax_grad = torch.softmax(output_geometric, dim=1)

                # ── Calcolo gate Dice ─────────────────────────────────────────
                with torch.no_grad():
                    pred_fg = output_softmax_grad[:, 1]  # [N, H, W]
                    if isinstance(target, list):
                        tgt_raw = target[0][:n_samples]
                    else:
                        tgt_raw = target[:n_samples]
                    if tgt_raw.ndim == 4:
                        tgt_raw = tgt_raw[:, 0]
                    tgt_b = (tgt_raw > 0.5).float()
                    inter = (pred_fg * tgt_b).sum(dim=(1, 2))
                    union = pred_fg.sum(dim=(1, 2)) + tgt_b.sum(dim=(1, 2))
                    dice_per_sample = 2 * inter / (union + 1e-5)

                    # Gate HARD per compactness: 0 se Dice<0.5, rampa a 1 se Dice=1.
                    # Impedisce alla compactness di rimpicciolire predizioni sbagliate.
                    dice_gate = ((dice_per_sample - 0.5) / 0.5).clamp(0.0, 1.0).mean()

                    # Gate SOFT per eccentricity+boundary: proporzionale al Dice grezzo.
                    # Attivo anche su fallimenti parziali (Dice 0.1–0.4) così i blob
                    # sbagliati vengono spinti verso forme circolari invece che elongate.
                    dice_gate_shape = dice_per_sample.mean().clamp(0.0, 1.0)
                # ─────────────────────────────────────────────────────────────

                # eccentricity + boundary: gate soft (attivo anche su fallimenti)
                loss_shape = (
                    self.geometric_loss_shape(output_softmax_grad) * warmup_scale * dice_gate_shape
                )
                # compactness: gate hard (solo quando il cerchio è già trovato)
                loss_compact = (
                    self.geometric_loss_compact(output_softmax_grad) * warmup_scale * dice_gate
                )
                loss_geometric = loss_shape + loss_compact

                # Aggiorna last_losses per il logging (usa l'oggetto completo)
                self.geometric_loss(output_softmax_grad)

                self.dice_gate_log.append(dice_gate.item())
                geometric_applied_this_step = True
            except Exception as e:
                print(f"⚠️  WARNING [Epoch {self.current_epoch}]: Geometric loss failed: {e}")
                loss_geometric = torch.tensor(0.0, device=self.device)
                self.verify_stats['geom_bypassed_exception_steps'] += 1
        else:
            self.verify_stats['geom_bypassed_warmup_steps'] += 1

        # Loss totale
        total_loss = loss_dice_ce + loss_geometric

        # SAFETY CHECK: Verifica NaN/Inf prima del backward.
        # Se total_loss è NaN/Inf, prova il fallback su loss_dice_ce.
        # Se anche loss_dice_ce è NaN/Inf, skippa l'intero step.
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n{'='*70}")
            print(f"⚠️  WARNING [Epoch {self.current_epoch}]: Loss totale è NaN/Inf!")
            print(f"   Dice+CE: {loss_dice_ce.item() if not torch.isnan(loss_dice_ce) else 'NaN'}")
            print(f"   Geometric: {loss_geometric.item() if not torch.isnan(loss_geometric) else 'NaN'}")
            if hasattr(self.geometric_loss, 'get_last_losses'):
                print(f"   Componenti: {self.geometric_loss.get_last_losses()}")
            print(f"{'='*70}\n")
            if geometric_applied_this_step:
                self.verify_stats['geom_bypassed_naninf_steps'] += 1
            # Fallback a Dice+CE; se anche quella è NaN skippa lo step
            total_loss = loss_dice_ce
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️  WARNING [Epoch {self.current_epoch}]: anche Dice+CE è NaN/Inf, step skippato.")
                self.optimizer.zero_grad(set_to_none=True)
                self.verify_stats['optimizer_skipped_steps'] += 1
                return {'loss': torch.tensor(0.0, device=self.device).cpu().numpy()}

        # Backward e step optimizer
        try:
            total_loss.backward()
        except RuntimeError as e:
            if 'nan' in str(e).lower() or 'inf' in str(e).lower():
                print(f"\n⚠️  WARNING [Epoch {self.current_epoch}]: Errore durante backward: {e}")
                print(f"   Skipping optimizer step per evitare corruzione.")
                self.optimizer.zero_grad(set_to_none=True)
                self.verify_stats['optimizer_skipped_steps'] += 1
                return {'loss': torch.tensor(0.0, device=self.device).cpu().numpy()}
            else:
                raise

        # Gradient clipping per stabilità (con controllo NaN)
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n⚠️  WARNING [Epoch {self.current_epoch}]: grad_norm è NaN/Inf dopo clipping!")
                self.optimizer.zero_grad(set_to_none=True)
                self.verify_stats['optimizer_skipped_steps'] += 1
                return {'loss': torch.tensor(0.0, device=self.device).cpu().numpy()}
        except RuntimeError as e:
            print(f"\n⚠️  WARNING [Epoch {self.current_epoch}]: Errore durante gradient clipping: {e}")
            self.optimizer.zero_grad(set_to_none=True)
            self.verify_stats['optimizer_skipped_steps'] += 1
            return {'loss': torch.tensor(0.0, device=self.device).cpu().numpy()}

        # NaN nei singoli gradienti è già implicitamente coperto da:
        # 1) check NaN su total_loss prima del backward
        # 2) clip_grad_norm_ che controlla la norma totale e skippa se NaN/Inf
        # Il loop per-parametro era costoso (itera tutta la rete ad ogni step) e ridondante.
        self.optimizer.step()
        if geometric_applied_this_step:
            self.verify_stats['geom_applied_steps'] += 1

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

            # Log su logger nnU-Net (ogni epoca)
            try:
                if hasattr(self.logger, 'my_fantastic_logging'):
                    for key in ('train_loss_dice_ce', 'train_loss_geometric'):
                        if key not in self.logger.my_fantastic_logging:
                            self.logger.my_fantastic_logging[key] = []
                self.logger.log('train_loss_dice_ce', avg_dice_ce, self.current_epoch)
                self.logger.log('train_loss_geometric', avg_geometric, self.current_epoch)
            except (AssertionError, AttributeError) as e:
                if self.current_epoch % 10 == 0:
                    print(f"⚠️  Warning: Logging fallito: {e}")

            # Log componenti geometriche su logger (ogni epoca)
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
                    pass
            else:
                last_losses = {}

            # Print periodico ogni 10 epoche (un solo blocco unificato)
            if self.current_epoch % 10 == 0:
                ramp_e = max(1, self.geometric_loss_warmup_ramp)
                w_scale = max(0.0, min(1.0,
                    (self.current_epoch - self.geometric_loss_warmup_epochs) / ramp_e
                ))
                avg_dice_gate = float(np.mean(self.dice_gate_log)) if self.dice_gate_log else 0.0
                print(f"\n{'='*65}")
                print(f"[Epoch {self.current_epoch}] Loss Summary")
                print(f"{'='*65}")
                print(f"  Dice+CE:        {avg_dice_ce:.6f}")
                print(f"  Geometric:      {avg_geometric:.6f}")
                print(f"  Total:          {avg_total:.6f}")
                print(f"  warmup_scale:   {w_scale:.3f}  "
                      f"(epoche {self.geometric_loss_warmup_epochs}→"
                      f"{self.geometric_loss_warmup_epochs + ramp_e})")
                print(f"  dice_gate:      {avg_dice_gate:.3f}  "
                      f"(0=geom silenziosa, 1=pieno peso)")
                if last_losses:
                    print(f"  Components: "
                          f"compactness={last_losses.get('compactness', 0):.6f}  "
                          f"boundary={last_losses.get('boundary', 0):.6f}  "
                          f"eccentricity={last_losses.get('eccentricity', 0):.6f}")
                # Gradient flow check (solo dopo warmup)
                if self.current_epoch >= self.geometric_loss_warmup_epochs:
                    grad_norms = [p.grad.abs().mean().item()
                                  for p in self.network.parameters()
                                  if p.grad is not None]
                    if grad_norms:
                        grad_mean = np.mean(grad_norms)
                        status = "✅ OK" if grad_mean > 1e-10 else "⚠️  PICCOLI"
                        print(f"  Gradient mean:  {grad_mean:.2e}  {status}")
                print(f"{'='*65}\n")

            # Reset log
            self.geometric_loss_log = []
            self.dice_gate_log = []

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
            f.write(f"  - Compactness:  {self.geometric_loss.weight_compactness}\n")
            f.write(f"  - Eccentricity: {self.geometric_loss.weight_eccentricity}\n")
            f.write(f"  - Boundary:     {self.geometric_loss.weight_boundary}\n\n")
            f.write(f"Other parameters:\n")
            f.write(f"  - Min area threshold: {self.geometric_loss.min_area} pixels\n")
            f.write(f"  - Batch size effettivo: 8 (ridotto per risparmiare memoria)\n")
            f.write(f"  - Loss geometrica calcolata su primi {self.geometric_loss_samples} campioni del batch\n")
            f.write(f"  - Geometric loss samples: {self.geometric_loss_samples} (solo primi N campioni del batch)\n")

        print(f"\n✅ Geometric loss config salvato: {config_file}\n")

        # Verify report: quanto la branch geometrica è stata davvero usata vs bypassata
        verify_file = os.path.join(self.output_folder, "verify.log")
        total_steps = self.verify_stats['total_steps']
        attempted = self.verify_stats['geom_attempted_steps']
        applied = self.verify_stats['geom_applied_steps']
        warmup_bypass = self.verify_stats['geom_bypassed_warmup_steps']
        exc_bypass = self.verify_stats['geom_bypassed_exception_steps']
        naninf_bypass = self.verify_stats['geom_bypassed_naninf_steps']
        opt_skipped = self.verify_stats['optimizer_skipped_steps']

        def _pct(n, d):
            return (100.0 * n / d) if d > 0 else 0.0

        with open(verify_file, 'w') as f:
            f.write("GEOMETRIC LOSS VERIFY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total training steps: {total_steps}\n")
            f.write(f"Geometric attempted:  {attempted} ({_pct(attempted, total_steps):.2f}% of total)\n")
            f.write(f"Geometric applied:    {applied} ({_pct(applied, total_steps):.2f}% of total, {_pct(applied, attempted):.2f}% of attempted)\n")
            f.write(f"Bypassed (warmup):    {warmup_bypass} ({_pct(warmup_bypass, total_steps):.2f}% of total)\n")
            f.write(f"Bypassed (exception): {exc_bypass} ({_pct(exc_bypass, total_steps):.2f}% of total)\n")
            f.write(f"Bypassed (NaN/Inf):   {naninf_bypass} ({_pct(naninf_bypass, total_steps):.2f}% of total)\n")
            f.write(f"Optimizer skipped:    {opt_skipped} ({_pct(opt_skipped, total_steps):.2f}% of total)\n\n")
            if attempted == 0:
                f.write("NOTE: Geometric branch mai tentata (probabile warmup >= total epochs).\n")
            elif _pct(applied, attempted) < 70.0:
                f.write("WARNING: Geometric branch spesso bypassata (<70% degli step tentati).\n")
            else:
                f.write("OK: Geometric branch usata in modo consistente.\n")

            # ── CONFIG IMPORT VERIFICATION ──────────────────────────────────
            f.write("\n\nCONFIG IMPORT VERIFICATION\n")
            f.write("=" * 60 + "\n\n")
            trainer_status = "OK" if _TRAINER_CONFIG_LOADED else "FAILED (fallback hardcoded)"
            losses_status  = "OK" if _LOSSES_CONFIG_LOADED  else "FAILED (fallback hardcoded)"
            f.write(f"geometric_config (nnUNetTrainerGeometric): {trainer_status}\n")
            f.write(f"geometric_config (geometric_losses):       {losses_status}\n\n")
            f.write(f"Pesi usati in questo training:\n")
            f.write(f"  compactness:  {self.geometric_loss.weight_compactness}\n")
            f.write(f"  eccentricity: {self.geometric_loss.weight_eccentricity}\n")
            f.write(f"  boundary:     {self.geometric_loss.weight_boundary}\n\n")
            if not self._config_load_ok:
                f.write("WARNING: geometric_config.py NON trovato nel sys.path del subprocess.\n")
                f.write("   I pesi configurati in run_experiment.py NON sono stati usati.\n")
                f.write("   Causa: entry-point nnUNetv2_train non include la CWD in sys.path.\n")
                f.write("   Fix:   PYTHONPATH=<geometrica/> viene impostato da run_experiment.py.\n")
            else:
                f.write("OK: geometric_config.py caricato correttamente dal file.\n")

        print(f"✅ Verify log salvato: {verify_file}")
