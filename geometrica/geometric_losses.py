"""
VERSIONE DIFFERENZIABILE COMPLETA con PROTEZIONI ANTI-NaN V2.4

Loss geometriche vettorizzate e sicure per training stabile.
Implementazione fully differentiable che opera sull'intero batch
senza cicli for, mantenendo il computational graph intatto.

Protezioni V2.4 (FIX GRADIENTI ESPLOSIVI):
- Clamp AGGRESSIVO (min=1e-2 invece di 1e-4) prima di sqrt
  → Riduce gradiente da 50 a 5 (10x più stabile!)
- Controllo area minima (50.0) prima di calcolare loss
- Protezioni aggressive in tutti i calcoli intermedi
- Gestione predizioni quasi vuote (ritorna 0.0 con computational graph)
- Controlli finali NaN in ogni funzione loss

MOTIVAZIONE V2.4:
Il gradiente di sqrt(x) è 1/(2*sqrt(x)):
- sqrt(1e-4) → gradiente = 50 (ESPLOSIONE!)
- sqrt(1e-2) → gradiente = 5 (STABILE)

Con predizioni incerte (epoca 5-20), ~28% dei pixel hanno
grad_mag_squared < 1e-2, causando esplosione gradienti nel backward.

Author: Francesco + Claude
Date: 2025-12-13
Version: 2.4 (fix gradienti esplosivi in sqrt)
"""

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
import math


class DifferentiableGeometricLossesV2:
    """
    Loss geometriche completamente differenziabili e vettorizzate.

    Processa l'intero batch in parallelo mantenendo gradienti.
    Include protezioni anti-NaN per training stabile.
    """

    def __init__(self,
                 weight_compactness: float = 0.01,
                 weight_boundary: float = 0.01,
                 weight_aspect: float = 0.005):
        """
        Versione semplificata con solo 3 loss differenziabili:
        - Compactness (area vs perimeter)
        - Boundary smoothness (second derivatives)
        - Aspect ratio (momenti di inerzia)

        NOTA: Pesi ridotti 10x rispetto a versione precedente per stabilità.
        """
        self.weight_compactness = weight_compactness
        self.weight_boundary = weight_boundary
        self.weight_aspect = weight_aspect

        self.last_losses = {}

    def __call__(self, pred_softmax: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_softmax: [B, C, H, W] dopo softmax, C=2

        Returns:
            Loss scalare differenziabile (0.0 se NaN)
        """
        # Estrai probabilità cerchi - NO thresholding!
        pred_soft = pred_softmax[:, 1, :, :]  # [B, H, W]

        # SAFETY CHECK: Verifica che pred_soft abbia abbastanza "massa" prima di calcolare loss
        # Se la predizione è quasi tutta background, i calcoli geometrici non hanno senso
        area_per_batch = pred_soft.sum(dim=(1, 2))  # [B]
        min_area_threshold = 50.0  # Area minima per considerare la predizione valida
        
        # Se tutti i batch hanno area troppo piccola, ritorna 0.0 mantenendo computational graph
        if (area_per_batch < min_area_threshold).all():
            return pred_soft.sum() * 0.0

        # Calcola loss components (vettorizzate) con protezioni
        # Se area è troppo piccola, le loss ritornano 0.0 mantenendo computational graph
        loss_compact = self._vectorized_compactness_loss(pred_soft)
        loss_bound = self._vectorized_boundary_loss(pred_soft)
        loss_aspect = self._vectorized_aspect_loss(pred_soft)

        # CHECK NaN: Se una loss è NaN, la setta a 0 MANTENENDO computational graph
        # IMPORTANTE: Usiamo pred_soft.sum() * 0.0 invece di torch.tensor(0.0)
        # per mantenere il computational graph e permettere gradient flow
        if torch.isnan(loss_compact) or torch.isinf(loss_compact):
            loss_compact = pred_soft.sum() * 0.0  # Mantiene computational graph!
        if torch.isnan(loss_bound) or torch.isinf(loss_bound):
            loss_bound = pred_soft.sum() * 0.0
        if torch.isnan(loss_aspect) or torch.isinf(loss_aspect):
            loss_aspect = pred_soft.sum() * 0.0

        # Logging
        self.last_losses = {
            'compactness': loss_compact.item(),
            'boundary': loss_bound.item(),
            'aspect': loss_aspect.item()
        }

        # Combinazione
        total = (self.weight_compactness * loss_compact +
                self.weight_boundary * loss_bound +
                self.weight_aspect * loss_aspect)

        # CHECK finale NaN - mantiene computational graph
        if torch.isnan(total) or torch.isinf(total):
            return pred_soft.sum() * 0.0

        return total

    def _vectorized_compactness_loss(self, pred_soft: torch.Tensor) -> torch.Tensor:
        """
        Compactness differenziabile usando soft area e soft perimeter.

        Compactness = 4π·Area / Perimeter²

        Args:
            pred_soft: [B, H, W] probabilità soft

        Returns:
            Loss scalare (protetto da NaN)
        """
        # Soft area (sum di probabilità)
        area = pred_soft.sum(dim=(1, 2))  # [B]

        # SAFETY CHECK: Se area è troppo piccola, ritorna 0.0 mantenendo computational graph
        min_area_threshold = 50.0
        
        # Gestisci batch con area insufficiente: usa mask per escluderli dal calcolo
        valid_mask = area >= min_area_threshold  # [B] boolean
        
        # Se nessun batch è valido, ritorna 0.0
        if not valid_mask.any():
            return pred_soft.sum() * 0.0

        # PROTEZIONE: Clampa area per evitare valori estremi
        area = torch.clamp(area, min=min_area_threshold, max=1e6)

        # Soft perimeter usando gradiente Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred_soft.dtype, device=pred_soft.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred_soft.dtype, device=pred_soft.device)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        # Aggiungi channel dimension
        pred_4d = pred_soft.unsqueeze(1)  # [B, 1, H, W]

        # Gradienti
        grad_x = F.conv2d(pred_4d, sobel_x, padding=1)
        grad_y = F.conv2d(pred_4d, sobel_y, padding=1)

        # Magnitudine = bordo soft
        # PROTEZIONE V2.4: Clamp AGGRESSIVO (min=1e-2) per evitare gradienti esplosivi
        # Il gradiente di sqrt(x) è 1/(2*sqrt(x)):
        #   - V2.3: sqrt(1e-4) → gradiente = 50 (ESPLOSIONE!)
        #   - V2.4: sqrt(1e-2) → gradiente = 5 (STABILE!)
        # Con predizioni incerte, ~28% pixel hanno grad_mag_squared < 1e-2
        grad_mag_squared = grad_x**2 + grad_y**2
        grad_mag_squared = torch.clamp(grad_mag_squared, min=1e-2)  # V2.4: min=1e-2 (era 1e-4)
        grad_mag = torch.sqrt(grad_mag_squared)

        # Soft perimeter
        perimeter = grad_mag.sum(dim=(1, 2, 3))  # [B]

        # PROTEZIONE: Clampa perimeter per evitare divisione per valori troppo piccoli
        perimeter = torch.clamp(perimeter, min=10.0, max=1e6)

        # Compactness con epsilon più grande per stabilità
        epsilon = 1e-2
        # PROTEZIONE: Assicura che il denominatore non sia zero o negativo
        denominator = perimeter**2 + epsilon
        denominator = torch.clamp(denominator, min=epsilon)  # Assicura almeno epsilon
        compactness = (4 * math.pi * area) / denominator

        # PROTEZIONE: Clampa compactness tra 0 e 1
        compactness = torch.clamp(compactness, min=0.0, max=1.0)

        # Loss = 1 - compactness (media solo su batch validi)
        loss_per_batch = 1.0 - compactness
        # Se alcuni batch non sono validi, usa solo quelli validi per la media
        if valid_mask.all():
            loss = loss_per_batch.mean()
        else:
            # Media solo sui batch validi
            loss = loss_per_batch[valid_mask].mean() if valid_mask.any() else pred_soft.sum() * 0.0

        # SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
        if torch.isnan(loss) or torch.isinf(loss):
            return pred_soft.sum() * 0.0

        return loss

    def _vectorized_boundary_loss(self, pred_soft: torch.Tensor) -> torch.Tensor:
        """
        Boundary smoothness usando Laplacian (second derivatives).

        Alta varianza del Laplacian = bordo irregolare

        Args:
            pred_soft: [B, H, W]

        Returns:
            Loss scalare (protetto da NaN)
        """
        # SAFETY CHECK: Verifica area prima di calcolare
        area_check = pred_soft.sum(dim=(1, 2))  # [B]
        min_area_threshold = 50.0
        
        # Gestisci batch con area insufficiente
        valid_mask = area_check >= min_area_threshold  # [B] boolean
        
        # Se nessun batch è valido, ritorna 0.0
        if not valid_mask.any():
            return pred_soft.sum() * 0.0

        # Laplacian kernel
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                dtype=pred_soft.dtype, device=pred_soft.device)
        laplacian = laplacian.view(1, 1, 3, 3)

        pred_4d = pred_soft.unsqueeze(1)  # [B, 1, H, W]

        # Calcola Laplacian
        lap_response = F.conv2d(pred_4d, laplacian, padding=1)  # [B, 1, H, W]
        lap_response = lap_response.squeeze(1)  # [B, H, W]

        # PROTEZIONE: Clampa lap_response prima di usarlo
        lap_response = torch.clamp(lap_response, min=-100.0, max=100.0)

        # Penalizza dove la maschera è attiva
        mask_active = (pred_soft > 0.1).float()

        # Weighted Laplacian
        lap_weighted = lap_response * mask_active

        # PROTEZIONE: Clampa lap_weighted per evitare valori estremi
        lap_weighted = torch.clamp(lap_weighted, min=-10.0, max=10.0)

        # Varianza + mean abs come loss
        # Calcola per batch e fai media solo sui batch validi
        # PROTEZIONE: Usa unbiased=False per evitare divisione per zero quando tutti i valori sono uguali
        var_per_batch = lap_weighted.view(lap_weighted.size(0), -1).var(dim=1, unbiased=False)
        mean_per_batch = lap_weighted.abs().view(lap_weighted.size(0), -1).mean(dim=1)

        # PROTEZIONE: Clampa varianza per evitare esplosioni
        var_per_batch = torch.clamp(var_per_batch, min=0.0, max=100.0)
        mean_per_batch = torch.clamp(mean_per_batch, min=0.0, max=10.0)

        # Media solo sui batch validi
        if valid_mask.all():
            loss = var_per_batch.mean() + 0.1 * mean_per_batch.mean()
        else:
            loss = (var_per_batch[valid_mask].mean() + 0.1 * mean_per_batch[valid_mask].mean()) if valid_mask.any() else pred_soft.sum() * 0.0

        # SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
        if torch.isnan(loss) or torch.isinf(loss):
            return pred_soft.sum() * 0.0

        return loss

    def _vectorized_aspect_loss(self, pred_soft: torch.Tensor) -> torch.Tensor:
        """
        Aspect ratio usando momenti di inerzia (fully differentiable).

        Penalizza forme allungate (ellissi vs cerchi).

        Args:
            pred_soft: [B, H, W]

        Returns:
            Loss scalare (protetto da NaN)
        """
        B, H, W = pred_soft.shape

        # SAFETY CHECK: Verifica area prima di calcolare
        area_check = pred_soft.sum(dim=(1, 2))  # [B]
        min_area_threshold = 50.0
        
        # Gestisci batch con area insufficiente
        valid_mask = area_check >= min_area_threshold  # [B] boolean
        
        # Se nessun batch è valido, ritorna 0.0
        if not valid_mask.any():
            return pred_soft.sum() * 0.0

        # Coordinate grids
        y_coords = torch.arange(H, dtype=pred_soft.dtype, device=pred_soft.device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, dtype=pred_soft.dtype, device=pred_soft.device).view(1, 1, W).expand(B, H, W)

        # Soft area per batch (con protezione)
        area = pred_soft.sum(dim=(1, 2), keepdim=True) + 1e-2  # [B, 1, 1] - epsilon più grande
        area = torch.clamp(area, min=min_area_threshold)  # PROTEZIONE: area minima più alta

        # Centro di massa ponderato
        x_center = (pred_soft * x_coords).sum(dim=(1, 2), keepdim=True) / area  # [B, 1, 1]
        y_center = (pred_soft * y_coords).sum(dim=(1, 2), keepdim=True) / area  # [B, 1, 1]

        # PROTEZIONE: Clampa centro di massa per evitare valori estremi
        x_center = torch.clamp(x_center, min=-W, max=2*W)
        y_center = torch.clamp(y_center, min=-H, max=2*H)

        # Differenze dal centro
        x_diff = x_coords - x_center
        y_diff = y_coords - y_center

        # PROTEZIONE: Clampa differenze per evitare overflow
        x_diff = torch.clamp(x_diff, min=-W*2, max=W*2)
        y_diff = torch.clamp(y_diff, min=-H*2, max=H*2)

        # Momenti di secondo ordine
        mu_20 = (pred_soft * x_diff**2).sum(dim=(1, 2)) / area.squeeze()  # [B]
        mu_02 = (pred_soft * y_diff**2).sum(dim=(1, 2)) / area.squeeze()  # [B]
        mu_11 = (pred_soft * x_diff * y_diff).sum(dim=(1, 2)) / area.squeeze()  # [B]

        # PROTEZIONE: Clampa momenti per evitare valori estremi
        mu_20 = torch.clamp(mu_20, min=1e-2, max=1e6)
        mu_02 = torch.clamp(mu_02, min=1e-2, max=1e6)
        mu_11 = torch.clamp(mu_11, min=-1e6, max=1e6)

        # Eigenvalues (assi principali)
        trace = mu_20 + mu_02
        det = mu_20 * mu_02 - mu_11**2

        # PROTEZIONE: Assicura che det non sia troppo negativo e che trace sia positivo
        trace = torch.clamp(trace, min=1e-2, max=1e6)
        det = torch.clamp(det, min=-1e6, max=1e6)

        # Calcola sqrt_term con protezione più aggressiva
        # PROTEZIONE V2.4: Clamp AGGRESSIVO (min=1e-2) per evitare gradienti esplosivi
        # Come in compactness_loss: sqrt(1e-2) → gradiente = 5 (STABILE)
        discriminant = trace**2 - 4*det
        discriminant = torch.clamp(discriminant, min=1e-2, max=1e12)  # V2.4: min=1e-2 (era 1e-4)
        sqrt_term = torch.sqrt(discriminant)
        
        lambda1 = (trace + sqrt_term) / 2 + 1e-2  # Maggiore - epsilon più grande
        lambda2 = (trace - sqrt_term) / 2 + 1e-2  # Minore - epsilon più grande

        # PROTEZIONE: Assicura che lambda2 non sia troppo piccolo e che lambda1 >= lambda2
        lambda1 = torch.clamp(lambda1, min=0.1, max=1e6)
        lambda2 = torch.clamp(lambda2, min=0.1, max=1e6)
        # Assicura lambda2 <= lambda1 (usa min per forzare upper bound)
        lambda2 = torch.min(lambda2, lambda1)

        # Aspect ratio
        aspect_ratio = lambda1 / (lambda2 + 1e-6)  # Aggiungi epsilon al denominatore

        # PROTEZIONE: Clampa aspect_ratio per evitare valori estremi
        aspect_ratio = torch.clamp(aspect_ratio, min=0.1, max=10.0)

        # Loss: penalizza ratio lontano da 1 (cerchio perfetto = 1)
        loss_per_batch = torch.abs(aspect_ratio - 1.0) / (aspect_ratio + 1.0 + 1e-6)  # Aggiungi epsilon anche qui
        
        # Media solo sui batch validi
        if valid_mask.all():
            loss = loss_per_batch.mean()
        else:
            loss = loss_per_batch[valid_mask].mean() if valid_mask.any() else pred_soft.sum() * 0.0

        # SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
        if torch.isnan(loss) or torch.isinf(loss):
            return pred_soft.sum() * 0.0

        return loss

    def get_last_losses(self) -> dict:
        """Ritorna componenti loss."""
        return self.last_losses.copy()


class GeometricLosses:
    """
    Wrapper per DifferentiableGeometricLossesV2 per compatibilità con trainer.

    Mappa i parametri del trainer (weight_solidity, weight_eccentricity)
    alla versione V2 semplificata (weight_aspect).
    """

    def __init__(self,
                 weight_compactness: float = 0.01,
                 weight_solidity: float = 0.01,
                 weight_eccentricity: float = 0.005,
                 weight_boundary: float = 0.005,
                 min_area: int = 10):
        """
        Args:
            weight_compactness: Peso per compactness loss
            weight_solidity: Peso per solidity (non usato in V2, mappato ad aspect)
            weight_eccentricity: Peso per eccentricity (non usato in V2, mappato ad aspect)
            weight_boundary: Peso per boundary smoothness loss
            min_area: Area minima (non usato in V2, mantenuto per compatibilità)

        NOTA: Pesi di default ridotti 10x per stabilità training.
        """
        # Salva tutti i parametri per compatibilità con trainer
        self.weight_compactness = weight_compactness
        self.weight_solidity = weight_solidity
        self.weight_eccentricity = weight_eccentricity
        self.weight_boundary = weight_boundary
        self.min_area = min_area

        # Combina solidity + eccentricity in aspect (V2 semplificata)
        # Aspect ratio copre sia solidity (convessità) che eccentricity (ellitticità)
        weight_aspect = weight_solidity + weight_eccentricity

        # Inizializza la classe V2 differenziabile con protezioni anti-NaN
        self._v2_loss = DifferentiableGeometricLossesV2(
            weight_compactness=weight_compactness,
            weight_boundary=weight_boundary,
            weight_aspect=weight_aspect
        )

        self.last_losses = {}

    def __call__(self, pred_softmax: torch.Tensor) -> torch.Tensor:
        """
        Calcola loss geometrica usando DifferentiableGeometricLossesV2.

        Args:
            pred_softmax: [B, C, H, W] dopo softmax, C=2

        Returns:
            Loss scalare differenziabile (0.0 se NaN)
        """
        # Calcola loss usando V2
        loss = self._v2_loss(pred_softmax)

        # Mappa i last_losses per compatibilità
        v2_losses = self._v2_loss.get_last_losses()
        self.last_losses = {
            'compactness': v2_losses.get('compactness', 0.0),
            'boundary': v2_losses.get('boundary', 0.0),
            'aspect': v2_losses.get('aspect', 0.0),
            # Per compatibilità, mappa aspect a solidity/eccentricity
            'solidity': v2_losses.get('aspect', 0.0) * (self.weight_solidity / (self.weight_solidity + self.weight_eccentricity + 1e-8)),
            'eccentricity': v2_losses.get('aspect', 0.0) * (self.weight_eccentricity / (self.weight_solidity + self.weight_eccentricity + 1e-8))
        }

        return loss

    def get_last_losses(self) -> dict:
        """Ritorna componenti loss."""
        return self.last_losses.copy()


# Test
if __name__ == "__main__":
    print("=" * 80)
    print("TEST: Differentiable Geometric Losses V2.4 (fix gradienti esplosivi)")
    print("=" * 80)

    # Crea batch fittizio - SIMULA TRAINING REALE
    batch_size = 4
    img_size = 128

    # IMPORTANTE: Simula logits della rete (prima di softmax)
    # In training vero, la rete output logits che poi vengono softmax
    logits = torch.randn(batch_size, 2, img_size, img_size, requires_grad=True)

    # Aggiungi pattern distinguibile ai logits
    y, x = torch.meshgrid(torch.arange(img_size, dtype=torch.float32),
                          torch.arange(img_size, dtype=torch.float32), indexing='ij')

    # Cerchio (alta prob classe 1 al centro)
    dist = torch.sqrt((x - 64.0)**2 + (y - 64.0)**2)
    circle_logit = 5.0 * (1.0 - dist / 40.0)  # Logit alto = prob alta dopo softmax

    # Modifica logits
    with torch.no_grad():
        logits[0, 1] += circle_logit  # Classe cerchi
        logits[0, 0] -= circle_logit  # Classe background

    # Ora applica softmax DENTRO il computational graph
    pred_softmax = torch.softmax(logits, dim=1)

    # Test con GeometricLosses wrapper
    print("\n--- Test GeometricLosses wrapper ---")
    geom_loss_wrapper = GeometricLosses(
        weight_compactness=0.01,
        weight_solidity=0.01,
        weight_eccentricity=0.005,
        weight_boundary=0.005
    )
    loss = geom_loss_wrapper(pred_softmax)

    print(f"\n✅ Total Loss: {loss.item():.6f}")
    print(f"   Componenti: {geom_loss_wrapper.get_last_losses()}")

    # Check NaN
    if torch.isnan(loss):
        print("\n   ❌ WARNING: Loss è NaN!")
    else:
        print("\n   ✅ Loss è un numero valido")

    # BACKWARD TEST
    print("\n🔍 Testing gradient flow...")
    loss.backward()

    # Check gradients on LEAF tensor (logits), not on pred_softmax
    if logits.grad is not None:
        grad_mean = logits.grad.abs().mean().item()
        grad_max = logits.grad.abs().max().item()
        grad_std = logits.grad.std().item()

        print(f"   Gradient statistics (on logits - leaf tensor):")
        print(f"      mean = {grad_mean:.8f}")
        print(f"      max  = {grad_max:.8f}")
        print(f"      std  = {grad_std:.8f}")

        # Check NaN nei gradienti
        if torch.isnan(logits.grad).any():
            print(f"\n   ❌ WARNING: Gradienti contengono NaN!")
        elif grad_mean > 1e-10:
            print(f"\n   ✅✅✅ GRADIENT FLOW OK!")
            print(f"   I gradienti sono presenti e non-zero - la rete può imparare!")
        else:
            print(f"\n   ⚠️  Gradienti molto piccoli ma validi")
    else:
        print(f"\n   ❌ No gradients on logits!")

    print("\n" + "=" * 80)
    print("✅ Test completato!")
    print("=" * 80)
