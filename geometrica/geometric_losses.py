"""
VERSIONE DIFFERENZIABILE COMPLETA con PROTEZIONI ANTI-NaN V2.6

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

Miglioramenti V2.5 (FIX ECCENTRICITY):
- Aspect loss ora ottimizza direttamente l'eccentricità al quadrato
- Formula: L = (1 - lambda_min/lambda_max)² invece di |AR-1|/(AR+1)
- Penalizzazione quadratica più aggressiva sulle deviazioni dalla circolarità
- Pesi aspect aumentati: 0.03 totale (era 0.015 in V2.4)

MOTIVAZIONE V2.4:
Il gradiente di sqrt(x) è 1/(2*sqrt(x)):
- sqrt(1e-4) → gradiente = 50 (ESPLOSIONE!)
- sqrt(1e-2) → gradiente = 5 (STABILE)

MOTIVAZIONE V2.5:
La vecchia formula |AR-1|/(AR+1) era troppo debole:
- Leggera ellisse (AR=1.1): loss = 0.048, ma e = 0.417!
- Nuova formula (1-ratio)²: loss = 0.01 per ratio=0.9
- Penalizzazione quadratica allineata meglio con eccentricity

Miglioramenti V2.6 (GRADIENT FLOW + EFFICIENZA):
- Boundary loss: maschera soft (pred_soft) invece di soglia hard (>0.1)
  → gradienti fluiscono per tutti i pixel, non solo quelli sopra threshold
- Compactness + Aspect loss: sqrt(clamp(x,min=0)+1e-2) elimina la zona morta
  del gradiente che clamp(min=1e-2)+sqrt creava per x in [0,1e-2]
  → bound massimo gradiente invariato (~5), ma mai zero
- Aspect loss: lazy cache per coordinate grids (y_coords, x_coords)
  → tensori GPU non ricreati a ogni forward pass se H,W,device,dtype non cambiano

Miglioramenti V2.7 (AREA RAMP):
- __call__: area ramp morbida (0→1 tra 50px e 300px) invece di hard threshold.
  Impedisce alla geometric loss di combattere Dice+CE quando la predizione
  è ancora piccola/incorretta (fascia 50–300px).
  La zona morta assoluta <50px è mantenuta come safety net nelle singole loss.

Miglioramenti V2.8 (CIRCLE TEMPLATE LOSS):
- Sostituisce l'eccentricity loss (momenti di inerzia) con la Circle Template Loss.
  
  PROBLEMA CON V2.5 (momenti di inerzia):
  La formula (1 - lambda_min/lambda_max)² ha gradiente ~0 per predizioni già
  quasi circolari (ratio ≈ 0.95 → loss = 0.0025, gradiente quasi zero).
  Nella pratica la loss non influenza il training su target già circolari.

  SOLUZIONE (Circle Template Loss):
  1. Calcola il "cerchio atteso": centroide e r = sqrt(area/π) (stop gradient)
  2. Costruisce template = sigmoid(k · (r − dist_from_center))
  3. Loss = MSE(pred_soft, template)
  
  Vantaggi:
  - Gradienti direzionali: rimuove "angoli" dell'ellisse, riempie i "poli"
  - Scala loss ~50-100× più grande per predizioni ellittiche → gradiente effettivo
  - Per cerchio perfetto: loss ≈ 0 (gradienti solo al bordo, già ottimi)
  - Funziona anche su predizioni piccole/sbagliate (con area ramp)

Author: Francesco + Claude
Date: 2026-04-29
Version: 2.8 (Circle Template Loss)
"""

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
import math

# Importa configurazione centralizzata
# _GEOMETRIC_CONFIG_LOADED: True se il file è stato trovato, False se si usano valori fallback
_GEOMETRIC_CONFIG_LOADED = False
try:
    from geometric_config import (
        WEIGHT_COMPACTNESS, WEIGHT_ECCENTRICITY,
        WEIGHT_BOUNDARY, MIN_AREA_THRESHOLD
    )
    _GEOMETRIC_CONFIG_LOADED = True
except ImportError:
    # Fallback se geometric_config non è disponibile
    WEIGHT_COMPACTNESS = 0.01
    WEIGHT_ECCENTRICITY = 0.03
    WEIGHT_BOUNDARY = 0.005
    MIN_AREA_THRESHOLD = 50.0


class DifferentiableGeometricLossesV2:
    """
    Loss geometriche completamente differenziabili e vettorizzate.

    Processa l'intero batch in parallelo mantenendo gradienti.
    Include protezioni anti-NaN per training stabile.
    """

    def __init__(self,
                 weight_compactness: float = WEIGHT_COMPACTNESS,
                 weight_boundary: float = WEIGHT_BOUNDARY,
                 weight_eccentricity: float = WEIGHT_ECCENTRICITY):
        """
        3 loss differenziabili:
        - Compactness (area vs perimeter)
        - Boundary smoothness (second derivatives)
        - Circle Template (V2.8): penalizza deviazione dal cerchio atteso
          [sostituisce la vecchia eccentricity loss a momenti di inerzia]

        NOTA: Pesi ridotti 10x rispetto a versione precedente per stabilità.
        """
        self.weight_compactness = weight_compactness
        self.weight_boundary = weight_boundary
        self.weight_eccentricity = weight_eccentricity

        self.last_losses = {}

        # Cache per le griglie di coordinate.
        # Chiave: (H, W, device, dtype) — così si ricalcolano solo se cambia dimensione o device.
        # Usata sia da _vectorized_eccentricity_loss che da _vectorized_circle_template_loss.
        # Evita di ricreare tensori su GPU a ogni forward pass (miglioramento efficienza).
        self._coord_cache: dict = {}

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
        min_area_threshold = MIN_AREA_THRESHOLD  # Area minima per considerare la predizione valida
        
        # Se tutti i batch hanno area troppo piccola, ritorna 0.0 mantenendo computational graph
        if (area_per_batch < min_area_threshold).all():
            return pred_soft.sum() * 0.0

        # Calcola loss components (vettorizzate) con protezioni
        # Se area è troppo piccola, le loss ritornano 0.0 mantenendo computational graph
        loss_compact = self._vectorized_compactness_loss(pred_soft)
        loss_bound = self._vectorized_boundary_loss(pred_soft)
        # V2.8: Circle Template Loss invece di _vectorized_eccentricity_loss
        loss_eccentricity = self._vectorized_circle_template_loss(pred_soft)

        # CHECK NaN: Se una loss è NaN, la setta a 0 MANTENENDO computational graph
        # IMPORTANTE: Usiamo pred_soft.sum() * 0.0 invece di torch.tensor(0.0)
        # per mantenere il computational graph e permettere gradient flow
        if torch.isnan(loss_compact) or torch.isinf(loss_compact):
            loss_compact = pred_soft.sum() * 0.0  # Mantiene computational graph!
        if torch.isnan(loss_bound) or torch.isinf(loss_bound):
            loss_bound = pred_soft.sum() * 0.0
        if torch.isnan(loss_eccentricity) or torch.isinf(loss_eccentricity):
            loss_eccentricity = pred_soft.sum() * 0.0

        # Logging
        self.last_losses = {
            'compactness': loss_compact.item(),
            'boundary': loss_bound.item(),
            'eccentricity': loss_eccentricity.item()  # ora = circle template loss
        }

        # Area ramp: scala la loss combinata in base all'area media del batch.
        #
        # PROBLEMA che risolve: le singole loss restituiscono 0.0 quando
        # area < MIN_AREA_THRESHOLD (50px) — zona morta dove la geometric loss
        # non contribuisce e non aiuta a far crescere la predizione.
        # Ma nella fascia intermedia (50–300px) la geometric loss è attiva e
        # può combattere il Dice gradient che vuole espandere la predizione.
        # La ramp crea una transizione morbida:
        #   area <  50px: area_scale ≈ 0  (geometric loss quasi silenziosa)
        #   area = 150px: area_scale = 0.5  (peso dimezzato)
        #   area > 300px: area_scale = 1.0  (peso pieno)
        #
        # In questo modo la geometric loss non si oppone mai alla crescita della
        # predizione quando il cerchio non è ancora stato trovato correttamente.
        TARGET_AREA = 6.0 * min_area_threshold  # 300px con default MIN_AREA=50
        area_scale = (area_per_batch / TARGET_AREA).clamp(0.0, 1.0).mean()

        # Combinazione con area ramp
        total = (self.weight_compactness * loss_compact +
                self.weight_boundary * loss_bound +
                self.weight_eccentricity * loss_eccentricity) * area_scale

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
        # V2.6: sqrt(clamp(x, min=0) + 1e-2) invece di clamp(x, min=1e-2) + sqrt(x).
        # La differenza è cruciale:
        #   - Vecchio (V2.4): clamp crea zona morta [0, 1e-2] dove gradiente = 0
        #   - Nuovo (V2.6):   gradiente = 1/(2*sqrt(x+1e-2)), sempre non-zero
        #                     Gradiente max a x=0: 1/(2*sqrt(1e-2)) ≈ 5  (stesso bound di V2.4)
        # Il clamp(min=0) prima dell'epsilon previene valori negativi da errori numerici.
        grad_mag_squared = grad_x**2 + grad_y**2
        grad_mag = torch.sqrt(torch.clamp(grad_mag_squared, min=0.0) + 1e-2)

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

        # Maschera soft: usa pred_soft direttamente come peso invece di una soglia netta.
        # Con soglia hard (>0.1) il gradiente era zero per tutti i pixel sotto threshold.
        # Con pred_soft il gradiente fluisce ovunque, pesato per la confidenza della predizione.
        # Il Laplacian individua già i bordi da solo, quindi non serve un taglio netto.
        mask_active = pred_soft

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

    def _vectorized_eccentricity_loss(self, pred_soft: torch.Tensor) -> torch.Tensor:
        """
        Eccentricity loss usando momenti di inerzia (fully differentiable).

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

        # Coordinate grids con lazy cache: ricalcolate solo se H, W, device o dtype cambiano.
        # expand() crea una view (non copia memoria), quindi il costo in cache è minimo.
        cache_key = (H, W, pred_soft.device, pred_soft.dtype)
        if cache_key not in self._coord_cache:
            y_base = torch.arange(H, dtype=pred_soft.dtype, device=pred_soft.device)
            x_base = torch.arange(W, dtype=pred_soft.dtype, device=pred_soft.device)
            self._coord_cache[cache_key] = (
                y_base.view(1, H, 1).expand(B, H, W),
                x_base.view(1, 1, W).expand(B, H, W),
            )
        y_coords, x_coords = self._coord_cache[cache_key]

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

        # V2.6: sqrt(clamp(discriminant, min=0) + 1e-2) elimina la zona morta del gradiente.
        # Il discriminant può essere negativamente leggermente negativo per errori numerici
        # (es. det leggermente sovrastimato): il clamp(min=0) lo gestisce prima dell'epsilon.
        # Gradiente max: 1/(2*sqrt(1e-2)) ≈ 5, stesso bound di stabilità di V2.4.
        discriminant = trace**2 - 4*det
        discriminant = torch.clamp(discriminant, max=1e12)  # solo upper bound, lower gestito da epsilon
        sqrt_term = torch.sqrt(torch.clamp(discriminant, min=0.0) + 1e-2)
        
        lambda1 = (trace + sqrt_term) / 2 + 1e-2  # Maggiore - epsilon più grande
        lambda2 = (trace - sqrt_term) / 2 + 1e-2  # Minore - epsilon più grande

        # PROTEZIONE: Assicura che lambda2 non sia troppo piccolo e che lambda1 >= lambda2
        lambda1 = torch.clamp(lambda1, min=0.1, max=1e6)
        lambda2 = torch.clamp(lambda2, min=0.1, max=1e6)
        # Assicura lambda2 <= lambda1 (usa min per forzare upper bound)
        lambda2 = torch.min(lambda2, lambda1)

        # ASPECT LOSS V2.5: Ottimizza direttamente l'eccentricità al quadrato
        # Formula eccentricity: e = sqrt(1 - (lambda_min/lambda_max)²)
        # Per penalizzare di più, usiamo: L = (1 - lambda_min/lambda_max)²
        # Questo è equivalente a e² ma più stabile numericamente (no sqrt)

        # Calcola rapporto asse minore/maggiore (inverso di aspect ratio)
        ratio_min_max = lambda2 / (lambda1 + 1e-6)  # [0, 1], 1 = cerchio perfetto

        # PROTEZIONE: Clampa tra 0 e 1
        ratio_min_max = torch.clamp(ratio_min_max, min=0.0, max=1.0)

        # Loss: (1 - ratio)² penalizza quadraticamente le deviazioni dalla circolarità
        # - Cerchio perfetto: ratio=1 → loss=0
        # - Leggera ellisse: ratio=0.9 → loss=0.01 (era 0.048 con formula vecchia)
        # - Ellisse media: ratio=0.8 → loss=0.04 (era 0.111)
        # - Ellisse forte: ratio=0.5 → loss=0.25 (era 0.333)
        loss_per_batch = (1.0 - ratio_min_max) ** 2
        
        # Media solo sui batch validi
        if valid_mask.all():
            loss = loss_per_batch.mean()
        else:
            loss = loss_per_batch[valid_mask].mean() if valid_mask.any() else pred_soft.sum() * 0.0

        # SAFETY CHECK finale: Se loss è NaN, ritorna 0.0
        if torch.isnan(loss) or torch.isinf(loss):
            return pred_soft.sum() * 0.0

        return loss

    def _vectorized_circle_template_loss(self, pred_soft: torch.Tensor) -> torch.Tensor:
        """
        Circle Template Loss (V2.8): penalizza la deviazione dalla forma circolare ideale.

        APPROCCIO:
          1. Stop-gradient su centroide e raggio: calcola il "cerchio atteso" dalla
             predizione corrente (stesso centroide, r = sqrt(area/π)) senza propagare
             gradienti attraverso queste quantità.
          2. Costruisce template circolare: sigmoid(k · (r − dist_from_center))
             con k=5 → bordo quasi-binario, coerente con predizioni confident.
          3. Loss = MSE(pred_soft, template):
             - Pixel dentro l'ellisse ma fuori dal cerchio → diff > 0 → riduce pred  (rimuove angoli)
             - Pixel fuori dall'ellisse ma dentro il cerchio → diff < 0 → aumenta pred (riempie poli)
             - Pixel dentro il cerchio e dentro la predizione → diff ≈ 0 → nessun push

        VANTAGGIO vs eccentricity loss (momenti di inerzia):
          - La vecchia formula (1-ratio)² ha gradiente ~0.1 × piccolo per ratio≈0.95
            → la loss è praticamente inattiva su predizioni quasi-circolari.
          - Circle Template Loss ha |diff| ≈ 0.5–1.0 sui pixel "fuori posto"
            → gradienti 50-100× più grandi e direzionalmente corretti.

        Args:
            pred_soft: [B, H, W] probabilità soft (canale foreground)

        Returns:
            Loss scalare differenziabile (0.0 se NaN o area < threshold)
        """
        B, H, W = pred_soft.shape

        area_check = pred_soft.sum(dim=(1, 2))  # [B]
        min_area_threshold = MIN_AREA_THRESHOLD
        valid_mask = area_check >= min_area_threshold  # [B] boolean

        if not valid_mask.any():
            return pred_soft.sum() * 0.0

        # Lazy cache per griglie coordinate (condivisa con eccentricity)
        cache_key = (H, W, pred_soft.device, pred_soft.dtype)
        if cache_key not in self._coord_cache:
            y_base = torch.arange(H, dtype=pred_soft.dtype, device=pred_soft.device)
            x_base = torch.arange(W, dtype=pred_soft.dtype, device=pred_soft.device)
            self._coord_cache[cache_key] = (
                y_base.view(1, H, 1).expand(B, H, W),
                x_base.view(1, 1, W).expand(B, H, W),
            )
        y_coords, x_coords = self._coord_cache[cache_key]

        # Stop gradient: centroide e raggio sono il "target" fisso, non si aggiornano
        # attraverso il template — solo pred_soft riceve gradiente via MSE.
        with torch.no_grad():
            area = area_check.clamp(min=min_area_threshold)           # [B]
            cx = (pred_soft * x_coords).sum(dim=(1, 2)) / area        # [B]
            cy = (pred_soft * y_coords).sum(dim=(1, 2)) / area        # [B]
            r  = torch.sqrt(area / math.pi).clamp(min=1.0, max=max(H, W) / 2.0)
            cx = cx.view(B, 1, 1)
            cy = cy.view(B, 1, 1)
            r  = r.view(B, 1, 1)

        # Distanza euclidea dal centroide (epsilon per stabilità numerica)
        dx   = x_coords - cx                                          # [B, H, W]
        dy   = y_coords - cy
        dist = torch.sqrt(dx**2 + dy**2 + 1e-2)                      # [B, H, W]

        # Template circolare con bordo netto (k=5 ≈ bordo quasi-binario)
        # sigmoid(k*(r-dist)): ≈1 dentro il cerchio, ≈0 fuori
        TEMPLATE_SHARPNESS = 5.0
        soft_circle = torch.sigmoid(TEMPLATE_SHARPNESS * (r - dist)) # [B, H, W]

        # MSE tra predizione e template normalizzata per area del template.
        #
        # PERCHÉ: mediare su tutti i pixel H*W (512×512=262144) diluisce il segnale
        # 50-100× perché il cerchio occupa solo il 3-8% dell'immagine.
        # Normalizzare per l'area del template (∑soft_circle) concentra la loss
        # sui pixel rilevanti e rende il peso scale-invariante alla dimensione del cerchio.
        # Con questa normalizzazione WEIGHT_ECCENTRICITY=0.05 produce un gradiente
        # comparabile a WEIGHT_COMPACTNESS=0.03 (entrambi ~0.003-0.005 sul totale).
        diff_sq = (pred_soft - soft_circle) ** 2                      # [B, H, W]
        template_area = soft_circle.sum(dim=(1, 2)).clamp(min=1.0)    # [B]
        loss_per_batch = diff_sq.sum(dim=(1, 2)) / template_area      # [B]

        if valid_mask.all():
            loss = loss_per_batch.mean()
        else:
            loss = loss_per_batch[valid_mask].mean() if valid_mask.any() else pred_soft.sum() * 0.0

        if torch.isnan(loss) or torch.isinf(loss):
            return pred_soft.sum() * 0.0

        return loss

    def get_last_losses(self) -> dict:
        """Ritorna componenti loss."""
        return self.last_losses.copy()


class GeometricLosses:
    """
    Interfaccia principale per le loss geometriche del trainer.

    3 termini: compactness, boundary, eccentricity.
    """

    def __init__(self,
                 weight_compactness: float = WEIGHT_COMPACTNESS,
                 weight_eccentricity: float = WEIGHT_ECCENTRICITY,
                 weight_boundary: float = WEIGHT_BOUNDARY,
                 min_area: int = 10):
        """
        Args:
            weight_compactness: Peso per compactness loss (area vs perimetro²)
            weight_eccentricity: Peso per eccentricity loss (momenti di inerzia)
            weight_boundary: Peso per boundary smoothness loss
            min_area: Area minima (mantenuto per compatibilità)
        """
        self.weight_compactness = weight_compactness
        self.weight_eccentricity = weight_eccentricity
        self.weight_boundary = weight_boundary
        self.min_area = min_area

        self._v2_loss = DifferentiableGeometricLossesV2(
            weight_compactness=weight_compactness,
            weight_boundary=weight_boundary,
            weight_eccentricity=weight_eccentricity
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

        v2_losses = self._v2_loss.get_last_losses()
        self.last_losses = {
            'compactness': v2_losses.get('compactness', 0.0),
            'boundary': v2_losses.get('boundary', 0.0),
            'eccentricity': v2_losses.get('eccentricity', 0.0),
        }

        return loss

    def get_last_losses(self) -> dict:
        """Ritorna componenti loss."""
        return self.last_losses.copy()


# Test
if __name__ == "__main__":
    print("=" * 80)
    print("TEST: Differentiable Geometric Losses V2.8 (3 termini: compactness, boundary, circle_template)")
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
        weight_compactness=WEIGHT_COMPACTNESS,
        weight_eccentricity=WEIGHT_ECCENTRICITY,
        weight_boundary=WEIGHT_BOUNDARY
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
