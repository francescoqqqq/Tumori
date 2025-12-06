"""
Loss functions geometriche per segmentazione di cerchi.

Implementa penalità per forme non circolari:
- Compactness: quanto la forma è simile a un cerchio
- Solidity: quanto la forma riempie il suo convex hull
- Eccentricity: quanto la forma è ellittica vs circolare
- Boundary Smoothness: quanto i bordi sono regolari

Author: Francesco + Claude
Date: 2025-12-05
"""

import torch
import numpy as np
from scipy import ndimage
from skimage import measure
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class GeometricLosses:
    """
    Collezione di loss functions per vincoli geometrici su cerchi.

    Combina diverse metriche di forma per penalizzare predizioni che non
    rispettano le proprietà geometriche dei cerchi.
    """

    def __init__(self,
                 weight_compactness: float = 0.1,
                 weight_solidity: float = 0.1,
                 weight_eccentricity: float = 0.05,
                 weight_boundary: float = 0.05,
                 min_area: int = 10):
        """
        Inizializza le loss geometriche.

        Args:
            weight_compactness: Peso per compactness loss (circolarità)
            weight_solidity: Peso per solidity loss (convessità)
            weight_eccentricity: Peso per eccentricity loss (ellitticità)
            weight_boundary: Peso per boundary smoothness loss (bordi regolari)
            min_area: Area minima (pixels) per considerare una componente
        """
        self.weight_compactness = weight_compactness
        self.weight_solidity = weight_solidity
        self.weight_eccentricity = weight_eccentricity
        self.weight_boundary = weight_boundary
        self.min_area = min_area

        # Per logging
        self.last_losses = {}

    def __call__(self, pred_softmax: torch.Tensor) -> torch.Tensor:
        """
        Calcola loss geometrica totale.

        Args:
            pred_softmax: Predizione dopo softmax [B, C, H, W]
                         C=2 (background + cerchi)

        Returns:
            Loss geometrica totale (scalare)
        """
        # Estrai canale cerchi (classe 1) e binarizza
        # pred_softmax[:, 1] = probabilità dei cerchi
        pred_binary = (pred_softmax[:, 1, :, :] > 0.5).float()

        # Calcola componenti loss
        loss_compact = self._compactness_loss(pred_binary)
        loss_solid = self._solidity_loss(pred_binary)
        loss_eccent = self._eccentricity_loss(pred_binary)
        loss_bound = self._boundary_smoothness_loss(pred_binary)

        # Salva per logging
        self.last_losses = {
            'compactness': loss_compact.item(),
            'solidity': loss_solid.item(),
            'eccentricity': loss_eccent.item(),
            'boundary': loss_bound.item()
        }

        # Combinazione pesata
        total_loss = (
            self.weight_compactness * loss_compact +
            self.weight_solidity * loss_solid +
            self.weight_eccentricity * loss_eccent +
            self.weight_boundary * loss_bound
        )

        return total_loss

    def _compactness_loss(self, pred_binary: torch.Tensor) -> torch.Tensor:
        """
        Calcola compactness loss (circolarità).

        Compactness = (4π · Area) / (Perimetro²)
        Per cerchio perfetto: C = 1.0
        Per forme irregolari: C < 1.0

        Loss = 1 - Compactness (quindi 0 per cerchio perfetto)

        Args:
            pred_binary: Tensor [B, H, W] con valori {0, 1}

        Returns:
            Loss scalare (media su batch)
        """
        batch_size = pred_binary.shape[0]
        losses = []

        for b in range(batch_size):
            mask = pred_binary[b].cpu().numpy().astype(np.uint8)

            # Trova componenti connesse (cerchi separati)
            labeled, num_components = ndimage.label(mask)

            for i in range(1, num_components + 1):
                component = (labeled == i).astype(np.uint8)

                # Calcola area
                area = np.sum(component)

                if area < self.min_area:
                    continue

                # Calcola perimetro usando contorni
                try:
                    contours = measure.find_contours(component, 0.5)
                    if len(contours) == 0:
                        continue

                    # Somma lunghezze di tutti i contorni
                    perimeter = sum(len(c) for c in contours)

                    if perimeter > 0:
                        compactness = (4 * np.pi * area) / (perimeter ** 2)
                        # Clamp a 1.0 per evitare valori > 1 dovuti a discretizzazione
                        compactness = min(compactness, 1.0)
                        loss = 1.0 - compactness
                        losses.append(loss)
                except Exception:
                    # In caso di errore, salta questa componente
                    continue

        # Media delle loss o 0 se nessuna componente valida
        mean_loss = np.mean(losses) if len(losses) > 0 else 0.0
        return torch.tensor(mean_loss, dtype=torch.float32, device=pred_binary.device)

    def _solidity_loss(self, pred_binary: torch.Tensor) -> torch.Tensor:
        """
        Calcola solidity loss (convessità).

        Solidity = Area / Area_ConvexHull
        Per cerchio perfetto: S ≈ 1.0
        Per forme con concavità: S < 1.0

        Loss = 1 - Solidity

        Args:
            pred_binary: Tensor [B, H, W] con valori {0, 1}

        Returns:
            Loss scalare (media su batch)
        """
        batch_size = pred_binary.shape[0]
        losses = []

        for b in range(batch_size):
            mask = pred_binary[b].cpu().numpy().astype(np.uint8)

            # Trova componenti connesse
            labeled, num_components = ndimage.label(mask)

            for i in range(1, num_components + 1):
                component = (labeled == i).astype(np.uint8)

                # Controlla area minima
                area = np.sum(component)
                if area < self.min_area:
                    continue

                try:
                    # Usa skimage.measure.regionprops per solidity
                    props = measure.regionprops(component)
                    if len(props) > 0 and hasattr(props[0], 'solidity'):
                        solidity = props[0].solidity
                        loss = 1.0 - solidity
                        losses.append(loss)
                except Exception:
                    continue

        mean_loss = np.mean(losses) if len(losses) > 0 else 0.0
        return torch.tensor(mean_loss, dtype=torch.float32, device=pred_binary.device)

    def _eccentricity_loss(self, pred_binary: torch.Tensor) -> torch.Tensor:
        """
        Calcola eccentricity loss (ellitticità).

        Eccentricity = √(1 - (minor_axis/major_axis)²)
        Per cerchio perfetto: E ≈ 0.0
        Per ellisse allungata: E → 1.0

        Loss = Eccentricity (direttamente)

        Args:
            pred_binary: Tensor [B, H, W] con valori {0, 1}

        Returns:
            Loss scalare (media su batch)
        """
        batch_size = pred_binary.shape[0]
        losses = []

        for b in range(batch_size):
            mask = pred_binary[b].cpu().numpy().astype(np.uint8)

            # Trova componenti connesse
            labeled, num_components = ndimage.label(mask)

            for i in range(1, num_components + 1):
                component = (labeled == i).astype(np.uint8)

                # Controlla area minima
                area = np.sum(component)
                if area < self.min_area:
                    continue

                try:
                    # Usa skimage.measure.regionprops per eccentricity
                    props = measure.regionprops(component)
                    if len(props) > 0 and hasattr(props[0], 'eccentricity'):
                        eccentricity = props[0].eccentricity
                        losses.append(eccentricity)
                except Exception:
                    continue

        mean_loss = np.mean(losses) if len(losses) > 0 else 0.0
        return torch.tensor(mean_loss, dtype=torch.float32, device=pred_binary.device)

    def _boundary_smoothness_loss(self, pred_binary: torch.Tensor) -> torch.Tensor:
        """
        Calcola boundary smoothness loss (regolarità bordi).

        Penalizza bordi frastagliati calcolando la variazione della curvatura.

        Loss = Var(curvature) + 0.1 · Mean(|curvature|)

        Args:
            pred_binary: Tensor [B, H, W] con valori {0, 1}

        Returns:
            Loss scalare (media su batch)
        """
        batch_size = pred_binary.shape[0]
        losses = []

        for b in range(batch_size):
            mask = pred_binary[b].cpu().numpy().astype(np.uint8)

            # Trova contorni
            try:
                contours = measure.find_contours(mask, 0.5)
            except Exception:
                continue

            for contour in contours:
                if len(contour) < 5:  # Servono almeno 5 punti
                    continue

                # Calcola curvatura discreta
                curvatures = []
                for i in range(1, len(contour) - 1):
                    p1, p2, p3 = contour[i-1], contour[i], contour[i+1]

                    # Vettori
                    v1 = p2 - p1
                    v2 = p3 - p2

                    # Lunghezze
                    len_v1 = np.linalg.norm(v1)
                    len_v2 = np.linalg.norm(v2)

                    if len_v1 < 1e-6 or len_v2 < 1e-6:
                        continue

                    # Angolo tra vettori
                    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

                    # Normalizza tra -π e π
                    while angle > np.pi:
                        angle -= 2 * np.pi
                    while angle < -np.pi:
                        angle += 2 * np.pi

                    curvatures.append(np.abs(angle))

                if len(curvatures) > 1:
                    # Loss = varianza + 0.1 * media valore assoluto
                    var_loss = np.var(curvatures)
                    mean_loss = np.mean(curvatures)
                    loss = var_loss + 0.1 * mean_loss
                    losses.append(loss)

        mean_loss = np.mean(losses) if len(losses) > 0 else 0.0
        return torch.tensor(mean_loss, dtype=torch.float32, device=pred_binary.device)

    def get_last_losses(self) -> dict:
        """Ritorna le loss individuali dell'ultima chiamata (per logging)."""
        return self.last_losses.copy()


# Test rapido
if __name__ == "__main__":
    print("Test GeometricLosses...")

    # Crea batch fittizio con un cerchio perfetto
    batch_size = 2
    img_size = 128

    # Batch con 2 canali (background + cerchi)
    pred_softmax = torch.zeros(batch_size, 2, img_size, img_size)

    # Prima immagine: cerchio perfetto
    from skimage.draw import disk
    rr, cc = disk((64, 64), 30)
    pred_softmax[0, 1, rr, cc] = 1.0  # Classe cerchi
    pred_softmax[0, 0] = 1.0 - pred_softmax[0, 1]  # Background

    # Seconda immagine: forma irregolare
    pred_softmax[1, 1, 30:90, 40:100] = 1.0  # Rettangolo
    pred_softmax[1, 0] = 1.0 - pred_softmax[1, 1]

    # Calcola loss
    geom_loss = GeometricLosses()
    loss = geom_loss(pred_softmax)

    print(f"\nTotal Loss: {loss.item():.4f}")
    print(f"Componenti loss: {geom_loss.get_last_losses()}")
    print("\n✅ Test completato!")
