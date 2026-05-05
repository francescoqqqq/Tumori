import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import os
import random
import math
import json

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_FOLDER    = "imagesTr"
LABEL_FOLDER  = "labelsTr"
METADATA_FILE = "metadata.json"

NUM_IMAGES = 500
IMG_SIZE   = (512, 512)

# PARAMETRI MULTI-CIRCLE
MIN_CIRCLES    = 2
MAX_CIRCLES    = 5
MAX_DISTRACTORS = 15

# PARAMETRI SINGLE-CIRCLE
SINGLE_CIRCLE_MIN_RADIUS = 20
SINGLE_CIRCLE_MAX_RADIUS = 80
SINGLE_CIRCLE_MARGIN     = 100   # > SINGLE_CIRCLE_MAX_RADIUS → cerchio sempre interno

# PARAMETRI DIFFICOLTÀ
MIN_OCCLUSIONS    = 5 #2
MAX_OCCLUSIONS    = 8 #5
OCCLUSION_MIN_SIZE = 30 #40
OCCLUSION_MAX_SIZE = 90 #120
NOISE_LEVEL       = 30
BLUR_KERNEL       = 3    # deve essere dispari (validato in _validate_params)

# Range colore piatto condiviso tra cerchi "uguale" e distrattori
_COLOR_RANGE = (180, 240)

_VALID_COLOR_STYLES = {"differente", "uguale", "identico"}
_VALID_CIRCLE_ALONE = {"si", "no"}


# ──────────────────────────────────────────────────────────────────────────────
#  FUNZIONI DI SUPPORTO
# ──────────────────────────────────────────────────────────────────────────────

def _validate_params(color_style: str, blur_kernel: int, circle_alone: str) -> None:
    """Valida i parametri prima della generazione."""
    if color_style not in _VALID_COLOR_STYLES:
        raise ValueError(
            f"color_style='{color_style}' non valido. "
            f"Usa uno tra: {_VALID_COLOR_STYLES}"
        )
    if blur_kernel > 0 and blur_kernel % 2 == 0:
        raise ValueError(
            f"BLUR_KERNEL={blur_kernel} deve essere dispari (es. 1, 3, 5) "
            f"per cv2.GaussianBlur."
        )
    if circle_alone not in _VALID_CIRCLE_ALONE:
        raise ValueError(
            f"circle_alone='{circle_alone}' non valido. "
            f"Usa uno tra: {_VALID_CIRCLE_ALONE}"
        )


def get_polygon_coords(center, radius, n_sides, angle_deg):
    """Calcola i vertici di un poligono regolare ruotato."""
    angle_rad = math.radians(angle_deg)
    points = [
        [
            int(center[0] + radius * math.cos(angle_rad + 2 * math.pi * i / n_sides)),
            int(center[1] + radius * math.sin(angle_rad + 2 * math.pi * i / n_sides)),
        ]
        for i in range(n_sides)
    ]
    return np.array([points], dtype=np.int32)


def add_gradient_to_circle(img, center, radius, base_color):
    """
    Riempie un cerchio con un gradiente radiale (più luminoso al centro).
    Usato in color_style="differente" per distinguere i cerchi dai poligoni piatti.

    Args:
        center: (row, col) — convenzione numpy (y, x)
    """
    cy, cx = center
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    circle_mask = dist <= radius
    gradient = np.clip(base_color - (dist / radius) * 50, 150, 255)
    img[circle_mask] = np.maximum(img[circle_mask], gradient[circle_mask].astype(np.uint8))
    return img


def _draw_circle(img, mask, cx, cy, radius, color_style, shared_color=None):
    """
    Disegna un cerchio su img (con stile colore scelto) e sulla mask (sempre binario).
    """
    if color_style == "uguale":
        flat_color = random.randint(*_COLOR_RANGE)
        cv2.circle(img, (cx, cy), radius, flat_color, -1)
    elif color_style == "identico":
        if shared_color is None:
            raise ValueError("shared_color deve essere valorizzato quando color_style='identico'")
        cv2.circle(img, (cx, cy), radius, shared_color, -1)
    else:
        base_color = random.randint(200, 255)
        img = add_gradient_to_circle(img, (cy, cx), radius, base_color)

    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return img


def _distance_point_to_segment(px, py, ax, ay, bx, by):
    """Distanza minima punto-segmento."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _poly_overlaps_any_circle(pts, circles_info):
    """True se il poligono interseca almeno un cerchio target."""
    poly = pts[0]
    n = len(poly)
    for circle in circles_info:
        cx, cy = circle["center"]
        radius = circle["radius"]

        # Centro cerchio dentro il poligono
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            return True

        # Vertice del poligono dentro/ai bordi del cerchio
        for vx, vy in poly:
            if math.hypot(float(vx) - cx, float(vy) - cy) <= radius:
                return True

        # Intersezione lato-poligono con bordo cerchio
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            if _distance_point_to_segment(cx, cy, float(ax), float(ay), float(bx), float(by)) <= radius:
                return True
    return False


def _add_distractors(img, img_size, color_style, shared_color=None, circles_info=None, avoid_circle_overlap=False):
    """Disegna poligoni di disturbo sull'immagine. Ritorna il conteggio."""
    num = random.randint(0, MAX_DISTRACTORS)
    drawn = 0
    max_attempts = 80
    circles = circles_info if circles_info is not None else []

    for _ in range(num):
        placed = False
        for _ in range(max_attempts):
            cx    = random.randint(50, img_size[1] - 50)
            cy    = random.randint(50, img_size[0] - 50)
            r     = random.randint(20, 60)
            angle = random.randint(0, 360)
            sides = random.choice([3, 4, 5, 6])
            pts   = get_polygon_coords((cx, cy), r, sides, angle)

            if avoid_circle_overlap and _poly_overlaps_any_circle(pts, circles):
                continue

            if color_style == "identico":
                if shared_color is None:
                    raise ValueError("shared_color deve essere valorizzato quando color_style='identico'")
                color = shared_color
            else:
                color = random.randint(*_COLOR_RANGE)
            cv2.fillPoly(img, pts, color=color)
            drawn += 1
            placed = True
            break

        if not placed:
            continue
    return drawn


def _add_occlusions(img, img_size):
    """Aggiunge rettangoli neri di occlusione solo all'immagine. Ritorna la lista."""
    num  = random.randint(MIN_OCCLUSIONS, MAX_OCCLUSIONS)
    info = []
    for _ in range(num):
        x1 = random.randint(0, img_size[1] - OCCLUSION_MAX_SIZE)
        y1 = random.randint(0, img_size[0] - OCCLUSION_MAX_SIZE)
        w  = random.randint(OCCLUSION_MIN_SIZE, OCCLUSION_MAX_SIZE)
        h  = random.randint(OCCLUSION_MIN_SIZE, OCCLUSION_MAX_SIZE)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 0, -1)
        info.append({'x': int(x1), 'y': int(y1), 'width': int(w), 'height': int(h)})
    return info


def _add_noise(img):
    """Applica rumore gaussiano, salt&pepper e blur gaussiano."""
    noise = np.random.normal(0, NOISE_LEVEL, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    sp = np.random.rand(*img.shape)
    img[sp < 0.01] = 255
    img[sp > 0.99] = 0

    if BLUR_KERNEL > 0:
        img = cv2.GaussianBlur(img, (BLUR_KERNEL, BLUR_KERNEL), 0)
    return img


def _circles_overlap(cx, cy, radius, circles_info, min_gap):
    """Ritorna True se il nuovo cerchio collide con quelli già piazzati."""
    for circle in circles_info:
        ex, ey = circle['center']
        er = circle['radius']
        min_dist = radius + er + min_gap
        if math.hypot(cx - ex, cy - ey) < min_dist:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  GENERATORE COMUNE  (usato da entrambe le modalità)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_images(output_dir, num_images, img_size, color_style, circle_alone, circles_fn):
    """
    Genera num_images immagini applicando la pipeline completa.

    Args:
        circles_fn: callable(img, mask, img_size, color_style, shared_color)
                    Disegna i cerchi target e ritorna (img, circles_info).
                    Tutta la logica specifica di modalità è qui.
    """
    os.makedirs(os.path.join(output_dir, IMG_FOLDER),   exist_ok=True)
    os.makedirs(os.path.join(output_dir, LABEL_FOLDER), exist_ok=True)

    metadata = []

    for i in range(num_images):
        img  = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        shared_color = random.randint(*_COLOR_RANGE) if color_style == "identico" else None

        enforce_circle_alone = (circle_alone == "si" and color_style == "identico")
        if enforce_circle_alone:
            img, circles_info = circles_fn(img, mask, img_size, color_style, shared_color)
            num_distractors = _add_distractors(
                img,
                img_size,
                color_style,
                shared_color,
                circles_info=circles_info,
                avoid_circle_overlap=True,
            )
        else:
            num_distractors = _add_distractors(img, img_size, color_style, shared_color)
            img, circles_info = circles_fn(img, mask, img_size, color_style, shared_color)
        occlusions_info   = _add_occlusions(img, img_size)
        img               = _add_noise(img)

        filename  = f"shape_{i:04d}.png"
        cv2.imwrite(os.path.join(output_dir, IMG_FOLDER,   filename), img)
        cv2.imwrite(os.path.join(output_dir, LABEL_FOLDER, filename), mask)

        metadata.append({
            'filename':        filename,
            'num_circles':     len(circles_info),
            'circles':         circles_info,
            'num_occlusions':  len(occlusions_info),
            'occlusions':      occlusions_info,
            'num_distractors': num_distractors,
            'color_style':     color_style,
            'circle_alone':    circle_alone,
        })

        if (i + 1) % 50 == 0:
            print(f"  Generati {i + 1}/{num_images} immagini...")

    with open(os.path.join(output_dir, METADATA_FILE), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ──────────────────────────────────────────────────────────────────────────────
#  API PUBBLICA
# ──────────────────────────────────────────────────────────────────────────────

def generate_single_circle_dataset(output_dir, num_images=None, img_size=None,
                                   color_style="differente", circle_alone="no", seed=None):
    """
    Genera dataset con UN SOLO cerchio per immagine.

    Args:
        output_dir:   Cartella di output.
        num_images:   Numero immagini. Default: NUM_IMAGES.
        img_size:     Tupla (H, W). Default: IMG_SIZE.
        color_style:  "differente" (gradiente) | "uguale" (piatto, no shortcut) |
                      "identico" (tutte le shape della stessa immagine stesso colore).
        circle_alone: "si" evita overlap distrattori/cerchio quando color_style="identico".
                      "no" mantiene il comportamento storico.
        seed:         Seed per riproducibilità. None = non deterministico.
    """
    _num_images = num_images if num_images is not None else NUM_IMAGES
    _img_size   = img_size   if img_size   is not None else IMG_SIZE
    _validate_params(color_style, BLUR_KERNEL, circle_alone)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"MODALITÀ: SINGLE-CIRCLE")
    print(f"{'='*60}")
    print(f"Immagini: {_num_images}  |  Size: {_img_size}  |  Color: {color_style}")
    if seed is not None:
        print(f"Seed: {seed}")

    def _single_circle_fn(img, mask, img_size, color_style, shared_color):
        radius = random.randint(SINGLE_CIRCLE_MIN_RADIUS, SINGLE_CIRCLE_MAX_RADIUS)
        # Il margine è >= SINGLE_CIRCLE_MAX_RADIUS → cerchio sempre completamente interno
        margin = max(SINGLE_CIRCLE_MARGIN, radius)
        cx = random.randint(margin, img_size[1] - margin)
        cy = random.randint(margin, img_size[0] - margin)
        img = _draw_circle(img, mask, cx, cy, radius, color_style, shared_color)
        return img, [{'center': [int(cx), int(cy)], 'radius': int(radius)}]

    _generate_images(output_dir, _num_images, _img_size, color_style, circle_alone, _single_circle_fn)

    print(f"\n✓ Dataset single-circle salvato in '{output_dir}'")
    print(f"  Raggio: {SINGLE_CIRCLE_MIN_RADIUS}–{SINGLE_CIRCLE_MAX_RADIUS} px")
    print(f"  Occlusioni: {MIN_OCCLUSIONS}–{MAX_OCCLUSIONS}  |  Rumore: {NOISE_LEVEL}")


def generate_multi_circle_dataset(output_dir, num_images=None, img_size=None,
                                  color_style="differente", circle_alone="no", seed=None):
    """
    Genera dataset con numero VARIABILE di cerchi per immagine (MIN–MAX).

    Args:
        output_dir:   Cartella di output.
        num_images:   Numero immagini. Default: NUM_IMAGES.
        img_size:     Tupla (H, W). Default: IMG_SIZE.
        color_style:  "differente" (gradiente) | "uguale" (piatto, no shortcut) |
                      "identico" (tutte le shape della stessa immagine stesso colore).
        circle_alone: "si" evita overlap distrattori/cerchi quando color_style="identico".
                      "no" mantiene il comportamento storico.
        seed:         Seed per riproducibilità. None = non deterministico.
    """
    _num_images = num_images if num_images is not None else NUM_IMAGES
    _img_size   = img_size   if img_size   is not None else IMG_SIZE
    _validate_params(color_style, BLUR_KERNEL, circle_alone)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"MODALITÀ: MULTI-CIRCLE ({MIN_CIRCLES}–{MAX_CIRCLES} cerchi)")
    print(f"{'='*60}")
    print(f"Immagini: {_num_images}  |  Size: {_img_size}  |  Color: {color_style}")
    if seed is not None:
        print(f"Seed: {seed}")

    def _multi_circle_fn(img, mask, img_size, color_style, shared_color):
        num_circles  = random.randint(MIN_CIRCLES, MAX_CIRCLES)
        circles_info = []
        min_gap = 4
        max_attempts = 100

        for _ in range(num_circles):
            placed = False
            for _ in range(max_attempts):
                radius = random.randint(25, 55)
                # Margine = raggio + buffer minimo → cerchio sempre completamente interno
                margin = radius + min_gap
                cx = random.randint(margin, img_size[1] - margin)
                cy = random.randint(margin, img_size[0] - margin)

                if _circles_overlap(cx, cy, radius, circles_info, min_gap):
                    continue

                img = _draw_circle(img, mask, cx, cy, radius, color_style, shared_color)
                circles_info.append({'center': [int(cx), int(cy)], 'radius': int(radius)})
                placed = True
                break

            if not placed:
                break
        return img, circles_info

    _generate_images(output_dir, _num_images, _img_size, color_style, circle_alone, _multi_circle_fn)

    print(f"\n✓ Dataset multi-circle salvato in '{output_dir}'")
    print(f"  Cerchi: {MIN_CIRCLES}–{MAX_CIRCLES}  |  Occlusioni: {MIN_OCCLUSIONS}–{MAX_OCCLUSIONS}")
    print(f"  Rumore: {NOISE_LEVEL}  |  Color: {color_style}")


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT MANUALE (legacy)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Menu interattivo — usa run_experiment.py per il workflow automatico."""
    print("\n" + "="*60)
    print("GENERATORE DATASET CERCHI")
    print("="*60)
    print("\n  [1] Single-circle (1 cerchio per immagine)")
    print("  [2] Multi-circle  (2-5 cerchi per immagine)")
    print()

    while True:
        choice = input("Scelta (1 o 2): ").strip()
        if choice == "1":
            generate_single_circle_dataset(os.path.join(BASE_DIR, "dataset_shapes_one"))
            break
        elif choice == "2":
            generate_multi_circle_dataset(os.path.join(BASE_DIR, "dataset_shapes"))
            break
        else:
            print("Scelta non valida.")


if __name__ == "__main__":
    main()
