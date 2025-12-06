import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import os
import random
import math
import json

# --- CONFIGURAZIONE ---
# Crea dataset_shapes dentro la cartella geometrica
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_shapes")
IMG_FOLDER = "imagesTr"
LABEL_FOLDER = "labelsTr"
METADATA_FILE = "metadata.json"

NUM_IMAGES = 500          # Più immagini per task difficile
IMG_SIZE = (512, 512)     # Dimensioni (Altezza, Larghezza)
MIN_CIRCLES = 2           # Minimo cerchi per immagine
MAX_CIRCLES = 5           # Massimo cerchi per immagine
MAX_DISTRACTORS = 15      # Massimo figure geometriche (non cerchi)

# --- PARAMETRI DI DIFFICOLTÀ ---
MIN_OCCLUSIONS = 2        # Minimo occlusioni per immagine
MAX_OCCLUSIONS = 5        # Massimo occlusioni per immagine
OCCLUSION_MIN_SIZE = 40   # Dimensione minima occlusione
OCCLUSION_MAX_SIZE = 120  # Dimensione massima occlusione
NOISE_LEVEL = 30          # Intensità rumore (0-50)
BLUR_KERNEL = 3           # Kernel per blur (3 o 5)

# Assicuriamoci che le cartelle esistano
os.makedirs(os.path.join(OUTPUT_DIR, IMG_FOLDER), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, LABEL_FOLDER), exist_ok=True)

def get_polygon_coords(center, radius, n_sides, angle_deg):
    """
    Calcola i vertici di un poligono regolare ruotato.
    """
    points = []
    angle_rad = math.radians(angle_deg)
    
    for i in range(n_sides):
        theta = angle_rad + (2 * math.pi * i / n_sides)
        x = int(center[0] + radius * math.cos(theta))
        y = int(center[1] + radius * math.sin(theta))
        points.append([x, y])
    
    return np.array([points], dtype=np.int32)

def add_gradient_to_circle(img, center, radius, base_color):
    """
    Riempie un cerchio con un gradiente radiale per renderlo meno uniforme.
    """
    cy, cx = center
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Maschera del cerchio
    circle_mask = distance_from_center <= radius
    
    # Gradiente: più luminoso al centro, più scuro ai bordi
    gradient = np.clip(base_color - (distance_from_center / radius) * 50, 150, 255)
    
    # Applica il gradiente solo dentro il cerchio
    img[circle_mask] = np.maximum(img[circle_mask], gradient[circle_mask].astype(np.uint8))
    
    return img

def generate_dataset():
    print(f"Inizio generazione di {NUM_IMAGES} immagini (DIFFICOLTÀ ELEVATA)...")
    
    metadata = []
    
    for i in range(NUM_IMAGES):
        # 1. Crea sfondi neri
        img = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)
        mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)

        circles_info = []

        # --- 2. GENERAZIONE FIGURE DI DISTURBO ---
        num_distractors = random.randint(0, MAX_DISTRACTORS)
        
        for _ in range(num_distractors):
            cx = random.randint(50, IMG_SIZE[1] - 50)
            cy = random.randint(50, IMG_SIZE[0] - 50)
            radius = random.randint(20, 60)
            angle = random.randint(0, 360)
            sides = random.choice([3, 4, 5, 6])
            
            color = random.randint(180, 240)
            
            pts = get_polygon_coords((cx, cy), radius, sides, angle)
            cv2.fillPoly(img, pts, color=color)

        # --- 3. GENERAZIONE CERCHI (Target) con gradiente ---
        num_circles = random.randint(MIN_CIRCLES, MAX_CIRCLES)
        
        for _ in range(num_circles):
            cx = random.randint(60, IMG_SIZE[1] - 60)
            cy = random.randint(60, IMG_SIZE[0] - 60)
            radius = random.randint(25, 55)
            
            base_color = random.randint(200, 255)
            
            # Disegna cerchio con gradiente nell'immagine
            img = add_gradient_to_circle(img, (cy, cx), radius, base_color)
            
            # Maschera rimane binaria (cerchio completo)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            
            circles_info.append({
                'center': [int(cx), int(cy)],
                'radius': int(radius)
            })

        # --- 4. AGGIUNTA OCCLUSIONI (solo sull'immagine!) ---
        num_occlusions = random.randint(MIN_OCCLUSIONS, MAX_OCCLUSIONS)
        occlusions_info = []
        
        for _ in range(num_occlusions):
            x1 = random.randint(0, IMG_SIZE[1] - OCCLUSION_MAX_SIZE)
            y1 = random.randint(0, IMG_SIZE[0] - OCCLUSION_MAX_SIZE)
            w = random.randint(OCCLUSION_MIN_SIZE, OCCLUSION_MAX_SIZE)
            h = random.randint(OCCLUSION_MIN_SIZE, OCCLUSION_MAX_SIZE)
            
            # Rettangolo nero che occlude i cerchi
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), 0, -1)
            
            occlusions_info.append({
                'x': int(x1), 'y': int(y1),
                'width': int(w), 'height': int(h)
            })

        # --- 5. AGGIUNTA RUMORE ---
        # Gaussian noise
        noise = np.random.normal(0, NOISE_LEVEL, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Salt and pepper noise
        salt_pepper = np.random.rand(*img.shape)
        img[salt_pepper < 0.01] = 255  # Salt
        img[salt_pepper > 0.99] = 0    # Pepper
        
        # Blur per confondere i bordi
        if BLUR_KERNEL > 0:
            img = cv2.GaussianBlur(img, (BLUR_KERNEL, BLUR_KERNEL), 0)

        # --- 6. SALVATAGGIO ---
        filename = f"shape_{i:04d}.png"
        img_path = os.path.join(OUTPUT_DIR, IMG_FOLDER, filename)
        mask_path = os.path.join(OUTPUT_DIR, LABEL_FOLDER, filename)
        
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
        
        # Salva metadata
        metadata.append({
            'filename': filename,
            'num_circles': num_circles,
            'circles': circles_info,
            'num_occlusions': num_occlusions,
            'occlusions': occlusions_info,
            'num_distractors': num_distractors
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Generati {i + 1}/{NUM_IMAGES} immagini...")

    # Salva metadata in JSON
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Completato! Dataset salvato in '{OUTPUT_DIR}'")
    print(f"✓ Metadata salvati in '{metadata_path}'")
    print(f"\n--- STATISTICHE ---")
    print(f"Immagini totali: {NUM_IMAGES}")
    print(f"Cerchi per immagine: {MIN_CIRCLES}-{MAX_CIRCLES}")
    print(f"Occlusioni per immagine: {MIN_OCCLUSIONS}-{MAX_OCCLUSIONS}")
    print(f"Livello rumore: {NOISE_LEVEL}")

if __name__ == "__main__":
    generate_dataset()