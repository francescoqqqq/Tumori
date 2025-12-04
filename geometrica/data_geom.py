import cv2  # pyright: ignore[reportMissingImports]
import numpy as np
import os
import random
import math

# --- CONFIGURAZIONE ---
OUTPUT_DIR = "dataset_shapes"
IMG_FOLDER = "imagesTr"
LABEL_FOLDER = "labelsTr"
NUM_IMAGES = 100          # Quante immagini generare
IMG_SIZE = (512, 512)     # Dimensioni (Altezza, Larghezza)
MIN_CIRCLES = 1           # Minimo cerchi per immagine
MAX_CIRCLES = 5           # Massimo cerchi per immagine
MAX_DISTRACTORS = 10      # Massimo figure geometriche (non cerchi)

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
        # Distribuiamo i punti equamente lungo la circonferenza
        theta = angle_rad + (2 * math.pi * i / n_sides)
        x = int(center[0] + radius * math.cos(theta))
        y = int(center[1] + radius * math.sin(theta))
        points.append([x, y])
    
    return np.array([points], dtype=np.int32)

def generate_dataset():
    print(f"Inizio generazione di {NUM_IMAGES} immagini...")
    
    for i in range(NUM_IMAGES):
        # 1. Crea sfondi neri
        # img: immagine input (grayscale)
        # mask: ground truth (dove 0=sfondo, 1 (o 255)=cerchio)
        img = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)
        mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)

        # --- 2. GENERAZIONE FIGURE DI DISTURBO (Triangoli, Quadrati, ecc.) ---
        # Queste vanno solo su 'img', NON su 'mask'
        num_distractors = random.randint(0, MAX_DISTRACTORS)
        
        for _ in range(num_distractors):
            # Parametri casuali
            cx = random.randint(50, IMG_SIZE[1] - 50)
            cy = random.randint(50, IMG_SIZE[0] - 50)
            radius = random.randint(20, 60)
            angle = random.randint(0, 360)
            sides = random.choice([3, 4, 5, 6]) # Triangolo, Quadrato, Pentagono, Esagono
            
            # Intensità grigio casuale (tra 50 e 200 per non essere bianco puro)
            color = random.randint(50, 200) 
            
            # Calcola coordinate e disegna
            pts = get_polygon_coords((cx, cy), radius, sides, angle)
            cv2.fillPoly(img, pts, color=color)

        # --- 3. GENERAZIONE CERCHI (Target) ---
        # Queste vanno su 'img' E su 'mask'
        num_circles = random.randint(MIN_CIRCLES, MAX_CIRCLES)
        
        for _ in range(num_circles):
            cx = random.randint(50, IMG_SIZE[1] - 50)
            cy = random.randint(50, IMG_SIZE[0] - 50)
            radius = random.randint(20, 60)
            
            # Colore per l'immagine (variabile)
            color = random.randint(100, 255)
            
            # Disegna su immagine
            cv2.circle(img, (cx, cy), radius, color, -1) # -1 riempie il cerchio
            
            # Disegna su maschera (Ground Truth)
            # Per nnU-Net solitamente le classi sono 1, 2, 3... 
            # Qui usiamo 255 per visibilità, ma in training andrà convertito a 1
            cv2.circle(mask, (cx, cy), radius, 255, -1)

        # --- 4. SALVATAGGIO ---
        # Nome file univoco
        filename = f"shape_{i:03d}.png"
        
        img_path = os.path.join(OUTPUT_DIR, IMG_FOLDER, filename)
        mask_path = os.path.join(OUTPUT_DIR, LABEL_FOLDER, filename)
        
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    print(f"Completato! Dataset salvato in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    generate_dataset()