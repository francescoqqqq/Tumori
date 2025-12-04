# Dataset Geometrico - Test su Figure Semplici

Questo progetto genera e utilizza un dataset sintetico semplice con figure geometriche (cerchi, triangoli, quadrati, ecc.) per testare nuove metriche, loss functions e architetture di nnU-Net prima di applicarle ai dati clinici.

## 🎯 Obiettivo

Testare e sviluppare nuove metodologie su problemi semplici e controllati, per poi applicarle al contesto clinico più complesso.

**Task di segmentazione**: Riconoscere solo i **cerchi** da immagini che contengono anche altre figure geometriche (triangoli, quadrati, pentagoni, esagoni) come distrattori.

## 📁 Struttura

```
geometrica/
├── data_geom.py              # Generatore dataset sintetico
├── dataset_shapes/           # Dataset pre-generato (100 immagini PNG)
│   ├── imagesTr/            # Immagini di input (grayscale)
│   └── labelsTr/             # Maschere di segmentazione (cerchi = 255, sfondo = 0)
├── convert_to_nnunet.py     # Converti PNG → formato nnU-Net (NIfTI)
├── train_geometric.py        # Script per training nnU-Net
├── test_geometric.py         # Script per test e valutazione
├── nnUNet_raw/              # Dataset convertito (formato nnU-Net)
├── nnUNet_preprocessed/      # Dataset preprocessato
├── nnUNet_results/          # Risultati training
└── risultati/               # Risultati test e visualizzazioni
```

## 🚀 Utilizzo Completo

### 1. Generare il Dataset (se necessario)

```bash
cd geometrica
python data_geom.py
```

Questo genererà 100 immagini in `dataset_shapes/` con:
- **Target**: Cerchi da segmentare (1-5 cerchi per immagine)
- **Distrattori**: Triangoli, quadrati, pentagoni, esagoni (0-10 per immagine)
- **Dimensioni**: 512x512 pixel grayscale

### 2. Convertire in Formato nnU-Net

```bash
python convert_to_nnunet.py
```

Questo script:
- Converte le immagini PNG in formato NIfTI (richiesto da nnU-Net)
- Crea la struttura corretta delle cartelle (`nnUNet_raw/Dataset501_Shapes/`)
- Genera il `dataset.json` con le informazioni del dataset
- Dataset ID: **501** (diverso dal dataset clinico che usa 500)

### 3. Setup Variabili d'Ambiente

```bash
export nnUNet_raw="/workspace/geometrica/nnUNet_raw"
export nnUNet_preprocessed="/workspace/geometrica/nnUNet_preprocessed"
export nnUNet_results="/workspace/geometrica/nnUNet_results"
```

Oppure, se sei nella directory `geometrica/`:

```bash
cd geometrica
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"
```

### 4. Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

Questo creerà i piani di training e preprocesserà il dataset.

### 5. Training

**Opzione A: Usando lo script Python**
```bash
python train_geometric.py -d 501 -c 2d -f 0
```

**Opzione B: Comando diretto nnU-Net**
```bash
nnUNetv2_train 501 2d 0
```

### 6. Test e Valutazione

```bash
python test_geometric.py -d 501 -c 2d -f 0
```

Questo script:
- Esegue inference sul test set
- Calcola metriche (Dice, IoU)
- Genera visualizzazioni confrontando input, ground truth e predizioni
- Salva risultati in `risultati/`

## 📊 Formato Dati

### Dataset Originale (PNG)
- **Immagini**: Grayscale PNG (0-255)
- **Maschere**: Binary PNG (0=sfondo, 255=cerchio target)

### Dataset nnU-Net (NIfTI)
- **Immagini**: NIfTI 3D (1 slice) con canale singolo
- **Maschere**: NIfTI 3D (1 slice) con classi (0=background, 1=circle)

## ⚙️ Configurazione

Puoi modificare i parametri in `data_geom.py`:
- `NUM_IMAGES`: Numero di immagini da generare (default: 100)
- `IMG_SIZE`: Dimensioni delle immagini (default: 512x512)
- `MIN_CIRCLES` / `MAX_CIRCLES`: Range di cerchi target (default: 1-5)
- `MAX_DISTRACTORS`: Numero massimo di figure di disturbo (default: 10)

## 💡 Vantaggi del Dataset Geometrico

- **Semplicità**: Problema ben definito e controllato
- **Velocità**: Training e test molto più rapidi rispetto ai dati clinici
- **Debugging**: Facile identificare problemi nelle metriche/loss
- **Validazione**: Test rapido di nuove idee prima di applicarle ai dati clinici
- **Isolamento**: Permette di testare solo la capacità di distinguere forme specifiche

## 🔄 Workflow Completo

```bash
# 1. Genera dataset (se necessario)
python data_geom.py

# 2. Converti in formato nnU-Net
python convert_to_nnunet.py

# 3. Setup variabili d'ambiente
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

# 4. Preprocessing
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

# 5. Training
python train_geometric.py -d 501 -c 2d -f 0

# 6. Test
python test_geometric.py -d 501 -c 2d -f 0
```

## 📈 Risultati Attesi

Con questo dataset semplice, nnU-Net dovrebbe raggiungere:
- **Dice Score**: > 0.95 (molto alto, dato che il problema è semplice)
- **IoU**: > 0.90

Se le metriche sono molto basse, potrebbe indicare problemi nella configurazione o nel formato dei dati.

## 🐛 Troubleshooting

**Errore "Dataset not found"**: Assicurati di aver eseguito `convert_to_nnunet.py` e di aver impostato correttamente le variabili d'ambiente.

**Errore durante preprocessing**: Verifica che il dataset sia stato convertito correttamente e che `dataset.json` esista.

**Training molto lento**: Per dataset piccoli come questo, il training dovrebbe essere veloce. Se è lento, controlla la configurazione GPU.
