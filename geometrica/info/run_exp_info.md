# run_experiment.py - guida completa

Questo documento riassume in modo operativo cosa fa `run_experiment.py`, come e dove vengono usati i trainer custom, e come avviene il training delle due reti.

## 1) Obiettivo del file

`run_experiment.py` e' il master script della pipeline:
1. genera dataset sintetico PNG
2. fa split train/test
3. converte train in NIfTI per nnU-Net
4. esegue preprocess nnU-Net
5. installa/patcha trainer custom
6. allena baseline e/o geometrica
7. esegue inference sul test
8. calcola metriche, visualizzazioni, confronto finale

Ogni run e' isolata in `experiments/<FOLDER_NAME>/`.

## 2) Blocco configurazione (top del file)

Le variabili iniziali controllano tutta la pipeline:
- nome run (`FOLDER_NAME`)
- modalita' automatica o interattiva (`AUTOMATIC`)
- dataset (`IMG_SIZE`, `NUM_IMAGES`, `TARGET_MODE`, `COLOR_STYLE`)
- training (`DATASET_ID`, `RETI_DA_ALLENARE`, `EPOCHS`, `BATCH_SIZE`, `WARMUP_EPOCHS`)
- pesi loss geometrica

In pratica, il file viene parametrizzato da qui.

## 3) Path di esperimento + variabili ambiente nnU-Net

Subito dopo gli import base, lo script costruisce i path della run e imposta:
- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

Questo passaggio e' fondamentale per evitare che nnU-Net usi cartelle globali condivise.

## 4) Import locali e funzioni metriche

Lo script importa:
- generatori dataset da `data_geom.py`
- funzioni metriche/plot da `test.py` tramite `importlib`

Le funzioni importate da `test.py` vengono poi usate nello STEP 5 per confronto finale.

## 5) Utility runtime

Funzioni di supporto:
- `_separator()`: output leggibile
- `_get_nets_to_run()`: valida e traduce `RETI_DA_ALLENARE`
- `_run_subprocess()`: wrapper dei comandi CLI con log file, filtro output e gestione errori

Questa parte rende la pipeline robusta e tracciabile.

## 6) Modalita' interattiva (opzionale)

Se `AUTOMATIC == "no"`:
- chiede nome cartella run
- mostra e permette modifica parametri dataset
- dopo dataset chiede conferma/modifica training
- dopo training chiede se eseguire anche confronto finale

Se `AUTOMATIC == "si"`, salta tutte le domande.

## 7) STEP 1 - Dataset

Blocchi principali:
- `create_experiment_structure()`: crea albero cartelle run
- `generate_dataset()`: genera PNG sintetici
- `split_and_organize()`: split train/test con seed 42
- `cleanup_raw()`: elimina folder temporanea
- `save_config_yaml()`: salva riepilogo configurazione usata

Output: `1_dataset/train` e `1_dataset/test` pronti.

## 8) STEP 2 - Conversione e preprocess nnU-Net

- `convert_train_to_nnunet()` converte il train da PNG a NIfTI:
  - immagini in float32
  - label binarie 0/1
  - genera `dataset.json` richiesto da nnU-Net
- `run_preprocessing()` lancia `nnUNetv2_plan_and_preprocess`

Output: dataset preprocessato in `2_nnunet_engine/nnUNet_preprocessed`.

## 9) STEP 3 - Come vengono usati i trainer custom

### 9.1 Installazione trainer nel package nnU-Net

Funzione: `install_trainer_files()`

Copia dal progetto al package installato:
- `nnUNetTrainerBaseline.py`
- `nnUNetTrainerGeometric.py`
- `geometric_losses.py`

Quindi nnU-Net, quando invocato da CLI, trova questi trainer con i nomi attesi.

### 9.2 Patch parametri training nel package

Funzione: `_patch_epochs_in_trainer()`

Effettua patch testuale su file trainer nel package per allineare:
- `self.num_epochs`
- `BATCH_SIZE`
- (nel geometrico) warmup e override batch safety

Serve a far combaciare i parametri del blocco config con cio' che i trainer eseguono realmente.

### 9.3 Generazione config geometrica

Funzione: `_write_geometric_config()`

Genera `geometric_config.py` con pesi e iperparametri correnti e lo scrive in:
- cartella trainer del package nnU-Net
- root del progetto (`SCRIPT_DIR`)

Motivo: assicurare che subprocess e import risolvano la stessa configurazione.

## 10) Processo di allenamento: baseline vs geometrica

### 10.1 Selezione reti da allenare

`_get_nets_to_run()` traduce:
- `"baseline"` -> solo baseline
- `"geometrica"` -> solo geometrica
- `"entrambe"` -> baseline poi geometrica

### 10.2 Lancio training via CLI nnU-Net

`run_training_single(net_type)` esegue:
- `nnUNetv2_train <DATASET_ID> 2d 0 -tr <trainer_name>`

dove `<trainer_name>` e':
- `nnUNetTrainerBaseline` per baseline
- `nnUNetTrainerGeometric` per geometrica

`run_all_training()` itera sulle reti scelte.

### 10.3 Cosa fa esattamente nnUNetTrainerBaseline

File: `nnUNetTrainerBaseline.py`

Caratteristiche:
- estende `nnUNetTrainer` standard
- non aggiunge loss geometriche
- forza batch size nei plans e in configuration manager
- usa loss standard nnU-Net (Dice + CE)
- usa numero epoche patchato dal master script

Quindi e' la rete di controllo "classica".

### 10.4 Cosa fa esattamente nnUNetTrainerGeometric

File: `nnUNetTrainerGeometric.py`

Caratteristiche:
- estende `nnUNetTrainer`
- importa parametri da `geometric_config.py`
- inizializza `GeometricLosses`
- train step custom:
  - calcola `loss_dice_ce` standard
  - dopo warmup calcola `loss_geometric` su primi N campioni batch
  - `total_loss = loss_dice_ce + loss_geometric`
- include molte safety:
  - fallback a Dice+CE se loss geometrica esplode
  - check NaN/Inf su loss e gradienti
  - gradient clipping
  - eventuale skip optimizer step su gradienti corrotti
- logging dettagliato per epoca e salvataggio config loss a fine training

In sintesi: stessa base nnU-Net, ma con vincoli geometrici aggiuntivi e protezioni di stabilita'.

## 11) STEP 4 - Inference

Per ogni rete allenata:
1. verifica checkpoint (niente NaN/Inf nei pesi)
2. converte test PNG -> NIfTI temporanei
3. chiama `nnUNetv2_predict` col trainer corretto
4. converte predizioni NIfTI (0/1) -> PNG (0/255)

Output in `3_predizioni/`.

## 12) STEP 5 - Confronto finale

`run_comparison()`:
- legge GT test e predizioni
- calcola metriche per immagine
- aggrega mean/std/min/max
- salva:
  - `metrics_comparison.json`
  - `metrics_comparison.txt`
  - visualizzazioni
  - grafico comparativo (se entrambe le reti)

## 13) Sequenza reale in main()

`main()` orchestrazione:
1. setup (eventuale interattivo)
2. STEP 1 dataset
3. STEP 2 conversione/preprocess
4. STEP 3 install trainer + training
5. STEP 4 inference
6. STEP 5 confronto
7. riepilogo finale

## 14) Collegamento diretto trainer <-> master script

Il collegamento avviene in tre punti chiave:
1. mapping nomi trainer:
   - `"baseline"` -> `"nnUNetTrainerBaseline"`
   - `"geometrica"` -> `"nnUNetTrainerGeometric"`
2. installazione/copia trainer custom nel package nnU-Net
3. chiamata CLI con `-tr <trainer_name>` per forzare il trainer desiderato

Quindi il master script non allena "manualmente": prepara ambiente+file e poi delega il training a `nnUNetv2_train` scegliendo il trainer custom tramite nome.
