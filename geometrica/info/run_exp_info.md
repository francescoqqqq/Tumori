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

### 9.1 Discovery dei trainer custom (nessuna copia nel package nnU-Net)

Funzione: `prepare_trainer_environment()` (in precedenza `install_trainer_files()`)

I trainer custom **non vengono piu' copiati** dentro il package `nnunetv2`
installato (site-packages): restano sempre in `geometrica/` insieme al resto
del progetto. Questo evita di richiedere permessi di scrittura su
site-packages, un problema reale su ambienti condivisi/read-only.

La discovery avviene invece a runtime tramite `_run_nnunet_entry()` +
`NNUNET_BOOTSTRAP_TEMPLATE`: invece di lanciare le CLI `nnUNetv2_train` /
`nnUNetv2_predict`, lo script lancia `sys.executable -c <bootstrap>` dove il
bootstrap, PRIMA di importare nnunetv2:
1. inserisce `SCRIPT_DIR` (geometrica/) in `sys.path`
2. estende `nnunetv2.training.nnUNetTrainer.__path__` con `SCRIPT_DIR`, cosi'
   `from nnunetv2.training.nnUNetTrainer.geometric_losses import ...` risolve
   direttamente `geometrica/geometric_losses.py`
3. patcha `recursive_find_python_class` con un fallback che cerca anche in
   `SCRIPT_DIR`, cosi' `-tr nnUNetTrainerBaseline` / `-tr nnUNetTrainerGeometric`
   vengono trovati anche se non installati nel package

Quindi nnU-Net, quando invocato, trova questi trainer con i nomi attesi senza
alcun file extra copiato/installato.

### 9.2 Parametri training (epoche, batch size)

Nessun patch testuale sui file trainer: `nnUNetTrainerBaseline.py` e
`nnUNetTrainerGeometric.py` leggono direttamente `NUM_EPOCHS` / `BATCH_SIZE`
(e, per il geometrico, i pesi delle loss + `WARMUP_EPOCHS`) da
`geometric_config.py` con un `try/except ImportError` di fallback. Questo
allinea sempre i parametri realmente eseguiti al blocco config dell'esperimento
corrente, senza dover riscrivere il sorgente dei trainer.

### 9.3 Generazione config geometrica

Funzione: `_write_geometric_config()`

Genera `geometric_config.py` con pesi e iperparametri correnti e lo scrive
solo nella root del progetto (`SCRIPT_DIR`): il bootstrap inserisce
`SCRIPT_DIR` in `sys.path` prima di importare i trainer, quindi la risoluzione
e' sempre coerente indipendentemente da cwd o `PYTHONPATH`.

## 10) Processo di allenamento: baseline vs geometrica

### 10.1 Selezione reti da allenare

`_get_nets_to_run()` traduce:
- `"baseline"` -> solo baseline
- `"geometrica"` -> solo geometrica
- `"entrambe"` -> baseline poi geometrica

### 10.2 Lancio training

`run_training_single(net_type)` esegue, tramite `_run_nnunet_entry()` (bootstrap
in-process, non la CLI `nnUNetv2_train`):
- entry point `nnunetv2.run.run_training.run_training_entry` con argomenti
  `<DATASET_ID> 2d 0 -tr <trainer_name> -device <device>`

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
- usa `NUM_EPOCHS` letto da `geometric_config.py` (fallback 100 se assente)

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
3. chiama l'entry point `predict_entry_point` (via `_run_nnunet_entry()`) col trainer corretto
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
4. STEP 3 prepara config trainer + training
5. STEP 4 inference
6. STEP 5 confronto
7. riepilogo finale

## 14) Collegamento diretto trainer <-> master script

Il collegamento avviene in tre punti chiave:
1. mapping nomi trainer:
   - `"baseline"` -> `"nnUNetTrainerBaseline"`
   - `"geometrica"` -> `"nnUNetTrainerGeometric"`
2. bootstrap dei path (`NNUNET_BOOTSTRAP_TEMPLATE`) che rende i trainer in
   `geometrica/` scopribili da nnU-Net senza copiarli nel package installato
3. chiamata all'entry point con `-tr <trainer_name>` per forzare il trainer desiderato

Quindi il master script non allena "manualmente": prepara ambiente+file e poi
delega il training all'entry point `run_training_entry` di nnU-Net (in-process,
via `_run_nnunet_entry()`) scegliendo il trainer custom tramite nome.
