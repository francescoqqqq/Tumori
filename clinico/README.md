# BraTS 2023 Hybrid Dataset - Segmentazione Tumori Cerebrali

Questo progetto prepara un dataset ibrido (Reale + Sintetico) per testare e confrontare diverse Loss Function (Dice, HD, clDice) su nnU-Net nella segmentazione di tumori cerebrali.

## ⚠️ Nota Importante: Dataset non inclusi nel Repository

**I dataset non sono inclusi in questo repository** perché troppo pesanti (~1.7 GB). Le cartelle `nnUNet_raw/`, `nnUNet_preprocessed/` e `nnUNet_results/` sono esclusi da Git tramite `.gitignore`.

Per ottenere il dataset completo, segui le istruzioni nella sezione **"Script Utilizzati"** qui sotto per rigenerarlo da zero.

## Struttura del Dataset
Una volta generato, il dataset si trova in `nnUNet_raw/Dataset500_Hybrid_BraTS` ed è composto da **112 casi validi** (dopo sanitizzazione):

1.  **Casi Reali (19 casi)**:
    -   Estratti dal dataset BraTS 2023 GLI Training (Synapse ID: `syn51514132`).
    -   Pazienti con tutte e 4 le modalità (T1, T1c, T2, FLAIR) complete.
    -   Prefisso: `BraTS_Real_XXX`.

2.  **Casi Sintetici "Frankenstein" (100 casi)**:
    -   Generati usando la tecnica di **Lesion Inpainting**.
    -   Tumori reali estratti dai 19 pazienti e "trapiantati" in posizioni casuali su cervelli sani (host).
    -   Include rotazioni e deformazioni (flip) per aumentare la varianza.
    -   Prefisso: `BraTS_Synth_XXX`.

## Split Train/Val/Test
Lo split è stato salvato nel file `hybrid_splits.json` (incluso nel repository).
-   **Train**: ~79 casi (Misto Reale/Sintetico)
-   **Validation**: ~30 casi
-   **Test**: ~10 casi Stratificati
    -   3 Casi Reali
    -   7 Casi Sintetici (scelti casualmente per testare generalizzazione su anatomie "nuove")

**Nota:** I numeri esatti dipendono dal risultato della sanitizzazione. Dopo la generazione, esegui `scripts/sanitize_dataset.py` per ottenere i conteggi precisi.

## Script Utilizzati (salvati in `scripts/`)
Per **rigenerare il dataset completo** da zero (necessario al primo setup o dopo un clone del repository):

1.  **Scaricare i casi reali da Synapse:**
    ```bash
    python scripts/download_brats_stratified.py
    ```
    Richiede un **Synapse Auth Token** (vedi [Synapse.org](https://www.synapse.org/) -> Account Settings -> Personal Access Tokens). Lo script ti chiederà di inserirlo se il login automatico fallisce.

2.  **Generare i casi sintetici "Frankenstein":**
    ```bash
    python scripts/generate_frankenstein_dataset.py
    ```
    Questo script mescola i tumori reali in posizioni nuove, creando 100 casi sintetici.

3.  **Assemblare il dataset finale:**
    ```bash
    python scripts/finalize_hybrid_dataset.py
    ```
    Crea la struttura `nnUNet_raw/Dataset500_Hybrid_BraTS` pronta per nnU-Net.

4.  **Sanitizzazione (opzionale, ma consigliato):**
    ```bash
    python scripts/sanitize_dataset.py
    ```
    Rimuove casi incompleti e aggiorna il `dataset.json` con il conteggio corretto.

## Prossimi Passi
1.  Installare nnU-Net: `pip install nnunetv2`
2.  Impostare variabili d'ambiente:
    ```bash
    export nnUNet_raw="/workspace/clinico/nnUNet_raw"
    export nnUNet_preprocessed="/workspace/clinico/nnUNet_preprocessed"
    export nnUNet_results="/workspace/clinico/nnUNet_results"
    ```
3.  Lanciare preprocessing:
    ```bash
    nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity
    ```
4.  Lanciare training (esempio Fold 0):
    ```bash
    nnUNetv2_train 500 2d 0
    ```
5.  Testare il modello:
    ```bash
    python test_nnunet.py -d 500 -c 2d -f 0
    ```

## Monitoraggio Training
Per monitorare il training in tempo reale:
```bash
bash monitor_training.sh
```

