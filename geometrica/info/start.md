# Start — Riepilogo rapido del progetto `geometrica`

Scopo: migliorare le segmentazioni di cerchi ottenute con nnU-Net aggiungendo loss geometriche differenziabili che guidino la rete verso forme più circolari e bordi più regolari, mantenendo alta la performance di overlap (Dice).

Punti chiave
- Obiettivo: integrare penalità su compattezza (compactness), liscezza del bordo (boundary smoothness) e rapporto assi (aspect/eccentricity) insieme a Dice+CE.
- Implementazione: tutte le metriche sono approssimate in modo differenziabile (no hard threshold, no conversione a NumPy, operazioni PyTorch vettorizzate).
- Tre termini principali:
  - Compactness: area soft / perimetro² (perimetro ≈ magnitudine del gradiente via Sobel). Loss = 1 − compactness.
  - Boundary smoothness: varianza/mean del Laplacian su `pred_soft` per penalizzare bordi frastagliati.
  - Aspect / Circle Template: sostituisce la vecchia loss a momenti; costruisce un template circolare soft e minimizza la deviazione normalizzata.

Stabilità e safety
- Fix critico: evitare operazioni che rompono il computational graph (es. thresholding hard o .cpu().numpy()); usare only-PyTorch e fallback del tipo `pred_soft.sum() * 0.0` quando serve.
- Protezioni: clamp aggressivi su sqrt, controllo area minima, area-ramp (scala 0→1 per attivare la geometria), check NaN/Inf pre/post backward e skip optimizer se gradienti corrotti.

Allenamento e configurazione
- Warm-up: default ~15–20 epoche in cui si addestra solo con Dice+CE; poi si abilita gradualmente la loss geometrica.
- Pesi consigliati (esempi usati): `compactness=0.01`, `boundary=0.005`, `aspect/eccentricity~0.015–0.03`.
- Parametri pratici: batch_size ridotto (es. 8), `GEOMETRIC_LOSS_SAMPLES` per ridurre memoria, num_epochs geometriche ≈100.

Metriche e allineamento
- Attenzione: definizioni usate in training (soft) e valutazione (discrete, OpenCV) possono divergere — soprattutto per eccentricity. È raccomandato allineare le definizioni o documentare la differenza.
- Valutare sia metriche standard (Dice, IoU, Hausdorff, Boundary IoU) sia metriche geometriche (compactness, solidity, eccentricity) e riportare min/max/std oltre alla media.

Workflow essenziale
- Genera dataset: `python data_geom.py` → converti con strumenti nnU-Net → `python run_experiment.py` per l'intera pipeline.
- Trainer custom: `nnUNetTrainerGeometric` (usa `geometric_losses.py` e `geometric_config.py`); `nnUNetTrainerBaseline` per il confronto.

Raccomandazioni rapide
- Allineare loss e metriche di valutazione (es. eccentricity) o modificare la valutazione per usare proxy coerenti.
- Se compaiono NaN: aumentare `WARMUP_EPOCHS`, ridurre i pesi geometrici, abilitare i safety checks, verificare gradient flow.
- Per multi-oggetto (multi-cerchi) valutare Hausdorff e metriche per contorno (media/min/max) invece di usare solo il contorno più grande.

Dove approfondire
- Vedi i file in `geometrica/info/` per dettagli tecnici: `GEOMETRIC_MODIFICATIONS.md`, `TEORIA_DIFFERENZIABILITA.md`, `ANALISI_METRICHE.md`, `VERSIONE_1.0_GEOMETRIC_LOSS.md`, `run_exp_info.md`.

---
Questo file è un riassunto sintetico: per debug o modifica dei pesi, apri `config.py` e `geometric_config.py`, e per implementazione guarda `geometric_losses.py` e `nnUNetTrainerGeometric.py`.
