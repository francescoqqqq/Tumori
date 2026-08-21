# Metriche di Valutazione — Come Funzionano Oggi

Documento di riferimento su come sono implementate **attualmente** le metriche
di valutazione in [`metrics_utils.py`](../metrics_utils.py), usate da
`run_experiment.py` (Step 5) e da `test.py` per confrontare le predizioni
delle reti (baseline e geometrica) contro il ground truth. Tutte le funzioni
operano su maschere già binarizzate a soglia 0.5.

Per la loss usata in training vedi invece [`LOSS_GEOMETRIC.md`](LOSS_GEOMETRIC.md).

---

## Indice

1. [Dice Score](#dice-score)
2. [IoU](#iou)
3. [Compactness](#compactness)
4. [Eccentricity](#eccentricity)
5. [Hausdorff Distance](#hausdorff-distance)
6. [Boundary IoU](#boundary-iou)
7. [Aggregazione su più immagini](#aggregazione-su-più-immagini)
8. [Visualizzazioni e vis_bad](#visualizzazioni-e-vis_bad)

---

## Dice Score

```python
pred_b = (pred > 0.5).astype(np.float32)
gt_b   = (gt   > 0.5).astype(np.float32)
dice = 2 * |pred_b ∩ gt_b| / (|pred_b| + |gt_b|)
```

Calcolato a livello di pixel su tutta l'immagine: se ci sono più cerchi,
contribuiscono tutti insieme alla stessa cifra. Se sia predizione che GT sono
vuoti, ritorna `1.0` (nessun errore); se solo uno dei due è vuoto, `0.0`.

## IoU

```python
iou = |pred_b ∩ gt_b| / |pred_b ∪ gt_b|
```

Stessa logica del Dice (pixel-level, tutta l'immagine, stessa gestione dei
casi vuoti).

## Compactness

$$
C = \min\!\left(\frac{4\pi \cdot \text{Area}}{\text{Perimetro}^2},\; 1.0\right)
$$

- Si binarizza la maschera (`> 0.5`), si estraggono i contorni con
  `cv2.findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
- Si seleziona il **contorno di area massima** (`max(contours, key=cv2.contourArea)`).
- Area e perimetro (`cv2.contourArea`, `cv2.arcLength`) sono calcolati su
  quel singolo contorno.
- Il risultato è clampato a un massimo di `1.0` (un cerchio perfetto vale 1.0;
  il clamp mantiene la scala interpretabile in `[0,1]`).
- Ritorna `nan` se non ci sono contorni, o se l'area è ≤ 10 px, o se il
  perimetro è zero.

## Eccentricity

$$
e = \sqrt{1 - (b/a)^2}, \qquad a \ge b \text{ semiassi dell'ellisse fittata}
$$

- Stessa binarizzazione e stessa selezione del contorno di area massima
  usata da Compactness.
- L'ellisse viene fittata sul contorno con `cv2.fitEllipse` (richiede almeno
  5 punti sul contorno e area ≥ 10 px, altrimenti ritorna `nan`).
- `0.0` = cerchio perfetto, valori vicino a `1.0` = forma molto allungata.

## Hausdorff Distance

$$
H(P,G) = \max\big(\,\sup_{p\in P}\inf_{g\in G}\|p-g\|,\ \sup_{g\in G}\inf_{p\in P}\|g-p\|\,\big)
$$

- Contorni estratti con `CHAIN_APPROX_NONE` (tutti i punti, non semplificati)
  sia per predizione che per GT.
- Da ciascun set di contorni si prende quello di **area massima**
  (`cv2.contourArea`), non quello con più punti.
- Distanza calcolata con `scipy.spatial.distance.directed_hausdorff` in
  entrambe le direzioni, si tiene il massimo.
- Se predizione o GT sono completamente vuoti, ritorna la diagonale
  dell'immagine (caso peggiore convenzionale), invece di `nan`.
- La media di questa metrica su un dataset è sensibile a outlier estremi:
  per confronti aggregati è preferibile guardare la mediana oltre alla media.

## Boundary IoU

- Bordo estratto per sottrazione morfologica: `maschera - erosione(maschera)`,
  con kernel quadrato di lato `thickness` (default 3 px).
- IoU calcolato solo tra i pixel di bordo di predizione e GT.
- Stessa gestione dei casi vuoti di Dice/IoU.

## Aggregazione su più immagini

`_aggregate_for_vis_bad()` (usata anche da `run_experiment.py`/`test.py` con
funzioni equivalenti) calcola, per ciascuna metrica, su tutte le immagini del
test set:

- media, mediana, 95° percentile, deviazione standard, min, max, e numero di
  campioni validi (`_n`).
- I valori `nan` (predizioni vuote per compactness/eccentricity) vengono
  esclusi dal calcolo di questi aggregati, non trattati come zero.

## Visualizzazioni e vis_bad

- `create_visualization()` / `create_comparison_visualization()`: pannelli
  immagine originale / GT / predizione / overlay TP-FN-FP, con le metriche
  principali scritte in overlay.
- `create_metrics_comparison_chart()`: grafico a barre baseline vs geometrica
  su Dice, IoU, Compactness, Eccentricity, Boundary IoU, Hausdorff (95°
  percentile), con colorazione verde/rosso per la rete migliore su ciascuna
  metrica.
- `create_vis_bad()`: isola le N predizioni con Dice più basso (default 10)
  per rete, per ispezione visiva mirata dei casi peggiori.
