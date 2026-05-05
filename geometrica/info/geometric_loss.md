# Loss geometriche differenziabili — documentazione tecnica

**Progetto:** `geometrica` — nnU-Net con vincoli geometrici su segmentazioni circolari (estensione futura: altri domini, es. tumori).

**File principale:** `geometric_losses.py` (classe `DifferentiableGeometricLossesV2` e wrapper `GeometricLosses`).

**Trainer:** `nnUNetTrainerGeometric.py` integra la loss geometrica con Dice + Cross-Entropy e applica *warmup*, *rampa* e *Dice gate*.

**Configurazione:** `geometric_config.py` è generato automaticamente da `run_experiment.py` e copiato insieme a `geometric_losses.py` nella cartella del trainer nnU-Net usata in training.

---

## Indice

1. [Obiettivo e principi](#obiettivo-e-principi)
2. [Dove vive il codice e come viene usato](#dove-vive-il-codice-e-come-viene-usato)
3. [Formula della loss totale nel trainer](#formula-della-loss-totale-nel-trainer)
4. [I tre termini geometrici](#i-tre-termini-geometrici)
5. [Chronologia delle versioni (V2.4 → V2.8)](#chronologia-delle-versioni-v24--v28)
6. [Circle Template Loss (V2.8) — sostituto dell’eccentricità a momenti](#circle-template-loss-v28--sostituto-delleccentricità-a-momenti)
7. [Area ramp (V2.7)](#area-ramp-v27)
8. [Altri miglioramenti (V2.5–V2.6, stabilità)](#altri-miglioramenti-v25v62-stabilità)
9. [Trainer: warmup progressivo e split Dice gate](#trainer-warmup-progressivo-e-split-dice-gate)
10. [Configurazione (`geometric_config.py`)](#configurazione-geometric_configpy)
11. [Training vs valutazione: perché la metrica “eccentricity” può divergere dalla loss](#training-vs-valutazione-perché-la-metrica-eccentricity-può-divergere-dalla-loss)
12. [Suggerimenti pratici (tuning, debug)](#suggerimenti-pratici-tuning-debug)
13. [Documentazione correlata in `info/`](#documentazione-correlata-in-info)

---

## Obiettivo e principi

La loss geometrica aggiunge, oltre a Dice e CE (localizzazione e overlap), un **prior di forma**: cercare maschere **compatte**, con **contorno regolare** e quanto più possibile **circolari** (nel setup attuale i target sono cerchi).

**Vincoli di progetto:**

| Principio | Cosa significa |
|-----------|----------------|
| **Differenziabilità completa** | Nessuna soglia hard che azzeri i gradienti (`pred_soft` continuo, niente NumPy nel forward della loss). |
| **Vettorizzazione** | Batch intero senza cicli Python sulle immagini. |
| **Stabilità numerica** | Clamp su sqrt, epsilon, gestione NaN/Inf con tensori legati al grafo (`pred_soft.sum() * 0.0`). |
| **Allineamento operativo** | Pesi e soglie letti da `geometric_config` quando il file è nel `PYTHONPATH` del processo di training. |

---

## Dove vive il codice e come viene usato

| Componente | Ruolo |
|------------|--------|
| `geometric_losses.py` | Implementazione dei termini geometrici e combinazione con **area ramp**. |
| `nnUNetTrainerGeometric.py` | Somma `loss_dice_ce + loss_geometric`, applica **warmup + ramp**, **Dice gate spaccato** (shape vs compactness), subset di campioni per batch (`GEOMETRIC_LOSS_SAMPLES`). |
| `run_experiment.py` | All’avvio training copia in cartella trainer: `nnUNetTrainerBaseline.py`, `nnUNetTrainerGeometric.py`, **`geometric_losses.py`**, genera **`geometric_config.py`**. |
| `geometric_config.py` | Valori effettivi di pesi, epoche, `MIN_AREA_THRESHOLD`, ecc. (sovrascritto a ogni run). |

Se `geometric_config.py` **non** è importabile (es. `PYTHONPATH` errato nel subprocess), trainer e loss usano **fallback** hardcoded: in console compare un **CONFIG WARNING** — in quel caso i pesi impostati in `run_experiment.py` **non** si applicano.

---

## Formula della loss totale nel trainer

Per ogni step (dopo il warmup e con `warmup_scale > 0`):

```text
total_loss = loss_dice_ce + loss_geometric

loss_geometric = loss_shape + loss_compact

loss_shape  = geometric_loss_shape(softmax) × warmup_scale × dice_gate_shape
loss_compact = geometric_loss_compact(softmax) × warmup_scale × dice_gate
```

- `geometric_loss_shape`: pesi **solo** `WEIGHT_ECCENTRICITY` e `WEIGHT_BOUNDARY` (compactness = 0).  
  Internamente il termine “eccentricity” è la **Circle Template Loss** (V2.8), non più i momenti di inerzia.
- `geometric_loss_compact`: pesi **solo** `WEIGHT_COMPACTNESS`.

L’oggetto `self.geometric_loss` (tutti e tre i termini) viene ancora chiamato per **logging** delle componenti (`get_last_losses`), non per sommare due volte il contributo geometrico.

---

## I tre termini geometrici

Tutti operano su `pred_soft = softmax[:, 1]`, **senza binarizzazione**.

### 1. Compactness

- **Idea:** \(C = 4\pi A / P^2\) (1 per cerchio ideale, &lt; 1 per forme irregolari).
- **Implementazione:** area = somma di `pred_soft`; perimetro ≈ integrale della magnitudine del gradiente Sobel su `pred_soft`.
- **Loss:** `1 - C` (clamp di C in [0,1], medie solo su campioni con area ≥ soglia minima).

### 2. Boundary (smoothness)

- **Idea:** penalizzare **alta varianza** del Laplaciano e contributo medio |Lap| nelle regioni dove c’è massa (maschera soft = `pred_soft`, non soglia fissa), per bordi meno frastagliati.

### 3. “Eccentricity” nel codice = Circle Template Loss (V2.8)

- **Nome storico:** il peso è ancora `WEIGHT_ECCENTRICITY` e la chiave di log `'eccentricity'`, ma **non** coincide più con la metrica OpenCV di valutazione.
- **Comportamento:** si confronta `pred_soft` con un template circolare morbido derivato da centroide e raggio attesi ([dettaglio sotto](#circle-template-loss-v28--sostituto-delleccentricità-a-momenti)).

---

## Chronologia delle versioni (V2.4 → V2.8)

| Versione | Focus |
|----------|--------|
| **V2.4** | Clamp più aggressivi su radici (`sqrt` con `eps` più grande); area minima; riduzione esplosione gradienti. |
| **V2.5** | “Aspect / eccentricità” tramite **autovalori dei momenti** centrali: \((1 - \lambda_{\min}/\lambda_{\max})^2\). |
| **V2.6** | Boundary con maschera **soft**; `sqrt(clamp(x,0)+eps)` per evitare zone morte di gradiente; **cache** delle griglie coordinate. |
| **V2.7** | **Area ramp** sulla loss combinata (scala 0→1 tra ~50 px e ~300 px di “massa” soft media nel batch). |
| **V2.8** | **Circle Template Loss** al posto della loss a momenti; normalizzazione per area del template. |

La docstring in cima a `geometric_losses.py` elenca ancora la storia V2.4–V2.6 per riferimento.

---

## Circle Template Loss (V2.8) — sostituto dell’eccentricità a momenti

### Problema della vecchia loss a momenti (V2.5–V2.7)

Per predizioni già **quasi circolari** (rapporto \(\lambda_{\min}/\lambda_{\max} \approx 0.95\)), la loss \((1 - \text{ratio})^2\) è **piccolissima** e i gradienti rispetto ai logit diventano trascurabili rispetto a Dice+CE. In pratica il termine **non guidava** abbastanza la forma quando la rete era già vicina a un cerchio — proprio il regime in cui vorresti rifinire l’ellitticità residua.

### Idea della Circle Template Loss

1. **Stop-gradient** su statistiche globali: dal `pred_soft` corrente si calcolano (senza propagare gradiente attraverso queste quantità) il **centroide** e l’**area** somma, poi il raggio del cerchio con la stessa area:  
   \(r = \sqrt{A/\pi}\) (con clamp per evitare valori estremi).
2. Si costruisce un **template circolare soft**:
   \[
   T_{ij} = \sigma\bigl(k \cdot (r - d_{ij})\bigr),
   \]
   dove \(d_{ij}\) è la distanza dal centroide al pixel \((i,j)\) e \(k\) è una nitidezza del bordo (nel codice: `TEMPLATE_SHARPNESS = 5.0`).  
   Dentro il cerchio ideale \(T \approx 1\), fuori \(T \approx 0\).
3. **Loss:** media dei quadrati delle differenze **normalizzata per la somma del template** (non per \(H \times W\)):
   \[
   L = \frac{\sum_{ij} (p_{ij} - T_{ij})^2}{\sum_{ij} T_{ij} + \epsilon}
   \]
   così il segnale non viene **diluito** dal fatto che l’immagine sia grande (es. 512×512) e il cerchio occupi solo una piccola frazione di pixel.

### Effetto sui gradienti

- Dove la predizione è **ellittica** rispetto al cerchio con stessa area: ci sono ampie regioni in cui \(p\) e \(T\) **non coincidono** (vertici dell’ellisse vs “buchi” lungo l’asse minore) → contributo alla loss **sostanzioso** e gradienti direzionali chiari.
- Il metodo storico `_vectorized_eccentricity_loss` (momenti) **resta nel file** per riferimento / esperimenti, ma **`__call__` usa `_vectorized_circle_template_loss`**.

### Test rapido

Eseguendo `python geometric_losses.py` si verifica forward, assenza di NaN e **flusso del gradiente** fino ai logit di prova.

---

## Area ramp (V2.7)

Dopo il calcolo delle tre componenti grezze e dei pesi:

```text
area_scale = mean( clamp(area_per_batch / TARGET_AREA, 0, 1) )
TARGET_AREA = 6 * MIN_AREA_THRESHOLD   # default: 6 * 50 = 300 px
total = (w_c * L_c + w_b * L_b + w_e * L_e) * area_scale
```

- Se l’area soft è molto piccola, `area_scale → 0`: la geometria non **contrasta** la fase in cui Dice+CE devono ancora far **crescere** la blob.
- Tra ~50 px e ~300 px c’è una **transizione graduale** invece di un passaggio netto dal “tutto geometrico” al “zero”.

---

## Altri miglioramenti (V2.5–V2.6, stabilità)

- **`sqrt(clamp(..., min=0) + eps)`** laddove serve: evita derivate \(\rightarrow \infty\) e zone con derivata nulla nell’interno delle radici quadrate.
- **Boundary:** uso di `pred_soft` come peso invece di maschera binaria, così i gradienti non si annullano sui pixel sotto soglia.
- **Cache coordinate** `(H, W, device, dtype)` per non ricreare griglie a ogni forward.

---

## Trainer: warmup progressivo e split Dice gate

Definito in `geometric_config.py` / `run_experiment.py`:

| Parametro | Ruolo tipico |
|-----------|----------------|
| `WARMUP_EPOCHS` | Epoche in cui `warmup_scale = 0`: solo Dice+CE. |
| `geometric_loss_warmup_ramp` | Lunghezza (in epoche) della rampa lineare \(0 \to 1\) subito dopo il warmup. Nel trainer è inizializzato uguale a `WARMUP_EPOCHS`; il divisore effettivo è `max(1, geometric_loss_warmup_ramp)`. |
| `GEOMETRIC_LOSS_SAMPLES` | Numero massimo di esempi per batch su cui si calcola la loss geometrica (risparmio GPU). |

**Dice gate (split):**

| Simbolo | Formula (idea) | Applicato a |
|---------|----------------|-------------|
| `dice_gate` | Lineare da 0 (Dice &lt; 0.5) a 1 (Dice = 1), poi media sul batch | **Solo compactness** |
| `dice_gate_shape` | Media del Dice per campione, clamp [0,1] | **Circle template + boundary** |

Motivazione: se il Dice è basso, **non** spegnere del tutto la parte “forma” (altrimenti restano blob allungati); la **compactness** invece può **restringe** predizioni sbagliate, quindi resta legata a un gate più **duro**.

---

## Configurazione (`geometric_config.py`)

Variabili tipiche scritte da `run_experiment.py` (i valori numerici dipendono dall’esperimento):

- `WEIGHT_COMPACTNESS`, `WEIGHT_ECCENTRICITY`, `WEIGHT_BOUNDARY`
- `WARMUP_EPOCHS`, `NUM_EPOCHS`, `BATCH_SIZE`, `GEOMETRIC_LOSS_SAMPLES`
- `MIN_AREA_THRESHOLD` (usato nella loss e coerente con le soglie interne a 50 px nelle singole funzioni dove indicato)

**Nota:** il nome `WEIGHT_ECCENTRICITY` pesa ora la **Circle Template Loss**; rinominare il parametro richiederebbe allineare trainer, config e documentazione — per ora è volutamente conservativo.

---

## Training vs valutazione: perché la metrica “eccentricity” può divergere dalla loss

- **In training** (Circle Template): confronto **soft** `pred_soft` vs template continuo, con normalizzazione sull’area del template; obiettivo = forma circolare coerente con massa e centroide attuali.
- **In valutazione** (`metrics_utils.calculate_eccentricity`): maschera **binaria** `pred > 0.5`, contorno principale con OpenCV, **`cv2.fitEllipse`**, poi  
  \(\sqrt{1 - (b/a)^2}\) con \(a \ge b\) assi ellisse.

Sono **due definizioni diverse**:  
- rumore di quantizzazione sul bordo pixelato;  
- soft vs hard;  
- “cerchio atteso dalla massa soft” vs “ellisse minima sul contorno discreto”.

Pertanto **non** aspettarsi che un miglioramento del termine in training si traduca sempre in un miglioramento monotono della colonna *eccentricity* nel `metrics_comparison.txt`. Metriche come **compactness** (definizione affine in valutazione) o **Hausdorff** possono allinearsi meglio all’intento pratico.

---

## Suggerimenti pratici (tuning, debug)

1. **Verificare il log di training** per `Config caricata da file: OK` — altrimenti i pesi non sono quelli dell’esperimento.
2. **Pesi:** partire dai valori in `run_experiment.py`; dopo V2.8 la Circle Template è comparabile in ordine di grandezza ai termini grazie alla normalizzazione, ma richiede comunque prove sui dati.
3. **Dataset piccolo / alta varianza:** medie aggregate oscillano molto; usare `vis_bad`, mediane, confronti stratificati.
4. **Post-processing inference** (`MIN_COMPONENT_PX` in pipeline): influenza **solo** export/metriche su NIfTI→PNG dopo inferenza, **non** il gradiente in training — utile per non far contare micromacchie nell’eccentricità OpenCV, ma è un problema distinto dall’ottimizzazione della loss.

---

## Documentazione correlata in `info/`

| File | Contenuto |
|------|-----------|
| `GEOMETRIC_MODIFICATIONS.md` | Storia più ampia (fix differenziabile, architettura, workflow). |
| `TEORIA_DIFFERENZIABILITA.md` | Note teoriche sulla differenziabilità. |
| `ANALISI_METRICHE.md` | Interpretazione delle metriche di valutazione. |
| `VERSIONE_1.0_GEOMETRIC_LOSS.md` | Snapshot della prima versione. |
| `run_exp_info.md` | Pipeline `run_experiment.py`. |

---

*Documento aggiornato alla versione delle loss descritta in codice come V2.8 (Circle Template Loss), con integrazione trainer (warmup, rampa, Dice gate splittato) e configurazione tramite `geometric_config.py`.*
