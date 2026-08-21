# Loss Geometrica — Documentazione Definitiva (stato attuale del codice)

Questo documento descrive **esattamente** cosa è implementato oggi in
[`geometric_losses.py`](../geometric_losses.py) (versione **V2.8 — Circle
Template Loss**) e in [`nnUNetTrainerGeometric.py`](../nnUNetTrainerGeometric.py),
così come effettivamente usato dall'ultimo esperimento configurato in
[`config.py`](../config.py) (`test_40_4_pesi`). È stato scritto leggendo il
codice riga per riga, non le versioni storiche descritte negli altri file di
`info/`.

> Nota terminologica importante: nel codice il peso e la chiave di log si
> chiamano ancora `WEIGHT_ECCENTRICITY` / `'eccentricity'`, ma dalla V2.8 in poi
> quel termine **non è più** la vecchia loss a momenti di inerzia — è la
> **Circle Template Loss** (vedi Parte 2). Il nome non è stato cambiato per non
> dover riallineare trainer, config e log.

---

## Indice

**Parte 1 — Implementazione tecnica**
1. [Architettura dei file e flusso dei dati](#1-architettura-dei-file-e-flusso-dei-dati)
2. [Come viene costruita la loss totale, passo per passo](#2-come-viene-costruita-la-loss-totale-passo-per-passo)
3. [Warmup, rampa e Dice gate "splittato"](#3-warmup-rampa-e-dice-gate-splittato)
4. [Safety net anti-NaN durante il training](#4-safety-net-anti-nan-durante-il-training)
5. [Configurazione effettiva e pesi finali usati](#5-configurazione-effettiva-e-pesi-finali-usati)

**Parte 2 — I vincoli geometrici: formule e motivazione**
6. [Principio comune a tutti i termini: operare su probabilità soft](#6-principio-comune-a-tutti-i-termini-operare-su-probabilità-soft)
7. [Termine 1 — Compactness](#7-termine-1--compactness)
8. [Termine 2 — Boundary smoothness](#8-termine-2--boundary-smoothness)
9. [Termine 3 — Circle Template Loss (il vero termine "eccentricity" in uso)](#9-termine-3--circle-template-loss-il-vero-termine-eccentricity-in-uso)
10. [Il termine a momenti di inerzia: presente nel codice ma NON usato](#10-il-termine-a-momenti-di-inerzia-presente-nel-codice-ma-non-usato)
11. [Perché questi tre termini insieme hanno senso geometricamente](#11-perché-questi-tre-termini-insieme-hanno-senso-geometricamente)
12. [Limiti noti e disallineamento training/valutazione](#12-limiti-noti-e-disallineamento-trainingvalutazione)

---

# Parte 1 — Implementazione tecnica

## 1. Architettura dei file e flusso dei dati

| File | Ruolo |
|------|-------|
| `geometric_losses.py` | Definisce `DifferentiableGeometricLossesV2` (i tre termini + combinazione) e il wrapper `GeometricLosses` usato dal trainer. |
| `nnUNetTrainerGeometric.py` | Sottoclasse di `nnUNetTrainer` che sovrascrive `train_step()`: calcola Dice+CE standard, poi somma la loss geometrica con warmup/rampa/gate, gestisce i safety-check anti-NaN. |
| `config.py` | Unico file che l'utente modifica: contiene i pesi (`WEIGHT_COMPACTNESS`, `WEIGHT_ECCENTRICITY`, `WEIGHT_BOUNDARY`), `WARMUP_EPOCHS`, `EPOCHS`, `BATCH_SIZE`, `GEOMETRIC_LOSS_SAMPLES`. |
| `run_experiment.py` | Ad ogni run genera `geometric_config.py` con i valori di `config.py` e lo mette in `sys.path` prima che nnU-Net importi i trainer, così `geometric_losses.py` e `nnUNetTrainerGeometric.py` leggono sempre i pesi dell'esperimento corrente (con fallback hardcoded se il file non viene trovato — stampa un `CONFIG WARNING` a schermo). |

Flusso per ogni batch di training (rete geometrica):

```
batch → network(data) → output logits [B, 2, H, W]
                              │
              ┌───────────────┴───────────────┐
              │                               │
      loss_dice_ce = self.loss(...)   softmax(output) → pred_soft = softmax[:,1]
      (Dice + CE standard nnU-Net)             │
              │                    calcolo loss geometrica (solo sui
              │                    primi GEOMETRIC_LOSS_SAMPLES campioni
              │                    del batch, per risparmiare memoria GPU)
              │                               │
              └──────────► total_loss = loss_dice_ce + loss_geometric ◄──┘
                                    │
                          backward() + grad-clip + optimizer.step()
```

## 2. Come viene costruita la loss totale, passo per passo

Nel `train_step()` di `nnUNetTrainerGeometric`:

1. **Forward** della rete → `output` (logit pre-softmax).
2. **`loss_dice_ce = self.loss(output, target)`** — loss standard nnU-Net
   (combinazione Dice + Cross-Entropy, invariata rispetto al trainer baseline).
3. Se `use_geometric_loss` è `True` e siamo oltre il warmup (`warmup_scale > 0`,
   vedi §3):
   - Si prendono solo i primi `n_samples = min(GEOMETRIC_LOSS_SAMPLES, batch_size)`
     campioni del batch (default `GEOMETRIC_LOSS_SAMPLES = 4`), per limitare il
     costo di calcolo/memoria dei termini geometrici.
   - Si calcola `output_softmax_grad = softmax(output_geometric, dim=1)` — **con
     gradiente attivo**, nessun `.detach()`.
   - Si calcola un **Dice "grezzo" per campione** (`dice_per_sample`, dentro
     `torch.no_grad()`, usato solo per pesare la loss, non per allenare):
     ```python
     inter = (pred_fg * target_bin).sum(dim=(1,2))
     union = pred_fg.sum(dim=(1,2)) + target_bin.sum(dim=(1,2))
     dice_per_sample = 2 * inter / (union + 1e-5)
     ```
   - Da `dice_per_sample` si derivano **due gate distinti** (dettagli in §3):
     `dice_gate` (hard, per compactness) e `dice_gate_shape` (soft, per gli
     altri due termini).
   - Si calcolano **due loss geometriche separate**, ciascuna un'istanza di
     `GeometricLosses` con pesi diversi:
     ```python
     loss_shape   = geometric_loss_shape(output_softmax_grad)   * warmup_scale * dice_gate_shape
     loss_compact = geometric_loss_compact(output_softmax_grad) * warmup_scale * dice_gate
     loss_geometric = loss_shape + loss_compact
     ```
     dove:
     - `geometric_loss_shape` è costruita con `weight_compactness=0`,
       `weight_eccentricity=WEIGHT_ECCENTRICITY`, `weight_boundary=WEIGHT_BOUNDARY`
       → include **solo** Circle Template + Boundary.
     - `geometric_loss_compact` è costruita con `weight_compactness=WEIGHT_COMPACTNESS`,
       gli altri due pesi a `0` → include **solo** Compactness.
   - In più, `self.geometric_loss(output_softmax_grad)` (l'oggetto con **tutti e
     tre** i pesi) viene chiamato una terza volta, ma **solo per popolare
     `last_losses` usato nel logging** (`get_last_losses()`), non contribuisce
     al gradiente sommato in `total_loss`.
4. **`total_loss = loss_dice_ce + loss_geometric`**.
5. Check NaN/Inf → eventuale fallback (§4) → `total_loss.backward()` →
   gradient clipping (`clip_grad_norm_(..., max_norm=12)`) → `optimizer.step()`.

Da notare: dentro ciascuna chiamata a `DifferentiableGeometricLossesV2.__call__`
(invocata internamente da `GeometricLosses.__call__`) viene applicata anche una
**area ramp** (§7/§8/§9), indipendente dal Dice gate del trainer:

```python
area_scale = clamp(area_per_batch / TARGET_AREA, 0, 1).mean()   # TARGET_AREA = 6·MIN_AREA_THRESHOLD = 300 px
total = (w_c·L_compact + w_b·L_boundary + w_e·L_circle_template) * area_scale
```

Quindi ogni termine geometrico è modulato **due volte**: una volta internamente
per l'area (evita di combattere Dice+CE quando la maschera predetta è ancora
troppo piccola), e una volta esternamente nel trainer per il Dice gate (evita
di penalizzare la forma quando la rete ha completamente fallito la
localizzazione).

## 3. Warmup, rampa e Dice gate "splittato"

**Warmup + rampa** (in `train_step`):

```python
ramp_epochs   = max(1, geometric_loss_warmup_ramp)     # = WARMUP_EPOCHS
warmup_scale  = clamp((current_epoch - WARMUP_EPOCHS) / ramp_epochs, 0, 1)
```

- Epoche `[0, WARMUP_EPOCHS)`: `warmup_scale = 0` → training è **solo Dice+CE**.
- Epoche `[WARMUP_EPOCHS, 2·WARMUP_EPOCHS)`: rampa lineare `0 → 1`.
- Epoche successive: `warmup_scale = 1` (peso pieno).

Con `WARMUP_EPOCHS = 20` (valore corrente in `config.py`): niente geometria fino
all'epoca 20, rampa fino all'epoca 40, poi peso pieno.

**Dice gate "splittato" in due varianti** (calcolate `torch.no_grad()`, usate
solo come moltiplicatori scalari, non propagano gradiente):

| Gate | Formula (esatta dal codice) | Applicato a | Motivazione |
|------|------------------------------|-------------|-------------|
| `dice_gate` (hard) | `mean( clamp((dice_per_sample - 0.5)/0.5, 0, 1) )` — il clamp è **per-campione**, la media viene dopo | **Compactness** | Ogni campione sotto Dice 0.5 contribuisce 0 al gate (non solo la media del batch); sale linearmente a 1 per campione quando il suo Dice va da 0.5 a 1.0, poi si fa la media sul sotto-batch. Impedisce alla compactness di "restringere" predizioni ancora sbagliate/vuote. |
| `dice_gate_shape` (soft) | `clamp( mean(dice_per_sample), 0, 1 )` — qui invece si fa prima la media, poi il clamp | **Boundary + Circle Template** | Resta attivo (parzialmente) anche con Dice medio basso (es. 0.1–0.4): l'idea è spingere anche i blob sbagliati verso una forma più circolare/liscia, invece di lasciarli completamente liberi di essere allungati mentre aspettano che il Dice migliori. |

> Attenzione: l'ordine `clamp` → `mean` non è intercambiabile con `mean` →
> `clamp` in generale. Il codice usa deliberatamente ordini **diversi** per i
> due gate (clamp-poi-media per `dice_gate`, media-poi-clamp per
> `dice_gate_shape`) — non è un refuso, sono due implementazioni distinte.

La separazione nasce da un problema osservato: con un gate unico e "duro" i
campioni con training fallito (Dice basso) restavano privi di qualunque
pressione geometrica, producendo blob allungati che peggioravano
l'eccentricità media aggregata a fine training.

## 4. Safety net anti-NaN durante il training

Diversi livelli di protezione, tutti pensati per **non rompere il
computational graph** (mai sostituire un tensore con `torch.tensor(0.0)` "nudo":
si usa sempre un'espressione tipo `pred_soft.sum() * 0.0`, che vale
matematicamente zero ma resta collegata al grafo per il backward):

1. **Dentro ogni singola loss geometrica** (`_vectorized_compactness_loss`,
   `_vectorized_boundary_loss`, `_vectorized_circle_template_loss`): se l'area
   soft di *tutti* i campioni nel sotto-batch è sotto `MIN_AREA_THRESHOLD`
   (50 px), la funzione ritorna `pred_soft.sum() * 0.0` prima di fare qualunque
   calcolo. Se solo alcuni campioni sono validi, la loss si media solo su
   quelli (`loss_per_batch[valid_mask]`).
2. **Clamp aggressivi prima di ogni `sqrt`** (es. `sqrt(clamp(x, min=0) + 1e-2)`)
   per evitare gradienti che esplodono quando l'argomento tende a 0.
3. **Check NaN/Inf per ogni componente** subito dopo il calcolo, con fallback a
   `pred_soft.sum() * 0.0`.
4. **Nel trainer**, prima del backward: se `total_loss` è NaN/Inf, si ripiega su
   `loss_dice_ce`; se anche quella è NaN/Inf, lo step viene **skippato**
   (`optimizer.zero_grad()`, nessun update).
5. **Durante/dopo il backward**: `try/except` su `RuntimeError` contenenti
   "nan"/"inf" → skip; **gradient clipping** con `clip_grad_norm_(..., 12)` e
   controllo che la norma risultante non sia NaN/Inf → in tal caso skip
   dell'`optimizer.step()`.
6. **Contatori diagnostici** (`verify_stats`) che tracciano quante volte la
   loss geometrica è stata tentata, applicata, bypassata per warmup, bypassata
   per eccezione o per NaN/Inf, e quante volte l'optimizer step è stato
   skippato — utili per verificare a posteriori quanto la loss geometrica ha
   realmente influenzato il training di un dato esperimento.

## 5. Configurazione effettiva e pesi finali usati

Valori correnti in [`config.py`](../config.py) (esperimento `test_40_4_pesi`,
l'ultimo configurato nel repository):

```python
WEIGHT_COMPACTNESS   = 0.015
WEIGHT_ECCENTRICITY  = 0.025   # in realtà pesa la Circle Template Loss (V2.8)
WEIGHT_BOUNDARY      = 0.006
WARMUP_EPOCHS        = 20
EPOCHS               = 100
BATCH_SIZE           = 8
GEOMETRIC_LOSS_SAMPLES = 4     # campioni per batch su cui si calcola la loss geometrica
MIN_AREA_THRESHOLD   = 50.0    # hardcoded nel template di _write_geometric_config() in run_experiment.py:
                                # non è una variabile di config.py, quindi oggi NON è modificabile
                                # dall'utente senza toccare run_experiment.py.
```

Questi valori vengono scritti in `geometric_config.py` da `run_experiment.py`
a ogni run e letti sia da `geometric_losses.py` che da
`nnUNetTrainerGeometric.py` (con fallback hardcoded identici a questi valori
se il file di config non fosse importabile — di fatto i default nel codice
sono già allineati ai valori "di produzione").

---

# Parte 2 — I vincoli geometrici: formule e motivazione

## 6. Principio comune a tutti i termini: operare su probabilità soft

Tutti e tre i termini operano su `pred_soft = softmax(logits)[:, 1, :, :]`
(probabilità continua della classe "cerchio" per ogni pixel), **mai** su una
maschera binarizzata. Questo è il vincolo di progettazione più importante:
qualunque operazione non differenziabile (soglia hard, `cv2.findContours`,
convex hull discreto, `.cpu().numpy()`) romperebbe il grafo computazionale di
PyTorch e la rete non riceverebbe alcun gradiente dalla geometria — problema
che nella cronologia del progetto ha effettivamente causato un bug critico
(loss geometrica sempre a zero gradiente nella primissima versione).

## 7. Termine 1 — Compactness

**Definizione geometrica classica** (isoperimetric inequality): per una figura
piana con area $A$ e perimetro $P$,

$$
C = \frac{4\pi A}{P^2}, \qquad C \le 1
$$

con uguaglianza $C=1$ **se e solo se** la figura è un cerchio. È la formula
standard usata anche in analisi d'immagine per misurare "circolarità": penalizza
sia perimetri eccessivi a parità di area (bordi frastagliati) sia forme non
circolari (es. ellissi molto allungate hanno perimetro grande rispetto
all'area).

**Approssimazione differenziabile implementata:**

- **Area soft:** $A = \sum_{i,j} p_{ij}$ (somma delle probabilità, non conteggio
  di pixel binari).
- **Perimetro soft:** magnitudine del gradiente spaziale di $p$, ottenuta con
  filtri di Sobel convoluzionali:
  $$
  g_x = \text{Sobel}_x * p, \qquad g_y = \text{Sobel}_y * p
  $$
  $$
  P = \sum_{i,j} \sqrt{\,\text{clamp}(g_x^2+g_y^2,\,0) + \varepsilon\,}, \qquad \varepsilon = 10^{-2}
  $$
  L'idea è che il gradiente spaziale di una mappa di probabilità è alto
  esattamente nelle zone di transizione foreground/background, cioè lungo il
  "bordo soft" — la sua somma è quindi un proxy ragionevole della lunghezza del
  contorno.
- **Compactness e loss:**
  $$
  C = \operatorname{clamp}\!\left(\frac{4\pi A}{P^2+\varepsilon},\,0,\,1\right), \qquad
  \mathcal{L}_{\text{compact}} = 1 - C
  $$

**Perché il clamp $C\le 1$:** con approssimazioni soft $C$ potrebbe
teoricamente superare leggermente 1 per rumore numerico; il clamp forza la
loss a restare $\ge 0$ e coerente con la definizione teorica (un cerchio
perfetto è il massimo di compattezza tra le forme "semplici" considerate qui).

**Perché ha senso:** è il termine più "generale" — non richiede sapere dove
sia il centro o quale sia il raggio atteso, penalizza in un colpo solo sia
irregolarità del bordo che allungamento. È il termine scelto per rimanere
attivo **solo quando la rete ha già trovato il cerchio** (gate Dice "hard",
§3), perché agire su una predizione ancora sbagliata rischierebbe di spingerla
verso zero (un'area piccola e "pulita" ha comunque $C$ più alta di un blob
grande e frastagliato, quindi la compactness da sola tende a *rimpicciolire*
le predizioni incerte).

## 8. Termine 2 — Boundary smoothness

**Idea geometrica:** un cerchio ha curvatura costante lungo tutto il bordo;
forme irregolari o frastagliate hanno variazioni di curvatura elevate. Il
proxy usato è l'operatore Laplaciano (derivata seconda spaziale), che risponde
fortemente nelle zone dove il bordo cambia direzione bruscamente.

**Implementazione:**

$$
\Delta p = \text{Laplaciano}_{3\times3} * p, \qquad \text{kernel} =
\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}
$$

Il Laplaciano viene pesato per la probabilità stessa (non per una maschera
binaria a soglia, per non azzerare i gradienti fuori da una soglia arbitraria).
Nel codice la sequenza esatta di clamp è a **due stadi**:

$$
\Delta p' = \operatorname{clamp}(\Delta p,\,-100,\,100) \quad\text{(subito dopo la convoluzione)}
$$
$$
\widetilde{\Delta p} = \operatorname{clamp}\big(\Delta p' \cdot p,\; -10,\; 10\big) \quad\text{(dopo la moltiplicazione per } p\text{)}
$$

$$
\mathcal{L}_{\text{boundary}} = \operatorname{Var}\!\big(\widetilde{\Delta p}\big) + 0.1\cdot \operatorname{mean}\big(|\widetilde{\Delta p}|\big)
$$

(varianza calcolata con `unbiased=False`, per immagine, poi mediata sul batch;
stesso per la media assoluta). Il primo clamp $[-100,100]$ limita la risposta
grezza della convoluzione, il secondo $[-10,10]$ limita il prodotto pesato —
sono due protezioni numeriche distinte, non una singola operazione.

**Perché ha senso:** la varianza del Laplaciano penalizza le zone dove il
bordo "salta" (spigoli, rientranze locali, rumore ad alta frequenza sul
contorno); il termine di media assoluta scoraggia anche curvature
sistematicamente troppo alte. Insieme spingono verso un contorno liscio, che è
condizione necessaria (ma non sufficiente) per un cerchio.

## 9. Termine 3 — Circle Template Loss (il vero termine "eccentricity" in uso)

Questo è il termine che nel codice ha nome storico `eccentricity` ma dalla
**V2.8** in poi è implementato in modo completamente diverso rispetto a una
classica loss basata su momenti d'inerzia (vedi §10 per la versione precedente,
ancora presente nel file ma non chiamata).

### Perché è stato sostituito il termine a momenti

La vecchia formula (V2.5, `_vectorized_eccentricity_loss`) calcolava
$L = (1-\lambda_{\min}/\lambda_{\max})^2$ dai momenti secondi d'inerzia. Il
problema: per predizioni **già quasi circolari** (rapporto assi $\approx 0.95$)
questa loss vale $\approx 0.0025$ e il suo gradiente rispetto ai logit è quasi
nullo — proprio nel regime in cui servirebbe rifinire l'ultima parte di
ellitticità residua, il termine smette di dare segnale utile.

### La soluzione: confronto diretto con un cerchio "template"

1. **Stop-gradient sulle statistiche globali** — da `pred_soft` corrente si
   calcolano, **senza** propagare gradiente attraverso queste quantità
   (`torch.no_grad()`):
   - il centroide pesato:
     $$
     c_x = \frac{\sum_{ij} p_{ij}\,x_{ij}}{A}, \qquad c_y = \frac{\sum_{ij} p_{ij}\,y_{ij}}{A}
     $$
   - l'area soft $A = \sum_{ij} p_{ij}$ e il raggio del cerchio che avrebbe la
     stessa area:
     $$
     r = \sqrt{\frac{A}{\pi}}, \qquad r \in \Big[1,\ \tfrac{\max(H,W)}{2}\Big]
     $$
2. Si costruisce un **template circolare soft**, con bordo nitido ma continuo
   tramite una sigmoide:
   $$
   d_{ij} = \sqrt{(x_{ij}-c_x)^2 + (y_{ij}-c_y)^2 + \varepsilon}
   $$
   $$
   T_{ij} = \sigma\big(k\,(r - d_{ij})\big), \qquad k = \text{TEMPLATE\_SHARPNESS} = 5.0
   $$
   Dentro il cerchio ideale (di raggio $r$ e centro $(c_x,c_y)$) $T\approx 1$,
   fuori $T\approx 0$, con transizione morbida ma stretta attorno al bordo.
3. **Loss = errore quadratico medio tra predizione e template, normalizzato
   per l'area del template** (non per $H\times W$):
   $$
   \mathcal{L}_{\text{circle}} = \frac{\sum_{ij}(p_{ij}-T_{ij})^2}{\max\!\big(\sum_{ij}T_{ij},\,1.0\big)}
   $$
   (nel codice il denominatore è ottenuto con `.clamp(min=1.0)`, non con
   un'aggiunta di epsilon — la differenza è rilevante solo per template quasi
   vuoti, dove clamp forza almeno area 1.0 invece di un epsilon infinitesimo).

**Perché normalizzare per l'area del template e non per $H\times W$:**
mediare su tutta l'immagine (es. $512\times512=262144$ pixel) diluirebbe il
segnale di un fattore 50–100×, dato che il cerchio occupa tipicamente solo il
3–8% dei pixel. Normalizzando per l'area del template il segnale resta
proporzionato alla dimensione effettiva del cerchio, e il peso
`WEIGHT_ECCENTRICITY` risulta comparabile in ordine di grandezza a
`WEIGHT_COMPACTNESS` senza dover essere ri-tarato per ogni scala di immagine.

**Perché è più efficace della loss a momenti:** dove la predizione è ellittica
rispetto al cerchio di pari area, esistono ampie regioni in cui $p$ e $T$ **non
coincidono** — i "vertici" allungati dell'ellisse (dove $p$ è alto ma $T\approx
0$) e i "poli" mancanti lungo l'asse minore (dove $T$ è alto ma $p$ è basso).
Questo produce un contributo alla loss sostanziale e un gradiente
**direzionale**: spinge la rete a *ridurre* la probabilità nei vertici
allungati e ad *aumentarla* dove il cerchio atteso la richiede — invece del
gradiente quasi piatto della vecchia formula sui rapporti tra autovalori.

**Nota concettuale importante:** a differenza del nome "eccentricity" suggerito
dal codice, questo termine non misura l'eccentricità di un'ellisse fittata
sulla predizione — confronta la predizione con il **cerchio di pari area e
baricentro**. Cattura quindi sia deviazioni dalla forma circolare (ellitticità)
sia asimmetrie locali della massa di probabilità attorno al proprio baricentro,
in un unico segnale di errore pixel-per-pixel.

## 10. Il termine a momenti di inerzia: presente nel codice ma NON usato

Nel file `geometric_losses.py` è ancora definito il metodo
`_vectorized_eccentricity_loss` (V2.5), mantenuto **solo per riferimento /
esperimenti futuri**. Formula:

$$
\mu_{20}=\frac{\sum p\,(x-c_x)^2}{A},\quad
\mu_{02}=\frac{\sum p\,(y-c_y)^2}{A},\quad
\mu_{11}=\frac{\sum p\,(x-c_x)(y-c_y)}{A}
$$

$$
\lambda_{1,2} = \frac{(\mu_{20}+\mu_{02}) \pm \sqrt{(\mu_{20}+\mu_{02})^2 - 4(\mu_{20}\mu_{02}-\mu_{11}^2)}}{2}
$$

$$
\mathcal{L}_{\text{aspect}} = \left(1-\frac{\lambda_{\min}}{\lambda_{\max}}\right)^2
$$

Il metodo `__call__` di `DifferentiableGeometricLossesV2` **chiama invece
sempre** `_vectorized_circle_template_loss` (§9) per il contributo
"eccentricity". Questo metodo a momenti non è collegato a nessun gradiente
usato in training nella versione corrente del codice.

## 11. Perché questi tre termini insieme hanno senso geometricamente

| Termine | Cosa cattura | Cosa NON cattura da solo |
|---------|--------------|---------------------------|
| Compactness | Rapporto globale area/perimetro² — penalizza sia allungamento sia frastagliatura, in un solo numero | Non dice *dove* la forma sbaglia, né distingue un'ellisse liscia da una forma frastagliata con la stessa compattezza |
| Boundary smoothness | Regolarità locale del contorno (assenza di spigoli/rumore ad alta frequenza) | Un contorno può essere liscio ma comunque molto ellittico (es. un'ellisse ha bordo liscio ma bassa compactness) |
| Circle Template | Confronto diretto pixel-per-pixel con il cerchio "atteso" (stessa area, stesso baricentro) — dà gradiente direzionale forte anche vicino alla convergenza | Da solo non penalizza rumore ad alta frequenza dentro l'area del template (che compactness/boundary invece colgono) |

I tre termini sono complementari: la compactness dà un segnale globale forte
quando la forma è ancora molto lontana da un cerchio; il boundary rifinisce la
regolarità locale del contorno; la Circle Template Loss fornisce il gradiente
più "chirurgico" nelle fasi avanzate del training, quando le altre due loss
tenderebbero a saturare. La combinazione con **area ramp** (attivazione
progressiva tra 50 e 300 px di massa soft) e **Dice gate splittato** (§3) evita
che questi tre segnali combattano contro Dice+CE nelle fasi in cui la rete sta
ancora imparando a localizzare il cerchio.

## 12. Limiti noti e disallineamento training/valutazione

Va tenuto presente (approfondito anche in
[`ANALISI_METRICHE.md`](ANALISI_METRICHE.md)) che le definizioni usate **in
training** (soft, su `pred_soft`) non coincidono esattamente con le metriche
usate **in valutazione** (`metrics_utils.py`, su maschere binarizzate a 0.5 con
OpenCV):

- **Compactness:** in training il perimetro è una somma di gradienti Sobel su
  probabilità continue; in valutazione è `cv2.arcLength` su un contorno
  discreto binarizzato. Sono proxy della stessa quantità ma non identici
  numericamente.
- **"Eccentricity":** in training è la Circle Template Loss (confronto con un
  cerchio di pari area/baricentro); in valutazione è l'eccentricità vera di
  un'ellisse fittata col contorno (`cv2.fitEllipse`,
  $\sqrt{1-(b/a)^2}$). Sono **concettualmente diverse**: la rete non sta
  ottimizzando direttamente l'eccentricità riportata nei risultati finali, ma
  un segnale correlato (deviazione dal cerchio atteso).

Per questo motivo un miglioramento della loss in training non garantisce un
miglioramento monotono della colonna "eccentricity" in
`metrics_comparison.txt`; la compactness (definizione più affine tra training e
valutazione) tende ad allinearsi meglio.

---
