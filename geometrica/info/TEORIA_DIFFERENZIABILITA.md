# Teoria: Differenziabilità e Computational Graph nelle Loss Geometriche

## Contesto del Problema

Nel progetto di segmentazione dei cerchi con nnU-Net, abbiamo implementato loss geometriche custom per migliorare la qualità delle predizioni. Queste loss calcolano proprietà geometriche (compattezza, solidità, eccentricità, smoothness dei bordi) per guidare la rete verso forme più circolari.

## Il Bug Critico: Interruzione del Computational Graph

### Cosa è successo

Durante il training, la rete raggiungeva l'epoca 65-76 e improvvisamente tutti i pesi diventavano NaN, rendendo il modello inutilizzabile. Il problema non era nei valori delle loss geometriche stesse, ma nel modo in cui gestivamo i casi edge.

### La Causa Nascosta

Quando una loss geometrica produceva un valore NaN (es. per immagini senza foreground), il codice la sostituiva con:

```python
loss = torch.tensor(0.0, device=device, dtype=dtype)
```

Questo codice, apparentemente innocuo, crea un **tensore foglia** (leaf tensor) completamente disconnesso dal computational graph. È come tagliare un ramo da un albero: il nodo esiste ancora, ma non ha più connessioni con la radice.

### Perché questo causa NaN nei pesi?

Durante il backward pass, PyTorch propaga i gradienti attraverso il computational graph. Quando incontra un tensore foglia disconnesso:

1. **Gradient Flow Interrotto**: I gradienti non possono fluire attraverso quel nodo
2. **Gradienti Instabili**: Alcune parti della rete ricevono gradienti, altre no
3. **Accumulo di Instabilità**: Epoch dopo epoch, l'instabilità si accumula
4. **Collasso Finale**: I pesi raggiungono valori così grandi da diventare NaN

È un problema insidioso perché non si manifesta immediatamente - ci vogliono 60+ epoche prima che l'instabilità diventi critica.

## La Soluzione: Mantenere il Computational Graph Intatto

### Principio Fondamentale

**Ogni operazione differenziabile deve mantenere connessioni al computational graph originale.**

Invece di creare un nuovo tensore da zero, usiamo operazioni che mantengono la connessione:

```python
loss = pred_soft.sum() * 0.0
```

Questa semplice espressione:
- Calcola `pred_soft.sum()` - tensore connesso al graph
- Lo moltiplica per 0.0 - operazione differenziabile
- Risultato: zero matematico, ma con gradient flow intatto

### Perché funziona?

Durante il backward:
- `d/dx (x * 0.0) = 0.0` - gradiente matematicamente corretto
- Il gradiente può propagarsi attraverso `.sum()` fino a `pred_soft`
- La rete riceve gradienti consistenti anche quando la loss geometrica è zero
- Nessuna interruzione nel computational graph

## Differenziabilità delle Loss Geometriche

### Challenge: Operazioni Geometriche Non Differenziabili

Le proprietà geometriche classiche (compattezza, area, perimetro) coinvolgono:
- **Contorni discreti**: `cv2.findContours()` - non differenziabile
- **Operazioni booleane**: threshold binari - non differenziabili
- **Conteggi**: numero di pixel - non differenziabile

### Soluzione: Approssimazioni Differenziabili

Abbiamo implementato versioni soft (differenziabili) di tutte le metriche:

#### 1. Area Soft
Invece di contare pixel binari: `sum(mask == 1)`
Usiamo: `sum(softmax_probabilities)`

La probabilità soft è un proxy differenziabile dell'area.

#### 2. Perimetro Soft
Invece di estrarre contorni discreti, calcoliamo il gradiente spaziale della probabilità:
- Gradient X e Y tramite convoluzione
- Magnitudine del gradiente = "bordo soft"
- Somma = perimetro approssimato

#### 3. Momenti Geometrici Soft
Per calcolare centroide ed eccentricità:
- Momenti del primo ordine (posizione): `sum(x * p(x,y))`
- Momenti del secondo ordine (forma): `sum(x² * p(x,y))`
- Tutti differenziabili perché usano moltiplicazioni e somme

#### 4. Boundary Smoothness Soft
Invece di analizzare contorni discreti:
- Convoluzione del gradiente per rilevare discontinuità
- Varianza locale del gradiente = smoothness inversa
- Tutto tramite operazioni convoluzionali differenziabili

## Safety Checks: Difesa in Profondità

Oltre al fix del computational graph, abbiamo implementato tre livelli di protezione:

### Livello 1: Validazione Pre-Backward
Prima di chiamare `.backward()`, verifichiamo se la loss totale è NaN:
- Se sì: fallback alla loss standard (Dice+CE)
- Se no: procediamo normalmente

Questo previene la propagazione di NaN nel computational graph.

### Livello 2: Validazione Gradienti
Dopo `.backward()` ma prima di `.optimizer.step()`, scanniamo tutti i gradienti:
- Se troviamo NaN in qualsiasi parametro: skip optimizer step
- Altrimenti: applica update normalmente

Questo previene la corruzione dei pesi anche se NaN appare nei gradienti.

### Livello 3: Logging Diagnostico
Ogni 10 epoche logghiamo:
- Breakdown dettagliato delle loss (Dice+CE, Geometric, componenti)
- Warning immediati se appare NaN
- Informazioni sul gradient flow

Questo permette diagnosi rapida se qualcosa va storto.

## Risultato Finale

Con queste modifiche (V2.2):

### Prima (V2.1)
- Training normale fino a epoca 60-76
- Loss improvvisamente → NaN
- Tutti i pesi corrotti
- Modello inutilizzabile

### Dopo (V2.2)
- Computational graph sempre intatto
- Gradienti consistenti durante tutto il training
- Safety checks prevengono casi edge
- Training completo fino a 100 epoche senza crash

## Lezione Chiave: PyTorch e il Computational Graph

**In PyTorch, non è sufficiente che un'operazione produca il valore matematico corretto - deve anche mantenere le connessioni del computational graph per permettere la backpropagation.**

Operazioni apparentemente equivalenti dal punto di vista matematico possono avere comportamenti drasticamente diversi durante il training:

- `torch.tensor(0.0)` → Valore corretto, gradient flow ROTTO
- `x.sum() * 0.0` → Valore corretto, gradient flow INTATTO

Questa sottile differenza determina se il training avrà successo o collasserà.

## Le Regole Geometriche: Quali e Perché

Per guidare la rete a produrre segmentazioni circolari migliori, abbiamo scelto tre proprietà geometriche fondamentali che caratterizzano un cerchio perfetto.

### 1. Compactness (Compattezza)

**Definizione:** Il rapporto tra area e perimetro quadrato.

**Formula:**
```
Compactness = (4π · Area) / Perimetro²
```

**Perché è importante:**
- Un cerchio perfetto ha compactness = 1.0 (valore massimo possibile)
- Forme irregolari, allungate o frastagliate hanno compactness < 1.0
- Questa metrica penalizza simultaneamente:
  - Forme con perimetro eccessivo rispetto all'area (bordi frastagliati)
  - Forme non compatte (es. ellissi molto allungate)

**Implementazione differenziabile:**
- **Area soft**: `sum(probabilità_pixel)` invece di contare pixel binari
- **Perimetro soft**: magnitudine del gradiente spaziale (operatore Sobel)
  - `gradient_x = conv2d(prob, sobel_x)`
  - `gradient_y = conv2d(prob, sobel_y)`
  - `perimetro = sum(√(gx² + gy²))`

**Loss associata:**
```
L_compactness = 1 - Compactness
```
Minimizzare questa loss spinge la forma verso compactness = 1.0 (cerchio).

### 2. Boundary Smoothness (Lisciezza del Bordo)

**Definizione:** Regolarità delle curvature lungo il perimetro.

**Formula (via Laplacian):**
```
Laplacian = ∇²f = ∂²f/∂x² + ∂²f/∂y²
Smoothness = -Var(Laplacian)
```

**Perché è importante:**
- I cerchi hanno curvatura costante lungo tutto il perimetro
- Forme irregolari mostrano alta varianza nelle derivate seconde
- Penalizza:
  - Angoli acuti o spigoli
  - Irregolarità nel contorno
  - "Rumore" nella maschera di segmentazione

**Implementazione differenziabile:**
- Kernel Laplaciano discreto 3×3: `[[0,1,0], [1,-4,1], [0,1,0]]`
- Convoluzione sulla mappa di probabilità
- La varianza alta della risposta Laplaciana indica irregolarità

**Loss associata:**
```
L_boundary = Var(Laplacian(prob)) + 0.1·Mean(|Laplacian(prob)|)
```
Due componenti:
1. Varianza: penalizza cambi di curvatura
2. Media assoluta: penalizza curvature eccessive

### 3. Aspect Ratio (Rapporto Assi)

**Definizione:** Rapporto tra asse maggiore e minore della forma.

**Formula (via momenti di inerzia):**
```
Centro di massa:
  x̄ = Σ(x·p(x,y)) / Σp(x,y)
  ȳ = Σ(y·p(x,y)) / Σp(x,y)

Momenti centrali di secondo ordine:
  μ₂₀ = Σ((x-x̄)²·p(x,y)) / Area
  μ₀₂ = Σ((y-ȳ)²·p(x,y)) / Area
  μ₁₁ = Σ((x-x̄)(y-ȳ)·p(x,y)) / Area

Eigenvalues (assi principali):
  trace = μ₂₀ + μ₀₂
  det = μ₂₀·μ₀₂ - μ₁₁²
  λ₁ = (trace + √(trace² - 4·det)) / 2
  λ₂ = (trace - √(trace² - 4·det)) / 2

Aspect Ratio:
  AR = λ₁ / λ₂
```

**Perché è importante:**
- Un cerchio ha aspect ratio = 1.0 (assi uguali)
- Ellissi allungate hanno AR > 1
- Questa metrica cattura sia:
  - **Eccentricità**: quanto la forma è allungata
  - **Solidity**: quanto riempie il suo bounding box

**Implementazione differenziabile:**
- Tutti i momenti calcolati come prodotti pesati e somme
- Nessuna operazione non differenziabile (no threshold, no contorni discreti)
- La radice quadrata include protezione: `√(max(x, 0))` per stabilità numerica

**Loss associata:**
```
L_aspect = |AR - 1| / (AR + 1)
```
Normalizzata per avere range [0, 1], minimizza quando AR = 1.

## Scelta dei Pesi: Bilanciamento e Stabilità

### Pesi Configurati (V2.2)

```
weight_compactness  = 0.01   (1%)
weight_boundary     = 0.005  (0.5%)
weight_aspect       = 0.015  (1.5%)  [combinazione di solidity + eccentricity]
──────────────────────────────────
Total geometric     = 0.03   (3%)
```

### Rapporto con Loss Standard

La loss totale durante training è:
```
L_total = L_dice_ce + L_geometric
L_total = L_dice_ce + 0.01·L_comp + 0.005·L_bound + 0.015·L_aspect
```

Dove:
- **L_dice_ce**: Loss standard nnU-Net (Dice + Cross-Entropy), tipicamente range [0.5, 1.5]
- **L_geometric**: Contributo geometrico, range [0, 0.03]

### Perché Questi Pesi?

**1. Predominanza della Loss Standard (97%)**
- La segmentazione base deve funzionare bene (Dice alta)
- Le loss geometriche sono una "guida delicata", non una forzatura
- Se i pesi fossero troppo alti (es. 0.1), la rete ignorerebbe Dice e produrrebbe cerchi perfetti nel posto sbagliato

**2. Compactness Dominante (0.01)**
- È la metrica più fondamentale: cattura l'essenza di "essere un cerchio"
- Peso più alto perché è la più importante

**3. Boundary Smoothness Ridotto (0.005)**
- Contributo secondario: affina i bordi dopo che la forma è grossolanamente corretta
- Peso minore perché troppa smoothness può causare perdita di dettagli ai bordi

**4. Aspect Ratio Intermedio (0.015)**
- Peso combinato: originariamente `solidity (0.01) + eccentricity (0.005)`
- Previene forme allungate ma non sovra-penalizza piccole deviazioni dalla circolarità perfetta

### Evoluzione dei Pesi

**Versione Iniziale (V1.0):**
```
weight_compactness  = 0.1    ❌ Troppo alto → instabilità
weight_boundary     = 0.05   ❌ Troppo alto → instabilità
weight_aspect       = 0.05   ❌ Troppo alto → instabilità
```
Risultato: Training collassava con NaN dopo poche epoche.

**Versione Stabile (V2.2):**
```
weight_compactness  = 0.01   ✅ Ridotto 10×
weight_boundary     = 0.005  ✅ Ridotto 10×
weight_aspect       = 0.015  ✅ Ridotto ~3×
```
Risultato: Training stabile fino a 100 epoche.

### Warm-up Strategy

Per ulteriore stabilità, attiviamo le loss geometriche gradualmente:

```
Epoche 0-19:   Solo Dice+CE (warm-up)
Epoche 20-99:  Dice+CE + Geometric (training completo)
```

Durante il warm-up (prime 20 epoche):
- La rete impara segmentazione base senza pressione geometrica
- I pesi si stabilizzano in una configurazione ragionevole
- Evita instabilità iniziali quando i pesi sono randomici

## Riferimenti Teorici

- **Automatic Differentiation**: Ogni operazione PyTorch mantiene informazioni sulla sua derivata
- **Computational Graph**: Struttura DAG (Directed Acyclic Graph) che traccia le dipendenze tra tensori
- **Leaf Tensors**: Tensori senza genitori nel graph - terminano la backpropagation
- **In-place Operations**: Modifiche dirette ai tensori possono corrompere il graph (PyTorch le rileva)
- **Detached Tensors**: Tensori esplicitamente disconnessi dal graph via `.detach()`
- **Image Moments**: Momenti statistici di ordine N per caratterizzare forme geometriche
- **Sobel Operator**: Filtro convoluzionale per derivate spaziali del primo ordine
- **Laplacian Operator**: Filtro convoluzionale per derivate spaziali del secondo ordine

La nostra correzione rispetta questi principi mantenendo tutte le operazioni connesse al graph originale attraverso operazioni differenziabili standard (somma, moltiplicazione, convoluzione).
