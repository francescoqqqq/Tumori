# Progetto nnU-Net: Segmentazione Clinica e Test Geometrici

Questo repository è organizzato in due progetti principali per testare e migliorare nnU-Net:

## 📁 Struttura del Progetto

### 🏥 `clinico/` - Segmentazione Tumori Cerebrali
Progetto dedicato alla segmentazione di tumori cerebrali usando dataset BraTS 2023. 
Qui si testano i limiti di nnU-Net nella segmentazione clinica reale.

**Contenuto:**
- Dataset ibrido BraTS (Reale + Sintetico "Frankenstein")
- Script per download, generazione e preprocessing
- Training e valutazione con nnU-Net
- Metriche e visualizzazioni

Vedi `clinico/README.md` per dettagli completi.

### 🔷 `geometrica/` - Test su Figure Geometriche
Progetto per testare nuove metriche, loss functions e architetture su dataset sintetici semplici (figure geometriche).
Questo permette di validare approcci su problemi più semplici prima di applicarli ai dati clinici.

**Contenuto:**
- Generatore di dataset sintetici con cerchi e figure geometriche
- Dataset pre-generato (`dataset_shapes/`)
- Script per test e validazione

Vedi `geometrica/README.md` per dettagli completi.

## 🎯 Obiettivo

1. **Clinico**: Valutare le performance di nnU-Net su dati clinici reali e identificare limiti e aree di miglioramento
2. **Geometrica**: Sviluppare e testare nuove metodologie su dataset semplici, poi applicarle al contesto clinico

## 🚀 Quick Start

### Setup Clinico
```bash
cd clinico
# Segui le istruzioni in clinico/README.md
```

### Setup Geometrico
```bash
cd geometrica
# Segui le istruzioni in geometrica/README.md
```
