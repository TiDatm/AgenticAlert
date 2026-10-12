# Ukrainian Air Raid Defense Forecasting Engine & Spatio-Temporal STGNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning and survival analysis framework for real-time forecasting of air raid alerts across Ukraine. The architecture features a **Hierarchical Spatio-Temporal Graph Neural Network (STGNN)** modeling 25 macro-regions (oblasts + Kyiv) and 136 micro-regions (raions), tactical kinetic flight corridors, Weibull time-to-incursion survival modeling, PCGrad multi-task gradient conflict resolution, and zero-leakage causal rolling evaluation.

---

## 🧠 Interactive Conversation Brain & Development Transcript

The entire design, research, implementation, iterative debugging, and mathematical derivations for this project have been archived directly inside this repository.

* 🌐 **Live Interactive Web Viewer (Instant Preview)**:  
  👉 **[Open Interactive Conversation Viewer](https://htmlpreview.github.io/?https://github.com/TiDatm/AgenticAlert/blob/main/docs/index.html)**
* 📖 **GitHub Pages Deployment**:  
  When GitHub Pages is enabled on this repo (pointing to `/docs`), access the viewer at:  
  👉 **[https://TiDatm.github.io/AgenticAlert/](https://tidatm.github.io/AgenticAlert/)**
* 💻 **Local Offline Viewing**:  
  Open [`docs/index.html`](docs/index.html) or [`brain/conversation_viewer.html`](brain/conversation_viewer.html) in any web browser.
* 📦 **Archived Brain Data**:  
  Located in [`brain/`](brain/) with full transcripts ([`brain/logs/transcript_full.jsonl`](brain/logs/transcript_full.jsonl)), research blueprints ([`brain/artifacts/`](brain/artifacts/)), and high-resolution diagnostic dashboards ([`brain/media/`](brain/media/)).

---

## ⚡ Architectural Highlights

```
                               ┌────────────────────────────────────────────────────────┐
                               │  Tactical Air Force Ingestion (Telegram / Sirens API)  │
                               └───────────────────────────┬────────────────────────────┘
                                                           ▼
                                ┌──────────────────────────────────────────────────────┐
                                │   Causal Dual-Stream Resampling (15-min / 1-hour)    │
                                └──────────────────────────┬───────────────────────────┘
                                                           ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                    Hierarchical STGNN Backbone                                         │
  │                                                                                                        │
  │   1. Tactical Kinetic Encoder (GRU + Flight Isochrones + Frontier Proximity)                           │
  │   2. Strategic Campaign Encoder (24h Diurnal / Seasonality / Launch Wave Signals)                      │
  │   3. Raion Spatial Refinement (136 Rayons -> Dynamic Graph Laplacian Convolutions)                     │
  │   4. Directed Flight Graph Message Passing (Azimuth transit corridors & ballistic ingress paths)       │
  │   5. Multi-Task Heads: 1h / 4h / 24h Alert Classification + Weibull Time-to-Incursion Survival Heads   │
  └───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                      ▼
                       ┌──────────────────────────────────────────────────────────────┐
                       │  Multi-Task Defense Loss with PCGrad (Conflicting Gradients) │
                       │    - Asymmetric Cost-Sensitive Focal Loss (γ=2.0, α=0.70)    │
                       │    - Censored Survival Log-Likelihood                        │
                       └──────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Dashboards & Research Blueprints

* **Interactive Dashboards**:
  * [`regional_accuracy_dashboard.html`](regional_accuracy_dashboard.html) — Regional precision, recall, and PR-AUC breakdown across all 25 Ukrainian administrative regions.
  * [`regional_onset_dashboard.html`](regional_onset_dashboard.html) — First-hour transition/incursion prediction diagnostics.
* **Core Research Blueprints**:
  * [`BLUEPRINT_KINEMATIC_ISOCHRONE_AND_SURVIVAL_DEFENSE.md`](BLUEPRINT_KINEMATIC_ISOCHRONE_AND_SURVIVAL_DEFENSE.md)
  * [`RESEARCH_AIR_RAID_DEFENSE.md`](RESEARCH_AIR_RAID_DEFENSE.md)
  * [`RESEARCH_DEEP_LEARNING_STGNN.md`](RESEARCH_DEEP_LEARNING_STGNN.md)
  * [`RESEARCH_ML_INFERENCE.md`](RESEARCH_ML_INFERENCE.md)
  * [`RESEARCH_MULTIMODAL_TEXT_AND_SYSTEM_IMPROVEMENT.md`](RESEARCH_MULTIMODAL_TEXT_AND_SYSTEM_IMPROVEMENT.md)

---

## 🚀 Quick Start & Verification

### 1. Installation
```bash
git clone https://github.com/TiDatm/AgenticAlert.git
cd AgenticAlert
pip install -r air_raid_forecast/requirements.txt # or pip install -e .
```

### 2. Run Test Suite
Verify anti-leakage guards, graph builders, and hierarchical model inference:
```bash
python -m unittest discover air_raid_forecast/tests
```

### 3. Generate or Refresh Interactive Conversation Viewer
To re-export or update the interactive HTML viewer:
```bash
python scripts/export_conversation_html.py
```

---

## 📂 Repository Structure

```
├── README.md                                          # Project overview and entry point
├── .gitignore                                         # Git ignore configuration
├── setup.py                                           # Package setup
├── docs/
│   └── index.html                                     # Interactive Conversation Brain Viewer (GitHub Pages)
├── scripts/
│   └── export_conversation_html.py                    # Standalone HTML exporter
├── brain/
│   ├── conversation_id.txt                            # Session identifier
│   ├── conversation_viewer.html                       # Standalone copy of HTML viewer
│   ├── logs/                                          # Full transcripts (transcript.jsonl, transcript_full.jsonl)
│   ├── artifacts/                                     # Research plans, walkthroughs, and blueprints
│   └── media/                                         # Generated maps, heatmaps, and diagnostic plots
└── air_raid_forecast/
    ├── src/
    │   ├── data/                                      # Graph builders, datasets, isochrone calculations
    │   ├── models/                                    # Hierarchical STGNN, layers, PCGrad backward pass
    │   ├── training/                                  # Rolling-origin training loop, evaluation metrics
    │   └── inference/                                 # Real-time sub-15ms inference engine
    ├── checkpoints/                                   # Trained model checkpoints (.pt)
    └── tests/                                         # Comprehensive unit tests
```
