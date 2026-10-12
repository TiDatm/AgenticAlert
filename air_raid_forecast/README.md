# PROJECT BLUEPRINT & SYNTHESIS OF RESEARCH ARTIFACTS
## Real-Time Ukrainian Air Raid Alert Defense Forecasting Engine

This document consolidates and translates the findings from all three foundational research artifacts:
1. [`RESEARCH_AIR_RAID_DEFENSE.md`](../RESEARCH_AIR_RAID_DEFENSE.md): Problem formulation, target definitions, and rolling-origin validation.
2. [`RESEARCH_PIPELINE_CHALLENGES.md`](../RESEARCH_PIPELINE_CHALLENGES.md): Real-time ingestion resilience, training-serving parity, zero-LLM deterministic regex parsing, and class-imbalance safeguards.
3. [`RESEARCH_DEEP_LEARNING_STGNN.md`](../RESEARCH_DEEP_LEARNING_STGNN.md): Multi-resolution dual-stream tensor schemas, hybrid adaptive graph convolutions, and multi-task loss.

---

### 1. Directory Structure

The project has been organized into modular components under `air_raid_forecast/`:

```
air_raid_forecast/
├── data/
│   ├── raw/                 # Ingested datasets (e.g. official_data_uk.csv)
│   └── processed/           # Binary grid arrays, adjacency matrices, preprocessed tensors
├── checkpoints/             # Saved model weights (.pt) and exported ONNX models
├── tests/                   # Anti-leakage unit tests, offline-online parity checks
└── src/
    ├── data/                # Ingestion, graph topology, text feature extraction, dataset generator
    ├── models/              # Adaptive graph conv layers, temporal encoders, PyTorch ST-GNN
    ├── training/            # Rolling-origin training loop, mixed precision (AMP), evaluation
    └── inference/           # Sub-15ms live streaming prediction engine & state ring buffer
```

---

### 2. Synthesis of Technical Decisions

| Pipeline Component | Research Finding / Requirement | Concrete Implementation In This Project |
| :--- | :--- | :--- |
| **Data Ingestion** | Primary ground-truth: `Vadimkin/ukrainian-air-raid-sirens-dataset`. Public, clean, daily updates since March 2022. | `src/data/download_alerts.py` to fetch `official_data_uk.csv` directly into `data/raw/`. |
| **Spatial Graph** | 25 administrative regions (24 oblasts + Kyiv City) connected via land borders and aerial transit corridors. | `src/data/build_graph.py` to construct static $25 \times 25$ adjacency matrix $\mathbf{A}_{\text{geo}}$ + dynamic corridor lookups. |
| **Leakage Guardrails** | Duration leakage ($t_{\text{end}}$ of ongoing alerts must never be visible at time $t$); rolling-origin splits. | `src/data/multi_res_dataset.py` with causal windowing and strict right-boundary clamping; unit tests in `tests/test_leakage.py`. |
| **Tactical Text Ingestion** | Sub-second latency, zero API costs, resilient during strikes. No external LLMs in production. | `src/data/text_features.py`: Pre-compiled regex engine for Air Force alerts (`МіГ-31К`, `шахеди`, `балістика`). |
| **Deep Learning Backbone** | Multi-Resolution Adaptive Spatio-Temporal GNN combining fast (15m, 2h) and slow (1h, 24h) streams. | `src/models/st_gnn.py` in PyTorch with `AdaptiveGraphConv` + Causal Temporal GRU. |
| **Multi-Horizon Outputs** | Simultaneously predicts $H \in \{1\text{h}, 4\text{h}, 24\text{h}\}$ across all 25 regions. | Multi-task projection heads with Asymmetric Cost-Sensitive Focal Loss ($\gamma=2.0, \alpha=0.75$). |
| **Hardware & Acceleration** | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.1). | PyTorch AMP (`torch.amp.autocast`) for high-throughput GPU training; CPU/ONNX export for production inference. |
| **Evaluation Strategy** | Avoid naive accuracy due to class imbalance; test on latest period. | Rolling-origin temporal evaluation computing PR-AUC, ROC-AUC, Brier score, and $F_2$ utility. |
