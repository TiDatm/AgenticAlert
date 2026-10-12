# Graph Report - elegant-turing  (2026-10-11)

## Corpus Check
- 71 files · ~179,129 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 635 nodes · 1039 edges · 38 communities (37 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d287b6ff`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_layered.py
- Strategic Blueprint: Kinematic Isochrone & Survival-Hazard Defense Architecture
- Spatio-Temporal Time Series Forecasting of Air Raid Alerts in Ukraine
- load_and_clean_alerts
- DEEP LEARNING ARCHITECTURAL SPECIFICATION: MULTI-RESOLUTION SPATIO-TEMPORAL GNN
- DEEP ML & INFERENCE RESEARCH: REAL-TIME UKRAINIAN AIR RAID ALERT PREDICTION
- COMPREHENSIVE RESEARCH REPORT & EMPIRICAL EXPERIMENTAL FINDINGS
- train_rolling.py
- train_pipeline.py
- DEFINITIVE ARCHITECTURAL RESEARCH: WHAT WILL GENUINELY FIX THE MODEL
- COMPREHENSIVE RESEARCH REPORT: METRIC DISPARITIES & ROC CURVE DYNAMICS
- Investigation & Research Report: Regional Metric Flattening, Degenerate Slices, and UI Artifacts
- Analysis of Regional Performance Asymmetry on Ukrainian Air Raid Alert Map
- ProgressBar
- 2. Multi-Horizon Overfitting Diagnostics & Benchmarks
- 2. Analysis of Candidate Approaches per Problem Cluster
- AirRaidDashboard
- advanced_features.py
- Strategic Blueprint: Hierarchical Raion-Level Defense Forecasting
- Comprehensive Multi-Perspective Architecture & Optimization Audit: Ukrainian Air Defense Forecasting System
- Blueprint & Research Report: Advanced Model Improvement, Anti-Overfitting Architecture & Defense-Grade Generalization
- .forward
- evaluate_model_on_loader
- Strategic Research & Engineering Blueprint: Multimodal Text/OSINT Ingestion and Architecture Improvements
- LayeredDefenseDataset
- MultiResSTGNN
- evaluate_horizon_metrics
- MultiTaskFocalLoss
- TestLayeredSTGNN
- test_layered_model.py

## God Nodes (most connected - your core abstractions)
1. `build_adjacency_matrix()` - 27 edges
2. `AirRaidDashboard` - 21 edges
3. `load_and_clean_alerts()` - 19 edges
4. `TestHierarchicalRaionDefense` - 19 edges
5. `MultiResolutionDataset` - 18 edges
6. `train_and_evaluate_layered_system()` - 18 edges
7. `AirRaidInferenceEngine` - 17 edges
8. `HierarchicalSTGNN` - 17 edges
9. `MultiResSTGNN` - 17 edges
10. `TestLayeredSTGNN` - 17 edges

## Surprising Connections (you probably didn't know these)
- `TestHierarchicalRaionDefense` --uses--> `AirRaidDashboard`  [INFERRED]
  air_raid_forecast/tests/test_hierarchical_defense.py → air_raid_forecast/show_map.py
- `TestLayeredSTGNN` --uses--> `AirRaidDashboard`  [INFERRED]
  air_raid_forecast/tests/test_layered_model.py → air_raid_forecast/show_map.py
- `PyTorchGridDataset` --uses--> `MultiResolutionDataset`  [INFERRED]
  air_raid_forecast/src/training/train_rolling.py → air_raid_forecast/src/data/multi_res_dataset.py
- `TestAntiLeakageAndParity` --uses--> `MultiResolutionDataset`  [INFERRED]
  air_raid_forecast/tests/test_leakage.py → air_raid_forecast/src/data/multi_res_dataset.py
- `AirRaidInferenceEngine` --uses--> `HierarchicalSTGNN`  [INFERRED]
  air_raid_forecast/src/inference/predict_engine.py → air_raid_forecast/src/models/hierarchical_stgnn.py

## Import Cycles
- None detected.

## Communities (38 total, 1 thin omitted)

### Community 0 - "train_layered.py"
Cohesion: 0.06
Nodes (64): build_adjacency_matrix(), build_directed_transition_matrices(), get_normalized_laplacian(), get_static_distance_features(), ndarray, Geographic Adjacency Graph and Spatial Mappings for Ukraine's 25 Administrative…, Constructs a symmetric binary adjacency matrix A of shape (25, 25)., Calculates symmetric normalized Laplacian with self-loops: A_norm = D^(-1/2) *… (+56 more)

### Community 1 - "Strategic Blueprint: Kinematic Isochrone & Survival-Hazard Defense Architecture"
Cohesion: 0.10
Nodes (19): 1. Executive Vision: Why the Current Baseline Fails Physically, 2. The Four Pillars of the Solution Architecture, 3. Pillar 1: Population & Capital-Weighted Regional De-Saturation, 4. Pillar 2: Kinematic Isochrone Horizons (Distance-Calibrated Warning Envelopes), 5. Pillar 3: Continuous Survival / Time-to-Impact Hazard Modeling, 6. Pillar 4: Metric Calibration (Isochrone Cumulative PR-AUC), 7. Comparative Architecture Matrix, 8. Summary of Design Decisions (+11 more)

### Community 2 - "Spatio-Temporal Time Series Forecasting of Air Raid Alerts in Ukraine"
Cohesion: 0.12
Nodes (16): 1.1 Alert Sources, 1.2 Text / Media Auxiliary Ingestion, 1. Data Landscape & Ingestion Architecture, 2.1 Spatial Graph Representation, 2.2 Feature Dictionary, 2.3 Strict Leakage Prevention Rules, 2. Spatio-Temporal Feature Engineering & Leakage Prevention, 3.1 Model Hierarchy (+8 more)

### Community 3 - "load_and_clean_alerts"
Cohesion: 0.10
Nodes (26): End-to-End Orchestrator for the Ukrainian Air Raid Defense Forecasting…, run_pipeline(), download_official_alerts(), load_and_clean_alerts(), DataFrame, Module for downloading and caching the official air raid alerts dataset.…, Resolves data file path cleanly whether run from root, src, or parent…, Downloads official Ukrainian air raid dataset if not already present. Returns… (+18 more)

### Community 4 - "DEEP LEARNING ARCHITECTURAL SPECIFICATION: MULTI-RESOLUTION SPATIO-TEMPORAL GNN"
Cohesion: 0.05
Nodes (39): 1. Directory Structure, 2. Synthesis of Technical Decisions, PROJECT BLUEPRINT & SYNTHESIS OF RESEARCH ARTIFACTS, Real-Time Ukrainian Air Raid Alert Defense Forecasting Engine, RESEARCH REPORT & TECHNICAL SPECIFICATION, 1.1 Fast Stream (High-Frequency Tactical Window), 1.2 Slow Stream (Macro-Scale Operational Window), 1. Input Tensor Schema & Multi-Resolution Preprocessing (+31 more)

### Community 5 - "DEEP ML & INFERENCE RESEARCH: REAL-TIME UKRAINIAN AIR RAID ALERT PREDICTION"
Cohesion: 0.11
Nodes (18): 1.1 Datasets Directly Accessible Without Proprietary Credentials, 1. Concrete Data Landscape & Access Analysis, 2.1 Geospatial Adjacency & Wave Propagation, 2.2 Diurnal & Temporal Periodicity, 2.3 Weapon-Specific Alert Duration Dynamics, 2. Intrinsic Dependencies & Mathematical Relationships, 3. Analytical Model Comparative Research, 4. End-to-End Prediction Process Mechanics (+10 more)

### Community 6 - "COMPREHENSIVE RESEARCH REPORT & EMPIRICAL EXPERIMENTAL FINDINGS"
Cohesion: 0.11
Nodes (17): 1. Empirical Discovery: The "December 2024 Administrative Shift", 2. Experimental Verification: Testing the "New Alert Onset" Formulation, 3. Is 24 Hours Ahead Too Naive to Predict? (Theory vs. Empirical Test), 4. What Additional Data We Need & How to Ingest It, 5. Concrete Action Plan & Implementation Blueprint, COMPREHENSIVE RESEARCH REPORT & EMPIRICAL EXPERIMENTAL FINDINGS, Data Source 1: Strategic Bomber Base Movements (Air Force Telegram), Data Source 2: Weather & Celestial Visibility (Lunar Phase & Cloud Cover) (+9 more)

### Community 7 - "train_rolling.py"
Cohesion: 0.19
Nodes (10): evaluate_predictions(), find_optimal_threshold(), ndarray, Evaluation Metrics & Calibration for Air Raid Alert Forecasting. Computes PR-…, Computes comprehensive evaluation metrics across all samples., Finds probability threshold maximizing F_beta utility on validation set., Dataset, PyTorchGridDataset (+2 more)

### Community 8 - "train_pipeline.py"
Cohesion: 0.23
Nodes (7): MultiResolutionDataset, PyTorch-compatible dataset builder generating (X_fast, X_slow, Targets) without…, Dataset, PyTorchGridDataset, Production Training Pipeline with Early Stopping, Regularization, and Automated…, Executes regularized training, tracks generalization gap across epochs, applies…, train_with_overfitting_checks()

### Community 9 - "DEFINITIVE ARCHITECTURAL RESEARCH: WHAT WILL GENUINELY FIX THE MODEL"
Cohesion: 0.15
Nodes (12): 1. What Physically Causes the Problem (The Physics of Air Defense), 3. The 3 Architectural Changes That Will Truly Fix the Model, 4. Summary: The Realistic vs. Unrealistic Matrix, 5. Conclusion, DEFINITIVE ARCHITECTURAL RESEARCH: WHAT WILL GENUINELY FIX THE MODEL, Executive Thesis, Layer 1: The Tactical Early-Warning Model (Horizon: 30 min – 2 hours), Layer 2: The Operational Attack Wave Predictor (Horizon: 2 – 6 hours) (+4 more)

### Community 10 - "COMPREHENSIVE RESEARCH REPORT: METRIC DISPARITIES & ROC CURVE DYNAMICS"
Cohesion: 0.17
Nodes (11): 1. Why Regional Metrics Produce Outputs Close to 0 (The Western Cluster), 2. Why Other Regions Produce Metrics Close to 80%–97% (The Capital & Corridors), 3. Dissecting the ROC Curve: How It Operates Across Cutoffs, 4. Synthesis of Metric Behaviors, COMPREHENSIVE RESEARCH REPORT: METRIC DISPARITIES & ROC CURVE DYNAMICS, Empirical ROC Curve Coordinates (1-Hour Tactical Horizon):, Executive Summary, Key Insights from the ROC Curve: (+3 more)

### Community 11 - "Investigation & Research Report: Regional Metric Flattening, Degenerate Slices, and UI Artifacts"
Cohesion: 0.11
Nodes (18): 1. Executive Summary & Empirical Evidence, 2. Core Problem 1: Hardcoded Global Fallback in Dashboard Rendering, 3. Core Problem 2: Missing Per-Region Computation in Training Evaluation, 4. Additional Problem 3: Degenerate Single-Class Slices Defaulting to 0.5 in Layer 1, 5.1 Colorbar Scaling Bug, 5.2 Geometric Collision with Control Headers, 5.3 Sidebar Panel Aspect Ratio Clipping, 5. Additional Problem 4: Test Holdout Duration vs. Kinetic Strike Frequency (+10 more)

### Community 12 - "Analysis of Regional Performance Asymmetry on Ukrainian Air Raid Alert Map"
Cohesion: 0.18
Nodes (10): 1. Root Cause 1: District (Raion) Aggregation vs. Oblast Saturation, 2. Root Cause 2: Extreme Label Imbalance Polarizing the Accuracy Metric, 3. Root Cause 3: The Threat Vector Transit Corridor Phenomenon, 4. Root Cause 4: Permanent Territorial Occupation & Static Sirens (Luhansk & Crimea), 5. Actionable Engineering Fixes to Remove the Distortion, Analysis of Regional Performance Asymmetry on Ukrainian Air Raid Alert Map, Executive Diagnosis, ROOT CAUSE INVESTIGATION: EXTREME ACCURACY POLARIZATION IN FRONTLINE & WESTERN REGIONS (+2 more)

### Community 13 - "ProgressBar"
Cohesion: 0.22
Nodes (5): ProgressBar, Lightweight, Zero-Dependency Progress Bar & Training Status Indicator. Provides…, Zero-dependency visual progress bar with ETA, speed (it/s), and dynamic status…, Convenience wrapper function matching the tqdm API., track_progress()

### Community 14 - "2. Multi-Horizon Overfitting Diagnostics & Benchmarks"
Cohesion: 0.22
Nodes (8): 1. Concrete Overfitting Failure Modes in Defense Time Series, 2.1 Generalization Ratio, 2.2 Spatial Climatology & Persistence Baseline Lift, 2.3 Expected Calibration Error (ECE), 2. Multi-Horizon Overfitting Diagnostics & Benchmarks, Executive Overview, RESEARCH SPECIFICATION: ROBUST TEMPORAL TRAINING, EVALUATION, AND OVERFITTING DETECTION, Spatio-Temporal Time Series Modeling for Ukrainian Air Raid Alerts

### Community 15 - "2. Analysis of Candidate Approaches per Problem Cluster"
Cohesion: 0.09
Nodes (21): 1. Executive Problem Formulation, 2. Analysis of Candidate Approaches per Problem Cluster, 3. Best Methods & Definitive Architecture Comparison, 4. Implementation Blueprint (Execution Phases), Cluster A: Per-Region 4h Operational Corridor Metrics, Cluster B: Eliminating Degenerate Single-Class Slices (The 0.500 Problem), Cluster C: Colorbar Clamping & Dynamic Auto-scaling, Cluster D: Dashboard Layout Geometry & Viewport Collision Resolution (+13 more)

### Community 16 - "AirRaidDashboard"
Cohesion: 0.12
Nodes (14): AirRaidDashboard, load_data(), Interactive Evaluation Dashboard for Ukraine Air Raid Defense Forecasting.…, Cleanly wipes ax_metric and builds new RadioButtons with valid choices., Interactive click listener to inspect individual raions and oblasts., Retrieve the exact numeric value for an individual raion according to active…, Retrieve the exact numeric value according to current mode, horizon, and metric., Cleanly wipes dynamic elements and redraws without leaking axes. (+6 more)

### Community 17 - "advanced_features.py"
Cohesion: 0.33
Nodes (5): compute_advanced_feature_tensors(), DatetimeIndex, ndarray, Advanced Multi-Horizon Target and Feature Engine. Implements: 1. Dual Target…, Computes unified feature tensors: - X_kinetic: [N_steps, 25, F_kinetic] (15m…

### Community 18 - "Strategic Blueprint: Hierarchical Raion-Level Defense Forecasting"
Cohesion: 0.12
Nodes (15): 1. Deconstructing the Idea: What "Raion-Level Alert" Truly Means, 2. What It Provides: Positive Operational Outcomes, 3. What Flaws and Serious Pitfalls It Adds: The Critical Trade-Offs, 4.1 Cross-Scale Mathematical Formulation, 4. The Architectural Blueprint: Hierarchical Multi-Scale ST-GNN (H-STGNN), 5.1 Smooth Kinematic Isochrones Across 136 Raions, 5.2 Resolution of the Frontline Saturation Failure, 5. Integrating Raions into the Kinematic Isochrone & Survival Blueprint (+7 more)

### Community 27 - "Comprehensive Multi-Perspective Architecture & Optimization Audit: Ukrainian Air Defense Forecasting System"
Cohesion: 0.11
Nodes (17): 1. Executive Diagnostic Summary, 2.1 The Mathematical Floor Calculation, 2.2 Gradient Dominance & Destructive Multi-Task Interference, 2.3 Optimization Infrastructure Shortcomings, 2. Perspective 1: Optimization & Loss Mechanics (Why Loss Plateaus at ~0.40), 3.1 The 100% Autoregressive Blindness, 3.2 The Complete Disconnect with OSINT & Air Force Telegram Alerts, 3.3 Conflation of Frontline Artillery with Strategic Aerial Strikes (+9 more)

### Community 28 - "Blueprint & Research Report: Advanced Model Improvement, Anti-Overfitting Architecture & Defense-Grade Generalization"
Cohesion: 0.10
Nodes (20): 1. Executive Synthesis & Architectural Diagnosis, 2.1 Spatio-Temporal Graph Regularization (Protecting Kyiv City), 2.2 Purged & Embargoed Chronological Walk-Forward Cross-Validation, 2.3 Optimization Regularizers, 2.4 Pareto Defense Utility Early Stopping ($F_2$-Aligned), 2. Advanced Regularization & Anti-Overfitting Framework, 3.1 Dual-Diffusion Advection ($P_{\text{fwd}}$, $P_{\text{bwd}}$), 3.2 8-Hour Kinetic Causal WaveNet Temporal Block (+12 more)

### Community 30 - "evaluate_model_on_loader"
Cohesion: 0.12
Nodes (12): ConnectivityPreservingDropEdge, evaluate_model_on_loader(), ModelEMA, Any, Tensor, Maintains an Exponential Moving Average of model weights for superior out-of-…, DropEdge with degree-based protection: Drops edges with probability p only…, Randomly zeros out features for a fraction of nodes during training… (+4 more)

### Community 31 - "Strategic Research & Engineering Blueprint: Multimodal Text/OSINT Ingestion and Architecture Improvements"
Cohesion: 0.14
Nodes (13): 1. Tactical Threat Disaggregation: Four Divergent Flight Regimes, 2.1 Authorized Public Feeds Under Strict Historical Anti-Leakage, 2. Ingesting Public Text & Media Data: Sources & Schema, 3.1 Deterministic Entity & Threat Extraction Specification, 3.2 Spatio-Temporal Text Feature Embedding, 3. NLP Feature Extraction Pipeline (Sub-2ms Latency, Zero LLM API Cost), 4.1 Air Defense Interception Decay Layer (Fixing the Western PR-AUC Deficit), 4.2 Metric Normalization in Dashboard (Information Gain Lift) (+5 more)

### Community 32 - "LayeredDefenseDataset"
Cohesion: 0.22
Nodes (5): LayeredDefenseDataset, ndarray, Produces strictly causal multi-tier tensors: - X_kinetic: [Batch, 25, 32, 16]…, Dataset, TorchLayeredDataset

### Community 33 - "MultiResSTGNN"
Cohesion: 0.15
Nodes (10): AdaptiveGraphConv, Graph Convolution combining static geographic Laplacian and learnable adaptive…, Args: x: Node feature tensor [Batch, Num_Nodes, In_Features] static_adj_norm:…, MultiResSTGNN, Tensor, PyTorch Multi-Resolution Adaptive Spatio-Temporal Graph Neural Network…, Forward pass producing raw prediction logits for all 3 horizons. Returns:…, Verify that features at forecast step t do not change if we modify future… (+2 more)

### Community 34 - "evaluate_horizon_metrics"
Cohesion: 0.16
Nodes (13): compute_expected_calibration_error(), compute_persistence_baseline(), diagnose_overfitting(), evaluate_horizon_metrics(), Any, ndarray, Comprehensive Evaluation, Benchmarking & Overfitting Diagnostic Engine.…, Synthesizes training vs. validation metrics into a formal Overfitting… (+5 more)

### Community 35 - "MultiTaskFocalLoss"
Cohesion: 0.29
Nodes (4): MultiTaskFocalLoss, Tensor, Multi-Task Cost-Sensitive Focal Loss for Air Raid Alert Forecasting. Heavily…, Computes Asymmetric Focal Loss across 3 horizons: 1h, 4h, 24h. L = w1 * L_1h +…

### Community 36 - "TestLayeredSTGNN"
Cohesion: 0.07
Nodes (23): MicroRaionDirectedGCN, Micro-scale graph convolutional layer for 136 raions with: - Directed forward…, LayeredDefenseLoss, LayeredSTGNN, Tensor, Layered Spatio-Temporal Graph Neural Network (Layered ST-GNN v2). Decouples…, Forward pass. Returns: - logit_tactical_1h: [B, 25] - logit_operational_4h: [B,…, Combines: 1. Masked Focal Loss on Tactical Onset (computed ONLY on currently… (+15 more)

### Community 37 - "test_layered_model.py"
Cohesion: 0.08
Nodes (27): build_tactical_text_grid(), load_cached_telegram_messages(), Any, DatetimeIndex, ndarray, Air Force & OSINT Tactical Telegram Indicator Ingestion & Discretization.…, Builds synchronized [N_steps, 25, 6] tactical OSINT signal grid. Applies strict…, Loads raw Telegram messages from JSONL file if available. Each line format:… (+19 more)

## Knowledge Gaps
- **166 isolated node(s):** `The Real-World Operational Context in Ukraine:`, `2. What It Provides: Positive Operational Outcomes`, `Pitfall A: The "Oblast-Level Inheritance" Data Asymmetry`, `Pitfall B: Extreme Rare-Event Class Imbalance ($136\times$ Sparsity)`, `Pitfall C: Graph Complexity & Message-Passing Dilution` (+161 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AirRaidDashboard` connect `AirRaidDashboard` to `train_layered.py`, `TestLayeredSTGNN`, `test_layered_model.py`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `build_adjacency_matrix()` connect `train_layered.py` to `load_and_clean_alerts`, `train_rolling.py`, `train_pipeline.py`, `AirRaidDashboard`, `advanced_features.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `MultiResSTGNN` connect `MultiResSTGNN` to `train_layered.py`, `TestLayeredSTGNN`, `test_layered_model.py`, `train_rolling.py`, `train_pipeline.py`, `AirRaidDashboard`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `AirRaidDashboard` (e.g. with `TestHierarchicalRaionDefense` and `TestLayeredSTGNN`) actually correct?**
  _`AirRaidDashboard` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `TestHierarchicalRaionDefense` (e.g. with `AirRaidDashboard` and `AirRaidInferenceEngine`) actually correct?**
  _`TestHierarchicalRaionDefense` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MultiResolutionDataset` (e.g. with `PyTorchGridDataset` and `PyTorchGridDataset`) actually correct?**
  _`MultiResolutionDataset` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `The Real-World Operational Context in Ukraine:`, `2. What It Provides: Positive Operational Outcomes`, `Pitfall A: The "Oblast-Level Inheritance" Data Asymmetry` to the rest of the system?**
  _166 weakly-connected nodes found - possible documentation gaps or missing edges._