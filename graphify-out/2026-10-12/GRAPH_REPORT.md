# Graph Report - elegant-turing  (2026-10-12)

## Corpus Check
- 78 files · ~312,431 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 730 nodes · 1176 edges · 46 communities (44 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d287b6ff`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_layered.py
- Strategic Blueprint: Kinematic Isochrone & Survival-Hazard Defense Architecture
- TECHNICAL CHALLENGES, PIPELINE RISKS & PRODUCTION COST-EFFICIENCY RESEARCH
- load_and_clean_alerts
- Spatio-Temporal Time Series Forecasting of Air Raid Alerts in Ukraine
- DEEP ML & INFERENCE RESEARCH: REAL-TIME UKRAINIAN AIR RAID ALERT PREDICTION
- COMPREHENSIVE RESEARCH REPORT & EMPIRICAL EXPERIMENTAL FINDINGS
- train_pipeline.py
- Mathematical & Empirical ML Research: Why Loss Floors at ~0.24 and How to Break the Learning Limit
- DEFINITIVE ARCHITECTURAL RESEARCH: WHAT WILL GENUINELY FIX THE MODEL
- COMPREHENSIVE RESEARCH REPORT: METRIC DISPARITIES & ROC CURVE DYNAMICS
- Investigation & Research Report: Regional Metric Flattening, Degenerate Slices, and UI Artifacts
- Analysis of Regional Performance Asymmetry on Ukrainian Air Raid Alert Map
- ProgressBar
- 2. Multi-Horizon Overfitting Diagnostics & Benchmarks
- 2. Analysis of Candidate Approaches per Problem Cluster
- AirRaidDashboard
- 1. The Mathematical Mechanism of the Failure
- Comprehensive Multi-Perspective Architecture & Optimization Audit: Ukrainian Air Defense Forecasting System
- Blueprint & Research Report: Advanced Model Improvement, Anti-Overfitting Architecture & Defense-Grade Generalization
- Root Cause Analysis: Air Raid Defense Model Performance Collapse
- evaluate_model_on_loader
- Strategic Research & Engineering Blueprint: Multimodal Text/OSINT Ingestion and Architecture Improvements
- LayeredDefenseDataset
- TestLayeredSTGNN
- evaluate_horizon_metrics
- Blueprint: Next-Generation Kinematic Trajectory & Survival Defense Architecture
- test_layered_model.py
- compute_advanced_feature_tensors
- build_clean_raions_geojson.py
- .__init__
- .forward
- layered_stgnn.py
- HierarchicalSTGNN
- .__init__
- AirDefenseInterceptionDecay

## God Nodes (most connected - your core abstractions)
1. `build_adjacency_matrix()` - 29 edges
2. `AirRaidDashboard` - 25 edges
3. `HierarchicalSTGNN` - 21 edges
4. `TestLayeredSTGNN` - 21 edges
5. `train_and_evaluate_layered_system()` - 20 edges
6. `TestHierarchicalRaionDefense` - 20 edges
7. `load_and_clean_alerts()` - 19 edges
8. `MultiResolutionDataset` - 18 edges
9. `AirRaidInferenceEngine` - 17 edges
10. `MultiResSTGNN` - 17 edges

## Surprising Connections (you probably didn't know these)
- `TestHierarchicalRaionDefense` --uses--> `AirRaidDashboard`  [INFERRED]
  air_raid_forecast/tests/test_hierarchical_defense.py → air_raid_forecast/show_map.py
- `TestLayeredSTGNN` --uses--> `AirRaidDashboard`  [INFERRED]
  air_raid_forecast/tests/test_layered_model.py → air_raid_forecast/show_map.py
- `TestLayeredSTGNN` --uses--> `LayeredDefenseDataset`  [INFERRED]
  air_raid_forecast/tests/test_layered_model.py → air_raid_forecast/src/data/layered_dataset.py
- `AirRaidInferenceEngine` --uses--> `HierarchicalSTGNN`  [INFERRED]
  air_raid_forecast/src/inference/predict_engine.py → air_raid_forecast/src/models/hierarchical_stgnn.py
- `AirRaidInferenceEngine` --uses--> `LayeredSTGNN`  [INFERRED]
  air_raid_forecast/src/inference/predict_engine.py → air_raid_forecast/src/models/layered_stgnn.py

## Import Cycles
- None detected.

## Communities (46 total, 2 thin omitted)

### Community 0 - "train_layered.py"
Cohesion: 0.07
Nodes (55): Advanced Multi-Horizon Target and Feature Engine. Implements: 1. Dual Target…, build_adjacency_matrix(), build_directed_transition_matrices(), compute_regional_threat_distances(), get_normalized_laplacian(), get_static_distance_features(), ndarray, Geographic Adjacency Graph and Spatial Mappings for Ukraine's 25 Administrative… (+47 more)

### Community 1 - "Strategic Blueprint: Kinematic Isochrone & Survival-Hazard Defense Architecture"
Cohesion: 0.10
Nodes (19): 1. Executive Vision: Why the Current Baseline Fails Physically, 2. The Four Pillars of the Solution Architecture, 3. Pillar 1: Population & Capital-Weighted Regional De-Saturation, 4. Pillar 2: Kinematic Isochrone Horizons (Distance-Calibrated Warning Envelopes), 5. Pillar 3: Continuous Survival / Time-to-Impact Hazard Modeling, 6. Pillar 4: Metric Calibration (Isochrone Cumulative PR-AUC), 7. Comparative Architecture Matrix, 8. Summary of Design Decisions (+11 more)

### Community 2 - "TECHNICAL CHALLENGES, PIPELINE RISKS & PRODUCTION COST-EFFICIENCY RESEARCH"
Cohesion: 0.11
Nodes (18): 1.1 Upstream API Downtime, Rate Limits & Blackouts, 1.2 Clock Drift, Event Out-of-Order Delivery & Timestamp Misalignment, 1.3 Label Asymmetry & Concept Drift Between War Phases, 1. Data Pipeline Failures & Operational Challenges, 2.1 The Pitfalls of Using LLMs in Production, 2.2 The Deterministic Alternative: Compiled Regex & Keyword Ontology, 2.3 Challenges in Deterministic Text Parsing, 2. Text/Media Pipeline Challenges (Zero-LLM Architecture) (+10 more)

### Community 3 - "load_and_clean_alerts"
Cohesion: 0.09
Nodes (27): End-to-End Orchestrator for the Ukrainian Air Raid Defense Forecasting…, run_pipeline(), download_official_alerts(), load_and_clean_alerts(), DataFrame, Module for downloading and caching the official air raid alerts dataset.…, Resolves data file path cleanly whether run from root, src, or parent…, Downloads official Ukrainian air raid dataset if not already present. Returns… (+19 more)

### Community 4 - "Spatio-Temporal Time Series Forecasting of Air Raid Alerts in Ukraine"
Cohesion: 0.05
Nodes (37): 1. Directory Structure, 2. Synthesis of Technical Decisions, PROJECT BLUEPRINT & SYNTHESIS OF RESEARCH ARTIFACTS, Real-Time Ukrainian Air Raid Alert Defense Forecasting Engine, 1.1 Alert Sources, 1.2 Text / Media Auxiliary Ingestion, 1. Data Landscape & Ingestion Architecture, 2.1 Spatial Graph Representation (+29 more)

### Community 5 - "DEEP ML & INFERENCE RESEARCH: REAL-TIME UKRAINIAN AIR RAID ALERT PREDICTION"
Cohesion: 0.11
Nodes (18): 1.1 Datasets Directly Accessible Without Proprietary Credentials, 1. Concrete Data Landscape & Access Analysis, 2.1 Geospatial Adjacency & Wave Propagation, 2.2 Diurnal & Temporal Periodicity, 2.3 Weapon-Specific Alert Duration Dynamics, 2. Intrinsic Dependencies & Mathematical Relationships, 3. Analytical Model Comparative Research, 4. End-to-End Prediction Process Mechanics (+10 more)

### Community 6 - "COMPREHENSIVE RESEARCH REPORT & EMPIRICAL EXPERIMENTAL FINDINGS"
Cohesion: 0.11
Nodes (17): 1. Empirical Discovery: The "December 2024 Administrative Shift", 2. Experimental Verification: Testing the "New Alert Onset" Formulation, 3. Is 24 Hours Ahead Too Naive to Predict? (Theory vs. Empirical Test), 4. What Additional Data We Need & How to Ingest It, 5. Concrete Action Plan & Implementation Blueprint, COMPREHENSIVE RESEARCH REPORT & EMPIRICAL EXPERIMENTAL FINDINGS, Data Source 1: Strategic Bomber Base Movements (Air Force Telegram), Data Source 2: Weather & Celestial Visibility (Lunar Phase & Cloud Cover) (+9 more)

### Community 7 - "train_pipeline.py"
Cohesion: 0.06
Nodes (35): MultiResolutionDataset, PyTorch-compatible dataset builder generating (X_fast, X_slow, Targets) without…, AdaptiveGraphConv, PyTorch Layers for Adaptive Spatio-Temporal Graph Neural Network: -…, Graph Convolution combining static geographic Laplacian and learnable adaptive…, Causal unidirectional temporal encoder operating on node sequences. Ensures…, TemporalCausalGRU, MultiTaskFocalLoss (+27 more)

### Community 8 - "Mathematical & Empirical ML Research: Why Loss Floors at ~0.24 and How to Break the Learning Limit"
Cohesion: 0.11
Nodes (18): 1. Mathematical Anatomy of `HierarchicalDefenseLoss`, 2. Theoretical Bayes Optimal Floor & Aleatoric Entropy, 3. Multi-Task Gradient Sabotage & Conflict, 4. Optimization Dynamics & Hyperparameter Friction, 5. Architectural & Algorithmic Blueprint to Break the Plateau, 6. Summary Comparison Table, A. Focal Loss Lower Bound Formulation, Actionable Roadmap: (+10 more)

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
Cohesion: 0.09
Nodes (18): AirRaidDashboard, load_data(), Interactive Evaluation Dashboard for Ukraine Air Raid Defense Forecasting.…, Keyboard navigation: 'h' to cycle child hromadas, 'escape' to clear hromada…, Fast <0.1ms spatial click handler using R-tree spatial indexing., Retrieve numeric value for an individual leaf node (hromada or childless raion)., Retrieve the exact numeric value for an individual raion according to active…, Retrieve numeric value according to current mode, horizon, and metric. (+10 more)

### Community 17 - "1. The Mathematical Mechanism of the Failure"
Cohesion: 0.13
Nodes (14): 1. The Mathematical Mechanism of the Failure, 2. Empirical Verification & Reproduction, 3. The Definitive Solution: Normalized Convex Combination, 4. Implementation Details, A. The Kendall & Gal Objective, B. The Gradient with Respect to $s_k$, C. Why This Breaks Down When $\mathcal{L}_k < 1.0$, Consequence 1: Crossing Zero ($0.0000$) (+6 more)

### Community 27 - "Comprehensive Multi-Perspective Architecture & Optimization Audit: Ukrainian Air Defense Forecasting System"
Cohesion: 0.11
Nodes (17): 1. Executive Diagnostic Summary, 2.1 The Mathematical Floor Calculation, 2.2 Gradient Dominance & Destructive Multi-Task Interference, 2.3 Optimization Infrastructure Shortcomings, 2. Perspective 1: Optimization & Loss Mechanics (Why Loss Plateaus at ~0.40), 3.1 The 100% Autoregressive Blindness, 3.2 The Complete Disconnect with OSINT & Air Force Telegram Alerts, 3.3 Conflation of Frontline Artillery with Strategic Aerial Strikes (+9 more)

### Community 28 - "Blueprint & Research Report: Advanced Model Improvement, Anti-Overfitting Architecture & Defense-Grade Generalization"
Cohesion: 0.10
Nodes (20): 1. Executive Synthesis & Architectural Diagnosis, 2.1 Spatio-Temporal Graph Regularization (Protecting Kyiv City), 2.2 Purged & Embargoed Chronological Walk-Forward Cross-Validation, 2.3 Optimization Regularizers, 2.4 Pareto Defense Utility Early Stopping ($F_2$-Aligned), 2. Advanced Regularization & Anti-Overfitting Framework, 3.1 Dual-Diffusion Advection ($P_{\text{fwd}}$, $P_{\text{bwd}}$), 3.2 8-Hour Kinetic Causal WaveNet Temporal Block (+12 more)

### Community 29 - "Root Cause Analysis: Air Raid Defense Model Performance Collapse"
Cohesion: 0.11
Nodes (18): 1. Kinematic Breakdown: Threat Physics vs. Arbitrary Temporal Horizons, 2. Mathematical & Metric Pathology: The PR-AUC Illusion, 3. Deep Learning & GNN Architecture Failure: Diffusion vs. Trajectory, 4. Data & Feature Pipeline Starvation: The Auto-Regressive Trap, 5. Synthesis: Summary Matrix of Failure Points, 6. The Verdict: Does the System Need a Scratch Rebuild?, A. Graph Convolution Models Heat Diffusion, Not Aerodynamic Flight, A. Siren History is Self-Referential (+10 more)

### Community 30 - "evaluate_model_on_loader"
Cohesion: 0.12
Nodes (12): ConnectivityPreservingDropEdge, evaluate_model_on_loader(), ModelEMA, Any, Tensor, Maintains an Exponential Moving Average of model weights for superior out-of-…, DropEdge with degree-based protection: Drops edges with probability p only…, Randomly zeros out features for a fraction of nodes during training… (+4 more)

### Community 31 - "Strategic Research & Engineering Blueprint: Multimodal Text/OSINT Ingestion and Architecture Improvements"
Cohesion: 0.14
Nodes (13): 1. Tactical Threat Disaggregation: Four Divergent Flight Regimes, 2.1 Authorized Public Feeds Under Strict Historical Anti-Leakage, 2. Ingesting Public Text & Media Data: Sources & Schema, 3.1 Deterministic Entity & Threat Extraction Specification, 3.2 Spatio-Temporal Text Feature Embedding, 3. NLP Feature Extraction Pipeline (Sub-2ms Latency, Zero LLM API Cost), 4.1 Air Defense Interception Decay Layer (Fixing the Western PR-AUC Deficit), 4.2 Metric Normalization in Dashboard (Information Gain Lift) (+5 more)

### Community 32 - "LayeredDefenseDataset"
Cohesion: 0.21
Nodes (5): LayeredDefenseDataset, ndarray, Produces strictly causal multi-tier tensors: - X_kinetic: [Batch, 25, 32, 16]…, Dataset, TorchLayeredDataset

### Community 33 - "TestLayeredSTGNN"
Cohesion: 0.17
Nodes (6): AirRaidInferenceEngine, Updates instantaneous alert states and advances circular ring buffer for: 1. 25…, Sub-15ms real-time prediction engine for all 25 Ukrainian regions and 136…, Test production inference engine outputs all 136 raion forecasts and respects…, Test inference engine calculates separated leaf forecasts for all 204 nodes., TestLayeredSTGNN

### Community 34 - "evaluate_horizon_metrics"
Cohesion: 0.16
Nodes (13): compute_expected_calibration_error(), compute_persistence_baseline(), diagnose_overfitting(), evaluate_horizon_metrics(), Any, ndarray, Comprehensive Evaluation, Benchmarking & Overfitting Diagnostic Engine.…, Synthesizes training vs. validation metrics into a formal Overfitting… (+5 more)

### Community 36 - "Blueprint: Next-Generation Kinematic Trajectory & Survival Defense Architecture"
Cohesion: 0.12
Nodes (16): 1. Executive Summary: The Paradigmatic Shift, 2.1 Why Replace GCN with a Trajectory Transformer?, 2.2 Mathematical Formulation of the Kinematic Bearing Bias $\mathcal{B}_{\text{kinematic}}$, 2. Subsystem 1: Kinematic Trajectory Transformer (KTT), 3.1 Physics-Grounded Tiered Operational Horizons, 3.2 Continuous Survival / Hazard Function Head, 3. Subsystem 2: Restructuring Temporal Horizons & Continuous Survival Modeling, 4.1 New Feature Tensor Specification (+8 more)

### Community 37 - "test_layered_model.py"
Cohesion: 0.11
Nodes (24): build_tactical_text_grid(), load_cached_telegram_messages(), Any, DatetimeIndex, ndarray, Air Force & OSINT Tactical Telegram Indicator Ingestion & Discretization.…, Builds synchronized [N_steps, 25, 6] tactical OSINT signal grid. Applies strict…, Resolves raw Telegram jsonl path robustly across workspace configurations. (+16 more)

### Community 38 - "compute_advanced_feature_tensors"
Cohesion: 0.50
Nodes (4): compute_advanced_feature_tensors(), DatetimeIndex, ndarray, Computes unified feature tensors: - X_kinetic: [N_steps, 25, F_kinetic] (15m…

### Community 40 - ".__init__"
Cohesion: 0.33
Nodes (4): HierarchicalDefenseLoss, MicroRaionDirectedGCN, Unified Hierarchical Multi-Scale Loss with Prevalence-Balanced Reductions: 1.…, Micro-scale graph convolutional layer for 136 raions with: - Directed forward…

### Community 43 - ".forward"
Cohesion: 0.33
Nodes (3): Tensor, Synthesizes calibrated raion predictions from discrete leaf predictions: -…, Forward pass. Returns: - logit_1h: [B, 25] - logit_4h: [B, 25] - hours_24h: [B,…

### Community 44 - "layered_stgnn.py"
Cohesion: 0.16
Nodes (10): LayeredDefenseLoss, LayeredSTGNN, Tensor, Layered Spatio-Temporal Graph Neural Network (Layered ST-GNN v2). Decouples…, Forward pass. Returns: - logit_tactical_1h: [B, 25] - logit_operational_4h: [B,…, Combines: 1. Masked Focal Loss on Tactical Onset (computed ONLY on currently…, DilatedCausalWaveNet, DirectedAdvectiveGCN (+2 more)

### Community 46 - "HierarchicalSTGNN"
Cohesion: 0.13
Nodes (9): HierarchicalSTGNN, Test physical de-saturation: Nikopol alone does NOT trigger Dnipropetrovsk…, Test HierarchicalSTGNN forward pass and gradient flow through all heads., Test backward compatibility when x_raion is None., Test 3-level Downward Hierarchical Propagation: Alert_leaf = Alert_local_leaf |…, Test raion defense report contains separated metrics for all 136 raions and 204…, Invariant: Exactly 136 raions, orthogonal membership S sum across rows == 1., Invariant: Border raions have fast horizons (<= 1h), deep-rear western raions… (+1 more)

### Community 49 - ".__init__"
Cohesion: 0.08
Nodes (13): CausalConv1d, ContinuousWeibullSurvivalHead, KinematicBearingAttention, KinematicTrajectoryTransformer, Tensor, Args: x: Node features [B, N, F] p_fwd: Forward transition matrix [N, N] or [B,…, Kinematic Bearing-Modulated Full-Graph Cross-Attention. Replaces 2-hop GCN with…, Multi-layer Kinematic Trajectory Transformer Block. Combines… (+5 more)

## Knowledge Gaps
- **201 isolated node(s):** `1. Executive Vision: Why the Current Baseline Fails Physically`, `2. The Four Pillars of the Solution Architecture`, `Why We Chose This:`, `The Mathematical Specification:`, `Why This Fixes Dnipropetrovsk & Donbas:` (+196 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AirRaidDashboard` connect `AirRaidDashboard` to `train_layered.py`, `TestLayeredSTGNN`, `test_layered_model.py`, `HierarchicalSTGNN`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `build_adjacency_matrix()` connect `train_layered.py` to `LayeredDefenseDataset`, `load_and_clean_alerts`, `test_layered_model.py`, `train_pipeline.py`, `AirRaidDashboard`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `HierarchicalSTGNN` connect `HierarchicalSTGNN` to `train_layered.py`, `TestLayeredSTGNN`, `train_pipeline.py`, `.__init__`, `.forward`, `layered_stgnn.py`, `.__init__`, `AirDefenseInterceptionDecay`, `evaluate_model_on_loader`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `AirRaidDashboard` (e.g. with `TestHierarchicalRaionDefense` and `TestLayeredSTGNN`) actually correct?**
  _`AirRaidDashboard` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HierarchicalSTGNN` (e.g. with `AirRaidInferenceEngine` and `AirDefenseInterceptionDecay`) actually correct?**
  _`HierarchicalSTGNN` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `TestLayeredSTGNN` (e.g. with `AirRaidDashboard` and `LayeredDefenseDataset`) actually correct?**
  _`TestLayeredSTGNN` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `1. Executive Vision: Why the Current Baseline Fails Physically`, `2. The Four Pillars of the Solution Architecture`, `Why We Chose This:` to the rest of the system?**
  _201 weakly-connected nodes found - possible documentation gaps or missing edges._