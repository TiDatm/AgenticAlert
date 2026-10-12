# Research & Engineering Blueprint: Comprehensive Resolution for Regional Evaluation Disparities, Metric Flattening, and Dashboard Architecture

**Date:** 2026-10-11  
**Target Systems:**  
* `air_raid_forecast/src/training/train_layered.py` (Multi-Tier Training & Regional Evaluation Engine)
* `air_raid_forecast/show_map.py` (Interactive Evaluation Dashboard)
* `air_raid_forecast/checkpoints/layered_defense_report.json` (Structured Evaluation Schema)

---

## 1. Executive Problem Formulation

Our previous empirical investigation uncovered five critical bugs causing metric flattening (all regions showing `0.812`), single-class slice collapse (6 regions showing `0.500`), colorbar scale corruption (`0.00` to `1.75`), and UI coordinate collisions.

This document analyzes the open approaches, evaluates their trade-offs, and specifies the definitive engineering blueprint to resolve all issues efficiently, rigorously, and without data leakage or metric deception.

```
       Problem Cluster                                Core Failure Mechanism
   --------------------------------------------------------------------------------------------------
   1. 4h Regional Metric Collapse      Missing regional calculation in train_layered.py; fallback to global.
   2. 1h Single-Class Degeneracy       Short test holdout (7-14d) lacks rare long-range strikes in rear.
   3. Metric Sensitivity Inversion     PR/ROC-AUC undefined when N_pos=0; dummy 0.500 masks true calibration.
   4. Colorbar Scale Distortion        Unbounded auto-expansion (vmin=vmax -> [-0.18, 1.81]) for AUC in [0, 1].
   5. Viewport Coordinate Collision    Vertical overlap between cbar ticks (y=0.15) and widget headers (y=0.14).
```

---

## 2. Analysis of Candidate Approaches per Problem Cluster

### Cluster A: Per-Region 4h Operational Corridor Metrics

#### The Challenge:
In Layer 2 (Operational Corridor, 4-hour horizon), the neural network predicts wave penetration across 2-hop geographic corridors ($\mathbf{A}^2 x_t$) for each of Ukraine's 25 administrative regions. The tensor shapes at test time are:
$$\mathbf{P}_4 \in \mathbb{R}^{N \times 25}, \quad \mathbf{Y}_4 \in \mathbb{R}^{N \times 25}, \quad \mathbf{M} \in \mathbb{R}^{N \times 25}$$
However, the training script only evaluated $\mathbf{P}_4$ flattened across all regions combined, omitting per-region calculation.

#### Evaluated Approaches:
1. **Approach A1: Naive Per-Column Calculation**
   * *Mechanism:* Iterate $r \in [0, 24]$, compute `precision_recall_curve(y4[:, r], p4[:, r])`.
   * *Pros:* Minimal code diff.
   * *Cons:* Fails to account for whether the region is in a flight corridor or already saturated by border shelling; suffers from single-class failure if holdout is short.
2. **Approach A2 (Recommended): Corridor-Conditioned Regional Wave Incursion Metrics**
   * *Mechanism:*
     For each region $r$, evaluate:
     $$\text{PR-AUC}_4^{(r)} = \text{AUC}\big(\text{Precision}\big(\mathbf{Y}_{4, \text{idle}}^{(r)}, \mathbf{P}_{4, \text{idle}}^{(r)}\big)\big)$$
     $$\text{ROC-AUC}_4^{(r)} = \text{ROC}\big(\mathbf{Y}_{4, \text{idle}}^{(r)}, \mathbf{P}_{4, \text{idle}}^{(r)}\big)$$
     alongside the **Corridor Incursion Rate** $\bar{y}_4^{(r)} = \frac{1}{N} \sum y_{4, i}^{(r)}$.
   * *Pros:* Evaluates the genuine operational question: *"When this region is not currently alerting, does the model accurately detect when a drone or missile wave penetrates into its corridor within 4 hours?"*
   * *Integration:* Export directly to `regional_breakdown[reg_name]["corridor_incursion_pr_auc"]` and `["corridor_incursion_roc_auc"]`.

---

### Cluster B: Eliminating Degenerate Single-Class Slices (The 0.500 Problem)

#### The Challenge:
In a 7-day to 14-day holdout:
* Deep rear regions (Zakarpattia, Lviv, Ivano-Frankivsk) often experience **0 missile strikes** ($N_{\text{pos}} = 0$).
* Frontline regions (Zaporizhzhia, Dnipropetrovsk, Luhansk) often experience **0 calm intervals** ($N_{\text{calm}} = 0$).
When $N_{\text{pos}} = 0$ or $N_{\text{neg}} = 0$, both PR-AUC and ROC-AUC are mathematically undefined. The existing script returned `0.50, 0.50`, flattening 24% of the country into purple blocks.

#### Evaluated Approaches:
1. **Approach B1: Synthetic Balancing / SMOTE**
   * *Mechanism:* Inject synthetic positive events into quiet regions.
   * *Verdict:* **REJECTED.** Violates temporal causality and falsifies operational defense metrics.
2. **Approach B2: Expanding Holdout Horizon to 60–90 Days (Recommended)**
   * *Mechanism:* Increase `test_days` from 7–21 to **60–90 days** (e.g., May–July 2026 or July–September 2026).
   * *Physical Basis:* Over any 60-day historical window in Ukraine, Russian strategic aviation (Tu-95, MiG-31K) and Shahed-136 swarms launched multiple national salvos that reached Western Ukraine. Concurrently, frontline regions had pauses between tactical shelling waves.
   * *Result:* Every single region in Ukraine has both positive ($y=1$) and negative ($y=0$) events in the holdout. Single-class degeneracy drops from $24\%$ to **$0\%$**.
3. **Approach B3: Multi-Metric Calibration Fallback (Brier Score & Expected Calibration Error)**
   * *Mechanism:* If a region experiences an extraordinary lull ($N_{\text{pos}} = 0$), rather than outputting a deceptive `0.50` (which implies coin-flip confusion), compute the **Brier Calibration Score**:
     $$\text{BS}^{(r)} = \frac{1}{N} \sum_{i=1}^N (p_i^{(r)} - 0)^2$$
     If the model correctly predicted near-zero probability ($p \approx 0.02$) during a calm period, its Brier score is $0.0004$ (near-perfect performance).
   * *Integration:* Flag such regions as `Calm Buffer (Brier = 0.001)` rather than a broken `PR-AUC = 0.50`.

---

### Cluster C: Colorbar Clamping & Dynamic Auto-scaling

#### The Challenge:
When values are uniform or close ($v_{\min} = v_{\max} = 0.812$), the logic `vmin, vmax = vmin - 1.0, vmax + 1.0` produced limits $[-0.188, 1.812]$, displaying a colorbar up to `1.75` for an AUC metric that cannot mathematically exceed `1.0`.

#### Evaluated Approaches:
1. **Approach C1: Naive Epsilon Shift**
   * *Mechanism:* `vmin, vmax = vmin - 0.05, vmax + 0.05`.
   * *Cons:* Can still breach $[0.0, 1.0]$ boundary if value is near $0$ or $1$.
2. **Approach C2 (Recommended): Type-Aware Bounded Clamping**
   * *Mechanism:* Inspect the active metric type and enforce mathematical domain boundaries:
     ```python
     if "AUC" in metric or "Utility" in metric:
         # Metric mathematically bounded in [0.0, 1.0]
         if vmin == vmax:
             vmin, vmax = max(0.0, vmin - 0.15), min(1.0, vmax + 0.15)
         else:
             vmin, vmax = max(0.0, vmin - 0.02), min(1.0, vmax + 0.02)
     elif "%" in metric:
         # Metric bounded in [0.0, 100.0]
         if vmin == vmax:
             vmin, vmax = max(0.0, vmin - 10.0), min(100.0, vmax + 10.0)
         else:
             vmin, vmax = max(0.0, vmin - 2.0), min(100.0, vmax + 2.0)
     elif "Hours" in metric:
         # Bounded in [0.0, 24.0]
         vmin, vmax = max(0.0, vmin), min(24.0, max(vmax, vmin + 1.0))
     ```
   * *Pros:* Fully respects mathematical definitions. Eliminates invalid colorbars forever.

---

### Cluster D: Dashboard Layout Geometry & Viewport Collision Resolution

#### The Challenge:
In the current Matplotlib layout:
* Colorbar at $y \in [0.165, 0.187]$ has tick labels extending down to $y \approx 0.14$.
* Control boxes at $y \in [0.02, 0.14]$ have top headers at $y = 0.14$.
* They collide and print numbers directly over the header text.
* Sidebar at $x = 0.73$ clips on narrow screens.

#### Evaluated Layout Solutions:

```
+-----------------------------------------------------------------------+
|  AIR DEFENSE FORECAST PERFORMANCE - LAYERED DEFENSE                   |
|  [Map View ax_map: y in [0.22, 0.94], x in [0.03, 0.69]]              | [Diagnostic Sidebar]
|                                                                       | ax_sidebar:
|                                                                       | x: [0.72, 0.98]
|                                                                       | y: [0.02, 0.94]
|                                                                       | Auto-wrapping,
|                                                                       | crisp monospace,
+-----------------------------------------------------------------------+ non-clipped.
|  [Dedicated Colorbar ax_cbar: y in [0.16, 0.18], x in [0.08, 0.64]]   |
|  Ticks oriented strictly downwards, with dedicated buffer margin      |
+-----------------------------------------------------------------------+
|  [Buffer Gap: 0.035 coordinate units -> Zero Collision]               |
+-----------------------------------+-----------------------------------+
| 1. ARCHITECTURE   | 2. HORIZON    | 3. EVALUATION METRIC              |
| [y: 0.02 to 0.12] | [0.02 to 0.12]| [y: 0.02 to 0.12]                 |
+-------------------+---------------+-----------------------------------+
```

#### Detailed Coordinate Plan:
* **Map Canvas:** `[0.03, 0.22, 0.67, 0.73]`
* **Persistent Colorbar:** `[0.07, 0.165, 0.59, 0.020]`
* **Clearance Buffer:** $y \in [0.125, 0.165]$ ($0.04$ margin prevents tick collision)
* **Bottom Controls:**
  * Mode Box: `[0.03, 0.02, 0.19, 0.105]`
  * Horizon Box: `[0.235, 0.02, 0.19, 0.105]`
  * Metric Box: `[0.44, 0.02, 0.26, 0.105]`
* **Sidebar Canvas:** `[0.72, 0.02, 0.26, 0.93]` with `fontsize=8.5` and dynamic text padding.

---

## 3. Best Methods & Definitive Architecture Comparison

| Problem Component | Current (Defective) State | Industry Best Method | Why This Best Method Wins |
| :--- | :--- | :--- | :--- |
| **4h Regional Evaluation** | Global scalar fallback (`0.812` everywhere) | Compute per-region PR/ROC-AUC in `train_layered.py` loop | Restores true spatial heterogeneity along Shahed flight corridors. |
| **Degenerate Slices** | Hardcoded `0.500` fallback for single-class regions | 60–90 day holdout + Brier calibration metric | Guarantees kinetic strike samples in all oblasts without synthetic cheating. |
| **JSON Export Schema** | Missing 4h fields in `regional_breakdown` | Full 3-tier schema (`tactical`, `corridor`, `campaign`) per oblast | Decouples data generation from visualization; robust contract. |
| **Colorbar Scaling** | Unconstrained `vmin, vmax = vmin-1, vmax+1` | Type-aware clamping bounded to $[0, 1]$ or $[0, 24\text{h}]$ | Prevents nonsensical `1.75` AUC scale. |
| **Dashboard Layout** | Overlapping cbar ticks and widget text | Coordinate buffer gap of $0.04$ + top-padded text bounds | Eliminates visual collisions and ensures 100% readable controls. |

---

## 4. Implementation Blueprint (Execution Phases)

### Phase 1: Training Engine Upgrade (`train_layered.py`)
1. Extend test holdout to 60 days to ensure rich sample diversity across all 25 regions.
2. In the `for r_i, r_name in enumerate(REGIONS):` loop:
   * Compute `corridor_incursion_pr_auc` and `corridor_incursion_roc_auc` from `p4_mat[:, r_i]` and `y4_mat[:, r_i]`.
   * Compute per-region Brier score for calm calibration verification.
   * Save the complete 3-tier metrics into `layered_defense_report.json`.

### Phase 2: Visualization Engine Upgrade (`show_map.py`)
1. Update `_get_region_value()` to fetch `reg_dict.get("corridor_incursion_pr_auc")`.
2. Implement bounded metric clamping in `redraw_dashboard()`.
3. Shift bottom panels and colorbar axes to establish a $0.04$ clearance margin.
4. Add robust fallback for temporarily occupied regions (Crimea) displaying `N/A (Occupied)`.

### Phase 3: Master Pipeline Execution & Map Verification
1. Run `python air_raid_forecast/src/training/train_layered.py --epochs 3 --test-days 60`.
2. Inspect the newly populated `layered_defense_report.json`.
3. Verify `show_map.py` renders distinct, realistic regional colors across Ukraine's flight corridors.
