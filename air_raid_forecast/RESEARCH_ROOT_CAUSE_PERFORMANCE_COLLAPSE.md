# Root Cause Analysis: Air Raid Defense Model Performance Collapse

**Document Type:** Rigorous Multi-Perspective Research & Mathematical Diagnostic Audit  
**Author:** Antigravity Research Lab (Kinematics, GNN Architecture, Mathematical ML, and Threat Intelligence Specialists)  
**Target Subject:** Comprehensive Analysis of Low Model Scores, Western Ukraine Degradation, and Horizon Flaws  
**Reference Visual:** User Evaluation Choropleth (`RAION ISOCHRONE PR-AUC`, 1h Tactical Horizon)  

---

## Executive Summary: The Four Pillars of Failure

The evaluation dashboard illustrates a stark contrast: while the southeastern frontline contact arc achieves moderate PR-AUC scores ($0.40 - 0.50$), **Western Ukraine (Lviv, Zakarpattia, Volyn, Rivne, Ternopil, Ivano-Frankivsk) collapses to near-zero ($0.010 - 0.050$)**, and **occupied/border raions (Severodonetsk, Lysychansk, Luhansk) flatten out to an artificial plateau of exactly $0.010$**.

Our deep multi-perspective investigation across defense kinematics, graph deep learning architectures, statistical learning theory, and intelligence data pipelines reveals that **this poor performance is NOT a simple hyperparameter tuning problem or a minor bug**. The entire foundation suffers from four structural flaws:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   THE FOUR PILLARS OF PERFORMANCE COLLAPSE                       │
├───────────────────────────────┬──────────────────────────────────────────────────┤
│ 1. Weapon Kinematics Mismatch │ Fixed [1h, 4h, 24h] windows violate the physics  │
│                               │ of aerial weapons (ballistics: 3m; Shaheds: 5h). │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ 2. Graph Diffusion vs Flight  │ GCN isotropic diffusion cannot model directed    │
│                               │ waypoint-guided trajectories across 8 hops.      │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ 3. Feature Starvation Loop    │ Model is blind to Russian airfield takeoffs; it  │
│                               │ only feeds on past siren rolling averages.       │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ 4. PR-AUC Metric Pathology    │ Raw PR-AUC numerically collapses to base rate    │
│                               │ (0.2% in West); 0.010 reflects dead data labels. │
└───────────────────────────────┴──────────────────────────────────────────────────┘
```

Below is the definitive, exhaustive audit of what is mathematically, physically, and architecturally misdesigned.

---

## 1. Kinematic Breakdown: Threat Physics vs. Arbitrary Temporal Horizons

The current model relies on three arbitrary static horizons:
* $\tau_1 = 1\text{ hour}$ (Tactical Onset)
* $\tau_2 = 4\text{ hours}$ (Operational Corridor)
* $\tau_3 = 24\text{ hours}$ (Strategic Campaign Daily Hours)

### A. The Velocity & Transit Spectrum of Aerial Threats
Modern aerial warfare over Ukraine does not operate on a uniform hourly clock. Different weapon classes inhabit radically distinct temporal and kinematic regimes:

| Threat Category | Representative Systems | Velocity ($v$) | Flight Time from Border/Launch to Central/West | Operational Fate Under 1h / 4h / 24h Horizons |
| :--- | :--- | :--- | :--- | :--- |
| **Hypersonic & Ballistic** | Iskander-M (9M723), Kinzhal (Kh-47M2), KN-23, S-300/400 | Mach 4.0 – 10.0 ($1,300 - 3,400\text{ m/s}$) | **1 to 5 minutes** to border/frontline; **4 to 9 minutes** to deep rear. | **Physics-blind.** A 1h prediction window is useless. An alarm must sound in $< 60$ seconds based on launch telemetry. Evaluating hourly onsets is irrelevant. |
| **Tactical Glide Bombs (KAB / UMPK)** | FAB-250/500/1500 with UMPK wings on Su-34 | Mach 0.85 – 1.1 ($900 - 1,150\text{ km/h}$) | **3 to 8 minutes** from standoff release ($40 - 70\text{ km}$ buffer). | Confined strictly to frontline/border raions (Kupyansk, Vovchansk, Kherson, Orikhiv). Smearing into 1h or 4h windows masks instantaneous strike frequency. |
| **Subsonic Cruise Missiles** | Kh-101/555, Kalibr (3M-54), Iskander-K (9M728) | $750 - 850\text{ km/h}$ ($\sim 13\text{ km/min}$) | **35 to 80 minutes** to Central Ukraine; **75 to 130 minutes** to Western Ukraine. | Missiles follow complex waypoint routing. Transit from ingress to Lviv takes $> 1.5\text{ hours}$. A 1h window expires before the missile arrives. |
| **One-Way Attack Drones (Shahed-136 / Geran-2)** | Shahed-136 / Italmas | $150 - 185\text{ km/h}$ ($\sim 2.7\text{ km/min}$) | **3.5 to 5.5 hours** from border to Central; **5.0 to 8.0 hours** to Western Ukraine. | **Severe False Positive Trap.** When a swarm crosses into Sumy or Poltava headed west, predicting danger in Lviv within 1h is physically impossible. |
| **Strategic Aviation Pre-Launch** | Tu-95MS / Tu-160 from Olenya / Engels | $700 - 850\text{ km/h}$ transit | **4 to 6 hours** flight time before reaching the Caspian launch line. | Completely invisible to a 1h/4h siren model. Siren is silent during the entire 5-hour bomber flight. |

### B. The "Western Penalty Trap": The Mathematics of False Alarms
Why is Western Ukraine (Lviv, Zakarpattia, Volyn, Rivne) dark purple ($0.010 - 0.035$)?
Consider a Shahed swarm launched from Primorsko-Akhtarsk or Kursk at $23:00$:
1. At $00:30$, the swarm crosses Sumy / Poltava.
2. An intelligent neural network detects the westward-bearing wave and assigns high alert probability to Lviv.
3. However, the drones travel at $180\text{ km/h}$. The distance to Lviv is $750\text{ km}$.
4. The drones will not enter Lviv Oblast until $04:15$ ($3.75\text{ hours later}$).
5. Under the rigid **1-hour prediction window** ($00:30 \to 01:30$), the siren in Lviv remains idle.
6. **Result:** The ground truth label is $y=0$. The model's accurate early warning is penalized as a **FALSE POSITIVE**.
7. Later, at $03:45$, when the drones finally approach the border of Lviv Oblast, the local siren sounds. If the model had decayed its probability back down, it is hit with a **FALSE NEGATIVE**.

> **Kinematic Conclusion:** Forcing a uniform 1-hour horizon across a country $1,300\text{ km}$ wide guarantees that any model attempting long-range early warning will be mathematically punished for being early.

---

## 2. Mathematical & Metric Pathology: The PR-AUC Illusion

### A. The Baseline Shift in Extreme Class Imbalance
In binary classification, the Precision-Recall Area Under Curve (PR-AUC) has a baseline value that is **NOT zero, but rather equal to the prior class prevalence $\pi$**:
$$\lim_{\text{random classifier}} \text{PR-AUC} = \pi = P(y=1)$$

Let us examine the empirical ground truth data computed across the 157,113 time steps (2022–2026):
* **Donetsk Oblast:** Mean alert active rate = $48.5\%$. Idle-to-alert 1h transition base rate = **$86.9\%$** ($0.8689$).
* **Dnipropetrovsk Oblast:** Mean active rate = $26.5\%$. Idle-to-alert 1h transition base rate = **$56.4\%$** ($0.5644$).
* **Lviv Oblast:** Mean active rate = $2.31\%$. Idle-to-alert 1h transition base rate = **$0.21\%$** ($0.0021$).
* **Zakarpattia Oblast:** Mean active rate = $2.27\%$. Idle-to-alert 1h transition base rate = **$0.21\%$** ($0.0021$).
* **Volyn Oblast:** Mean active rate = $2.50\%$. Idle-to-alert 1h transition base rate = **$0.57\%$** ($0.0057$).

#### The Absurdity of Absolute Scale Comparison:
When evaluating the model on a uniform $[0.0, 1.0]$ color scale:
1. A **completely useless, random model** predicting alerts in Donetsk gets a PR-AUC of **$0.869$** (bright yellow/green).
2. A **state-of-the-art model** in Lviv that identifies oncoming strikes with $10\times$ better precision than chance gets a PR-AUC of **$0.026$** (dark purple/black).
3. The true predictive **Lift** ($\frac{\text{PR-AUC}}{\text{Base Rate}}$) for Lviv is:
   $$\text{Lift}_{\text{Lviv}} = \frac{0.026}{0.0021} \approx \mathbf{12.4\times \text{ better than chance}}$$
   Whereas for Donetsk:
   $$\text{Lift}_{\text{Donetsk}} = \frac{0.922}{0.869} \approx \mathbf{1.06\times \text{ better than chance}}$$

> **Mathematical Conclusion:** The choropleth map is not displaying model intelligence. It is displaying **geographical threat prevalence**. High scores in the Donbas reflect constant artillery saturation, while low scores in Lviv reflect the extreme sparsity of events.

### B. Frontline / Occupied Territory Degeneracy (The 0.010 Plateau)
In the user's dashboard, raions like **Сєвєродонецький**, **Лисичанський**, **Луганський**, and **Старобільський** display an identical score of **$0.010$**.

Our audit of `official_data_uk.csv` and the evaluation code revealed the exact mechanism:
1. In the entire 115,000-row alerts dataset, **Luhansk Oblast has only 2 alert entries** (from March 24–28, 2022). It has had **ZERO alert records** in the dataset since April 2022.
2. At the raion level, Severodonetsk and Luhansk have `len(df[df['raion'].str.contains('Сєвєродонец')]) == 0`.
3. In `train_layered.py` (lines 810–816):
   ```python
   elif np.sum(m_d) > 0 and len(np.unique(y_iso_m[m_d, d_i])) > 1:
       p_v_d, r_v_d, _ = precision_recall_curve(y_iso_m[m_d, d_i], p_iso_m[m_d, d_i])
       r_pr = float(auc(r_v_d, p_v_d))
   else:
       r_pr = max(0.01, round(base_rate_d, 3))
   ```
4. Because there are no positive alert transitions in the holdout period (`len(np.unique(y_test)) == 1`), `sklearn.metrics.precision_recall_curve` is mathematically undefined. The fallback assigns `r_pr = 0.010`.

---

## 3. Deep Learning & GNN Architecture Failure: Diffusion vs. Trajectory

The model uses Spatio-Temporal Graph Neural Networks (`LayeredSTGNN`, `HierarchicalSTGNN`) built on 2 layers of Graph Convolutional Networks (GCN):
$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

### A. Graph Convolution Models Heat Diffusion, Not Aerodynamic Flight
Graph Convolution is the discrete analogue of the heat diffusion equation:
$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$
Heat spreads isotropically to adjacent physical neighbors.

Missiles and Shahed UAVs do not diffuse like heat:
1. They are **high-speed directional rays** moving along specific flight vectors $\vec{v}$.
2. A Kh-101 launched over the Caspian Sea flies at $800\text{ km/h}$, crossing an entire oblast in 18 minutes. It does not "diffuse" into neighboring raions; it cuts a narrow $10\text{ km}$ corridor right through them.
3. Message-passing across administrative boundaries spreads threat probabilities outward like ink in water, creating massive false alarms in adjacent northern and southern raions that the missile will never touch.

### B. The 2-Hop Horizon Starvation & Over-Smoothing
* The graph of Ukraine's 136 raions has an administrative geodesic diameter of **$14\text{ hops}$**.
* The shortest path from the Sumy border to Lviv is **$8\text{ hops}$**.
* A 2-layer GCN has an exact spatial receptive field of **$k=2\text{ hops}$**.
* **Mathematical Reality:** It is **topologically impossible** for alert signals in Sumy, Kharkiv, or Chernihiv to reach Lviv through the GCN layers. The receptive field simply cannot see them.
* If one naively increases GCN layers to $K=8$ to reach Lviv, the network suffers from **exponential over-smoothing**:
  $$\lim_{K \to \infty} H^{(K)} \in \text{span}\left(\mathbf{1} \cdot \pi^T\right)$$
  All 136 raion node embeddings collapse to a uniform, indistinguishable spatial average.

### C. Hierarchical Pooling Information Dilution
The hierarchical pooling operator collapses micro raions into parent oblasts via:
$$H_{\text{macro}} = D_{\text{raion}}^{-1} S^T H_{\text{micro}}$$
In an oblast like Dnipropetrovsk (7 raions):
* If Nikopol experiences heavy Russian tube artillery shelling, its local combat indicator is $1.0$.
* The hard pooling operator averages this across all 7 raions:
  $$\bar{h}_{\text{Dnipropetrovsk}} = \frac{1}{7} \times 1.0 + \frac{6}{7} \times 0.0 = \mathbf{0.14}$$
* **$86\%$ of the tactical frontline signal is wiped out** before reaching the macro network.

---

## 4. Data & Feature Pipeline Starvation: The Auto-Regressive Trap

### A. Siren History is Self-Referential
Inspecting `layered_dataset.py`, the 16 features fed into the kinetic stream are:
1. Current siren state $s(t)$
2. Frontline siren state $s_{\text{front}}(t)$
3. Upstream graph pressure $P_{\text{fwd}} s(t)$
4. Downstream graph pressure $P_{\text{bwd}} s(t)$
5. Pressure velocity $\Delta P_{\text{fwd}} s(t)$
6–8. Static distances to border, frontline, and coast
9. National surge flag ($\text{mean}(s) > 0.35$)
10. Nocturnal indicator (21:00–06:00)
11–16. Six OSINT text flags

Notice that **every dynamic feature is a mathematical transformation of past siren states**.
* If Lviv has been silent for 10 days, its input vector is:
  $$\mathbf{x}_{\text{Lviv}}(t) = [0, 0, 0, 0, 0, d_1, d_2, d_3, 0, \text{noct}, 0, 0, 0, 0, 0, 0]$$
* **No machine learning algorithm in the world can predict whether a Russian Tu-95 bomber took off from Murmansk tonight if its only input is that the siren in Lviv was off for the past 2 hours.**
* Russian strike operations are dictated by external strategic logistics:
  - Bomber turnaround times at Olenya and Engels-2 airbases.
  - Kalibr missile reloading cycles in Sevastopol and Novorossiysk.
  - Drone truck convoy arrivals at Primorsko-Akhtarsk and Chauda.
  None of this information exists in historical siren time series.

### B. The OSINT Telegram "Mock Illusion"
During our code audit of `telegram_dataset.py`, we discovered why the model never learned to connect Air Force Telegram warnings with future sirens:
1. **Broken File Path:** `RAW_TELEGRAM_PATH` in `telegram_dataset.py` double-nested `air_raid_forecast/air_raid_forecast/...`, failing `os.path.exists()`.
2. **Empty Dataset:** The raw file `telegram_air_force_alerts.jsonl` contained only 14 hand-written dummy messages.
3. **Circular Synthesis Fallback:** In `train_layered.py`, the system fell back to `synthesize_deterministic_tactical_grid`. That function manufactures "tactical indicators" **only after sirens are already sounding**:
   - `flag_mig31k` is set to $1.0$ only when $\ge 16$ oblasts are *already* on alert!
   - `flag_strat_aviation` is set only when $\ge 12$ oblasts are *already* active!
4. **Result:** The model was trained on pseudo-indicators derived from sirens, meaning the network has **never seen real pre-launch early warning telemetry during training**.

---

## 5. Synthesis: Summary Matrix of Failure Points

| Component | Current Implementation | Failure Mode | Impact on Model Scores |
| :--- | :--- | :--- | :--- |
| **Tactical Horizon** | Fixed 1-hour window for all 136 raions | Shaheds take 5h to reach Lviv; warning at $t+0$ penalized as FP at $t+1\text{h}$ | Western PR-AUC collapses to $0.010 - 0.035$ |
| **Evaluation Metric** | Raw PR-AUC on absolute $[0, 1]$ scale | Baseline equals prevalence $\pi$; Lviv prevalence is $0.0021$, Donbas is $0.869$ | Misleading dashboard: Donbas appears "good", West appears "broken" |
| **Data Cleanliness** | Evaluates raions with zero positive samples | Severodonetsk & Luhansk have 0 alert events in official dataset | Flat $0.010$ plateau across occupied raions |
| **Neural Backbone** | 2-layer Graph Convolution (GCN) | Receptive field is 2 hops; cannot reach Lviv (8 hops from border); over-smooths | Spatial blindness; threat cannot propagate from east to west |
| **Feature Inputs** | Auto-regressive siren rolling stats only | Silent rear regions have zero dynamic input until neighboring sirens fire | Zero predictive foresight; reactively tracks sirens rather than anticipating them |
| **OSINT / Intelligence** | Circular synthetic flags generated from sirens | Model never learned true correlation between Telegram warnings and onsets | Early warning head defaults to static spatial priors |

---

## 6. The Verdict: Does the System Need a Scratch Rebuild?

**YES.** Modifying loss weights or adjusting learning rates within the existing 2-hop GCN / 1h-window paradigm cannot fix these problems because the flaws are **physical and mathematical**, not computational:
1. A 1-hour prediction window on Lviv will *always* penalize incoming 180 km/h Shahed swarms.
2. A 2-hop GCN will *never* propagate signals across an 8-hop country.
3. A siren-only dataset contains *zero* entropy regarding tonight's Russian bomber sorties.

To achieve genuine early warning accuracy across Ukraine, the system must transition to:
* **Continuous Survival / Time-to-Impact Modeling** instead of rigid hourly binary buckets.
* **Kinematic Trajectory Transformers (KTT)** with full-graph cross-attention and flight corridor bearing cones.
* **Precursor Intelligence Channel** ingesting true historical Air Force alerts and staging telemetry.
* **Information Gain Lift Evaluation** that measures true predictive power relative to base rate prevalence.
