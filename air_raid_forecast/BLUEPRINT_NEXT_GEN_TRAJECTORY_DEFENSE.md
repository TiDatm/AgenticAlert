# Blueprint: Next-Generation Kinematic Trajectory & Survival Defense Architecture

**Document Type:** Strategic Architectural Blueprint & Ground-Up Redesign Specification  
**Project:** Ukrainian Civil Defense & Air Threat Forecasting Engine (`air_raid_forecast v3`)  
**Target:** Full Resolution of Horizon Mismatch, False Alarm Inversion, GNN Over-Smoothing, and Metric Distortion  

---

## 1. Executive Summary: The Paradigmatic Shift

The current system attempts to solve an impossible problem: **classifying whether a siren will sound in a fixed 1-hour window using only the historical on/off state of adjacent sirens**. 

As proven in our research audit (`RESEARCH_ROOT_CAUSE_PERFORMANCE_COLLAPSE.md`), this baseline fails because:
1. Aerial threats move along **directed kinematic trajectories** spanning hundreds of kilometers, not isotropic graph diffusion.
2. A 1-hour prediction window **penalizes long-range early warnings** for subsonic drones and cruise missiles.
3. The siren history contains **zero precursor telemetry** on bomber sorties or drone launches.
4. Raw PR-AUC on extreme class imbalances collapses to the prior prevalence rate ($\pi \approx 0.002$).

To achieve state-of-the-art predictive performance, the system must undergo a complete architectural transformation across four core subsystems:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               THE NEXT-GENERATION TRAJECTORY DEFENSE PARADIGM                          │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ OLD ARCHITECTURE (v2)    │ NEXT-GENERATION ARCHITECTURE (v3)                           │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2-Hop Graph Conv (GCN)   │ Kinematic Trajectory Transformer (KTT) with Bearing Cones   │
│ Discrete [1h, 4h, 24h]   │ Continuous Survival / Hazard Function & Tiered Kinematics   │
│ Auto-Regressive Sirens   │ Multimodal Ingestion of Precursor Air Force & Airfield OSINT│
│ Raw PR-AUC Choropleths   │ Prevalence-Normalized Precision Lift (λ_PR) & Brier Scores  │
│ Degenerate Sentinels     │ Explicit "No-Telemetry / Active Combat" Exclusion Masking   │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 2. Subsystem 1: Kinematic Trajectory Transformer (KTT)

### 2.1 Why Replace GCN with a Trajectory Transformer?
Message-passing graph neural networks ($A X W$) discretize heat diffusion. Aerial threats are **directed ballistic or aerodynamic vectors**. An incoming cruise missile or drone swarm does not propagate from raion to raion through administrative adjacency; it flies through air corridors at velocities $v \in [180, 900]\text{ km/h}$.

The **Kinematic Trajectory Transformer (KTT)** replaces local 2-hop message-passing with **full-graph Cross-Attention modulated by geometric corridor bearing cones**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + \mathcal{B}_{\text{kinematic}} \right) V$$

### 2.2 Mathematical Formulation of the Kinematic Bearing Bias $\mathcal{B}_{\text{kinematic}}$
For any pair of nodes (raion $i$ to raion $j$):
1. Let $\vec{c}_i, \vec{c}_j \in \mathbb{R}^2$ be the spatial coordinates (latitude, longitude).
2. Let $\vec{u}_{i \to j} = \frac{\vec{c}_j - \vec{c}_i}{\|\vec{c}_j - \vec{c}_i\|}$ be the unit vector pointing from $i$ to $j$.
3. Let $\vec{v}_{\text{threat}}(i) \in \mathbb{R}^2$ be the estimated threat heading vector at node $i$ (extracted from upstream border ingress or Air Force Telegram alerts).
4. The **Directional Alignment Factor** is:
   $$\cos(\theta_{ij}) = \vec{v}_{\text{threat}}(i) \cdot \vec{u}_{i \to j}$$
5. The **Kinematic Bearing Bias Matrix** $\mathcal{B}_{\text{kinematic}} \in \mathbb{R}^{N \times N}$ is defined as:
   $$\mathcal{B}_{\text{kinematic}}(i, j) = \begin{cases}
   \beta_{\text{fwd}} \cdot \max(0, \cos(\theta_{ij})) - \lambda \cdot \frac{\|\vec{c}_i - \vec{c}_j\|}{v_{\text{effective}}} & \text{if } \cos(\theta_{ij}) > 0 \\
   -\infty \quad (\text{masked out}) & \text{if } \cos(\theta_{ij}) \le -0.2 \text{ (upstream backflow)}
   \end{cases}$$

```
                Threat Heading Vector v_threat(i)
                             ▲
                             │  / +1.0 (Directly along flight corridor)
                             │ /
                 Raion i ────┼──── Raion j (Target)
                             │ \
                             │  \ -∞ (Upstream / Opposing direction)
```

**Key Advantages:**
* **Instant Receptive Field:** Any raion in Ukraine can attend to an active threat at the border in a single attention layer without hop limits ($8\text{ hops} \to 1\text{ step}$).
* **Zero Over-Smoothing:** Attention weights dynamically concentrate only on raions lying inside the forward threat cone, leaving lateral and upstream regions unaffected.
* **Velocity-Calibrated Delays:** The spatial distance penalty $-\lambda \frac{d}{v}$ directly encodes physical flight times into the attention logits.

---

## 3. Subsystem 2: Restructuring Temporal Horizons & Continuous Survival Modeling

The static $[1\text{h}, 4\text{h}, 24\text{h}]$ tripartite structure is fundamentally flawed. We replace it with **Physics-Grounded Tiered Horizons** and a **Continuous Hazard Rate Head**.

### 3.1 Physics-Grounded Tiered Operational Horizons

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHYSICS-GROUNDED OPERATIONAL HORIZONS                           │
├───────────────────────┬──────────────┬────────────────────────┬────────────────────────┤
│ Regime                │ Window       │ Target Threat Classes  │ Target Regions         │
├───────────────────────┼──────────────┼────────────────────────┼────────────────────────┤
│ 1. Instantaneous      │ 15 Minutes   │ Ballistics (Iskander), │ Frontline & Border     │
│    Tactical Reaction  │ (0 - 15m)    │ KAB Glide Bombs, MLRS  │ Raions (0 - 80 km)     │
├───────────────────────┼──────────────┼────────────────────────┼────────────────────────┤
│ 2. Kinematic Corridor │ Dynamic      │ Cruise Missiles,       │ All 136 Raions         │
│    Incursion Envelope │ τ(r) ∈ [1,5h]│ Shahed-136 UAV Swarms  │ (Distance-Calibrated)  │
├───────────────────────┼──────────────┼────────────────────────┼────────────────────────┤
│ 3. Strategic Overnight│ 12 Hours     │ Mass Salvo Campaign    │ All Ukraine (Oblasts)  │
│    Salvo Window       │ (21:00-09:00)│ Replenishment Cycles   │                        │
└───────────────────────┴──────────────┴────────────────────────┴────────────────────────┘
```

#### Formulation of the Regional Kinematic Horizon $\tau(r)$:
$$\tau(r) = \max\left(1.0\text{ h}, \; \alpha \cdot \frac{d_{\text{threat}}(r)}{v_{\text{effective}}}\right)$$
* Border / Frontline Raions (Kharkiv, Sumy, Kherson): $\tau = \mathbf{1.0\text{ h}}$ ($4$ steps).
* Central Transit Corridor (Kyiv, Poltava, Dnipro): $\tau = \mathbf{2.0 - 2.5\text{ h}}$ ($8 - 10$ steps).
* Deep Rear Buffer (Lviv, Zakarpattia, Volyn, Rivne): $\tau = \mathbf{4.5 - 5.0\text{ h}}$ ($18 - 20$ steps).

### 3.2 Continuous Survival / Hazard Function Head
Instead of forcing the network to make a discrete binary guess (*"Will a siren sound within the next 60 minutes?"*), the model outputs the parameters of a **Continuous Conditional Survival Function**:

$$S_r(t \mid \mathcal{F}_{\tau}) = P(T_{\text{siren}} > t \mid T_{\text{siren}} > 0)$$

Using a Weibull-parameterized neural hazard head:
$$\lambda_r(t) = \frac{\kappa_r}{\eta_r} \left( \frac{t}{\eta_r} \right)^{\kappa_r - 1}$$
$$S_r(t) = \exp\left( -\left(\frac{t}{\eta_r}\right)^{\kappa_r} \right)$$
where $\eta_r = \text{softplus}(W_\eta h_r)$ is the **Scale Parameter** (characteristic time to impact) and $\kappa_r = \text{softplus}(W_\kappa h_r) + 0.5$ is the **Shape Parameter**.

#### Why This Resolves the False Alarm Penalty:
1. If a Shahed swarm is detected at $t=0$ heading for Lviv, the model predicts $\eta_{\text{Lviv}} = 4.2\text{ hours}$.
2. Under the old 1h binary bucket, minute 61 was scored as a $100\%$ false alarm.
3. Under the continuous survival likelihood, the model's prediction that the siren would arrive at hour $4.2 \pm 0.3\text{h}$ is scored with **maximum log-likelihood**.
4. The user dashboard displays an intuitive operational readout:
   > **Lviv District:** *Expected time until sirens: 4 hours 15 minutes (Confidence: 89%).*

---

## 4. Subsystem 3: Multimodal Intelligence & Precursor Feature Ingestion

A model that only watches past sirens cannot predict future strikes. The feature pipeline must ingest **true external precursor telemetry**.

```
  External Telemetry Sources
  ══════════════════════════════════════════════════════════════════════════════
  1. Russian Airbase Sorties    ──► Olenya (Tu-95MS), Engels (Tu-160), Savasleyka (MiG-31K)
  2. Drone Launch Facilities    ──► Primorsko-Akhtarsk, Yeysk, Chauda, Kursk
  3. Ukrainian Air Force OSINT  ──► Real-time @kpszsu parsing (corridor, weapon type, count)
  4. Atmospheric ERA5 Weather   ──► Low cloud ceiling (AGL < 200m), wind vectors (850 hPa)
```

### 4.1 New Feature Tensor Specification

| Feature Channel | Dim | Physical Meaning & Precursor Warning Value |
| :--- | :---: | :--- |
| **Airfield Staging Dynamics** | **4** | Hours since bomber takeoff from Olenya/Engels; MiG-31K airborne endurance timer; expected missile release window ($T_{\text{takeoff}} + 4.5\text{h}$). |
| **Launch Facility Staging** | **4** | Rolling 24h launch activity at Primorsko-Akhtarsk, Chauda, and Kursk; estimated drone wave size. |
| **True Historical OSINT Ingestion** | **6** | Verified regex-parsed indicators from Ukrainian Air Force (@kpszsu) historical message archives: `[mig31k, ballistics, strat_aviation, shahed_group, kab_tactical, all_clear]`. |
| **Corridor Ingress Vectors** | **3** | Dynamic flight bearing vector $(\cos \theta, \sin \theta)$ and inbound wave speed estimated from radar reports. |
| **Atmospheric ERA5 Reanalysis** | **3** | Low cloud ceiling (degrades mobile fire group visual tracking); surface wind speed and headwind component along corridor. |
| **Adjacency & Kinematics** | **8** | Directed forward/backward advection, border/frontline distances, nocturnal indicator. |
| **Total Kinematic Input Dimension** | **28** | *(Upgraded from legacy 16-dim circular siren features).* |

---

## 5. Subsystem 4: Statistically Sound Evaluation & Information Gain Lift

### 5.1 The Metric Fallacy Solution: Normalized Information Gain Lift
To eliminate the dark-purple artifact across Western Ukraine caused by extreme class imbalance ($\pi \approx 0.002$), the dashboard must evaluate:

$$\text{Lift}(r) = \frac{\text{PR-AUC}(r)}{\pi(r)}$$

where $\pi(r) = P(Y_r = 1 \mid \text{Idle}_r = 1)$ is the regional base rate.

* **$\text{Lift} = 1.0$**: Model provides zero information (equivalent to a dummy coin flip).
* **$\text{Lift} > 5.0$**: Model increases alert precision by $500\%$ over chance.
* **$\text{Lift} > 15.0$**: Exceptional early warning capability.

On the revised choropleth:
- Western Ukraine (Lviv, Zakarpattia, Volyn) with $\text{PR-AUC} = 0.035$ and $\pi = 0.0021$ achieves **$\text{Lift} = 16.7\times$**, rendering in **bright vibrant green/gold**.
- Frontline Donetsk with $\text{PR-AUC} = 0.90$ and $\pi = 0.869$ achieves **$\text{Lift} = 1.04\times$**, correctly showing that high raw precision is simply an artifact of permanent contact.

### 5.2 Explicit Telemetry Masking for Occupied & Continuous-Combat Nodes
Raions with degenerate sample support must be explicitly categorized rather than assigned a misleading `0.010` sentinel:
* **Category 1: `Normal Evaluated`** ($N_{\text{idle}} \ge 50$ and transitions observed): Scored via Lift and Continuous Hazard MAE.
* **Category 2: `Continuous Combat Contact`** (Base rate $\ge 90\%$, idle mask $< 5\%$): Scored via Clearance/All-Clear accuracy rather than onset.
* **Category 3: `Telemetry Severed / Occupied`** (Luhansk, Severodonetsk, Mariupol): Rendered with **diagonal hatched gray pattern** labeled *"No Civil Siren Telemetry Available"*.

---

## 6. Implementation Roadmap & Architecture Comparison

| Dimension | Legacy Model (v2) | Next-Gen Architecture (v3) | Expected Impact |
| :--- | :--- | :--- | :--- |
| **Spatial Model** | 2-Hop GCN ($A X W$) | Kinematic Trajectory Transformer (KTT) with Bearing Cones | Full-country receptive field in 1 layer; zero over-smoothing. |
| **Temporal Horizon** | Rigid 1h / 4h / 24h buckets | Continuous Survival Function $\hat{T}_{\text{impact}}$ + Isochrone Windows | Eliminates false-alarm penalties for accurate early warnings. |
| **Western PR-AUC** | $0.010 - 0.035$ (Dark Purple) | **$0.65 - 0.75$** under 4.5h isochrone; **$15\times - 20\times$ Lift** | Western defense commands receive accurate, actionable alerts. |
| **Data Ingestion** | Siren time series only | Airfield staging telemetry + real historical Telegram logs | Anticipates bomber and drone waves hours before border ingress. |
| **Occupied Nodes** | $0.010$ sentinel fallback | Explicit hatched masking (`No Telemetry`) | Clean, transparent operational dashboard. |
| **Loss Function** | Unbalanced Focal Loss + Huber | Weibull Continuous Hazard NLL + Balanced Focal Loss | Eliminates gradient starvation on rare rear events. |
