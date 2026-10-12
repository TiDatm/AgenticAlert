# ROOT CAUSE INVESTIGATION: EXTREME ACCURACY POLARIZATION IN FRONTLINE & WESTERN REGIONS
## Analysis of Regional Performance Asymmetry on Ukrainian Air Raid Alert Map

---

### Executive Diagnosis
Looking closely at the map generated on screen, there is a striking polarization:
1. **Western Regions (Lviv: 100%, Zakarpattia: 100%, Ivano-Frankivsk: 100%, Chernivtsi: 99.0%)**:
   - **Mechanism**: **All-Negative Trivial Memorization (The True Negative Flood)**.
   - The empirical positive rate during the test period was **$0.0\%$**. Because alarms rarely reach western borders during quiet periods, a model predicting constant zero achieves 100% accuracy while having $\text{Recall} = 0.0$ and $\text{Precision} = 0.0$.
2. **Frontline & Eastern Regions (Kharkiv: 100%, Donetsk: 100%, Zaporizhzhia: 100%, Dnipro: 100%)**:
   - **Mechanism**: **All-Positive Saturation & Aggregation Artifact (The True Positive Flood)**.
   - The empirical positive rate at the Oblast level was **$100.0\%$**. Sirens were virtually never silent in these regions during the test period. A model predicting constant 1 achieves 100% accuracy trivially.
3. **Transition Corridors (Poltava: 74.0%, Kirovohrad: 75.0%, Mykolaiv: 71.2%, Kherson: 64.4%, Kyiv: 85.6%)**:
   - **Mechanism**: **True Non-Trivial Modeling Zone**.
   - These intermediate regions have balanced positive/negative distributions ($30\%\text{–}60\%$ alert prevalence), forcing the model to make genuine non-trivial predictions rather than exploiting class saturation.

---

### 1. Root Cause 1: District (Raion) Aggregation vs. Oblast Saturation

In the dataset schema, examine the `level` column breakdown during the latest test period:
- `raion` (District level): **4,194 records**
- `hromada` (Community level): **187 records**
- `oblast` (Whole region): **54 records**

#### The Aggregation Bug / Artifact:
Since late 2024 / 2025, Ukraine adopted localized district-level sirens (`raion`). For instance, Kharkiv Oblast contains 7 raions (Bohodukhiv, Izium, Krasnohrad, Kupyansk, Lozova, Kharkiv, Chuhuiv).
- Frontline districts (e.g. Kupyansk, Chuhuiv) are under almost **continuous shelling or drone surveillance alarms 24/7**.
- Rear districts in the same oblast (e.g. Krasnohrad) might be calm.
- **When aggregated to the Oblast level**: If *any single district* is on alert, the whole Oblast grid cell $G_{t, \text{Kharkiv}} = 1$.
- **Result**: The continuous border shelling in Kupyansk causes Kharkiv Oblast's label to stay permanently stuck at **$1.0$ (100% active)**. The model trivially outputs $p \approx 0.99$ and achieves 100% accuracy without forecasting anything dynamic.

---

### 2. Root Cause 2: Extreme Label Imbalance Polarizing the Accuracy Metric

Examining the actual underlying metrics exported in `regional_accuracy.json`:

```json
"Харківська область": {
  "accuracy": 100.0,
  "positive_rate": 100.0,    <-- 100% of time steps had alerts active
  "precision": 100.0,
  "recall": 100.0,
  "roc_auc": 0.5             <-- Zero discriminatory variance (ROC-AUC drops to baseline 0.5!)
}
```

Compare this to Lviv:
```json
"Львівська область": {
  "accuracy": 100.0,
  "positive_rate": 0.0,      <-- 0% of time steps had alerts active
  "precision": 0.0,
  "recall": 0.0,
  "roc_auc": 0.5             <-- Zero discriminatory variance
}
```

#### Why `ROC-AUC = 0.5` is the Smoking Gun:
Notice that for **both Kharkiv (100% positive) and Lviv (0% positive)**, `ROC-AUC` is exactly **0.50** (the theoretical minimum score for a completely uninformative model).
- Accuracy reports **100%** because there were no negative instances in Kharkiv to get wrong ($\text{TN} = 0, \text{FN} = 0$), and no positive instances in Lviv to get wrong ($\text{TP} = 0, \text{FP} = 0$).
- Standard Accuracy treats both trivial edge cases as "perfect", creating a false visual impression on the map.

---

### 3. Root Cause 3: The Threat Vector Transit Corridor Phenomenon

Why do the central transit regions show 64%–75% accuracy?
- **Kherson (64.4%)**, **Mykolaiv (71.2%)**, **Odesa (73.1%)**, **Kirovohrad (75.0%)**, and **Poltava (74.0%)** form the actual **aerial transit corridors** for Shahed/Geran drone swarms launching from Crimea and the Sea of Azov.
- In these regions:
  - An alert is not continuous (unlike Kupyansk or Nikopol).
  - An alert is not nonexistent (unlike Lviv during lulls).
  - Alarms turn on and off based on whether a specific drone cluster penetrates air defenses, turns north towards Kyiv, or veers west towards Vinnytsia.
- **This is the hardest mathematical prediction zone**: The model must predict flight path branching. That is why intermediate transit corridors have lower raw accuracy (64%–79%), even though they are the **only regions where the model is performing genuine dynamic inference**.

---

### 4. Root Cause 4: Permanent Territorial Occupation & Static Sirens (Luhansk & Crimea)

- **Luhansk Oblast**: Permanently marked as on alert or completely unmonitored in open APIs (often logged as continuous alert or zero records depending on the scraper source).
- **Crimea**: Shows an artificial static value (85.0%) because civilian siren networks in occupied territories are not integrated with Ukraine's State Emergency Service API.

---

### 5. Actionable Engineering Fixes to Remove the Distortion

To eliminate these measurement distortions and display true operational accuracy on the map:

1. **Target Disaggregation (Switch from "Any Alert" to "New Alert Trigger")**:
   - Instead of predicting: *"Is an alert active at $t+H$?"* (which is trivially 1 in Kharkiv and 0 in Lviv),
   - Predict: *"Will a **NEW alert start** in region $r$ during window $[t, t+H]$ given that it is currently idle?"*
   - This eliminates the trivial saturation of continuous sirens in frontline oblasts.
2. **Filter by Administrative Level**:
   - Filter `df[df['level'] == 'oblast']` or require a threshold of active districts (e.g. $\ge 50\%$ of raions on alert) before marking the Oblast active, preventing a single border artillery skirmish from turning the whole oblast positive.
3. **Replace Map Metric from Raw Accuracy to Balanced Accuracy or PR-AUC**:
   - **Balanced Accuracy**: $\frac{\text{Recall} + \text{Specificity}}{2}$.
   - For Lviv (where positives are 0) and Kharkiv (where negatives are 0), Balanced Accuracy correctly penalizes class collapse rather than showing a misleading 100%.
