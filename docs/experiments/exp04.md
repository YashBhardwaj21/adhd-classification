# Experiment 4: ComBat Multi-Site Harmonization & The Clinical Variance Trade-off

## 1. Scientific Overview & Objectives
Given the pronounced cross-site batch effects identified in Experiment 3, multi-site neuroimaging studies require harmonization before features can be pooled for population-level modeling. Experiment 4 evaluates **NeuroCombat** (empirical Bayes harmonization) applied to the windowed graph metrics.

The primary objectives are:
1. Remove additive and multiplicative scanner-specific batch effects from topological network features.
2. Preserve genuine biological variation associated with age, sex, and diagnostic status.
3. Quantify the impact of harmonization on both site identifiability and diagnostic classification accuracy.

---

## 2. Methodology & Model Formulation

### 2.1 NeuroCombat Empirical Bayes Framework
For subject $j$ at site $i$ and feature $v$, the observed metric $y_{ijv}$ is modeled as:
$$y_{ijv} = \alpha_v + \mathbf{X}_{ij}^T \boldsymbol{\beta}_v + \gamma_{iv} + \delta_{iv} \epsilon_{ijv}$$
where:
- $\alpha_v$: Overall baseline feature mean.
- $\mathbf{X}_{ij}$: Design matrix of protected biological covariates (age, sex, diagnosis).
- $\boldsymbol{\beta}_v$: Regression coefficients for biological covariates.
- $\gamma_{iv}$: Additive site batch effect.
- $\delta_{iv}$: Multiplicative scale site batch effect.
- $\epsilon_{ijv} \sim \mathcal{N}(0, \sigma_v^2)$: Residual error.

Harmonized values $y_{ijv}^*$ are reconstructed by adjusting for empirical Bayes estimates $\hat{\gamma}_{iv}^*$ and $\hat{\delta}_{iv}^*$:
$$y_{ijv}^* = \frac{y_{ijv} - \hat{\alpha}_v - \mathbf{X}_{ij}^T \hat{\boldsymbol{\beta}}_v - \hat{\gamma}_{iv}^*}{\hat{\delta}_{iv}^*} + \hat{\alpha}_v + \mathbf{X}_{ij}^T \hat{\boldsymbol{\beta}}_v$$

---

## 3. Audited Harmonization Effects on Graph Metrics
Traced directly to [`results/exp04/comparison_table.csv`](../../results/exp04/comparison_table.csv):

| Metric | Raw Mean $\pm$ Std | ComBat Mean $\pm$ Std | Mean Difference ($\Delta$) | Effect Size |
| :--- | :--- | :--- | :--- | :--- |
| **Clustering** | $0.333271 \pm 0.034303$ | $0.333361 \pm 0.027542$ | $+0.000090$ | Variance reduced ($19.7\%$) |
| **Global Efficiency** | $0.582024 \pm 0.009142$ | $0.582107 \pm 0.008395$ | $+0.000083$ | Variance reduced ($8.2\%$) |
| **Local Efficiency** | $0.722478 \pm 0.011567$ | $0.722443 \pm 0.009197$ | $-0.000035$ | Variance reduced ($20.5\%$) |
| **Path Length** | $0.568345 \pm 0.057977$ | $0.568185 \pm 0.029857$ | $-0.000161$ | Variance reduced ($48.5\%$) |
| **Gamma ($\gamma$)** | $1.032007 \pm 0.002830$ | $1.031970 \pm 0.002234$ | $-0.000037$ | Variance reduced ($21.1\%$) |
| **Lambda ($\lambda$)**| $1.018686 \pm 0.004430$ | $1.018652 \pm 0.004151$ | $-0.000034$ | Variance reduced ($6.3\%$) |
| **Sigma ($\sigma$)** | $1.013170 \pm 0.004715$ | $1.013199 \pm 0.003937$ | $+0.000029$ | Small-world preserved |

Topological preservation analysis in `results/exp04/graph_topology_preservation.csv` demonstrates that global topological rankings and relative metric ordering were preserved ($r > 0.98$).

---

## 4. The Harmonization Trade-off: Site Identifiability vs Diagnostic Accuracy

To verify whether site effects were removed, linear classifiers were trained on raw versus ComBat-harmonized features under two tasks:
1. **Site Prediction** (Predicting which of the 8 scanners collected the data).
2. **Diagnosis Prediction** (Predicting ADHD vs TDC).

Traced directly to [`results/exp04/site_prediction_results.csv`](../../results/exp04/site_prediction_results.csv) and [`results/exp04/diagnosis_prediction_results.csv`](../../results/exp04/diagnosis_prediction_results.csv):

| Prediction Task | Raw Features Accuracy | ComBat Harmonized Accuracy | Delta ($\Delta$) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Site Classification** (8-way) | **$58.64\%$** | **$31.29\%$** | **$-27.35\%$** | Substantial attenuation of scanner signatures |
| **Diagnostic Classification** (Binary) | **$64.53\%$** | **$59.56\%$** | **$-4.97\%$** | Diagnostic accuracy dropped by ~5 percentage points |

---

## 5. Critical Scientific Discussion: Non-Overclaiming Interpretation
> [!IMPORTANT]
> **Interpretation of the Diagnostic Accuracy Drop**:
> - **Sensitivity to Site-Associated Variance**: The decrease in ADHD classification accuracy from $64.53\%$ to $59.56\%$ demonstrates that downstream diagnostic models were partially relying on site-associated variance in the raw data.
> - **What This Does NOT Prove**: It does not prove that the variance removed by ComBat was "true clinical signal" nor that raw performance was "purely artifactual." In multi-site observational datasets, clinical presentation, participant recruitment demographics (e.g., socioeconomic status, local comorbidity profiles), and scanner characteristics are frequently collinear. 
> - When ComBat removes site-correlated variance, it necessarily removes any biological variance that is collinear with site, while eliminating non-biological scanner offsets.

---

## 6. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp04/04_combat_harmonization.ipynb`](../../notebooks/exp04/04_combat_harmonization.ipynb)
- **Result Artifacts**:
  - [`results/exp04/comparison_table.csv`](../../results/exp04/comparison_table.csv) (raw vs ComBat metric table)
  - [`results/exp04/site_prediction_results.csv`](../../results/exp04/site_prediction_results.csv) (8-site classification performance)
  - [`results/exp04/diagnosis_prediction_results.csv`](../../results/exp04/diagnosis_prediction_results.csv) (ADHD vs TDC classification performance)
  - [`results/exp04/graph_topology_preservation.csv`](../../results/exp04/graph_topology_preservation.csv) (correlation preservation metrics)
  - [`results/exp04/effect_size_before_after.csv`](../../results/exp04/effect_size_before_after.csv) (standardized mean differences)
