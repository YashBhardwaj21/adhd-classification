# Experiment 3: Topological Feature Distribution & Cross-Site ANOVA Analysis

## 1. Scientific Overview & Objectives
Experiment 3 extracts and characterizes graph-theoretic metrics from the windowed brain networks constructed in Experiment 2. In multi-site neuroimaging consortia such as ADHD-200, topological measurements are prone to confounding scanner effects, radio-frequency coil differences, and pulse-sequence variations.

The primary objective of Experiment 3 is twofold:
1. Establish the empirical population-level distribution of dynamic network metrics across all 31,060 temporal windows.
2. Formally test for the existence of site-associated variance across acquisition sites via one-way Analysis of Variance (ANOVA).

---

## 2. Statistical Methodology

### 2.1 Metric Characterization
For each windowed graph, seven core topological metrics and four surrogate null metrics are tracked:
- **Clustering Coefficient ($C$)**: Local segregation and triangle density.
- **Transitivity ($T$)**: Ratio of triangles to connected triples across the whole network.
- **Global Efficiency ($E_{\text{glob}}$)**: Average inverse shortest path length, measuring information integration.
- **Characteristic Path Length ($L$)**: Average shortest path distance between all node pairs.
- **Normalized Metrics ($\gamma, \lambda, \sigma$)**: Scaled relative to degree-preserving null surrogates.

### 2.2 Cross-Site Variance Analysis
To assess scanner heterogeneity, a one-way ANOVA was conducted for each metric across the 8 acquisition sites:
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\sum_{j=1}^k n_j (\bar{y}_j - \bar{y})^2 / (k - 1)}{\sum_{j=1}^k \sum_{i=1}^{n_j} (y_{ij} - \bar{y}_j)^2 / (N - k)}$$
where $k = 8$ sites and $N = 31,060$ dynamic observations.

---

## 3. Audited Empirical Metric Distributions
Traced directly to [`results/exp03/feature_statistics.csv`](../../results/exp03/feature_statistics.csv) ($N = 31,060$ windows):

| Feature | Mean | Std | 25% | Median (50%) | 75% | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clustering** | $0.3480$ | $0.0479$ | $0.3145$ | $0.3431$ | $0.3755$ | $0.2324$ | $0.7294$ |
| **Transitivity** | $0.3468$ | $0.0601$ | $0.3048$ | $0.3382$ | $0.3787$ | $0.2121$ | $0.7770$ |
| **Efficiency** | $0.3929$ | $0.0157$ | $0.3835$ | $0.3923$ | $0.4015$ | $0.3429$ | $0.4700$ |
| **Path Length** | $2.8327$ | $0.1004$ | $2.7881$ | $2.8262$ | $2.8717$ | $2.5544$ | $4.9108$ |
| **Null Clustering** | $0.3373$ | $0.0470$ | $0.3045$ | $0.3326$ | $0.3641$ | $0.2249$ | $0.7260$ |
| **Null Transitivity**| $0.3471$ | $0.0618$ | $0.3037$ | $0.3383$ | $0.3804$ | $0.2082$ | $0.7775$ |
| **Null Efficiency**  | $0.4001$ | $0.0156$ | $0.3912$ | $0.3993$ | $0.4080$ | $0.3501$ | $0.4776$ |
| **Null Path Length**| $2.7818$ | $0.1087$ | $2.7299$ | $2.7764$ | $2.8311$ | $2.4561$ | $4.9110$ |
| **Gamma ($\gamma$)** | $1.0318$ | $0.0048$ | $1.0286$ | $1.0314$ | $1.0346$ | $1.0016$ | $1.0617$ |
| **Lambda ($\lambda$)**| $1.0185$ | $0.0103$ | $1.0122$ | $1.0197$ | $1.0258$ | $0.9782$ | $1.0520$ |
| **Sigma ($\sigma$)** | $1.0132$ | $0.0103$ | $1.0059$ | $1.0120$ | $1.0194$ | $0.9756$ | $1.0623$ |

---

## 4. Cross-Site ANOVA Statistical Audit
Traced directly to [`results/exp03/site_anova.csv`](../../results/exp03/site_anova.csv):

| Feature | $F$-Statistic | Unadjusted $p$-Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **path_length** | $4,993.87$ | $< 10^{-300}$ | Extreme site variance across scanners |
| **efficiency** | $4,609.76$ | $< 10^{-300}$ | Extreme site variance across scanners |
| **clustering** | $585.85$ | $< 10^{-300}$ | Significant site-associated offset |
| **transitivity** | $436.92$ | $< 10^{-300}$ | Significant site-associated offset |
| **gamma** ($\gamma$) | $460.07$ | $< 10^{-300}$ | Residual site variance survives null normalization |
| **sigma** ($\sigma$) | $214.91$ | $< 10^{-300}$ | Small-world index varies systematically by site |
| **lambda** ($\lambda$)| $84.29$ | $8.36 \times 10^{-139}$ | Highly significant site heterogeneity |

---

## 5. Critical Boundaries & Non-Overclaim Restraint
> [!IMPORTANT]
> **Strict Scientific Boundaries**:
> 1. **Site Effects vs Diagnostic Signal**: The ANOVA $p$-values in `site_anova.csv` test for differences **between acquisition sites**, not between ADHD patients and healthy controls. They demonstrate that raw topological features are heavily contaminated by scanner protocol variations, providing motivation for harmonization in Experiment 4.
> 2. **No Clinical Group Inferences**: Prior drafts claimed these statistics provided "artifact-verified biological proof of altered brain wiring." Such claims are scientifically unsupported: high $F$-statistics across imaging sites reflect technical batch effects, not biological pathology.
> 3. **Subject-Level Dependencies**: Because dynamic windows from the same subject are serially correlated, the degrees of freedom in window-level ANOVA are inflated. These statistics are diagnostic of scanner-level batch effects, but should not be interpreted as independent degrees of freedom for hypothesis testing.

---

## 6. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp03/03_topological_feature_extraction.ipynb`](../../notebooks/exp03/03_topological_feature_extraction.ipynb)
- **Result Artifacts**:
  - [`results/exp03/feature_statistics.csv`](../../results/exp03/feature_statistics.csv) (empirical distribution parameters)
  - [`results/exp03/site_anova.csv`](../../results/exp03/site_anova.csv) (cross-site ANOVA $F$ and $p$ values)
  - [`results/exp03/subject_graph_features.csv`](../../results/exp03/subject_graph_features.csv) (subject-averaged feature table)
