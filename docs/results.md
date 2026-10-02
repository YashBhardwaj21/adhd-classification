# Canonical Empirical Results

This document compiles the verified numerical findings, evaluation tables, statistical tests, and interpretation boundaries across all nine experiments.

---

## 1. Executive Results Summary

The empirical findings demonstrate:
1. Resting-state dynamic functional connectivity exhibits smooth, monotonic temporal autocorrelation decay ($0.8990 \to 0.5467$ across lags 1–4) and continuous Frobenius distance expansion ($32.58 \to 70.61$).
2. Temporal dispersion in dynamic graph metrics captures significant cross-site scanner variation ($F > 4600$, $p < 10^{-15}$) but does not survive multiple testing correction for diagnostic group separation under full cohort validation.
3. ComBat harmonization substantially suppresses scanner site prediction accuracy from **58.64%** down to **31.29%** (balanced accuracy: 50.19% down to 23.56%) while preserving intrinsic graph metric distributions.
4. Under matched single-split conditions on 33 held-out test subjects, the isotropic Classical GCN achieved **0.7293 AUC** (69.70% accuracy) compared to **0.6429 AUC** (60.61% accuracy) for the 6-qubit hybrid QGCNN.
5. Under strict 7-fold Leave-One-Site-Out (LOSO) cross-validation across 497 subjects, graph neural network architectures achieved modest out-of-distribution discrimination (GAT: **0.5752 AUC**, GraphSAGE: **0.5502 AUC**, GCN: **0.5468 AUC**, GIN: **0.5437 AUC**). Classical ML models combining functional connectivity with phenotypic covariates achieved the highest observed generalization (SVM with Graph+Phenotype: **0.6493 AUC**; Extra Trees with FC+Phenotype: **0.6328 AUC**).

---

## 2. Experiment 01 — Dynamic FC Temporal Validation

Evaluation across 1,193 imaging acquisitions (764 subjects, 31,060 sliding windows, $W=30$ TRs, stride=5 TRs) demonstrates continuous, monotonic temporal autocorrelation decay across consecutive temporal window lags:

| Metric | Lag 1 (1 window / 5 TRs) | Lag 2 (2 windows / 10 TRs) | Lag 3 (3 windows / 15 TRs) | Lag 4 (4 windows / 20 TRs) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Temporal Similarity ($r$)** | $0.8990 \pm 0.041$ | $0.7789 \pm 0.068$ | $0.6609 \pm 0.091$ | $0.5467 \pm 0.112$ |
| **Mean Frobenius Distance ($D_F$)** | $32.58 \pm 4.21$ | — | — | $70.61 \pm 9.14$ |

*Note: Intermediate Frobenius distance values at lags 2 and 3 were not retained in the canonical summary; exact empirical values are $D_F=32.58$ at lag 1 and $D_F=70.61$ at lag 4 (`results/exp01/temporal_similarity_validation.csv`). Static versus dynamic functional connectivity Frobenius distance across all 1,193 acquisitions averaged $13.83 \pm 4.68$ (`results/exp01/static_dynamic_fc_comparison.csv`).*

---

## 3. Experiment 02 — Graph Construction and Null-Model Evaluation

The Kruskal Minimum Spanning Tree combined with Proportional Thresholding (MST+PT) at density $\rho = 0.20$ guarantees network connectedness across all 31,060 windows:
- **Active ROIs**: 190 nodes (CC200 atlas).
- **Target Edge Count**: Exactly 3,591 edges per window ($190 \times 189 / 2 \times 0.20 = 3,591$).
- **Connected Components**: Exactly 1.0 (connected network guaranteed) across all 31,060 windows.
- **Distance Metric**: $d_{ij} = 1 - |r_{ij}|$ for MST edge filtering followed by proportional edge retention.

---

## 4. Experiment 03 — Graph-Theoretic Diagnosis Effects

Topological network metrics were extracted across all 31,060 sliding windows and evaluated for site dependence across the phenotypically complete subset (`results/exp03/site_effect_anova.csv`):

| Graph Metric | Description / Symbol | Window-Level Mean ($\mu \pm \sigma$, $N=31,060$) | Site Effect ANOVA $F$-Statistic | Site Effect $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **Clustering Coefficient** | $C$ | $0.3480 \pm 0.0479$ | 585.85 | $< 10^{-15}$ |
| **Transitivity** | $T$ | $0.3468 \pm 0.0601$ | 436.92 | $< 10^{-15}$ |
| **Global Efficiency** | $E_g$ | $0.3929 \pm 0.0157$ | 4,609.76 | $< 10^{-15}$ |
| **Characteristic Path Length** | $L$ | $2.8327 \pm 0.1004$ | 4,993.87 | $< 10^{-15}$ |
| **Null Global Efficiency** | $E_{g,\text{null}}$ | $0.4001 \pm 0.0156$ | 5,523.08 | $< 10^{-15}$ |
| **Null Characteristic Path** | $L_{\text{null}}$ | $2.7818 \pm 0.1087$ | 3,786.81 | $< 10^{-15}$ |
| **Normalized Clustering** | $\gamma = C / C_{\text{null}}$ | $1.0318 \pm 0.0048$ | 460.07 | $< 10^{-15}$ |
| **Normalized Path Length** | $\lambda = L / L_{\text{null}}$ | $1.0185 \pm 0.0103$ | 84.29 | $8.36 \times 10^{-139}$ |
| **Small-Worldness** | $\sigma = \gamma / \lambda$ | $1.0132 \pm 0.0103$ | 214.91 | $< 10^{-15}$ |

**Key Finding**: Cross-site scanner variation exerted massive statistical influence across all graph-theoretic metrics ($F > 4600$ for global efficiency and path length), dominating any subtle diagnostic group separation.

---

## 5. Experiment 04 — Site Effects and Harmonization

Application of empirical Bayes ComBat harmonization to CC200 graph metrics yielded substantial attenuation of scanner bias while preserving underlying topological metric distributions:

### Scanner Site and Diagnostic Prediction (`results/exp04/site_prediction_combat_comparison.csv` and `diagnosis_prediction_combat_comparison.csv`)

| Task / Dataset | Accuracy | Balanced Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- |
| **Site Prediction (Raw)** | **58.64%** | **50.19%** | **0.5061** |
| **Site Prediction (ComBat)** | **31.29%** | **23.56%** | **0.2437** |
| **Diagnosis Prediction (Raw)** | 64.53% | 59.43% | 0.5953 |
| **Diagnosis Prediction (ComBat)** | 59.56% | 51.60% | 0.4992 |

*Diagnostic diagnostic tool: Random Forest classifier.*

### Topological Preservation Across Harmonization (`results/exp04/combat_harmonization_comparison.csv`)

| Metric | Raw Mean $\pm$ Std | ComBat Mean $\pm$ Std | Mean Difference ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Clustering Coefficient** | $0.333271 \pm 0.034303$ | $0.333361 \pm 0.027542$ | $+0.000090$ |
| **Global Efficiency** | $0.582024 \pm 0.009142$ | $0.582107 \pm 0.008395$ | $+0.000083$ |
| **Local Efficiency** | $0.722478 \pm 0.011567$ | $0.722443 \pm 0.009197$ | $-0.000035$ |
| **Characteristic Path Length** | $0.568345 \pm 0.057977$ | $0.568185 \pm 0.029857$ | $-0.000161$ |
| **Normalized Clustering ($\gamma$)** | $1.032007 \pm 0.002830$ | $1.031970 \pm 0.002234$ | $-0.000037$ |
| **Normalized Path ($\lambda$)** | $1.018686 \pm 0.004430$ | $1.018652 \pm 0.004151$ | $-0.000034$ |
| **Small-Worldness ($\sigma$)** | $1.013170 \pm 0.004715$ | $1.013199 \pm 0.003937$ | $+0.000029$ |

---

## 6. Experiment 05 — Dynamic Brain State Discovery

Unsupervised $k$-means clustering across 31,060 sliding-window correlation matrices (764 subjects, 1,193 acquisitions, 9 sites) identified $K=3$ distinct, recurring connectivity micro-states (`results/exp05/subject_feature_summary.csv` and `dynamic_state_transitions.csv`):

| State Descriptor | State 0 (Dominant Integrated) | State 1 (Segregated / Sparse) | State 2 (Transitional / Intermediate) |
| :--- | :--- | :--- | :--- |
| **Subject-Level Mean Occupancy** | **49.05%** ($\pm 30.54\%$) | **27.32%** ($\pm 27.57\%$) | **23.63%** ($\pm 16.66\%$) |
| **Window-Level Aggregate Distribution** | 48.16% (14,957 / 31,060) | 27.27% (8,471 / 31,060) | 24.57% (7,632 / 31,060) |
| **Mean Dwell Time (Windows)** | **6.24 windows** ($\pm 6.46$) | **3.44 windows** ($\pm 5.56$) | **2.83 windows** ($\pm 2.45$) |
| **Mean Self-Transition ($P_{ii}$)** | **0.6939** | 0.3954 | 0.4918 |
| **Cohort Switching Rate** | \multicolumn{3}{c|}{$0.2048 \pm 0.1151$} |
| **Transition Entropy** | \multicolumn{3}{c|}{$0.6925 \pm 0.3699$} |
| **Clustering Stability** | \multicolumn{3}{c|}{MeanRank selected $K=3$ (Rank 1, MeanRank=2.333); 30-run ARI = $0.9950 \pm 0.0024$} |

State 0 represents a strongly integrated configuration with the highest temporal persistence and dwell time.

---

## 7. Experiment 06 — Semi-Supervised Pseudo-Labeling

Semi-supervised pseudo-labeling was benchmarked on 955 subjects parcellated with the AAL-116 atlas ($d=6,670$ upper-triangle features):
- **Procedure I (Self-Training with Thresholding)**: Class-weighted Logistic Regression ($\tau = 0.75$) generated **552 pseudo-labels** from 564 candidate unlabeled samples.
- **Procedure II (Multi-Model Ensemble Consensus)**: 4-model weighted ensemble consensus voting (Random Forest, Gradient Boosting, Logistic Regression, Calibrated SVM with $\bar{p} \ge 0.75$) generated **484 high-confidence pseudo-labels**.
- **Holdout Evaluation ($N=79$)**:
  - Clean-only training baseline accuracy: **67.09%**
  - Pseudo-label augmented training accuracy: **72.15%** ($\Delta = +5.06\%$)

> **Important Cohort Distinction**: The 484 and 552 pseudo-labels generated in Exp 06 are methodological evaluation records on the 391/564 split. They are distinct from the **713 downstream pseudo-labels** historically combined with 103 clean training samples in Exp 07.

---

## 8. Experiment 07 — Classical GCN versus Hybrid QGCNN

Evaluation on the **33 held-out clean test subjects** under matched graph density ($\rho=0.15$), unweighted message passing, and 117-dimensional node features:

| Architecture | Test AUC | Accuracy | Precision | Recall | F1 Score | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical GCN** | **0.7293** | **69.70%** (23/33) | 0.6941 | 0.6970 | **0.6929** | 5,522 |
| **Hybrid QGCNN** | **0.6429** | **60.61%** (20/33) | 0.6204 | 0.6061 | **0.6082** | 2,188 |

### Confusion Matrices ($N=33$: 19 TDC, 14 ADHD)
- **Classical GCN**:
  $$\begin{bmatrix} 15 & 4 \\ 6 & 8 \end{bmatrix} \quad (\text{Specificity} = 78.9\%, \text{Sensitivity} = 57.1\%)$$
- **Hybrid QGCNN**:
  $$\begin{bmatrix} 11 & 8 \\ 5 & 9 \end{bmatrix} \quad (\text{Specificity} = 57.9\%, \text{Sensitivity} = 64.3\%)$$

*Execution Context*: Single train/validation/test split (`seed=42`), single initialization seed, noiseless state-vector simulation via PennyLane.

---

## 9. Experiment 08 — Volumetric and Temporal Baselines

Deep learning baselines evaluated on raw 4D functional BOLD volumes and temporal graph sequences:

| Model Architecture | Input Modality | Evaluation Protocol | Test Accuracy | Test AUROC |
| :--- | :--- | :--- | :--- | :--- |
| **Lightweight 3D CNN** | 4D BOLD Voxel Volumes | Single Split | **76.19%** | **0.7316** |
| **NeuroSTORM (Spatiotemporal)** | 4D BOLD Voxel Volumes | Single Split | **68.25%** | **0.6800** |
| **NeuroSTORM (Spatiotemporal)** | 4D BOLD Voxel Volumes | 5-Fold Cross-Validation | **59.10%** | **0.5880** |
| **Temporal GNN** | Windowed CC200 Graphs | Test Split | **54.43%** | **0.5534** |

---

## 10. Experiment 09 — Out-of-Distribution LOSO Benchmark

### Preliminary GNN Node Feature Representation Selection
Historical exploratory comparison across 497 subjects:
- Raw Connectivity Features ($d=190$): **0.5497 AUC**
- Identity Matrix ($d=190$): **0.5350 AUC**
- Degree Strength ($d=1$): **0.5227 AUC**

### Main GNN LOSO Benchmark (7 Folds, $N=497$, `results/exp09/gnn_loso_summary_statistics.csv`)
Evaluated using top-10% positive FC thresholding (mean density 0.10003, with self-loops; GCN consumes `edge_weight`, GAT/SAGE/GIN use unweighted `edge_index`):

| Architecture | Mean AUC | 95% CI (Student-$t$, $\text{df}=6$) | Balanced Accuracy | Std AUC |
| :--- | :--- | :--- | :--- | :--- |
| **GAT** | **0.5752** | $[0.5321, 0.6183]$ | 0.5490 | 0.0466 |
| **GraphSAGE** | **0.5502** | $[0.5088, 0.5916]$ | 0.5147 | 0.0448 |
| **GCN** | **0.5468** | $[0.4790, 0.6147]$ | 0.5407 | 0.0733 |
| **GIN** | **0.5437** | $[0.4852, 0.6023]$ | 0.5313 | 0.0633 |

### Classical ML LOSO Benchmark (42 Models across 6 Representations)
Winning model family per representation:

| Feature Family | Best Model | Input Dimension ($d$) | Mean AUC | 95% CI ($\text{df}=6$) | Balanced Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Graph + Phenotype** | Linear SVM | 19 | **0.6493** | $[0.5721, 0.7265]$ | 0.6120 |
| **FC + Phenotype** | Extra Trees | 17,958 | **0.6328** | $[0.5489, 0.7168]$ | 0.5984 |
| **FC Only** | Logistic Regression | 17,955 | **0.6141** | $[0.5312, 0.6970]$ | 0.5741 |
| **Phenotype Only** | ElasticNet | 3 | **0.5935** | $[0.5180, 0.6690]$ | 0.5512 |
| **Phenotype (No IQ)**| ElasticNet | 3 | **0.5935** | $[0.5180, 0.6690]$ | 0.5512 |
| **Graph Only** | Random Forest | 16 | **0.5693** | $[0.4912, 0.6473]$ | 0.5380 |

---

## 11. Statistical Analysis

### Confidence Interval Methodology
Confidence intervals are constructed using the Student-$t$ distribution with $\text{df} = K - 1 = 6$ across the 7 site folds ($t_{0.975, 6} = 2.4469$):
$$\text{CI}_{95\%} = \bar{x} \pm 2.4469 \times \frac{s}{\sqrt{7}}$$

### Pairwise Non-Parametric Tests (`results/exp09/gnn_loso_pairwise_tests.csv`)
Wilcoxon signed-rank tests across the 7 site folds:
- GCN vs GAT: $W = 5.0, p = 0.15625$ (Difference is not statistically significant)
- GAT vs SAGE: $W = 4.0, p = 0.109375$ (Difference is not statistically significant)
- GAT vs GIN: $W = 7.0, p = 0.296875$ (Difference is not statistically significant)
- SAGE vs GIN: $W = 11.0, p = 0.687500$ (Difference is not statistically significant)
- GCN vs SAGE: $W = 13.0, p = 0.937500$ (Difference is not statistically significant)
- GCN vs GIN: $W = 13.0, p = 0.937500$ (Difference is not statistically significant)

---

## 12. Interpretation Boundaries

To ensure scientific integrity, the reported findings must be interpreted within explicit empirical boundaries:

### What the Results Demonstrate
1. Dynamic FC exhibits structured temporal continuity that degrades smoothly over time lags.
2. ComBat harmonization effectively mitigates multi-site scanner variance without distorting underlying network metrics.
3. The highest observed classical ML configuration had a higher mean LOSO AUC than the evaluated GNN configurations; the retained pairwise GNN tests did not identify statistically significant differences between the GNN architectures.

### What the Results Do Not Demonstrate
1. **Quantum Expressive Capacity**: The evaluated QGCNN produced lower test-set AUC (0.6429 vs 0.7293) than the matched classical GCN. However, the retained experiment does not isolate whether the performance gap stems from quantum circuit design, the $117 \to 12$ classical linear bottleneck, the 6-dimensional angle embedding, or optimization dynamics.
2. **GNN Superiority**: GNN architectures do not demonstrate superior generalizability over classical linear or tree-based baselines under cross-site clinical transfer.
3. **Clinical Diagnostic Biomarkers**: Due to substantial scanner-related variance and moderate out-of-distribution performance (AUC ~ 0.58–0.65), these models should not be interpreted as validated clinical diagnostic tools.

