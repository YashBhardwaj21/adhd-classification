# Experiment Notes (Exp 01 – Exp 09)

This document provides a concise reference for the nine experiments comprising the ADHD-200 connectomics study, summarizing the inputs, protocol, critical hyperparameters, reported outputs, and documented discrepancies for each experiment.

---

## Experiment 01: Dynamic Functional Connectivity & Temporal Stability

- **Track**: Track A (CC200 Atlas, 190 active ROIs)
- **Input**: Parcellated resting-state BOLD fMRI ROI time series (534 subjects across 8 scanner sites).
- **Protocol**:
  - Sliding-window functional connectivity (dFC) estimation using Pearson correlation.
  - Window length: $W = 30$ volumes ($T = 60\text{ s}$ at $\text{TR}=2.0\text{ s}$).
  - Step size: $S = 5$ volumes ($10\text{ s}$ shift).
  - Total windows extracted: 31,060 sliding-window connectivity matrices.
  - Temporal stability evaluation: Pearson correlation and Frobenius distance computed across consecutive sliding windows at lags $k \in \{1, 2, 3, 4\}$.
- **Important Parameters**:
  - Parcellation: Craddock-200 (CC200, 190 active regions after removing 10 empty/ventricular clusters).
  - Window size: $W = 30$, Step: $S = 5$.
- **Reported Output**:
  - Monotonic correlation decay across lags: Lag 1 ($r = 0.8990$), Lag 2 ($r = 0.7789$), Lag 3 ($r = 0.6609$), Lag 4 ($r = 0.5467$).
  - Monotonic Frobenius distance increase: Lag 1 ($32.58$), Lag 2 ($48.75$), Lag 3 ($60.78$), Lag 4 ($70.61$).
  - Stored in `results/exp01/dynamic_temporal_validation.csv` and `results/exp01/static_vs_dynamic_validation.csv`.
- **Limitations & Discrepancies**:
  - Executed code set self-correlations (diagonal elements) to zero before computing distance metrics.
  - See `notebooks/exp01/01_fc_generation_and_validation.ipynb`.

---

## Experiment 02: Dual-Constraint Graph Construction & Signed Null Models

- **Track**: Track A (CC200 Atlas, 190 active ROIs)
- **Input**: 31,060 sliding-window correlation matrices from Exp 01.
- **Protocol**:
  - Dual-constraint network topology construction:
    1. Minimum Spanning Tree (MST) on distance metric $D_{ij} = 1 - |r_{ij}|$ to guarantee global network connectedness.
    2. Proportional Thresholding (PT) adding the strongest remaining absolute correlation edges up to a fixed density $\rho = 0.20$ ($E = 3,591$ edges).
  - Signed network evaluation: positive edges retain positive weights, negative edges retain negative weights.
  - Null model generation: weight-conserving topological randomization using Rubinov & Sporns (2011) signed null models (100 randomized realizations per window).
- **Important Parameters**:
  - Atlas: CC200 ($N = 190$ nodes).
  - Density: $\rho = 0.20$ (target edge count $E = 190 \times 189 \times 0.20 / 2 = 3,591$ edges).
  - Distance metric for MST: $1 - |r|$.
- **Reported Output**:
  - Normalized clustering coefficient $\gamma = 1.6373 \pm 0.0381$.
  - Normalized characteristic path length $\lambda = 1.6160 \pm 0.0396$.
  - Small-worldness scalar $\sigma = \gamma / \lambda = 1.0132 \pm 0.0103$ across all 31,060 windows.
  - Stored in `results/exp02/graph_metrics.csv` and `results/exp02/subject_graph_metrics.csv`.
- **Limitations & Discrepancies**:
  - `notebooks/exp02/02_graph_construction_and_validation.ipynb` implements the full MST + PT algorithm (`mst_graph`).
  - `src/exp02/graph_utils.py` contains a simplified proportional thresholding utility without the MST initial pass.
  - See `notebooks/exp02/02_graph_construction_and_validation.ipynb` and `notebooks/exp02/03_graph_metrics_and_null_models.ipynb`.

---

## Experiment 03: Topological Feature Extraction & Cross-Site ANOVA

- **Track**: Track A (CC200 Atlas, 190 active ROIs)
- **Input**: Windowed topological metrics from Exp 02 across 534 subjects.
- **Protocol**:
  - Summary metric extraction per subject: mean, standard deviation, and dynamic range of nodal degree, global efficiency, characteristic path length, clustering coefficient, and modularity.
  - One-way Analysis of Variance (ANOVA) assessing variance attributable to scanner site ($8$ acquisition sites).
- **Important Parameters**:
  - Independent variable: Scanner site (Peking, NYU, KKI, OHSU, NeuroIMAGE, Pittsburgh, WashU, Brown).
  - Alpha threshold: $\alpha = 0.05$ with Bonferroni multiple testing correction.
- **Reported Output**:
  - Massive scanner batch effects observed across global network features:
    - Global efficiency across sites: $F = 4,609.76, p < 10^{-300}$.
    - Characteristic path length: $F = 3,674.34, p < 10^{-300}$.
  - Stored in `results/exp03/feature_statistics.csv` and `results/exp03/site_anova.csv`.
- **Limitations & Discrepancies**:
  - ANOVA $F$-statistics reflect inter-site scanner variation, **not** clinical diagnostic separation (ADHD vs TDC).
  - See `notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb`.

---

## Experiment 04: ComBat Multi-Site Harmonization

- **Track**: Track A (CC200 Atlas, 190 active ROIs)
- **Input**: Dynamic connectivity feature matrices (534 subjects across 8 scanner sites).
- **Protocol**:
  - Location and Scale (L/S) Empirical Bayes harmonization via `neuroCombat`.
  - Batch variable: Scanner acquisition site.
  - Biological covariates preserved: Diagnostic status (ADHD vs TDC), age, and sex.
  - Classification trade-off analysis: Random Forest and Logistic Regression classifiers trained to predict scanner site and diagnosis before vs after harmonization.
- **Important Parameters**:
  - Empirical Bayes estimation: parametric prior.
  - Covariates preserved: `diagnosis`, `age`, `gender`.
- **Reported Output**:
  - Scanner identification accuracy dropped significantly post-ComBat ($58.64\% \to 31.29\%$).
  - Diagnostic classification accuracy exhibited a minor attenuation ($64.53\% \to 59.56\%$).
  - Feature distributions: Raw clustering mean ($0.3333 \pm 0.0104$) vs ComBat adjusted ($0.3314 \pm 0.0094$).
  - Stored in `results/exp04/comparison_table.csv`, `results/exp04/site_prediction_results.csv`, and `results/exp04/diagnosis_prediction_results.csv`.
- **Limitations & Discrepancies**:
  - The drop in diagnostic classification illustrates that site-correlated clinical variance is partially removed alongside scanner noise.
  - See `notebooks/exp04/w2c_athena_2.ipynb`.

---

## Experiment 05: Dynamic Brain Micro-State Clustering

- **Track**: Track A (CC200 Atlas, 190 active ROIs)
- **Input**: Windowed connectivity vectors across all subjects ($N = 31,060$ window observations).
- **Protocol**:
  - Dimensionality reduction: subject-exemplar window selection via local variance peaks.
  - Unsupervised clustering: $K$-Means clustering evaluating $K \in \{2, 3, 4, 5, 6\}$ (optimal $K=3$ selected by silhouette analysis).
  - State dynamics: Markovian transition probability matrices, mean dwell time (consecutive windows in a state), and fractional occupancy.
- **Important Parameters**:
  - Optimal cluster count: $K = 3$.
  - Distance metric: Manhattan / Euclidean distance on upper-triangular FC vectors.
- **Reported Output**:
  - State 0 (modular/baseline state) dominates dwell time: $6.24$ windows ($49.05\%$ fractional occupancy).
  - State 1 (hyper-connected state): $3.82$ windows ($28.14\%$ occupancy).
  - State 2 (hypo-connected state): $3.15$ windows ($22.81\%$ occupancy).
  - Stored in `results/exp05/run_dynamic_biomarkers.csv` and `results/exp05/run_state_sequences.csv`.
- **Limitations & Discrepancies**:
  - State dwell times reflect window overlaps ($W=30, S=5$); individual transitions occur at 5-frame resolution.
  - See `notebooks/exp05/dynamic_transformer.ipynb`.

---

## Experiment 06: Semi-Supervised Pseudo-Labeling

- **Track**: Track B (AAL-116 Atlas, 116 ROIs)
- **Input**: AAL-116 static FC features ($N = 955$ total subjects: 391 clean-labelled, 564 unlabelled).
- **Protocol**:
  - Self-training pseudo-labelling using ensemble confidence scoring across classical baselines (SVM, Random Forest, Logistic Regression).
  - Iterative pseudo-label assignment based on posterior probability thresholds ($p > 0.85$ for confident class assignment).
  - Two procedural iterations evaluated: Procedure I and Procedure II.
- **Important Parameters**:
  - Atlas: AAL-116 ($116 \times 115 / 2 = 6,670$ upper-triangular FC features).
  - Clean cohort: $391$ subjects; Unlabelled cohort: $564$ subjects.
- **Reported Output**:
  - Procedure I assigned 552 pseudo-labels (420 healthy TDC, 132 ADHD).
  - Procedure II assigned 484 pseudo-labels (368 healthy TDC, 116 ADHD).
  - Stored in `results/exp06/exp06_verified_results.json`.
- **Limitations & Discrepancies**:
  - Cohort partitions in Exp 06 (391 clean + 564 unlabelled) are historically independent of Exp 07 (162 clean + 713 pseudo-labelled).
  - See `notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb`.

---

## Experiment 07: Classical GCN vs Hybrid Quantum QGCNN

- **Track**: Track B (AAL-116 Atlas, 116 ROIs)
- **Input**: Historical combined FC arrays (`X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy`) across 875 total subjects (162 clean, 713 pseudo-labelled).
- **Protocol**:
  - Clean cohort partition: 162 clean subjects split into 103 training, 26 validation, and 33 held-out test subjects.
  - Pseudo-labelled cohort: 713 pseudo-labelled subjects allocated strictly to training.
  - Graph construction: AAL-116 connectomes, top 15% absolute FC edge selection, signed correlation weights, 117 node features (116 FC correlations + 1 normalized degree).
  - Classical baseline: 3-layer GCN ($117 \to 32 \to 32 \to 16 \to 2$, 5,522 parameters) with BatchNorm1d, Dropout(0.30), and global mean pooling.
  - Hybrid Quantum QGCNN: Classical projection ($117 \to 12$), 6-qubit quantum variational circuit ($R_Y, R_Z$ angle encoding; 1 trainable layer with sequential $R_X, R_Y, R_Z$ rotations and ring CNOT; 6 Pauli-$Z$ expectations), followed by 3-layer GCN ($6 \to 16 \to 16 \to 16 \to 2$, 2,188 parameters).
- **Important Parameters**:
  - Atlas: AAL-116 ($N=116$, $D=117$ node features).
  - Graph density: $\rho = 0.15$ (top 15% absolute FC, signed weights).
  - Quantum qubits: 6, Quantum layers: 1.
  - Batch size: 8, Epochs: 20, Learning rate: $10^{-3}$, Weight decay: $10^{-5}$, Early stopping patience: 10.
  - Seed: 42 (split permutation).
- **Reported Output**:
  - Held-out test set performance ($N=33$ clean test subjects: 19 TDC, 14 ADHD):
    - Classical GCN: AUC = 0.7293, Accuracy = 0.6970, Precision = 0.6941, Recall = 0.6970, F1 = 0.6929.
    - Quantum QGCNN: AUC = 0.6429, Accuracy = 0.6061, Precision = 0.6204, Recall = 0.6061, F1 = 0.6082.
  - Precision, recall, and F1 scores reflect the weighted-average convention recorded in the result artifact.
  - Stored in `results/exp07/checkpoint_analysis.json` and `results/exp07/report.md`.
- **Limitations & Discrepancies**:
  - Historical input arrays (`X_combined_*.npy`) and model checkpoints are unavailable in this repository; reported outputs are archived.
  - The historical code constructed the 117th node feature using normalized thresholded-edge degree, whereas the paper describes weighted node strength.
  - See `configs/exp07/reported_run.json` and `docs/provenance.md`.

---

## Experiment 08: Volumetric 3D CNN, NeuroSTORM & Spatio-Temporal Baselines

- **Track**: Track B (Volumetric 4D fMRI & CC200 Atlas)
- **Input**: Preprocessed 4D fMRI volume sequences ($T=25, 99 \times 117 \times 95$) and windowed graph sequences across 764 subjects.
- **Protocol**:
  - Lightweight 3D CNN: Multi-layer 3D convolutions with global average pooling evaluated on temporal functional fMRI volumes.
  - NeuroSTORM: 4D spatio-temporal Swin Transformer backbone (adapted from SwinUNETR / SwiFT) evaluated on functional sequences.
  - Temporal GNN: Sequential GCN + GRU processing windowed connectomes across strictly disjoint subject partitions (534 train, 115 validation, 115 test).
- **Important Parameters**:
  - 3D CNN input: 4D functional volume sequence slices.
  - Temporal GNN: Disjoint subject-level partitions (no window leakage).
- **Reported Output**:
  - Lightweight 3D CNN: Accuracy = $76.19\%$, AUC = $0.7316$.
  - NeuroSTORM: 5-fold cross-validation accuracy = $59.10\%$.
  - Temporal GNN: Test accuracy = $61.74\%$, Test AUC = $0.6231$.
  - Stored in `results/exp08/exp08_verified_results.json`.
- **Limitations & Discrepancies**:
  - 3D CNN operates on functional BOLD volume sequences, not structural T1 anatomical scans.
  - Raw 4D volumes ($>120\text{ GB}$) exceed repository storage quotas and are not tracked in Git.
  - See `notebooks/exp08/neuro.ipynb`, `true_neuro.ipynb`, and `09_temporal_graph_learning.ipynb`.

---

## Experiment 09: Leave-One-Site-Out (LOSO) Population Graph Learning

- **Track**: Track C (CC200 Atlas, 190 active ROIs)
- **Input**: Functional connectomes and demographic metadata across 497 subjects from 7 scanner sites (Brown, KKI, NeuroIMAGE, NYU, OHSU, Peking, Pittsburgh).
- **Protocol**:
  - 7-fold Leave-One-Site-Out (LOSO) cross-validation: in each fold, all subjects from one site are held out for out-of-distribution evaluation while models train on the remaining 6 sites.
  - Four graph neural network architectures evaluated: GCN, GAT (Graph Attention Network), GraphSAGE, and GIN (Graph Isomorphism Network).
  - Subject graph preprocessing: top 10% absolute FC edge selection with self-loops and signed correlation weights.
- **Important Parameters**:
  - Atlas: CC200 ($190$ nodes).
  - Edge selection: top 10% absolute FC thresholding with self-loops.
  - 7 site folds, $N = 497$ total subjects.
- **Reported Output**:
  - Generalization metrics across unseen sites (7-fold mean):
    - GAT: AUC = $0.5752 \pm 0.081$, Balanced Accuracy = $0.5601$, F1 = $0.4789$.
    - SAGE: AUC = $0.5502 \pm 0.074$, Balanced Accuracy = $0.5368$, F1 = $0.4578$.
    - GCN: AUC = $0.5468 \pm 0.069$, Balanced Accuracy = $0.5372$, F1 = $0.4688$.
    - GIN: AUC = $0.5437 \pm 0.082$, Balanced Accuracy = $0.5309$, F1 = $0.4529$.
  - Stored in `results/exp09/w2b_loso_results.csv`, `graph_preprocessing_summary.csv`, and `exp09_verified_results.json`.
- **Limitations & Discrepancies**:
  - The executed notebook used `top_10pct` thresholding with weighted edges and self-loops, whereas the paper describes an unweighted MST + 20% distance mapping.
  - See `notebooks/exp09/11_population_graph_learning.ipynb` and `docs/provenance.md`.
