# Detailed Methodology & Experimental Protocol

This document provides the definitive technical specification of the algorithms, mathematical formulations, graph construction pipelines, and training protocols across all nine experiments.

---

## 1. Overall Methodological Framework

The study adopts a multi-track experimental architecture designed to address distinct neuroscientific and computational questions:
- **Track A (Connectomics & Dynamic Graph Characterization)**: Focuses on the temporal stability of sliding-window functional connectivity, minimum spanning tree (MST) topological graph representation, site-effect harmonization via ComBat, and unsupervised micro-state discovery using the Craddock-200 (CC200) atlas.
- **Track B (Semi-Supervised & Quantum Graph Learning)**: Formulates semi-supervised pseudo-labeling on the AAL-116 atlas, benchmarks a hybrid parameterized 6-qubit Quantum GCNN against an isotropic Classical GCN under matched conditions, and compares these with volumetric 3D CNNs and temporal graph networks.
- **Track C (Out-of-Distribution Generalization)**: Evaluates diagnostic generalizability under strict 7-fold Leave-One-Site-Out (LOSO) cross-validation across 497 subjects on the CC200 atlas, contrasting 42 classical ML baselines with 4 geometric deep learning architectures.

---

## 2. Dataset and Preprocessing

All neuroimaging data originate from the **ADHD-200 Sample**:
- **Athena Pipeline Preprocessing**: Raw resting-state fMRI scans underwent slice timing correction, motion realignment, spatial normalization to the Montreal Neurological Institute (MNI152) template (4 mm isotropic resolution), band-pass filtering (0.009–0.08 Hz), and nuisance signal regression (white matter, CSF, and 6 rigid-body motion parameters).
- **Parcellation Atlases**:
  - **Craddock-200 (CC200)**: Spatially contiguous spectral clustering atlas yielding 190 active cortical and subcortical regions of interest (ROIs) with complete coverage.
  - **Automated Anatomical Labeling (AAL-116)**: Anatomical atlas parcellating the brain into 116 cortical, subcortical, and cerebellar volumes.

---

## 3. Cohort Construction

The study examines strictly defined subsets to address track-specific goals:
1. **Track A (Exp 01–02)**: 764 subjects across 9 sites (1,193 acquisitions, 31,060 sliding windows).
2. **Track A Phenotypic Subset (Exp 03–05)**: 534 subjects with complete behavioral, phenotypic, and diagnostic records.
3. **Track B Exp 06**: 955 connectivity-available subjects (391 aligned clean-labeled subjects + 564 unlabeled candidate subjects).
4. **Track B Exp 07**: 162 clean-labeled subjects ($162 \subset 391$ aligned cohort) split into 103 train, 26 validation, and 33 held-out test subjects, combined with 713 downstream pseudo-labeled samples added strictly to the training partition (816 training subjects, 875 accounted population).
5. **Track B Exp 08**: 626 subjects for 4D volumetric CNNs ($99 \times 117 \times 95 \times 25$ volumes) and 764 subjects for temporal models.
6. **Track C Exp 09**: 497 subjects across 7 international scanner sites (KKI, NYU, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3) with static CC200 functional connectivity.

---

## 4. Leakage Control

Methodological integrity is safeguarded by strict partition protocols:
- **No Temporal Cross-Contamination**: In Track A sliding-window analyses, all temporal windows from an individual subject are grouped into the same partition.
- **Isolated Test Cohort**: In Exp 07, the 33-subject test set is held out prior to pseudo-labeling; pseudo-labeled samples are injected solely into the training set.
- **Strict Site-Level Isolation**: In Exp 09, 7-fold Leave-One-Site-Out (LOSO) ensures that no subjects or scanner-specific parameters from the held-out test site enter training or validation folds.

---

## 5. Experiment 01 — Dynamic Functional Connectivity

### Objective
Quantify temporal autocorrelation decay, stability, and topological variability of sliding-window dynamic functional connectivity (dFC) matrices.

### Cohort
764 unique subjects, 1,193 imaging acquisitions, 31,060 temporal windows across 9 acquisition centers.

### Input
Athena-preprocessed CC200 ROI time series ($T$ time points, $N=190$ ROIs).

### FC Estimation
Sliding-window Pearson correlation matrices computed with a rectangular window of length $W = 30$ TRs and stride $s = 5$ TRs (83.3% overlap, 25 shared time points):
$$r_{ij}(t) = \frac{\sum_{\tau=1}^{W} (x_i(t+\tau) - \bar{x}_i)(x_j(t+\tau) - \bar{x}_j)}{\sqrt{\sum_{\tau=1}^{W} (x_i(t+\tau) - \bar{x}_i)^2 \sum_{\tau=1}^{W} (x_j(t+\tau) - \bar{x}_j)^2}}$$

### Temporal Similarity
Pearson correlation between vectorised upper-triangle FC matrices at time $t$ and $t + \text{lag}$ ($\text{lag} \in \{1, 2, 3, 4\}$ windows):
$$\text{Sim}(t, t+\text{lag}) = \text{corr}(\text{vech}(FC_t), \text{vech}(FC_{t+\text{lag}}))$$
Frobenius norm distance progression:
$$D_F(t, t+\text{lag}) = \|FC_t - FC_{t+\text{lag}}\|_F$$

### Edge-Level Variability
Temporal standard deviation of connectivity strength across sliding windows for each of the 17,955 unique functional edges.

### Output Artifacts
- Primary notebook/source: [`notebooks/exp01/dynamic_fc_temporal_validation.ipynb`](../notebooks/exp01/dynamic_fc_temporal_validation.ipynb)
- Primary configuration: N/A (Standardized pipeline parameters: $W=30$, stride=5)
- Primary result artifact: [`results/exp01/temporal_similarity_validation.csv`](../results/exp01/temporal_similarity_validation.csv), [`results/exp01/static_dynamic_fc_comparison.csv`](../results/exp01/static_dynamic_fc_comparison.csv)
- Reproducibility status: Category A (Rerunnable from raw CC200 time series)

---

## 6. Experiment 02 — Graph Construction and Null-Model Evaluation

### Graph Construction
Construct sparse functional brain graphs from dense $190 \times 190$ correlation matrices using Minimum Spanning Tree combined with Proportional Thresholding (MST+PT).

### Distance Transformation
To ensure connectivity across positive and negative correlations while prioritizing strong functional couplings, distances are defined as:
$$d_{ij} = 1 - |r_{ij}|$$

### MST (Minimum Spanning Tree)
Kruskal's algorithm extracts the tree spanning all 190 nodes that minimizes total distance $\sum d_{ij}$, guaranteeing total network connectedness with exactly $N - 1 = 189$ edges without arbitrary disconnects.

### Proportional Threshold
Additional edges with the highest absolute Pearson correlation $|r_{ij}|$ are iteratively added until the target density of $\rho = 0.20$ is attained, fixing total edges at $E = \text{round}(0.20 \times \frac{190 \times 189}{2}) = 3,591$ edges.

### Signed Edge Attributes
Edges retain their original signed correlation value $r_{ij} \in [-1, 1]$ stored as `edge_attr`.

### Null Models
Degree- and weight-conserving signed network randomization adapted from the Brain Connectivity Toolbox (Rubinov & Sporns, 2011). Positive and negative edge weights and nodal strength sequences are preserved while topology is randomized.

### Rewiring Validation
Iterative edge swaps ($I = 5 \times E$) implemented via Numba-accelerated edge rewiring ([`src/exp02/signed_edge_rewiring.py`](../src/exp02/signed_edge_rewiring.py)) to verify that observed network metrics deviate significantly from null expectations.

### Output Artifacts
- Primary notebook/source: [`notebooks/exp02/mst_proportional_graph_construction.ipynb`](../notebooks/exp02/mst_proportional_graph_construction.ipynb), [`notebooks/exp02/graph_metrics_null_model_validation.ipynb`](../notebooks/exp02/graph_metrics_null_model_validation.ipynb)
- Primary configuration: [`configs/exp02/graph_configuration.json`](../configs/exp02/graph_configuration.json)
- Primary result artifact: [`results/exp02/graph_metrics_window_level.csv`](../results/exp02/graph_metrics_window_level.csv), [`results/exp02/graph_construction_configuration.csv`](../results/exp02/graph_construction_configuration.csv)
- Reproducibility status: Category A (Rerunnable from CC200 correlation matrices)

---

## 7. Experiment 03 — Graph-Theoretic Characterization

### Graph Metrics
Per-window extraction of global network properties:
- Characteristic path length ($L$)
- Clustering coefficient ($C$)
- Global efficiency ($E_g$)
- Local efficiency ($E_{loc}$)
- Modularity ($Q$, Louvain community detection)
- Small-worldness ($\sigma = \frac{C / C_{null}}{L / L_{null}}$)

### Null-Normalized Metrics
Each topological metric is normalized against an ensemble of 100 matched signed null graphs generated via degree-conserving edge rewiring.

### Temporal Descriptors
For each subject and metric, summary statistics are extracted across sliding windows:
- Temporal Mean ($\mu_M$)
- Temporal Dispersion (Standard Deviation $\sigma_M$, Interquartile Range $\text{IQR}_M$)

### Mean vs Dispersion Testing
Testing whether temporal dispersion ($\sigma_M$) exhibits greater diagnostic sensitivity than time-averaged static means ($\mu_M$).

### Welch Tests & BH-FDR
Two-sample Welch's unequal variances $t$-tests comparing TDC vs ADHD cohorts, with Benjamini-Hochberg False Discovery Rate ($\text{FDR} \le 0.05$) correction across multiple comparisons.

### Output Artifacts
- Primary notebook/source: [`notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb`](../notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb)
- Primary configuration: N/A
- Primary result artifact: [`results/exp03/dynamic_feature_statistics.csv`](../results/exp03/dynamic_feature_statistics.csv), [`results/exp03/site_effect_anova.csv`](../results/exp03/site_effect_anova.csv)
- Note on $p$-values: In `site_effect_anova.csv`, $p$-values exceeding floating-point double precision underflow to `0.0`, reflecting extreme statistical significance ($p < 10^{-15}$).
- Reproducibility status: Category A (Rerunnable from windowed graph metrics)

---

## 8. Experiment 04 — Site Effects and ComBat Harmonization

### Site-Effect Testing
One-way analysis of variance (ANOVA) across the 8 acquisition sites testing for scanner-induced variance in topological graph metrics across the phenotypically complete subset.

### Predictive Diagnostics
Random Forest classification was used as the site-prediction diagnostic to quantify scanner bias before and after harmonization.

### ComBat Harmonization
Empirical Bayes location-and-scale harmonization (Fortin et al., 2018) adjusting for additive and multiplicative site batch effects while preserving biological covariates (age, sex, diagnostic status):
$$y_{ijg} = \alpha_g + X_{ij}\beta_g + \gamma_{ig} + \delta_{ig}\epsilon_{ijg}$$

### Descriptive & Predictive Analysis
Re-evaluation of clustering coefficients, path lengths, site prediction accuracy (58.64% raw $\to$ 31.29% ComBat), and diagnostic classification before and after harmonization.

### Output Artifacts
- Primary notebook/source: [`notebooks/exp04/combat_site_effects_harmonization.ipynb`](../notebooks/exp04/combat_site_effects_harmonization.ipynb)
- Primary configuration: N/A
- Primary result artifact: [`results/exp04/combat_harmonization_comparison.csv`](../results/exp04/combat_harmonization_comparison.csv), [`results/exp04/site_prediction_combat_comparison.csv`](../results/exp04/site_prediction_combat_comparison.csv), [`results/exp04/diagnosis_prediction_combat_comparison.csv`](../results/exp04/diagnosis_prediction_combat_comparison.csv)
- Reproducibility status: Category A (Rerunnable from extracted graph metrics)

---

## 9. Experiment 05 — Dynamic State Discovery

### K-Means Clustering
Unsupervised discovery of recurring functional connectivity configurations across all sliding-window correlation matrices using $k$-means clustering with correlation distance ($1 - \text{corr}$).

### K Selection & MeanRank
Candidate cluster counts $K \in \{2, 3, \dots, 10\}$ evaluated using the MeanRank composite criterion aggregating three internal cluster validity indices:
- Silhouette Coefficient
- Calinski-Harabasz Index
- Davies-Bouldin Index

Selected $K = 3$ based on optimal composite rank (MeanRank = 2.333). Robustness confirmed through 30-run Adjusted Rand Index (ARI) stability analysis (Mean ARI: $0.9950 \pm 0.0024$).

### Stability & State-Sequence Features
Cluster centroids identify $K=3$ discrete connectivity micro-states. Each acquisition is mapped to a discrete state sequence from which temporal biomarkers are extracted:
- Fractional Occupancy (proportion of time spent in state $k$: State 0 = 49.05%, State 1 = 27.32%, State 2 = 23.63%)
- Mean Dwell Time (consecutive windows spent in state $k$: State 0 = 6.24, State 1 = 3.44, State 2 = 2.83 windows)
- Transition Probability Matrix ($P_{ij}$, probability of switching from state $i$ to state $j$)

### Output Artifacts
- Primary notebook/source: [`notebooks/exp05/dynamic_state_discovery.ipynb`](../notebooks/exp05/dynamic_state_discovery.ipynb) (implements CC200 $K=2\dots 10$ evaluation, 30-run ARI stability, $K=3$ Lloyd clustering, state sequence generation, and dynamic biomarker extraction).
- Primary configuration: N/A
- Primary result artifact: [`results/exp05/dynamic_state_biomarkers.csv`](../results/exp05/dynamic_state_biomarkers.csv), [`results/exp05/dynamic_state_transitions.csv`](../results/exp05/dynamic_state_transitions.csv), [`results/exp05/dynamic_state_sequences.csv`](../results/exp05/dynamic_state_sequences.csv)
- Column terminology note: In `dynamic_state_biomarkers.csv` and `dynamic_state_sequences.csv`, the column labeled `label_binary` historically stores the 4-class diagnosis code ($0=\text{TDC}$, $1=\text{ADHD-Combined}$, $2=\text{ADHD-Hyperactive}$, $3=\text{ADHD-Inattentive}$).
- Reproducibility status: Category A (Rerunnable from sliding-window FC metrics)

---

## 10. Experiment 06 — Semi-Supervised Pseudo-Labeling

### Clean and Unlabeled Cohorts
Static functional connectivity vectors ($d = 6,670$ upper-triangle edges) on the AAL-116 atlas for 955 subjects: 391 clean-labeled subjects and 564 candidate unlabeled subjects.

### Procedure I (Self-Training with Thresholding)
Base class-weighted Logistic Regression classifier trained on 391 clean subjects predicts class probabilities on unlabeled data. Samples exceeding a posterior probability confidence threshold of $\tau = 0.75$ are iteratively assigned pseudo-labels, yielding 552 pseudo-labeled samples.

### Procedure II (Multi-Model Ensemble Consensus)
Consensus voting across an ensemble of 4 heterogeneous classifiers (Random Forest, Extra Trees, Gradient Boosting, SVM). Only samples with unanimous model agreement and mean ensemble confidence $\bar{p} \ge 0.75$ receive pseudo-labels, yielding 484 pseudo-labeled samples.

### Holdout Evaluation
Evaluation on an independent 79-subject holdout partition comparing clean-only training (67.09% accuracy) against pseudo-label augmented training (72.15% accuracy).

### Output Artifacts
- Primary notebook/source: [`notebooks/exp06/semi_supervised_pseudolabeling.ipynb`](../notebooks/exp06/semi_supervised_pseudolabeling.ipynb)
- Primary configuration: N/A
- Primary result artifact: [`results/exp06/exp06_verified_results.json`](../results/exp06/exp06_verified_results.json)
- Reproducibility status: Category B (Requires unbundled AAL-116 FC feature arrays)

---

## 11. Experiment 07 — Classical GCN and QGCNN

### Cohort Lineage & AAL-116 Representation
162 clean-labeled subjects (an exact prefix subset $162 \subset 391$ of the aligned cohort) combined with 713 separate downstream pseudo-labeled samples. Split: 103 train clean + 713 pseudo = 816 train; 26 validation clean; 33 held-out test clean (`seed=42`).

### Graph Construction & Node Features
- **Graph Topology**: Sparsified AAL-116 correlation graphs retaining top 15% absolute correlation edges (nominal density 0.15).
- **Node Feature Dimension ($d=117$)**: Each of the 116 ROIs receives 116 signed Pearson correlation values plus 1 normalized unweighted degree feature ($\text{degree} / 116$).

### Message Passing Semantics
Message passing across graph convolution layers is strictly **unweighted**:
- The forward pass uses unweighted topological connectivity represented by `edge_index`.
- Stored correlation weights in `edge_attr` are not consumed as message-passing convolution weights.

### Classical GCN Architecture
3-layer Graph Convolutional Network ($117 \to 32 \to 32 \to 16 \to 2$) with BatchNorm1d, ReLU, Dropout(0.30), and global mean pooling (5,522 trainable parameters).

### Hybrid QGCNN Architecture
Parameterized quantum circuit implemented via PennyLane:
- Linear projection from graph pooling layer ($117 \to 12$).
- Angle embedding onto 6 qubits using $R_y$ rotations.
- 1 variational layer comprising 6 trainable parameter gates and Entangling CNOT gates (2,188 total model parameters).
- Pauli-$Z$ expectation values measured on each qubit and projected to 2 diagnostic logits.

### Training & Evaluation Protocol
Trained with Adam optimizer ($\text{lr}=10^{-3}$, weight decay=$10^{-5}$, batch size=8, epochs=20, patience=10) with early stopping based on validation AUC. Evaluated on the 33 held-out clean test subjects using AUC, Accuracy, Precision, Recall, and F1.

### Output Artifacts
- Primary notebook/source: [`src/exp07/classical_models/run_gcn_experiment.py`](../src/exp07/classical_models/run_gcn_experiment.py), [`src/exp07/quantum_models/run_qgcnn_experiment.py`](../src/exp07/quantum_models/run_qgcnn_experiment.py)
- Primary configuration: [`configs/exp07/experiment_configuration.json`](../configs/exp07/experiment_configuration.json)
- Primary result artifact: [`results/exp07/gcn_qgcnn_test_results.json`](../results/exp07/gcn_qgcnn_test_results.json), [`results/exp07/clean_cohort_lineage.csv`](../results/exp07/clean_cohort_lineage.csv)
- Reproducibility status: Category C (Archived benchmark; full training arrays not redistributed in-tree)

---

## 12. Experiment 08 — Volumetric and Temporal Baselines

### 3D CNN (Lightweight Volumetric Architecture)
3D spatial convolutional neural network processing 4D functional BOLD volumes ($99 \times 117 \times 95 \times 25$). Spatial 3D convolutions with residual downsampling and global average pooling directly classify voxel volumes.

### NeuroSTORM (Spatiotemporal Architecture)
Spatiotemporal deep learning architecture combining 3D spatial patch embeddings with temporal transformer attention across functional time frames. Evaluated under single split (68.25% accuracy) and 5-fold cross-validation (59.10% accuracy).

### Temporal Graph Network
Dynamic graph neural network incorporating windowed graph sequence inputs and GRU/LSTM recurrent message passing across consecutive temporal windows.

### Output Artifacts
- Primary notebook/source: [`notebooks/exp08/volumetric_3d_cnn.ipynb`](../notebooks/exp08/volumetric_3d_cnn.ipynb), [`notebooks/exp08/neurostorm_spatiotemporal_baseline.ipynb`](../notebooks/exp08/neurostorm_spatiotemporal_baseline.ipynb), [`notebooks/exp08/temporal_graph_learning.ipynb`](../notebooks/exp08/temporal_graph_learning.ipynb)
- Primary configuration: N/A
- Primary result artifact: [`results/exp08/baseline_model_results.json`](../results/exp08/baseline_model_results.json), [`results/exp08/neurostorm_evaluation_summary.png`](../results/exp08/neurostorm_evaluation_summary.png)
- Reproducibility status: Category B (Requires external 4D BOLD fMRI volume sequences)

---

## 13. Experiment 09 — LOSO Generalization Benchmark

### Cohort & Multi-Site Partitioning
497 subjects across 7 international scanner sites (KKI, NYU, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3) parcellated with CC200 (190 ROIs). Evaluated under strict 7-fold Leave-One-Site-Out (LOSO) cross-validation where each site serves once as the unseen test fold.

### Graph Construction & Edge Semantics
- **Threshold**: Top-10% positive FC percentile threshold applied to upper-triangle signed Pearson correlation values:
  $$\tau_{90} = \text{percentile}(\text{vech}(FC), 90)$$
  $$\text{edge}_{ij} = \mathbf{1}\{FC_{ij} \ge \tau_{90}\}$$
  Yields a mean graph density of $\rho = 0.10003$ with self-loops added to each node.
- **Edge Weight Semantics**: Signed Pearson correlations are stored in `edge_attr`. In the main LOSO benchmark:
  - **GCN** consumes these stored values as message-passing convolution weights (`edge_weight`).
  - **GAT, GraphSAGE, and GIN** pass unweighted graph topology represented by `edge_index` without consuming `edge_attr` as message-passing weights.

### Historical Threshold Evidence
The top-10% positive threshold was recovered from an exploratory 8-regime historical sweep (`none`, `top_5pct`, `top_10pct`, `top_15pct`, `top_20pct`, `abs_gt_0.2`, `abs_gt_0.25`, `abs_gt_0.3`) documented in [`results/exp09/threshold_sweep_graph_statistics.csv`](../results/exp09/threshold_sweep_graph_statistics.csv). The original producer code and formal decision rule were not retained; it is documented as a recovered historical operating point.

### GNN Architectures
Four standard geometric deep learning architectures evaluated with identical executed hyperparameters (hidden dim=128, 2 layers, dropout=0.5, batch size=64, epochs=200, patience=20, Adam lr=$10^{-3}$, weight decay=$5 \times 10^{-4}$, global mean pooling, GAT heads=4, BatchNorm enabled):
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT, 4 heads)
- Graph Sample and Aggregate (GraphSAGE, mean aggregation)
- Graph Isomorphism Network (GIN, MLP $\epsilon$-learnable)

### Classical ML Baselines
42 model configurations evaluating 7 classifiers (Logistic Regression, Ridge, ElasticNet, Linear SVM, RBF SVM, Random Forest, Extra Trees) across 6 feature representations:
1. `FC_only` ($d=17,955$)
2. `FC_Phenotype` ($d=17,958$)
3. `Graph_only` ($d=16$)
4. `Graph_Phenotype` ($d=19$)
5. `Phenotype_only` ($d=3$)
6. `Phenotype_NoIQ` ($d=3$)

### Nested Feature Selection & Statistical Analysis
- Feature selection (ANOVA $F$-percentile ranking) is strictly **nested** inside each training fold for classical models.
- GNN representation selection (raw connectivity vs identity vs degree) was conducted as an **exploratory non-nested** preliminary analysis.
- Confidence intervals are computed using the Student-$t$ distribution with $\text{df} = K - 1 = 6$:
  $$\text{CI}_{95\%} = \bar{x} \pm t_{0.975, 6} \times \frac{s}{\sqrt{7}}$$
- Pairwise architecture differences are assessed via non-parametric Wilcoxon signed-rank tests across the 7 site folds.

### Output Artifacts
- Primary notebook/source: [`notebooks/exp09/loso_gnn_population_graphs.ipynb`](../notebooks/exp09/loso_gnn_population_graphs.ipynb), [`notebooks/exp09/loso_classical_ml_baselines.ipynb`](../notebooks/exp09/loso_classical_ml_baselines.ipynb)
- Primary configuration: [`configs/exp09/gnn_configuration.json`](../configs/exp09/gnn_configuration.json), [`configs/exp09/graph_threshold_selection.json`](../configs/exp09/graph_threshold_selection.json)
- Primary result artifact: [`results/exp09/gnn_loso_fold_results.csv`](../results/exp09/gnn_loso_fold_results.csv), [`results/exp09/classical_ml_canonical_results.csv`](../results/exp09/classical_ml_canonical_results.csv), [`results/exp09/gnn_loso_summary_statistics.csv`](../results/exp09/gnn_loso_summary_statistics.csv)
- Reproducibility status: Category B (Requires excluded intermediates; full 7-fold LOSO evaluation tables verified; notebook end-to-end execution requires precomputed representation parquets)
