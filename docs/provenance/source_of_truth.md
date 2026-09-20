# Authoritative Source of Truth & Evidence Ledger

**Repository**: `ADHD200_GitHub_Staging`  
**Purpose**: Primary reference ledger mapping every experiment to its canonical notebook, source code, execution outputs, and staged result artifacts.  
**Integrity Rule**: Every entry distinguishes verified artifact data from exploratory notebook outputs, documents exact numerical values, notes discrepancies, and avoids over-claiming.

---

## Master Track Architecture

The repository comprises three distinct experimental tracks rather than a single linear pipeline:

1. **Track A (Experiments 1–5)**:
   - **Representation**: Craddock CC200 functional parcellation preprocessed via the Athena pipeline down to 190 valid regions.
   - **Cohort**: 764 unique subjects across 1,193 acquisitions from 9 scanning sites (`Brown`, `KKI`, `NYU`, `NeuroIMAGE`, `OHSU`, `Peking_1`, `Peking_2`, `Peking_3`, `WashU`).
   - **Methodological Scope**: Sliding-window dynamic functional connectivity ($W=30$, $S=5$ TRs; 31,060 windows), signed degree-preserving null models, temporal dispersion vs. mean statistical analysis, ComBat multi-site harmonization, and K-Means ($K=3$) dynamic brain state discovery.

2. **Track B (Experiments 6–8)**:
   - **Representation**: AAL-116 anatomical parcellation (116 regions / 117-dimensional node features including degree) and temporal 4D fMRI volume sequences.
   - **Methodological Scope**: Semi-supervised pseudo-labeling (Self-Training vs. 4-model ensemble), matched Classical GCN vs. 6-qubit Quantum GCNN on a separate staged cohort (162 clean, 713 pseudo-labeled), and deep baselines (Lightweight 3D CNN on temporal fMRI volumes, NeuroSTORM 4D Swin Transformer, and temporal graph learning).

3. **Track C (Experiment 9)**:
   - **Representation**: Transductive population graph connecting human subjects based on phenotypic and connectomic similarity.
   - **Cohort**: 497 subjects across 7 sites (`KKI`, `NYU`, `NeuroIMAGE`, `OHSU`, `Peking_1`, `Peking_2`, `Peking_3`).
   - **Methodological Scope**: Strict Leave-One-Site-Out (LOSO) 7-fold cross-validation evaluating GCN, GAT, GraphSAGE, and GIN.

---

## Experiment 1: Dynamic FC Generation & Temporal Autocorrelation Validation

- **Scientific Goal**: Verify whether sliding-window dynamic functional connectivity (dFC) matrices exhibit continuous temporal autocorrelation decaying with time lag, as opposed to stationary or white-noise sampling fluctuations.
- **Canonical Notebook**: [`notebooks/exp01/01_fc_generation_and_validation.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp01/01_fc_generation_and_validation.ipynb)
- **Primary Input**: Athena CC200 BOLD time-series ($T_i \times 190$).
- **Parameters**: Window length $W = 30$ TRs, Stride $S = 5$ TRs.
- **Staged Result Artifacts**:
  - [`results/exp01/dynamic_manifest.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp01/dynamic_manifest.csv): 1,193 acquisitions across 764 subjects from 9 sites.
  - [`results/exp01/dynamic_temporal_validation.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp01/dynamic_temporal_validation.csv): Per-subject temporal lag similarity and Frobenius distance for lags 1 to 4.
  - [`results/exp01/static_vs_dynamic_validation.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp01/static_vs_dynamic_validation.csv): 1,193 acquisitions comparing static FC against mean dynamic FC.
- **Audited Numerical Results** (Verified from Cell 53, lines 2273–2278 of canonical notebook and `dynamic_temporal_validation.csv`):
  - **Lag 1**: Similarity Mean = **0.899042 ± 0.016444**, Frobenius Distance Mean = **32.578362 ± 3.449146**
  - **Lag 2**: Similarity Mean = **0.778905 ± 0.032205**, Frobenius Distance Mean = **48.749290 ± 4.721958**
  - **Lag 3**: Similarity Mean = **0.660890 ± 0.045131**, Frobenius Distance Mean = **60.780884 ± 5.382166**
  - **Lag 4**: Similarity Mean = **0.546714 ± 0.057422**, Frobenius Distance Mean = **70.605774 ± 5.968088**
- **Verified Implementation Detail**:
  - `generate_dynamic_fc()` in Cell 9 leaves the matrix diagonal as 1.0 without zeroing self-correlations.
  - Static FC generation applies Fisher $z$-transform ($\operatorname{arctanh}$), while windowed dynamic FC in this notebook is stored as raw Pearson correlation coefficients in $[-1.0, 1.0]$.
- **Status**: **ARTIFACT-VERIFIED**.

---

## Experiment 2: Graph Construction, Graph Metrics & Null Models

- **Scientific Goal**: Construct stable, connected brain networks across dynamic windows and evaluate whether the empirical connectome exhibits small-world topological properties against degree-preserving null models.
- **Canonical Notebooks**:
  - [`notebooks/exp02/02_graph_construction_and_validation.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp02/02_graph_construction_and_validation.ipynb)
  - [`notebooks/exp02/03_graph_metrics_and_null_models.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp02/03_graph_metrics_and_null_models.ipynb)
- **Source Scripts**:
  - [`src/exp02/graph_utils.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/graph_utils.py): MST + Proportional Thresholding (PT) graph builder.
  - [`src/exp02/null_model_und_sign_fixed.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/null_model_und_sign_fixed.py): Rubinov-Sporns signed degree-preserving network rewiring.
  - [`src/exp02/randmio_und_signed_fast.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/randmio_und_signed_fast.py): Numba JIT-compiled inner loop for signed edge swapping.
  - [`src/exp02/null_model_und_sign_fast.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/null_model_und_sign_fast.py): **Dead code** (0-byte empty file, replaced by `null_model_und_sign_fixed.py`).
- **Configuration**:
  - [`configs/exp02/graph_config.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/configs/exp02/graph_config.json): density $\rho = 0.20$, nodes $N = 190$, edges $M = 3,591$, mean degree $\langle k \rangle = 37.8$.
- **Staged Result Artifacts**:
  - [`results/exp02/graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/graph_metrics.csv): Window-level graph metrics for 31,060 dynamic windows (`avg_degree`, `avg_strength`, `clustering`, `transitivity`, `global_efficiency`, `local_efficiency`, `path_length`).
  - [`results/exp02/subject_graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/subject_graph_metrics.csv): Subject-level summaries (mean, std, median, min, max) for 764 subjects.
  - [`results/exp02/site_graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/site_graph_metrics.csv): Site-level summaries across 9 scanning sites.
  - [`results/exp02/graph_construction_strategy.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/graph_construction_strategy.csv): Density sweep validating $\rho = 0.20$ as preserving connectivity without over-densification.
- **Provenance Clarification**:
  - `results/exp04/comparison_table.csv` is specifically the Exp 4 ComBat harmonization comparison table and must not be cited as the primary output of Exp 2.
- **Status**: **ARTIFACT-VERIFIED**.

---

## Experiment 3: Dynamic Graph Feature Extraction & Statistical Diagnosis Analysis

- **Scientific Goal**: Aggregate window-level graph topological metrics into subject-level temporal moment descriptors and analyze whether diagnostic divergence is concentrated in temporal dispersion rather than temporal averages.
- **Canonical Notebook**: [`notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb)
- **Supporting Notebook**: [`notebooks/exp04/w2c_athena_2.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp04/w2c_athena_2.ipynb) (Phase VII-C)
- **Input**: 31,060 dynamic window graphs from Exp 2.
- **Staged Result Artifacts**:
  - [`results/exp03/subject_graph_features.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp03/subject_graph_features.csv): 764 subjects with 42 temporal features (mean, std, var, range, CV, IQR across 7 graph metrics).
  - [`results/exp03/feature_statistics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp03/feature_statistics.csv): Window-level distribution summary across all 31,060 windows.
  - [`results/exp03/site_anova.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp03/site_anova.csv): Cross-site ANOVA statistics.
- **Audited Findings**:
  - In `w2c_athena_2.ipynb` Phase VII-C, temporal dispersion metrics (`range`, `std`, `cv`, `iqr`) show statistically significant group separation between ADHD and TDC ($p < 10^{-5}$ uncorrected), whereas temporal means show no significant difference ($p > 0.15$).
  - Large node-level parquet intermediates (143 MB) were excluded from staging due to repository size policies.
- **Status**: **PARTIALLY MAPPED AT STATISTIC LEVEL**. The qualitative finding is verified from code and notebook executions; individual paper-reported p-values must not be claimed as standalone verified artifacts unless directly present in staged CSVs.

---

## Experiment 4: Acquisition-Site Effects & ComBat Harmonization

- **Scientific Goal**: Quantify scanner-associated confounding variance across multi-center acquisitions and evaluate the impact of empirical Bayes ComBat harmonization on site classification, diagnosis classification, and graph topology preservation.
- **Canonical Notebook**: [`notebooks/exp04/w2c_athena_2.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp04/w2c_athena_2.ipynb)
- **Staged Result Artifacts**:
  - [`results/exp04/site_prediction_results.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp04/site_prediction_results.csv):
    - Raw Site Accuracy: **58.64%** (Balanced Acc: 50.19%, Macro F1: 50.61%)
    - ComBat Site Accuracy: **31.29%** (Balanced Acc: 23.56%, Macro F1: 24.37%)
  - [`results/exp04/diagnosis_prediction_results.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp04/diagnosis_prediction_results.csv):
    - Raw Diagnosis Accuracy: **64.53%** (Balanced Acc: 59.43%, Macro F1: 59.53%)
    - ComBat Diagnosis Accuracy: **59.56%** (Balanced Acc: 51.60%, Macro F1: 49.92%)
  - [`results/exp04/comparison_table.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp04/comparison_table.csv): Metric distributions before and after ComBat:
    - Clustering: Raw Mean $0.3333 \pm 0.0343$ $\to$ ComBat Mean $0.3334 \pm 0.0275$
    - Global Efficiency: Raw Mean $0.5820 \pm 0.0091$ $\to$ ComBat Mean $0.5821 \pm 0.0084$
    - Local Efficiency: Raw Mean $0.7225 \pm 0.0116$ $\to$ ComBat Mean $0.7224 \pm 0.0092$
    - Path Length: Raw Mean $0.5683 \pm 0.0580$ $\to$ ComBat Mean $0.5682 \pm 0.0299$
    - Gamma ($\gamma$): Raw Mean $1.0320 \pm 0.0028$ $\to$ ComBat Mean $1.0320 \pm 0.0022$
    - Lambda ($\lambda$): Raw Mean $1.0187 \pm 0.0044$ $\to$ ComBat Mean $1.0187 \pm 0.0042$
    - Sigma ($\sigma$): Raw Mean $1.0132 \pm 0.0047$ $\to$ ComBat Mean $1.0132 \pm 0.0039$
  - [`results/exp04/graph_topology_preservation.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp04/graph_topology_preservation.csv): Feature correlation before and after ComBat:
    - $\lambda$: 0.9137, $E_{glob}$: 0.8838, $\gamma$: 0.8348, $\sigma$: 0.8150, $C$: 0.7712, $E_{loc}$: 0.7659, $L$: 0.4579.
  - [`results/exp04/effect_size_before_after.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp04/effect_size_before_after.csv): Kruskal-Wallis site effect size reduction.
- **Scientific Interpretation**:
  - The drop in diagnosis prediction accuracy from 64.53% to 59.56% indicates sensitivity to site-associated variance in the unharmonized feature set. It demonstrates that scanner-associated variance and diagnostic variance are entangled in the raw multicenter cohort; it does not constitute proof that the removed variance was purely biological signal.
- **Status**: **ARTIFACT-VERIFIED**.

---

## Experiment 5: Dynamic Brain-State Discovery

- **Scientific Goal**: Discover discrete recurring whole-brain dynamic connectivity states via unsupervised K-Means clustering ($K=3$) and characterize state occupancy, mean dwell time, switching rates, and transition entropy.
- **Canonical Notebook**: [`notebooks/exp05/dynamic_transformer.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp05/dynamic_transformer.ipynb)
- **Staged Result Artifacts**:
  - [`results/exp05/subject_feature_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_feature_summary.csv) (764 subjects):
    - **State 0 Occupancy**: Mean = **0.4905 ± 0.3054** (Median 0.4878)
    - **State 1 Occupancy**: Mean = **0.2732 ± 0.2757** (Median 0.1951)
    - **State 2 Occupancy**: Mean = **0.2363 ± 0.1666** (Median 0.2174)
    - **State 0 Dwell Time**: Mean = **6.236 ± 6.465 windows** (Max 46.0)
    - **State 1 Dwell Time**: Mean = **3.436 ± 5.558 windows** (Max 33.0)
    - **State 2 Dwell Time**: Mean = **2.829 ± 2.453 windows** (Max 46.0)
    - **Switching Rate**: Mean = **0.2048 ± 0.1151** transitions/window (Max 0.60)
    - **Transition Entropy**: Mean = **0.6925 ± 0.3699** (Max 1.478)
  - [`results/exp05/run_transition_dynamics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_transition_dynamics.csv): 1,193 acquisitions with transition counts ($C_{00}$ to $C_{22}$) and transition probabilities ($P_{00}$ to $P_{22}$).
  - [`results/exp05/run_state_sequences.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_state_sequences.csv): Complete temporal state index sequences per acquisition.
  - [`results/exp05/run_dynamic_biomarkers.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_dynamic_biomarkers.csv): Per-run occupancy, dwell, and transition rate metrics.
  - [`results/exp05/subject_dynamic_dataset.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_dynamic_dataset.csv), [`results/exp05/subject_feature_correlation.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_feature_correlation.csv).
- **Notebook Implementation Note**:
  - `dynamic_transformer.ipynb` contains exploratory code investigating a 3-layer `DynamicFCTransformer` on voxel/ROI timeseries (5-fold CV: Mean Acc $0.585 \pm 0.014$, Mean AUROC $0.546 \pm 0.038$), which is distinct from the primary K=3 state clustering pipeline whose outputs are staged in `results/exp05/`.
- **Status**: **ARTIFACT-VERIFIED**.

---

## Experiment 6: Semi-Supervised Pseudo-Labeling

- **Scientific Goal**: Exploit unannotated neuroimaging scans in the ADHD-200 repository via semi-supervised self-training and multi-model consensus ensembles on AAL-116 extracted features.
- **Canonical Notebook**: [`notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb)
- **Data Cohort**: 955 total subjects:
  - Clean Labeled Pool: 391 subjects (Train: 273, Val: 59, Test: 59).
  - Unlabeled Pool: 564 subjects.
- **Staged Result Artifact**: [`results/exp06/exp06_verified_results.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp06/exp06_verified_results.json)
- **Audited Implementations**:
  - **Procedure I (Self-Training Classifier)**:
    - Base Classifier: `LogisticRegression(class_weight='balanced', C=1.0)`
    - Preprocessing: `SimpleImputer(strategy='median')` + `StandardScaler()` fit on train split only.
    - Self-training Parameters: threshold $\tau = 0.75$, $k_{best} = 10$, $\max\_iter = 10$.
    - Output: **552 newly accepted pseudo-labels**. Test accuracy: **0.6441** (AUC 0.6385).
  - **Procedure II (Weighted 4-Model Ensemble)**:
    - Base Models: Random Forest, Gradient Boosting, Logistic Regression, Calibrated SVM.
    - Selection Rule: Consensus thresholding and class agreement.
    - Output: **484 selected pseudo-labeled subjects**. Augmented validation accuracy: **0.7215**.
- **Critical Pipeline Decoupling**:
  - The pseudo-labels generated in Experiment 6 (552 from Proc I, 484 from Proc II) are **NOT consumed by Experiment 7**. Experiment 7 evaluates a distinct, pre-staged cohort.
- **Status**: **NOTEBOOK-OUTPUT & VERIFIED JSON ONLY**. Standalone CSVs of the accepted subject IDs were not saved as separate public files.

---

## Experiment 7: Matched Classical GCN vs. Quantum GCNN

- **Scientific Goal**: Conduct a head-to-head empirical comparison between a classical Graph Convolutional Network (GCN) and a Variational Quantum Graph Convolutional Neural Network (QGCNN).
- **Source Scripts**:
  - [`src/exp07/classical_models/model_classical_gcn.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/classical_models/model_classical_gcn.py)
  - [`src/exp07/quantum_models/quantum_embedding_broadcast.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/quantum_models/quantum_embedding_broadcast.py)
  - [`src/exp07/train_classical_gcn.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/train_classical_gcn.py)
  - [`src/exp07/quantum_models/train_qgcnn_vectorized.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/quantum_models/train_qgcnn_vectorized.py)
- **Architectural Specifications**:
  - Parcellation: AAL-116 pipeline where node feature dimension is 117 (116 connectivity weights + 1 degree/topological attribute), specified via `NODE_FEATURE_DIM`.
  - Classical GCN: `GCNConv(117, 32)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `GCNConv(32, 32)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `GCNConv(32, 16)` $\to$ `global_mean_pool` $\to$ `Linear(16, 2)`.
  - Quantum GCNN: Linear pre-projection (117 $\to$ 12), 6-qubit PennyLane variational circuit ($R_y, R_z$ angle encoding, CNOT ring entanglement, trainable $R_y$ rotations, Pauli-Z expectation measurements) $\to$ GCN layer $\to$ `global_mean_pool` $\to$ `Linear(16, 2)`.
- **Downstream Staged Cohort**:
  - Total: 875 subjects (162 clean labels: 93 TDC, 69 ADHD; 713 pseudo-labels: 535 TDC, 178 ADHD).
  - Held-out Test Set: Exactly **33 subjects**.
- **Staged Result Artifacts**:
  - [`results/exp07/checkpoint_analysis.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp07/checkpoint_analysis.json)
  - [`results/exp07/report.md`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp07/report.md)
- **Audited Numerical Results** (Evaluated on the 33 test subjects):
  - **Classical GCN**:
    - **AUROC**: **0.7293**
    - **Accuracy**: **0.6970** (23 / 33 correct)
    - **Precision**: **0.6941**
    - **Recall**: **0.6970**
    - **F1 Score**: **0.6929**
    - Confusion Matrix: $\begin{bmatrix} 15 & 4 \\ 6 & 8 \end{bmatrix}$ (4 false positives, 6 false negatives)
  - **Quantum GCNN (6 Qubits)**:
    - **AUROC**: **0.6429**
    - **Accuracy**: **0.6061** (20 / 33 correct)
    - **Precision**: **0.6204**
    - **Recall**: **0.6061**
    - **F1 Score**: **0.6082**
    - Confusion Matrix: $\begin{bmatrix} 14 & 5 \\ 8 & 6 \end{bmatrix}$ (5 false positives, 8 false negatives)
- **Submission Blocker / Missing Utility Dependency**:
  - Standalone training scripts import `from utils.config import ...` and `from utils.data_loader import ...`. The `utils/` package was omitted from staging in `src/exp07/`, preventing out-of-the-box execution of `train_classical_gcn.py` until dependency wrappers or mocks are restored.
- **Status**: **ARTIFACT-VERIFIED**.

---

## Experiment 8: Volumetric and Temporal Baselines

- **Scientific Goal**: Benchmark connectomic GNNs against three baseline paradigms: 3D spatio-temporal CNN, 4D foundation transformer (NeuroSTORM), and dynamic graph sequence learning.
- **Canonical Notebooks**:
  - [`notebooks/exp08/neuro.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/neuro.ipynb)
  - [`notebooks/exp08/true_neuro.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/true_neuro.ipynb)
  - [`notebooks/exp08/09_temporal_graph_learning.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/09_temporal_graph_learning.ipynb)
- **Baseline 1: Lightweight 3D CNN (`neuro.ipynb`)**:
  - **Input Modality**: Temporal 4D BOLD fMRI volume sequences of shape $(B, T=50, X=99, Y=117, Z=95)$, temporally subsampled by a factor of 2 to $T_s = 25$, normalized to $[-1, 1]$ and stored as `int8`.
  - **Architecture**: `Conv3d(1, 16, k=5, s=3, p=2)` $\to$ `BatchNorm3d` $\to$ `ReLU` $\to$ `Conv3d(16, 32, k=3, s=2, p=1)` $\to$ `BatchNorm3d` $\to$ `ReLU` $\to$ `Conv3d(32, 64, k=3, s=2, p=1)` $\to$ `BatchNorm3d` $\to$ `ReLU` $\to$ `AdaptiveAvgPool3d((2, 2, 2))` $\to$ temporal `AdaptiveAvgPool1d(1)` $\to$ `Linear(512, 256)` $\to$ `Linear(256, 2)`. Total parameters: **203,298**.
  - **Cohort**: 626 subjects (Train: 500, Val: 63, Test: 63).
  - **Result**: Test Accuracy = **0.7619** (48 / 63), Test AUROC = **0.7316** (Control F1 0.83, ADHD F1 0.62).
  - **Origin**: Fallback architecture implemented when the official NeuroSTORM foundation model encountered environment dependency issues in the initial staging.
- **Baseline 2: NeuroSTORM 4D Swin Transformer (`true_neuro.ipynb`)**:
  - **Input Modality**: 4D fMRI BOLD volumes ($1 \times 16 \times 96 \times 96 \times 96$).
  - **Architecture**: Official NeuroSTORM / SwiFT spatio-temporal Swin Transformer backbone fine-tuned from the pre-trained `neurostorm_mae.pth` checkpoint.
  - **Results**:
    - Single held-out test split: Accuracy = **0.6825**, AUROC = **0.6684**.
    - 5-fold cross-validation: Mean Accuracy = **0.5910 ± 0.014**, Mean AUROC = **0.5880 ± 0.038**.
  - **Third-Party Attribution**: Licensed under Apache-2.0 by CUHK-AIM-Group. Requires explicit attribution notice.
- **Baseline 3: Temporal Graph Learning (`09_temporal_graph_learning.ipynb`)**:
  - **Dataset Split**: Strictly subject-wise split across 764 subjects:
    - Training: 534 subjects (21,496 dynamic window graphs).
    - Validation: 115 subjects (4,887 dynamic window graphs).
    - Testing: 115 subjects (4,677 dynamic window graphs).
    - Disjoint verification: `assert set(train_subjects).isdisjoint(test_subjects)` explicitly executed in Cell 22.
  - **Section 1 (Window-Level Model)**:
    - Evaluated on 4,677 test graphs originating from the 115 test subjects.
    - Test Loss = **1.2331**, Accuracy = **0.5443**, Balanced Acc = **0.5340**, Precision = **0.3991**, Recall = **0.4958**, F1 = **0.4422**, AUROC = **0.5534**.
    - Confusion Matrix: $\begin{bmatrix} 1661 & 1242 \\ 839 & 825 \end{bmatrix}$.
  - **Section 2 (Subject-Level Sequence Learning)**:
    - Reorganizes the dynamic graphs into chronological per-subject sequences (average 40.65 windows/subject) for sequence modeling. Do not conflate the window-level 0.5443 result with subject-level sequence models.
  - **Leakage Status**: **NO LEAKAGE**. The earlier claim that overlapping windows were randomly split across train/test sets is refuted by Cell 22's subject-level disjoint assertions.
- **Status**: **ARTIFACT-VERIFIED**.

---

## Experiment 9: Transductive Population Graph Learning under Leave-One-Site-Out (LOSO)

- **Scientific Goal**: Evaluate cross-center generalization by holding out entire acquisition sites during transductive graph node classification.
- **Canonical Notebook**: [`notebooks/exp09/11_population_graph_learning.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp09/11_population_graph_learning.ipynb)
- **Cohort**: Exactly **497 subjects** across **7 sites** (`configs/exp09/w2b_dataset_manifest.json`):
  - Fold 0 (`KKI`): $N = 83$ test subjects
  - Fold 1 (`NYU`): $N = 93$ test subjects
  - Fold 2 (`NeuroIMAGE`): $N = 48$ test subjects
  - Fold 3 (`OHSU`): $N = 79$ test subjects
  - Fold 4 (`Peking_1`): $N = 85$ test subjects
  - Fold 5 (`Peking_2`): $N = 67$ test subjects
  - Fold 6 (`Peking_3`): $N = 42$ test subjects
  - Total: $83 + 93 + 48 + 79 + 85 + 67 + 42 = 497$.
- **Staged Result Artifacts**:
  - [`results/exp09/w2b_loso_results.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/w2b_loso_results.csv)
  - [`results/exp09/architecture_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/architecture_summary.csv)
  - [`results/exp09/graph_preprocessing_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/graph_preprocessing_summary.csv)
  - [`results/exp09/training_curves.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/training_curves.csv)
- **Audited Numerical Results across All 7 Site Folds**:

| Fold | Holdout Site | $N_{\text{test}}$ | GCN AUC | GAT AUC | GraphSAGE AUC | GIN AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `KKI` | 83 | 0.6021 | **0.6282** | 0.5238 | 0.5119 |
| 1 | `NYU` | 93 | 0.4653 | **0.5670** | 0.5427 | 0.5256 |
| 2 | `NeuroIMAGE` | 48 | 0.5374 | **0.5548** | 0.5304 | 0.4591 |
| 3 | `OHSU` | 79 | 0.4929 | 0.5142 | 0.5373 | **0.5553** |
| 4 | `Peking_1` | 85 | 0.5478 | 0.5656 | 0.5225 | **0.5799** |
| 5 | `Peking_2` | 67 | 0.5027 | **0.5491** | 0.5446 | 0.5152 |
| 6 | `Peking_3` | 42 | **0.6796** | 0.6476 | 0.6499 | 0.6590 |
| **Mean** | **Across All 7 Folds** | **497** | **0.5468** | **0.5752** | **0.5502** | **0.5437** |

- **Discrepancy: Paper Description vs. Executed Code**:
  - *Paper Description*: MST + 20% proportional thresholding, inverse-distance edge mapping $d = 1/|r|$, unweighted edges, no self-loops.
  - *Executed Code* ([`configs/exp09/graph_preprocessing_config.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/configs/exp09/graph_preprocessing_config.json) & notebook): Top-10% thresholding (empirical density 0.10003, mean edges 3,782), raw weighted connectivity features, self-loops added ($A + I$).
  - *Status*: **DOCUMENTED DISCREPANCY**.
- **Recorded Execution Environment**:
  - Frozen in [`configs/exp09/w2b_environment.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/configs/exp09/w2b_environment.json): Python 3.12.13, PyTorch 2.5.1+cu121, CUDA 12.1, PyG 2.8.0, NumPy 2.4.6, Pandas 2.3.3, Scikit-Learn 1.8.0, Nilearn 0.13.1, Nibabel 5.4.2, PennyLane 0.44.1, MONAI 1.5.2, running on NVIDIA A100-SXM4-80GB (Azure Linux 6.17 glibc 2.39).
- **Status**: **ARTIFACT-VERIFIED**.

---

## Provenance Matrix Summary

| Exp | Core Representation | Canonical Notebook | Primary Result Artifact | Verification Status | Key Caveat / Audited Detail |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | CC200 / 190 ROI dFC | `01_fc_generation_and_validation.ipynb` | `dynamic_temporal_validation.csv` | **Artifact-Verified** | Lags 1–4 are 0.8990 $\to$ 0.7789 $\to$ 0.6609 $\to$ 0.5467. Diagonals left as 1.0. |
| **2** | MST+PT Brain Graphs ($\rho=0.2$) | `02_graph_construction...ipynb`, `03_graph_metrics...ipynb` | `graph_metrics.csv`, `subject_graph_metrics.csv` | **Artifact-Verified** | 31,060 windows, 3,591 edges/window. `null_model_und_sign_fast.py` is 0 bytes. |
| **3** | Temporal Graph Moments (42 feats) | `04_dynamic_graph_feature_extraction.ipynb` | `subject_graph_features.csv` | **Partially Mapped** | Dispersion shows group separation; individual p-values are notebook-derived. |
| **4** | ComBat Multi-Site Harmonization | `w2c_athena_2.ipynb` | `comparison_table.csv`, `site_prediction_results.csv` | **Artifact-Verified** | Site acc drops 58.6% $\to$ 31.3%; diag acc drops 64.5% $\to$ 59.6%. |
| **5** | Dynamic Brain States ($K=3$) | `dynamic_transformer.ipynb` | `subject_feature_summary.csv`, `run_transition_dynamics.csv` | **Artifact-Verified** | Occupancies: 49.1%, 27.3%, 23.6%. 1,193 transition runs staged. |
| **6** | AAL-116 Pseudo-Labeling | `exp06_semi_supervised_pseudolabeling.ipynb` | `exp06_verified_results.json` | **Notebook / JSON Only** | Proc I yields 552; Proc II yields 484. Not consumed by Exp 7. |
| **7** | Matched Classical vs Quantum GCN | `train_classical_gcn.py`, `train_qgcnn_vectorized.py` | `checkpoint_analysis.json`, `report.md` | **Artifact-Verified** | 33 test subjects. Classical AUC 0.7293 vs Quantum 0.6429. Missing `utils/` import. |
| **8** | Volumetric & Temporal Baselines | `neuro.ipynb`, `true_neuro.ipynb`, `09_temporal_graph...ipynb` | `exp08_verified_results.json` | **Artifact-Verified** | 3D CNN (0.7619) operates on temporal fMRI volumes. Temporal graph (0.5443) has subject split. |
| **9** | Transductive Population Graph (LOSO)| `11_population_graph_learning.ipynb` | `w2b_loso_results.csv`, `w2b_environment.json` | **Artifact-Verified** | 7 site folds, 497 subjects. Top-10% executed vs MST+20% paper description. |
