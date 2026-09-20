# Paper vs. Code Discrepancy & Audit Ledger

**Repository**: `ADHD200_GitHub_Staging`  
**Purpose**: Rigorously document every divergence, design difference, and methodological nuance between paper descriptions/drafts and the actually executed code in this repository.  
**Auditing Standard**: Do not silently reconcile contradictions. Present both the written claim and the physical implementation.

---

## Discrepancy 1: Experiment 9 Population Graph Construction

- **Claim in Paper / Written Protocol**:
  - Population graph nodes represent human subjects.
  - Edges constructed using a Minimum Spanning Tree (MST) followed by 20% proportional thresholding (fixed density $\rho = 0.20$).
  - Edge distance metric: Inverse-distance mapping $d_{ij} = \frac{1}{|r_{ij}|}$.
  - Edges are unweighted binary connections; self-loops are excluded.
- **Actually Executed Implementation**:
  - Code reference: [`configs/exp09/graph_preprocessing_config.json`](../../configs/exp09/graph_preprocessing_config.json) and [`notebooks/exp09/11_population_graph_learning.ipynb`](../../notebooks/exp09/11_population_graph_learning.ipynb).
  - Method: `top_10pct` proportional thresholding (empirical edge density $0.10003$, mean edges $3,782$).
  - Edge weights: Continuous weighted correlation values are retained as edge attributes.
  - Self-loops: Explicitly added ($A + I$) for GCN and GAT message-passing stability.
  - Node features: 190-dimensional raw mean regional connectivity profiles.
- **Impact**:
  - The spectral radius, message-passing dilation, and graph Laplacian eigenvalues differ significantly between a 20% MST-backbone graph and a 10% weighted self-loop graph.
- **Resolution Status**: **UNRESOLVED DISCREPANCY**. Explicitly retained and documented in `docs/provenance/reproducibility_status.md` and `docs/provenance/source_of_truth.md`.

---

## Discrepancy 2: Experiment 6 to Experiment 7 Cohort Decoupling

- **Claim in Paper / Conceptual Pipeline**:
  - Semi-supervised pseudo-labeling in Track B (Exp 6) expands the scarce training pool by annotating unlabeled scans, which are then passed into the downstream Classical GCN and Quantum GCNN models in Exp 7.
- **Actually Executed Implementation**:
  - Exp 6 executed in [`notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb`](../../notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb) on an AAL-116 dataset (391 clean, 564 unlabeled pool), generating 552 pseudo-labels via Procedure I (Logistic Regression self-training) and 484 selected subjects via Procedure II (Ensemble).
  - Exp 7 evaluated in [`results/exp07/checkpoint_analysis.json`](../../results/exp07/checkpoint_analysis.json) consumed an independent staging cohort consisting of **162 clean labels** (93 TDC, 69 ADHD) and **713 pseudo-labels** (535 TDC, 178 ADHD), evaluated on 33 held-out test subjects.
  - Neither the 552 nor the 484 pseudo-labels from Exp 6 are directly consumed by the Exp 7 scripts.
- **Impact**:
  - The pipeline between Exp 6 and Exp 7 is decoupled; Exp 7 evaluates a downstream model trained on a larger, separately staged pseudo-labeled corpus.
- **Resolution Status**: **CONFIRMED PIPELINE DECOUPLING**. Must be explicitly stated in repository documentation rather than masked as a seamless pipeline.

---

## Discrepancy 3: Experiment 1 Functional Connectivity Diagonal Self-Correlations

- **Claim in Paper / Connectomics Standard**:
  - Functional connectivity correlation matrices are zeroed along the diagonal ($R_{ii} = 0$) because self-correlations ($r=1.0$) do not represent inter-regional functional communication and distort graph algorithms.
- **Actually Executed Implementation**:
  - In [`notebooks/exp01/01_fc_generation_and_validation.ipynb`](../../notebooks/exp01/01_fc_generation_and_validation.ipynb) (Cell 9), `generate_dynamic_fc()` executes:
    ```python
    fc = np.corrcoef(x.T)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    ```
  - The diagonal elements remain $1.0$ in the stored `.npy` arrays.
  - In contrast, static FC generation in later cells specifically executes `np.fill_diagonal(fc, 0)`.
- **Impact**:
  - Graph construction routines in Exp 2 must explicitly ignore self-loops during Kruskal's MST extraction and thresholding.
- **Resolution Status**: **IMPLEMENTATION DETAIL PRESERVED**. Acknowledged in `reproducibility_status.md`.

---

## Discrepancy 4: Experiment 4 ComBat Harmonization & The "ComBat Paradox"

- **Claim in Paper**:
  - Multi-site ComBat harmonization purges non-biological scanner hardware effects while fully preserving biological and clinical diagnostic variance.
- **Actually Executed Implementation**:
  - Verified from [`results/exp04/site_prediction_results.csv`](../../results/exp04/site_prediction_results.csv) and [`results/exp04/diagnosis_prediction_results.csv`](../../results/exp04/diagnosis_prediction_results.csv):
    - Site classification accuracy drops from **58.64%** (raw) to **31.29%** (post-ComBat).
    - Diagnosis classification accuracy also drops from **64.53%** (raw) to **59.56%** (post-ComBat), with balanced accuracy dropping from 59.43% to 51.60%.
- **Scientific Reality**:
  - In the ADHD-200 multi-center dataset, clinical phenotype distributions (ADHD subtype ratios, age, sex) and scanner hardware are partially correlated across sites.
  - Removing site variance via empirical Bayes location/scale adjustments removes variance that models were using for diagnosis. This is empirical evidence of sensitivity to site-associated variance, not proof that the eliminated variance was solely non-biological noise.
- **Resolution Status**: **DOCUMENTED SCIENTIFIC PHENOMENON**. Stated neutrally with exact numerical support.

---

## Discrepancy 5: Experiment 8 Lightweight 3D CNN Modality

- **Erroneous Statement in Prior Drafts**:
  - Prior descriptions characterized the Lightweight 3D CNN in `notebooks/exp08/neuro.ipynb` as a structural T1 anatomical volume model.
- **Actually Executed Implementation**:
  - In [`notebooks/exp08/neuro.ipynb`](../../notebooks/exp08/neuro.ipynb) (Cells 10 and 22), the model accepts input of shape:
    $$(B, T=50, X=99, Y=117, Z=95)$$
  - Subsamples time points by 2 ($T_s = 25$), feeds 3D spatial slices through 3D convolutional blocks, pools over time with `AdaptiveAvgPool1d(1)`, and classifies through a dense layer (203,298 parameters).
  - The inputs are derived from **4D resting-state BOLD fMRI volume sequences**, not structural T1 images.
- **Resolution Status**: **CORRECTED IN DOCUMENTATION**.

---

## Discrepancy 6: Experiment 8 Temporal Graph Data Leakage Clarification

- **Erroneous Statement in Prior Drafts**:
  - Prior drafts asserted that overlapping dynamic windows ($W=30, S=5$) were randomly partitioned into train and test sets, causing severe data leakage across windows.
- **Actually Executed Implementation**:
  - In [`notebooks/exp08/09_temporal_graph_learning.ipynb`](../../notebooks/exp08/09_temporal_graph_learning.ipynb) (Cells 18–23):
    ```python
    train_subjects, temp_subjects = train_test_split(subjects, test_size=0.30, random_state=42)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.50, random_state=42)
    assert set(train_subjects).isdisjoint(test_subjects)
    ```
  - Splits were partitioned strictly by **subject ID**: 534 training subjects (21,496 graphs), 115 validation subjects (4,887 graphs), 115 test subjects (4,677 graphs).
  - The 0.5443 accuracy / 0.5534 AUC result was evaluated on 4,677 window graphs originating exclusively from the 115 held-out test subjects.
- **Resolution Status**: **CORRECTED IN DOCUMENTATION**. Data leakage claim dismissed based on verified code assertions.
