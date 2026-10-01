# Provenance, Discrepancies & Execution Ledger

This document provides a factual record of discrepancies between manuscript descriptions, early project notes, and the executed code artifacts across the study:
*ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical–Quantum Graph Learning, and Cross-Site Generalization*  
**Authors**: Shantanu Khare, Bhavana Deepthi, Yash Bhardwaj, Shridevi S, Daehan Won (Vellore Institute of Technology & Binghamton University).  
**Status**: Manuscript in preparation (2026).

---

## 1. Executive Discrepancy Table

| Experiment | Earlier Manuscript / Early Project Description | Executed Code Artifact | Status |
| :---: | :--- | :--- | :---: |
| **Exp 01** | Static vs dynamic FC stability comparison | Diagonal zeroed before distance computation; lag correlation decay $0.8990 \to 0.5467$ | Documented |
| **Exp 02** | MST + 20% proportional thresholding | `mst_graph` in notebook and `src/exp02/graph_utils.py` both implement canonical MST + PT ($\rho=0.20$, 3,591 edges, $D = 1 - |r|$) | Documented |
| **Exp 03** | Diagnostic group separation analysis (Earlier draft: Mann–Whitney U) | ANOVA captures inter-site scanner variation ($F > 4,000$); diagnostic separation evaluated via Welch unequal-variance $t$-test + BH-FDR output directly to notebook cells in `notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb` | Documented |
| **Exp 04** | ComBat removes batch effects while retaining signal | Scanner accuracy drops $58.64\% \to 31.29\%$, and static diagnostic accuracy drops $64.53\% \to 59.56\%$; dynamic micro-state diagnosis improves $0.626 \to 0.675$ | Documented |
| **Exp 05** | Dynamic micro-state clustering | $K=3$ selected via multi-metric MeanRank across Silhouette, Calinski–Harabasz, and Davies–Bouldin ($K \in [2, 10]$) and confirmed by 30-run ARI stability ($0.9950 \pm 0.0024$); State 0 dominates dwell time ($6.24$ windows) | Documented |
| **Exp 06** | Semi-supervised pseudo-labelling | Procedure I yielded 552 pseudo-labels; Procedure II yielded 484 consensus labels on 391 clean aligned cohort; 79-subject holdout accuracy $0.6709 \to 0.7215$ | Documented |
| **Exp 07** | Earlier manuscript draft: Density 0.20, weighted node strength, 2 quantum layers | Executed config used nominal density $0.15$ (`>=` threshold), 1 quantum layer, 3-layer Classical GCN, 20 epochs, 117 node features, stored `edge_attr` not passed to `GCNConv`, 162 clean + 713 selected pseudo cohort (103/26/33 clean split; 816 train); 162 clean prefix proven subset of Exp 06 391 aligned cohort | Documented / Archived |
| **Exp 08** | Earlier manuscript draft: Lightweight 3D CNN on structural T1 scans | Lightweight 3D CNN (Acc 76.19%) and NeuroSTORM (Acc 68.25% / 59.10%) evaluated on 4D functional BOLD volume sequences across 626 scans; Temporal GNN (Acc 54.43%) evaluated on 764-subject CC200 timeseries cohort | Documented |
| **Exp 09** | Earlier manuscript draft: Unweighted MST + 20% distance mapping | Executed notebook used `top_10pct` positive-FC thresholding with signed correlation attributes and self-loops across 497 subjects and 7 sites; Classical ML baselines evaluated 7 classifier types across 6 feature families; edge weights consumed during message passing only by GCN | Resolved |

---

## 2. Detailed Experiment Discrepancies

### Experiment 01: Dynamic FC Stability
- **Diagonal Handling**: The executed notebook (`notebooks/exp01/01_fc_generation_and_validation.ipynb`) zeroed the diagonal of correlation matrices before computing Pearson similarities and Frobenius distances across sliding windows.
- **Numbers**: Monotonic correlation decay ($0.8990 \to 0.7789 \to 0.6609 \to 0.5467$) and Frobenius distance progression ($32.58 \to 48.75 \to 60.78 \to 70.61$) match the stored artifact `results/exp01/dynamic_temporal_validation.csv`.

### Experiment 02: CC200 Graph Construction
- **Algorithm**: The executed notebook (`notebooks/exp02/02_graph_construction_and_validation.ipynb`, cell 71) implemented `mst_graph(fc, density=0.20)` by first extracting a Minimum Spanning Tree on distance matrix $D = 1 - |r|$ and then adding the highest absolute correlation edges until reaching 3,591 edges (20% density for 190 nodes).
- **Algorithm Alignment**: `src/exp02/graph_utils.py` implements canonical MST + Proportional Thresholding (MST+PT) matching the notebook implementation (`mst_graph`, 20% density, 3,591 edges, $D = 1 - |r|$). Topological metrics reported in `results/exp02/graph_metrics.csv` are derived from the canonical MST+PT construction across 31,060 sliding windows.

### Experiment 03: Cross-Site Scanner ANOVA & Diagnostic Group Comparison
- **Scope of $F$-Statistics**: The large $F$-statistics reported ($F > 4,000, p < 10^{-300}$ for global efficiency and path length) quantify variance attributable to acquisition site (scanner differences across 8 participating clinics), **not** diagnostic variance between ADHD and typically developing controls (TDC).
- **Diagnostic Group Comparisons**: Diagnostic separation between ADHD and TDC was evaluated using Welch unequal-variance $t$-tests with Benjamini–Hochberg False Discovery Rate (BH-FDR) correction ($\alpha=0.05$) across 26 dynamic topological metrics. Results exist directly within the executed notebook cell outputs of `notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb`.

### Experiment 04: ComBat Harmonization
- **Clinical Variance Attenuation**: Empirical Bayes ComBat reduced scanner identification accuracy from $58.64\%$ to $31.29\%$ (`results/exp04/site_prediction_results.csv`), while diagnostic classification accuracy also decreased from $64.53\%$ to $59.56\%$ (`results/exp04/diagnosis_prediction_results.csv`).
- **Dynamic Micro-State Gains**: In contrast, dynamic micro-state features classified with Random Forest (500 trees, 5-fold CV) improved from $0.6256$ to $0.6754$ ($0.626 \to 0.675$, $+0.0497$).
- **Correction Note**: Stale claims that scanner classification dropped $95.7\% \to 14.8\%$ represent preliminary unharmonized exploratory passes. The canonical verified benchmark records $58.64\% \to 31.29\%$.

### Experiment 05: Data-Driven Dynamic Connectivity States
- **Cluster Count Selection**: Dynamic micro-state clustering evaluated candidate state counts $K \in [2, 10]$ across Silhouette Coefficient, Calinski–Harabasz (CH) index, and Davies–Bouldin (DB) index.
- **MeanRank Procedure**: A composite MeanRank metric selected $K=3$ as the optimal state count, driven by a pronounced Calinski–Harabasz peak (12,892.17) alongside robust Silhouette (0.2772) and Davies–Bouldin (1.1399) scores. Clustering stability was confirmed via 30 repeated random initializations, yielding an Adjusted Rand Index (ARI) of $0.9950 \pm 0.0024$.
- **Dwell Time & Occupancy**: State 0 exhibited the longest mean dwell time ($6.24$ windows) and highest fractional occupancy ($49.05\%$).

### Experiment 06: Semi-Supervised Pseudo-Labeling
- **Cohort Definition**: Experiment 06 operated on 955 AAL-116 FC candidate matrices inner-joined with 691 phenotypic records, establishing a phenotypically aligned clean cohort of $N=391$ subjects (`aligned_subjects.npy`) and 564 unlabelled subjects.
- **Procedure Metrics**: Procedure I (Logistic Regression self-training, $\tau=0.75$) generated 552 pseudo-labels; Procedure II (4-model weighted ensemble) generated 484 consensus labels.
- **Holdout Evaluation**: In an independent 79-subject holdout evaluation, pseudo-label augmentation improved classification accuracy from $0.6709$ to $0.7215$ (`results/exp06/exp06_verified_results.json`).

### Experiment 07: Classical GCN and Quantum GCNN Evaluation
- **Cohort Lineage Provenance**:
  - Experiment 06 produced a 391-subject phenotypically aligned clean cohort.
  - Experiment 07 used a historical combined 875-subject array in which the first 162 entries were treated as the clean benchmark cohort and the remaining 713 entries as the pseudo-labelled/training extension.
  - At the subject-ID level, the 162 clean Exp07 subjects are verified as an exact prefix subset of the 391 Exp06 aligned cohort ($162 \subset 391$, documented in `results/exp07/clean_subjects_manifest.csv`). The historical scientific rationale for selecting the integer 162 prefix was not recovered.
  - The clean cohort was partitioned deterministically (`random_state=42`, stratified): 103 training, 26 validation, and 33 held-out test subjects.
  - Total training set: 103 clean + 713 pseudo = 816 graphs.
  - Held-out test set: 33 clean subjects (19 TDC, 14 ADHD).
- **Graph Construction**:
  - Atlas: AAL-116 ($N=116$ ROIs, 6,670 upper-triangular FC features).
  - Thresholding: Nominal 15% density threshold on upper-triangle absolute FC values (`abs_fc >= threshold`).
  - Edge Attributes: Retained signed Pearson correlation values in `edge_attr`. Stored `edge_attr` is not passed as edge weights to `GCNConv` layers.
  - Node Features: 117 dimensions (116 signed FC row values + 1 normalized degree `degree / 116`).
- **Architectures & Evaluation**:
  - Classical GCN: 3-layer GCN ($117 \to 32 \to 32 \to 16 \to 2$, BatchNorm, Dropout 0.30, 20 epochs, 5,522 parameters). Test metrics: **AUC = 0.7293**, Accuracy = 0.6970 (15/19 TDC, 8/14 ADHD).
  - Quantum QGCNN: Classical projection ($117 \to 12$), 6-qubit PQC ($N_{\text{layers}}=1$, 18 parameters), followed by 3 GCN layers ($6 \to 16 \to 16 \to 16 \to 2$, 2,188 parameters). Test metrics: **AUC = 0.6428**, Accuracy = 0.6061 (11/19 TDC, 9/14 ADHD).
  - Archival notice: Model evaluation metrics are verified in `results/exp07/checkpoint_analysis.json`.

### Experiment 08: Deep-Learning Baselines
- **Cohort & Modality Distinctions**:
  - **4D Volumetric Cohort (626 Scans)**: Lightweight 3D CNN (Acc 76.19%) and NeuroSTORM spatio-temporal transformer (Acc 68.25% / 59.10%) were evaluated on sequences of 4D functional BOLD volumes ($T=25, 99 \times 117 \times 95$), **not** structural T1 anatomical scans.
  - **Timeseries Cohort (764 Subjects)**: Temporal GNN (Acc 54.43%) was evaluated on extracted 1D BOLD timeseries from CC200 parcellations across 764 subjects.
- **Partitioning**: Strictly disjoint subject splits (534 train, 115 validation, 115 test) without subject leakage across splits (`results/exp08/exp08_verified_results.json`).

### Experiment 09: Leave-One-Site-Out Population Graphs
- **Graph Construction Protocol**: Subject-level graphs constructed using `top_10pct` positive-FC thresholding (`fc >= 90th percentile`) with self-loops across 497 subjects from 7 clinical sites and 190 CC200 ROIs (`results/exp09/graph_preprocessing_summary.csv`).
- **Edge Weight Semantics**:
  - Signed Pearson correlation values are stored as `edge_attr` in the PyG Data objects with self-loops.
  - In the main 7-fold LOSO benchmark, only isotropic GCN consumes `edge_weight=edge_weight` during message passing.
  - GAT, SAGE, and GIN execute message passing over topological connectivity defined by `edge_index` without consuming `edge_attr` as weights (`configs/exp09/w2b_best_config.json`).
- **GAT Interpretation**: GAT achieved the highest mean AUC (**0.5752**) among the four evaluated GNN architectures under 7-fold LOSO. The empirical data demonstrate superior cross-site AUC without proving that attention weights specifically suppress scanner noise.
- **Threshold Selection Provenance**: The recovered 8-regime threshold sweep ledger is preserved in `results/exp09/graph_statistics.csv` (`top_10pct` $\sigma \approx 3.81$, 80.48% connected vs `top_5pct` $\sigma \approx 6.02$, 6.44% connected), accompanied by downstream diagnostic validation ledgers (`gnn_diagnostic_positive.csv`, `gnn_diagnostic_absolute.csv`). The original producer script and exact historical selection rule were not retained in the surviving tree (`configs/exp09/selected_threshold.json`).
- **Pareto Front Artifact Clarification**: `w2_pareto_front.csv` records a multi-objective comparison of feature representations (ComBat FC vs Raw FC vs GraphPheno) against site prediction balanced accuracy. It does not represent a threshold sweep and is disassociated from threshold metadata.
- **Classical Baselines & Phenotypic Equivalence**: Evaluated across 7 classifiers and 6 feature families (canonical 42 configurations in `w1_canonical_model_summary.csv`, 52 historical records in `w1_model_summary.csv`). Elastic Net demographic baseline achieved **0.5935 mean AUC**. 'Phenotype' and 'Phenotype without IQ' evaluations are identical because `master_cohort.csv` contained no cognitive test columns.

---

## 3. Seed & Determinism Provenance

- **Dataset Split Seed**: Random seed `42` is verified as the deterministic seed for the 162 clean-subject partition (103 train / 26 val / 33 test) in Experiment 07.
- **Global RNG Initialization**: While `seed = 42` was specified across configurations, global random number generator state across GPU CUDA kernels and PennyLane quantum simulator backends was not independently verified for every historical run. Deterministic bitwise replication of trained weights cannot be asserted without the original execution environment.

---

## 4. Removed Legacy Files & File Hash Ledger

Prior to repository cleanup, several alternative and legacy implementation variants were present in the tree. To maintain a lean research repository, these variants have been removed from the public working tree. Their SHA256 hashes are recorded below for provenance:

| File Path (Prior to Removal) | SHA256 Hash | Historical Role |
| :--- | :--- | :--- |
| `src/exp07/quantum_models/quantum_embedding_vectorized.py` | `3CA08DF107479CC8C197E68A40286394B20B58AF1354F6C62001CE31D1153CE0` | Legacy standalone implementation variant |
| `src/exp07/quantum_models/quantum_embedding_vmap.py` | `009EBC8DA049B69137E67AD3B154E820CA9F5379FE3C414470664D697CA5A797` | Experimental Jax/vmap quantum circuit variant |
| `src/exp07/quantum_models/resume.py` | `FC2803A39EEADAD35A98FD3BB7154E502E67301471EC0AEF66B2863EFB995303` | Training checkpoint resumption utility |
| `scripts/clean_notebooks.py` | `6A01A093DE8D3FF3690B8100529D7D62FBBEB93375815610DB7023BF4DDE4BFD` | Maintenance utility for notebook output stripping |
| `scripts/fix_markdown_links.py` | `28F0FA1DF334A2E7ED4965377DA9CEAFE91A6E77A8EEAE49D6A1296C8395BBA4` | Maintenance utility for markdown link validation |

---

## 5. Recovered Historical Evidence Artifacts

The following lightweight artifacts were recovered from historical cloud development archives to provide empirical provenance for key experimental decisions:

| Artifact Path | Source Archive Origin | SHA256 Hash | Status | Producer Code Availability |
| :--- | :--- | :--- | :--- | :--- |
| `results/exp07/clean_subjects_manifest.csv` | Derived from `aligned_subjects.npy` and Exp 07 split logic | `e60971bc3a90c5d50696b3d94de4b99a17a36fdc65866b918fa8b3fcca54b630` | Provenance Manifest | Fully reproducible via `scratch/generate_manifest.py` |
| `results/exp09/graph_statistics.csv` | `06_results/workflow2b/graph_statistics.csv` | `cd0a714e43f4ce905268754b17a33d855bb316cc7a9b948b1dc7207180de0512` | Recovered Historical Ledger | Producer script not retained in repository |
| `results/exp09/gnn_diagnostic_positive.csv` | `06_results/workflow2b/gnn_diagnostic_positive.csv` | `ae4b6153d370c1af73fc734e8e6a84c3afe5a485d852cf3e0668370a61510dc3` | Recovered Historical Validation | Producer script not retained in repository |
| `results/exp09/gnn_diagnostic_absolute.csv` | `06_results/workflow2b/gnn_diagnostic_absolute.csv` | `a024403690ef1dd198441d3e22dfce382f4071f7ebcbb7aca29320931d3be662` | Recovered Historical Validation | Producer script not retained in repository |
| `results/exp09/w2b_wilcoxon_tests.csv` | `06_results/workflow2b/w2b_wilcoxon_tests.csv` | `9dbe6cdd7aed848a253d2ecfca5963be926bb94d00ddd87273ff4f845240b532` | Recovered Statistical Ledger | Derived statistical tests |
| `results/exp09/w2b_summary_with_ci.csv` | `06_results/workflow2b/w2b_summary_with_ci.csv` | `7418c381bd946b8c36b52f4542d4df0f5c1b20539ed3706aae56eb7458167afe` | Recovered Statistical Ledger | Derived summary table with CIs |
| `results/exp09/w2b_error_analysis.csv` | `06_results/workflow2b/w2b_error_analysis.csv` | `7d87a833b1b146d876113f96713427dda9a907d860584d9f2d7d9566f0f91f2d` | Recovered Diagnostic Ledger | Derived error analysis |

---

## 6. Third-Party Code & Licensing Determinations

| Bundled Component | File Path | Origin / Upstream | Upstream License | Status in Repository |
| :--- | :--- | :--- | :--- | :--- |
| **Brain Connectivity Toolbox** | `src/exp02/null_model_und_sign_fixed.py`, `src/exp02/randmio_und_signed_fast.py` | Rubinov & Sporns (2011) / `bctpy` | GNU General Public License v3.0 or later (GPL-3.0-or-later) | Bundled algorithm implementation (see `third_party/bctpy.md`) |
| **NeuroSTORM Transformer** | `src/exp08/neurostorm/neurostorm.py` | CUHK-AIM-Group (derived from MONAI / SwiFT) | Apache License, Version 2.0 (Apache-2.0) | Bundled model implementation (see `NOTICE`) |
| **neuroCombat** | N/A (external package dependency) | Fortin et al. (2018) | MIT | Pip dependency; not bundled |

### Brain Connectivity Toolbox (BCT) Provenance

The historical implementation contains BCT-derived graph-randomization code. In the historical source, `null_model_und_sign_fixed.py` depended on BCTPY utilities and the `randmio_und_signed` routine; the current repository version retains the derived null-model implementation and BCT utility imports. `randmio_und_signed_fast.py` is a Numba-accelerated reimplementation of the BCTPY `randmio_und_signed` routine. The relevant historical source is therefore treated as BCT-derived/adapted code rather than as an independently authored graph-randomization algorithm.

Upstream project: aestrivex/bctpy (https://github.com/aestrivex/bctpy). The upstream BCTPY repository is GPL-3.0. The exact BCTPY version and upstream commit used during the historical experiment were not recorded in the available provenance evidence.

The historical `null_model_und_sign_fixed.py` file carries a Rubinov 2011 attribution for the undirected signed null model:
Mikail Rubinov and Olaf Sporns. "Weight-conserving characterization of complex functional brain networks." *NeuroImage*, 2011; 56(4): 2068–2079. DOI: 10.1016/j.neuroimage.2011.03.069.

See [third_party/bctpy.md](../third_party/bctpy.md) for the complete third-party provenance document and GPL-3.0-or-later licensing notices.
