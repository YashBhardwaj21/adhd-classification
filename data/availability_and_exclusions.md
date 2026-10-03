# Data Availability and Exclusions

This document defines the data availability policy for the repository, itemizes large data artifacts excluded from Git tracking, documents retained lightweight artifacts, and specifies external inputs required to reproduce experimental tracks.

---

## 1. Raw ADHD-200 Data Availability

Raw and preprocessed neuroimaging data from the **ADHD-200 Sample** are not redistributed in this Git repository. The ADHD-200 Sample is an open-access multi-site dataset collected under institutional review board (IRB) approvals across 8 international testing centers and distributed via the **1000 Functional Connectomes Project / International Neuroimaging Data-sharing Initiative (INDI)**.

Researchers can request and download the raw 4D BOLD resting-state fMRI scans and phenotypic records from the official distribution portals:
- **NITRC ADHD-200 Portal**: [https://www.nitrc.org/projects/adhd-200/](https://www.nitrc.org/projects/adhd-200/)
- **Neurobureau Preprocessed Connectomes**: [http://neurobureau.projects.nitrc.org/ADHD200/Data.html](http://neurobureau.projects.nitrc.org/ADHD200/Data.html) (Athena pipeline, Craddock-200 parcellation, and AAL-116 parcellation).

All users of these data must abide by the original ADHD-200 Data Use Agreement, acknowledging the contributing imaging centers and attributing the consortium.

---

## 2. Excluded Large Artifacts

Due to file size quotas, institutional data governance, and long-term storage limitations, large intermediate arrays, 4D neuroimaging volumes, and trained deep-learning model checkpoints are excluded from Git:

| Excluded Artifact | Scope / Location | Approximate Size | Rationale for Exclusion |
| :--- | :--- | :--- | :--- |
| **Raw 4D BOLD fMRI Scans** | All sites / `data/raw/` | > 120 GB | High-volume raw NIfTI files; subject to external distribution terms. |
| **Craddock-200 Extracted Time Series** | Track A (`exp01`–`exp05`) | ~ 5 GB | Intermediate ROI time series; regenerated from preprocessed NIfTI files. |
| **T1-Weighted Structural MRI Scans** | Track B (`exp08`) | ~ 40 GB | High-resolution anatomical scans; not required for functional connectome pipelines. |
| **Exp 07 Combined Training Arrays** | `X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy` | > 50 MB | Historical combined training arrays (162 clean + 713 pseudo-labeled); archived externally. |
| **Pretrained Deep Learning Checkpoints** | Exp 07 (`*.pth`), Exp 08 NeuroSTORM (`*.ckpt`) | ~ 2.5 GB | Binary checkpoint weights; full numerical test evaluations are preserved in results. |
| **Intermediate Feature Parquets** | Exp 09 (historical internal names: `w1_pheno_features.parquet`, `w1_graph_features.parquet`) | ~ 150 MB | Intermediate extracted feature tables; upstream representation ledgers are retained in `results/exp09/`. |

---

## 3. Retained Lightweight Artifacts

All lightweight summary manifests, cross-validation tables, confusion matrices, configuration schemas, and execution logs are retained directly in the repository under [`results/`](../results/) and [`configs/`](../configs/):

| Retained Artifact | Experiment | Primary Purpose |
| :--- | :--- | :--- |
| **`results/exp01/temporal_similarity_validation.csv`** | Exp 01 | Window temporal similarity correlation decay across lag 1–4 ($N=764$, 1,193 acquisitions). |
| **`results/exp01/static_dynamic_fc_comparison.csv`** | Exp 01 | Static versus dynamic FC Frobenius norm distance progression. |
| **`results/exp02/graph_metrics_window_level.csv`** | Exp 02 | Topological graph metrics across all 31,060 sliding windows. |
| **`results/exp02/graph_metrics_subject_level.csv`** | Exp 02 | Subject-averaged topological graph metrics ($N=764$). |
| **`results/exp03/dynamic_feature_statistics.csv`** | Exp 03 | Dynamic topological feature distributions and diagnostic group statistics. |
| **`results/exp03/site_effect_anova.csv`** | Exp 03 | One-way ANOVA cross-site effect sizes and $F$-statistics. |
| **`results/exp04/combat_harmonization_comparison.csv`** | Exp 04 | Pre- and post-ComBat topological preservation and site clustering. |
| **`results/exp04/site_prediction_combat_comparison.csv`** | Exp 04 | Scanner site prediction balanced accuracy before and after harmonization. |
| **`results/exp04/diagnosis_prediction_combat_comparison.csv`** | Exp 04 | Diagnostic prediction performance before and after harmonization. |
| **`results/exp05/dynamic_state_biomarkers.csv`** | Exp 05 | Dynamic brain state occupancy and dwell time metrics ($K=3$). |
| **`results/exp05/dynamic_state_transitions.csv`** | Exp 05 | Transition probability matrices between discrete connectivity micro-states. |
| **`results/exp06/exp06_verified_results.json`** | Exp 06 | Semi-supervised pseudo-labeling metrics (Procedure I: 552 labels, Procedure II: 484 labels). |
| **`results/exp07/gcn_qgcnn_test_results.json`** | Exp 07 | Held-out test evaluation on 33 clean subjects for Classical GCN vs Quantum QGCNN. |
| **`results/exp07/clean_cohort_lineage.csv`** | Exp 07 | Complete 391-subject lineage mapping demonstrating the 162-subject clean prefix subset. |
| **`results/exp08/baseline_model_results.json`** | Exp 08 | Benchmark evaluation metrics for 3D CNN (76.19%), NeuroSTORM (59.10%), and Temporal GNN (54.43%). |
| **`results/exp09/gnn_loso_fold_results.csv`** | Exp 09 | Complete 7-fold Leave-One-Site-Out (LOSO) cross-validation results across 4 GNN architectures ($N=497$). |
| **`results/exp09/classical_ml_canonical_results.csv`** | Exp 09 | Full 7-fold LOSO evaluation table for 42 classical ML baseline models across 6 feature representations. |
| **`results/exp09/classical_ml_family_summary.csv`** | Exp 09 | Winning classical models and mean AUCs per feature representation family. |
| **`results/exp09/representation_pareto_analysis.csv`** | Exp 09 | Multi-objective Pareto evaluation across feature representations (diagnostic AUC vs site prediction BA). |
| **`results/exp09/threshold_sweep_graph_statistics.csv`** | Exp 09 | Recovered historical 8-regime graph threshold sweep ledger. |
| **`results/manifest.csv`** | All | Central cryptographic manifest indexing all retained results with sample sizes and SHA-256 hashes. |

---

## 4. External Inputs Required for Reproduction

To execute specific experimental tracks from raw inputs, the following files must be placed in `data/`:

1. **Track A (Experiments 01–05)**:
   - Athena preprocessed resting-state time series parcellated with the Craddock-200 (CC200) atlas (`*nii.gz` or pre-extracted `.1D` / `.npy` files).
   - ADHD-200 phenotypic demographic table (`ADHD200_phenotypic.csv`).

2. **Track B (Experiments 06–08)**:
   - Preprocessed resting-state fMRI parcellated with the AAL-116 atlas (for Exp 06–07).
   - 4D functional BOLD NIfTI volumes or pre-extracted 3D spatial volumes (for Exp 08).
   - Aligned phenotypic records containing diagnostic status and quality control flags.

3. **Track C (Experiment 09)**:
   - Pre-computed static functional connectivity matrices (CC200 atlas, 190 active ROIs, 497 subjects across 7 sites).
   - Site acquisition identifiers and diagnostic class labels (TDC vs ADHD).
