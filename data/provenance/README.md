# Data Provenance & Exclusion Ledger

## 1. Overview
This repository adheres to standard biomedical research and open-source publication practices. Due to GitHub's individual file size cap (files $>100\text{ MiB}$ are blocked) and institutional data redistribution agreements, large raw imaging volumes and high-dimensional intermediate matrices are excluded from the public Git tree.

## 2. Excluded Large Artifacts
| Excluded Artifact | Approximate Size | Original Path / Scope | Rationale for Exclusion |
| :--- | :--- | :--- | :--- |
| **Craddock-200 Athena Raw Time Series** | $143.79\text{ MB}$ | `01_fc_generation/.../cc200_time_series.parquet` | Exceeds GitHub $100\text{ MiB}$ file size limit. Excluded to ensure repository health. |
| **Raw 4D BOLD fMRI NIfTI Scans** | $>120\text{ GB}$ | NeuroIMAGE, NYU, Peking, etc. | High-volume medical imaging files subject to ADHD-200 usage agreements. |
| **Model Weight Checkpoints** | $10\text{ MB} - 500\text{ MB}$ | `src/exp07/classical_best_model.pth`, NeuroSTORM checkpoints | Model weights can be reproduced using provided training scripts and seeds. |

## 3. Retained In-Tree Artifacts
All summary manifests, evaluation tables, confusion matrices, and execution logs are retained in the `results/` and `notebooks/` directories:
- Window indexing manifests (`results/exp01/dynamic_manifest.csv`)
- Full graph metric outputs across all 31,060 windows (`results/exp02/graph_metrics.csv`)
- Empirical topological feature parameters (`results/exp03/feature_statistics.csv`, `site_anova.csv`)
- Harmonization comparative metrics (`results/exp04/comparison_table.csv`)
- Dynamic micro-state sequences and transition dynamics (`results/exp05/`)
- Semi-supervised pseudo-label evaluation records (`results/exp06/exp06_verified_results.json`)
- Classical vs Quantum test set evaluation across all 33 test subjects (`results/exp07/checkpoint_analysis.json`)
- Volumetric and temporal benchmark test metrics (`results/exp08/exp08_verified_results.json`)
- Complete 7-fold Leave-One-Site-Out results table across all 497 subjects (`results/exp09/w2b_loso_results.csv`)
