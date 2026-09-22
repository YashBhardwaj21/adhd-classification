# Data Provenance & Exclusion Ledger

This document lists large data artifacts excluded from this Git repository and explains their provenance.

---

## 1. Excluded Artifacts

| Excluded Artifact | Original Scope | Rationale |
| :--- | :--- | :--- |
| **Raw 4D BOLD fMRI Scans** | All sites | High-volume imaging files governed by ADHD-200 open data distribution; not tracked in Git. |
| **Craddock-200 Parcellated Time Series** | Track A | High-dimensional intermediate arrays; reconstructed from preprocessed connectomes. |
| **Exp 07 Combined Arrays** | `X_combined_full.npy`, `y_combined.npy` | Historical training arrays for Exp 07; not redistributed in this repository. |
| **Model Checkpoints** | Exp 07, NeuroSTORM | Trained model weights from historical runs; not redistributed in this repository. |

---

## 2. Retained In-Tree Artifacts

All lightweight summary manifests, cross-validation tables, confusion matrices, and execution logs are retained in [`results/`](../results/):
- Window indexing manifests (`results/exp01/dynamic_temporal_validation.csv`)
- Full graph metric outputs across all 31,060 windows (`results/exp02/graph_metrics.csv`)
- Topological feature statistics and cross-site ANOVA tables (`results/exp03/`)
- Harmonization comparative metrics (`results/exp04/comparison_table.csv`)
- Dynamic connectivity state sequences and transition dynamics (`results/exp05/`)
- Semi-supervised pseudo-label evaluation records (`results/exp06/exp06_verified_results.json`)
- Classical vs Quantum held-out test evaluation on 33 test subjects (`results/exp07/checkpoint_analysis.json`)
- Volumetric and temporal benchmark test metrics (`results/exp08/exp08_verified_results.json`)
- Complete 7-fold Leave-One-Site-Out results table across all 497 subjects (`results/exp09/w2b_loso_results.csv`)
