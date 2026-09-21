# Experiment 09: Leave-One-Site-Out Population Graph Learning

This directory contains evaluation outputs from the 7-fold Leave-One-Site-Out (LOSO) cross-validation experiments evaluating population graph learning on the CC200 atlas (497 subjects across 7 scanner sites: Brown, KKI, NeuroIMAGE, NYU, OHSU, Peking, Pittsburgh).

## Retained Artifacts

- **`w2b_loso_results.csv`**: Full fold-level evaluation table across all 4 evaluated architectures (GCN, GAT, SAGE, GIN) recording AUC, Balanced Accuracy (BA), Sensitivity, Specificity, and F1 score per test site.
- **`graph_preprocessing_summary.csv`**: Summary statistics of subject-level graph construction using top 10% absolute FC edge selection with signed correlation weights.
- **`exp09_verified_results.json`**: Aggregate performance summary reporting mean and standard deviation per architecture across the 7 site folds.
- **`confusion_matrices/`**: Site-fold confusion matrices for each evaluated model.
- **`predictions/`**: Subject-level predictions and posterior probabilities across cross-validation folds.
- **`training_curves.csv`**: Epoch-by-epoch loss and validation metrics recorded during LOSO training.
- **`architecture_summary.csv`**: Model layer dimensions and hyperparameter configurations.

## Summary Results (7-Fold LOSO Mean Performance)

| Architecture | AUC | Balanced Accuracy | Sensitivity | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GAT** | 0.5752 | 0.5601 | 0.4489 | 0.6713 | 0.4789 |
| **SAGE** | 0.5502 | 0.5368 | 0.4357 | 0.6379 | 0.4578 |
| **GCN** | 0.5468 | 0.5372 | 0.4518 | 0.6226 | 0.4688 |
| **GIN** | 0.5437 | 0.5309 | 0.4312 | 0.6306 | 0.4529 |

*Source: Executed notebook `notebooks/exp09/11_population_graph_learning.ipynb`.*
