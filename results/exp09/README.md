# Experiment 09: Leave-One-Site-Out Population Graph Learning

This directory contains evaluation outputs from the 7-fold Leave-One-Site-Out (LOSO) cross-validation experiments evaluating population graph learning on the CC200 atlas (497 subjects across 7 scanner sites: Brown, KKI, NeuroIMAGE, NYU, OHSU, Peking, Pittsburgh).

## Retained Artifacts

- **`w2b_loso_results.csv`**: Full fold-level evaluation table across all 4 evaluated architectures (GCN, GAT, SAGE, GIN) recording AUC, Balanced Accuracy (BA), Sensitivity, Specificity, and F1 score per test site.
- **`graph_preprocessing_summary.csv`**: Summary statistics of subject-level graph construction using top 10% absolute FC edge selection with signed correlation weights.
- **`exp09_verified_results.json`**: Executed-protocol metadata and aggregate mean AUC values for the four evaluated architectures.
- **`architecture_summary.csv`**: Aggregate mean and standard deviation of AUC, balanced accuracy, and F1 across the 7 LOSO folds, with fold counts by architecture.
- **`training_curves.csv`**: Epoch-level training and validation metrics across LOSO folds.
- **`confusion_matrices/`**: Site-fold confusion matrices for each evaluated model.
- **`predictions/`**: Subject-level predictions and posterior probabilities across folds.

## Summary Results (7-Fold LOSO Mean Performance)

| Architecture | AUC | Balanced Accuracy | Sensitivity | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| GAT | 0.5752 | 0.5490 | 0.4970 | 0.6010 | 0.4827 |
| SAGE | 0.5502 | 0.5147 | 0.3664 | 0.6631 | 0.3309 |
| GCN | 0.5468 | 0.5407 | 0.5497 | 0.5317 | 0.4802 |
| GIN | 0.5437 | 0.5313 | 0.5325 | 0.5301 | 0.4690 |

*Source: Executed notebook `notebooks/exp09/11_population_graph_learning.ipynb`.*
