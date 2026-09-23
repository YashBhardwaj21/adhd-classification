# Experimental Results

This directory contains lightweight experimental outputs, summary tables, and selected machine-readable evaluation artifacts derived from executed experiments.

Large raw neuroimaging data (4D BOLD fMRI scans), model checkpoints, and intermediate binary arrays exceed public repository quotas and are not included. See [`data/README.md`](../data/README.md) and [`data/provenance.md`](../data/provenance.md) for details on external data requirements and excluded files.

A machine-readable index of the canonical retained result artifacts is available in [`manifest.csv`](manifest.csv). Additional supplementary and validation artifacts are retained within the experiment-specific result directories.

## Result Summary by Experiment

- **[`exp01/`](exp01/)**: Dynamic functional connectivity (dFC) sliding-window lag correlation decay and Frobenius distance tables (`01_fc_generation_and_validation.ipynb`).
- **[`exp02/`](exp02/)**: Topological graph metrics across 31,060 sliding windows on the CC200 atlas (`03_graph_metrics_and_null_models.ipynb`).
- **[`exp03/`](exp03/)**: Distributional statistics of dynamic network features and cross-site scanner variation ANOVA F-tests (`04_dynamic_graph_feature_extraction.ipynb`).
- **[`exp04/`](exp04/)**: Empirical Bayes ComBat scanner harmonization comparison tables and site vs. diagnostic prediction trade-offs (`w2c_athena_2.ipynb`).
- **[`exp05/`](exp05/)**: Dynamic brain micro-state dwell times, fractional occupancy rates, and Markovian state transition sequences (`dynamic_transformer.ipynb`).
- **[`exp06/`](exp06/)**: Semi-supervised pseudo-labelling verification records and label distributions on AAL-116 (`exp06_semi_supervised_pseudolabeling.ipynb`).
- **[`exp07/`](exp07/)**: Held-out test set evaluation on 33 clean subjects for Classical GCN and Quantum QGCNN (`report.md`), including `checkpoint_analysis.json` which contains aggregate performance metrics along with per-subject ground-truth labels, predictions, and predicted diagnostic probabilities.
- **[`exp08/`](exp08/)**: Deep learning benchmark test results: Lightweight 3D CNN, NeuroSTORM, and temporal GNN (`exp08_verified_results.json`).
- **[`exp09/`](exp09/)**: 7-fold Leave-One-Site-Out (LOSO) population graph learning cross-validation tables across GCN, GAT, SAGE, and GIN (`w2b_loso_results.csv`, `README.md`).
