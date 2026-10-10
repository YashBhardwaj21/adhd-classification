# Experimental Results

This directory contains lightweight experimental outputs, summary tables, and selected machine-readable evaluation artifacts derived from executed experiments.

Large raw neuroimaging data (4D BOLD fMRI scans), model checkpoints, and intermediate binary arrays exceed public repository quotas and are not included. See [`data/README.md`](../data/README.md) and [`data/availability_and_exclusions.md`](../data/availability_and_exclusions.md) for details on external data requirements and excluded files.

A machine-readable index of the canonical retained result artifacts is available in [`manifest.csv`](manifest.csv). Additional supplementary, historical, and validation artifacts are retained within the experiment-specific result directories.

## Result Summary by Experiment

- **[`exp01/`](exp01/)**: Dynamic functional connectivity (dFC) sliding-window lag correlation decay and Frobenius distance tables (`temporal_similarity_validation.csv`, `static_dynamic_fc_comparison.csv`; executed via `notebooks/exp01/dynamic_fc_temporal_validation.ipynb`).
- **[`exp02/`](exp02/)**: Topological graph metrics across 31,060 sliding windows on the CC200 atlas (`graph_metrics_window_level.csv`, `graph_metrics_subject_level.csv`; executed via `notebooks/exp02/graph_metrics_null_model_validation.ipynb`).
- **[`exp03/`](exp03/)**: Distributional statistics of dynamic network features and cross-site scanner variation ANOVA F-tests (`dynamic_feature_statistics.csv`, `site_effect_anova.csv`; executed via `notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb`).
- **[`exp04/`](exp04/)**: Empirical Bayes ComBat scanner harmonization comparison tables and site vs. diagnostic prediction trade-offs (`combat_harmonization_comparison.csv`, `site_prediction_combat_comparison.csv`; executed via `notebooks/exp04/combat_site_effects_harmonization.ipynb`).
- **[`exp05/`](exp05/)**: Dynamic brain micro-state dwell times, fractional occupancy rates, and Markovian state transition sequences (`dynamic_state_biomarkers.csv`, `dynamic_state_sequences.csv`; executed via `notebooks/exp05/dynamic_state_discovery.ipynb`).
- **[`exp06/`](exp06/)**: Semi-supervised pseudo-labelling verification records and label distributions on AAL-116 (`exp06_verified_results.json`; executed via `notebooks/exp06/semi_supervised_pseudolabeling.ipynb`).
- **[`exp07/`](exp07/)**: Held-out test set evaluation on 33 clean subjects for Classical GCN and Quantum QGCNN (`experiment_report.md`, `gcn_qgcnn_test_results.json`) and the complete 391-subject aligned cohort lineage manifest (`clean_cohort_lineage.csv`).
- **[`exp08/`](exp08/)**: Deep learning benchmark test results: Lightweight 3D CNN, NeuroSTORM, and temporal GNN (`baseline_model_results.json`).
- **[`exp09/`](exp09/)**: 7-fold Leave-One-Site-Out (LOSO) population graph learning cross-validation tables across GCN, GAT, SAGE, and GIN (`gnn_loso_fold_results.csv`, `README.md`), along with recovered eight-regime threshold sweep statistics (`threshold_sweep_graph_statistics.csv`) and diagnostic evaluation ledgers.
