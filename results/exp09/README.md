# Experiment 09: Leave-One-Site-Out Population Graph Learning

This directory contains evaluation outputs from the 7-fold Leave-One-Site-Out (LOSO) cross-validation experiments evaluating population graph learning on the CC200 atlas (497 subjects across 7 scanner sites: KKI, NYU, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3).

## Retained Artifacts

### 1. Primary Benchmark & Canonical Results
- **`gnn_loso_fold_results.csv`** *(historical: `w2b_loso_results.csv`)*: Full fold-level evaluation table across all 4 evaluated architectures (GCN, GAT, SAGE, GIN) recording AUC, Balanced Accuracy (BA), Sensitivity, Specificity, and F1 score per test site.
- **`gnn_loso_summary_statistics.csv`** *(historical: `w2b_summary_with_ci.csv`)*: Model performance summary with 95% confidence intervals calculated from the seven LOSO fold AUC values using the Student-t distribution with 6 degrees of freedom.
- **`exp09_verified_results.json`**: Executed-protocol metadata and aggregate mean AUC values for the four evaluated architectures.
- **`graph_preprocessing_summary.csv`**: Summary statistics of subject-level graph construction using top-10% positive-FC percentile thresholding with self-loops and stored signed correlation edge attributes (`edge_attr`).

### 2. Recovered Historical Evidence Ledgers
- **`threshold_sweep_graph_statistics.csv`** *(historical: `graph_statistics.csv`)*: Recovered historical eight-regime threshold sweep (`none`, `top_5pct`, `top_10pct`, `top_15pct`, `top_20pct`, `abs_gt_0.2`, `abs_gt_0.25`, `abs_gt_0.3`) measuring graph-topological metrics (density, small-worldness $\sigma$, connected components ratio). *Note: Producer script was not retained in surviving tree; preserved as recovered empirical sweep evidence.*
- **`threshold_positive_fc_diagnostic.csv`** *(historical: `gnn_diagnostic_positive.csv`)*: Downstream 7-fold LOSO diagnostic evaluation under positive proportional thresholding across architectures (Mean AUC = 0.5687).
- **`threshold_absolute_fc_diagnostic.csv`** *(historical: `gnn_diagnostic_absolute.csv`)*: Downstream 7-fold LOSO diagnostic evaluation under absolute thresholding across architectures (Mean AUC = 0.5534).
- **`gnn_loso_pairwise_tests.csv`** *(historical: `w2b_wilcoxon_tests.csv`)*: Pairwise Wilcoxon signed-rank tests across LOSO fold AUCs between architectures.
- **`gnn_loso_error_analysis.csv`** *(historical: `w2b_error_analysis.csv`)*: Detailed fold-level error breakdown and confusion dynamics across sites.

### 3. Classical Baselines & Representation Analysis
- **`classical_ml_canonical_results.csv`** *(historical: `w1_canonical_model_summary.csv`)*: Canonical 7-fold LOSO summary containing all 42 evaluated model configurations (7 classifiers across 6 feature families with n_folds=7).
- **`classical_ml_historical_results.csv`** *(historical: `w1_model_summary.csv`)*: Master historical ledger of all 52 recorded classical baseline runs (including intermediate 3-, 5-, and 6-fold evaluations and winner duplicates).
- **`classical_ml_family_summary.csv`** *(historical: `w1_family_winners.csv`)*: Canonical winner summary across the 6 evaluated feature families.
- **`representation_pareto_analysis.csv`** *(historical: `w2_pareto_front.csv`)*: Multi-objective Pareto evaluation across feature representations (diagnostic ADHD AUC vs scanner site prediction balanced accuracy). *Note: This artifact evaluates representation types (ComBat FC vs Raw FC vs GraphPheno), not edge thresholding.*

### 4. Forensic Audit Archives (`audit_source_files/exp09/`)
The raw, uncurated training execution traces and per-fold subject-level prediction dumps are archived under `audit_source_files/exp09/` for provenance verification:
- `training_curves.csv`: Epoch-level training loss and validation metrics across all folds.
- `confusion_matrices/`: 28 raw fold-level confusion matrix files across architectures.
- `predictions/`: 28 raw subject-level posterior probability prediction CSVs across architectures.
- `architecture_summary.csv`: Intermediate multi-metric rollup.

---

## Stored Edge Attributes and Message-Passing Semantics

In subject-level graph construction, signed Pearson correlation values are stored as edge attributes (`edge_attr`) alongside self-loops.
- **GCN**: Consumes stored `edge_attr` as `edge_weight` during message passing (`conv(x, edge_index, edge_weight=edge_weight)`).
- **GAT, SAGE, GIN**: Do **not** consume `edge_attr` as edge weights; they execute message passing over topological connectivity defined by unweighted `edge_index` (`conv(x, edge_index)`).

---

## Summary Results (7-Fold LOSO Mean Performance)

| Architecture | AUC | Balanced Accuracy | Sensitivity | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| GAT | 0.5752 | 0.5490 | 0.4970 | 0.6010 | 0.4827 |
| SAGE | 0.5502 | 0.5147 | 0.3664 | 0.6631 | 0.3309 |
| GCN | 0.5468 | 0.5407 | 0.5497 | 0.5317 | 0.4802 |
| GIN | 0.5437 | 0.5313 | 0.5325 | 0.5301 | 0.4690 |

*Source: Executed notebook `notebooks/exp09/loso_gnn_population_graphs.ipynb`.*
