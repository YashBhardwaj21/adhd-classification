# Forensic Research Provenance & Evidence Ledger

This document establishes the evidence classification model, cohort lineage records, historical discrepancy reconciliations, and cryptographic verification ledger governing all retained artifacts.

---

## 1. Evidence Model

To ensure transparency, artifacts and claims across this repository are categorized into three distinct evidentiary tiers:

### Canonical Current Protocol
Methods, source code implementations, and execution configurations that define the current manuscript's benchmark protocol. These represent the definitive, active operational procedures of the repository.

### Recovered Historical Evidence
Lightweight empirical data ledgers, parameter logs, and evaluation summaries recovered from past execution runs. These artifacts verify reported figures, tables, and claims even where large intermediate binary files are no longer tracked in Git.

### Unresolved Provenance
Procedures or decisions for which the empirical results survive but the original producer code, intermediate scripts, or formal algorithmic selection rules were not preserved in surviving project records.

---

## 2. Cohort Lineage & Disjoint Population Mapping

The lineage mapping connecting the initial ADHD-200 releases to the evaluated model populations is documented below:

```text
955 Available AAL-116 Connectivity Subjects
  │
  ├── 391 Aligned Clean Cohort (Exp 06)
  │     │   (Verified phenotypic and imaging quality control)
  │     │
  │     └── 162 Verified Exp 07 Clean Subset
  │           │   (Proven mathematical prefix subset: 162 ⊂ 391)
  │           ├── 103 Training Clean Subjects
  │           ├── 26 Validation Clean Subjects
  │           └── 33 Held-Out Test Clean Subjects
  │
  └── 564 Unlabeled Candidate Subjects
        │   (Exp 06 candidate self-training pool)
        │
        └── 713 Downstream Pseudo-Labeled Samples (Exp 07)
              │   (Historical expansion cohort)
              │
              └── Combined with 103 Train Clean
                    ├── 816 Total Training Graphs
                    └── 875 Accounted Exp 07 Population (816 train + 26 val + 33 test)
```

- **Proof of Subset Membership**: Analysis of subject identifiers confirms that all 162 clean Exp 07 subjects exist within the 391 Exp 06 aligned cohort ([`results/exp07/clean_cohort_lineage.csv`](../results/exp07/clean_cohort_lineage.csv)).
- **Historical Uncertainty**: The original rationale for selecting the specific 162-subject prefix rather than all 391 aligned subjects was not recorded in surviving project documentation.

---

## 3. Separation of Graph Construction Pipelines

Experiments 02 and 09 implement distinct graph pipelines that must not be conflated:

| Attribute | Experiment 02 Pipeline | Experiment 09 Pipeline |
| :--- | :--- | :--- |
| **Pipeline Type** | Dynamic sliding-window connectomics | Static population graph classification |
| **Parcellation** | Craddock-200 (190 ROIs) | Craddock-200 (190 ROIs) |
| **Algorithm** | Kruskal MST + Proportional Thresholding (MST+PT) | Top-10% positive FC percentile thresholding |
| **Distance Metric** | $d_{ij} = 1 - |r_{ij}|$ | $r_{ij} \ge \text{percentile}(r, 90)$ |
| **Edge Density** | Exactly $\rho = 0.20$ ($E = 3,591$) | Mean $\rho = 0.10003$ |
| **Connectedness** | Guaranteed 1 connected component | Variable (average 1.05 components) |
| **Edge Attribute Semantics** | Retains signed $r_{ij}$ as `edge_attr` | Retains signed $r_{ij}$ as `edge_attr`; GCN consumes as `edge_weight`; GAT/SAGE/GIN use unweighted `edge_index` |

---

## 4. Historical Corrections & Reconciliations

Systematic code and execution audits identified and corrected several discrepancies between early working drafts and verified execution artifacts:

### Experiment 02: Distance Formulation
- *Historical Draft Claim*: $d_{ij} = \sqrt{2(1 - r_{ij})}$.
- *Audited Execution Reality*: Distance is defined as $d_{ij} = 1 - |r_{ij}|$ to accommodate positive and negative correlations while prioritizing strong absolute couplings for Kruskal's MST.

### Experiment 06: Procedure I Confidence Threshold
- *Historical Draft Claim*: Posterior confidence threshold $\tau = 0.85$.
- *Audited Execution Reality*: The actual executed self-training code applied a threshold of $\tau = 0.75$, yielding 552 pseudo-labels.

### Experiment 07: Architecture and Graph Parameters
- *Nominal Density*: 15% proportional thresholding (nominal density 0.15).
- *Quantum Layer Count*: Configured with $N_{\text{layers}} = 1$ (1 variational layer, 18 trainable parameters), rather than 2 layers.
- *Node Feature Dimension*: $d = 117$ composed of 116 signed Pearson correlation values plus 1 normalized unweighted degree feature ($\text{degree} / 116$).
- *Message Passing*: Message passing across graph convolution layers is strictly **unweighted**; stored correlation weights are not consumed as convolution edge weights.

### Experiment 08: Imaging Modality
- *Historical Draft Claim*: T1-weighted structural MRI inputs.
- *Audited Execution Reality*: Volumetric 3D CNN and NeuroSTORM models process 4D functional BOLD volumes ($99 \times 117 \times 95 \times 25$).

### Experiment 09: Threshold and Message Passing Semantics
- *Threshold Formulation*: Top-10% positive FC percentile threshold applied to upper-triangle signed Pearson correlation values with self-loops.
- *Message Passing Semantics*: Signed correlation values are stored in `edge_attr`. GCN consumes them as message-passing `edge_weight`; GAT, GraphSAGE, and GIN pass unweighted `edge_index`.

---

## 5. Threshold Provenance & Selection Status

The top-10% positive FC threshold in Experiment 09 represents a **recovered historical operating point**, not a reconstructable optimization result:
1. An exploratory 8-regime sweep (`none`, `top_5pct`, `top_10pct`, `top_15pct`, `top_20pct`, `abs_gt_0.2`, `abs_gt_0.25`, `abs_gt_0.3`) was recovered from execution artifacts ([`results/exp09/threshold_sweep_graph_statistics.csv`](../results/exp09/threshold_sweep_graph_statistics.csv)).
2. Downstream diagnostic validation confirmed superior performance for the positive threshold over absolute thresholding ([`results/exp09/threshold_positive_fc_diagnostic.csv`](../results/exp09/threshold_positive_fc_diagnostic.csv)).
3. The original producer script and exact historical selection rule were not preserved in surviving project records.
4. Consequently, this threshold is documented as a recovered empirical finding ([`configs/exp09/graph_threshold_selection.json`](../configs/exp09/graph_threshold_selection.json)).

---

## 6. Missing and Excluded Artifacts

The following artifacts cannot be redistributed or reconstructed from the public repository:
- **Raw Imaging Data**: Raw 4D BOLD NIfTI files (>120 GB) governed by ADHD-200 distribution terms.
- **Combined Training Arrays (Exp 07)**: `X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy` (875 subjects).
- **Deep Learning Checkpoints**: Checkpoint weights for Classical GCN, QGCNN, and NeuroSTORM.
- **Intermediate Feature Parquets (Exp 09)**: `w1_pheno_features.parquet`, `w1_graph_features.parquet`.

---

## 7. Retained Empirical Evidence Ledger

The central cryptographic manifest ([`results/manifest.csv`](../results/manifest.csv)) indexes all retained canonical results:

| Retained Artifact | Originating Experiment | Dataset / Atlas | Sample Size ($N$) | Evidentiary Status | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `results/exp01/temporal_similarity_validation.csv` | Exp 01 | CC200 | 764 (1,193 runs) | Derived | `d7a48c...` |
| `results/exp01/static_dynamic_fc_comparison.csv` | Exp 01 | CC200 | 764 (1,193 runs) | Derived | `a1f9e2...` |
| `results/exp02/graph_metrics_window_level.csv` | Exp 02 | CC200 | 764 (31,060 windows)| Derived | `38b4c1...` |
| `results/exp02/graph_construction_configuration.csv`| Exp 02 | CC200 | N/A | Derived | `520e18...` |
| `results/exp03/dynamic_feature_statistics.csv` | Exp 03 | CC200 | 534 | Derived | `6e82a9...` |
| `results/exp03/site_effect_anova.csv` | Exp 03 | CC200 | 534 | Derived | `b27c3d...` |
| `results/exp04/combat_harmonization_comparison.csv` | Exp 04 | CC200 | 534 | Derived | `295a0f...` |
| `results/exp05/dynamic_state_biomarkers.csv` | Exp 05 | CC200 | 534 | Derived | `7c41b8...` |
| `results/exp06/exp06_verified_results.json` | Exp 06 | AAL-116 | 955 | Verified | `8f3e21...` |
| `results/exp07/gcn_qgcnn_test_results.json` | Exp 07 | AAL-116 | 33 (Test) | Verified | `1b490f...` |
| `results/exp07/clean_cohort_lineage.csv` | Exp 07 | AAL-116 | 391 ($162 \subset 391$)| Provenance | `e60971...` |
| `results/exp08/baseline_model_results.json` | Exp 08 | 4D / CC200 | 626 / 764 | Verified | `4c82d3...` |
| `results/exp09/gnn_loso_fold_results.csv` | Exp 09 | CC200 | 497 (7 folds) | Derived | `1688b1...` |
| `results/exp09/classical_ml_canonical_results.csv` | Exp 09 | CC200 | 497 (42 models) | Derived | `4a9f3c...` |
| `results/exp09/classical_ml_family_summary.csv` | Exp 09 | CC200 | 497 (6 families) | Derived | `e2b804...` |
| `results/exp09/representation_pareto_analysis.csv` | Exp 09 | CC200 | 497 | Derived | `f9c182...` |
| `results/exp09/threshold_sweep_graph_statistics.csv`| Exp 09 | CC200 | 497 | Recovered | `0d8e41...` |

---

## 8. Known Uncertainties

1. **Selection of 162 Exp 07 Subjects**: While mathematically verified as a strict prefix of the 391 aligned Exp 06 cohort, the scientific decision rule used to select the integer 162 was not recovered.
2. **Threshold Optimization Rule**: The selection of `top_10pct` in Exp 09 survives as an empirical data point supported by diagnostic sweeps, but the algorithmic optimization script is not retained.
3. **Hardware Platform for Initial Checkpoints**: Exact physical GPU hardware and low-level driver versions utilized during the initial training of the Exp 07 classical/quantum checkpoints were not recorded.
