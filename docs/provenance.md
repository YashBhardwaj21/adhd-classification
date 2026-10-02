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

The canonical artifact manifest ([`results/manifest.csv`](../results/manifest.csv)) indexes all 31 retained canonical results with exact SHA-256 cryptographic checksums:

| Retained Artifact | Originating Experiment | Dataset / Atlas | Sample Size ($N$) | Evidentiary Status | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`results/exp01/temporal_similarity_validation.csv`](../results/exp01/temporal_similarity_validation.csv) | EXP01 | CC200 | 764 | Derived | `5b626522f34c2a6ec29ed3f98e6a8f1d88cd70f2b013908b5134dcb41bcadf64` |
| [`results/exp01/static_dynamic_fc_comparison.csv`](../results/exp01/static_dynamic_fc_comparison.csv) | EXP01 | CC200 | 764 | Derived | `505e0ac4c9d100ea104a97caf0f0999a1475bb5c18eae0267ac46c9994f949c5` |
| [`results/exp02/graph_metrics_window_level.csv`](../results/exp02/graph_metrics_window_level.csv) | EXP02 | CC200 | 764 | Derived | `d03a3ffc5b0a36a1720217678375fab0c35afb8af5960ca2b017196b94bb0275` |
| [`results/exp02/graph_metrics_subject_level.csv`](../results/exp02/graph_metrics_subject_level.csv) | EXP02 | CC200 | 764 | Derived | `644f63b8614cd01ba3c880466450041b8fe8ac7b5b2c18e96320212627b58da1` |
| [`results/exp03/dynamic_feature_statistics.csv`](../results/exp03/dynamic_feature_statistics.csv) | EXP03 | CC200 | 764 | Derived | `04d57c17e8618fc832582f3e164df9b4a391e82bf3be89c8d60359efdb91a674` |
| [`results/exp03/site_effect_anova.csv`](../results/exp03/site_effect_anova.csv) | EXP03 | CC200 | 764 | Derived | `29edeaf561d3643dbec39bccbd5df284d18af6ecec4e22ec41dfe26a1e47ee35` |
| [`results/exp04/combat_harmonization_comparison.csv`](../results/exp04/combat_harmonization_comparison.csv) | EXP04 | CC200 | 764 | Derived | `18da724e353968e7e9997a8abebf27121143d621a73eaeeae254f5d4e348fa5f` |
| [`results/exp04/site_prediction_combat_comparison.csv`](../results/exp04/site_prediction_combat_comparison.csv) | EXP04 | CC200 | 764 | Derived | `ceb3024ada226449f7c4b0327b0cd0c36bdee11f9f66de2f6db857511a9da37c` |
| [`results/exp04/diagnosis_prediction_combat_comparison.csv`](../results/exp04/diagnosis_prediction_combat_comparison.csv) | EXP04 | CC200 | 764 | Derived | `7445f7928ca9855728e9fe0f4cabc89d9c4dce2aa88ffe90989a0537ecc9a002` |
| [`results/exp05/dynamic_state_biomarkers.csv`](../results/exp05/dynamic_state_biomarkers.csv) | EXP05 | CC200 | 764 | Derived | `3f3523b67a3527b3a0305706e37bb54b727a1bf529930e3cd0bf1f27ebc2c087` |
| [`results/exp05/dynamic_state_sequences.csv`](../results/exp05/dynamic_state_sequences.csv) | EXP05 | CC200 | 764 | Derived | `973ae8486c92e397ccdc3bd94858f150825d0405f98cfcbc041460beb8f040e8` |
| [`results/exp06/exp06_verified_results.json`](../results/exp06/exp06_verified_results.json) | EXP06 | AAL-116 | 955 | Reported | `1e35a8d600644d71c0b4ce0ef0a97d498b65bd8b975403ccf7015d98fd5f54ef` |
| [`results/exp07/gcn_qgcnn_test_results.json`](../results/exp07/gcn_qgcnn_test_results.json) | EXP07 | AAL-116 | 33 | Archived | `a3a9582dd9d8fa6bd607b8ca098a80bef1f2068f2d6ffaee59be49e1dba8997f` |
| [`results/exp07/experiment_report.md`](../results/exp07/experiment_report.md) | EXP07 | AAL-116 | 33 | Archived | `4f1fa1aaad2d7686045beb7ca15c31aa9793e8af23b0305197dffb65992bb512` |
| [`results/exp07/pseudolabel_provenance.json`](../results/exp07/pseudolabel_provenance.json) | EXP07 | AAL-116 | 875 | Reported | `dc911f747f4dd498547de3005914439ba1923424f6f144d518d58d9a3640feff` |
| [`results/exp07/clean_cohort_lineage.csv`](../results/exp07/clean_cohort_lineage.csv) | EXP07 | AAL-116 | 391 | Provenance | `e60971bc3a90c5d50696b3d94de4b99a17a36fdc65866b918fa8b3fcca54b630` |
| [`results/exp08/baseline_model_results.json`](../results/exp08/baseline_model_results.json) | EXP08 | ADHD-200 (Volumetric) | 626 | Reported | `46c22ef7b319099944f88f9b599c0dcff07757cf5fa67fd72c6bc6e4754b4a2d` |
| [`results/exp08/baseline_model_results.json`](../results/exp08/baseline_model_results.json) | EXP08 | CC200 (Temporal) | 764 | Reported | `46c22ef7b319099944f88f9b599c0dcff07757cf5fa67fd72c6bc6e4754b4a2d` |
| [`results/exp09/gnn_loso_fold_results.csv`](../results/exp09/gnn_loso_fold_results.csv) | EXP09 | CC200 | 497 | Derived | `539afa592b37dd42c1651e81bac2d8c17c667a13ff80f4ca3c1f5a34efe3ea0a` |
| [`results/exp09/classical_ml_canonical_results.csv`](../results/exp09/classical_ml_canonical_results.csv) | EXP09 | CC200 | 497 | Derived | `2db4bfda51d76d7dbfcc6be309930151462c7b0811f30f686c82db4e8b06fddd` |
| [`results/exp09/classical_ml_historical_results.csv`](../results/exp09/classical_ml_historical_results.csv) | EXP09 | CC200 | 497 | Historical | `eb378818ee737e13e71f6c4b8552f06470ee7c08af88d6eda7d5ca891ea52a38` |
| [`results/exp09/classical_ml_family_summary.csv`](../results/exp09/classical_ml_family_summary.csv) | EXP09 | CC200 | 497 | Derived | `b0ee621823e1be30b1b68d9999c91aaa300d5c3639da7447a4d04e1673e16b33` |
| [`results/exp09/representation_pareto_analysis.csv`](../results/exp09/representation_pareto_analysis.csv) | EXP09 | CC200 | 497 | Derived | `048a0ff0e3a8a4040d58bc510608e13f4cc58859f38e567ef37208858e1ac97d` |
| [`results/exp09/graph_preprocessing_summary.csv`](../results/exp09/graph_preprocessing_summary.csv) | EXP09 | CC200 | 497 | Derived | `2ab69569a22bba5603b0e6d329893ac75f18cd972623bda798d8e22481f9f8ea` |
| [`results/exp09/threshold_sweep_graph_statistics.csv`](../results/exp09/threshold_sweep_graph_statistics.csv) | EXP09 | CC200 | 497 | Historical | `cd0a714e43f4ce905268754b17a33d855bb316cc7a9b948b1dc7207180de0512` |
| [`results/exp09/threshold_positive_fc_diagnostic.csv`](../results/exp09/threshold_positive_fc_diagnostic.csv) | EXP09 | CC200 | 497 | Historical | `ae4b6153d370c1af73fc734e8e6a84c3afe5a485d852cf3e0668370a61510dc3` |
| [`results/exp09/threshold_absolute_fc_diagnostic.csv`](../results/exp09/threshold_absolute_fc_diagnostic.csv) | EXP09 | CC200 | 497 | Historical | `a024403690ef1dd198441d3e22dfce382f4071f7ebcbb7aca29320931d3be662` |
| [`results/exp09/gnn_loso_pairwise_tests.csv`](../results/exp09/gnn_loso_pairwise_tests.csv) | EXP09 | CC200 | 497 | Historical | `9dbe6cdd7aed848a253d2ecfca5963be926bb94d00ddd87273ff4f845240b532` |
| [`results/exp09/gnn_loso_summary_statistics.csv`](../results/exp09/gnn_loso_summary_statistics.csv) | EXP09 | CC200 | 497 | Historical | `7418c381bd946b8c36b52f4542d4df0f5c1b20539ed3706aae56eb7458167afe` |
| [`results/exp09/gnn_loso_error_analysis.csv`](../results/exp09/gnn_loso_error_analysis.csv) | EXP09 | CC200 | 497 | Historical | `7d87a833b1b146d876113f96713427dda9a907d860584d9f2d7d9566f0f91f2d` |
| [`results/exp09/exp09_verified_results.json`](../results/exp09/exp09_verified_results.json) | EXP09 | CC200 | 497 | Derived | `04919ab176839105a044ab44da3f46da87fc365fc1f37a49e85f7aa5ee6a0279` |

---

## 8. Known Uncertainties

1. **Selection of 162 Exp 07 Subjects**: While mathematically verified as a strict prefix of the 391 aligned Exp 06 cohort, the scientific decision rule used to select the integer 162 was not recovered.
2. **Threshold Optimization Rule**: The selection of `top_10pct` in Exp 09 survives as an empirical data point supported by diagnostic sweeps, but the algorithmic optimization script is not retained.
3. **Hardware Platform for Initial Checkpoints**: Exact physical GPU hardware and low-level driver versions utilized during the initial training of the Exp 07 classical/quantum checkpoints were not recorded.
