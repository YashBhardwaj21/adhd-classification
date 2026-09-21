# Provenance, Discrepancies & Execution Ledger

This document provides a factual record of discrepancies between manuscript descriptions, early project notes, and the executed code artifacts across the study:
*ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical–Quantum Graph Learning, and Cross-Site Generalization*  
**Authors**: Shantanu Khare, Bhavana Deepthi, Yash Bhardwaj (Vellore Institute of Technology).  
**Status**: Manuscript in preparation (2026).

---

## 1. Executive Discrepancy Table

| Experiment | Manuscript / Initial Description | Executed Code Artifact | Status |
| :---: | :--- | :--- | :---: |
| **Exp 01** | Static vs dynamic FC stability comparison | Diagonal zeroed before distance computation; lag correlation decay $0.8990 \to 0.5467$ | Documented |
| **Exp 02** | MST + 20% proportional thresholding | `mst_graph` in notebook implements MST + PT ($\rho=0.20$); `src/exp02/graph_utils.py` provides simplified PT | Documented |
| **Exp 03** | Diagnostic group separation analysis | One-way ANOVA $F$-tests capture inter-site scanner variation ($F > 4,000$), not clinical separation | Documented |
| **Exp 04** | ComBat removes batch effects while retaining signal | Scanner accuracy drops $58.6\% \to 31.3\%$, but diagnostic accuracy also drops $64.5\% \to 59.6\%$ | Documented |
| **Exp 05** | Dynamic micro-state clustering | $K=3$ selected by silhouette analysis; State 0 dominates dwell time ($6.24$ windows, $49.05\%$ occupancy) | Documented |
| **Exp 06** | Semi-supervised pseudo-labelling | Procedure I yielded 552 pseudo-labels; Procedure II yielded 484; independent cohort from Exp 07 | Documented |
| **Exp 07** | Density 0.20, weighted node strength, 2 quantum layers | Executed config used density $0.15$, 1 quantum layer, normalized degree feature; inputs unavailable | Documented / Archived |
| **Exp 08** | Lightweight 3D CNN on structural T1 scans | CNN evaluated on functional 4D BOLD volume sequences ($T=25, 99 \times 117 \times 95$), not T1 scans | Documented |
| **Exp 09** | Unweighted MST + 20% distance mapping | Executed notebook used `top_10pct` thresholding with weighted edges and self-loops | Documented |

---

## 2. Detailed Experiment Discrepancies

### Experiment 01: Dynamic FC Stability
- **Diagonal Handling**: The executed notebook (`notebooks/exp01/01_fc_generation_and_validation.ipynb`) zeroed the diagonal of correlation matrices before computing Pearson similarities and Frobenius distances across sliding windows.
- **Numbers**: Monotonic correlation decay ($0.8990 \to 0.7789 \to 0.6609 \to 0.5467$) and Frobenius distance progression ($32.58 \to 48.75 \to 60.78 \to 70.61$) match the stored artifact `results/exp01/dynamic_temporal_validation.csv`.

### Experiment 02: CC200 Graph Construction
- **Algorithm**: The executed notebook (`notebooks/exp02/02_graph_construction_and_validation.ipynb`, cell 71) implemented `mst_graph(fc, density=0.20)` by first extracting a Minimum Spanning Tree on distance matrix $D = 1 - |r|$ and then adding the highest absolute correlation edges until reaching 3,591 edges (20% density for 190 nodes).
- **Utility vs Notebook**: `src/exp02/graph_utils.py` contains a standalone PyG graph generation utility that performs proportional thresholding without the initial MST pass. Topological metrics reported in `results/exp02/graph_metrics.csv` were computed from the notebook implementation.

### Experiment 03: Cross-Site Scanner ANOVA
- **Scope of $F$-Statistics**: The large $F$-statistics reported ($F > 4,000, p < 10^{-300}$ for global efficiency and path length) quantify variance attributable to acquisition site (scanner differences across 8 participating clinics), **not** diagnostic variance between ADHD and typically developing controls (TDC).

### Experiment 04: ComBat Harmonization
- **Clinical Variance Attenuation**: Empirical Bayes ComBat successfully reduced scanner identification accuracy ($58.64\% \to 31.29\%$), but diagnostic classification accuracy also decreased from $64.53\%$ to $59.56\%$. This reflects that diagnostic status was non-uniformly distributed across collection sites, leading ComBat to adjust partially for site-associated clinical variance.

### Experiment 05: Micro-State Clustering
- **Cluster Number**: Analysis evaluated $K \in \{2, 3, 4, 5, 6\}$; $K=3$ was selected as the optimal partition by silhouette coefficient. State 0 exhibited the longest mean dwell time ($6.24$ windows) and highest occupancy ($49.05\%$).

### Experiment 06: Semi-Supervised Pseudo-Labeling
- **Cohort Decoupling**: Experiment 06 operated on a cohort of 955 subjects (391 clean-labelled and 564 unlabelled). Two procedural variants were executed: Procedure I generated 552 pseudo-labels, and Procedure II generated 484 pseudo-labels. This partition is historically decoupled from the 162 clean + 713 pseudo cohort utilized in Experiment 07.

### Experiment 07: Classical GCN vs Quantum QGCNN
- **Density Correction**: Early project notes referenced a graph density of $0.20$. Historical configuration records verify that the reported experiment was executed with `DENSITY = 0.15` (strongest 15% absolute FC edges, signed weights, no MST).
- **Architecture Parameters**: The reported quantum model used 1 variational layer (`n_layers: 1`), 6 qubits (`n_qubits: 6`), batch size 8 (`batch_size: 8`), and weight decay $10^{-5}$ (`weight_decay: 0.00001`), as recorded in `configs/exp07/reported_run.json`.
- **Node Feature Construction**: The historical executed code constructed the 117th node feature as a normalized thresholded-edge degree (`np.sum(edge_mask, axis=1) / (n_rois - 1)`), whereas early draft text described a weighted node strength. The historical implementation is preserved in `src/exp07/utils/graph_utils.py` without rewriting it to match the paper description.
- **Quantum Circuit Gate Ordering**: The variational layer uses parameterized single-qubit rotations `qml.RX`, `qml.RY`, `qml.RZ` followed by ring `qml.CNOT` entanglement, rather than generic `qml.Rot` abstractions.
- **Metric Conventions**: Precision ($0.6941$ Classical, $0.6204$ Quantum), recall ($0.6970$ Classical, $0.6061$ Quantum), and F1 ($0.6929$ Classical, $0.6082$ Quantum) reported in `results/exp07/checkpoint_analysis.json` represent the weighted-average convention across classes.
- **Execution & Reproduction Status**: The original large combined arrays (`X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy`) and trained checkpoint weights exceed public repository limits and are unavailable here. The reported test evaluation is archived; the experiment cannot currently be rerun end-to-end from scratch.

### Experiment 08: Deep-Learning Baselines
- **Input Modality**: The Lightweight 3D CNN was evaluated on 4D functional BOLD volume sequences ($T=25, 99 \times 117 \times 95$), **not** structural T1 anatomical scans.
- **Subject-Level Partitioning**: Temporal graph learning was evaluated on strictly disjoint subject partitions (534 train, 115 validation, 115 test), disproving conjectures of sliding-window data leakage across splits.

### Experiment 09: Leave-One-Site-Out Population Graphs
- **Graph Construction Protocol**: The executed notebook (`notebooks/exp09/11_population_graph_learning.ipynb`) constructed subject graphs using `top_10pct` thresholding with weighted edges and self-loops, rather than the unweighted MST + 20% distance mapping mentioned in manuscript text.

---

## 3. Seed & Determinism Provenance

- **Dataset Split Seed**: Random seed `42` is verified as the deterministic seed for the 162 clean-subject partition (103 train / 26 val / 33 test) in Experiment 07.
- **Global RNG Initialization**: While `seed = 42` was specified across configurations, global random number generator state across GPU CUDA kernels and PennyLane quantum simulator backends was not independently verified for every historical run. Deterministic bitwise replication of trained weights cannot be asserted without the original execution environment.

---

## 4. Removed Legacy Files & File Hash Ledger

Prior to repository cleanup, several alternative and legacy implementation variants were present in the tree. To maintain a lean research repository, these variants have been removed from the public working tree. Their SHA256 hashes are recorded below for provenance:

| File Path (Prior to Removal) | SHA256 Hash | Historical Role |
| :--- | :--- | :--- |
| `src/exp07/quantum_models/quantum_embedding_vectorized.py` | `3CA08DF107479CC8C197E68A40286394B20B58AF1354F6C62001CE31D1153CE0` | Legacy standalone implementation variant |
| `src/exp07/quantum_models/quantum_embedding_vmap.py` | `009EBC8DA049B69137E67AD3B154E820CA9F5379FE3C414470664D697CA5A797` | Experimental Jax/vmap quantum circuit variant |
| `src/exp07/quantum_models/resume.py` | `FC2803A39EEADAD35A98FD3BB7154E502E67301471EC0AEF66B2863EFB995303` | Training checkpoint resumption utility |
| `scripts/clean_notebooks.py` | `6A01A093DE8D3FF3690B8100529D7D62FBBEB93375815610DB7023BF4DDE4BFD` | Maintenance utility for notebook output stripping |
| `scripts/fix_markdown_links.py` | `28F0FA1DF334A2E7ED4965377DA9CEAFE91A6E77A8EEAE49D6A1296C8395BBA4` | Maintenance utility for markdown link validation |

### Canonical Exp 07 File Hashes (Retained)

| File Path | SHA256 Hash |
| :--- | :--- |
| `src/exp07/classical_models/model_classical_gcn.py` | `903D570CC18B095E82A1AE79BA237D60B5911BA578064C769EE7C0E800663919` |
| `src/exp07/classical_models/train_classical_gcn.py` | `579F1B8C22BB403E6710371CA48B87A55E7DD43DB2041172C7401127AEE9D877` |
| `src/exp07/classical_models/run_experiment.py` | `F63C5FBEA7EF613836AA7D5FB9DF2AD6095D876A0F4AFEBFD2A55F0C00C0714F` |
| `src/exp07/quantum_models/quantum_embedding_broadcast.py` | `5A09340DD92E0916DCFB8C0D7E78069FCEC31B535AC1814CF02B402B93098C40` |
| `src/exp07/quantum_models/train_qgcnn_vectorized.py` | `256CFFABA44BF9D9388788519096BB6495F34B68C8E839A48B6D35B771E8E67F` |
| `src/exp07/quantum_models/run_experiment.py` | `E796EA1CEEB80661191C468D1D8A5287F9C4C85396E122B6888A98C7EFB7C331` |
| `src/exp07/utils/config.py` | `27F9F9995DB1BE05185C2A3C39F494177958025BC13786AC7CE753F06ED3C164` |
| `src/exp07/utils/data_loader.py` | `A8AB8FB7FF3AF68DDC5E0CCC69CA5E9B2EC270AD87E13C56CCFA5494E1F01E4E` |
| `src/exp07/utils/graph_utils.py` | `056CB5C3480B47097A46EE2D03AFCA05266A652474CB3F8F8CEA0C732164D4BD` |
| `src/exp07/utils/training_utils.py` | `52197F53F2DE0468805E6F105F3EA7ACF7584B3C1083F14AC24213820F9335E0` |

---

## 5. Third-Party Code & Licensing Determinations

| Bundled Component | File Path | Origin / Upstream | Upstream License | Status in Repository |
| :--- | :--- | :--- | :--- | :--- |
| **Brain Connectivity Toolbox** | `src/exp02/null_model_und_sign_fixed.py`, `src/exp02/randmio_und_signed_fast.py` | Rubinov & Sporns (2011) / `bctpy` | GNU GPL v3.0 | Bundled algorithm implementation |
| **NeuroSTORM Transformer** | `src/exp08/neurostorm/neurostorm.py` | CUHK-AIM-Group (derived from MONAI / SwiFT) | Apache License 2.0 | Bundled model implementation |
| **neuroCombat** | N/A (external package dependency) | Fortin et al. (2018) | MIT | Pip dependency; not bundled |
