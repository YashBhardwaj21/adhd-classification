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
| **Exp 07** | Density 0.20, weighted node strength, 2 quantum layers | Executed config used nominal density $0.15$ (`>=` threshold), 1 quantum layer, 117 node features (116 FC + unweighted normalized degree), stored `edge_attr` not passed to `GCNConv`, 162 clean + 713 selected pseudo cohort (103/26/33 clean split; 816 train) | Documented / Archived |
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
- **Clinical Variance Attenuation**: Empirical Bayes ComBat reduced scanner identification accuracy from $58.64\%$ to $31.29\%$, while diagnostic classification accuracy also decreased from $64.53\%$ to $59.56\%$. This reflects that diagnostic status was non-uniformly distributed across collection sites, leading ComBat to adjust partially for site-associated clinical variance.

### Experiment 05: Data-Driven Dynamic Connectivity States
- **Cluster Number**: Analysis evaluated $K \in \{2, 3, 4, 5, 6\}$; $K=3$ was selected as the optimal partition by silhouette coefficient. State 0 exhibited the longest mean dwell time ($6.24$ windows) and highest occupancy ($49.05\%$).

### Experiment 06: Semi-Supervised Pseudo-Labeling
- **Cohort Decoupling**: Experiment 06 operated on a cohort of 955 subjects (391 clean-labelled and 564 unlabelled). Two procedural variants were executed: Procedure I generated 552 pseudo-labels, and Procedure II generated 484 pseudo-labels. This partition is historically decoupled from the 162 clean + 713 pseudo cohort utilized in Experiment 07.

### Experiment 07: Classical GCN and Quantum GCNN Evaluation

- **Required Provenance Statement**:
  "Experiment 7 evaluates Classical GCN and Quantum GCNN models on AAL-116 functional-connectivity graphs. The historical graph construction uses 116 ROIs and a 15% nominal density threshold based on the percentile of upper-triangle absolute FC values. Signed FC values are retained as edge attributes, while node features consist of the 116 signed FC values for each ROI plus normalized graph degree. The stored edge attributes are not passed as edge weights to the historical GCNConv layers."

- **Cohort Provenance**:
  - "Experiment 7 used a separately prepared cohort consisting of 162 clean labeled subjects and 713 additionally selected pseudo-labeled subjects."
  - "`part1.ipynb` is a collection of exploratory and comparative experiments rather than a single pseudo-label generator. Multiple candidate approaches were evaluated, after which a selected cohort was exported through the later `11_ensemble_labeling` production workflow. The resulting Experiment 7 input cohort is documented downstream as 162 clean labeled subjects and 713 selected pseudo-labeled subjects."
  - "The historical Experiment 7 cohort contains 162 clean labeled subjects and 713 additionally selected pseudo-labeled subjects. The pseudo-labeled subjects are added only to the training set; the validation and test sets contain clean labeled subjects."
  - Pseudo-label distribution (713 subjects): healthy = 535, ADHD = 178.
  - "The clean cohort was split using stratified train/test and train/validation splits with random_state=42, resulting in 103 training, 26 validation, and 33 test subjects."
  - Total training cohort: 103 clean + 713 pseudo = 816 training subjects.
  - Held-out test cohort: 33 clean subjects (clean labeled subjects only).
  - Reproducibility: "Random state 42 was used for the clean train/validation/test split. Full deterministic training reproducibility was not established from the historical trainer code." ("Random state 42 was verified for the clean data split. Full deterministic training reproducibility was not established from the historical checkpoint-producing trainer code.")

- **Input Representation & Graph Construction**:
  - Atlas: AAL-116 (116 ROIs, 6670 unique FC features: $116 \times 115 / 2 = 6670$).
  - Full vs Reduced Features: The full 6670-dimensional features (`X_combined_full.npy`) were used for graph reconstruction. The 2000-dimensional reduced representation (`X_combined_reduced.npy`) was not used for Exp 07 graph reconstruction.
  - Nominal density parameter: $\text{DENSITY} = 0.15$.
  - Thresholding definition: Computed on upper-triangle absolute FC values:
    `abs_fc = np.abs(fc_matrix)`
    `upper_vals = abs_fc[triu_idx]`
    `threshold = np.percentile(upper_vals, 100 * (1 - density))` (85th percentile).
  - Edge selection: `edge_mask = abs_fc >= threshold` (`>=` comparison; percentile ties can yield slightly more edges than the nominal 15%).
  - Edge attributes: `edge_attr = torch.tensor(fc_matrix[edge_mask], dtype=torch.float32)` retains original signed FC values.
  - Message-passing behavior: "edge_attr is stored in the graph data object but is not passed as edge weights to the historical GCNConv message-passing layers."
  - Node features: 117-dimensional: 116 signed FC row values + 1 normalized degree (`degree = np.sum(abs_fc >= threshold, axis=1, keepdims=True) / n_rois; node_features = np.hstack([fc_matrix, degree])`). The degree feature is based on unweighted thresholded adjacency, not weighted node strength. The inspected historical Exp07 graph-construction code does not apply a final z-score/StandardScaler transformation to the 117-dimensional node-feature matrix.

- **Model Architectures & Hyperparameters**:
  - Classical GCN: $117 \to 32 \to 32 \to 16 \to 2$ with BatchNorm1d, ReLU, Dropout($p=0.30$), and global mean pooling (5,522 parameters).
  - Quantum QGCNN: Classical projection ($117 \to 12$), 6-qubit quantum variational circuit ($N_{\text{qubits}}=6, N_{\text{layers}}=1$; $R_Y, R_Z$ angle encoding; 1 trainable layer with parameterized $R_X, R_Y, R_Z$ rotations and ring CNOT entanglement; 6 Pauli-$Z$ expectations), followed by 3 GCN layers ($6 \to 16 \to 16 \to 16 \to 2$) and global mean pooling (2,188 parameters).
  - Hyperparameters: `SEED=42`, `N_QUBITS=6`, `N_LAYERS=1`, `BATCH_SIZE=8`, `LEARNING_RATE=1e-3`, `WEIGHT_DECAY=1e-5`, `DENSITY=0.15`, `EPOCHS=20`, `PATIENCE=10`, `CHECKPOINT_INTERVAL=1`.

- **Checkpoints & Independent Verified Results**:
  - Historical checkpoint-producing code wrote:
    - Classical: `classical_checkpoint_epoch_{epoch}.pth`, `classical_best_model.pth`
    - Quantum: `quantum_checkpoint_epoch_{epoch}.pth`, `quantum_best_model.pth`
  - Historical checkpoint analysis evaluated `classical_best_model.pth` and `quantum_best_model.pth` on the 33 held-out test subjects.
  - **Classical GCN Test Results** (33 clean test samples):
    - AUC: 0.7293233082706767
    - Accuracy: 0.696969696969697
    - Weighted Precision: 0.6940836940836941
    - Weighted Recall: 0.696969696969697
    - Weighted F1: 0.6928904428904429
    - Confusion Matrix: `[[15, 4], [6, 8]]`
  - **Quantum GCNN Test Results** (33 clean test samples):
    - AUC: 0.6428571428571428
    - Accuracy: 0.6060606060606061
    - Weighted Precision: 0.6204322638146168
    - Weighted Recall: 0.6060606060606061
    - Weighted F1: 0.6082390727552018
    - Confusion Matrix: `[[11, 8], [5, 9]]`
  - Each model's metrics are reported independently without comparative ranking, tiers, or verdicts.
  - Stored in `results/exp07/checkpoint_analysis.json` and `results/exp07/report.md`.

- **Paper-vs-Code Discrepancies**:
  - Text referenced 0.20 density; historical code executed with nominal density 0.15.
  - Manuscript described weighted node strength; historical code used unweighted normalized degree.
  - Manuscript implied weighted convolutions; historical code did not pass `edge_attr` to `GCNConv`.
  - Manuscript described 2 quantum layers; historical code executed with 1 quantum layer.

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
| **Brain Connectivity Toolbox** | `src/exp02/null_model_und_sign_fixed.py`, `src/exp02/randmio_und_signed_fast.py` | Rubinov & Sporns (2011) / `bctpy` | GNU General Public License v3.0 or later (GPL-3.0-or-later) | Bundled algorithm implementation (see `third_party/bctpy.md`) |
| **NeuroSTORM Transformer** | `src/exp08/neurostorm/neurostorm.py` | CUHK-AIM-Group (derived from MONAI / SwiFT) | Apache License, Version 2.0 (Apache-2.0) | Bundled model implementation (see `NOTICE`) |
| **neuroCombat** | N/A (external package dependency) | Fortin et al. (2018) | MIT | Pip dependency; not bundled |

### Brain Connectivity Toolbox (BCT) Provenance

The historical implementation contains BCT-derived graph-randomization code. `null_model_und_sign_fixed.py` imports utilities and the `randmio_und_signed` routine from the BCTPY package, while `randmio_und_signed_fast.py` is a Numba-accelerated reimplementation of the BCTPY `randmio_und_signed` routine. The relevant historical source is therefore treated as BCT-derived/adapted code rather than as an independently authored graph-randomization algorithm.

Upstream project: aestrivex/bctpy (https://github.com/aestrivex/bctpy). The upstream BCTPY repository is GPL-3.0. The exact BCTPY version and upstream commit used during the historical experiment were not recorded in the available provenance evidence.

The historical `null_model_und_sign_fixed.py` file carries a Rubinov 2011 attribution for the undirected signed null model:
Mikail Rubinov and Olaf Sporns. "Weight-conserving characterization of complex functional brain networks." *NeuroImage*, 2011; 56(4): 2068–2079. DOI: 10.1016/j.neuroimage.2011.03.069.

See [third_party/bctpy.md](../third_party/bctpy.md) for the complete third-party provenance document and GPL-3.0-or-later licensing notices.
