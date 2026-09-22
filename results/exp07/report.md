# Experiment 7: Classical GCN and Quantum GCNN Evaluation

Experiment 7 evaluates Classical GCN and Quantum GCNN models on AAL-116 functional-connectivity graphs. The historical graph construction uses 116 ROIs and a 15% nominal density threshold based on the percentile of upper-triangle absolute FC values. Signed FC values are retained as edge attributes, while node features consist of the 116 signed FC values for each ROI plus normalized graph degree. The stored edge attributes are not passed as edge weights to the historical GCNConv layers.

---

## 1. Cohort and Data Split Provenance

Experiment 7 used a separately prepared cohort consisting of 162 clean labeled subjects and 713 additionally selected pseudo-labeled subjects.

`part1.ipynb` is a collection of exploratory and comparative experiments rather than a single pseudo-label generator. Multiple candidate approaches were evaluated, after which a selected cohort was exported through the later `11_ensemble_labeling` production workflow. The resulting Experiment 7 input cohort is documented downstream as 162 clean labeled subjects and 713 selected pseudo-labeled subjects.

The historical Experiment 7 cohort contains 162 clean labeled subjects and 713 additionally selected pseudo-labeled subjects. The pseudo-labeled subjects are added only to the training set; the validation and test sets contain clean labeled subjects.

The clean cohort was split using stratified train/test and train/validation splits with random_state=42, resulting in 103 training, 26 validation, and 33 test subjects.

- **Clean Cohort (162 subjects)**:
  - First split: `test_size = 0.20`, `random_state = 42`, `stratify = labels`
  - Second split: `test_size = 0.20`, `random_state = 42`, `stratify = labels`
  - Split counts: 103 train, 26 validation, 33 held-out test
- **Pseudo-Labeled Cohort (713 subjects)**:
  - Pseudo-label distribution: 535 healthy (TDC), 178 ADHD
  - Allocated strictly to training
- **Final Partition Composition**:
  - Training: 816 subjects (103 clean + 713 pseudo-labeled)
  - Validation: 26 clean subjects
  - Held-out Test: 33 clean subjects (19 TDC, 14 ADHD)

Random state 42 was verified for the clean data split. Full deterministic training reproducibility was not established from the historical checkpoint-producing trainer code.

---

## 2. Graph Representation and Edge Attribute Handling

- **Atlas**: AAL-116 ($N=116$ regions of interest).
- **Input Features**: Full 6670-dimensional upper-triangular FC correlation features ($116 \times 115 / 2 = 6670$). Graph reconstruction uses `X_combined_full.npy`, reconstructing a symmetric $116 \times 116$ signed correlation matrix.
- **Thresholding**: Nominal 15% density threshold computed on upper-triangle absolute FC values:
  `abs_fc = np.abs(fc_matrix)`
  `threshold = np.percentile(upper_vals, 100 * (1 - density))` (85th percentile).
  Edge selection rule: `edge_mask = abs_fc >= threshold` (with `>=`, ties may produce more edges than the nominal target).
- **Edge Attributes**: Stored edges retain original signed FC values: `edge_attr = torch.tensor(fc_matrix[edge_mask], dtype=torch.float32)`.
- **Message-Passing Behavior**: `edge_attr` is stored in the graph data object but is not passed as edge weights to the historical GCNConv message-passing layers.
- **Node Features**: 117-dimensional features consisting of 116 signed FC values + 1 normalized degree:
  `degree = np.sum(abs_fc >= threshold, axis=1, keepdims=True) / n_rois`
  `node_features = np.hstack([fc_matrix, degree])`.
  The degree feature is based on unweighted thresholded adjacency, not weighted node strength. No final z-score standardization was applied to the 117-dimensional node feature matrix.

---

## 3. Checkpoint and Evaluation Provenance

Historical checkpoint-producing code wrote:
- Classical: `classical_checkpoint_epoch_{epoch}.pth`, `classical_best_model.pth`
- Quantum: `quantum_checkpoint_epoch_{epoch}.pth`, `quantum_best_model.pth`

Historical checkpoint analysis evaluated `classical_best_model.pth` and `quantum_best_model.pth` on the 33 held-out test subjects. The verified results below come directly from that checkpoint analysis (`checkpoint_analysis.json`).

### 3.1 Classical GCN Test Results

- **Architecture**: 3-layer GCN ($117 \to 32 \to 32 \to 16 \to 2$) with BatchNorm1d, ReLU, Dropout(0.30), and global mean pooling (5,522 parameters).
- **Test Samples**: 33 clean subjects
- **AUC**: 0.7293233082706767
- **Accuracy**: 0.696969696969697
- **Weighted Precision**: 0.6940836940836941
- **Weighted Recall**: 0.696969696969697
- **Weighted F1 Score**: 0.6928904428904429
- **Confusion Matrix**:
  ```text
  [[15, 4],
   [ 6, 8]]
  ```

### 3.2 Quantum GCNN Test Results

- **Architecture**: Classical projection ($117 \to 12$), 6-qubit quantum embedding ($R_Y, R_Z$ angle encoding; 1 trainable layer with sequential $R_X, R_Y, R_Z$ rotations and ring CNOT entanglement; 6 Pauli-$Z$ expectations), followed by 3 GCNConv layers ($6 \to 16 \to 16 \to 16 \to 2$) and global mean pooling (2,188 parameters).
- **Test Samples**: 33 clean subjects
- **AUC**: 0.6428571428571428
- **Accuracy**: 0.6060606060606061
- **Weighted Precision**: 0.6204322638146168
- **Weighted Recall**: 0.6060606060606061
- **Weighted F1 Score**: 0.6082390727552018
- **Confusion Matrix**:
  ```text
  [[11, 8],
   [ 5, 9]]
  ```

---

## 4. Paper-vs-Code Discrepancy Statement

| Attribute | Manuscript / Early Description | Historical Executed Code | Verified Artifact Status |
| :--- | :--- | :--- | :--- |
| **Atlas & ROIs** | AAL-116 (116 regions) | AAL-116 (116 regions) | Matched |
| **FC Dimension** | 6670 unique features | `X_combined_full.npy` (6670 features) | Matched (not reduced 2000) |
| **Density Parameter** | Text referenced 0.20 density | `DENSITY = 0.15` (85th percentile threshold) | Executed code used 0.15 |
| **Edge Selection** | Exact percentage implied | `abs_fc >= threshold` (ties may exceed 15%) | Preserved in `graph_utils.py` |
| **Edge Attributes** | Signed weights described | Stored in `edge_attr` | Stored in Data object |
| **GCN Message Passing** | Weighted convolutions implied | `edge_attr` not passed to `GCNConv` | Forward pass uses unweighted edges |
| **Node Features** | Weighted node strength | Normalized thresholded degree (`degree / 116`) | Executed code used degree |
| **Quantum Layers** | 2 variational layers | `n_layers = 1` (1 variational layer, 18 weights) | Recorded in config and model |
| **Clean Split** | 162 subjects | 103 train / 26 val / 33 test (`seed=42`) | Deterministic split verified |
| **Pseudo-Labels** | Single procedure implied | 713 selected subjects added to train only | Historical selected cohort |
| **Metric Reporting** | Class-level metrics | Weighted-average precision, recall, and F1 | Recorded in `checkpoint_analysis.json` |
