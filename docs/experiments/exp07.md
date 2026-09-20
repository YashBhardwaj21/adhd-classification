# Experiment 7: Quantum vs Classical Graph Convolutional Networks

## 1. Scientific Overview & Objectives
Experiment 7 evaluates whether parameterizing Graph Convolutional Networks (GCN) with Variational Quantum Circuits (VQC) provides inductive advantage or expressive compression for psychiatric neuroimaging classification.

The core experimental questions are:
1. Can a 6-qubit parameterized quantum embedding compress 117-dimensional brain regional node features into a 6-dimensional Hilbert space representation without catastrophic loss of diagnostic signal?
2. Does a hybrid Quantum Graph Convolutional Neural Network (QGCNN) match or exceed the generalization performance of an equivalent classical GCN on out-of-sample test subjects?

---

## 2. Parcellation & Graph Construction
- **Atlas**: Automated Anatomical Labeling (AAL-116), defining $N = 116$ cortical and subcortical regions.
- **Node Feature Construction**:
  Each node $i \in \{1, \dots, 116\}$ is represented by a **117-dimensional feature vector**:
  - $116$ Pearson functional connectivity correlation values with all other ROIs in the network.
  - $1$ normalized node degree feature measuring local topological centrality.
  $$\mathbf{x}_i = [\mathbf{C}_{i, 1}, \mathbf{C}_{i, 2}, \dots, \mathbf{C}_{i, 116}, \text{deg}(i) / (N-1)]^T \in \mathbb{R}^{117}$$
- **Edge Definition**: Proportional thresholding at density $\rho = 0.20$ based on absolute correlation magnitude $|\mathbf{C}_{i,j}|$.

---

## 3. Network Architectures

### 3.1 Classical GCN Baseline
- **Input**: Node features $\mathbf{X} \in \mathbb{R}^{116 \times 117}$, edge index $\mathbf{E} \in \mathbb{R}^{2 \times E}$.
- **Layer 1**: $\text{GCNConv}(117 \to 32) \to \text{BatchNorm1d}(32) \to \text{ReLU} \to \text{Dropout}(0.30)$
- **Layer 2**: $\text{GCNConv}(32 \to 32) \to \text{BatchNorm1d}(32) \to \text{ReLU} \to \text{Dropout}(0.30)$
- **Layer 3**: $\text{GCNConv}(32 \to 16) \to \text{ReLU}$
- **Readout**: $\text{GlobalMeanPool}(\cdot) \to \text{Linear}(16 \to 2)$
- **Total Parameters**: ~5,842 learnable parameters.

### 3.2 Hybrid Quantum GCNN (6 Qubits)
- **Classical Projection**: $\text{Linear}(117 \to 12)$ projecting into $2 \times N_{\text{qubits}}$ rotation parameters $[\boldsymbol{\theta}, \boldsymbol{\phi}]$.
- **Quantum Variational Circuit**:
  - 6 qubits initialized in state $|0\rangle^{\otimes 6}$.
  - State preparation via angle encoding: $R_Y(\theta_q) R_Z(\phi_q)$ for $q \in \{1, \dots, 6\}$.
  - 2 strongly entangling layers: circular CNOT gates followed by parameterized rotations $U(\alpha, \beta, \gamma)$.
  - Measurement: Expectation values of Pauli-$Z$ operators: $\langle Z_q \rangle \in [-1, 1]$.
  - PennyLane native broadcasting: Evaluates all 116 nodes in a single batched quantum circuit call.
- **Graph Convolution Backbone**:
  - **Layer 1**: $\text{GCNConv}(6 \to 16) \to \text{BatchNorm1d}(16) \to \text{ReLU} \to \text{Dropout}(0.30)$
  - **Layer 2**: $\text{GCNConv}(16 \to 16) \to \text{BatchNorm1d}(16) \to \text{ReLU} \to \text{Dropout}(0.30)$
  - **Layer 3**: $\text{GCNConv}(16 \to 16) \to \text{ReLU}$
  - **Readout**: $\text{GlobalMeanPool}(\cdot) \to \text{Linear}(16 \to 2)$

---

## 4. Cohort Partitioning & Verification
- **Total Dataset**: 875 subjects
  - Clean Labeled Subjects: 162 (93 TDC, 69 ADHD)
  - Pseudo-Labeled Subjects: 713 (535 TDC, 178 ADHD)
- **Held-Out Test Set**: **33 subjects** strictly drawn from the clean-labeled cohort (19 TDC, 14 ADHD) and held out from all training, pseudo-labeling, and validation.

---

## 5. Audited Quantitative Results
Traced directly to [`results/exp07/checkpoint_analysis.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp07/checkpoint_analysis.json) and [`results/exp07/report.md`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp07/report.md) ($N = 33$ test subjects):

| Model Architecture | Test AUC | Accuracy | Precision | Recall | F1 Score | Confusion Matrix [TDC, ADHD] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical GCN** | **$0.7293$** | **$69.70\%$** ($23/33$) | $0.6941$ | $0.6970$ | **$0.6929$** | $\begin{bmatrix} 15 & 4 \\ 6 & 8 \end{bmatrix}$ |
| **Quantum GCNN (6 qubits)** | **$0.6429$** | **$60.61\%$** ($20/33$) | $0.6204$ | $0.6061$ | **$0.6082$** | $\begin{bmatrix} 14 & 5 \\ 8 & 6 \end{bmatrix}$ |

### Key Findings
1. **Classical Superiority**: The Classical GCN outperforms the Quantum GCNN by **$+0.0865$ AUC** ($0.7293$ vs $0.6429$) and **$+9.09\%$ accuracy** ($69.70\%$ vs $60.61\%$).
2. **Quantum Information Bottleneck**: Compressing 117-dimensional continuous FC features through a 6-qubit quantum bottleneck produces a performance penalty, demonstrating that NISQ-scale variational circuits face representational constraints on dense continuous neuroimaging connectomes.

---

## 6. Execution Utilities & Staged Code
To ensure reproducibility and standalone execution, the required utility dependencies are staged under `src/exp07/utils/`:
- `src/exp07/utils/config.py`: Architecture hyperparams ($N=116, D=117, Q=6$).
- `src/exp07/utils/data_loader.py`: Dataset loader supporting the 875-subject Track B cohort.
- `src/exp07/utils/graph_utils.py`: Converts correlation matrices into PyG graphs with degree features.
- `src/exp07/utils/training_utils.py`: Epoch training loop, validation metrics, and timed checkpointing.

---

## 7. Source Code & Result Provenance
- **Model Scripts**:
  - [`src/exp07/classical_models/model_classical_gcn.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/classical_models/model_classical_gcn.py)
  - [`src/exp07/classical_models/train_classical_gcn.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/classical_models/train_classical_gcn.py)
  - [`src/exp07/quantum_models/train_qgcnn_vectorized.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/quantum_models/train_qgcnn_vectorized.py)
  - [`src/exp07/quantum_models/quantum_embedding_broadcast.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/quantum_models/quantum_embedding_broadcast.py)
  - [`src/exp07/quantum_models/resume.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp07/quantum_models/resume.py)
- **Result Artifacts**:
  - [`results/exp07/checkpoint_analysis.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp07/checkpoint_analysis.json)
  - [`results/exp07/report.md`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp07/report.md)
