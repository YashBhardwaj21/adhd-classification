# Scientific Limitations & Future Research Directions

This document details the methodological constraints, statistical boundaries, and confounding factors of the study, and outlines concrete directions for future research.

---

## 1. Methodological & Empirical Limitations

### 1. Single-Configuration Quantum Comparison
The quantum-classical comparison in Experiment 07 is limited to a single architectural configuration: a 6-qubit parameterized quantum circuit with 1 variational layer and angle embedding, evaluated against a 3-layer isotropic GCN. The observed performance gap (Classical AUC 0.7293 vs Quantum AUC 0.6429) cannot be attributed definitively to intrinsic quantum expressive limits versus sub-optimal hyperparameter choices (such as learning rate, circuit depth, or optimizer selection).

### 2. Single Seed and Split
Experiment 07 was trained and evaluated on a single deterministic train/validation/test split (`seed=42`) with 33 held-out test subjects. Due to the computational cost of quantum state-vector simulation, multi-seed cross-validation was not performed. Consequently, variance across initialization seeds and partition splits remains unquantified for the quantum model.

### 3. Classical-to-Quantum Dimensional Bottleneck
To interface 117-dimensional node features with a 6-qubit quantum circuit, the model applies a linear dimensional projection ($117 \to 12$) followed by angle embedding onto 6 rotation angles. This severe dimensionality reduction constitutes a classical information bottleneck that may discard diagnostic variance prior to quantum state preparation.

### 4. Cohort and Label Alignment Ambiguity
While the 162 clean Exp 07 subjects are verified as an exact mathematical subset of the 391 aligned Exp 06 cohort ($162 \subset 391$), the historical rationale for filtering the cohort down to 162 subjects was not recovered from project archives.

### 5. Pseudo-Label Uncertainty & Downstream Protocol
In Experiment 06, semi-supervised self-training and ensemble consensus generated 552 and 484 pseudo-labels respectively on a 391-subject clean cohort. In Experiment 07, however, 713 pseudo-labels were combined with 103 clean training samples. The exact selection protocol linking Exp 06 candidates to the 713 downstream samples was not preserved.

### 6. Acquisition-Site Confounding
Cross-site scanner variations exert massive statistical effects on resting-state connectomes ($F > 4600$, $p < 10^{-15}$). Although ComBat harmonization reduces site prediction accuracy from 58.64% to 31.29% (balanced accuracy: 50.19% to 23.56%), residual site-specific noise and batch differences persist across international imaging centers.

### 7. Heterogeneous Cohorts, Atlases, and Representations
The study spans multiple atlases (CC200 with 190 ROIs, AAL-116 with 116 ROIs, and raw 4D voxel volumes) and distinct graph representations (MST+PT at density 0.20, proportional threshold at density 0.15, and top-10% positive thresholding). The three tracks are methodologically independent and must not be treated as a single unified benchmark.

### 8. Non-Nested GNN Representation Selection
In Experiment 09, preliminary node feature representation selection (comparing raw connectivity features, identity matrices, and degree strength) was conducted as an exploratory analysis across the full cohort rather than nested inside the cross-validation folds. In contrast, classical baseline feature selection was strictly nested.

### 9. Seven-Fold Statistical Power
Out-of-distribution evaluation in Experiment 09 relies on 7 site folds corresponding to the 7 available testing centers. With $K=7$ degrees of freedom ($\text{df}=6$), statistical power for detecting subtle differences between GNN architectures (e.g., GCN vs GAT, $p = 0.15625$) is constrained by the small number of clinical sites.

### 10. Sliding-Window Overlap & Autocorrelation
In Track A, dynamic functional connectivity is estimated via overlapping rectangular sliding windows ($W=30$ TRs, stride=5 TRs). Successive temporal windows share 25 time points (83.3% overlap), inducing intrinsic mathematical autocorrelation in similarity decay and state transition metrics.

---

## 2. Future Research Directions

### A. Multi-Seed GCN and QGCNN Replication
Execute systematic 10-fold cross-validation repeated across multiple random seeds ($\ge 5$) on the classical GCN and QGCNN architectures to establish rigorous confidence intervals and assess split stability.

### B. AAL-116 Cohort Alignment & Full-Sample Training
Re-train the classical GCN and quantum architectures across the full 391-subject aligned clean cohort without artificial downsampling to 162 subjects.

### C. Explicit Downstream Pseudo-Label Protocol
Reconstruct and benchmark an end-to-end reproducible pseudo-label pipeline linking high-confidence ensemble predictions directly to graph neural network training sets.

### D. Quantum Architecture & Ansatz Ablations
Systematically benchmark alternative quantum variational circuits:
- Variational depth sweeps ($N_{\text{layers}} \in \{1, 2, 4\}$)
- Alternative entangling topologies (ring, all-to-all, alternating)
- Amplitude embedding and tensor-network quantum representations to mitigate the linear dimensionality bottleneck

### E. Matched Quantum and Classical LOSO Benchmark
Scale the hybrid QGCNN to the 7-fold Leave-One-Site-Out (LOSO) CC200 benchmark (Exp 09) to directly test quantum out-of-distribution generalizability across clinical imaging sites.

### F. Hardware-Aware and Noisy Quantum Execution
Evaluate the hybrid QGCNN on physical quantum processing units (QPUs) using error-mitigation techniques (Zero-Noise Extrapolation, Readout Error Mitigation) to quantify real-world noise resilience.
