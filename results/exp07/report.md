# Experiment 7 Results: Classical GCN vs Quantum QGCNN

This report summarizes the evaluated test-set metrics for Experiment 7 (Track B) recorded in [`checkpoint_analysis.json`](checkpoint_analysis.json).

## Cohort and Test Set Scope

The held-out test evaluation was performed on 33 clean-labelled subjects partitioned from the 162 clean-labelled subject cohort:
- Total clean cohort: 162 subjects (103 train, 26 validation, 33 held-out test)
- Pseudo-labelled cohort: 713 subjects (allocated strictly to training)
- Held-out test set: 33 subjects (19 TDC / healthy, 14 ADHD)

## Test Set Performance

| Model | AUC | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical GCN** | 0.7293 | 0.6970 | 0.6941 | 0.6970 | 0.6929 |
| **Quantum QGCNN** (6 qubits) | 0.6429 | 0.6061 | 0.6204 | 0.6061 | 0.6082 |

> [!NOTE]
> **Metric Convention**: Precision, recall, and F1 score metrics reported above reproduce the weighted-average convention recorded in the result artifact (`checkpoint_analysis.json`). Differences between this weighted convention and manuscript-reported class-level metrics are documented in [`docs/provenance.md`](../../docs/provenance.md).

## Evaluated Architectures

- **Classical GCN**: 3 GCNConv layers (117 $\to$ 32 $\to$ 32 $\to$ 16 $\to$ 2) with BatchNorm1d, Dropout(0.30), and global mean pooling (5,522 trainable parameters).
- **Hybrid Quantum QGCNN**: Classical projection (117 $\to$ 12), 6-qubit variational circuit (1 trainable layer with sequential $R_X, R_Y, R_Z$ rotations, ring CNOT entanglement, 6 Pauli-$Z$ expectation measurements), followed by 3 GCNConv layers (6 $\to$ 16 $\to$ 16 $\to$ 16 $\to$ 2) and global mean pooling (2,188 trainable parameters).
- **Graph Construction**: AAL-116 parcellation (116 nodes), top 15% absolute FC edge selection, signed correlation weights, 117 node features (116 FC correlations + 1 normalized degree).

## Interpretation

Under the evaluated configuration, the two models produced different test-set performance on the 33 held-out test subjects. The experiment evaluates the specific parameterizations tested and does not establish a general advantage for either architecture across all possible hyperparameter configurations.
