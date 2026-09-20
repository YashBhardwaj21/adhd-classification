# ADHD Classification: Classical vs Quantum GCNN

## 1. Dataset Summary

| Type | Count | Healthy | ADHD |
|------|-------|---------|------|
| Clean Labels | 162 | 93 | 69 |
| Pseudo Labels | 713 | 535 | 178 |
| Test Set | 33 | - | - |

## 2. Model Performance

| Model | AUC | Accuracy | Precision | Recall | F1 |
|-------|-----|----------|-----------|--------|----|
| Classical GCN | 0.7293 | 0.6970 | 0.6941 | 0.6970 | 0.6929 |
| Quantum GCNN (6 qubits) | 0.6429 | 0.6061 | 0.6204 | 0.6061 | 0.6082 |

## 3. Key Findings

- **Classical GCN outperforms quantum** by 0.0865 AUC
- Classical: AUC = 0.7293
- Quantum: AUC = 0.6429
- Performance gap is moderate

## 4. Generated Plots

| Plot | Description |
|------|-------------|
| `roc_curves.png` | ROC curves for both models |
| `confusion_matrices.png` | Confusion matrices |
| `comparison_bars.png` | Performance comparison |
| `label_distribution.png` | Clean vs pseudo labels |
| `label_comparison.png` | Label distribution comparison |
| `summary_table.png` | Results summary table |
| `previous_runs.png` | Previous runs history |

## 5. Conclusion

Based on the results:

- **Classical GCN is recommended** for this task
- Quantum GCNN needs more training or architectural changes
