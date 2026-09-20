# Experiment 6: Semi-Supervised Pseudo-Labeling on Unlabeled Cohorts

## 1. Scientific Overview & Objectives
A central bottleneck in clinical neuroimaging is the scarcity of high-confidence diagnostic ground truth relative to large uncurated repositories. Experiment 6 evaluates semi-supervised learning strategies to leverage unannotated scans from the ADHD-200 repository.

The primary objectives are:
1. Develop iterative self-training and ensemble pseudo-labeling algorithms on functional connectivity features.
2. Quantify label confidence thresholds and selection criteria to prevent confirmation bias and label noise propagation.
3. Benchmark diagnostic accuracy gains when augmenting clean training sets with pseudo-labeled subjects.

---

## 2. Cohort Partitioning & Parcellation
- **Atlas**: Automated Anatomical Labeling (AAL-116 parcellation, Track B feature space).
- **Cohort Structure**:
  - **Clean Labeled Cohort**: 391 subjects
    - Training Set ($70\%$): 273 subjects
    - Validation Set ($15\%$): 59 subjects
    - Held-out Clean Test Set ($15\%$): 59 subjects
  - **Unlabeled Target Pool**: 564 subjects from ADHD-200.

---

## 3. Pseudo-Labeling Procedures & Audited Results
Traced directly to [`results/exp06/exp06_verified_results.json`](../../results/exp06/exp06_verified_results.json) and executed output in `notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb`:

### 3.1 Procedure I: Iterative Self-Training (Logistic Regression)
- **Base Classifier**: `LogisticRegression(class_weight='balanced', C=1.0)` with `StandardScaler`.
- **Selection Parameters**: Minimum probability threshold $p \ge 0.75$, $k_{\text{best}} = 10$ candidates per class per iteration, maximum 10 iterations.
- **Assigned Pseudo-Labels**: **552 subjects** from the 564 unlabeled pool.
- **Validation Accuracy**: $0.5593$
- **Held-Out Clean Test Accuracy**: **$0.6441$** ($38/59$ correct).

### 3.2 Procedure II: Weighted 4-Classifier Ensemble
- **Ensemble Members**: Random Forest, Gradient Boosting, Logistic Regression, Calibrated Linear SVM.
- **Selection Criterion**: Unanimous or high-confidence consensus probability across all four architectures.
- **Assigned Pseudo-Labels**: **484 subjects**.
- **Clean-Only Validation Accuracy**: $0.6709$
- **Pseudo-Label Augmented Validation Accuracy**: **$0.7215$** ($+5.06\%$ absolute gain on internal validation).

### 3.3 Fully Supervised Ensembling Baselines (Clean Test Set, $N=59$)
- **Hard Voting Test Accuracy**: $0.7119$ ($42/59$ correct)
- **Soft Voting Test Accuracy**: $0.6780$ ($40/59$ correct)

---

## 4. Architectural Decoupling: Non-Linear Pipeline Provenance
> [!WARNING]
> **Track B Historical Decoupling**:
> In prior documentation drafts, Experiment 6 was described as directly feeding its 552 (or 484) pseudo-labeled subjects into Experiment 7's Graph Convolutional Network. Source code and artifact audits reveal this is factually incorrect:
> - **Experiment 6 Cohort**: 391 clean labeled + 564 unlabeled subjects $\to$ produced 552 (Procedure I) or 484 (Procedure II) pseudo-labels.
> - **Experiment 7 Cohort**: 162 clean labeled + 713 pseudo-labeled subjects $\to$ 875 total subjects with a 33-subject test partition.
> - **Scientific Meaning**: Experiment 6 and Experiment 7 represent **independent experimental tracks** within Track B, conducted with distinct data preparation cohorts. They are not a single contiguous training pipeline.

---

## 5. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb`](../../notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb)
- **Result Artifact**:
  - [`results/exp06/exp06_verified_results.json`](../../results/exp06/exp06_verified_results.json)
- **Historical Model Artifact**: `mnt/ADHD200/06_semi_supervised/models/self_training_model.pkl` (earlier Random Forest iteration preserved for audit history).
