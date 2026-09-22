# Data Availability & Requirements

Raw ADHD-200 neuroimaging data are not redistributed in this repository.

To reproduce the experiments, obtain the ADHD-200 data from the official distribution and follow the current access and usage terms.

---

## 1. Required Data by Experiment

- **Experiments 01–05 (Connectomics & Harmonization)**:
  - Parcellated resting-state BOLD fMRI ROI time series on the Craddock-200 (CC200) atlas (Athena preprocessing pipeline).
  - Phenotypic metadata table with diagnostic status (`ADHD` vs `TDC`), acquisition site, age, and sex.

- **Experiment 06 (Semi-Supervised Learning)**:
  - Static functional connectivity correlation matrices on the AAL-116 atlas for 955 subjects (391 clean-labelled, 564 unlabelled).

- **Experiment 07 (Classical GCN vs Quantum QGCNN)**:
  - Historical execution inputs (`X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy`) across 875 subjects (162 clean-labelled, 713 pseudo-labelled).
  - *Status*: These original combined arrays are **not** redistributed in this repository. The reported test results (N=33) are archived in `results/exp07/`. See [`docs/provenance.md`](../docs/provenance.md).

- **Experiment 08 (Volumetric 3D CNN, NeuroSTORM & Temporal GNN)**:
  - Preprocessed 4D functional BOLD volume sequences ($T=25, 99 \times 117 \times 95$) for volumetric CNN evaluation.
  - Windowed CC200 graph sequences across disjoint subject splits for temporal GNN.

- **Experiment 09 (Leave-One-Site-Out Population Graphs)**:
  - Static CC200 functional connectivity matrices (190 ROIs) and phenotypic metadata across 497 subjects from 7 scanner sites.

---

## 2. Expected Input Formats

- **Time Series**: NumPy arrays (`.npy`) or tabular files (`.csv`) of shape `(T, N_ROIS)`.
- **Connectivity Matrices**: Symmetric correlation matrices of shape `(N_ROIS, N_ROIS)` or upper-triangular vectors of shape `(N_ROIS * (N_ROIS - 1) / 2,)`.
- **Phenotypic Manifests**: CSV tables containing columns `subject_id`, `diagnosis`, `site`, `age`, and `gender`.

---

## 3. Not Included in this Repository

- Raw 4D BOLD fMRI NIfTI files
- Intermediate sliding-window correlation arrays
- Model checkpoints from prior training runs
- Experiment 07 historical combined arrays (`X_combined_full.npy`, etc.)

See [`data/provenance.md`](provenance.md) for full exclusion details.

---

## 4. Obtaining the ADHD-200 Dataset

Obtain the ADHD-200 data from the official distribution and follow the current access and usage terms:
1. Visit the [ADHD-200 Consortium Preprocessed Connectomes Project (NITRC)](https://www.nitrc.org/projects/fcon_1000/).
2. Access the Athena preprocessed pipelines.
3. For phenotypic data and site descriptions, consult the official ADHD-200 documentation.
