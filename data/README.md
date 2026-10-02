# Data Requirements and Management

This directory defines the neuroimaging and phenotypic data inputs required to execute experiments across the repository, specifies expected file formats, and outlines data access procedures.

---

## 1. Data Availability Policy

Raw neuroimaging scans and intermediate feature arrays from the **ADHD-200 Consortium** are not redistributed in this Git repository. Users wishing to execute experiments from primary inputs must obtain the dataset directly from official distribution repositories in accordance with the ADHD-200 Data Use Agreement.

For a detailed ledger of excluded large files, retained lightweight result tables, and external dependencies, see [`data/availability_and_exclusions.md`](availability_and_exclusions.md).

---

## 2. Required Data by Experimental Track

### Track A — Dynamic Connectivity and Graph Analysis (Experiments 01–05)
- **Atlas**: Craddock-200 (CC200) atlas with 190 active cortical/subcortical regions.
- **Imaging Inputs**: Resting-state BOLD fMRI ROI time series generated via the Athena preprocessing pipeline.
- **Cohort**: 764 unique subjects across 9 international acquisition centers (1,193 imaging acquisitions, 31,060 sliding temporal windows).
- **Phenotypic Data**: Demographic manifest (`ADHD200_phenotypic.csv`) with `subject_id`, `diagnosis` (TDC vs ADHD), `site`, `age`, and `gender`. A 534-subject phenotypically complete subset is evaluated for behavioral and diagnosis association tests (Exp 03–05).

### Track B — Semi-Supervised and Quantum Graph Classification (Experiments 06–08)
- **Atlases & Modalities**:
  - Automated Anatomical Labeling (AAL-116) atlas for Exp 06–07.
  - 4D functional BOLD NIfTI volumes ($99 \times 117 \times 95 \times T$) for volumetric 3D CNN and NeuroSTORM (Exp 08).
- **Cohorts**:
  - Exp 06: 955 connectivity-available subjects partitioned into a 391-subject aligned clean cohort and 564 unlabeled subjects.
  - Exp 07: 162 clean-labeled subjects (proven subset of the 391 aligned cohort) combined with 713 downstream pseudo-labeled samples. *Note: As documented in [`availability_and_exclusions.md`](availability_and_exclusions.md), these large training arrays are archived externally and not tracked in Git.*
  - Exp 08: 626 volumetric subjects (4D fMRI) and 764 temporal subjects.

### Track C — Independent Cross-Site Generalization (Experiment 09)
- **Atlas**: CC200 atlas (190 ROIs).
- **Imaging Inputs**: Static functional connectivity Pearson correlation matrices.
- **Cohort**: 497 subjects across 7 scanner sites (KKI, NYU, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3) evaluated under strict Leave-One-Site-Out (LOSO) cross-validation.

---

## 3. Expected Input Formats

When placing external data into `data/`, adhere to the following directory layout and formats:

```text
data/
├── raw/
│   └── adhd200/                     # Raw 4D fMRI NIfTI files (*.nii, *.nii.gz)
├── preprocessed/
│   ├── cc200/                       # CC200 ROI time series (*.npy, shape: [T, 190])
│   ├── aal116/                      # AAL-116 ROI time series (*.npy, shape: [T, 116])
│   └── static_fc/                   # Static FC correlation matrices (*.npy, shape: [N_ROI, N_ROI])
└── phenotypic/
    └── master_cohort.csv            # Clean phenotypic table with columns:
                                     # [subject_id, site, diagnosis, age, gender, handedness]
```

---

## 4. Obtaining the Dataset

1. Register for access via the [NITRC ADHD-200 Portal](https://www.nitrc.org/projects/fcon_1000/).
2. Download preprocessed functional connectomes (Athena pipeline) from the [Neurobureau ADHD-200 repository](http://neurobureau.projects.nitrc.org/ADHD200/Data.html).
3. Ensure site acquisition IDs match the canonical identifiers used across repository configs (`KKI`, `NYU`, `NeuroIMAGE`, `OHSU`, `Peking_1`, `Peking_2`, `Peking_3`, `Pittsburgh`, `WashU`).
