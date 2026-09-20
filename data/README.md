# ADHD-200 Data Governance & Directory Layout

## 1. Data Availability & Policies

Raw ADHD-200 neuroimaging data (functional BOLD fMRI scans, anatomical T1 scans, and derived ROI time series) are governed by consortium data use agreements and exceed GitHub storage limits. Consequently, raw imaging scans and large intermediate binary arrays are not tracked in this Git repository.

All lightweight derived tabular metrics, execution manifests, cross-validation results, and summary figures are tracked in [`results/`](../results/) and executed [`notebooks/`](../notebooks/).

---

## 2. Directory Layout for External Data

When executing notebooks or training scripts on a local workstation or HPC cluster, organize external data according to the structure below or point the corresponding environment variables to your data directory:

```text
data/
├── RawDataBIDS/                      # Raw BOLD fMRI in BIDS format (Exp 08)
│   ├── sub-0010001/
│   │   └── func/
│   │       └── sub-0010001_task-rest_bold.nii.gz
│   └── ...
├── 02_timeseries/                    # Parcellated ROI timeseries (CC200 / AAL116)
│   ├── 0010001_roi_timeseries.npy
│   └── ...
├── 03_fc_matrices/                   # Static and dynamic FC matrices (Exp 01, 02)
│   ├── static/
│   │   └── static_manifest.csv
│   ├── dynamic/
│   │   └── dynamic_manifest.csv
│   ├── X_fc.npy
│   ├── X_fc_clean.npy
│   └── X_fc_norm.npy
├── exp07/                            # AAL-116 connectomes for Classical & Quantum GCN (Exp 07)
│   ├── aal116_fc_features.npz
│   └── aal116_labels.csv
└── neurostorm_data/                  # Preprocessed 4D volume sequences (Exp 08)
    └── ...
```

---

## 3. Environment Variable Configuration

All data paths throughout the codebase and notebooks can be configured dynamically without editing source files:

| Variable | Default Fallback | Purpose / Experiment Scope |
| :--- | :--- | :--- |
| `ADHD200_DATA_DIR` | `data` | Root directory for datasets and manifests |
| `ADHD200_FC_ROOT` | `data/03_fc_matrices` | Static and dynamic FC matrices (Exp 01, 02) |
| `ADHD200_RESULTS_DIR` | `results` | Output root for generated tables and figures |
| `ADHD200_CHECKPOINTS_DIR` | `checkpoints` | Storage location for model weights |
| `ADHD200_PROJECT_ROOT` | `.` | Repository root path |

Example configuration in bash:
```bash
export ADHD200_DATA_DIR="/path/to/my/adhd_data"
export ADHD200_FC_ROOT="/path/to/my/adhd_data/03_fc_matrices"
export ADHD200_RESULTS_DIR="results"
```

Example in PowerShell:
```powershell
$env:ADHD200_DATA_DIR = "D:\data\adhd200"
$env:ADHD200_FC_ROOT = "D:\data\adhd200\03_fc_matrices"
```

---

## 4. Obtaining the ADHD-200 Consortium Dataset

To download the original ADHD-200 dataset:
1. Visit the [ADHD-200 Consortium Preprocessed Connectomes Project (NITRC)](https://www.nitrc.org/projects/fcon_1000/).
2. Request access and download the Athena or NIAK preprocessed pipelines.
3. For phenotypic information and site descriptions, consult the official ADHD-200 release documentation.

For detailed instructions on running each experiment once data is mounted, refer to [`docs/reproduction.md`](../docs/reproduction.md).
