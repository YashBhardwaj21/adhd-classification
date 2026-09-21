# Third-Party Dependencies and Attributions

This document records third-party packages, algorithms, and bundled code utilized within this repository.

---

## 1. NeuroSTORM Spatio-Temporal Transformer
- **Included File**: `src/exp08/neurostorm/neurostorm.py`
- **Upstream Project**: NeuroSTORM (CUHK-AIM-Group)
- **Source**: https://github.com/CUHK-AIM-Group/NeuroSTORM
- **License**: Apache License 2.0
- **Implementation Status**: Adapted from upstream implementation (incorporates architectural blocks derived from MONAI SwinUNETR and SwiFT) with dataloading interfaces for 4D functional fMRI sequences.
- **Publication Reference**: NeuroSTORM: A Foundation Model for Neuroimaging Spatio-Temporal Representation Learning.

---

## 2. Brain Connectivity Toolbox (BCT) Null Models
- **Included Files**:
  - `src/exp02/null_model_und_sign_fixed.py`
  - `src/exp02/randmio_und_signed_fast.py`
- **Upstream Project**: `bctpy` (Python port of the Brain Connectivity Toolbox by Roan LaPlante)
- **Source**: https://github.com/aestrivex/bctpy
- **License**: GNU General Public License v3.0 (GPL-3.0)
- **Implementation Status**: Copied directly from `bctpy` utilities for weight-conserving signed matrix randomization.
- **Licensing Note**: While the general repository code is released under the MIT License, these two files are governed by GPL-3.0 as derived works of `bctpy`.
- **Publication Reference**: Rubinov, M., & Sporns, O. (2011). "Weight-conserving characterization of complex brain networks: measure development and application to normal and brain-damaged subjects." *NeuroImage*, 56(4), 2068-2079.

---

## 3. neuroCombat (External Dependency)
- **Component**: ComBat harmonization protocol (Experiment 04)
- **Upstream Project**: `neuroCombat` (Python package by Jean-Philippe Fortin)
- **Source**: https://github.com/Jfortin1/neuroCombat
- **License**: MIT License
- **Status in Repository**: External package dependency installed via pip (`pip install neuroCombat`); no source code bundled in-tree.
- **Publication Reference**: Fortin, J. P., et al. (2018). "Harmonization of cortical thickness measurements across scanners and sites." *NeuroImage*, 167, 104-120.
