"""Configuration parameters for Experiment 2 (Graph Construction & Analysis)."""

# Atlas specifications
N_ROIS = 190  # CC200 / Craddock-200 parcellation (190 active ROIs)

# Graph construction parameters
DENSITY = 0.20  # Minimum Spanning Tree (MST) + Proportional Thresholding (PT) at 20%
STRATEGY = "MST+PT"
