#!/usr/bin/env python
# ============================================================================
# RUN CLASSICAL GCN EXPERIMENT (Track B Baseline)
# ============================================================================

import argparse
import sys
from pathlib import Path

from exp07.utils.config import SEED

try:
    from exp07.classical_models.train_classical_gcn import run_experiment
except ModuleNotFoundError as err:
    _missing = err.name
    def run_experiment(*args, **kwargs):
        raise RuntimeError(
            f"Required dependency '{_missing}' is not installed.\n"
            "Track B requires PyTorch Geometric and its dependencies.\n"
            "Please install dependencies via: pip install -r environment/track_b/requirements.txt"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run Classical Graph Convolutional Network (GCN) experiment on AAL-116 connectomes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing authentic aal116_fc_features.npz and aal116_labels.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory where training history and evaluation metrics will be written.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Directory where model checkpoints will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Deterministic random seed (default: {SEED}).",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RUNNING CLASSICAL GCN EXPERIMENT (Track B)")
    print("=" * 60 + "\n")

    try:
        model, history = run_experiment(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            checkpoints_dir=args.checkpoints_dir,
            seed=args.seed,
        )
        print("\nExperiment execution completed successfully.")
    except (FileNotFoundError, RuntimeError) as err:
        print(f"\n[EXECUTION HALTED: Precondition Not Met]\n{err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
