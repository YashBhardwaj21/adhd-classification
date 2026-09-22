#!/usr/bin/env python
# Run vectorized Quantum GCNN experiment (Track B hybrid model).

import argparse
import sys
from pathlib import Path

from exp07.utils.config import RANDOM_SEED

try:
    from exp07.quantum_models.train_qgcnn_vectorized import run_experiment
except ModuleNotFoundError as err:
    _missing = err.name
    def run_experiment(*args, **kwargs):
        raise RuntimeError(
            f"Required dependency '{_missing}' is not installed.\n"
            "Track B quantum acceleration requires PyTorch Geometric and PennyLane (lightning.gpu).\n"
            "Please install dependencies via: pip install -r environment/track_b/requirements.txt"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid Quantum-Classical GCNN experiment on AAL-116 connectomes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing historical X_combined_full.npy, y_combined.npy, subjects_combined.npy.",
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
        default=RANDOM_SEED,
        help=f"Deterministic random seed (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--quantum-device",
        type=str,
        default="lightning.gpu",
        help="PennyLane quantum simulator device (default: 'lightning.gpu'). Use 'default.qubit' for CPU checks.",
    )

    args = parser.parse_args()

    print("\nRunning Vectorized Quantum GCNN Experiment (Track B)\n")

    try:
        model, history = run_experiment(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            checkpoints_dir=args.checkpoints_dir,
            seed=args.seed,
            quantum_device=args.quantum_device,
        )
        print("\nExperiment execution completed successfully.")
    except (FileNotFoundError, RuntimeError) as err:
        print(f"\n[EXECUTION HALTED: Precondition Not Met]\n{err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
