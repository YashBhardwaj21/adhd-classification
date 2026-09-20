#!/usr/bin/env python
# ============================================================================
# RUN VECTORIZED QUANTUM EXPERIMENT
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_models.train_qgcnn_vectorized import run_experiment

if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("RUNNING VECTORIZED QUANTUM EXPERIMENT")
    print("🚀"*30 + "\n")
    model, history = run_experiment()
    print("\n✅ Experiment complete!")