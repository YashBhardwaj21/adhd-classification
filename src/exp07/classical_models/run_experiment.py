#!/usr/bin/env python
# ============================================================================
# RUN CLASSICAL GCN EXPERIMENT
# ============================================================================

from classical_models.train_classical_gcn import run_experiment

if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("RUNNING CLASSICAL GCN EXPERIMENT")
    print("🚀"*30 + "\n")
    model, history = run_experiment()
    print("\n✅ Experiment complete!")