"""Unit tests for configuration integrity and environment metadata validation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_w2b_environment_config():
    """Verify that the recorded environment metadata exists and matches historical packages."""
    env_file = ROOT / "configs/exp09/w2b_environment.json"
    assert env_file.exists(), "configs/exp09/w2b_environment.json must exist"

    with open(env_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Core environment assertions
    assert meta.get("python_version") == "3.12.13"
    assert "2.5.1" in meta.get("torch_version", "")
    assert meta.get("cuda_version") == "12.1"
    assert meta.get("pyg_version") == "2.8.0"
    assert "A100" in meta.get("gpu_name", "")
    assert meta.get("git_commit") == "not recorded during original execution"

    # Check pip_freeze contains PennyLane packages
    pip_freeze = meta.get("pip_freeze", [])
    assert any("pennylane==0.44.1" in pkg for pkg in pip_freeze)
    assert any("pennylane_lightning==0.44.0" in pkg for pkg in pip_freeze)
    assert any("pennylane_lightning_gpu==0.44.0" in pkg for pkg in pip_freeze)


def test_track_b_reported_run_config():
    """Verify that Track B configuration matches verified historical research parameters."""
    cfg_file = ROOT / "configs/exp07/reported_run.json"
    assert cfg_file.exists(), "configs/exp07/reported_run.json must exist"

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["experiment"] == "exp07"
    assert cfg["status"] == "historical_reported_run"
    assert cfg["graph"]["n_rois"] == 116
    assert cfg["graph"]["density"] == 0.15
    assert cfg["quantum"]["n_qubits"] == 6
    assert cfg["quantum"]["n_layers"] == 1
    assert cfg["training"]["batch_size"] == 8
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["learning_rate"] == 0.001
    assert cfg["training"]["weight_decay"] == 0.00001

    from exp07.utils.config import BATCH_SIZE, DENSITY, LEARNING_RATE, N_LAYERS, N_QUBITS, N_ROIS, NODE_FEATURE_DIM, WEIGHT_DECAY

    assert N_ROIS == 116, "Track B uses AAL-116 parcellation"
    assert NODE_FEATURE_DIM == 117, "Node features must be 116 FC + 1 degree"
    assert N_QUBITS == 6, "Quantum circuit must use 6 qubits"
    assert N_LAYERS == 1, "Verified historical quantum circuit used 1 variational layer"
    assert DENSITY == 0.15, "Verified historical Exp 07 density is 0.15"
    assert BATCH_SIZE == 8, "Verified historical batch size is 8"
    assert LEARNING_RATE == 1e-3, "Verified historical learning rate is 1e-3"
    assert WEIGHT_DECAY == 1e-5, "Verified historical weight decay is 1e-5"
