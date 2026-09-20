"""Unit tests for configuration integrity and environment metadata validation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_w2b_environment_config():
    """Verify that the recorded environment metadata exists and matches audited packages."""
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

    # Check pip_freeze contains PennyLane packages
    pip_freeze = meta.get("pip_freeze", [])
    assert any("pennylane==0.44.1" in pkg for pkg in pip_freeze)
    assert any("pennylane_lightning==0.44.0" in pkg for pkg in pip_freeze)
    assert any("pennylane_lightning_gpu==0.44.0" in pkg for pkg in pip_freeze)


def test_track_b_hyperparams():
    """Verify that Track B configuration matches audited research parameters."""
    from exp07.utils.config import N_LAYERS, N_QUBITS, N_ROIS, NODE_FEATURE_DIM

    assert N_ROIS == 116, "Track B uses AAL-116 parcellation"
    assert NODE_FEATURE_DIM == 117, "Node features must be 116 FC + 1 degree"
    assert N_QUBITS == 6, "Quantum circuit must use 6 qubits"
    assert N_LAYERS == 2, "Quantum circuit must use 2 variational layers"
