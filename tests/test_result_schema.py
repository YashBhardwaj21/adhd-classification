"""Schema tests for selected canonical CSV and JSON result artifacts."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_exp01_temporal_validation_schema():
    path = ROOT / "results/exp01/dynamic_temporal_validation.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"subject_id", "site", "lag", "mean_similarity", "std_similarity", "mean_frobenius", "windows"}
    assert expected_cols.issubset(set(df.columns))
    assert set(df["lag"].unique()) == {1, 2, 3, 4}
    assert len(df) > 0


def test_exp04_comparison_table_schema():
    path = ROOT / "results/exp04/comparison_table.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"Metric", "Raw Mean", "Raw Std", "ComBat Mean", "ComBat Std", "Mean Difference"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 7


def test_exp07_checkpoint_analysis_schema():
    path = ROOT / "results/exp07/checkpoint_analysis.json"
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    for model_key in ["classical", "quantum"]:
        assert model_key in data
        assert "auc" in data[model_key]
        assert "accuracy" in data[model_key]
        assert "precision" in data[model_key]
        assert "recall" in data[model_key]
        assert "confusion_matrix" in data[model_key]
        assert "y_true" in data[model_key]
        assert len(data[model_key]["y_true"]) == 33


def test_exp08_verified_results_schema():
    path = ROOT / "results/exp08/exp08_verified_results.json"
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "models" in data
    assert "Lightweight_3D_CNN" in data["models"]
    assert "NeuroSTORM" in data["models"]
    assert "Temporal_Graph" in data["models"]


def test_exp09_loso_results_schema():
    path = ROOT / "results/exp09/w2b_loso_results.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"Architecture", "Fold", "Test_Site", "N_Test", "AUC", "BA", "Sensitivity", "Specificity", "F1"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 28  # 4 architectures x 7 folds
    assert set(df["Architecture"].unique()) == {"GCN", "GAT", "SAGE", "GIN"}
    assert len(df["Test_Site"].unique()) == 7


def test_results_manifest_schema():
    path = ROOT / "results/manifest.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"experiment", "artifact", "source", "status", "dataset", "n_subjects", "seed", "notes"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 9
    experiments = set(df["experiment"].unique())
    for exp_num in range(1, 10):
        assert f"exp0{exp_num}" in experiments, f"Missing exp0{exp_num} in manifest"
