"""Schema tests for selected canonical CSV and JSON result artifacts."""

import json
from pathlib import Path

import numpy as np
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


def test_exp07_clean_subjects_manifest_schema():
    path = ROOT / "results/exp07/clean_subjects_manifest.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {
        "subject_id",
        "site",
        "diagnosis",
        "exp06_aligned",
        "exp07_clean_prefix",
        "exp07_combined_index",
        "exp07_split",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 391
    assert df["subject_id"].nunique() == 391
    clean = df[df["exp07_clean_prefix"] == 1]
    assert len(clean) == 162
    splits = clean["exp07_split"].value_counts().to_dict()
    assert splits.get("train") == 103
    assert splits.get("validation") == 26
    assert splits.get("test") == 33
    assert set(clean["subject_id"]).issubset(set(df["subject_id"]))


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


def test_exp09_canonical_model_summary_schema():
    path = ROOT / "results/exp09/w1_canonical_model_summary.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"family", "model", "mean_auc", "std_auc", "ci_lower", "ci_upper", "mean_ba", "std_ba", "n_folds", "n_sites"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 42  # 7 classifiers x 6 feature families
    assert (df["n_folds"] == 7).all()
    assert (df["n_sites"] == 7).all()


def test_exp09_graph_statistics_schema():
    path = ROOT / "results/exp09/graph_statistics.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"Threshold", "Mean_Density", "Mean_Small_Worldness", "Pct_Connected"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 8
    strategies = set(df["Threshold"])
    assert "top_10pct" in strategies
    assert "top_5pct" in strategies
    t10 = df[df["Threshold"] == "top_10pct"].iloc[0]
    assert np.isclose(t10["Mean_Density"], 0.10, atol=1e-2)
    assert np.isclose(t10["Mean_Small_Worldness"], 3.8095, atol=1e-2)


def test_exp09_diagnostic_ledgers_schema():
    pos_path = ROOT / "results/exp09/gnn_diagnostic_positive.csv"
    abs_path = ROOT / "results/exp09/gnn_diagnostic_absolute.csv"
    assert pos_path.exists()
    assert abs_path.exists()
    pos = pd.read_csv(pos_path)
    ab = pd.read_csv(abs_path)
    expected_cols = {"fold", "test_site", "auc", "ba", "n_test"}
    assert expected_cols.issubset(set(pos.columns))
    assert expected_cols.issubset(set(ab.columns))
    assert len(pos) == 7
    assert len(ab) == 7


def test_exp09_statistical_ledgers_schema():
    w_tests = ROOT / "results/exp09/w2b_wilcoxon_tests.csv"
    w_ci = ROOT / "results/exp09/w2b_summary_with_ci.csv"
    w_err = ROOT / "results/exp09/w2b_error_analysis.csv"
    for p in [w_tests, w_ci, w_err]:
        assert p.exists()
        df = pd.read_csv(p)
        assert len(df) > 0


def test_exp09_selected_threshold_config_schema():
    cfg_path = ROOT / "configs/exp09/selected_threshold.json"
    assert cfg_path.exists()
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg.get("selected_threshold") == "top_10pct"
    assert (ROOT / cfg.get("threshold_sweep_ledger")).exists()
    assert cfg.get("sweep_producer_code_retained") is False


def test_results_manifest_schema():
    path = ROOT / "results/manifest.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected_cols = {"experiment", "artifact", "source", "status", "dataset", "n_subjects", "seed", "notes"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 25
    experiments = set(df["experiment"].unique())
    for exp_num in range(1, 10):
        assert f"exp0{exp_num}" in experiments, f"Missing exp0{exp_num} in manifest"
