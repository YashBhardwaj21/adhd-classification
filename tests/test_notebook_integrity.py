"""
Tests asserting notebook structural integrity, metadata headers, and execution cleanliness.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_NOTEBOOKS = [
    "notebooks/exp01/01_fc_generation_and_validation.ipynb",
    "notebooks/exp02/02_graph_construction_and_validation.ipynb",
    "notebooks/exp02/03_graph_metrics_and_null_models.ipynb",
    "notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb",
    "notebooks/exp04/w2c_athena_2.ipynb",
    "notebooks/exp05/dynamic_transformer.ipynb",
    "notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb",
    "notebooks/exp08/09_temporal_graph_learning.ipynb",
    "notebooks/exp08/neuro.ipynb",
    "notebooks/exp08/true_neuro.ipynb",
    "notebooks/exp09/11_population_graph_learning.ipynb",
]


def get_all_notebooks():
    nbs = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    return nbs


def test_notebook_count_and_names():
    nbs = get_all_notebooks()
    rel_paths = [str(nb.relative_to(ROOT)).replace("\\", "/") for nb in nbs]
    assert len(nbs) == 11, f"Expected 11 notebooks, found {len(nbs)}"
    for expected in EXPECTED_NOTEBOOKS:
        assert expected in rel_paths, f"Missing expected notebook: {expected}"


@pytest.mark.parametrize("nb_path", EXPECTED_NOTEBOOKS)
def test_individual_notebook_integrity(nb_path):
    path = ROOT / nb_path
    assert path.exists(), f"Notebook {nb_path} does not exist"

    with open(path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, f"{nb_path} missing 'cells' key"
    cells = nb_data["cells"]
    assert len(cells) > 0, f"{nb_path} has no cells"

    # Verify Cell 0 is standard metadata header
    cell_0 = cells[0]
    assert cell_0.get("cell_type") == "markdown", f"{nb_path}: Cell 0 must be markdown header"
    header_text = "".join(cell_0.get("source", []))
    assert "# Exp" in header_text or "# 0" in header_text or "# w2c" in header_text, (
        f"{nb_path}: Header markdown must introduce the experiment"
    )

    # Verify no error outputs remain in code cells and no banned host paths in code source
    banned_in_code = ["/home/nvidia", "/lp-dev/", "file:///"]
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", (
                    f"{nb_path} Cell {idx} contains an unhandled error output: "
                    f"{out.get('ename')}: {out.get('evalue')}"
                )

            src_text = "".join(cell.get("source", []))
            for b in banned_in_code:
                assert b not in src_text, (
                    f"{nb_path} Cell {idx} contains hardcoded '{b}' in source code"
                )
