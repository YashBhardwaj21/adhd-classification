"""
Tests asserting notebook structural integrity, metadata headers, and execution cleanliness.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_NOTEBOOKS = [
    "01_fc_generation_and_validation.ipynb",
    "02_graph_construction_and_validation.ipynb",
    "03_graph_metrics_and_null_models.ipynb",
    "04_dynamic_graph_feature_extraction.ipynb",
    "w2c_athena_2.ipynb",
    "dynamic_transformer.ipynb",
    "exp06_semi_supervised_pseudolabeling.ipynb",
    "09_temporal_graph_learning.ipynb",
    "neuro.ipynb",
    "true_neuro.ipynb",
    "11_population_graph_learning.ipynb",
]


def get_all_notebooks():
    nbs = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    return nbs


def test_notebook_count_and_names():
    nbs = get_all_notebooks()
    nb_names = [nb.name for nb in nbs]
    assert len(nbs) == 11, f"Expected 11 notebooks, found {len(nbs)}"
    for expected in EXPECTED_NOTEBOOKS:
        assert expected in nb_names, f"Missing expected notebook: {expected}"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_individual_notebook_integrity(nb_name):
    matches = list((ROOT / "notebooks").rglob(nb_name))
    assert len(matches) == 1, f"Expected exactly one match for {nb_name}"
    nb_path = matches[0]

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, f"{nb_name} missing 'cells' key"
    cells = nb_data["cells"]
    assert len(cells) > 0, f"{nb_name} has no cells"

    # Verify Cell 0 is standard metadata header
    cell_0 = cells[0]
    assert cell_0.get("cell_type") == "markdown", f"{nb_name}: Cell 0 must be markdown header"
    header_text = "".join(cell_0.get("source", []))
    assert "# Exp" in header_text or "# 0" in header_text or "# w2c" in header_text, (
        f"{nb_name}: Header markdown must introduce the experiment"
    )

    # Verify no error outputs remain in code cells
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", (
                    f"{nb_name} Cell {idx} contains an unhandled error output: "
                    f"{out.get('ename')}: {out.get('evalue')}"
                )

            # Verify no host paths in code source
            src_text = "".join(cell.get("source", []))
            assert "/home/nvidia" not in src_text, (
                f"{nb_name} Cell {idx} contains hardcoded '/home/nvidia' in source code"
            )
