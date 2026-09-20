#!/usr/bin/env python
# ============================================================================
# REPOSITORY AUDIT SCRIPT
# Validates directory structure, notebook validity, artifact non-emptiness,
# and flags any 0-byte or corrupted staging files.
# ============================================================================

import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CANONICAL_NOTEBOOKS = [
    "notebooks/exp01/01_fc_generation_and_validation.ipynb",
    "notebooks/exp02/02_graph_construction_and_null_models.ipynb",
    "notebooks/exp03/03_topological_feature_extraction.ipynb",
    "notebooks/exp04/04_combat_harmonization.ipynb",
    "notebooks/exp05/05_dynamic_state_modeling.ipynb",
    "notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb",
    "notebooks/exp08/neuro.ipynb",
    "notebooks/exp08/true_neuro.ipynb",
    "notebooks/exp08/09_temporal_graph_learning.ipynb",
    "notebooks/exp09/11_population_graph_learning.ipynb",
]

KEY_RESULT_ARTIFACTS = [
    "results/exp01/dynamic_temporal_validation.csv",
    "results/exp01/static_vs_dynamic_validation.csv",
    "results/exp02/graph_metrics.csv",
    "results/exp02/subject_graph_metrics.csv",
    "results/exp03/feature_statistics.csv",
    "results/exp03/site_anova.csv",
    "results/exp04/comparison_table.csv",
    "results/exp04/site_prediction_results.csv",
    "results/exp04/diagnosis_prediction_results.csv",
    "results/exp05/run_dynamic_biomarkers.csv",
    "results/exp05/run_state_sequences.csv",
    "results/exp06/exp06_verified_results.json",
    "results/exp07/checkpoint_analysis.json",
    "results/exp07/report.md",
    "results/exp08/exp08_verified_results.json",
    "results/exp09/w2b_loso_results.csv",
    "results/exp09/exp09_verified_results.json",
    "results/exp09/graph_preprocessing_summary.csv",
]

DOCUMENTATION_FILES = [
    "docs/provenance/source_of_truth.md",
    "docs/provenance/paper_vs_code.md",
    "docs/provenance/reproducibility_status.md",
    "docs/reproduction.md",
    "docs/experiments/exp01.md",
    "docs/experiments/exp02.md",
    "docs/experiments/exp03.md",
    "docs/experiments/exp04.md",
    "docs/experiments/exp05.md",
    "docs/experiments/exp06.md",
    "docs/experiments/exp07.md",
    "docs/experiments/exp08.md",
    "docs/experiments/exp09.md",
]


def audit_notebooks():
    print("\n--- 1. Auditing Canonical Notebooks ---")
    errors = 0
    for rel_path in CANONICAL_NOTEBOOKS:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            print(f"❌ MISSING NOTEBOOK: {rel_path}")
            errors += 1
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                num_cells = len(data.get("cells", []))
                print(f"✅ {rel_path} ({num_cells} cells, {full_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"❌ INVALID JSON in {rel_path}: {e}")
            errors += 1
    return errors


def audit_artifacts():
    print("\n--- 2. Auditing Key Result Artifacts ---")
    errors = 0
    for rel_path in KEY_RESULT_ARTIFACTS:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            print(f"❌ MISSING ARTIFACT: {rel_path}")
            errors += 1
        elif full_path.stat().st_size == 0:
            print(f"❌ ZERO-BYTE ARTIFACT: {rel_path}")
            errors += 1
        else:
            print(f"✅ {rel_path} ({full_path.stat().st_size:,} bytes)")
    return errors


def audit_zero_byte_files():
    print("\n--- 3. Checking for 0-Byte or Orphaned Source Files ---")
    errors = 0
    for root, dirs, files in os.walk(ROOT_DIR / "src"):
        for file in files:
            file_path = Path(root) / file
            if file_path.stat().st_size == 0:
                print(f"⚠️ ZERO-BYTE SOURCE FILE FOUND: {file_path.relative_to(ROOT_DIR)}")
                errors += 1
    if errors == 0:
        print("✅ No zero-byte source files found in src/.")
    return errors


def audit_documentation():
    print("\n--- 4. Auditing Documentation Coverage ---")
    errors = 0
    for rel_path in DOCUMENTATION_FILES:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            print(f"❌ MISSING DOC: {rel_path}")
            errors += 1
        elif full_path.stat().st_size == 0:
            print(f"❌ EMPTY DOC: {rel_path}")
            errors += 1
        else:
            print(f"✅ {rel_path} ({full_path.stat().st_size:,} bytes)")
    return errors


def main():
    print("=" * 60)
    print("ADHD-200 REPOSITORY INTEGRITY AUDIT")
    print("=" * 60)
    print(f"Root Directory: {ROOT_DIR}")

    err_nb = audit_notebooks()
    err_art = audit_artifacts()
    err_zero = audit_zero_byte_files()
    err_doc = audit_documentation()

    total_errors = err_nb + err_art + err_zero + err_doc
    print("\n" + "=" * 60)
    if total_errors == 0:
        print("🎉 ALL AUDIT CHECKS PASSED: Repository is complete and well-formed!")
        sys.exit(0)
    else:
        print(f"⚠️ AUDIT FAILED WITH {total_errors} ISSUES. Review details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
