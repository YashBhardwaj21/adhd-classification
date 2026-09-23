#!/usr/bin/env python
# Repository audit script.
# Audits canonical notebook existence, JSON validity, and reports their cell counts.
# Validates directory structure, key artifact non-emptiness,
# and checks for 0-byte or corrupted staging files.


import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent

CANONICAL_NOTEBOOKS = [
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

KEY_RESULT_ARTIFACTS = [
    "results/exp01/dynamic_temporal_validation.csv",
    "results/exp01/static_vs_dynamic_validation.csv",
    "results/exp02/graph_construction_strategy.csv",
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
    "docs/reproduction.md",
    "docs/experiment_notes.md",
    "docs/provenance.md",
    "data/README.md",
    "data/provenance.md",
]


def audit_notebooks() -> int:
    print("\n--- 1. Auditing Canonical Notebook Integrity ---")
    errors = 0
    for rel_path in CANONICAL_NOTEBOOKS:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            print(f"[MISSING NOTEBOOK] {rel_path}")
            errors += 1
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                num_cells = len(data.get("cells", []))
                print(f"[OK] {rel_path} ({num_cells} cells, {full_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"[INVALID JSON] {rel_path}: {e}")
            errors += 1
    return errors


def audit_artifacts() -> int:
    print("\n--- 2. Auditing Key Result Artifacts ---")
    errors = 0
    for rel_path in KEY_RESULT_ARTIFACTS:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            print(f"[MISSING ARTIFACT] {rel_path}")
            errors += 1
        elif full_path.stat().st_size == 0:
            print(f"[ZERO-BYTE ARTIFACT] {rel_path}")
            errors += 1
        else:
            print(f"[OK] {rel_path} ({full_path.stat().st_size:,} bytes)")
    return errors


def audit_zero_byte_files() -> int:
    print("\n--- 3. Checking for 0-Byte or Orphaned Source Files ---")
    errors = 0
    for root, dirs, files in os.walk(ROOT_DIR / "src"):
        for file in files:
            file_path = Path(root) / file
            if file_path.name == "__init__.py":
                continue
            if file_path.stat().st_size == 0:
                print(f"[ZERO-BYTE SOURCE] {file_path.relative_to(ROOT_DIR)}")
                errors += 1
    if errors == 0:
        print("[OK] No zero-byte source files found in src/.")
    return errors


def audit_documentation() -> int:
    print("\n--- 4. Auditing Documentation Coverage ---")
    errors = 0
    for rel_path in DOCUMENTATION_FILES:
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            print(f"[MISSING DOC] {rel_path}")
            errors += 1
        elif full_path.stat().st_size == 0:
            print(f"[EMPTY DOC] {rel_path}")
            errors += 1
        else:
            print(f"[OK] {rel_path} ({full_path.stat().st_size:,} bytes)")
    return errors


def audit_stale_references_and_obsolete_files() -> int:
    print("\n--- 5. Auditing for Stale References & Obsolete Files ---")
    errors = 0
    obsolete_files = [
        "src/exp07/quantum_models/quantum_embedding_vectorized.py",
        "src/exp07/quantum_models/quantum_embedding_vmap.py",
        "src/exp07/quantum_models/resume.py",
        "scripts/clean_notebooks.py",
        "scripts/fix_markdown_links.py",
        "data/provenance/README.md",
        "docs/experiments/exp01.md",
        "docs/provenance/paper_vs_code.md",
    ]
    for obs in obsolete_files:
        if (ROOT_DIR / obs).exists():
            print(f"[FOUND OBSOLETE FILE] {obs} should have been removed from repository tree.")
            errors += 1

    # Check that active source files do not import obsolete Exp 07 files
    for py_file in (ROOT_DIR / "src").rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            for obsolete_mod in ["quantum_embedding_vmap", "resume"]:
                if f"import {obsolete_mod}" in content or f"from .{obsolete_mod}" in content:
                    print(f"[STALE IMPORT] {py_file.relative_to(ROOT_DIR)} imports obsolete module '{obsolete_mod}'")
                    errors += 1
        except Exception as e:
            print(f"[READ ERROR] {py_file}: {e}")
            errors += 1

    if errors == 0:
        print("[OK] No obsolete files or stale imports found in active source tree.")
    return errors


def audit_markdown_links() -> int:
    print("\n--- 6. Auditing Internal Markdown Links ---")
    import re
    errors = 0
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    for md_file in ROOT_DIR.rglob("*.md"):
        if ".git" in md_file.parts or ".pytest_cache" in md_file.parts:
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
            for match in link_pattern.finditer(text):
                target = match.group(2).strip()
                if (
                    target.startswith("http://")
                    or target.startswith("https://")
                    or target.startswith("mailto:")
                    or target.startswith("#")
                    or target.startswith("file:")
                ):
                    continue
                # Strip anchor fragments
                clean_target = target.split("#")[0]
                if not clean_target:
                    continue
                target_path = (md_file.parent / clean_target).resolve()
                if not target_path.exists():
                    print(f"[BROKEN LINK] {md_file.relative_to(ROOT_DIR)}: '{target}' -> missing {target_path}")
                    errors += 1
        except Exception as e:
            print(f"[LINK SCAN ERROR] {md_file}: {e}")
            errors += 1

    if errors == 0:
        print("[OK] All relative internal Markdown links resolve successfully.")
    return errors


def audit_terminology() -> int:
    print("\n--- 7. Auditing for Neutral Technical Terminology ---")
    errors = 0
    discouraged_phrases = [
        "massive scanner batch effects",
        "authoritative source of truth",
        "world-class",
    ]
    for md_file in (ROOT_DIR / "docs").glob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8").lower()
            for phrase in discouraged_phrases:
                if phrase in text:
                    print(f"[LOADED TERMINOLOGY] {md_file.relative_to(ROOT_DIR)} contains '{phrase}'")
                    errors += 1
        except Exception as e:
            print(f"[TERM SCAN ERROR] {md_file}: {e}")
            errors += 1

    if errors == 0:
        print("[OK] Documentation uses neutral, factual technical terminology.")
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
    err_stale = audit_stale_references_and_obsolete_files()
    err_links = audit_markdown_links()
    err_term = audit_terminology()

    total_errors = err_nb + err_art + err_zero + err_doc + err_stale + err_links + err_term
    print("\n" + "=" * 60)
    if total_errors == 0:
        print("Repository structure checks passed.")
        sys.exit(0)
    else:
        print(f"AUDIT FAILED WITH {total_errors} ISSUES. Review details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
