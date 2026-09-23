"""
Automated lint test asserting no host-specific hardcoded paths or file:/// URLs exist in repository code, docs, configs, or notebooks.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BANNED_PATTERNS = [
    (re.compile(r'/home/nvidia'), "Host path '/home/nvidia'"),
    (re.compile(r'/lp-dev/'), "Host path '/lp-dev/'"),
    (re.compile(r'file:///'), "Absolute URL scheme 'file:///'"),
]


def test_no_banned_patterns_in_source_code():
    """Verify that all Python source files in src/ are free of host paths and file:/// URLs."""
    src_files = list((ROOT / "src").rglob("*.py"))
    assert len(src_files) > 0, "No source files found in src/"

    violations = []
    for py_file in src_files:
        content = py_file.read_text(encoding="utf-8")
        for pattern, desc in BANNED_PATTERNS:
            match = pattern.search(content)
            if match:
                violations.append(f"{py_file.relative_to(ROOT)}: contains {desc}")

    assert not violations, "Banned path patterns detected in source code:\n" + "\n".join(violations)


def test_no_banned_patterns_in_documentation():
    """Verify that no banned patterns exist in any documentation markdown files."""
    md_files = list(ROOT.rglob("*.md"))
    assert len(md_files) > 0, "No markdown files found"

    violations = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        for pattern, desc in BANNED_PATTERNS:
            match = pattern.search(content)
            if match:
                violations.append(f"{md_file.relative_to(ROOT)}: contains {desc}")

    assert not violations, "Banned path patterns detected in documentation:\n" + "\n".join(violations)


def test_no_hardcoded_paths_in_configs():
    """Verify that configuration files do not contain personal host directory paths or file:/// URLs."""
    config_files = list((ROOT / "configs").rglob("*.json"))
    violations = []
    for cfg in config_files:
        if cfg.name == "w2b_environment.json":
            continue
        content = cfg.read_text(encoding="utf-8")
        for pattern, desc in BANNED_PATTERNS:
            match = pattern.search(content)
            if match:
                violations.append(f"{cfg.relative_to(ROOT)}: contains {desc}")

    assert not violations, "Banned path patterns detected in configs:\n" + "\n".join(violations)


def test_no_banned_patterns_in_notebook_sources():
    """Verify that all canonical notebook code cells are free of banned host path patterns."""
    nb_files = list((ROOT / "notebooks").rglob("*.ipynb"))
    assert len(nb_files) > 0, "No notebooks found"

    violations = []
    for nb_path in nb_files:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
        for idx, cell in enumerate(nb_data.get("cells", [])):
            src_text = "".join(cell.get("source", []))
            for pattern, desc in BANNED_PATTERNS:
                if pattern.search(src_text):
                    violations.append(f"{nb_path.relative_to(ROOT)} Cell {idx}: contains {desc}")

    assert not violations, "Banned path patterns detected in notebook sources:\n" + "\n".join(violations)
