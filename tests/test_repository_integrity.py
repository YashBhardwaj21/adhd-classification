import csv, os, re
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(p):
    return open(os.path.join(REPO_ROOT, p), encoding="utf-8").read()


def _manifest_rows():
    mp = os.path.join(REPO_ROOT, "results", "manifest.csv")
    with open(mp, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("row", _manifest_rows(), ids=lambda r: r.get("source", r.get("artifact", "?")))
def test_manifest_file_exists(row):
    rel = row.get("source", "").strip()
    assert rel, "Empty source in manifest row"
    assert os.path.isfile(os.path.join(REPO_ROOT, rel)), f"Manifest source missing: {rel}"


REQUIRED_CONFIGS = [
    "configs/exp09/gnn_configuration.json",
    "configs/exp09/graph_preprocessing_configuration.json",
    "configs/exp09/execution_environment.json",
    "configs/exp09/dataset_manifest.json",
    "configs/exp09/node_feature_selection.json",
    "configs/exp09/graph_threshold_selection.json",
]


@pytest.mark.parametrize("rel", REQUIRED_CONFIGS)
def test_required_config_exists(rel):
    assert os.path.isfile(os.path.join(REPO_ROOT, rel)), f"Config missing: {rel}"


PUBLIC_DOCS = [
    "README.md",
    "docs/methods.md",
    "docs/results.md",
    "docs/study_design.md",
    "docs/reproduction.md",
    "docs/limitations.md",
    "environment/README.md",
]

STALE_PATTERNS = [r"\bw1_[a-z]", r"\bw2b_[a-z]", r"\bw2c_[a-z]"]


@pytest.mark.parametrize("doc", PUBLIC_DOCS)
@pytest.mark.parametrize("pat", STALE_PATTERNS)
def test_no_stale_prefix(doc, pat):
    # Strip parenthetical historical-name notes before checking
    cleaned = re.sub(r"\(historical[^)]*\)", "", _read(doc))
    hits = re.findall(pat, cleaned)
    assert not hits, f"{doc} has stale pattern {pat!r}: {hits}"


def test_reproduction_no_conda_env_yml():
    bad = [
        ln for ln in _read("docs/reproduction.md").splitlines()
        if "environment.yml" in ln and "conda env create" in ln
    ]
    assert not bad, "stale 'conda env create -f environment.yml' found: " + str(bad)


NO_AUDIT = [
    "README.md",
    "docs/results.md",
    "docs/study_design.md",
    "docs/methods.md",
    "docs/reproduction.md",
    "docs/limitations.md",
    "results/exp09/README.md",
]


@pytest.mark.parametrize("doc", NO_AUDIT)
def test_no_audit_source_files_ref(doc):
    hits = re.findall(r"audit_source_files/", _read(doc))
    assert not hits, f"{doc} references non-public audit_source_files/"
