"""resolved.yaml: the config plus provenance; reading it back drops provenance."""

import importlib.metadata as md
import re

import yaml
from qda_quant.calib.sources import SOURCES
from qda_quant.output import record
from qda_quant.output.record import plugins, provenance, without_provenance, write_resolved
from qda_quant.registry import Registry


def _installed(dist: str) -> bool:
    try:
        md.distribution(dist)
    except md.PackageNotFoundError:
        return False
    return True


def test_qda_commit_names_this_checkout():
    got = record.qda_commit()
    assert got == "unknown" or re.fullmatch(r"[0-9a-f]{40}(-dirty)?", got)


def test_qda_commit_of_an_untracked_copy_is_unknown(tmp_path, monkeypatch):
    """A package copied into another git repository (a venv inside a project) is not that repository's commit."""
    import shutil
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    copy = tmp_path / "record.py"
    shutil.copy(record.__file__, copy)
    record.qda_commit.cache_clear()
    monkeypatch.setattr(record, "__file__", str(copy))
    try:
        assert record.qda_commit() == "unknown"
    finally:
        record.qda_commit.cache_clear()


def test_qda_commit_outside_a_checkout_is_unknown(monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")

    record.qda_commit.cache_clear()
    monkeypatch.setattr(record.subprocess, "run", no_git)
    try:
        assert record.qda_commit() == "unknown"
    finally:
        record.qda_commit.cache_clear()


def test_resolved_yaml_round_trips_without_provenance(tmp_path):
    cfg = {"calib": {"file": {"path": "c.jsonl", "samples": 4, "seqlen": 8, "seed": 0}}}
    prov = provenance(plugins([(SOURCES, "file", None)]))
    package, version = SOURCES.provenance(
        "file"
    )  # the entry point's distribution, or the module from a source tree
    assert prov["plugins"] == {"file": {"group": "qda.calib", "package": package, "version": version}}
    if _installed("qda-quant"):  # installed, the record names the distribution and its version
        assert (package, version) == ("qda-quant", md.version("qda-quant"))
    back = yaml.safe_load(write_resolved(tmp_path, cfg, prov).read_text())
    assert back["provenance"] == prov and without_provenance(back) == cfg
    assert not list(tmp_path.glob(".*.tmp"))


def test_one_name_from_two_groups_keeps_both():
    """Registries keyed by architecture share names (a target table and its binding rule are both ``sm90``)."""
    other = Registry("other")
    other.register("file", type("File", (), {"name": "file"}))
    got = plugins([(SOURCES, "file", None), (other, "file", None)])
    assert got["file"]["group"] == "qda.calib" and got["qda.other:file"]["group"] == "qda.other"


def test_an_unregistered_object_is_recorded_without_a_group():
    local = type("Local", (), {"name": "file"})
    assert plugins([(SOURCES, "file", local)])["file"]["group"] is None
