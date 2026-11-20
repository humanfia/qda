"""Tests for source package layering."""

from tools.check_package_layers import check_tree


def test_nested_imports_are_checked_and_vendored_sources_are_skipped(tmp_path):
    """Scan aliases and function bodies while ignoring third-party code and similarly named modules."""
    quant = tmp_path / "packages/qda-quant/src/qda_quant"
    kl = tmp_path / "packages/qda-kl/src/qda_kl"
    quant.mkdir(parents=True)
    kl.mkdir(parents=True)
    (quant / "bad.py").write_text(
        "def score():\n    from qda_kl.score import score\nimport os, qda_agent as a\n"
    )
    (kl / "bad.py").write_text("def search():\n    import qda_agent.judge\n")
    (quant / "good.py").write_text("from . import qda_kl\nimport qda_kl_helpers\n")
    (kl / "good.py").write_text("import qda_quant\n")
    for directory in ("third_party", "vendor", "_vendor"):
        vendor = quant / directory
        vendor.mkdir()
        (vendor / "bad.py").write_text("import qda_kl\n")
    assert check_tree(tmp_path) == [
        "packages/qda-quant/src/qda_quant/bad.py:3: qda_quant imports qda_agent",
        "packages/qda-quant/src/qda_quant/bad.py:2: qda_quant imports qda_kl.score",
        "packages/qda-kl/src/qda_kl/bad.py:2: qda_kl imports qda_agent.judge",
    ]
