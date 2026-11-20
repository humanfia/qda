"""A fine-tune as the search's own model: its own KL reference, and nothing wired to a model cache.

A run scores every candidate against `judge.kl_ref`, so the reference must be the one built from THIS
run's original model; a fine-tune is its own original. These tests pin the check that says so, the two
cases it cannot check (a Hub id, a manifest without the hash) which are reported not refused, and that
qda-agent names no model-cache path of its own.
"""

import ast
import hashlib
import json
import re
import warnings
from pathlib import Path

import pytest
from qda_agent.config import RunConfig
from tests_support import filled

CONFIGS = Path(__file__).parents[1] / "configs"
SOURCES = Path(__file__).parents[1] / "src" / "qda_agent"


def a_model(tmp_path: Path, name: str = "finetune") -> Path:
    """A local checkpoint directory: a config.json is what the reference records a hash of."""
    model = tmp_path / name
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "name": name}))
    return model


def a_reference(tmp_path: Path, model: Path | None, *, record: bool = True) -> Path:
    """A qda-kl reference directory; `model` is the original whose config hash it records."""
    reference = tmp_path / "reference"
    reference.mkdir(exist_ok=True)
    manifest: dict = {"build": {"tp": 1}}
    if record and model is not None:
        sha = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
        manifest["model"] = {"name": model.name, "config_sha256": sha}
    (reference / "manifest.json").write_text(json.dumps(manifest))
    return reference


def config(tmp_path: Path, *, model: Path | str, reference: Path | None, anchor: Path | None) -> RunConfig:
    """The shipped search config with this model, anchor and reference."""
    path = filled(CONFIGS / "search.yaml", tmp_path / "cfg")
    overrides = [f"model={model}", f"run_dir={tmp_path / 'run'}"]
    overrides.append("judge.anchor=null" if anchor is None else f"judge.anchor={anchor}")
    overrides.append("judge.kl_ref=null" if reference is None else f"judge.kl_ref={reference}")
    return RunConfig.load(path, overrides)


@pytest.mark.parametrize("level", ["none", "strict"])
def test_a_local_finetune_with_its_own_reference_loads(tmp_path, level):
    """A fine-tune is a local directory like any original; its own reference passes the check."""
    model = a_model(tmp_path)
    reference = a_reference(tmp_path, model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to report: the reference is this model's own
        cfg = config(tmp_path, model=model, reference=reference, anchor=model)
    assert cfg.model == str(model) and cfg.isolation.level in ("none", "strict")
    cfg = RunConfig.load(
        filled(CONFIGS / "search.yaml", tmp_path / "cfg2"),
        [f"model={model}", f"judge.anchor={model}", f"judge.kl_ref={reference}", f"isolation.level={level}"],
    )
    assert cfg.isolation.level == level


def test_a_reference_from_another_model_is_refused(tmp_path):
    """The failure this check exists for: a fine-tune scored against the model it was fine-tuned from."""
    finetune = a_model(tmp_path, "finetune")
    base = a_model(tmp_path, "base")
    reference = a_reference(tmp_path, base)  # built from the base model
    with pytest.raises(ValueError, match="built from another model") as raised:
        config(tmp_path, model=finetune, reference=reference, anchor=finetune)
    assert "qda-kl ref --model" in str(raised.value) and str(finetune) in str(raised.value)


def test_a_hub_id_cannot_be_checked_and_says_so_once(tmp_path):
    """A Hub id names no local config.json: the check says what it could not check, and does not refuse."""
    reference = a_reference(tmp_path, a_model(tmp_path))
    with pytest.warns(UserWarning, match="not a local checkpoint directory") as caught:
        config(tmp_path, model="acme/finetune-27b", reference=reference, anchor=None)
    assert len(caught) == 1


def test_a_manifest_without_the_hash_says_so_once(tmp_path):
    model = a_model(tmp_path)
    reference = a_reference(tmp_path, model, record=False)
    with pytest.warns(UserWarning, match="records no model.config_sha256") as caught:
        config(tmp_path, model=model, reference=reference, anchor=model)
    assert len(caught) == 1


def test_an_unreadable_manifest_says_so_once(tmp_path):
    model = a_model(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    with pytest.warns(UserWarning, match="missing or unreadable") as caught:
        config(tmp_path, model=model, reference=reference, anchor=model)
    assert len(caught) == 1


def test_the_same_unchecked_reference_is_reported_once(tmp_path):
    """Loading the same config twice (a run re-reads it) repeats the reason once, not once per load."""
    reference = a_reference(tmp_path, a_model(tmp_path))
    first = config(tmp_path, model="acme/finetune-27b", reference=reference, anchor=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # already reported: a second load is silent
        second = config(tmp_path, model=first.model, reference=reference, anchor=None)
    assert second.judge.kl_ref == reference


def test_a_config_without_a_reference_is_silent(tmp_path):
    model = a_model(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config(tmp_path, model=model, reference=None, anchor=model)


#: An absolute path into a model cache or snapshot: `/.../hf_cache/...`, `/.../models--X/...`, `/.../snapshots/...`.
CACHE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.\-])/(?:[^/\s]+/)*(?:hf_cache|models--[^/\s]*|snapshots)(?![A-Za-z0-9_.\-])"
)


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Every string constant in a source file, with its line (docstrings included)."""
    tree = ast.parse(path.read_text())
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _cache_path_hits(sources: list[Path]) -> list[str]:
    """Where these sources hold an absolute path into a model cache or snapshot, and the path."""
    hits: list[str] = []
    for source in sorted(sources):
        for line, value in _string_literals(source):
            hits.extend(f"{source}:{line}: {match.group(0)}" for match in CACHE_PATH.finditer(value))
    return hits


def test_qda_agent_hard_wires_no_model_cache_path():
    """No string in qda-agent is an absolute path into a model cache or snapshot: a fine-tune is `model`.

    A model cache or snapshot is named by the operator (`model:` is a local directory or a Hub id), never
    by us. The rule is on the string, so no file is exempt: the hub-snapshot link helper (`bwrap.py`)
    passes because it names only the bare layout token and elided spellings, not a path.
    """
    assert _cache_path_hits(list(SOURCES.rglob("*.py"))) == []


def test_a_planted_checkpoint_path_fails_even_in_the_hub_link_helper(tmp_path):
    """A hard-wired checkpoint path anywhere, `bwrap.py` included, is a hit: file name buys no exemption."""
    planted = tmp_path / "bwrap.py"
    planted.write_text('CHECKPOINT = "/opt/models--Qwen--x/snapshots/abc"\n')
    hits = _cache_path_hits([planted])
    assert hits != [] and "/opt/models--Qwen--x/snapshots" in hits[0]


def test_the_bare_hub_layout_token_is_not_a_path(tmp_path):
    """The layout as a helper names it (`models--`, an elided `.../models--X/snapshots/<rev>`) is no path."""
    helper = tmp_path / "bwrap.py"
    helper.write_text('A = ".../models--X/snapshots/<rev>/config.json"\nB = "models--"\nC = "snapshots"\n')
    assert _cache_path_hits([helper]) == []
