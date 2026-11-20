"""Every loader reads ``org/name@rev`` the same way, through `intake.model_ref.parse`."""

import json
from pathlib import Path

import huggingface_hub
import pytest
import transformers
from qda_quant.intake.model_ref import parse

REPO, REV = "org/tiny", "0123abc"


def test_a_hub_reference_splits_at_its_revision_and_a_path_never_does(tmp_path, monkeypatch):
    assert parse(f"{REPO}@{REV}") == (REPO, REV)
    assert parse(REPO) == (REPO, None) and parse(f"{REPO}@") == (REPO, None)
    assert parse(f"{REPO}@{REV}").hub() == {"revision": REV} and parse(REPO).hub() == {}
    local = tmp_path / "ckpt@v2"
    local.mkdir()
    assert parse(str(local)) == (str(local), None)
    monkeypatch.chdir(tmp_path)
    assert parse("ckpt@v2") == ("ckpt@v2", None)  # an existing relative path is a path
    assert parse(Path("missing@v2")) == ("missing@v2", None)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake hub that knows only `REPO` at `REV`: a Qwen3 config, served to transformers and the hub client."""
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 64,
            }
        )
    )
    asked: list[tuple[str, str | None]] = []
    original = transformers.AutoConfig.from_pretrained

    def config(name, *args, revision=None, **kw):
        asked.append((str(name), revision))
        if (str(name), revision) != (REPO, REV):
            raise OSError(f"{name} is not a local folder and is not a valid model identifier")
        return original(str(snap), *args, **kw)

    def download(repo_id, revision=None, **_):
        asked.append((repo_id, revision))
        return str(snap)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return asked


def test_load_config_reads_a_pinned_hub_id(hub):
    from qda_quant.intake.loading import load_config

    assert load_config(f"{REPO}@{REV}").model_type == "qwen3"
    assert hub == [(REPO, REV)]


def test_load_model_hands_transformers_the_repo_and_its_revision(hub, monkeypatch):
    from qda_quant.intake.loading import load_model

    seen = {}

    def weights(name, *, config, **kw):
        seen.update(name=name, **kw)
        return transformers.AutoModelForCausalLM.from_config(config)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", weights)
    load_model(f"{REPO}@{REV}")
    assert seen == {"name": REPO, "revision": REV}


def test_local_source_downloads_the_pinned_revision(hub):
    from qda_quant.engine.planning import local_source

    local_source(f"{REPO}@{REV}", weights=False)
    assert hub == [(REPO, REV)]


def test_a_probe_reads_the_pinned_revision_through_the_hub_api(monkeypatch):
    from qda_quant.intake.probe import Source

    asked = []

    class Api:
        def list_repo_files(self, repo, revision=None):
            asked.append((repo, revision))
            return ["config.json"]

    def file(repo, name, revision=None):
        asked.append((repo, revision))
        raise AssertionError("stop here: the revision is what this test checks")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", file)
    with pytest.raises(AssertionError, match="stop here"):
        Source.open(f"{REPO}@{REV}")
    assert asked == [(REPO, REV), (REPO, REV)]


def test_a_rotated_draft_and_a_source_identity_read_the_reference_the_same_way(tmp_path):
    from qda_kl.config import source_identity
    from qda_quant.output.rotate_draft import beside

    assert beside(tmp_path / "ckpt", f"org/dflash-tiny@{REV}") == tmp_path / "ckpt-dflash-tiny"
    assert source_identity(f"{REPO}@{REV}")["revision"] == REV


SHA = "c" * 40


def _pinned_cache(root: Path, *, main: bool = False) -> None:
    """What ``hf download org/tiny --revision SHA`` leaves: the snapshot, and ``refs/main`` only when asked."""
    repo = root / "models--org--tiny"
    (repo / "snapshots" / SHA).mkdir(parents=True)
    if main:
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(SHA)


@pytest.fixture
def offline(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    return tmp_path


def test_offline_a_cached_revision_or_ref_passes(offline, monkeypatch):
    from huggingface_hub import constants
    from qda_quant.intake.model_ref import offline_refusal

    _pinned_cache(offline)
    assert offline_refusal(f"{REPO}@{SHA}") is None  # the pinned snapshot is there
    _pinned_cache(offline / "other", main=True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(offline / "other"))
    assert offline_refusal(REPO) is None  # refs/main maps the bare id
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    assert offline_refusal("org/never-downloaded") is None  # online, the hub answers
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    for path in (
        "missing-dir",
        "./a/b",
        "/abs/c",
        "~/d/e",
    ):  # cannot be org/name: the caller says it is missing
        assert offline_refusal(path) is None


def test_offline_a_pinned_download_without_refs_main_names_the_cached_revision(offline):
    """The documented offline download pins a commit, so the cache has no ``refs/main`` for the bare id."""
    from qda_quant.exit import ENVIRONMENT
    from qda_quant.intake.model_ref import offline_refusal

    _pinned_cache(offline)
    got = offline_refusal(REPO)
    assert got is not None and got.code == ENVIRONMENT
    assert str(got) == (
        f"{REPO}: the local Hugging Face cache has no `main` ref for it (it holds {SHA}) "
        "and the hub cannot be reached (HF_HUB_OFFLINE is set)"
    )
    assert got.fix == (
        f"name the cached revision, {REPO}@{SHA}, or download it once with network access: hf download {REPO}"
    )
    other = "d" * 40
    got = offline_refusal(f"{REPO}@{other}")
    assert got is not None and f"no snapshot at revision {other}" in str(got)
    assert got.fix.endswith(f"hf download {REPO} --revision {other}")
    got = offline_refusal("org/never-downloaded@v1")
    assert got is not None and "holds no snapshot of it" in str(got)
    assert got.fix == "download it once with network access: hf download org/never-downloaded --revision v1"
