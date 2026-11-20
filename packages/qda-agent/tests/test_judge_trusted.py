"""The judge's own inputs (anchor, baselines, format reference) may be Hugging Face cache snapshots: a
directory of links into its repository's `blobs/`, themselves links into the cache's shared store. The
served files' links must stay in that repository; the release's other files are ignored."""

from pathlib import Path

import pytest
from qda_agent.judge.checks import tensor_fingerprint, trusted_files_problem
from qda_agent.judge.config import JudgeConfig
from safetensors_kit import anchor_dir, quant_dir
from test_judge import PY, Rig, ckpt

#: What a public release carries besides the engine's files (HF and ModelScope metadata, a licence).
EXTRAS = (".gitattributes", ".mdl", ".msc", ".mv", "README.md", "LICENSE", "configuration.json")


def release(d: Path) -> Path:
    """`d` with a release's other files next to the engine's."""
    for name in EXTRAS:
        (d / name).write_text("x")
    return d


def snapshot(d: Path, hub: Path) -> Path:
    """`d`'s files as a hub cache keeps them: snapshot links -> the repository's `blobs/` -> the shared store."""
    repo = hub / "models--X"
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (hub / "blobs" / "12").mkdir(parents=True)
    for f in sorted(d.iterdir()):
        real = hub / "blobs" / "12" / f"{f.name}.{len(f.read_bytes())}"
        real.write_bytes(f.read_bytes())
        (repo / "blobs" / f.name).symlink_to(f"../../blobs/12/{real.name}")
        (snap / f.name).symlink_to(f"../../blobs/{f.name}")
    return snap


def test_a_snapshot_is_a_trusted_input_for_each_setting(tmp_path):
    snap = snapshot(release(anchor_dir(tmp_path / "real")), tmp_path / "hub")
    cfg = JudgeConfig(
        contamination="none",
        kl_cmd=["score", "{ckpt}", "{out}"],
        anchor=snap,
        baselines=[snap],
        format_reference=snap,
    )
    assert (cfg.anchor, cfg.baselines, cfg.format_reference) == (snap, [snap], snap)
    assert (
        trusted_files_problem(snap) is None
    )  # links into the repo, and a release's other files are not read


def test_a_snapshot_baseline_is_admitted_and_hashed_through_its_links(tmp_path):
    anchor = release(anchor_dir(tmp_path / "anchor"))
    real = release(quant_dir(tmp_path / "real", b"base"))
    snap = snapshot(real, tmp_path / "hub")
    r = Rig(tmp_path, anchor=anchor, baselines=[snap], allow_any_format=True, qda_cmd=[PY])
    r.host.start()
    try:
        # the baseline is its regular-file self: the fingerprint reads what the links resolve to
        assert r.host.baseline_digests[snap] == tensor_fingerprint(real)
        ckpt(r.work / "a", b"w2")
        rep = r.call("handin", "a")
        assert rep.exit == 0, rep.stderr
    finally:
        r.host.stop()


@pytest.mark.parametrize("escape", ["outside", "store"])
def test_a_snapshot_whose_served_file_leaves_its_repo_is_refused(tmp_path, escape):
    snap = snapshot(release(anchor_dir(tmp_path / "real")), tmp_path / "hub")
    (snap / "config.json").unlink()
    if escape == "outside":
        secret = tmp_path / "secret.json"
        secret.write_text("{}")
        (snap / "config.json").symlink_to(secret)
    else:  # straight into the shared store, not through this repository's own blobs/ entry
        other = tmp_path / "hub" / "blobs" / "34" / "y"
        other.parent.mkdir(parents=True)
        other.write_text("{}")
        (snap / "config.json").symlink_to("../../../blobs/34/y")
    with pytest.raises(ValueError, match=r"baselines\[0\].*not a regular file inside the model"):
        Rig(tmp_path, baselines=[snap], allow_any_format=True)


def test_a_snapshots_other_files_are_never_read_so_their_links_are_not_checked(tmp_path):
    snap = snapshot(release(anchor_dir(tmp_path / "real")), tmp_path / "hub")
    secret = tmp_path / "secret"
    secret.write_text("x")
    for name in EXTRAS:
        (snap / name).unlink()
        (snap / name).symlink_to(secret)
    assert trusted_files_problem(snap) is None


def test_a_directory_that_is_not_a_snapshot_may_not_link_out_of_itself(tmp_path):
    model = release(anchor_dir(tmp_path / "model"))
    secret = tmp_path / "secret.json"
    secret.write_text("{}")
    (model / "config.json").unlink()
    (model / "config.json").symlink_to(secret)
    assert "not a regular file inside the model directory" in (trusted_files_problem(model) or "")
