"""Local custody remains host-only while published checkpoint contents remain readable."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from qda_agent.config import RunConfig
from qda_agent.isolation.wall import bwrap_argv as wall_argv
from qda_agent.judge.custody import validate
from qda_agent.sched.sandbox import bwrap_argv, private_args, published_entries
from test_build_receipt import OPS, _trusted, check_reads_and_denied_writes, prepare_work
from test_judge import Rig, ckpt


def test_full_card_moe_profile_validates_the_serialized_eight_card_layout(tmp_path, monkeypatch):
    model = tmp_path / "fp8"
    model.mkdir()
    (model / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}}')
    ref = tmp_path / "ref"
    ref.mkdir()
    source_hash = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    (ref / "manifest.json").write_text(
        json.dumps({"build": {"tp": 8}, "model": {"config_sha256": source_hash}})
    )
    raw = {
        "model": str(model),
        "format": "w4a16",
        "budget_gpu_hours": 8,
        "contamination": "none",
        "cards": list(range(8)),
        "run_dir": str(tmp_path / "run"),
        "toolset": {"name": "qda", "components": ["qda", "kl", "qsched"]},
        "judge": {
            "full_card_moe": True,
            "require_build_receipt": True,
            "anchor": str(model),
            "kl_ref": str(ref),
            "kl_layout": {"tp": 8},
            "custody_dir": str(tmp_path / "custody"),
            "max_run_bytes": 380_000_000_000,
            "max_copy_bytes": 190_000_000_000,
        },
    }
    cfg = RunConfig.model_validate(raw)
    assert cfg.judge.full_card_moe and cfg.judge.kl_layout.cards == len(cfg.cards)
    assert cfg.judge.gencheck_cards == cfg.judge.check_cards == 8
    assert cfg.judge.check_cmd[-2:] == ["--tp", "{cards}"]
    restored = RunConfig.model_validate(cfg.model_dump())
    assert restored.judge.check_cmd == cfg.judge.check_cmd
    assert restored.judge.full_card_moe
    for key, value in (
        ("cards", [0]),
        ("max_run_bytes", 1),
        ("kl_layout", {"tp": 4}),
        ("require_build_receipt", False),
        ("anchor", tmp_path / "different"),
    ):
        changed = dict(raw)
        if key == "cards":
            changed[key] = value
        else:
            changed["judge"] = {**raw["judge"], key: value}
        with pytest.raises(ValueError, match="full_card_moe"):
            RunConfig.model_validate(changed)
    (ref / "manifest.json").write_text('{"build": {"tp": 8}, "model": {"config_sha256": "other"}}')
    with pytest.raises(ValueError, match="config_sha256"):
        RunConfig.model_validate(raw)
    (ref / "manifest.json").write_text(
        json.dumps({"build": {"tp": 8}, "model": {"config_sha256": source_hash}})
    )
    read_text = Path.read_text
    mountinfo = "1 0 0:1 / / rw - ext4 /dev/root rw\n"
    mountinfo += f"2 1 0:2 / {tmp_path / 'run'} rw - nfs4 shared:/run rw\n"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            mountinfo if str(path) == "/proc/self/mountinfo" else read_text(path, *args, **kwargs)
        ),
    )
    with pytest.raises(ValueError, match="filesystem nfs4"):
        RunConfig.model_validate(raw)
    mountinfo = mountinfo.replace(str(tmp_path / "run"), str(tmp_path / "custody"))
    with pytest.raises(ValueError, match="filesystem nfs4"):
        RunConfig.model_validate(raw)


@pytest.mark.parametrize("bad", ["existing", "link", "writable", "foreign", "overlap", "missing"])
def test_local_custody_refuses_untrusted_placement(tmp_path, monkeypatch, bad):
    parent = tmp_path / "parent"
    parent.mkdir()
    local = parent / "local"
    inputs = []
    if bad == "existing":
        local.mkdir()
    elif bad == "link":
        parent.rename(tmp_path / "actual")
        parent.symlink_to(tmp_path / "actual", target_is_directory=True)
    elif bad == "writable":
        parent.chmod(0o777)
    elif bad == "foreign":
        monkeypatch.setattr(os, "getuid", lambda: -1)
    elif bad == "overlap":
        inputs = [parent]
    else:
        local = parent / "absent" / "local"
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(local, inputs)


def test_custody_is_an_output_and_requires_both_sandboxes(tmp_path):
    config = {
        "model": "m",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "contamination": "none",
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "toolset": {"name": "qda", "components": []},
        "judge": {"custody_dir": str(tmp_path / "local"), "max_run_bytes": 1 << 20},
    }
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    cfg = RunConfig.load(path)
    assert cfg.judge.custody_dir == tmp_path / "local"
    assert not cfg.judge.custody_dir.exists()
    assert cfg.resolved()["judge"]["custody_dir"] == str(tmp_path / "local")
    for overrides in ({"sandbox": "none"}, {"scheduler.job_sandbox": "none"}):
        with pytest.raises(ValueError, match="bwrap"):
            RunConfig.load(path, overrides)
    config["judge"] = {}
    path.write_text(json.dumps(config))
    assert "custody_dir" not in RunConfig.load(path).resolved()["judge"]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
@pytest.mark.parametrize("wall", [False, True])
def test_local_custody_publication_is_readonly_and_private_trees_are_hidden(tmp_path, wall):
    from qda_agent.judge import build_receipt

    local = tmp_path / "local"
    original = ckpt(tmp_path / "original")
    r = Rig(
        tmp_path,
        custody_dir=local,
        max_run_bytes=1 << 30,
        max_copy_bytes=1 << 20,
        require_build_receipt=True,
        model=str(original),
    )
    r.host.qda_argv = ["/usr/bin/python3", "-m", "qda_quant.cli"]
    _, private = _trusted(r, "candidate")
    build_receipt.publish(private)
    prepare_work(r.work)
    assert local.stat().st_uid == os.getuid() and local.stat().st_mode & 0o777 == 0o700
    assert (r.host.jdir / "builds").resolve() == local / "builds"
    assert (r.host.jdir / "frozen").resolve() == local / "frozen"
    (local / "frozen" / "secret").write_text("private frozen")
    paths = [local / "frozen" / "secret", private / "receipt.json"]
    extra = "\nfor p in " + repr([str(p) for p in paths]) + ":\n assert not Path(p).exists(), p\n"
    code = OPS + extra
    readonly = published_entries(r.run, r.work)
    if wall:
        argv = wall_argv(
            ["/usr", "/lib", "/lib64", "/bin"],
            [str(r.work)],
            [local, r.host.jdir],
            keep=[r.work],
            binds=dict(readonly),
            minimal=True,
            pid="private",
        )
        argv += ["/usr/bin/python3", "-I", "-c", code, str(r.work)]
    else:
        argv = bwrap_argv(
            ["/usr/bin/python3", "-I", "-c", code, str(r.work)],
            str(r.work),
            writable=[r.work],
            hide=r.host.sched.cfg.hide,
            readonly=readonly,
            pid="private",
        )
    with private_args(argv) as (command, fds):
        result = subprocess.run(command, capture_output=True, text=True, pass_fds=fds)
    assert result.returncode == 0, result.stderr
    check_reads_and_denied_writes(json.loads(result.stdout.removeprefix("PROBE ")))
