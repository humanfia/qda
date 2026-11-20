import os
import shutil
from pathlib import Path

import pytest


def disk_space(cfg):
    """The start's `disk space` guard for `cfg`, walked as `qda doctor --search` walks it (`STATIC_GUARDS`)."""
    from datetime import UTC, datetime

    from qda_agent.judge.config import JudgeConfig
    from qda_agent.judge.startup import STATIC_GUARDS, Start
    from qda_agent.sched import SchedConfig
    from qda_quant.guards import SEARCH_START, all_problems

    sched = SchedConfig.model_validate(cfg.sched_config(datetime.now(UTC)))
    start = Start.of(JudgeConfig.model_validate(cfg.judge_config()), sched, cfg.run_dir, cfg.workdir)
    (found,) = [v for v in all_problems(STATIC_GUARDS, {SEARCH_START}, start) if v.guard.name == "disk space"]
    return found


def two_hop_cache(tmp_path, unreadable=False):
    """A snapshot whose repository blob links into the shared store's second hop, as a real cache has it."""
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    blob = "a" * 64
    store = hub / "blobs" / "aa"
    store.mkdir(parents=True)
    weights = store / blob
    weights.write_text("weights\n")
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / blob).symlink_to(f"../../blobs/aa/{blob}")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(f"../../blobs/{blob}")
    if unreadable:
        weights.chmod(0)  # the final store file, which only the second hop reaches
    return snap


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_the_sandbox_check_reads_a_jobs_weight(tmp_path):
    """`qda doctor --search` opens a weight through the job's binds, so a two-hop cache passes."""
    from qda_agent import doctor as d
    from qda_agent.sched import SchedConfig

    snap = two_hop_cache(tmp_path)
    c = d.check_bwrap(SchedConfig(cards=[0], readable=[snap]))
    assert c.ok, c.detail
    assert "reads model.safetensors (8 bytes)" in c.detail


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file with no permissions")
def test_the_sandbox_check_refuses_a_weight_a_job_cannot_read(tmp_path):
    """A store file a job cannot open is a MISS naming the file, not a sandbox that starts."""
    from qda_agent import doctor as d
    from qda_agent.sched import SchedConfig

    snap = two_hop_cache(tmp_path, unreadable=True)
    c = d.check_bwrap(SchedConfig(cards=[0], readable=[snap]))
    assert not c.ok and not c.skipped
    assert "model.safetensors" in c.detail and "cannot read" in c.detail


def bound_config(tmp_path, unreadable=False):
    """A run's settings: the model readable, a baseline bound at `work/baselines/0` as a run binds it."""
    from qda_agent.sched import SchedConfig

    snap = two_hop_cache(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    weight = baseline / "small.safetensors"
    weight.write_text("weights\n")
    if unreadable:
        weight.chmod(0)
    return SchedConfig(
        cards=[0],
        gpus=False,
        readable=[snap],
        binds={run / "work" / "baselines" / "0": baseline},
        run_dir=run,
    )


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_a_bound_baseline_a_job_can_read_passes(tmp_path):
    """The configured baseline mounts under the workdir, as a run mounts `work/baselines` then `.../0`."""
    from qda_agent import doctor as d

    c = d.check_bwrap(bound_config(tmp_path))
    assert c.ok, c.detail


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file with no permissions")
def test_a_bound_baseline_a_job_cannot_read_is_a_miss(tmp_path):
    """The baseline is read at the path a job gets it: unreadable is a MISS naming it."""
    from qda_agent import doctor as d

    c = d.check_bwrap(bound_config(tmp_path, unreadable=True))
    assert not c.ok and not c.skipped
    inside = tmp_path / "run" / "work" / "baselines" / "0" / "small.safetensors"
    assert str(inside) in c.detail and "cannot read" in c.detail


def test_card_order(tmp_path, monkeypatch):
    import subprocess

    from qda_agent import doctor as d
    from qda_agent.sched import SchedConfig

    proc = tmp_path / "gpus"
    for minor, bus in (
        (7, "0000:18:00.0"),
        (6, "0000:2a:00.0"),
        (0, "0000:3a:00.0"),
        (1, "0000:5d:00.0"),
        (2, "0000:9a:00.0"),
        (3, "0000:ab:00.0"),
    ):
        (proc / bus).mkdir(parents=True)
        (proc / bus / "information").write_text(
            f"GPU UUID: \t GPU-{minor}\nDevice Minor: \t {minor}\nBus Location: \t {bus}\n"
        )

    def smi(out):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, out, "")

    cfg = SchedConfig(cards=[4, 5])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # a whole node; a one-card job numbers its own
    monkeypatch.setattr(d.subprocess, "run", smi("4, 00000000:9A:00.0\n5, 00000000:AB:00.0\n"))
    assert d.card_order_info(cfg, proc).detail.endswith("nvidia-smi index == PCI-bus-order ordinal")
    monkeypatch.setattr(d.subprocess, "run", smi("4, 00000000:AB:00.0\n5, 00000000:9A:00.0\n"))
    c = d.card_order_info(cfg, proc)
    assert "[4, 5]" in c.detail and "!=" in c.detail
    assert d.card_order_info(cfg, tmp_path / "none").detail == "no NVIDIA driver"


def gpu_table(tmp_path):
    """A driver table of six cards, minors [7,6,0,1,2,3] in bus order, with their device nodes."""
    proc, dev = tmp_path / "gpus", tmp_path / "dev"
    dev.mkdir()
    for minor, bus in (
        (7, "0000:18:00.0"),
        (6, "0000:2a:00.0"),
        (0, "0000:3a:00.0"),
        (1, "0000:5d:00.0"),
        (2, "0000:9a:00.0"),
        (3, "0000:ab:00.0"),
    ):
        (proc / bus).mkdir(parents=True)
        (proc / bus / "information").write_text(
            f"GPU UUID: \t GPU-{minor}\nDevice Minor: \t {minor}\nBus Location: \t {bus}\n"
        )
        (dev / f"nvidia{minor}").touch()
    return proc, dev


def test_a_run_whose_cards_include_a_faulty_card_is_refused(tmp_path, monkeypatch):
    """A fatal ECC fault on one card of eight; cards are matched to nvidia-smi's rows by PCI bus."""
    import subprocess

    from qda_agent import doctor as d

    proc, _ = gpu_table(tmp_path)
    buses = ["18", "2A", "3A", "5D", "9A", "AB"]  # ordinals 0-5

    def smi(faulty: dict[str, str]):
        rows = "".join(f"00000000:{b}:00.0, {faulty.get(b, '0, [N/A]')}\n" for b in buses)
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, rows, "")

    monkeypatch.setattr(d.subprocess, "run", smi({}))
    assert d.card_health([4, 5], proc) == (None, None)
    monkeypatch.setattr(d.subprocess, "run", smi({"9A": "2, [N/A]"}))  # ordinal 4
    bad, _ = d.card_health([4, 5], proc)
    assert bad == "card 4: 2 uncorrectable ECC error(s) since the driver loaded"
    assert d.card_health([0, 1], proc) == (None, None)  # a run on other cards does not need it

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(d.subprocess, "run", missing)
    assert d.card_health([4], proc) == (None, "no nvidia-smi listing")  # judged nowhere: a skip, not a pass


def test_cards_outside_cuda_visible_devices_are_refused_naming_no_device(tmp_path, monkeypatch):
    """`cards` are physical numbers; a card this shell was not given is refused in one sentence that
    never shows a device node. The start's own `cards visible` guard is the only judge of it."""
    from qda_agent.isolation.devices import invisible_cards

    proc, _ = gpu_table(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    why = invisible_cards([0], proc=proc)
    assert why is not None and "physical card numbers" in why and "nvidia" not in why
    assert invisible_cards([5], proc=proc) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-2,GPU-3")  # the same cards by UUID
    assert invisible_cards([4, 5], proc=proc) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,GPU-3")  # a number and a UUID name the two cards too
    assert invisible_cards([4, 5], proc=proc) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # a CPU session hides the cards: nothing to mismatch
    assert invisible_cards([4, 5], proc=proc) is None


def test_a_run_refuses_cards_outside_cuda_visible_devices(tmp_path, monkeypatch, capsys):
    from qda_agent import cli
    from qda_agent.config import RunConfig

    cfg = RunConfig.load(_shipped(tmp_path), [f"run_dir={tmp_path / 'run'}", "cards=[0]"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert cli.execute(cfg, worker=object()) == 3  # the same command works once the shell names the card
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "cards [0] are not in CUDA_VISIBLE_DEVICES='5'" in err
    assert not (tmp_path / "run").exists()


def _shipped(tmp_path):
    """The shipped search config, filled under `tmp_path`, with a baseline that is a checkpoint."""
    from tests_support import filled

    path = filled(Path(__file__).parents[1] / "configs" / "search.yaml", tmp_path / "cfg")
    (tmp_path / "cfg" / "path" / "to" / "baseline-ckpt" / "config.json").write_text("{}")
    return path


def test_doctor_writes_nothing_under_run_dir_so_run_starts_after_it(tmp_path, monkeypatch, capsys):
    """The job-sandbox probe mounts the baselines under a throwaway workdir, never `run_dir/work`."""
    from qda_agent import cli
    from qda_agent.doctor import search
    from qda_quant.doctor import Context

    path, run = _shipped(tmp_path), tmp_path / "run"
    search(Context(search=path, overrides=(f"run_dir={run}",)))
    assert not run.exists() or not any(run.iterdir())
    monkeypatch.setattr(
        cli, "pidfd_refusal", lambda cfg: "stand-in: the run got past its empty-run-dir check"
    )
    assert cli.main(["run", str(path), "--set", f"run_dir={run}"]) == 3  # a Python that cannot hold the fence
    err = capsys.readouterr().err
    assert "error: stand-in: the run got past" in err and "not empty" not in err


def test_a_start_refusal_is_one_line_and_exit_2(tmp_path, monkeypatch, capsys):
    """A refused input ends `qda-agent run` with one line and status 2, not a traceback."""
    from types import SimpleNamespace

    from qda_agent import backends, cli
    from qda_agent.config import RunConfig
    from qda_agent.flow import CalibTemplateRefused

    def refuse():
        raise CalibTemplateRefused(
            "data set corpus x.jsonl: of 1 rows,\n1 rows hold pre-rendered chat markup"
        )

    cfg = RunConfig.load(_shipped(tmp_path), [f"run_dir={tmp_path / 'run'}"])
    monkeypatch.setattr(
        backends, "make_run", lambda cfg, resolved, worker: SimpleNamespace(run=refuse, epic=None)
    )
    assert cli.execute(cfg, worker=object()) == 2
    err = capsys.readouterr().err
    # the progress line comes first: the refusal is one of the lines, and the last thing written
    assert "error: data set corpus x.jsonl: of 1 rows, 1 rows hold" in err and "Traceback" not in err
    assert err.rstrip().splitlines()[-1].startswith("error: ")


def test_the_room_line_names_the_variable_that_points_at_a_short_disk(tmp_path, monkeypatch, fake_room):
    """The cache and temp disks are `room.cache_room`'s lines, with or without a search config."""
    from qda_agent import room as R
    from qda_agent.doctor import disk_room
    from qda_quant.doctor import Context

    monkeypatch.setattr(R, "_mount_table", lambda: [(Path("/"), "xfs")])
    monkeypatch.setattr(
        R,
        "places",
        lambda: [R.Place("HF_HOME", tmp_path, "the Hub downloads"), R.Place("TMPDIR", tmp_path, "t")],
    )
    fake_room(1_200_000_000, 3_400)
    (line,) = disk_room(Context())
    assert not line.ok and line.name == "disk room" and "HF_HOME and TMPDIR are on " in line.detail
    assert "export HF_HOME=<a larger disk>" in line.fix
    # the run directory's own disk is check_disk_cap's: a search config adds no line here
    assert disk_room(Context(search=tmp_path / "search.yaml")) == disk_room(Context())


def test_the_disk_cap_line_states_the_floor_without_a_cap_or_misses_above_the_free_space(tmp_path):
    """The 27B search's 107 GB cap was above its disk's free space, and the run filled the disk: a MISS now,
    from the same check the host refuses to start on. Without a cap (the default) there is no
     estimate to warn about: the line states the free-space floor the run is guarded by."""
    import shutil as sh

    from qda_agent.config import RunConfig
    from qda_agent.doctor import check_disk_cap

    path, run = _shipped(tmp_path), tmp_path / "runs" / "new"  # run_dir does not exist yet: its disk is read
    free = sh.disk_usage(tmp_path).free
    c = check_disk_cap(RunConfig.load(path, [f"run_dir={run}", "judge.min_free_bytes=1"]))
    assert c.ok and "unset" in c.detail and "judge.min_free_bytes floor" in c.detail
    c = check_disk_cap(RunConfig.load(path, [f"run_dir={run}", f"judge.max_run_bytes={free * 2}"]))
    assert c.missed and "above the" in c.detail and str(tmp_path) in c.detail
    c = check_disk_cap(RunConfig.load(path, [f"run_dir={run}", "judge.max_run_bytes=1000"]))
    assert c.ok


def test_the_disk_cap_misses_below_what_the_run_must_hold(tmp_path):
    """Doctor reports the run's own start check, not a second one; a cap under it is a MISS with a
    pasteable `--set judge.max_run_bytes`."""
    from qda_agent.config import RunConfig
    from qda_agent.doctor import check_disk_cap

    path, run = _shipped(tmp_path), tmp_path / "runs" / "new"
    cand0 = tmp_path / "cfg" / "path" / "to" / "baseline-ckpt"
    (cand0 / "model.safetensors").write_bytes(b"x" * 1000)
    build = 1000 + 2  # the weights just written + the config.json `_shipped` made
    need = build * (1 + 1 + 1)  # the hand-in's own copy + 1 card + 1 in flight
    short = check_disk_cap(RunConfig.load(path, [f"run_dir={run}", f"judge.max_run_bytes={need - 1}"]))
    assert short.missed and short.fix == f"--set judge.max_run_bytes={need}"
    assert "the run must hold" in short.detail
    ok = check_disk_cap(RunConfig.load(path, [f"run_dir={run}", f"judge.max_run_bytes={need}"]))
    assert ok.ok


def test_a_run_without_custody_refuses_the_same_cap_at_start_and_in_doctor(tmp_path):
    """The scenario is a run without local custody: the host start must check it too, and doctor must
    print that same refusal (one check_space, not a second opinion)."""
    from qda_agent.config import RunConfig
    from qda_agent.doctor import check_disk_cap
    from test_judge import Rig

    cand0 = tmp_path / "cand0"
    cand0.mkdir()
    (cand0 / "config.json").write_text("{}")
    (cand0 / "model.safetensors").write_bytes(b"x" * 1000)
    build = 1000 + 2  # what candidate 0's tree measures
    need = build * (1 + 2 + 1)  # the hand-in's own copy + 2 cards + 1 in flight
    with pytest.raises(ValueError) as refused:
        Rig(tmp_path, format_reference=cand0, max_run_bytes=need - 1)  # the Rig's scheduler has cards [0, 1]
    assert "the run must hold" in str(refused.value)
    assert f"--set judge.max_run_bytes={need}" in str(refused.value)
    run = Path(os.path.realpath(tmp_path / "run"))  # the host measures its resolved run_dir
    cfg = RunConfig.load(
        _shipped(tmp_path),
        [
            f"run_dir={run}",
            "cards=[0, 1]",
            f"judge.format_reference={cand0}",
            f"judge.max_run_bytes={need - 1}",
        ],
    )
    check = check_disk_cap(cfg)
    assert check.missed and check.detail == " ".join(str(refused.value).split())
    assert disk_space(cfg).problem.text == check.detail, "the line is the start's `disk space` guard"


def confined_smi(table: dict[str, str]):
    """A stand-in nvidia-smi that lists only the card whose bus is the one value of `table`."""
    import subprocess

    (bus,) = table.values()

    def run(argv, **kwargs):  # noqa: ARG001
        if any("index,pci.bus_id" in a for a in argv):
            out = f"0, {bus}\n"  # the job's only card, renumbered 0 as Slurm's device cgroup makes it
        elif any("pci.bus_id," in a for a in argv):
            out = f"{bus}, 0, [N/A]\n"
        else:
            out = ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    return run


def test_a_confined_job_resolves_the_card_nvidia_smi_shows(tmp_path, monkeypatch):
    """A Slurm one-card job sees one GPU through nvidia-smi while /proc lists the node: cards are job-local."""
    import subprocess

    from qda_agent import doctor as d
    from qda_agent.sched import SchedConfig

    proc, _ = gpu_table(tmp_path)  # PCI ordinals 0..5, minors [7,6,0,1,2,3] in bus order
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(subprocess, "run", confined_smi({"visible": "00000000:3A:00.0"}))  # ordinal 3
    cfg = SchedConfig(cards=[0])
    assert d.card_health(cfg.cards, proc) == (None, None)
    order = d.card_order_info(cfg, proc)
    assert "job-local" in order.detail and "1 of the node's 6" in order.detail, order.detail


def test_a_confined_job_refuses_a_card_it_cannot_see(tmp_path, monkeypatch):
    """Never pick a card silently: a number outside the job's set is one sentence naming what it can see."""
    import subprocess

    from qda_agent import doctor as d
    from qda_agent.isolation.devices import CardError, resolve_cards
    from qda_agent.sched import SchedConfig

    proc, _ = gpu_table(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(subprocess, "run", confined_smi({"visible": "00000000:3A:00.0"}))
    with pytest.raises(CardError) as refused:  # the start's own `cards` guard says it
        resolve_cards([1], proc)
    assert "this job sees 1 of the node's 6 GPU(s), numbered 0..0" in str(refused.value)
    assert "job-local index" in str(refused.value)
    # That guard already says why; the card-order fact adds no second line (doctor skips None).
    assert d.card_order_info(SchedConfig(cards=[1]), proc) is None


def _shipped_cfg(tmp_path, *sets):
    """The shipped search.yaml's defaults, filled under `tmp_path` (`isolation.level: none`, no job_sandbox)."""
    from qda_agent.config import RunConfig
    from tests_support import filled

    path = filled(Path(__file__).parents[1] / "configs" / "search.yaml", tmp_path / "cfg")
    (tmp_path / "cfg" / "path" / "to" / "baseline-ckpt" / "config.json").write_text("{}")
    return RunConfig.load(path, [f"run_dir={tmp_path / 'run'}", "cards=[0]", *sets])


@pytest.mark.parametrize("sets", [(), ("scheduler.job_sandbox=none",)], ids=["defaults", "job_sandbox-none"])
def test_doctor_and_the_run_read_one_job_sandbox(tmp_path, sets):
    """The job sandbox doctor resolves is the one `resolved.yaml` records and the judge decides on."""
    from datetime import UTC, datetime

    from qda_agent.sched import SchedConfig

    cfg = _shipped_cfg(tmp_path, *sets)
    doctor_side = SchedConfig.model_validate(cfg.sched_config(datetime.now(UTC))).job_sandbox
    assert doctor_side == cfg.resolved()["scheduler"]["job_sandbox"] == cfg.job_sandbox()
    # the shipped level is none: unset follows it, the jobs run on the host
    assert cfg.job_sandbox() == "none"


def _reference_with_prompts(base, *, recorded, rows):
    """A reference directory whose manifest records `rows` at the relative path `recorded`."""
    import hashlib
    import json

    ref = Path(base) / "ref"
    (ref / recorded).parent.mkdir(parents=True, exist_ok=True)
    (ref / recorded).write_text(rows)
    (ref / "manifest.json").write_text(
        json.dumps(
            {
                "k": 1,
                "vocab": 1,
                "build": {
                    "prompts": recorded,
                    "prompts_sha256": hashlib.sha256(rows.encode()).hexdigest(),
                },
            }
        )
    )
    return ref


def test_a_reference_whose_prompt_file_is_gone_refuses_the_start_as_unreadable(tmp_path):
    """A cache entry that kept no copy of its prompt file (its builder's run was deleted) is the start's
    `ReferenceUnreadable`, raised by the one derivation `static_checks` reaches, not a FileNotFoundError."""
    import json
    from types import SimpleNamespace

    from qda_agent.judge.startup import ReferenceUnreadable, reference_index

    ref = tmp_path / "ref"
    ref.mkdir()
    gone = tmp_path / "deleted-run" / "data" / "corpus.search96.jsonl"
    (ref / "manifest.json").write_text(
        json.dumps({"k": 1, "vocab": 1, "build": {"prompts": str(gone), "prompts_sha256": "0" * 64}})
    )
    judge = SimpleNamespace(kl_reference=lambda: ref, kl_holdout_reference=lambda: ref)
    with pytest.raises(ReferenceUnreadable, match="cannot read"):
        reference_index(judge)


def test_a_reference_that_kept_its_prompts_derives_the_index(tmp_path):
    """The entry's own copy resolves, for the reference and the holdout set alike."""
    from types import SimpleNamespace

    from qda_agent.judge.startup import reference_index

    ref = _reference_with_prompts(
        tmp_path / "a", recorded="inputs/prompts.jsonl", rows='{"id": "a", "prompt": "x"}\n'
    )
    hold = _reference_with_prompts(
        tmp_path / "h", recorded="inputs/prompts.jsonl", rows='{"id": "h", "prompt": "y"}\n'
    )
    judge = SimpleNamespace(kl_reference=lambda: ref, kl_holdout_reference=lambda: hold)
    index = reference_index(judge)
    assert sorted(s.role for s in index.sources) == ["eval", "holdout"]
