import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qda_agent.sched import FakeBackend, FakeClock, Kind, LocalBackend, SchedConfig, Scheduler, State, sandbox
from qda_agent.sched.sandbox import (
    CardError,
    bwrap_argv,
    card_uuids,
    device_nodes,
    resolve_cards,
    sandbox_env,
)

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")


@needs_bwrap
def test_a_run_loaded_through_a_symlink_has_readable_job_binds(tmp_path):
    """Config paths and work binds share one canonical root, including a not-yet-created run."""
    import yaml
    from qda_agent.config import RunConfig

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    model = real / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights!")
    for name in ("ref", "hidden"):
        (real / name).mkdir()
    (real / "prompts.jsonl").write_text('{"text":"input"}\n')
    (real / "heldout.jsonl").write_text('{"text":"heldout"}\n')
    venv = tmp_path / "venv"
    venv.symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    config = tmp_path / "search.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": str(alias / "model"),
                "format": "w4a16",
                "budget_gpu_hours": 1,
                "contamination": "none",
                "cards": [0],
                "run_dir": str(alias / "run"),
                "baselines": [str(alias / "model")],
                "calib": str(alias / "prompts.jsonl"),
                "judge": {
                    "kl_ref": str(alias / "ref"),
                    "gencheck_prompts": {"file": {"path": str(alias / "heldout.jsonl")}},
                },
                "toolset": {"python": str(venv / "bin" / "python"), "components": ["baselines"]},
                "sandbox": {"hide": [str(alias / "hidden")]},
            }
        )
    )
    cfg = RunConfig.load(config)
    assert cfg.run_dir == real / "run" and cfg.model == str(model)
    assert cfg.baselines == [model] and cfg.judge.kl_ref == real / "ref"
    assert cfg.calib == str(real / "prompts.jsonl")
    assert cfg.judge.gencheck_prompts["file"]["path"] == str(real / "heldout.jsonl")
    expected_python = Path(sys.executable).parent.resolve() / "python"
    assert cfg.toolset.python == expected_python
    assert cfg.sandbox.hide == [alias / "hidden", real / "hidden"]
    cfg.workdir.mkdir(parents=True)
    (cfg.workdir / "input.txt").write_text("work-readable")
    (cfg.run_dir / "overlay" / "baselines").mkdir(parents=True)
    (cfg.run_dir / "overlay" / "baselines" / "0").mkdir()
    (cfg.run_dir / "overlay" / "baselines" / "README.md").write_text("overlay-readable")
    sched = cfg.sched_config(datetime.now(UTC))
    sandbox.preflight(
        sched["readable"],
        sched["binds"],
        writable=[cfg.workdir],
        run_dir=cfg.run_dir,
        host_written=sched["host_written"],
        hide=cfg.sandbox.hide,
    )
    result = subprocess.run(
        bwrap_argv(
            ["/bin/cat", "input.txt", "baselines/0/model.safetensors"],
            str(cfg.workdir),
            writable=[cfg.workdir],
            readable=sched["readable"],
            binds=sched["binds"],
            hide=cfg.sandbox.hide,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "work-readableweights!"


PROBE = """
import os, socket, sys
out = []
try:
    open(sys.argv[1]).read(); out.append("judge:readable")
except OSError:
    out.append("judge:hidden")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); out.append("net:open")
except OSError:
    out.append("net:closed")
open("written.txt", "w").write("ok"); out.append("work:writable")
out.append("home:" + os.environ.get("HOME", ""))
out.append("secret:" + os.environ.get("QDA_TEST_SECRET", "none"))
out.append("cvd:" + os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
print("PROBE " + " ".join(out))
"""


def run_job(tmp_path, sandbox, origin="agent"):
    run = tmp_path / "run"
    work = run / "work"
    judge = run / "judge"
    work.mkdir(parents=True)
    judge.mkdir()
    (judge / "ledger.jsonl").write_text("secret\n")
    (work / "probe.py").write_text(PROBE)
    s = Scheduler(
        run,
        SchedConfig(cards=[5], budget_s=600, job_sandbox=sandbox),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    kind = Kind.score if origin == "judge" else Kind.build
    s.submit(
        [os.path.normpath(Path(sys.executable).absolute()), "probe.py", str(judge / "ledger.jsonl")],
        str(work),
        name="p",
        expected_s=60,
        root=work if origin == "agent" else None,
        origin=origin,
        kind=kind,
    )
    t = time.time() + 20
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.05)
    assert s.get("p").state is State.done, s.log("p")
    assert "Error processing" not in s.log("p"), s.log("p")  # every .pth of the venv works in the sandbox
    line = next(x for x in s.log("p").splitlines() if x.startswith("PROBE "))
    return line.split()[1:], work


@needs_bwrap
def test_a_job_naming_the_symlinked_venv_starts(tmp_path, monkeypatch):
    """A venv reached through a link, with the command spelled that way, must still exec inside."""
    from qda_agent.sched import sandbox

    real = tmp_path / "venvs" / "gate-0123456789"
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "python").symlink_to(Path(sys.executable).resolve())
    alias = tmp_path / "venvs" / "dev"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(sandbox.sys, "prefix", str(alias))
    work = tmp_path / "work"
    work.mkdir()

    command = [str(alias / "bin" / "python"), "-c", 'print("job-started")']
    got = subprocess.run(
        sandbox.bwrap_argv(command, str(work), writable=[work]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == "job-started"


@needs_bwrap
def test_agent_job_is_sandboxed(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_TEST_SECRET", "leak")
    out, work = run_job(tmp_path, "bwrap")
    assert out[:5] == ["judge:hidden", "net:closed", "work:writable", f"home:{work}", "secret:none"]
    assert out[5] == "cvd:" + ",".join(card_uuids([5]))  # inside the sandbox cards are named by UUID
    assert (work / "written.txt").read_text() == "ok"


@needs_bwrap
def test_job_dir_snapshot_is_read_only(tmp_path):
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    (work / "x.sh").write_text('echo hacked > "$0"; echo rc=$?\n')
    s = Scheduler(run, SchedConfig(cards=[0], budget_s=600), LocalBackend(1), FakeClock(time.time()))
    job = s.submit([], str(work), name="x", expected_s=60, root=work, freeze="x.sh")
    while s.pending():
        s.tick()
        time.sleep(0.05)
    assert "rc=1" in s.log("x") and Path(job.argv[1]).read_text().startswith("echo hacked >")


def test_sandbox_none_and_judge_jobs_run_on_the_host(tmp_path):
    out = run_job(tmp_path / "a", "none")[0]
    assert out[0] == "judge:readable" and out[-1] == "cvd:" + ",".join(
        card_uuids([5])
    )  # unsandboxed: by UUID
    out = run_job(tmp_path / "b", "bwrap", origin="judge")[0]
    assert out[0] == "judge:readable" and out[-1] == "cvd:" + ",".join(card_uuids([5]))  # host job: by UUID


def test_scheduler_wraps_agent_jobs_only(tmp_path):
    clock = FakeClock()
    be = FakeBackend(clock)
    (tmp_path / "model").mkdir()  # a readable source must be there to bind
    s = Scheduler(tmp_path, SchedConfig(cards=[0, 1], budget_s=600, readable=[tmp_path / "model"]), be, clock)
    s.submit(["true"], str(tmp_path), name="a", expected_s=1)
    s.submit(
        ["true"],
        str(tmp_path),
        name="j",
        expected_s=1,
        origin="judge",
        kind=Kind.score,
        env={"PYTHONPATH": "/e"},
    )
    s.tick()
    assert be.argvs["0001-a"][0] == "bwrap" and str(tmp_path / "model") in be.argvs["0001-a"]
    assert be.argvs["0002-j"] == ["true"] and be.envs["0002-j"]["PYTHONPATH"] == "/e"
    argv = be.argvs["0001-a"]
    inner = {argv[i + 1]: argv[i + 2] for i, x in enumerate(argv) if x == "--setenv"}
    assert "--clearenv" in argv and be.envs["0001-a"] is None  # bwrap itself gets the host's env
    assert inner["HOME"] == str(tmp_path)
    assert inner["TZ"] == "UTC"  # the clock the deadline and qsched speak
    assert inner["CUDA_VISIBLE_DEVICES"] == ",".join(card_uuids([1]))  # by UUID; the judge job took card 0
    assert "QDA_TEST_SECRET" not in inner and inner["PATH"] != os.environ.get("PATH")


def test_a_configured_tz_wins_in_an_agent_job(tmp_path):
    """`toolset.env.TZ` and a job's own `env` override the sandbox's UTC, as they do any clean variable."""
    clock = FakeClock()
    be = FakeBackend(clock)
    s = Scheduler(
        tmp_path, SchedConfig(cards=[0, 1], budget_s=600, agent_job_env={"TZ": "Asia/Tokyo"}), be, clock
    )
    s.submit(["true"], str(tmp_path), name="a", expected_s=1)
    s.submit(["true"], str(tmp_path), name="b", expected_s=1, env={"TZ": "Europe/Paris"})
    s.tick()
    inner = {}
    for name in ("0001-a", "0002-b"):
        argv = be.argvs[name]
        inner[name] = {argv[i + 1]: argv[i + 2] for i, x in enumerate(argv) if x == "--setenv"}
    assert inner["0001-a"]["TZ"] == "Asia/Tokyo" and inner["0002-b"]["TZ"] == "Europe/Paris"


@needs_bwrap
def test_a_job_prints_utc_dates_and_file_times(tmp_path):
    """The job sandbox's own `date` and `ls -l` agree with the UTC clock, not the host's local time."""
    work = tmp_path / "work"
    work.mkdir()
    stamp = 1_700_000_000  # 2023-11-14 22:13:20 UTC; 17:13 in New York
    (work / "f").write_text("")
    os.utime(work / "f", (stamp, stamp))
    probe = "date +%Z; ls -l --time-style=+%Y-%m-%dT%H:%M f"
    argv = bwrap_argv(["/bin/sh", "-c", probe], str(work), writable=[work], env=sandbox_env(work))
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True).stdout.split()
    expected = datetime.fromtimestamp(stamp, UTC).strftime("%Y-%m-%dT%H:%M")
    assert out[0] == "UTC" and out[-2] == expected, out


@needs_bwrap
def test_a_source_swapped_into_the_run_dir_after_validation_fails_the_job(tmp_path):
    """The check ran when the config loaded; a link swapped before the bind must not reach the judge's dir."""
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    judge = run / "judge"
    judge.mkdir()
    (judge / "planted.txt").write_text("judge secret")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "data"
    link.symlink_to(outside)  # a source outside the run dir, as the config check saw it
    s = Scheduler(
        run,
        SchedConfig(
            cards=[0],
            budget_s=600,
            gpus=False,
            readable=[link],
            run_dir=run,
            host_written=[run / "overlay"],
        ),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    leak, planted = work / "leak.txt", link / "planted.txt"
    script = work / "read.py"
    script.write_text(f"open({str(leak)!r}, 'w').write(open({str(planted)!r}).read())\n")
    s.submit(
        [os.path.normpath(Path(sys.executable).absolute()), str(script)],
        str(work),
        name="p",
        expected_s=60,
        root=work,
    )
    link.unlink()
    link.symlink_to(judge)  # swapped after the config check, before the job starts
    while s.pending():
        s.tick()
        time.sleep(0.01)
    job = s.get("p")
    assert job.state is State.failed, job.reason
    assert "could not start" in job.reason and "refusing to bind it" in job.reason
    assert not leak.exists()  # the job never ran, so the judge's file was not read


def test_a_gpu_job_gets_the_writable_proc_and_a_cpu_run_a_namespace_of_its_own(tmp_path, monkeypatch):
    """Where a fresh procfs comes only with the host's pid namespace, a GPU job's sandbox takes it: the
    driver writes `/proc/self/task/<tid>/comm`. A run with `scheduler.gpus: false` has nothing of the
    driver's to give, so its jobs keep a pid namespace of their own and the host's `/proc` read-only."""
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "shared"))
    proc = fake_proc(tmp_path, PERMUTED)

    def argv_of(name, **cfg):
        clock = FakeClock()
        be = FakeBackend(clock)
        s = Scheduler(tmp_path / name, SchedConfig(cards=[0], budget_s=600, **cfg), be, clock, proc=proc)
        s.submit(["true"], str(tmp_path), name="a", expected_s=1, root=tmp_path)
        s.tick()
        return be.argvs["0001-a"]

    gpu = argv_of("gpu")
    assert gpu[gpu.index("--unshare-user") :][:3] == ["--unshare-user", "--proc", "/proc"]  # writable /proc
    assert "--unshare-pid" not in gpu
    cpu = argv_of("cpu", gpus=False)
    assert (
        "--unshare-pid" in cpu and "--proc" not in cpu
    )  # the host's /proc, read-only, in a namespace of its own
    assert cpu[cpu.index("--ro-bind", cpu.index("--unshare-user")) :][:3] == ["--ro-bind", "/proc", "/proc"]


def test_a_pinned_source_is_bound_by_descriptor(tmp_path):
    """bwrap binds the opened descriptor, so a swap after the check cannot change what it binds."""
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    pinned = sandbox.pin_sources([data], {})
    try:
        argv = bwrap_argv(["true"], str(work), writable=[work], readable=[data], pinned=pinned)
        (pin,) = pinned.values()
        i = argv.index("--ro-bind-fd")
        assert argv[i + 1] == str(pin.fd) and argv[i + 2] == str(data)
    finally:
        for pin in pinned.values():
            os.close(pin.fd)


@needs_bwrap
def test_sandbox_with_relative_interpreter(tmp_path):
    """The suite started as `../venv/bin/python -m pytest` (the review's case) still passes."""
    rel = os.path.relpath(sys.executable, tmp_path)
    p = subprocess.run(
        [rel, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{__file__}::test_agent_job_is_sandboxed"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[1] / "src"), *sys.path]),
        },
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr


def test_python_roots_follow_venv_links(tmp_path, monkeypatch):
    from qda_agent.sched import sandbox

    prefix = tmp_path / "venv"
    site = prefix / "lib" / "python3" / "site-packages"
    site.mkdir(parents=True)
    target = tmp_path / "tools" / "guard.py"
    target.parent.mkdir()
    target.write_text("")
    (site / "guard.py").symlink_to(target)
    editable = tmp_path / "repo" / "src"
    editable.mkdir(parents=True)
    (site / "_editable.pth").write_text(f"{editable}\nimport os\n/does/not/exist\n")
    monkeypatch.setattr(sandbox.sys, "prefix", str(prefix))
    monkeypatch.setattr(sandbox.sys, "path", [str(site)])
    roots = sandbox.python_roots()
    assert target.resolve() in roots and editable.resolve() in roots and Path("/does/not/exist") not in roots


def test_python_roots_mount_an_editable_dir_by_its_pth_spelling(tmp_path, monkeypatch):
    """An editable install whose `.pth` line goes through a link (a home dir that is a link to another
    filesystem): the sandboxed interpreter adds the line as written, so that spelling is mounted too."""
    from qda_agent.sched import sandbox

    prefix = tmp_path / "venv"
    site = prefix / "lib" / "python3" / "site-packages"
    site.mkdir(parents=True)
    real = tmp_path / "fs" / "home" / "u"
    (real / "repo" / "src").mkdir(parents=True)
    home = tmp_path / "home-u"
    home.symlink_to(real)
    (site / "_editable_impl_pkg.pth").write_text(f"{home / 'repo' / 'src'}\n")
    monkeypatch.setattr(sandbox.sys, "prefix", str(prefix))
    monkeypatch.setattr(sandbox.sys, "path", [str(site)])
    roots = sandbox.python_roots()
    assert (real / "repo" / "src").resolve() in roots and home / "repo" / "src" in roots


def test_python_roots_include_the_base_a_uv_venv_names(tmp_path, monkeypatch):
    """A uv venv's `pyvenv.cfg` names a base outside the venv (often under the user's home); the job's
    interpreter reads that base's standard library, so `python_roots` mounts it, as spelled and for real."""
    from qda_agent.isolation.bwrap import interpreter_roots
    from qda_agent.sched import sandbox

    real_home = tmp_path / "fs" / "home" / "u"
    base = real_home / ".local" / "share" / "uv" / "python" / "cpython-3.12.3-linux-x86_64-gnu"
    (base / "bin").mkdir(parents=True)
    (base / "bin" / "python3.12").write_text("")
    home = tmp_path / "home-u"
    home.symlink_to(real_home)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {home}/.local/share/uv/python/{base.name}/bin\n")
    (venv / "bin" / "python3.12").symlink_to(base / "bin" / "python3.12")

    # the function both the sandbox mounts and the judge's qda binds start from
    roots = interpreter_roots(venv / "bin" / "python3.12")
    assert base in roots and base.resolve() in roots  # the spelling pyvenv.cfg uses, and its real path
    assert venv in roots

    # the sandbox's own roots read the running interpreter's prefix, base and pyvenv.cfg
    monkeypatch.setattr(sandbox.sys, "executable", str(venv / "bin" / "python3.12"))
    monkeypatch.setattr(sandbox.sys, "prefix", str(venv))
    monkeypatch.setattr(sandbox.sys, "path", [])
    assert base in sandbox.python_roots()


def test_python_is_spelled_under_the_prefixs_real_path(tmp_path, monkeypatch):
    """A venv reached through a link: the sandbox mounts the real prefix, so the command must be spelled there."""
    from qda_agent.sched import sandbox

    real = tmp_path / "venvs" / "gate-a351929c2bc9"
    (real / "bin").mkdir(parents=True)
    base = tmp_path / "usr" / "bin" / "python3.12"
    base.parent.mkdir(parents=True)
    base.write_text("")
    # the interpreter itself stays a symlink (resolving it would leave the venv)
    (real / "bin" / "python3.12").symlink_to(base)
    alias = tmp_path / "venvs" / "dev"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(sandbox.sys, "executable", str(alias / "bin" / "python3.12"))
    monkeypatch.setattr(sandbox.sys, "prefix", str(alias))

    assert sandbox.python() == str(real / "bin" / "python3.12")
    assert Path(sandbox.python()).is_relative_to(real)


def fake_proc(tmp_path, rows):
    """A fake /proc/driver/nvidia/gpus: one directory per (bus id, minor, uuid), named by its bus id."""
    proc = tmp_path / "proc-gpus"
    for bus, minor, uuid in rows:
        d = proc / bus
        d.mkdir(parents=True)
        (d / "information").write_text(
            f"Model: \t\t NVIDIA H200\nIRQ:   \t\t 42\nGPU UUID: \t {uuid}\n"
            f"Device Minor: \t {minor}\nBus Location: \t {bus}\n"
        )
    return proc


#: index (= PCI bus order), minor and directory-listing order all differ: bus order is the ordinal
PERMUTED = [  # (bus id, minor, uuid); the uuid names the bus-order ordinal
    ("0000:18:00.0", 0, "GPU-o0"),
    ("0000:2a:00.0", 1, "GPU-o1"),
    ("0000:3a:00.0", 4, "GPU-o2"),
    ("0000:5d:00.0", 3, "GPU-o3"),
    ("0000:9a:00.0", 2, "GPU-o4"),
    ("0000:ab:00.0", 5, "GPU-o5"),
    ("0000:ba:00.0", 7, "GPU-o6"),
    ("0000:db:00.0", 6, "GPU-o7"),
]


def test_cards_are_pci_bus_order_not_minors(tmp_path):
    proc = fake_proc(tmp_path, reversed(PERMUTED))  # created in another order than bus order
    cards = resolve_cards([2, 4, 6], proc)
    assert [c.uuid for c in cards] == ["GPU-o2", "GPU-o4", "GPU-o6"]
    assert [c.minor for c in cards] == [4, 2, 7]  # /dev/nvidia4 is ordinal 2, not ordinal 4
    with pytest.raises(CardError, match=r"\[8\]"):
        resolve_cards([1, 8], proc)  # never "all", never a guess
    assert resolve_cards([5], tmp_path / "no-driver") == []  # no NVIDIA driver: no GPU to show


def test_device_binding_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "private"))  # the process namespace probe is another test's
    proc = fake_proc(tmp_path, PERMUTED)
    dev = tmp_path / "dev"
    dev.mkdir()
    for m in range(8):
        (dev / f"nvidia{m}").touch()
    nodes = device_nodes([2, 4], proc, dev)
    assert sorted(n.name for n in nodes if n.parent == dev) == [
        "nvidia2",
        "nvidia4",
    ]  # minors of ordinals 4, 2
    argv = bwrap_argv(["true"], "/w", writable=[Path("/w")], devices=nodes)
    for flag in ("--unshare-net", "--unshare-pid", "--die-with-parent"):
        assert flag in argv
    assert argv[-2:] == ["--", "true"]
    assert argv[argv.index("--bind") : argv.index("--bind") + 3] == ["--bind", "/w", "/w"]


def _setenv(argv):
    return {argv[i + 1]: argv[i + 2] for i, x in enumerate(argv) if x == "--setenv"}


def _devbinds(argv):
    return [argv[i + 1] for i, x in enumerate(argv) if x == "--dev-bind"]


def test_exact_card_path_binds_the_assigned_gpus(tmp_path, monkeypatch):
    """Agent (sandboxed) and judge (host) jobs on an exact-card backend, with permuted minors."""
    proc = fake_proc(tmp_path, PERMUTED)
    monkeypatch.setattr(
        sandbox, "device_nodes", lambda cards, p=None, dev=None: [c.node for c in resolve_cards(cards, proc)]
    )
    from qda_agent.sched import queue

    monkeypatch.setattr(queue.devices, "device_nodes", sandbox.device_nodes)
    clock = FakeClock()
    be = FakeBackend(clock)
    s = Scheduler(tmp_path / "run", SchedConfig(cards=[2, 4, 6], budget_s=600), be, clock, proc=proc)
    s.submit(["true"], str(tmp_path), name="j", expected_s=1, origin="judge", kind=Kind.score)
    s.submit(["true"], str(tmp_path), name="a", ncards=2, expected_s=1, root=tmp_path)
    s.tick()
    j, a = s.get("j"), s.get("a")
    assert j.cards == [2] and a.cards == [4, 6]
    assert be.envs["0001-j"]["CUDA_VISIBLE_DEVICES"] == "GPU-o2"  # the judge's host job too
    assert be.envs["0001-j"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    argv = be.argvs["0002-a"]
    assert _setenv(argv)["CUDA_VISIBLE_DEVICES"] == "GPU-o4,GPU-o6"
    assert sorted(_devbinds(argv)) == ["/dev/nvidia2", "/dev/nvidia7"]  # minors of ordinals 4 and 6


def test_no_driver_refuses_at_start(tmp_path):
    """Without an NVIDIA driver the queue refuses to start while `gpus` is true, and names the fix."""
    none = tmp_path / "no-driver"  # no /proc/driver/nvidia/gpus
    clock = FakeClock()
    with pytest.raises(CardError, match="no NVIDIA driver.*scheduler.gpus: false"):
        Scheduler(tmp_path / "run", SchedConfig(cards=[0], budget_s=600), LocalBackend(1), clock, proc=none)
    s = Scheduler(
        tmp_path / "run", SchedConfig(cards=[0], budget_s=600, gpus=False), LocalBackend(1), clock, proc=none
    )
    s.submit(["true"], str(tmp_path), name="a", expected_s=1, origin="judge", kind=Kind.score)
    s.tick()
    assert s.get("a").state is State.running


def test_cards_without_gpus_are_queue_slots(tmp_path):
    """`gpus: false`: any card numbers start, and no job sees a GPU, on the host or in bwrap."""
    proc = fake_proc(tmp_path, PERMUTED[:2])  # a driver with two GPUs: cards 8 and 9 name none of them
    clock = FakeClock()
    be = FakeBackend(clock)
    s = Scheduler(tmp_path / "run", SchedConfig(cards=[8, 9], budget_s=600, gpus=False), be, clock, proc=proc)
    s.submit(["true"], str(tmp_path), name="j", expected_s=1, origin="judge", kind=Kind.score)
    s.submit(["true"], str(tmp_path), name="a", expected_s=1, root=tmp_path)
    s.tick()
    assert s.get("j").cards == [8] and s.get("a").cards == [9]
    assert be.envs["0001-j"]["CUDA_VISIBLE_DEVICES"] == ""
    argv = be.argvs["0002-a"]
    assert _setenv(argv)["CUDA_VISIBLE_DEVICES"] == "" and _devbinds(argv) == []


def test_unknown_pool_card_is_refused_at_start(tmp_path):
    proc = fake_proc(tmp_path, PERMUTED[:2])
    with pytest.raises(CardError, match=r"\[5\]"):
        Scheduler(
            tmp_path / "run",
            SchedConfig(cards=[0, 5], budget_s=600),
            FakeBackend(FakeClock()),
            FakeClock(),
            proc=proc,
        )


def test_local_backend_keeps_the_schedulers_uuids(tmp_path, monkeypatch):
    """A judge job on the host gets its cards by UUID; LocalBackend must not overwrite them."""
    proc = fake_proc(tmp_path, PERMUTED)
    run = tmp_path / "run"
    s = Scheduler(
        run, SchedConfig(cards=[3, 4], budget_s=600), LocalBackend(1), FakeClock(time.time()), proc=proc
    )
    s.submit(
        ["bash", "-c", "echo $CUDA_DEVICE_ORDER $CUDA_VISIBLE_DEVICES"],
        str(tmp_path),
        name="j",
        expected_s=5,
        origin="judge",
        kind=Kind.score,
    )
    t = time.time() + 10
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.02)
    assert s.get("j").state is State.done and s.log("j").split() == ["PCI_BUS_ID", "GPU-o3"]


@needs_bwrap
def test_agent_jobs_get_the_toolsets_python(tmp_path):
    """`toolset.python` is the `python` of every agent job (its bin/ first on PATH), not qda-agent's own."""
    from qda_agent.config import RunConfig

    venv = Path(os.path.normpath(Path(sys.prefix).absolute()))  # stands for a venv with torch
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[5],
        run_dir=tmp_path / "run",
        toolset={"python": str(venv / "bin" / "python"), "env": {"QDA_TOOLSET": "yes"}},
    )
    sched = cfg.sched_config(deadline=None)
    assert sched["agent_job_env"]["PATH"].split(":")[0] == str(venv / "bin")
    assert cfg.judge_config()["qda_cmd"] == [str(venv / "bin" / "python"), "-m", "qda_quant.cli"]
    work = cfg.workdir
    work.mkdir(parents=True)
    s = Scheduler(
        cfg.run_dir, SchedConfig.model_validate({**sched, "budget_s": 600}), LocalBackend(1), FakeClock()
    )
    s.submit(
        ["/bin/sh", "-c", 'echo "PATH=$PATH TOOLSET=$QDA_TOOLSET PY=$(command -v python)"'],
        str(work),
        name="p",
        expected_s=60,
        root=work,
    )
    t = time.time() + 20
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.05)
    out = s.log("p")
    assert f"PATH={venv}/bin:" in out and "TOOLSET=yes" in out and f"PY={venv}/bin/python" in out, out


def test_a_host_job_keeps_the_launching_path_and_a_sandboxed_job_does_not(tmp_path, monkeypatch):
    """`scheduler.job_sandbox: none` runs jobs on the host, so their PATH keeps the user's tools."""
    from qda_agent.config import RunConfig

    host_path = "/opt/site/gcc/bin:/usr/local/bin:/usr/bin:/bin"
    monkeypatch.setenv("PATH", host_path)
    common = dict(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[5],
        toolset={"python": sys.executable},
    )
    cfg = RunConfig(**common, run_dir=tmp_path / "host", scheduler={"job_sandbox": "none"})
    assert cfg.agent_job_env()["PATH"].endswith(f":{host_path}")
    assert cfg.sched_config(deadline=None)["agent_job_env"] == cfg.agent_job_env()
    # `isolation.level: none` runs the jobs on the host whatever the legacy knob says (job_sandbox(...))
    plain = RunConfig(**common, run_dir=tmp_path / "plain", isolation={"level": "none"})
    assert plain.agent_job_env()["PATH"].endswith(f":{host_path}")
    sandboxed = RunConfig(**common, run_dir=tmp_path / "sandbox", scheduler={"job_sandbox": "bwrap"})
    assert sandboxed.agent_job_env()["PATH"].endswith(":/usr/local/bin:/usr/bin:/bin")
    assert host_path not in sandboxed.agent_job_env()["PATH"]


def _bwrap_result(returncode, stdout, stderr):
    return subprocess.CompletedProcess(["bwrap"], returncode, stdout, stderr)


def _refused_proc(*_a, **_k):
    return _bwrap_result(1, "", "bwrap: Can't mount proc on /newroot/proc: Operation not permitted\n")


def test_the_probe_prefers_a_fresh_procfs_in_a_pid_namespace_of_its_own(monkeypatch):
    """Where this host mounts one: a fresh procfs, and only the sandbox's processes in its pid namespace."""
    monkeypatch.setattr(sandbox, "_ns", None)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []

    def run(argv, **k):
        calls.append(argv)
        return _bwrap_result(0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    assert sandbox.proc_form() == "fresh"
    assert sandbox.pid_form() == "private"
    assert len(calls) == 1 and "--unshare-pid" in calls[0]
    assert sandbox.proc_args() == ["--proc", "/proc"]


def test_a_refused_procfs_keeps_the_fresh_proc_with_the_hosts_pids(monkeypatch):
    """The DGX hosts: a fresh procfs mounts only outside a pid namespace of the sandbox's own.

    The shared form keeps the user namespace: with the host's processes in `/proc` now reachable by pid, it is
    what leaves another same-uid process's environ, maps, descriptors and root unreadable."""
    monkeypatch.setattr(sandbox, "_ns", None)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: f"/usr/bin/{name}")

    def run(argv, **k):
        return _refused_proc() if "--unshare-pid" in argv else _bwrap_result(0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    assert sandbox.proc_form() == "fresh"
    assert sandbox.pid_form() == "shared"
    assert sandbox.proc_args() == ["--unshare-user", "--proc", "/proc"]
    argv = bwrap_argv(["true"], "/w", writable=[Path("/w")])
    for flag in ("--unshare-user", "--proc"):
        assert flag in argv
    assert "--unshare-pid" not in argv


def test_proc_binds_the_hosts_read_only_where_a_fresh_one_is_refused(monkeypatch):
    """A host that refuses every fresh procfs gets the host's /proc read-only instead.

    The fallback also unshares the user namespace: outside it the kernel lets a same-uid process read
    `/proc/<pid>/environ` (the probe this fallback was measured with), inside a namespace of its own it does not."""
    monkeypatch.setattr(sandbox, "_ns", None)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sandbox.subprocess, "run", _refused_proc)
    assert sandbox.proc_form() == "host"
    assert sandbox.pid_form() == "shared"  # the bound /proc names the host's processes: the pid ns must match
    assert sandbox.proc_args() == ["--unshare-user", "--ro-bind", "/proc", "/proc"]
    argv = bwrap_argv(["true"], "/w", writable=[Path("/w")])
    assert "--proc" not in argv and "--unshare-pid" not in argv
    i = argv.index("--unshare-user")
    assert argv[i : i + 5] == ["--unshare-user", "--ro-bind", "/proc", "/proc", "--dev"]


def test_another_bwrap_failure_keeps_the_best_form(monkeypatch):
    """Only a refused proc mount selects a fallback: another failure must surface from the real argv."""
    monkeypatch.setattr(sandbox, "_ns", None)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda *a, **k: _bwrap_result(1, "", "bwrap: setting up uid map: Permission denied\n"),
    )
    assert sandbox.proc_form() == "fresh"
    assert sandbox.pid_form() == "private"
    assert sandbox.proc_args() == ["--proc", "/proc"]
    assert sandbox.pid_args() == ["--unshare-pid"]


def test_a_snapshots_links_bring_their_targets_and_nothing_else(tmp_path):
    """A HF snapshot is links into the sibling `blobs/`: bind the named ones, not the rest of the cache."""
    repo = tmp_path / "hub" / "models--org--name"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_text("A\n")
    (blobs / "other").write_text("B\n")
    snap = repo / "snapshots" / "rev"
    (snap / "lib").mkdir(parents=True)
    (snap / "config.json").symlink_to("../../blobs/a")
    (snap / "lib" / "sub").mkdir()
    (snap / "lib" / "sub" / "w.bin").symlink_to("../../../blobs/a")  # nested, and a target already named
    (snap / "alias").symlink_to("config.json")  # stays inside the snapshot
    (snap / "broken").symlink_to("../../blobs/none")
    assert sandbox.expand_links([snap]) == [snap, blobs / "a"]


def test_links_of_a_readable_dir_that_is_not_a_snapshot_are_not_expanded(tmp_path):
    """Only a hub snapshot's links are followed; any other readable directory keeps its links as they are."""
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "elsewhere").write_text("x")
    (data / "link").symlink_to(tmp_path / "elsewhere")
    assert sandbox.expand_links([data]) == [data]


def test_a_snapshot_link_outside_its_repository_is_refused(tmp_path):
    """A planted link to held-out data is refused, naming the link and its target."""
    repo = tmp_path / "hub" / "models--org--name"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_text("A\n")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").symlink_to("../../blobs/a")
    held = tmp_path / "heldout"
    held.mkdir()
    (held / "item.txt").write_text("held out\n")
    (snap / "planted").symlink_to(held, target_is_directory=True)
    with pytest.raises(ValueError) as caught:
        sandbox.expand_links([snap])
    assert str(snap / "planted") in str(caught.value)
    assert str(held) in str(caught.value)


def test_a_snapshot_link_that_reaches_out_through_another_link_is_refused(tmp_path):
    """The real path decides: a link that stays inside the repository but resolves outside it is refused."""
    repo = tmp_path / "hub" / "models--org--name"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    held = tmp_path / "heldout"
    held.mkdir()
    (held / "item.txt").write_text("held out\n")
    (blobs / "alias").symlink_to(held / "item.txt")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "item.txt").symlink_to("../../blobs/alias")
    with pytest.raises(ValueError, match="outside the model repository"):
        sandbox.expand_links([snap])


def test_a_snapshot_binds_the_shared_stores_second_hop(tmp_path):
    """A repository blob is a link into `hub/blobs/`: bind that file too, and nothing else there."""
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    blob = "a" * 64
    store = hub / "blobs" / "aa"
    store.mkdir(parents=True)
    (store / blob).write_text("weights\n")
    (hub / "blobs" / "bb").mkdir()
    (hub / "blobs" / "bb" / ("b" * 64)).write_text("another model\n")  # never named by this snapshot
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / blob).symlink_to(f"../../blobs/aa/{blob}")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(f"../../blobs/{blob}")
    assert sandbox.expand_links([snap]) == [snap, blobs / blob, store / blob]


@needs_bwrap
@pytest.mark.parametrize("when", ["before-pin", "after-pin"])
def test_a_store_file_swap_cannot_change_what_the_job_reads(tmp_path, when):
    """A checked store file swapped to held-out data must be refused or keep its pinned original."""
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    digest = "a" * 64
    final = hub / "blobs" / "aa" / digest
    final.parent.mkdir(parents=True)
    final.write_text("allowed weight")
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / digest).symlink_to(f"../../blobs/aa/{digest}")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    weight = snap / "model.safetensors"
    weight.symlink_to(f"../../blobs/{digest}")
    held = tmp_path / "heldout"
    held.write_text("held-out secret")

    expanded = sandbox.expand_links([snap])
    pinned = sandbox.pin_sources(expanded, {}) if when == "after-pin" else None
    final.rename(final.with_name("original"))
    final.symlink_to(held)
    if pinned is None:
        try:
            pinned = sandbox.pin_sources(expanded, {})
        except ValueError:  # a swap before pinning must not authorize the new target
            return
    try:
        argv = sandbox.bwrap_argv(
            [sandbox.python(), "-c", "import sys; print(open(sys.argv[1]).read())", str(weight)],
            str(snap),
            writable=[],
            readable=expanded,
            pinned=pinned,
            env=sandbox.sandbox_env(Path("/tmp")),
            pid="private",
        )
        got = subprocess.run(
            argv, capture_output=True, text=True, pass_fds=tuple(p.fd for p in pinned.values()), timeout=60
        )
        assert got.returncode == 0, got.stderr
        assert got.stdout.strip() == "allowed weight", "the swap exposed held-out data to the job"
    finally:
        for pin in pinned.values():
            os.close(pin.fd)


def test_a_snapshot_link_to_another_models_blob_is_not_bound(tmp_path):
    """A link that hops through another repository is refused, naming the link and what it reaches."""
    hub = tmp_path / "hub"
    other = hub / "models--org--other" / "blobs"
    other.mkdir(parents=True)
    (other / "x").write_text("another model\n")
    repo = hub / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "planted").symlink_to("../../../models--org--other/blobs/x")
    with pytest.raises(ValueError, match="outside the model repository") as caught:
        sandbox.expand_links([snap])
    assert str(snap / "planted") in str(caught.value)


def test_a_planted_store_root_is_not_taken_for_the_shared_store(tmp_path):
    """`hub/blobs` itself a link to held-out data is no store; the link is refused."""
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    digest = "a" * 64
    held = tmp_path / "heldout"
    (held / "aa").mkdir(parents=True)
    (held / "aa" / digest).write_text("HELDOUT SHOULD NOT BE READABLE")
    (repo / "blobs").mkdir(parents=True)
    (hub / "blobs").symlink_to(held, target_is_directory=True)  # not a cache: a planted root
    (repo / "blobs" / digest).symlink_to(f"../../blobs/aa/{digest}")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(f"../../blobs/{digest}")
    with pytest.raises(ValueError, match="outside the model repository"):
        sandbox.expand_links([snap])


@needs_bwrap
def test_the_preflight_reads_one_weight_through_the_jobs_binds(tmp_path):
    """The run's preflight opens the largest file's first bytes, binary header and all.

    The first 8 bytes of a safetensors file are a little-endian length (128 here, so `0x80` first),
    not UTF-8 text; the check must not decode them.
    """
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    blob = "a" * 64
    store = hub / "blobs" / "aa"
    store.mkdir(parents=True)
    weight = (128).to_bytes(8, "little") + b" " * 128 + b"\0" * 4096  # a valid F32[1] header
    (store / blob).write_bytes(weight)  # the weight, larger than the config
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / blob).symlink_to(f"../../blobs/aa/{blob}")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").symlink_to(f"../../blobs/{blob}")
    assert sandbox.preflight([snap], {}) == snap / "model.safetensors"


def test_a_planted_snapshot_link_refuses_the_job(tmp_path):
    """A snapshot link to a held-out directory fails the job before it starts."""
    repo = tmp_path / "hub" / "models--org--name"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_text("A\n")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").symlink_to("../../blobs/a")
    held = tmp_path / "heldout"
    held.mkdir()
    (held / "item.txt").write_text("held out\n")
    (snap / "planted").symlink_to(held, target_is_directory=True)
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    s = Scheduler(
        run,
        SchedConfig(cards=[0], budget_s=600, job_sandbox="bwrap", readable=[snap]),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    s.submit([sandbox.python(), "-c", "print('ran')"], str(work), name="p", expected_s=60, root=work)
    t = time.time() + 20
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.05)
    job = s.get("p")
    assert job.state is State.failed
    assert str(snap / "planted") in job.reason and str(held) in job.reason, job.reason


@needs_bwrap
def test_a_judge_job_reads_a_snapshot_whose_files_are_links_outside_it(tmp_path):
    """The snapshot is bound, but its links target the sibling `blobs/`, which must be bound too."""
    repo = tmp_path / "hub" / "models--org--name"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_text("weights\n")
    (blobs / "other").write_text("another model\n")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").symlink_to("../../blobs/a")
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    (work / "probe.py").write_text(
        "import sys\n"
        "print('snapshot:' + open(sys.argv[1]).read().strip())\n"
        "try:\n"
        "    open(sys.argv[2]).read()\n"
        "    print('other:readable')\n"
        "except OSError:\n"
        "    print('other:hidden')\n"
    )
    s = Scheduler(
        run,
        SchedConfig(cards=[0], budget_s=600, job_sandbox="bwrap", readable=[snap]),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    s.submit(
        [
            os.path.normpath(Path(sys.executable).absolute()),
            "probe.py",
            str(snap / "config.json"),
            str(blobs / "other"),
        ],
        str(work),
        name="p",
        expected_s=60,
        root=work,
    )
    t = time.time() + 20
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.05)
    log = s.log("p")
    assert s.get("p").state is State.done, log
    assert "snapshot:weights" in log and "other:hidden" in log, log


@needs_bwrap
def test_the_chosen_proc_form_runs_a_job(tmp_path):
    """Whichever /proc form this host allows, a job starts in it."""
    argv = bwrap_argv([shutil.which("true") or "/usr/bin/true"], str(tmp_path), writable=[tmp_path])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


@needs_bwrap
def test_the_host_proc_fallback_keeps_another_processes_environ_unreadable(tmp_path, monkeypatch):
    """Where sandboxes get the host's /proc, each is in a user namespace of its own: a same-uid process
    outside it keeps its environ (API keys), maps and descriptors unreadable; only cmdline is visible."""
    monkeypatch.setattr(sandbox, "_ns", ("host", "shared"))
    sibling = subprocess.Popen(["sleep", "60"], env={**os.environ, "QDA_PROBE_SECRET": "not-a-real-key"})
    try:
        probe = (
            f"cat /proc/{sibling.pid}/environ; echo rc=$?; "
            f"readlink /proc/{sibling.pid}/root; echo root=$?; "
            f"cat /proc/{sibling.pid}/cmdline"
        )
        argv = bwrap_argv(["/bin/sh", "-c", probe], str(tmp_path), writable=[tmp_path])
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert "not-a-real-key" not in r.stdout
        assert "rc=1" in r.stdout and "Permission denied" in r.stderr  # environ
        assert "root=1" in r.stdout  # root points at the host's /
        assert "sleep" in r.stdout  # cmdline is the one thing a host procfs shows
    finally:
        sibling.kill()
        sibling.wait()


@needs_bwrap
def test_the_fresh_proc_with_the_hosts_pids_keeps_another_process_sealed(tmp_path, monkeypatch):
    """Where a sandbox shares the host's pid namespace to get a fresh procfs, the user namespace it
    also unshares still refuses another same-uid process's environ, maps, descriptors and root. What it does
    not take away is the signal: the kernel compares user ids, not namespaces, so the sandbox can still signal
    a same-user process -- which is why `pid: shared` wants the node to itself (`isolation.allocation`)."""
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "shared"))
    sibling = subprocess.Popen(["sleep", "60"], env={**os.environ, "QDA_PROBE_SECRET": "not-a-real-key"})
    try:
        probe = (
            f"cat /proc/{sibling.pid}/environ; echo rc=$?; "
            f"cat /proc/{sibling.pid}/maps >/dev/null; echo maps=$?; "
            f"readlink /proc/{sibling.pid}/fd/0; echo fd=$?; "
            f"readlink /proc/{sibling.pid}/root; echo root=$?; "
            f"kill -0 {sibling.pid}; echo kill=$?; "
            f"cat /proc/{sibling.pid}/cmdline"
        )
        argv = bwrap_argv(["/bin/sh", "-c", probe], str(tmp_path), writable=[tmp_path])
        assert "--unshare-pid" not in argv and "--proc" in argv and "--unshare-user" in argv
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert "not-a-real-key" not in r.stdout
        assert "rc=1" in r.stdout and "Permission denied" in r.stderr  # environ
        assert "maps=1" in r.stdout and "fd=1" in r.stdout
        assert "root=1" in r.stdout
        assert (
            "kill=0" in r.stdout
        )  # a signal does reach a same-user process: `pid: shared` wants a node of its own
        assert "sleep" in r.stdout
    finally:
        sibling.kill()
        sibling.wait()


def _run_hidden_job(tmp_path, code, *, hide, readable=(), agent_env=None):
    """Run one agent job in the real scheduler sandbox and return its log."""
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    s = Scheduler(
        run,
        SchedConfig(
            cards=[0],
            budget_s=600,
            gpus=False,  # no driver needed: the mount layout is the same
            hide=list(hide),
            readable=list(readable),
            agent_job_env=dict(agent_env or {}),
        ),
        LocalBackend(1),
        FakeClock(time.time()),
    )
    s.submit([sandbox.python(), "-c", code], str(work), name="h", expected_s=60, root=work, kind=Kind.build)
    t = time.time() + 20
    while s.pending():
        assert time.time() < t
        s.tick()
        time.sleep(0.05)
    log = s.log("h")
    assert s.get("h").state is State.done, log
    line = next(x for x in log.splitlines() if x.startswith("PROBE "))
    return json.loads(line.split(" ", 1)[1])


@needs_bwrap
def test_a_hidden_file_in_a_python_root_is_unreadable_at_both_spellings(tmp_path, monkeypatch):
    """`sandbox.hide` reaches the job at both the spelling it hides and the resolved one, and the mask is
    mounted after `python_roots()` -- a package whose data file is hidden still imports."""
    site = tmp_path / "site"
    (site / "qda_kl" / "data").mkdir(parents=True)
    (site / "qda_kl" / "__init__.py").write_text("MARKER = 'synthetic'\n")
    (site / "qda_kl" / "data" / "gencheck-prompts.jsonl").write_text("PRIVATE\n")
    (site / "qda_kl" / "data" / "public.jsonl").write_text("PUBLIC\n")
    alias = site / "alias"
    alias.symlink_to("qda_kl", target_is_directory=True)
    real = sandbox.python_roots()
    monkeypatch.setattr(sandbox, "python_roots", lambda: [*real, site])
    code = f"""
import json
from pathlib import Path
import qda_kl

def read(p):
    try:
        p.read_text()
        return "readable"
    except OSError as e:
        return type(e).__name__

data = Path(qda_kl.__file__).parent / "data"
print("PROBE " + json.dumps({{
    "import": qda_kl.MARKER,
    "resolved": read(data / "gencheck-prompts.jsonl"),
    "alias": read(Path({str(alias)!r}) / "data" / "gencheck-prompts.jsonl"),
    "public": (data / "public.jsonl").read_text().strip(),
}}))
"""
    got = _run_hidden_job(
        tmp_path, code, hide=[alias / "data" / "gencheck-prompts.jsonl"], agent_env={"PYTHONPATH": str(site)}
    )
    assert got == {
        "import": "synthetic",
        "resolved": "PermissionError",
        "alias": "PermissionError",
        "public": "PUBLIC",
    }


@needs_bwrap
def test_a_hidden_data_dir_under_a_mounted_prefix_is_hidden_at_its_alias_too(tmp_path, monkeypatch):
    """`python_roots()` mounts a prefix reached through a link by both spellings; each is a mount of its own
    inside, so hiding the resolved data directory must mask the alias spelling's copy too."""
    real = tmp_path / "venvs" / "gate-0123456789"
    data = real / "qda_kl" / "data"
    data.mkdir(parents=True)
    (real / "qda_kl" / "__init__.py").write_text("MARKER = 'synthetic'\n")
    (data / "gencheck-prompts.jsonl").write_text("PRIVATE\n")
    (real / "qda_kl" / "public.jsonl").write_text("PUBLIC\n")
    alias = tmp_path / "venvs" / "dev"
    alias.symlink_to(real, target_is_directory=True)
    roots = sandbox.python_roots()
    monkeypatch.setattr(sandbox, "python_roots", lambda: [*roots, real, alias])
    code = f"""
import json
from pathlib import Path
import qda_kl

def read(p):
    try:
        p.read_text()
        return "readable"
    except OSError as e:
        return type(e).__name__

real, alias = Path({str(real)!r}), Path({str(alias)!r})
print("PROBE " + json.dumps({{
    "import": qda_kl.MARKER,
    "real": read(real / "qda_kl" / "data" / "gencheck-prompts.jsonl"),
    "alias": read(alias / "qda_kl" / "data" / "gencheck-prompts.jsonl"),
    "alias_public": read(alias / "qda_kl" / "public.jsonl"),
}}))
"""
    got = _run_hidden_job(tmp_path, code, hide=[data], agent_env={"PYTHONPATH": str(real)})
    assert got == {
        "import": "synthetic",  # the package still imports with its data directory hidden
        "real": "FileNotFoundError",
        "alias": "FileNotFoundError",
        "alias_public": "readable",
    }


@needs_bwrap
def test_a_hidden_calib_slice_is_absent_in_a_job(tmp_path):
    """A pool slice directory hidden under a readable pool is an empty tmpfs in the job, its files absent."""
    pool = tmp_path / "pool"
    (pool / "if-heldout").mkdir(parents=True)
    (pool / "if-heldout" / "slice.jsonl").write_text("PRIVATE\n")
    (pool / "calib").mkdir()
    (pool / "calib" / "calib.jsonl").write_text("PUBLIC\n")
    code = f"""
import json
from pathlib import Path

def read(p):
    try:
        p.read_text()
        return "readable"
    except OSError as e:
        return type(e).__name__

hidden, public = Path({str(pool / "if-heldout")!r}), Path({str(pool / "calib")!r})
try:
    entries = sorted(p.name for p in hidden.iterdir())
except OSError as e:
    entries = type(e).__name__
print("PROBE " + json.dumps({{"entries": entries, "inner": read(hidden / "slice.jsonl"),
                             "public": read(public / "calib.jsonl")}}))
"""
    got = _run_hidden_job(tmp_path, code, hide=[pool / "if-heldout"], readable=[pool])
    assert got == {"entries": [], "inner": "FileNotFoundError", "public": "readable"}


@needs_bwrap
def test_a_hidden_path_outside_every_mount_is_masked_in_a_job(tmp_path):
    """The job's root is the host's (no `--tmpfs /`), so a hide no readable mount covers is visible through
    it and must still be masked: the root counts as a bind of `/`."""
    # not under TMPDIR: the sandbox replaces /tmp with a tmpfs, which hides anything there for free
    folder = Path(tempfile.mkdtemp(prefix="qda-job-hide-", dir="/var/tmp"))
    try:
        private = folder / "private.txt"
        private.write_text("PRIVATE\n")
        code = f"""
import json
from pathlib import Path
try:
    Path({str(private)!r}).read_text()
    got = "readable"
except OSError as e:
    got = type(e).__name__
print("PROBE " + json.dumps({{"hidden": got}}))
"""
        got = _run_hidden_job(tmp_path, code, hide=[private])
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert got == {"hidden": "PermissionError"}


def test_a_hide_that_covers_a_path_the_job_needs_is_refused(tmp_path):
    """A hide of the workdir, a readable source or a bind is refused at start naming both paths: a mask
    mounted last would take the needed path with it."""
    pool = tmp_path / "pool"
    (pool / "slice").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError) as e:
        bwrap_argv(["true"], str(work), writable=[work], readable=[pool / "slice"], hide=[pool])
    assert str(pool) in str(e.value) and os.path.realpath(pool / "slice") in str(e.value)
    with pytest.raises(ValueError) as e:
        bwrap_argv(["true"], str(work), writable=[work], hide=[work])
    assert str(work) in str(e.value) and "covers" in str(e.value)


@needs_bwrap
def test_the_job_self_check_refuses_a_hidden_path_a_mount_still_shows(tmp_path, monkeypatch):
    """The check runs inside the job's own mounts before its command: a mask that did not hide a path stops
    the job with the path named, exactly as the wall refuses a turn."""
    private = tmp_path / "private.txt"
    private.write_text("PRIVATE\n")
    monkeypatch.setattr(
        sandbox, "mask_after", lambda mounts, hide: []
    )  # as if the mask were mounted too early
    argv = bwrap_argv(["/bin/true"], str(tmp_path), writable=[tmp_path], hide=[private], pid="private")
    got = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert got.returncode == 70, got
    assert str(private) in got.stderr and "still visible" in got.stderr, got.stderr


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("whole_source", [False, True])
def test_turn_masks_and_checks_a_hidden_object_at_each_bind_alias(tmp_path, directory, whole_source):
    """A hidden source reachable only at a relocated read-only bind must stay hidden in the turn."""
    from qda_agent.isolation import wall

    source = tmp_path / "source"
    source.mkdir()
    hidden = source if whole_source else source / "private"
    if not whole_source:
        hidden.mkdir() if directory else hidden.write_text("private")
    view = tmp_path / "absent-view"
    specs = []
    argv = wall.bwrap_argv([], [], [hidden], binds={view: source}, minimal=True, hidden_checks=specs)
    dest = view if whole_source else view / "private"
    i = argv.index(str(dest), argv.index(str(view)) + 1)
    if hidden.is_dir():
        assert argv[i - 1] == "--tmpfs"
    else:
        assert argv[i - 2 : i] == ["--ro-bind", "/dev/null"]
    st = hidden.stat()
    assert (str(dest), st.st_dev, st.st_ino) in specs
    mounts = [("tmpfs", "/", None), ("--ro-bind", str(view), str(source))]
    assert sandbox.mask_after(iter(mounts), [hidden]) == [(dest, hidden)]


def test_a_python_root_that_is_also_readable_is_bound_once(monkeypatch):
    """A venv named by prepare's config and by `python_roots` is mounted once.

    Asserted on the argv, not by running bwrap: the duplicate is only fatal from bubblewrap 0.10, where a
    `--ro-bind-fd` op closes its descriptor and `fstat`s it for a swap, so the second op on that fd fails
    with `Can't stat fd N`. CI installs 0.9.0, which has no such check and would pass either way.
    """
    venv = Path(sys.prefix)
    monkeypatch.setattr(sandbox, "python_roots", lambda: [venv])
    argv = sandbox.bwrap_argv(["/bin/true"], "/tmp", writable=[Path("/tmp")], readable=[venv])

    mounted = [argv[i + 2] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert mounted.count(str(venv)) == 1, f"{venv} bound {mounted.count(str(venv))} times: {mounted}"
