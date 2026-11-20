import functools
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # the tests' helpers (safetensors_kit, tests_support)

from testkit import sandbox_helpers  # noqa: E402, F401 (shared fixture)
from tests_support import endpoint  # noqa: E402, F401 (the stub endpoint a `dsh` run is pointed at)


@pytest.fixture(autouse=True)
def no_api_probe(request, monkeypatch):
    """No test opens a connection to an agent's model API (`api_route`): the probe holds, unless the test
    asks for `real_api_route` (and then replaces the socket itself)."""
    if "real_api_route" not in request.fixturenames:
        monkeypatch.setattr("qda_agent.api_route.api_route_problem", lambda host, port, *, direct: None)


@pytest.fixture
def real_api_route():
    """Opt out of `no_api_probe`: the test runs the real `api_route.api_route_problem`."""


@pytest.fixture(scope="session")
def _agent_cli_stubs(tmp_path_factory):
    stubs = tmp_path_factory.mktemp("agent-clis")
    for cli in ("claude", "codex", "qwen"):  # qwen: `tests_support.NoConfigCli`, a plugin backend
        (stubs / cli).write_text("#!/bin/sh\nexit 0\n")
        (stubs / cli).chmod(0o755)
    return stubs


@pytest.fixture(autouse=True)
def agent_clis_on_path(_agent_cli_stubs, monkeypatch):
    """Stub agent CLIs on `PATH`: a run refuses an agent CLI it cannot find before its clock
    (`AgentBackend.turn_problem`), and CI installs neither. The flow tests drive a stubbed Humanize, which
    never starts them; a test of the missing-CLI refusal sets its own `PATH`."""
    monkeypatch.setenv("PATH", f"{_agent_cli_stubs}{os.pathsep}{os.environ.get('PATH', '')}")


@pytest.fixture(autouse=True)
def humanize_home(tmp_path_factory, monkeypatch):
    """A fresh Humanize home per test: Humanize writes its epics there, never under the real `~/.hmz`."""
    home = tmp_path_factory.mktemp("hmz-home")
    monkeypatch.setenv("HUMANIZE_HOME", str(home))
    return home


#: The cards the fake driver table lists; tests name cards up to 7 (`cards: [0, 1]`, a full-card run's 8).
FAKE_GPUS = 8


@pytest.fixture(autouse=True)
def gpu_table(tmp_path_factory, monkeypatch):
    """The driver's GPU table: the real one when it lists at least `FAKE_GPUS` cards, else (CI with no driver,
    or a one-card machine) a fake one of 8 cards.

    The scheduler refuses to start without a driver while `gpus` is true, and refuses a card the table does
    not list (a run on `cards: [0, 1]` on a one-card machine); the fake table lets the queue tests run there.
    Its device minors (100 and up) name no `/dev/nvidia*` node, so no job sees a GPU. A subprocess
    the code starts still reads the real `/proc`. `QDA_TEST_FAKE_GPUS=1` uses
    the fake table on a GPU machine too (to check what CI sees).

    The fake cards are on no `nvidia-smi` either: the start's card health check (`doctor.card_health`) on the
    host's own table is left unjudged, as on CI, instead of refusing card 0 for a bus `nvidia-smi` does not
    list. A test that hands it a table (`proc`) still gets the real check.
    """
    from qda_agent import doctor
    from qda_agent.isolation.bwrap import cards_by_bus
    from qda_agent.sched import sandbox

    real = sandbox.NV_PROC.is_dir() and os.environ.get("QDA_TEST_FAKE_GPUS") != "1"
    if real and len(cards_by_bus(sandbox.NV_PROC)) >= FAKE_GPUS:
        return sandbox.NV_PROC
    proc = tmp_path_factory.mktemp("nvidia-gpus")
    for i in range(FAKE_GPUS):
        bus = f"0000:{0x18 + 0x10 * i:02x}:00.0"
        (proc / bus).mkdir()
        (proc / bus / "information").write_text(
            f"Model: \t\t NVIDIA H200\nGPU UUID: \t GPU-fake-{i}\nBus Location: \t {bus}\nDevice Minor: \t {100 + i}\n"
        )
    monkeypatch.setattr(sandbox, "NV_PROC", proc)
    health = doctor.card_health
    monkeypatch.setattr(
        doctor,
        "card_health",
        lambda cards, proc=None: (None, "no nvidia-smi listing") if proc is None else health(cards, proc),
    )
    return proc


@pytest.fixture(autouse=True)
def own_tmpdir(tmp_path, tmp_path_factory, monkeypatch):
    """A `$TMPDIR` the bwrap sandbox would grant the agent holds every `tmp_path`: point it at a fresh directory
    beside the test's own instead, as on CI.

    `Bwrap.writable()` names a `$TMPDIR` other than `/tmp` or `/var/tmp` as agent-writable, so a run refuses a
    model or reference under it. A machine with a small `/tmp` sets `TMPDIR` to a large disk, which puts each
    test's model there; this keeps the product rule and moves only the test's `$TMPDIR` aside.
    """
    from qda_agent.sandbox_settings import Bwrap

    if any(tmp_path.is_relative_to(d) for d in Bwrap().writable() or ()):
        monkeypatch.setenv("TMPDIR", str(tmp_path_factory.mktemp("tmpdir")))


@pytest.fixture
def fake_room(monkeypatch):
    """`fake_room(free_bytes, free_inodes, total_inodes=...)`: what `qda_agent.room.room` reads for every path.

    The figures are faked; the path is still the real nearest existing directory, so its filesystem is real.
    """
    from qda_agent import room as R

    def set_room(free_bytes: int, free_inodes: int, total_inodes: int = 10**9) -> None:
        def fake(path):
            return R.Room(R.existing(path), free_bytes, free_inodes, 1 << 40, total_inodes)

        monkeypatch.setattr(R, "room", fake)

    return set_room


def pytest_addoption(parser):
    parser.addoption(
        "--torch-python",
        default=None,
        help="python of a venv with torch: the GPU wall test also checks that torch sees no card through it",
    )


@pytest.fixture(autouse=True)
def no_other_work_on_the_node(monkeypatch):
    """Stand in for a dedicated node: the wall's `pid: shared` check would refuse this shared machine.

    Every wall test starts a turn on a node that is one Slurm job shared by every track, so no test could
    start a wall with the real check. The check itself is exercised in `test_pid_shared.py`, which patches this
    fixture's answer back.
    """
    monkeypatch.setattr("qda_agent.isolation.wall.why_not_dedicated", lambda: None)


# A host that cannot do what CI does. Each cause is probed on the host, whatever fails; a failed test is reported as
# skipped, with the cause, only when its probe fails here AND the failure's own text names that cause. Where the
# probes pass (CI), and for any failure that does not name the cause, every failure stays a failure.

#: The base a test's temporary paths sit under on CI (`/tmp/pytest-of-runner/pytest-0`).
CI_BASETEMP = len("/tmp/pytest-of-runner/pytest-0")

#: The flags a sandbox test starts bwrap with (`isolation.bwrap.bwrap_argv`), less the mounts.
BWRAP_FLAGS = ["--die-with-parent", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--new-session"]


@functools.cache
def bwrap_cannot_start() -> list[str]:
    """bwrap's own error lines (`bwrap: ...`) when it is installed but cannot start a sandbox here (no user
    namespaces), probed with the flags the tests use; empty when it starts, is missing, or fails without one."""
    if (bwrap := shutil.which("bwrap")) is None:
        return []  # the tests that need it skip on `shutil.which("bwrap")`
    argv = [bwrap, *BWRAP_FLAGS, "--ro-bind", "/", "/", "--dev", "/dev", "true"]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if done.returncode == 0:
        return []
    return [line for line in done.stderr.splitlines() if line.startswith("bwrap: ")]


@pytest.fixture(scope="session", autouse=True)
def host_bwrap_calls(tmp_path_factory):
    """Where this host's bwrap cannot start: a `bwrap` first on `PATH` that notes a sandbox start in a file, then
    runs the real one; None elsewhere.

    `--help` and `--version` (the feature probes) are not noted: they start no sandbox. Session-wide, so it is
    on `PATH` before any test's `monkeypatch` saves it, and a test's `monkeypatch.undo()` keeps it.
    """
    if not bwrap_cannot_start():
        yield None
        return
    shim = tmp_path_factory.mktemp("bwrap-shim")
    log = shim / "calls"
    real = shutil.which("bwrap")
    (shim / "bwrap").write_text(
        f'#!/bin/sh\ncase "$1" in --help|--version) ;; *) : >> {log} ;; esac\nexec {real} "$@"\n'
    )
    (shim / "bwrap").chmod(0o755)
    path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{shim}{os.pathsep}{path}"
    yield log
    os.environ["PATH"] = path


@pytest.fixture(autouse=True)
def bwrap_calls(request, host_bwrap_calls):
    """Each test starts with no bwrap start noted."""
    if host_bwrap_calls is not None:
        host_bwrap_calls.unlink(missing_ok=True)
        request.node.stash[_BWRAP_LOG] = host_bwrap_calls


_BWRAP_LOG = pytest.StashKey[Path]()


def _unsafe_parent(d: Path) -> bool:
    """Group- or world-writable and not a root-owned sticky directory (custody's rule, read here on its own)."""
    st = d.lstat()
    return bool(st.st_mode & 0o022) and not (st.st_uid == 0 and st.st_mode & stat.S_ISVTX)


@functools.cache
def _unsafe_ancestors(base: Path) -> list[Path]:
    """The ancestors of `base` custody would refuse here (probed on this host, whatever failed)."""
    return [a for a in base.parents if _unsafe_parent(a)]


def _temp_base_cause(base: Path, text: str) -> str | None:
    """Why `$TMPDIR` (not the code) fails this test. Two probes on this host (a base longer than CI's; an unsafe
    ancestor), each also named by the failure: a socket path too long that fits under CI's base, or that ancestor."""
    from qda_agent.rpc import SUN_PATH_MAX

    if len(str(base)) > CI_BASETEMP:
        for m in re.finditer(r"socket (/\S+) is too long for a unix socket", text):
            sock = Path(m.group(1))
            if sock.is_relative_to(base) and len(str(sock)) - len(str(base)) + CI_BASETEMP <= SUN_PATH_MAX:
                return f"$TMPDIR is too long for this test's unix sockets ({base}): set TMPDIR=/tmp"
    unsafe = _unsafe_ancestors(base)
    for m in re.finditer(r"writable ancestor (/[^\s'\"]+)", text):
        if Path(m.group(1)) in unsafe:
            return f"$TMPDIR has a group- or world-writable ancestor {m.group(1)}: set TMPDIR=/tmp"
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if not report.failed:
        return report
    text = report.longreprtext + "".join(content for _, content in report.sections)
    log = item.stash.get(_BWRAP_LOG, None)
    why = None
    if log is not None and log.is_file():
        # a failure that is not bwrap's own refusal to start (a wrong exit code, a missing denial) stays a failure
        if error := next((line for line in bwrap_cannot_start() if line in text), None):
            why = f"this host's bwrap cannot start a sandbox ({error}) and the test ran bwrap"
    if why is None:
        why = _temp_base_cause(item.config._tmp_path_factory.getbasetemp(), text)  # type: ignore[attr-defined]
    if why is not None:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {why}")
    return report
