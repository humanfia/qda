"""The agent's fence, held on this machine.

`isolation.SANDBOX` is a Humanize `Permission`; Humanize draws it into a `Fence` with `Fence.of` (the
four scopes, plus the system minimum) and runs every turn of the CLI inside `hmz internal fence`. These
tests draw the same fence and run a probe inside it, so they check what the agent's CLI would meet.
Skipped where the machine cannot hold a fence (no Landlock with TCP rules, ptrace forbidden, ...).
"""

import contextlib
import dataclasses
import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

fence = pytest.importorskip("hmz.coganchor.fence")

from qda_agent import isolation  # noqa: E402
from qda_agent.config import RunConfig  # noqa: E402
from qda_agent.isolation import CLI_CONFIG, SANDBOX, wall  # noqa: E402
from tests_support import NoConfigCli, agent, assert_job_process  # noqa: E402

needs_fence = pytest.mark.skipif(
    # Humanize 0.1's probe raises AttributeError on a Python without os.pidfd_open (see doctor.pidfd_problem).
    sys.platform != "linux" or not hasattr(os, "pidfd_open") or not fence.enforceable(net=True),
    reason="this machine cannot hold a network fence",
)
#: The wall's turns need the filesystem fence and bwrap only: the network is the sandbox's own namespace, so a
#: machine that refuses `pidfd_getfd` (a DGX node) can hold them.
needs_wall = pytest.mark.skipif(
    sys.platform != "linux" or not fence.enforceable(net=False) or shutil.which("bwrap") is None,
    reason="this machine cannot hold the wall's filesystem fence or run bubblewrap",
)

PROBE = r"""
import json, os, socket, sys, urllib.request
out = {}
def t(name, fn):
    try:
        fn(); out[name] = "ok"
    except Exception as e:
        out[name] = type(e).__name__
t("write_workdir", lambda: open("notes.md", "w").write("x"))
t("read_outside", lambda: os.listdir(sys.argv[1]))
t("read_home", lambda: os.listdir(os.path.expanduser("~")))
t("write_tmp", lambda: open(sys.argv[2], "w").write("x"))
def rpc():
    s = socket.socket(socket.AF_UNIX); s.connect(".qda/rpc.sock"); assert s.recv(5) == b"pong\n"
t("rpc_socket", rpc)
t("net_other", lambda: urllib.request.urlopen("https://pypi.org", timeout=10))
print(json.dumps(out))
"""


def _fence(work: Path, read=(), hosts=("api.anthropic.com",)):
    return fence.Fence.of(
        local=str(SANDBOX.local),
        user=str(SANDBOX.user),
        system=str(SANDBOX.system),
        online=False,
        workdir=work,
        home=Path.home(),
        hosts=list(hosts),
        read=read,
    )


def _run_fenced(work: Path, outside: Path, tmp_file: Path) -> dict:
    import json

    p = subprocess.run(
        [*fence.wrapper(_fence(work)), "/usr/bin/python3", "-c", PROBE, str(outside), str(tmp_file)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@needs_fence
def test_fence_workdir_socket_only(tmp_path):
    work = tmp_path / "work"
    # `/` is above every scope the fence grants and is no temporary directory, which `local=ALL` writes. A path
    # taken from `__file__` is only out of reach where the tree happens to be checked out: a copy under /tmp is
    # writable inside the fence, and one under the interpreter's own tree is readable.
    outside = Path("/")
    (work / ".qda").mkdir(parents=True)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(work / ".qda" / "rpc.sock"))
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        c.sendall(b"pong\n")
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    # Under /tmp, where the fence writes, but a fresh name: a fixed one would be left by an earlier run of
    # another user, and `fs.protected_regular` forbids opening another user's file there for writing.
    tmp_file = Path("/tmp") / f"qda-fence-probe-{os.getpid()}-{os.urandom(4).hex()}"
    try:
        out = _run_fenced(work, outside, tmp_file)
    finally:
        tmp_file.unlink(missing_ok=True)
    assert out["write_workdir"] == "ok"
    assert out["rpc_socket"] == "ok"  # the four commands reach the host
    assert out["read_outside"] == "PermissionError"  # run dir, judge, ledger: nothing above the workdir
    assert out["read_home"] == "PermissionError"
    assert out["net_other"] != "ok"  # only the model's API host passes the proxy
    # local=ALL also writes the temporary directories: a run dir must not be under one (isolation.check).
    assert out["write_tmp"] == "ok"


WALL_PROBE = r"""
import glob, json, os, socket, subprocess, sys, urllib.request
out = {"nvidia_nodes": glob.glob("/dev/nvidia*"), "home": None}
try:
    os.close(os.open("/dev/nvidiactl", os.O_RDWR)); out["nvidiactl"] = "opened"
except OSError as e:
    out["nvidiactl"] = e.errno
try:
    out["home"] = os.listdir(os.path.expanduser("~"))
except Exception as e:
    out["home"] = type(e).__name__
try:
    s = socket.socket(socket.AF_UNIX); s.connect(".qda/rpc.sock"); out["rpc"] = s.recv(5).decode()
except Exception as e:
    out["rpc"] = type(e).__name__
try:
    urllib.request.urlopen("https://pypi.org", timeout=10); out["net_other"] = "ok"
except Exception as e:
    out["net_other"] = type(e).__name__
if len(sys.argv) > 1:
    # CUDA_VISIBLE_DEVICES names a card of the run's pool, so no env guard hides it: only the sandbox can.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "4"}
    p = subprocess.run([sys.argv[1], "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True, env=env)
    out["cuda"] = p.stdout.strip() or p.stderr.strip()[-200:]
print(json.dumps(out))
"""


@needs_wall
@pytest.mark.skipif(  # the probe opens /dev/nvidiactl; a runner may have /dev/dri but no NVIDIA node
    not Path("/dev/nvidiactl").exists(),
    reason="needs an NVIDIA GPU",
)
def test_bwrap_wall_hides_gpus(tmp_path_factory, monkeypatch, request):
    # The run dir must not be under /tmp (the fence writes it), so build it under the package.
    import json
    import shutil as sh

    # A torch venv (--torch-python) may live under the real home, which the wall binds back only through its
    # allowlist (test_home_binds_is_an_allowlist); this test is about GPUs, so it hides a fake home instead.
    home = Path(__file__).parents[1] / "runs" / "gpu-home"  # not under /tmp, which the fence opens
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    # The CLI's config dir (`live_state` copies it), kept out of the hidden home: never the launcher's, whose
    # `sessions` another process may hold or write while pytest runs.
    cli = home.parent / "gpu-cli"
    cli.mkdir(parents=True, exist_ok=True)
    (cli / "settings.json").write_text("{}\n")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cli))

    run = tmp_path_factory.mktemp("wall") / "run"
    sh.rmtree(run, ignore_errors=True)
    work = run / "work"
    (work / ".qda").mkdir(parents=True)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(work / ".qda" / "rpc.sock"))
    srv.listen(1)

    def serve():
        while True:
            c, _ = srv.accept()
            c.sendall(b"pong\n")
            c.close()

    threading.Thread(target=serve, daemon=True).start()

    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    torch_py = request.config.getoption("--torch-python")
    quant = Path(torch_py).parents[1] if torch_py else None  # the venv, mounted back read-only
    f = _fence(work, read=[str(quant)] if quant else [])
    probe = ["/usr/bin/python3", "-c", WALL_PROBE]

    def run_probe(cmd, *extra):
        p = subprocess.run([*cmd, *probe, *extra], cwd=work, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout.strip().splitlines()[-1])

    # Control: the fence the wall puts inside the namespace (the filesystem's alone) does not hide the GPUs;
    # bwrap does. A network fence is not needed for it, so the control runs where the wall can start.
    bare = run_probe(fence.wrapper(dataclasses.replace(f, online=True)))
    assert bare["nvidiactl"] == "opened"  # (it cannot list /dev, but it can open the GPU nodes)
    with isolation.gpu_wall(cfg):  # the turn's way out is the wall's bridge: probe while it is up
        argv = fence.wrapper(f)  # what Humanize spawns each fenced turn with: bwrap ... -- hmz internal fence
        options = json.loads(Path(argv[4]).read_text())  # bwrap's own arguments, carried in a private file
        assert options[0] == "bwrap" and "--dev" in options
        out = run_probe(argv, *([torch_py] if quant else []))
    assert out["nvidia_nodes"] == []  # ls /dev/nvidia* finds nothing
    assert out["nvidiactl"] == errno.ENOENT  # the node is not there at all, not just refused
    assert out["rpc"] == "pong\n"  # the four commands still reach the host
    assert out["net_other"] != "ok"  # only the model's API host
    assert out["home"] in ([], "PermissionError")  # home hidden (tmpfs) and fenced
    if quant:
        assert out["cuda"] == "0"  # torch sees no GPU even with CUDA_VISIBLE_DEVICES=4
    assert fence.wrapper.__name__ == "wrapper"  # restored after the run
    sh.rmtree(run, ignore_errors=True)
    sh.rmtree(home, ignore_errors=True)
    sh.rmtree(cli, ignore_errors=True)


#: A stand-in for the CLI inside the wall: it uses the proxy `live_state` set, and nothing else.
WALL_NET = r"""
import json, os, socket, urllib.parse
p = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
out = {"proxy": f"{p.hostname}:{p.port}"}
def ask(target):
    s = socket.create_connection((p.hostname, p.port), timeout=20)
    s.sendall(("CONNECT " + target + " HTTP/1.1\r\nHost: " + target + "\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        part = s.recv(4096)
        if not part:
            break
        head += part
    return s, head.split(b"\r\n")[0].decode()
s, out["allowed"] = ask("127.0.0.1:%(port)d")
s.sendall(b"ping")
out["echo"] = s.recv(64).decode()
s.close()
_, out["refused"] = ask("example.com:443")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    out["direct"] = "ok"
except OSError as why:
    out["direct"] = type(why).__name__
print(json.dumps(out))
"""


def test_the_wall_gives_a_turn_a_network_of_its_own(tmp_path):
    """The argv cuts the network, binds the turn's socket in, and leaves only the filesystem fence inside."""
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    f = _fence(work, hosts=["api.anthropic.com"])
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(f)
    argv = json.loads(Path(argv[4]).read_text()) + argv[5:]
    assert "--unshare-net" in argv
    # The host's proxy socket: a private path bound read-only at a path of the sandbox's fresh /tmp.
    sock = next(a for a in argv if a.endswith(".sock") and "qda-net-" in a)
    dest = argv[argv.index(sock) + 1]
    assert argv[argv.index(sock) - 1] == "--ro-bind" and dest == f"/tmp/{Path(sock).stem}-wall.sock"
    # `live_state` is handed the destination for its relay, and the fence put inside is the filesystem's alone.
    live = argv.index(isolation.LIVE_SOURCE)
    assert argv[live + 2] == dest
    policy = json.loads(next(a for a in argv if a.startswith("--policy=")).removeprefix("--policy="))
    assert policy["online"] is True and policy["hosts"] == ["api.anthropic.com"]


@needs_wall
def test_the_wall_spells_the_home_from_its_real_path(tmp_path, monkeypatch):
    """A home reached through a link: bwrap cannot make the parents of a mount point under the link, so the
    wall spells its binds from the real home, and the link itself stays."""
    real = tmp_path / "real-home"
    (real / ".local/bin").mkdir(parents=True)
    (real / ".local/bin/claude").write_text("#!/bin/sh\n")
    link = tmp_path / "home"
    link.symlink_to(real)
    monkeypatch.setenv("HOME", str(link))
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    f = _fence(work, read=[str(link / ".local/bin/claude")])
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(f)
        options = json.loads(Path(argv[4]).read_text())
        p = subprocess.run([*argv, "/usr/bin/true"], cwd=work, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert str(real / ".local/bin/claude") in options  # the same file, spelled from the real home
    assert not any(a.startswith(f"{link}/") for a in options)  # no mount point hangs off the link


def test_the_wall_shows_the_turn_the_fences_paths_not_the_host_root(tmp_path):
    """The root starts empty: a host tree bound read-only shows every pathname socket, and `connect()` does not
    write. `--ro-bind / /` has no place in a turn's argv; the system minimum comes back one path at a time."""
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(_fence(work))
    argv = json.loads(Path(argv[4]).read_text())
    triples = [argv[i : i + 3] for i in range(len(argv) - 2)]
    # The root is the first mount made, and it is an empty tmpfs (pid args, if any, sit before it).
    first = next(i for i, a in enumerate(argv) if a in ("--tmpfs", "--ro-bind", "--bind", "--dev"))
    assert argv[first : first + 2] == ["--tmpfs", "/"]
    assert ["--ro-bind", "/", "/"] not in triples and ["--bind", "/", "/"] not in triples
    assert ["--ro-bind", "/usr", "/usr"] in triples  # the fence's system minimum is mounted back
    assert ["--ro-bind", "/srv", "/srv"] not in triples  # a host dir the fence does not name is not there


#: Read the receipted output, copy it elsewhere, then try every way to change it in place.
_RECEIPTED_READONLY = r"""
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
target = root / "candidate" / "model.safetensors"
def t(fn):
    try:
        fn()
    except OSError as e:
        return type(e).__name__
    return "ok"
out = {
    "read": t(lambda: target.read_bytes()),
    "alias_read": t(lambda: (root / "alias" / "candidate" / "model.safetensors").read_bytes()),
    "copy": t(lambda: (root / "copy.safetensors").write_bytes(target.read_bytes())),
    "write": t(lambda: target.write_bytes(b"x")),
    "append": t(lambda: target.open("ab").write(b"x")),
    "truncate": t(lambda: os.truncate(target, 0)),
    "rename": t(lambda: target.rename(target.with_name("moved"))),
    "chmod": t(lambda: target.chmod(0o644)),
    "unlink": t(lambda: target.unlink()),
}
print(json.dumps(out))
"""


@needs_wall
def test_a_receipted_build_output_is_read_only_in_a_wall_turn(tmp_path):
    """A receipted build is mounted read-only over its destination in the turn's own sandbox, after the
    writable workdir: the turn reads and copies it, and every in-place change fails."""
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    tree = run / "judge" / "builds" / "job1" / "published"
    tree.mkdir(parents=True)
    (tree / "model.safetensors").write_bytes(b"w1")
    (run / "judge" / "builds" / "job1" / "receipt.json").write_text(
        json.dumps({"job_id": "job1", "destination": "candidate", "workdir": str(work)})
    )
    (work / "alias").symlink_to(".", target_is_directory=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(_fence(work))
        argv = json.loads(Path(argv[4]).read_text()) + argv[5:]
        triples = [argv[i : i + 3] for i in range(len(argv) - 2)]
        assert ["--ro-bind", str(tree), str(work / "candidate")] in triples
        got = subprocess.run(
            [*argv, "/usr/bin/python3", "-I", "-S", "-c", _RECEIPTED_READONLY, str(work)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert got.returncode == 0, got.stderr
    seen = json.loads(got.stdout.strip().splitlines()[-1])
    assert seen["read"] == "ok" and seen["alias_read"] == "ok" and seen["copy"] == "ok", seen
    for op in ("write", "append", "truncate", "rename", "chmod", "unlink"):
        assert seen[op] != "ok", (op, seen)
    assert (tree / "model.safetensors").read_bytes() == b"w1"


@needs_wall
def test_a_receipted_build_is_read_only_at_a_two_hop_workdir_spelling(tmp_path, monkeypatch):
    """The config names the workdir through a home alias that hides a second hop (`~/site/scratch` ->
    `/site/scratch/...`) while the fence is given the resolved path. The build must be mounted at the
    resolved spelling the turn works in, not only at the alias spelling."""
    real = tmp_path / "realhome"
    (real / "site").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    (real / "site" / "scratch").symlink_to(scratch, target_is_directory=True)
    home = tmp_path / "home"
    home.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    run = home / "site" / "scratch" / "run"  # the config's spelling: two symlink hops
    work = run / "work"
    (scratch / "run" / "work").mkdir(parents=True)  # at the target: `run` itself traverses links
    resolved = Path(os.path.realpath(work))  # JudgeHost's workdir and the fence's
    assert resolved != work
    (work / "alias").symlink_to(".", target_is_directory=True)
    tree = run / "judge" / "builds" / "job1" / "published"
    tree.mkdir(parents=True)
    (tree / "model.safetensors").write_bytes(b"w1")
    (tree.parent / "receipt.json").write_text(
        json.dumps({"job_id": "job1", "destination": "candidate", "workdir": str(resolved)})
    )
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    assert cfg.workdir == run / "work" and cfg.workdir != resolved
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(_fence(resolved))
        argv = json.loads(Path(argv[4]).read_text()) + argv[5:]
        triples = [argv[i : i + 3] for i in range(len(argv) - 2)]
        assert ["--ro-bind", str(Path(os.path.realpath(tree))), str(resolved / "candidate")] in triples
        got = subprocess.run(
            [*argv, "/usr/bin/python3", "-I", "-S", "-c", _RECEIPTED_READONLY, str(resolved)],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert got.returncode == 0, got.stderr
    seen = json.loads(got.stdout.strip().splitlines()[-1])
    assert seen["read"] == "ok" and seen["alias_read"] == "ok" and seen["copy"] == "ok", seen
    for op in ("write", "append", "truncate", "rename", "chmod", "unlink"):
        assert seen[op] != "ok", (op, seen)
    assert (tree / "model.safetensors").read_bytes() == b"w1"


@needs_wall
def test_a_host_pathname_socket_outside_the_grants_is_not_reachable(tmp_path):
    """A host socket the fence never granted must be absent inside a turn (ENOENT), not merely read-only.

    A read-only bind refuses writes, not `connect()`, and the network namespace holds IP sockets alone: a
    listener under a host path the fence does not name answers a turn through the host root. The turn's root
    holds only what the fence reads and writes, so the socket is not there at all.
    """
    # Beside the checkout: a host dir under no home, temp or run dir, and one the fence's minimum omits.
    beside = Path(__file__).resolve().parents[3].parent
    home = Path(os.path.realpath(Path.home()))
    temp = [Path(t) for t in wall.TEMPORARY]
    if any(beside.is_relative_to(p) for p in (home, Path.home(), *temp)):
        pytest.skip(f"{beside} is inside the home or a temp dir, which the wall hides anyway")
    try:
        room = Path(tempfile.mkdtemp(prefix="qda-wall-sock-", dir=beside))
    except OSError:
        pytest.skip(f"{beside} is not writable")
    server = socket.socket(socket.AF_UNIX)
    at = room / "rpc.sock"
    try:
        server.bind(str(at))
        server.listen(1)
        with socket.socket(socket.AF_UNIX) as control:  # the listener answers on the host
            control.connect(str(at))
        run = tmp_path / "run"
        work = run / "work"
        work.mkdir(parents=True)
        cfg = RunConfig(
            contamination="none", model="m", format="w4a16", budget_gpu_hours=1, cards=[0], run_dir=run
        )
        code = (
            "import socket, sys\n"
            "s = socket.socket(socket.AF_UNIX)\n"
            "try:\n"
            "    s.connect(sys.argv[1])\n"
            "except OSError as why:\n"
            "    print(why.errno, type(why).__name__)\n"
            "else:\n"
            "    print('ok')\n"
        )
        with isolation.gpu_wall(cfg):
            argv = fence.wrapper(_fence(work))
            p = subprocess.run(
                [*argv, "/usr/bin/python3", "-c", code, str(at)],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=300,
            )
    finally:
        server.close()
        shutil.rmtree(room, ignore_errors=True)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == f"{errno.ENOENT} FileNotFoundError", p.stdout  # absent, not refused


@needs_wall
def test_a_wall_turn_reaches_only_its_hosts_proxy(tmp_path_factory):
    """A real turn under the wall: no route out, and the one proxy it has passes only the fence's hosts."""
    run = tmp_path_factory.mktemp("wallnet") / "run"
    work = run / "work"
    work.mkdir(parents=True)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    seen: list = []

    def serve() -> None:
        while True:
            try:
                client, _ = srv.accept()
            except OSError:
                return
            seen.append(client.recv(4096))
            client.sendall(b"pong")
            client.close()

    threading.Thread(target=serve, daemon=True).start()
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    f = _fence(work, hosts=[f"127.0.0.1:{port}"])
    with isolation.gpu_wall(cfg):
        argv = fence.wrapper(f)
        probe = ["/usr/bin/python3", "-c", WALL_NET % {"port": port}]
        p = subprocess.run([*argv, *probe], cwd=work, capture_output=True, text=True, timeout=300)
        srv.close()
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["proxy"].startswith("127.0.0.1:")
    assert out["allowed"].startswith("HTTP/1.1 200"), out
    assert out["echo"] == "pong" and seen == [b"ping"]  # it went through the proxy, not around it
    assert out["refused"].startswith("HTTP/1.1 403"), out  # another host is refused, as under the fence alone
    assert out["direct"] != "ok", out  # no route out of the namespace


FAKE_CLAUDE = r"""#!/usr/bin/python3
import glob, json, os, subprocess, sys
if "--version" in sys.argv:
    print("2.1.317 (Claude Code)")
    sys.exit(0)
# Walked from `/proc/self`, not from `os.getpid()`: where the wall's pid namespace could not mount a
# fresh procfs and binds the host's instead, a number the kernel gave this process (`os.getpid()`) is not
# the one the host's `/proc` lists it under, and `/proc/self` is the only name this process has there.
chain = []
at = "/proc/self"
try:
    while True:
        chain.append(open(at + "/cmdline", "rb").read().replace(b"\0", b" ").decode()[:200])
        pid = int(open(at + "/stat").read().rsplit(")", 1)[1].split()[1])
        if pid <= 0:
            break
        at = f"/proc/{pid}"
except OSError:
    pass
H = os.path.expanduser("~")
seen = {"claude": os.path.lexists(H + "/.claude"), "claude.json": os.path.lexists(H + "/.claude.json"),
        "anthropic": os.path.lexists(H + "/.config/anthropic"),
        "claude dirs": sorted(os.path.basename(p) for p in glob.glob(H + "/.claude*") if os.path.isdir(p)),
        "dir entries": {os.path.basename(p): sorted(os.listdir(p)) for p in glob.glob(H + "/.claude*") if os.path.isdir(p)}}
ccd = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(H, ".claude")  # unset: the CLI's default
seen["config dir"] = ccd
try:  # the CLI's own session write lands in the run's epic (cred --keep), which must be mounted
    os.makedirs(os.path.join(ccd, "projects", "probe"), exist_ok=True)
    with open(os.path.join(ccd, "projects", "probe", "s.jsonl"), "a") as f:
        f.write("{}\n")
    seen["session write"] = "ok"
except (OSError, TypeError) as e:
    seen["session write"] = type(e).__name__
sec = os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR") or ""
seen["sign-in"] = os.path.isfile(os.path.join(sec, ".credentials.json")) if sec else None
seen["sign-in dir"] = sec

def probe(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True).stdout.split()
    except OSError:
        return []
seen["tz"] = os.environ.get("TZ")
seen["zone"] = (probe(["date", "+%Z"]) or [""])[0]
seen["date"] = (probe(["date", "+%H:%M"]) or [""])[0]
ls = probe(["ls", "-l", "--time-style=+%H:%M", "stamp"])
seen["ls"] = ls[-2] if len(ls) > 1 else ""
with open("LOG", "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "chain": chain, "nvidia": glob.glob("/dev/nvidia*"), **seen}) + "\n")
sys.exit(3)
"""

PROBE_FLOW = """
from hmz.flows import Agent, AgentCollection, Env, EnvCollection, FlowContext, FlowParams, flow
from qda_agent.isolation import SANDBOX


class W(Agent):
    _permission = SANDBOX


class A(AgentCollection):
    worker: W


class E(EnvCollection):
    workspace: Env


@flow(agents=A, envs=E, params=FlowParams)
async def probe(task: str, *, agents: A, envs: E, params: FlowParams, ctx: FlowContext) -> str:
    w = agents["worker"]
    return await w.run(task, session=await w.spawn(), env=envs["workspace"])
"""


#: A file time far from "now": 2023-11-14 22:13:20 UTC (17:13 in New York, 07:13 in Tokyo).
STAMP = 1_700_000_000


def _spawn_a_turn(tmp_path, monkeypatch, **cfg_kwargs):
    """Run one fake-CLI turn inside the wall; return the run dir, workdir and the recorded turns."""
    from hmz.sdk import Hmz

    run = tmp_path / "spawn"
    if run.exists():
        import shutil as sh

        sh.rmtree(run)
    work = run / "work"
    work.mkdir(parents=True)
    (work / "LOG").write_text("")
    (work / "stamp").write_text("")
    os.utime(work / "stamp", (STAMP, STAMP))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "claude").write_text(FAKE_CLAUDE.replace('"LOG"', repr(str(work / "LOG"))))
    (bindir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    (run / "probe_flow.py").write_text(PROBE_FLOW)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        **cfg_kwargs,
    )
    with isolation.gpu_wall(cfg):
        r = Hmz(workspace=work).run(
            f"{run}/probe_flow.py:probe",
            "hi",
            agents={"worker": "claude/claude-haiku-5-5:low"},
            envs={"workspace": f"local{work}"},
            budget={"duration": 60},
        )
        with pytest.raises(Exception):  # noqa: B017 -- the fake CLI fails every turn after recording it
            r.run()
    turns = [json.loads(x) for x in (work / "LOG").read_text().splitlines()]
    return run, work, [t for t in turns if "--version" not in t["argv"]]


def _hhmm_window(start: datetime, end: datetime) -> set[str]:
    """The `%H:%M` strings a clock may show from a minute before `start` to a minute after `end`."""
    return {
        (start + timedelta(minutes=m)).strftime("%H:%M")
        for m in range(-1, int((end - start).total_seconds() // 60) + 2)
    }


@needs_wall
def test_a_spawned_turn_runs_inside_the_wall(tmp_path, monkeypatch):
    run, work, turns = _spawn_a_turn(tmp_path, monkeypatch)
    assert turns, "the fake CLI never took a turn"
    for t in turns:
        chain = " | ".join(t["chain"])
        assert "internal fence" in chain and "bwrap" in chain, chain  # CLI <- fence <- bwrap
        assert (
            "--bind " + str(Path.home() / ".claude") + " " not in chain
        )  # The user's CLI state is not mounted
        assert t["nvidia"] == []
        # nothing of the user's CLI state, however the test was launched (CLAUDE_CONFIG_DIR set or not)
        assert not t["claude"] and not t["claude.json"] and not t["anthropic"] and not t["claude dirs"], t
        assert not t["config dir"].startswith(str(Path.home())), (
            t
        )  # Humanize's per-session dir, not a home dir
        assert t["config dir"] == str(run / "live" / "claude"), t  # the live dir only the wall sees
        assert t["session write"] == "ok", t  # the CLI's sessions are written there
        assert t["sign-in dir"] == str(run / "cli" / "claude"), t  # the run's own config dir
    from hmz.runtime import epic

    assert (run / "live" / "claude" / "projects" / "probe" / "s.jsonl").is_file()
    # Humanize's session store is read-only inside: nothing of the turn lands where the host follows links
    assert not list(epic.under(work).glob("*/sessions/*/projects/probe/s.jsonl"))
    assert isolation.wall_hook().wrapper.__name__ == "wrapper"  # restored


@needs_wall
def test_the_walls_shell_runs_in_utc(tmp_path, monkeypatch):
    """Run 1's agent read local `ls -l` times as UTC; the shell's own `date` and `ls -l` are UTC."""
    monkeypatch.delenv("TZ", raising=False)
    before = datetime.now(UTC)
    _, _, turns = _spawn_a_turn(tmp_path, monkeypatch)
    after = datetime.now(UTC)
    expected = datetime.fromtimestamp(STAMP, UTC).strftime("%H:%M")
    assert turns, "the fake CLI never took a turn"
    for t in turns:
        assert t["tz"] == "UTC" and t["zone"] == "UTC", t
        assert t["date"] in _hhmm_window(before, after), t
        assert t["ls"] == expected, t


@needs_wall
def test_a_configured_tz_wins_in_the_wall_and_the_hosts_is_untouched(tmp_path, monkeypatch):
    """`toolset.env.TZ` reaches the shell; the flow never writes the host's own TZ."""
    monkeypatch.setenv("TZ", "America/Chicago")
    tokyo = timezone(timedelta(hours=9))
    before = datetime.now(tokyo)
    _, _, turns = _spawn_a_turn(tmp_path, monkeypatch, toolset={"env": {"TZ": "Asia/Tokyo"}})
    after = datetime.now(tokyo)
    expected = datetime.fromtimestamp(STAMP, tokyo).strftime("%H:%M")
    assert turns, "the fake CLI never took a turn"
    for t in turns:
        assert t["tz"] == "Asia/Tokyo" and t["zone"] == "JST", t
        assert t["date"] in _hhmm_window(before, after), t
        assert t["ls"] == expected, t
    assert os.environ["TZ"] == "America/Chicago"  # the host's environment is the caller's, not the wall's


@needs_wall
def test_the_wall_answers_for_the_networks_part_of_the_fence(tmp_path):
    """A driver says the wall holds the network: what Humanize must put around the CLI is the filesystem alone."""
    from hmz.coganchor.agents import driver
    from hmz.coganchor.fence import Fence, enforceable
    from qda_agent.isolation.cli_dirs import clis

    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path,
        sandbox="bwrap",
    )
    cls = driver(clis(cfg)[0][0])[0]
    said = cls.__dict__.get("natively")
    fence = Fence.of(
        local="read",
        user="read",
        system="read",
        online=False,
        workdir=tmp_path,
        home=Path.home(),
        hosts=("api.anthropic.com",),
    )
    with isolation.gpu_wall(cfg):
        rest = cls.natively(None, fence)  # as Humanize asks: `AgentBase._fences` does exactly this
        assert rest.online and not rest.open and enforceable(net=not rest.online)
        assert (rest.read, rest.write, rest.hosts) == (fence.read, fence.write, fence.hosts)
        assert (
            fence.online is False
        )  # the agent's own fence is untouched: its CLI is still told it is offline
    assert cls.__dict__.get("natively") is said  # restored


def test_hold_problem_uses_the_wall_for_the_network_the_native_fence_cannot_cut(tmp_path, monkeypatch):
    """Where the machine cannot cut the network, bwrap's netns wall holds it; `fence` alone cannot."""
    import hmz.coganchor.fence as hmz_fence

    monkeypatch.setattr(hmz_fence, "enforceable", lambda *, net: not net)  # the filesystem fence holds
    monkeypatch.setattr(isolation.shutil, "which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    walls = {"model": "m", "format": "w4a16", "budget_gpu_hours": 1, "cards": [0], "run_dir": tmp_path / "w"}
    assert isolation.hold_problem(RunConfig(contamination="none", **walls, sandbox={"name": "bwrap"})) is None


def test_hold_problem_refuses_only_where_the_wall_and_the_native_fence_both_fail(tmp_path, monkeypatch):
    """Neither the wall (no filesystem fence, no bubblewrap) nor the native fence holds: say both, refuse."""
    import hmz.coganchor.fence as hmz_fence

    monkeypatch.setattr(hmz_fence, "enforceable", lambda *, net: False)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "w",
        sandbox={"name": "bwrap"},
    )
    monkeypatch.setattr(isolation.shutil, "which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    problem = isolation.hold_problem(cfg)
    assert problem and "no filesystem fence" in problem and "sandbox docker" in problem
    monkeypatch.setattr(isolation.shutil, "which", lambda name: None)
    problem = isolation.hold_problem(cfg)
    assert problem and "no bubblewrap" in problem


def test_wall_refuses_without_hook_and_is_exclusive(tmp_path, monkeypatch):
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path,
        sandbox="bwrap",
    )
    hook = isolation.wall_hook()
    original = hook.wrapper
    with isolation.gpu_wall(cfg), pytest.raises(RuntimeError, match="another run"), isolation.gpu_wall(cfg):
        pass  # one wall per process
    assert hook.wrapper is original
    for exc in (ValueError("boom"), KeyboardInterrupt()):  # restored on an error and on ctrl+c / cancel
        with pytest.raises(type(exc)), isolation.gpu_wall(cfg):
            assert hook.wrapper is not original
            raise exc
        assert hook.wrapper is original
    monkeypatch.setattr(hook, "wrapper", lambda fence, extra: [])  # another Humanize: refuse to start
    with pytest.raises(RuntimeError, match="no hmz.coganchor.fence.wrapper"), isolation.gpu_wall(cfg):
        pass
    monkeypatch.delattr(hook, "wrapper")
    with pytest.raises(RuntimeError, match="no hmz.coganchor.fence.wrapper"):
        isolation.wall_hook()


async def test_wall_restored_on_cancellation(tmp_path):
    import asyncio

    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path,
        sandbox="bwrap",
    )
    original = isolation.wall_hook().wrapper

    async def hold():
        with isolation.gpu_wall(cfg):
            await asyncio.sleep(10)

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    assert isolation.wall_hook().wrapper is not original
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert isolation.wall_hook().wrapper is original


def test_wall_hides_the_users_cli_state_and_tmp(tmp_path):
    home = Path.home()
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    state = [str(home / ".claude"), str(home / ".claude.json"), str(home / ".config" / "anthropic")]
    scratch = tmp_path / "hmz-home-x"  # Humanize's per-session config dir lives under a temp dir
    scratch.mkdir()
    argv = isolation.bwrap_argv(
        read=["/usr", str(home / ".claude-7")],
        write=[str(work), "/tmp", "/var/tmp", *state, str(scratch), "/dev/nvidia0"],
        hide=[],
        hide_early=[home, run],
        keep=[work],
    )
    pairs = [
        (argv[i], argv[i + 1]) for i in range(len(argv) - 1) if argv[i] in ("--bind", "--tmpfs", "--ro-bind")
    ]
    binds = [b for f, b in pairs if f == "--bind"]
    assert (
        ("--tmpfs", "/tmp") in pairs
        and ("--tmpfs", "/var/tmp") in pairs
        and ("--tmpfs", str(home.resolve())) in pairs  # a hidden root is resolved: HOME may be a link
    )
    assert str(work) in binds and not any(b in binds for b in state) and "/tmp" not in binds
    assert str(scratch.resolve()) in binds or str(scratch) in binds  # Humanize's per-session dir is kept
    assert not any(b.startswith("/dev/nvidia") for b in binds)


def test_humanizes_session_homes_are_known_by_both_prefixes():
    tmp = tempfile.gettempdir()
    assert wall._humanize_home(f"{tmp}/hmz-home-ab12")
    assert wall._humanize_home(f"{tmp}/hmz-home-ab12/hmz-home-cd34")
    assert wall._humanize_home(f"{tmp}/hmz-homes-ab12")  # the parent of native and anchor turns' homes
    assert wall._humanize_home("/var/tmp/hmz-homes-ab12/x")
    assert not wall._humanize_home(f"{tmp}/hmz-hom-ab12")
    assert not wall._humanize_home(f"{tmp}/other/x")
    assert not wall._humanize_home("/srv/hmz-home-ab12")  # not in a temp dir


def test_overlap_compares_normalised_paths_and_keeps_links(tmp_path):
    a = tmp_path / "a"
    assert wall._overlaps(f"{a}/b/../c", [f"{a}/c"])
    assert wall._overlaps(f"{a}/./c//d", [f"{a}/c"])  # below
    assert wall._overlaps(str(a), [f"{a}/c/./d"])  # above
    assert not wall._overlaps(f"{a}/cc", [f"{a}/c"])
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert not wall._overlaps(str(tmp_path / "link"), [str(tmp_path / "real")])  # a link is not followed


def test_live_state_starts_without_the_interpreters_site_packages(tmp_path):
    """`live_state` runs in a venv whose site-packages holds a `.pth` file: under `LIVE_PYTHON` the `.pth` never
    runs (it does under `-I` alone), and the program still fills the live dir and moves the variable."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    site = subprocess.run(
        [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    marker = tmp_path / "pth-ran"
    Path(site, "zz_probe.pth").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n")
    subprocess.run([python, "-I", "-c", "pass"], check=True)
    assert marker.exists()  # the probe works: `-I` alone runs the .pth
    marker.unlink()

    home, live = tmp_path / "home", tmp_path / "live"
    home.mkdir()
    (home / "settings.json").write_text("{}")
    (tmp_path / "skill").mkdir()
    (home / "skills").symlink_to(tmp_path / "skill")
    spec = json.dumps({"CLAUDE_CONFIG_DIR": str(live)})
    show = "import os; print(os.environ['CLAUDE_CONFIG_DIR'])"
    live_argv = [python, *isolation.LIVE_PYTHON[1:], isolation.LIVE_SOURCE, spec, "-", "--"]
    out = subprocess.run(
        [*live_argv, python, "-I", "-S", "-c", show],
        env={"CLAUDE_CONFIG_DIR": str(home), "PATH": os.environ["PATH"]},
        check=True,
        capture_output=True,
        text=True,
    )
    assert not marker.exists()
    assert out.stdout.strip() == str(live)
    assert (live / "settings.json").read_text() == "{}" and (live / "skills").is_dir()
    assert not (live / "skills").is_symlink()


def test_private_cli_config_is_built_from_scratch(tmp_path, monkeypatch):
    """Each CLI's config dir is the run's own, never the launcher's; a sign-in file only when the CLI
    signs in by a file, and then a link into the wall's read-only mount of the default account dir."""
    home = tmp_path / "home"
    launcher = home / ".claude-launcher"  # the account of the session that started the run
    (launcher / "projects").mkdir(parents=True)
    (launcher / ".credentials.json").write_text("launcher")
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text("default")
    (home / ".codex").mkdir()
    (home / ".codex" / "auth.json").write_text("default")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(launcher))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    for t in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    ):
        monkeypatch.delenv(t, raising=False)
    run = tmp_path / "run"
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        agent=agent("claude/m", "codex/m"),
    )
    with isolation.private_cli_config(cfg) as dirs:
        assert dirs == {"claude": run / "cli" / "claude", "codex": run / "cli" / "codex"}
        assert os.environ["CLAUDE_CONFIG_DIR"] == str(dirs["claude"])
        assert os.environ["CODEX_HOME"] == str(dirs["codex"])
        assert sorted(p.name for p in dirs["claude"].iterdir()) == [".credentials.json", "settings.json"]
        assert (dirs["claude"] / ".credentials.json").is_symlink()  # never the launcher's copy of a file
        assert (dirs["claude"] / ".credentials.json").readlink() == Path(
            "../../signin/claude/.credentials.json"
        )
        assert sorted(p.name for p in dirs["codex"].iterdir()) == ["auth.json", "config.toml"]
        assert (dirs["codex"] / "auth.json").is_symlink()
        assert isolation.sign_in_accounts(cfg) == {"claude": home / ".claude", "codex": home / ".codex"}
    assert os.environ["CLAUDE_CONFIG_DIR"] == str(launcher) and "CODEX_HOME" not in os.environ  # restored
    # a token in the environment, or an account's provider: no sign-in file at all
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        agent=agent("claude/m", "codex@acct/m"),
    )
    with isolation.private_cli_config(cfg) as dirs:
        assert [p.name for p in dirs["claude"].iterdir()] == ["settings.json"]
        assert [p.name for p in dirs["codex"].iterdir()] == ["config.toml"]
    with pytest.raises(ValueError, match="no qda_agent.backends plugin 'qwen' is installed"):
        RunConfig(
            contamination="none",
            model="m",
            format="w4a16",
            budget_gpu_hours=1,
            cards=[0],
            run_dir=run,
            agent=agent("qwen/m"),
        )
    # a backend plugin whose CLI's config dir cannot be moved: the wall refuses it (it would be the user's)
    no_dir = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",  # the wall is what refuses a CLI whose config dir cannot be moved
        agent={"backend": NoConfigCli(model="m")},
    )
    with pytest.raises(ValueError, match="CLI_CONFIG"):
        isolation.check(no_dir)


def test_private_cli_config_links_the_sign_in_into_the_account_mount(tmp_path, monkeypatch, caplog):
    """`agent.backend.sign_in_dir` is the run's sign-in: the run's config dir links its sign-in file into the
    empty `account` path the wall mounts the account dir at, so the live file is what is read and a refresh on
    the host (`os.replace`) shows through. A `fence` run, with no mount namespace, keeps the read-only copy,
    which no refresh reaches. A missing dir or file, a second sign-in source, two dirs for one CLI, or a
    sign-in with no usable expiry is refused at start; under the deadline plus 30 min it only warns when the
    run names its own account dir (something on the host may refresh that), and the CLI's default dir is
    refused; every refusal and warning names `claude setup-token` and `CLAUDE_CODE_OAUTH_TOKEN`."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    for var in (
        "CLAUDE_CONFIG_DIR",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    chosen, run = tmp_path / "chosen", tmp_path / "run"
    (chosen / "projects").mkdir(parents=True)
    (chosen / "projects" / "x.jsonl").write_text("another session")
    (chosen / "history.jsonl").write_text("another session")
    sign_in = chosen / ".credentials.json"

    def _ms(hours: float) -> int:
        return int((datetime.now(UTC) + timedelta(hours=hours)).timestamp() * 1000)

    def token(expires_at=None):
        inner = {"accessToken": "SECRET"}
        if expires_at is not None:
            inner["expiresAt"] = expires_at
        return json.dumps({"claudeAiOauth": inner})

    sign_in.write_text(token(_ms(4)))

    def cfg(extra_claude=False, sandbox="bwrap", at=None, **backend):
        return RunConfig(
            contamination="none",
            model="m",
            format="w4a16",
            budget_gpu_hours=1,
            cards=[0],
            run_dir=at or run,
            sandbox={"name": sandbox},
            agent={
                "backend": {"name": "claude", "model": "m", **backend},
                "fallbacks": [{"name": "claude", "model": "m"}] if extra_claude else [],
            },
        )

    with isolation.private_cli_config(cfg(sign_in_dir=str(chosen))) as dirs:
        d, live = dirs["claude"], run / "live" / "claude"
        assert sorted(p.name for p in d.iterdir()) == [".credentials.json", "settings.json"]
        link = Path("..", "..", "signin", "claude", ".credentials.json")
        assert (d / ".credentials.json").readlink() == link  # no copy: a link to the wall's mount
        assert (live / ".credentials.json").readlink() == link  # the dir the CLI works in, not a copy
        assert not (d / ".credentials.json").is_file()  # the old copy path is gone
        assert not list((run / "signin" / "claude").iterdir())  # the wall's mount point, empty on the host
        assert isolation.sign_in_accounts(cfg(sign_in_dir=str(chosen)))["claude"] == chosen
        isolation.check_sign_in(cfg(sign_in_dir=str(chosen)), datetime.now(UTC) + timedelta(hours=1))
    deadline = datetime.now(UTC) + timedelta(hours=1)
    limit = deadline + timedelta(minutes=30)
    # the live file is what a walled run reads: a refresh after the run started is seen, not a stale copy
    sign_in.write_text(token(_ms(-1)))
    with isolation.private_cli_config(cfg(sign_in_dir=str(chosen))):
        sign_in.write_text(token(_ms(4)))
        isolation.check_sign_in(cfg(sign_in_dir=str(chosen)), deadline)  # no refusal
    # under the limit a named account dir only warns, naming the one-year token as the way to last the run
    sign_in.write_text(token(int(limit.timestamp() * 1000) - 1000))
    with (
        caplog.at_level("WARNING", logger="qda_agent.isolation"),
        isolation.private_cli_config(cfg(sign_in_dir=str(chosen))),
    ):
        isolation.check_sign_in(cfg(sign_in_dir=str(chosen)), deadline)
    assert "claude setup-token" in caplog.text and "CLAUDE_CODE_OAUTH_TOKEN" in caplog.text
    # the CLI's default dir: the same sign-in is refused, and the refusal says how to sign in for a long run
    (home / ".claude" / ".credentials.json").write_text(token(int(limit.timestamp() * 1000) - 1000))
    with pytest.raises(isolation.SignInRefused, match="claude setup-token") as short:
        isolation.check_sign_in(cfg(), deadline)
    assert "export the token it prints as CLAUDE_CODE_OAUTH_TOKEN" in str(short.value)
    assert "refresh loop" not in str(short.value)  # qda ships no refresh loop: the refusal never asks for one
    for bad in (None, "soon", 1e999):  # no usable expiry in the live file
        sign_in.write_text(token(bad))
        with pytest.raises(isolation.SignInRefused, match="no usable expiry") as caught:
            isolation.check_sign_in(cfg(sign_in_dir=str(chosen)), deadline)
        assert ".credentials.json" not in str(caught.value)  # no file name, path or content in the refusal
        assert "claude setup-token" in str(caught.value) and "remove agent.backend.sign_in_dir" in str(
            caught.value
        )
    sign_in.write_text(token(_ms(4)))

    with pytest.raises(ValueError, match="does not exist"):  # refused when the config is loaded
        cfg(sign_in_dir=str(tmp_path / "absent"))
    gone = tmp_path / "gone"
    gone.mkdir()
    missing = cfg(sign_in_dir=str(gone))
    gone.rmdir()  # gone between the config load and the run's start
    with pytest.raises(ValueError, match="is not a directory"), isolation.private_cli_config(missing):
        pass
    empty = tmp_path / "empty"
    empty.mkdir()
    with (
        pytest.raises(ValueError, match="no sign-in file"),
        isolation.private_cli_config(cfg(sign_in_dir=str(empty))),
    ):
        pass
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    with (
        pytest.raises(ValueError, match="name one sign-in"),
        isolation.private_cli_config(cfg(sign_in_dir=str(chosen))),
    ):
        pass
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    # an account's provider signs in instead
    with (
        pytest.raises(ValueError, match="name one sign-in"),
        isolation.private_cli_config(cfg(sign_in_dir=str(chosen), account="acct")),
    ):
        pass
    # the fallback would pick another sign-in
    with (
        pytest.raises(ValueError, match="different dirs"),
        isolation.private_cli_config(cfg(sign_in_dir=str(chosen), extra_claude=True)),
    ):
        pass
    (home / ".claude-x").mkdir()
    expanded = cfg(sign_in_dir="~/.claude-x")
    assert expanded.agent.backend.sign_in_dir == home / ".claude-x"  # `~` names the home


def _none_run(tmp_path: Path, **backend: object) -> tuple[RunConfig, Path]:
    """A level-none config and its run dir for the sign-in tests."""
    run = tmp_path / "run"
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        isolation={"level": "none"},
        agent={"backend": {"name": "claude", "model": "m", **backend}},
    )
    return cfg, run


def _no_claude_token(monkeypatch) -> None:
    """Clear the variables that would sign the CLI in without a file."""
    for var in ("CLAUDE_CONFIG_DIR", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_level_none_gets_the_runs_config_dir_with_the_users_sign_in_linked(tmp_path, monkeypatch):
    """Level none: the run's own config dir, an empty settings file, and a SYMLINK to the user's sign-in file.

    Everything the CLI writes (CLAUDE.md, commands/, history) stays in the run's dir; the user's dir is
    unchanged after the run, and the link target is the user's own file, never a copy.
    """
    home = tmp_path / "home"
    account = home / ".claude"
    (account / "commands").mkdir(parents=True)
    (account / "CLAUDE.md").write_text("the user's instructions\n")
    (account / "commands" / "brand.md").write_text("the user's command\n")
    (account / "history.jsonl").write_text("another session\n")
    sign_in = account / ".credentials.json"
    sign_in.write_text('{"claudeAiOauth": {"accessToken": "x", "expiresAt": 4102444800000}}\n')
    monkeypatch.setenv("HOME", str(home))
    _no_claude_token(monkeypatch)
    files = {p.relative_to(account): p.read_bytes() for p in account.rglob("*") if p.is_file()}
    entries = {p.relative_to(account) for p in account.rglob("*")}
    cfg, run = _none_run(tmp_path)
    with isolation.gpu_wall(cfg):  # the level-none branch sets the run's own config dirs up
        d = run / "cli" / "claude"
        assert os.environ["CLAUDE_CONFIG_DIR"] == str(d)
        assert (d / "settings.json").read_text() == "{}\n"
        link = d / ".credentials.json"
        assert link.is_symlink() and link.readlink() == sign_in  # the user's file, not a copy
        assert sorted(p.name for p in d.iterdir()) == [".credentials.json", "settings.json"]
        # what the CLI writes lands in the run's dir
        (d / "CLAUDE.md").write_text("agent instructions\n")
        (d / "commands").mkdir()
        (d / "commands" / "agent.md").write_text("agent command\n")
        (d / "history.jsonl").write_text("this run\n")
    assert "CLAUDE_CONFIG_DIR" not in os.environ  # restored: the launcher's was not passed on
    assert {p.relative_to(account): p.read_bytes() for p in account.rglob("*") if p.is_file()} == files
    assert {p.relative_to(account) for p in account.rglob("*")} == entries  # no new entry either


def test_level_none_refreshes_the_users_sign_in_through_the_link(tmp_path, monkeypatch):
    """A CLI refresh written through the link updates the user's own file in place, and the link stays a link."""
    home = tmp_path / "home"
    account = home / ".claude"
    account.mkdir(parents=True)
    sign_in = account / ".credentials.json"
    sign_in.write_text('{"claudeAiOauth": {"accessToken": "old", "expiresAt": 1}}\n')
    monkeypatch.setenv("HOME", str(home))
    _no_claude_token(monkeypatch)
    cfg, run = _none_run(tmp_path)
    refreshed = '{"claudeAiOauth": {"accessToken": "refreshed", "expiresAt": 4102444800000}}\n'
    with isolation.private_cli_config(cfg):
        link = run / "cli" / "claude" / ".credentials.json"
        link.write_text(refreshed)
        assert link.is_symlink()  # the write followed the link; it did not replace it
    assert sign_in.read_text() == refreshed


def test_level_none_links_the_named_account_dirs_sign_in(tmp_path, monkeypatch):
    """`agent.backend.sign_in_dir` at level none: the link points at that dir's sign-in file."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    _no_claude_token(monkeypatch)
    chosen = tmp_path / "account"
    chosen.mkdir()
    sign_in = chosen / ".credentials.json"
    sign_in.write_text("{}\n")
    cfg, run = _none_run(tmp_path, sign_in_dir=str(chosen))
    with isolation.private_cli_config(cfg):
        assert (run / "cli" / "claude" / ".credentials.json").readlink() == sign_in


def test_level_none_leaves_a_backend_without_a_config_dir_alone(tmp_path, monkeypatch):
    """A backend whose CLI has no `CLI_CONFIG` entry has no dir to move: level none neither refuses it nor
    changes the environment (not even the launcher's `CLAUDE_CONFIG_DIR` / `CODEX_HOME` / `DSH_HOME`)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    tmp_path.joinpath("run").mkdir()
    for var in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "DSH_HOME"):
        monkeypatch.setenv(var, str(home / var))
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        isolation={"level": "none"},
        agent={"backend": NoConfigCli(model="m")},
    )
    before = dict(os.environ)
    with isolation.gpu_wall(cfg) as _:
        assert dict(os.environ) == before
        assert not (tmp_path / "run" / "cli").exists()
    assert dict(os.environ) == before


def test_level_none_creates_no_link_when_a_token_or_an_account_signs_in(tmp_path, monkeypatch):
    """A token variable or an account provider signs the CLI in: the private dir holds only its settings."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    _no_claude_token(monkeypatch)
    assert "CLAUDE_CODE_OAUTH_TOKEN" in CLI_CONFIG["claude"][3]  # a documented auth variable, like the keys
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "t")
    cfg, _ = _none_run(tmp_path)
    assert isolation.sign_in_accounts(cfg) == {"claude": None}
    with isolation.private_cli_config(cfg) as dirs:
        assert sorted(p.name for p in dirs["claude"].iterdir()) == ["settings.json"]
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    account_cfg, _ = _none_run(tmp_path, account="acct")
    with isolation.private_cli_config(account_cfg) as dirs:
        assert sorted(p.name for p in dirs["claude"].iterdir()) == ["settings.json"]


@pytest.mark.parametrize("level", ["sandbox", "strict"])
def test_the_oauth_token_is_kept_through_the_wall(tmp_path, monkeypatch, level):
    """`CLAUDE_CODE_OAUTH_TOKEN` is the claude token variable the run selected: the wall keeps it like the API
    keys it already passes, and still drops every other secret-like name the launcher session carries."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    _no_claude_token(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-token")
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "launcher-only")
    monkeypatch.setenv("SOME_MESSAGING_KEY", "launcher-only")
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run",
        isolation={"level": level},
        agent=agent("claude/m"),
    )
    cfg.workdir.mkdir(parents=True)
    assert "CLAUDE_CODE_OAUTH_TOKEN" in isolation.sign_in_tokens(cfg)
    monkeypatch.setattr(wall, "pid_form", lambda _: "private")
    seen = []

    class StopAfterArgv(Exception):
        pass

    def spy(*args, **kwargs):
        seen.append(kwargs)
        raise StopAfterArgv

    monkeypatch.setattr(wall, "bwrap_argv", spy)
    with isolation.gpu_wall(cfg):
        with pytest.raises(StopAfterArgv):
            fence.wrapper(
                fence.Fence(read=(str(cfg.workdir),), write=(str(cfg.workdir),), online=False, hosts=())
            )
        assert not (cfg.run_dir / "cli" / "claude" / ".credentials.json").exists()  # a token, not a file
    assert len(seen) == 1
    unset = seen[0]["unset"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in unset  # the selected sign-in reaches the agent
    assert "CLAUDE_CODE_MESSAGING_TOKEN" in unset and "SOME_MESSAGING_KEY" in unset  # the launcher's stay out
    # a CLI the run does not use keeps nothing: the same name is a launcher secret for a codex-only run
    codex = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=tmp_path / "run2",
        isolation={"level": level},
        agent=agent("codex/m"),
    )
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in isolation.sign_in_tokens(codex)


@pytest.mark.parametrize(
    ("cli", "file", "unset"),
    [
        (
            "claude",
            ".credentials.json",
            ("CLAUDE_CONFIG_DIR", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        ),
        ("codex", "auth.json", ("CODEX_HOME", "OPENAI_API_KEY", "CODEX_API_KEY")),
    ],
)
@needs_wall
def test_the_wall_shows_only_the_sign_in_file(tmp_path, monkeypatch, cli, file, unset):
    """Each wall binds only the account dir's sign-in file into an empty dir, so nothing else is
    reachable -- neither the entries the dir held at the start (another session's transcripts and history) nor
    ones the host creates after the launch. A wall records the file it bound
    (`sign_in_mounted`), so a refresh on the host is visible as a stale bind to the flow, which recycles the
    process; the next wall binds the new file."""
    home = tmp_path / "home"
    account = home / f".{cli}-acc"
    (account / "projects").mkdir(parents=True)
    (account / "projects" / "x.jsonl").write_text("another session\n")
    (account / "history.jsonl").write_text("another session\n")
    sign_in = account / file
    sign_in.write_text("FIRST\n")
    monkeypatch.setenv("HOME", str(home))
    for var in unset:
        monkeypatch.delenv(var, raising=False)
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        agent={"backend": {"name": cli, "model": "m", "sign_in_dir": str(account)}},
    )
    d = run / "signin" / cli  # the empty dir the wall binds the sign-in into
    live = run / "live" / cli / file  # what the CLI reads: a link to the bound file
    probe = r"""
import json, os, sys
out = {}
for p in sys.argv[1:]:
    try:
        out[p] = {"entries": sorted(os.listdir(p))} if os.path.isdir(p) else {"read": open(p).read()}
    except OSError as e:
        out[p] = {"error": type(e).__name__}
print(json.dumps(out))
"""
    gone = ("projects", "history.jsonl", "late.jsonl", "sessions")
    inside = [live, d / file, d, *(d / name for name in gone)]

    def seen() -> dict:
        p = subprocess.run(
            [*fence.wrapper(_fence(work)), "/usr/bin/python3", "-I", "-S", "-c", probe, *map(str, inside)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout.strip().splitlines()[-1])

    with isolation.gpu_wall(cfg):
        assert live.is_symlink() and not live.exists()  # a link to the mount, never a copy
        assert isolation.sign_in_mounted(cfg) is None  # no wall (process) is up yet
        # what the wall binds: inode + mtime, never the file's content
        before = isolation.sign_in_witness(cfg)
        assert not isolation.sign_in_refreshed(cfg, before)
        first = seen()
        assert isolation.sign_in_mounted(cfg) == before  # the process holds the file stat it started with
        (account / "late.jsonl").write_text("created after the launch\n")  # another session's turn ...
        (account / "sessions").mkdir()
        (account / "sessions" / "late.jsonl").write_text("created after the launch\n")  # ... and history
        tmp = sign_in.with_name(sign_in.name + ".tmp")
        tmp.write_text("SECOND\n")
        tmp.replace(sign_in)  # a refresh on the host: the same path, a new inode
        assert isolation.sign_in_refreshed(cfg, before)  # the flow resends on the new file
        assert isolation.sign_in_refreshed(cfg, isolation.sign_in_mounted(cfg))  # the warm bind is stale too
        second = seen()
        # the new wall binds the new file
        assert isolation.sign_in_mounted(cfg) == isolation.sign_in_witness(cfg)
    assert isolation.sign_in_mounted(cfg) is None  # the run's wall is down
    for got, want in ((first, "FIRST\n"), (second, "SECOND\n")):
        assert got[str(live)]["read"] == want, got  # the CLI's own sign-in, through the link
        assert got[str(d / file)]["read"] == want, got  # the file the wall bound
        assert got[str(d)]["entries"] == [file], got  # nothing else is there
        for name in gone:  # nor anything the dir held at the start or gained after it
            assert got[str(d / name)] == {"error": "FileNotFoundError"}, got


@needs_wall
def test_the_wall_refuses_an_account_dir_inside_a_hidden_root(tmp_path, monkeypatch):
    """The account dir is bound back from the host's own tree, so it may not lie under a path the wall hides
    (nor the run dir): a mask over it would take the sign-in with it. The run is refused before the wall is
    taken, so the lock is free for the next run."""
    home = tmp_path / "home"
    account = home / ".claude-acc"
    account.mkdir(parents=True)
    (account / ".credentials.json").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    for var in (
        "CLAUDE_CONFIG_DIR",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox={"name": "bwrap", "hide": [str(home)]},
        agent={"backend": {"name": "claude", "model": "m", "sign_in_dir": str(account)}},
    )
    with pytest.raises(isolation.WallRefused, match="hides"), isolation.gpu_wall(cfg):
        pass


FAKE_CLAUDE_HOOK = r"""#!/usr/bin/python3
import json, os, shlex, subprocess, sys
if "--version" in sys.argv:
    print("2.1.317 (Claude Code)")
    sys.exit(0)
argv = sys.argv[1:]
settings = json.loads(argv[argv.index("--settings") + 1]) if "--settings" in argv else {}
fired = []
for group in (settings.get("hooks") or {}).get("PreToolUse", []):
    for h in group.get("hooks", []):
        payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "kl list"},
                   "session_id": "s", "cwd": os.getcwd()}
        p = subprocess.run(shlex.split(h["command"]), input=json.dumps(payload), text=True, capture_output=True)
        fired.append([p.returncode, p.stderr[-300:]])
        sock = shlex.split(h["command"])[-1]
        try:
            os.unlink(sock)
            fired.append(["unlinked"])
        except OSError as e:
            import errno
            fired.append([errno.errorcode.get(e.errno, str(e.errno))])
import socket
env = sorted(k for k in os.environ if k.startswith("CLAUDE_CODE_") or "MESSAGING" in k)
try:  # a socket in the user's runtime dir (as Claude Code's messaging sockets are)
    s = socket.socket(socket.AF_UNIX)
    s.connect("RTSOCK")
    seen = "connected"
except OSError as e:
    seen = type(e).__name__
open("HOOKLOG", "a").write(json.dumps({"fired": fired, "env": env, "runtime": seen}) + "\n")
sys.exit(3)
"""


@pytest.fixture
def runtime_socket():
    """A socket path short enough for AF_UNIX in the user's runtime dir, removed even if the test fails."""
    rt = Path(f"/run/user/{os.getuid()}")
    if not rt.is_dir():
        pytest.skip("no /run/user/$UID here")
    path = rt / f"qda-wall-test-{os.getpid()}.sock"
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


@needs_wall
def test_supervisor_tool_log_is_written_inside_the_wall(tmp_path, monkeypatch, runtime_socket):
    """A real Humanize spawn of a fake claude inside the wall: its PreToolUse hook reaches tools.jsonl."""
    import shutil as sh

    import yaml
    from qda_agent import backends

    run = Path(__file__).parents[1] / "runs" / "tw"  # short: the socket path must fit; not under /tmp
    if run.exists():
        sh.rmtree(run)
    run.parent.mkdir(parents=True, exist_ok=True)  # no other test may have made runs/ (CI skips the GPU one)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    rtsock = runtime_socket
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(rtsock))
    listener.listen(4)
    fake = FAKE_CLAUDE_HOOK.replace('"HOOKLOG"', repr(str(run / "work" / "HOOKLOG"))).replace(
        "RTSOCK", str(rtsock)
    )
    (bindir / "claude").write_text(fake)
    (bindir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    # as when qda-agent is started from inside a Claude Code session: its messaging socket and token
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_SOCKET", f"/run/user/{os.getuid()}/cc-socks/x.sock")
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "secret")
    monkeypatch.setenv("SOME_MESSAGING_KEY", "secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-token")  # the run's own claude sign-in
    config = run.parent / "tw.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": "m",
                "format": "w4a16",
                "budget_gpu_hours": 30 / 3600,
                "contamination": "none",
                "cards": [0],
                "run_dir": str(run),
                "agent": agent("claude/claude-haiku-5-5:low", resend_wait_s=1),
                "sandbox": "bwrap",
                "host": "testkit:fake_host",
                "commands": None,
                "drain_timeout_s": 1,
                "judge": {"eval_max_tokens": 32768},  # the task statement states the final evaluation's cap
            }
        )
    )
    cfg = RunConfig.load(config)
    cfg.workdir.mkdir(parents=True)
    r = backends.make_run(cfg, config)
    with isolation.gpu_wall(cfg):
        r.run()
    logs = [json.loads(x) for x in (run / "work" / "HOOKLOG").read_text().splitlines()]
    inherited = set(
        isolation.inherited_secrets(os.environ, isolation.sign_in_tokens(cfg))
    )  # the parent session's, incl. messaging socket+token; the run's own token is not one
    assert logs and all(not inherited & set(x["env"]) for x in logs), logs  # none reach the agent
    assert all("CLAUDE_CODE_OAUTH_TOKEN" in x["env"] for x in logs), logs  # the selected sign-in does
    assert all(not any("MESSAGING" in k for k in x["env"]) for x in logs), (
        logs
    )  # (Humanize's own settings stay)
    assert all(x["runtime"] != "connected" for x in logs), logs  # /run/user/$UID (cc-socks, buses) hidden
    listener.close()
    fired = [x["fired"] for x in logs]
    assert fired and fired[0] and fired[0][0][0] == 0, fired  # the relay reached the flow's hook socket
    # ... which the agent cannot remove: a read-only bind (EROFS), whatever the temp dir is
    assert all(f[1] == ["EROFS"] for f in fired), fired
    tools = [json.loads(x) for x in (run / "tools.jsonl").read_text().splitlines()]
    assert tools and tools[0]["tool"] == "Bash" and tools[0]["input"] == {"command": "kl list"}
    sh.rmtree(run)
    config.unlink()


def test_home_binds_is_an_allowlist(tmp_path, caplog):
    """Under the home only the run's dirs and the CLIs' install trees are bound back."""
    home = tmp_path / "home"
    epic = home / ".hmz" / "epics" / "w"
    sign = home / ".claude-x" / ".credentials.json"  # The user's sign-in: never bound (the run has its own)
    conda = home / "miniconda3"
    (conda / "lib").mkdir(parents=True)
    (conda / "lib" / "libnode.so.147").write_text("")
    allowed = [
        home / "run" / "work",
        epic / "s" / "sessions" / "claude",
        home / ".local" / "bin" / "claude",
        home / ".local" / "share" / "claude" / "versions" / "2.1.317",
        home / ".npm-global" / "lib" / "node_modules" / "@anthropic-ai" / "claude-code",
        conda / "bin" / "node",
        Path("/usr"),
    ]
    refused = [
        sign,
        home / ".claude",
        home / ".claude.json",
        home / ".claude-x",
        home / ".claude-x" / "projects",
        home / ".claude-x" / ".claude.json",
        home / ".config" / "anthropic",
        home / ".cache" / "claude-cli-nodejs",
        home / ".local" / "share" / "claude",
        home / ".local" / "share" / "claude" / "versions",
        home / ".codex" / "sessions",
        home / ".hmz" / "epics" / "other",
        home / ".bashrc",
        home / "secret.txt",
    ]
    paths = [str(p) for p in [*allowed, *refused, conda]]
    patterns = isolation.INSTALLS["claude"]
    with caplog.at_level("DEBUG", logger="qda_agent.isolation"):
        dropped: set[str] = set()
        kw = {
            "dirs": [home / "run" / "work", epic],
            "patterns": patterns,
            "dropped": dropped,
        }
        kept = isolation.home_binds(paths, home, **kw)
        isolation.home_binds(paths, home, **kw)  # a second turn logs nothing new
    # conda's node loads libnode.so from its prefix's lib: the prefix is not bound, its lib is
    assert kept == [*(str(p) for p in allowed), str(conda / "lib")]
    assert dropped == {str(p) for p in refused}
    logged = [r.getMessage() for r in caplog.records]
    assert len(logged) == len(refused) and all("not on the home allowlist" in m for m in logged)
    # a run dir that is the home itself allows nothing more
    assert isolation.home_binds(paths, home, dirs=[home, tmp_path]) == ["/usr"]
    # install trees are read-only: a writable list keeps only the run's dirs
    assert isolation.home_binds(paths, home, dirs=kw["dirs"]) == [*map(str, allowed[:2]), "/usr"]


def test_installs_refuses_an_unknown_cli(tmp_path):
    """A backend plugin of a CLI the wall knows no install trees of (or Humanize knows not at all) is refused."""
    from typing import ClassVar, Literal

    from qda_agent.agents import AgentBackend, Codex

    class Qwen(AgentBackend):  # as a third-party backend plugin would be
        cli: ClassVar[str] = "qwen"
        name: Literal["qwen"] = "qwen"

    class Nosuch(AgentBackend):
        cli: ClassVar[str] = "nosuchcli"
        name: Literal["nosuch"] = "nosuch"

    cfg = RunConfig(
        contamination="none", model="m", format="w4a16", budget_gpu_hours=1, cards=[0], run_dir=tmp_path
    )
    assert "bin/claude" in isolation.installs(cfg)
    cfg.agent.fallbacks = [Codex(model="gpt-6.1-sol", effort="low")]
    assert "node_modules/@openai/codex" in isolation.installs(cfg)
    cfg.agent.fallbacks = [Qwen(model="m")]  # a CLI Humanize knows, but not its install trees
    with pytest.raises(ValueError, match="INSTALLS"):
        isolation.installs(cfg)
    cfg.agent.fallbacks = [Nosuch(model="m")]
    with pytest.raises(ValueError, match="knows no agent CLI"):
        isolation.installs(cfg)


FAKE_CLAUDE_STATE = r"""#!/usr/bin/python3
import json, os, sys
if "--version" in sys.argv:
    print("2.1.317 (Claude Code)")
    sys.exit(0)
out = {}
for p in PLANTED + [SIGN_IN]:
    try:
        open(p).read()
        out[p] = "ok"
    except OSError as e:
        out[p] = type(e).__name__
    try:  # what its directory shows
        out[p + " listed"] = os.path.basename(p) in os.listdir(os.path.dirname(p))
    except OSError:
        out[p + " listed"] = False
open(LOG, "a").write(json.dumps(out) + "\n")
sys.exit(3)
"""


@contextmanager
def humanize_lists(paths):
    """Inside `gpu_wall`: every turn's fence lists `paths` too, to read, as if Humanize asked for them. Then
    only the wall's own home allowlist keeps them out (with it switched off, they are readable). (A path listed
    to write as well is dropped from the read list by the `Fence`, and bwrap_argv binds no unkept write.)"""
    hook = isolation.wall_hook()
    walled = hook.wrapper
    extra = tuple(str(p) for p in paths)
    hook.wrapper = lambda f: walled(dataclasses.replace(f, read=(*f.read, *extra)))
    try:
        yield
    finally:
        hook.wrapper = walled


#: The user's CLI state and private files, planted in the (fake) real home: none may be readable inside.
PLANTED = [
    ".claude/projects/x.jsonl",
    ".claude/history.jsonl",
    ".claude/settings.json",
    ".claude.json",
    ".config/anthropic/x",
    ".codex/sessions/x",
    ".codex/history.jsonl",
    ".codex/auth.json",
    ".codex-x/sessions/x",
    ".codex-x/history.jsonl",
    ".hmz/epics/other-workdir/x",
    ".local/share/claude/x",
    "secret.txt",
]


@needs_wall
@pytest.mark.parametrize("ccd", [None, ".claude-x"])
def test_the_users_session_transcripts_are_unreadable_inside(tmp_path, monkeypatch, caplog, ccd):
    """The user's CLI state and files planted anywhere in the real home (outside /tmp, which the
    wall replaces anyway) cannot be read or listed by the CLI inside the wall, with CLAUDE_CONFIG_DIR unset
    and set (the launcher's account); the CLI itself runs from its install tree in that home. The user's
    sign-in files are not readable either: the run signs in with its own copy or a token."""
    import shutil as sh

    from hmz.sdk import Hmz

    base = Path(__file__).parents[1] / "runs" / f"n2-{ccd or 'unset'}"  # not under /tmp
    sh.rmtree(base, ignore_errors=True)
    home = base / "home"
    planted = [home / p for p in PLANTED]
    config = home / (ccd or ".claude")
    if ccd:
        planted += [config / "projects" / "x.jsonl", config / "history.jsonl", config / ".claude.json"]
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    else:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    for p in planted:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("a transcript of another session")
    sign_in = config / ".credentials.json"
    sign_in.write_text("{}")
    run = base / "run"
    work = run / "work"
    work.mkdir(parents=True)
    bindir = home / ".local" / "bin"  # an install tree INSTALLS names
    bindir.mkdir(parents=True)
    log = work / "STATELOG"
    fake = (
        FAKE_CLAUDE_STATE.replace("PLANTED", repr([str(p) for p in planted]))
        .replace("SIGN_IN", repr(str(sign_in)))
        .replace("LOG", repr(str(log)))
    )
    (bindir / "claude").write_text(fake)
    (bindir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    (run / "probe_flow.py").write_text(PROBE_FLOW)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    with (
        caplog.at_level("DEBUG", logger="qda_agent.isolation"),
        isolation.gpu_wall(cfg),
        humanize_lists([*planted, *{p.parent for p in planted}]),
    ):
        r = Hmz(workspace=work).run(
            f"{run}/probe_flow.py:probe",
            "hi",
            agents={"worker": "claude/claude-haiku-5-5:low"},
            envs={"workspace": f"local{work}"},
            budget={"duration": 60},
        )
        with pytest.raises(Exception):  # noqa: B017 -- the fake CLI fails every turn after recording it
            r.run()
    seen = [json.loads(x) for x in log.read_text().splitlines()]
    assert seen, "the fake CLI never took a turn"
    for s in seen:
        assert all(v is False or v not in ("ok", True) for v in s.values()), s  # neither readable nor listed
    dropped = [r.getMessage() for r in caplog.records if "not on the home allowlist" in r.getMessage()]
    assert any(f"{home / '.claude.json'} is" in m for m in dropped), dropped  # Humanize asked; dropped
    assert len(dropped) == len(set(dropped)), dropped  # once each, however many turns
    sh.rmtree(base)


def test_a_launcher_symlink_is_made_again_as_a_link(tmp_path):
    """A launcher symlink under a hidden dir is made again as a link, not bound: a bind would put its target's
    content at the link's path (Node then looks for the package next to `bin/`)."""
    home = tmp_path / "home"
    pkg = home / "conda" / "lib" / "node_modules" / "@openai" / "codex" / "bin"
    pkg.mkdir(parents=True)
    (pkg / "codex.js").write_text("")
    link = home / "conda" / "bin" / "codex"
    link.parent.mkdir()
    link.symlink_to("../lib/node_modules/@openai/codex/bin/codex.js")
    argv = isolation.bwrap_argv([str(link), str(pkg)], [], [], hide_early=[home])
    k = argv.index("--symlink")
    assert argv[k : k + 3] == ["--symlink", "../lib/node_modules/@openai/codex/bin/codex.js", str(link)]
    assert argv.count(str(link)) == 1  # not bound as well


FAKE_CODEX = r"""#!/usr/bin/python3
import json, os, sys
if "--version" in sys.argv:
    print("codex-cli 0.149.0")
    sys.exit(0)
if sys.argv[1:2] != ["app-server"]:  # Humanize's own probe on the host (`codex features list`), not a turn
    sys.exit(0)
out = {"launcher is a link": os.path.islink(LAUNCHER)}
for p in PLANTED:
    try:
        open(p).read()
        out[p] = "ok"
    except OSError as e:
        out[p] = type(e).__name__
try:  # its config dir is writable (Humanize's per-session dir, linked into the run's own)
    open(os.path.join(os.environ["CODEX_HOME"], "logs_2.sqlite"), "a").write("x")
    out["state write"] = "ok"
except (OSError, KeyError) as e:
    out["state write"] = type(e).__name__
open(LOG, "a").write(json.dumps(out) + "\n")
sys.exit(3)
"""


@needs_wall
def test_a_codex_turn_inside_the_wall(monkeypatch):
    """A real Humanize spawn of a fake codex, started through an npm-style launcher symlink in a home outside
    /tmp: the launcher stays a link, the user's Codex state (`~/.codex`, `~/.codex-x`) is unreadable, and its
    config dir is writable and none of it lands in the user's dir."""
    import shutil as sh

    from hmz.sdk import Hmz

    base = Path(__file__).parents[1] / "runs" / "cx"  # not under /tmp
    sh.rmtree(base, ignore_errors=True)
    home = base / "home"
    planted = [home / p for p in PLANTED]
    for p in planted:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("a rollout of another session")
    run = base / "run"
    work = run / "work"
    work.mkdir(parents=True)
    log = work / "CODEXLOG"
    pkg = home / ".npm-global" / "lib" / "node_modules" / "@openai" / "codex" / "bin"
    pkg.mkdir(parents=True)
    launcher = home / ".npm-global" / "bin" / "codex"
    launcher.parent.mkdir(parents=True)
    fake = (
        FAKE_CODEX.replace("PLANTED", repr([str(p) for p in planted]))
        .replace("LAUNCHER", repr(str(launcher)))
        .replace("LOG", repr(str(log)))
    )
    (pkg / "codex.js").write_text(fake)
    (pkg / "codex.js").chmod(0o755)
    launcher.symlink_to("../lib/node_modules/@openai/codex/bin/codex.js")
    monkeypatch.setenv("PATH", f"{launcher.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    (run / "probe_flow.py").write_text(PROBE_FLOW)
    spec = "codex/gpt-6.1-sol:low"
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        agent=agent(spec),
    )
    with isolation.gpu_wall(cfg), humanize_lists([*planted, *{p.parent for p in planted}]):
        r = Hmz(workspace=work).run(
            f"{run}/probe_flow.py:probe",
            "hi",
            agents={"worker": spec},
            envs={"workspace": f"local{work}"},
            budget={"duration": 60},
        )
        with pytest.raises(Exception):  # noqa: B017 -- the fake CLI fails every turn after recording it
            r.run()
    seen = [json.loads(x) for x in log.read_text().splitlines()]
    assert seen, "the fake codex never took a turn"

    for s in seen:
        assert s.pop("launcher is a link") is True, s
        assert s.pop("state write") == "ok", s
        assert all(v != "ok" for v in s.values()), s  # no planted file readable, ~/.codex/auth.json included
    assert not (home / ".codex" / "logs_2.sqlite").exists()  # nothing of the run in the user's dir
    sh.rmtree(base)


FAKE_WRITE_BACK = r"""#!/usr/bin/python3
import json, os, sys
if "--version" in sys.argv:
    print(VERSION)
    sys.exit(0)
if IS_CODEX and sys.argv[1:2] != ["app-server"]:  # Humanize's own probe on the host, not a turn
    sys.exit(0)
seen = []  # every file of the user's CLI dirs in the agent's whole view (by its planted name)
for top, dirs, files in os.walk("/", onerror=lambda e: None):
    if top in ("/proc", "/sys", "/dev"):
        dirs[:] = []
        continue
    seen += [os.path.join(top, f) for f in files if f.startswith("qda-planted-")]
d = os.environ["CODEX_HOME" if IS_CODEX else "CLAUDE_CONFIG_DIR"]
other = os.environ.get("CLAUDE_CONFIG_DIR" if IS_CODEX else "CODEX_HOME")
done = []
for n in sorted(os.listdir(d)):  # replace every link of the session's config dir with the agent's own
    f = os.path.join(d, n)
    if not os.path.islink(f):
        continue
    os.unlink(f)
    open(f, "w").write("PWNED")
    done.append(n)
for n in ("CLAUDE.md", "AGENTS.md", "history.jsonl", "commands", "prompts"):  # and write its own
    f = os.path.join(d, n)
    if n in ("commands", "prompts"):
        os.makedirs(f, exist_ok=True)
        f = os.path.join(f, "evil.md")
    open(f, "w").write("PWNED")
open(LOG, "a").write(json.dumps({"seen": seen, "replaced": done, "dir": d, "other": other}) + "\n")
sys.exit(3)
"""

#: A sign-in's content: no file of the run may hold it, under any name (a copy, a write-back, a log).
SECRET = "qda-planted-sign-in-7f3c"

#: The user's CLI dirs planted in the (fake) real home: what a person, other sessions and the launcher keep.
CLI_DIRS = (".claude", ".claude-a", ".claude-launcher", ".codex", ".codex-a", ".codex-launcher")


@needs_wall
@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_the_agent_sees_and_changes_nothing_of_the_users_cli_dirs(tmp_path, monkeypatch, cli):
    """Launched with CLAUDE_CONFIG_DIR / CODEX_HOME pointing at a planted dir, the agent's
    whole view (a walk of `/`) holds no file of any real `~/.claude*` or `~/.codex*` dir; it replaces every link of
    its config dir and writes CLAUDE.md, AGENTS.md, commands, prompts and history there; afterwards every real dir
    is byte-identical to a snapshot taken before the run. The CLI signs in by the planted file (token variables
    unset: with one set, no sign-in is made and the credential checks below hold vacuously); the run's template dir
    holds only a link into the empty host side of the sign-in mount, and no file of the run holds the sign-in."""

    from hmz.sdk import Hmz

    base = tmp_path / f"wb-{cli}"
    home = base / "home"
    for d in CLI_DIRS:
        for n in (
            "CLAUDE.md",
            "AGENTS.md",
            "history.jsonl",
            "commands/ok.md",
            "prompts/ok.md",
            "projects/p/x.jsonl",
        ):
            f = home / d / n
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"ORIGINAL {d}/{n}")
            (f.parent / f"qda-planted-{n.replace('/', '-')}").write_text("planted")
        (home / d / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": SECRET, "expiresAt": 4102444800000}})
        )
        (home / d / "auth.json").write_text(json.dumps({"tokens": {"access_token": SECRET}}))
    (home / ".claude.json").write_text("{}")

    def snapshot() -> dict:
        return {f: f.read_bytes() for d in CLI_DIRS for f in (home / d).rglob("*") if f.is_file()} | {
            home / ".claude.json": (home / ".claude.json").read_bytes()
        }

    before = snapshot()
    run = base / "run"
    work = run / "work"
    work.mkdir(parents=True)
    log = work / "WBLOG"
    bindir = home / ".npm-global" / "bin"  # an install tree INSTALLS names, for both CLIs
    bindir.mkdir(parents=True)
    version = "2.1.317 (Claude Code)" if cli == "claude" else "codex-cli 0.149.0"
    fake = (
        FAKE_WRITE_BACK.replace("VERSION", repr(version))
        .replace("IS_CODEX", repr(cli == "codex"))
        .replace("LOG", repr(str(log)))
    )
    (bindir / cli).write_text(fake)
    (bindir / cli).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude-launcher"))  # as from inside an agent session
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex-launcher"))
    for name in CLI_CONFIG:  # with a token variable set, the CLI never signs in by the file
        for var in CLI_CONFIG[name][3]:
            monkeypatch.delenv(var, raising=False)
    (run / "probe_flow.py").write_text(PROBE_FLOW)
    spec = "claude/claude-haiku-5-5:low" if cli == "claude" else "codex/gpt-6.1-sol:low"
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        agent=agent(spec),
    )
    assert cfg.sandbox.name == "bwrap", cfg.sandbox  # the link checks below are the bwrap wall's sign-in
    assert isolation.sign_in_accounts(cfg)[cli] == home / f".{cli}"  # signs in by the planted file
    with isolation.gpu_wall(cfg):
        r = Hmz(workspace=work).run(
            f"{run}/probe_flow.py:probe",
            "hi",
            agents={"worker": spec},
            envs={"workspace": f"local{work}"},
            budget={"duration": 60},
        )
        with pytest.raises(Exception):  # noqa: B017 -- the fake CLI fails every turn after its writes
            r.run()
    turns = [json.loads(x) for x in log.read_text().splitlines()]
    assert turns, "the fake CLI never took a turn"
    for t in turns:
        assert t["seen"] == [], t["seen"]  # nothing of any real ~/.claude* or ~/.codex* dir in the whole view
        assert not t["dir"].startswith(str(home)), t  # its config dir is Humanize's per-session one
        assert t["other"] is None, t  # the launcher's dir of the other CLI is not passed on
    assert snapshot() == before  # byte-identical: nothing written back, nothing added
    assert os.environ["CLAUDE_CONFIG_DIR"] == str(home / ".claude-launcher")  # restored after the run
    assert (run / "cli" / cli).is_dir()  # the run's own template dir
    sign_in = Path(CLI_CONFIG[cli][2]).name
    mount = run / isolation.SIGNIN / cli
    link = run / "cli" / cli / sign_in  # the template dir's (the live dir's is the agent's to replace)
    assert link.is_symlink(), link  # the sign-in path was taken: a link, never a copy
    assert (link.parent / link.readlink()).resolve() == (mount / sign_in).resolve(), link  # into the mount
    assert not any(mount.iterdir())  # the mount's host side is empty: the file is bound only inside a turn
    leaked = [f for f in run.rglob("*") if f.is_file() and SECRET in f.read_text(errors="replace")]
    assert not leaked, leaked  # no byte of the sign-in anywhere in the run, under any name


def test_home_binds_a_symlinked_home_and_a_python_prefix(tmp_path):
    """A home reached through a link is matched in both spellings. Of a Python prefix in the home
    (`~/.local`) only its bin, lib and pyvenv.cfg come back, never `share/` or the rest."""
    real = tmp_path / "real-home"
    home = tmp_path / "home"
    home.symlink_to(real)
    local = real / ".local"
    for d in ("bin", "lib/python3.12", "share/claude"):
        (local / d).mkdir(parents=True)
    paths = [
        str(real / ".claude"),  # the resolved spelling of a state dir
        str(home / ".claude.json"),
        str(real / "secret.txt"),
        str(local),  # Humanize lists the interpreter's prefix whole
        str(local / "lib" / "python3.12"),
        str(local / "share" / "claude"),
        str(home / ".local" / "bin" / "claude"),
    ]
    narrow = [home / ".local" / p for p in isolation.PYTHON_PARTS]
    dropped: set[str] = set()
    kept = isolation.home_binds(
        paths, home, narrow=narrow, patterns=isolation.INSTALLS["claude"], dropped=dropped
    )
    assert kept == [str(local / "bin"), str(local / "lib"), str(local / "lib" / "python3.12"), paths[-1]]
    assert dropped == {*paths[:3], str(local / "share" / "claude")}


@pytest.mark.skipif(not isolation.shutil.which("bwrap"), reason="needs bwrap")
def test_a_symlinked_cli_under_a_bound_dir_runs(tmp_path):
    """A CLI launcher that is a link inside a directory already mounted read-only (the venv that runs the fence)
    is left as that mount shows it: bwrap cannot make a mount point inside a read-only bind."""
    home = Path(__file__).parents[1] / "runs" / "symcli"  # not under /tmp
    import shutil as sh

    sh.rmtree(home, ignore_errors=True)
    venv = home / "venv"
    pkg = venv / "node_modules" / "@openai" / "codex" / "bin"
    pkg.mkdir(parents=True)
    (pkg / "codex.py").write_text("#!/usr/bin/python3\nprint('ran')\n")
    (pkg / "codex.py").chmod(0o755)
    (venv / "bin").mkdir()
    launcher = venv / "bin" / "codex"
    launcher.symlink_to("../node_modules/@openai/codex/bin/codex.py")
    other = home / ".local" / "bin" / "codex"  # a link of its own under the hidden home: made again as a link
    other.parent.mkdir(parents=True)
    other.symlink_to(launcher)
    argv = isolation.bwrap_argv(
        [str(venv), str(launcher), str(pkg / "codex.py"), str(other)], [], [], hide_early=[home]
    )
    dests = {argv[k + 2]: x for k, x in enumerate(argv[:-2]) if x in ("--ro-bind", "--bind", "--symlink")}
    assert str(launcher) not in dests and dests[str(other)] == "--symlink"  # inside the venv's bind: as it is
    for cmd in (launcher, other):
        p = subprocess.run([*argv, str(cmd)], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0 and p.stdout.strip() == "ran", p.stderr
    sh.rmtree(home)


TWO_TURNS_FLOW = PROBE_FLOW.replace(
    '    return await w.run(task, session=await w.spawn(), env=envs["workspace"])',
    """    out = []
    for i in range(2):  # two turns, each a fresh session: Humanize builds and drops a home per turn
        try:
            out.append(await w.run(f"{task} {i}", session=await w.spawn(), env=envs["workspace"]))
        except Exception as e:  # the fake CLI exits 3
            out.append(type(e).__name__)
    return str(out)""",
)

#: A fake CLI that attacks Humanize's host-side home build and drop: in turn 1 it plants links to the user's
#: real files in every dir the host builds a home from or drops one into (the run's template CLI dir, the
#: session homes, the epic's session store), plants the same links in its own config dir, and leaves a
#: process behind; in turn 2 it replaces every link of its config dir with a file of its own, and looks for
#: the user's secret in its view.
FAKE_LINKS = r"""#!/usr/bin/python3
import glob, json, os, subprocess, sys
if "--version" in sys.argv:
    print(VERSION)
    sys.exit(0)
if IS_CODEX and sys.argv[1:2] != ["app-server"]:
    sys.exit(0)
d = os.environ["CODEX_HOME" if IS_CODEX else "CLAUDE_CONFIG_DIR"]
first = not os.path.exists(MARK)
out = {"turn": 1 if first else 2, "dir": d, "planted": {}, "secret": []}

def plant(where, name, to):
    f = os.path.join(where, name)
    try:
        if os.path.isdir(f) and not os.path.islink(f):
            os.rename(f, f + ".moved")
        elif os.path.lexists(f):
            os.unlink(f)
        os.symlink(to, f)
        return "planted"
    except OSError as e:
        return type(e).__name__

if first:
    homes = [h for h in glob.glob("/tmp/hmz-home-*/hmz-home-*") + glob.glob(TMP + "/hmz-home-*/hmz-home-*")]
    for where in [RUNCLI, EPIC, *homes, d]:
        for name, to in ((NOTE, os.path.join(REAL, NOTE)), ("history.jsonl", os.path.join(REAL, "history.jsonl")),
                         (CMDS, os.path.join(REAL, CMDS)), (SETTINGS, SECRET)):
            out["planted"][where + "/" + name] = plant(where, name, to)
    bg = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", BG], start_new_session=True,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import time
    time.sleep(0.3)
    out["background"] = bg.poll()  # None: still running as the turn ends
    open(MARK, "w").write("1")
else:  # replace every link of its config dir with its own file: a drop writes such files back
    for n in sorted(os.listdir(d)):
        f = os.path.join(d, n)
        if os.path.islink(f):
            os.unlink(f)
            if n == CMDS:
                os.mkdir(f)
                f = os.path.join(f, "evil.md")
            open(f, "w").write("PWNED")
            out["planted"][f] = "replaced"
for top, dirs, files in os.walk(d):
    for f in files:
        try:
            if SECRET_TEXT in open(os.path.join(top, f), errors="ignore").read():
                out["secret"].append(os.path.join(top, f))
        except OSError:
            pass
open(LOG, "a").write(json.dumps(out) + "\n")
sys.exit(3)
"""


def _alive(marker: str) -> list[str]:
    """Pids of processes whose command line holds `marker`."""
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if marker.encode() in (p / "cmdline").read_bytes():
                out.append(p.name)
        except OSError:
            continue
    return out


@needs_wall
@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_humanizes_home_build_and_drop_see_nothing_the_agent_wrote(tmp_path, monkeypatch, cli):
    """Over two turns the agent plants links to the user's real files in the run's template CLI dir, in
    Humanize's session homes, in the epic's session store and in its own config dir, and leaves a process
    behind. Humanize builds and drops each home on the host, following links. Afterwards the user's dirs are
    byte-identical, no turn saw the user's secret, every dir the host works on refused the links (read-only
    inside), and no process of turn 1 was alive when the host dropped a home."""
    from hmz.coganchor import homes
    from hmz.sdk import Hmz

    base = tmp_path / "f1b"
    home = base / "home"
    real = home / (".claude" if cli == "claude" else ".codex")
    note, cmds, settings = (
        ("CLAUDE.md", "commands", "settings.json")
        if cli == "claude"
        else (
            "AGENTS.md",
            "prompts",
            "config.toml",
        )
    )
    for d in (real, home / f"{real.name}-launcher"):
        for n in (note, "history.jsonl", f"{cmds}/ok.md"):
            (d / n).parent.mkdir(parents=True, exist_ok=True)
            (d / n).write_text(f"ORIGINAL {d.name}/{n}")
            os.utime(d / n, (1, 1))  # old, so any write-back would count as newer
    secret_text = "qda-f1b-private-value"
    (home / "secret.json").write_text(json.dumps({"secret": secret_text}))
    (real / ".credentials.json").write_text('{"who": "DEFAULT"}')
    (real / "auth.json").write_text('{"who": "DEFAULT"}')

    def snapshot() -> dict:
        return {
            f: f.read_bytes()
            for f in home.rglob("*")
            if f.is_file() and not {".npm-global", ".hmz"} & set(f.parts)
        }

    run = base / "run"
    work = run / "work"
    work.mkdir(parents=True)
    log, mark, bg = work / "F1BLOG", work / "TURN1", str(tmp_path / f"background-{cli}")
    bindir = home / ".npm-global" / "bin"
    bindir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    epic = isolation.agent_state(work)
    fake = FAKE_LINKS
    for k, v in {
        "VERSION": "2.1.317 (Claude Code)" if cli == "claude" else "codex-cli 0.149.0",
        "IS_CODEX": cli == "codex",
        "MARK": str(mark),
        "LOG": str(log),
        "RUNCLI": str(run / "cli" / cli),
        "EPIC": str(epic),
        "TMP": tempfile.gettempdir(),
        "REAL": str(real),
        "SECRET_TEXT": secret_text,
        "SECRET": str(home / "secret.json"),
        "NOTE": note,
        "CMDS": cmds,
        "SETTINGS": settings,
        "BG": bg,
    }.items():
        fake = fake.replace(k, repr(v))
    (bindir / cli).write_text(fake)
    (bindir / cli).chmod(0o755)
    before = snapshot()
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude-launcher"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex-launcher"))
    probe = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", bg])  # `_alive` sees one
    try:
        assert _alive(bg) == [str(probe.pid)]
    finally:
        probe.kill()
        probe.wait()
    survivors: list[list[str]] = []
    drop = homes.Home.drop

    def watched_drop(self: homes.Home) -> None:
        survivors.append(_alive(bg))  # the host's turn-end work starts here
        drop(self)

    monkeypatch.setattr(homes.Home, "drop", watched_drop)
    (run / "flow.py").write_text(TWO_TURNS_FLOW)
    spec = "claude/claude-haiku-5-5:low" if cli == "claude" else "codex/gpt-6.1-sol:low"
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
        agent=agent(spec),
    )
    try:
        with isolation.gpu_wall(cfg):
            Hmz(workspace=work).run(
                f"{run}/flow.py:probe",
                "hi",
                agents={"worker": spec},
                envs={"workspace": f"local{work}"},
                budget={"duration": 90},
            ).run()
    finally:
        left = _alive(bg)
        for pid in left:
            with contextlib.suppress(OSError):
                assert_job_process(int(pid), tmp_path, Path(bg))
                os.kill(int(pid), 9)
    turns = [json.loads(x) for x in log.read_text().splitlines()]
    assert [t["turn"] for t in turns] == [1, 2], turns
    assert snapshot() == before  # the user's dirs and the secret: byte-identical, nothing added
    for t in turns:
        assert t["secret"] == [], t  # no file of its config dir holds the secret, in either turn
    planted = turns[0]["planted"]
    hosted = {k: v for k, v in planted.items() if not k.startswith(str(run / "live"))}
    assert any(k.startswith(str(run / "cli")) for k in hosted) and any("hmz-home-" in k for k in hosted)
    assert all(v != "planted" for v in hosted.values()), hosted  # read-only inside, every one
    assert all(v == "planted" for k, v in planted.items() if k not in hosted)  # its own dir is its own
    assert turns[0]["background"] is None  # turn 1 left a process running
    assert survivors and all(s == [] for s in survivors), survivors  # nothing of turn 1 outlived it
    assert left == []
    assert all(t["dir"] == str(run / "live" / cli) for t in turns), turns  # the live dir, every turn
    for f in run.rglob("*"):  # nothing the host keeps holds the secret
        if f.is_file() and not f.is_symlink() and not f.is_relative_to(run / "live"):
            assert secret_text not in f.read_text(errors="ignore"), f


def test_home_guard_refuses_a_link_out_of_the_run(tmp_path, monkeypatch):
    """The extra layer: a link out of the run in a dir Humanize builds homes from stops the build and the drop."""
    from hmz.coganchor import homes

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    run = tmp_path / "run"
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox="bwrap",
    )
    secret = tmp_path / "home" / "secret.json"
    secret.parent.mkdir()
    secret.write_text("{}")
    calls = []
    monkeypatch.setattr(homes, "build", lambda *a, **k: calls.append("build"))
    monkeypatch.setattr(homes.Home, "drop", lambda self: calls.append("drop"))
    with isolation.private_cli_config(cfg), wall._home_guard(cfg):
        homes.build()  # clean: passes through
        (run / "cli" / "claude" / "inside").symlink_to(run / "cli" / "claude" / "settings.json")
        homes.build()  # a link inside the run is fine
        (run / "cli" / "claude" / "CLAUDE.md").symlink_to(secret)
        with pytest.raises(isolation.UnsafeHome, match="CLAUDE.md"):
            homes.build()
        with pytest.raises(isolation.UnsafeHome):
            homes.Home.drop(object())
        (run / "cli" / "claude" / "CLAUDE.md").unlink()
        os.mkfifo(run / "cli" / "claude" / "pipe")
        assert isolation.unsafe_entries(run / "cli", [run]) == [f"{run}/cli/claude/pipe: not a file or a dir"]
    assert calls == ["build", "build"]


@pytest.fixture
def scratch():
    """`scratch(dir)`: a new `qda-test-*` directory in `dir` (outside pytest's tmp), removed after the test."""
    made: list[Path] = []

    def make(parent: str) -> Path:
        made.append(Path(tempfile.mkdtemp(prefix="qda-test-", dir=parent)))
        return made[-1]

    yield make
    for d in made:
        shutil.rmtree(d, ignore_errors=True)


def _xdg(kind: str, scratch) -> str | None:
    """A `$XDG_RUNTIME_DIR` of each kind the fence could reach."""
    run_user = f"/run/user/{os.getuid()}"
    if kind == "unset":
        return None
    if kind == "tmp":
        return str(scratch("/tmp"))
    if kind == "shm":
        return str(scratch("/dev/shm"))
    if kind == "link":  # a link in /tmp to the real runtime dir: the agent could replace the link
        link = scratch("/tmp") / "lnk"
        link.symlink_to(run_user)
        return str(link)
    return run_user


@pytest.mark.skipif(not isolation.shutil.which("bwrap"), reason="needs bwrap")
@pytest.mark.parametrize("deep", [True, False])
def test_bwrap_refuses_a_private_socket_base_the_wall_binds_writable(tmp_path, monkeypatch, deep):
    """The fence grants `$TMPDIR`; one outside the temporary dirs and the home is bound writable in the wall. A
    `$XDG_RUNTIME_DIR` under it is refused (statically, from the real fence); a hidden one is allowed. A run dir
    short enough for the socket is held to it too: the judge's jobs name their temp dirs by links made there."""
    import test_components as tc

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    tmpdir = tc.ROOT / "runs" / "tdir"  # not a temporary dir, not under the (stand-in) home
    (tmpdir / "xdg").mkdir(parents=True, exist_ok=True)
    try:
        monkeypatch.setenv("TMPDIR", str(tmpdir))
        monkeypatch.setattr(tempfile, "tempdir", None)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmpdir / "xdg"))
        cfg = tc._inputs(tmp_path, run_dir=tc.ROOT / "runs" / ("r" * (110 if deep else 1)), sandbox="bwrap")
        assert cfg.needs_private_socket is deep
        with pytest.raises(ValueError, match=f"which the agent can write \\(through {tmpdir}\\)"):
            isolation.check(cfg)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(home))  # hidden in the wall
        isolation.check(cfg)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_exposed_compares_both_spellings(tmp_path):
    real = tmp_path / "real"
    (real / "base").mkdir(parents=True)
    (tmp_path / "lnk").symlink_to(real)
    argv = ["bwrap", "--tmpfs", str(tmp_path), "--bind", str(tmp_path / "lnk"), str(tmp_path / "lnk"), "--"]
    assert wall.exposed(argv, real / "base") == str(
        tmp_path / "lnk"
    )  # the bind is a link to the base's parent
    argv = ["bwrap", "--bind", str(real), str(real), "--"]
    assert wall.exposed(argv, tmp_path / "lnk" / "base") == str(real)  # the base is named through a link
    assert wall.exposed(["bwrap", "--ro-bind", str(real), str(real), "--"], real / "base") is None


# --- `sandbox.hide`: a private path must stay hidden wherever the turn reaches it ---------------------

_SYSTEM = ["/usr", "/lib", "/lib64", "/bin"]

_PRIVATE_READ = """
import json, os, sys
out = {}
for p in sys.argv[1:]:
    try:
        s = os.stat(p)
        out[p] = {"read": open(p).read(), "dev": s.st_dev, "ino": s.st_ino}
    except OSError as e:
        out[p] = {"error": type(e).__name__}
print(json.dumps(out))
"""


def _private_wall(tmp_path: Path, hidden, read=(), binds=None):
    """A turn's mounts over a fake private tree; returns the argv and the paths to read inside."""
    resolved = tmp_path / "resolved"
    private = resolved / "private"
    private.mkdir(parents=True, exist_ok=True)
    marker = private / "heldout.txt"
    marker.write_text("SYNTHETIC PRIVATE\n")
    public = resolved / "public.txt"
    public.write_text("public control\n")
    alias = tmp_path / "alias"
    if not alias.is_symlink():
        alias.symlink_to(resolved, target_is_directory=True)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    sentinel = work / "control.txt"
    sentinel.write_text("work control\n")
    argv = isolation.bwrap_argv(
        [*_SYSTEM, *read],
        [str(work)],
        hidden,
        keep=[work],
        binds=binds or {},
        minimal=True,
        links={str(alias): str(resolved)},
    )
    return argv, (marker, alias / "private" / "heldout.txt", public, sentinel)


def _refuses_when_shown(argv, hidden):
    with pytest.raises(wall.WallRefused, match="still visible"):
        wall._verify_hidden(argv, wall._hidden_specs(hidden))


def _read_inside(argv, paths):
    got = subprocess.run(
        [*argv, "/usr/bin/python3", "-I", "-S", "-c", _PRIVATE_READ, *map(str, paths)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


needs_bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bwrap")


@needs_bwrap
@pytest.mark.parametrize("spelling", ["alias", "resolved"])
def test_a_private_dir_is_hidden_at_both_spellings_after_a_later_read_bind(tmp_path, spelling):
    """A holdout stayed readable at its resolved spelling after a bind of its parent mounted over the
    early hide tmpfs. Both spellings are checked, and the configured spelling alone must suffice."""
    resolved = tmp_path / "resolved"
    hidden = tmp_path / "alias" / "private" if spelling == "alias" else resolved / "private"
    argv, (marker, alias_marker, public, sentinel) = _private_wall(tmp_path, [hidden], read=[str(resolved)])
    seen = _read_inside(argv, [marker, alias_marker, public, sentinel])
    assert "error" in seen[str(marker)] and "error" in seen[str(alias_marker)], seen
    assert seen[str(public)]["read"] == "public control\n" and seen[str(sentinel)]["read"] == "work control\n"


@needs_bwrap
def test_a_private_dir_is_hidden_after_a_later_explicit_parent_bind(tmp_path):
    resolved = tmp_path / "resolved"
    argv, (marker, alias_marker, public, sentinel) = _private_wall(
        tmp_path, [resolved / "private"], binds={resolved: resolved}
    )
    seen = _read_inside(argv, [marker, alias_marker, public, sentinel])
    assert "error" in seen[str(marker)] and "error" in seen[str(alias_marker)], seen
    assert seen[str(public)]["read"] == "public control\n"


@needs_bwrap
def test_a_private_file_is_masked_by_dev_null_after_a_later_read_bind(tmp_path):
    resolved = tmp_path / "resolved"
    marker = resolved / "private" / "heldout.txt"
    argv, (host_marker, alias_marker, public, sentinel) = _private_wall(
        tmp_path, [marker], read=[str(resolved)]
    )
    seen = _read_inside(argv, [host_marker, alias_marker, public, sentinel])
    host = marker.stat()
    for p in (host_marker, alias_marker):
        got = seen[str(p)]
        assert got.get("read") != "SYNTHETIC PRIVATE\n"  # not the host's file, at either spelling
        if "dev" in got:  # `/dev/null` on the wall's nodev root refuses the open; elsewhere it reads empty
            assert (got["dev"], got["ino"]) != (host.st_dev, host.st_ino), seen
    assert seen[str(public)]["read"] == "public control\n"


@needs_bwrap
def test_a_private_file_inside_a_bound_directory_is_masked(tmp_path):
    """A holdout inside an otherwise bound tree (a copy of `qda_kl/data/gencheck-prompts.jsonl` under the
    workdir): the workdir is bound read/write, so the file's mask must come after that bind, its siblings
    stay readable, and the self-check reports the file while a bind still shows it."""
    work = tmp_path / "work"
    data = work / "src" / "qda_kl" / "data"
    data.mkdir(parents=True)
    marker = data / "gencheck-prompts.jsonl"
    marker.write_text("SYNTHETIC PRIVATE\n")
    control = data / "public.jsonl"
    control.write_text("public control\n")
    argv = isolation.bwrap_argv(_SYSTEM, [str(work)], [marker], keep=[work], minimal=True)
    seen = _read_inside(argv, [marker, control])
    got = seen[str(marker)]
    assert got.get("read") != "SYNTHETIC PRIVATE\n", seen
    if "dev" in got:  # `/dev/null` on the wall's nodev root refuses the open; elsewhere it reads empty
        host = marker.stat()
        assert (got["dev"], got["ino"]) != (host.st_dev, host.st_ino), seen
    assert seen[str(control)]["read"] == "public control\n"
    _refuses_when_shown(isolation.bwrap_argv(_SYSTEM, [str(work)], [], keep=[work], minimal=True), [marker])


def test_a_hide_that_covers_a_path_the_turn_needs_is_refused(tmp_path):
    """A hide that is a parent of the run dir (or of the workdir, a CLI dir, a readable bind) must be a
    refusal naming both paths: a mask is mounted last, so it would silently take the run dir with it."""
    tree = tmp_path / "tree"
    run = tree / "run"
    work = run / "work"
    work.mkdir(parents=True)
    readable = tmp_path / "data"
    readable.mkdir()
    assert wall.hidden_covers([tree], [run, work, readable]) == (tree, run)  # both paths, not a silent mask
    assert wall.hidden_covers([run], [run]) == (run, run)  # the run dir itself is not a narrower hide
    assert wall.hidden_covers([readable / "slice"], [run, work, readable]) is None  # a child covers nothing


@needs_bwrap
def test_the_wall_refuses_a_hide_that_covers_the_run_dir(tmp_path):
    """The refusal is raised by the wall at start, and its message names the hide and the covered path."""
    from hmz.coganchor import fence as hmz_fence

    tree = tmp_path / "tree"
    run = tree / "run"
    work = run / "work"
    work.mkdir(parents=True)
    cfg = RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox={"name": "bwrap", "hide": [str(tree)]},
    )
    f = hmz_fence.Fence.of(
        local="all",
        user="none",
        system="none",
        online=False,
        workdir=work,
        home=Path.home(),
        hosts=["cloud.infini-ai.com"],
    )
    with isolation.gpu_wall(cfg), pytest.raises(isolation.WallRefused, match="covers") as err:
        hmz_fence.wrapper(f)
    assert str(tree) in str(err.value) and str(run) in str(err.value), err.value


@needs_bwrap
def test_a_private_pool_slice_directory_is_hidden_after_a_later_read_bind(tmp_path):
    """A hidden calibration slice of the pool (a directory with its own tree) stays absent at every path
    while the pool root is read-bound; a sibling of the slice stays readable, and the self-check reports
    the slice."""
    pool = tmp_path / "pool"
    slice_ = pool / "if-heldout"
    (slice_ / "inner").mkdir(parents=True)
    (slice_ / "prompts.jsonl").write_text("SYNTHETIC PRIVATE\n")
    (slice_ / "inner" / "trace.jsonl").write_text("SYNTHETIC PRIVATE\n")
    control = pool / "public.txt"
    control.write_text("public control\n")
    work = tmp_path / "work"
    work.mkdir()
    argv = isolation.bwrap_argv([*_SYSTEM, str(pool)], [str(work)], [slice_], keep=[work], minimal=True)
    paths = [slice_, slice_ / "prompts.jsonl", slice_ / "inner", slice_ / "inner" / "trace.jsonl", control]
    seen = _read_inside(argv, paths)
    for p in paths[:-1]:
        assert "error" in seen[str(p)], seen
    assert seen[str(control)]["read"] == "public control\n"
    shown = isolation.bwrap_argv([*_SYSTEM, str(pool)], [str(work)], [], keep=[work], minimal=True)
    _refuses_when_shown(shown, [slice_])


@needs_bwrap
def test_the_hidden_path_self_check_refuses_a_path_a_bind_shows(tmp_path):
    """The wall's own pre-start check (`_verify_hidden`): a path a bind still shows is a refusal, a masked
    one passes. Both kinds are covered, a directory by an empty tmpfs and a file by `/dev/null`."""
    resolved = tmp_path / "resolved"
    marker = resolved / "private" / "heldout.txt"
    shown, _ = _private_wall(tmp_path, [], read=[str(resolved)])
    _refuses_when_shown(shown, [marker])
    for hidden in ([resolved / "private"], [marker]):
        masked, _ = _private_wall(tmp_path, hidden, read=[str(resolved)])
        wall._verify_hidden(masked, wall._hidden_specs(hidden))  # no refusal: masked


def test_the_walls_dev_is_mounted_before_its_binds(tmp_path):
    """`--dev /dev` mounted after the binds covered every path bound under /dev/shm (a run dir, a home, a
    $TMPDIR); it comes first now, a path under /dev/shm is bound, and /dev/shm itself is never the host's."""
    from qda_agent.isolation import wall

    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("no writable /dev/shm")
    inside = Path(tempfile.mkdtemp(prefix="qda-wall-", dir=shm))
    try:
        (inside / "seen").write_text("ok\n")
        argv = wall.bwrap_argv([str(inside)], [], [], minimal=True)
        dev = argv.index("--dev")
        at = argv.index(str(inside))
        assert argv[dev + 1] == "/dev" and dev < at and argv[at - 1] == "--ro-bind"
        assert argv.count("--dev") == 1
        assert str(shm) not in wall.bwrap_argv([str(shm)], [str(shm)], [], minimal=True)
        if shutil.which("bwrap") is None:
            return
        system = [d for d in ("/usr", "/bin", "/lib", "/lib64") if Path(d).exists()]
        argv = wall.bwrap_argv([*system, str(inside)], [], [], minimal=True)
        done = subprocess.run([*argv, "cat", str(inside / "seen")], capture_output=True, text=True)
        if "Unknown option" in done.stderr or "No permissions" in done.stderr:
            pytest.skip(f"this bubblewrap cannot run the wall: {done.stderr.strip()}")
        assert done.returncode == 0 and done.stdout == "ok\n", done.stderr
    finally:
        shutil.rmtree(inside)


def test_long_run_fix_names_each_clis_own_long_run_sign_in():
    """A sign-in that cannot last the run is fixed by the CLI's own long-lived credential: claude's one-year
    setup token, another CLI's token variables; never a refresh loop qda does not ship."""
    claude = isolation.long_run_fix("claude")
    assert "claude setup-token" in claude and "CLAUDE_CODE_OAUTH_TOKEN" in claude and "one year" in claude
    codex = isolation.long_run_fix("codex")
    assert "OPENAI_API_KEY or CODEX_API_KEY" in codex and "setup-token" not in codex
    assert "DEEPSEEK_API_KEY" in isolation.long_run_fix("dsh")
    assert all("refresh loop" not in isolation.long_run_fix(n) for n in ("claude", "codex", "dsh"))
    # a token beside agent.backend.sign_in_dir is refused, so the fix for a named dir removes it first
    named = isolation.long_run_fix("claude", Path("/acct"))
    assert named.startswith("remove agent.backend.sign_in_dir /acct") and "claude setup-token" in named
