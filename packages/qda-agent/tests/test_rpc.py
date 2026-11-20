import contextlib
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest
from qda_agent.rpc import Reply, Request, Server, _read_line, call


def echo(req: Request) -> Reply:
    if req.verb == "kl":
        raise RuntimeError("secret /host/path")
    return Reply(0, json.dumps([req.verb, req.argv, req.cwd]), "")


@pytest.fixture
def server(tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (work / "out").symlink_to(tmp_path)
    srv = Server(tmp_path / "rpc.sock", echo, work)
    srv.start()
    yield srv, work
    srv.stop()
    assert not (tmp_path / "rpc.sock").exists()


def test_round_trip(server, tmp_path):
    srv, work = server
    r = call(srv.path, "qsched", ["list"], "sub")
    assert r.exit == 0 and json.loads(r.stdout) == ["qsched", ["list"], str(work / "sub")]
    r = call(srv.path, "kl", ["d"])
    assert r.exit == 1 and "internal error" in r.stderr and "/host" not in r.stderr
    r = call(srv.path, "gencheck", [])  # any plain name reaches the handler, which knows the toolset's
    assert r.exit == 0 and json.loads(r.stdout)[0] == "gencheck"
    for bad in ("../rm", "Rm", "", "x" * 33, 7):
        r = call(srv.path, bad, [])
        assert r.exit == 2 and "unknown verb" in r.stderr
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(srv.path))
        s.sendall(b'{"verb": "qda", "argv": [1], "cwd": "."}\n')
        assert json.loads(s.makefile().readline())["exit"] == 2


@pytest.mark.parametrize("cwd", ["/", "/tmp", "..", "sub/../..", "out"])
def test_cwd_must_stay_in_workdir(server, cwd):
    srv, _ = server
    r = call(srv.path, "qsched", ["list"], cwd)
    assert r.exit == 2 and "work directory" in r.stderr


def test_socket_is_owner_only(server, tmp_path):
    srv, _ = server
    mode = stat.S_IMODE(srv.path.stat().st_mode)
    assert stat.S_ISSOCK(srv.path.stat().st_mode) and mode == 0o600
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".rpc-")] == []


def test_deep_or_huge_requests_get_a_reply(server):
    srv, _ = server
    for line in (b"[" * 100000 + b"]" * 100000, b'{"verb": [[[[[["x"]]]]]], "argv": [], "cwd": "."}'):
        with socket.socket(socket.AF_UNIX) as s:
            s.connect(str(srv.path))
            s.sendall(line + b"\n")
            got = json.loads(s.makefile().readline())
        assert got["exit"] == 2 and "too deep" in got["stderr"]
    a, b = socket.socketpair()

    def flood():
        with contextlib.suppress(OSError):  # the reader stops at MAX_LINE and the peer closes
            a.sendall(b"x" * (2 << 20))
        a.close()

    t = threading.Thread(target=flood)
    t.start()
    with pytest.raises(ValueError, match="too long"):
        _read_line(b)
    b.close()
    t.join()
    assert call(srv.path, "qsched", ["list", 'x"[[[[[[["'], ".").exit == 0  # brackets inside strings are fine


def test_bind_path_len_matches_the_real_bind(tmp_path):
    from qda_agent.rpc import BIND_PREFIX, bind_path_len

    sock = tmp_path / "rpc.sock"
    seen = []
    import tempfile

    real = tempfile.mkdtemp

    def spy(**kw):
        d = real(**kw)
        seen.append(len(str(Path(d) / "s")))
        return d

    tempfile.mkdtemp = spy
    try:
        Server(sock, echo, tmp_path).server_close()
    finally:
        tempfile.mkdtemp = real
    assert seen and bind_path_len(sock) == max(seen[0], len(str(sock))) and BIND_PREFIX == ".rpc-"


def test_bind_path_len_counts_bytes(tmp_path):
    from qda_agent.rpc import BIND_EXTRA, bind_path_len

    d = tmp_path / ("é" * 10)  # 2 bytes each in UTF-8
    sock = d / "rpc.sock"
    assert bind_path_len(sock) == len(os.fsencode(d)) + BIND_EXTRA
    assert bind_path_len(sock) > len(str(d)) + BIND_EXTRA


def _inherited_setgid(parent: Path) -> int:
    """The setgid bit a directory made under `parent` takes from it (shared scratch); no other extra bit is allowed."""
    return stat.S_ISGID if parent.stat().st_mode & stat.S_ISGID else 0


def test_private_dir_is_owner_only_and_removed_however_the_run_ends(tmp_path, monkeypatch):
    from qda_agent.rpc import private_dir, private_dir_of

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt), private_dir(run) as d:
        assert d.parent == tmp_path and d.name.startswith("qda-") and stat.S_IMODE(d.stat().st_mode) == 0o700 | _inherited_setgid(tmp_path)
        assert private_dir_of(run) == d
        (d / "rpc.sock").write_text("")
        raise KeyboardInterrupt
    assert not d.exists() and private_dir_of(run) is None
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    with private_dir(run) as d:
        assert d.parent == Path("/tmp")
    assert not d.exists()


def test_private_dir_is_resolved(tmp_path, monkeypatch):
    """A base named through a link gives the directory's real path: no name on the way to it is ever bound."""
    from qda_agent.rpc import private_dir

    (tmp_path / "real").mkdir()
    (tmp_path / "lnk").symlink_to(tmp_path / "real")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "lnk"))
    with private_dir(tmp_path / "run") as d:
        assert d.parent == tmp_path / "real"


def test_a_caller_gone_before_its_reply_prints_no_traceback(tmp_path, capfd):
    """A caller that hangs up while its request runs (a turn that ended, a killed `qsched`) leaves the reply
    nowhere to go: the server drops it quietly instead of printing a BrokenPipeError traceback."""
    started, release = threading.Event(), threading.Event()

    def slow(req: Request) -> Reply:
        if req.argv != ["big"]:
            return Reply(0, "", "")
        started.set()
        release.wait(10)
        return Reply(0, "x" * (1 << 20), "")  # more than the socket buffers: the write meets the closed end

    work = tmp_path / "work"
    work.mkdir()
    srv = Server(tmp_path / "rpc.sock", slow, work)
    srv.start()
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.connect(str(srv.path))
            s.sendall(b'{"verb": "qsched", "argv": ["big"], "cwd": "."}\n')
            assert started.wait(10)
        release.set()
        r = call(srv.path, "qsched", [])  # the server still answers the next caller
        assert r.exit == 0
        deadline = time.monotonic() + 10  # the big reply's handler meets the closed end after `call` returns
        while any("process_request_thread" in t.name for t in threading.enumerate()) and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        srv.stop()
    err = capfd.readouterr().err
    assert "Traceback" not in err and "BrokenPipeError" not in err
