"""The wall's one way out: a host proxy behind a unix socket, and the relay that carries loopback to it."""

import socket
import subprocess
import sys
import threading

import pytest

fence = pytest.importorskip("hmz.coganchor.fence")

from qda_agent import isolation  # noqa: E402
from qda_agent.isolation import hostnet  # noqa: E402


def _target() -> tuple[int, list]:
    """A stand-in for the model's API on the host's loopback: it records what reaches it."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
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
    return srv.getsockname()[1], seen


#: A stand-in for the agent CLI: it reaches the API through the proxy `live_state` set.
CLI = r"""
import json, os, socket, urllib.parse
p = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
out = {"proxy": os.environ["HTTPS_PROXY"], "no_proxy": os.environ.get("NO_PROXY")}
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
print(json.dumps(out))
"""


def test_the_relay_carries_a_connection_to_the_hosts_proxy(sandbox_helpers):
    """`live_state` starts a relay on the sandbox's loopback: it reaches the host's proxy, and only its hosts."""
    port, seen = _target()
    bridge = hostnet.NetBridge()
    try:
        sock = bridge.socket([f"127.0.0.1:{port}"])
        argv = [
            *isolation.LIVE_PYTHON,
            isolation.LIVE_SOURCE,
            "{}",
            str(sock),
            "--",
            sys.executable,
            "-I",
            "-S",
            "-c",
            CLI % {"port": port},
        ]
        process = sandbox_helpers(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
    finally:
        bridge.stop()
    assert process.returncode == 0, stderr
    import json

    out = json.loads(stdout.strip().splitlines()[-1])
    assert out["proxy"].startswith("http://127.0.0.1:") and out["no_proxy"] == ""
    assert out["allowed"].startswith("HTTP/1.1 200"), out
    assert out["echo"] == "pong" and seen == [b"ping"]  # it went through the proxy, not around it
    assert out["refused"].startswith("HTTP/1.1 403"), out  # a host the fence does not name is refused


def test_the_command_does_not_run_when_the_hosts_socket_is_missing(tmp_path, sandbox_helpers):
    """Without the socket the CLI would have no way out: refuse before it starts, saying why."""
    argv = [
        *isolation.LIVE_PYTHON,
        isolation.LIVE_SOURCE,
        "{}",
        str(tmp_path / "absent.sock"),
        "--",
        sys.executable,
        "-I",
        "-S",
        "-c",
        "print('the command ran')",
    ]
    process = sandbox_helpers(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode != 0 and "cannot reach the host's proxy" in stderr, stderr
    assert "the command ran" not in stdout


def test_a_socket_path_too_long_for_a_unix_socket_is_refused(tmp_path, monkeypatch):
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    monkeypatch.setattr(hostnet.rpc, "private_base", lambda: deep)
    bridge = hostnet.NetBridge()
    try:
        with pytest.raises(ValueError, match="too long for a unix socket"):
            bridge.socket(["api.anthropic.com"])
    finally:
        bridge.stop()
