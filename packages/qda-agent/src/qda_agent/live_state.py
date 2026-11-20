"""Prepare writable CLI state inside the wall, and its one way out, before starting the agent CLI.

Humanize creates each session's home on the host and removes it after the turn, following its links.
That home, the CLI template and the epic's session store stay read-only inside the wall. The CLI writes
to `<run_dir>/live/<name>` instead; the host reads this state only through the snapshot helper used by
`codexlog` after the run.

The wall starts this stdlib-only program with `python -I -S -c SOURCE SPEC NET -- argv`, without loading
`.pth` files or qda-agent sources. `SPEC` maps environment variables to live directories. For each
variable Humanize set, copy its home into the live directory, update the variable, then exec `argv`.

`NET` is the unix socket the wall bound in, the host's proxy for the turn's hosts, or `-` for none. With
one, a relay process of this program listens on the sandbox's loopback (the only network there is) and
carries every connection to the socket; the proxy variables point the CLI at it. The relay is a process,
not a thread: exec ends every thread but the one that calls it. It detaches from the standard streams, so
the driver's pipes still end with the CLI.

Links can reach only paths mounted inside the wall, so copies use data the agent can already read.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import sys
import threading
from pathlib import Path

#: The variables every common HTTP client reads its proxy from, as Humanize's fence sets them
#: (`hmz.coganchor.fence.wrap.PROXIES`): this program runs before that module is importable (-I -S).
PROXIES = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")

#: How much one side of a relayed connection reads before it hands the bytes to the other.
_CHUNK = 64 * 1024


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.is_symlink() or path.exists():
        path.unlink()


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False, ignore_dangling_symlinks=True)
    elif src.is_file():
        shutil.copy2(src, dst)


def sync(home: Path, live: Path) -> None:
    """Refresh turn-specific files in `live` from `home`, preserving existing linked-entry state."""
    live.mkdir(mode=0o700, parents=True, exist_ok=True)
    for src in sorted(home.iterdir()):
        dst = live / src.name
        if src.is_symlink():
            if not (dst.is_symlink() or dst.exists()):
                _copy(src, dst)
        else:
            _remove(dst)
            _copy(src, dst)


def _pump(src: socket.socket, dst: socket.socket) -> None:
    """Carry what `src` sends to `dst` until it closes, then half-close `dst`."""
    try:
        while data := src.recv(_CHUNK):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


def _carry(client: socket.socket, sock: str) -> None:
    """Carry one loopback connection to the unix socket `sock` (and back)."""
    try:
        upstream = socket.socket(socket.AF_UNIX)
        upstream.connect(sock)
    except OSError:
        client.close()
        return
    with client, upstream:
        back = threading.Thread(target=_pump, args=(client, upstream), daemon=True)
        back.start()
        _pump(upstream, client)
        back.join()


def _serve(listener: socket.socket, sock: str) -> None:
    """Accept on the loopback port for the life of the command, one thread per connection."""
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        threading.Thread(target=_carry, args=(client, sock), daemon=True).start()


def _relay(sock: str) -> int:
    """Start the loopback-to-socket relay in a child of this process and return its port.

    The child detaches from the standard streams (a driver reading the CLI's output must see it end with
    the CLI), checks that the socket answers before the command is started, and then serves it. It ends
    with the sandbox's process namespace, which goes when the wall's turn does.

    Raises:
        SystemExit: The socket is missing or refuses a connection: the command must not run without it.
    """
    read_fd, write_fd = os.pipe()
    if os.fork() == 0:
        try:
            os.close(read_fd)
            null = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(null, fd)
            probe = socket.socket(socket.AF_UNIX)
            probe.connect(sock)  # the wall bound the host's socket in; fail before the command runs
            probe.close()
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))  # the sandbox's loopback, which is all of its network
            listener.listen(128)
            os.write(write_fd, str(listener.getsockname()[1]).encode())
            os.close(write_fd)
            _serve(listener, sock)
        except BaseException as why:  # noqa: BLE001 -- the child must never run the parent's code again
            with contextlib.suppress(OSError):
                os.write(write_fd, f"!{why}".encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        said = pipe.read(64)
    if not said or said.startswith(b"!"):
        raise SystemExit(
            f"live_state: cannot reach the host's proxy at {sock}: "
            f"{said[1:].decode(errors='replace') if said else 'no answer'}"
        )
    return int(said)


def main(argv: list[str]) -> None:
    """Parse `SPEC NET -- PROGRAM ARGS...`, sync live directories, add the proxy and exec."""
    spec, net, sep, *rest = argv
    if sep != "--" or not rest:
        raise SystemExit("usage: live_state SPEC NET -- PROGRAM [ARGS...]")
    env = dict(os.environ)
    for var, live in json.loads(spec).items():
        home = env.get(var)
        if home and Path(home).is_dir() and Path(home).resolve() != Path(live).resolve():
            sync(Path(home), Path(live))
            env[var] = live
    if net != "-":
        port = _relay(net)
        env.update(dict.fromkeys(PROXIES, f"http://127.0.0.1:{port}"))
        # Emptied rather than removed: a NO_PROXY somebody exported would send a host around the proxy.
        env.update(NO_PROXY="", no_proxy="", NODE_USE_ENV_PROXY="1")
    os.execvpe(rest[0], rest, env)


if __name__ == "__main__":
    main(sys.argv[1:])
