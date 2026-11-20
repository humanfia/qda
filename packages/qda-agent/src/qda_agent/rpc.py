"""The request channel between the in-sandbox commands and the host.

Wire format (frozen; agreed with the flow): a unix stream socket at `<run_dir>/rpc.sock`, one request
and one response per connection, each a single JSON line (UTF-8, at most `MAX_LINE` bytes)::

    request   {"verb": "qda" | "kl" | "qsched" | "handin" | <a toolset's command>, "argv": [str, ...], "cwd": str}
    response  {"exit": int, "stdout": str, "stderr": str}

`argv` excludes the command name; `cwd` is the caller's working directory relative to the agent's work
directory (`.` at its top). The server refuses an absolute `cwd` and one that resolves outside the work
directory (symlinks included); handlers get it resolved to the host path. The socket is owner-only
(0600; bound in a private 0700 directory, then moved into place). The server never raises into the
client: a bad request gets exit 2, an internal error exit 1 with one line. Stdlib only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import socketserver
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERB = re.compile(r"[a-z][a-z0-9_-]{0,31}")  # what a command name looks like on the wire
MAX_LINE = 1 << 20
MAX_DEPTH = 4  # a request nests 2 deep ({..., "argv": [...]})
BIND_PREFIX = ".rpc-"
#: the temporary bind path is `<dir>/.rpc-XXXXXXXX/s` (mkdtemp adds 8 characters)
BIND_EXTRA = len("/" + BIND_PREFIX + "XXXXXXXX" + "/s")
SUN_PATH_MAX = 107  # sockaddr_un.sun_path, less its NUL


def bind_path_len(path: Path) -> int:
    """The longest path, in filesystem-encoded bytes, the server binds for the socket `path` (checked by doctor).

    That is the temporary `<dir>/.rpc-XXXXXXXX/s`, which is longer than the final path.
    """
    p = Path(path).absolute()
    return max(len(os.fsencode(p)), len(os.fsencode(p.parent)) + BIND_EXTRA)


def fits(path: Path) -> bool:
    """Whether the server can bind the socket `path` (`bind_path_len` within `SUN_PATH_MAX`)."""
    return bind_path_len(path) <= SUN_PATH_MAX


PRIVATE_PREFIX = "qda-"


def private_base() -> Path:
    """Where `private_dir` makes its directories: `$XDG_RUNTIME_DIR`, or `/tmp` without one."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    return Path(base) if base and Path(base).is_dir() else Path("/tmp")  # noqa: S108 (mkdtemp: 0700, random name)


#: The private socket directory of each run this process holds open, by resolved `run_dir` (`private_dir`).
_PRIVATE: dict[Path, Path] = {}


def private_dir_of(run_dir: Path) -> Path | None:
    """The private socket directory `private_dir` holds open for `run_dir`, or None."""
    return _PRIVATE.get(Path(run_dir).resolve())


@contextlib.contextmanager
def private_dir(run_dir: Path) -> Iterator[Path]:
    """Hold a short, private directory for the host's socket of one run, removed on the way out.

    The directory is `mkdtemp` (mode 0700, a random name, made without a race) in `$XDG_RUNTIME_DIR`, or in
    `/tmp` without one. It is for a run whose own socket path would not fit a unix socket: the sandbox shows it
    at that path, read-only (`RunConfig.socket_binds`). The run's config objects find it by `run_dir`
    (`private_dir_of`) while it is held, in this process only.
    """
    key = Path(run_dir).resolve()
    if key in _PRIVATE:
        raise RuntimeError(f"{run_dir} already holds a socket directory")
    # resolved: what is bound is the directory itself, never a name (a link in its base) on the way to it
    d = Path(os.path.realpath(tempfile.mkdtemp(prefix=PRIVATE_PREFIX, dir=private_base())))
    _PRIVATE[key] = d
    try:
        yield d
    finally:
        del _PRIVATE[key]
        shutil.rmtree(d, ignore_errors=True)


def _depth(line: bytes) -> int:
    """The bracket nesting of a JSON text (strings skipped), without parsing it."""
    depth = most = 0
    in_str = esc = False
    for c in line.decode("utf-8", "replace"):
        if in_str:
            esc, in_str = (False, True) if esc else (c == "\\", c != '"')
        elif c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
            most = max(most, depth)
        elif c in "]}":
            depth -= 1
    return most


@dataclass
class Request:
    """One command call from the sandbox."""

    verb: str
    argv: list[str]
    cwd: str


@dataclass
class Reply:
    """What the command prints and exits with."""

    exit: int = 0
    stdout: str = ""
    stderr: str = field(default="")

    @classmethod
    def refuse(cls, verb: str, why: str) -> Reply:
        """Exit 2 with `<verb>: refused: <why>` on stderr."""
        return cls(2, "", f"{verb}: refused: {why}\n")


Handler = Callable[[Request], Reply]


def parse(line: bytes, root: Path) -> Request:
    """Decode and validate one request line; its `cwd` comes back resolved under `root`.

    The verb must be a plain command name; which commands a search has is the handler's to decide.

    Raises:
        ValueError: The line is not a well-formed request.
    """
    if _depth(line) > MAX_DEPTH:
        raise ValueError("request nested too deep")
    try:
        d = json.loads(line)
    except RecursionError:
        raise ValueError("request nested too deep") from None
    if not isinstance(d, dict) or set(d) != {"verb", "argv", "cwd"}:
        raise ValueError("a request is {verb, argv, cwd}")
    verb, argv, cwd = d["verb"], d["argv"], d["cwd"]
    if not isinstance(verb, str) or not VERB.fullmatch(verb):
        raise ValueError(f"unknown verb {verb!r}")
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv) or not isinstance(cwd, str):
        raise ValueError("argv must be a list of strings and cwd a string")
    if Path(cwd).is_absolute():
        raise ValueError("cwd must be relative to the work directory")
    real = (root / cwd).resolve()
    if not real.is_relative_to(root.resolve()):
        raise ValueError("cwd is outside the work directory")
    return Request(verb, argv, str(real))


def _read_line(sock: socket.socket) -> bytes:
    buf = bytearray()
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > MAX_LINE:
            raise ValueError("request too long")
    return bytes(buf.split(b"\n", 1)[0])


class _Handle(socketserver.BaseRequestHandler):
    server: Server

    def handle(self) -> None:
        try:
            req = parse(_read_line(self.request), self.server.root)
        except ValueError as e:
            reply = Reply(2, "", f"rpc: bad request: {e}\n")
        else:
            try:
                reply = self.server.handler(req)
            except Exception as e:  # noqa: BLE001  (one line to the caller, never a traceback)
                reply = Reply(1, "", f"{req.verb}: internal error ({type(e).__name__})\n")
        # The caller may be gone by now (its turn ended, a timeout killed it): no one is left to read the reply,
        # and `socketserver` would print a traceback to the search's stderr for it.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.request.sendall(json.dumps(asdict(reply)).encode() + b"\n")


class Server(socketserver.ThreadingUnixStreamServer):
    """The host end: one thread per connection, every request through `handler`.

    Args:
        path: Socket path (`<run_dir>/rpc.sock`).
        handler: Maps a request to its reply.
        root: The agent's work directory; every request's `cwd` must resolve inside it.
    """

    daemon_threads = True

    def __init__(self, path: Path, handler: Handler, root: Path) -> None:
        self.path, self.handler, self.root = Path(path), handler, Path(root).resolve()
        self.path.unlink(missing_ok=True)
        private = Path(tempfile.mkdtemp(prefix=BIND_PREFIX, dir=self.path.parent))  # mode 0700
        try:
            super().__init__(str(private / "s"), _Handle)
            (private / "s").chmod(0o600)
            (private / "s").rename(self.path)
        finally:
            private.rmdir()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Serve in a background thread."""
        self._thread = threading.Thread(target=self.serve_forever, name="qda-rpc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and remove the socket."""
        self.shutdown()
        self.server_close()
        self.path.unlink(missing_ok=True)


def call(path: Path | str, verb: str, argv: list[str], cwd: str = ".", timeout_s: float = 600.0) -> Reply:
    """Send one request and wait for its reply (`cwd` relative to the work directory)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        s.connect(str(path))
        s.sendall(json.dumps({"verb": verb, "argv": argv, "cwd": cwd}).encode() + b"\n")
        d = json.loads(_read_line(s))
    return Reply(int(d["exit"]), str(d["stdout"]), str(d["stderr"]))
