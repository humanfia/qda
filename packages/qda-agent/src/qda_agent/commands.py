"""The four commands the agent sees: `qda`, `kl`, `qsched`, `handin`.

Each is a thin client of the host (see `qda_agent.rpc` for the wire format): it sends its argv and
working directory and prints what the host answers. This file is stdlib only and self-contained;
`install_bin` copies it into the sandbox's `bin/` as `qda-rpc`, links the four names to it and writes
`rpc.path`: the socket path and the work directory, one per line, as the sandbox sees them. The
working directory goes to the host relative to the work directory.

    qda [--cards N] [--expected DUR] [--name X] [--after J] build CONFIG.yaml [--set k=v ...]
                                              queue a build (CONFIG copied now)
    qda ...                                   any other qda subcommand, run on the host without a GPU
    kl DIR | kl list | kl show ID             score a checkpoint (frozen copy) / the readings so far
    qsched add [--cards N] --expected DUR [--name X] [--after J] SCRIPT
                                              SCRIPT (in the work directory) is copied now, run as
                                              `bash <copy>`; interpreters and flags go inside it.
                                              DUR: 90s, 20m, 1.5h or minutes. --after repeats or a,b.
    qsched list | qsched log NAME [-n N] | qsched cancel NAME ...
    qsched show NAME | qsched eta NAME [--layers N]    one job; when it should end (from its progress lines)
    qsched status | qsched usage                       the cards and the queue; card-hours by job kind
    qsched wait --job NAME... | --any [--timeout DUR]
                                              block (at most 590 s); the last line starts MET, FAILED, TIMEOUT
    handin DIR [DIR...] | handin status       submit a hand-in list, best first (each checked on a copy)

A toolset's other commands (`gencheck DIR [--max-tokens N] [--per-domain K] | gencheck list | gencheck show ID`)
are links to the same client.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

NAMES = ("qda", "kl", "qsched", "handin")
SCRIPT = "qda-rpc"
PATH_FILE = "rpc.path"


def install_bin(
    bindir: Path, socket_path: Path, workdir: Path | None = None, names: tuple[str, ...] = NAMES
) -> list[Path]:
    """Write the client and its links into `bindir`.

    Args:
        bindir: Directory put on the agent's PATH.
        socket_path: The host socket as seen from inside the sandbox.
        workdir: The work directory as seen from inside the sandbox (default: `bindir`'s parent).
        names: The commands to link (the toolset's; default all four).

    Returns:
        The command paths.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    script = bindir / SCRIPT
    script.write_text("#!/usr/bin/env python3\n" + Path(__file__).read_text())
    script.chmod(0o755)
    (bindir / PATH_FILE).write_text(f"{socket_path}\n{workdir or bindir.parent}\n")
    out = []
    for n in names:
        link = bindir / n
        link.unlink(missing_ok=True)
        link.symlink_to(SCRIPT)
        out.append(link)
    return out


def _connect(s: socket.socket, path: str) -> None:
    """Connect to `path`; one too long for a unix socket address is reached through its directory's descriptor."""
    p = Path(path)
    if len(os.fsencode(path)) <= 107 or not Path("/proc/self/fd").is_dir():
        s.connect(path)
        return
    fd = os.open(p.parent, getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY)
    try:
        s.connect(f"/proc/self/fd/{fd}/{p.name}")
    finally:
        os.close(fd)


def request(sock_path: str, verb: str, argv: list[str], cwd: str) -> dict:
    """One round trip: send the request line, return the response object."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        _connect(s, sock_path)
        s.sendall(json.dumps({"verb": verb, "argv": argv, "cwd": cwd}).encode() + b"\n")
        buf = bytearray()
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(bytes(buf).split(b"\n", 1)[0])


def main(argv: list[str] | None = None) -> int:
    """Entry point of the four commands; the command is the name it was called by."""
    argv = sys.argv if argv is None else argv
    verb = Path(argv[0]).name
    if (
        verb == SCRIPT
    ):  # a toolset's other commands (`gencheck`, ...) are links too; the host refuses unknown ones
        print(f"call this by a command's name ({', '.join(NAMES)}, ...)", file=sys.stderr)
        return 2
    here = Path(os.path.realpath(argv[0])).parent
    try:
        sock, root = (here / PATH_FILE).read_text().split("\n")[:2]
        cwd = os.path.relpath(Path.cwd(), root)
        if cwd == ".." or cwd.startswith("../"):
            print(f"{verb}: run it inside the work directory {root}", file=sys.stderr)
            return 2
        r = request(sock, verb, list(argv[1:]), cwd)
    except (OSError, ValueError) as e:
        print(f"{verb}: the host is not reachable ({type(e).__name__})", file=sys.stderr)
        return 1
    sys.stdout.write(r.get("stdout", ""))
    sys.stderr.write(r.get("stderr", ""))
    return int(r.get("exit", 1))


if __name__ == "__main__":
    sys.exit(main())
