"""The bwrap wall: every fenced turn of the agent runs in a mount namespace without GPUs (see this package).

`gpu_wall` wraps Humanize's fence wrapper while a run is under way, so that the command it spawns is
`bwrap ... -- hmz internal fence ... -- <CLI>`. The mounts are drawn from that very fence (`bwrap_argv`),
and of the home only an allowlist comes back (`home_binds`, `INSTALLS`).

The turn's network is a namespace of its own (`--unshare-net`): loopback and no route out. Its one way out
is a unix socket bound in read-only (`NetBridge`), which carries the CLI's proxy connection to a host-side
`Proxy` that passes only the turn's `Fence.hosts`; the fence inside the wall is the filesystem's alone.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from qda_agent import rpc
from qda_agent.sandbox_settings import Bwrap

from .allocation import why_not_dedicated
from .bwrap import (
    _HIDDEN_CHECK as _CHECKER,
)
from .bwrap import (
    hidden_covers,
    hidden_specs,
    mask_after,
    mount_points,
    pid_args,
    pid_form,
    private_bwrap_argv,
    proc_args,
    published_entries,
)
from .cli_dirs import CLI_CONFIG, agent_state, cli_config_dirs, clis, live_dirs, sign_in_file
from .cli_signin import (
    SIGNIN,
    private_cli_config,
    sign_in_accounts,
    sign_in_mount,
    sign_in_tokens,
    sign_in_unmount,
)
from .hostnet import NetBridge

if TYPE_CHECKING:
    from hmz.coganchor.fence import Fence
    from hmz.coganchor.homes import Home

    from qda_agent.config import RunConfig, SharedNodeAcceptance

_log = logging.getLogger(__name__)


def bwrap_argv(
    read: Sequence[str],
    write: Sequence[str],
    hide: Sequence[Path],
    keep: Sequence[str | Path] = (),
    unset: Sequence[str] = (),
    setenv: Mapping[str, str] | None = None,
    binds: Mapping[Path, Path] | None = None,
    tmpfs: Sequence[Path] = (),
    pid: str = "auto",
    minimal: bool = False,
    links: Mapping[str, str] | None = None,
    hide_early: Sequence[Path] = (),
    hidden_checks: list[tuple[str, int, int]] | None = None,
) -> list[str]:
    """The bwrap command (up to `--`) that holds a fence's paths in a mount namespace without GPUs.

    Args:
        read: Paths to mount back read-only (the fence's `read`: the system minimum, the CLI's programs). A
            symlink under a hidden directory is made again as a symlink, not bound: a bind would put its
            target's content at the link's path (a Node launcher such as conda's `bin/codex ->
            ../lib/node_modules/...` then resolves its package from the wrong directory), and a link grants
            nothing its target's own entry does not. A path inside one already mounted read-only (a CLI
            launcher inside the venv that runs the fence) is left as that mount shows it.
        write: Paths the fence lets the CLI write. Device nodes are skipped (`/dev` is a fresh one with only
            the standard nodes). `/tmp` and `/var/tmp` become private tmpfs mounts. A path under a hidden
            directory is mounted back only when it is in `keep` or under one: the user's own CLI state
            (`~/.claude*`, `~/.config/anthropic`: every session's transcripts, memory and credentials on
            this machine) stays hidden; the CLI works in the run's own config dir, which Humanize makes.
        hide: Private paths (`sandbox.hide`: a holdout, a reference trace) masked after every bind, so a
            bind of the path or a parent cannot mount over the mask and re-expose it. Each spelling the
            caller needs (a home alias and its resolved target) is named separately. A directory becomes an
            empty tmpfs, a regular file an empty read-only `/dev/null`; a path no bind shows is skipped
            (the empty root or an earlier tmpfs already hides it), as is one a bind's source does not hold.
            The wall refuses a hide that covers a path the turn needs (its run dir, workdir, CLI dirs or a
            readable bind) instead of letting the mask silently take it (`hidden_covers`).
        keep: Writable paths under a hidden directory to mount back (the agent's workdir).
        unset: Environment variables removed for everything inside (see `inherited_secrets`).
        setenv: Environment variables set for everything inside over the inherited ones (`TZ` for the
            agent's shell, unless the run configures one).
        binds: Read-only binds at another path, `{path inside: host source}`, made last (after
            `bwrap.mount_points` made their mount points safe).
        tmpfs: Empty directories mounted just before `binds`, so a bind into one shows its file with
            nothing of the directory it came from beside it (the run's sign-in file). The mount point a
            bind makes inside one is created in the fresh tmpfs (nothing is written on the host).
        pid: The pid namespace: `"auto"`, `"private"` or `"shared"` (see `bwrap.pid_form`).
        minimal: Start from an empty root (`--tmpfs /`) and mount back only `read`, `write` and `binds`,
            rather than showing the host's whole tree read-only. A read-only host root still answers
            `connect()` to every pathname unix socket on it (a read-only bind refuses writes, not
            `connect`), and the network namespace does not cover pathname sockets; a foreign socket absent
            from the tree is one the turn cannot reach at all (ENOENT). Jobs and the grader keep the host
            root (`read` there is a small list, not the fence's system minimum).
        links: With `minimal`, `{path: target}` symlinks to make again after the empty root and its tmpfs
            mounts, so a path that names another through a link (a home reached by the link and by its
            target) still resolves. Binds must still name the target's own path: bwrap cannot make the
            parents of a mount point under a link.
        hide_early: Directories covered by an empty tmpfs before the paths above are mounted back (the
            home, the run dir, the runtime dirs): a writable path under one is bound back only when kept.

        hidden_checks: When supplied, collect host inode specifications for the caller's refusal check,
            including hidden objects exposed at relocated bind destinations.

    Returns:
        The argv, ending in `--`.
    """
    argv = [
        "bwrap",
        "--die-with-parent",
        *pid_args(pid),
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        *(["--tmpfs", "/"] if minimal else ["--ro-bind", "/", "/"]),
    ]
    # In an empty root the private temp dirs are the base every other hidden dir may sit on (a session home
    # under /tmp), so they are mounted first: a later tmpfs over /tmp would hide them.
    #: Every mount made, in order, as `(kind, destination, source)`: the last one covering a private path
    #: decides what a later mask has to do (a bind shows it, an empty tmpfs or nothing does not).
    mounts: list[tuple[str, str, str | None]] = [("tmpfs", "/", None)] if minimal else [("bind", "/", "/")]
    # The fresh /dev before anything is mounted under it: mounted after the binds, it covered every one
    # under /dev/shm -- a run dir, a home, a $TMPDIR or a CLI kept there -- and the turn found none of them.
    # The job sandbox mounts it first too (`isolation.bwrap`).
    argv += ["--dev", "/dev"]
    mounts.append(("tmpfs", "/dev", None))
    order = (*TEMPORARY, *hide_early) if minimal else (*hide_early, *TEMPORARY)
    roots = [Path(h).resolve() for h in order]
    for r in dict.fromkeys(str(r) for r in roots):
        argv += ["--tmpfs", r]
        mounts.append(("tmpfs", r, None))
    if minimal:
        for dest, target in sorted((links or {}).items(), key=lambda one: len(Path(one[0]).parts)):
            argv += ["--symlink", target, dest]
    roots += [Path(os.path.normpath(h)) for h in hide_early]  # a path may name a hidden dir by a link
    # A kept path by both spellings: as given (the spelling its bind is made at) and resolved (a link on the
    # way, such as a `$TMPDIR` under the home that is a link to a scratch filesystem).
    kept = list(dict.fromkeys(q for k in keep for q in (Path(k).resolve(), Path(os.path.normpath(k)))))
    # Read/write binds first, read-only ones last: a later bind wins, so a read-only path under a writable
    # bind (the hook socket's dir under a writable $TMPDIR) is made read-only again.
    bound: list[Path] = []
    ro: list[Path] = []
    for flag, paths in (("--bind", write), ("--ro-bind", read)):
        for p in paths:
            q = Path(p)
            if _device(q) or str(q) in TEMPORARY or not q.exists():
                continue
            hidden = any(q.is_relative_to(r) for r in roots)
            # Humanize's own scratch in the temp dirs is kept -- but never anything under a hidden root (a home
            # or run dir that itself sits in a temp dir)
            temp = any(q.is_relative_to(Path(t)) for t in TEMPORARY) and not any(
                q.is_relative_to(Path(h).resolve()) for h in hide_early
            )
            if flag == "--bind" and hidden and not temp and not any(q.is_relative_to(k) for k in kept):
                continue
            if flag == "--bind":
                bound.append(q)
                argv += [flag, p, p]
                mounts.append((flag, p, p))
            elif any(q.is_relative_to(b) for b in ro) or (
                q.is_symlink() and any(q.is_relative_to(b) for b in bound)
            ):
                continue  # already there, as it is: bwrap cannot make a mount point inside a read-only bind
            elif hidden and q.is_symlink():
                argv += ["--symlink", str(q.readlink()), p]
            elif minimal or hidden or any(q.is_relative_to(b) for b in bound):
                # An empty root shows nothing it did not mount, so every read path is mounted; a host root
                # already shows a path outside the hidden dirs, and a bind there would be busy.
                ro.append(q)
                argv += [flag, p, p]
                mounts.append((flag, p, p))
    for p in tmpfs:
        argv += ["--tmpfs", str(p)]
        mounts.append(("tmpfs", str(p), None))
    for dst, src in (binds or {}).items():
        argv += ["--ro-bind", str(src), str(dst)]
        mounts.append(("--ro-bind", str(dst), str(src)))

    # The private paths last: a mask mounted before a bind of the path or a parent is mounted over by it,
    # which is how a holdout stayed readable at its resolved spelling while the alias looked hidden.
    masks = mask_after(mounts, hide)
    for view, host in sorted(masks, key=lambda one: len(one[0].parts)):
        if host.is_file():
            argv += ["--ro-bind", "/dev/null", str(view)]
        else:
            argv += ["--tmpfs", str(view)]
    if hidden_checks is not None:
        hidden_checks.extend(hidden_specs([*((p, p) for p in hide), *masks]))
    for name in unset:
        argv += ["--unsetenv", name]
    for name, value in sorted((setenv or {}).items()):
        argv += ["--setenv", name, value]
    return [*argv, *proc_args(pid=pid), "--"]


def _device(path: Path) -> bool:
    """Whether the wall's own `/dev` or `/proc` provides `path`, which is then never bound from the host.

    A directory under `/dev/shm` (a run dir, a home, a `$TMPDIR` there) is a host path like any other and is
    bound; `/dev/shm` itself stays the wall's own, never the host's shared memory.
    """
    shm = Path("/dev/shm")
    inside_shm = path.is_relative_to(shm) and path != shm
    return path.is_relative_to("/proc") or (path.is_relative_to("/dev") and not inside_shm)


def _hidden_specs(paths: Sequence[Path]) -> list[tuple[str, int, int]]:
    """Each existing path with the host's `(device, inode)`: what a mask must change inside the wall."""
    return hidden_specs((Path(p), Path(p)) for p in dict.fromkeys(os.path.normpath(str(h)) for h in paths))


def _verify_hidden(argv: Sequence[str], specs: Sequence[tuple[str, int, int]]) -> None:
    """Refuse to start unless every private path is a mask inside the wall, not the host's own object.

    The check runs the turn's own mounts (its `argv`, ending in `--`) with a stat: a hidden directory is an
    empty tmpfs and a hidden file `/dev/null`, so each shows a different `(device, inode)` than the host's;
    a path absent inside is hidden too. A bind that re-exposed the path shows the host's pair, which is the
    refusal.
    """
    got = subprocess.run(
        [*argv, "/usr/bin/python3", "-I", "-S", "-c", _CHECKER, json.dumps(specs)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if got.returncode:
        raise WallRefused(f"the wall's hidden-path self-check could not run: {got.stderr.strip()}")
    bad = json.loads(got.stdout or "[]")
    if bad:
        raise WallRefused(
            f"the wall would start with a private path still visible ({bad[0]}): a bind shows the host's own "
            "file or directory there; keep it out of sandbox.hide's spellings and of every bound path"
        )


def exposed(argv: Sequence[str], path: Path) -> str | None:
    """The writable bind of a bwrap `argv` that shows `path` (or a parent) to the agent, or None.

    Both spellings are compared, as written and with links followed, on each side.
    """
    ps = {Path(os.path.abspath(path)), Path(os.path.realpath(path))}  # noqa: PTH100 (links kept)
    for i, a in enumerate(argv[:-2]):
        if a == "--bind":
            dst = argv[i + 2]
            ds = {Path(os.path.abspath(dst)), Path(os.path.realpath(dst))}  # noqa: PTH100
            if any(p.is_relative_to(d) for p in ps for d in ds):
                return dst
    return None


#: Inherited variables that would let the agent reach whatever started the run: a Claude Code session's
#: messaging socket and token, its session ids. Matched by prefix / substring.
SECRET_PREFIXES = ("CLAUDE_CODE_",)
SECRET_PARTS = ("MESSAGING",)


def inherited_secrets(environ: Mapping[str, str], keep: Collection[str] = ()) -> list[str]:
    """The names in `environ` (the flow's own environment) the agent must not inherit.

    ``keep``: names that sign one of the run's CLIs in (`sign_in_tokens`, from `CLI_CONFIG`). Such a token is
    the run's own credential, not the launcher session's, so it stays. The keep set is derived from the product's
    auth-variable table, not a hand-written allow-list; every other name matching a secret prefix or part goes.
    """
    kept = frozenset(keep)
    return sorted(
        k
        for k in environ
        if k not in kept and (k.startswith(SECRET_PREFIXES) or any(x in k for x in SECRET_PARTS))
    )


def runtime_dirs(environ: Mapping[str, str]) -> list[Path]:
    """The user's runtime dir (`$XDG_RUNTIME_DIR`, `/run/user/$UID`): session buses, agent sockets."""
    out = [Path(environ["XDG_RUNTIME_DIR"])] if environ.get("XDG_RUNTIME_DIR") else []
    out.append(Path(f"/run/user/{os.getuid()}"))
    return [d for d in dict.fromkeys(out) if d.is_dir()]


def home_links(home: Path) -> dict[str, str]:
    """The symlinks on the way to `home`, to make again in an empty root (`{path: target}`).

    A cluster may spell a home twice, one path a link to the other: the wall binds the real spelling,
    and the turn's `$HOME`, its cwd and any path it builds keep the link's. In an empty root the link is
    not there unless it is made again. The bind destinations themselves stay on the link's target: bwrap
    cannot make the parents of a mount point under a link.
    """
    links = {}
    for p in (home, *home.parents):
        q = Path(os.path.normpath(p))
        if q != Path("/") and q.is_symlink():
            links[str(q)] = str(q.readlink())
    return links


def interpreter_links(exe: str, home: Path, base: str) -> dict[str, str]:
    """The symlinks under `home` on the way from `exe` to the interpreter it runs (`{path: target}`).

    A venv made by `uv venv --managed-python` runs `bin/python` -> `~/.local/share/uv/python/cpython-3.12-...`,
    a link to the patch release's directory. The wall binds that directory (`sys.base_prefix`) but hides the
    rest of the home, so without the link the turn cannot start its Python (`execvp ... No such file or
    directory`). Paths are spelled from the home's real path, as the wall's binds are. Only a link into
    `base` (the interpreter's `sys.base_prefix`, which the wall binds) or to a directory that holds it
    (`~/pythons -> /scratch/pythons`) is made: bwrap cannot make a mount
    point under a link, so a link to a directory the wall binds elsewhere (`~/proj -> /scratch/proj` with
    the venv in `~/proj/.venv`) would stop every turn (`bwrap: Can't mkdir parents for .../.venv/bin`).
    """
    real = Path(os.path.realpath(home))
    inside = Path(os.path.realpath(base))
    links: dict[str, str] = {}
    p = Path(os.path.abspath(exe))  # noqa: PTH100 (links kept)
    for _ in range(40):  # the kernel's own hop limit
        hop = next((a for a in (*reversed(p.parents[:-1]), p) if a.is_symlink()), None)
        if hop is None:
            break
        target = hop.parent / hop.readlink()
        to = Path(os.path.realpath(target))
        if Path(os.path.realpath(hop.parent)).is_relative_to(real) and (
            to.is_relative_to(inside) or inside.is_relative_to(to)
        ):
            at = Path(os.path.realpath(hop.parent)) / hop.name
            links[str(at)] = str(hop.readlink())
        p = Path(os.path.normpath(target / p.relative_to(hop)))
    return links


#: The `mkdtemp` prefixes of the dirs Humanize builds session homes in: `hmz-home-` (`homes.build` and the
#: session's own parent) and `hmz-homes-` (the dir that holds the homes of native and anchor turns, 0.2.2).
HUMANIZE_HOME_PREFIXES = ("hmz-home-", "hmz-homes-")


def _humanize_home(path: str) -> bool:
    """Whether a fence path is one of Humanize's per-session homes or their parents, in a temp dir."""
    parts = Path(path).parts
    return any(part.startswith(HUMANIZE_HOME_PREFIXES) for part in parts) and any(
        Path(path).is_relative_to(t) or Path(os.path.realpath(path)).is_relative_to(os.path.realpath(t))
        for t in (*TEMPORARY, tempfile.gettempdir())
    )


#: The install trees each agent CLI runs from, by Humanize backend name. Under the real home the wall binds back
#: only these (read-only), the run's own dirs and the session's sign-in file (`home_binds`): an allowlist, so
#: that the user's CLI state -- every session's transcripts, memory, history and credentials in `~/.claude*`,
#: `~/.claude.json`, `~/.codex`, `~/.config/anthropic`, ... -- never comes back, whatever Humanize lists for
#: the CLI to read or write, now or in a later version. Each entry is a home-relative path pattern matched
#: from the right (`bin/claude` is `~/.local/bin/claude` or `~/.npm-global/bin/claude`) against a path
#: Humanize lists or one of its parents; `*` stands for one path component. A CLI not named here cannot run
#: in the wall (`installs` refuses it): add its install trees, never its state dir.
INSTALLS: dict[str, tuple[str, ...]] = {
    # The native installer: a launcher in ~/.local/bin and one file per version; or npm, global or the
    # older per-user `~/.claude/local` (an npm prefix holding only the package).
    "claude": (
        "bin/claude",
        ".local/share/claude/versions/*",
        ".claude/local",
        "node_modules/@anthropic-ai/claude-code",
        "bin/node",
    ),
    # The npm package and its native binary's platform package (`@openai/codex-linux-x64`), wherever npm put it.
    "codex": ("bin/codex", "node_modules/@openai/codex", "node_modules/@openai/codex-*", "bin/node"),
    # DeepSeek Harness has no install tree of its own under the home: Humanize drives it through its Python
    # SDK, whose bundled runtime is one executable in the site-packages of the interpreter that runs the fence
    # (`PYTHON_PARTS` keeps that interpreter). Its state dir (`~/.dsh`) is not named, so it never comes back.
    "dsh": (),
}


def installs(cfg: RunConfig) -> tuple[str, ...]:
    """The install-tree patterns (`INSTALLS`) of every CLI the run may use: its spec's and its fallbacks'.

    Raises:
        ValueError: A CLI Humanize does not know, or one with no entry in `INSTALLS`.
    """
    out: list[str] = []
    for name, _ in clis(cfg):
        if name not in INSTALLS:
            raise ValueError(
                f"the bwrap wall knows no install trees for the agent CLI {name!r}; name them in "
                "qda_agent.isolation.INSTALLS (never its state dir), or use sandbox docker"
            )
        out += INSTALLS[name]
    return tuple(dict.fromkeys(out))


def _install(rel: Path, patterns: Sequence[str]) -> bool:
    return any(a.match(g) for g in patterns for a in (rel, *rel.parents) if a != Path())


def home_binds(
    paths: Sequence[str],
    home: Path,
    *,
    dirs: Sequence[str | Path] = (),
    patterns: Sequence[str] = (),
    narrow: Sequence[str | Path] = (),
    dropped: set[str] | None = None,
) -> list[str]:
    """The paths of a fence's list that may be bound back: those under `home` only when allowed.

    A path under `home` is kept when it is inside one of `dirs` (the run's own: the workdir and its Humanize
    epic), is inside one of `narrow` (the parts of the Python
    that runs the fence: its `bin`, `lib`, `pyvenv.cfg`), or lies in an install tree (`patterns`, from
    `INSTALLS`). A listed directory that holds entries of `narrow` (the Python prefix itself) becomes those
    entries; a Node runtime's prefix becomes its `lib` (`<prefix>/bin/node` loads `<prefix>/lib/libnode.so`
    in a conda or Homebrew node). Everything else under `home` is dropped and logged once at debug level.
    `home` is matched in both spellings, as given and with its links followed. Paths outside `home` pass
    unchanged (`bwrap_argv` rules them).

    Args:
        paths: The fence's `read` or `write` list.
        home: The real home directory.
        dirs: The run's own directories under `home` (`home` itself, or one outside it, is ignored).
        patterns: Install-tree patterns; pass none for a writable list (install trees are read-only).
        narrow: Parts of a listed directory that may be bound in its place (read-only lists only).
        dropped: Paths already logged; updated.

    Returns:
        The paths to bind, in order (what a listed directory becomes in its place).
    """
    spellings = list(dict.fromkeys((Path(os.path.normpath(home)), Path(os.path.realpath(home)))))

    def under(p: str | Path) -> tuple[Path, Path] | None:
        q = Path(os.path.normpath(p))
        return next(((h, q.relative_to(h)) for h in spellings if q.is_relative_to(h)), None)

    def rels(ps: Sequence[str | Path]) -> list[Path]:
        return [u[1] for u in map(under, ps) if u is not None]

    own = [d for d in rels(dirs) if d != Path()]
    parts = rels(narrow)
    listed = set(rels(paths))
    seen = set() if dropped is None else dropped
    out: list[str] = []
    for p in paths:
        if (u := under(p)) is None:
            out.append(p)
            continue
        h, rel = u
        inner = [n for n in parts if n.is_relative_to(rel) and n != rel and (h / n).exists()]
        if any(rel.is_relative_to(d) for d in [*own, *parts]) or _install(rel, patterns):
            out.append(p)
        elif inner:
            out += [str(h / n) for n in inner]
        elif (
            "bin/node" in patterns
            and rel / "bin" / "node" in listed
            and any((h / rel / "lib").glob("libnode.so*"))
        ):
            out.append(str(h / rel / "lib"))
        elif p not in seen:
            seen.add(p)
            _log.debug("agent wall: %s is not on the home allowlist; not bound", p)
    return out


#: The parts of a Python prefix the fence's own interpreter needs; never the whole prefix (`~/.local` holds
#: `share/` of every program).
PYTHON_PARTS = ("bin", "lib", "lib64", "pyvenv.cfg")


#: One wall per process: the hook is a module attribute of Humanize's, shared by every run in it.
_WALL = threading.Lock()

#: The program that starts every fenced turn inside the wall (`live_state`), passed by value: the wall need not
#: show qda-agent's own files.
LIVE_SOURCE = (Path(__file__).parent.parent / "live_state.py").read_text()
#: The program that holds one turn's processes: it runs the turn in a session of its own and kills what it
#: leaves behind (`turn_guard`), so a `pid: shared` turn leaks no process into the next one.
TURN_SOURCE = (Path(__file__).parent / "turn_guard.py").read_text()
#: How the wall runs `LIVE_SOURCE`: isolated (`-I`: no PYTHON* variables, cwd or user site) and without the
#: site module (`-S`), so no `.pth` file of the interpreter's site-packages runs before it. It needs only the
#: stdlib.
LIVE_PYTHON = (sys.executable, "-I", "-S", "-c")
#: Directories bwrap replaces with private tmpfs mounts (Humanize's per-session dirs in them are kept).
TEMPORARY = ("/tmp", "/var/tmp")

#: The backends whose driver writes files into the fence's scratch directory on the host for the runtime it
#: starts to read. DeepSeek Harness is started with `--patch <scratch>/hmz-dsh-*/humanize.patch.yml`, and it
#: unpacks its native modules beside that file. The wall's `/tmp` is a fresh tmpfs, where `hmz internal fence`
#: makes the directory again, empty, and the runtime does not start. For these the host's own is bound.
HOST_SCRATCH = ("dsh",)


def host_scratch(cfg: RunConfig, tmp: str) -> list[str]:
    """The fence's scratch directory, as writable paths to add, when the wall must bind the host's own.

    Args:
        cfg: The run config: a CLI of it named in `HOST_SCRATCH` asks for the bind.
        tmp: The fence's `tmp`, which Humanize makes on the host before it builds the turn's command.

    Returns:
        `[tmp]` for a run with such a CLI whose scratch directory exists on the host, else `[]`: the turn then
        works in the empty one the wall's `/tmp` gives it. Wherever `$TMPDIR` is -- under the home too -- the
        wall binds it back: it is Humanize's per-turn `mkdtemp`, not the user's state, and is kept by the same
        rule as the turn's other per-turn dirs (`turn_dirs`).
    """
    if not tmp or not Path(tmp).is_dir() or not any(name in HOST_SCRATCH for name, _ in clis(cfg)):
        return []
    return [tmp]


def tmpdir_refusal(cfg: RunConfig) -> str | None:
    """Why the wall cannot show Humanize's per-turn dirs in this process's `$TMPDIR`; None when it can.

    Humanize makes each turn's own dirs in the temp dir (`tempfile.gettempdir()`, which reads `$TMPDIR`): the
    fence's scratch (`hmz-fence-*`, with DeepSeek Harness's `hmz-dsh-*` patch inside), the hook sockets the
    CLI reports its tool calls to (`humanize-hook-*`) and the session homes (`HUMANIZE_HOME_PREFIXES`). The
    wall binds exactly those back, wherever the temp dir is (`turn_dirs`), even under the hidden home. It
    cannot where a mask the wall puts on last covers the temp dir (`sandbox.hide`, `judge.custody_dir`), or
    where the wall's own `/dev` or `/proc` replaces it. The strict run refuses that at its start and `qda
    doctor --search` says so (`isolation.check`), rather than a turn failing minutes in.

    Args:
        cfg: The run config: its hidden paths.

    Returns:
        One line naming `TMPDIR` and the fix, or None.
    """
    temp = tempfile.gettempdir()
    spellings = {Path(os.path.abspath(temp)), Path(os.path.realpath(temp))}  # noqa: PTH100 (links kept)
    custody = getattr(getattr(cfg, "judge", None), "custody_dir", None)
    fix = "set TMPDIR to a directory of your own outside every hidden path (for example /tmp or a scratch)"
    for h in [*getattr(cfg.sandbox, "hide", ()), *([custody] if custody else [])]:
        hs = {Path(os.path.abspath(h)), Path(os.path.realpath(h))}  # noqa: PTH100
        if any(t.is_relative_to(x) for t in spellings for x in hs):
            return (
                f"TMPDIR={temp} lies under {h}, which the wall masks, so the agent's turn could not see "
                f"Humanize's per-turn dirs there (its scratch, its hook sockets, its session home): {fix}"
            )
    if any(_device(t / "hmz-fence-x") for t in spellings):
        return f"TMPDIR={temp} is under /dev or /proc, which the wall replaces with its own: {fix}"
    return None


def turn_dirs(
    paths: Sequence[str], hidden: Sequence[Path], temp: str, spellings: Sequence[Path] = ()
) -> list[str]:
    """Those of Humanize's per-turn dirs (`paths`) that lie in a hidden root, each checked to be a temp dir.

    The wall hides the home, the run dir and the runtime dirs, and mounts back only an allowlist; a `$TMPDIR`
    there (`~/tmp`, common on a cluster whose `/tmp` is small) holds the dirs Humanize made for this turn: the
    ones this process recorded (the fence's `tmp`, each hook socket `Gate.address` made) and the session
    homes the fence lists. They are not user state and are bound back by their exact paths, never by a name
    pattern searched under the home. Each must sit in this process's temp dir (any spelling), be a directory
    (not a link), be this user's and not be writable by others; anything else under the home stays hidden.

    Args:
        paths: The per-turn dirs, as the wall spells them.
        hidden: The roots the wall hides before it binds (`bwrap_argv`'s `hide_early`).
        temp: This process's temp dir (`tempfile.gettempdir()`).
        spellings: Other spellings of `temp` the paths may use (through the home's real path).

    Returns:
        The paths in a hidden root, to bind back.

    Raises:
        WallRefused: One of them is not a directory this user made in the temp dir.
    """
    temps = list(
        dict.fromkeys((Path(os.path.abspath(temp)), Path(os.path.realpath(temp)), *spellings))  # noqa: PTH100
    )
    roots = [q for h in hidden for q in (Path(os.path.normpath(h)), Path(os.path.realpath(h)))]
    out: list[str] = []
    for p in dict.fromkeys(paths):
        q = Path(os.path.normpath(p))
        if not os.path.lexists(q):
            continue  # nothing to bind yet (`bwrap_argv` skips it too)
        if not any(x.is_relative_to(r) for x in (q, Path(os.path.realpath(q))) for r in roots):
            continue  # not hidden: `bwrap_argv` rules it as before
        at = next((t for t in temps if q.is_relative_to(t) and q != t), None)
        why = "it is not in TMPDIR" if at is None else _not_ours(at / q.relative_to(at).parts[0])
        if why is not None:
            raise WallRefused(
                f"Humanize's per-turn dir {p} cannot be bound into the wall from TMPDIR={temp} ({why}): set "
                "TMPDIR to a directory of your own (for example /tmp or a scratch dir)"
            )
        out.append(str(q))
    return out


def _not_ours(path: Path) -> str | None:
    """Why `path` is not a directory of this user's alone (checked with `lstat`, no link followed); None."""
    try:
        st = os.lstat(path)
    except OSError as e:
        return e.strerror
    if not stat.S_ISDIR(st.st_mode):
        return "not a directory"
    if st.st_uid != os.getuid():
        return "not this user's"
    if st.st_mode & 0o022:
        return "writable by others"
    return None


def _shown(argv: Sequence[str], path: str, *, writable: bool) -> bool:
    """Whether a bwrap `argv` shows the host's `path` at its own path: the last mount over it binds it."""
    p = Path(os.path.normpath(path))
    last: tuple[str, str | None, Path] | None = None
    for i, a in enumerate(argv):
        if a == "--":
            break
        if a in ("--bind", "--ro-bind") and i + 2 < len(argv):
            src, dst = argv[i + 1], Path(os.path.normpath(argv[i + 2]))
        elif a in ("--tmpfs", "--dev", "--proc") and i + 1 < len(argv):
            src, dst = None, Path(os.path.normpath(argv[i + 1]))
        else:
            continue
        if p.is_relative_to(dst):
            last = (a, src, dst)
    if last is None:
        return False
    kind, src, dst = last
    if src is None or (writable and kind != "--bind"):
        return False
    return os.path.realpath(Path(src) / p.relative_to(dst)) == os.path.realpath(p)


def wall_hook() -> ModuleType:
    """Humanize's fence module, checked to still have the hook `gpu_wall` wraps.

    Returns:
        `hmz.coganchor.fence`.

    Raises:
        RuntimeError: The installed Humanize has no `fence.wrapper(fence)`: the wall cannot be put up, and
            the run must not start without it.
    """
    from hmz.coganchor import fence as hmz_fence

    fn = getattr(hmz_fence, "wrapper", None)
    if not callable(fn) or list(inspect.signature(fn).parameters) != ["fence"]:
        raise RuntimeError(
            "this Humanize has no hmz.coganchor.fence.wrapper(fence), which the bwrap wall wraps; "
            "use sandbox docker, or the hmz version range qda-agent pins"
        )
    return hmz_fence


def shared_pid_refusal(pid: str, *, shared_node: SharedNodeAcceptance | None = None) -> str | None:
    """Why a `shared` pid form cannot be held on this node; None when it can.

    `shared` gives the agent's process the host's pid namespace, so a same-user signal or a ptrace reaches
    every other process of this user on the node: it is acceptable only on a node the run has to itself
    (`isolation.allocation.why_not_dedicated`). The wall asks it for every agent turn, and the `strict`
    level's preflight asks it before the run's clock starts.

    Args:
        pid: The requested pid form (`sched.sandbox.pid_form` decides what a request gets).
        shared_node: The run's operator acceptance (`isolation.shared_node`), or None. Naming one lifts
            THIS refusal only, for a run an operator is watching: the file wall, the network wall, the host
            proxy and the user namespace are built exactly as before. It comes from the run config, never
            from the environment, so nothing inside a turn can change it.

    Returns:
        The wall's refusal sentence when the node is not the run's and no acceptance is named; otherwise None.
    """
    if pid_form(pid) != "shared":
        return None
    why = why_not_dedicated()
    if why is None:
        return None
    if shared_node is not None:
        _log.warning(
            "sandbox: pid: shared on a node this run does NOT hold (%s). Accepted for this run by %s: %s. "
            "Only the dedication refusal is lifted: file and network walls unchanged.",
            why,
            shared_node.accepted_by,
            shared_node.reason,
        )
        return None
    return (
        "pid: shared gives the agent's process the host's pid namespace (this node refuses a fresh procfs), "
        f"so the run needs the node to itself, and it does not have it: {why}. Ask for a dedicated allocation "
        "(an exclusive node or container for this run)."
    )


@contextmanager
def gpu_wall(cfg: RunConfig) -> Iterator[None]:
    """While the run is under way, give its CLIs their own config dir and put every fenced turn inside `bwrap_argv`.

    The private config dir (`private_cli_config`) is set up at every isolation level; the bwrap wall only for
    `sandbox: bwrap` outside `isolation.level: none`, where no bwrap runs. Humanize's wrapper and the variables
    are restored on the way out however the run ends (an error, a cancellation, ctrl+c).

    Every turn's network is the sandbox's own (`--unshare-net`): its one way out is the host's proxy for the
    turn's hosts, behind a unix socket bound in (`NetBridge`), and the fence Humanize puts inside the wall is
    the filesystem's alone (`online=True`) -- the namespace holds the network, so no `pidfd_getfd`, which a
    DGX node refuses, is needed.

    The turn's root is empty and holds only what its fence reads and writes (`bwrap_argv(minimal=True)`): a
    read-only host root would still let it `connect()` to every pathname unix socket on the machine, the
    netns covering IP sockets alone.

    Paths that name the home through a link are spelled from the home's real path: bwrap cannot make the
    parents of a mount point under a link, and a cluster may reach every home through one. The link
    stays, so either spelling reaches the same file inside.

    Args:
        cfg: The run config: its home directory, `run_dir` and `sandbox.hide` are hidden.

    Raises:
        RuntimeError: Another run in this process holds the wall, or Humanize has no hook for it.
        ValueError: A CLI of the run the wall has no install trees for (`installs`).
        WallRefused: An agent turn cannot be held: the proxy's socket does not fit or is the agent's to reach,
            or `$TMPDIR` lies where the wall cannot show Humanize's per-turn dirs (`tmpdir_refusal`, `turn_dirs`).
    """
    # A none run walls nothing, so it starts no bwrap and probes nothing. It still gets the run's own config
    # dirs (a private dir per CLI whose sign-in is a link to the user's file, never the user's dir itself), so
    # the CLI cannot leave files in the user's config dir. A stub config of the wall's own tests may omit the
    # level: it then gets the bwrap wall, as a sandboxed run does.
    if getattr(getattr(cfg, "isolation", None), "level", None) == "none":
        # Nothing starts bwrap at this level. Each CLI still gets the run's own config dir here: a saved
        # settings file would otherwise name the endpoint the run's environment sets (see `private_cli_config`),
        # and a file the CLI writes would land in the user's directory.
        with private_cli_config(cfg):
            yield
        return
    if cfg.sandbox.name != "bwrap":
        with private_cli_config(cfg):
            yield
        return
    # The pid namespace the turn gets. `shared` is forced where the host refuses a fresh procfs: the host's
    # `/proc` is bound, and Humanize's cred reads it by pid, so a private namespace would have it resolve the
    # host's process of that number instead of the tracee. It is acceptable with the node to itself, or with
    # the run's own operator acceptance (`isolation.shared_node`), which logs what else is on the node.
    accepted = getattr(getattr(cfg, "isolation", None), "shared_node", None)
    pid = pid_form(cast(Bwrap, cfg.sandbox).pid)
    from qda_agent.isolation import pid_refusal  # the package's interface, not this module's internals

    if (problem := pid_refusal(cfg)) is not None:  # the rule and what it applies to: `isolation.pid_refusal`
        raise WallRefused(problem)
    if pid == "shared" and accepted is None:  # an accepted shared node already logged why it is not held
        _log.warning("sandbox: pid: shared (the host refuses a fresh procfs); this run holds its node")
    hmz_fence = wall_hook()
    home = Path.home()
    # bwrap cannot make the parents of a mount point under a link, and a cluster may spell every home twice
    # (one path a link to the other): a path that names the home through a link is spelled from the home's
    # real path. The link itself stays, so either spelling reaches the same file inside.
    root = Path(os.path.realpath(home))

    def spelled(p: str | Path) -> Path:
        q = Path(p)
        return root.joinpath(q.relative_to(home)) if q.is_relative_to(home) else q

    patterns = installs(cfg)
    # Humanize's per-turn dirs live in this process's temp dir, wherever `$TMPDIR` puts it; one the wall cannot
    # show is refused here, at the start, not by a turn failing minutes in (`tmpdir_refusal`).
    if (problem := tmpdir_refusal(cfg)) is not None:
        raise WallRefused(problem)
    temp = tempfile.gettempdir()
    temps = [spelled(Path(os.path.abspath(temp)))]  # noqa: PTH100 -- the spelling the wall binds at
    # The run's sign-in by file: each CLI process's wall mounts an empty tmpfs at `<run_dir>/signin/<name>`
    # and binds only the account dir's sign-in file read-only into it, which the CLI's config dir and live dir
    # link their own sign-in path to (`private_cli_config`). The account dir itself is never mounted, so no
    # other entry can be reached -- the ones it held at the start or any the host's sessions and refresh loop
    # add later. The bind is made when the process starts (a streaming CLI keeps it across goals), so the
    # flow recycles a warm process when the host file changed (`sign_in_mounted`); a refresh during a turn is
    # not seen until that turn ends. The account dir may sit
    # under the hidden home (a bind source is read from the host's own tree); it may not sit under a hide or
    # the run dir. Refused before the wall is taken, so a refused run leaves it to the next.
    sign_in: dict[Path, Path] = {}
    empty: list[Path] = []
    hides = [*cast(Bwrap, cfg.sandbox).hide, *([cfg.judge.custody_dir] if cfg.judge.custody_dir else [])]
    for name, src in sign_in_accounts(cfg).items():
        if src is None:
            continue
        for h in [*hides, cfg.run_dir]:
            if Path(os.path.realpath(src)).is_relative_to(Path(os.path.realpath(h))):
                raise WallRefused(
                    f"agent.backend.sign_in_dir {src} lies under {h}, which the wall hides or the run owns: "
                    "keep the account dir outside every hidden path and the run dir"
                )
        at = spelled(cfg.run_dir) / SIGNIN / name
        sign_in[at / sign_in_file(name)] = src / sign_in_file(name)
        empty.append(at)  # the tmpfs the file is bound into: nothing else of the dir is shown
    # the interpreter that runs `hmz internal fence` inside the wall
    python = [
        spelled(Path(x) / part) for x in dict.fromkeys((sys.prefix, sys.base_prefix)) for part in PYTHON_PARTS
    ]
    if not _WALL.acquire(blocking=False):
        raise RuntimeError("another run in this process holds the GPU wall; run one search per process")
    inner = hmz_fence.wrapper
    early = [root, spelled(cfg.run_dir), *runtime_dirs(os.environ)]
    # A private path is named by every spelling the turn may use: the config's (a home alias included, both
    # as written and spelled through the home's real path) and its fully resolved target. They are masked
    # last: mounted before the binds, a later parent bind puts the tree back.
    hidden: list[Path] = []
    for h in [*cast(Bwrap, cfg.sandbox).hide, *([cfg.judge.custody_dir] if cfg.judge.custody_dir else [])]:
        for s in (Path(h), spelled(h), Path(os.path.realpath(h)), Path(os.path.realpath(spelled(h)))):
            if s not in hidden:
                hidden.append(s)
    unset = inherited_secrets(os.environ, sign_in_tokens(cfg))
    dropped: set[str] = set()
    gates = _hook_gates()
    links = home_links(home)  # the turn's paths keep the link's spelling (see `bwrap_argv`'s `links`)
    links |= interpreter_links(sys.executable, home, sys.base_prefix)  # a uv-managed Python's version link

    lives = {name: spelled(d) for name, d in live_dirs(cfg).items()}
    spec = json.dumps({CLI_CONFIG[name][0]: str(d) for name, d in lives.items()})
    bridge: NetBridge | None = None  # made with the first turn: the hosts are the fence's
    checked = False  # the private paths are checked against the wall's own mounts once, at the run's start

    def walled(fence: Fence) -> list[str]:
        nonlocal bridge, checked
        # Read-only inside: what Humanize builds a session's home from and drops it into, on the host, with
        # links followed (homes.build / Home.drop): the home itself, the run's template CLI dirs and the
        # epic's session store. The CLI works in its live dir instead (`live_state`), which the host never
        # touches.
        frozen = [spelled(agent_state(cfg.workdir)), *(spelled(d) for d in cli_config_dirs(cfg).values())]
        sessions = [str(spelled(w)) for w in fence.write if _humanize_home(w)]
        keep = [spelled(cfg.workdir), *lives.values()]
        fence = fence.granting(write=[str(d) for d in lives.values()])
        fence = dataclasses.replace(
            fence,
            read=tuple(str(spelled(r)) for r in fence.read),
            write=tuple(str(spelled(w)) for w in fence.write),
        )
        # The flow's own hook endpoints (`hmz internal hook --at <dir>/hook.sock`, made by this process
        # under the temp dir the wall replaces): without them the CLI's PreToolUse never reaches the flow.
        # Read-only: connecting to the socket still works, but the agent cannot remove or replace it (and
        # so silence or fake its own tool log).
        hooks = [str(spelled(d)) for d in gates.dirs if d.is_dir()]
        scratch = [str(spelled(t)) for t in host_scratch(cfg, fence.tmp)]
        # Humanize's per-turn dirs are not the user's state: with `$TMPDIR` under the hidden home (`~/tmp`)
        # they are bound back by their exact paths -- the scratch and hook dirs this process recorded, the
        # session homes the fence lists -- and nothing else of the home comes back (`turn_dirs`).
        turn = turn_dirs([*scratch, *hooks, *sessions], early, temp, temps)
        ro = [*hooks, *sessions, *(str(p) for p in frozen)]
        write = [
            *fence.write,
            *scratch,
            *(str(p) for p in keep if str(p) not in fence.write),
        ]
        write = [w for w in write if not _overlaps(w, ro)]
        write = home_binds(write, root, dirs=[*keep, *turn], dropped=dropped)
        read = home_binds(
            [*fence.read, *ro],
            root,
            dirs=[*keep, *frozen, *turn],
            patterns=patterns,
            narrow=python,
            dropped=dropped,
        )
        # The turn's one way out: a unix socket to the host's proxy for this fence's hosts, bound at a fresh
        # tmpfs path the relay `live_state` starts finds (a mount point the agent can neither move nor replace).
        if bridge is None:
            bridge = NetBridge()
        try:
            sock = bridge.socket(fence.hosts)
        except ValueError as why:
            raise WallRefused(str(why)) from why
        way_out = f"/tmp/{sock.stem}-wall.sock"
        # The toolset's read-only entries (the wiki): host-owned sources outside the workdir, bound over their
        # mount points in it, which are made real (not links) first: a planted link never redirects a bind or
        # stops the turn from starting. So is the host's private socket directory, when the run has one: at
        # `work/.qda`, where the commands look, and busy, so the agent can neither move nor replace it. The
        # proxy socket is not under a writable root: `bwrap_argv` makes its mount point in the fresh `/tmp`.
        # The shell's read-only entries come from the run's workspace manifest, the same record TASK.md and
        # the contamination scan read (a bind the manifest does not name is not mounted).
        mounts = {spelled(k): spelled(v) for k, v in cfg.manifest().agent_binds().items()}
        # The turn's workdir may be named by the config's spelling, the home's real spelling or either one's
        # resolved target (a home alias can hide a second hop, `~/site/scratch` -> `/site/scratch/...`), and the
        # fence binds it at the spelling it was given. A mount on one spelling is not seen through another, so
        # a receipted build is mounted at every spelling the wall binds the workdir at (bwrap refuses to start
        # when it cannot make a mount point's parents, so a spelling nothing is mounted at cannot be used).
        candidates = {Path(cfg.workdir), spelled(cfg.workdir)}
        candidates |= {Path(os.path.realpath(p)) for p in candidates}
        bound = {Path(os.path.abspath(p)) for p in (*write, *read)}  # noqa: PTH100
        roots = [w for w in candidates if Path(os.path.abspath(w)) in bound]  # noqa: PTH100
        # Receipted build outputs are mounted read-only over their published paths, after the writable
        # workdir: the turn can read and copy a build but cannot change it in place.
        receipts: dict[Path, Path] = {}
        for dst, tree in published_entries(cfg.run_dir, cfg.workdir):
            if tree.is_dir():
                for w in roots:
                    receipts.setdefault(w / Path(dst).relative_to(cfg.workdir), tree)
        needed = {
            **mounts,
            **{spelled(k): spelled(v) for k, v in cfg.socket_binds().items()},
        }
        binds = mount_points({**needed, **receipts}, roots, deep=receipts)
        # A hide over a receipted build masks it (mounts are made before masks); only a hide over what the
        # turn really needs refuses the run, as before.
        if (hit := hidden_covers(hidden, [cfg.run_dir, *keep, *frozen, *read, *needed])) is not None:
            raise WallRefused(
                f"sandbox.hide {hit[0]} covers {hit[1]}, which the turn needs (the run dir, its workdir or "
                "live CLI dirs, or a readable bind); hide a narrower path"
            )
        specs: list[tuple[str, int, int]] = []
        argv = bwrap_argv(
            read,
            write,
            hidden,
            keep=[*keep, *frozen, *turn],
            unset=unset,
            setenv={"TZ": cfg.toolset.env.get("TZ", "UTC")},
            binds={**binds, Path(way_out): sock, **sign_in},
            tmpfs=empty,
            pid=pid,
            minimal=True,  # the fence's system minimum only: no host tree, so no foreign pathname socket
            links=links,
            hide_early=early,
            hidden_checks=specs,
        )
        private = rpc.private_dir_of(cfg.run_dir)
        # Both host socket dirs must stay out of a writable bind, or the agent could rename one and leave a link
        # there, which a bind would follow: the run's private socket directory (when it has one), and the
        # bridge's (the proxy socket's host dir). The first keeps its wording; the proxy's is new with the
        # network namespace, so a refused run names which one.
        checks = [] if private is None else [(f"the private socket directory {private}", private.parent)]
        checks.append((f"the proxy socket directory {bridge.base.parent}", bridge.base.parent))
        for named, base in checks:
            if (via := exposed(argv, base)) is not None:
                raise WallRefused(f"{named} is writable in the wall (through {via})")
        # Never a silent drop: each per-turn dir the turn needs is shown at its own path, the scratch writable.
        for d, writable in [*((d, True) for d in scratch), *((d, False) for d in [*hooks, *sessions])]:
            if os.path.lexists(d) and not os.path.islink(d) and not _shown(argv, d, writable=writable):
                raise WallRefused(
                    f"Humanize's per-turn dir {d} (TMPDIR={temp}) would not be "
                    f"{'writable' if writable else 'visible'} in the wall: set TMPDIR to a directory of your "
                    "own outside every hidden path (for example /tmp or a scratch dir)"
                )
        if not checked and specs:
            _verify_hidden(argv, specs)
            checked = True
        # The sign-in file this process is about to hold (stat only): a streaming CLI keeps its mounts across
        # goals, so this, not the goal, is what its credential is (`sign_in_mounted`); the flow recycles the
        # process when the host refreshes the file after this.
        sign_in_mount(cfg)
        live = [*LIVE_PYTHON, LIVE_SOURCE, spec, way_out, "--"]
        # The namespace holds the network, so the fence Humanize puts inside is the filesystem's alone. The
        # whole turn runs under the guard, which ends it with no process left behind.
        turn = [*LIVE_PYTHON, TURN_SOURCE, "--", *live, *inner(dataclasses.replace(fence, online=True))]
        return private_bwrap_argv([*argv, *turn], cfg.run_dir)

    try:
        with private_cli_config(cfg), _home_guard(cfg), _network_held(cfg):
            setattr(hmz_fence, "wrapper", walled)  # noqa: B010 - undeclared attribute on hmz's module
            yield
    finally:
        setattr(hmz_fence, "wrapper", inner)  # noqa: B010 - undeclared attribute on hmz's module
        sign_in_unmount(cfg)  # no wall of this run is up once its wrapper is gone
        if bridge is not None:
            bridge.stop()
        gates.undo()
        _WALL.release()


def _overlaps(path: str, others: Sequence[str]) -> bool:
    """Whether `path` is one of `others`, under one, or above one, compared as absolute normalised paths.

    `os.path.abspath` drops `.`, `..` and repeated slashes but keeps links, the same rule as the config's
    paths: two spellings of one dir match, and a link is not swapped for its target.
    """
    p, qs = Path(os.path.abspath(path)), [Path(os.path.abspath(q)) for q in others]  # noqa: PTH100 (no resolve())
    return any(p.is_relative_to(q) or q.is_relative_to(p) for q in qs)


class WallRefused(RuntimeError):
    """The wall will not start a turn, and no retry would change that: the run must end with this error.

    Humanize wraps it in a retried `HarnessError`; the flow finds it in the cause chain and ends the run.
    """


class UnsafeHome(WallRefused):
    """A dir Humanize is about to build a home from or drop one into holds what the run never put there."""


def unsafe_entries(root: Path, inside: Sequence[Path]) -> list[str]:
    """What under `root` Humanize must not work on.

    That is a link that leads out of `inside`, or a node that is not a regular file, a dir or a link. Walked
    with `lstat` only (no link is followed); `[]` for a missing root.
    """
    allowed = [Path(os.path.realpath(r)) for r in inside]
    bad: list[str] = []
    stack = [root]
    while stack:
        top = stack.pop()
        try:
            entries = list(os.scandir(top))
        except FileNotFoundError:
            continue
        except OSError as e:
            bad.append(f"{top}: {e.strerror}")
            continue
        for e in entries:
            st = e.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                if not any(Path(os.path.realpath(e.path)).is_relative_to(r) for r in allowed):
                    bad.append(f"{e.path} -> {Path(e.path).readlink()}")
            elif stat.S_ISDIR(st.st_mode):
                stack.append(Path(e.path))
            elif not stat.S_ISREG(st.st_mode):
                bad.append(f"{e.path}: not a file or a dir")
    return bad


@contextmanager
def _home_guard(cfg: RunConfig) -> Iterator[None]:
    """Before Humanize builds a home (`homes.build`) or drops one (`homes.Home.drop`), check its sources.

    An extra layer under the read-only binds of `gpu_wall`: the run's template CLI dirs and the epic's session
    store must hold no link out of the run's own dirs and nothing but files, dirs and links. If they do, the
    step is refused (`UnsafeHome`, logged) and nothing is repaired. Both hooks exist in hmz 0.1.0 to 0.2.2. No
    agent process is alive during the walk: every turn's processes are gone when its fence exits (see
    `gpu_wall`), and qsched jobs never see these dirs.
    """
    from hmz.coganchor import homes

    sources = [*cli_config_dirs(cfg).values(), agent_state(cfg.workdir)]
    inside = [cfg.run_dir, agent_state(cfg.workdir)]

    def check(step: str) -> None:
        bad = [b for root in sources for b in unsafe_entries(root, inside)]
        if bad:
            _log.error("refused Humanize's home %s: %s", step, bad)
            raise UnsafeHome(f"Humanize's home {step} refused: {len(bad)} unsafe entr(ies): {bad[:3]}")

    real_build, real_drop = homes.build, homes.Home.drop

    def build(*args: Any, **kw: Any) -> Home:  # the wrapped signature, forwarded unchanged
        check("build")
        return real_build(*args, **kw)

    def drop(self: Home) -> None:
        check("drop")
        real_drop(self)

    homes.build, homes.Home.drop = build, drop  # type: ignore[assignment]
    try:
        yield
    finally:
        homes.build, homes.Home.drop = real_build, real_drop  # type: ignore[assignment]


#: A class that says nothing of its own `natively`: restoring it means removing what the wall set.
_UNSET = object()


def _network_inside(self: Any, fence: Fence) -> Fence:
    """What a driver answers while the wall is up: the namespace holds the network, so the fence is the filesystem's.

    The method :func:`_network_held` puts in place of each driver's `natively`, called by Humanize with the
    fence the agent is held to. `self` is not read: the answer is the same for every CLI.
    """
    del self
    return fence.without(network=True)


@contextmanager
def _network_held(cfg: RunConfig) -> Iterator[None]:
    """While the wall is up, answer each agent driver's `natively` for the network bwrap's namespace holds.

    Humanize asks a driver what its CLI holds natively (`AgentBase._fences`) and refuses the session where
    what is left needs a way of cutting the network this machine has not got -- Landlock ABI 4 with seccomp,
    and a descriptor taken out of the CLI's processes, which a DGX node refuses. Inside the wall the network
    is already held by `--unshare-net`, whose one way out is the host's proxy (`NetBridge`), so the driver
    is answered as if it held the network itself: only the fence it is handed changes, and nothing is left
    for `hmz internal fence` to cut but the filesystem.

    The fence in the agent's own config is left as it was written, so a CLI that is told to take its web
    tools off for an offline fence (`ClaudeCodeAgent._offline`, `CodexAgent`'s apps) goes on hearing it.

    Args:
        cfg: The run config, whose CLIs are the drivers answered for.
    """
    from hmz.coganchor.agents import driver

    said: dict[type, Any] = {}
    try:
        for name, _ in clis(cfg):
            try:
                cls, _ = driver(name)
            except KeyError:
                continue  # `installs` has already refused a CLI this wall cannot run
            if cls not in said:
                said[cls] = cls.__dict__.get("natively", _UNSET)
                cls.natively = _network_inside  # type: ignore[method-assign]
        yield
    finally:
        for cls, real in said.items():
            if real is _UNSET:
                with contextlib.suppress(AttributeError):
                    del cls.natively
            else:
                cls.natively = real  # type: ignore[method-assign]


class _Gates:
    """The directories of the hook sockets Humanize's gates serve in this process, while the wall is up."""

    def __init__(self) -> None:
        self.dirs: list[Path] = []
        self.undo = lambda: None


def _hook_gates() -> _Gates:
    """Record each hook socket Humanize's `Gate.address` makes, so the wall binds exactly those back."""
    gates = _Gates()
    try:
        from hmz.coganchor.agents import hooks as hmz_hooks
    except ImportError:
        return gates
    gate = getattr(hmz_hooks, "Gate", None)
    real = getattr(gate, "address", None)
    if real is None:
        return gates

    def address(self: object) -> str:
        at = real(self)
        if at and (d := Path(at).parent) not in gates.dirs:
            gates.dirs.append(d)
        return at

    gate.address = address  # type: ignore[union-attr]

    def undo() -> None:
        gate.address = real  # type: ignore[union-attr]

    gates.undo = undo
    return gates
