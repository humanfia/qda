"""The agent's sandbox, as Humanize is asked to hold it (see this package).

Two layers, outermost first:

- **bwrap** (`sandbox: {name: bwrap}`, the default): the agent's CLI runs in a mount namespace with a fresh `/dev`
  (no GPU nodes), `/` read-only, the home directory and the run dir hidden under empty tmpfs mounts, and
  only the paths the fence grants mounted back (of the home, only an allowlist: `home_binds`); its own
  PID, IPC, UTS and network namespaces. The network namespace has loopback and no route out: the CLI
  reaches its model's API through a unix socket bound into it, to a host-side proxy for that turn's hosts
  (`wall.NetBridge`).
- **the Humanize fence** (Humanize `Permission`): Landlock (Seatbelt on macOS) on paths -- the workdir
  read/write, the system minimum readable -- and, where the sandbox does not cut the network itself, a
  proxy that passes only the model's API host. Inside the bwrap wall it is the filesystem's alone.

Humanize puts its fence outermost of every turn it spawns and has no hook for a layer outside it; bwrap
cannot start inside the fence (Landlock forbids `mount`). So while a run is under way, `gpu_wall` wraps
Humanize's fence wrapper (`hmz.coganchor.fence.wrapper`, looked up at each turn) so that the command it
spawns is `bwrap ... -- hmz internal fence ... -- <CLI>`. The bwrap mounts are drawn from that very
fence, so they never grant more than it does.

The commands in `bin/` reach the host through the RPC socket; Landlock and Seatbelt do not govern connecting
to a Unix socket, and the socket's directory is mounted back, so neither layer needs a hole for it.

This module holds the static checks a run must pass before it starts (`check`) and the fence's settings;
`wall` builds the bwrap wall, `cli_dirs` says where each CLI's dirs are, and `cli_signin` decides what signs
it in.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hmz.flows import Permission, PermissionKind

from qda_agent import rpc
from qda_agent.sandbox_settings import Bwrap, Sandbox

from . import wall as _wall
from .bwrap import pid_form, published_entries, python_roots, register_published
from .cli_dirs import CLI_CONFIG, agent_state, cli_config_dirs, live_dirs, token_vars
from .cli_signin import (
    CODEX_PROVIDER_FIX,
    SIGNIN,
    SignInRefused,
    check_sign_in,
    long_run_fix,
    private_cli_config,
    sign_in_accounts,
    sign_in_conflicts,
    sign_in_dirs,
    sign_in_mount,
    sign_in_mounted,
    sign_in_refreshed,
    sign_in_report,
    sign_in_tokens,
    sign_in_unmount,
    sign_in_witness,
)
from .wall import (
    HOST_SCRATCH,
    INSTALLS,
    LIVE_PYTHON,
    LIVE_SOURCE,
    PYTHON_PARTS,
    TEMPORARY,
    UnsafeHome,
    WallRefused,
    bwrap_argv,
    exposed,
    gpu_wall,
    home_binds,
    inherited_secrets,
    installs,
    runtime_dirs,
    shared_pid_refusal,
    tmpdir_refusal,
    unsafe_entries,
    wall_hook,
)

if TYPE_CHECKING:
    from qda_agent.config import RunConfig


__all__ = [
    "SIGNIN",
    "CLI_CONFIG",
    "HOST_SCRATCH",
    "INSTALLS",
    "LIVE_PYTHON",
    "LIVE_SOURCE",
    "PYTHON_PARTS",
    "SANDBOX",
    "TEMPORARY",
    "SignInRefused",
    "UnsafeHome",
    "CODEX_PROVIDER_FIX",
    "WallRefused",
    "agent_state",
    "bwrap_argv",
    "check",
    "check_sign_in",
    "cli_config_dirs",
    "env_spec",
    "exposed",
    "gpu_wall",
    "hold_problem",
    "home_binds",
    "inherited_secrets",
    "installs",
    "live_dirs",
    "permission",
    "pid_refusal",
    "private_cli_config",
    "published_entries",
    "register_published",
    "runtime_dirs",
    "shared_node_record",
    "pid_form",
    "shared_pid_refusal",
    "sign_in_accounts",
    "sign_in_conflicts",
    "sign_in_dirs",
    "sign_in_mount",
    "sign_in_mounted",
    "sign_in_refreshed",
    "sign_in_report",
    "long_run_fix",
    "sign_in_tokens",
    "sign_in_unmount",
    "sign_in_witness",
    "tmpdir_refusal",
    "token_vars",
    "unsafe_entries",
    "wall_hook",
]

#: Workdir read/write; the rest of home and the system unreadable (beyond the CLI's minimum); network to
#: the model's API only.
SANDBOX = Permission(
    local=PermissionKind.ALL,
    user=PermissionKind.NONE,
    system=PermissionKind.NONE,
    online=PermissionKind.NONE,
)


def permission(iso: Sandbox) -> Permission:
    """The permission the worker's sessions run under."""
    return Permission() if iso.name == "none" else SANDBOX


def env_spec(iso: Sandbox, workdir: Path) -> str:
    """The Humanize environment spec (`-e workspace=...`) the agent works in.

    Args:
        iso: The sandbox settings.
        workdir: The agent's workdir (absolute).

    Returns:
        `local/<workdir>`.
    """
    del iso
    workdir = workdir.resolve()
    return f"local{workdir}"


def pid_refusal(cfg: RunConfig) -> str | None:
    """Why the run cannot hold the pid form its wall asks for, or None: the wall's rule, in the wall's words.

    The rule applies exactly when a wall is put up (`sandbox: bwrap` above `isolation.level: none`), and the
    three callers ask it at their own moment: `wall.gpu_wall` as the wall goes up, `levels.strict`'s preflight
    before the run's clock, and `cli.start_refusal` before the run creates anything. One place decides what
    the rule applies to, so a run can neither skip it nor hear it twice.
    """
    level = getattr(getattr(cfg, "isolation", None), "level", None)  # a stub config (a level's own test)
    if cfg.sandbox.name != "bwrap" or level == "none":
        return None
    from qda_agent.sandbox_settings import Bwrap

    accepted = getattr(getattr(cfg, "isolation", None), "shared_node", None)
    return shared_pid_refusal(cast(Bwrap, cfg.sandbox).pid, shared_node=accepted)


def hold_problem(cfg: RunConfig) -> str | None:
    """Why this machine cannot hold the run's agent CLI to its fence, or None when it can.

    The agent's CLI does not hold the network natively (on Linux), so a `fence` run needs this machine to cut
    the network from outside (Landlock ABI 4 with seccomp, or Seatbelt). A `bwrap` run does not: the wall's
    network namespace holds the network, and the fence it puts inside a turn is the filesystem's alone
    (`isolation.wall`), so only the wall has to start. A run is refused here only where neither holds.

    Args:
        cfg: The run config: its sandbox and its agent's CLI.

    Returns:
        One sentence saying why the agent cannot be held (for `qda-agent run` and `qda doctor --search`), or
        None. An unsandboxed run, a container, and an in-process driver (a backend with no CLI) are held by
        their own process and pass.
    """
    iso = cfg.sandbox
    if cfg.isolation.level == "none" or cfg.agent.backend.cli is None:
        return None
    from hmz.coganchor.fence import enforceable

    if iso.name == "bwrap":
        missing = []
        if shutil.which("bwrap") is None:
            missing.append("no bubblewrap (`bwrap`) on PATH")
        if not enforceable(net=False):  # the filesystem fence the wall puts inside a turn
            missing.append("no filesystem fence (Landlock ABI 4, Linux 6.7 or later)")
        try:
            wall_hook()
        except RuntimeError as why:
            missing.append(str(why))
        if not missing:
            return None
        reason = "the bwrap wall cannot start: " + "; ".join(missing)
        if enforceable(net=True):  # the native fence alone would hold it: no wall needed
            return reason + "; this machine can hold the native network fence, so use sandbox fence"
        return (
            reason
            + ", and this machine cannot hold the native network fence either: use sandbox docker, or a "
            "machine that can"
        )
    return None


def shared_node_record(cfg: RunConfig) -> dict[str, str | None] | None:
    """The run's shared-node acceptance, with what else the wall finds on the node; None without one.

    For `result.json` and `END-REPORT.md`: who accepted a shared node and why (`isolation.shared_node`), and
    the wall's own scan at the moment this is read -- other jobs and process names and counts, never command
    lines. A run that names no acceptance records nothing at all, so its result files are unchanged.

    Args:
        cfg: The run config.

    Returns:
        `{"accepted_by", "reason", "other_work"}`, or None when the run names no acceptance.
    """
    accepted = cfg.isolation.shared_node
    if accepted is None:
        return None
    return {
        "accepted_by": accepted.accepted_by,
        "reason": accepted.reason,
        "other_work": _wall.why_not_dedicated(),
    }


#: Where a sandboxed run's directory goes instead (`run_dir_in_tmp`); `qda-agent search` and `qda doctor --search`
#: print it as the refusal's fix.
RUN_DIR_FIX = (
    "put run_dir outside /tmp and /var/tmp (qda-agent prepare --out DIR on your own storage, or "
    "--set run_dir=DIR); keep the agent's builds node-local with --set work_dir=/var/tmp/<fresh dir>"
)


#: What a run refused by the wall's pid rule (`pid_refusal`) does instead; `qda-agent search` and `qda doctor
#: --search` print it as the fix.
PID_FIX = (
    "run on a node no other process of yours uses, or set isolation.shared_node with both accepted_by "
    "(who accepts the shared node) and reason (docs/search-config.md#isolation)"
)


def run_dir_in_tmp(cfg: RunConfig) -> str | None:
    """Why a sandboxed run's directory cannot be where it is, or None.

    At ``isolation.level`` sandbox or strict the agent can read and write the temporary directories, so a
    run directory there would expose the ledger, FINAL.json and the tool-call log to it. `check` raises this
    line, the search's start refuses on it (`cli.START_GUARDS`) and `qda doctor --search` prints it.
    """
    if cfg.isolation.level == "none":
        return None
    run = cfg.run_dir.resolve()
    for tmp in ("/tmp", "/var/tmp", tempfile.gettempdir()):
        if run.is_relative_to(Path(tmp).resolve()):
            return (
                f"run_dir {run} is under {tmp}, which the agent can read and write at isolation.level "
                f"{cfg.isolation.level}: it would reach the run's ledger, FINAL.json and tool-call log"
            )
    return None


def check(cfg: RunConfig) -> None:
    """Refuse a sandbox that would not hold.

    Raises:
        ValueError: In any mode, a `judge.gencheck_prompts` the agent's jobs read. In a sandboxed mode: an agent
            CLI whose config dir `CLI_CONFIG` cannot move, or a `run_dir` under a temporary directory (the fence
            makes it writable, exposing the ledger, FINAL.json and the tool-call log). For `sandbox: bwrap`: no
            `bwrap`, an agent CLI whose install trees `INSTALLS` does not name, a `$TMPDIR` the wall cannot show
            Humanize's per-turn dirs from (`tmpdir_refusal`), or a `judge.gencheck_prompts` the wall would show. For `sandbox: fence`: a machine with GPUs, unless `allow_visible_gpus`. For `fence`
            and `bwrap`, a run dir too long for the socket whose private socket directory the agent could write
            (`_private_socket_unwritable`; always under `fence`), and for `bwrap` a private base the agent could
            write at all: the judge's jobs name their temp dirs by links made there (`judge.jobtmp`).
    """
    iso = cfg.sandbox
    _prompts_unread(cfg)
    if cfg.isolation.level == "none":
        return
    cli_config_dirs(cfg)
    if (why := run_dir_in_tmp(cfg)) is not None:
        raise ValueError(why)
    if iso.name == "bwrap":
        if shutil.which("bwrap") is None:
            raise ValueError("sandbox bwrap needs bubblewrap (`bwrap`) on PATH; or use docker")
        wall_hook()
        installs(cfg)
        if (why := tmpdir_refusal(cfg)) is not None:  # Humanize's per-turn dirs in $TMPDIR, out of reach
            raise ValueError(why)
        _prompts_hidden(cfg)
    if iso.name == "bwrap":  # the judge's jobs name their temp dirs by links in the private base (`jobtmp`)
        _private_socket_unwritable(cfg)


def _private_socket_unwritable(cfg: RunConfig) -> None:
    """Refuse a private socket directory the agent could rename or replace.

    A job binds that directory by path, so an agent that can write its base could leave a link in its place, which
    the bind would follow to any host directory. The judge's jobs name their temp dirs by links made under the
    same base (`judge.jobtmp`): an agent that could replace one would receive a job's scratch. `fence` has no
    mount namespace and writes the temporary directories, `/dev/shm`, `$TMPDIR` and the workdir: a fenced run is
    refused whenever its socket does not fit.
    `bwrap` is refused when its wall, drawn from the real fence (`Fence.of`), would bind the base writable.
    """
    from hmz.coganchor.fence import Fence

    home = Path.home()
    f = Fence.of(
        local=str(SANDBOX.local),
        user=str(SANDBOX.user),
        system=str(SANDBOX.system),
        online=False,
        workdir=cfg.workdir,
        home=home,
    )
    early = [home, cfg.run_dir, *runtime_dirs(os.environ)]
    hidden = [
        p for h in cast(Bwrap, cfg.sandbox).hide for p in (h, Path(os.path.realpath(h)))
    ]  # masked last by the wall, so they do not drop a writable path here
    write = home_binds([*f.write, str(cfg.workdir)], home, dirs=[cfg.workdir])
    argv = bwrap_argv([], write, hidden, keep=[cfg.workdir], hide_early=early)
    if (via := exposed(argv, rpc.private_base())) is not None:
        raise ValueError(
            f"the judge jobs' short temp-dir links (and the RPC socket of a run_dir too long for one) would be "
            f"made under {rpc.private_base()}, which the agent can write (through {via}): set $XDG_RUNTIME_DIR "
            "to a directory the agent cannot write"
        )


def _hidden_prompts(cfg: RunConfig) -> Path | None:
    """The file of a long-generation prompt set the agent may not read; None for a public set or no check."""
    dataset = cfg.judge.prompt_set()
    if dataset is None or dataset.public or "gencheck" not in cfg.toolset.commands(cfg.run):
        return None
    return dataset.local()


def _prompts_unread(cfg: RunConfig) -> None:
    """Refuse a long-generation prompt set that lies under a path the agent's jobs read (in any sandbox mode).

    The jobs read the model, the paths `RunConfig.read_paths` binds (the baselines included), the toolset's
    interpreter's installation and the host interpreter's roots.
    """
    src = _hidden_prompts(cfg)
    if src is None:
        return
    q = Path(os.path.realpath(src))
    model = [Path(cfg.model)] if Path(cfg.model).exists() else []
    python = [cfg.toolset.python.parent.parent] if cfg.toolset.python is not None else []
    readable, binds = cfg.read_paths()
    read = [
        *model,
        *readable,
        *binds.values(),
        *python,
        *python_roots(),
    ]
    if hit := next((r for r in read if q.is_relative_to(os.path.realpath(r))), None):
        raise ValueError(
            f"judge.gencheck_prompts {src} lies under {hit}, which the agent's jobs read: keep it outside every "
            "path bound into them"
        )


def _prompts_hidden(cfg: RunConfig) -> None:
    """Refuse a long-generation prompt set the wall would show: it shows `/` but the home, the run dir, `hide`."""
    src = _hidden_prompts(cfg)
    if src is None:
        return
    q = Path(os.path.realpath(src))
    if not any(q.is_relative_to(os.path.realpath(h)) for h in (Path.home(), *cast(Bwrap, cfg.sandbox).hide)):
        raise ValueError(
            f"judge.gencheck_prompts {src} would be readable from the agent's shell: keep it under the home, "
            "or list its directory in sandbox.hide"
        )
