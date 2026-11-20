"""The run's sign-in: which credential each agent CLI uses, whether it will outlast the run, and the config dir
that carries it (see this package).

Humanize builds each session's config dir from "the user's" dir and writes back what the CLI changed, so
pointing it at the real `~/.claude` or `~/.codex` would let the agent plant instructions for the user's later
sessions. `private_cli_config` points it at `<run_dir>/cli/<name>` for the length of the run, at every
isolation level. The dir holds an empty settings file and, only for a CLI that signs in by a file, a link to
that file: at `none` (no wall) to the user's own file, so the CLI refreshes it in place; at a walled level
into the wall's read-only mount of it. This module also decides what signs the run in, and refuses a walled
run whose sign-in would expire before the deadline.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qda_quant.exit import ENVIRONMENT, USAGE

from .cli_dirs import CLI_CONFIG, cli_config_dirs, clis, live_dirs, sign_in_file

if TYPE_CHECKING:
    from qda_agent.config import RunConfig

_log = logging.getLogger(__name__)

#: The run dir's subdirectory where the wall mounts each CLI's account dir read-only (`<run_dir>/signin/<name>`).
SIGNIN = "signin"


class SignInRefused(ValueError):
    """The run's sign-in cannot last it: refused at the start, before any turn.

    `code` is the exit status (`qda_quant.exit`): `ENVIRONMENT` where nothing signs this run in -- the
    credential the config names is not there, or no account or token variable names one -- so the same command
    works once the machine has a sign-in; `USAGE` where the config itself names two sign-ins, or a directory
    that is not one.
    """

    def __init__(self, why: str, *, code: int = USAGE) -> None:
        super().__init__(why)
        self.code = code


def sign_in_dirs(cfg: RunConfig) -> dict[str, Path | None]:
    """The `agent.backend.sign_in_dir` of each CLI the run uses, by CLI name.

    Raises:
        SignInRefused: Two agents of the same CLI name different dirs (one of them None counts as a different
            choice): the run would silently copy whichever it met first.
    """
    out: dict[str, Path | None] = {}
    for backend in (cfg.agent.backend, *cfg.agent.fallbacks):
        if backend.cli is None:
            continue
        chosen = getattr(backend, "sign_in_dir", None)
        if backend.name in out and out[backend.name] != chosen:
            raise SignInRefused(
                f"agent.backend.sign_in_dir: the run's {backend.name} agents name different dirs; give one"
            )
        out[backend.name] = chosen
    return out


def _config_dirs(cfg: RunConfig) -> dict[str, Path]:
    """The run's own config dir of each CLI the run uses, by name.

    A sandboxed run refuses a CLI whose config dir cannot be moved (`cli_config_dirs`). At
    `isolation.level: none` a CLI outside `CLI_CONFIG` (a backend with no config dir variable) is left alone
    instead: there is no wall that the user's own directories would defeat, and today's unconfined run must
    not start refusing it.
    """
    if getattr(getattr(cfg, "isolation", None), "level", None) == "none":
        return {name: cfg.run_dir / "cli" / name for name, _ in clis(cfg) if name in CLI_CONFIG}
    return cli_config_dirs(cfg)


def sign_in_tokens(cfg: RunConfig) -> frozenset[str]:
    """The token variables (`CLI_CONFIG`) of the CLIs this run uses: each signs its CLI in, so the wall keeps it.

    The wall drops the flow's inherited `CLAUDE_CODE_*` names as launcher secrets; a token the run itself needs
    is not one of them. Only the variables `CLI_CONFIG` names for a CLI of this run are kept -- never a prefix,
    and never a variable of a CLI the run does not use.
    """
    return frozenset(t for name in _config_dirs(cfg) for t in CLI_CONFIG[name][3])


def sign_in_accounts(cfg: RunConfig) -> dict[str, Path | None]:
    """The account dir each CLI of the run signs in from by a file, by CLI name; None where it does not.

    A CLI signs in by a file unless its backend names an account (the account's provider signs in on the host)
    or one of its token variables is set (`CLI_CONFIG`). The dir is `agent.backend.sign_in_dir` when the run
    names one (whether it holds the sign-in file or not: `private_cli_config` refuses one that does not), else
    the CLI's default dir (`~/.claude`, `~/.codex`) when it holds the sign-in file. Nothing here reads the file.
    """
    accounts = dict(clis(cfg))
    configured = sign_in_dirs(cfg)
    out: dict[str, Path | None] = {}
    for name in _config_dirs(cfg):
        sign_in = CLI_CONFIG[name][2]
        if sign_in is None or accounts[name] or any(os.environ.get(t) for t in CLI_CONFIG[name][3]):
            out[name] = None  # a backend with no sign-in file (`dsh`) never signs in by one
        elif (chosen := configured.get(name)) is not None:
            out[name] = chosen
        else:
            default = Path.home() / Path(sign_in).parent
            out[name] = default if (default / Path(sign_in).name).is_file() else None
    return out


#: The model provider `codex_token_provider` names in the run's codex config.
CODEX_TOKEN_PROVIDER = "qda-openai"
#: Where that provider sends a turn when `OPENAI_BASE_URL` is not set: codex's own default.
OPENAI_BASE = "https://api.openai.com/v1"


def codex_token_provider(environ: Mapping[str, str]) -> str | None:
    """The `config.toml` that signs codex in by its token variable, or None when none is set.

    `codex app-server` (what Humanize drives) signs in only from `auth.json` -- what `codex login
    --with-api-key` writes -- and reads neither `OPENAI_API_KEY` nor `CODEX_API_KEY` (only `codex exec` reads the
    latter), nor `OPENAI_BASE_URL`: a run with only the variable set fails its first turn with `401 Missing
    bearer`. A model provider with `env_key` makes codex read the key from the environment at each request, so
    the key is never written to a file; its `base_url` is `OPENAI_BASE_URL` when set (the token's endpoint,
    which `flow._endpoint_env` keeps), else the official API. `OPENAI_API_KEY` is used when both are set.
    """
    var = next((t for t in CLI_CONFIG["codex"][3] if environ.get(t)), None)
    if var is None:
        return None
    base = environ.get("OPENAI_BASE_URL", "").strip() or OPENAI_BASE
    return (
        f"model_provider = {json.dumps(CODEX_TOKEN_PROVIDER)}\n\n"
        f"[model_providers.{CODEX_TOKEN_PROVIDER}]\n"
        f'name = "OpenAI ({var})"\n'
        f"base_url = {json.dumps(base)}\n"
        f"env_key = {json.dumps(var)}\n"
        'wire_api = "responses"\n'
    )


#: What to do when the user's codex config names its own model provider (`codex_own_provider`).
CODEX_PROVIDER_FIX = (
    "qda does not use the provider in the sign-in dir's config.toml; set OPENAI_API_KEY and OPENAI_BASE_URL to "
    "that provider's key and URL"
)


def codex_own_provider(src: Path) -> str | None:
    """The `model_provider` the codex config in `src` names, when it is not codex's default (`openai`).

    `src` is the dir the sign-in file is read from (`agent.backend.sign_in_dir`, else `~/.codex`). The run's
    codex config is written afresh (`private_cli_config`), so a provider of the user's own `config.toml` never
    reaches the run: a codex signed in by a file would send its turns to the official API instead. The
    provider is the active profile's (`profile = "x"` at the top level, `[profiles.x] model_provider`) when
    that sets one, else the top level's, as codex resolves it. Only those keys are read; a file that is missing
    or does not parse names none.
    """
    try:
        config = tomllib.loads((src / "config.toml").read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    profile = config.get("profile")
    profiles = config.get("profiles")
    active = profiles.get(profile) if isinstance(profile, str) and isinstance(profiles, dict) else None
    provider = active.get("model_provider") if isinstance(active, dict) else None
    if not isinstance(provider, str):
        provider = config.get("model_provider")
    if not isinstance(provider, str) or provider.strip() in ("", "openai"):
        return None
    return provider


def sign_in_conflicts(cfg: RunConfig) -> dict[str, str]:
    """Name CLIs whose configured sign-in directory conflicts with an account or token variable."""
    configured = sign_in_dirs(cfg)
    out = {}
    for name, account in clis(cfg):
        if name not in CLI_CONFIG or configured.get(name) is None:
            continue
        tokens = [t for t in CLI_CONFIG[name][3] if os.environ.get(t)]
        if account or tokens:
            source = "account" if account else " / ".join(tokens)
            out[name] = f"{name}: agent.backend.sign_in_dir and {source} both set; name one sign-in"
    # A codex key variable is handed to codex in the run's one codex config (`codex_token_provider`), which an
    # agent signed in by an account would then also read: both kinds in one run cannot share it.
    used = clis(cfg)
    if ("codex", True) in used and ("codex", False) in used and "codex" not in out:
        if tokens := [t for t in CLI_CONFIG["codex"][3] if os.environ.get(t)]:
            out["codex"] = (
                f"codex: one agent signs in by an account and another by {' / '.join(tokens)}, and they share "
                "the run's codex config; sign both in the same way"
            )
    return out


def sign_in_report(cfg: RunConfig, deadline: datetime) -> tuple[bool, bool, list[str]]:
    """Describe the run's sign-in sources and expiry coverage, without exposing any secret values."""
    configured = sign_in_dirs(cfg)
    conflicts = sign_in_conflicts(cfg)
    ok, optional, lines = True, False, []
    kinds = {
        "ANTHROPIC_API_KEY": "API key",
        "ANTHROPIC_AUTH_TOKEN": "gateway token",
        "CLAUDE_CODE_OAUTH_TOKEN": "setup-token OAuth token",
    }
    for backend in (cfg.agent.backend, *cfg.agent.fallbacks):
        name = backend.name
        if backend.cli is None or name not in CLI_CONFIG:
            continue
        if name in conflicts:
            ok = False
            lines.append(conflicts[name])
            continue
        if account := getattr(backend, "account", None):
            lines.append(f"{name}: account dir of account {account}")
            continue
        tokens = CLI_CONFIG[name][3]
        if present := [t for t in tokens if os.environ.get(t)]:
            if name == "codex":  # one key: the provider of the run's codex config (`codex_token_provider`)
                present = present[:1]
            line = f"{name}: " + ", ".join(f"{kinds.get(t, 'API key')} via {t}" for t in present)
            if name == "codex":
                endpoint = "OPENAI_BASE_URL" if os.environ.get("OPENAI_BASE_URL", "").strip() else OPENAI_BASE
                line += f"; endpoint {endpoint}"
            lines.append(line)
            continue
        sign_in, expiry = CLI_CONFIG[name][2], CLI_CONFIG[name][4]
        if sign_in is None:
            ok = False
            lines.append(f"{name}: no sign-in; set {' / '.join(tokens)} or agent.backend.account")
            continue
        src = configured.get(name) or Path.home() / Path(sign_in).parent
        path = src / Path(sign_in).name
        source = "agent.backend.sign_in_dir" if configured.get(name) else "the default CLI config dir"
        line = f"{name}: credential file in {source}"
        if not path.is_file():
            ok = False
            lines.append(line + f" missing; sign in with {name} or set {' / '.join(tokens)}")
            continue
        if name == "codex" and codex_own_provider(src) is not None:
            # said, not judged: the sign-in file still signs the run in, against the official backend
            optional = True
            lines.append(f"codex: {CODEX_PROVIDER_FIX}")
            continue
        if expiry is not None:
            covered = False
            try:
                when = _expiry(path, expiry)
                covered = when >= deadline + timedelta(minutes=30)
                optional = optional or not covered
                line += (
                    f"; expires {when.isoformat()}; covers deadline + 30 min: {'yes' if covered else 'no'}"
                )
            except SignInRefused:
                ok = False
                line += "; expiry unavailable; covers deadline + 30 min: no"
            if not covered or cfg.wall_s >= 3600:
                line += "; for a long run, use `claude setup-token` or an API key"
        else:
            line += "; expiry coverage unknown (the file format records no expiry)"
        lines.append(line)
    return ok and not optional, ok and optional, list(dict.fromkeys(lines))


def sign_in_witness(cfg: RunConfig) -> dict[str, tuple[int, int] | None]:
    """The identity of each sign-in file the wall binds, by CLI name, without reading it.

    The value is the file's `(inode, mtime in ns)`, or None where the dir holds none. A host refresh replaces
    the file (`os.replace`, a new inode) or rewrites it (a new mtime), so a witness that differs from the one a
    wall was built with is a credential that wall does not hold. Only `stat` is used; the file is never opened.
    """
    if getattr(getattr(cfg, "isolation", None), "level", None) == "none" or cfg.sandbox.name != "bwrap":
        return {}  # no wall binds the file: the run reads the user's own file (level none) or no mount exists
    out: dict[str, tuple[int, int] | None] = {}
    for name, src in sign_in_accounts(cfg).items():
        if src is None:
            continue
        try:
            st = (src / sign_in_file(name)).stat()
        except OSError:
            out[name] = None
        else:
            out[name] = (st.st_ino, st.st_mtime_ns)
    return out


#: What each run's wall was built with: `{run_dir: sign_in_witness}` at the moment its CLI process started.
#: The wall's wrapper records it (`sign_in_mount`); a streaming CLI is one long-lived process that keeps the
#: mounts it started with across goals, so this -- not the goal -- is the sign-in file it actually holds.
_MOUNTED: dict[Path, dict[str, tuple[int, int] | None]] = {}


def sign_in_mounted(cfg: RunConfig) -> dict[str, tuple[int, int] | None] | None:
    """The witness the run's running wall was built with, or None while no wall is up.

    A `claude`/`codex` stream session reuses its process across goals (`_start` returns the live one), so a host
    refresh between goals is in this mount's past: the flow recycles the process when it differs from
    `sign_in_witness`.
    """
    return _MOUNTED.get(Path(cfg.run_dir))


def sign_in_mount(cfg: RunConfig) -> None:
    """Record the witness a wall is about to bind; `wall.gpu_wall`'s wrapper calls it at each process start."""
    _MOUNTED[Path(cfg.run_dir)] = sign_in_witness(cfg)


def sign_in_unmount(cfg: RunConfig) -> None:
    """Forget the wall's witness as the run's wall comes down (`wall.gpu_wall`)."""
    _MOUNTED.pop(Path(cfg.run_dir), None)


def sign_in_refreshed(cfg: RunConfig, before: Mapping[str, tuple[int, int] | None]) -> bool:
    """Whether the host refreshed a sign-in file since the witness `before`.

    True when an inode or mtime changed, a file the account dir held is gone, or one it lacked has appeared.
    A turn or a warm process holding `before` has the old credential; the flow recycles the process, whose
    next wall binds the new file (once per change; a refusal with no change stays fatal).
    """
    return sign_in_witness(cfg) != dict(before)


#: The sign-in that lasts a long run, by CLI: what a refusal of a short-lived sign-in file tells the user to do.
_LONG_RUN = {
    "claude": "run `claude setup-token` and export the token it prints as CLAUDE_CODE_OAUTH_TOKEN (valid one "
    "year, never refreshed), then rerun `qda doctor --search`",
}


def long_run_fix(name: str, named: Path | None = None) -> str:
    """What to do when the sign-in file of the agent CLI `name` would expire before the run ends.

    `claude`: its one-year setup token. Any other CLI: one of its token variables (`CLI_CONFIG`), which the run
    passes through the wall like the claude token. `named`: the run's `agent.backend.sign_in_dir`, which must go
    first, since a token beside it is refused (`sign_in_conflicts`).
    """
    if name in _LONG_RUN:
        fix = _LONG_RUN[name]
    elif tokens := (CLI_CONFIG[name][3] if name in CLI_CONFIG else ()):
        fix = f"export {' or '.join(tokens)}, then rerun `qda doctor --search`"
    else:
        return f"sign in to {name} again, then rerun `qda doctor --search`"
    if named is not None:
        return f"remove agent.backend.sign_in_dir {named} (a token beside it is refused), {fix}"
    return fix


def check_sign_in(cfg: RunConfig, deadline: datetime) -> None:
    """Refuse a run whose sign-in expires before the deadline plus 30 min with no refresh loop over it.

    Only the expiry field is read (`_expiry`); no message names the file or shows its content. A `bwrap` run
    reads the live file: each CLI process's wall binds the account dir's sign-in file into an empty dir, and
    the flow recycles a warm process when the host refreshes it (so a refresh by `os.replace` is the file the
    next wall binds). Under the limit it only warns when
    the run names `agent.backend.sign_in_dir`: the host's refresh loop extends a named account dir while it is
    below 2 h left, and it writes the file the wall binds. The CLI's default dir has no such loop (a refresh
    there would rotate the user's own session), so that one is refused. A `fence` run reads the named dir's file
    through the link its own config dir holds to it, which no refresh loop reaches, and its named dir is
    refused as before.

    Raises:
        SignInRefused: The sign-in records no usable expiry, or one earlier than `deadline` plus 30 min with no
            refresh loop over the file.
    """
    if cfg.isolation.level == "none":
        return  # the run's config dir links the user's own sign-in file, and the CLI refreshes it in place
    walled = cfg.sandbox.name == "bwrap"
    accounts, limit = sign_in_accounts(cfg), deadline + timedelta(minutes=30)
    for name, named in sign_in_dirs(cfg).items():
        expiry = CLI_CONFIG[name][4]
        src = accounts.get(name)
        if expiry is None or src is None:  # no expiry recorded, or an account or token variable signs it in
            continue
        if walled:
            path, refreshed = src / sign_in_file(name), named is not None
        elif named is None:  # a fence run has never held the default dir's copy to an expiry
            continue
        else:
            # No wall binds the file, so the run's config dir holds a link to the named dir's
            # (`private_cli_config`): the credential is that dir's file, whether or not the run has made the
            # link yet (`qda doctor --search` asks this before it exists).
            path, refreshed = src / sign_in_file(name), False
        try:
            when = _expiry(path, expiry)
        except SignInRefused as why:
            raise SignInRefused(f"{why}: {long_run_fix(name, named)}", code=why.code) from None
        if when > limit:
            _log.info("sign-in (%s) expires %s", src, when.isoformat())
        elif refreshed:
            _log.warning(
                "sign-in (%s) expires at %s, before the run's deadline plus 30 min (%s): the run lasts only if "
                "something on the host refreshes that file; otherwise %s",
                src,
                when.isoformat(),
                limit.isoformat(),
                long_run_fix(name, named),
            )
        elif named is not None:
            raise SignInRefused(
                f"agent.backend.sign_in_dir {named}: the sign-in expires at {when.isoformat()}, before the "
                f"run's deadline plus 30 min ({limit.isoformat()}): {long_run_fix(name, named)}",
                code=ENVIRONMENT,
            )
        else:
            raise SignInRefused(
                f"the CLI's default sign-in dir: the sign-in expires at {when.isoformat()}, before the run's "
                f"deadline plus 30 min ({limit.isoformat()}): {long_run_fix(name)}",
                code=ENVIRONMENT,
            )


def _expiry(path: Path, keys: tuple[str, ...]) -> datetime:
    """One field of the sign-in: its expiry, in milliseconds since the epoch.

    Raises:
        SignInRefused: The field is absent, unreadable, or not a finite number. The message names neither the
            file nor any of its content (the underlying error is dropped, not chained).
    """
    problem = SignInRefused(
        "agent.backend.sign_in_dir: the sign-in records no usable expiry", code=ENVIRONMENT
    )
    try:
        node: Any = json.loads(path.read_text())
        for key in keys:
            node = node[key]
        ms = float(node)
    except (OSError, KeyError, TypeError, ValueError):
        raise problem from None
    if not math.isfinite(ms):
        raise problem from None
    try:
        return datetime.fromtimestamp(ms / 1000, UTC)
    except (OverflowError, OSError, ValueError):
        raise problem from None


@contextmanager
def private_cli_config(cfg: RunConfig) -> Iterator[dict[str, Path]]:
    """While the run is under way, its CLIs read and write a config dir of the run's own, never the user's.

    Humanize starts each session in a per-session config dir made of links to "the user's" config dir, which it
    takes from this process's `CLAUDE_CONFIG_DIR` / `CODEX_HOME` (or `~/.claude`, `~/.codex`), and as it drops
    that dir after the turn it writes back every link the CLI replaced. Pointed at the user's dir -- or at the
    launcher's, an agent session's own account -- that lets the agent plant instructions (`CLAUDE.md`,
    `commands/`, `AGENTS.md`, `prompts/`, history) that the user's later sessions load, and shows it every other
    session's transcripts. So each CLI gets a fresh dir, `<run_dir>/cli/<name>`, built from scratch, and the
    variables point at it for the length of the run: whatever is linked or written back stays in the run.

    The dir holds an empty settings file and, only for a CLI that signs in by a file -- no account in its spec
    (an account's provider signs in on the host) and none of its token variables set -- its sign-in, from
    `agent.backend.sign_in_dir` when the CLI names one, else from its default dir (`~/.claude`, `~/.codex`).
    When no wall binds a copy (`isolation.level: none`, or a run whose sandbox is no bwrap), the config dir's
    sign-in path is a SYMLINK to the user's own file, never a copy: a CLI that refreshes its credential writes
    through it, so the user's file is updated in place, while everything else the CLI writes (`CLAUDE.md`,
    `commands/`, history) stays in the run's dir. A `bwrap` run never copies it either: the wall of each CLI
    process mounts an empty `<run_dir>/signin/<name>` (`SIGNIN`) and binds only the sign-in file read-only into
    it, and both the config dir's sign-in path and the live dir's (the dir the CLI works in inside the wall) are
    links to that path. Nothing else in the account dir is ever mounted, so an entry it gains after the run
    starts is not reachable either; a refresh on the host (`os.replace`, the same path with a new inode) is the
    file the next wall binds (the flow recycles a stream session's warm process, whose wall then binds it). A
    configured dir that is missing, holds no sign-in file, or meets a second sign-in source (an account or a
    token variable) is refused at the start, naming only the dir. Nothing comes from the launcher's own config
    dir (the variables of CLIs the run does not use are unset too), and nothing is copied back when the run
    ends. Inside the wall this dir is read-only and the CLI works in its live dir (`live_dirs`, `live_state`);
    a CLI that refreshes its sign-in by rewriting the file cannot, which is why the host refreshes the account
    dir the wall binds; the flow recycles the process when it changes (`check_sign_in`).

    Args:
        cfg: The run config. At `isolation.level: none` every CLI the run uses still gets the run's own config
            dir: an empty settings file matters there too (a dsh run's runtime reads its endpoint from
            `$DSH_HOME/settings.yaml`, and that outranks the run's environment), and a sign-in file is linked
            rather than the user's dir being read.

    Yields:
        The dirs, by CLI name.
    """
    dirs = _config_dirs(cfg)
    if not dirs:  # a CLI outside `CLI_CONFIG` (a plugin backend): nothing to move, and the environment is
        yield {}  # left exactly as the run found it (today's unconfined behaviour)
        return
    saved: dict[str, str | None] = {}
    by_name = dict(clis(cfg))
    configured = sign_in_dirs(cfg)
    if conflicts := sign_in_conflicts(cfg):
        raise SignInRefused("; ".join(conflicts.values()))
    accounts = sign_in_accounts(cfg)
    # What binds a read-only copy of the account dir's sign-in file: the bwrap wall. Without it (level none,
    # or a sandbox that is not bwrap) the run's config dir links the user's own file instead.
    walled = cfg.sandbox.name == "bwrap" and cfg.isolation.level != "none"
    try:
        for name, d in dirs.items():
            var, settings, sign_in, tokens, _ = CLI_CONFIG[name]
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, mode=0o700)
            (d / settings).write_text("{}\n" if settings.endswith(".json") else "")
            if name == "codex" and not by_name[name] and (provider := codex_token_provider(os.environ)):
                # The key stays in the environment: only the variable's name is written.
                (d / settings).write_text(provider)
            sign_in_from = configured.get(name)
            if sign_in is None:  # no sign-in file: an account or a token variable signs it in
                if sign_in_from is not None:
                    raise SignInRefused(
                        f"agent.backend.sign_in_dir {sign_in_from}: {name} has no sign-in file"
                    )
                if not (by_name[name] or any(os.environ.get(t) for t in tokens)):
                    raise SignInRefused(
                        f"{name} signs in by a Humanize account or a token variable ({', '.join(tokens)}); "
                        "the run names neither: set agent.backend.account or export the variable",
                        code=ENVIRONMENT,
                    )
            elif sign_in_from is not None:  # the CLI's own account dir: it decides this run's sign-in
                if not sign_in_from.is_dir():
                    raise SignInRefused(f"agent.backend.sign_in_dir: {sign_in_from} is not a directory")
                if not (sign_in_from / sign_in_file(name)).is_file():
                    raise SignInRefused(
                        f"agent.backend.sign_in_dir {sign_in_from}: no sign-in file for {name}",
                        code=ENVIRONMENT,
                    )
            src = accounts.get(name)
            if src is not None and walled:  # the wall binds the sign-in file under <run_dir>/signin/<name>
                at = cfg.run_dir / SIGNIN / name
                shutil.rmtree(at, ignore_errors=True)
                at.mkdir(parents=True, mode=0o700)  # the host side of its empty tmpfs
                (d / sign_in_file(name)).symlink_to(_sign_in_link(name))  # never a copy: the bound file
            elif src is not None:  # no wall: link the user's file, which the CLI refreshes in place
                (d / sign_in_file(name)).symlink_to(src / sign_in_file(name))
            saved.setdefault(var, os.environ.get(var))
            os.environ[var] = str(d)
        if walled:
            for name, live in live_dirs(cfg).items():
                shutil.rmtree(live, ignore_errors=True)
                live.mkdir(parents=True, mode=0o700)
                if accounts.get(name) is not None:  # the CLI works in this dir: link, not copy
                    (live / sign_in_file(name)).symlink_to(_sign_in_link(name))
        for var, *_ in CLI_CONFIG.values():  # the launcher's dir of a CLI the run does not use: not passed on
            if var not in saved:
                saved[var] = os.environ.pop(var, None)
        yield dirs
    finally:
        for var, was in saved.items():
            if was is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = was


def _sign_in_link(name: str) -> Path:
    """The link from a CLI's config or live dir (`<run_dir>/cli|live/<name>`) to its account mount's sign-in."""
    return Path("..") / ".." / SIGNIN / name / sign_in_file(name)
