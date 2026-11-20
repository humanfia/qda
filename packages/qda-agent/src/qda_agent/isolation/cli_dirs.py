"""Where each agent CLI of the run works: its config dir and its live dir, and the run's Humanize epics.

The dirs are the run's own, never the user's; `cli_signin` fills them for the length of the run and decides
what signs it in (see this package).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qda_agent.config import RunConfig

#: Each agent CLI's config dir, by Humanize backend name: `(the variable that moves it, the settings file made in
#: it, its sign-in file under the default dir -- None for a backend that has none --, the variables that sign it
#: in by a token instead, the keys leading to its expiry in that file -- milliseconds since the epoch -- or None
#: when it records none)`.
CLI_CONFIG: dict[str, tuple[str, str, str | None, tuple[str, ...], tuple[str, ...] | None]] = {
    "claude": (
        "CLAUDE_CONFIG_DIR",
        "settings.json",
        ".claude/.credentials.json",
        # The three auth variables Claude Code documents (env-vars reference): an Anthropic API key, a bearer
        # token for a gateway, and the one-year OAuth token from `claude setup-token`. Any of them signs the
        # CLI in without its credential file; the OAuth token has no refresh, it is replaced by regenerating.
        ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        ("claudeAiOauth", "expiresAt"),
    ),
    "codex": (
        "CODEX_HOME",
        "config.toml",
        ".codex/auth.json",
        ("OPENAI_API_KEY", "CODEX_API_KEY"),
        None,
    ),
    # DeepSeek Harness: Humanize drives it through its SDK and reads the key on the host -- from the account
    # in the spec, or from `DEEPSEEK_API_KEY` -- so the runtime inside the wall is handed the key and no file of
    # `~/.dsh` is ever mounted. `DSH_HOME` is where the runtime keeps its profile and sessions.
    "dsh": (
        "DSH_HOME",
        "settings.yaml",
        None,
        ("DEEPSEEK_API_KEY",),
        None,
    ),
}


def sign_in_file(name: str) -> str:
    """The sign-in file's name in the account dir of a CLI that signs in by a file (`.credentials.json`, ...).

    Raises:
        ValueError: The CLI has no sign-in file (`CLI_CONFIG`): nothing may ask where it is.
    """
    sign_in = CLI_CONFIG[name][2]
    if sign_in is None:
        raise ValueError(f"the agent CLI {name!r} has no sign-in file")
    return Path(sign_in).name


def token_vars(name: str) -> tuple[str, ...]:
    """The variables that sign the agent CLI `name` in by a token instead of its sign-in file."""
    return CLI_CONFIG[name][3]


def clis(cfg: RunConfig) -> list[tuple[str, bool]]:
    """The agent CLIs the run may use (`agent.backend`, then `agent.fallbacks`): `(backend name, has an account)`.

    Raises:
        ValueError: A CLI Humanize does not know.
    """
    from hmz.coganchor.backends import named

    out = []
    for spec in cfg.agent.specs:
        cli = spec.split("/", 1)[0]
        profile = named(cli.split("@", 1)[0])
        if profile is None:
            raise ValueError(f"Humanize knows no agent CLI {cli!r}")
        out.append((profile.name, "@" in cli))
    return list(dict.fromkeys(out))


def cli_config_dirs(cfg: RunConfig) -> dict[str, Path]:
    """The run's own config dir of each of its CLIs: `<run_dir>/cli/<name>`.

    Raises:
        ValueError: A CLI with no `CLI_CONFIG` entry (its config dir cannot be moved, so it would be the user's).
    """
    out = {}
    for name, _ in clis(cfg):
        if name not in CLI_CONFIG:
            raise ValueError(
                f"qda-agent cannot give the agent CLI {name!r} a config dir of the run's own; "
                "name it in qda_agent.isolation.CLI_CONFIG"
            )
        out[name] = cfg.run_dir / "cli" / name
    return out


def live_dirs(cfg: RunConfig) -> dict[str, Path]:
    """The config dir each CLI of the run works in inside the wall: `<run_dir>/live/<name>` (`live_state`).

    Only the wall binds it (read-write); the host makes it empty at the start. After that the host never opens
    live/ directly; it copies it out only through the sandboxed snapshot helper (`codexlog`, after the run).
    """
    return {name: cfg.run_dir / "live" / name for name in cli_config_dirs(cfg)}


def agent_state(workdir: Path) -> Path:
    """The run's own Humanize epics, `~/.hmz/epics/<workdir>` (`hmz.runtime.epic.under`), created.

    Every session store Humanize keeps for this workdir's runs (`sessions/<backend>`, which `cred --keep` points
    the CLI's `projects/`, `sessions/`, `todos/` ... at), so a re-sent `/goal` resumes its session; mounted back
    read-write even where the fence lists no path for it. Other workspaces' epics and the rest of `~/.hmz` stay
    hidden.
    """
    from hmz.runtime import epic

    epics = epic.under(workdir)
    epics.mkdir(parents=True, exist_ok=True)
    return epics
