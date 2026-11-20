"""Configure agent CLIs and in-process drivers for the worker role.

CLI settings form Humanize's `CLI[@ACCOUNT]/MODEL[:EFFORT]` spec. Humanize requires shell tools for
workers and reports an unsupported backend with its own diagnostic.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator, Field, SerializeAsAny, field_validator, model_validator
from qda_quant.registry import Registry

from qda_agent import dsh
from qda_agent.settings import Plugin, Strict, pick, validate_plugin

BACKENDS = Registry(
    "backends",
    {
        "claude": "qda_agent.agents:Claude",
        "codex": "qda_agent.agents:Codex",
        "dsh": "qda_agent.agents:Dsh",
    },
    group="qda_agent.backends",
    on_add=partial(validate_plugin, "qda_agent.backends"),
)

if TYPE_CHECKING:
    from qda_agent.config import RunConfig

__all__ = ["AgentBackend", "AgentConfig", "AnyBackend", "Backend", "Claude", "Codex", "Dsh"]


#: Endpoint variables of each agent CLI that its Humanize profile does not list (`endpoint_vars` adds those it
#: does): inherited by a CLI that signs in by its file or an account, they send that sign-in elsewhere.
ENDPOINT_VARS: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_BEDROCK_BASE_URL",),
}


def endpoint_vars(cli: str) -> tuple[str, ...]:
    """Every variable that says where a turn of the agent CLI `cli` goes, by its canonical name, sorted.

    The CLI's Humanize profile names the one that says where a turn goes (`endpoint`) and the other variables it
    would take an account from (`ambient`); of those, the URLs (`*_URL`) are endpoints. `ENDPOINT_VARS` adds any
    the profile does not list. A CLI Humanize does not know has none.
    """
    from hmz.coganchor.backends import named

    profile = named(cli)
    names = set(ENDPOINT_VARS.get(cli, ()))
    if profile is not None:
        names |= {v for v in (profile.endpoint, *profile.ambient) if v.endswith("_URL")}
    return tuple(sorted(names))


def canonical_cli(spec: str) -> str:
    """The canonical CLI name of an agent spec (`claude-code@work/m` -> `claude`)."""
    from hmz.coganchor.backends import named

    cli = spec.split("/", 1)[0].split("@", 1)[0]
    profile = named(cli)
    return profile.name if profile is not None else cli


def log_codex_calls(cfg: RunConfig, tools: Any, log: Any) -> None:
    """Append a Codex agent's tool calls to tools.jsonl, read from its session log after the run.

    The hook does not see Codex's calls, so they are logged after the fact, not checked live. Codex is the
    one CLI whose calls are read from its own session log (`qda_agent.codexlog`): inside the wall from a
    sandboxed copy of its live dir, at `isolation.level: none` from the run's Humanize epic, in place.
    """
    from qda_agent import codexlog, isolation

    try:
        if "codex" not in dict(isolation.cli_dirs.clis(cfg)):
            return
        if cfg.isolation.level == "none":  # no wall, no live dir: Humanize keeps the sessions in the epic
            from hmz.runtime import epic

            runs = epic.epics(cfg.workdir)  # oldest first; the last is this run's, still open
            if not runs:
                return
            entries, refused = codexlog.epic_calls(runs[-1] / "sessions" / "codex" / "sessions")
        else:
            live = isolation.live_dirs(cfg).get("codex")
            if live is None:
                return
            entries, refused = codexlog.session_calls(live, cfg.run_dir)
        tools.extend(entries)
    except Exception as e:  # noqa: BLE001 -- agent-written input: nothing here may cost the run its result.json
        log("host_error", step="codex_log", error=type(e).__name__, message=str(e)[:500])
        return
    log("codex_log", entries=len(entries), refused=refused)


@dataclass(frozen=True)
class TurnProblem:
    """What a backend's first-turn check found, and whether another try could take the turn.

    Attributes:
        said: What to report: the endpoint's own words, or the SDK this interpreter lacks.
        transient: The endpoint did not answer, or answered that it is busy, throttled or failing. That is
            what `agent.retries` and Humanize's own retry policy are for; a refusal it answered with it will
            answer again, so only that one is a fact about the run's config.
    """

    said: str
    transient: bool = False


def sdk_missing(modules: Sequence[str]) -> list[str]:
    """List the modules of `modules` this interpreter cannot import: one rule, for doctor and for a run."""
    return [m for m in modules if importlib.util.find_spec(m) is None]


def sdk_install(cli: str) -> str:
    """Return Humanize's own install line for the SDK that drives `cli`, which it validates on load."""
    from hmz.coganchor.backends import named

    profile = named(cli)
    return getattr(profile, "installs", None) or f"install the {cli} SDK"


class Backend(Plugin):
    """Configure a worker's Humanize spec, CLI dependency and optional in-process driver."""

    #: The Humanize CLI name (an executable on `PATH`); None: the backend starts no CLI.
    cli: ClassVar[str | None]
    #: The Python modules Humanize drives the backend through, for one that has no executable on `PATH`
    #: (`qda doctor --search` looks for these instead of `cli`); empty: `cli` is an executable.
    modules: ClassVar[tuple[str, ...]] = ()
    #: Whether a session of this backend takes another instruction once a turn of it has ended. False: the
    #: flow starts a new session for every goal it sends.
    resumes: ClassVar[bool] = True

    @property
    def spec(self) -> str:
        """Return the Humanize agent spec used in logs and reports."""
        raise NotImplementedError

    @property
    def place(self) -> str:
        """Return the Humanize place, which omits the reasoning effort."""
        raise NotImplementedError

    def driver(self, cfg: RunConfig) -> Any:
        """Return an in-process driver, or None to let Humanize launch the CLI."""
        del cfg
        return None

    def turn_problem(self) -> TurnProblem | None:
        """Return why the first turn of this backend cannot be taken as configured, or None when it can.

        The check a run makes before its clock starts and `qda doctor --search` makes as its `agent turn`
        line (`flow.check_agent_turn`): one backend, one answer. A CLI is looked up on `PATH`; beyond that it
        can only be asked by taking a turn of it, which costs what a turn costs, so the run's own first turn is
        the check; an SDK-driven backend whose endpoint answers for a token's worth (`dsh`) answers here.
        """
        return None

    def turn_asked(self) -> str:
        """Return what `turn_problem` asks of this backend, for the line that reports it passing."""
        return "nothing to ask before its first turn"


class AgentBackend(Backend):
    """Configure a Humanize CLI, model, reasoning effort and account.

    Attributes:
        model: The model the CLI runs.
        effort: The reasoning effort (`low`, `high`, ...); None: the CLI's default.
        account: The Humanize account of the CLI (`claude@ACCOUNT`); None: the default sign-in.
        sign_in_dir: The CLI config dir whose sign-in file the run links into its own config dir
            (`isolation.private_cli_config`: a wall binds it read-only, or at level none the link is to the
            user's file so a refresh lands there); None: the CLI's default dir (`~/.claude`, `~/.codex`).
    """

    cli: ClassVar[str]

    model: str
    effort: str | None = None
    account: str | None = None
    sign_in_dir: Path | None = None

    @field_validator("sign_in_dir")
    @classmethod
    def _expand_home(cls, value: Path | None) -> Path | None:
        """Expand `~` so a config may name the dir as `~/.claude-NAME`, also when loaded without `load`."""
        return None if value is None else Path(value).expanduser()

    @property
    def place(self) -> str:
        """Return the `CLI[@ACCOUNT]/MODEL` place without its reasoning effort."""
        return f"{self.cli}{'@' + self.account if self.account else ''}/{self.model}"

    @property
    def spec(self) -> str:
        """Return the `CLI[@ACCOUNT]/MODEL[:EFFORT]` agent spec."""
        return f"{self.place}:{self.effort}" if self.effort else self.place

    def turn_problem(self) -> TurnProblem | None:
        """Return that the CLI is not on `PATH`, else None (a CLI can only be asked by taking a turn of it).

        Humanize starts the CLI by name from this `PATH` at the run's first turn, after candidate 0's trial
        run; no retry installs it, so it is never transient. An SDK-driven backend (`modules`) has no
        executable to find and answers for itself.
        """
        if self.modules or shutil.which(self.cli) is not None:
            return None
        return TurnProblem(
            f"{self.cli} is not on PATH (fix: install {self.cli} and put it on PATH, or change agent.backend)"
        )


class Claude(AgentBackend):
    """Claude Code."""

    cli: ClassVar[str] = "claude"
    name: Literal["claude"] = "claude"
    model: str = "claude-opus-5-5"
    effort: str | None = "high"


class Codex(AgentBackend):
    """OpenAI Codex CLI."""

    cli: ClassVar[str] = "codex"
    name: Literal["codex"] = "codex"
    model: str = "gpt-5.6-sol"
    effort: str | None = "high"


class Dsh(AgentBackend):
    """DeepSeek Harness, which Humanize drives through its Python SDK (there is no `dsh` on `PATH` to find).

    The SDK and its bundled runtime are installed into the interpreter that runs `qda-agent`. The backend
    signs in through a Humanize account (`account`: its key, and its endpoint when the account was made by a
    gateway way) or `DEEPSEEK_API_KEY` in the run's environment. Humanize reads the key on the host and hands
    it to the runtime, so there is no sign-in file to mount and `sign_in_dir` is refused.

    Where the key goes: a named account's own endpoint (and a run that names one ignores
    `DEEPSEEK_BASE_URL`), else `DEEPSEEK_BASE_URL` (an OpenAI-compatible base URL that the runtime appends
    `/chat/completions` to, with no version path), else the SDK's default, the official API.

    The SDK starts a conversation and never resumes one, so a session is carried only by the runtime that took
    its first turn. Under the wall that runtime is not there for a second instruction, which Humanize then
    refuses as unrecoverable: a session that has taken a turn takes no second one (`resumes` is False).
    """

    cli: ClassVar[str] = "dsh"
    modules: ClassVar[tuple[str, ...]] = ("deepseek_harness", "deepseek_harness_runtime")
    resumes: ClassVar[bool] = False
    name: Literal["dsh"] = "dsh"
    # what the endpoint a run names no other for serves; a gateway's own name comes from the validator
    model: str = dsh.OFFICIAL_MODEL

    @model_validator(mode="before")
    @classmethod
    def _model_follows_the_endpoint(cls, data: Any) -> Any:
        """Take the model name this run's endpoint serves when the config names no model.

        The official API serves `deepseek-flash` and a gateway serves its own names, so the default is
        the one the endpoint the key goes to takes (`dsh.default_model`); a model the config names is
        kept, and a config that names none resolves to a name that is written out in `resolved.yaml`.
        """
        if not isinstance(data, dict) or data.get("model"):
            return data
        account = data.get("account")
        return {**data, "model": dsh.default_model(account if isinstance(account, str) else None)}

    @field_validator("sign_in_dir")
    @classmethod
    def _no_sign_in_file(cls, value: Path | None) -> Path | None:
        """Refuse `sign_in_dir`: the harness has no sign-in file for the run to mount."""
        if value is not None:
            raise ValueError(
                "agent.backend.sign_in_dir: dsh has no sign-in file; name a Humanize account "
                "(agent.backend.account) or set DEEPSEEK_API_KEY"
            )
        return value

    def turn_problem(self) -> TurnProblem | None:
        """Return the SDK modules this interpreter lacks, else what the run's endpoint says of its model.

        Both are one fact -- what a `dsh` run against this endpoint needs -- and neither is visible until a
        turn is taken: Humanize drives the harness in this interpreter, and an endpoint refuses a model name
        it does not serve only when a turn is sent. A module this interpreter lacks is not transient: no
        retry installs it.
        """
        if missing := sdk_missing(self.modules):
            return TurnProblem(
                f"{self.cli} is driven through its Python SDK: {', '.join(missing)} not installed in "
                f"{sys.prefix} (fix: {sdk_install(self.cli)} (into the interpreter that runs qda-agent), or "
                "change agent.backend)"
            )
        return dsh.turn_problem(self.model, self.account)

    def turn_asked(self) -> str:
        """Return the endpoint and the model its check asks it to take a turn at."""
        return f"{dsh.base_url(self.account)} takes {self.model}"


def _backend(value: Any) -> Any:
    return pick(BACKENDS, value)


AnyBackend = Annotated[SerializeAsAny[Backend], BeforeValidator(_backend)]


class AgentConfig(Strict):
    """Select the agent backend and Humanize's retry and fallback settings.

    Attributes:
        backend: The agent backend (`qda_agent.backends`): its CLI, model, effort and account, or a script.
        fallbacks: CLI backends whose places take the turn, in order, once `backend`'s retries are spent.
        retries: Extra tries at the same place before falling back (Humanize's own retry).
        retry_policy: Humanize wait policy between tries (`exponential-jitter`, `linear`, ...).
        retry_timeout_s: Total limit on one place's retries, seconds (0 = none).
        cost_usd: API spending limit of the whole search, USD (None = unlimited).
        goal: Use the CLI's own `/goal` (True), a plain prompt re-sent on idle (False), or `/goal`
            where the CLI has one (None).
        resend_wait_s: The least time between two sends of the goal (after an idle return or a failed turn,
            once Humanize's retries and fallbacks are spent); one fixed tick, not a backoff schedule.
    """

    backend: AnyBackend = Field(default_factory=Claude)
    fallbacks: list[AnyBackend] = Field(default_factory=list)
    retries: int = Field(default=3, ge=0)
    retry_policy: str = "exponential-jitter"
    retry_timeout_s: float = Field(default=0.0, ge=0)
    cost_usd: float | None = Field(default=None, gt=0)
    goal: bool | None = None
    resend_wait_s: float = Field(default=5.0, ge=0)

    @field_validator("goal", mode="before")
    @classmethod
    def _goal_is_a_flag(cls, goal: object) -> object:
        # pydantic's lax bool would turn "no" into False and 1 into True: a mode change nobody asked for
        if goal is not None and not isinstance(goal, bool):
            raise ValueError(
                f"agent.goal is a mode flag (true/false/null), not {type(goal).__name__} {goal!r}; "
                "the goal text is TASK.md's"
            )
        return goal

    @field_validator("fallbacks")
    @classmethod
    def _clis_only(cls, fallbacks: list[Backend]) -> list[Backend]:
        if bad := [f.name for f in fallbacks if f.cli is None]:
            raise ValueError(f"agent.fallbacks: {', '.join(bad)} starts no agent CLI and cannot take a turn")
        return fallbacks

    @property
    def spec(self) -> str:
        """Return the backend's Humanize spec, `CLI[@ACCOUNT]/MODEL[:EFFORT]`."""
        return self.backend.spec

    @property
    def specs(self) -> list[str]:
        """List CLI specs in startup order: the primary backend, then its fallbacks."""
        return [b.spec for b in (self.backend, *self.fallbacks) if b.cli is not None]

    @property
    def place(self) -> str:
        """Return the backend's Humanize place, `CLI[@ACCOUNT]/MODEL`."""
        return self.backend.place

    @property
    def fallback_places(self) -> list[str]:
        """List Humanize places for the configured fallbacks, in order."""
        return [f.place for f in self.fallbacks]
