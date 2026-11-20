"""Assemble agent commands, workspace files and job settings from ordered components.

A toolset names its components (`qda_agent.components`) and the interpreter the agent's jobs run::

    toolset: {name: qda, python: venvs/quant/bin/python}              # the default preset
    toolset: {name: qda, components: [qda, kl, qsched, handin]}       # the same with fewer parts

An explicit `components` list replaces the preset. Each component supplies commands, verbs, read-only
files, TASK.md sections and job dependencies. With `python` set, its `bin/` leads the job `PATH` and
`qda` runs as `python -m qda_quant.cli`. Otherwise jobs use the host interpreter and `qda` on its `PATH`.

Presets:

- `qda`: builds, KL scoring, generation checks, the queue, submission, the methods wiki and inputs:
  datasets, a serving environment, baselines, model metadata and library sources.
- `public`: reserved and refused until a separate job interpreter and host path can exclude qda;
  see the `isolation` package.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator, Field, SerializeAsAny, model_validator
from qda_quant.registry import Registry

from qda_agent.components import COMPONENTS, Component, Run
from qda_agent.settings import Plugin, pick, validate_plugin

TOOLSETS = Registry(
    "toolsets",
    {"public": "qda_agent.toolsets:Public", "qda": "qda_agent.toolsets:Qda"},
    group="qda_agent.toolsets",
    on_add=partial(validate_plugin, "qda_agent.toolsets"),
)

__all__ = ["DEFAULT", "Public", "Qda", "Toolset"]

#: The system part of a job's PATH, after the toolset's own `bin/`.
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"

#: The default preset's key.
DEFAULT = "qda"


def _components(value: Any) -> Any:
    if not isinstance(value, list):
        raise ValueError(f"toolset.components must be a list of component keys, got {value!r}")
    return [pick(COMPONENTS, v) for v in value]


class Toolset(Plugin):
    """Configure an ordered set of components and their shared job environment.

    Attributes:
        components: The parts, in TASK.md order (keys or `{name: KEY, ...settings}`); default: the preset's.
        python: The interpreter of a virtualenv with torch and qda-quant; its installation is bound read-only
            into the job sandbox (as `qda`'s is). None: the host's.
        qda: The `qda` command (an argv) when it is not `python -m qda_quant.cli`.
        env: More environment of every agent job and `qda` build (e.g. `PYTHONPATH` to a source tree, which
            must then be in `scheduler.readable`); a `PATH` here replaces the toolset's. The agent's code can
            read it: it must hold no secret. A job that runs on the host (`scheduler.job_sandbox: none`)
            keeps the launching `PATH` after the toolset's own directories, so a site compiler still
            resolves; a job sandbox has only the system directories it mounts.
    """

    #: The preset's components.
    preset: ClassVar[tuple[str, ...]] = ()

    components: Annotated[list[SerializeAsAny[Component]], BeforeValidator(_components)] = Field(
        default_factory=list
    )
    python: Path | None = None
    qda: list[str] | None = Field(default=None, min_length=1)
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _preset(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "components" not in raw:
            raw = {**raw, "components": list(cls.preset)}
        return raw

    @model_validator(mode="after")
    def _distinct(self) -> Toolset:
        keys = [c.name for c in self.components]
        if len(set(keys)) != len(keys):
            raise ValueError(f"toolset.components names a component twice: {keys}")
        return self

    def check_shipped_data(self, model: str) -> None:
        """Refuse a `data` set file holding pre-rendered chat markup, before the clock (`qda_agent.shipped_data`)."""
        from qda_agent.shipped_data import check

        check(self, model)

    def offered(self, run: Run) -> list[Component]:
        """The components this run gets, in TASK.md order: those whose `offers` is true."""
        return [comp for comp in self.components if comp.offers(run)]

    def commands(self, run: Run) -> tuple[str, ...]:
        """List unique programs installed in this run's `work/bin`, preserving component order."""
        return tuple(dict.fromkeys(c for comp in self.offered(run) for c in comp.commands))

    def subcommands(self, run: Run) -> dict[str, list[str]]:
        """Collect allowed verbs for commands that declare them, preserving order and removing duplicates."""
        out: dict[str, list[str]] = {}
        for comp in self.offered(run):
            for cmd, verbs in comp.subcommands.items():
                out.setdefault(cmd, []).extend(v for v in verbs if v not in out.get(cmd, []))
        return out

    def handlers(self, run: Run) -> dict[str, str]:
        """Map component-handled commands to their `module:function` handlers."""
        return {k: v for comp in self.offered(run) for k, v in comp.handlers.items()}

    def sections(self, run: Run) -> list[str]:
        """Build TASK.md sections with command descriptions before workspace inputs.

        Preserve component order within each group and introduce inputs with "Your workspace also has:".
        """
        offered = self.offered(run)
        tools = [s for c in offered if (c.commands or c.subcommands) and (s := c.section(run))]
        other = [s for c in offered if not (c.commands or c.subcommands) and (s := c.section(run))]
        if not other:
            return tools
        if tools and tools[-1].endswith(";\n"):
            tools[-1] = tools[-1][: -len(";\n")] + ".\n"
        if not tools:  # nothing to introduce the inputs with: "also" would have no antecedent
            return other
        return [*tools, "Your workspace also has:\n", *other]

    def install(self, run: Run) -> None:
        """Install each offered component's workspace files in component order."""
        for comp in self.offered(run):
            comp.install(run)

    def protected(self, run: Run) -> list[str]:
        """List unique top-level workspace names supplied by the read-only overlay."""
        return list(dict.fromkeys(n for comp in self.offered(run) for n in comp.protected(run)))

    def mounts(self, run: Run, jobs: bool = True) -> dict[Path, Path]:
        """Map workspace destinations to read-only host sources, with overlay parents first.

        Add component binds under protected names when `jobs` is True. The agent's shell receives
        only the overlay; baseline weights remain available to jobs alone.

        Raises:
            ValueError: A bind outside the protected names, or a source inside the workdir.
        """
        names = self.protected(run)
        out = {run.workdir / n: run.ro(n) for n in names}
        for comp in self.offered(run) if jobs else ():
            for dst, src in comp.binds(run).items():
                rel = dst.relative_to(run.workdir)
                if rel.parts[0] not in names or len(rel.parts) < 2:
                    raise ValueError(
                        f"component {comp.name}: bind {dst} is not under a protected name {names}"
                    )
                out[dst] = src
        for src in out.values():
            if Path(os.path.abspath(src)).is_relative_to(os.path.abspath(run.workdir)):  # noqa: PTH100
                raise ValueError(f"a read-only bind's source {src} lies in the agent's workdir")
        return out

    def readable(self) -> list[Path]:
        """Collect component job inputs outside the work directory."""
        return [p for comp in self.components for p in comp.readable()]

    def baseline_names(self, run: Run) -> list[str] | None:
        """Use the first offered component's baseline names, or None to let TASK.md use their paths."""
        return next((n for comp in self.offered(run) if (n := comp.baseline_names(run)) is not None), None)

    def job_env(self, run: Run | None = None, *, base_path: str | None = None) -> dict[str, str]:
        """Build job environment overrides, with explicit `env` taking precedence over components.

        `PATH` starts with the interpreter's `bin/` and the component directories. After them come
        `base_path` — the launching `PATH` of jobs that run on the host (`scheduler.job_sandbox: none`), so a
        site, conda or module compiler a build needs keeps working — or, in a job sandbox, the fixed
        system directories (`SYSTEM_PATH`), whose paths are the ones the sandbox really mounts.
        """
        dirs = [
            *([self.python.parent] if self.python is not None else []),
            *(d for c in self.components for d in c.path()),
        ]
        tail = base_path if base_path else SYSTEM_PATH
        parts = [*map(str, dirs), tail] if (dirs or base_path) else []
        path = {"PATH": ":".join(parts)} if parts else {}
        comps = (
            {k: v for comp in self.offered(run) for k, v in comp.env(run).items()} if run is not None else {}
        )
        return {**path, **comps, **self.env}

    def qda_cmd(self) -> list[str]:
        """Select the explicit `qda` argv, configured Python module invocation or `qda` on `PATH`.

        The host binds this program's installation read-only for jobs. Without the `qda` component,
        it identifies the job interpreter but is never run by the host.
        """
        if self.qda is not None:
            return list(self.qda)
        return [str(self.python), "-m", "qda_quant.cli"] if self.python is not None else ["qda"]

    def qda_env(self, run: Run | None = None, *, trusted: bool = False) -> dict[str, str]:
        """Build `qda` environment overrides from components, then explicit `env` settings.

        `trusted` leaves out every component value that names a path in the agent's workdir (the `source`
        component's `PYTHONPATH` to the copy the agent patches): the judge's own builds, smokes and probes run
        the run's qda-quant, never code the agent can write. The agent's builds keep the whole environment.
        """
        comps: dict[str, str] = {}
        if run is not None:
            for comp in self.offered(run):
                for k, v in comp.env(run).items():
                    if not (trusted and _in_workdir(v, run.workdir)):
                        comps[k] = v
        return {**comps, **self.env}


def _in_workdir(value: str, workdir: Path) -> bool:
    """Whether any `os.pathsep` part of an environment value is a path inside `workdir`."""
    root = Path(os.path.abspath(workdir))
    return any(
        Path(os.path.abspath(p)).is_relative_to(root) for p in value.split(os.pathsep) if os.path.isabs(p)
    )


class Qda(Toolset):
    """Provide the default build, scoring, generation-check, submission and workspace components."""

    name: Literal["qda"] = "qda"
    preset: ClassVar[tuple[str, ...]] = (
        *("qda", "bpw", "kl", "gencheck", "qsched", "handin", "wiki", "fisher", "probe"),
        *("data", "serve", "baselines", "views", "source"),
    )


class Public(Toolset):
    """Reserve a public-library preset, refused until jobs can use an interpreter without qda."""

    name: Literal["public"] = "public"

    @model_validator(mode="after")
    def _refused(self) -> Public:
        raise ValueError(
            "toolset `public` (public libraries only, no qda) is not available yet: the agent's jobs and its "
            "wall still bind the Python that runs qda-agent, so qda_quant would stay importable. It needs a "
            "component with its own job interpreter (see the `isolation` package)"
        )
