"""The run config models: one search configured from YAML, with named plugins for each replaceable part.

`qda_agent.config` is the public name and imports these on first use, so importing it pulls neither the
judge nor the scheduler (their settings classes are named by fields here).

Plugins use a key and settings (`qda_agent.settings`). Example (`configs/search.yaml`)::

    model: Qwen/Qwen3-8B
    format: w4a16
    budget_gpu_hours: 4         # over all cards: 1 card -> a 4 h search
    cards: [4]
    run_dir: runs/q8b-w4a16     # relative paths are relative to this file
    agent:
      backend: {name: claude, model: claude-opus-5-5, effort: high}
      fallbacks: [{name: codex, model: gpt-5.6-sol, effort: high}]
    sandbox: {name: bwrap}
    scheduler: {name: qsched}
    launcher: {name: local}
    toolset: {name: qda, python: venvs/quant/bin/python}
    contamination: {name: ngram, guard: guard.txt}

`RunConfig.load` applies dotted overrides and resolves configured paths: relative to the YAML file when written
there, relative to the current directory when given with `--set`.
`resolved` adds commit and plugin provenance for `OUT/resolved.yaml`, which can be loaded again.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import warnings
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    SerializeAsAny,
    ValidationError,
    field_validator,
    model_validator,
)
from qda_kl.datasets import DATASETS
from qda_kl.eval.suites import SUITES, parse_suite
from qda_quant.format import Format

from qda_agent import rpc
from qda_agent.agents import BACKENDS, AgentConfig
from qda_agent.components import Run
from qda_agent.configfile import ConfigError
from qda_agent.configfile import load as load_config
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.contamination import CHECKS
from qda_agent.judge.final_rules import FINAL_RULES
from qda_agent.launchers import LAUNCHERS, Launcher, Local
from qda_agent.run_sources import check
from qda_agent.sandbox_settings import SANDBOXES, Bwrap, Sandbox
from qda_agent.sched.config import SCHEDULERS, Qsched, Scheduler
from qda_agent.settings import (
    Plugin,
    Strict,
    git_commit,
    pick,
    provenance,
)
from qda_agent.toolsets import DEFAULT, TOOLSETS, Qda, Toolset

__all__ = ["IsolationSettings", "RunConfig"]


def _plugin(registry: Any, default: str) -> Any:
    return BeforeValidator(lambda value: pick(registry, value, default))


def _legacy_level(sandbox: Sandbox, scheduler: Any) -> str:
    """The level the legacy knobs describe (`_isolation_level`): bwrap on both sides is `strict`."""
    jobs = getattr(scheduler, "job_sandbox", None)
    if sandbox.name == "none":
        return "none"
    return "sandbox" if jobs == "none" else "strict"


def _suite_plugins(suite: dict[str, Any] | None, prefix: str = "judge.suite") -> dict[str, tuple[Any, str]]:
    """The plugin of `judge.suite` and every dataset plugin nested in its settings, for the run's record.

    The settings are parsed the way the evaluator parses them, so the record names the plugins that will
    run: the suite by its registered name, and each dataset of the suite's `datasets` in list order.
    """
    if suite is None:
        return {}
    parsed = parse_suite(suite, prefix)
    found: dict[str, tuple[Any, str]] = {prefix: (SUITES, parsed.name)}
    for i, dataset in enumerate(getattr(parsed, "datasets", None) or ()):
        found[f"{prefix}.datasets.{i}"] = (DATASETS, dataset.name)
    return found


@functools.cache
def _site_dirs(exe: str) -> tuple[Path, ...]:
    """`exe`'s site-packages directories (sysconfig's `purelib` and `platlib`), as real paths.

    One subprocess per interpreter per process: `sysconfig` describes the site-packages of the Python that
    actually runs the jobs, which a path-shaped rule over the read paths cannot tell apart from a copy.
    """
    code = (
        "import json, sysconfig; "
        "print(json.dumps(sorted({sysconfig.get_paths()[k] for k in ('purelib', 'platlib')})))"
    )
    try:
        out = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=60, check=True)
        return tuple(sorted({Path(p).resolve() for p in json.loads(out.stdout)}))
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()


def _vendored_ifbench() -> Path | None:
    """The vendored IFBench checkers inside the real `qda_kl` package (importlib), as a real path."""
    try:
        spec = importlib.util.find_spec("qda_kl")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    d = Path(next(iter(spec.submodule_search_locations))) / "third_party" / "ifbench"
    return d.resolve() if d.is_dir() else None


def _every_family_at_every_group(menu: list[str]) -> None:
    """Refuse a resolved menu that offers a quantized family at one group and not at another.

    One checkpoint stores every quantized module at one group, so a family missing at a group the menu offers
    is unusable by every checkpoint at that group: a g64 checkpoint whose menu has `w8a16` at g128 only can
    promote nothing. A family its format fixes per channel (`w8a8-fp8`, whose scales are one an output row)
    has no other group to be offered at, so it takes part in neither side of the rule.
    """
    groups: dict[str, set[int]] = {}
    for name in menu:
        if name == "bf16":
            continue
        fmt = Format.parse(name)
        if fmt.group_size == -1:
            continue
        family = name.split("-")[0]  # the spelling's family prefix: `w8a16` of `w8a16-sym-g64`
        groups.setdefault(family, set()).add(fmt.group_size)
    offered = set().union(*groups.values()) if groups else set()

    def shown(g: int) -> str:
        return "channel" if g == -1 else f"g{g}"

    for family, at in groups.items():
        if missing := sorted(offered - at):
            raise ValueError(
                f"judge.format_menu: {family} is offered at {', '.join(shown(g) for g in sorted(at))} but not at "
                f"{', '.join(shown(g) for g in missing)}; one checkpoint takes one group, so a checkpoint at "
                f"{shown(missing[0])} could not use {family}: offer every family at every group the menu offers"
            )


def _local_storage(path: Path) -> None:
    """Refuse shared or unknown filesystems for a full-card search's writable directories."""
    resolved = path.resolve()
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, _, filesystem = line.partition(" - ")
        mount = fields.split()[4]
        for escaped, character in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
            mount = mount.replace(escaped, character)
        root = Path(mount)
        if resolved.is_relative_to(root):
            mounts.append((len(root.parts), filesystem.split()[0]))
    kind = max(mounts)[1] if mounts else "unknown"
    if kind not in {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay"}:
        raise ValueError(f"full_card_moe {path}: filesystem {kind} is not supported node-local storage")


class SharedNodeAcceptance(Strict):
    """An operator's acceptance that the agent's turns may share this node with the user's other work.

    `strict` and `sandbox` ask for a `pid: shared` turn (the host's pid namespace) where the host refuses a
    fresh procfs, and refuse the run when this user has anyone else's work on the node: a same-user signal
    or a `ptrace` would reach it. Naming this acceptance lifts exactly that refusal, for a run the operator
    is watching; the file wall, the network wall, the host proxy and the user namespace are unchanged, and
    so is the refusal for every config that names no acceptance. The values are part of `resolved.yaml` and
    the wall writes them, with what else is on the node, to the launch log when it goes up.

    Attributes:
        accepted_by: Who accepted a shared node for this run.
        reason: Why, in one sentence: the run's purpose and the control that stands in for a dedicated node.
    """

    accepted_by: str = Field(min_length=1, pattern=r"\S")
    reason: str = Field(min_length=1, pattern=r"\S")


class IsolationSettings(Strict):
    """How a run holds its processes apart (`isolation.level`).

    Attributes:
        level: `none` runs plain subprocesses (no bwrap, network namespace, host proxy, Slurm or
            dedicated-node check): the product default, for a workstation, a cloud VM or a container.
            `sandbox` runs the agent's turns and its jobs in bwrap with read-only workspace mounts, no
            network and their own pid namespace where the host allows one. `strict` is the experiment
            wall: everything `sandbox` holds, plus the agent's network namespace with the model API through
            the host proxy, the dedicated-allocation check and no reach into other runs. A run that names
            neither the level nor a legacy sandbox setting runs at `none`; one that names a legacy setting
            takes its level from it (see `RunConfig._isolation_level`).
        tamper_check: At `none`, `digest` detects a change to the judge's state, the reference or the
            scorer at every job (so before every reading) and at the end, and refuses that reading;
            `none` runs without the check. It is detection, not prevention (a wall is what prevents), so
            the run records what it checked (`run_dir/trust.jsonl`).
        shared_node: An operator's acceptance of a node this user already shares (`SharedNodeAcceptance`):
            it lifts the wall's dedication refusal for a `pid: shared` turn and nothing else. Unset, the
            refusal stands exactly as before; it is not read from the environment, so a turn cannot set it.

    The level is the run's choice in one place. `qda_agent.isolation.api` applies it: the flow and the
    scheduler move their isolation call sites to that interface in the following steps.
    """

    level: Literal["none", "sandbox", "strict"] = "none"
    tamper_check: Literal["digest", "none"] = "digest"
    shared_node: SharedNodeAcceptance | None = None


#: What the reference check already reported in this process, by (reference, reason): one line per reason.
_REFERENCE_SKIPS: set[tuple[str, str]] = set()


def _report_reference_skip(reference: Path, why: str) -> None:
    """Warn once per process and reason that a KL reference's model could not be checked."""
    key = (str(reference), why)
    if key in _REFERENCE_SKIPS:
        return
    _REFERENCE_SKIPS.add(key)
    warnings.warn(f"judge.kl_ref {reference}: {why}", UserWarning, stacklevel=2)


#: The first line of every search config `qda-agent prepare` writes (`prepare.HEADER`).
PREPARE_MARK = "# Written by `qda-agent prepare`"


def prepare_again(path: Path) -> str | None:
    """The `qda-agent prepare` command that writes `path` again, when prepare wrote it; None otherwise.

    The model is the Hub id `prepare.json` beside it records, else the config's `model`; the format, cards and
    budget are the config's. Prepare skips every step already done and rewrites only the config.
    """
    try:
        text = path.read_text()
        raw = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not text.startswith(PREPARE_MARK) or not isinstance(raw, dict):
        return None
    model = raw.get("model")
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        model = json.loads((path.parent / "prepare.json").read_text())["model"]["repo"]
    argv = ["qda-agent", "prepare", "--model", str(model), "--out", str(path.resolve().parent)]
    if raw.get("format"):
        argv += ["--format", str(raw["format"])]
    if isinstance(raw.get("cards"), list):
        argv += ["--cards", ",".join(map(str, raw["cards"]))]
    if isinstance(raw.get("budget_gpu_hours"), (int, float)):
        argv += ["--budget-gpu-hours", f"{raw['budget_gpu_hours']:g}"]
    backend = ((raw.get("agent") or {}).get("backend") or {}).get("name")
    if backend and backend != "claude":
        argv += ["--backend", str(backend)]
    return shlex.join(argv)


class RunConfig(Strict):
    """Validate one search and derive its scheduler, judge and recorded configuration.

    Attributes:
        model: The original model (bf16 or FP8), a path or a hub id.
        format: The format preset to search in (`w4a16`, `w4a8`, `w3a16`).
        budget_gpu_hours: GPU budget over all cards; wall time = budget / len(cards).
        cards: The GPUs the search holds for its whole budget.
        run_dir: Where the run writes everything; the agent's workdir is `run_dir/work` unless `work_dir` is set.
        work_dir: The agent's workdir in a fresh directory of its own instead of `run_dir/work`: on node-local
            storage, the agent's own builds and every job's writes land there, as the judge's builds land in
            `judge.custody_dir`, while the records, the jobs' logs and the exported winner stay in `run_dir`. It
            is checked as `judge.custody_dir` is (fresh, below trusted ancestors, apart from every input, bind,
            `run_dir` and `judge.custody_dir`), and counts in `judge.max_run_bytes`. Node loss before the
            export loses the agent's work, never the run's records.
        agent: The worker and its fallbacks (`qda_agent.backends`).
        sandbox: The agent's sandbox (`qda_agent.sandbox_settings`).
        isolation: The run's isolation level (`qda_agent.isolation.api`); a config that names no level
            takes it from `sandbox.name` and `scheduler.job_sandbox` (see `_isolation_level`).
        launcher: How the whole run is started (`qda_agent.launchers`).
        scheduler: The GPU scheduler (`qda_agent.schedulers`) over the allocated cards;
            the cards, the deadline and the drain window are filled in from this config.
        toolset: What the agent's jobs run (`qda_agent.toolsets`): their Python, `PATH` and `qda`.
        contamination: The contamination check (`qda_agent.contamination_checks`), run over every file the
            agent can read before it starts and on every checkpoint the judge freezes.
        judge: How the host scores and checks (KL and check commands, the format gate, ...); the workdir,
            socket, baselines, `qda` command and contamination check are filled in from this config.
        task: A task-statement template replacing the built-in one (`string.Template`, see
            `qda_agent.prompt`).
        calib: The default calibration file named in the task statement.
        baselines: Baseline checkpoints, in fallback order; `judge.starts` labels them as the row's pre-built
            candidate 0s (when it is empty they are unlabelled, as before).
        drain_timeout_s: After the deadline, how long pending checks and KL readings may take; None: one
            KL reading of the anchor (at least 900 s), or 900 s without an anchor.
        commands: `module:function` installing command wrappers into the workdir,
            `function(bindir, socket_path)`; None installs none (a host with no socket; tests).
        host: `module:factory` of the host side, called `factory(run_dir, sched, judge, isolation=,
            run_config=)` with the dicts of `sched_config` and `judge_config`, the run's isolation and the run
            config (tests swap in a fake host).
    """

    model: str
    format: str
    budget_gpu_hours: float = Field(gt=0)
    cards: list[int] = Field(min_length=1)
    run_dir: Path
    work_dir: Path | None = None
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sandbox: Annotated[SerializeAsAny[Sandbox], _plugin(SANDBOXES, "bwrap")] = Field(default_factory=Bwrap)
    isolation: IsolationSettings = Field(default_factory=IsolationSettings)
    scheduler: Annotated[SerializeAsAny[Scheduler], _plugin(SCHEDULERS, "qsched")] = Field(
        default_factory=Qsched
    )
    launcher: Annotated[SerializeAsAny[Launcher], _plugin(LAUNCHERS, "local")] = Field(default_factory=Local)
    toolset: Annotated[SerializeAsAny[Toolset], _plugin(TOOLSETS, DEFAULT)] = Field(default_factory=Qda)
    contamination: Annotated[SerializeAsAny[Plugin], _plugin(CHECKS, "ngram")] = Field(
        default="ngram", validate_default=True
    )
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    #: Whether the run named `isolation.level` itself (a copy must keep a named level, not re-derive it).
    _level_named: bool = PrivateAttr(default=False)
    task: Path | None = None
    calib: str | None = None
    baselines: list[Path] = Field(default_factory=list)
    drain_timeout_s: float | None = Field(default=None, ge=0)
    commands: str | None = "qda_agent.commands:install_bin"
    host: str = "qda_agent.judge:make_host"

    #: The workspace manifest `prepare` wrote (`qda_agent.workspace`), while this config object lives. Not a
    #: field: it is derived, never configured or recorded in `resolved.yaml`.
    _manifest: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _row_format_menu(self) -> RunConfig:
        """Spell every menu entry in full: family, group and symmetry; refuse a promoted entry that is not sym.

        A bare entry (``w4a16``) takes the row's group, and the row's symmetry when it is the row's own family;
        a wider family (a promoted module) is symmetric. An entry may name its own group and symmetry
        (``w4a16-asym-g64``), so a row can offer several groups, and a row that names no symmetry both
        symmetries of its body; a wider family
        named asymmetric is refused: promoted modules are stored symmetric. An entry of the row's own family that
        names the other symmetry is refused when the row names its symmetry: the row's spelling fixes it, and a
        menu entry could otherwise move the whole checkpoint to the other one. A menu that offers a family at one
        group and not at another is refused (`_every_family_at_every_group`). The judge's menu gate checks the
        family, group and symmetry of every quantized module.
        """
        if self.judge.engine == "llama.cpp":
            return self
        row = Format.parse(self.format)
        if "format_menu" not in self.judge.model_fields_set:
            if row.weight_bits == 3:
                self.judge.format_menu = ["bf16", "w3a16", "w4a16", "w8a16"]
            elif row.act == "nvfp4_block":
                # W4A4: the card serves NVFP4 dense and FP8 W8A8 for a wider module; the W4 default
                # (w8a16 / w4a16 at group 16) is refused by the sm100 and sm103 tables
                self.judge.format_menu = ["bf16", "w8a8-fp8", row.spelling]
            elif row.act != "bf16":  # W4A8: FP8 activations, so the row's own family is in its menu
                self.judge.format_menu = [*self.judge.format_menu, row.spelling]
        menu = []
        for name in self.judge.format_menu:
            if name == "bf16":
                menu.append(name)
                continue
            fmt = Format.parse(name)
            own = (fmt.weight_bits, fmt.act) == (row.weight_bits, row.act)
            if not ("-g" in name or name.endswith("-pc")):
                fmt = replace(fmt, group_size=row.group_size)
            if fmt.symmetric_defaulted and not (own and row.symmetric_defaulted):
                # a row whose format names no symmetry lets the planner store banks sym beside asym dense
                # modules, so its own family keeps admitting both; every other bare entry is pinned
                fmt = replace(fmt, symmetric=row.symmetric if own else True, symmetric_defaulted=False)
            if fmt.weight_bits > row.weight_bits and not fmt.symmetric:
                raise ValueError(
                    f"judge.format_menu: {name} is a wider format than the row's {self.format} and asymmetric; "
                    "promoted modules are stored symmetric"
                )
            if own and not row.symmetric_defaulted and fmt.symmetric != row.symmetric:
                theirs, ours = ("sym", "asym") if fmt.symmetric else ("asym", "sym")
                raise ValueError(
                    f"judge.format_menu: {name} is {theirs} and the row's {self.format} is {ours}; one checkpoint "
                    "takes one symmetry per family and the row's spelling fixes it for its own family: spell the "
                    f"entry {ours}, or leave its symmetry out so it follows the row"
                )
            menu.append(fmt.spelling if fmt.symmetric_defaulted else fmt.name)
        self.judge.format_menu = list(dict.fromkeys(menu))
        _every_family_at_every_group(self.judge.format_menu)
        return self

    @model_validator(mode="after")
    def _fixed_map_statement(self) -> RunConfig:
        """Refuse a fixed allocation (`judge.format_map`) stated by a template that offers per-module formats.

        The built-in statement's `${group_line}` and `${module_formats_line}` tell the agent that bit width
        may vary per module and how to widen one; under a fixed map that is false. So a run with a map names its
        own task template (`task`), and that template uses neither placeholder: it states the fixed map itself.
        """
        if self.judge.format_map is None or self.judge.engine == "llama.cpp":
            return self
        if self.task is None:
            raise ValueError(
                "judge.format_map fixes every module's format, but no task template is named: the built-in "
                "statement offers per-module formats; name a task file (task:) that states the fixed map"
            )
        if self.task.is_file():
            text = self.task.read_text(errors="replace")
            for key in ("group_line", "module_formats_line"):
                if f"${{{key}}}" in text or re.search(rf"\${key}\b", text):
                    raise ValueError(
                        f"judge.format_map fixes every module's format, but task {self.task} renders ${{{key}}}, "
                        "which offers per-module formats: drop it and state the fixed map in the template"
                    )
        return self

    @model_validator(mode="after")
    def _serve_default(self) -> RunConfig:
        """Default a serving component's interpreter to `judge.serve_python` when available.

        This shares the judge's vLLM environment with the agent's generation jobs. Omit the serving component
        from `toolset.components` to leave it unavailable.
        """
        for c in self.toolset.components:
            c.default_python(self.judge.serve_python)
        return self

    @model_validator(mode="after")
    def _starts_scope(self) -> RunConfig:
        """A start is our own build or the upstream GSQ (W3A16 rows only), and it labels a baseline.

        The standing rule: no published or vendor checkpoint is a start; the one exception is an upstream GSQ
        build, and only on a W3A16 row. A start's path is one of `baselines` (the agent's jobs read it there),
        so `run_sources`' own rule decides where it may lie.
        """
        family = self.format.split("-")[0]
        for i, start in enumerate(self.judge.starts):
            if start.origin == "gsq" and family != "w3a16":
                raise ValueError(
                    f"judge.starts[{i}] ({start.name}): a gsq start is only allowed on a W3A16 row, and this "
                    f"row's format is {self.format}: an upstream GSQ build is W3A16 only"
                )
        if self.judge.starts:
            paths = [os.path.normpath(start.path) for start in self.judge.starts]
            if len(set(paths)) != len(paths):
                raise ValueError("judge.starts names one path twice: every start is one checkpoint")
            if sorted(paths) != sorted(os.path.normpath(b) for b in self.baselines):
                raise ValueError(
                    "judge.starts must label every baselines path once: a start is a baseline the "
                    "agent's jobs read, and the bar is the lowest valid KL of the starts"
                )
            names = [start.name for start in self.judge.starts]
            if len(set(names)) != len(names):
                raise ValueError("judge.starts names one label twice: TASK.md shows each start once")
        return self

    @model_validator(mode="after")
    def _isolation_level(self) -> RunConfig:
        """Fill `isolation.level` from the legacy sandbox settings when the run names no level.

        A config that names neither `sandbox` nor `scheduler.job_sandbox` keeps the product default, `none`:
        agent search must run on a workstation, a cloud VM or a container. The two old knobs describe one
        level: `sandbox.name` says how the agent's turns run and `scheduler.job_sandbox` how its jobs do.
        bwrap on both sides is `strict`, which every search config that sets them gets; neither side is
        `none`. A config that names one bwrap side only is `sandbox`, the level that holds both, and warns
        where that differs from what it ran before. An explicit `isolation.level` wins, and the legacy
        settings stay in the record.
        """
        self._level_named = (
            "isolation" in self.model_fields_set and "level" in self.isolation.model_fields_set
        )
        if self._level_named:
            return self
        legacy = "sandbox" in self.model_fields_set or "job_sandbox" in self.scheduler.model_fields_set
        if not legacy:  # the product default: a workstation, a cloud VM or a container needs no sandbox
            return self
        jobs = getattr(self.scheduler, "job_sandbox", None)
        level = _legacy_level(self.sandbox, self.scheduler)
        if self.sandbox.name == "none" and jobs == "bwrap":
            warnings.warn(
                "sandbox: none with scheduler.job_sandbox: bwrap runs the agent's jobs confined and the "
                "agent unconfined; isolation.level is none, which runs both as plain processes. Name "
                "isolation.level to choose.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif self.sandbox.name != "none" and jobs == "none":
            warnings.warn(
                "scheduler.job_sandbox: none with a bwrap agent sandbox runs the agent confined and its jobs "
                "unconfined; isolation.level is sandbox, which runs both in bwrap. Name isolation.level to "
                "choose.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.isolation = self.isolation.model_copy(update={"level": level})
        return self

    @model_validator(mode="after")
    def _reference_matches_model(self) -> RunConfig:
        """Refuse a KL reference built from another model, and say so when the check cannot be made.

        A search scores every candidate against `judge.kl_ref`, the reference `qda-kl ref` built from the
        original model. A fine-tune of a released model is the run's own original: a reference built from
        the model it was fine-tuned from would score every candidate against the wrong model, and the
        readings would look plausible. When the reference's manifest records its model's `config_sha256`
        and the run's original is a local checkpoint, the two must agree.

        When the check cannot be made it is reported once, naming why (the original is a Hub id, the
        manifest records no `config_sha256`, or it cannot be read), and the run is not refused: the config
        may be an operator's intent, and the hash is only recorded by recent `qda-kl ref` builds.
        """
        reference = self.judge.kl_ref
        if not isinstance(reference, Path):
            return self
        original = Path(self.judge.anchor) if self.judge.anchor is not None else Path(self.model)
        try:
            manifest = json.loads((reference / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            _report_reference_skip(
                reference,
                f"its manifest.json is missing or unreadable, so its model was not checked against {original}",
            )
            return self
        model = manifest.get("model") if isinstance(manifest, dict) else None
        recorded = model.get("config_sha256") if isinstance(model, dict) else None
        config = original / "config.json"
        if not recorded or not config.is_file():
            why = (
                "its manifest records no model.config_sha256"
                if not recorded
                else f"{original} is not a local checkpoint directory with config.json (a Hub id is not checked)"
            )
            _report_reference_skip(reference, f"{why}, so its model was not checked against {original}")
            return self
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        if digest != recorded:
            raise ValueError(
                f"judge.kl_ref {reference} was built from another model: its manifest records "
                f"model.config_sha256 {recorded}, and {config} hashes {digest}. Build the reference from "
                f"this run's model: qda-kl ref --model {original} ..."
            )
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> RunConfig:
        """Copy, keeping the isolation level the copy should have.

        `model_copy` runs no validator, so a copy that changes `sandbox` or `scheduler` for settings the
        legacy mapping reads differently would otherwise keep this object's level, and every check and
        record read afterwards would name the wrong isolation. A copy that changes a knob follows it (as a
        config naming that knob would); a level the run named itself is kept; a copy that supplies
        `isolation` names the level it supplies, so the next copy keeps it.
        """
        copy = super().model_copy(update=update, deep=deep)
        changed = False
        if update is not None:
            changed = ("sandbox" in update and update["sandbox"] != self.sandbox) or (
                "scheduler" in update and update["scheduler"] != self.scheduler
            )
        if update and "isolation" in update:
            # This copy supplied the level record: it names its level, and the next copy must keep it.
            copy._level_named = True
        elif changed and not self._level_named:
            # A copy that changes a legacy knob follows it, exactly as a config that named that knob would.
            # A level the run named itself (even one the knobs would also describe) is that choice and is
            # kept, and an unrelated copy (the same settings copied) leaves the level where it was.
            copy.isolation = copy.isolation.model_copy(
                update={"level": _legacy_level(copy.sandbox, copy.scheduler)}
            )
        return copy

    @model_validator(mode="after")
    def _sources_outside_the_run(self) -> RunConfig:
        """Refuse a read-only input that lies in the run dir, or holds it (see `qda_agent.run_sources`)."""
        check(self)
        return self

    @model_validator(mode="after")
    def _receipt_custody(self) -> RunConfig:
        """Require process confinement for receipt custody; the normal run-dir wall hides it."""
        if (self.judge.require_build_receipt or self.judge.custody_dir is not None) and (
            self.sandbox.name != "bwrap" or getattr(self.scheduler, "job_sandbox", None) != "bwrap"
        ):
            raise ValueError(
                "judge.require_build_receipt / judge.custody_dir needs bwrap for the agent and its jobs"
            )
        return self

    @model_validator(mode="after")
    def _full_card_moe(self) -> RunConfig:
        """Validate the single-node 8-card search profile before opening a run."""
        if not self.judge.full_card_moe:
            return self
        if len(self.cards) != 8:
            raise ValueError("full_card_moe needs exactly 8 allocated cards")
        if self.judge.kl_layout.tp != 8 or self.judge.kl_layout.cards != 8:
            raise ValueError("full_card_moe needs judge.kl_layout.tp=8 and one pipeline stage")
        if self.judge.suite is not None:
            raise ValueError("full_card_moe uses judge.kl_layout; judge.suite is not supported")
        if not self.judge.require_build_receipt:
            raise ValueError("full_card_moe requires judge.require_build_receipt: true")
        for setting in ("gencheck_cards", "check_cards"):
            if setting not in self.judge.model_fields_set:
                setattr(self.judge, setting, len(self.cards))
            if getattr(self.judge, setting) != len(self.cards):
                raise ValueError(f"full_card_moe needs judge.{setting} to cover all allocated cards")
        if "check_cmd" not in self.judge.model_fields_set and self.judge.check_cmd is not None:
            self.judge.check_cmd = [*self.judge.check_cmd, "--tp", "{cards}"]
        if self.judge.anchor is None or Path(self.model).resolve() != self.judge.anchor.resolve():
            raise ValueError(
                "full_card_moe requires model and judge.anchor to name the same local FP8 source"
            )
        model_config = Path(self.model) / "config.json"
        try:
            source_bytes = model_config.read_bytes()
            source = json.loads(source_bytes)
        except (OSError, json.JSONDecodeError):
            raise ValueError("full_card_moe model must be a local checkpoint with config.json") from None
        quantization = source.get("quantization_config", {}) if isinstance(source, dict) else {}
        if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
            raise ValueError("full_card_moe model config must declare quantization_config.quant_method: fp8")
        if not isinstance(self.judge.kl_ref, Path):
            raise ValueError("full_card_moe judge.kl_ref must be a local qda-kl reference directory")
        manifest = self.judge.kl_ref / "manifest.json"
        try:
            reference = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            raise ValueError("full_card_moe judge.kl_ref must contain a qda-kl manifest") from None
        build = reference.get("build") if isinstance(reference, dict) else None
        if not isinstance(build, dict) or build.get("tp") != 8:
            raise ValueError("full_card_moe judge.kl_ref manifest must record build.tp=8")
        original = reference.get("model")
        if (
            not isinstance(original, dict)
            or original.get("config_sha256") != hashlib.sha256(source_bytes).hexdigest()
        ):
            raise ValueError("full_card_moe judge.kl_ref must record the FP8 source config_sha256")
        if self.judge.custody_dir is None or self.judge.max_run_bytes is None:
            raise ValueError("full_card_moe requires judge.custody_dir and finite judge.max_run_bytes")
        if self.judge.max_copy_bytes is None or self.judge.max_run_bytes < 2 * self.judge.max_copy_bytes:
            raise ValueError(
                "full_card_moe requires judge.max_copy_bytes and judge.max_run_bytes >= 2 * judge.max_copy_bytes"
            )
        _local_storage(self.run_dir)
        _local_storage(self.judge.custody_dir)
        return self

    @field_validator("cards")
    @classmethod
    def _distinct(cls, cards: list[int]) -> list[int]:
        if len(set(cards)) != len(cards):
            raise ValueError(f"cards {cards} name a card twice")
        return cards

    @property
    def wall_s(self) -> float:
        """Convert the total GPU-hour budget to wall-clock seconds across the allocated cards."""
        return self.budget_gpu_hours * 3600.0 / len(self.cards)

    def drain_window_s(self) -> float:
        """Seconds pending judge jobs may finish after the deadline: `drain_timeout_s` or the derived one."""
        return self.judge.drain_window_s() if self.drain_timeout_s is None else self.drain_timeout_s

    def time_ratio(self, wall_s: float) -> float | None:
        """How long the search ran against candidate 0's build: None unless `judge.candidate0_build_s` is set."""
        build = self.judge.candidate0_build_s
        return None if build is None else round(wall_s / build, 3)

    @property
    def workdir(self) -> Path:
        """Return the agent's writable work directory: `work_dir`, else `run_dir/work`."""
        return Path(os.path.abspath(self.work_dir)) if self.work_dir is not None else self.run_dir / "work"

    @property
    def socket(self) -> Path:
        """Locate the command RPC socket inside the work directory.

        The workdir is the one directory of the run that every sandboxed run mounts into its sandbox.
        """
        return self.workdir / ".qda" / "rpc.sock"

    @property
    def host_socket(self) -> Path:
        """Where the host listens: `socket`, or `rpc.sock` in the run's private directory while one is held.

        The flow holds one (`rpc.private_dir`) when `socket` is too long for a unix socket (`needs_private_socket`).
        """
        d = rpc.private_dir_of(self.run_dir)
        return self.socket if d is None else d / "rpc.sock"

    @property
    def needs_private_socket(self) -> bool:
        """Whether the host must listen in a short private directory.

        That is when `socket` does not fit a unix socket and the sandbox keeps one (`Sandbox.private_socket`).
        """
        return self.sandbox.private_socket and not rpc.fits(self.socket)

    @property
    def agent_socket(self) -> Path:
        """The socket as the agent's commands name it.

        That is `socket` where the sandbox maps it over its parent, else the host's path (a sandbox that sees
        the host's files, `Sandbox.host_paths`).
        """
        return self.host_socket if self.sandbox.host_paths else self.socket

    def socket_binds(self) -> dict[Path, Path]:
        """`{work/.qda: private directory}` while one is held: the read-only bind that shows it at `socket`."""
        d = rpc.private_dir_of(self.run_dir)
        return {} if d is None else {self.socket.parent: d}

    def manifest(self) -> Any:
        """The workspace manifest: what this run gives the agent (`qda_agent.workspace.build_manifest`).

        `prepare` builds the full record (files, texts and bind digests) and stores it here; before that, and
        for a config check or the wall, the plan alone is built (no file system walk) and cached. The wall and
        the scheduler binds read this one object, so they cannot disagree with TASK.md or the contamination
        scan.
        """
        if self._manifest is None:
            from qda_agent.workspace import build_manifest

            self._manifest = build_manifest(self, self.toolset)
        return self._manifest

    def read_paths(self) -> tuple[list[Path], dict[Path, Path]]:
        """The host paths the agent's jobs read, and the binds that show them under the workdir.

        Both come from the workspace manifest: the list holds the paths bound read-only at the same path
        (`scheduler.readable`, `toolset.readable()`, the baselines not bound under the workdir, and an
        unwalled run's private socket directory); the mapping is the manifest's job binds. `sched_config`
        hands both to the scheduler; `qda_agent.flow.readable_inputs` and `qda_agent.isolation` scan them.
        The plan is rebuilt here rather than read from `manifest()`: a check runs against the config as it
        is now, and a record cached before a later edit would hide that edit.
        """
        from qda_agent.workspace import build_manifest

        manifest = build_manifest(self, self.toolset)
        return [Path(p) for p in manifest.readable], manifest.job_binds()

    def interpreters(self) -> list[Path]:
        """The interpreters this run uses: the flow's own, the judge's, and the toolset's and its components'.

        `contamination_skips` asks each for its site-packages; a configured path that is not one of these (a
        copy merely named `site-packages`) is not an install and stays scanned.
        """
        found = [Path(sys.executable), *([self.judge.serve_python] if self.judge.serve_python else [])]
        found += [
            p
            for p in [self.toolset.python, *(c.job_python() for c in self.toolset.components)]
            if p is not None
        ]
        return list(dict.fromkeys(found))

    def contamination_skips(self) -> list[Path]:
        """The installed trees the pre-start contamination scan leaves out, as exact paths.

        The site-packages directories of every interpreter the run uses (sysconfig of that interpreter) and the
        vendored IFBench checkers inside the real `qda_kl` package (importlib); both are compared as real paths
        by prefix, never by a name tail or a pattern, so a copy of a checker tree, or a directory merely named
        `site-packages` under the read paths, is still scanned. Only entries that bear on a read path are
        listed. `resolved` records them, `flow.readable_inputs` leaves them out, and the run's configured data
        (workspace, calibration file and prompt set) is scanned regardless.
        """
        readable, binds = self.read_paths()
        roots = [r.resolve() for r in (*readable, *binds.values())]
        if self.calib:
            roots.append(Path(self.calib).resolve())
        if (prompts := self.judge.prompts_path()) is not None:
            roots.append(prompts.resolve())
        found = {p for exe in self.interpreters() for p in _site_dirs(str(exe))}
        if (vendor := _vendored_ifbench()) is not None:
            found.add(vendor)
        return sorted(p for p in found if any(p.is_relative_to(r) or r.is_relative_to(p) for r in roots))

    def task_renders(self, placeholder: str) -> bool:
        """Whether the run's task template carries `placeholder` (the built-in one carries its whole list)."""
        if self.task is None:
            return True
        try:
            return placeholder in self.task.read_text()
        except OSError:
            return True  # a missing template is the task-file check's to report

    def require_task_values(self) -> None:
        """Refuse, before the clock, a task statement whose values the config does not carry.

        Raises:
            ValueError: The statement states the evaluation's generation cap and the row does not.
        """
        if self.judge.eval_max_tokens is None and self.task_renders("${eval_max_tokens}"):
            raise ValueError(
                "the task statement states the evaluation's generation cap ('answers as long as the "
                "evaluation allows'): set judge.eval_max_tokens (the final evaluation's max_tokens for this row)"
            )

    def job_sandbox(self) -> Literal["bwrap", "none"]:
        """How the run's jobs run: what `sched_config` hands the judge and `resolved.yaml` records.

        One resolution, `isolation.api.job_sandbox(isolation.level, scheduler.job_sandbox)`, for the jobs the
        isolation starts, for the judge (`judge.startup.qda_resolution` decides a `qda build` job's `PATH` on
        it: `bwrap` keeps only the directories a job sandbox mounts) and for `qda doctor --search`. An unset
        `scheduler.job_sandbox` follows the level: `none` at `isolation.level: none`, so a judge build keeps
        the shell's `PATH`; `bwrap` at `sandbox` and `strict`, so no job leaves the sandbox unasked.
        """
        from .isolation.api import job_sandbox

        knob = getattr(self.scheduler, "job_sandbox", "bwrap")
        return cast(Literal["bwrap", "none"], job_sandbox(self.isolation.level, knob))

    def agent_job_env(self) -> dict[str, str]:
        """The environment of the agent's jobs, from its one owner: the toolset.

        A job that runs on the host — `isolation.level: none`, or `scheduler.job_sandbox: none`
        (`isolation.api.job_sandbox`) — keeps the launching `PATH` after the toolset's own directories, so
        the compiler and other tools the user set up still resolve; a job sandbox keeps the fixed system
        directories its mounts provide.
        """
        host = self.job_sandbox() != "bwrap"
        return self.toolset.job_env(self.run, base_path=os.environ.get("PATH") if host else None)

    def sched_config(self, deadline: datetime) -> dict[str, Any]:
        """Build scheduler settings with the run's cards, deadline, mounts and job environment.

        Reserve the drain window and the final rules' checks for judge admission past the deadline.
        """
        sched = self.scheduler.model_dump()
        readable, binds = self.read_paths()
        run = self.run
        return {
            **sched,
            "job_sandbox": self.job_sandbox(),
            "cards": self.cards,
            "deadline": deadline,
            "judge_grace_s": self.drain_window_s() + sum(rule.reserve_s() for rule in self.judge.final_rules),
            "readable": list(dict.fromkeys(map(str, readable))),
            "binds": {str(k): str(v) for k, v in binds.items()},
            "agent_job_env": self.agent_job_env(),
            # read again when a job starts: a link swapped after the check must not reach the run dir
            "run_dir": os.path.realpath(self.run_dir),
            "host_written": [] if run.overlay is None else [os.path.realpath(run.overlay)],
            # as written: the job sandbox masks every spelling (the home alias's real path and each target)
            "hide": [str(p) for p in getattr(self.sandbox, "hide", [])],
        }

    @property
    def run(self) -> Run:
        """Build the run context passed to toolset components.

        The commands and verbs the toolset installs are part of it: a component's TASK.md text names another
        component's command only when this run gives it (the workspace manifest records the same set).
        """
        base = Run(
            workdir=self.workdir,
            overlay=self.run_dir / "overlay",
            walled=self.sandbox.walled,
            model=self.model,
            format=self.format,
            baselines=tuple(self.baselines),
            engine=self.judge.engine,
        )
        return replace(
            base,
            commands=self.toolset.commands(base),
            subcommands={k: tuple(v) for k, v in self.toolset.subcommands(base).items()},
        )

    def judge_config(self) -> dict[str, Any]:
        """Build judge settings with run paths, baselines, commands and the contamination check."""
        return {
            **self.judge.model_dump(exclude_unset=True),
            "contamination": self.contamination,
            "workdir": self.workdir,
            "model": self.model,
            "format": self.format,
            "socket": self.host_socket,
            "baselines": self.baselines,
            "qda_cmd": self.toolset.qda_cmd(),
            "qda_env": self.toolset.qda_env(self.run, trusted=True),
            "agent_qda_env": self.toolset.qda_env(self.run),
            "commands": list(self.toolset.commands(self.run)),
            "subcommands": self.toolset.subcommands(self.run),
            "handlers": self.toolset.handlers(self.run),
        }

    def resolved(self) -> dict[str, Any]:
        """Serialize the run config with commit and plugin provenance for `OUT/resolved.yaml`.

        The size-derived defaults are recorded as their values: `drain_timeout_s`, `judge.max_copy_bytes` and
        `scheduler.job_sandbox` (`job_sandbox`: an unset knob follows `isolation.level`).
        An anchor the host refuses keeps its configured `max_copy_bytes`, so a replay refuses it the same way.
        `contamination_skipped` names the directories the startup scan leaves out, so the record shows them.
        A `judge.suite` is recorded with the package and version of the suite plugin it names, and of each
        dataset plugin nested in its settings.
        `RunConfig.load` accepts this record and discards provenance before validation.
        """
        d = self.model_dump(mode="json")
        if self.work_dir is None:
            d.pop("work_dir")  # preserve default run records byte for byte
        if self.judge.custody_dir is None:
            d["judge"].pop("custody_dir")  # preserve default run records byte for byte
        if self.judge.max_scratch_bytes is None:
            d["judge"].pop("max_scratch_bytes")
        if self.isolation.shared_node is None:
            d["isolation"].pop("shared_node")  # preserve default run records byte for byte
        if "job_sandbox" in d["scheduler"]:
            d["scheduler"]["job_sandbox"] = (
                self.job_sandbox()
            )  # what the jobs ran as: unset follows the level
        d["drain_timeout_s"] = self.drain_window_s()
        try:
            cap = self.judge.max_checkpoint_bytes()
        except ValueError:
            pass  # the anchor is unusable: the host refuses at start, and the replay must too
        else:
            d["judge"] = {**d["judge"], "max_copy_bytes": cap}
        plugins = {
            "agent.backend": (BACKENDS, self.agent.backend.name),
            **{f"agent.fallbacks.{i}": (BACKENDS, f.name) for i, f in enumerate(self.agent.fallbacks)},
            "sandbox": (SANDBOXES, self.sandbox.name),
            "scheduler": (SCHEDULERS, self.scheduler.name),
            "launcher": (LAUNCHERS, self.launcher.name),
            "toolset": (TOOLSETS, self.toolset.name),
            "contamination": (CHECKS, self.contamination.name),
            **{f"judge.final_rules.{i}": (FINAL_RULES, r.name) for i, r in enumerate(self.judge.final_rules)},
            **{
                f"judge.{key}": (DATASETS, next(iter(named)))
                for key in ("kl_ref", "gencheck_prompts")
                if isinstance(named := getattr(self.judge, key), dict)
            },
            **_suite_plugins(self.judge.suite),
            **{
                key: value
                for i, rule in enumerate(self.judge.final_rules)
                for key, value in _suite_plugins(
                    getattr(rule, "suite", None), f"judge.final_rules.{i}.suite"
                ).items()
            },
        }
        return {
            **d,
            "provenance": {
                "commit": git_commit(Path(__file__).parent),
                "plugins": {k: {"name": n, **provenance(r, n)} for k, (r, n) in plugins.items()},
                "contamination_skipped": [str(p) for p in self.contamination_skips()],
            },
        }

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | dict[str, Any] | None = None) -> RunConfig:
        """Load a run YAML or resolved config, applying overrides before validation and path resolution.

        See `qda_agent.configfile.load`; this is that loader bound to `RunConfig`. A config `qda-agent prepare`
        wrote that no longer validates was written by an earlier version: its refusal says so, and its `fix` is
        the prepare command that writes it again (`prepare_again`).
        """
        try:
            return load_config(cls, path, overrides)
        except ConfigError as e:
            if not isinstance(e.__cause__, ValidationError) or (again := prepare_again(Path(path))) is None:
                raise
            raise ConfigError(
                f"{e} (an earlier qda-agent prepare wrote it; this version does not read those keys)",
                fix=again,
            ) from e
