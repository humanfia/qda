"""Toolset components (`qda_agent.components`): the parts a toolset is assembled from.

A toolset (`qda_agent.toolsets`) is a list of components. Each component declares everything it adds to the
agent's world, and nothing outside it refers to a component by name:

| what | where it goes |
|---|---|
| `offers`: whether this run gets it at all | `offers(run)`; false: nothing in this table applies |
| `commands`: programs in `work/bin` | `commands.install_bin` links them; the host answers them |
| `subcommands`: verbs of a command | the host refuses a verb no component of the toolset adds |
| `handlers`: host code of a new command | `JudgeHost.handle` calls it (the built-in commands need none) |
| `install`, `protected`: files in the workdir | written at prepare; read-only ones in a host-owned overlay, bound over `work/<name>` |
| `binds`: read-only binds at another path | the job sandbox (`baselines/<N>`) |
| `section`: its lines of TASK.md | the statement's Tools list, in the toolset's order |
| `readable`, `path`, `env`: what its jobs need | bound read-only into the job sandbox / on `PATH` / set in every job |

So a toolset without `wiki` neither installs the skill nor says a word about it, and an ablation that removes
or adds a part is a list edit in the run config. A part that belongs to one engine's contract withholds itself
for the other engine (`bpw` prices bits per quantized weight, which a llama.cpp search does not use). A new
part (a named read-only dataset, a second interpreter for jobs, another `qsched` verb) is one more `Component`
subclass, registered here or through another package's `qda.plugins` hook.

The built-ins:

| key | adds |
|---|---|
| `qda` | `qda build` (a GPU job) and `qda plan` |
| `kl` | `kl DIR`, `kl list`, `kl show` |
| `gencheck` | `gencheck DIR [--max-tokens N] [--per-domain K]`, `gencheck list`, `gencheck show`: long generations, counts only |
| `qsched` | `qsched add / list / log / show / eta / status / usage / wait / cancel` |
| `handin` | `handin DIR [DIR...]`, `handin status` |
| `wiki` | the `quant-methods` skill, read-only, where the agent CLI finds skills |
| `data` | named read-only datasets for the jobs, listed in `DATA.md` |
| `serve` | a vLLM virtualenv on the jobs' `PATH` |
| `baselines` | each baseline at `baselines/<N>` in the jobs (neutral names) |
| `views` | the model's and baselines' config / tokenizer / index files, readable by the shell |
| `source` | a read-write copy of `qda_quant` in `src/`, imported by the agent's jobs |
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shutil
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from qda_kl.serve import NO_JIT_ENV
from qda_quant.intake.fisher import CACHE_RULE, CARD_RULE
from qda_quant.output.vllm_env import cache_env, disabled_kernels
from qda_quant.registry import Registry

from qda_agent.settings import Plugin, validate_plugin

__all__ = ["COMPONENTS", "Component", "Run"]

_log = logging.getLogger(__name__)

BUILTINS = {
    "baselines": "qda_agent.components:Baselines",
    "bpw": "qda_agent.components:Bpw",
    "data": "qda_agent.components:Data",
    "gencheck": "qda_agent.components:Gencheck",
    "handin": "qda_agent.components:Handin",
    "kl": "qda_agent.components:Kl",
    "qda": "qda_agent.components:Qda",
    "qsched": "qda_agent.components:Qsched",
    "serve": "qda_agent.components:Serve",
    "source": "qda_agent.components:Source",
    "views": "qda_agent.components:Views",
    "wiki": "qda_agent.components:Wiki",
}
COMPONENTS = Registry[type[Plugin]](
    "components",
    BUILTINS,
    group="qda_agent.components",
    on_add=partial(validate_plugin, "qda_agent.components"),
)


@dataclass(frozen=True)
class Run:
    """What a component may know about the run it is part of.

    Attributes:
        workdir: The agent's work directory.
        overlay: A host-owned directory outside the workdir (under the run dir, which the agent never sees).
            A component writes its read-only files here, under a top-level name of the workdir
            (`overlay/.claude/...` for `work/.claude/...`), and lists that name in `protected`.
        walled: The agent's shell is in the bwrap wall, which binds the overlay read-only over the workdir; else
            its shell gets a writable copy of each entry (its jobs still get the overlay).
        model: The original model (a path or a hub id).
        format: The format the search quantizes to (`format`), for a page that names one.
        baselines: The baseline checkpoints, in fallback order (host paths).
        engine: The serving engine the checkpoints are made for (`judge.engine`): `vllm` or `llama.cpp`.
            A component whose tool belongs to one engine's contract reads it here.
        commands: The commands this run's toolset installs (`Toolset.commands`), or None when the caller did
            not give a toolset (a direct `sections()` call): a section that names another component's command
            then keeps the name, as it did before the workspace manifest.
        subcommands: The verbs each installed command offers (`Toolset.subcommands`).
    """

    workdir: Path
    overlay: Path | None = None
    walled: bool = True
    model: str = ""
    format: str = ""
    baselines: tuple[Path, ...] = ()
    engine: str = "vllm"
    commands: tuple[str, ...] | None = None
    subcommands: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def ro(self, name: str) -> Path:
        """Where a component writes the read-only entry `name` (a top-level name of the workdir)."""
        if self.overlay is None:
            raise ValueError("this run has no overlay directory for read-only files")
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"{name!r} is not a top-level name of the workdir")
        return self.overlay / name


def _gives(run: Run, name: str) -> bool:
    """Whether this run installs the command `name`; a `Run` built without a toolset says yes (as before)."""
    return run.commands is None or name in run.commands


def _wrap(text: str) -> str:
    """`text` wrapped into the statement's clauses, continuation lines indented like the rest of the item."""
    return textwrap.fill(
        text, width=100, subsequent_indent="  ", break_on_hyphens=False, break_long_words=False
    )


class Component(Plugin):
    """Settings of one toolset component; the class attributes say what it adds.

    Class attributes:
        commands: Programs it puts in `work/bin` (each a link to the RPC client, answered by the host).
        subcommands: For each command, the verbs (first argument) it adds; a command's verbs are the union
            over the toolset, and the host refuses any other. A command absent here takes any argument.
        handlers: For a command the host does not answer itself (not `qda`, `kl`, `gencheck`, `qsched`, `handin`),
            `module:function`, called as `function(host, argv, cwd) -> qda_agent.rpc.Reply`.
        section_text: Its lines of TASK.md when they are a constant; "" for none.
    """

    commands: ClassVar[tuple[str, ...]] = ()
    subcommands: ClassVar[dict[str, tuple[str, ...]]] = {}
    handlers: ClassVar[dict[str, str]] = {}
    section_text: ClassVar[str] = ""

    def offers(self, run: Run) -> bool:  # noqa: ARG002
        """Whether this run gets the component at all (default: yes).

        A component whose tool belongs to one engine's contract says so here. One that does not offer itself
        contributes nothing to the run: no command, verb, handler, file, mount, environment or TASK.md line.
        The run-independent hooks (`readable`, `path`, `job_python`) are not gated, so a component that ever
        withholds itself declares nothing there.
        """
        return True

    def section(self, run: Run) -> str:  # noqa: ARG002
        """Its part of TASK.md: Markdown list items, or "" for none."""
        return self.section_text

    def install(self, run: Run) -> None:  # noqa: ARG002
        """Write its files before the agent starts: read-only ones under `run.ro(name)`, its own ones in the workdir."""

    def protected(self, run: Run) -> list[str]:  # noqa: ARG002
        """Top-level names of the workdir it writes under `run.overlay`.

        Each is bound read-only from the overlay over `work/<name>` in the wall, in every agent job and in the
        host's `qda` calls; the agent never writes the source.
        """
        return []

    def binds(self, run: Run) -> dict[Path, Path]:  # noqa: ARG002
        """More read-only binds of the agent's jobs, `{path in the workdir: host path}`.

        The path must lie under one of the toolset's `protected` names (so its parents are the host's), and the
        host path outside the workdir.
        """
        return {}

    def readable(self) -> list[Path]:
        """Paths outside the workdir its jobs read (bound read-only into the job sandbox at the same path)."""
        return []

    def path(self) -> list[Path]:
        """Directories put on every agent job's `PATH`, after the toolset's interpreter's `bin/`."""
        return []

    def env(self, run: Run) -> dict[str, str]:  # noqa: ARG002
        """Environment it adds to every agent job (the agent's code can read it: no secret)."""
        return {}

    def baseline_names(self, run: Run) -> list[str] | None:  # noqa: ARG002
        """How TASK.md names the baselines to the agent, when this component decides it (None: it does not)."""
        return None

    def job_python(self) -> Path | None:
        """The interpreter it adds to the agent's jobs, bound read-only into them (None: it adds none)."""
        return None

    def default_python(self, python: Path | None) -> None:  # noqa: ARG002
        """Take `python` (the judge's serving interpreter) as its own job interpreter when it has none."""


# ---------------------------------------------------------------------------------------------- the built-in commands


class Qda(Component):
    """`qda build` as a GPU job on the run's cards, and `qda plan` on the host without a GPU."""

    name: Literal["qda"] = "qda"
    commands: ClassVar[tuple[str, ...]] = ("qda",)
    subcommands: ClassVar[dict[str, tuple[str, ...]]] = {"qda": ("build", "plan")}

    section_text: ClassVar[str] = (
        "- `qda build CONFIG.yaml [--set key=value ...]` builds a checkpoint as a GPU job. A config names `model`,\n"
        "  `format`, `steps` (the methods in order, each with its settings), `calib` (the calibration data) and `out`\n"
        "  (the output dir), and may set `hessian`, `batch_size` and `target`; leave `devices` unset (the job sees\n"
        "  only its own cards). A top-level `seed` must stay 0: a step that draws random numbers has its own `seed`\n"
        "  setting (`rotate`, `gsq`, `scale_kd`), and `calib.seed` shuffles the calibration rows. `--set` overrides\n"
        "  one key by its dotted path: `--set out=ckpt/b`, and step N's settings as `steps.N.KEY`\n"
        "  (`--set steps.0.damp=0.01`). Every build writes `resolved.yaml` into its output\n"
        "  dir, which builds the same checkpoint again. `qda plan CONFIG.yaml` checks a config without a GPU.\n"
        "  `calib` names one source by its key: `calib: {file: {path: data.jsonl, samples: 128, seqlen: 2048}}`;\n"
        "  the other sources are `mix`, `slice`, `selfgen` and `hf`;\n"
    )


class Bpw(Component):
    """`bpw DIR` prices a checkpoint with the judge's own budget rule; `bpw --plan RECIPE` prices a recipe."""

    name: Literal["bpw"] = "bpw"
    commands: ClassVar[tuple[str, ...]] = ("bpw",)

    section_text: ClassVar[str] = (
        "- `bpw DIR` prices a checkpoint you made with the judge's own rule (bits per quantized weight: its\n"
        "  decoder Linears' packed codes, scales and zero points over the original's fixed decoder-Linear weight\n"
        "  count, and with the head lease the same after the head's lease) and compares it with the budget;\n"
        "  `bpw --plan RECIPE.yaml` prices a recipe's resolved per-module formats against the original's shapes,\n"
        "  per format class and in total, without building it;\n"
    )

    def offers(self, run: Run) -> bool:
        """Bits per quantized weight is the vLLM budget's rule; a llama.cpp search is bounded by .gguf bytes."""
        return run.engine != "llama.cpp"


class Kl(Component):
    """`kl DIR` scores a checkpoint against the original model (a judge job); `kl list`, `kl show ID`."""

    name: Literal["kl"] = "kl"
    commands: ClassVar[tuple[str, ...]] = ("kl",)

    section_text: ClassVar[str] = (
        "- `kl DIR` scores a checkpoint you made against the original model (overall, per domain, human text);\n"
    )


class Gencheck(Component):
    """`gencheck DIR` generates long answers with a checkpoint and the original model (a judge job), any time.

    Like `kl`, it is a search tool, not a step before a submission: the agent checks intermediate candidates
    whenever it likes, and the results may change its recipes. It gates nothing and never counts at the end.

    The reply counts, per domain and for both models, the answers cut at the length limit, the answers that
    end in a repetition loop and the mean answer length; the texts stay the judge's. The built-in prompt set is
    public and readable by the agent's jobs (it ships in `qda_kl`); for a hidden set, give
    `judge.gencheck_prompts`.
    """

    name: Literal["gencheck"] = "gencheck"
    commands: ClassVar[tuple[str, ...]] = ("gencheck",)

    section_text: ClassVar[str] = (
        "- `gencheck DIR [--max-tokens N] [--per-domain K]` has a checkpoint write long answers and counts, per\n"
        "  domain and against the original model, the answers cut at N tokens or ending in a loop, and their mean\n"
        "  length; N defaults to `judge.gencheck_max_tokens` and K to `judge.gencheck_per_domain`. A check\n"
        "  generates text, so it takes longer than a scoring run (`gencheck list`, `gencheck show ID`, each with\n"
        "  when it was asked and finished);\n"
    )


class Qsched(Component):
    """The run's GPU queue for any other job: `qsched add`, `list`, `log`, `cancel` and their read-only views."""

    name: Literal["qsched"] = "qsched"
    commands: ClassVar[tuple[str, ...]] = ("qsched",)

    section_text: ClassVar[str] = (
        "- `qsched` runs any other GPU job (`add`, `list`, `log`, `cancel`). A job is one script in your workspace\n"
        "  (`qsched add --cards N --expected DUR [--after JOB] [--name NAME] scripts/x.sh`); it runs as\n"
        "  `bash <copy>`, so interpreters and flags go inside the script. The script is copied when you submit\n"
        "  it, it runs on cards no other job uses, and no job starts if it would run past the deadline.\n"
        "  `qsched show NAME`, `qsched eta NAME` (its end time, from the job's progress lines), `qsched status`\n"
        "  (the cards and the queue), `qsched usage` (card-hours) and `qsched wait --job NAME | --any` (one\n"
        "  blocking call; its last line starts MET, FAILED or TIMEOUT: call it again on TIMEOUT instead of a\n"
        "  sleep loop);\n"
    )


class Handin(Component):
    """`handin DIR [DIR...]` submits one hand-in list, best first (each checked on a frozen copy)."""

    name: Literal["handin"] = "handin"
    commands: ClassVar[tuple[str, ...]] = ("handin",)

    section_text: ClassVar[str] = (
        "- `handin DIR [DIR...]` hands in checkpoint directories you made in this search, best first;\n"
        "  a later call replaces the list (`handin status` lists them);\n"
    )


# ---------------------------------------------------------------------------------------------- the methods wiki


#: Where each agent CLI looks for project skills, relative to its working directory (the workdir). Claude Code
#: reads `.claude/skills/<name>/SKILL.md`; Codex reads `.agents/skills/<name>/SKILL.md` from the repository it
#: runs in. The skill is written once and bound read-only at both places.
SKILL_DIRS = (".claude/skills", ".agents/skills")


class Wiki(Component):
    """The `quant-methods` skill: mechanism cards, read-only in the workdir.

    The pages ship with qda-agent (`qda_agent/wiki/skill/`). `capability.md` and `stacking.md` are generated
    at prepare from the installed `qda_quant` step registry (`qda_agent.wiki.pages`), so they name exactly
    the methods this library has.
    """

    name: Literal["wiki"] = "wiki"

    @model_validator(mode="before")
    @classmethod
    def _no_manual(cls, raw: object) -> object:
        """Refuse the retired `manual` setting with its reason, not a bare unknown-key error."""
        if isinstance(raw, dict) and "manual" in raw:
            raise ValueError(
                "`manual` was removed: qda-agent ships no experience manual; "
                "the quant-methods skill carries mechanism cards only"
            )
        return raw

    def section(self, run: Run) -> str:
        """Its line of TASK.md: the skill, its index and its page of methods this library implements."""
        how = "read-only" if run.walled else "your own copy"
        return (
            f"- `.claude/skills/quant-methods/` ({how}; also at `.agents/skills/quant-methods/`), a skill of\n"
            "  mechanism cards: start at its `SKILL.md`. The pages describe methods, not results;\n"
            "  `references/capability.md` says which of them this library implements;\n"
        )

    def install(self, run: Run) -> None:
        """Copy the skill and write its generated pages under each `SKILL_DIRS` entry of the overlay."""
        from qda_agent.wiki import install

        for d in SKILL_DIRS:
            top, _, rest = d.partition("/")
            dst = run.ro(top) / rest / "quant-methods"
            if dst.exists():
                shutil.rmtree(dst)
            install(dst)

    def protected(self, run: Run) -> list[str]:  # noqa: ARG002
        """`.claude` and `.agents`, whole: their every parent inside the workdir is then a mount point."""
        return [d.partition("/")[0] for d in SKILL_DIRS]


# ---------------------------------------------------------------------------------------------- extra qda verbs


class Fisher(Component):
    """`qda fisher map` (a GPU job) and `qda fisher allocate` (here): a per-unit sensitivity map and a greedy pick."""

    name: Literal["fisher"] = "fisher"
    subcommands: ClassVar[dict[str, tuple[str, ...]]] = {"qda": ("fisher",)}

    section_text: ClassVar[str] = (
        "- `qda fisher map --model DIR --calib FILE --menu F1,F2 [--samples N] [--seqlen L]\n"
        "  [--device cuda:0[,cuda:1,...]] --out MAP.json` gives, for every unit (the modules one engine\n"
        "  kernel serves together, per layer) and every menu format, its bits in the judge's counting and a\n"
        "  predicted KL increase (diagonal Fisher of the model's own outputs times the round-to-nearest\n"
        "  error); run it in a `qsched` job\n"
        "  (`python -m qda_quant.cli fisher map`). It ranks and does not calibrate: the prediction is exact\n"
        "  only for small deltas, and on a mixture-of-experts layer routing flips are discrete and second\n"
        "  order cannot see them, so read an MoE ranking as a rough ordering rather than a KL estimate.\n"
        f"  Cards: {_wrap(CARD_RULE)}\n"
        f"  Cache: {_wrap(CACHE_RULE)}\n"
        "  `qda fisher allocate MAP --body FORMAT --budget-bpw B` (here, no GPU) picks one format per unit by\n"
        "  predicted KL saved per bit inside the budget and prints a `formats:` block a build accepts. The\n"
        "  ranking can understate some modules: verify a chosen allocation with a build and `kl` against the\n"
        "  previous one;\n"
    )

    def section(self, run: Run) -> str:
        """Its line, naming only commands the run installs (`qda fisher` needs `qda`; the map needs a queue)."""
        if not _gives(run, "qda"):
            return ""
        if _gives(run, "qsched"):
            return self.section_text
        return self.section_text.replace(
            "run it in a `qsched` job\n  (`python -m qda_quant.cli fisher map`)",
            "run it in a job\n  (`python -m qda_quant.cli fisher map`)",
        )

    def offers(self, run: Run) -> bool:
        """It prices bits per quantized weight, the vLLM budget's rule."""
        return run.engine != "llama.cpp"


class Probe(Component):
    """`python -m qda_agent.probe` in a GPU job: an RTN build of a format scheme, read on the agent's own data."""

    name: Literal["probe"] = "probe"

    section_text: ClassVar[str] = (
        "- `python -m qda_agent.probe CONFIG --anchor ORIGINAL --data FILE.jsonl --work DIR [--sequences 32]` (in a\n"
        "  `qsched` job, one card, with the interpreter that has vLLM) builds CONFIG's `format`/`formats:` by round to\n"
        "  nearest, prices it in the judge's bits per weight and reads its KL against the original on `--sequences`\n"
        "  rows of your own FILE (`text`, `prompt` or `messages` rows), next to the same config with every unit in\n"
        "  the body format; one line out. The first probe of a model builds that reference once under DIR. It is\n"
        "  not `kl`: another reference, an RTN build, and a ranking signal only. One probe is one build and one\n"
        "  short reading, on one card;\n"
    )

    def section(self, run: Run) -> str:
        """Its line, naming `qsched` and `kl` only when the run gives them."""
        text = self.section_text
        if not _gives(run, "qsched"):
            text = text.replace(
                "(in a\n  `qsched` job, one card, with the interpreter that has vLLM)",
                "(one card, with the interpreter that has vLLM)",
            )
        if not _gives(run, "kl"):
            text = text.replace(
                "It is\n  not `kl`: another reference, an RTN build, and a ranking signal only.",
                "It is a\n  ranking signal only: another reference and an RTN build.",
            )
        return text

    def offers(self, run: Run) -> bool:
        """It prices bits per quantized weight, the vLLM budget's rule."""
        return run.engine != "llama.cpp"


# ---------------------------------------------------------------------------------------------- inputs of the jobs


class Data(Component):
    """Named read-only datasets for the agent's jobs, listed in a generated `DATA.md`.

    Attributes:
        sets: `{name: path}` of each dataset (a file or a directory); the jobs read it at that path.
    """

    name: Literal["data"] = "data"
    sets: dict[str, Path] = Field(default_factory=dict)

    @field_validator("sets")
    @classmethod
    def _names(cls, sets: dict[str, Path]) -> dict[str, Path]:
        """Refuse a dataset name that is not plain: it is printed in `DATA.md` and TASK.md."""
        bad = [k for k in sets if not re.fullmatch(r"[A-Za-z0-9][\w.-]*", k)]
        if bad:
            raise ValueError(f"data.sets: names {bad} are not plain names (letters, digits, _ . -)")
        return sets

    def section(self, run: Run) -> str:  # noqa: ARG002
        """The `DATA.md` line."""
        if not self.sets:
            return ""
        return "- `DATA.md` lists the datasets your GPU jobs can read (read-only), by name and path;\n"

    def install(self, run: Run) -> None:
        """Write `DATA.md`: one row per dataset with its path, kind and size."""
        if not self.sets:
            return
        rows = ["# Data", "", "Read-only datasets your GPU jobs can read at these paths.", ""]
        rows += ["| name | path | kind | size |", "|---|---|---|---|"]
        for name, path in sorted(self.sets.items()):
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            size = sum(p.stat().st_size for p in files)
            kind = "file" if path.is_file() else f"directory, {len(files)} files"
            rows.append(f"| {name} | `{path}` | {kind} | {_size(size)} |")
        rows += [
            "",
            "Rows are `messages` (a build renders them in the model's own chat template) or text already in it.",
            "Copy a file into your work dir, add rows of your own, and point `calib.file.path` at it.",
        ]
        out = run.ro("DATA.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rows) + "\n")

    def protected(self, run: Run) -> list[str]:  # noqa: ARG002
        """`DATA.md`."""
        return ["DATA.md"] if self.sets else []

    def readable(self) -> list[Path]:
        """Every dataset."""
        return list(self.sets.values())


def _size(n: int) -> str:
    """`n` bytes in the largest binary unit below 1024 (`3.2 GiB`)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


class Serve(Component):
    """A vLLM virtualenv on the jobs' `PATH`, so the agent can serve and sample its own checkpoints.

    Attributes:
        python: The vLLM virtualenv's interpreter; the venv is bound read-only into every job and its `bin/`
            comes after the toolset's on `PATH` (`vllm` and the venv's other programs). Unset: the judge's
            `serve_python` (`RunConfig`), else nothing is added.
    """

    name: Literal["serve"] = "serve"
    python: Path | None = None

    def section(self, run: Run) -> str:  # noqa: ARG002
        """The serving line (none without `python`).

        The interpreter is named by the job environment's variable, not by its host path: TASK.md carries no
        machine path, and the manifest records the value as part of what the run gives the agent.
        """
        if self.python is None:
            return ""
        return (
            "- your GPU jobs can serve a checkpoint with vLLM: `vllm` is on their `PATH`, and\n"
            "  `$QDA_VLLM_PYTHON` is the interpreter with vLLM installed (use it for your own long "
            "generations);\n"
        )

    def env(self, run: Run) -> dict[str, str]:  # noqa: ARG002
        """`QDA_VLLM_PYTHON` (the interpreter TASK.md names, without printing its host path) and the engine env.

        `qda_kl.serve.NO_JIT_ENV` is read here, not restated: the judge's own engines already start with it, and
        the agent's vLLM jobs start from the job environment. Without it a first run JIT-builds FlashInfer's
        sampler (and all-reduce) inside the job, which flock the shared workspace and failed on a 27B model.
        The compile caches are the judge's engines' too (`qda_quant.output.vllm_env.cache_env`): the directories
        the user named, else the node-local ones under `/tmp`, never the `$HOME` defaults, where two engines
        building a kernel at once fail on its flock (EIO). A sandboxed job gets its own tmpfs at `/tmp`
        (`isolation.bwrap`), so its caches live and die with the job. Machete is off in the agent's engines as in
        every other (`qda_quant.output.vllm_env.disabled_kernels`).
        """
        engine = cache_env(disabled_kernels(NO_JIT_ENV))
        return {**engine, "QDA_VLLM_PYTHON": str(self.python)} if self.python is not None else engine

    def readable(self) -> list[Path]:
        """The virtualenv (the interpreter's `bin/`'s parent, the link kept as given), when the interpreter is there.

        A missing interpreter binds nothing: binding its directory instead would hand the jobs whatever holds it.
        """
        if self.python is None or not self.python.exists():
            return []
        return [self.python.parent.parent]

    def path(self) -> list[Path]:
        """The virtualenv's `bin/`."""
        return [self.python.parent] if self.python is not None else []

    def job_python(self) -> Path | None:
        """The vLLM interpreter it binds for the jobs (None without `python`)."""
        return self.python

    def default_python(self, python: Path | None) -> None:
        """Take `python` as its interpreter when it has none."""
        if self.python is None and python is not None:
            self.python = python


#: Top-level files of a checkpoint that hold no weights (config, tokenizer, index), as the agent's shell sees
#: them in `model/` and `baselines/<N>/`.
VIEW = (
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
)


#: Suffixes of weight files, which a view never copies (even through a link with a `VIEW` name).
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _view(src: Path, dst: Path) -> None:
    """Copy the `VIEW` files of checkpoint `src` into `dst`, and make the copy read-only.

    A link is copied as its target's content only when it stays in the checkpoint or leads to a hub snapshot's
    blob (a Hugging Face cache snapshot holds only links into `blobs/`); any other link is skipped with a log
    line, so a planted `config.json -> ~/secret` never reaches the agent. A link to a weights file is skipped too.

    The copy is a host-owned input, not disk the agent may reclaim: its files go 0444 and its directories
    0555 (`workspace.read_only`), so deleting or editing them with an ordinary command fails.
    """
    from qda_quant.intake.fingerprint import link_inside

    from qda_agent.workspace import read_only

    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if not any(fnmatch.fnmatchcase(f.name, g) for g in VIEW):
            continue
        if f.is_symlink() and (not link_inside(src, f) or Path(os.path.realpath(f)).suffix in WEIGHTS):
            _log.warning("views: skipped %s, a link to %s", f, os.path.realpath(f))
            continue
        if f.is_file():
            shutil.copyfile(f, dst / f.name)
    read_only(dst)


class Baselines(Component):
    """The baselines under neutral names: each one at `baselines/<N>` in the jobs (no method or model name).

    TASK.md names them `baselines/0`, `baselines/1`, ... in fallback order; the jobs see the whole checkpoint
    there, bound read-only over the workdir. With `views`, the agent's shell sees their non-weight files there.
    """

    name: Literal["baselines"] = "baselines"

    def section(self, run: Run) -> str:
        """The baselines line."""
        if not run.baselines:
            return ""
        return "- the baselines are at `baselines/0`, `baselines/1`, ... in fallback order (read-only in your jobs);\n"

    def install(self, run: Run) -> None:
        """Make the mount points `baselines/<N>` in the overlay."""
        for n in range(len(run.baselines)):
            (run.ro("baselines") / str(n)).mkdir(parents=True, exist_ok=True)

    def protected(self, run: Run) -> list[str]:
        """`baselines` (its `<N>` mount points are the host's)."""
        return ["baselines"] if run.baselines else []

    def binds(self, run: Run) -> dict[Path, Path]:
        """`baselines/<N>` -> the N-th baseline."""
        return {run.workdir / "baselines" / str(n): b for n, b in enumerate(run.baselines)}

    def baseline_names(self, run: Run) -> list[str] | None:
        """`baselines/<N>`."""
        return [f"baselines/{n}" for n in range(len(run.baselines))]


class Views(Component):
    """The original model's and the baselines' non-weight files (config, tokenizer, index), readable by the shell.

    Copied to `model/` and `baselines/<N>/` in the workdir and sealed read-only (files 0444, directories
    0555), so the agent can read a config or a tokenizer without a GPU job and cannot reclaim them as disk.
    The weights stay out of the shell's reach.
    """

    name: Literal["views"] = "views"

    def section(self, run: Run) -> str:
        """The views line (none when there is nothing to show)."""
        if not Path(run.model).is_dir() and not any(b.is_dir() for b in run.baselines):
            return ""
        return (
            "- `model/` and `baselines/<N>/` hold the original model's and the baselines' config, tokenizer and\n"
            "  index files (not their weights): read-only inputs, not disk you can reclaim;\n"
        )

    def install(self, run: Run) -> None:
        """Copy the views."""
        if Path(run.model).is_dir():
            _view(Path(run.model), run.ro("model"))
        for n, b in enumerate(run.baselines):
            if b.is_dir():
                _view(b, run.ro("baselines") / str(n))

    def protected(self, run: Run) -> list[str]:
        """`model` and `baselines`, when there is something to show."""
        return [*(["model"] if Path(run.model).is_dir() else []), *(["baselines"] if run.baselines else [])]


class Source(Component):
    """A read-write copy of the qda library in the workdir, which the agent may read and patch.

    The agent's jobs (its `qda build`s and `qsched` scripts) import it first (`PYTHONPATH`), so a change the
    agent makes runs in its own builds. The judge never imports it: KL readings and hand-in checks run with
    the judge's own evaluator and interpreter.

    Attributes:
        package: The `qda_quant` package directory to copy (default: the one installed with qda-agent).
    """

    name: Literal["source"] = "source"
    package: Path | None = None

    def _src(self) -> Path:
        """The package directory to copy."""
        if self.package is not None:
            return self.package
        import qda_quant

        return Path(qda_quant.__file__).parent

    def readable(self) -> list[Path]:
        """The package it copies (read on the host only; listed so the config can check where it is)."""
        return [self._src()] if self.package is not None else []

    section_text: ClassVar[str] = (
        "- `src/qda_quant/` is a copy of the library: read it, and change it if you need to; your jobs\n"
        "  import this copy, and so does a build unless the build-receipt policy is enabled;\n"
    )

    def install(self, run: Run) -> None:
        """Copy the package (no byte code); keep a real copy already there, refuse a link in its place."""
        dst = run.workdir / "src" / "qda_quant"
        for p in (dst.parent, dst):
            if p.is_symlink() or (os.path.lexists(p) and not p.is_dir()):
                raise ValueError(f"{p} is a link or a file, not the source copy's directory")
        if not os.path.lexists(dst):
            shutil.copytree(self._src(), dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def env(self, run: Run) -> dict[str, str]:
        """`PYTHONPATH` to the copy."""
        return {"PYTHONPATH": str(run.workdir / "src")}
