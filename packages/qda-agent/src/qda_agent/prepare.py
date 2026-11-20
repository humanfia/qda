"""`qda-agent prepare`: every input of one search, made in one command, and the run config that names them.

    qda-agent prepare --model Qwen/Qwen3-8B --out work/q8b [--format w4a16] [--cards 0,1] [--dry-run]

Each step writes into `--out` (DIR) and is skipped when its output is already there and matches what this
run would make, so an interrupted prepare is run again with the same command:

| step | what it runs | writes |
|---|---|---|
| model | `hf download REPO --revision SHA` (a Hub id; a local directory is used as is) | DIR/model |
| corpus | `qda-kl corpus` (no guard: the manifest says `guard: none`) | DIR/data/corpus{,.eval,.holdout}.jsonl |
| ref | `qda-kl ref --cache` on the 96 `search96` prompts and, in the same engine load (`--group`), the holdout split `evaluate` reads; `--tp` 1 when the model's weights fit one card (`REF_CARD_FILL`), else the most of the cards vLLM splits the model over (each built once per model) | DIR/ref (a link into `$QDA_CACHE/kl-ref`); the holdout reference in the same cache |
| calib | `qda calib render` for candidate 0, and the public pools' `messages` rows as the search's default (`--calib FILE`: that file for both, as is) | DIR/data/calib.jsonl, DIR/data/agent-calib.jsonl |
| imatrix | llama.cpp only: `llama-imatrix` over that calibration, once per model and llama.cpp commit | DIR/imatrix/model.imatrix.gguf |
| sensitivity | `--sensitivity` only (llama.cpp): one `qda sensitivity` sweep of the classes at their types, allocated into a type map candidate 0 reads | DIR/sensitivity/{rows,allocation,type_map}.json |
| cand0 | `qda build` of candidate 0 (`qda_agent.cand0`) | DIR/cand0-METHOD |
| floor | llama.cpp only: a bf16 GGUF scored by llama.cpp against the vLLM reference | DIR/floor |
| config | the run config, with every derived field filled once, the budget (`--budget-bpw`) included | DIR/search.yaml |

Before the first step it refuses when a cache or temporary disk is short (`room.cache_room`, the lines
`qda doctor` prints), naming the variable that points there; the run directory's room is `require_room`'s.

`prepare.json` in DIR records what each step resolved (the model's revision, the sampling the reference used)
and how long it took. Every command is a public CLI run by subprocess (`qda`, `qda-kl`, `hf`) with this
interpreter. The search's defaults are the product's: isolation and contamination `none`.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION
from qda_quant.calib.render import POOL, parse_names
from qda_quant.exit import ENVIRONMENT, USAGE, refuse
from qda_quant.format import Format
from qda_quant.gguf.allocate import one_allocation
from qda_quant.step import Refused

from qda_agent import budget_source, cand0, configfile, group_size, summary
from qda_agent.host import serve_line
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.custody import free_floor, space_problem
from qda_agent.room import Place, cache_room, full_line, room_notes
from qda_agent.run_config import PREPARE_MARK
from qda_agent.toolsets import Qda

if TYPE_CHECKING:
    from qda_agent.agents import Backend

__all__ = ["Prepare", "Refusal", "Step", "calib_thinking_note", "main", "plan", "search_config"]

#: The run's format -> candidate 0's build method.
FORMATS = {"w4a16": "gptq", "w4a8": "gptq", "w3a16": "gptq", "w4a4": "gptq", "gguf-q4km": cand0.GGUF}


def _requested_format(raw: str) -> tuple[str, Format, bool]:
    """`--format` as (the preset its build method is keyed by, the `Format` asked for, group named explicitly).

    A bare preset (`w4a16`) names no group size, so prepare may lower its 128 to one that stores the model; a
    spelling that names one (`w4a16-g64`, `w4a16-pc`) is taken as given, and refused when it does not work.
    """
    if raw == "gguf-q4km":
        return raw, Format.parse("w4a16"), False
    try:
        fmt = Format.parse(raw)
    except Refused as e:  # an unknown spelling
        raise Refusal(f"--format {raw!r}: {e}") from None
    preset = f"w{fmt.weight_bits}a{fmt.act_bits}"  # NVFP4's activations are 4-bit, so `w4a4`, not `w4a8`
    if preset not in FORMATS:
        raise Refusal(f"--format {raw!r}: one of {', '.join(FORMATS)}")
    return preset, fmt, "-g" in raw.lower() or raw.lower().endswith("-pc")


STEPS = ("model", "corpus", "ref", "calib", "imatrix", "sensitivity", "cand0", "floor", "config")
#: The public pool mix: the `messages` pools only. `d1.jsonl` is left out although it is the largest pool: it is
#: text pre-rendered in one model's chat template, published so our own builds of that model can be reproduced,
#: and it is the one pool not filtered for benchmark overlap (its dataset card counts 48 MMLU-Pro test items).
#: Reproducing a released build reads that build's own rendered calibration, not this mix. `reasoning.jsonl`
#: holds prompts without answers and is left out too (`qda calib render` records that).
POOL_NAMES = "chat=3,multilingual=2"
#: Candidate 0's calibration: samples of seqlen tokens (qda-quant's file-source defaults, named here once), which
#: the calib step renders enough rows for and the cand0 recipe reads.
CALIB_SAMPLES, CALIB_SEQLEN = 128, 2048
#: The pools the agent's jobs get as data sets (`DATA.md`): exactly the pools `POOL_NAMES` mixes, derived here
#: so the mix names its pools once. Each is a `messages` file the agent renders itself; `d1` (pre-rendered text,
#: not a data set) and `reasoning` (no answers) are not in the mix.
AGENT_POOLS = tuple(parse_names(POOL_NAMES))
#: The reference's named subset a search reads every candidate on; the sensitivity reads score it too.
SEARCH_SUBSET = "search96"
#: A measured KL increment below this counts as zero in the sensitivity allocation, fixed at the baseline's floor.
SENSITIVITY_FLOOR = 0.0
#: The share of a read's cards the bf16 template may fill; the rest is the server's cache and buffers.
SENSITIVITY_CARD_FILL = 0.75
#: The share of one card (`ref_card_bytes`) the model's weights may fill for the reference to read on
#: that card alone, without tensor parallelism; the rest is vLLM's KV cache for the reference's generations.
#: Qwen3-8B (16.4 GB) reads on one 32 GB card or larger; a 27B (55.6 GB) on one H200 but on two H100 80GB.
REF_CARD_FILL = 0.6
#: Candidate 0's bytes over the bf16 model's, for the disk estimate (Qwen3-8B: 5.03 GB Q4_K_M of 16.4 GB).
#: A W4A4 candidate stores about 4.25 bits a weight plus a 16-bit head: the released Qwen3.8-27B W4A4 is
#: 21.9 GB of its 55.6 GB bf16 (0.394), rounded up.
SIZE_RATIO = {"w4a16": 0.35, "w4a8": 0.35, "w3a16": 0.30, "w4a4": 0.40, "gguf-q4km": 0.31}
#: A search's run directory in candidate 0 sizes: the agent's own builds and the judge's frozen copies of what it
#: read and handed in. It has no cap (builds stop only at the free-space floor); measured 4.1x (0.25 GPU-h,
#: Qwen3-0.6B), 5.2x (1 GPU-h, Qwen3.5-4B) and 6.3x (4 GPU-h, Qwen3-Coder-30B-A3B), so this is a typical size.
RUN_DIR_FACTOR = 6.5

#: The build docs' commands for the programs a GGUF search runs (docs/formats.md, GGUF for llama.cpp);
#: CMAKE_CUDA_ARCHITECTURES is the run's cards' (`group_size.cuda_architectures`: 90 on an H100, 120 on a
#: PRO 6000), `native` where they are not known.
LLAMA_BUILD = """\
cmake -S "{llama}" -B "{build}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="{arch}" -DBUILD_SHARED_LIBS=ON \\
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "{build}" --target llama-server llama-imatrix -j8
c++ -std=c++17 -O2 -shared -fPIC "{probe}" \\
  -I "{llama}/include" -I "{llama}/src" -I "{llama}/ggml/include" \\
  -L "{build}/bin" -Wl,-rpath,"{build}/bin" -lllama -lggml -lggml-base -ldl -o "{build}/probe.so\""""


def probe_source() -> Path:
    """The `probe.cpp` this installation ships in `qda_kl.gguf`."""
    from importlib.resources import files

    return Path(str(files("qda_kl.gguf") / "probe.cpp"))


def llama_build(llama: Path, build: Path, cards: Sequence[int] | None = None) -> str:
    """`LLAMA_BUILD` for this checkout and the run's `cards` (their CUDA architecture), with ``CUDACXX=`` on the configure line when CMake would not find nvcc.

    CMake looks for ``nvcc`` on ``PATH``; a toolkit installed off it (``CUDA_HOME``, ``CUDA_PATH`` or
    ``/usr/local/cuda``, found by qda-quant's one nvcc lookup) is named to it through ``CUDACXX``. With no nvcc
    anywhere, or ``CUDACXX`` already set, the commands are unchanged and CMake's own error names the compiler.
    """
    from qda_quant.output.vllm_env import _nvcc

    arch = group_size.cuda_architectures(cards)
    commands = LLAMA_BUILD.format(llama=llama, build=build, probe=probe_source(), arch=arch)
    if os.environ.get("CUDACXX") or shutil.which("nvcc"):
        return commands
    nvcc = _nvcc(os.environ)
    return f'CUDACXX="{nvcc}" {commands}' if nvcc else commands


class Refusal(Exception):
    """A step cannot run; the message says what to do.

    Attributes:
        code: The status the command exits with (`qda_quant.exit`): ``ENVIRONMENT`` where the step needs
            something this machine does not have -- a card the process cannot see, a program that is not
            installed, a llama.cpp checkout that is missing or unbuilt -- and the default ``USAGE`` where the
            step list, a path or an argument has to change.
    """

    def __init__(self, message: str, *, code: int = USAGE, fix: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.fix = fix


def sha256(path: Path) -> str:
    """The file's sha256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        got = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return got if isinstance(got, dict) else {}


def calib_thinking_note(calib: str | Path | None) -> str:
    """One line for a default calibration that holds no thinking text for a thinking model.

    The facts are the calibration's own metadata, `qda calib render`'s `.meta.json` beside it, never a second
    parse of the data: `thinking.template.think_markers` (or `reasoning`) says the model's chat template
    writes think blocks, and `thinking`'s `turns with reasoning...` counts (present only when non-zero) say
    whether any row carried reasoning. A calibration without metadata, one that does carry reasoning, or a
    template without a thinking format says nothing.

    Args:
        calib: The default calibration file the run names, or None.

    Returns:
        The sentence to print after "the default calibration file is ...", or "" when there is nothing to say.
    """
    if calib is None:
        return ""
    meta = _json(Path(calib).with_name(Path(calib).name + ".meta.json"))
    thinking = meta.get("thinking") or {}
    template = thinking.get("template") or {}
    if not (template.get("think_markers") or template.get("reasoning")):
        return ""
    if any(str(key).startswith("turns with reasoning") for key in thinking):
        return ""
    return (
        "the default calibration holds no thinking text, and this model's chat template has a thinking "
        "format: for a thinking model, generate its own answers and traces with "
        "`qda traces <model> <prompts.jsonl> <prefix> --max-tokens <tokens> --seed <seed>` and use them as "
        "the calibration or `scale_kd` data"
    )


def _backend_setting(value: str | dict[str, Any]) -> str | dict[str, Any]:
    """`--backend` as `agent.backend` holds it: a registry name, or the YAML/JSON mapping a config would hold."""
    if not isinstance(value, str) or not value.lstrip().startswith("{"):
        return value
    try:
        got = yaml.safe_load(value)
    except yaml.YAMLError as e:
        raise Refusal(f"--backend {value!r}: not YAML ({e})") from None
    if not isinstance(got, dict) or not isinstance(got.get("name"), str):
        raise Refusal(f"--backend {value!r}: expected a backend name, or a mapping with `name: KEY`")
    return got


@dataclass
class Prepare:
    """One prepare: what was asked, and where each output goes."""

    model: str
    out: Path
    format: str = "w4a16"
    cards: list[int] = field(default_factory=lambda: [0])
    budget_gpu_hours: float = 4.0
    #: `--budget-bpw` as given (`budget_source.parse`); None: the row's default (`budget_source.DEFAULT_BPW`), or
    #: candidate 0's size for a llama.cpp search.
    budget_bpw: str | None = None
    #: `--head-lease` (`judge.head_lease`); None: on for a W4A4 row, off otherwise.
    head_lease: bool | None = None
    #: Measure the class sensitivity sweep and give candidate 0 its type map (GGUF only; off by default).
    sensitivity: bool = False
    #: `--calib`: candidate 0's and the search's calibration file, used as is; None: the calib step renders one.
    calib_file: Path | None = None
    #: A backend name (`qda_agent.backends`), or the mapping a config would hold (`{name: codex, model: ...}`).
    backend: str | dict[str, Any] = "claude"
    #: The interpreters that build (torch + qda-quant) and read (vLLM): this one; a test passes a stand-in.
    python: Path = Path(sys.executable)
    serve_python: Path = Path(sys.executable)
    sign_in_dir: Path | None = None
    #: The backend the registry resolved from `backend` and `sign_in_dir`, with its own defaults and validators.
    agent: Backend = field(init=False)

    def __post_init__(self) -> None:
        from qda_agent.agents import BACKENDS
        from qda_agent.isolation.devices import invisible_cards
        from qda_agent.settings import pick

        if why := invisible_cards(self.cards):  # each step names its cards to CUDA as physical numbers
            raise Refusal(why, code=ENVIRONMENT)  # name the cards you hold, or unset CUDA_VISIBLE_DEVICES
        self.out = Path(self.out).expanduser().resolve()
        self.format, self.given, self.explicit_group = _requested_format(self.format)
        self.fmt = self.given
        #: What `resolve_format` chose, as `prepare.json` records it (`format_choice`); None until it decides.
        self.format_choice: dict[str, Any] | None = None
        self._check_budget()
        if self.sensitivity and self.format != "gguf-q4km":
            raise Refusal("--sensitivity is a GGUF option: it allocates a .gguf's per-tensor types")
        if self.calib_file is not None:
            self.calib_file = Path(self.calib_file).expanduser().resolve()
            if not self.calib_file.is_file():
                raise Refusal(f"--calib {self.calib_file}: not a file")
        self.backend = _backend_setting(self.backend)
        if self.sign_in_dir is not None:
            self.sign_in_dir = Path(self.sign_in_dir).expanduser().resolve()
            if not self.sign_in_dir.is_dir():
                raise Refusal(f"--sign-in-dir {self.sign_in_dir} is not a directory")
        try:  # the registry's defaults and validators: a backend's model, a dsh run's endpoint, its sign-in
            self.agent = pick(BACKENDS, self._backend_value())
        except ValueError as e:
            raise Refusal(f"--backend {self.backend!r}: {e}") from None
        local = Path(self.model).expanduser()
        self.local = local.resolve() if local.exists() else None
        self.repo, _, self.revision = self.model.partition("@")
        self.state_path = self.out / "prepare.json"
        self.state: dict[str, Any] = _json(self.state_path)
        self._read_hub()
        self.resolve_format()

    def _check_budget(self) -> None:
        """Read `--budget-bpw` and `--head-lease` against the row, before any step runs.

        A `hf:` checkpoint is read here (headers and sizes only), so a GGUF price on a vLLM search, or a
        safetensors one on a llama.cpp search, is refused before the first GPU hour.

        Raises:
            Refusal: a malformed `--budget-bpw`, a number of bits on a llama.cpp search (its budget is
                bytes), a Hub checkpoint of the other engine or one the rule cannot price, or `--head-lease`
                on a llama.cpp search (its byte budget already holds the output tensors).
        """
        if self.budget_bpw is not None:
            try:
                self.budget = budget_source.parse(self.budget_bpw)
            except ValueError as e:
                raise Refusal(str(e)) from None
        elif self.gguf:
            self.budget = budget_source.BudgetSpec("candidate0")
        else:
            self.budget = budget_source.BudgetSpec("number", bpw=budget_source.DEFAULT_BPW[self.format])
        if self.head_lease is None:
            self.head_lease = self.format == "w4a4"
        if self.gguf and self.head_lease:
            raise Refusal(
                "--head-lease: a llama.cpp search's byte budget already holds the output and token_embd "
                "tensors; drop --head-lease"
            )
        if self.gguf and self.budget.kind == "number":
            raise Refusal(
                f"--budget-bpw {self.budget}: a llama.cpp search states its budget in bytes; give candidate0 "
                "or hf:REPO:FILE.gguf"
            )
        #: The Hub checkpoint's price (`hf:`), read once (`_read_hub`).
        self.hub: budget_source.HubPrice | None = None

    def _read_hub(self) -> None:
        """Price an `hf:` budget: from prepare.json when this prepare already read it, else from the Hub.

        The first run that executes records the repository, the commit it read and the price
        (`budget_source.HubPrice.record`, written by `execute`), so running the same prepare again prices the
        same commit and does not need the Hub.

        Raises:
            Refusal: the Hub cannot be read or prices the checkpoint by no rule, or the checkpoint is of the
                other engine.
        """
        if self.budget.kind != "hf":
            return
        saved = self.state.get("budget_source")
        if isinstance(saved, dict) and saved.get("spec") == str(self.budget):
            self.hub = budget_source.HubPrice.from_record(saved)
        else:
            try:
                self.hub = budget_source.hub_price(self.budget)
            except budget_source.BudgetRefused as e:
                raise Refusal(str(e), fix=e.fix) from None
            except ValueError as e:
                raise Refusal(str(e)) from None
        if (self.hub.kind == "gguf") != self.gguf:
            raise Refusal(
                f"--budget-bpw {self.budget}: a {self.hub.kind} checkpoint prices "
                f"{'a llama.cpp' if self.hub.kind == 'gguf' else 'a vLLM'} search, and this one is "
                f"{'llama.cpp' if self.gguf else 'vLLM'} (--format {self.format})"
            )

    def budget_value(self) -> float | int:
        """The search's budget: bits per quantized weight on vLLM, bytes on llama.cpp (`judge.budget_bytes`).

        A number is itself; `candidate0` and a safetensors `hf:` checkpoint are priced by the judge's own rule
        against this model (`judge.checks.budget`), so they need the model and, for candidate 0, its build.
        With `--sensitivity` and the `candidate0` budget, the number is the Q4_K_M rule's file size the sweep
        allocated under, which the sensitivity step recorded.

        Raises:
            Refusal: candidate 0 is not built, or the Hub checkpoint is not a quantization of this model (its
                decoder Linears are not this model's).
        """
        from qda_agent.judge.checks import CheckpointError
        from qda_agent.judge.checks.budget import bpw, denominator, linear_names

        if self.budget.kind == "number":
            return float(cast(float, self.budget.bpw))
        if self.hub is not None and self.hub.size is not None:
            return self.hub.size
        if self.budget.kind == "candidate0" and self.gguf:
            if not self.sensitivity:
                return gguf_file(self.cand0).stat().st_size
            # the sweep allocated inside the Q4_K_M rule's file, not candidate 0's smaller file: one number
            # for both, recorded by the sensitivity step
            if (budget := self.sensitivity_budget()) is None:
                raise Refusal(
                    "--sensitivity: the search's byte budget is the Q4_K_M rule's file size the sweep "
                    "allocated under; run the sensitivity step before the config step"
                )
            return budget
        try:
            if self.hub is None:
                return round(bpw(self.cand0, self.model_dir), 4)
            if set(self.hub.linears) != linear_names(self.model_dir):
                raise Refusal(
                    f"--budget-bpw {self.budget}: its decoder Linears are not {self.model_dir.name}'s; name a "
                    "quantization of this model"
                )
            return round(cast(int, self.hub.bits) / denominator(self.model_dir), 4)
        except CheckpointError as e:
            raise Refusal(f"--budget-bpw {self.budget}: {e}") from None

    def resolve_format(self, *, save: bool = False) -> dict[str, Any] | None:
        """Store the format candidate 0 is built in, from the model's own widths, before any step runs.

        The widths come from `config.json` and the safetensors headers of the local model, or, before the
        download, from the Hub's safetensors metadata (range reads). One decision: `self.fmt` is what candidate
        0's recipe and `search.yaml` both carry. The default group size is lowered only across 128 > 64 > 32;
        per-channel is a different preset and is refused unless `--format` named it.
        """
        if self.gguf:
            return None
        modules = None
        source = ""
        try:
            if (self.model_dir / "config.json").is_file():
                modules = group_size.local_widths(self.model_dir)
                source = f"{self.model_dir.name}: config.json + safetensors headers"
            elif self.local is None:
                modules = group_size.hub_widths(self.repo, self.revision or None)
                source = f"{self.repo}@{self.revision or 'main'}: Hub safetensors metadata"
        # offline or not a Hub repo (`IntakeRefused`, a ValueError), no config.json or safetensors, a Hub HTTP
        # error (`HfHubHTTPError`, an OSError): the model step decides instead
        except (OSError, ValueError) as e:
            print(
                f"qda-agent prepare: warning: group size is decided after the model step ({e})",
                file=sys.stderr,
            )
            return None
        if not modules:
            return None
        try:
            decision = group_size.auto_format(
                self.given, modules, group_size.target_for(self.cards), explicit=self.explicit_group
            )
        except Refused as e:
            raise Refusal(str(e), code=e.code) from None
        shown = self.format_choice  # a real run resolves twice (Hub metadata, then the headers); print once
        self.fmt = decision.fmt
        trigger = decision.trigger
        self.format_choice = {
            "format": self.fmt.name,
            "group_size": self.fmt.group_size,
            "explicit": self.explicit_group,
            "trigger": (
                None
                if trigger is None
                else {"module": trigger.name, "width": trigger.cols, "reason": trigger.reason}
            ),
            "feasible_group_sizes": list(decision.feasible),
            "source": source,
        }
        if trigger is not None and (
            shown is None
            or shown.get("trigger") != self.format_choice["trigger"]
            or shown.get("format") != self.format_choice["format"]
        ):
            print(
                f"qda-agent prepare: {self.format}: group size {self.fmt.group_size} ({trigger.name}: {trigger.cols} {trigger.reason})"
            )
        if save:
            self.save(format_choice=self.format_choice)
        return self.format_choice

    def _backend_value(self) -> str | dict[str, Any]:
        """The `agent.backend` value the registry resolves: the name or mapping, with `--sign-in-dir` when given."""
        if self.sign_in_dir is None:
            return self.backend
        given = {"name": self.backend} if isinstance(self.backend, str) else dict(self.backend)
        return {**given, "sign_in_dir": str(self.sign_in_dir)}

    @property
    def gguf(self) -> bool:
        """Whether this search is a llama.cpp one."""
        return self.format == "gguf-q4km"

    def source_identity(self) -> str | None:
        """The original this prepare reads, to tell a reused output from a stale one.

        A Hub model is its resolved revision (recorded by the model step); a local directory is its own digest
        (`qda_agent.workspace.tree_sha256`: content for small files, size and mtime for the weights), so
        replaced weights change it. None while a Hub model's revision is not recorded yet.

        The reference and candidate 0 record the identity they were built from, so a later prepare into the
        same DIR with another model refuses to reuse them instead of mixing two models.
        """
        if self.local is not None:
            from qda_agent.workspace import tree_sha256

            digest, _, _, method = tree_sha256(self.local)
            return f"local {self.local} {method} {digest}"
        revision = (self.state.get("model") or {}).get("revision")
        return f"{self.repo}@{revision}" if revision else None

    def made_from(self) -> dict[str, str | None]:
        """What a record beside an artefact this prepare made keeps of the model (the importance matrix's).

        ``source`` and ``source_sha256`` name the model as the template's record and the judge do
        (`workspace.model_source`; the judge compares them with the run's model directory), and
        ``source_identity`` is this prepare's own reuse key (`source_identity`: a Hub model's ``repo@revision``).
        """
        from qda_agent.workspace import model_source

        return {**model_source(self.model_dir), "source_identity": self.source_identity()}

    @property
    def model_dir(self) -> Path:
        """The original model's directory: the local one, or the download."""
        return self.local or self.out / "model"

    corpus = property(lambda self: self.out / "data" / "corpus.jsonl")
    eval_split = property(lambda self: self.out / "data" / "corpus.eval.jsonl")
    search = property(lambda self: self.out / "data" / "corpus.search96.jsonl")
    search_ids = property(lambda self: self.out / "data" / "corpus.search96.ids")
    holdout = property(lambda self: self.out / "data" / "corpus.holdout.jsonl")
    ref = property(lambda self: self.out / "ref")
    calib = property(lambda self: self.out / "data" / "calib.jsonl")
    #: The search's default calibration: the agent's `messages` pools, row for row, so a build may name any
    #: `samples` the pools supply. Candidate 0's recipe reads :attr:`calib`, the least rows its 128 samples pack.
    agent_calib = property(lambda self: self.out / "data" / "agent-calib.jsonl")
    imatrix = property(lambda self: self.out / "imatrix" / "model.imatrix.gguf")
    sensitivity_rows = property(lambda self: self.out / "sensitivity" / "rows.json")
    sensitivity_record = property(lambda self: self.out / "sensitivity" / "measurement.json")
    allocation = property(lambda self: self.out / "sensitivity" / "allocation.json")
    type_map = property(lambda self: self.out / "sensitivity" / "type_map.json")
    pools = property(lambda self: self.out / "data" / "pools")
    cand0 = property(lambda self: self.out / f"cand0-{FORMATS[self.format]}")
    bf16 = property(lambda self: self.out / "bf16" / "model-bf16.gguf")
    floor = property(lambda self: self.out / "floor")
    config = property(lambda self: self.out / "search.yaml")

    def env(self) -> dict[str, str]:
        """The environment of a GPU step: exactly the prepare's cards."""
        return {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.cards))}

    def save(self, **record: Any) -> None:
        """Record what a step resolved in prepare.json."""
        self.state.update(record)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=1, sort_keys=True) + "\n")

    def qda(self) -> list[str]:
        return [str(self.python), "-m", "qda_quant.cli"]

    def qda_kl(self) -> list[str]:
        return [str(self.serve_python), "-m", "qda_kl.cli"]

    def recipe(self, *, template: bool = True, imatrix: bool = True) -> dict[str, Any]:
        """Candidate 0's recipe: the calibration it reads is what the calib step renders rows for.

        A llama.cpp search's candidate 0 reads the bf16 GGUF (`bf16`) as its template, the file the engine floor
        scores and every search build reads, and the imatrix step's file as `exporter.gguf.imatrix`. With
        `--sensitivity`, its `exporter.gguf.tensor_types` are the sensitivity step's allocation and
        `exporter.gguf.libggml` is the checkout's built library those rules' Q5_K cells need.
        `template=False` and `imatrix=False` are the recipe without those, which writes the same file (an
        earlier prepare's recipe).
        """
        recipe = cand0.candidate0_recipe(
            self.build_inputs(),
            method=FORMATS[self.format],
            out=self.cand0,
            calib_samples=CALIB_SAMPLES,
            calib_seqlen=CALIB_SEQLEN,
            template=self.bf16 if self.gguf and template else None,
            imatrix=self.imatrix if self.gguf and imatrix else None,
            tensor_types=self.tensor_types(),
            libggml=self.libggml(),
        )
        # the download's repo, so a checkpoint built from `<out>/model` is served under the model's name
        repo = (self.state.get("model") or {}).get("repo")
        if isinstance(repo, str) and repo:
            recipe["meta"] = {**recipe.get("meta", {}), "source_repo": repo}
        return recipe

    def libggml(self) -> Path | None:
        """The checkout's built `libggml-base`, which a measured type map's Q5_K needs; None without one."""
        return _libggml(llama_programs(self.cards)) if self.tensor_types() else None

    def tensor_types(self) -> list[dict[str, str]] | None:
        """The type-map rules the sensitivity step allocated, or None (the step off, or not run yet)."""
        if not (self.gguf and self.sensitivity) or not self.type_map.is_file():
            return None
        try:
            got = json.loads(self.type_map.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return got if isinstance(got, list) and got else None

    def sensitivity_tp(self) -> int:
        """Cards per sensitivity read: the fewest that hold the bf16 template with room for the server's cache.

        The reads are independent and run one per group of this many cards over `--cards`, so a model that fits
        one card is read on every card at once. Without CUDA's card sizes, or before the template is built (a
        dry run, or a plan printed before the floor step), one read takes every card. The card size is the
        smallest of the physical `--cards` (`ref_card_bytes`), not of the first few cards CUDA numbers.
        """
        if not self.bf16.exists():
            return len(self.cards)
        per = ref_card_bytes(self.cards)
        if per is None:  # the driver will not say: the widest read always fits
            return len(self.cards)
        need = self.bf16.stat().st_size / SENSITIVITY_CARD_FILL
        return next((tp for tp in range(1, len(self.cards) + 1) if tp * per >= need), len(self.cards))

    def sensitivity_budget(self) -> int | None:
        """The bytes the sensitivity sweep allocated inside, when it is known without measuring.

        An `hf:` GGUF's file size is priced at construction; the Q4_K_M rule's own file size is known once the
        sweep's plan has run and recorded it. None: not known yet.
        """
        if self.hub is not None and self.hub.size is not None:
            return self.hub.size
        if (got := (self.state.get("sensitivity") or {}).get("budget")) is not None:
            return int(got) if isinstance(got, int) and got > 0 else None
        return None

    def make_bf16(self) -> Path:
        """The bf16 GGUF of the model, converted once: candidate 0's template, the engine floor's checkpoint and
        every search build's template (`judge.llamacpp.template`)."""
        programs = llama_programs(self.cards)
        try:
            return cand0.make_template(
                self.python,
                programs["llama_cpp"],
                self.model_dir,
                self.bf16,
                cache=template_cache(self.model_dir, programs["llama_cpp"]),
            )
        except ValueError as e:
            raise Refusal(str(e)) from None

    def llama_commit(self) -> str | None:
        """The commit of the `QDA_LLAMA_CPP` checkout, or None when none is named or it cannot be read.

        The imatrix step keys on it; None (a plan, or a machine without llama.cpp) is "not known", never a
        guessed commit.
        """
        from qda_quant.gguf.llama_cpp import ENV, commit_of
        from qda_quant.step import Refused

        root = os.environ.get(ENV)
        if not root:
            return None
        try:
            return commit_of(Path(root).expanduser())
        except Refused:
            return None

    @property
    def calib_need(self) -> dict[str, int]:
        """The samples and seqlen candidate 0's recipe reads from the calibration file."""
        return {k: self.recipe()["calib"]["file"][k] for k in ("samples", "seqlen")}

    @property
    def calib_path(self) -> Path:
        """Candidate 0's calibration file: `--calib`, or the one the calib step renders."""
        return self.calib_file or self.calib

    def build_inputs(self) -> cand0.Inputs:
        """What candidate 0 is built from."""
        fmt = "w4a16" if self.gguf else group_size.spelling(self.fmt)
        return cand0.Inputs(str(self.model_dir), fmt, str(self.calib_path), self.qda(), {})


@dataclass
class Step:
    """One step: its output, its commands (for the plan), why it is already done, and how it runs."""

    name: str
    output: Path
    commands: Callable[[], list[list[str]]]
    done: Callable[[], str | None]
    run: Callable[[], str | None]  # a line for the run's log, or None


# ------------------------------------------------------------------------------------------------- the steps


def _hf(p: Prepare) -> str:
    beside = p.python.parent / "hf"
    found = str(beside) if beside.is_file() else shutil.which("hf")
    if not found:
        raise Refusal(
            'no `hf` command beside this interpreter or on PATH: pip install "huggingface_hub[cli]"',
            code=ENVIRONMENT,
        )
    return found


def _remember_source(p: Prepare, revision: str) -> None:
    """Leave the downloaded repo in the model directory, where every build from it reads it."""
    from qda_quant.config import SOURCE_RECORD

    p.model_dir.mkdir(parents=True, exist_ok=True)
    record = {"repo": p.repo, "revision": revision}
    (p.model_dir / SOURCE_RECORD).write_text(json.dumps(record, indent=2) + "\n")


def _model_step(p: Prepare) -> Step:
    def commands() -> list[list[str]]:
        if p.local:
            return []
        rev = p.revision or "main"
        return [["hf", "download", p.repo, "--revision", rev, "--local-dir", str(p.model_dir)]]

    def done() -> str | None:
        if p.local:
            return f"a local model, used as is: {p.local}"
        got = p.state.get("model") or {}
        if (p.model_dir / "config.json").is_file() and got.get("repo") == p.repo:
            if not p.revision or p.revision in (got.get("requested"), got.get("revision")):
                return f"{p.model_dir} holds {p.repo}@{got.get('revision')}"
        return None

    def run() -> None:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(p.repo, revision=p.revision or None).sha
        subprocess.run(
            [_hf(p), "download", p.repo, "--revision", str(sha), "--local-dir", str(p.model_dir)], check=True
        )
        _remember_source(p, str(sha))  # any build from this directory names the original, not "model"
        p.save(model={"repo": p.repo, "requested": p.revision or "main", "revision": sha})
        p.resolve_format(save=True)  # the model is here: read its widths before the reference runs

    return Step("model", p.model_dir, commands, done, run)


def _corpus_step(p: Prepare) -> Step:
    def done() -> str | None:
        manifest = _json(p.corpus.with_suffix(".manifest.json"))
        splits = manifest.get("splits") or {}
        files = [(p.corpus, manifest.get("sha256"))] + [
            (Path(s["path"]), s.get("sha256")) for s in splits.values()
        ]
        if {"eval", "holdout", "search96"} <= set(splits) and all(
            f.is_file() and sha256(f) == want for f, want in files
        ):
            return f"{p.corpus} and its splits match their manifest"
        return None

    def run() -> None:
        if p.corpus.exists():
            raise Refusal(
                f"{p.corpus} does not match its manifest; remove DIR/data/corpus* to build it again"
            )
        p.corpus.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*p.qda_kl(), "corpus", "--out", str(p.corpus)], check=True)

    return Step("corpus", p.corpus, lambda: [[*p.qda_kl(), "corpus", "--out", str(p.corpus)]], done, run)


def sampling(model_dir: Path) -> dict[str, Any] | None:
    """The sampling `qda-kl ref` is given: None when the model's card or generation_config.json names it."""
    from qda_kl.reference import default_sampling

    return default_sampling(model_dir)


def ref_command(
    p: Prepare,
    model: Path,
    prompts: Path,
    out: Path,
    extra: list[str] | None = None,
    *,
    holdout: bool = False,
) -> list[str]:
    """`qda-kl ref` on `prompts`, on the prepare's cards, with the explicit sampling a model without one needs.

    With `holdout`, the same command also builds the holdout reference `evaluate` reads (`qda-kl ref --group`
    naming no output: it warms the shared reference cache, and `evaluate` links it into the run when needed),
    so one engine load settles both prompt sets.
    """
    command = [*p.qda_kl(), "ref", "--model", str(model), "--prompts", str(prompts), "--out", str(out)]
    command += [
        "--tp",
        str(ref_layout(p, model)[0]),
        "--gpu-memory-utilization",
        str(DEFAULT_GPU_MEMORY_UTILIZATION),
    ]
    command += list(extra or [])
    if (model / "config.json").is_file() and (named := sampling(model)) is not None:
        command += ["--sampling", json.dumps(named)]
    if holdout:
        command += ["--group", json.dumps({"prompts": str(p.holdout)})]
    return command


def ref_card_bytes(cards: Sequence[int]) -> int | None:
    """What one of `cards` holds (`fisher.CARD_SHARE` of the smallest one's memory), read from the driver.

    `cards` are `--cards` numbers (physical, as `group_size.target_for` reads them), mapped to their GPUs the
    way `resolve_cards` does and sized by nvidia-smi, so a node whose cards differ in size measures the cards
    the run uses and not the first few CUDA would number. None when the driver will not say.
    """
    from qda_quant.intake.fisher import CARD_SHARE

    from qda_agent.isolation.devices import resolve_cards, smi_memory

    try:
        memory = {card.bus: total for card, total in smi_memory() or ()}
        sizes = [memory[card.bus] for card in resolve_cards(cards)]
    except (OSError, ValueError, KeyError):  # no driver table, a card this job cannot see, or one smi omits
        return None
    return int(min(sizes) * CARD_SHARE) if sizes else None


def ref_layout(p: Prepare, model: Path) -> tuple[int, str]:
    """The reference's tensor-parallel size over `--cards`, and the reason the step's log gives for it.

    A model whose weights fill at most `REF_CARD_FILL` of one card reads on one card, without tensor
    parallelism, however many cards `--cards` names. Otherwise -- and whenever the weights or the card size
    are unknown (a dry run that cannot size the download, no CUDA) -- it reads on the most of the cards vLLM
    splits it over (`ref_tp`).
    """
    cards = len(p.cards)
    weights = _model_bytes(p) if model == p.model_dir else None
    per = ref_card_bytes(p.cards) if cards > 1 and weights else None
    if weights and per and weights <= REF_CARD_FILL * per:
        return 1, (
            f"the model's {weights / 1e9:.1f} GB of weights fit one {per / 1e9:.0f} GB card: one card, "
            "no tensor parallelism"
        )
    tp = ref_tp(model, cards)
    if tp < cards:
        return tp, f"vLLM splits this model's attention heads over at most {tp} of them"
    return tp, ""


def ref_tp(model: Path, cards: int) -> int:
    """The most of `cards` vLLM splits `model` over: the attention heads divide by it, and the KV heads divide
    by it or it by them (vLLM's own checks). A model without a readable config.json takes every card."""
    try:
        cfg = json.loads((model / "config.json").read_text())
    except (OSError, ValueError):
        return cards
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    heads = text.get("num_attention_heads")
    if not isinstance(heads, int) or heads < 1:
        return cards
    kv = text.get("num_key_value_heads") or heads

    def fits(tp: int) -> bool:
        return heads % tp == 0 and (kv % tp == 0 if tp <= kv else tp % kv == 0)

    return next(tp for tp in range(cards, 0, -1) if fits(tp))


def ref_split(manifest: dict[str, Any]) -> str:
    """Where the reference's build time went, from its own record: engine start, generation, teacher forcing,
    and the rest (process start, tokenizer checks, writing). It is the build's record even when this step
    reused a cached reference, so nothing here is inferred from how long this step took."""
    seconds = (manifest.get("build") or {}).get("seconds") or {}
    parts = {k: float(seconds.get(k) or 0.0) for k in ("engine_start", "generate", "teacher_force")}
    other = max(float(seconds.get("total") or 0.0) - sum(parts.values()), 0.0)
    shown = ", ".join(f"{k.replace('_', ' ')} {v:.1f} s" for k, v in parts.items())
    return f"      ref (from its manifest): {shown}, other {other:.1f} s (process start, tokenizer checks, writing)"


def reference_prompts_readable(ref: Path) -> bool:
    """Whether the reference's recorded prompt file still resolves, by the one reader of that record.

    A reference whose prompt file cannot be read cannot give the search its calibration index, so the ref step
    is not done: rerunning it hands the entry its own copy from this run's corpus (the cache heals an entry
    that recorded a path into a run's directory), where before `done` said the step was finished and the start
    failed.
    """
    from qda_quant.calib.contamination import reference_prompt_file, reference_rows
    from qda_quant.step import Refused

    try:
        reference_rows(reference_prompt_file(ref)[0])
    except (Refused, OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _ref_step(p: Prepare) -> Step:
    """The search reference and the holdout reference `evaluate` reads, in one engine load.

    The search reference reads the 96 `search96` prompts, every candidate's reading takes minutes on any
    engine; the full eval split would make a 27B reference take half an hour and a llama.cpp reading over an
    hour, so `qda-agent evaluate` reads the full holdout split instead. Both prompt sets read the same original
    with the same settings, so one `qda-kl ref --cache` builds both: one engine load and one copy of the model
    on the cards, not two. Each lands in the reference cache; `evaluate` links the holdout one into the run
    when it needs it, and builds it then if this prepare did not warm it.
    """

    def command() -> list[str]:
        # one build per model and prompts
        extra = ["--subset", f"{SEARCH_SUBSET}={p.search_ids}", "--cache"]
        return ref_command(p, p.model_dir, p.search, p.ref, extra, holdout=True)

    def done() -> str | None:
        built = _json(p.ref / "manifest.json").get("build") or {}
        source = p.source_identity()
        if (
            source is not None
            and p.state.get("ref_source") == source
            and p.search.is_file()
            and built.get("prompts_sha256") == sha256(p.search)
            and reference_prompts_readable(p.ref)
        ):
            return f"{p.ref} was built from {source} on {p.search.name}"
        return None

    def run() -> str:
        source = p.source_identity()
        if source is None:
            raise Refusal(
                f"{p.model_dir}: no model revision is recorded; run the model step first (it records the "
                "revision a reference is built from)"
            )
        if p.ref.exists() and not p.ref.is_symlink():  # a link into the reference cache is replaced
            raise Refusal(
                f"{p.ref} was not built from {source} on {p.search.name}; remove it to build the reference again"
            )
        subprocess.run(command(), check=True, env=p.env())
        # the search reference is the link's manifest; the holdout entry, when warmed, lives in the cache only
        tp, why = ref_layout(p, p.model_dir)
        used = f"      ref on {tp} of {len(p.cards)} card(s)" + (f" ({why})" if why else "")
        detail = ref_split(_json(p.ref / "manifest.json")) + "\n" + used
        p.save(
            sampling=sampling(p.model_dir) or "the model's released card or generation_config.json",
            search96={"ids": str(p.search_ids), "sha256": sha256(p.search_ids), "prompts": str(p.search)},
            ref_source=source,
            gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        )
        return detail

    return Step("ref", p.ref, lambda: [command()], done, run)


def _calib_command(p: Prepare) -> list[str]:
    return [
        *p.qda(),
        *("calib", "render", "--model", str(p.model_dir), "--names", POOL_NAMES),
        *("--samples", str(p.calib_need["samples"]), "--seqlen", str(p.calib_need["seqlen"])),
        *("--out", str(p.calib), "--thinking", "keep"),
    ]


def _pools_command(p: Prepare, revision: str) -> list[str]:
    files = [f"{name}.jsonl" for name in AGENT_POOLS]
    return [
        "hf",
        "download",
        POOL,
        *files,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(p.pools),
    ]


def _agent_mix() -> tuple[tuple[str, int], ...]:
    """The default calibration's mix: `POOL_NAMES`' weights over the pools the agent gets as data sets.

    The weights are the render's own table (`qda_quant.calib.render.parse_names`), and `AGENT_POOLS` is
    derived from the same table, so every pool the agent gets has a weight.
    """
    weights = parse_names(POOL_NAMES)
    return tuple((name, weights[name]) for name in AGENT_POOLS)


def _write_agent_calib(p: Prepare) -> dict[str, Any]:
    """Write :attr:`Prepare.agent_calib`: the agent's `messages` pools, interleaved at the render's weights.

    This is the search's default calibration, and it is a row source rather than a cut: every row the pools
    hold reaches the file, so a build may name any ``calib.file.samples`` the rows pack. Candidate 0's recipe
    reads :attr:`Prepare.calib` (the least rows its samples need) instead. The agent may copy this file and
    add rows of its own; the file is written atomically so a rerun replaces it whole.
    """
    mix = _agent_mix()
    queues = [(p.pools / f"{name}.jsonl").read_text().splitlines(keepends=True) for name, _ in mix]
    rows: list[str] = []
    while any(queues):
        for (_, weight), queue in zip(mix, queues, strict=True):  # queues is one entry per mix item
            take, queue[:] = queue[:weight], queue[weight:]
            rows += take
    tmp = p.agent_calib.with_name(p.agent_calib.name + ".tmp")
    tmp.write_text("".join(rows))
    tmp.replace(p.agent_calib)
    return {"sha256": sha256(p.agent_calib), "rows": len(rows), "mix": dict(mix)}


#: Runs in the build's interpreter: the builder's own sample-count rule decides whether a `--calib` file fills
#: candidate 0's samples x seqlen; a short or unreadable file exits 2 with the builder's message.
_CHECK_CALIB = """\
import sys
from qda_quant.calib.sources import FileSource
from qda_quant.engine.calib import load_samples
from qda_quant.step import Refused

path, samples, seqlen, model = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
try:
    load_samples(FileSource(path=path, samples=samples, seqlen=seqlen), model)
except Refused as e:
    print(e, file=sys.stderr)
    raise SystemExit(2)
"""


def _check_calib(p: Prepare) -> dict[str, Any]:
    """Refuse a `--calib` file candidate 0 cannot fill, before any build; its record for `prepare.json`.

    Raises:
        Refusal: the builder cannot take candidate 0's samples of seqlen from the file (its own message).
    """
    need = p.calib_need
    path = cast(Path, p.calib_file)
    command = [str(p.python), "-c", _CHECK_CALIB, str(path), str(need["samples"]), str(need["seqlen"])]
    done = subprocess.run([*command, str(p.model_dir)], check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise Refusal(
            f"--calib {path}: {(done.stderr or done.stdout).strip() or f'exited {done.returncode}'}"
        )
    return {"path": str(path), "sha256": sha256(path), **need}


def _calib_step(p: Prepare) -> Step:
    """Candidate 0's calibration, the search's default calibration, and the agent's `messages` pools.

    With `--calib` the file is both calibrations, as is: it is checked to fill candidate 0's samples, and the
    pools are still downloaded as the agent's data sets.
    """

    def meta() -> dict[str, Any]:
        return _json(p.calib.with_name(p.calib.name + ".meta.json"))

    def given() -> bool:
        got = p.state.get("calib_file") or {}
        need = p.calib_need
        return got.get("sha256") == sha256(cast(Path, p.calib_file)) and all(
            got.get(k) == need[k] for k in need
        )

    def rendered() -> bool:
        if p.calib_file is not None:
            return given()
        got = meta()
        need = p.calib_need
        same = got.get("model") == str(p.model_dir) and (got.get("samples"), got.get("seqlen")) == (
            need["samples"],
            need["seqlen"],
        )
        return same and p.calib.is_file() and got.get("sha256") == sha256(p.calib)

    def pooled() -> bool:
        files = (p.state.get("calib_file") or {}).get("pools") if p.calib_file else meta().get("files")
        files = files or {}
        return all((f := p.pools / f"{n}.jsonl").is_file() and sha256(f) == files.get(n) for n in AGENT_POOLS)

    def merged() -> bool:
        got = p.state.get("agent_calib") or {}
        return p.agent_calib.is_file() and got.get("sha256") == sha256(p.agent_calib)

    def done() -> str | None:
        if p.calib_file is not None and rendered() and pooled():
            return f"{p.calib_file} is candidate 0's and the search's calibration; the agent's pools are in {p.pools}"
        if rendered() and pooled() and merged():
            return (
                f"{p.calib} was rendered for {p.model_dir.name}; the agent's pools and default calibration "
                f"({p.agent_calib.name}) are in {p.pools.parent}"
            )
        return None

    def run() -> None:
        if p.calib_file is not None:
            record = _check_calib(p)
            from huggingface_hub import HfApi

            repo, _, revision = POOL.partition("@")
            sha = str(HfApi().dataset_info(repo, revision=revision or None).sha)
            subprocess.run([_hf(p), *_pools_command(p, sha)[1:]], check=True)
            pools = {n: sha256(p.pools / f"{n}.jsonl") for n in AGENT_POOLS}
            p.save(calib_file={**record, "pools": pools, "pool_revision": sha})
            return
        if not rendered():
            if p.calib.exists():
                raise Refusal(f"{p.calib} was made for another model or mix; remove it and its .meta.json")
            subprocess.run(_calib_command(p), check=True)
        command = _pools_command(p, str(meta()["pool"]["revision"]))
        subprocess.run([_hf(p), *command[1:]], check=True)
        if not pooled():
            raise Refusal(f"{p.pools}: the pools differ from the ones {p.calib} was rendered from")
        p.save(agent_calib=_write_agent_calib(p))

    def commands() -> list[list[str]]:
        if p.calib_file is not None:
            return [_pools_command(p, "<the pool's current revision>")]
        return [_calib_command(p), _pools_command(p, "<the render's revision>")]

    return Step("calib", p.calib_path, commands, done, run)


def _imatrix_step(p: Prepare) -> Step:
    """The importance matrix, measured once per (model, calibration file, llama.cpp commit) on this machine.

    `llama-imatrix` measures the bf16 template over the calibration text; its file is what a GGUF build's
    `exporter.gguf.imatrix` weights the tensors taken from the template with. The measurement is cached
    (`qda_agent.imatrix_cache`) and placed at `DIR/imatrix/model.imatrix.gguf` as a hard link (a copy across
    file systems): every build reads that path, and a sandbox binds it like any other file of the run.
    """

    def need() -> dict[str, int]:
        return p.calib_need

    def text() -> Path:
        from qda_agent import imatrix_cache

        return p.out / "data" / imatrix_cache.TEXT

    def commands() -> list[list[str]]:
        from qda_agent import imatrix_cache

        try:
            program = imatrix_program(p.cards)
        except Refusal:  # the dry run's last line names the missing llama.cpp and its build commands
            return []
        return [
            imatrix_cache.command(
                program,
                p.bf16,
                text(),
                p.imatrix,
                context=need()["seqlen"],
                chunks=need()["samples"],
            )
        ]

    def done() -> str | None:
        record = _json(p.imatrix.with_name(p.imatrix.name + ".json"))
        commit = p.llama_commit()
        if not p.imatrix.is_file() or not record or commit is None:
            return None
        if record.get("llama_cpp") != commit or record.get("calib_sha256") != sha256(p.calib_path):
            return None
        source = p.source_identity()
        if source is not None and record.get("source_identity") != source:
            return None
        return f"{p.imatrix} was measured on {p.calib_path.name} by llama.cpp {commit[:9]}"

    def run() -> None:
        from qda_agent import imatrix_cache

        source = p.source_identity()
        if source is None:
            raise Refusal(
                f"{p.model_dir}: no model revision is recorded; run the model step first (the imatrix is "
                "measured on the model the search reads)"
            )
        commit = p.llama_commit()
        if commit is None:
            raise Refusal(
                "the importance matrix is measured by llama-imatrix, which needs QDA_LLAMA_CPP at the pinned "
                "commit; name it and run the imatrix step again",
                code=ENVIRONMENT,
            )
        program = imatrix_program(p.cards)  # before the conversion: the refusal carries the build commands
        if not p.calib_path.is_file():
            raise Refusal(f"{p.calib_path}: no calibration file; run the same prepare again")
        p.make_bf16()  # the file llama-imatrix measures, and every GGUF build's template
        sizes = need()
        measured, built = imatrix_cache.cached(
            program=program,
            template=p.bf16,
            text=imatrix_cache.text_file(p.calib_path, text()),
            model=p.model_dir,
            calib=p.calib_path,
            commit=commit,
            context=sizes["seqlen"],
            chunks=sizes["samples"],
            env=p.env(),
        )
        imatrix_cache.place(measured, p.imatrix)
        record = {
            **p.made_from(),  # the judge reads `source`; `done` reads `source_identity`
            "llama_cpp": commit,
            "calib_sha256": sha256(p.calib_path),
            "sha256": sha256(p.imatrix),
            "cache": str(measured),
            "context": sizes["seqlen"],
            "chunks": sizes["samples"],
            "measured_by_this_prepare": built,
        }
        p.imatrix.with_name(p.imatrix.name + ".json").write_text(json.dumps(record, indent=1) + "\n")
        p.save(imatrix={"path": str(p.imatrix), "cache": str(measured), "llama_cpp": commit})

    return Step("imatrix", p.imatrix, commands, done, run)


def _libggml(programs: dict[str, Path]) -> Path | None:
    """The checkout's built `libggml-base`, which a Q5_K read needs (llama.cpp's own quantizer)."""
    path = programs["llama_cpp"] / "build" / "bin" / "libggml-base.so"
    return path if path.is_file() else None


def _qda_output(command: list[str], env: dict[str, str] | None = None, log: Path | None = None) -> str:
    """Run a `qda` command whose output is captured; a failure is refused with its own last stderr line.

    The captured command prints nothing itself, so `execute`'s "its output is above" would be empty; the
    refusal names what the command said instead. With `log`, a failed command's whole output is kept there
    and the refusal names the file: an engine's last line ("see root cause above") is not its cause.
    """
    done = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if done.returncode:
        said = [line for line in (done.stderr or done.stdout or "").splitlines() if line.strip()]
        detail = said[-1] if said else "no output"
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(done.stdout + done.stderr)
            detail += f" (its whole output: {log})"
        raise Refusal(f"`{' '.join(command)}` exited {done.returncode}: {detail}")
    return done.stdout


def _sensitivity_read(
    p: Prepare,
    base: Path,
    cell: tuple[str, str] | None,
    work: Path,
    cards: list[int],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """One `qda sensitivity read` on `cards`, returning its row."""
    name = "bf16" if cell is None else f"{cell[0]}-{cell[1]}"
    out = work / "reads" / name
    command = [
        *p.qda(),
        *("sensitivity", "read", str(base)),
        "--ref",
        str(p.ref),
        "--out",
        str(out),
        "--subset",
        SEARCH_SUBSET,
        "--tp",
        str(len(cards)),
        "--json",
    ]
    if cell is not None:
        command += ["--class", cell[0], "--type", cell[1]]
    if (lib := _libggml(llama_programs(cards))) is not None:
        command += ["--libggml", str(lib)]
    env = {**p.env(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))}
    log(f"sensitivity: {name} on cards {cards}")
    _qda_output(command, env=env, log=out.parent / f"{name}.log")
    row = json.loads((out / "row.json").read_text())
    return row


def _sensitivity_step(p: Prepare) -> Step:
    """The class sweep candidate 0's type map is allocated from (`--sensitivity`; off by default).

    `qda sensitivity base` lays the bf16 checkpoint out one class per shard once, then every cell (a class at
    one type) is one `qda sensitivity read`: a fake-quantized variant scored against the run's own reference on
    the `search96` subset. The reads are independent, so they run one per group of `sensitivity_tp()` cards.
    The rows are cached per (model, template, reference, quantizer, arm set); the allocation inside the budget
    is re-derived from them with `qda sensitivity table`, and its rules with `qda sensitivity types`.

    The budget is `--budget-bpw`'s: an `hf:` GGUF's file size, or the Q4_K_M rule's own file size for the
    `candidate0` default. A number of bits is refused for a llama.cpp search before any step.
    """
    work = p.out / "sensitivity" / ".work"

    def plan_rows() -> list[dict[str, Any]]:
        text = _qda_output([*p.qda(), *("sensitivity", "plan", str(p.bf16), "--json")])
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def reference() -> dict[str, Any]:
        return {
            "content_sha256": _json(p.ref / "manifest.json").get("content_sha256"),
            "subset": SEARCH_SUBSET,
        }

    def record_of() -> dict[str, Any]:
        return _json(p.sensitivity_record)

    def done() -> str | None:
        record = record_of()
        if not (p.sensitivity_rows.is_file() and record and p.allocation.is_file() and p.type_map.is_file()):
            return None
        commit = p.llama_commit()
        if commit is None or record.get("llama_cpp") != commit:
            return None
        source = p.source_identity()
        if source is not None and record.get("source") != source:
            return None
        if record.get("reference") != reference() or record.get("floor") != SENSITIVITY_FLOOR:
            return None
        if record.get("budget_spec") != str(p.budget):
            return None
        if (want := p.sensitivity_budget()) is not None and record.get("budget") != want:
            return None
        return f"{p.sensitivity_rows} holds {len(record.get('arms', []))} measured arms"

    def run() -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from queue import Queue

        from qda_agent import sensitivity

        source = p.source_identity()
        if source is None:
            raise Refusal(
                f"{p.model_dir}: no model revision is recorded; run the model step first (the reads score "
                "the model the search reads)"
            )
        commit = p.llama_commit()
        if commit is None:
            raise Refusal(
                "the sensitivity sweep needs QDA_LLAMA_CPP at the pinned commit (the quantizer's bytes); "
                "name it and run prepare with --sensitivity again"
            )
        llama_programs(p.cards)  # pinned, and the libggml a Q5_K read loads
        if not (p.ref / "manifest.json").is_file():
            raise Refusal(f"{p.ref}: no reference; run the ref step first")
        ref = reference()
        if not ref["content_sha256"]:
            raise Refusal(f"{p.ref / 'manifest.json'}: the reference records no content hash")
        if not p.cards:
            raise Refusal("--cards: a sensitivity read needs at least one card")
        if p.budget.kind == "hf" and (p.hub is None or p.hub.size is None):
            raise Refusal(f"--budget-bpw {p.budget}: the Hub checkpoint's file size was not read")
        p.make_bf16()
        rows = plan_rows()
        totals = next((row["totals"] for row in rows if "totals" in row), {})
        arms = sensitivity.cells(rows)
        if not arms:
            raise Refusal(f"{p.bf16}: the plan names no class to read")
        payload = sensitivity.key_payload(p.model_dir, p.bf16, reference=ref, arms=arms)
        target = sensitivity.entry(payload)
        hit = sensitivity.lookup(payload)
        if hit is None:
            with sensitivity.measuring(target):
                hit = sensitivity.lookup(payload)
                if hit is None:  # another prepare may have measured it while this one waited
                    shutil.rmtree(work, ignore_errors=True)
                    base = work / "base"
                    print(f"sensitivity: base {base} ({len(arms)} arms to read)")
                    subprocess.run(
                        [*p.qda(), *("sensitivity", "base", str(p.model_dir), "--out", str(base))],
                        check=True,
                        env=p.env(),
                    )
                    tp = p.sensitivity_tp()
                    groups = [p.cards[i : i + tp] for i in range(0, len(p.cards), tp)]
                    baseline = _sensitivity_read(p, base, None, work, groups[0], print)
                    measured: dict[tuple[str, str], dict[str, Any]] = {}
                    # a read takes whichever group is free and gives it back: two engines never share a card
                    free: Queue[list[int]] = Queue()
                    for group in groups:
                        free.put(group)

                    def read(cell: tuple[str, str]) -> dict[str, Any]:
                        cards = free.get()
                        try:
                            return _sensitivity_read(p, base, cell, work, cards, print)
                        finally:
                            free.put(cards)

                    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                        futures = {pool.submit(read, cell): cell for cell in arms}
                        for future in as_completed(futures):
                            cell = futures[future]
                            row = future.result()
                            measured[cell] = {
                                "class": cell[0],
                                "type": cell[1],
                                "kl": row["kl"],
                                "d_kl": row["kl"] - float(baseline["kl"]),
                            }
                            print(f"sensitivity: {cell[0]} {cell[1]} dKL {measured[cell]['d_kl']:.6g}")
                    work_rows = work / "rows.json"
                    work_rows.write_text(json.dumps([measured[cell] for cell in arms], indent=1) + "\n")
                    sensitivity.store(work_rows, payload, baseline_kl=float(baseline["kl"]))
                    hit = sensitivity.rows_path(target)
                    shutil.rmtree(work, ignore_errors=True)
        p.sensitivity_rows.parent.mkdir(parents=True, exist_ok=True)
        p.sensitivity_rows.write_bytes(hit.read_bytes())
        baseline_kl = float(_json(sensitivity.record_path(target)).get("baseline_kl") or 0.0)
        # the sweep's allocation budget is --budget-bpw's: an hf: GGUF's bytes, or the Q4_K_M rule's own file
        if p.budget.kind == "hf":
            if p.hub is None or p.hub.size is None:
                raise Refusal(f"--budget-bpw {p.budget}: the Hub checkpoint's file size was not read")
            budget = int(p.hub.size)
        else:
            budget = int(totals.get("rule_file_bytes") or 0)
        if budget <= 0:
            raise Refusal("no budget for the type map: the plan stated no rule file size")
        allocation = json.loads(
            _qda_output(
                [
                    *p.qda(),
                    *("sensitivity", "table", str(p.bf16), str(p.sensitivity_rows)),
                    "--budget-bytes",
                    str(budget),
                    "--floor",
                    str(SENSITIVITY_FLOOR),
                    "--json",
                ]
            )
        )
        p.allocation.write_text(json.dumps(allocation, indent=1) + "\n")
        rules = _qda_output(
            [*p.qda(), *("sensitivity", "types", str(p.bf16), str(p.allocation), "--compact")]
        )
        p.type_map.write_text(json.dumps(json.loads(rules), indent=1) + "\n")
        chosen = one_allocation(allocation)
        record = {
            "source": source,
            "llama_cpp": commit,
            "reference": ref,
            "arms": [list(cell) for cell in arms],
            "cache": str(hit),
            "budget": budget,
            "budget_spec": str(p.budget),
            "floor": SENSITIVITY_FLOOR,
            "file_bytes": chosen.get("file_bytes"),
            "d_kl": chosen.get("d_kl"),
            "baseline_kl": baseline_kl,
        }
        p.sensitivity_record.write_text(json.dumps(record, indent=1) + "\n")
        picked = sum(1 for kind in (chosen.get("picks") or {}).values() if kind != "bf16")
        print(
            f"sensitivity: {len(arms)} arms, type map {picked} classes, "
            f"{chosen.get('file_bytes')} B, dKL {float(chosen.get('d_kl') or 0):.6g}"
        )
        p.save(
            sensitivity={
                "rows": str(p.sensitivity_rows),
                "cache": str(hit),
                "arms": len(arms),
                "budget": budget,
                "file_bytes": chosen.get("file_bytes"),
                "d_kl": chosen.get("d_kl"),
            }
        )

    def commands() -> list[list[str]]:
        return [
            [*p.qda(), *("sensitivity", "plan", str(p.bf16), "--json")],
            [*p.qda(), *("sensitivity", "base", str(p.model_dir), "--out", "<DIR/sensitivity/.work/base>")],
            [
                *p.qda(),
                *("sensitivity", "read", "<base>", "--class", "<class>", "--type", "<type>"),
                *("--ref", str(p.ref), "--out", "<DIR/sensitivity/.work/reads/...>"),
                *("--subset", SEARCH_SUBSET, "--tp", str(p.sensitivity_tp()), "--json"),
            ],
            [
                *p.qda(),
                *("sensitivity", "table", str(p.bf16), str(p.sensitivity_rows)),
                *("--budget-bytes", "<the rule's file size>", "--json"),
            ],
        ]

    return Step("sensitivity", p.sensitivity_rows, commands, done, run)


def _cand0_step(p: Prepare) -> Step:
    method = FORMATS[p.format]

    def recipe(template: bool = True) -> str:
        return yaml.safe_dump(p.recipe(template=template), sort_keys=False)

    def done() -> str | None:
        written = p.cand0.with_name(f"{p.cand0.name}.recipe.yaml")
        source = p.source_identity()
        # a recipe with or without the template writes the same file: an earlier prepare's build still counts
        if (
            source is not None
            and (p.state.get("cand0") or {}).get("source") == source
            and (p.cand0 / "build.json").is_file()
            and written.is_file()
            and written.read_text() in (recipe(), recipe(template=False))
        ):
            return f"{p.cand0} was built from {written.name}"
        return None

    def run() -> None:
        source = p.source_identity()
        if source is None:
            raise Refusal(
                f"{p.model_dir}: no model revision is recorded; run the model step first (it records the "
                "revision candidate 0 is built from)"
            )
        if p.gguf and not p.imatrix.is_file():
            raise Refusal(
                f"no importance matrix at {p.imatrix}: run the same prepare again (a GGUF "
                "candidate 0 hands exporter.gguf.imatrix the run's one, and without it the tensors taken "
                "from the template are fitted unweighted)"
            )
        if p.sensitivity and not p.tensor_types():
            raise Refusal(
                f"no sensitivity type map at {p.type_map}: run prepare again with --sensitivity, "
                "or drop the flag to build the default Q4_K_M mix"
            )
        if p.cand0.exists():
            recorded = (p.state.get("cand0") or {}).get("source")
            why = f"the model is now {source}" if recorded and recorded != source else "it is unfinished"
            raise Refusal(
                f"{p.cand0} is an unfinished or other build ({why}); remove it and its .recipe.yaml"
            )
        types = p.tensor_types()
        libggml = p.libggml() if types else None
        if types and libggml is None:
            lib = llama_programs(p.cards)["llama_cpp"] / "build" / "bin" / "libggml-base.so"
            raise Refusal(
                f"no {lib}: a measured type map's Q5_K needs llama.cpp's native quantizer; build llama.cpp "
                "so build/bin/libggml-base.so exists (docs/formats.md, GGUF for llama.cpp)",
                code=ENVIRONMENT,
            )
        need = p.calib_need
        built = cand0.build_candidate0(
            p.build_inputs(),
            method=method,
            out=p.cand0,
            cards=p.cards,
            calib_samples=need["samples"],
            calib_seqlen=need["seqlen"],
            template=p.make_bf16() if p.gguf else None,
            imatrix=p.imatrix if p.gguf else None,
            tensor_types=types,
            libggml=libggml,
        )
        seconds = built.build_seconds if built.build_seconds is not None else built.wall_seconds
        p.save(
            cand0={"path": str(built.out), "method": method, "build_s": round(seconds, 3), "source": source}
        )

    def commands() -> list[list[str]]:
        build = [*p.qda(), "build", str(p.cand0.with_name(f"{p.cand0.name}.recipe.yaml"))]
        return ([_convert_command(p)] if p.gguf and not p.bf16.is_file() else []) + [build]

    return Step("cand0", p.cand0, commands, done, run)


def _convert_command(p: Prepare) -> list[str]:
    """llama.cpp's bf16 conversion of the model into `p.bf16` (`cand0.template_command`), for the plan."""
    llama = os.environ.get("QDA_LLAMA_CPP", "llama.cpp")
    return cand0.template_command(p.python, llama, p.model_dir, p.bf16)


def llama_checkout() -> tuple[Path, str]:
    """The QDA_LLAMA_CPP checkout and its commit, which must be the pinned one.

    Raises:
        Refusal: QDA_LLAMA_CPP is unset, is not a checkout, or is at another commit; the message holds the
            commands that make it (docs/formats.md, GGUF for llama.cpp).
    """
    from qda_quant.gguf.llama_cpp import ENV, PINNED, checkout_command, commit_of
    from qda_quant.step import Refused

    if not os.environ.get(ENV):
        raise Refusal(
            f"a GGUF search needs llama.cpp: {checkout_command()} && export {ENV}=$PWD/llama.cpp",
            code=ENVIRONMENT,
        )
    root = Path(os.environ[ENV]).expanduser().resolve()
    try:
        commit = commit_of(root)
    except Refused as e:
        raise Refusal(f"{ENV}: {e}; {checkout_command()}", code=ENVIRONMENT) from None
    if commit != PINNED:
        raise Refusal(f"{ENV}: {root} is at {commit}; git -C {root} checkout {PINNED}", code=ENVIRONMENT)
    return root, commit


def llama_programs(cards: Sequence[int] | None = None) -> dict[str, Path]:
    """The llama.cpp checkout, `llama-server` and the probe bridge a llama.cpp search runs, from QDA_LLAMA_CPP.

    `cards` are the run's, which the build commands in a refusal are written for (their CUDA architecture).

    Raises:
        Refusal: QDA_LLAMA_CPP is unset, not the pinned checkout, or its programs are not built; the message
            holds the commands that make them (docs/formats.md, GGUF for llama.cpp).
    """
    root, _ = llama_checkout()
    build = root / "build"
    programs = {"llama_cpp": root, "server": build / "bin" / "llama-server", "bridge": build / "probe.so"}
    if missing := [str(path) for path in programs.values() if not path.exists()]:
        raise Refusal(
            f"llama.cpp is not built: no {', '.join(missing)}; build it as docs/formats.md says "
            f"(GGUF for llama.cpp):\n{llama_build(root, build, cards)}",
            code=ENVIRONMENT,
        )
    return programs


def template_cache(model: Path, llama_cpp: str | Path) -> Path | None:
    """``$QDA_CACHE/gguf-template/<model fingerprint>/<llama.cpp commit>/model-bf16.gguf``, or None.

    The conversion is a pure function of the model's files and the converter, so it is made once per machine.
    The path is computed here rather than in `cand0` because `cand0` never imports `qda_quant`; a checkout git
    cannot read has no commit to key on and is not cached.
    """
    from qda_quant.intake.fingerprint import model_fingerprint
    from qda_quant.io import qda_cache

    commit = cand0._llama_commit(llama_cpp)
    if commit is None or not Path(model).is_dir():
        return None
    return qda_cache("gguf-template", model_fingerprint(model)[:16], commit[:12], "model-bf16.gguf")


#: How the floor was read: a floor read another way (llama.cpp's own kernels) is measured again.
FLOOR_READER = "decode-gguf+topk"


def floor_cache(p: Prepare, programs: dict[str, Path]) -> Path | None:
    """``$QDA_CACHE/gguf-floor/<key>``: the engine floor's reading, made once per machine, or None.

    The floor is the reading of the bf16 template (decoded, read with vLLM) against the reference: a pure
    function of the model's files, the converter (the pinned llama.cpp commit) and the reference's content. A
    local model is keyed by its content fingerprint; a reference without a recorded content hash, or a checkout
    git cannot read, is not cached.
    """
    from qda_quant.intake.fingerprint import model_fingerprint
    from qda_quant.io import qda_cache

    reference = _json(p.ref / "manifest.json").get("content_sha256")
    commit = cand0._llama_commit(programs["llama_cpp"])
    if not reference or commit is None or not p.model_dir.is_dir():
        return None
    payload = {
        "model": model_fingerprint(p.model_dir),
        "llama_cpp": commit,
        "reader": FLOOR_READER,
        "reference": reference,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return qda_cache("gguf-floor", digest[:24])


def imatrix_program(cards: Sequence[int] | None = None) -> Path:
    """The pinned checkout's `llama-imatrix`, the one program the imatrix step runs.

    `cards` are the run's, which the build commands in a refusal are written for (as `llama_programs`).

    Raises:
        Refusal: the checkout is missing or unpinned (`llama_checkout`), or `llama-imatrix` is not built; the
            message holds the build commands (docs/formats.md, GGUF for llama.cpp).
    """
    root, _ = llama_checkout()
    build = root / "build"
    program = build / "bin" / "llama-imatrix"
    if not program.is_file():
        raise Refusal(
            f"llama.cpp is not built with llama-imatrix: no {program}; build it as docs/formats.md says "
            f"(GGUF for llama.cpp):\n{llama_build(root, build, cards)}",
            code=ENVIRONMENT,
        )
    return program


def _floor_fits_one_card(size: int, cards: Sequence[int]) -> None:
    """Refuse a search whose bf16 decode cannot load on one card, at prepare rather than mid-search.

    The floor reading, every candidate's KL reading and every long-generation check (`judge.gencheck_cards`,
    1 in the config prepare writes) each load a `.gguf`'s bf16 decode in one vLLM engine at tp=1. The card
    size is the smallest of the physical `cards` (`--cards`, as `ref_card_bytes` reads them), not of the first
    card CUDA numbers.

    Raises:
        Refusal: the driver reports a card smaller than the decode; unknown card sizes pass.
    """
    per = ref_card_bytes(cards)
    if per is None:  # the driver will not say: the reading's own load says what failed
        return
    if size > per:
        raise Refusal(
            f"the engine floor, each KL reading and each long-generation check load the bf16 model "
            f"({size / 1e9:.1f} GB) with vLLM on one card, and a card here "
            f"holds {per / 1e9:.1f} GB of weights: a GGUF search of this model does not fit this card",
            code=ENVIRONMENT,
        )


def _floor_step(p: Prepare) -> Step:
    score = p.floor / "score.json"
    config = p.out / "floor.yaml"

    def commands() -> list[list[str]]:
        return [_convert_command(p), [*p.qda_kl(), "score", str(config)]]

    def done() -> str | None:
        reference = _json(p.ref / "manifest.json").get("content_sha256")
        floor = p.state.get("floor") or {}
        if (
            score.is_file()
            and reference
            and floor.get("ref") == reference
            and floor.get("reader") == FLOOR_READER
        ):
            return f"{score} read the bf16 GGUF against this reference"
        return None

    def run() -> None:
        programs = llama_programs(p.cards)
        p.make_bf16()  # candidate 0's template already, when the cand0 step made it
        if p.bf16.is_file():
            _floor_fits_one_card(p.bf16.stat().st_size, p.cards)
        # the bf16 GGUF decoded against the model and read with vLLM, as every candidate's reading is
        body = {
            "checkpoint": str(p.bf16),
            "snapshot": str(p.model_dir),
            "reference": str(p.ref),
            "out": str(p.floor),
            "scorer": {"topk": {}},
        }
        config.write_text(yaml.safe_dump(body, sort_keys=False))
        cached = floor_cache(p, programs)
        if cached is not None and (cached / "score.json").is_file():
            shutil.rmtree(p.floor, ignore_errors=True)
            shutil.copytree(cached, p.floor)
        else:
            # one reading of the whole reference, and the number the judge's reading watchdog needs: the decode
            # and the load of a GGUF reading take longer than `judge.kl_score_s`'s constant estimate
            print(
                "      floor: decoding the bf16 GGUF and scoring it against the reference on one card",
                flush=True,
            )
            subprocess.run(commands()[1], check=True, env=p.env())
            if cached is not None:
                tmp = cached.with_name(f".{cached.name}.{os.getpid()}")
                shutil.copytree(p.floor, tmp, dirs_exist_ok=True)
                try:
                    tmp.rename(cached)  # one rename: a reader sees the whole reading or none
                except OSError:  # another prepare put the same reading there first
                    shutil.rmtree(tmp, ignore_errors=True)
        got = _json(score)
        p.save(
            floor={
                "kl": float(got["kl"]),
                "ref": _json(p.ref / "manifest.json").get("content_sha256"),
                # the whole reading `qda-kl score` timed (decode, load, scoring); `wall_s` is the scoring alone
                "seconds": _json(p.floor / "gguf-decode.json").get("reading_s") or got.get("wall_s"),
                "reader": FLOOR_READER,
                "cards": 1,  # the topk scorer's default `tp`
            }
        )

    return Step("floor", score, commands, done, run)


def gguf_file(checkpoint: Path) -> Path:
    """The one `.gguf` a GGUF build wrote into `checkpoint`."""
    files = sorted(checkpoint.glob("*.gguf"))
    if len(files) != 1:
        raise Refusal(f"{checkpoint}: {len(files)} .gguf files; candidate 0 is one GGUF build")
    return files[0]


#: The final evaluation's generation cap when the model's context holds it (`judge.eval_max_tokens`).
EVAL_MAX_TOKENS: int = 32768


def eval_max_tokens(model: Path) -> int:
    """`EVAL_MAX_TOKENS`, capped by the model's context: the number TASK.md promises the agent.

    The context is read where the reference reads it (`qda_kl.reference.context_length`), so the two
    never disagree about what the model holds.
    """
    from qda_kl.reference import context_length

    return min(EVAL_MAX_TOKENS, context_length(str(model)))


def require_room(p: Prepare) -> None:
    """Refuse before any GPU step when the run's disk is already below its free-space floor.

    The check is the run's own (`custody.space_problem` with a default `JudgeSettings`, so the floor is
    `custody.free_floor`'s, the one `qda doctor --search` reports and the judge host refuses to start on), read
    live on the disk `run_dir` will be on. The model may not be downloaded yet, so its weights are sized from
    the Hub when needed (`_model_bytes`). Prepare writes no disk cap and estimates no run size: the floor is
    watched while the run goes.
    """
    problem = space_problem(
        p.out / "run",
        cap=None,
        build=None,
        cards=len(p.cards),
        judge=JudgeSettings(),
        weights=_model_bytes(p),
    )
    if problem is not None:
        raise Refusal(str(problem))


def require_compiler() -> None:
    """Refuse where the ref step's engine start would (`qda_quant.output.vllm_env.vllm_environ`), before step 1.

    A missing C compiler or a Python without its headers otherwise stops the ref step after the model download
    and the corpus. The steps run this interpreter.
    """
    from qda_quant.output.vllm_env import vllm_environ

    try:
        with vllm_environ({}):
            pass
    except Refused as refused:
        raise Refusal(str(refused), code=refused.code, fix=refused.fix or None) from None


def require_cache_room() -> None:
    """Refuse when a cache or temporary disk is short (`room.cache_room`, the lines `qda doctor` prints).

    A note (a cache defaulting into a network `$HOME`, `room.room_notes`) is a warning on stderr, with its fix.
    """
    for note in room_notes():
        print(f"qda-agent prepare: warning: {note.detail}", file=sys.stderr)
    if misses := [c for c in cache_room() if c.missed]:
        # a short disk is the machine (the contract's row 3): free it, or point the cache elsewhere
        raise Refusal("; ".join(f"{c.detail}; fix: {c.fix}" for c in misses), code=ENVIRONMENT)


def head_rule(model_dir: Path) -> str:
    """The output head's rule the source checkpoint itself states.

    `16-bit` unless the source's own release stores the head quantized: its `quantization_config` gives
    lm_head a group of its own (the W3A16 releases store it INT4 g64), or a GGUF source's `output.weight`
    is a quantized type. Read from the config or the GGUF header; no weights are read. `judge.head`, TASK.md
    and both judges read the one value, so an exception model's builds keep the released head.
    """
    from qda_agent.judge.checks.formats import HEAD_16BIT

    found = sorted(p for p in model_dir.glob("*.gguf") if p.is_file())
    if found:  # a GGUF source states the head's stored type in its header
        try:
            from qda_quant.gguf.file import read_table, type_name

            table = read_table(found[0])
        except (OSError, ValueError, ImportError):
            return HEAD_16BIT
        head = next((t for t in table.tensors if t.name == "output.weight"), None)
        if head is None:
            return HEAD_16BIT
        name = type_name(head.tensor_type)
        return HEAD_16BIT if name in ("F32", "F16", "BF16") else name
    config = model_dir / "config.json"
    if not config.is_file():
        return HEAD_16BIT
    try:
        cfg = json.loads(config.read_text())
    except (OSError, ValueError):
        return HEAD_16BIT
    if not isinstance(cfg, dict):
        return HEAD_16BIT
    q = cfg.get("quantization_config")
    q = q if isinstance(q, dict) else {}
    ignored = [str(x) for x in q.get("ignore") or ()]
    if any(x == "lm_head" or x.endswith(".lm_head") for x in ignored):
        return HEAD_16BIT
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)):
        return HEAD_16BIT  # a tied head has no tensor of its own
    for group in (q.get("config_groups") or {}).values():
        targets = group.get("targets") if isinstance(group, dict) else None
        if not any("lm_head" in str(t) for t in targets or ()):
            continue
        weights = group.get("weights") or {}
        bits, size, sym = weights.get("num_bits"), weights.get("group_size"), weights.get("symmetric")
        if isinstance(bits, int) and isinstance(size, int):
            return f"w{bits}a16-{'sym' if sym else 'asym'}-g{size}"
    return HEAD_16BIT


def search_config(p: Prepare) -> dict[str, Any]:
    """The run config: every input prepare made, and each derived field filled once from its one source.

    `judge.anchor` and `scheduler.readable` are the model; `baselines[0]` is candidate 0, which is also the
    vLLM search's `judge.format_reference`; candidate 0's KL is read at host start (`select_candidate0`).
    The budget is `--budget-bpw`'s (`Prepare.budget_value`): `judge.budget_bpw`, with `judge.head_lease`, on
    vLLM; `judge.budget_bytes` on llama.cpp, whose `engine_floor` is the bf16 GGUF's KL.

    Raises:
        Refusal: a step this config needs has not run (the message names it).
    """
    made = p.state.get("cand0") or {}
    if not made.get("build_s") or not p.cand0.is_dir():
        raise Refusal(f"no candidate 0 at {p.cand0}: run the same prepare again to build it")
    # The registry's backend with its own defaults (`qda_agent.backends`), as the config holds it.
    backend = p.agent.model_dump(mode="json", exclude_none=True)
    if backend.get("name") == "dsh" and not (isinstance(p.backend, dict) and p.backend.get("model")):
        # dsh serves different names on the official API and on a gateway, and which one the run reaches is
        # decided by its account and environment, not by this prepare's: a config that names no model takes
        # the one its endpoint serves (docs/search-config.md), so a gateway account edits nothing.
        backend.pop("model", None)
    model = str(p.model_dir)
    sets = {name: str(p.pools / f"{name}.jsonl") for name in AGENT_POOLS}
    components = [{"name": "data", "sets": sets} if c == "data" else c for c in Qda.preset]
    judge: dict[str, Any] = {
        "serve_python": str(p.serve_python),
        "kl_ref": str(p.ref),
        "kl_holdout_ref": str(p.holdout),
        "anchor": model,
        "eval_max_tokens": eval_max_tokens(p.model_dir),
        "head": head_rule(p.model_dir),
        "select_candidate0": True,
        "candidate0_build_s": made["build_s"],
    }
    if p.gguf:
        floor = p.state.get("floor") or {}
        if "kl" not in floor:
            raise Refusal("no engine floor measured yet: run the same prepare again to measure it")
        programs = llama_programs(p.cards)
        judge |= {
            "engine": "llama.cpp",
            "budget_bytes": p.budget_value(),
            "engine_floor": floor["kl"],
            # every GGUF build reads prepare's one bf16 conversion and one importance matrix, outside the
            # run's cap (the imatrix weights the tensors taken from the template)
            "llamacpp": {
                **{k: str(v) for k, v in programs.items()},
                "template": str(p.bf16),
                "imatrix": str(p.imatrix),
            },
        }
        # the floor step measured this backend's reading on this host: the judge's watchdog reads it instead
        # of estimating llama.cpp with vLLM's `kl_score_s`. An earlier prepare (no measured seconds) falls
        # back on that step's own wall time, which also covers the conversion.
        measured = floor.get("seconds") or ((p.state.get("steps") or {}).get("floor") or {}).get("seconds")
        if measured:
            judge["reading_s"] = float(measured)
            if floor.get("cards"):
                judge["measured_cards"] = int(floor["cards"])
    else:
        judge |= {"format_reference": str(p.cand0), "budget_bpw": p.budget_value()}
        if p.head_lease:
            judge["head_lease"] = True
    row_format = p.format if p.gguf else group_size.spelling(p.fmt)  # the task and end report name it
    if not p.gguf and p.fmt.act == "nvfp4_block":
        # W4A4 rows: NVFP4 dense (or mixed with FP8 W8A8); the W4 default's g16 formats are refused on sm100
        judge["format_menu"] = ["bf16", "w8a8-fp8", row_format]
    return {
        "model": model,
        "format": row_format,
        "budget_gpu_hours": p.budget_gpu_hours,
        "cards": list(p.cards),
        "run_dir": str(p.out / "run"),
        "calib": str(p.calib_file or p.agent_calib),
        "baselines": [str(p.cand0)],
        "agent": {"backend": backend},
        "sandbox": {"name": "none"},
        "isolation": {"level": "none"},
        "scheduler": {"name": "qsched", "readable": [model]},
        "launcher": {"name": "local"},
        "toolset": {"name": "qda", "python": str(p.python), "components": components},
        "contamination": {"name": "none"},
        "judge": judge,
    }


HEADER = (
    PREPARE_MARK
    + """; run it with `qda-agent search {out}`. Every input below was made by prepare.
# The product defaults: no sandbox, no contamination check. To publish or report scores, build a guarded corpus
# and set `contamination: {{name: ngram, guard: FILE}}`. To isolate the agent and its jobs, see
# docs/search-config.md#isolation.
"""
)

#: The note beside `isolation:` in every search config: the one `prepare` writes and the shipped `configs/`
#: files say the same sentence (`test_isolation_config.py` holds them to it), so the default and how to leave it
#: are stated once.
ISOLATION_NOTE = "set strict (needs bubblewrap) to keep the agent away from the reference and cut its network"


def _annotate_isolation(body: str) -> str:
    """`body` (a dumped config) with `ISOLATION_NOTE` as a comment line above its top-level `isolation:` key.

    `yaml.safe_dump` writes no comments, so the note is put back on the one line it belongs to; the YAML itself
    stays the dump's.
    """
    key = "\nisolation:\n"
    if body.count(key) != 1:
        raise Refusal("the written config has no single top-level `isolation:` key to annotate")
    return body.replace(key, f"\n# {ISOLATION_NOTE}{key}", 1)


def budget_note(p: Prepare) -> str:
    """Where the search's budget came from, for the comment above it in the config."""
    if p.hub is not None:
        return f"budget: --budget-bpw {p.budget}, read from {p.hub.source}"
    if p.budget_bpw is not None:
        return f"budget: --budget-bpw {p.budget}"
    if p.budget.kind == "candidate0":
        if p.sensitivity:
            return "budget: the Q4_K_M rule's file size, the sensitivity sweep's allocation"
        return "budget: candidate 0's size (the default for a GGUF row)"
    return f"budget: the {p.format} row's default"


def _annotate_budget(p: Prepare, body: str) -> str:
    """Put `budget_note` above the judge's budget key; a config without one is returned as it is."""
    for key in ("\n  budget_bpw:", "\n  budget_bytes:"):
        if body.count(key) == 1:
            return body.replace(key, f"\n  # {budget_note(p)}{key}", 1)
    return body


def _config_step(p: Prepare) -> Step:
    def text() -> str:
        dump = yaml.safe_dump(search_config(p), sort_keys=False)
        return HEADER.format(out=p.out) + _annotate_budget(p, _annotate_isolation(dump))

    def done() -> str | None:
        try:
            same = p.config.is_file() and p.config.read_text() == text()
        except Refusal:
            return None
        return f"{p.config} is current" if same else None

    def run() -> None:
        from datetime import UTC, datetime, timedelta

        from qda_agent.config import RunConfig
        from qda_agent.flow import StartRefused, check_inputs
        from qda_agent.isolation import SignInRefused, sign_in_report

        # Write beside it and move it in only once it passes the run's own start checks: a refused config left
        # in place would compare equal on the next prepare and be reported current.
        tmp = p.config.with_name(p.config.name + ".tmp")
        tmp.write_text(text())
        try:  # the run's own start checks: a config prepare writes is never refused at run start
            cfg = RunConfig.load(tmp)
            check_inputs(cfg)
        except (configfile.ConfigError, StartRefused) as e:
            tmp.unlink(missing_ok=True)
            raise Refusal(f"the written config does not load: {e}") from None
        tmp.replace(p.config)
        # The search refuses an agent nothing signs in; say so now, while there is time to sign in.
        # `sign_in_report` is the run's own rule (it calls `sign_in_conflicts`), so prepare keeps no copy of it.
        try:
            deadline = datetime.now(UTC) + timedelta(seconds=cfg.wall_s)
            ok, _optional, lines = sign_in_report(cfg, deadline)
        except SignInRefused as e:
            ok, lines = False, [str(e)]
        if not ok:
            for line in lines:
                print(f"qda-agent prepare: warning: {line}", file=sys.stderr)

    return Step("config", p.config, lambda: [], done, run)


def plan(p: Prepare, steps: tuple[str, ...] = STEPS) -> list[Step]:
    """The steps to consider, in order (`imatrix` and `floor` only for a llama.cpp search)."""
    makers = {
        "model": _model_step,
        "corpus": _corpus_step,
        "ref": _ref_step,
        "calib": _calib_step,
        "imatrix": _imatrix_step,
        "sensitivity": _sensitivity_step,
        "cand0": _cand0_step,
        "floor": _floor_step,
        "config": _config_step,
    }
    llama = {"imatrix", "floor"}
    return [
        makers[name](p)
        for name in STEPS
        if name in steps
        and (name not in llama or p.gguf)
        and (name != "sensitivity" or (p.gguf and p.sensitivity))
    ]


def _model_bytes(p: Prepare) -> int | None:
    if p.model_dir.is_dir():
        return sum(f.stat().st_size for f in p.model_dir.glob("*.safetensors")) or None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(p.repo, revision=p.revision or None, files_metadata=True)
    except Exception:  # noqa: BLE001 - offline or unknown: the estimate says so
        return None
    return sum(s.size or 0 for s in info.siblings or () if s.rfilename.endswith(".safetensors")) or None


def disk_estimate(p: Prepare) -> list[tuple[str, int | None]]:
    """About how many bytes each output takes (None: unknown); the reference's is an upper bound."""
    from qda_kl.corpus import DOMAINS, SEARCH

    model = _model_bytes(p)
    rows = min(SEARCH, sum(n - held for *_, n, held in DOMAINS))
    out: list[tuple[str, int | None]] = [] if p.local else [("model download", model)]  # a local one stays
    out.append(("reference (at most: 2 answers x 8192 tokens x top-128)", rows * 2 * 8192 * 128 * 8))
    holdout_rows = sum(held for *_, _n, held in DOMAINS)
    out.append(
        ("holdout reference (at most: 2 answers x 8192 tokens x top-128)", holdout_rows * 2 * 8192 * 128 * 8)
    )
    cand0 = int(model * SIZE_RATIO[p.format]) if model else None
    out.append(("candidate 0", cand0))
    if p.gguf:
        out.append(("bf16 GGUF (engine floor, and every build's template)", model))
        # a KL reading and a long-generation check of a .gguf can run at once, each decoding into its own out
        out.append(
            (
                "bf16 decodes of the running GGUF jobs (a KL reading and a long-generation check at once, "
                "each in its out/decoded, removed after it)",
                2 * model if model else None,
            )
        )
    if p.gguf and p.sensitivity:  # one checkpoint per class until the sweep ends, then removed
        out.append(("sensitivity scratch (removed after the sweep)", model))
    out.append(
        (
            f"search run directory (typical: {RUN_DIR_FACTOR:g} x candidate 0)",
            int(cand0 * RUN_DIR_FACTOR) if cand0 else None,
        )
    )
    return out


def _gb(n: int | None) -> str:
    return "unknown" if n is None else f"{n / 1e9:.1f} GB"


def show_plan(p: Prepare, steps: list[Step]) -> str:
    """The dry run: each step, whether it would run, its commands, and the disk estimate."""
    shown = p.format if p.gguf else group_size.spelling(p.fmt)
    lines = [f"qda-agent prepare: plan for {p.model} -> {p.out} ({shown}, cards {p.cards})"]
    unit = "bytes" if p.gguf else "bpw"
    read = ""
    if p.hub is not None:  # a safetensors price is divided by the local model's weights
        value = "priced after the download"
        if p.gguf or (p.model_dir / "config.json").is_file():
            value = str(p.budget_value()) if p.gguf else f"{p.budget_value():g}"
        read = f", read from {p.hub.source}: {value}"
    lines.append(
        f"  budget: {p.budget} ({unit}){read}"
        + ("" if p.gguf else f"; head lease {'on' if p.head_lease else 'off'}")
    )
    lines.append(  # the search's deadline is the GPU-hours over the cards (`RunConfig.wall_s`)
        f"  budget_gpu_hours: {p.budget_gpu_hours:g} over {len(p.cards)} card(s): "
        f"the search stops {p.budget_gpu_hours / len(p.cards):g} h after it starts"
    )
    for i, step in enumerate(steps, 1):
        why = step.done()
        lines.append(
            f"  [{i}/{len(steps)}] {step.name}: " + (f"skip ({why})" if why else f"run -> {step.output}")
        )
        if not why:
            lines += [f"      $ {' '.join(command)}" for command in step.commands()]
        if not why and step.name == "ref" and not (p.model_dir / "config.json").is_file():
            from qda_kl.reference import DEFAULT_SAMPLING

            lines.append(
                f"      (with --sampling '{json.dumps(DEFAULT_SAMPLING)}' if its card and generation_config.json name none)"
            )
        if not why and step.name == "ref" and not p.holdout.is_file():
            lines.append(
                f"      (and --group for {p.holdout} once the model and corpus steps have run: "
                "one engine load, both)"
            )
    if p.gguf:
        try:
            llama_programs(p.cards)
        except Refusal as e:
            lines.append(f"  llama.cpp: the run will refuse until this is done: {e}")
    sizes = disk_estimate(p)
    lines.append("  disk (about): " + ", ".join(f"{name} {_gb(n)}" for name, n in sizes))
    known = [n for _, n in sizes if n is not None]
    floor = free_floor(JudgeSettings(), _model_bytes(p))  # the run's own default floor (`custody.free_floor`)
    lines.append(
        f"  total about {_gb(sum(known))}"
        + (" plus the unknown items" if len(known) < len(sizes) else "")
        + f", and the run stops builds below {_gb(floor)} free (judge.min_free_bytes): keep both free"
    )
    return "\n".join(lines) + "\n"


def execute(p: Prepare, steps: list[Step], log: Callable[[str], None] = print) -> int:
    """Run each step that is not done, in order, with a log line and its time; the first failure stops it."""
    if p.gguf and any(s.name in ("imatrix", "sensitivity", "floor", "config") for s in steps):
        llama_programs(p.cards)  # before any GPU hour: the refusal carries the build commands
        if any(s.name == "imatrix" for s in steps):
            imatrix_program(p.cards)  # llama-imatrix itself, the same refusal before the measurement
    p.out.mkdir(parents=True, exist_ok=True)
    require_room(p)  # before the first GPU step: a disk already below the run's floor refuses here
    if p.hub is not None and (record := {"spec": str(p.budget), **p.hub.record()}) != p.state.get(
        "budget_source"
    ):
        p.save(budget_source=record)  # a rerun prices this commit again, offline (`Prepare._read_hub`)
    if p.format_choice is not None and p.state.get("format_choice") != p.format_choice:
        p.save(format_choice=p.format_choice)  # the model step may be skipped: record the decision anyway
    if any(s.name == "ref" and not s.done() for s in steps):
        require_compiler()  # before the download: what the ref step's engine start refuses on
    for i, step in enumerate(steps, 1):
        head = f"prepare [{i}/{len(steps)}] {step.name}"
        if why := step.done():
            log(f"{head}: skip ({why})")
            continue
        log(f"{head}: -> {step.output}")
        started = time.monotonic()
        try:
            detail = step.run()
        except subprocess.CalledProcessError as e:
            if full := full_line([Place("--out", p.out, "the run's outputs")]):
                # a quota fills mid-write in the child, which prints its own traceback: name what is full
                raise Refusal(full, code=ENVIRONMENT) from None
            raise Refusal(
                f"{step.name}: `{' '.join(map(str, e.cmd))}` exited {e.returncode} (its output is above); "
                "fix it and run the same prepare again: finished steps are skipped"
            ) from None
        seconds = round(time.monotonic() - started, 1)
        p.save(steps={**p.state.get("steps", {}), step.name: {"seconds": seconds}})
        if detail:
            log(detail)
        log(f"{head}: done in {seconds} s")
    if line := serve_line(p.cand0):  # the same line `qda build` and the search's end report print
        log(f"serve candidate 0 with:\n  {line}")
    return 0


def add_parser(sub: Any) -> Any:
    """Add the `prepare` subcommand."""
    import argparse

    p = sub.add_parser("prepare", help="make every input of one search and its run config")
    p.add_argument("--model", required=True, help="a Hub id (REPO[@REVISION]) or a local model directory")
    p.add_argument("--out", required=True, type=Path, help="the directory every output goes into")
    p.add_argument(
        "--format",
        default="w4a16",
        help="the search's format preset (default w4a16), or a spelling that names a group size "
        "(w4a16-g64, w4a16-pc); a bare preset lowers its 128 only to a group that stores the model",
    )
    p.add_argument(
        "--cards",
        default="0",
        help="the GPUs, comma separated (default 0); the reference reads on one of them when the model fits "
        "one card, else on the most of them vLLM splits the model over",
    )
    p.add_argument("--budget-gpu-hours", type=float, default=4.0, help="the search's GPU budget (default 4)")
    p.add_argument(
        "--backend",
        default="claude",
        help="the agent backend (`qda_agent.backends`), or the mapping a config holds: "
        "'{name: codex, model: gpt-5.6-sol}' (default claude)",
    )
    p.add_argument(
        "--sign-in-dir",
        type=Path,
        help="the agent CLI's config dir that holds its sign-in (agent.backend.sign_in_dir; default ~/.claude)",
    )
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="GGUF only: measure the class sensitivity sweep and give candidate 0 its measured type map "
        "(off by default; a few GPU-hours)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and the disk estimate; run nothing")
    p.add_argument(
        "--budget-bpw",
        metavar="N|candidate0|hf:REPO[:FILE]",
        help="the size the search must stay within: N bits per quantized weight, candidate 0's own price, or a "
        "published checkpoint's (a GGUF file's bytes, or a compressed-tensors/GPTQ/AWQ checkpoint's bpw, read "
        "without downloading its weights); default "
        + ", ".join(f"{k} {v:g}" for k, v in budget_source.DEFAULT_BPW.items())
        + ", and candidate 0's size for gguf-q4km",
    )
    p.add_argument(
        "--head-lease",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="for a search that may also shrink the output head: the bits a narrower head saves are lent to the "
        "body under --budget-bpw (judge.head_lease); default on for w4a4, off otherwise; not for gguf-q4km",
    )
    p.add_argument(
        "--calib",
        type=Path,
        metavar="FILE",
        help="candidate 0's and the search's calibration file (JSONL), used as is instead of the public pool mix",
    )
    return p


def main(args: Any) -> int:
    """The `prepare` subcommand."""
    try:
        try:
            cards = cand0.parse_cards([args.cards])
        except ValueError as e:
            raise Refusal(str(e)) from None
        p = Prepare(
            args.model,
            args.out,
            format=args.format,
            cards=cards,
            budget_gpu_hours=args.budget_gpu_hours,
            budget_bpw=args.budget_bpw,
            head_lease=args.head_lease,
            sensitivity=args.sensitivity,
            calib_file=args.calib,
            backend=args.backend,
            sign_in_dir=args.sign_in_dir,
        )
        chosen = plan(p)
        if args.dry_run:
            sys.stdout.write(show_plan(p, chosen))
            return 0
        require_cache_room()  # before any output exists to be half written
        execute(p, chosen)
    except Refusal as e:
        return refuse(str(e), e.fix, code=e.code)
    except OSError as e:
        if e.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        # the parent itself hit a full disk or quota: name the place that is full, not a guess
        line = full_line([Place("--out", args.out, "the run's outputs")])
        return refuse(
            line or f"{e.filename or 'a cache'} is full; fix: free room or move the cache", code=ENVIRONMENT
        )
    if note := calib_thinking_note(p.calib_path):
        print(f"note: {note}.")
    if text := summary.card(p.cand0, original=p.model_dir, head_lease=bool(p.head_lease)):
        print(text)
    if p.config.is_file():
        print(f"next: qda-agent search {p.out}")
    return 0
