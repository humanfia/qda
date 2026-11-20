"""The run config's `judge` section: host scoring, generation checks and builds."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BeforeValidator,
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)
from qda_kl.datasets import KL_REFERENCE, PROMPTS, Dataset, parse_dataset, with_defaults
from qda_quant.gguf.llama_cpp import PINNED, commit_of
from qda_quant.output.vllm_env import DEFAULT_TP, cache_env, disabled_kernels
from qda_quant.target import GGUF_MENU

from qda_agent.commands import NAMES
from qda_agent.settings import Plugin, Strict, pick

from .checks import DEFAULT_MENU, HEAD_16BIT, UNSCORED, format_key
from .contamination import CHECKS
from .final_rules import FINAL_RULES, FinalRule, default_rules
from .freeze import model_size

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DRAIN_FLOOR_S",
    "LLAMACPP_DISK_CAP_FACTOR",
    "MIN_FREE_BYTES",
    "READING_START_S",
    "JudgeConfig",
    "JudgeSettings",
    "KLLayout",
    "Start",
]

#: The judge's cap when neither `max_copy_bytes` nor an anchor names one.
DEFAULT_MAX_BYTES = 200 << 30
#: The least drain window; the fixed default, so small-model runs drain as they always did.
DRAIN_FLOOR_S = 900.0
#: A reading's engine start that `reading_expected_s` does not count: process start, weight load,
#: torch.compile, CUDA-graph capture and warm-up. About 106 s on the 4B rows; 300 s leaves room on a cold
#: cache or a busy node.
READING_START_S = 300.0
#: The candidate `.gguf` files (`judge.budget_bytes`) an explicit `judge.max_run_bytes` on a llama.cpp search is
#: compared with at launch: below it, the preflight warns with how many builds fit. It is never a default cap.
LLAMACPP_DISK_CAP_FACTOR = 10
#: The least free space the run's disk keeps (`judge.min_free_bytes`) when no original model sizes it, in bytes
#: (decimal: 10 GB). It is measured live with `statvfs`, never estimated. A setting left at its default keeps
#: twice the original model's bf16 weights instead (`custody.free_floor`), whatever the disk's size.
MIN_FREE_BYTES = 10_000_000_000


def _unscored() -> tuple[str, ...]:
    return UNSCORED


def _final_rule(value: Any) -> Any:
    return pick(FINAL_RULES, value)


#: Float storage types a llama.cpp menu may name besides the target's (`qda_quant.target.GGUF_MENU`): a row may
#: admit a weight stored BF16 or F16 by listing it. No default menu lists them; F32 is not nameable (a tensor
#: is stored F32 only where candidate 0 stores it so, `checks.gguf.gguf_problem`), nor is any FP8 type.
GGUF_FLOAT_MENU = ("BF16", "F16")


class KLLayout(Strict):
    """The card layout one KL reading runs on.

    Attributes:
        tp: Tensor-parallel size; a `tensor_parallel_size` in `serve_args` is refused (tp names it).
        serve_args: The other `vllm.LLM` arguments that split the model, such as `pipeline_parallel_size`
            and `enable_expert_parallel`.
    """

    tp: int = Field(default=DEFAULT_TP, ge=1)
    serve_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_card_count(self) -> KLLayout:
        if "tensor_parallel_size" in self.serve_args:
            raise ValueError(
                f"serve_args names tensor_parallel_size={self.serve_args['tensor_parallel_size']}; "
                f"set kl_layout.tp={self.tp} instead"
            )
        p = self.serve_args.get("pipeline_parallel_size", 1)
        if not isinstance(p, int) or isinstance(p, bool) or p < 1:
            raise ValueError(f"serve_args.pipeline_parallel_size {p!r} is not a positive integer")
        return self

    @property
    def cards(self) -> int:
        """Cards one reading needs: `tp` times its pipeline stages."""
        return self.tp * int(self.serve_args.get("pipeline_parallel_size", 1))


class LlamaCpp(Strict):
    """The programs a llama.cpp search runs.

    Attributes:
        bridge: The compiled probe shared library the admission check preloads into `llama-server` to read
            the loaded tensors, as a file.
        llama_cpp: The llama.cpp checkout directory the checks run with: the pinned commit, with `gguf-py`
            beside `convert_hf_to_gguf.py`.
        server: The `llama-server` binary built from that checkout, which the admission check
            (`judge.llama_cmd`) starts. A CPU-only build serves the check.
        mtp: Whether a candidate `.gguf` carries a multi-token-prediction (MTP/NextN) head: `required`
            refuses a file without one, `forbidden` refuses a file with any, `any` (the default) does not
            decide. A row keeps one choice. The tensors are the pinned llama.cpp's own names
            (`nextn.pre_projection`, `nextn.post_projection`, `blk.N.nextn.*`,
            `src/llama-arch.cpp:728-738`); admission names the tensors and their count in the refusal.
        template: A bf16 GGUF of the run's model (`convert_hf_to_gguf.py --outtype bf16` at the pinned commit).
            Every GGUF build the judge admits reads it instead of converting the source again
            (`qda_agent.judge.gguf_template`). It is used as is, outside the run directory and its disk cap.
            If a `<file>.json` record sits beside it (`qda-agent prepare` writes one), the record must name this
            model and commit. None: the judge converts the source once at start into the run directory, where
            the cap counts it once.
        imatrix: A `llama-imatrix` GGUF of the run's model (`qda-agent prepare` measures it once per model,
            calibration file and llama.cpp commit). Every GGUF build the judge admits reads it as
            `exporter.gguf.imatrix`: it weights the fit of the tensors taken from the template. Used as is,
            outside the run directory and its disk cap; a recipe naming another one is refused. If a
            `<file>.json` record sits beside it, the record must name this model and commit. None: a build
            fits those tensors unweighted unless its own recipe supplies a matrix.
        lock: Tensors a candidate must store exactly as candidate 0 (`baselines[0]`) does: a list of regexes,
            each searched in the GGUF tensor name (`re.search`), e.g. ``["^blk\\.64\\."]`` for a row whose
            MTP block stays at candidate 0's types. A matching tensor of candidate 0 the candidate leaves out,
            or stores in another type, is refused at admission, on the menu or not. Empty (the default): no
            lock. Needs `baselines`.
    """

    bridge: Path
    llama_cpp: Path
    server: Path
    template: Path | None = None
    imatrix: Path | None = None
    mtp: Literal["required", "forbidden", "any"] = "any"
    lock: list[str] = Field(default_factory=list)

    @field_validator("lock")
    @classmethod
    def _lock_compiles(cls, v: list[str]) -> list[str]:
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"judge.llamacpp.lock: {pattern!r} is not a regex ({e})") from None
        return v


class Start(Strict):
    """One pre-built starting checkpoint of the row (a candidate 0).

    Attributes:
        name: The build method TASK.md and the manifest show the agent (`gptq`, `awq`, `gsq`, `rtn`): one of
            `qda_quant.steps.BUILTINS`, never a path, a release name or a publisher's name.
        path: The checkpoint. It is one of `baselines` (the agent's jobs read it there) and lies
            outside the run dir, where every baseline lives.
        origin: `own` for a checkpoint this work built, `gsq` for the upstream GSQ build (W3A16 rows only).
    """

    name: str
    path: Path
    origin: Literal["own", "gsq"]

    @field_validator("name")
    @classmethod
    def _label(cls, value: str) -> str:
        """A name is a build method qda_quant knows, never a release, a vendor or a publisher's name."""
        from qda_quant.steps import BUILTINS

        if value not in BUILTINS:
            raise ValueError(
                "a start name is a build method qda_quant knows "
                f"({', '.join(sorted(BUILTINS))}): {value!r} is not one; never a release or a publisher's name"
            )
        return value


class JudgeSettings(Strict):
    """Configure host scoring, generation checks and builds in the run's `judge:` section.

    Command templates are argv lists with these slots: `{python}` the serving interpreter, `{ckpt}` the
    frozen checkpoint, `{out}` the JSON file the command must write, `{ref}` the KL reference, `{config}` a
    `qda-kl score` config the judge writes into its own dir for this reading (`checkpoint`, `reference`, `out`;
    the reading is then read from `<out>/score.json`), `{eval}` the read-only evaluator package copy (also
    first on `PYTHONPATH`), `{tools}` the read-only copy of the judge's own scripts and `{suite}` the JSON of
    `suite`. The KL command writes
    a `qda_kl` reading or, for another suite, a `SuiteResult` (`{tools}/suite_job.py` runs a suite by name) -- the
    agent sees only the metrics, `domains` and `domains_text`; `per_seq`, `flip_rate`, token counts and the rest
    of a suite's `detail` stay host-side. A nonzero exit is no score (a refusal exits 2 and writes no file). The
    check command writes `{"ok": bool, "reason": str}`.

    The `{cards}` command slot is the scheduled job's card count (for example `serve_check.py --tp {cards}`).

    Attributes:
        anchor: The original model; when set, tokenizer files, chat template and generation config of a
            checkpoint must equal its own byte for byte.
        llama_cmd: The admission check, with `{ckpt}`, `{out}`, `{python}`, `{tools}`; it writes
            `{"ok": bool, "reason": str}`. The default `judge/tools/gguf_check.py` starts the pinned server,
            strict-checks its load through `judge.llamacpp` (its `gguf-py` checkout and bridge), and
            completes once.
        serve_python: The interpreter of the serving venv (vLLM + qda_kl's dependencies); None: the host's.
        job_env: Extra environment of the judge's jobs (e.g. `PATH`, `CUDA_HOME` of the serving venv). A
            `PYTHONPATH` here comes after the evaluator copy (`job_environ`), e.g. qda-quant's sources for a
            serving venv without qda-quant (`qda-kl score CONFIG.yaml` imports `qda_quant.output.record`).
        metric: The metric the search ranks readings by, a key of the score command's `metrics` (the `kl` suite's
            `kl` by default). Its direction comes from the reading's `higher_is_better`.
        max_kl: A reading whose `kl` exceeds this is refused at once, and every entry of the same build's file
            set becomes ineligible: the checkpoint is broken, not merely worse. 1.0: a KL of 1 is the
            reference's own entropy scale, so a reading over it means a rotation/basis error, not a bad search.
        final_rules: The rules that decide the checkpoint a search ends on (`qda.final_rules`), in the order they
            are tried: `KEY` or `{name: KEY, ...settings}`. The task statement tells the agent this order. A run
            that names none uses `handin`: the result is the first valid entry of the agent's latest hand-in
            list, and with no valid hand-in candidate 0 as-is; `FINAL.json` records the path as `chosen_by`.
        eval_max_tokens: The final evaluation's generation cap for this row, in tokens; TASK.md's long-generation
            paragraph states it (the long-generation paragraph). A run whose statement renders that
            paragraph must set it: the run is refused when TASK.md is rendered, before the agent starts.
        kl_noise: A relative tie band for this row's KL readings (`0.02` = 2 %): TASK.md states
            "Two readings within 2 % are a tie." None: no tie sentence is stated. Only a measured figure belongs
            here; no number is invented.
        suite: The eval suite a reading runs, one `{key: settings}` of a `qda eval` config (`qda.eval_suites`, so
            its package must be installed on the host too): `kl_cmd` takes it as `{suite}`, and a reading takes
            the cards it declares. None: `kl_cmd` names what it measures, on `kl_layout`'s cards.
        kl_ref: The KL reference of the original model (`qda-kl ref`), or the reference a suite job reads: a path,
            or one `{key: settings}` from `qda.datasets`; never visible to the agent.
        kl_cmd: The scoring command (`qda-kl score CONFIG.yaml`), with `{moe}` standing for `kl_moe_args` on
            a checkpoint with routed experts (dropped on a dense one).
        kl_moe_args: Extra scorer arguments for a checkpoint with routed experts: their scores are not
            bit-identical across repeats on vLLM 0.33, so the judge reads the mean of three (`kl` is the
            mean; `repeats`, `spread`, `deterministic` stay judge-side).
        kl_moe_factor: The scoring part of a reading's expected time is multiplied by this for such a
            checkpoint (the engine loads it once; `kl_moe_args` reads it three times).
        kl_layout: The card layout of a reading: `tp` and the other `vllm.LLM` split arguments. The cards a
            reading takes are derived from it (`tp` times `pipeline_parallel_size`); a reading whose command
            would run its scorer with another `tp` or `serve_args` is refused when it is queued.
        kl_load_s_per_gib: Seconds one card takes to load a gibibyte of a checkpoint; with `kl_score_s` it
            gives a reading's expected duration (admission only).
        kl_score_s: The fixed scoring time of one reading (admission only), on vLLM.
        reading_s: A whole reading's seconds as `prepare` measured them on this host and backend (its `floor`
            step decodes the bf16 GGUF and scores it once, for the engine floor). It replaces the formula in
            `reading_expected_s`; None (a vLLM search, which prepare does not measure) keeps the formula.
        measured_cards: The cards `reading_s` was measured on, so a reading on another count moves only its load.
        gencheck_cmd: The long-generation check (`qda-kl gencheck CONFIG.yaml`); `{config}` is a gencheck config
            the judge writes for this check, and the result is read from `<out>/gencheck.json`.
        gencheck_cards: Cards per long-generation check (its tensor parallel size).
        gencheck_expected_s: Expected duration of a long-generation check at `gencheck_max_tokens` and
            `gencheck_per_domain`, both models included (admission only; a check the agent asks longer or wider
            costs in proportion to the answers it generates; the original model's answers are generated once
            per run and reused).
        gencheck_max_tokens: The longest answer, in tokens (8192: minutes per check; 32768 is the evaluation's
            length); the default of a check the agent does not size itself (`gencheck DIR --max-tokens N`).
        gencheck_per_domain: Prompts per domain (the first N of each domain of the prompt set; 16 for the
            evaluation's count); the default of a check the agent does not size itself (`--per-domain K`).
        gencheck_prompts: A JSONL prompt set (`{id, domain, prompt}` rows), a path or one `{key: settings}` from
            `qda.datasets`, copied into the judge's dir at start; None: the evaluator's built-in set, which is
            public (the agent's jobs can read it). Unless the dataset declares itself public, keep the source
            outside what the agent's shell sees (under the home or `sandbox.hide`) and outside every path its
            jobs read (`isolation.check` refuses it otherwise).
        check_cmd: The serving check of a submission (stock vLLM load + bounded generation,
            `judge/tools/serve_check.py`); None skips it.
        check_cards: Cards per check.
        check_expected_s: Expected duration of a check.
        build_expected_s: Expected duration of a `qda build` that does not state one.
        require_build_receipt: Mark a checkpoint whose files are no trusted `qda build` job's recorded
            outputs ineligible (default false): it is still scored and reported, but no final rule may pick
            it. Receipts bind the whole checkpoint and its build metadata before the agent receives a copy.
        build_receipt_smoke: Before the run opens, build one tiny synthetic model through the same trusted
            build verb a candidate build takes (the run's `qda_cmd`/`qda_env`, bwrap, the trusted builder),
            in the run's format, and refuse the start unless it comes back with a receipt and a build
            record (default true). A build environment missing a package the build path imports would
            otherwise fail every build from launch, with nothing eligible. The smoke directory is outside
            the run's custody and ledger and is removed either way. Applies only when
            `require_build_receipt` is on (and not to a llama.cpp search, whose menu names no uniform
            format); set false to open the run without the check.
        candidate0_smoke: The run's trial run, minutes before the agent's first turn: build the run's own
            model (`model`) once, uniform `rtn` in the run's own `format`, as a queued trusted build on one
            card; run the judge's own admission gate on it (the one a hand-in passes, without the bpw budget,
            which judges an allocation); then start an engine on it with `check_cmd`, as a hand-in is checked.
            Any failure ends the run with candidate 0 and `FINAL.json`'s reason (default true). It is what
            proves this machine can build and serve at all, instead of every candidate failing after the run
            has been spent. Its request, custody and output are removed; it is no candidate and no ledger
            line. It costs one build and one engine start of the model, within the run's deadline. A CPU run,
            a test's fake GPUs and a run naming no model do no trial run.
        reading_smoke: Before the run opens (and with `require_build_receipt`), read candidate 0
            (`baselines[0]`) once through the run's own `kl_cmd`, `kl_layout` and `kl_ref`, and fold the
            result through the judge's own completeness rules; refuse the start unless it is a valid,
            complete reading (default true). A scorer that measures only a subset of the reference, writes
            no per-sequence readings, or cannot start would otherwise refuse every candidate's reading after
            the agent had spent the run. The reading is candidate 0's, which a search needs anyway; it is
            recorded as no candidate and in no ledger line, and its job's result files are removed (nothing
            the check did not create is deleted). It costs one reading's load and scoring time, charged
            within the run's deadline. A run with `select_candidate0` reads its baselines at start through
            the same fold (and refuses an invalid one), so it skips the check; a run with no `baselines[0]`
            is refused, naming the fix, because the reading could not be made at all. Set false to open the
            run without it.
        custody_dir: Fresh host-owned per-run checkpoint directory on node-local storage. None keeps
            checkpoints under `run_dir/judge`. Requires bwrap for the agent and its jobs; records and the
            exported winner stay in `run_dir`. Node loss before export loses candidate weights.
        max_run_bytes: The largest byte count the run and custody directories may hold: published custody, the frozen
            hand-in copies, the judge's own files and the agent's work/ (its local scheduler builds
            included), each inode once (hardlinks count once; the explicit custody root is walked even
            when linked from the run). Off by default (None: no cap, custody or not, and `qda-agent prepare`
            writes none): the run is then guarded by `min_free_bytes` alone. A value set by hand is kept as
            given, and on a llama.cpp row the preflight warns when it is below 10 x `budget_bytes`, with how
            many builds fit. A new
            trusted build, agent job or hand-in copy is refused once the on-disk count plus the pending
            jobs' reservations would pass it, with a `disk budget:` message stating each part in decimal GB
            (1e9) and naming the published builds no prune pass frees (never scored) with their sizes;
            no running job is stopped. None: no cap. A checkpoint written by a job that did not
            declare it can overshoot by at most the concurrent jobs' writes before the next refusal.
        min_free_bytes: The live free-space floor (default: twice the bf16 weights of `judge.anchor`, else
            `MIN_FREE_BYTES` (10 GB), `custody.free_floor`; a value set by hand is used as is; 0 turns it off): the
            free bytes `room()` (one `statvfs`) reads on the disks the run's builds land on (the custody root,
            else `run_dir`, and the agent's workdir). The default holds one build being written and one being
            read on any disk, small or shared. Below it the start is refused
            (`custody.check_space`, which `qda doctor --search` reports), a new trusted build, agent job or
            hand-in copy is refused (`prune.check_room`), and each scheduler pass that finds it stops the
            agent's running jobs and records the stop, so the `handin` rule submits the best scored eligible
            entry (`prune.watch_floor`, said in two `DISK FLOOR` lines on the terminal). It is the only disk
            guard when `max_run_bytes` is unset.
        job_reserve_bytes: What an agent job that does not declare an output checkpoint reserves against
            `max_run_bytes` (default 2 GB). A trusted build reserves its plan's output estimate; an agent
            job that declares one (`qda build`, or `qsched add --checkpoint DIR`) reserves the original's
            quantized-size estimate.
        max_copy_bytes: Largest checkpoint the judge copies. None: twice the anchor's size + 1 GiB, or 200 GiB
            without an anchor. An explicit value replaces that bound (an anchor with no files, or with a link
            that is broken or escapes it, is refused at start; a Hugging Face hub snapshot's links into its
            blobs count as files).
        max_files: Most entries (files and directories) a checkpoint may hold (the judge's listing cap).
        budget_bpw: The largest bits per quantized weight the task admits: the decoder Linears' packed
            codes, scales and zero points, and 16 bits a weight for a Linear kept bf16, over the fixed
            count of decoder-Linear weights llm-compressor's uniform default quantizes (the anchor's,
            never the candidate's; `judge.checks.budget`). The output head is not in it (`head_lease`). A
            candidate over it is refused at admission, and `baselines[0]` (candidate 0) over it is refused
            at start. None: no budget. A vLLM search's budget; a llama.cpp search states bytes instead
            (`engine`).
        head_lease: For a search that may also shrink the output head. Off (the default): the head stays
            16-bit and costs nothing. On: the head may be stored in a narrower format its row offers
            (`checks.budget.HEAD_LEASE`: bf16, w8a16 or w4a16 beside a W4A16 or W3A16 body, bf16 or w8a16
            beside W4A8, bf16 or FP8 beside W4A4), it is charged 16 bits a weight, and the bits it saves are lent to the body under
            `budget_bpw`. A vLLM search with `budget_bpw` and the 16-bit `head` rule only.
        engine: The serving engine a search's checkpoints are made for: `vllm` (compressed-tensors, the
            default) or `llama.cpp` (one `.gguf` file). llama.cpp prices the file by its weight bytes,
            not bits per weight: it needs `budget_bytes` and `engine_floor`, and refuses `budget_bpw` and a
            final rule that runs a suite, since no suite has a llama.cpp engine.
        budget_bytes: The most bytes a candidate `.gguf` may cost (the llama.cpp search's budget). Its basis
            is the whole file by default (`budget_basis: file`): `os.stat` of the frozen `.gguf` the scorer
            reads, header and padding included. Needs `engine: llama.cpp`; a vLLM search states
            `budget_bpw`.
        budget_basis: Which `.gguf` figure `budget_bytes` screens: `file` (the default; the whole file the
            vendor ships and the scorer reads) or `payload` (only the stored tensor bytes, without the
            header: an explicit opt-in). The payload figure is always recorded beside the file figure.
        engine_floor: The answer KL the bf16 GGUF reads (decoded, read with vLLM) against the sealed bf16
            reference: the reading's own noise, part of every candidate's KL, so the floor sits beside a
            reading, not under it. It is stated in FINAL.json and TASK.md; a candidate's bar is
            `candidate0_kl`, which is separate. Needs `engine: llama.cpp`; a vLLM search has no engine
            floor.
        llamacpp: The programs a llama.cpp search runs: `bridge` (the compiled probe shared library, a file),
            `llama_cpp` (the checkout directory; the pinned commit, with `gguf-py` beside
            `convert_hf_to_gguf.py`) and `server` (the `llama-server` built from it, which the admission check
            starts), and optionally `template` (a bf16 GGUF of the model that every GGUF
            build reads). Required by `engine: llama.cpp`, refused at load when a path
            is missing, the bridge or server is not a file, the checkout is not the pinned commit, or `gguf-py`
            is absent. A vLLM search states none.
        format_reference: An optional checkpoint directory of the task's format (a public baseline). It must
            itself be in `format_menu` and keep the engine's fused-unit facts at host start. A judged
            checkpoint's format is the menu's and the engine's facts', not the reference's tensor inventory
            or quantization config, so a search does not need one; when given it is also a baseline the
            near-copy identity check compares against.
        format_map: A checkpoint directory whose per-module stored formats are the task's FIXED allocation
            (a paper checkpoint: `config.json` with its `quantization_config`, and its safetensors), or None
            (the menu alone gates formats). Every module of a judged checkpoint must be stored exactly as
            there, module by module (`checks.stored_formats`: the effective format each module's tensors
            and config give it, so `ignore` lists that spell one allocation differently agree), lm_head
            included: a quantized head of the map is read back from the candidate's tensors, a float one
            keeps the bf16 rule. The first module that differs is named. At host start the map must be
            readable, admitted by itself, and `format_menu` must hold exactly its quantized formats outside
            lm_head (`checks.map_menu_problem`). The judge reads it; keep it out of the agent's view
            (`sandbox.hide`). A vLLM search only; not with `allow_any_format`.
        allow_any_format: Skip the format gate (the menu and the engine's facts) for development stand-ins;
            a judged search leaves it false. A search no longer needs a format reference either way.
        format_menu: The formats the task admits, as names: `bf16`, `w8a16` and `w4a16` for a W4 search;
            a W3 search defaults to `bf16`, `w3a16`, `w4a16` and `w8a16`. Format.parse reads each spelling. Bare
            names inherit the run format's fixed group; an explicit group must agree. Every quantized
            module's family and group must be one of them (`checks.format_problem`); the statement gives
            the agent this menu.
            A llama.cpp search names GGUF types instead, `Q4_K` and `Q6_K`, and only the types the
            llama.cpp serves (`qda_quant.target.GGUF_MENU`) are admitted; the default is those, including
            template-fitted Q8_0 and IQ4_XS. These need no solved-format override. Such a menu may also name
            the float storage types `BF16` and `F16` (`GGUF_FLOAT_MENU`), which no default lists. Every tensor
            of a hand-in is stored as a menu type, except where candidate 0 stores that tensor off the menu
            (its F32 norms, say): there it keeps candidate 0's type.
        head: The output head's storage rule, owned here for the task statement and the judge alike:
            `16-bit` (the default) keeps lm_head — `output.weight` in a GGUF — in a 16-bit float type (BF16
            or F16), independent of the menu, and the judge admits such a head even where no menu entry names
            it. An exception row, whose source model was released with a quantized head, names that release's
            rule instead: a GGUF type (`Q8_0`) under `llama.cpp`, which `format_menu` must admit, or a format
            name (`w4a16-sym-g64`, or the FP8 head `w8a8-fp8`) on vLLM; the head must then be stored exactly
            so. TASK.md's lm_head
            sentence is rendered from this one value (`prompt._head_line`), so the agent's instructions and
            the check cannot disagree. `qda-agent prepare` writes it when its source's own config or GGUF
            header stores a quantized head; every other row keeps the default.
        unscored: Tensor-name globs no KL reading scores, left out of the baseline fingerprint.
        evaluator: The evaluator package tree, copied read-only to `<run_dir>/judge/eval/` at start; KL and
            check jobs import it from there. None: the installed `qda_kl` package.
        candidate0_kl: Candidate 0's KL, read by the same scorer and reference as the candidates'.
        select_candidate0: Read and admit every baseline at host start, then use the lowest-KL one as
            candidate 0. Default off; cannot be combined with an explicit `candidate0_kl`.
        starts: The row's pre-built starts, in menu order (`{name, path, origin}`): each is a candidate 0 and
            the lowest valid KL is the bar every candidate is compared with. Each must label one of
            `baselines` (the jobs read it there): a row gives every baseline a start, or leaves
            `starts` empty and keeps today's unlabelled baselines (candidate 0 = `baselines[0]`). With starts
            the host reads every one at start, so `candidate0_kl` is read, never configured.
        candidate0_build_s: Candidate 0's build time in seconds, for the run's report
            (`time_ratio = search wall time / candidate0_build_s`).
        full_card_moe: Serialize trusted builds and KL scores over eight cards, using an FP8 original.
    """

    anchor: Path | None = None
    llama_cmd: list[str] = Field(
        default_factory=lambda: ["{python}", "{tools}/gguf_check.py", "{ckpt}", "{out}"]
    )
    serve_python: Path | None = None
    job_env: dict[str, str] = Field(default_factory=dict)
    metric: str = Field(default="kl", min_length=1)
    max_kl: float = Field(default=1.0, gt=0)
    final_rules: list[Annotated[SerializeAsAny[FinalRule], BeforeValidator(_final_rule)]] = Field(
        default_factory=default_rules, min_length=1
    )
    #: The final evaluation's generation cap for this row, in tokens: TASK.md's long-generation paragraph
    #: states it (the long-generation paragraph). Required when that paragraph is rendered.
    eval_max_tokens: int | None = Field(default=None, gt=0)
    #: A relative tie band for KL readings of this row (0.02 = 2 %); None: TASK.md states no tie band.
    #: Only a measured figure belongs here.
    kl_noise: float | None = Field(default=None, gt=0, lt=1)
    suite: dict[str, Any] | None = None
    kl_ref: Path | dict[str, Any] | None = None
    #: The prompt set held out from `kl_ref`, named explicitly (a ``qda-kl ref`` directory or a prompt JSONL).
    #: The builds' calibration filter matches against the UNION of `kl_ref`'s eval set and this set; a search
    #: never starts without it, and a reference record that declares a different holdout refuses the start.
    kl_holdout_ref: Path | None = None
    kl_cmd: list[str] = Field(
        default_factory=lambda: [
            "{python}",
            "-m",
            "qda_kl.cli",
            "score",
            "{config}",
            "{moe}",
        ]
    )
    kl_moe_args: list[str] = Field(
        default_factory=lambda: [
            "--set",
            "scorer.topk.repeat=3",
            "--set",
            "scorer.topk.allow_nondeterministic=true",
        ]
    )
    kl_moe_factor: float = Field(default=3.0, ge=1)
    kl_layout: KLLayout = Field(default_factory=KLLayout)
    kl_load_s_per_gib: float = Field(default=7.5, ge=0)
    kl_score_s: float = Field(default=60.0, gt=0)
    reading_s: float | None = Field(default=None, gt=0)
    measured_cards: int | None = Field(default=None, ge=1)
    gencheck_cmd: list[str] = Field(
        default_factory=lambda: ["{python}", "-m", "qda_kl.cli", "gencheck", "{config}"]
    )
    gencheck_cards: int = Field(default=1, ge=1)
    gencheck_expected_s: float = Field(default=600.0, gt=0)
    gencheck_max_tokens: int = Field(default=8192, ge=1)
    gencheck_per_domain: int = Field(default=4, ge=1)
    gencheck_prompts: Path | dict[str, Any] | None = None
    check_cmd: list[str] | None = Field(
        default_factory=lambda: ["{python}", "{tools}/serve_check.py", "{ckpt}", "{out}"]
    )
    check_cards: int = Field(default=1, ge=1)
    check_expected_s: float = Field(default=300.0, gt=0)
    build_expected_s: float = Field(default=1800.0, gt=0)
    require_build_receipt: bool = False
    build_receipt_smoke: bool = True
    candidate0_smoke: bool = True
    reading_smoke: bool = True
    custody_dir: Path | None = None
    max_run_bytes: int | None = Field(default=None, gt=0)
    min_free_bytes: int = Field(default=MIN_FREE_BYTES, ge=0)
    #: Separate cap for run_dir/workdir (jobs, records and durable export included). None: only the total
    #: cap applies. Local trusted builds reserve job_reserve_bytes here for records/logs, not their weights;
    #: so do agent jobs when the run's `work_dir` puts the workdir outside run_dir (it is not counted here).
    max_scratch_bytes: int | None = Field(default=None, gt=0)
    job_reserve_bytes: int = Field(default=2_000_000_000, gt=0)
    max_copy_bytes: int | None = Field(default=None, gt=0)
    max_files: int = Field(default=4096, gt=0)
    engine: Literal["vllm", "llama.cpp"] = "vllm"
    budget_bpw: float | None = Field(default=None, gt=0)
    head_lease: bool = False
    budget_bytes: int | None = Field(default=None, gt=0)
    budget_basis: Literal["file", "payload"] = "file"
    engine_floor: float | None = Field(default=None, ge=0)
    llamacpp: LlamaCpp | None = None
    format_reference: Path | None = None
    format_map: Path | None = Field(default=None, exclude_if=lambda value: value is None)
    allow_any_format: bool = False
    format_menu: list[str] = Field(default_factory=lambda: list(DEFAULT_MENU))
    #: The output head's rule (`HEAD_16BIT`, a GGUF type under llama.cpp, or a format name on vLLM); it must
    #: be spelled after `engine` and `format_menu`, whose validators it reads.
    head: str = HEAD_16BIT
    unscored: list[str] = Field(default_factory=lambda: list(_unscored()))
    evaluator: Path | None = None
    candidate0_kl: float | None = Field(default=None, ge=0)
    select_candidate0: bool = False
    starts: list[Start] = Field(default_factory=list)
    candidate0_build_s: float | None = Field(default=None, gt=0)
    full_card_moe: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def _candidate0_source(self) -> JudgeSettings:
        """The candidate-0 reading comes from the scorer or the config, never both."""
        if self.select_candidate0 and self.candidate0_kl is not None:
            raise ValueError("judge.select_candidate0 cannot be combined with judge.candidate0_kl")
        return self

    @model_validator(mode="after")
    def _start_labels(self) -> JudgeSettings:
        """A labelled start reads candidate 0 at start, so it cannot also be configured."""
        if self.starts and self.candidate0_kl is not None:
            raise ValueError(
                "judge.starts reads candidate 0 at start: drop judge.candidate0_kl (the bar is the lowest "
                "valid KL of the starts)"
            )
        return self

    def disk_cap_line(self) -> str | None:
        """The launch preflight's disk-cap line: the cap, the rule that set it, and any warning; None: no cap.

        A llama.cpp search that converts its bf16 template into the run directory holds it once, inside the
        cap (`gguf_template.run_template_bytes`). The line says so, and counts the builds that fit beside it.
        """
        if self.max_run_bytes is None:
            return None
        from .gguf_template import run_template_bytes

        cap = self.max_run_bytes / 1e9
        template = run_template_bytes(self)
        held = (
            f"; the run's bf16 GGUF template holds about {template / 1e9:.1f} GB of it, once"
            if template
            else ""
        )
        line = f"disk cap: {cap:.1f} GB (judge.max_run_bytes, explicit){held}"
        if self.budget_bytes is not None and (
            self.max_run_bytes < LLAMACPP_DISK_CAP_FACTOR * self.budget_bytes or template
        ):
            budget = self.budget_bytes / 1e9
            rule = LLAMACPP_DISK_CAP_FACTOR * self.budget_bytes / 1e9
            fits = max(0, self.max_run_bytes - template) // self.budget_bytes
            beside = " beside the template" if template else ""
            if self.max_run_bytes < LLAMACPP_DISK_CAP_FACTOR * self.budget_bytes:
                line += f" WARNING: below {LLAMACPP_DISK_CAP_FACTOR} x budget_bytes ({rule:.1f} GB);"
            else:
                line += ";"
            line += f" about {fits} builds of {budget:.1f} GB fit{beside}"
        return line

    @model_validator(mode="before")
    @classmethod
    def _llama_menu(cls, data: Any) -> Any:
        """A llama.cpp search that names no menu admits the types the llama.cpp target declares.

        The default vLLM menu (`bf16`, `w8a16`, `w4a16`) names no GGUF type, so it cannot stand for one. An
        explicit empty menu is the task's own statement (no GGUF type admitted) and is held as it is.
        """
        if isinstance(data, dict) and data.get("engine") == "llama.cpp" and data.get("format_menu") is None:
            return {**data, "format_menu": list(GGUF_MENU)}
        return data

    @field_validator("engine_floor")
    @classmethod
    def _finite_floor(cls, value: float | None) -> float | None:
        """Refuse a floor that is not a finite number: inf or NaN serializes to no value at all."""
        if value is not None and not math.isfinite(value):
            raise ValueError(
                f"judge.engine_floor: {value!r} is not a finite KL; JSON would store null and a replay "
                "would refuse the floor as missing"
            )
        return value

    @field_validator("format_menu")
    @classmethod
    def _menu(cls, value: list[str], info: ValidationInfo) -> list[str]:
        """Refuse a menu entry that names no format; preserve its group spelling, in order.

        Under `engine: llama.cpp` the entries are served GGUF types, including native template fits,
        held as the menu spells them. A menu type need not have a solved grid.
        """
        if info.data.get("engine", "vllm") == "llama.cpp":
            named = (*GGUF_MENU, *GGUF_FLOAT_MENU)
            types = {name.lower(): name for name in named}
            menu = []
            for name in value:
                if (spelled := types.get(name.strip().lower())) is None:
                    raise ValueError(
                        f"judge.format_menu: {name!r} is no GGUF type the llama.cpp target declares "
                        f"({', '.join(named)})"
                    )
                menu.append(spelled)
            return list(dict.fromkeys(menu))
        for name in value:
            format_key(name)
        return list(dict.fromkeys(name.strip().lower() for name in value))

    @field_validator("head")
    @classmethod
    def _head(cls, value: str, info: ValidationInfo) -> str:
        """Hold the head rule to one thing: the 16-bit default, a GGUF type, or a weight-only format name.

        Under `llama.cpp` an exception's type must be one the target declares and one `judge.format_menu`
        admits: the head is then read like any stored tensor (`checks.gguf`). On vLLM the exception is a
        `Format` spelling (`w4a16-sym-g64`), read back from the head's tensors (`checks.formats`).
        """
        value = value.strip()
        if value.lower() == HEAD_16BIT:
            return HEAD_16BIT
        if info.data.get("engine", "vllm") == "llama.cpp":
            named = {name.lower(): name for name in (*GGUF_MENU, *GGUF_FLOAT_MENU)}
            if (spelled := named.get(value.lower())) is None:
                raise ValueError(
                    f"judge.head: {value!r} is neither {HEAD_16BIT} nor a GGUF type the llama.cpp target "
                    f"declares ({', '.join(named.values())})"
                )
            menu = info.data.get("format_menu")
            if menu is not None and spelled not in menu:
                raise ValueError(
                    f"judge.head: {spelled} is not in judge.format_menu ({', '.join(menu)}): an exception "
                    "model's head keeps a type its own menu admits"
                )
            return spelled
        try:
            format_key(value)
        except ValueError as e:
            raise ValueError(
                f"judge.head: {value!r} is neither {HEAD_16BIT} nor a format name ({e})"
            ) from None
        return value.lower()

    @field_validator("suite")
    @classmethod
    def _recorded_suite(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Hold the suite with every default filled in, nested plugins included, for the record and the job."""
        if value is None:
            return None
        from qda_kl.eval.suites import parse_suite  # the evaluator's own parser: one reading of a suite

        return parse_suite(value, "judge.suite").to_dict()

    @field_validator("kl_ref", "gencheck_prompts")
    @classmethod
    def _dataset(
        cls, value: Path | dict[str, Any] | None, info: ValidationInfo
    ) -> Path | dict[str, Any] | None:
        """Refuse a dataset that is unknown, misconfigured or of another use; record its defaults."""
        if value is None:
            return None
        use, where = (KL_REFERENCE if info.field_name == "kl_ref" else PROMPTS), f"judge.{info.field_name}"
        dataset = parse_dataset(value, where, use)
        if use == KL_REFERENCE and dataset.public:
            raise ValueError(
                f"judge.kl_ref.{dataset.name}: a public dataset, and the agent may never read a KL reference"
            )
        return with_defaults(value, where, use)

    @model_validator(mode="before")
    @classmethod
    def _renamed_keys(cls, data: Any) -> Any:
        """Name the keys that replaced the one-card KL settings, instead of only refusing them."""
        if isinstance(data, dict):
            if "kl_cards" in data:
                raise ValueError(
                    "kl_cards was replaced by kl_layout: the cards are derived from it "
                    "(kl_layout.tp times its pipeline_parallel_size)"
                )
            if "kl_expected_s" in data:
                raise ValueError(
                    "kl_expected_s was replaced by kl_load_s_per_gib and kl_score_s "
                    "(one reading = load per GiB times the checkpoint's GiB over its cards, plus the score)"
                )
        return data

    @model_validator(mode="after")
    def _engine_budget(self) -> JudgeSettings:
        """One budget per engine, both stated: bits per weight on vLLM, bytes and a floor on llama.cpp."""
        if self.engine == "llama.cpp":
            if self.budget_bpw is not None:
                raise ValueError(
                    "judge.engine: llama.cpp admits a .gguf by its weight bytes, not bits per weight: "
                    "set judge.budget_bytes and drop judge.budget_bpw"
                )
            if self.budget_bytes is None:
                raise ValueError(
                    "judge.engine: llama.cpp needs judge.budget_bytes: the most bytes a candidate .gguf "
                    "may cost (the whole file, unless judge.budget_basis: payload)"
                )
            if self.engine_floor is None:
                raise ValueError(
                    "judge.engine: llama.cpp needs judge.engine_floor: the answer KL the bf16 GGUF reads "
                    "against the sealed bf16 reference (qda-agent prepare's floor step measures it)"
                )
            if self.llamacpp is None:
                raise ValueError(
                    "judge.engine: llama.cpp needs judge.llamacpp.bridge and judge.llamacpp.llama_cpp: the "
                    "probe shared library the admission check loads and the pinned checkout it was built against"
                )
            self._check_llamacpp()
        else:
            if self.budget_bytes is not None:
                raise ValueError(
                    "judge.budget_bytes needs judge.engine: llama.cpp; a vLLM search's budget is "
                    "judge.budget_bpw"
                )
            if self.budget_basis != "file":
                raise ValueError(
                    "judge.budget_basis needs judge.engine: llama.cpp; a vLLM search's budget is "
                    "judge.budget_bpw"
                )
            if self.engine_floor is not None:
                raise ValueError(
                    "judge.engine_floor needs judge.engine: llama.cpp, whose scorer compares it; a vLLM "
                    "search has no engine floor"
                )
            if self.llamacpp is not None:
                raise ValueError(
                    "judge.llamacpp needs judge.engine: llama.cpp, whose admission check runs through its bridge"
                )
        if self.engine == "llama.cpp":
            for rule in self.final_rules:
                if rule.runs_suite:
                    raise ValueError(
                        f"judge.final_rules: {rule.name} runs a sanity suite, and judge.engine is llama.cpp: "
                        "no suite has a llama.cpp engine, so its final check would answer the .gguf with vLLM"
                    )
        return self

    def _check_llamacpp(self) -> None:
        """Refuse a bridge, server or checkout a llama.cpp run could not start, at config load.

        Raises:
            ValueError: the bridge, server or template is not a file, the checkout is not a directory, it has
                no `gguf-py`, its commit cannot be read, or it is not the pinned commit.
        """
        assert self.llamacpp is not None
        bridge, root = self.llamacpp.bridge, self.llamacpp.llama_cpp
        if not bridge.is_file():
            raise ValueError(
                f"judge.llamacpp.bridge: {bridge} is not a file (the compiled probe shared library)"
            )
        if not self.llamacpp.server.is_file():
            raise ValueError(
                f"judge.llamacpp.server: {self.llamacpp.server} is not a file (the llama-server binary built "
                "from that checkout)"
            )
        if not root.is_dir():
            raise ValueError(f"judge.llamacpp.llama_cpp: {root} is not a directory (a llama.cpp checkout)")
        if not (root / "gguf-py").is_dir():
            raise ValueError(f"judge.llamacpp.llama_cpp: {root} has no gguf-py (not a llama.cpp checkout)")
        try:
            commit = commit_of(root)
        except Exception as e:  # qda_quant.step.Refused: not a checkout, or git cannot read its history
            raise ValueError(f"judge.llamacpp.llama_cpp: {e}") from None
        if commit != PINNED:
            raise ValueError(
                f"judge.llamacpp.llama_cpp: {root} is at {commit}, not the pinned llama.cpp {PINNED}"
            )
        template = self.llamacpp.template
        if template is not None and not template.is_file():
            raise ValueError(
                f"judge.llamacpp.template: {template} is not a file (a bf16 GGUF of the run's model)"
            )
        imatrix = self.llamacpp.imatrix
        if imatrix is not None and not imatrix.is_file():
            raise ValueError(
                f"judge.llamacpp.imatrix: {imatrix} is not a file (a llama-imatrix GGUF of the run's model)"
            )

    @model_validator(mode="after")
    def _one_suite(self) -> JudgeSettings:
        """Refuse a suite the command does not run, or a layout that a suite's own cards would ignore."""
        if any("{suite}" in a for a in self.kl_cmd) != (self.suite is not None):
            raise ValueError(
                "judge.suite and the {suite} slot of judge.kl_cmd go together: "
                '["{python}", "{tools}/suite_job.py", "{suite}", "{ckpt}", "{ref}", "{out}"] runs a suite'
            )
        if self.suite is not None and self.kl_layout != KLLayout():
            raise ValueError("judge.kl_layout: a reading by judge.suite takes the cards the suite declares")
        if self.engine == "llama.cpp":
            if self.format_reference is not None:
                raise ValueError(
                    "judge.format_reference gates safetensors formats; a llama.cpp run has no format gate"
                )
            if self.format_map is not None:
                raise ValueError(
                    "judge.format_map fixes safetensors formats; a llama.cpp run has no such map"
                )
        if self.format_map is not None and self.allow_any_format:
            raise ValueError(
                "judge.format_map fixes every module's format and judge.allow_any_format skips the format gate: "
                "set one"
            )
        if self.budget_bpw is not None and self.anchor is None:
            raise ValueError(
                "judge.budget_bpw needs judge.anchor: the budget's denominator is the original model's "
                "decoder-Linear weight count"
            )
        if self.head_lease and (
            self.engine != "vllm" or self.budget_bpw is None or self.head != HEAD_16BIT or self.format_map
        ):
            raise ValueError(
                "judge.head_lease lends a narrower head's bits to the body under judge.budget_bpw: it needs a "
                "vLLM search with judge.budget_bpw, the 16-bit judge.head rule and no judge.format_map"
            )
        self.reading_cards()  # an unknown suite or a bad setting is refused here, not at the first reading
        return self

    def kl_reference(self) -> Path | None:
        """The local path of `kl_ref`, or None without one.

        Raises:
            ValueError: The dataset cannot make it available.
        """
        return None if self.kl_ref is None else parse_dataset(self.kl_ref, "judge.kl_ref").local()

    def kl_holdout_reference(self) -> Path | None:
        """The explicit holdout prompt set (`kl_holdout_ref`), or None when unset."""
        return self.kl_holdout_ref

    def prompt_set(self) -> Dataset | None:
        """The dataset of `gencheck_prompts`; None: the evaluator's built-in set."""
        prompts = self.gencheck_prompts
        return None if prompts is None else parse_dataset(prompts, "judge.gencheck_prompts")

    def prompts_path(self) -> Path | None:
        """The local file `gencheck_prompts` names; None: the evaluator's built-in set.

        Raises:
            ValueError: The dataset cannot make it available.
        """
        prompts = self.prompt_set()
        return None if prompts is None else prompts.local()

    def reading_cards(self) -> int:
        """Cards one reading takes: what `suite` declares, else `kl_layout`'s.

        Raises:
            ValueError: `suite` is unknown or its settings are invalid.
        """
        if self.suite is None:
            return self.kl_layout.cards
        from qda_kl.eval.suites import parse_suite  # the evaluator's own parser: one reading of a suite

        return parse_suite(self.suite, "judge.suite").cards()

    def anchor_bytes(self) -> int | None:
        """File bytes of `judge.anchor`, or None without one.

        Raises:
            ValueError: The anchor is not a usable model directory.
        """
        if self.anchor is None:
            return None
        try:
            return model_size(self.anchor)
        except ValueError as e:
            raise ValueError(f"judge.anchor: {e}") from None

    def max_checkpoint_bytes(self) -> int:
        """The largest checkpoint the judge takes: `max_copy_bytes`, else twice the anchor + 1 GiB, else 200 GiB.

        The anchor is sized even when `max_copy_bytes` is set: an anchor that cannot be read is refused at start
        whatever the cap.

        Raises:
            ValueError: The anchor is not a usable model directory.
        """
        anchor = self.anchor_bytes()
        if self.max_copy_bytes is not None:
            return self.max_copy_bytes
        return DEFAULT_MAX_BYTES if anchor is None else 2 * anchor + (1 << 30)

    def reading_expected_s(
        self, size_bytes: int, cards: int, moe: bool = False, fraction: float = 1.0
    ) -> float:
        """Expected seconds of one KL reading: the load over its cards, plus the scoring.

        A measured `reading_s` wins over the formula: a GGUF reading decodes the file before it loads it, which
        no constant estimates, and `prepare` already ran that very reading to measure the engine floor. It is a whole reading of this
        model on this backend, so `moe` and `fraction` are in it; `measured_cards`, when set, moves the load for
        a reading on another card count at the formula's own per-gib rate.

        Args:
            size_bytes: The frozen checkpoint's bytes.
            cards: The reading's cards (`reading_cards`).
            moe: A checkpoint with routed experts, which `kl_moe_args` reads `kl_moe_factor` times.
            fraction: The share of the reference's scored positions the reading covers (a screening reading
                loads the whole checkpoint but scores only its subset).
        """
        load = self.kl_load_s_per_gib * (size_bytes / (1 << 30))
        if self.reading_s is not None:
            if self.measured_cards is None or self.measured_cards == cards:
                return self.reading_s
            return self.reading_s + load * (1 / cards - 1 / self.measured_cards)
        return load / cards + self.kl_score_s * (self.kl_moe_factor if moe else 1.0) * fraction

    def reading_watchdog_s(self, expected_s: float) -> float:
        """How long `read()` may wait before cancelling a reading: `max(2 * expected, expected + 300 s)`.

        `reading_expected_s` counts the load and the scoring only; a reading also starts an engine
        (process start, weight load, torch.compile, CUDA graphs, warm-up), which is about 106 s on the
        4B rows. A watchdog that fired at the bare estimate killed healthy readings still in engine
        start, so the wait keeps a start-up allowance
        and doubles the estimate for large models, where the scoring term dominates.

        This is the wait only: queue admission still uses `reading_expected_s`, which decides whether
        a job could START before the deadline, not how long a started job may run.
        """
        return max(2.0 * expected_s, expected_s + READING_START_S)

    def drain_window_s(self) -> float:
        """How long pending judge jobs may finish: one anchor-sized KL reading, and never under 900 s.

        An unusable anchor raises on the host's own startup check; the estimate keeps the floor here.
        """
        try:
            anchor = self.anchor_bytes()
        except ValueError:
            return DRAIN_FLOOR_S
        if anchor is None:
            return DRAIN_FLOOR_S
        return max(DRAIN_FLOOR_S, self.reading_expected_s(anchor, self.reading_cards()))


def _check_plugin(value: Any) -> Any:
    return pick(CHECKS, value)


class JudgeConfig(JudgeSettings):
    """Extend judge settings with paths, commands and plugins supplied by `RunConfig.judge_config`.

    Attributes:
        workdir: The agent's work directory (default `<run_dir>/work`); every path it names must lie in it.
        model: The original model as the run names it (a path or a hub id); the long-generation check
            generates with `anchor` when set, else with this.
        format: The run's format (`RunConfig.format`); the launch check builds candidate 0 in it
            (`candidate0_smoke`).
        baselines: Baseline checkpoints: at most two starting points, by default the tier's best community
            checkpoint when one exists, else the library's own AWQ and GPTQ builds with library defaults.
            `baselines[0]` is candidate 0: the result a run ends on as-is when nothing valid was submitted
            or scored, and the checkpoint `judge.candidate0_kl` is read for.
        qda_cmd: The `qda` CLI of `qda_quant` the `qda` command forwards to (a `qda` script, or
            `[<venv python>, -m, qda_quant.cli]`); it must come from a virtualenv or a system prefix. From
            the toolset.
        qda_env: Extra environment of the judge's own `qda` (the trial run's builds, trusted
            receipt builds), from the toolset's trusted settings: never a path into the agent's workdir. A
            `PATH` here goes first.
        agent_qda_env: Extra environment of the agent's `qda` verbs and builds: `qda_env` plus what the
            agent-facing components add (the `source` component's `PYTHONPATH` to the agent's copy).
        contamination: The contamination check (`qda_agent.contamination_checks`) run on every frozen
            checkpoint; the flow runs the same check on the agent's files before it starts. Required, with no
            default of its own: the loader fills it from the run's `contamination` (default `ngram`).
        socket: The RPC socket path (default `<run_dir>/rpc.sock`).
        commands: The commands the agent has (from the toolset); the host refuses the others.
        subcommands: For each command whose verbs the toolset declares, the verbs it offers; the host refuses
            any other first argument.
        handlers: `command -> module:function` for commands the host does not answer itself (from the
            toolset's components).
    """

    workdir: Path | None = None
    model: str | None = None
    format: str | None = None
    baselines: list[Path] = Field(default_factory=list)
    qda_cmd: list[str] = Field(default_factory=lambda: ["qda"], min_length=1)
    qda_env: dict[str, str] = Field(default_factory=dict)
    agent_qda_env: dict[str, str] = Field(default_factory=dict)
    contamination: Annotated[SerializeAsAny[Plugin], BeforeValidator(_check_plugin)]
    socket: Path | None = None
    #: The reference contamination index (hashes only, `qda_quant.calib.contamination`): built host-side from
    #: the KL reference's prompts and passed to every build job, whose calibration filter matches against it
    #: (`QDA_CONTAMINATION_INDEX`, required). The agent never reads it for a prompt: it holds sha256 digests.
    contamination_index: Path | None = None
    commands: list[str] = Field(default_factory=lambda: list(NAMES))
    subcommands: dict[str, list[str]] = Field(default_factory=dict)
    handlers: dict[str, str] = Field(default_factory=dict)

    def job_environ(self, evaluator_parent: Path | None, *, tmp_dir: Path | None = None) -> dict[str, str]:
        """Build judge job overrides with the evaluator copy first on `PYTHONPATH`.

        Prepend `serve_python`'s `bin/` to the configured or host `PATH` so engine subprocesses use
        the serving environment's programs.

        Args:
            evaluator_parent: The dir holding the evaluator package copy, or None (the installed package).
            tmp_dir: The absolute name of the job's private temp directory (`jobtmp.JobTemps.make`): short
                enough for the sockets bound in it, and the same from any cwd a process of the job runs in.
        """
        # The engines' compile caches: the node-local directories every engine of qda uses
        # (`qda_quant.output.vllm_env.cache_env`), unless `judge.job_env` or the host's environment (a job
        # inherits it, `Scheduler._command`) names one. Never under the run dir, which is often on a network
        # filesystem whose lock two engines' first compiles cannot both hold.
        env = cache_env(disabled_kernels(self.job_env))
        if self.engine == "llama.cpp":
            if self.llamacpp is not None:  # the admission's strict check reads the same pair the reading does
                env["QDA_LLAMA_SERVER"] = str(self.llamacpp.server)
                env["QDA_LLAMA_CPP"] = str(self.llamacpp.llama_cpp)
                env["QDA_LLAMA_BRIDGE"] = str(self.llamacpp.bridge)
        if self.serve_python is not None:
            rest = env.get("PATH", os.environ.get("PATH", ""))
            env["PATH"] = os.pathsep.join(p for p in (str(self.serve_python.parent), rest) if p)
        if evaluator_parent is not None:
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (str(evaluator_parent), env.get("PYTHONPATH")) if p
            )
        if tmp_dir is not None:
            # Sockets are bound under it: torch's symmetric memory at `$TMPDIR/symm_mem-<id>` (a multi-card
            # job), vLLM's IPC at the base below. Absolute, never relative to a cwd: a child that changes
            # directory (ninja running nvcc in a build dir) must find the same directory.
            if not tmp_dir.is_absolute():
                raise ValueError(f"a judge job's temp dir must be an absolute path, not {tmp_dir}")
            short = str(tmp_dir)
            env |= {"TMPDIR": short, "TMP": short, "TEMP": short}
            env.setdefault("VLLM_RPC_BASE_PATH", short)
        env |= self.contamination_env()
        return env

    def contamination_env(self) -> dict[str, str]:
        """The build's calibration-filter variables: the reference index, and that a search requires it.

        Every build job of a search gets them, so a build cannot run unfiltered by omission: with the
        variable set and the index unreadable or empty, `qda build` refuses before it loads a weight. A run
        with no reference (no `kl_ref`) has no index and no variable, and its build record says the filter
        did not run (`build.json` `calibration.filter`).
        """
        if self.contamination_index is None:
            return {}
        return {"QDA_CONTAMINATION_INDEX": str(self.contamination_index), "QDA_CONTAMINATION_REQUIRED": "1"}

    @model_validator(mode="after")
    def _starts_label_the_baselines(self) -> JudgeConfig:
        """Every start is one baseline, once: the jobs read the baseline and the judge ranks the start."""
        if not self.starts:
            return self
        paths = [os.path.normpath(start.path) for start in self.starts]
        if len(set(paths)) != len(paths):
            raise ValueError("judge.starts names one path twice: every start is one checkpoint")
        if sorted(paths) != sorted(os.path.normpath(b) for b in self.baselines):
            raise ValueError(
                "judge.starts must label every baselines path once: a start is a baseline the agent's "
                "jobs read, and the bar is the lowest valid KL of the starts"
            )
        names = [start.name for start in self.starts]
        if len(set(names)) != len(names):
            raise ValueError("judge.starts names one label twice: TASK.md shows each start once")
        return self

    def lease_formats(self) -> tuple[str, ...]:
        """The formats the output head may take under `head_lease`; empty when the lease is off."""
        from .checks.budget import lease_formats

        return lease_formats(self.format) if self.head_lease else ()

    @model_validator(mode="after")  # at the host's start: a run config may leave these to a later --set
    def _check(self) -> JudgeConfig:
        try:
            self.lease_formats()
        except ValueError as e:
            raise ValueError(f"judge.head_lease: {e}") from None
        if self.select_candidate0 and not self.baselines:
            raise ValueError("judge.select_candidate0 needs at least one configured baseline")
        if self.llamacpp is not None and self.llamacpp.lock and not self.baselines:
            raise ValueError("judge.llamacpp.lock needs candidate 0 to lock to: set baselines[0]")
        if self.kl_ref is None and any(slot in a for a in self.kl_cmd for slot in ("{ref}", "{config}")):
            raise ValueError(
                "judge.kl_cmd needs the KL reference ({ref} or {config}): set judge.kl_ref "
                "(the original model's, from `qda-kl ref`)"
            )
        if "gencheck" in self.commands and self.anchor is None and not self.model:
            raise ValueError(
                "the gencheck command needs the original model: set judge.anchor (or the run's model)"
            )
        if "gencheck" in self.commands and self.engine == "llama.cpp":
            original = Path(str(self.anchor or self.model)).expanduser()
            if not (original / "config.json").is_file():
                raise ValueError(
                    f"the gencheck command decodes each .gguf against the original's HF snapshot, and {original} "
                    "is not a local one: set judge.anchor to the local HF snapshot the .gguf files are "
                    "converted from"
                )
        if self.format_map is not None and not (self.format_map / "config.json").is_file():
            raise ValueError(
                f"judge.format_map {self.format_map} is not a checkpoint directory "
                "(the fixed allocation's checkpoint, with its config.json and safetensors shards)"
            )
        if self.format_reference is not None and not (self.format_reference / "config.json").is_file():
            raise ValueError(
                f"judge.format_reference {self.format_reference} is not a checkpoint directory "
                "(a baseline of the task's format, with its config.json and safetensors shards)"
            )
        if (self.candidate0_kl is not None or self.candidate0_build_s is not None) and not any(
            rule.accepts_candidate0 for rule in self.final_rules
        ):
            raise ValueError(
                "judge.candidate0_kl and judge.candidate0_build_s describe candidate 0 for a rule that "
                "reads it: add judge.final_rules: [handin], or remove them"
            )
        return self
