"""The ``full`` eval preset: each original model's released-card benchmark settings, one table for every entry point.

``qda eval`` and ``qda-agent evaluate`` expand ``full`` from here; a model not in the table gets defaults and a note.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError
from qda_quant.calib.selfgen import SAMPLING_FIELDS, sampling_of
from qda_quant.output.generation import CARD_SAMPLING, hub_id
from qda_quant.step import Refused

from qda_kl.eval.suites import Suite

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "FAMILIES",
    "FullSettings",
    "FullSuite",
    "REPEATS",
    "Applied",
    "Family",
    "applied",
    "card_sampling",
    "row_of",
    "template_think_end",
]


#: The answer cap of a model the table does not name, before it is capped by the model's context.
DEFAULT_MAX_TOKENS = 32768

#: The engine window of a model the table does not name: the widest window the released cards served.
DEFAULT_MAX_MODEL_LEN = 114688

#: The sampling seed: sample ``j`` uses ``seed + j`` (the paper's 17 + j, as every released card did).
SEED = 17

#: Repeats per task, as the released cards ran them; a family that runs the simple-evals GPQA has its own key.
REPEATS: dict[str, int] = {
    "paper_aime25": 32,
    "paper_hmmt_feb_2026": 32,
    "paper_gpqa_diamond": 16,
    "paper_gpqa_diamond_simple_evals": 16,
    "ifbench": 16,
    "ifeval": 16,
    "mmlu_pro": 1,
    "livecodebench": 4,
}

#: The headline metrics, as the cards scored them: IFBench loose, IFEval strict, MMLU-Pro exact match. The cards
#: include IFEval on the R1 model only; every other family runs the harness's strict headline for it.
METRICS: dict[str, str] = {
    "ifbench": "prompt_level_loose_acc",
    "ifeval": "prompt_level_strict_acc",
    "mmlu_pro": "exact_match",
}

#: The paper's `lm_eval` tasks besides GPQA, whose task name differs on the R1 family.
TASKS: tuple[str, ...] = ("paper_aime25", "paper_hmmt_feb_2026", "ifbench", "ifeval", "mmlu_pro")

#: Answer caps of the families whose cards serve every task at 81920 tokens (MMLU-Pro at 32768).
CAPS_81920: dict[str, int] = {
    "paper_aime25": 81920,
    "paper_hmmt_feb_2026": 81920,
    "paper_gpqa_diamond": 81920,
    "ifbench": 81920,
    "ifeval": 81920,
    "mmlu_pro": 32768,
    "livecodebench": 81920,
}

#: Qwen3.8-27B's cards cap GPQA, IFBench and IFEval at 32768 tokens.
CAPS_27B: dict[str, int] = {**CAPS_81920, "paper_gpqa_diamond": 32768, "ifbench": 32768, "ifeval": 32768}

#: The R1 cards cap every task at 32768 tokens, on a 49152-token window.
CAPS_R1: dict[str, int] = {
    "paper_aime25": 32768,
    "paper_hmmt_feb_2026": 32768,
    "paper_gpqa_diamond_simple_evals": 32768,
    "ifbench": 32768,
    "ifeval": 32768,
    "mmlu_pro": 32768,
    "livecodebench": 32768,
}

#: The serving arguments every card used: bf16 weights and KV cache, prefix caching off, 128 sequences.
SERVE_ARGS: dict[str, Any] = {
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "enable_prefix_caching": False,
    "max_num_seqs": 128,
}


@dataclass(frozen=True)
class Family:
    """One table row: the settings every task of one released model ran with.

    Attributes:
        max_tokens: Answer cap per task, keyed by the task the suite runs (``livecodebench`` is its own task).
        sampling: The sampling every task used, MMLU-Pro included.
        think_end_token: The model's closing thinking tag, or None for a model that does not think.
        chat_template_kwargs: Extra chat-template arguments (``{"reasoning_effort": "medium"}`` on 27B).
        enable_thinking: The chat template's thinking mode.
        gpu_memory_utilization: The engine's memory share.
        max_model_len: The engine window.
        gpqa_task: The GPQA task the card ran (the R1 card uses the simple-evals extraction).
        lcb_tp: The LiveCodeBench engine's tensor-parallel size (2 on the cards that split it).
        tokenizer_reference: Serve the reference's tokenizer, as the R1 cards did to keep one chat template
            across rows.
    """

    max_tokens: dict[str, int]
    sampling: dict[str, float]
    think_end_token: str | None
    chat_template_kwargs: dict[str, str] = field(default_factory=dict)
    enable_thinking: bool = True
    gpu_memory_utilization: float = 0.95
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpqa_task: str = "paper_gpqa_diamond"
    lcb_tp: int = 1
    tokenizer_reference: bool = False


#: The one table: the original model's HF id -> the settings its released cards record. Nine ids, seven rows'
#: worth of settings: the two Qwen3.5 sizes are one row, and the two Gemma sizes differ only in LCB's TP.
FAMILIES: dict[str, Family] = {
    "Qwen/Qwen3.5-4B": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["Qwen/Qwen3.5-4B"],
        think_end_token="</think>",
    ),
    "Qwen/Qwen3.5-9B": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["Qwen/Qwen3.5-9B"],
        think_end_token="</think>",
    ),
    "Qwen/Qwen3.6-35B-A3B": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["Qwen/Qwen3.6-35B-A3B"],
        think_end_token="</think>",
        gpu_memory_utilization=0.9,
        lcb_tp=2,
    ),
    "Qwen/Qwen3.8-27B": Family(
        max_tokens=CAPS_27B,
        sampling=CARD_SAMPLING["Qwen/Qwen3.8-27B"],
        think_end_token="</think>",
        chat_template_kwargs={"reasoning_effort": "medium"},
        gpu_memory_utilization=0.9,
    ),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["Qwen/Qwen3-Coder-30B-A3B-Instruct"],
        think_end_token=None,
        enable_thinking=False,
        lcb_tp=2,
    ),
    "google/gemma-4-26B-A4B-it": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["google/gemma-4-26B-A4B-it"],
        think_end_token="<channel|>",
    ),
    "google/gemma-4-31B-it": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["google/gemma-4-31B-it"],
        think_end_token="<channel|>",
        lcb_tp=2,
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": Family(
        max_tokens=CAPS_81920,
        sampling=CARD_SAMPLING["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"],
        think_end_token="</think>",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": Family(
        max_tokens=CAPS_R1,
        sampling=CARD_SAMPLING["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"],
        think_end_token="</think>",
        max_model_len=49152,
        gpqa_task="paper_gpqa_diamond_simple_evals",
        tokenizer_reference=True,
    ),
}

#: The table's closing thinking tags: a model the table does not name is matched to one by its chat template.
THINK_END_TOKENS: tuple[str, ...] = tuple(
    dict.fromkeys(row.think_end_token for row in FAMILIES.values() if row.think_end_token is not None)
)


class FullSettings(BaseModel):
    """What a user may override in a ``full`` entry; every unset field keeps the table's value.

    Attributes:
        max_tokens: An answer cap for every task, or per task (a task the table names may be overridden alone).
        repeats: Samples per problem, for every task or per task.
        sampling: Sampling fields merged over the row's (MMLU-Pro uses the same ones).
        think_end_token: The closing thinking tag.
        chat_template_kwargs: Template arguments, merged over the row's.
        enable_thinking: The chat template's thinking mode.
        metric: Headline metrics, merged over the row's.
        max_model_len: The engine window.
        gpu_memory_utilization: The engine's memory share.
        serve_args: Engine arguments, merged over the row's.
        seed: Seed of sample 0.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int | dict[str, int] | None = None
    repeats: int | dict[str, int] | None = None
    sampling: dict[str, float] | None = None
    think_end_token: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    enable_thinking: bool | None = None
    metric: dict[str, str] | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    serve_args: dict[str, Any] | None = None
    seed: int | None = None


@dataclass(frozen=True)
class Applied:
    """One expansion of the preset: the row it applied and the two ``qda eval`` suite entries.

    Attributes:
        note: The line that says which row was applied (the first line both entry points print).
        origin: The name the row was found by (the reference or a checkpoint's ``base_model``), or None.
        entries: ``{lm_eval: ...}`` and ``{livecodebench: ...}``, each valid ``qda eval`` input.
    """

    note: str
    origin: str | None
    entries: tuple[dict[str, Any], ...]


def _key(name: str) -> str:
    """The HF id a name denotes (`qda_quant.output.generation.hub_id`)."""
    return hub_id(name)


def _candidates(reference: str, checkpoints: Sequence[str]) -> list[str]:
    """The names to look up, in order: the reference, what its config says, then each checkpoint's original."""
    from qda_kl.kl import original_of

    names = [str(reference)]
    config = Path(reference) / "config.json"
    if config.is_file():
        try:
            named = json.loads(config.read_text()).get("_name_or_path")
        except (OSError, ValueError):
            named = None
        if isinstance(named, str) and named:
            names.append(named)
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        if not path.is_dir():
            continue
        try:
            if base := original_of(path):
                names.append(base)
        except (OSError, ValueError):
            continue
    return names


def row_of(reference: str, checkpoints: Sequence[str] = ()) -> tuple[Family, str] | tuple[None, str | None]:
    """The table row for a run's original, and the name it was found by; ``(None, name)`` when no row matches."""
    names = _candidates(reference, checkpoints)
    for name in names:
        if (key := _key(name)) in FAMILIES:
            return FAMILIES[key], name
    return None, names[0] if names else None


def card_sampling(reference: str) -> tuple[dict[str, float], str] | None:
    """The released cards' sampling for ``reference``'s original and the row's HF id, or None (a copy).

    ``qda-kl ref`` uses it in place of the model's own ``generation_config.json``, so a model's reference and
    its ``full`` eval are sampled alike.
    """
    row, origin = row_of(reference)
    return None if row is None or origin is None else (dict(row.sampling), _key(origin))


def template_think_end(model: str) -> str | None:
    """The table's closing thinking tag that ``model``'s chat template writes, or None (also for a Hub id).

    Only a local directory's template is read (``chat_template.jinja``, ``chat_template.json`` or
    ``tokenizer_config.json``'s), never the vocabulary: a thinking family's other models carry the tag too.
    """
    templates: list[str] = []
    for name in ("chat_template.jinja", "chat_template.json", "tokenizer_config.json"):
        path = Path(model) / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
            if name.endswith(".json"):  # tokenizer_config.json also lists the added tokens: the template only
                text = json.dumps(json.loads(text).get("chat_template"))
        except (OSError, ValueError):
            continue
        templates.append(text)
    return next((tag for tag in THINK_END_TOKENS if any(tag in text for text in templates)), None)


def _refuse_two_lines(reason: str) -> Refused:
    """A refusal in the shape the preset promises: the reason, then the one change that fixes it."""
    return Refused(
        f"{reason}\n"
        "fix: give the preset's sampling explicitly, e.g. suites.0.full.sampling "
        "{temperature: 1.0, top_p: 0.95}"
    )


def _defaults(reference: str, origin: str | None) -> Family:
    """The row a model the table does not name gets: 32768-token caps, its own sampling and thinking tag."""
    from qda_kl.reference import context_length

    name = origin or str(reference)
    try:
        sampling = sampling_of(name)
    except (OSError, ValueError) as why:
        raise _refuse_two_lines(
            f"full: {name}: its generation_config.json could not be read ({why})"
        ) from None
    if not sampling:
        raise _refuse_two_lines(f"full: {name}: its generation_config.json is missing or names no sampling")
    try:
        context = context_length(name)
    except (OSError, ValueError):
        context = 1 << 30  # unreadable config: no context is known, so only the default cap applies
    cap = min(DEFAULT_MAX_TOKENS, context)
    caps = {task: cap for task in (*TASKS, "paper_gpqa_diamond")}
    caps["livecodebench"] = cap
    return Family(
        max_tokens=caps,
        sampling=sampling,
        think_end_token=template_think_end(name),
        max_model_len=min(context, DEFAULT_MAX_MODEL_LEN),
    )


def _per_task(value: int | dict[str, int], tasks: Sequence[str], what: str) -> dict[str, int]:
    """A per-task override, checked against ``tasks``: one value for every task, or a mapping naming tasks."""
    if isinstance(value, int):
        return {task: value for task in tasks}
    unknown = sorted(set(value) - set(tasks))
    if unknown:
        raise Refused(f"full: {what}: unknown task(s) {unknown}; tasks are {list(tasks)}")
    return dict(value)


def _serve_args(reference: str, row: Family, settings: FullSettings) -> dict[str, Any]:
    """The engine arguments of both suites: the card's serving settings, the window, and an R1 tokenizer if any."""
    args: dict[str, Any] = {
        **SERVE_ARGS,
        "gpu_memory_utilization": (
            row.gpu_memory_utilization
            if settings.gpu_memory_utilization is None
            else settings.gpu_memory_utilization
        ),
        "max_model_len": row.max_model_len if settings.max_model_len is None else settings.max_model_len,
    }
    if row.tokenizer_reference:
        args["tokenizer"] = str(reference)
    args.update(settings.serve_args or {})
    return args


def _entries(reference: str, row: Family, settings: FullSettings) -> tuple[dict[str, Any], ...]:
    """The ``lm_eval`` and ``livecodebench`` entries of one row, with the user's overrides applied."""
    tasks = [*TASKS[:2], row.gpqa_task, *TASKS[2:]]
    every = [*tasks, "livecodebench"]  # an override may name any task of the preset, either suite's
    caps = {task: row.max_tokens[task] for task in every}
    repeats = {task: REPEATS[task] for task in every}
    if settings.max_tokens is not None:
        caps.update(_per_task(settings.max_tokens, every, "max_tokens"))
    if settings.repeats is not None:
        repeats.update(_per_task(settings.repeats, every, "repeats"))
    sampling = {**row.sampling, **(settings.sampling or {})}
    unknown = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unknown:
        raise Refused(f"full: sampling: unknown {unknown[0]!r}; known: {', '.join(SAMPLING_FIELDS)}")
    tags = {**row.chat_template_kwargs, **(settings.chat_template_kwargs or {})}
    metric = {**METRICS, **(settings.metric or {})}
    think = row.think_end_token if settings.think_end_token is None else settings.think_end_token
    thinking = row.enable_thinking if settings.enable_thinking is None else settings.enable_thinking
    seed = SEED if settings.seed is None else settings.seed
    serve = _serve_args(reference, row, settings)
    lm_eval = {
        "tasks": tasks,
        "repeats": {task: repeats[task] for task in tasks},
        "max_tokens": {task: caps[task] for task in tasks},
        "sampling": sampling,
        "enable_thinking": thinking,
        "think_end_token": think,
        "chat_template_kwargs": tags,
        "metric": metric,
        "seed": seed,
        "serve_args": serve,
    }
    lcb_serve = dict(serve)
    if row.lcb_tp > 1:
        lcb_serve.setdefault("tensor_parallel_size", row.lcb_tp)
    livecodebench = {
        "protocol": "paper",
        "timeout_s": 8,
        "repeats": repeats["livecodebench"],
        "max_tokens": caps["livecodebench"],
        "sampling": sampling,
        "enable_thinking": thinking,
        "think_end_token": think,
        "chat_template_kwargs": tags,
        "seed": seed,
        "serve_args": lcb_serve,
    }
    return ({"lm_eval": lm_eval}, {"livecodebench": livecodebench})


def applied(
    reference: str, checkpoints: Sequence[str] = (), settings: dict[str, Any] | None = None
) -> Applied:
    """Expand ``full`` for one run: the table row (or the default), the user's overrides, and the note.

    Args:
        reference: The run's original model: an HF id, or a local directory.
        checkpoints: The run's checkpoints; a local one whose files name its ``base_model`` names the original
            when the reference itself does not.
        settings: The ``full`` entry's overrides (:class:`FullSettings`); every unset field keeps the table's.

    Raises:
        Refused: The overrides are not valid settings, or an unknown model's ``generation_config.json`` cannot
            supply the sampling (two lines: the reason, then the fix).
    """
    try:
        spec = FullSettings.model_validate(settings or {})
    except ValidationError as e:
        raise Refused(
            "full: " + "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        ) from None
    row, origin = row_of(reference, checkpoints)
    if row is not None:
        note = f"full: settings of {_key(origin)} (the released cards')"
    else:
        row = _defaults(reference, origin)
        # a thinking model's whole answer holds its thinking too, so the note says which text is scored
        think = row.think_end_token if spec.think_end_token is None else spec.think_end_token
        scored = (
            f"only the text after {think!r} scored"
            if think
            else "whole answers scored (no closing thinking tag in its chat template)"
        )
        note = (
            f"full: no released card for {origin or reference}; 32768-token caps, its "
            f"generation_config.json's sampling, and {scored}"
        )
    if settings:
        note += " (your overrides applied)"
    return Applied(note=note, origin=origin, entries=_entries(reference, row, spec))


class FullSuite(Suite):
    """The ``full`` entry of a ``qda eval`` config, replaced by :func:`applied`'s entries before any job runs.

    Its :class:`FullSettings` fields are validated only at expansion, once the reference row exists to merge over.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: ClassVar[str] = "full"

    def to_dict(self) -> dict[str, Any]:
        """The entry as written (overrides only): the runner expands it before any job exists."""
        return {self.name: dict(self.model_extra or {})}

    def expand(self, reference: str, checkpoints: Sequence[str] = ()) -> Applied:
        """Expand this entry for a run; see :func:`applied`."""
        return applied(reference, checkpoints, self.model_extra)
