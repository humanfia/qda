"""Check long generations for repetition loops and answers cut at the token limit."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from qda_quant.output.record import without_provenance
from qda_quant.recipe import apply_overrides, error_lines
from qda_quant.step import Refused

from qda_kl.datasets import DATASETS, PROMPTS, Dataset, parse_dataset, with_defaults
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_TP
from qda_kl.lifetime import command
from qda_kl.serve import NO_JIT_ENV as NO_JIT_ENV  # re-exported: the tests and callers name it here
from qda_kl.serve import ServeSettings, batch_args, no_jit, settings_for
from qda_kl.serve import llm as start_llm

RESULT = "gencheck.json"


class GencheckConfig(BaseModel):
    """Configure matched generations from a checkpoint and its original model.

    Defaults select four prompts per domain with an 8192-token answer limit.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    checkpoint: str = Field(
        min_length=1,
        description="Checkpoint directory served by vLLM, or a `.gguf` file, decoded to bf16 against `original`.",
    )
    original: str = Field(
        min_length=1, description="Original model directory or Hub id (a local directory for a `.gguf`)."
    )
    prompts: Annotated[str, Field(min_length=1)] | dict[str, Any] = Field(
        default="builtin",
        description="`builtin` (the shipped set), a prompt JSONL path, or one qda.datasets {key: settings}.",
    )
    domains: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Domains to select (default: every domain in the prompt set).",
    )
    per_domain: int = Field(
        default=4, ge=1, description="First prompts per domain (default 4; full check: 16)."
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        description="Answer token limit (default 8192; full check: the evaluation's length, e.g. 32768).",
    )
    seed: int = Field(default=17, ge=0, description="Sampling seed for both models.")
    temperature: float = Field(
        default=0.6, ge=0, description="Temperature; other sampling uses the original."
    )
    enable_thinking: bool = Field(
        default=True,
        description=(
            "Chat template's thinking mode (default on, as the benchmark models' cards show; false for a "
            "non-thinking model)."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(default=None, description="Extra chat-template args.")
    original_cache: str | None = Field(
        default=None, min_length=1, description="Optional original answer cache."
    )
    out: str = Field(min_length=1, description="Directory for gencheck.json, resolved.yaml and texts/.")
    tp: int = Field(default=DEFAULT_TP, ge=1, description="Tensor-parallel size for generation.")
    serve_args: dict[str, Any] = Field(
        default_factory=dict,
        description="``vllm.LLM`` keyword arguments for both models, over ``build.json``'s ``serve.args`` "
        "(e.g. ``max_model_len`` for a hybrid model whose default context does not fit one card).",
    )
    nccl_nvls: bool = Field(
        default=False,
        description="Let both engines' ranks join NCCL's NVLS collectives; off, as for a build.",
    )

    @field_validator("serve_args")
    @classmethod
    def _no_card_count(cls, serve_args: dict[str, Any]) -> dict[str, Any]:
        if "tensor_parallel_size" in serve_args:
            raise ValueError(
                f"serve_args names tensor_parallel_size={serve_args['tensor_parallel_size']}; set tp instead"
            )
        return serve_args

    @field_validator("prompts")
    @classmethod
    def _recorded_prompts(cls, prompts: str | dict[str, Any]) -> str | dict[str, Any]:
        return with_defaults(prompts, "prompts", PROMPTS)

    @field_validator("domains")
    @classmethod
    def _unique_domains(cls, domains: list[str] | None) -> list[str] | None:
        if domains is not None and len(set(domains)) != len(domains):
            raise Refused("domains must be unique")
        return domains

    @model_validator(mode="after")
    def _domains_from_prompts(self) -> GencheckConfig:
        """Make the prompt set the source of the domains; default to all it holds."""
        available = list(dict.fromkeys(prompt.domain for prompt in _load_prompts(self.prompts)))
        if not available:
            raise Refused("prompts: the prompt set has no domains")
        if self.domains is None:
            self.domains = available
        elif unknown := [domain for domain in self.domains if domain not in available]:
            raise Refused(
                f"domains: unknown {', '.join(map(repr, unknown))}; available: {', '.join(available)}"
            )
        return self

    @staticmethod
    def raw(text: str, overrides: Sequence[str] = ()) -> dict[str, Any]:
        """Read YAML and apply dotted overrides, accepting resolved configs as input."""
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise Refused(f"not valid YAML: {error}") from None
        if not isinstance(raw, dict):
            raise Refused("a qda-kl gencheck config is a YAML mapping")
        return apply_overrides(without_provenance(raw), list(overrides))

    @classmethod
    def from_yaml(cls, text: str, overrides: Sequence[str] = ()) -> GencheckConfig:
        """Validate a YAML config, reporting dotted paths for invalid keys."""
        try:
            return cls.model_validate(cls.raw(text, overrides))
        except ValidationError as error:
            raise Refused("; ".join(error_lines(error))) from None


class Prompt(BaseModel):
    """Identify a prompt and its domain independently of its position in the source file."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    source: str | None = Field(
        default=None, min_length=1, description="Optional source group for provenance."
    )


class Generation(BaseModel):
    """Store one answer for text exports and caching, separate from the counts-only report."""

    model_config = ConfigDict(extra="forbid")

    id: str
    domain: str
    text: str
    token_ids: list[int]
    token_count: int = Field(ge=0)
    finish_reason: Literal["stop", "length"]

    @model_validator(mode="after")
    def _matching_token_count(self) -> Generation:
        if self.token_count != len(self.token_ids):
            raise ValueError("token_count must equal len(token_ids)")
        return self


class _CacheManifest(BaseModel):
    """Record the identity, count and checksum of a completed original-answer cache."""

    model_config = ConfigDict(extra="forbid", strict=True)

    cache_key: str
    record_count: int = Field(ge=0)
    sha256: str


def _prompt_set(prompts: str | dict[str, Any]) -> Dataset:
    """The dataset of a config's ``prompts``; the word ``builtin`` is the shipped set."""
    return parse_dataset({"gencheck_builtin": {}} if prompts == "builtin" else prompts, "prompts", PROMPTS)


def _load_prompts(selected: str | dict[str, Any]) -> list[Prompt]:
    """Read every prompt of a config's ``prompts``, in file order."""
    source = _prompt_set(selected).local()
    try:
        prompts = [
            Prompt.model_validate_json(line) for line in source.read_text().splitlines() if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise Refused(f"invalid prompt file {source}: {error}") from None
    if len({prompt.id for prompt in prompts}) != len(prompts):
        raise Refused("prompt ids must be unique")
    return prompts


def read_prompts(config: GencheckConfig) -> list[Prompt]:
    """Select ``per_domain`` prompts from each configured domain in source-file order.

    Raises:
        Refused: The JSONL is invalid, has duplicate ids, or lacks enough prompts.
    """
    prompts = _load_prompts(config.prompts)
    assert config.domains is not None  # set by _domains_from_prompts
    counts = dict.fromkeys(config.domains, 0)
    selected = []
    for prompt in prompts:
        if prompt.domain in counts and counts[prompt.domain] < config.per_domain:
            selected.append(prompt)
            counts[prompt.domain] += 1
    for domain, count in counts.items():
        if count < config.per_domain:
            raise Refused(f"{domain}: need {config.per_domain} prompts, found {count}")
    return selected


def prompt_token_ids(
    tokenizer: Any, prompt: str, chat_template_kwargs: dict[str, Any] | None = None, chat: bool = True
) -> list[int]:
    """The prompt's token ids, through the chat template when ``chat`` and as plain text otherwise."""
    if chat:
        got = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            **(chat_template_kwargs or {}),
        )
        return list(got["input_ids"] if hasattr(got, "keys") else got)
    return list(tokenizer(prompt)["input_ids"])


def as_generation(prompt: Prompt, output: Any) -> Generation:
    """One engine answer as a :class:`Generation`.

    Raises:
        RuntimeError: the engine returned more or fewer than one answer, or did not stop normally.
    """
    if len(output.outputs) != 1 or output.outputs[0].finish_reason not in ("stop", "length"):
        raise RuntimeError(f"{prompt.id}: generation did not complete normally")
    answer = output.outputs[0]
    return Generation(
        id=prompt.id,
        domain=prompt.domain,
        text=answer.text,
        token_ids=list(answer.token_ids),
        token_count=len(answer.token_ids),
        finish_reason=answer.finish_reason,
    )


def build_report(
    checkpoint: Sequence[Generation],
    original: Sequence[Generation],
    config: GencheckConfig,
    serve: ServeSettings | None = None,
) -> dict[str, Any]:
    """Summarize matched answers by domain, with counts and deltas but no text or token ids.

    ``serve`` is the checkpoint's serve settings, recorded so a reader sees the kernel plan the answers came from.
    """
    assert config.domains is not None  # set by _domains_from_prompts
    identities = [{(row.id, row.domain) for row in rows} for rows in (checkpoint, original)]
    if identities[0] != identities[1] or any(
        len(identities[index]) != len(rows) for index, rows in enumerate((checkpoint, original))
    ):
        raise Refused("checkpoint and original answers must have matching, unique ids and domains")
    report: dict[str, Any] = {
        "max_tokens": config.max_tokens,
        "per_domain": config.per_domain,
        "serve": (serve or ServeSettings()).model_dump(mode="json"),
    }
    for name, rows in (("checkpoint", checkpoint), ("original", original)):
        report[name] = {}
        if any(row.domain not in config.domains for row in rows):
            raise Refused("answers contain an unselected domain")
        for domain in config.domains:
            group = [row for row in rows if row.domain == domain]
            if len(group) != config.per_domain:
                raise Refused(f"{name}.{domain}: expected {config.per_domain} answers, got {len(group)}")
            report[name][domain] = {
                "n": len(group),
                "cut": sum(row.finish_reason == "length" for row in group),
                "loops": sum(is_loop(row.token_ids) for row in group),
                "mean_tokens": sum(len(row.token_ids) for row in group) / len(group),
            }
    report["delta"] = {
        domain: {
            key: report["checkpoint"][domain][key] - report["original"][domain][key]
            for key in ("cut", "loops", "mean_tokens")
        }
        for domain in config.domains
    }
    report["total"] = {
        name: {key: sum(group[key] for group in report[name].values()) for key in ("n", "cut", "loops")}
        for name in ("checkpoint", "original")
    }
    return report


def run_gencheck(config: GencheckConfig) -> dict[str, Any]:
    """Generate checkpoint answers, reuse or generate original answers, and publish their comparison.

    A `.gguf` checkpoint is decoded to bf16 against `original` (`qda_kl.gguf_checkpoint`, as `qda-kl score`
    does) into ``<out>/decoded``, answered by vLLM, and the decode removed. Write answer texts and the resolved
    config before publishing the counts-only report.
    """
    from qda_quant.output.record import plugins, provenance, write_resolved

    from qda_kl import gguf_checkpoint
    from qda_kl.cli import publish

    out = Path(config.out)
    (out / RESULT).unlink(missing_ok=True)
    (out / "resolved.yaml").unlink(missing_ok=True)
    prompts = read_prompts(config)
    cache = None
    if config.original_cache is not None:
        cache = Path(config.original_cache) / original_cache_key(config, prompts) / "generations.jsonl"
    gguf = gguf_checkpoint.gguf_file(config.checkpoint)
    model, own, decoded = config.checkpoint, config, None
    if gguf is not None:
        snapshot = gguf_checkpoint.snapshot_of(config.original, key="original")
        print(f"qda-kl gencheck: decoding {gguf} against {snapshot}", file=sys.stderr, flush=True)
        decoded, manifest = gguf_checkpoint.decode(gguf, snapshot, out)
        model = str(decoded)
        if gguf_checkpoint.language_only(manifest):  # the decode holds no visual tower to load
            own = config.model_copy(update={"serve_args": {"language_model_only": True, **config.serve_args}})
    try:
        checkpoint = _generate(model, prompts, own)
        settings = gencheck_settings(model, own)
    finally:
        if decoded is not None:
            shutil.rmtree(decoded, ignore_errors=True)
    original = _cached_original(cache, prompts)
    cache_hit = original is not None
    if original is None:
        original = _generate(config.original, prompts, config)
    else:
        print("qda-kl: reusing cached original generations", file=sys.stderr)
    expected = {(prompt.id, prompt.domain) for prompt in prompts}
    for rows in (checkpoint, original):
        if {(row.id, row.domain) for row in rows} != expected:
            raise RuntimeError("generation results do not match the selected prompts")
    report = build_report(checkpoint, original, config, settings)
    if gguf is not None:
        report["gguf"] = {"sha256": manifest["gguf"]["sha256"]}
    if cache is not None and not cache_hit:
        _write_original_cache(cache, original)
    texts = out / "texts"
    texts.mkdir(parents=True, exist_ok=True)
    for name, rows in (("checkpoint", checkpoint), ("original", original)):
        publish(texts / f"{name}.jsonl", "".join(row.model_dump_json() + "\n" for row in rows))
    named = _prompt_set(config.prompts) if isinstance(config.prompts, dict) else None
    used = plugins([(DATASETS, named.name, type(named))]) if named else {}
    write_resolved(out, config.model_dump(mode="json"), provenance(used))
    publish(out / RESULT, json.dumps(report, indent=2) + "\n")
    return report


def original_cache_key(config: GencheckConfig, prompts: Sequence[Prompt]) -> str:
    """Hash the original's identity, serving settings, selected prompts and generation settings.

    Include sampling defaults and library versions to invalidate stale answers. Local weights use
    file sizes and modification times; Hub models include their resolved config revision.
    The serving settings are part of the key: answers generated on one engine are not answers of another.
    ``nccl_nvls`` is one of them: NVLS picks another NCCL collective algorithm, so a run with it on can
    generate other answers even at the same seed.
    Checkpoint and output paths do not affect the original's answers.
    """
    serve = gencheck_settings(config.original, config).model_dump()
    # Kernel names are sets; sort them so the key is JSON-encodable and stable across processes.
    serve["kernels"] = {kind: sorted(names) for kind, names in serve["kernels"].items()}
    data = {
        "schema": 3,
        "serve": serve,
        "original": _original_identity(config.original),
        # Domain order changes report order; selected prompts stay in the same file order.
        "prompts": [prompt.model_dump() for prompt in prompts],
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "temperature": config.temperature,
        "enable_thinking": config.enable_thinking,
        "chat_template_kwargs": config.chat_template_kwargs or {},
        "tp": config.tp,
        "versions": {},
    }
    for name in ("vllm", "transformers", "torch"):
        try:
            data["versions"][name] = version(name)
        except PackageNotFoundError:
            data["versions"][name] = None
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _original_identity(model: str) -> dict[str, Any]:
    """Identify model and sampling configs, plus metadata for local checkpoint files."""
    from transformers import AutoConfig, GenerationConfig

    model_config = AutoConfig.from_pretrained(model)
    try:
        sampling = GenerationConfig.from_pretrained(model)
    except OSError:
        sampling = GenerationConfig.from_model_config(model_config)
    identity = {
        "model": model,
        "revision": getattr(model_config, "_commit_hash", None),
        "config": model_config.to_dict(),
        "generation_config": sampling.to_dict(),
    }
    path = Path(model)
    if path.is_dir():
        identity["path"] = str(path.resolve())
        identity["files"] = {
            str(file.relative_to(path)): {
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in sorted(path.iterdir())
            if file.is_file() and file.suffix in (".safetensors", ".bin", ".json", ".model", ".txt")
        }
    return identity


def _cached_original(cache: Path | None, prompts: Sequence[Prompt]) -> list[Generation] | None:
    """Read cached answers only when their manifest, checksum and ordered prompt identities match."""
    if cache is None:
        return None
    try:
        manifest = _CacheManifest.model_validate_json((cache.parent / "complete.json").read_bytes())
        if manifest.cache_key != cache.parent.name or manifest.record_count != len(prompts):
            return None
        data = cache.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest.sha256:
            return None
        rows = [Generation.model_validate_json(line) for line in data.splitlines()]
    except (OSError, ValueError):
        return None
    if len(rows) != manifest.record_count:
        return None
    if [(row.id, row.domain) for row in rows] != [(prompt.id, prompt.domain) for prompt in prompts]:
        return None
    return rows


def _write_original_cache(cache: Path, rows: Sequence[Generation]) -> None:
    """Publish the answer JSONL atomically, then mark it complete with a checksum manifest."""
    from qda_kl.cli import publish

    cache.parent.mkdir(parents=True, exist_ok=True)
    completion = cache.parent / "complete.json"
    completion.unlink(missing_ok=True)
    publish(cache, "".join(row.model_dump_json() + "\n" for row in rows))
    manifest = _CacheManifest(
        cache_key=cache.parent.name,
        record_count=len(rows),
        sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
    )
    publish(completion, manifest.model_dump_json(indent=2) + "\n")


def gencheck_settings(model: str | Path, config: GencheckConfig) -> ServeSettings:
    """The serve settings both of a check's engines start with: `settings_for`, then `no_jit`."""
    return no_jit(settings_for(model, args=config.serve_args, nccl_nvls=config.nccl_nvls))


def _generate(model: str, prompts: list[Prompt], config: GencheckConfig) -> list[Generation]:
    """Generate answers in a child process so its engine exits before the next model loads.

    The child reports a failure to the ``failure`` file as well as its traceback: the traceback stays on the
    inherited stderr (the job log keeps it), and the exception names the child's own error, so the caller and
    the judge's reason carry the cause instead of ``exited 1`` alone (11-16: three runs of a failed check
    recorded only that).
    """
    with TemporaryDirectory(prefix="qda-gencheck-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.jsonl"
        failure = Path(directory) / "failure.txt"
        request.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompts": [prompt.model_dump() for prompt in prompts],
                    "config": config.model_dump(mode="json"),
                }
            )
        )
        result = subprocess.run(
            [sys.executable, "-m", "qda_kl.gencheck", str(request), str(response), str(failure)],
            check=False,
        )
        if result.returncode:
            why = failure.read_text(errors="replace").strip() if failure.is_file() else ""
            raise RuntimeError(
                f"generation failed for {model} (engine process exited {result.returncode})"
                + (f": {why}" if why else "")
            )
        return [Generation.model_validate_json(line) for line in response.read_text().splitlines()]


def _generate_in_process(model: str, prompts: list[Prompt], config: GencheckConfig) -> list[Generation]:
    """Generate answers with vLLM using the original model's sampling defaults and config overrides.

    A checkpoint's own serve settings (``build.json["serve"]``) are applied, so it is generated on the kernels
    and in the environment its build planned. The config's ``serve_args`` win over them for both models, so the
    checkpoint and the original run on the same engine settings, and both are started with ``nccl_nvls``.
    """
    from vllm import TokensPrompt

    settings = gencheck_settings(model, config)
    # The engine sizes its memory profile (a hybrid model's per-sequence state, the cudagraph sizes) for
    # max_num_seqs sequences; the check runs exactly its prompts, all in one batch either way (11-13: the 27B
    # original's 26 GiB profile at vLLM's default did not fit one card next to its weights).
    kwargs = {
        "tensor_parallel_size": config.tp,
        "seed": config.seed,
        "generation_config": config.original,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        **batch_args(settings.args, len(prompts)),
    }
    template_kwargs = {**(config.chat_template_kwargs or {}), "enable_thinking": config.enable_thinking}
    if "max_model_len" not in kwargs:
        # The model's own context (262144 for Qwen3.8-27B) does not fit one card next to its weights; the check
        # needs only its longest prompt plus max_tokens, and a context cap changes memory, not the answers.
        tokenizer = _context_tokenizer(model, kwargs)
        kwargs["max_model_len"] = config.max_tokens + max(
            len(prompt_token_ids(tokenizer, prompt.prompt, template_kwargs)) for prompt in prompts
        )
    llm = start_llm(settings, kwargs, model=model)
    tokenizer = llm.get_tokenizer()
    jobs = [
        TokensPrompt(prompt_token_ids=prompt_token_ids(tokenizer, prompt.prompt, template_kwargs))
        for prompt in prompts
    ]
    base = llm.get_default_sampling_params()
    base.max_tokens = config.max_tokens
    base.temperature = config.temperature
    base.seed = config.seed
    base.n = 1
    base.detokenize = True
    params = [base.clone() for _ in jobs]
    outputs = llm.generate(jobs, params, use_tqdm=False)
    return [as_generation(prompt, output) for prompt, output in zip(prompts, outputs, strict=True)]


def _context_tokenizer(model: str, kwargs: dict[str, Any]) -> Any:
    """The tokenizer the engine will load (its ``tokenizer`` argument, else the model's), before the engine."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        kwargs.get("tokenizer") or model, trust_remote_code=bool(kwargs.get("trust_remote_code", False))
    )


@command("qda-kl gencheck")
def _engine_process(request: str, response: str, failure: str | None = None) -> int:
    """One model's generations in this process (the child `_generate` starts), its engine owned by it."""
    try:
        request_data = json.loads(Path(request).read_text())
        generated = _generate_in_process(
            request_data["model"],
            [Prompt.model_validate(prompt) for prompt in request_data["prompts"]],
            GencheckConfig.model_validate(request_data["config"]),
        )
        Path(response).write_text("".join(row.model_dump_json() + "\n" for row in generated))
    except BaseException as error:
        # The parent names this line in its own error; the traceback stays on the inherited stderr.
        if failure is not None:
            try:
                Path(failure).write_text(f"{type(error).__name__}: {error}\n")
            except OSError:
                pass
        raise
    return 0


if __name__ == "__main__":
    sys.exit(_engine_process(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))


def is_loop(
    token_ids: Sequence[int],
    *,
    window: int = 2048,
    min_tokens: int = 256,
    max_period: int = 512,
    min_repeats: int = 4,
    mismatch_rate: float = 0.02,
) -> bool:
    """Detect an approximately periodic suffix of the answer's final window.

    A suffix must cover at least half the window and at least ``min_repeats``
    full periods. For each period, the most frequent token in each phase defines
    the repeating pattern; at most ``mismatch_rate`` of the tokens may differ.
    Short answers are ignored. Only complete periods ending at the answer's end
    are compared, so repetition earlier in an answer does not count.

    Args:
        token_ids: Generated token ids, excluding the prompt.
        window: Maximum tail length to inspect (default 2048).
        min_tokens: Minimum answer length to inspect (default 256).
        max_period: Largest repeating phrase length (default 512).
        min_repeats: Minimum number of complete repetitions (default 4).
        mismatch_rate: Maximum fraction differing from the pattern (default 2%).

    Returns:
        Whether a qualifying periodic suffix exists.

    Raises:
        Refused: Detector settings are outside their valid ranges.
    """
    if min(window, min_tokens, max_period) < 1 or min_repeats < 2:
        raise Refused("window, min_tokens and max_period must be positive; min_repeats must be at least 2")
    if not 0 <= mismatch_rate < 1:
        raise Refused("mismatch_rate must be in [0, 1)")
    if len(token_ids) < min_tokens:
        return False
    tail = token_ids[-window:]
    for period in range(1, min(max_period, len(tail) // min_repeats) + 1):
        counts: list[dict[int, int]] = [{} for _ in range(period)]
        maxima = [0] * period
        matches = 0
        for offset, token in enumerate(reversed(tail)):
            phase = offset % period
            count = counts[phase].get(token, 0) + 1
            counts[phase][token] = count
            if count > maxima[phase]:
                matches += 1
                maxima[phase] = count
            length = offset + 1
            if (
                length % period == 0
                and length >= min_repeats * period
                and length * 2 >= len(tail)
                and length - matches <= mismatch_rate * length
            ):
                return True
    return False
