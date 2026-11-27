"""The ``sanity`` eval suite: is a candidate worth a full score, in under ten minutes.

    suites:
      - sanity: {chat_template_kwargs: {enable_thinking: false}, max_tokens: 16384}

The suite loads ``checkpoint`` in vLLM once, answers the shipped gencheck prompts (:mod:`qda_kl.gencheck`, eight
per domain by default), and reports four checks:

* ``chat_template`` and ``generation_config``: whether the model has each (a missing template means plain-text
  prompts; a missing generation config means greedy sampling, so the other checks still run);
* ``cut_rate`` and ``loop_rate``: answers stopped at the token cap, and answers with a periodic suffix, from the
  comparison :func:`qda_kl.gencheck.build_report` makes against the original's answers;
* ``constraint_pass_rate``: the shipped instruction prompts ask for five numbered sections, a heading and an
  example in each, and a one-sentence summary; :mod:`qda_kl.eval.constraints` checks each answer. Those four rules
  belong to that prompt set, so a custom set with instruction rows is refused before an engine loads. A
  thinking model's answer is checked after its closing thinking tag (``think_end_token``): drafts in its
  thinking are not the answer, and an answer that never closes its thinking fails every rule;
* ``mtp_acceptance_rate`` and its ratio to the original's, when the checkpoint's config names MTP layers: the
  engine runs vLLM's MTP draft and :func:`qda_kl.eval.specdec.read_spec_metrics` reads the counters. A checkpoint
  that reports none while the original's record has them scores 0.0: it cannot silently drop the head.

The original is answered once in :meth:`SanitySuite.prepare`, under ``out/refs/<digest>``, so each candidate pays
one engine load and one batch of answers per seed. Both models sample with the config's settings over the
original's ``generation_config.json``, greedily when neither names a setting; several seeds refuse that greedy
fallback, since every seed would answer identically and the pooled counts would count one result several times.
Both serve with the same engine settings: the suite's ``serve_args``, since the original carries no
``build.json``. Each candidate's answers are kept as ``answers.jsonl`` beside its
``results/<checkpoint>/sanity.json``, in the original's format, so its rows can be recounted without another model
run; ``detail["answers"]`` names that file and its sha256.

With several ``seeds`` the engine answers every prompt once per seed, for the original and the candidate. The
headline numbers are the first seed's (``seed``); ``detail["per_seed"]`` holds each seed's cut and loop counts,
and ``detail["pooled"]`` pools all answers with the prompt ids that looped on each side, so a rule read over
several seeds comes from one run. Every seed is part of the original's reference key.

The W4 linear kernel can change the sampled answers, so each row names the kernel it ran with:
``detail["serve"]`` holds the ``original`` block (from the cached record) and the ``checkpoint`` block (from the
engine that just ran), each with the bound kernel classes by kind and where any backend override came from.

``max_tokens`` defaults to 16384 because a shorter cap truncates the original's own answers, and the cut and loop
checks then say little about the candidate. ``max_model_len`` reserves ``max_tokens`` plus 8192 prompt tokens, capped
by the model's context length. The engine serves every prompt in one batch: ``max_num_seqs`` is the prompt count
(:func:`qda_kl.serve.batch_args`) unless ``serve_args`` names it, and that effective batch is recorded and part of
both cache identities, so both models run it and a cached original's answers are reused only at the same batching.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, model_validator
from qda_quant.calib.selfgen import resolve_sampling, sampling_params

from qda_kl.cli import publish
from qda_kl.engine_params import DEFAULT_GPU_MEMORY_UTILIZATION
from qda_kl.eval import constraints
from qda_kl.eval.full_preset import think_end_of
from qda_kl.eval.lm_eval import answer_region
from qda_kl.eval.specdec import read_spec_metrics
from qda_kl.eval.suites import Layout, SuiteResult, locked, sha256_rows
from qda_kl.gencheck import (
    GencheckConfig,
    Generation,
    Prompt,
    as_generation,
    build_report,
    is_loop,
    prompt_token_ids,
    read_prompts,
)
from qda_kl.reference import max_model_len, versions
from qda_kl.serve import ServeSettings, backend_source, batch_args, check_tp, no_jit, vllm_version
from qda_kl.serve import llm as start_llm
from qda_kl.vllm_hooks import EXTENSION, merge_kernels

#: The original model's answers under ``out/refs/<digest>``.
ANSWERS = "answers.jsonl"
#: The sampling, prompt count and the original's spec-decode reading under ``out/refs/<digest>``.
META = "meta.json"


def _complete(out: Path) -> bool:
    """Whether the reference record has both files; ``meta.json`` is published last, so it marks completion."""
    return (out / ANSWERS).is_file() and (out / META).is_file()


def write_answers(path: Path, rows: list[Generation]) -> None:
    """Publish ``rows`` as ``answers.jsonl``: one :class:`Generation` per line, the original's format."""
    publish(path, "".join(row.model_dump_json() + "\n" for row in rows))


def _tally(rows: list[Generation]) -> dict[str, Any]:
    """Cut and loop counts of one side's answers, their rates, and the prompt ids that looped.

    ``loops`` counts answers (a prompt that loops under two seeds counts twice); ``loop_ids`` names each
    looping prompt once.
    """
    cuts = sum(row.finish_reason == "length" for row in rows)
    looped = [row.id for row in rows if is_loop(row.token_ids)]
    return {
        "n": len(rows),
        "cuts": cuts,
        "cut_rate": cuts / len(rows),
        "loops": len(looped),
        "loop_rate": len(looped) / len(rows),
        "loop_ids": sorted(set(looped)),
    }


def _served(settings: ServeSettings, kernels: dict[str, list[str]] | None) -> dict[str, Any]:
    """One half's bound kernels by kind (``dense``, ``moe``) and where each backend override came from."""
    backends = {}
    for kind in ("linear", "moe"):
        name = f"{kind}_backend"
        backends[kind] = {"value": settings.args.get(name), "source": backend_source(settings, name)}
    return {"kernels": kernels, "backends": backends}


class SanitySuite(Layout):
    """The checks every candidate passes before it earns a full score.

    Attributes:
        prompts: a shipped prompt set (``builtin``) or a JSONL path; the rows are
            :class:`qda_kl.gencheck.Prompt`.
        domains: the domains to answer (default: every domain in the prompt set).
        per_domain: prompts answered per domain.
        max_tokens: cap of one answer; long enough that the original stops on its own, since a shorter cap cuts
            the original's own answers and the cut and loop checks then see almost nothing.
        sampling: the sampling both models generate with (``temperature``, ``top_p``, ...); each field wins over
            the original's ``generation_config.json``. None: that file alone, or greedy when it names nothing
            (several ``seeds`` refuse greedy).
        chat_template_kwargs: extra chat-template arguments, such as ``{"enable_thinking": false}``.
        think_end_token: the closing thinking tag; an instruction answer is checked after its first
            occurrence, and one without it fails. Unset: the model's released card's tag, else its chat
            template's (:func:`qda_kl.eval.full_preset.think_end_of`); none when ``enable_thinking`` is
            false.
        seed: the first sampling seed, the same for both models; the headline numbers are this seed's.
        seeds: the sampling seeds, each answered by both models in the same engine (default: just ``seed``).
            The first is ``seed``. A seed is part of the original's reference key, because its answers differ.
        num_speculative_tokens: draft tokens proposed per decoding step when the model has MTP layers.
        gpu_memory_utilization: vLLM's share of the card, for both engines; lower it when other jobs share it.
        tp: tensor-parallel size of the engine (the layout's ``tp``).
        serve_args: vLLM keyword arguments, the same for the original and every candidate: the original has no
            ``build.json``, so a checkpoint's recorded serve section is not applied and the suite config is the
            one engine setting both models share. They may not name ``tensor_parallel_size`` (set ``tp``) or
            ``speculative_config`` (the suite sets the MTP draft). ``max_num_seqs`` defaults to the prompt count:
            the resolved settings and both cache identities record that effective batch.
    """

    name: ClassVar[str] = "sanity"

    prompts: str = Field(default="builtin", min_length=1)
    domains: list[str] | None = Field(default=None, min_length=1)
    per_domain: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=16384, ge=1)
    sampling: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    think_end_token: str | None = Field(default=None, min_length=1)
    seed: int = 17
    seeds: list[int] = Field(default_factory=list)
    num_speculative_tokens: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=DEFAULT_GPU_MEMORY_UTILIZATION, ge=0.05, le=0.95)

    @model_validator(mode="after")
    def _draft(self) -> SanitySuite:
        """Refuse a serve_args that competes with the suite's own draft, and resolve ``seeds``."""
        if "speculative_config" in self.serve_args:
            raise ValueError("serve_args: the draft is the model's own MTP layers; drop speculative_config")
        if not self.seeds:
            self.seeds = [self.seed]
        elif self.seeds[0] != self.seed:
            raise ValueError(f"seeds[0]={self.seeds[0]} must equal seed={self.seed}: seed is the first seed")
        elif len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds: {self.seeds} repeats a seed")
        return self

    def inputs(self) -> dict[str, str | None]:
        """SHA-256 of the prompts the suite answers (None when the prompt set refuses)."""
        try:
            rows = self._prompts()
        except ValueError:
            return {"prompts_sha256": None}
        return {"prompts_sha256": sha256_rows([row.model_dump() for row in rows])}

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings with the effective batch, which is then in ``resolved.yaml`` and the result digest.

        The batch is the prompt count unless ``serve_args`` names it, so the config alone cannot say it. A prompt
        set that refuses keeps the configured ``serve_args``: the job is refused before an engine starts.
        """
        settings = self.model_dump(mode="json")
        if settings["think_end_token"] is None:
            # resolved per model and recorded in the result; a config that does not set it keeps its digest
            del settings["think_end_token"]
        with contextlib.suppress(ValueError):
            settings["serve_args"] = self._engine_args()
        return {self.name: settings}

    def _engine_args(self) -> dict[str, Any]:
        """``serve_args`` plus the effective ``max_num_seqs``: the prompt count unless ``serve_args`` names it.

        Raises:
            ValueError: the prompt set refuses (a custom set with instruction rows).
        """
        return batch_args(self.serve_args, len(self._prompts()))

    def reference_dir(self, reference: str, work: Path) -> Path:
        """``work/refs/<digest>``: the original's answers for these prompts, sampling and layout.

        The detected MTP head count is part of the identity: the suite turns vLLM's draft on when it finds one,
        so the answers and their acceptance reading depend on it. A record written while the detector did not
        know a model family's name for the head (:func:`mtp_layers`) is therefore not reused after the detector
        learns it; the original is answered again and its acceptance rate recorded. The effective engine settings
        (including a derived ``max_num_seqs``) are part of it too, so answers taken at another batching are never
        reused as a pair for a candidate running at this one. Every seed is part of the identity as well: a
        record holds the answers of exactly the seeds it was measured with.
        """
        key = {
            "suite": self.name,
            "reference": reference,
            "prompts_sha256": self.inputs()["prompts_sha256"],
            "sampling": self.sampling,
            "chat_template_kwargs": self.chat_template_kwargs,
            "seeds": self.seeds,
            "max_tokens": self.max_tokens,
            "num_speculative_tokens": self.num_speculative_tokens,
            "mtp_layers": _mtp_layers(reference),
            "tp": self.tp,
            "serve_args": self._settings().args,
            "serve_env": self._settings().env,
        }
        return work / "refs" / hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    def prepare(self, reference: str, work: Path) -> None:
        """Answer the prompts on the original once per seed and cache them under :meth:`reference_dir`.

        Holds ``work/refs/<digest>.lock`` while it checks and builds, so parallel ``--only`` jobs sharing one
        ``out`` answer the original once. A record left incomplete by a killed run (both files are published
        separately, ``meta.json`` last) is replaced under the lock instead of being skipped. Sampling is resolved
        here, from the original alone, because both models must generate with the same settings; it is resolved
        before the record is checked, so a greedy record built before the refusal existed is refused too.
        """
        out = self.reference_dir(reference, work)
        sampling = self._sampling(reference)
        with locked(out.with_suffix(".lock")):
            if _complete(out):
                return
            for name in (ANSWERS, META):
                (out / name).unlink(missing_ok=True)
            prompts = self._prompts()
            settings = self._settings()
            rows, got = self._measure(reference, settings, prompts, sampling)
            out.mkdir(parents=True, exist_ok=True)
            write_answers(out / ANSWERS, rows)
            publish(
                out / META,
                json.dumps(
                    {
                        "sampling": sampling,
                        "serve": settings.args,
                        "kernels": got["kernels"],
                        "specdec": got["specdec"],
                        "prompts": len(prompts),
                    }
                )
                + "\n",
            )

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        """Answer the prompts on ``checkpoint`` and report the checks against the original's answers.

        Raises:
            RuntimeError: the original's answers are missing, or a generation did not complete normally.
        """
        stored = self.reference_dir(reference, work) / ANSWERS
        try:
            meta = json.loads((stored.parent / META).read_text())
            original = [Generation.model_validate_json(line) for line in stored.read_text().splitlines()]
        except (OSError, ValueError) as why:
            raise RuntimeError(
                f"the original model's sanity-check answers are missing ({stored.parent}): {why}"
            ) from None
        prompts = self._prompts()
        settings = self._settings()
        rows, got = self._measure(checkpoint, settings, prompts, meta["sampling"])
        answers = self._keep_answers(rows, work)

        checkpoint_by_seed = self._by_seed(rows, len(prompts))
        original_by_seed = self._by_seed(original, len(prompts))
        first = checkpoint_by_seed[self.seed]
        config = self._config(reference, checkpoint)
        report = build_report(first, original_by_seed[self.seed], config, settings)
        total = report["total"]["checkpoint"]
        metrics = {
            "cut_rate": total["cut"] / total["n"],
            "loop_rate": total["loops"] / total["n"],
            "chat_template": float(got["chat_template"]),
            "generation_config": float(got["generation_config"]),
        }
        higher = {"cut_rate": False, "loop_rate": False, "chat_template": True, "generation_config": True}
        detail: dict[str, Any] = {
            "gencheck": report,
            "sampling": meta["sampling"],
            "seed": self.seed,
            "prompts": {"rows": len(prompts), "sha256": sha256_rows([row.model_dump() for row in prompts])},
            "load_s": got["load_s"],
            "vllm": vllm_version(),
            "versions": versions(),
            "serve": {
                **settings.model_dump(mode="json"),
                "original": _served(settings, meta.get("kernels")),
                "checkpoint": _served(settings, got.get("kernels")),
            },
        }
        if answers is not None:
            detail["answers"] = answers
        think = self._think_end(checkpoint)
        scored = constraints.score(
            [answer_region(row.text, think) for row in first if row.domain == "instruction"]
        )
        if scored is not None:
            metrics["constraint_pass_rate"] = scored["instruction_rate"]
            higher["constraint_pass_rate"] = True
            detail["constraints"] = {**scored, "think_end_token": think}
        reading, base = got["specdec"], meta["specdec"]
        if reading is not None or base is not None:
            # a checkpoint that dropped the original's MTP head scores 0.0 rather than no number
            rate = reading["acceptance_rate"] if reading is not None else 0.0
            metrics["mtp_acceptance_rate"] = rate
            higher["mtp_acceptance_rate"] = True
            if (base or {}).get("acceptance_rate"):
                metrics["mtp_acceptance_rate_ratio"] = rate / base["acceptance_rate"]
                higher["mtp_acceptance_rate_ratio"] = True
            detail["specdec"] = {"checkpoint": reading, "original": base}
            if reading is None:
                detail["specdec"]["missing"] = (
                    "the original has an MTP head; the checkpoint reported no MTP counters"
                )
        if len(self.seeds) > 1:
            # The gate's pooled-seed rule reads these; the suite only reports them.
            detail["per_seed"] = [
                {
                    "seed": seed,
                    "checkpoint": _tally(checkpoint_by_seed[seed]),
                    "original": _tally(original_by_seed[seed]),
                }
                for seed in self.seeds
            ]
            detail["pooled"] = {"checkpoint": _tally(rows), "original": _tally(original)}
        return SuiteResult(suite=self.name, metrics=metrics, higher_is_better=higher, detail=detail)

    def _keep_answers(self, rows: list[Generation], work: Path) -> dict[str, Any] | None:
        """Write the candidate's answers beside its result, like the original's under ``refs/<digest>``.

        The suite keeps no candidate text otherwise, so a reader could not recount a candidate's own rows
        (:func:`write_answers` is the writer both use). Returns the file's path under ``work`` and its sha256,
        or None when the runner did not name a result file (a direct call in a test or a tool).
        """
        if self.result_path is None:
            return None
        path = self.result_path.parent / ANSWERS
        path.parent.mkdir(parents=True, exist_ok=True)
        write_answers(path, rows)
        return {
            "path": str(path.relative_to(work)) if path.is_relative_to(work) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _by_seed(self, rows: list[Generation], prompts: int) -> dict[int, list[Generation]]:
        """Split seed-major answers into one prompt set per seed.

        Raises:
            RuntimeError: the answers are not one full set per seed, so a seed's report would compare the
                wrong answers.
        """
        expected = prompts * len(self.seeds)
        if len(rows) != expected:
            raise RuntimeError(
                f"{len(rows)} answers for {prompts} prompts and {len(self.seeds)} seeds, not {expected}"
            )
        return {seed: rows[index * prompts : (index + 1) * prompts] for index, seed in enumerate(self.seeds)}

    def _config(self, reference: str, checkpoint: str | None = None) -> GencheckConfig:
        """The gencheck config of these prompts and limits; checkpoint and original only need to be known."""
        return GencheckConfig(
            checkpoint=checkpoint or reference,
            original=reference,
            prompts=self.prompts,
            domains=self.domains,
            per_domain=self.per_domain,
            max_tokens=self.max_tokens,
            seed=self.seed,
            chat_template_kwargs=self.chat_template_kwargs,
            out=".",
        )

    def _prompts(self) -> list[Prompt]:
        """The prompts of this suite, in source-file order; :class:`GencheckConfig` validates the selection.

        Raises:
            ValueError: a custom prompt set has instruction rows; the four shipped rules are not its rules.
        """
        rows = read_prompts(self._config(self.name))
        if self.prompts != "builtin" and any(row.domain == "instruction" for row in rows):
            raise ValueError(
                f"prompts: {self.prompts!r} has instruction rows, but the sanity check's constraint rules are the "
                "shipped set's; use prompts: builtin, or drop the instruction domain from the custom set"
            )
        return rows

    def _settings(self) -> ServeSettings:
        """The one engine setting every row uses: this suite's ``serve_args``, for the original and candidates.

        The original has no ``build.json``, so a checkpoint's recorded serve section cannot apply to it; applying
        it to the candidate alone would compare answers from two engines. The suite config is what both share,
        including the effective ``max_num_seqs`` (:meth:`_engine_args`), so the recorded settings name the batch
        the engines actually ran and a cached original is only reused at that same batch. Like every scoring and
        gencheck engine it starts under :func:`qda_kl.serve.no_jit`, so no FlashInfer kernel is compiled at run time.
        """
        return no_jit(ServeSettings(args=self._engine_args()))

    def _think_end(self, model: str) -> str | None:
        """The tag ``model``'s answers are cut at: the setting, else none with thinking off, else its own."""
        if self.think_end_token is not None:
            return self.think_end_token
        if self.chat_template_kwargs.get("enable_thinking") is False:
            return None
        return think_end_of(model, [model])  # a checkpoint's own files may name its original's card

    def _sampling(self, reference: str) -> dict[str, Any]:
        """The sampling both models use; greedy when neither an override nor the generation config names one.

        Raises:
            ValueError: several seeds with greedy decoding: every seed would answer identically, so the pooled
                counts would count one result several times.
        """
        try:
            sampling = resolve_sampling(reference, self.sampling, "sanity.sampling")
        except ValueError:
            if self.sampling is not None:
                raise
            sampling = {"temperature": 0.0}
        if len(self.seeds) > 1 and not sampling.get("temperature"):
            raise ValueError(
                f"seeds: {self.seeds} with greedy decoding give identical answers, so the pooled counts would "
                'count one result several times; set sanity.sampling, e.g. \'{"temperature": 1.0, "top_p": 0.95}\''
            )
        return sampling

    def _measure(
        self, model: str, settings: ServeSettings, prompts: list[Prompt], sampling: dict[str, Any]
    ) -> tuple[list[Generation], dict[str, Any]]:
        """Answer ``prompts`` on ``model`` in one engine, once per seed, on the kernels ``settings`` plan.

        The engine loads once; each seed is one more batch. The answers are seed-major: one prompt set per
        seed, in :attr:`seeds` order. The spec-decode counters are cumulative over the engine, so they are read
        after the first seed: the headline numbers are that seed's, as before ``seeds``.
        """
        from vllm import TokensPrompt

        t0 = time.time()
        mtp = _mtp_layers(model)
        kwargs = {
            "tensor_parallel_size": self.tp,
            "max_model_len": max_model_len(model, self.max_tokens),
            "seed": self.seed,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "worker_extension_cls": EXTENSION,  # reads the kernels the engine bound, below
            **settings.args,
        }
        check_tp(kwargs, self.tp, ValueError)
        if mtp:
            kwargs["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": self.num_speculative_tokens,
            }
            kwargs["disable_log_stats"] = False
        llm = start_llm(settings, kwargs, model=model)
        kernels = merge_kernels(llm.collective_rpc("qda_kl_kernels"))
        loaded = time.time()
        tokenizer = llm.get_tokenizer()
        chat = getattr(tokenizer, "chat_template", None) is not None
        jobs = [
            TokensPrompt(
                prompt_token_ids=prompt_token_ids(tokenizer, row.prompt, self.chat_template_kwargs, chat=chat)
            )
            for row in prompts
        ]
        rows: list[Generation] = []
        specdec: dict[str, Any] | None = None
        # A closing tag that is a special token (Gemma's) is stripped by vLLM's default decode, leaving no
        # boundary for answer_region() to cut at; keep specials only when the answers are cut.
        skip = self._think_end(model) is None
        for seed in self.seeds:
            outputs = llm.generate(
                jobs,
                [sampling_params(sampling, self.max_tokens, seed, skip_special_tokens=skip) for _ in jobs],
                use_tqdm=False,
            )
            rows += [as_generation(prompt, output) for prompt, output in zip(prompts, outputs, strict=True)]
            if mtp and specdec is None:  # the counters are cumulative: this is the first seed's reading
                specdec = read_spec_metrics(llm.get_metrics())
        got: dict[str, Any] = {
            "chat_template": chat,
            "generation_config": (Path(model) / "generation_config.json").is_file(),
            "load_s": round(loaded - t0, 1),
            "kernels": kernels,
            "specdec": specdec,
        }
        return rows, got


#: The names a model family gives its multi-token-prediction layer count: DeepSeek and Kimi declare
#: ``num_nextn_predict_layers``, Qwen3.5/3.8 declare ``mtp_num_hidden_layers`` on the text config.
MTP_HEAD_ATTRS = ("num_nextn_predict_layers", "mtp_num_hidden_layers")


def _mtp_layers(model: str) -> int:
    """The model's multi-token-prediction layer count, 0 when it has none (see :func:`mtp_layers`)."""
    from transformers import AutoConfig

    return mtp_layers(AutoConfig.from_pretrained(model))


def mtp_layers(config: Any) -> int:
    """The MTP head count a model config declares, under either family name.

    The head is named on the language model's config, which for a multimodal model is a sub-config: it is read
    through the config interface (``get_text_config()``), with the ``text_config`` / ``llm_config`` attributes as
    the fallback for a config object that has no method, so no family's own attribute is the only name known.

    Args:
        config: a Hugging Face model config (``transformers.AutoConfig.from_pretrained``) or any object with the
            same attributes; nested parts are read when the outer config does not name the head.

    Returns:
        The head's layer count; 0 when the config names no head.
    """
    parts: list[Any] = [config]
    text_of = getattr(config, "get_text_config", None)
    if callable(text_of):
        # A config naming several text parts raises: the attributes below pick one instead.
        with contextlib.suppress(ValueError):
            parts.append(text_of())
    parts += [getattr(config, "text_config", None), getattr(config, "llm_config", None)]
    seen: list[Any] = []
    for part in parts:
        if part is None or any(part is other for other in seen):
            continue
        seen.append(part)
        for attr in MTP_HEAD_ATTRS:
            layers = int(getattr(part, attr, 0) or 0)
            if layers:
                return layers
    return 0
