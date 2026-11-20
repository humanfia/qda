"""The ``livecodebench`` eval suite: LiveCodeBench v6 answers graded in bubblewrap sandboxes.

The suite generates ``repeats`` answers per problem with vLLM and grades each distinct answer. ``protocol``
picks the prompt and the case runner: ``official`` (the default) asks LiveCodeBench's own prompt and grades
with the official test runner (:mod:`qda_agent.third_party.livecodebench`) at ``timeout_s`` wall-clock seconds per
case; ``paper`` asks the paper's prompt and grades as described in the paper's Appendix B: every stdin case in a
real child interpreter, so code that reads ``sys.stdin.buffer`` or writes ``sys.stdout.buffer`` runs as it
would at a contest, and ``timeout_s`` CPU seconds per case with a wall-clock backstop (the paper used 8).
Every program runs in its own bubblewrap mount namespace built from an empty root: no network, only the
interpreter's trees mounted read-only, one scratch directory writable; ``sandbox`` chooses that: ``auto`` (the
default) sandboxes where bubblewrap is installed and stops with the fix where it is not, ``bwrap`` requires
it, and ``none`` runs the programs on this machine, unsandboxed, after one warning line. Grading uses 16
workers by default and retries timed-out programs once alone, after the parallel pass finishes. The engine
and graders use disjoint CPU sets. The sandbox carries no GPU. Resource caps inside it limit one program to
4 GiB, 64 processes and 64 MiB per file, and the tree is an allowlist, so anything not named is absent rather
than hidden. The sandbox's /proc is fresh by default (``proc: private``); a node whose container refuses a
fresh proc mount needs the explicit ``proc: host-readonly`` setting, which binds the host's /proc read-only.

Problems are LiveCodeBench v6: ``contest_date`` in ``[2024-08-01, 2025-05-01)``, which is 454 of the release's
1055. The test cases come from ``livecodebench/code_generation_lite`` at a pinned revision and are downloaded
into the data directory, never committed. The first three release files end before the window, so only
``test3.jsonl`` .. ``test6.jsonl`` are fetched. ``max_problems`` keeps a smoke run small; it changes the score
and is therefore part of the config digest.

Generation serves the checkpoint with the settings its ``build.json`` names
(:func:`qda_kl.serve.settings_for`) and samples with :func:`qda_quant.calib.selfgen.resolve_sampling`, so the
checkpoint's own ``generation_config.json`` applies unless the config overrides it. ``enable_thinking`` names the
chat template's thinking mode, as in the ``lm_eval`` suite. A configured ``think_end_token`` cuts each generation
to its answer region (:func:`qda_kl.eval.lm_eval.answer_region`) first, as ``lm_eval`` does: an unfinished
thinking block therefore scores nothing. The answer is the last fenced code block of that region, as
LiveCodeBench extracts it.

``detail["livecodebench_v6"]`` retains scores per question id and draw, and counts length-capped generations
with no extracted code per draw. Raw generations are saved as gzip JSONL under the run's work directory.

A suite measures one checkpoint and returns a :class:`SuiteResult`; it raises ``ValueError`` or ``RuntimeError``
to refuse, and then no result file is written. The dataset and prompts are in
:mod:`qda_agent.livecodebench_data`; the case sandbox and the trusted grader in
:mod:`qda_agent.livecodebench_sandbox`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field
from qda_kl.eval.lm_eval import answer_region
from qda_kl.eval.samples import write_samples
from qda_kl.eval.suites import Suite, SuiteResult
from qda_quant.calib.selfgen import resolve_sampling, sampling_params
from qda_quant.io import qda_cache

from qda_agent.livecodebench_data import (
    FILES,
    ITEM_VERSION,
    PROBLEMS,
    REVISION,
    SYSTEM,
    WINDOW,
    cases_of,
    code_of,
    download,
    in_window,
    item_of,
    rows,
)
from qda_agent.livecodebench_sandbox import GRADER, PAPER_RUNNER, VENDOR, grade_code

#: The ``SamplingParams`` fields a completions request carries as JSON. Every sampling setting the checkpoints
#: resolve is here; a field added to one path only would be a different sampler, so extend this deliberately.
SERVED_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "length_penalty",
    "min_tokens",
    "stop",
    "include_stop_str_in_output",
    "ignore_eos",
    "spaces_between_special_tokens",
    "n",
    "logprobs",
)

#: Requests in flight to the server at once. The server schedules; the client only keeps the queue short.
SERVED_CONCURRENCY = 64

#: The one line a run prints before grading generated code with no sandbox (``sandbox: none``).
UNSANDBOXED_WARNING = (
    "WARNING: livecodebench: sandbox none: the model's generated code runs unsandboxed on this machine, "
    "with your files and your network; run it only on code and a machine you accept that for"
)


class LiveCodeBenchSuite(Suite):
    """LiveCodeBench v6 pass@1, every program in a bubblewrap sandbox.

    Attributes:
        repeats: answers generated per problem (the paper uses 4); pass@1 averages over them.
        max_tokens: cap of one answer (the paper used 81920 on the Qwen models).
        timeout_s: limit of one test case, in seconds: wall clock under ``official`` (the official default 6),
            CPU time under ``paper`` (the paper used 8).
        protocol: ``official``: LiveCodeBench's prompt (system message included) and test runner. ``paper``:
            the paper's prompt (one user message) and runner (stdin cases in child processes, CPU-time limits),
            which reproduce its LiveCodeBench column.
        workers: concurrent graders (default 16); timed-out programs are retried once with one worker.
        grading_cpus: dedicated logical CPUs for grading. None reserves up to ``workers`` physical cores,
            leaving at least one for generation. Explicit IDs must be available and leave engine cores.
        seed: seed of draw 0; draw ``d`` uses ``seed + d``.
        sampling: sampling settings; its fields win over the checkpoint's ``generation_config.json``.
        enable_thinking: the chat template's thinking mode (default on, as the benchmark models' cards
            show); set false for a non-thinking model.
        think_end_token: the model's closing thinking tag; only the text after its first occurrence is graded,
            and a generation that never closes its thinking has no code (and scores nothing). Unset: the whole
            generation, as a non-thinking checkpoint produces it.
        chat_template_kwargs: extra chat-template arguments (``enable_thinking`` here is ignored; use the
            named field).
        serve_env: engine environment, over ``build.json``'s ``serve.env``.
        serve_args: ``vllm.LLM`` keyword arguments, over ``serve.args`` (e.g. ``gpu_memory_utilization``).
        server: base URL of a running vLLM completions server to generate on, instead of an engine in this
            process (:mod:`qda_kl.eval.served`). The prompts are rendered by the checkpoint's own tokenizer
            and sent as the same ids, with the same sampling as the in-process path; the engine path joins
            the config digest (a served run is a different measurement), the URL does not.
        data: directory holding the downloaded dataset; default ``$QDA_CACHE/livecodebench/<revision>``.
        max_problems: grade only this many window problems (a smoke run); default all 454.
        sandbox: what runs the generated programs. ``auto`` (the default) uses bubblewrap when it is
            installed and refuses when it is not, naming ``none`` as the way on; ``bwrap`` requires
            bubblewrap; ``none`` runs them on this machine, unsandboxed, after one warning line. Recorded in
            the result, and part of the config digest: the two are not the same measurement.
        proc: the /proc the grading sandbox sees. ``private`` mounts a fresh procfs, so a program sees only
            the sandbox's own processes; ``host-readonly`` binds the host's /proc read-only instead, for a
            node whose container refuses a fresh mount (then the host's process table is visible under the
            same uid's access checks). There is no automatic fallback: a node refuses in ``private`` and the
            operator opts in explicitly. Recorded in the result.
    """

    name: ClassVar[str] = "livecodebench"
    repeats: int = Field(default=4, ge=1)
    max_tokens: int = Field(default=65536, ge=1)
    timeout_s: int = Field(default=6, ge=1)
    protocol: Literal["official", "paper"] = "official"
    workers: int = Field(default=16, ge=1)
    grading_cpus: list[int] | None = None
    seed: int = 17
    sampling: dict[str, Any] | None = None
    enable_thinking: bool = True
    think_end_token: str | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    serve_env: dict[str, str] = Field(default_factory=dict)
    serve_args: dict[str, Any] = Field(default_factory=dict)
    server: str | None = Field(default=None, exclude=True)
    data: str | None = None
    max_problems: int | None = Field(default=None, ge=1)
    sandbox: Literal["auto", "bwrap", "none"] = "auto"
    proc: Literal["private", "host-readonly"] = "private"

    def backend(self) -> str:
        """What ``sandbox`` resolves to here: ``bwrap`` or ``none``.

        Raises:
            ValueError: bubblewrap is wanted (``bwrap``, or ``auto`` on a machine without it) and missing.
        """
        if self.sandbox == "none":
            return "none"
        if shutil.which("bwrap") is not None:
            return "bwrap"
        if self.sandbox == "bwrap":
            raise ValueError("livecodebench needs bubblewrap (bwrap) to grade test cases in a sandbox")
        raise ValueError(
            "livecodebench: bubblewrap (bwrap) is not installed, so the generated code cannot be "
            "sandboxed; install it (apt install bubblewrap), or accept running generated code "
            "unsandboxed on this machine with --lcb-sandbox none (sandbox: none on the suite)"
        )

    def inputs(self) -> dict[str, str | None]:
        """The pinned revision, window and grader code, so changing any of them makes old results stale."""
        return {
            "revision": REVISION,
            "window": f"{WINDOW[0]}..{WINDOW[1]}",
            "grader_sha256": hashlib.sha256(GRADER.read_bytes()).hexdigest(),
            "paper_runner_sha256": hashlib.sha256(PAPER_RUNNER.read_bytes()).hexdigest(),
            "testing_util_sha256": hashlib.sha256((VENDOR / "testing_util.py").read_bytes()).hexdigest(),
        }

    def _code_of(self, generation: str | None) -> str | None:
        """The program to grade: the last fence of the answer region, never of unfinished thinking."""
        if generation is None:
            return None
        return code_of(answer_region(generation, self.think_end_token))

    def data_dir(self) -> Path:
        """The directory the test cases are downloaded into."""
        if self.data:
            return Path(self.data)
        cache = qda_cache()
        return Path(cache) / "livecodebench" / REVISION

    def prepare(self, reference: str, work: Path) -> None:  # noqa: ARG002 - fixed Suite interface
        """Preflight the real grader, then download and select the v6 problems under ``work/livecodebench``.

        Raises:
            ValueError: ``sandbox`` asks for bubblewrap and it is not installed.
            RuntimeError: the preflight fails, a download fails, or the full window's count is not 454.
        """
        backend = self.backend()
        if backend == "none":  # lm_eval's --confirm_run_unsafe_code: said once, before any program runs
            print(UNSANDBOXED_WARNING, flush=True)
        root = Path(work) / "livecodebench"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="preflight-", dir=root) as scratch:
            if grade_code(
                "print(int(input()) * 2)",
                None,
                [{"input": "21\n", "output": "42\n"}],
                Path(scratch),
                self.timeout_s,
                run_dir=Path(work),
                host_written=[Path(scratch)],
                proc=self.proc,
                protocol=self.protocol,
                sandbox=backend,
            ) != (True, 1, 0, 0):
                raise RuntimeError(
                    "livecodebench: refused: the preflight could not grade a trivial program; "
                    "repair the grader's Python/NumPy runtime or use a node where the sandbox works"
                )
        key = {
            "version": ITEM_VERSION,
            "revision": REVISION,
            "window": list(WINDOW),
            "max_problems": self.max_problems,
            "protocol": self.protocol,
        }
        selection = root / "items.json"
        data = self.data_dir()
        if selection.is_file():
            cached = json.loads(selection.read_text())
            present = all((data / f["name"]).is_file() for f in cached.get("files", []))
            if cached.get("key") == key and present:
                return
        data.mkdir(parents=True, exist_ok=True)
        items, files = self._select(data)
        if self.max_problems is None and len(items) != PROBLEMS:
            raise RuntimeError(
                f"livecodebench: {len(items)} problems in {WINDOW} at revision {REVISION}, "
                f"expected {PROBLEMS}"
            )
        selection.write_text(
            json.dumps(
                {
                    "key": key,
                    "revision": REVISION,
                    "window": list(WINDOW),
                    "files": files,
                    "items": items,
                }
            )
        )

    def _select(self, data: Path) -> tuple[list[dict], list[dict]]:
        """Stream the candidate files in order and collect window problems, stopping at ``max_problems``."""
        items: list[dict] = []
        files: list[dict] = []
        seen: set[str] = set()
        for name in FILES:
            path = download(data, name)
            files.append({"name": name, "bytes": path.stat().st_size})
            for row in rows(path):
                if not in_window(row):
                    continue
                ident = str(row["question_id"])
                if ident in seen:
                    continue
                seen.add(ident)
                items.append(item_of(row, self.protocol))
                if self.max_problems is not None and len(items) >= self.max_problems:
                    return items, files
        return items, files

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:  # noqa: ARG002
        """Generate ``repeats`` answers per problem and grade each distinct answer.

        Raises:
            RuntimeError: the selection is missing, or the sandbox itself failed.
            ValueError: ``sandbox`` asks for a missing bubblewrap, or the sampling settings are invalid.
        """
        backend = self.backend()
        selected = Path(work) / "livecodebench" / "items.json"
        if not selected.is_file():
            raise RuntimeError(f"livecodebench: {selected} is missing; run prepare first")
        selection = json.loads(selected.read_text())
        items = selection["items"]
        missing = [
            f["name"] for f in selection.get("files", []) if not (self.data_dir() / f["name"]).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"livecodebench: dataset files {missing} are missing from {self.data_dir()}; run prepare again"
            )
        identity = self._served_identity(checkpoint) if self.server else None
        t0 = time.time()
        available = os.sched_getaffinity(0)
        grading_cpus = self._grading_cpus()
        engine_cpus = available - self._siblings(grading_cpus)
        # Existing interpreter threads and newly spawned engine processes must avoid the grading CPUs.
        saved = {
            int(task.name): os.sched_getaffinity(int(task.name)) for task in Path("/proc/self/task").iterdir()
        }
        try:
            for tid in saved:
                os.sched_setaffinity(tid, engine_cpus)
            answers, finishes, prompt_hashes = self._generate(checkpoint, [one["prompt"] for one in items])
        finally:
            for tid, cpus in saved.items():
                with contextlib.suppress(ProcessLookupError):
                    os.sched_setaffinity(tid, cpus)
        t_gen = time.time() - t0
        scratch = Path(work) / "livecodebench" / "scratch"
        got, grading = self._grade(items, answers, scratch, run_dir=Path(work), sandbox=backend)
        scored = len(got)
        means = {ident: sum(draws) / self.repeats for ident, draws in got.items()}
        value = sum(means.values()) / scored if scored else 0.0
        by_difficulty: dict[str, list[float]] = {}
        for one in items:
            by_difficulty.setdefault(str(one.get("difficulty")), []).append(means[one["id"]])
        raw = write_samples(
            work,
            checkpoint,
            self.name,
            self.model_dump(mode="json"),
            (
                {
                    "question_id": one["id"],
                    "seed": self.seed + draw,
                    "text": text,
                    "finish_reason": finishes[i][draw],
                    "score": got[one["id"]][draw],
                }
                for i, one in enumerate(items)
                for draw, text in enumerate(answers[i])
            ),
        )
        detail = {
            **grading,
            "engine": "server" if self.server else "in-process",
            "problems": scored,
            "generations": sum(len(v) for v in answers),
            "seed": self.seed,
            "repeats": self.repeats,
            "sandbox": backend,
            "proc": self.proc,
            "protocol": self.protocol,
            "workers": self.workers,
            "grading_cpus": sorted(grading_cpus),
            "engine_cpus": sorted(engine_cpus),
            "timeout_s": self.timeout_s,
            "revision": REVISION,
            "window": list(WINDOW),
            "data_files": selection.get("files", []),
            "pass_at_1_by_difficulty": {level: sum(v) / len(v) for level, v in sorted(by_difficulty.items())},
            "generate_s": round(t_gen, 1),
            "total_s": round(time.time() - t0, 1),
            "livecodebench_v6": {
                "per_question": got,
                "prompt_hashes": {
                    one["id"]: digest for one, digest in zip(items, prompt_hashes, strict=True)
                },
                "cap_hits": [
                    sum(
                        finishes[i][draw] == "length" and self._code_of(answers[i][draw]) is None
                        for i in range(len(items))
                    )
                    for draw in range(self.repeats)
                ],
                "raw_generations": str(raw.relative_to(work)),
            },
        }
        if identity is not None:
            # The engine that generated, read from the server: this client process need not import vLLM.
            detail["server"] = identity
            detail["vllm"] = identity["vllm"]
        return SuiteResult(
            suite=self.name,
            metrics={"livecodebench_v6": value},
            higher_is_better={"livecodebench_v6": True},
            detail=detail,
        )

    # ------------------------------------------------------------------ generation

    def _generate(
        self, checkpoint: str, prompts: Sequence[str]
    ) -> tuple[list[list[str | None]], list[list[str | None]], list[str]]:
        """Answer each prompt ``repeats`` times, retaining vLLM's finish reason for each draw.

        The third return value hashes each prompt as the chat template renders it, so the result records what
        every row asked and a paired comparison can refuse two rows that saw different prompts.

        Raises:
            RuntimeError: the engine's tokenizer encodes the probe differently from the ``tokenizer.json`` of
                the checkpoint (or of the ``tokenizer=`` override the engine loads), so every graded
                text would be the wrong text.
        """
        if self.server:
            return self._served_generate(checkpoint, prompts)
        from qda_kl.serve import settings_for, start_engine
        from qda_kl.tokenizer_check import tokenizer_refusal
        from vllm import LLM, TokensPrompt

        sampling = resolve_sampling(checkpoint, self.sampling, "livecodebench.sampling")
        serve = settings_for(checkpoint, self.serve_env, self.serve_args)
        llm = start_engine(serve, lambda: LLM(model=checkpoint, **serve.args))
        tok = llm.get_tokenizer()
        source = serve.args.get("tokenizer") or checkpoint  # the engine's own tokenizer, when overridden
        revision = serve.args.get("tokenizer_revision") or serve.args.get("revision")
        refusal = tokenizer_refusal(tok, source, revision=revision)
        if not refusal and prompts:
            refusal = tokenizer_refusal(tok, source, revision=revision, probe=prompts[0])
        if refusal:
            raise RuntimeError(refusal)
        template_kwargs = {**self.chat_template_kwargs, "enable_thinking": self.enable_thinking}
        prompts_ids, prompt_hashes = self._render(tok, template_kwargs, prompts)
        jobs = [(ids, draw) for ids in prompts_ids for draw in range(self.repeats)]
        flat = [TokensPrompt(prompt_token_ids=ids) for ids, _ in jobs]
        # A configured closing thinking tag is a special token, and answer_region() cuts at it; vLLM's default
        # (skip_special_tokens=True) strips it at decode time, leaving no boundary to cut at. Keep specials only
        # for a suite that cuts at one: a cell with no think_end_token keeps vLLM's default and its old records.
        params = [
            sampling_params(
                sampling, self.max_tokens, self.seed + draw, skip_special_tokens=self.think_end_token is None
            )
            for _, draw in jobs
        ]
        said = llm.generate(flat, params, use_tqdm=False)
        out: list[list[str | None]] = []
        finishes: list[list[str | None]] = []
        for i in range(len(prompts)):
            drawn = [
                said[i * self.repeats + draw].outputs[0].text
                if said[i * self.repeats + draw].outputs
                else None
                for draw in range(self.repeats)
            ]
            out.append(drawn)
            finishes.append(
                [
                    said[i * self.repeats + draw].outputs[0].finish_reason
                    if said[i * self.repeats + draw].outputs
                    else None
                    for draw in range(self.repeats)
                ]
            )
        return out, finishes, prompt_hashes

    def _served_identity(self, checkpoint: str) -> dict[str, Any]:
        """What the server says it is, refusing a server that is not the checkpoint's own engine.

        Raises:
            qda_kl.eval.served.Refused: the server names no vLLM version, runs another version than the
                checkpoint's settings name, or serves another checkpoint.
        """
        from qda_kl.eval.served import check_server_identity, server_identity
        from qda_kl.serve import settings_for

        serve = settings_for(checkpoint, self.serve_env, self.serve_args)
        identity = server_identity(str(self.server))
        check_server_identity(identity, checkpoint, serve.vllm)
        return identity

    def _render(
        self, tok: Any, template_kwargs: dict[str, Any], prompts: Sequence[str]
    ) -> tuple[list[list[int]], list[str]]:
        """Each prompt's token ids and rendered-text hash, exactly as the in-process engine asks for them.

        Both generation paths call this, so the served path cannot drift from the in-process one: the ids and
        the hashes a served row records are computed by the same code.
        """
        prompts_ids: list[list[int]] = []
        prompt_hashes: list[str] = []
        for text in prompts:
            messages = [
                *([{"role": "system", "content": SYSTEM}] if self.protocol == "official" else []),
                {"role": "user", "content": text},
            ]
            rendered = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            prompt_hashes.append(hashlib.sha256(rendered.encode()).hexdigest())
            got = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **template_kwargs,
            )
            prompts_ids.append(list(got["input_ids"] if hasattr(got, "keys") else got))
        return prompts_ids, prompt_hashes

    def _served_payload(self, ids: Sequence[int], draw: int, sampling: dict[str, Any]) -> dict[str, Any]:
        """One completions request: the in-process ``SamplingParams`` as a JSON payload, field for field.

        The payload is built from the same object the in-process engine is given, so a sampling setting
        cannot apply to one path and not the other; a field this list does not carry is one the in-process
        engine was never given either.
        """
        from qda_kl.eval.served import SERVED_NAME

        params = sampling_params(
            sampling, self.max_tokens, self.seed + draw, skip_special_tokens=self.think_end_token is None
        )
        payload: dict[str, Any] = {"model": SERVED_NAME, "prompt": list(ids)}
        for name in SERVED_SAMPLING_FIELDS:
            value = getattr(params, name, None)
            if value is not None:
                payload[name] = value
        # Named even at their defaults: the in-process engine was given exactly these.
        payload["max_tokens"] = int(params.max_tokens)
        payload["seed"] = int(params.seed)
        payload["skip_special_tokens"] = bool(params.skip_special_tokens)
        return payload

    def _served_generate(
        self, checkpoint: str, prompts: Sequence[str]
    ) -> tuple[list[list[str | None]], list[list[str | None]], list[str]]:
        """Answer each prompt ``repeats`` times on the server, sending the in-process path's own requests.

        Raises:
            RuntimeError: the tokenizer check refuses (see :meth:`_generate`), or the server answers a
                completion without a choice.
        """
        import transformers
        from qda_kl.eval.served import post_completions
        from qda_kl.serve import settings_for
        from qda_kl.tokenizer_check import tokenizer_refusal

        sampling = resolve_sampling(checkpoint, self.sampling, "livecodebench.sampling")
        serve = settings_for(checkpoint, self.serve_env, self.serve_args)
        source = serve.args.get("tokenizer") or checkpoint  # the engine's own tokenizer, when overridden
        revision = serve.args.get("tokenizer_revision") or serve.args.get("revision")
        tok = transformers.AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=bool(serve.args.get("trust_remote_code", False)),
            revision=revision,
        )
        refusal = tokenizer_refusal(tok, source, revision=revision)
        if not refusal and prompts:
            refusal = tokenizer_refusal(tok, source, revision=revision, probe=prompts[0])
        if refusal:
            raise RuntimeError(refusal)
        template_kwargs = {**self.chat_template_kwargs, "enable_thinking": self.enable_thinking}
        prompts_ids, prompt_hashes = self._render(tok, template_kwargs, prompts)
        jobs = [(ids, draw) for ids in prompts_ids for draw in range(self.repeats)]
        payloads = [self._served_payload(ids, draw, sampling) for ids, draw in jobs]
        if not payloads:
            return [], [], prompt_hashes
        with ThreadPoolExecutor(max_workers=min(SERVED_CONCURRENCY, len(payloads))) as pool:
            replies = list(pool.map(lambda payload: post_completions(str(self.server), payload), payloads))
        texts: list[str | None] = []
        finishes: list[str | None] = []
        for reply in replies:
            choice = (reply.get("choices") or [None])[0]
            if not choice:
                raise RuntimeError(
                    f"livecodebench: server {self.server} answered a completion without a choice"
                )
            texts.append(choice.get("text"))
            finishes.append(choice.get("finish_reason"))
        answers = [texts[i * self.repeats : (i + 1) * self.repeats] for i in range(len(prompts))]
        reasons = [finishes[i * self.repeats : (i + 1) * self.repeats] for i in range(len(prompts))]
        return answers, reasons, prompt_hashes

    # ------------------------------------------------------------------ grading

    def _grade(
        self,
        items: Sequence[dict],
        answers: Sequence[Sequence[str | None]],
        scratch: Path,
        *,
        run_dir: Path | None = None,
        sandbox: str,
    ) -> tuple[dict[str, list[float]], dict[str, int]]:
        """Grade the selected problems in one streaming pass, ``2 * workers`` problems resident at a time.

        Args:
            items: The selected problems.
            answers: The generated answers, ``repeats`` per problem.
            scratch: The caller's scratch directory.
            run_dir: The directory no sandbox source may come from (the eval's out directory).
            sandbox: What runs each program, as :meth:`backend` resolved it.

        Raises:
            RuntimeError: a selected problem has no dataset row (the cached selection and the data directory
                disagree); a score computed from the remaining rows would be silently wrong.
        """
        wanted = {one["id"]: i for i, one in enumerate(items)}
        scratch.mkdir(parents=True, exist_ok=True)
        per_item: dict[str, list[float]] = {}
        counts = {
            "cases_run": 0,
            "timeouts": 0,
            "timeouts_before_retry": 0,
            "errors": 0,
            "no_code": 0,
            "incomplete": 0,
        }
        found: set[str] = set()

        def problems() -> Iterator[tuple[str, list[str | None], list[dict], str | None]]:
            for row in self._rows_for(wanted):
                ident = str(row["question_id"])
                if ident in found:
                    continue
                found.add(ident)
                i = wanted[ident]
                codes = [self._code_of(text) for text in answers[i]]
                counts["no_code"] += sum(code is None for code in codes)
                cases, fn_name = cases_of(row)
                yield (ident, codes, cases, fn_name)

        self._grade_chunk(problems(), scratch, per_item, counts, run_dir=run_dir, sandbox=sandbox)
        counts["incomplete"] = len(items) - len(found)
        if counts["incomplete"]:
            absent = sorted(set(wanted) - found)
            raise RuntimeError(
                f"livecodebench: {len(absent)} of {len(items)} selected problems have no dataset row in "
                f"{self.data_dir()} (first: {absent[0]}); run prepare again"
            )
        for one in items:
            per_item.setdefault(one["id"], [0.0] * self.repeats)
        counts["timeouts_after_retry"] = counts["timeouts"]
        return per_item, counts

    def _grading_cpus(self) -> set[int]:
        """Resolve dedicated grader CPUs from the config and refuse an unavailable or engine-empty allocation."""
        available = os.sched_getaffinity(0)
        cores = {tuple(sorted(self._siblings({cpu}))): cpu for cpu in sorted(available)}
        cpus = set(
            self.grading_cpus
            if self.grading_cpus is not None
            else list(cores.values())[-min(self.workers, max(1, len(cores) - 1)) :]
        )
        if not cpus or not cpus <= available or not available - self._siblings(cpus):
            raise ValueError("livecodebench: set grading_cpus to available CPUs, leaving CPUs for the engine")
        return cpus

    @staticmethod
    def _siblings(cpus: set[int]) -> set[int]:
        """Include SMT siblings so the engine cannot contend on a grader's physical core."""
        siblings = set(cpus)
        for cpu in cpus:
            text = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
            for part in text.split(","):
                bounds = [int(n) for n in part.split("-")]
                siblings.update(range(bounds[0], bounds[-1] + 1))
        return siblings

    def _rows_for(self, wanted: dict[str, int]) -> Iterator[dict]:
        """Stream the dataset rows whose ``question_id`` is selected."""
        for name in FILES:
            path = self.data_dir() / name
            if not path.is_file():
                continue
            for row in rows(path):
                if str(row.get("question_id")) in wanted:
                    yield row

    def _grade_chunk(
        self,
        chunk: Iterable[tuple[str, list[str | None], list[dict], str | None]],
        scratch: Path,
        per_item: dict[str, list[float]],
        counts: dict[str, int],
        *,
        run_dir: Path | None = None,
        sandbox: str,
    ) -> None:
        """Grade the problems of ``chunk``, each distinct program once, with a bounded resident window.

        ``chunk`` is consumed as the pool drains it: at most ``2 * workers`` problems' cases are resident,
        counting both the in-flight window and the timed-out programs waiting for their retry, and once the
        oldest one's programs all finish, the next problem is read from the stream and submitted. When the
        retry list would push the resident count above the window, the pass drains the window and runs those
        retries alone before reading on, so timeouts do not accumulate. The pool therefore never waits for a
        batch's slowest program while the stream still has work. The case directories are made under
        ``scratch``, which the host writes, so it is the one area inside ``run_dir`` a sandbox mount may come
        from.
        """
        verdicts: dict[tuple[str, str], bool] = {}
        codes_of: dict[str, list[str | None]] = {}
        cpus = self._grading_cpus()
        retry = []
        meta_of: dict[Future[tuple[bool, int, int, int]], tuple[str, str, list[dict], str | None]] = {}

        def submit(
            pool: ThreadPoolExecutor,
            ident: str,
            codes: list[str | None],
            cases: list[dict],
            fn_name: str | None,
        ) -> set[Future[tuple[bool, int, int, int]]]:
            group: set[Future[tuple[bool, int, int, int]]] = set()
            for code in dict.fromkeys(code for code in codes if code is not None):
                future = pool.submit(
                    grade_code,
                    code,
                    fn_name,
                    cases,
                    scratch,
                    self.timeout_s,
                    run_dir=run_dir,
                    host_written=[scratch],
                    proc=self.proc,
                    protocol=self.protocol,
                    sandbox=sandbox,
                )
                meta_of[future] = (ident, code, cases, fn_name)
                group.add(future)
            return group

        def record(future: Future[tuple[bool, int, int, int]]) -> None:
            ident, code, cases, fn_name = meta_of.pop(future)
            passed, ran, timeouts, errors = future.result()
            verdicts[(ident, code)] = passed
            counts["cases_run"] += ran
            counts["timeouts_before_retry"] += timeouts
            counts["errors"] += errors
            if timeouts:
                retry.append((ident, code, cases, fn_name))

        def retry_alone() -> None:
            """Re-run each timed-out program once, alone, then release its cases.

            A complete re-run preserves state in the prefix before the timed-out case. Called only when no
            parallel program is running, mid-stream after draining the window or once at the end.
            """
            if not retry:
                return
            with ThreadPoolExecutor(
                max_workers=1, initializer=os.sched_setaffinity, initargs=(0, cpus)
            ) as alone:
                for ident, code, cases, fn_name in retry:
                    passed, ran, timeouts, errors = alone.submit(
                        grade_code,
                        code,
                        fn_name,
                        cases,
                        scratch,
                        self.timeout_s,
                        run_dir=run_dir,
                        host_written=[scratch],
                        proc=self.proc,
                        protocol=self.protocol,
                        sandbox=sandbox,
                    ).result()
                    verdicts[(ident, code)] = passed
                    counts["cases_run"] += ran
                    counts["timeouts"] += timeouts
                    counts["errors"] += errors
            retry.clear()

        window = self.workers * 2
        resident: set[str] = set()  # problems whose cases the window or the retry list still holds
        with ThreadPoolExecutor(
            max_workers=self.workers, initializer=os.sched_setaffinity, initargs=(0, cpus)
        ) as pool:
            pending: deque[tuple[str, set[Future[tuple[bool, int, int, int]]]]] = deque()
            for ident, codes, cases, fn_name in chunk:
                codes_of[ident] = codes
                pending.append((ident, submit(pool, ident, codes, cases, fn_name)))
                resident.add(ident)
                while len(resident) >= window:
                    if not pending:
                        retry_alone()
                        resident.clear()
                        break
                    held = len(retry)
                    drained, group = pending.popleft()
                    for future in as_completed(group):
                        record(future)
                    if len(retry) == held:
                        resident.discard(drained)
            while pending:
                held = len(retry)
                drained, group = pending.popleft()
                for future in as_completed(group):
                    record(future)
                if len(retry) == held:
                    resident.discard(drained)
        retry_alone()
        for ident, codes in codes_of.items():
            per_item[ident] = [float(code is not None and verdicts[(ident, code)]) for code in codes]
