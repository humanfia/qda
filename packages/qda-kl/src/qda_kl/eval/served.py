"""Score a checkpoint on a stock vLLM OpenAI server: one engine per card, every cell a client of it.

The per-cell path (:meth:`qda_kl.eval.lm_eval.LmEvalSuite.run`) starts an engine inside each cell and lets the
process take it down at the end, so a cell's last few generations run alone on the card. A tail is unavoidable
inside one cell; what costs the card is that the next cell starts a new engine instead of feeding its own
requests into the running one. One long-lived server per card and model removes that: cells stay separate
processes (their results, records and the sole-writer gate are unchanged) and only generation moves into the
shared server, which schedules every cell's requests together.

Prompts are sent as token ids from the same tokenizer, so the prompt is the one the in-process path sent;
sampling, the answer cap, the EOS-only engine stop of a thinking model, the post-generation stop cut and the
answer-region cut all go through the same harness helpers. What differs by design is the batch composition,
which vLLM does not make bit-reproducible: the two paths are compared per item against an old-vs-old control,
never for exact equality.

``server_command`` renders a checkpoint's resolved serve settings (:func:`qda_kl.serve.settings_for`) as a
``vllm serve`` command **and its environment** (the ``serve.env`` kernel pin, ``VLLM_PLUGINS``, NVLS, the
FlashInfer workspace). A setting either renders or is refused, never dropped. ``server_identity`` reads the server's own
vLLM version and model, which the result records beside the settings that were requested.

The client derives from the harness's plain ``TemplateLM``, because ``lm-eval[api]`` (which ``tenacity`` comes
from) is not installed in the eval venv. Importing this module does not need lm-evaluation-harness at all: the
harness is an optional extra, so ``ServedModel`` is built on first use (module ``__getattr__``) and the harness's
helpers are imported inside the methods that call them.
"""

from __future__ import annotations

import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from qda_quant.output.serve_command import SERVED_NAME, serve_argv
from qda_quant.step import Refused

from qda_kl.serve import ServeSettings, engine_environment
from qda_kl.server import wait_ready as wait_ready


def server_command(
    checkpoint: str,
    settings: ServeSettings,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    seed: int = 17,
    served_name: str = SERVED_NAME,
) -> tuple[list[str], dict[str, str]]:
    """The ``vllm serve`` argv and environment for ``settings`` (a server started without the env binds
    another kernel than the row it reproduces)."""
    argv = serve_argv(
        checkpoint,
        settings.args,
        name=served_name,
        host=host,
        port=port,
        seed=seed,
        disable_log_requests=True,
    )
    return argv, engine_environment(settings)


def _base_url(base_url: str) -> str:
    """The server's root: the base URL without its completions path (``.../v1/completions``) or ``/v1``."""
    return base_url.rstrip("/").removesuffix("/v1/completions").removesuffix("/v1")


def server_identity(base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """What the server says it is: its vLLM version and served models, read from the server.

    ``GET /version`` names the engine that generates; ``GET /v1/models`` names the model every request is
    answered by and the checkpoint directory it loaded. The result records these, not the request's settings:
    a served result must name the engine that produced it.

    Raises:
        Refused: the server did not answer one of the two endpoints with a usable body.
    """
    root = _base_url(base_url)
    try:
        version = requests.get(f"{root}/version", timeout=timeout)
        version.raise_for_status()
        models = requests.get(f"{root}/v1/models", timeout=timeout)
        models.raise_for_status()
        reported = version.json().get("version")
        data = models.json().get("data") or []
    except (requests.RequestException, ValueError) as why:
        raise Refused(f"server {root}: cannot read its version and models: {why}") from why
    served = next((m for m in data if m.get("id") == SERVED_NAME), None)
    return {
        "url": root,
        "vllm": reported,
        "model": None if served is None else {"id": served.get("id"), "root": served.get("root")},
        "models": [m.get("id") for m in data],
    }


def check_server_identity(
    identity: dict[str, Any], checkpoint: str, expected_vllm: str | None = None
) -> None:
    """Refuse a server that is not the engine the settings name, or not serving this checkpoint.

    Args:
        identity: what :func:`server_identity` read.
        checkpoint: the local checkpoint directory this run measures.
        expected_vllm: the vLLM version the checkpoint's settings name, or None when they name none.

    Raises:
        Refused: the server reported no vLLM version; it runs another version than ``expected_vllm``; it
            serves no model under :data:`SERVED_NAME`; or that model's root is not ``checkpoint``.
    """
    if not identity.get("vllm"):
        raise Refused(
            f"server {identity['url']}: /version named no vLLM version; the result cannot record its engine"
        )
    if expected_vllm is not None and identity["vllm"] != expected_vllm:
        raise Refused(
            f"server {identity['url']}: it runs vLLM {identity['vllm']}, the settings name {expected_vllm}"
        )
    model = identity.get("model")
    if model is None or model.get("id") != SERVED_NAME:
        raise Refused(
            f"server {identity['url']}: /v1/models names {identity.get('models')}, not {SERVED_NAME!r}"
        )
    root = model.get("root")
    if not root:
        raise Refused(
            f"server {identity['url']}: its model {SERVED_NAME!r} names no root, so the checkpoint it serves "
            f"cannot be checked against {checkpoint}"
        )
    if Path(root).resolve() != Path(checkpoint).resolve():
        raise Refused(f"server {identity['url']}: /v1/models root {root} is not the checkpoint {checkpoint}")


class _ServedClient:
    """lm-evaluation-harness model backed by a running vLLM completions server (``/v1/completions``).

    ``ServedModel`` is this class over the harness's ``TemplateLM`` (`_served_model`), so that importing the
    module does not import the harness.

    Built from the same checkpoint and chat-template settings as the in-process model, so the harness builds
    the same prompts; generation is one HTTP request per request, sent as token ids. It derives from the
    harness's ``TemplateLM`` rather than its API model, because that model's constructor requires
    ``lm-eval[api]`` (``tenacity``), which the ``eval`` extra does not install; only generation is offered here.

    Attributes:
        enable_thinking: the chat template's thinking mode, set by the suite (the harness leaves it unset,
            which renders a different prompt).
        chat_template_args: extra template arguments, set by the suite.
        think_end_token: the model's closing thinking tag; answers are cut to the text after it.
        qda_samples: ``(task name, document id) -> [raw completion per draw]``, which the suite publishes.
    """

    def __init__(
        self,
        *,
        base_url: str,
        checkpoint: str,
        max_model_len: int,
        seed: int = 17,
        trust_remote_code: bool = False,
        tokenizer: str | None = None,
        num_concurrent: int = 128,
        timeout: float = 86400.0,
        retries: int = 3,
    ) -> None:
        super().__init__()
        import transformers

        self.base_url = base_url
        self.model = SERVED_NAME
        self.checkpoint = checkpoint
        # The in-process model's tokenizer: the same source, asked for the same way, so the same ids. `tokenizer`
        # is the serve settings' override (vLLM's ``--tokenizer``) and MUST be honoured here: this client sends
        # prompts as token ids, so the server's tokenizer never sees the text, and a checkpoint whose own
        # tokenizer is broken would silently encode every prompt with it. On 2026-11-10 that mangled a whole
        # evaluation row -- a checkpoint carrying a defective tokenizer dropped every space and newline, and the
        # config's override reached only ``vllm serve``, which was never asked to encode anything.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer or checkpoint, trust_remote_code=trust_remote_code
        )
        self.max_length = int(max_model_len)
        self._max_gen_toks = 256
        self._batch_size = 1
        self._seed = int(seed)
        self._concurrent = max(1, int(num_concurrent))
        self._timeout = float(timeout)
        self._retries = max(1, int(retries))
        self._local = threading.local()
        self.enable_thinking = True
        self.chat_template_args: dict[str, Any] = {}
        self.think_end_token: str | None = None
        self.qda_samples: dict[tuple[str, str], list[dict[str, Any]]] = {}

    # -- the harness's model interface ----------------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int | None:
        """The checkpoint's end-of-text id, which vLLM stops on."""
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        """The harness's default answer cap when a request names none."""
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        """One request per call: the server schedules, so the client batches nothing."""
        return self._batch_size

    @property
    def tokenizer_name(self) -> str:
        """The checkpoint's name, as the harness's fingerprinting asks for it."""
        return str(self.checkpoint).replace("/", "__")

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        **kwargs: Any,  # noqa: ARG002 - the harness's own signature
    ) -> list[int]:
        """Encode as the in-process model does: no special tokens, optional left truncation."""
        encoding: list[int] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def loglikelihood(self, requests: Any, disable_tqdm: bool = False) -> Any:  # noqa: ARG002
        """Unsupported: the served path generates only (every paper suite here is a generation task).

        Raises:
            Refused: always.
        """
        raise Refused("served model: loglikelihood is not offered; only generation tasks can run on a server")

    def loglikelihood_rolling(self, requests: Any, disable_tqdm: bool = False) -> Any:  # noqa: ARG002
        """Unsupported: see :meth:`loglikelihood`.

        Raises:
            Refused: always.
        """
        raise Refused("served model: loglikelihood is not offered; only generation tasks can run on a server")

    def _loglikelihood_tokens(self, requests: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Unsupported: see :meth:`loglikelihood`.

        Raises:
            Refused: always.
        """
        raise Refused("served model: loglikelihood is not offered; only generation tasks can run on a server")

    # -- prompt ---------------------------------------------------------------------------------------------------
    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """The checkpoint's chat template with the suite's thinking mode, rendered as the in-process model does."""

        def render(messages: list[dict[str, str]]) -> str:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                chat_template=getattr(self.tokenizer, "chat_template", None),
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )

        try:
            return render(chat_history)
        except Exception:  # noqa: BLE001 - a template that rejects a system role: render again without it
            if "system" not in {m.get("role") for m in chat_history}:
                raise
            return render([m for m in chat_history if m.get("role") != "system"])

    # -- generation -----------------------------------------------------------------------------------------------
    def generate_until(
        self,
        requests: list[Any],
        disable_tqdm: bool = False,  # noqa: ARG002 - the harness's own signature
    ) -> list[str]:
        """Serve every request in parallel and return each answer cut to its answer region."""
        from lm_eval.models.utils import postprocess_generated_text

        from qda_kl.eval.lm_eval import answer_region

        if not requests:
            return []
        eos = self.tokenizer.decode(self.eot_token_id) or self.tokenizer.eos_token
        plans = [self._plan(request, eos) for request in requests]
        with ThreadPoolExecutor(max_workers=min(self._concurrent, len(plans))) as pool:
            responses = list(pool.map(self._post, (plan["payload"] for plan in plans)))
        outs = []
        for request, plan, response in zip(requests, plans, responses, strict=True):
            choice = response["choices"][0]
            tokens = choice.get("token_ids")
            ntoks = len(tokens) if tokens is not None else int(response["usage"]["completion_tokens"])
            # The raw text, before any stop cut: the suite counts cap hits on it and keeps it for regrading.
            self.qda_samples.setdefault((request.task_name, str(request.doc_id)), []).append(
                {"text": choice["text"], "finish_reason": choice["finish_reason"], "tokens": ntoks}
            )
            # The engine stopped on ``engine_stop``; cut there, then to the answer region, where the task's own
            # stop strings apply — so one quoted inside the thinking cannot end the answer.
            text = postprocess_generated_text(choice["text"], plan["engine_stop"], None)
            outs.append(answer_region(text, self.think_end_token, plan["task_stops"]))
        return outs

    def _plan(self, request: Any, eos: str) -> dict[str, Any]:
        """One request's token ids, sampling payload and stop lists, as the in-process path would send them."""
        from lm_eval.models.utils import handle_stop_sequences, maybe_truncate, normalize_gen_kwargs

        from qda_kl.eval.lm_eval import task_stops

        context, gen_kwargs = request.args[0], dict(request.args[1]) if len(request.args) > 1 else {}
        stops = task_stops(request)
        gen = normalize_gen_kwargs(gen_kwargs, default_max_gen_toks=self._max_gen_toks)
        until = handle_stop_sequences(gen.pop("until", None), eos=eos)
        max_gen_toks = int(gen.pop("max_gen_toks", self._max_gen_toks))
        gen.pop("do_sample", None)
        # HF defaults, as the in-process model sets them: special tokens stay in the text.
        gen = {"skip_special_tokens": False, "spaces_between_special_tokens": False} | gen
        tokens, max_gen_toks = maybe_truncate(
            self.tok_encode(context), max_gen_toks, self.max_length, side="left"
        )
        # While thinking, only EOS reaches the engine; the task's stops are applied to the answer region.
        engine_stop = [stop for stop in until if stop == eos] if self.think_end_token else until
        payload: dict[str, Any] = {
            **gen,
            "model": self.model,
            "prompt": tokens,
            "max_tokens": max_gen_toks,
            "stop": engine_stop,
            "return_token_ids": True,
        }
        if payload.get("seed") is None:
            payload.pop("seed", None)  # a request that names no seed sends none (the engine's own applies)
        return {"payload": payload, "engine_stop": engine_stop, "task_stops": stops}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one payload, retrying a refused or dropped connection; any other failure is raised."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
        return post_completions(
            self.base_url, payload, timeout=self._timeout, retries=self._retries, session=session
        )


def post_completions(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 86400.0,
    retries: int = 3,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """POST one completions payload, retrying a refused or dropped connection.

    Every served client (the harness model here and a suite that sends its own prompt ids, such as
    ``livecodebench``) uses this, so they treat a failure the same way: a 5xx or a transport error is retried
    with backoff; any other non-200 status, and a body that is not JSON, is raised.

    Raises:
        RuntimeError: the server answered a non-200 status, or the request loop ended without a response.
        requests.RequestException: the last attempt failed with a transport error.
    """
    owns = session is None
    session = requests.Session() if session is None else session
    try:
        for attempt in range(1, max(1, int(retries)) + 1):
            try:
                response = session.post(base_url, json=payload, timeout=timeout)
                if response.status_code >= 500 and attempt < retries:
                    time.sleep(min(2.0**attempt, 10.0))
                    continue
                if response.status_code != 200:
                    raise RuntimeError(
                        f"served model: {base_url} answered {response.status_code}: {response.text[:400]}"
                    )
                return response.json()
            except requests.RequestException:
                if attempt == retries:
                    raise
                time.sleep(min(2.0**attempt, 10.0))
        raise RuntimeError("served model: request loop ended without a response")  # pragma: no cover
    finally:
        if owns:
            session.close()


@functools.cache
def _served_model() -> type:
    """The harness model class: `_ServedClient` first, so its methods implement ``TemplateLM``'s abstract ones."""
    from lm_eval.api.model import TemplateLM

    # ``__module__`` is named here: ``ABCMeta.__new__`` would otherwise take it from ``abc``, and pickling, which
    # looks the class up by module and name, would fail.
    namespace = {"__doc__": _ServedClient.__doc__, "__module__": __name__}
    return type("ServedModel", (_ServedClient, TemplateLM), namespace)


if TYPE_CHECKING:
    from lm_eval.api.model import TemplateLM

    class ServedModel(_ServedClient, TemplateLM):  # noqa: D101 - the type checker's view of `_served_model`
        ...


def __getattr__(name: str) -> Any:
    """``ServedModel`` is built on first access, the only point that imports the harness."""
    if name == "ServedModel":
        return _served_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
