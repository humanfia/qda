"""The served-model client: what it sends, how it cuts answers, and the serve command it starts.

The harness is a real dependency here (the client subclasses its completions model), so the module skips when
the eval extra is absent; the tokenizer and the HTTP server are stand-ins, so no checkpoint and no GPU are
needed. What the fake server sees is the payload assertion: token ids, the sampling the suite resolved, and the
EOS-only stop of a thinking model with the task stops kept for the answer-region cut.
"""

from __future__ import annotations

import hashlib
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("lm_eval")

from qda_kl.eval.served import (  # noqa: E402 - after the harness check
    ServedModel,
    check_server_identity,
    server_command,
    server_identity,
)
from qda_kl.serve import ServeSettings, settings_for  # noqa: E402
from qda_quant.step import Refused  # noqa: E402

EOS = "</s>"
ANSWER = "thinking Question: a quoted stop <｜end▁of▁thinking｜>the answer"


class _Encoding:
    """What a tokenizer call returns."""

    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class _Tokenizer:
    """The smallest tokenizer the client uses: one id per code point, EOS id 1, a fixed template."""

    name_or_path = "stub"
    eos_token = EOS
    eos_token_id = 1
    bos_token = "<s>"
    bos_token_id = 2
    pad_token = EOS
    pad_token_id = 1
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.templates: list[dict] = []

    def __call__(self, text, add_special_tokens=False, truncation=False, return_attention_mask=False, **kw):
        return _Encoding([ord(c) for c in text])

    def decode(self, ids):
        ids = [ids] if isinstance(ids, int) else ids
        return "".join(EOS if i == 1 else chr(i) for i in ids)

    def batch_decode(self, ids):
        return [self.decode(i) for i in ids]

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append({"messages": messages, **kwargs})
        return "<prompt>"


class _Server:
    """A completions endpoint: records each body and answers with ``answer`` (or an echo of the prompt ids).

    It also answers what a real server does about itself: ``GET /version`` and ``GET /v1/models``, whose model
    root names the checkpoint it serves.
    """

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.answer = ANSWER
        self.token_ids = [10, 11, 12, 13]
        self.finish = "stop"
        self.vllm = "0.33.0"
        self.root = "/ckpt"
        self.model_id = "qda"
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # silence
                pass

            def _json(self, body: dict, status: int = 200) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802 - http.server's name
                if self.path == "/version":
                    self._json({"version": outer.vllm})
                elif self.path == "/v1/models":
                    self._json(
                        {
                            "object": "list",
                            "data": [{"id": outer.model_id, "object": "model", "root": outer.root}],
                        }
                    )
                else:
                    self._json({"error": f"no {self.path}"}, status=404)

            def do_POST(self) -> None:  # noqa: N802 - http.server's name
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                text = outer.answer if outer.answer is not None else "".join(chr(i) for i in body["prompt"])
                ids = outer.token_ids if outer.answer is not None else list(body["prompt"])[:3]
                self._json(
                    {
                        "id": "x",
                        "object": "text_completion",
                        "created": 0,
                        "model": body.get("model"),
                        "choices": [
                            {"index": 0, "text": text, "finish_reason": outer.finish, "token_ids": ids}
                        ],
                        "usage": {
                            "prompt_tokens": len(body["prompt"]),
                            "completion_tokens": len(ids),
                            "total_tokens": len(body["prompt"]) + len(ids),
                        },
                    }
                )

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def tokenizer(monkeypatch) -> _Tokenizer:
    """A stub in place of the checkpoint's tokenizer, so no checkpoint is read."""
    stub = _Tokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: stub)
    return stub


@pytest.fixture
def server():
    got = _Server()
    yield got
    got.close()


def _request(context: str, stops=("Question:",), doc_id: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        args=(
            context,
            {
                "until": list(stops),
                "max_gen_toks": 64,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "seed": 17,
                "do_sample": True,
            },
        ),
        task_name="paper_aime25",
        doc_id=doc_id,
    )


def _model(
    tokenizer: _Tokenizer, server: _Server, *, think: str | None = "<｜end▁of▁thinking｜>"
) -> ServedModel:
    model = ServedModel(base_url=server.url, checkpoint="stub", max_model_len=8192, seed=17)
    model.enable_thinking = True
    model.think_end_token = think
    return model


def test_payload_is_the_in_process_request_as_token_ids(tokenizer, server) -> None:
    """Prompt ids, sampling, seed and the EOS-only engine stop of a thinking model reach the server."""
    model = _model(tokenizer, server)
    assert model.generate_until([_request("hello")]) == ["the answer"]

    body = server.bodies[0]
    assert body["model"] == "qda"
    assert body["prompt"] == [ord(c) for c in "hello"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["top_k"] == 20
    assert body["min_p"] == 0.0
    assert body["presence_penalty"] == 1.5
    assert body["seed"] == 17
    assert body["skip_special_tokens"] is False
    assert body["spaces_between_special_tokens"] is False
    assert body["return_token_ids"] is True
    # The task's stop stays out of the engine: it is quoted inside the thinking and would cut the answer.
    assert body["stop"] == [EOS]


def test_answer_region_is_cut_after_thinking_and_raw_text_is_kept(tokenizer, server) -> None:
    """A task stop quoted in the thinking cannot end the answer, and the raw completion is recorded."""
    model = _model(tokenizer, server)
    assert model.generate_until([_request("hello")]) == ["the answer"]
    assert model.qda_samples[("paper_aime25", "0")] == [
        {"text": ANSWER, "finish_reason": "stop", "tokens": 4}
    ]


def test_without_thinking_the_task_stops_reach_the_engine(tokenizer, server) -> None:
    """A non-thinking model sends the whole stop list, as the in-process model does."""
    server.answer = "before Question: after"
    model = _model(tokenizer, server, think=None)
    # The engine stopped before "Question:", so the answer keeps the space that preceded it, as in-process.
    assert model.generate_until([_request("hello")]) == ["before "]
    assert server.bodies[0]["stop"] == ["Question:", EOS]


def test_results_keep_request_order(tokenizer, server) -> None:
    """Concurrent requests come back in the order they were given, not the order they finished."""
    server.answer = None
    model = _model(tokenizer, server, think=None)
    contexts = ["a", "bb", "ccc", "dddd"]
    outs = model.generate_until([_request(c, stops=(), doc_id=i) for i, c in enumerate(contexts)])
    assert outs == contexts


def test_serve_command_renders_every_setting() -> None:
    """The command carries the settings as flags: booleans as flags, mappings as JSON."""
    settings = ServeSettings(
        args={
            "max_model_len": 114688,
            "dtype": "bfloat16",
            "enable_prefix_caching": False,
            "max_num_seqs": 128,
            "tokenizer": "/tokenizer/ckpt",
            "tokenizer_mode": "auto",
            "compilation_config": {"inductor_compile_config": {"deterministic": True}},
        }
    )
    argv, _ = server_command("/ckpt", settings, port=8123, seed=17)
    assert argv[:3] == ["vllm", "serve", "/ckpt"]
    assert argv[argv.index("--served-model-name") + 1] == "qda"
    assert argv[argv.index("--port") + 1] == "8123"
    assert argv[argv.index("--seed") + 1] == "17"
    assert "--no-enable-prefix-caching" in argv
    assert argv[argv.index("--max-model-len") + 1] == "114688"
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
    assert argv[argv.index("--tokenizer") + 1] == "/tokenizer/ckpt"
    assert argv[argv.index("--tokenizer-mode") + 1] == "auto"
    assert json.loads(argv[argv.index("--compilation-config") + 1]) == {
        "inductor_compile_config": {"deterministic": True}
    }


def test_serve_command_resolves_the_checkpoints_own_serve_section(tmp_path) -> None:
    """A built checkpoint's build.json reaches the command: vllm, linear_backend, and its own args."""
    (tmp_path / "build.json").write_text(
        json.dumps({"serve": {"vllm": "0.33.0", "args": {"linear_backend": "marlin"}}})
    )
    settings = settings_for(str(tmp_path), args={"max_model_len": 8192})
    assert settings.vllm == "0.33.0" and settings.source == "build.json+override"
    argv, _ = server_command(str(tmp_path), settings, port=8123, seed=17)
    assert argv[argv.index("--linear-backend") + 1] == "marlin"
    assert argv[argv.index("--max-model-len") + 1] == "8192"


def test_serve_command_refuses_a_setting_it_cannot_render() -> None:
    """A setting without a flag is refused, not dropped: the command would start another engine."""
    with pytest.raises(Refused, match="no vllm serve flag"):
        server_command("/ckpt", ServeSettings(args={"something_new": 1}))


def test_serve_command_carries_the_environment_argv_cannot() -> None:
    """A checkpoint's kernel pin and plugin list ride in the returned environment, not silently nowhere."""
    settings = ServeSettings(
        env={"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}, plugins=["mypkg"], args={"dtype": "bfloat16"}
    )
    argv, env = server_command("/ckpt", settings, port=8123)
    assert env["VLLM_DISABLED_KERNELS"] == "MacheteLinearKernel"
    assert env["VLLM_PLUGINS"] == "mypkg"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"


def test_server_identity_is_read_from_the_server(server) -> None:
    """The server names its own vLLM version and the model root it serves, and a matching one passes."""
    identity = server_identity(server.url)
    assert identity["vllm"] == "0.33.0"
    assert identity["model"] == {"id": "qda", "root": "/ckpt"}
    check_server_identity(identity, "/ckpt", "0.33.0")


def _fake_harness(monkeypatch):
    """Patch the two harness entry points the served branch reaches; generation still goes to the server."""
    import lm_eval.evaluator
    import lm_eval.tasks

    class Manager:
        def __init__(self, include_path=None, metadata=None) -> None:  # noqa: ARG002 - the harness's own
            pass

        @staticmethod
        def load(tasks):
            return {"tasks": dict.fromkeys(tasks, object()), "groups": {}, "group_map": {}}

    def simple_evaluate(**kwargs):
        task = kwargs["tasks"][0]
        requests = [
            types.SimpleNamespace(
                args=(q, {"until": ["Question:"], "max_gen_toks": 64}), task_name=task, doc_id=i
            )
            for i, q in enumerate(["q1", "q2"])
        ]
        outs = kwargs["model"].generate_until(requests)
        return {
            "results": {task: {"name": task, "alias": task, "score,none": 0.5, "score_stderr,none": "N/A"}},
            "higher_is_better": {task: {"score": True}},
            "n-samples": {task: {"original": 2, "effective": 2}},
            "samples": {
                task: [
                    {
                        "doc_id": i,
                        "doc_hash": q,
                        "doc": {"question_id": i},
                        "prompt_hash": hashlib.sha256(q.encode()).hexdigest(),
                        "filter": "none",
                        "score": 0.5,
                        "filtered_resps": [text],
                    }
                    for i, (q, text) in enumerate(zip(["q1", "q2"], outs, strict=True))
                ]
            },
        }

    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", simple_evaluate)
    monkeypatch.setattr(lm_eval.tasks, "TaskManager", Manager)


def test_a_served_run_records_the_servers_own_version_and_model(
    monkeypatch, tmp_path, tokenizer, server
) -> None:
    """``LmEvalSuite.run`` on a server writes what the server reported, not what the settings requested."""
    from qda_kl.eval.lm_eval import LmEvalSuite

    _fake_harness(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "build.json").write_text(json.dumps({"serve": {"vllm": "0.33.0"}}))
    server.root = str(ckpt)
    suite = LmEvalSuite(
        tasks=["paper_aime25"],
        repeats=1,
        max_tokens=64,
        sampling={"temperature": 1.0},
        serve_args={"max_model_len": 8192},
        server=server.url,
    )
    got = suite.run(str(ckpt), "reference", tmp_path / "out")
    assert got.detail["engine"] == "server"
    assert got.detail["vllm"] == "0.33.0"  # read from GET /version, not settings.vllm
    assert got.detail["server"] == {
        "url": server.url,
        "vllm": "0.33.0",
        "model": {"id": "qda", "root": str(ckpt)},
        "models": ["qda"],
    }
    assert server.bodies  # generation went to the server


def test_a_served_run_refuses_another_engine_before_generating(
    monkeypatch, tmp_path, tokenizer, server
) -> None:
    """A server running another vLLM than the settings name refuses the run, and nothing is generated."""
    from qda_kl.eval.lm_eval import LmEvalSuite

    _fake_harness(monkeypatch)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "build.json").write_text(json.dumps({"serve": {"vllm": "0.33.0"}}))
    server.root, server.vllm = str(ckpt), "0.10.2"
    suite = LmEvalSuite(
        tasks=["paper_aime25"],
        repeats=1,
        max_tokens=64,
        sampling={"temperature": 1.0},
        serve_args={"max_model_len": 8192},
        server=server.url,
    )
    with pytest.raises(Refused, match="runs vLLM 0.10.2"):
        suite.run(str(ckpt), "reference", tmp_path / "out")
    assert server.bodies == []


def test_server_identity_refuses_another_version_or_another_checkpoint(server) -> None:
    """A server running another vLLM, serving another checkpoint, another name, or naming no root is refused."""
    server.vllm = "0.10.2"
    with pytest.raises(Refused, match="runs vLLM 0.10.2"):
        check_server_identity(server_identity(server.url), "/ckpt", "0.33.0")
    server.vllm, server.root = "0.33.0", "/another"
    with pytest.raises(Refused, match="is not the checkpoint"):
        check_server_identity(server_identity(server.url), "/ckpt", "0.33.0")
    server.root, server.model_id = "/ckpt", "other"
    with pytest.raises(Refused, match="not 'qda'"):
        check_server_identity(server_identity(server.url), "/ckpt", None)
    server.model_id, server.root = "qda", ""
    with pytest.raises(Refused, match="names no root"):
        check_server_identity(server_identity(server.url), "/ckpt", None)


def test_the_client_encodes_with_the_serve_settings_tokenizer_override(monkeypatch, server) -> None:
    """The client sends ids, so a broken checkpoint tokenizer silently mangles every prompt.

    On 2026-11-10 an evaluation row was served with a config that overrode the tokenizer, but the override
    reached only ``vllm serve --tokenizer``; the server was never asked to encode anything, because this
    client encodes. The checkpoint's own tokenizer dropped every space and newline, so the whole row
    measured a prompt-mangled model. The override has to reach the thing that encodes.
    """

    class _DropsWhitespace(_Tokenizer):
        """The defect: a Metaspace tokenizer that drops every space and newline."""

        def __call__(self, text, **kw):
            return _Encoding([ord(c) for c in text.replace(" ", "").replace("\n", "")])

    asked: list[str] = []
    drops_spaces = _DropsWhitespace()
    good = _Tokenizer()

    def pick(name, *a, **k):
        asked.append(name)
        return good if name == "/pin/good-tokenizer" else drops_spaces

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", pick)
    text = 'a pitch for "Sunnis ice cream".\nAnything else?'

    broken = ServedModel(base_url=server.url, checkpoint="/ckpt/paro", max_model_len=8192, seed=17)
    assert asked[-1] == "/ckpt/paro"
    assert broken.tokenizer(text).input_ids != [ord(c) for c in text]  # the defect, reproduced

    fixed = ServedModel(
        base_url=server.url,
        checkpoint="/ckpt/paro",
        max_model_len=8192,
        seed=17,
        tokenizer="/pin/good-tokenizer",
    )
    assert asked[-1] == "/pin/good-tokenizer"  # the override is what gets loaded
    assert fixed.tokenizer(text).input_ids == [ord(c) for c in text]  # round-trips exactly

    # no override: unchanged, still the checkpoint's own tokenizer
    plain = ServedModel(
        base_url=server.url, checkpoint="/ckpt/other", max_model_len=8192, seed=17, tokenizer=None
    )
    assert asked[-1] == "/ckpt/other"
    assert plain.checkpoint == "/ckpt/other"
