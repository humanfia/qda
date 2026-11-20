"""`qda-agent prepare`: every step runs once and is skipped after, the plan runs nothing, and the config it
writes holds no placeholder and fills each derived field from its one source.

CPU only: the public commands are a stand-in `subprocess.run` that writes what each command writes.
"""

import errno
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from qda_agent import cli, configfile, evaluate
from qda_agent import prepare as P
from qda_agent.config import RunConfig
from qda_agent.judge.config import MIN_FREE_BYTES
from qda_kl import corpus, refcache
from qda_kl.eval import rundir
from qda_kl.reference import DEFAULT_SAMPLING
from qda_quant.target import Target
from safetensors_kit import write_safetensors


def pool_text(name: str) -> str:
    """What the stand-in `hf download` writes as one messages pool: 256 numbered rows, identifiable by name."""
    rows = [
        {"messages": [{"role": "user", "content": f"{name} {i}"}, {"role": "assistant", "content": "a"}]}
        for i in range(256)
    ]
    return "".join(json.dumps(r) + "\n" for r in rows)


POOL_SHA = {
    name: hashlib.sha256(pool_text(name).encode()).hexdigest() for name in ("chat", "multilingual", "d1")
}

ROWS = [
    {"id": "klv2-chat-chat-00000", "group": "easy", "domain": "chat", "split": "eval", "prompt": "hello"},
    {"id": "klv2-chat-chat-00001", "group": "easy", "domain": "chat", "split": "holdout", "prompt": "bye"},
]


@pytest.fixture(autouse=True)
def local_mounts(monkeypatch):
    """No disk is a network one, whatever this host mounts: the warning a network `$HOME` adds has its own test."""
    from qda_agent import room as R

    monkeypatch.setattr(R, "_mount_table", lambda: [])


@pytest.fixture(autouse=True)
def all_cards_visible(monkeypatch):
    """`--cards` must be in CUDA_VISIBLE_DEVICES when it is set (the root conftest empties it).

    With it unset, `group_size.target_for` would start CUDA on every card of a GPU machine, and torch keeps that
    count for the rest of the pytest process; the target is the one a machine without a driver gets, as on CI.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(P.group_size, "target_for", lambda cards: Target())


#: The real `require_compiler`, before the autouse fixture below replaces it.
REQUIRE_COMPILER = P.require_compiler


@pytest.fixture(autouse=True)
def compiler_holds(monkeypatch):
    """The steps run on stand-in commands; the engine start's refusal (`require_compiler`) is tested alone."""
    monkeypatch.setattr(P, "require_compiler", lambda: None)


def model(tmp_path: Path, generation: dict | None = None) -> Path:
    m = tmp_path / "Qwen3-8B"
    m.mkdir(parents=True)
    (m / "config.json").write_text("{}")
    # a real safetensors header: prepare reads the widths to pick candidate 0's group size
    write_safetensors(
        m / "model.safetensors", {"model.layers.0.self_attn.q_proj.weight": ("BF16", (128, 128), None)}
    )
    if generation is not None:
        (m / "generation_config.json").write_text(json.dumps(generation))
    return m


#: Candidate 0's stored tensors: `model`'s one q_proj, packed 4-bit with a bf16 scale per 128 weights.
PACKED = {
    "model.layers.0.self_attn.q_proj.weight_packed": ("I32", (128, 16), None),
    "model.layers.0.self_attn.q_proj.weight_scale": ("BF16", (128, 1), None),
}


class Commands:
    """A stand-in for every public command prepare runs: it records the argv and writes the outputs."""

    def __init__(self, monkeypatch):
        self.calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", self)
        # no card size unless a test names one: the reference's layout must not depend on this machine's GPUs
        monkeypatch.setattr(P, "ref_card_bytes", lambda cards: None)

    def __call__(self, argv, check=False, env=None, **kw):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        arg = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        if "corpus" in argv:
            corpus.finish(Path(arg["--out"]), ROWS, None, tokenizer="t", tokenizer_revision="r")
        elif "ref" in argv:
            out = Path(arg["--out"])
            out.mkdir(parents=True)
            sha = P.sha256(Path(arg["--prompts"]))
            (out / "manifest.json").write_text(
                json.dumps(
                    {"build": {"prompts": arg["--prompts"], "prompts_sha256": sha}, "content_sha256": "c"}
                )
            )
        elif "render" in argv:
            out = Path(arg["--out"])
            out.write_text('{"text": "t"}\n')
            meta = {"model": arg["--model"], "rows": 300, "sha256": P.sha256(out)}
            meta |= {"samples": int(arg["--samples"]), "seqlen": int(arg["--seqlen"])}
            meta |= {"pool": {"dataset": "qda-lab/qda-calibration-pools", "revision": "abc"}}
            meta |= {"files": {n: POOL_SHA[n] for n in ("chat", "multilingual")}}
            out.with_name(out.name + ".meta.json").write_text(
                json.dumps(meta | {"thinking": {"policy": "keep"}})
            )
        elif "download" in argv and "--repo-type" in argv:
            assert arg["--revision"] == "abc" and "d1.jsonl" not in argv
            for name in argv[3:5]:
                Path(arg["--local-dir"]).mkdir(parents=True, exist_ok=True)
                (Path(arg["--local-dir"]) / name).write_text(pool_text(name.rsplit(".", 1)[0]))
        elif "build" in argv:
            recipe = yaml.safe_load(Path(argv[argv.index("build") + 1]).read_text())
            Path(recipe["out"]).mkdir(parents=True)
            (Path(recipe["out"]) / "build.json").write_text('{"load_to_saved_s": 42.5}')
            (Path(recipe["out"]) / "config.json").write_text("{}")
            if not recipe.get("exporter", {}).get(
                "gguf"
            ):  # a packed W4 g128 q_proj: 4.125 bpw by the judge's rule
                write_safetensors(Path(recipe["out"]) / "model.safetensors", PACKED)
        elif "--outtype" in argv:  # llama.cpp's convert_hf_to_gguf.py (a template)
            out = Path(arg["--outfile"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"GGUF-bf16")
        elif "llama-imatrix" in argv[0]:
            out = Path(argv[argv.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"GGUF-imatrix")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def prepared(tmp_path, **kw) -> P.Prepare:
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    (py.parent / "hf").touch()  # `hf` beside the interpreter, as a pip install puts it
    (tmp_path / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return P.Prepare(
        str(kw.pop("model", None) or model(tmp_path)), tmp_path / "out", python=py, serve_python=py, **kw
    )


def test_the_model_step_leaves_the_downloaded_repo_beside_the_model(tmp_path, monkeypatch):
    """The repo travels with the download, so an agent's own recipe (no meta) still names the original."""
    from qda_quant.config import SOURCE_RECORD

    class Info:
        sha = "c0ffee"

    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda *a, **k: SimpleNamespace(model_info=lambda *a, **k: Info())
    )

    def run(argv, check=False, **kw):  # noqa: ANN001, ARG001
        argv = [str(a) for a in argv]
        arg = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        out = Path(arg["--local-dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}")
        write_safetensors(
            out / "model.safetensors", {"model.layers.0.self_attn.q_proj.weight": ("BF16", (128, 128), None)}
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    p = prepared(tmp_path, model="Qwen/Qwen3-8B")
    next(s for s in P.plan(p, ("model",))).run()

    assert json.loads((p.model_dir / SOURCE_RECORD).read_text()) == {
        "repo": "Qwen/Qwen3-8B",
        "revision": "c0ffee",
    }


def test_the_default_calibration_note_is_decided_from_its_own_metadata(tmp_path):
    """A thinking template with no reasoning rows earns one line pointing at `qda traces`."""
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"text": "t"}\n')
    meta = calib.with_name(calib.name + ".meta.json")
    thinking = {
        "policy": "keep",
        "template": {"families": ["chatml"], "reasoning": True, "think_markers": True},
        "chats": 343,
    }
    meta.write_text(json.dumps({"thinking": thinking}))
    note = P.calib_thinking_note(calib)
    assert note.startswith("the default calibration holds no thinking text")
    assert "qda traces <model> <prompts.jsonl> <prefix> --max-tokens <tokens> --seed <seed>" in note
    # rows did carry reasoning: there is nothing to warn about
    meta.write_text(json.dumps({"thinking": {**thinking, "turns with reasoning, rendered": 2}}))
    assert P.calib_thinking_note(calib) == ""
    # a template without a thinking format: not a thinking model
    plain = {**thinking, "template": {"think_markers": False, "reasoning": False}}
    meta.write_text(json.dumps({"thinking": plain}))
    assert P.calib_thinking_note(calib) == ""
    # no metadata (a calibration the user made): nothing claimed
    meta.unlink()
    assert P.calib_thinking_note(calib) == ""
    assert P.calib_thinking_note(None) == ""


def test_prepare_prints_the_calibration_note(tmp_path, monkeypatch, capsys):
    """The note is a line of prepare's own output, decided once and used by TASK.md too."""
    Commands(monkeypatch)
    monkeypatch.setattr(
        P, "calib_thinking_note", lambda _calib: "the default calibration holds no thinking text"
    )
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0 and "note: the default calibration holds no thinking text." in out


def test_every_step_runs_once_and_a_second_prepare_runs_nothing(tmp_path, monkeypatch):
    commands = Commands(monkeypatch)
    p = prepared(tmp_path, cards=[0, 1])
    lines: list[str] = []
    P.execute(p, P.plan(p), log=lines.append)
    verbs = [
        next(v for v in ("corpus", "ref", "render", "download", "build") if v in argv)
        for argv in commands.calls
    ]
    assert verbs == ["corpus", "ref", "render", "download", "build"]
    ref = commands.calls[1]
    assert ref[ref.index("--tp") + 1] == "2" and ref[ref.index("--prompts") + 1].endswith(
        "corpus.search96.jsonl"
    )
    assert ref[ref.index("--subset") + 1] == f"search96={p.out / 'data' / 'corpus.search96.ids'}"
    assert lines[0].startswith("prepare [1/6] model: skip (a local model, used as is")
    render = commands.calls[2]  # as many rows as candidate 0's recipe packs into its samples
    recipe = yaml.safe_load((p.out / "cand0-gptq.recipe.yaml").read_text())["calib"]["file"]
    assert render[render.index("--samples") + 1] == str(recipe["samples"]) == "128"
    assert render[render.index("--seqlen") + 1] == str(recipe["seqlen"]) == "2048"
    # the search's default calibration is the pools' own rows, not candidate 0's cut
    assert recipe["path"] == str(p.calib) and p.calib != p.agent_calib
    merged = [json.loads(line)["messages"][0]["content"] for line in p.agent_calib.read_text().splitlines()]
    assert len(merged) == 512 and merged[:4] == ["chat 0", "chat 1", "chat 2", "multilingual 0"]
    assert yaml.safe_load(p.config.read_text())["calib"] == str(p.agent_calib)
    assert any(line.startswith("prepare [6/6] config: done in ") for line in lines)
    assert any(line.startswith("      ref (from its manifest): ") for line in lines)  # through execute's log

    commands.calls.clear()
    lines.clear()
    again = P.Prepare(p.model, p.out, cards=[0, 1], python=p.python, serve_python=p.serve_python)
    P.execute(again, P.plan(again), log=lines.append)
    assert commands.calls == [] and all(": skip (" in line for line in lines)
    state = json.loads((p.out / "prepare.json").read_text())
    assert (
        set(state["steps"]) == {"corpus", "ref", "calib", "cand0", "config"}
        and state["cand0"]["build_s"] == 42.5
    )
    assert state["search96"]["sha256"] == P.sha256(p.out / "data" / "corpus.search96.ids")


class ChatWords:
    """Every rendered chat is 64 tokens (w0..w63): one row is one sample of any seqlen up to 64."""

    def apply_chat_template(self, messages, tokenize=False, **kw):  # noqa: ANN001, ANN003, ARG002
        return " ".join(f"w{i}" for i in range(64))

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ARG002
        return {"input_ids": [int(word[1:]) for word in text.split()]}


def test_the_default_calibration_supplies_the_samples_a_build_asks_for(tmp_path, monkeypatch):
    """The agent's calib is a row source, so 256/512 samples are not refused; the ceiling is the rows."""
    from qda_quant.calib.sources import FileSource
    from qda_quant.engine.calib import load_samples
    from qda_quant.step import Refused

    Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p), log=lambda _: None)
    assert yaml.safe_load((p.out / "cand0-gptq.recipe.yaml").read_text())["calib"]["file"] == {
        "path": str(p.calib),
        "samples": 128,
        "seqlen": 2048,
    }  # candidate 0's recipe is unchanged
    for asked in (256, 512):
        got = load_samples(
            FileSource(path=str(p.agent_calib), samples=asked, seqlen=8), p.model_dir, ChatWords()
        )
        assert len(got) == asked
    with pytest.raises(Refused, match=r"512 samples of 8, recipe asks for 513"):
        load_samples(FileSource(path=str(p.agent_calib), samples=513, seqlen=8), p.model_dir, ChatWords())


def test_prepare_takes_the_backend_from_its_registry(tmp_path, monkeypatch):
    """A registered backend is accepted by name with the registry's own defaults, a mapping carries its
    settings, and an unknown name is the registry's refusal (there is no second backend list here)."""
    Commands(monkeypatch)
    p = prepared(tmp_path, backend="codex")
    P.execute(p, P.plan(p), log=lambda _: None)
    assert yaml.safe_load(p.config.read_text())["agent"]["backend"] == {
        "name": "codex",
        "model": "gpt-5.6-sol",
        "effort": "high",
    }
    mapped = P.Prepare(
        p.model,
        p.out,
        python=p.python,
        serve_python=p.serve_python,
        backend="{name: codex, model: gpt-5.6-mini, effort: low}",
    )
    assert mapped.agent.model == "gpt-5.6-mini" and mapped.agent.effort == "low"
    with pytest.raises(P.Refusal, match="no qda_agent.backends plugin"):
        P.Prepare(p.model, p.out, python=p.python, serve_python=p.serve_python, backend="nope")


def test_the_written_config_has_no_placeholder_and_one_source_per_derived_field(tmp_path, monkeypatch):
    Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p), log=lambda _: None)
    raw = yaml.safe_load(p.config.read_text())
    assert configfile.placeholders(raw) == []
    assert raw["judge"]["anchor"] == raw["model"] == raw["scheduler"]["readable"][0] == str(p.model_dir)
    assert raw["judge"]["format_reference"] == raw["baselines"][0] == str(p.cand0)
    assert raw["judge"]["select_candidate0"] is True and raw["judge"]["candidate0_build_s"] == 42.5
    assert raw["isolation"] == {"level": "none"} and raw["contamination"] == {"name": "none"}
    # the one note beside `isolation:` is put back after the dump, the same line the shipped configs carry
    assert f"# {P.ISOLATION_NOTE}\nisolation:\n" in p.config.read_text()
    assert raw["judge"]["kl_holdout_ref"].endswith("corpus.holdout.jsonl")
    (data,) = [c for c in raw["toolset"]["components"] if isinstance(c, dict) and c["name"] == "data"]
    assert sorted(data["sets"]) == ["chat", "multilingual"]  # never d1: pre-rendered text is refused at start
    cfg = RunConfig.load(p.config)
    assert cfg.baselines == [p.cand0] and cfg.judge.kl_ref == p.ref


def test_the_written_header_points_at_a_section_the_page_has():
    """The header's isolation pointer names a heading of docs/search-config.md, so a reader lands on it."""
    root = Path(__file__).parents[3]
    ((page, anchor),) = re.findall(r"(docs/[\w-]+\.md)#([\w-]+)", P.HEADER)
    headings = (root / page).read_text().splitlines()
    assert f"## {anchor.capitalize()}" in headings


def test_the_config_step_warns_of_an_agent_nothing_signs_in_and_takes_a_sign_in_dir(
    tmp_path, monkeypatch, capsys
):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p), log=lambda _: None)
    # `sign_in_report` (the run's own rule, shared with doctor) words this; prepare keeps no copy of it.
    err = capsys.readouterr().err
    assert "qda-agent prepare: warning: claude: credential file in" in err and "missing" in err
    assert "sign_in_dir" not in yaml.safe_load(p.config.read_text())["agent"]["backend"]

    account = tmp_path / "account"
    account.mkdir()
    # A real credential records its expiry: `sign_in_report` reads it (`claudeAiOauth.expiresAt`, ms) and
    # warns when it does not cover the deadline. An empty object has no expiry, which is itself a warning.
    far = int((datetime.now(UTC) + timedelta(days=30)).timestamp() * 1000)
    (account / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"expiresAt": far}}))
    again = P.Prepare(p.model, p.out, python=p.python, serve_python=p.serve_python, sign_in_dir=account)
    P.execute(again, P.plan(again), log=lambda _: None)
    assert "warning" not in capsys.readouterr().err
    assert RunConfig.load(p.config).agent.backend.sign_in_dir == account
    with pytest.raises(P.Refusal, match="dsh has no sign-in file"):
        P.Prepare(p.model, p.out, backend="dsh", sign_in_dir=account)


def test_a_dsh_config_names_no_model_unless_asked_to(tmp_path, monkeypatch):
    """The official API's `deepseek-flash` is not a gateway's name, and prepare cannot know which endpoint the
    run reaches (its account and `DEEPSEEK_BASE_URL` are the run's): a dsh backend that names no model is
    written without one, so the run takes what its endpoint serves; a model the user names is kept."""
    Commands(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    def backend(tmp, spec):
        p = prepared(tmp, backend=spec)
        P.execute(p, P.plan(p), log=lambda _: None)
        return yaml.safe_load(p.config.read_text())["agent"]["backend"], p

    got, p = backend(tmp_path / "bare", "dsh")
    assert got == {"name": "dsh"}
    assert RunConfig.load(p.config).agent.backend.model == "deepseek-flash"  # the official endpoint's own
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")
    got, p = backend(tmp_path / "gateway", "dsh")
    assert "model" not in got
    assert RunConfig.load(p.config).agent.backend.model != "deepseek-flash"  # follows the run's endpoint
    got, _ = backend(tmp_path / "account", "{name: dsh, account: gw}")
    assert got == {"name": "dsh", "account": "gw"}
    got, _ = backend(tmp_path / "named", "{name: dsh, model: my-model}")
    assert got["model"] == "my-model"
    got, _ = backend(tmp_path / "claude", "claude")
    assert got["model"], "another backend's default model is still written out"


def test_a_config_its_start_checks_refuse_is_not_left_behind(tmp_path, monkeypatch):
    """A refused config left in place compares equal on the next prepare and is reported current."""
    Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p, ("model", "corpus", "ref", "calib", "cand0")), log=lambda _: None)
    p.agent_calib.unlink()  # the run's own start check refuses the default calibration it names
    with pytest.raises(P.Refusal, match="does not load"):
        P.execute(p, P.plan(p, ("config",)), log=lambda _: None)
    assert not p.config.exists() and not p.config.with_name(p.config.name + ".tmp").exists()
    assert P.plan(p, ("config",))[0].done() is None  # the next prepare does not report it current


def test_the_config_writes_no_disk_cap(tmp_path, monkeypatch):
    """No estimate-based cap. Candidate 0 x (a kept count + 2 ...) under- and over-counted the frozen copies,
    refusing runs that fit and admitting ones that did not, so prepare writes no `judge.max_run_bytes` in
    search.yaml and no `disk_cap` in prepare.json: the run is guarded by its live free-space floor."""
    Commands(monkeypatch)
    p = prepared(tmp_path, cards=[0, 1])
    P.execute(p, P.plan(p, ("model", "corpus", "ref", "calib", "cand0")), log=lambda _: None)
    P.execute(p, P.plan(p, ("config",)), log=lambda _: None)
    judge = yaml.safe_load(p.config.read_text())["judge"]
    assert "keep_top" not in judge and "max_run_bytes" not in judge and "min_free_bytes" not in judge
    assert "disk_cap" not in json.loads((p.out / "prepare.json").read_text())
    loaded = RunConfig.load(p.config).judge
    assert loaded.max_run_bytes is None and loaded.min_free_bytes == MIN_FREE_BYTES


def test_prepare_refuses_before_a_gpu_step_when_the_disk_cannot_hold_two_copies_of_the_model(
    tmp_path, monkeypatch
):
    """The run's own check (`custody.space_problem`, the free-space floor alone: no cap, no size estimate),
    called before the first GPU step. The floor is twice the model's safetensors, not a share of the disk: a
    tiny model passes on a 4.3 GB disk, a 3 GB one is refused there with the bytes it needs and the fix."""
    Commands(monkeypatch)
    p = prepared(tmp_path)
    free = {"bytes": 4_300_000_000}
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=free["bytes"], f_frsize=1, f_favail=10**9, f_blocks=100 * 10**9, f_files=10**9
        ),
    )
    P.require_room(p)  # a tiny model: 2 x its weights fit
    with open(p.model_dir / "model.safetensors", "r+b") as f:
        f.truncate(3_000_000_000)  # sparse: a 3 GB bf16 model
    with pytest.raises(
        P.Refusal,
        match=r"DISK FLOOR: needs at least 6.0 GB free \(2 copies of the bf16 model\), .* has only 4.3 GB; "
        r"use --out on a larger disk",
    ):
        P.execute(p, P.plan(p, ("ref",)), log=lambda _: None)
    assert not p.ref.exists()  # no GPU step ran
    free["bytes"] = 6_000_000_000  # at the floor
    P.require_room(p)


def test_a_machine_whose_engine_would_not_start_is_refused_before_the_first_step(tmp_path, monkeypatch):
    """No C compiler, or a Python without its headers: the ref step's engine stopped after the download and the
    corpus. When the ref step is pending, prepare enters the same engine start (`vllm_environ`) before the first
    step runs; a plan without a pending ref step (``--steps model,corpus``, a finished prepare) does not."""
    from contextlib import contextmanager

    from qda_quant.output import vllm_env
    from qda_quant.step import Refused

    entered = []

    @contextmanager
    def refusing(env=None):
        entered.append(env)
        raise Refused(
            "C compiler: not found: Triton's first use builds its CUDA driver with it", "install gcc", code=3
        )
        yield

    commands = Commands(monkeypatch)
    p = prepared(tmp_path)
    monkeypatch.setattr(P, "require_compiler", REQUIRE_COMPILER)
    monkeypatch.setattr(vllm_env, "vllm_environ", refusing)
    P.execute(p, P.plan(p, ("model", "corpus")), log=lambda _: None)  # no ref step: runs, nothing entered
    assert p.corpus.exists() and entered == []
    P.execute(p, P.plan(p, ("corpus",)), log=lambda _: None)  # done: nothing to run, nothing entered
    assert entered == []
    calls = len(commands.calls)
    with pytest.raises(P.Refusal, match=r"^C compiler: not found") as refused:
        P.execute(p, P.plan(p, ("ref",)), log=lambda _: None)
    assert refused.value.code == 3 and refused.value.fix == "install gcc" and entered == [{}]
    assert len(commands.calls) == calls and not p.ref.exists()


def test_a_model_without_sampling_gets_a_recorded_explicit_one(tmp_path, monkeypatch):
    commands = Commands(monkeypatch)
    p = prepared(tmp_path)  # the model ships no generation_config.json
    P.execute(p, P.plan(p, ("corpus", "ref")), log=lambda _: None)
    ref = commands.calls[1]
    assert json.loads(ref[ref.index("--sampling") + 1]) == DEFAULT_SAMPLING
    assert json.loads((p.out / "prepare.json").read_text())["sampling"] == DEFAULT_SAMPLING
    named = P.Prepare(str(model(tmp_path / "other", {"temperature": 0.6})), tmp_path / "o2")
    assert "--sampling" not in P.ref_command(named, named.model_dir, named.eval_split, named.ref)


def test_the_reference_takes_the_most_cards_vllm_splits_the_model_over(tmp_path, monkeypatch):
    """Qwen3-0.6B's 16 heads (8 KV heads) refuse 3 cards in vLLM: the reference reads on 2 and says so."""
    commands = Commands(monkeypatch)
    m = model(tmp_path)
    (m / "config.json").write_text(json.dumps({"num_attention_heads": 16, "num_key_value_heads": 8}))
    p = prepared(tmp_path, model=m, cards=[0, 1, 2])
    lines: list[str] = []
    P.execute(p, P.plan(p, ("corpus", "ref")), log=lines.append)
    ref = commands.calls[1]
    assert ref[ref.index("--tp") + 1] == "2"
    assert (
        "      ref on 2 of 3 card(s) (vLLM splits this model's attention heads over at most 2 of them)"
        in "\n".join(lines)
    )
    assert [P.ref_tp(m, n) for n in (1, 2, 4, 8, 16)] == [1, 2, 4, 8, 16]  # 16 cards: each KV head on 2
    (m / "config.json").write_text(
        json.dumps({"text_config": {"num_attention_heads": 28, "num_key_value_heads": 4}})
    )
    assert [P.ref_tp(m, n) for n in (3, 8)] == [2, 4]  # 28 heads: 8 does not divide them


def test_a_model_that_fits_one_card_reads_its_reference_on_one_card(tmp_path, monkeypatch):
    """Weights within REF_CARD_FILL of one card: the reference reads on one card, no tensor parallelism, however
    many cards prepare holds; a model that does not fit, or an unknown card size, keeps the most cards vLLM
    splits it over (Qwen3-0.6B's 16 heads, 8 KV heads: 2 of 3)."""
    commands = Commands(monkeypatch)
    m = model(tmp_path)
    (m / "config.json").write_text(json.dumps({"num_attention_heads": 16, "num_key_value_heads": 8}))
    weights = (m / "model.safetensors").stat().st_size
    p = prepared(tmp_path, model=m, cards=[0, 1, 2, 3])
    asked: list[list[int]] = []

    def card(cards: list[int]) -> int:
        asked.append(list(cards))
        return int(weights / P.REF_CARD_FILL) + 1

    monkeypatch.setattr(P, "ref_card_bytes", card)
    lines: list[str] = []
    P.execute(p, P.plan(p, ("corpus", "ref")), log=lines.append)
    ref = commands.calls[1]
    assert ref[ref.index("--tp") + 1] == "1" and asked and all(a == [0, 1, 2, 3] for a in asked)
    used = next(line for line in "\n".join(lines).splitlines() if line.startswith("      ref on "))
    assert (
        used.startswith("      ref on 1 of 4 card(s) (the model's ")
        and "one card, no tensor parallelism" in used
    )

    three = prepared(tmp_path / "three", model=m, cards=[0, 1, 2])
    command = P.ref_command(three, m, three.search, three.ref)
    assert command[command.index("--tp") + 1] == "1"
    too_small = int(weights / P.REF_CARD_FILL) - 1
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: too_small)
    command = P.ref_command(three, m, three.search, three.ref)
    assert command[command.index("--tp") + 1] == "2"
    assert P.ref_layout(three, m) == (2, "vLLM splits this model's attention heads over at most 2 of them")
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: None)  # no CUDA: the card count alone decides
    assert P.ref_layout(three, m) == (2, "vLLM splits this model's attention heads over at most 2 of them")
    assert P.ref_layout(prepared(tmp_path / "four", model=m, cards=[0, 1, 2, 3]), m) == (4, "")

    def never(cards: list[int]) -> int:
        raise AssertionError("one card asks no card size")

    monkeypatch.setattr(P, "ref_card_bytes", never)
    assert P.ref_layout(prepared(tmp_path / "one", model=m, cards=[0]), m) == (1, "")


def test_ref_card_bytes_sizes_the_cards_named_not_the_first_ones(monkeypatch):
    """`--cards` holds physical card numbers: on a node whose cards 4-7 are smaller than 0-3, a run on 4-7 is
    sized by those, and one on 0-3 by those; a card the driver omits, or no driver, leaves the size unknown."""
    from types import SimpleNamespace

    from qda_agent.isolation import devices
    from qda_quant.intake.fisher import CARD_SHARE

    gib = 2**30
    table = [
        SimpleNamespace(ordinal=i, bus=f"0000:{i + 1:02x}:00.0", minor=i, uuid=f"GPU-{i}") for i in range(8)
    ]
    sizes = [80 * gib] * 4 + [48 * gib] * 4  # cards 4-7 are the smaller ones
    monkeypatch.setattr(devices, "smi_memory", lambda proc=None: list(zip(table, sizes, strict=True)))
    monkeypatch.setattr(
        devices, "resolve_cards", lambda cards, proc=None, visible=None: [table[c] for c in cards]
    )
    monkeypatch.setattr(
        "qda_quant.intake.fisher.card_bytes", lambda count: pytest.fail("torch's first cards were asked")
    )
    assert P.ref_card_bytes([0, 1, 2, 3]) == int(80 * gib * CARD_SHARE)
    assert P.ref_card_bytes([4, 5, 6, 7]) == int(48 * gib * CARD_SHARE)
    assert P.ref_card_bytes([2, 3, 4, 5]) == int(48 * gib * CARD_SHARE), "the smallest of the named cards"
    monkeypatch.setattr(devices, "smi_memory", lambda proc=None: list(zip(table[:4], sizes[:4], strict=True)))
    assert P.ref_card_bytes([4, 5]) is None, "a card nvidia-smi does not list"
    monkeypatch.setattr(devices, "smi_memory", lambda proc=None: None)
    assert P.ref_card_bytes([0, 1]) is None, "a driver that will not say"


def test_smi_memory_reads_each_cards_total_in_bytes(tmp_path, monkeypatch):
    from qda_agent.isolation import bwrap

    proc = tmp_path / "gpus"
    for i, bus in enumerate(("0000:18:00.0", "0000:28:00.0")):
        (proc / bus).mkdir(parents=True)
        (proc / bus / "information").write_text(
            f"GPU UUID: \t GPU-{i}\nBus Location: \t {bus}\nDevice Minor: \t {i}\n"
        )
    out = "1, 00000000:28:00.0, 49140\n0, 00000000:18:00.0, 81559\n2, 00000000:99:00.0, 1\n"
    monkeypatch.setattr(
        bwrap.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr="")
    )
    got = bwrap.smi_memory(proc)
    assert [(c.ordinal, total) for c, total in got] == [(0, 81559 * 2**20), (1, 49140 * 2**20)]
    monkeypatch.setattr(
        bwrap.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="")
    )
    assert bwrap.smi_memory(proc) is None


def test_the_dry_run_prints_the_plan_and_runs_nothing(tmp_path, monkeypatch, capsys):
    commands = Commands(monkeypatch)
    m = model(tmp_path)
    code = cli.main(
        ["prepare", "--model", str(m), "--out", str(tmp_path / "out"), "--cards", "0,1", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert code == 0 and commands.calls == [] and not (tmp_path / "out").exists()
    assert "[2/6] corpus: run -> " in out and "qda_kl.cli corpus --out" in out and "--sampling" in out
    assert "qda_kl.cli ref --model" in out and "--tp 2" in out
    assert "calib render" in out and "chat=3,multilingual=2" in out
    assert "model download" not in out and "candidate 0 0.0 GB" in out and "unknown" not in out
    # the search reference 96 prompts x 2 x 8192 x 128 x 8 bytes, the holdout one 201 prompts (both in the cache)
    assert "total about 5.0 GB" in out and "holdout reference" in out
    # the run directory scales with candidate 0 (0 GB for this tiny model), then the run's own free-space floor
    assert "search run directory (typical: 6.5 x candidate 0) 0.0 GB" in out
    assert "and the run stops builds below 0.0 GB free (judge.min_free_bytes): keep both free" in out


def test_the_dry_run_counts_the_run_directory_and_the_floor_beside_the_downloads(tmp_path, monkeypatch):
    """The printed total once left out the run directory (a 30B MoE run used 101 GB beside an "at most 87.4 GB"):
    it is now a typical multiple of candidate 0, and the run's free-space floor is named beside the total."""
    Commands(monkeypatch)
    p = prepared(tmp_path)
    with open(p.model_dir / "model.safetensors", "r+b") as f:
        f.truncate(3_000_000_000)  # sparse: a 3 GB bf16 model, so candidate 0 (w4a16) is about 1 GB
    out = P.show_plan(p, P.plan(p, ("ref", "cand0")))
    assert "candidate 0 1.0 GB" in out and "search run directory (typical: 6.5 x candidate 0) 6.8 GB" in out
    assert "total at most" not in out and "total about 12.9 GB" in out
    assert "and the run stops builds below 6.0 GB free (judge.min_free_bytes): keep both free" in out


def test_the_dry_run_prints_the_gpu_hour_budget_and_its_deadline(tmp_path, monkeypatch, capsys):
    Commands(monkeypatch)
    out = tmp_path / "out"
    argv = ["prepare", "--model", str(model(tmp_path)), "--out", str(out), "--cards", "0,1", "--dry-run"]
    assert cli.main(argv) == 0
    line = "  budget_gpu_hours: {} over 2 card(s): the search stops {} h after it starts\n"
    assert line.format(4, 2) in capsys.readouterr().out  # the default, `--budget-gpu-hours`'s
    assert cli.main([*argv, "--budget-gpu-hours", "0.5"]) == 0
    assert line.format(0.5, 0.25) in capsys.readouterr().out


def test_a_w4a4_prepare_is_a_gptq_build_on_its_own_preset(tmp_path, monkeypatch):
    """`--format w4a4` is priced as a W4A4 candidate, not as the W4A8 family the activation spelling named."""
    monkeypatch.setattr(P.group_size, "target_for", lambda cards: Target.parse("sm100"))
    p = prepared(tmp_path, format="w4a4")
    assert P.FORMATS[p.format] == "gptq"
    size = dict(P.disk_estimate(p))["candidate 0"]
    assert size is not None and size > 0


def test_a_w4a4_prepare_writes_a_menu_every_sm100_entry_can_plan(tmp_path, monkeypatch):
    """The W4 default (w8a16 / w4a16 at group 16) is refused on sm100; a W4A4 run names the formats it serves."""
    from qda_quant.format import Format

    monkeypatch.setattr(P.group_size, "target_for", lambda cards: Target.parse("sm100"))
    p = prepared(tmp_path, format="w4a4")
    p.cand0.mkdir(parents=True)
    p.save(cand0={"build_s": 10.0})
    raw = P.search_config(p)
    menu = raw["judge"]["format_menu"]
    assert menu == ["bf16", "w8a8-fp8", "w4a4-nvfp4"]
    for name in menu:
        if name != "bf16":
            assert Target.parse("sm100").plan(Format.parse(name), "dense").kernel
    # the head lease is on by default for W4A4, at the row's 4.8 budget, and the row offers a bf16 or FP8 head
    from qda_agent.judge.checks.budget import lease_formats

    assert raw["judge"]["head_lease"] is True and raw["judge"]["budget_bpw"] == 4.8
    assert lease_formats(raw["format"]) == ("bf16", "fp8")


def test_a_prepare_refuses_before_its_first_step_when_a_cache_disk_is_short(
    tmp_path, monkeypatch, capsys, fake_room
):
    """The lines `qda doctor` prints, read before any output exists to be half written."""
    from qda_agent import room as R

    commands = Commands(monkeypatch)
    # one place on the short disk: where several share this host's disk the line names each (`fix: A; export B`)
    monkeypatch.setattr(R, "places", lambda: [R.Place("TMPDIR", tmp_path / "tmp", "the temporary files")])
    fake_room(1 << 40, 500)
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 3 and commands.calls == [] and not (tmp_path / "out").exists()
    assert err.startswith("error: ")  # the contract's line
    assert "500 free inodes are below the 10,000-inode floor" in err
    assert "fix: export TMPDIR=<a larger disk>" in err


def _tiny_child(monkeypatch):
    """`subprocess.run` that runs a real child exiting 1, as prepare's own commands do with check=True."""
    real = subprocess.run

    def run(argv, check=False, **kw):
        return real([sys.executable, "-c", "import sys; sys.exit(1)"], check=True)

    monkeypatch.setattr(subprocess, "run", run)


def _probe_fails_under(monkeypatch, where: Path):
    from qda_agent import room as R

    real = R._write_probe

    def write(directory):
        if Path(directory).is_relative_to(where):
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        real(directory)

    monkeypatch.setattr(R, "_write_probe", write)


def test_a_failed_child_on_a_full_cache_names_the_variable(tmp_path, monkeypatch, capsys):
    """The quota fills in the child process, which dies on its own: prepare finds which place is full."""
    hf = tmp_path / "hf"
    hf.mkdir()
    from qda_agent import room as R

    monkeypatch.setattr(R, "_hf_home", lambda: hf)
    _tiny_child(monkeypatch)
    _probe_fails_under(monkeypatch, hf)
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 3
    assert (
        err == f"error: HF_HOME ({hf}) is full; fix: export HF_HOME=<a directory on a disk with more room>\n"
    )


def test_a_failed_child_with_room_everywhere_keeps_the_old_refusal(tmp_path, monkeypatch, capsys):
    _tiny_child(monkeypatch)
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2 and "run the same prepare again" in err and "is full" not in err
    assert not list((tmp_path / "out").glob(".qda-probe-*"))


def test_a_failed_child_on_a_full_run_directory_says_to_pass_out(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _tiny_child(monkeypatch)
    _probe_fails_under(monkeypatch, out)
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(out)])
    err = capsys.readouterr().err
    assert code == 3
    assert err == f"error: --out ({out}) is full; fix: pass --out <a directory on a larger disk>\n"


def test_a_network_home_warns_but_does_not_refuse_prepare(tmp_path, monkeypatch, capsys, fake_room):
    """A user quota is invisible to statvfs: name the variables and their fix, and let the prepare run."""
    from qda_agent import room as R

    Commands(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(R, "places", lambda: [R.Place("QDA_CACHE", tmp_path / ".cache", "c", defaulted=True)])
    monkeypatch.setattr(R, "_mount_table", lambda: [(Path("/"), "nfs4")])
    fake_room(1 << 40, 10**6, 10**6)
    monkeypatch.setattr(P, "execute", lambda *_a, **_k: 0)  # the steps are not what this test is about
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 0
    assert "qda-agent prepare: warning: QDA_CACHE defaults into $HOME on nfs4" in err
    assert "fix: export QDA_CACHE=<a local disk>" in err


def test_a_prepare_for_cards_it_cannot_see_is_the_machine(tmp_path, monkeypatch, capsys):
    """`CUDA_VISIBLE_DEVICES=1` and `--cards 0`: the card is not there, so the exit status says so (3)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    args = [
        "prepare",
        "--model",
        str(model(tmp_path)),
        "--out",
        str(tmp_path / "out"),
        "--cards",
        "0",
        "--dry-run",
    ]
    assert cli.main(args) == 3
    err = capsys.readouterr().err
    assert err.startswith("error: cards [0] are not in CUDA_VISIBLE_DEVICES='1'")


def test_a_step_that_failed_names_the_command_and_the_rerun(tmp_path, monkeypatch, capsys):
    def fail(argv, **kw):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", fail)
    code = cli.main(["prepare", "--model", str(model(tmp_path)), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2 and "corpus: `" in err and "run the same prepare again" in err


def test_a_gguf_prepare_without_llama_cpp_is_refused_with_the_build_commands(tmp_path, monkeypatch, capsys):
    from qda_quant.gguf.llama_cpp import PINNED

    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    args = [
        "prepare",
        "--model",
        str(model(tmp_path)),
        "--out",
        str(tmp_path / "out"),
        "--format",
        "gguf-q4km",
    ]
    assert cli.main(args) == 3  # no llama.cpp on this machine: install it, do not change the command
    assert "export QDA_LLAMA_CPP=" in capsys.readouterr().err
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setenv("QDA_LLAMA_CPP", str(root))
    monkeypatch.setattr("qda_quant.gguf.llama_cpp.commit_of", lambda _r: PINNED)
    assert cli.main(args) == 3  # the checkout is there but its programs are not built
    err = capsys.readouterr().err
    assert "cmake --build" in err and "--target llama-server" in err and "probe.so" in err
    # the printed configure line is for the run's own card, not an H100's
    monkeypatch.setattr(P.group_size, "_smi_caps", lambda cards: [(str(c), (12, 0)) for c in cards])
    assert cli.main(args) == 3
    err = capsys.readouterr().err
    assert '-DCMAKE_CUDA_ARCHITECTURES="120"' in err and "ARCHITECTURES=90" not in err


@pytest.mark.parametrize(
    ("caps", "arch"),
    [
        ([(12, 0)], "120"),  # a PRO 6000
        ([(10, 3)], "103"),  # a GB300
        ([(9, 0), (9, 0)], "90"),  # two H100s: one architecture
        ([(12, 0), (9, 0)], "90;120"),  # a mixed set names both
        ([], "native"),  # no card the driver lists: CMake builds for the machine's own
    ],
)
def test_the_printed_llama_build_is_for_the_runs_card_architecture(tmp_path, monkeypatch, caps, arch):
    monkeypatch.setattr(
        P.group_size, "_smi_caps", lambda cards: [(str(c), cap) for c, cap in zip(cards, caps, strict=False)]
    )
    printed = P.llama_build(tmp_path / "llama.cpp", tmp_path / "build", list(range(len(caps) or 1)))
    assert f'-DCMAKE_CUDA_ARCHITECTURES="{arch}"' in printed
    assert "ARCHITECTURES=90" not in printed or arch == "90"
    assert 'ARCHITECTURES="native"' in P.llama_build(tmp_path / "l", tmp_path / "b")  # no cards named


def test_the_floor_step_says_what_it_runs_and_keeps_the_reading_it_measured(tmp_path, monkeypatch, capsys):
    """The floor decodes the bf16 GGUF and reads it with vLLM on one card; its `wall_s` is the one measured
    reading the judge has: the step names itself and the state keeps the number."""
    from test_judge import llama_paths

    paths = llama_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("qda_agent.cand0._llama_commit", lambda _root: None)  # no floor cache: a real reading
    p = prepared(tmp_path, format="gguf-q4km", cards=[0, 1, 2, 3])
    monkeypatch.setattr(P.Prepare, "make_bf16", lambda self: self.bf16)
    p.ref.mkdir(parents=True)
    (p.ref / "manifest.json").write_text(json.dumps({"content_sha256": "c"}))
    seen: list[list[str]] = []

    def score(argv, **kw):  # noqa: ANN001, ANN003, ANN202 -- `subprocess.run`'s own signature
        seen.append(argv)
        p.floor.mkdir(parents=True, exist_ok=True)
        (p.floor / "score.json").write_text(json.dumps({"kl": 0.0003, "wall_s": 548.19}))
        (p.floor / "gguf-decode.json").write_text(json.dumps({"reading_s": 601.5}))  # decode + load + scoring

    monkeypatch.setattr(P.subprocess, "run", score)
    next(s for s in P.plan(p, ("floor",))).run()
    assert "floor: decoding the bf16 GGUF" in capsys.readouterr().out  # not minutes of silence
    config = Path(seen[0][seen[0].index("score") + 1])
    assert config.name == "floor.yaml"
    body = yaml.safe_load(config.read_text())
    assert body["scorer"] == {"topk": {}} and body["snapshot"] == str(p.model_dir)
    assert body["checkpoint"] == str(p.bf16)
    assert (p.state["floor"]["seconds"], p.state["floor"]["cards"]) == (601.5, 1)
    assert p.state["floor"]["reader"] == P.FLOOR_READER
    step = next(s for s in P.plan(p, ("floor",)))
    assert step.done()  # this reading stands
    p.save(floor={**p.state["floor"], "reader": None})  # a floor llama.cpp's kernels read is measured again
    assert not step.done()


def test_a_floor_whose_bf16_does_not_fit_one_card_is_refused_before_the_reading(tmp_path, monkeypatch):
    """The floor loads the whole bf16 decode on one card: a larger model is refused with both sizes, not an OOM."""
    from test_judge import llama_paths

    paths = llama_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    p = prepared(tmp_path, format="gguf-q4km")
    p.bf16.parent.mkdir(parents=True, exist_ok=True)
    p.bf16.write_bytes(b"\0" * 2000)
    monkeypatch.setattr(P.Prepare, "make_bf16", lambda self: self.bf16)
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: 1000)
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: pytest.fail("the reading ran"))
    with pytest.raises(P.Refusal, match="on one card") as got:
        next(s for s in P.plan(p, ("floor",))).run()
    assert got.value.code == 3  # ENVIRONMENT


def test_the_disk_estimate_counts_a_gguf_reading_s_bf16_decode(tmp_path, monkeypatch):
    """Every GGUF reading and long-generation check decodes to bf16 under its `out` first, and one of each can
    run at once: the estimate holds two decodes beside the template."""
    p = prepared(tmp_path, format="gguf-q4km")
    monkeypatch.setattr(P, "_model_bytes", lambda _p: 10**9)
    rows = dict(P.disk_estimate(p))
    key = (
        "bf16 decodes of the running GGUF jobs (a KL reading and a long-generation check at once, "
        "each in its out/decoded, removed after it)"
    )
    assert rows[key] == 2 * 10**9


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    ("on_path", "home_var", "in_home", "cuda_cxx", "expected"),
    [
        (True, "CUDA_HOME", True, None, None),  # CMake finds nvcc on PATH itself
        (False, "CUDA_HOME", True, None, "home"),
        (False, "CUDA_PATH", True, None, "home"),
        (False, "CUDA_HOME", False, None, None),  # no nvcc anywhere: CMake's own error names it
        (False, "CUDA_HOME", True, "/my/nvcc", None),  # the user's CUDACXX is left to CMake
    ],
)
def test_the_printed_llama_build_names_nvcc_through_cudacxx_only_when_cmake_would_miss_it(
    tmp_path, monkeypatch, on_path, home_var, in_home, cuda_cxx, expected
):
    bin_dir = tmp_path / "path-bin"
    bin_dir.mkdir()
    if on_path:
        executable(bin_dir / "nvcc")
    home = tmp_path / "cuda"
    home.mkdir()
    nvcc = executable(home / "bin" / "nvcc") if in_home else None
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDACXX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv(home_var, str(home))
    if cuda_cxx:
        monkeypatch.setenv("CUDACXX", cuda_cxx)
    llama, build = tmp_path / "llama.cpp", tmp_path / "llama.cpp" / "build"
    plain = P.LLAMA_BUILD.format(llama=llama, build=build, probe=P.probe_source(), arch="native")
    printed = P.llama_build(llama, build)
    assert printed == (f'CUDACXX="{nvcc}" {plain}' if expected else plain)
    if expected:
        assert printed.splitlines()[0].startswith(f'CUDACXX="{nvcc}" cmake -S ')


def test_the_printed_llama_build_compiles_the_installed_probe_cpp(tmp_path):
    """A pip-installed QDA prints a probe.cpp it ships, not a `bench/gguf` path from a source clone."""
    probe = P.probe_source()
    assert probe.is_file() and probe.name == "probe.cpp"
    printed = P.llama_build(tmp_path / "llama.cpp", tmp_path / "llama.cpp" / "build")
    assert f'"{probe}"' in printed
    assert "bench/gguf" not in printed


def test_the_gguf_config_takes_its_budget_and_floor_from_candidate_0_and_the_floor_reading(
    tmp_path, monkeypatch
):
    from test_judge import llama_paths

    paths = llama_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    p = prepared(tmp_path, format="gguf-q4km")
    p.cand0.mkdir(parents=True)
    (p.cand0 / "model-Q4_K_M.gguf").write_bytes(b"\0" * 1234)
    p.save(cand0={"build_s": 10.0}, floor={"kl": 0.0003, "ref": "c", "seconds": 548.19, "cards": 4})
    raw = P.search_config(p)
    assert raw["judge"]["budget_bytes"] == 1234 and raw["judge"]["engine_floor"] == 0.0003
    assert raw["judge"]["llamacpp"]["server"] == paths["server"] and "llama_server" not in raw["judge"]
    assert "format_reference" not in raw["judge"]
    # the floor step's own measured reading: the judge's watchdog estimates llama.cpp by it, not by kl_score_s
    assert raw["judge"]["reading_s"] == 548.19 and raw["judge"]["measured_cards"] == 4
    assert raw["baselines"] == [str(p.cand0)] and configfile.placeholders(raw) == []


def test_a_gguf_run_names_its_format_gguf_q4km_in_the_task_and_end_report(tmp_path, monkeypatch):
    """A `--format gguf-q4km` run is called that in search.yaml, TASK.md, the goal and END-REPORT, not w4a16."""
    import time

    from qda_agent import flow, prompt
    from qda_agent.host import Final
    from qda_agent.run_config import prepare_again
    from test_judge import llama_paths

    paths = llama_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    p = prepared(tmp_path, format="gguf-q4km")
    p.cand0.mkdir(parents=True)
    (p.cand0 / "model-Q4_K_M.gguf").write_bytes(b"\0" * 1234)
    p.save(cand0={"build_s": 10.0}, floor={"kl": 0.0003, "ref": "c", "seconds": 5.0})
    raw = P.search_config(p)

    def named_paths(node):  # noqa: ANN001, ANN202
        if isinstance(node, dict):
            return [x for v in node.values() for x in named_paths(v)]
        if isinstance(node, list):
            return [x for v in node for x in named_paths(v)]
        return [Path(node)] if isinstance(node, str) and node.startswith(str(p.out)) else []

    for made in named_paths(raw):  # every input the config names, as the earlier steps leave them
        if made != p.out / "run" and not made.exists():
            made.parent.mkdir(parents=True, exist_ok=True)
            made.write_text("{}\n") if made.suffix else made.mkdir()
    p.config.write_text(P.HEADER.format(out=p.out) + yaml.safe_dump(raw))
    cfg = RunConfig.load(p.config)
    assert cfg.format == "gguf-q4km" and "--format gguf-q4km" in prepare_again(p.config)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    assert "the gguf-q4km tier" in prompt.goal(cfg.format, t)
    flow.prepare(cfg, t, t, installed=True)
    task = (cfg.workdir / "TASK.md").read_text()
    assert task.startswith("# Goal: the best gguf-q4km tier") and "the best gguf-q4km quantization" in task
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    log = flow._Log(cfg.run_dir / "flow.jsonl", time.time)
    now = time.time()
    final = Final(rule="handin", choice=None, reason="nothing handed in")
    flow._end_report(cfg, final, log, log, flow.EndKind.DEADLINE, "at the deadline", None, now, now, 1.0)
    assert "format `gguf-q4km`" in (cfg.run_dir / "END-REPORT.md").read_text()


def test_evaluate_scores_the_winner_and_candidate_0_on_a_holdout_reference(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    holdout = tmp_path / "data" / "corpus.holdout.jsonl"
    holdout.parent.mkdir()
    holdout.write_text("")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text(
        json.dumps({"build": {"sampling": {"temperature": 1.0, "max_tokens": 64}}})
    )
    judge = {
        "kl_ref": str(ref),
        "kl_holdout_ref": str(holdout),
        "anchor": "m",
        "serve_python": sys.executable,
    }
    resolved = {"model": "m", "cards": [0], "baselines": [str(tmp_path / "c0")], "judge": judge}
    (run / "resolved.yaml").write_text(yaml.safe_dump(resolved))
    (run / "FINAL.json").write_text(
        json.dumps(
            {
                "choice": {"dir": str(run / "winner")},
                "export": {"winner": str(run / "winner")},
                "reading": False,  # FINAL.json's two facts, as the judge now records them
                "gencheck": False,
            }
        )
    )
    (run / "winner").mkdir()
    (run / "winner" / "resolved.yaml").write_text(
        yaml.safe_dump({"steps": [{"rotate": {"residual": True, "seed": 3}}]})
    )
    calls = []

    def fake(argv, **kw):
        calls.append(argv)
        if "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            kl = 0.01 if body["checkpoint"].endswith("winner") else 0.02
            Path(body["out"], "score.json").write_text(json.dumps({"kl": kl}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(["evaluate", str(run)]) == 0
    out = capsys.readouterr().out
    linked = Path(calls[0][calls[0].index("--out") + 1])
    assert linked == run / "evaluate" / "holdout.ref" and "--cache" in calls[0]
    assert calls[0][calls[0].index("--max-tokens") + 1] == "64"
    assert "verdict: the winner is better than candidate 0" in out and "0.500" in out
    # FINAL.json recorded neither a reading nor a check of the winner, and evaluate says so
    assert (
        "WARNING: The delivered checkpoint has no KL reading and no long-generation check (`gencheck`)."
        in out
    )

    # the winner's own record says it is residual-rotated, so a draft must be rotated with it
    assert "rotation: this checkpoint runs in a rotated residual basis" in out
    assert f"qda rotate-draft --checkpoint {run / 'winner'}" in out and "DFlash / DSpark" in out
    assert evaluate.winner(run) == run / "winner"
    # both facts recorded: nothing is named (the same FINAL.json fields END-REPORT.md reads)
    final = json.loads((run / "FINAL.json").read_text())
    final.update(reading=True, gencheck=True)
    (run / "FINAL.json").write_text(json.dumps(final))
    assert cli.main(["evaluate", str(run)]) == 0
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize(("rc", "want"), [(3, 3), (2, 2), (1, 1), (-15, 130)])
def test_evaluate_reports_the_childs_own_status(tmp_path, monkeypatch, capsys, rc, want):
    """The scoring child is a `qda` command: its status is the answer, and a kill it did not catch is 130."""
    run = tmp_path / "run"
    run.mkdir()
    holdout = tmp_path / "data" / "corpus.holdout.jsonl"
    holdout.parent.mkdir()
    holdout.write_text("")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text(
        json.dumps({"build": {"sampling": {"temperature": 1.0, "max_tokens": 64}}})
    )
    judge = {
        "kl_ref": str(ref),
        "kl_holdout_ref": str(holdout),
        "anchor": "m",
        "serve_python": sys.executable,
    }
    resolved = {"model": "m", "cards": [0], "baselines": [str(tmp_path / "c0")], "judge": judge}
    (run / "resolved.yaml").write_text(yaml.safe_dump(resolved))
    (run / "FINAL.json").write_text(
        json.dumps({"choice": {"dir": str(run / "winner")}, "export": {"winner": str(run / "winner")}})
    )

    def fail(argv, **kw):
        raise subprocess.CalledProcessError(rc, argv)

    monkeypatch.setattr(subprocess, "run", fail)
    assert cli.main(["evaluate", str(run)]) == want
    note, err = capsys.readouterr().err.split("\n", 1)  # the alias's note line, then the refusal
    assert note == evaluate.NOTE and err.startswith("error: `")


def test_a_reference_is_not_reused_across_a_downloaded_revision(tmp_path):
    """Prepare revA, then revB into the same DIR. The model step records revB's sha before the ref step
    runs, and a reference built from revA's is not reused (the download is faked here)."""
    p = P.Prepare("org/m@revA", tmp_path / "out")
    p.save(
        model={"repo": "org/m", "requested": "revA", "revision": "shaA"},
        ref_source="org/m@shaA",
        search96={"ids": "x", "sha256": "y", "prompts": "z"},
    )
    p.model_dir.mkdir(parents=True, exist_ok=True)
    (p.model_dir / "config.json").write_text("{}")
    p.search.parent.mkdir(parents=True, exist_ok=True)
    p.search.write_text("")
    p.ref.mkdir(parents=True)
    (p.ref / "manifest.json").write_text(
        json.dumps({"build": {"prompts": str(p.search), "prompts_sha256": P.sha256(p.search)}})
    )
    assert P.plan(p, ("ref",))[0].done() is not None
    other = P.Prepare("org/m@revB", p.out)
    other.save(
        model={"repo": "org/m", "requested": "revB", "revision": "shaB"}
    )  # what its model step records
    assert P.plan(other, ("ref",))[0].done() is None


def test_a_reference_and_candidate_0_are_not_reused_for_other_weights(tmp_path, monkeypatch):
    """Other weights in the same DIR leave the reference and candidate 0 of the old model; they are
    refused, never mixed with the new source (a replaced local file is the same case as another revision)."""
    Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p, ("model", "corpus", "ref", "calib", "cand0")), log=lambda _: None)
    (p.model_dir / "model.safetensors").write_bytes(b"x" * 1000)  # other weights, same tokenizer and config
    again = P.Prepare(p.model, p.out, python=p.python, serve_python=p.serve_python)
    assert P.plan(again, ("ref",))[0].done() is None
    assert P.plan(again, ("cand0",))[0].done() is None
    with pytest.raises(P.Refusal, match="remove it to build the reference again"):
        P.execute(again, P.plan(again, ("ref",)), log=lambda _: None)
    with pytest.raises(P.Refusal, match="unfinished or other build"):
        P.execute(again, P.plan(again, ("cand0",)), log=lambda _: None)


def test_a_holdout_reference_comes_from_the_cache_and_keeps_its_layout(tmp_path):
    """The holdout reference is the reference cache's (keyed there by the original's content, so two
    fine-tunes sharing a holdout file get one each), and it repeats the run reference's own tp and serve_args,
    not the number of cards the run held."""
    holdout = tmp_path / "corpus.holdout.jsonl"
    holdout.write_text("")
    run_ref = tmp_path / "ref"
    run_ref.mkdir()
    (run_ref / "manifest.json").write_text(
        json.dumps(
            {
                "build": {
                    "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 64},
                    "answers": 3,
                    "seed": 5,
                    "tp": 2,
                    "serve_args": {"max_num_seqs": 8},
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            }
        )
    )
    judge = {"kl_ref": str(run_ref), "kl_holdout_ref": str(holdout)}
    first = {"model": "org/a", "cards": [0], "judge": {**judge, "anchor": "org/a"}}
    second = {"model": "org/b", "cards": [0], "judge": {**judge, "anchor": "org/b"}}
    ref_a, command = rundir.holdout_reference(first, sys.executable, 4, tmp_path / "run-a")
    ref_b, other = rundir.holdout_reference(second, sys.executable, 4, tmp_path / "run-b")
    assert ref_a != ref_b and "--cache" in command
    assert command[command.index("--model") + 1] == "org/a" and other[other.index("--model") + 1] == "org/b"
    assert command[command.index("--tp") + 1] == "2"  # the reference's layout, not the four cards
    assert json.loads(command[command.index("--serve-args") + 1]) == {"max_num_seqs": 8}
    assert command[command.index("--answers") + 1] == "3"


def test_prepare_and_evaluate_ask_for_one_holdout_reference(tmp_path, monkeypatch):
    """Prepare's warmed holdout reference names no `--sampling`/`--serve-args` (the engine derives them);
    evaluate repeats the search reference's recorded layout as explicit flags. The cache key is the effective
    settings, so evaluate hits the entry prepare warmed instead of building the reference again."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    m = model(tmp_path, {"temperature": 1.0, "top_k": 20, "top_p": 0.95})
    prompts = tmp_path / "holdout.jsonl"
    prompts.write_text("")
    pin = {"compilation_config": {"inductor_compile_config": {"deterministic": True}}}
    layout = rundir.reading_layout(
        {
            "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
            "answers": 2,
            "seed": 17,
            "tp": 1,
            "serve_args": pin,
        },
        1,
    )
    prepared = {"answers": 2, "seed": 17, "max_tokens": 8192, "tp": 1}  # what prepare's ref command passes
    evaluated = {
        "answers": layout["answers"],
        "seed": layout["seed"],
        "max_tokens": layout["max_tokens"],
        "tp": layout["tp"],
        "sampling": layout["sampling"],
        "serve_args": layout["serve_args"],
        "chat_template_kwargs": layout["chat_template_kwargs"],
    }
    assert refcache.entry(refcache.key_payload(m.resolve(), prompts, prepared)) == refcache.entry(
        refcache.key_payload(m.resolve(), prompts, evaluated)
    )


def test_a_score_is_not_reused_after_the_holdout_prompts_change(tmp_path, monkeypatch):
    """Evaluate again after the holdout prompts changed builds a new reference and scores again; the old
    reading (against the old reference) is not printed beside the new one."""
    from qda_kl.reference import model_identity

    run, c0 = tmp_path / "run", tmp_path / "c0"
    run.mkdir()
    c0.mkdir()
    (c0 / "config.json").write_text("{}")
    holdout = tmp_path / "corpus.holdout.jsonl"
    holdout.write_text("a")
    run_ref = tmp_path / "ref"
    run_ref.mkdir()
    (run_ref / "manifest.json").write_text(
        json.dumps({"build": {"sampling": {"temperature": 1.0, "max_tokens": 64}}})
    )
    judge = {
        "kl_ref": str(run_ref),
        "kl_holdout_ref": str(holdout),
        "anchor": "m",
        "serve_python": sys.executable,
    }
    (run / "resolved.yaml").write_text(
        yaml.safe_dump({"model": "m", "cards": [0], "baselines": [str(c0)], "judge": judge})
    )
    (run / "FINAL.json").write_text(
        json.dumps({"baseline": str(c0), "chosen_by": "fallback_candidate0", "reason": "x"})
    )
    scores = []

    def fake(argv, **kw):
        if "ref" in argv:
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            sha = P.sha256(Path(argv[argv.index("--prompts") + 1]))
            (out / "manifest.json").write_text(
                json.dumps(
                    {"content_sha256": sha, "build": {"sampling": {"temperature": 1.0, "max_tokens": 64}}}
                )
            )
        elif "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            scores.append(body)
            content = json.loads((Path(body["reference"]) / "manifest.json").read_text())["content_sha256"]
            Path(body["out"], "score.json").write_text(
                json.dumps(
                    {
                        "kl": 0.01,
                        "checkpoint": model_identity(body["checkpoint"]),
                        "reference": {"content_sha256": content},
                    }
                )
            )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(["evaluate", str(run)]) == 0
    holdout.write_text("b")  # another prompt set: another reference, so another reading
    assert cli.main(["evaluate", str(run)]) == 0
    assert len(scores) == 2  # the first reading was not reused


def test_evaluate_on_a_run_that_kept_candidate_0_says_why_and_scores_only_it(tmp_path, monkeypatch, capsys):
    """A run whose agent never ran falls back to candidate 0: one line says so, and only candidate 0 is scored."""
    run, c0 = tmp_path / "run", tmp_path / "c0"
    run.mkdir()
    ref = tmp_path / "holdout-ref"
    ref.mkdir()
    judge = {"kl_ref": str(ref), "kl_holdout_ref": str(ref), "anchor": "m", "serve_python": sys.executable}
    resolved = {"model": "m", "cards": [0], "baselines": [str(c0), str(tmp_path / "c1")], "judge": judge}
    (run / "resolved.yaml").write_text(yaml.safe_dump(resolved))
    why = "the agent never ran (HarnessRefused: Not logged in); candidate 0 as-is"
    final = {"rule": "handin", "baseline": str(c0), "chosen_by": "fallback_candidate0", "reason": why}
    (run / "FINAL.json").write_text(json.dumps({**final, "agent_ran": False}))
    scored = []

    def fake(argv, **kw):
        body = yaml.safe_load(Path(argv[-1]).read_text())
        scored.append(body["checkpoint"])
        Path(body["out"], "score.json").write_text(json.dumps({"kl": 0.02}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(["evaluate", str(run)]) == 0
    out = capsys.readouterr().out
    assert scored == [str(c0)]
    assert f"verdict: the run kept candidate 0 ({why}); only candidate 0 is scored" in out


def test_search_runs_the_doctor_on_the_prepared_config_then_the_run_decides(tmp_path, monkeypatch, capsys):
    """A prepare directory's search.yaml goes to doctor and then to the run; doctor's status no longer stops the
    search: here doctor misses (1) and the run's own refusal of the empty config is the status (2)."""
    from qda_quant import cli as qda

    asked = []
    monkeypatch.setattr(qda, "main", lambda argv: asked.append(argv) or 1)
    (tmp_path / "search.yaml").write_text("")
    assert cli.main(["search", str(tmp_path), "--set", f"run_dir={tmp_path / 'r2'}"]) == 2
    assert asked == [
        ["doctor", "--search", str(tmp_path / "search.yaml"), "--set", f"run_dir={tmp_path / 'r2'}"]
    ]
    err = capsys.readouterr().err
    assert "qda doctor reported a miss above" in err and "error: " in err, err
    assert not (tmp_path / "r2").exists(), "the refused run created nothing"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, 32768),
        ({"max_position_embeddings": 16384}, 16384),
        ({"text_config": {"max_position_embeddings": 4096}}, 4096),
        ({"max_position_embeddings": 131072}, 32768),
    ],
)
def test_the_evaluation_cap_is_the_models_context_when_that_is_smaller(tmp_path, config, expected):
    """A 16k-context model's TASK.md promised 32768-token evaluations; the cap now reads the context the
    reference reads."""
    p = prepared(tmp_path)
    (p.model_dir / "config.json").write_text(json.dumps(config))
    p.cand0.mkdir(parents=True)
    p.save(cand0={"build_s": 10.0})
    assert P.search_config(p)["judge"]["eval_max_tokens"] == expected


def test_the_ref_step_says_where_its_time_went():
    """The step line's split comes from the reference's own record, never from this step's wall time."""
    built = {
        "build": {"seconds": {"engine_start": 100.0, "generate": 240.0, "teacher_force": 35.0, "total": 380}}
    }
    line = P.ref_split(built)
    assert "engine start 100.0 s" in line and "generate 240.0 s" in line and "teacher force 35.0 s" in line
    assert "other 5.0 s" in line


def _reference_recording(base, recorded: str, digest: str):
    """A reference directory whose manifest records `recorded` (and `digest`) as its prompt file."""
    ref = Path(base) / "ref"
    ref.mkdir(parents=True)
    (ref / "manifest.json").write_text(
        json.dumps({"k": 1, "vocab": 1, "build": {"prompts": recorded, "prompts_sha256": digest}})
    )
    return ref


ROWS_ONE = '{"id": "a", "prompt": "x"}\n'


def test_the_ref_step_is_not_done_while_its_prompt_file_is_gone(tmp_path):
    """A reference whose prompt file was deleted with another run's directory cannot be indexed, so
    prepare does not report the step finished; rerunning it hands the cache entry its own copy."""
    prompts = tmp_path / "run" / "data" / "corpus.search96.jsonl"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(ROWS_ONE)
    digest = hashlib.sha256(ROWS_ONE.encode()).hexdigest()
    ref = _reference_recording(tmp_path, str(prompts), digest)
    assert P.reference_prompts_readable(ref)
    prompts.unlink()
    assert not P.reference_prompts_readable(ref)


def test_the_ref_step_is_done_when_the_entry_keeps_its_prompt_copy(tmp_path):
    """A relative record beside the prompt file (a cache entry's own copy) resolves without the builder's run."""
    copy = tmp_path / refcache.PROMPTS_COPY
    copy.parent.mkdir(parents=True)
    copy.write_text(ROWS_ONE)
    digest = hashlib.sha256(ROWS_ONE.encode()).hexdigest()
    assert P.reference_prompts_readable(_reference_recording(tmp_path, refcache.PROMPTS_COPY, digest))


def test_the_bf16_template_is_converted_once_per_model_and_llama_cpp(tmp_path, monkeypatch):
    """A second prepare of the same model (another directory, the same files) links the cached template
    instead of converting 16 GB again; the link is a hard link, so a sandbox that binds the path reads it."""
    from qda_agent import cand0

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cand0, "_llama_commit", lambda _root: "a" * 40)
    converted = []

    def convert(argv, check=False, env=None, **kw):
        converted.append(argv)
        Path(argv[argv.index("--outfile") + 1]).write_bytes(b"GGUF" * 8)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", convert)
    outs = []
    for run in ("a", "b"):
        model = tmp_path / run / "model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"w" * 16)
        out = tmp_path / run / "bf16" / "model-bf16.gguf"
        cache = P.template_cache(model, tmp_path / "llama")
        assert (
            cand0.make_template(
                sys.executable, tmp_path / "llama", model, out, cache=cache, log=lambda _: None
            )
            == out
        )
        outs.append(out)
    assert len(converted) == 1
    assert outs[0].read_bytes() == outs[1].read_bytes() == b"GGUF" * 8
    assert outs[0].stat().st_ino == outs[1].stat().st_ino  # one file on disk
    assert json.loads(outs[1].with_name(outs[1].name + ".json").read_text())["source"] == str(
        (tmp_path / "b" / "model").resolve()
    )  # each directory's record still names its own source


def test_the_engine_floor_is_read_once_per_model_reference_and_llama_cpp(tmp_path, monkeypatch):
    """The floor (the bf16 GGUF read by llama.cpp against the reference, ~10 minutes on 8B) is a pure
    function of the model's files, llama.cpp and the reference's content: a second prepare reuses the reading."""
    from test_judge import llama_paths

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    paths = llama_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    monkeypatch.setattr("qda_agent.cand0._llama_commit", lambda _root: "a" * 40)
    monkeypatch.setattr(P.Prepare, "make_bf16", lambda self: self.bf16)
    scored = []

    def score(argv, check=False, env=None, **kw):
        body = yaml.safe_load(Path(argv[-1]).read_text())
        scored.append(body)
        Path(body["out"]).mkdir(parents=True, exist_ok=True)
        (Path(body["out"]) / "score.json").write_text(json.dumps({"kl": 0.00035}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", score)
    kls = []
    for run in ("a", "b"):
        p = prepared(tmp_path / run, format="gguf-q4km")
        p.ref.mkdir(parents=True)
        (p.ref / "manifest.json").write_text(json.dumps({"content_sha256": "c"}))
        P.execute(p, P.plan(p, ("floor",)), log=lambda _: None)
        kls.append(p.state["floor"]["kl"])
    assert len(scored) == 1 and kls == [0.00035, 0.00035]
    other = prepared(tmp_path / "c", format="gguf-q4km")
    other.ref.mkdir(parents=True)
    (other.ref / "manifest.json").write_text(json.dumps({"content_sha256": "another reference"}))
    P.execute(other, P.plan(other, ("floor",)), log=lambda _: None)
    assert len(scored) == 2  # another reference is another reading


def test_the_ref_step_warms_the_holdout_reference_in_the_same_engine_load(tmp_path, monkeypatch):
    """One `qda-kl ref --cache` builds the search reference and warms the holdout cache entry; the holdout
    group names no output (evaluate links it into the run when it needs it)."""
    commands = Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p, ("corpus", "ref")), log=lambda _: None)
    ref = commands.calls[1]
    assert "--cache" in ref
    group = json.loads(ref[ref.index("--group") + 1])
    assert group == {"prompts": str(p.holdout)}  # warmed in the cache, not linked under --out
    assert ref.count("--group") == 1


def _fake_llama(tmp_path: Path, monkeypatch) -> Path:
    """A pinned checkout whose llama.cpp programs, bridge and native quantizer library exist."""
    from qda_quant.gguf.llama_cpp import PINNED

    root = tmp_path / "llama.cpp"
    root.mkdir(exist_ok=True)
    (root / "convert_hf_to_gguf.py").write_text("")
    for name in ("llama-server", "llama-imatrix", "libggml-base.so"):
        p = root / "build" / "bin" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    (root / "build" / "probe.so").write_bytes(b"")
    monkeypatch.setenv("QDA_LLAMA_CPP", str(root))
    monkeypatch.setattr("qda_quant.gguf.llama_cpp.commit_of", lambda _root: PINNED)
    return root


def test_a_gguf_prepare_measures_one_imatrix_and_candidate_0_reads_it(tmp_path, monkeypatch):
    """The imatrix step measures llama-imatrix once over the calibration, and cand0's recipe names the file."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    commands = Commands(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km")
    steps = ("model", "corpus", "ref", "calib", "imatrix", "cand0")
    P.execute(p, P.plan(p, steps), log=lambda _line: None)
    measured = [c for c in commands.calls if "llama-imatrix" in c[0]]
    assert len(measured) == 1
    argv = measured[0]
    assert argv[argv.index("-c") + 1] == "2048" and argv[argv.index("--chunks") + 1] == "128"
    assert "--process-output" in argv and argv[argv.index("-m") + 1] == str(p.bf16)
    assert p.imatrix.is_file() and not p.imatrix.is_symlink()
    record = json.loads(p.imatrix.with_name(p.imatrix.name + ".json").read_text())
    assert record["calib_sha256"] == P.sha256(p.calib) and record["chunks"] == 128
    exporter = yaml.safe_load((p.out / "cand0-gguf-q4km.recipe.yaml").read_text())["exporter"]["gguf"]
    assert exporter["imatrix"] == str(p.imatrix) and exporter["template"] == str(p.bf16)

    commands.calls.clear()  # the cache makes the second prepare's imatrix step a skip, not a measurement
    again = P.Prepare(p.model, p.out, format="gguf-q4km", python=p.python, serve_python=p.serve_python)
    P.execute(again, P.plan(again, ("imatrix",)), log=lambda _line: None)
    assert commands.calls == []


def _hub_imatrix(tmp_path: Path, monkeypatch, revision: str) -> tuple[P.Prepare, list[str]]:
    """A GGUF prepare of the Hub model `org/m` at `revision` (already in `DIR/model`), its imatrix step stubbed.

    Returns the prepare and the list every real measurement appends the measured revision to.
    """
    _fake_llama(tmp_path, monkeypatch)
    measured: list[str] = []
    p = prepared(tmp_path, model="org/m", format="gguf-q4km")
    p.model_dir.mkdir(parents=True, exist_ok=True)
    (p.model_dir / "config.json").write_text("{}")
    p.save(model={"repo": "org/m", "requested": "main", "revision": revision})
    p.calib.parent.mkdir(parents=True, exist_ok=True)
    p.calib.write_text('{"text": "t"}\n')

    def cached(**kw):
        measured.append(revision)
        out = tmp_path / "cache" / f"{revision}.imatrix.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"GGUF-imatrix")
        return out, True

    from qda_agent import imatrix_cache

    monkeypatch.setattr(P, "imatrix_program", lambda *_: Path("llama-imatrix"))
    monkeypatch.setattr(P.Prepare, "make_bf16", lambda self: self.bf16)
    monkeypatch.setattr(imatrix_cache, "cached", cached)
    monkeypatch.setattr(imatrix_cache, "text_file", lambda calib, out: Path(out))
    return p, measured


def test_the_imatrix_record_names_the_model_as_the_judge_does_and_prepare_reuses_it(tmp_path, monkeypatch):
    """`source` is the model directory the run config names; prepare reuses on `source_identity`."""
    p, measured = _hub_imatrix(tmp_path, monkeypatch, "shaA")
    P.execute(p, P.plan(p, ("imatrix",)), log=lambda _line: None)
    record = json.loads(p.imatrix.with_name(p.imatrix.name + ".json").read_text())
    assert record["source"] == str(p.model_dir.resolve()) and record["source_identity"] == "org/m@shaA"
    assert record["source_sha256"] and measured == ["shaA"]

    again = P.Prepare(p.model, p.out, format="gguf-q4km", python=p.python, serve_python=p.serve_python)
    P.execute(again, P.plan(again, ("imatrix",)), log=lambda _line: None)
    assert measured == ["shaA"], "the same model at the same revision reuses its matrix"


def test_another_revision_of_the_model_measures_its_own_imatrix(tmp_path, monkeypatch):
    p, measured = _hub_imatrix(tmp_path, monkeypatch, "shaA")
    P.execute(p, P.plan(p, ("imatrix",)), log=lambda _line: None)
    p.save(model={"repo": "org/m", "requested": "main", "revision": "shaB"})  # the model step moved on
    again = P.Prepare(p.model, p.out, format="gguf-q4km", python=p.python, serve_python=p.serve_python)
    P.execute(again, P.plan(again, ("imatrix",)), log=lambda _line: None)
    assert len(measured) == 2
    record = json.loads(p.imatrix.with_name(p.imatrix.name + ".json").read_text())
    assert record["source_identity"] == "org/m@shaB" and record["source"] == str(p.model_dir.resolve())


def test_candidate_0_refuses_without_the_importance_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    Commands(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km")
    P.execute(p, P.plan(p, ("model", "corpus", "ref", "calib")), log=lambda _line: None)
    with pytest.raises(P.Refusal, match="no importance matrix"):
        P.execute(p, P.plan(p, ("cand0",)), log=lambda _line: None)


def test_the_gguf_config_names_the_run_imatrix(tmp_path, monkeypatch):
    from test_judge import llama_paths

    paths = llama_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    p = prepared(tmp_path, format="gguf-q4km")
    p.cand0.mkdir(parents=True)
    (p.cand0 / "model-Q4_K_M.gguf").write_bytes(b"\0" * 1234)
    p.imatrix.parent.mkdir(parents=True)
    p.imatrix.write_bytes(b"GGUF-imatrix")
    p.save(cand0={"build_s": 10.0}, floor={"kl": 0.0003, "ref": "c"})
    raw = P.search_config(p)
    assert raw["judge"]["llamacpp"]["imatrix"] == str(p.imatrix)


def test_the_head_rule_is_read_from_the_sources_own_release_config(tmp_path):
    """`prepare` writes the source's own head rule; the judge and TASK.md read that one value."""
    assert P.head_rule(model(tmp_path / "plain")) == "16-bit"
    keep = model(tmp_path / "keep")
    (keep / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "compressed-tensors", "ignore": ["lm_head"]}})
    )
    assert P.head_rule(keep) == "16-bit"
    w3 = model(tmp_path / "w3")
    (w3 / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": ["model.embed_tokens"],
                    "config_groups": {
                        "group_0": {"targets": ["Linear"], "weights": {"num_bits": 3, "group_size": 128}},
                        "group_head": {
                            "targets": ["re:.*lm_head$"],
                            "weights": {"num_bits": 4, "group_size": 64, "symmetric": True},
                        },
                    },
                }
            }
        )
    )
    assert P.head_rule(w3) == "w4a16-sym-g64"
    tied = model(tmp_path / "tied")
    (tied / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    assert P.head_rule(tied) == "16-bit"


def test_the_head_rule_reads_a_gguf_sources_output_type(tmp_path):
    """A GGUF release states its head in the header; the config path is not needed."""
    from qda_quant.gguf.rule import BF16, Q8_0
    from test_judge import write_gguf

    quantized = tmp_path / "quantized"
    quantized.mkdir()
    write_gguf(
        quantized / "m.gguf",
        [
            ("output.weight", (256, 2), Q8_0, b"\0" * (8 * 34 * 2)),
            ("blk.0.attn_norm.weight", (256,), BF16, b"\0" * 512),
        ],
    )
    assert P.head_rule(quantized) == "Q8_0"
    sixteen = tmp_path / "sixteen"
    sixteen.mkdir()
    write_gguf(
        sixteen / "m.gguf",
        [
            ("output.weight", (256, 2), BF16, b"\0" * 1024),
            ("blk.0.attn_norm.weight", (256,), BF16, b"\0" * 512),
        ],
    )
    assert P.head_rule(sixteen) == "16-bit"


def _finished_run(tmp_path: Path, *, kept: bool = False, anchor: str = "m") -> Path:
    """A run directory `evaluate` can read: the holdout is already a reference, both checkpoints exist."""
    run, c0, win = tmp_path / "run", tmp_path / "c0", tmp_path / "win"
    for path in (run, c0, win):
        path.mkdir(exist_ok=True)
    ref = tmp_path / "holdout-ref"
    ref.mkdir()
    judge = {"kl_ref": str(ref), "kl_holdout_ref": str(ref), "anchor": anchor, "serve_python": sys.executable}
    resolved = {"model": "m", "cards": [0], "baselines": [str(c0)], "judge": judge}
    (run / "resolved.yaml").write_text(yaml.safe_dump(resolved))
    final = (
        {"baseline": str(c0), "chosen_by": "fallback_candidate0", "reason": "the agent never ran"}
        if kept
        else {"baseline": str(c0), "choice": {"dir": str(win)}}
    )
    (run / "FINAL.json").write_text(json.dumps(final))
    return run


def test_evaluate_suite_runs_one_benchmark_and_prints_its_table(tmp_path, monkeypatch, capsys):
    """`--suite` writes a qda eval config from the run's own facts, runs that benchmark, then its table."""
    run = _finished_run(tmp_path)
    monkeypatch.setattr(rundir, "suite_ready", lambda _name: None)
    calls: list[list[str]] = []

    def fake(argv, **kw):
        calls.append([str(a) for a in argv])
        if "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            Path(body["out"], "score.json").write_text(json.dumps({"kl": 0.02}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    spec = "{lm_eval: {tasks: [gpqa_diamond], repeats: 2}}"
    assert cli.main(["evaluate", str(run), "--suite", spec]) == 0
    preset = yaml.safe_load((run / "evaluate" / "lm_eval" / "eval.yaml").read_text())
    assert preset["reference"] == "m"
    assert preset["checkpoints"] == {"candidate0": str(tmp_path / "c0"), "winner": str(tmp_path / "win")}
    assert preset["suites"][0]["lm_eval"]["tasks"] == ["gpqa_diamond"]
    assert preset["out"] == str(run / "evaluate" / "lm_eval")
    eval_cmd = next(c for c in calls if "eval" in c and "table" not in c)
    table_cmd = next(c for c in calls if "table" in c)
    assert eval_cmd[-1] == str(run / "evaluate" / "lm_eval" / "eval.yaml")
    assert table_cmd[table_cmd.index("--baseline") + 1] == "candidate0"
    assert "verdict:" in capsys.readouterr().out


def test_evaluate_suite_dry_run_lists_jobs_and_scores_nothing(tmp_path, monkeypatch, capsys):
    run = _finished_run(tmp_path)
    monkeypatch.setattr(rundir, "suite_ready", lambda _name: None)
    calls: list[list[str]] = []

    def fake(argv, **kw):
        calls.append([str(a) for a in argv])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    spec = "{lm_eval: {tasks: [gpqa_diamond]}}"
    assert cli.main(["evaluate", str(run), "--suite", spec, "--dry-run"]) == 0
    assert calls and calls[0][-1] == "--dry-run"
    assert not any("score" in c for c in calls)


def test_evaluate_suite_refuses_what_it_cannot_run(tmp_path, monkeypatch, capsys):
    """Refused through `qda eval RUN_DIR`: the stderr contract without the alias's note line."""
    from qda_quant.cli import main as qda

    run = _finished_run(tmp_path, kept=True)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0))
    spec = "{lm_eval: {tasks: [gpqa_diamond]}}"
    assert qda(["eval", str(run), "--suite", "nope"]) == 2  # the input: USAGE
    err = capsys.readouterr().err  # the contract's one `error:` line, naming the four presets
    assert err.startswith("error: ") and "no such preset" in err and "kl, sanity, full, specdec" in err
    assert err.count("\n") == 1
    assert qda(["eval", str(run), "--suite", "{nope: {}}"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "unknown suite" in err
    real = __import__("importlib").util.find_spec
    monkeypatch.setattr(
        __import__("importlib").util, "find_spec", lambda name: None if name == "lm_eval" else real(name)
    )
    assert qda(["eval", str(run), "--suite", spec]) == 3  # the machine: ENVIRONMENT
    assert capsys.readouterr().err == (
        'error: lm_eval needs the harness, which is not installed\nfix:   uv pip install -c constraints.txt "./packages/qda-kl[eval]"\n'
    )


def test_evaluate_suite_refuses_in_two_lines_when_the_run_cards_refuse_its_jobs(
    tmp_path, monkeypatch, capsys
):
    """A benchmark whose jobs are refused on the run's cards exits with the child's status, and stderr names the
    cards with the job's error, then the run's `cards` setting that picks others."""
    from qda_kl.eval.suites import SuiteResult
    from qda_quant.cli import main as qda

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    run = _finished_run(tmp_path, kept=True)
    resolved = yaml.safe_load((run / "resolved.yaml").read_text())
    (run / "resolved.yaml").write_text(yaml.safe_dump({**resolved, "cards": [3]}))
    monkeypatch.setattr(rundir, "suite_ready", lambda _name: None)

    def fake(argv, **kw):
        argv = [str(a) for a in argv]
        if "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            Path(body["out"], "score.json").write_text(json.dumps({"kl": 0.02}))
            return subprocess.CompletedProcess(argv, 0)
        out = Path(yaml.safe_load(Path(argv[-1]).read_text())["out"])  # the child's failure record
        failed = SuiteResult(
            suite="sanity",
            checkpoint="candidate0",
            metrics={},
            higher_is_better={},
            error="prepare: card 3 is busy",
        )
        (out / "results" / "candidate0").mkdir(parents=True)
        (out / "results" / "candidate0" / "sanity.json").write_text(failed.model_dump_json())
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(subprocess, "run", fake)
    assert qda(["eval", str(run), "--suite", "sanity"]) == 3
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2
    assert err[0].startswith("error: card 3 (the run's cards)") and err[0].endswith("prepare: card 3 is busy")
    assert err[1].startswith("fix:   set `cards` in ") and str(run / "resolved.yaml") in err[1]


def test_evaluate_suite_on_a_fallback_run_keeps_candidate_0_alone(tmp_path, monkeypatch):
    run = _finished_run(tmp_path, kept=True)
    monkeypatch.setattr(rundir, "suite_ready", lambda _name: None)
    calls: list[list[str]] = []

    def fake(argv, **kw):
        calls.append([str(a) for a in argv])
        if "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            Path(body["out"], "score.json").write_text(json.dumps({"kl": 0.02}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    spec = "{lm_eval: {tasks: [gpqa_diamond]}}"
    assert cli.main(["evaluate", str(run), "--suite", spec, "--dry-run"]) == 0
    preset = yaml.safe_load((run / "evaluate" / "lm_eval" / "eval.yaml").read_text())
    assert preset["checkpoints"] == {"candidate0": str(tmp_path / "c0")}


def test_the_evaluate_suite_presets_are_four_and_full_names_the_one_table():
    """`--suite` names exactly kl, sanity, full and specdec. `full` is one `qda eval` entry, expanded per
    model by `qda_kl.eval.full_preset` (the one table both entry points read), never a copy of its values."""
    from qda_kl.eval import presets as eval_presets
    from qda_kl.eval.suites import parse_suite

    assert list(eval_presets.PRESETS) == ["kl", "sanity", "full", "specdec"]
    assert eval_presets.PRESETS["kl"] == (eval_presets.Group("kl", ({"kl": {}},)),)
    assert eval_presets.PRESETS["sanity"] == (eval_presets.Group("sanity", ({"sanity": {}},)),)
    assert eval_presets.PRESETS["specdec"] == (eval_presets.Group("specdec", ({"specdec": {}},)),)
    assert eval_presets.PRESETS["full"] == (eval_presets.Group("full", ({"full": {}},)),)
    for name, groups in eval_presets.PRESETS.items():
        for group in groups:
            for entry in group.suites:
                assert parse_suite(entry, name).name in entry  # every entry is what qda eval takes


def test_the_eval_example_configs_copied_the_table_are_gone():
    """`configs/eval/paper.yaml` and `mmlu-pro.yaml` were copies of the old preset's two groups; the table is
    the one owner, so both are removed, and so is the AutoRound W4A16 benchmark config."""
    root = Path(__file__).parents[3]
    for name in ("paper.yaml", "mmlu-pro.yaml", "autoround-w4a16.yaml"):
        assert not (root / "configs" / "eval" / name).exists()


def test_evaluate_full_keeps_the_tables_refusal_status_and_fix(tmp_path, monkeypatch, capsys):
    """The released table's refusal is printed as the contract's two lines, with its own status."""
    from qda_kl.eval import full_preset
    from qda_quant.exit import ENVIRONMENT
    from qda_quant.step import Refused

    run = _finished_run(tmp_path, anchor="Qwen/Qwen3.5-9B")
    monkeypatch.setattr(rundir, "suite_ready", lambda _name: None)

    def refused(*_args, **_kw):
        raise Refused(
            "the released row needs a card it cannot find", "run it on a node with 4 cards", code=ENVIRONMENT
        )

    monkeypatch.setattr(full_preset, "applied", refused)
    from qda_quant.cli import main as qda

    assert qda(["eval", str(run), "--suite", "full", "--dry-run"]) == ENVIRONMENT
    assert capsys.readouterr().err == (
        "error: the released row needs a card it cannot find\nfix:   run it on a node with 4 cards\n"
    )


def test_evaluate_suite_full_dry_run_expands_the_released_row_first(tmp_path, monkeypatch, capsys):
    """`--suite full` is one `qda eval` run (lm_eval with MMLU-Pro, plus LiveCodeBench), listed without
    scoring anything; its first line names the table row applied."""
    run = _finished_run(tmp_path, anchor="Qwen/Qwen3.8-27B")
    monkeypatch.setattr(rundir, "suite_ready", lambda _names: None)
    calls: list[list[str]] = []

    def fake(argv, **kw):
        calls.append([str(a) for a in argv])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(["evaluate", str(run), "--suite", "full", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "full: settings of Qwen/Qwen3.8-27B (the released cards')"
    config = yaml.safe_load((run / "evaluate" / "full" / "eval.yaml").read_text())
    assert config["reference"] == "Qwen/Qwen3.8-27B"
    lm = next(e["lm_eval"] for e in config["suites"] if "lm_eval" in e)
    assert lm["tasks"] == [
        "paper_aime25",
        "paper_hmmt_feb_2026",
        "paper_gpqa_diamond",
        "ifbench",
        "ifeval",
        "mmlu_pro",
    ]
    assert lm["max_tokens"]["paper_gpqa_diamond"] == 32768  # 27B's card caps GPQA lower
    assert lm["sampling"]["presence_penalty"] == 0.0  # 27B's sampling, not the Qwen3.5 one
    assert lm["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert lm["serve_args"]["gpu_memory_utilization"] == 0.9
    lcb = next(e["livecodebench"] for e in config["suites"] if "livecodebench" in e)
    assert lcb["protocol"] == "paper" and lcb["timeout_s"] == 8 and lcb["repeats"] == 4
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"


def test_evaluate_names_the_holdout_rows_the_reference_skipped(tmp_path):
    """A reference with rows past the model's context says so, whether it was built now or reused."""
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text(
        json.dumps(
            {
                "k": 128,
                "vocab": 10,
                "build": {
                    "prompts": "corpus.holdout.jsonl",
                    "dropped_over_context": ["klv2-long-long-00037"],
                    "prompt_rows": 201,
                    "context": 16384,
                },
            }
        )
    )
    assert rundir.reference_note(ref) == (
        "corpus.holdout.jsonl: 200 of 201 rows kept, 1 over the model's 16384-token context"
    )
    assert rundir.reference_note(tmp_path) is None


def test_evaluate_and_qda_eval_expand_the_same_table(tmp_path):
    """The one table: `evaluate`'s entries and `qda eval`'s expansion of `- full: {}` are the same settings,
    and `--set` wins field by field over both."""
    import qda_agent.extensions as extensions

    extensions.register()
    from qda_kl.eval.config import EvalConfig
    from qda_kl.eval.full_preset import applied

    reference = "Qwen/Qwen3.6-35B-A3B"
    code = tmp_path / "c0"
    code.mkdir()
    text = yaml.safe_dump(
        {
            "reference": reference,
            "checkpoints": {"candidate0": str(code)},
            "out": str(tmp_path / "out"),
            "suites": [{"full": {}}],
        }
    )
    got = applied(reference, [str(code)])
    cfg = EvalConfig.from_yaml(text)
    assert cfg.full_note == got.note
    lm = next(s for s in cfg.suites if s.name == "lm_eval").to_dict()["lm_eval"]
    assert lm["tasks"] == got.entries[0]["lm_eval"]["tasks"]
    assert lm["max_tokens"] == got.entries[0]["lm_eval"]["max_tokens"]
    assert lm["repeats"] == got.entries[0]["lm_eval"]["repeats"]
    assert lm["sampling"] == got.entries[0]["lm_eval"]["sampling"]
    assert lm["serve_args"]["gpu_memory_utilization"] == 0.9
    assert lm["serve_args"]["max_model_len"] == 114688
    lcb = next(s for s in cfg.suites if s.name == "livecodebench").to_dict()["livecodebench"]
    assert lcb["protocol"] == got.entries[1]["livecodebench"]["protocol"]
    assert lcb["max_tokens"] == got.entries[1]["livecodebench"]["max_tokens"]
    # a --set value wins over the row, for `qda eval` exactly as for evaluate's overrides
    overridden = EvalConfig.from_yaml(text, ["suites.0.full.sampling.temperature=0.7"])
    lm = next(s for s in overridden.suites if s.name == "lm_eval")
    assert lm.sampling["temperature"] == 0.7
    assert lm.sampling["presence_penalty"] == 1.5  # the rest of the Qwen3.5/3.6 row stays
    assert overridden.full_note.endswith("(your overrides applied)")


def test_qda_eval_prints_the_applied_row_as_its_first_line(tmp_path, capsys):
    """`qda eval` reads the same `- full: {}` entry: its own output names the row before any job is listed."""
    import argparse

    import qda_agent.extensions as extensions

    extensions.register()
    from qda_kl.eval import cli as eval_cli

    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "reference": "Qwen/Qwen3.5-9B",
                "checkpoints": {"candidate0": str(tmp_path / "c0")},
                "out": str(tmp_path / "out"),
                "suites": [{"full": {}}],
            }
        )
    )
    parser = argparse.ArgumentParser()
    eval_cli.add_parser(parser.add_subparsers())
    args = parser.parse_args(["eval", str(config), "--dry-run"])
    assert args.run(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "full: settings of Qwen/Qwen3.5-9B (the released cards')"
    assert sum('"suite": "lm_eval"' in line for line in out) >= 1
    assert sum('"suite": "livecodebench"' in line for line in out) == 1


def test_evaluate_refuses_a_build_directory(tmp_path, capsys):
    """A `qda build` output holds a resolved.yaml (a build config, no judge) but no FINAL.json."""
    (tmp_path / "resolved.yaml").write_text(yaml.safe_dump({"steps": [{"rtn": {}}]}))
    assert cli.main(["evaluate", str(tmp_path)]) == 2
    assert "no FINAL.json: not a finished search run" in capsys.readouterr().err


def test_a_search_config_an_earlier_prepare_wrote_says_so_and_names_the_prepare_that_rewrites_it(
    tmp_path, monkeypatch
):
    """No migration: a prepare-written config with keys this version dropped (a `mix` component, a
    `judge.keep_top`) is refused, saying an earlier prepare wrote it, and its fix is the one prepare command
    (same model, out, format, cards and budget) that writes it again. A hand-written config keeps `edit`."""
    from qda_agent import doctor
    from qda_agent.configfile import ConfigError
    from qda_quant.doctor import Context

    Commands(monkeypatch)
    p = prepared(tmp_path, cards=[0, 1])
    P.execute(p, P.plan(p, ("model", "corpus", "ref", "calib", "cand0", "config")), log=lambda _: None)
    old = p.config.read_text().replace("  - qda\n", "  - qda\n  - mix\n", 1)
    old = old.replace("\njudge:\n", "\njudge:\n  keep_top: 4\n", 1)
    p.config.write_text(old)
    with pytest.raises(ConfigError) as refused:
        RunConfig.load(p.config)
    assert "plugin 'mix'" in str(refused.value) and "judge.keep_top" in str(refused.value)
    assert str(refused.value).endswith(
        "(an earlier qda-agent prepare wrote it; this version does not read those keys)"
    )
    assert refused.value.fix == (
        f"qda-agent prepare --model {p.model_dir} --out {p.out} --format w4a16 --cards 0,1 --budget-gpu-hours 4"
    )
    (line,) = doctor.search(Context(search=p.config))
    assert line.missed and line.fix == refused.value.fix

    p.config.write_text(
        old.split("\n", 3)[3]
    )  # the same keys without prepare's header: a hand-written config
    with pytest.raises(ConfigError) as refused:
        RunConfig.load(p.config)
    assert refused.value.fix is None and "earlier qda-agent prepare" not in str(refused.value)
