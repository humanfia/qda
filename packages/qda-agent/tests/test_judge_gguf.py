"""A llama.cpp candidate: one `.gguf`, the pinned server loads it and answers, and receipts cover its bytes.

The server here is a fake (CPU, no engine): it records its argv and environment, answers `/health` and
`/completion`, and can fail before it binds. The strict inventory check is the evaluator's
(`qda_kl.llamacpp.strict_check`); these tests stand a raising one in for it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from qda_agent.judge import build_receipt
from qda_agent.judge.checks.gguf import (
    _ggml_name,
    gguf_fingerprint,
    gguf_near_copy,
    gguf_problem,
    gguf_sizes,
    the_gguf,
)
from qda_agent.judge.config import JudgeConfig, JudgeSettings
from qda_agent.judge.freeze import digest
from qda_agent.sched import Job, Refused, State
from qda_quant.gguf.file import read_table
from qda_quant.gguf.rule import BF16, F16, F32, IQ4_XS, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0
from test_judge import Rig, ckpt, llama_paths, write_gguf

TOOLS = Path(__file__).parents[1] / "src" / "qda_agent" / "judge" / "tools"

FAKE_SERVER = """#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
record = os.environ.get("QDA_FAKE_RECORD")
if record:
    with open(record, "w") as f:
        json.dump({{"argv": argv, "env": dict(os.environ)}}, f)
if os.environ.get("QDA_FAKE_FAIL"):
    print("llama_model_load: error loading model: bad magic", flush=True)
    sys.exit(1)
print("QDA_TENSOR\\tblk.0.attn_q.weight\\tQ4_K\\t1\\t1\\t1\\t1", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{{"status": "ok"}}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if record:
            with open(record + ".body", "w") as f:
                f.write(body.decode())
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({{"tokens_predicted": 16, "truncated": False}}).encode())


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def gguf_dir(d: Path, data: bytes | None = None) -> Path:
    """A candidate directory holding one top-level `.gguf`: one Q4_K tensor, or `data` as given."""
    d.mkdir(parents=True, exist_ok=True)
    if data is not None:
        (d / "model.gguf").write_bytes(data)
    else:
        write_gguf(d / "model.gguf", [("blk.0.attn_q.weight", (256,), Q4_K, b"\0" * 144)])
    return d


def the_tool():
    """The shipped admission tool, loaded from the source tree."""
    spec = importlib.util.spec_from_file_location("judge_gguf_check", TOOLS / "gguf_check.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_server(tmp_path: Path) -> Path:
    p = tmp_path / "llama-server"
    p.write_text(FAKE_SERVER.format(python=sys.executable))
    p.chmod(0o755)
    return p


def a_shared_object() -> Path:
    """Some real shared library to name as the probe: it must load, whatever it observes."""
    for name in ("/lib64/libm.so.6", "/usr/lib64/libm.so.6", "/lib/x86_64-linux-gnu/libm.so.6"):
        if Path(name).is_file():
            return Path(name)
    pytest.skip("no shared library on this host to name as the probe")


# ---------------------------------------------------------------- the server's side of the admission


def test_the_tool_requests_the_measured_server_settings(tmp_path, monkeypatch):
    server = fake_server(tmp_path)
    record = tmp_path / "record.json"
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(server))
    monkeypatch.setenv("QDA_FAKE_RECORD", str(record))
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    monkeypatch.delenv("QDA_LLAMA_BRIDGE", raising=False)
    mod = the_tool()
    ckpt_dir = gguf_dir(tmp_path / "cand")
    ok, reason = mod.check(ckpt_dir, tmp_path / "server.log", 60)
    assert ok and reason == ""
    got = json.loads(record.read_text())
    assert got["argv"][:2] == ["-m", str(ckpt_dir / "model.gguf")]
    for flag, value in (("-c", "2048"), ("-np", "1"), ("-b", "128"), ("-ub", "128"), ("-fa", "on")):
        assert got["argv"][got["argv"].index(flag) + 1] == value, got["argv"]
    assert "--check-tensors" in got["argv"] and "--log-verbosity" in got["argv"]
    assert got["env"]["OMP_NUM_THREADS"] == "8"
    assert json.loads((record.parent / "record.json.body").read_text()) == mod.BODY
    assert mod.BODY == {
        "prompt": "Say hello.",
        "n_predict": 16,
        "temperature": 0,
        "seed": 17,
        "cache_prompt": False,
        "return_tokens": True,
    }


def test_a_gguf_the_server_cannot_load_is_refused(tmp_path, monkeypatch):
    server = fake_server(tmp_path)
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(server))
    monkeypatch.setenv("QDA_FAKE_FAIL", "1")
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    mod = the_tool()
    ok, reason = mod.check(gguf_dir(tmp_path / "cand"), tmp_path / "server.log", 60)
    assert not ok and "exited 1" in reason
    assert "bad magic" in (tmp_path / "server.log").read_text()


def test_a_strict_check_that_does_not_pass_is_refused(tmp_path, monkeypatch):
    error = ValueError  # any exception from the loader is the same refusal class
    server = fake_server(tmp_path)
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(server))
    monkeypatch.setenv("QDA_LLAMA_CPP", str(tmp_path / "llama.cpp"))
    monkeypatch.setenv("QDA_LLAMA_BRIDGE", str(a_shared_object()))
    mod = the_tool()

    def refuses(*_a):
        raise error("tensor blk.3.ffn_down.weight loaded as Q6_K, the file stores Q4_K")

    monkeypatch.setattr(mod, "strict", refuses)
    ok, reason = mod.check(gguf_dir(tmp_path / "cand"), tmp_path / "server.log", 60)
    assert not ok and "strict load check refused it" in reason
    assert error.__name__ in reason and "Q6_K" in reason


def test_a_strict_check_without_the_probe_is_refused(tmp_path, monkeypatch):
    server = fake_server(tmp_path)
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(server))
    monkeypatch.setenv("QDA_LLAMA_CPP", str(tmp_path / "llama.cpp"))
    monkeypatch.delenv("QDA_LLAMA_BRIDGE", raising=False)
    mod = the_tool()
    ok, reason = mod.check(gguf_dir(tmp_path / "cand"), tmp_path / "server.log", 60)
    assert not ok and "probe.so" in reason


def test_the_probe_is_preloaded_and_the_threads_pinned(tmp_path, monkeypatch):
    probe, record = a_shared_object(), tmp_path / "record.json"
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(fake_server(tmp_path)))
    monkeypatch.setenv("QDA_FAKE_RECORD", str(record))
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    monkeypatch.setenv("QDA_LLAMA_BRIDGE", str(probe))
    mod = the_tool()
    assert mod.check(gguf_dir(tmp_path / "cand"), tmp_path / "server.log", 60)[0]
    env = json.loads(record.read_text())["env"]
    assert env["LD_PRELOAD"] == str(probe) and env["OMP_NUM_THREADS"] == "8"


@pytest.mark.parametrize("given", ["file", "directory"])
def test_the_tool_serves_the_frozen_gguf_file_or_its_directory(tmp_path, monkeypatch, given):
    """The judge hands every llama.cpp job the frozen `.gguf` FILE; a hand-run names its directory."""
    record = tmp_path / "record.json"
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(fake_server(tmp_path)))
    monkeypatch.setenv("QDA_FAKE_RECORD", str(record))
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    monkeypatch.delenv("QDA_LLAMA_BRIDGE", raising=False)
    mod = the_tool()
    d = gguf_dir(tmp_path / "cand")
    ok, reason = mod.check(d / "model.gguf" if given == "file" else d, tmp_path / "server.log", 60)
    assert ok and reason == "", reason
    assert json.loads(record.read_text())["argv"][:2] == ["-m", str(d / "model.gguf")]


@pytest.mark.parametrize("bad", ["link", "none", "two", "not-gguf", "missing"])
def test_the_tool_refuses_what_is_not_one_gguf_before_starting_a_server(tmp_path, monkeypatch, bad):
    record = tmp_path / "record.json"
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(fake_server(tmp_path)))
    monkeypatch.setenv("QDA_FAKE_RECORD", str(record))
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    mod = the_tool()
    d = gguf_dir(tmp_path / "cand")
    if bad == "link":
        (tmp_path / "alias.gguf").symlink_to(d / "model.gguf")
        ckpt = tmp_path / "alias.gguf"
    elif bad == "none":
        (d / "model.gguf").unlink()
        ckpt = d
    elif bad == "two":
        (d / "other.gguf").write_bytes((d / "model.gguf").read_bytes())
        ckpt = d
    elif bad == "not-gguf":
        ckpt = d / "config.json"
        ckpt.write_text("{}")
    else:
        ckpt = d / "absent.gguf"
    ok, reason = mod.check(ckpt, tmp_path / "server.log", 60)
    assert not ok and reason == "no single top-level .gguf"
    assert not record.exists()  # no server was started on a guess


# ---------------------------------------------------------------- the judge's side of the admission


def llama_rig(tmp_path, monkeypatch, *, budget=1 << 20, llamacpp_extra=None, **kw):
    """A Rig on the llama.cpp engine whose `check` job is a fake run the test gives an effect to."""
    pair = llama_paths(tmp_path, monkeypatch)
    return Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=budget,
        engine_floor=0,
        llamacpp={**pair, **(llamacpp_extra or {})},
        llama_cmd=["check", "{ckpt}", "{out}"],
        **kw,
    )


def test_a_handin_reaches_the_shipped_tool_as_the_frozen_file_and_is_admitted(tmp_path, monkeypatch):
    """The 11-13 failure: the judge passes the frozen `.gguf` file and the tool globbed it as a directory, so every
    hand-in was refused with 'no single top-level .gguf'. Here the job runs the shipped tool on the argv it got."""
    r = llama_rig(tmp_path, monkeypatch)
    record = tmp_path / "record.json"
    monkeypatch.setenv("QDA_LLAMA_SERVER", str(fake_server(tmp_path)))
    monkeypatch.setenv("QDA_FAKE_RECORD", str(record))
    monkeypatch.delenv("QDA_LLAMA_CPP", raising=False)
    monkeypatch.delenv("QDA_LLAMA_BRIDGE", raising=False)
    tool = the_tool()
    seen = []

    def run_the_tool(argv):
        seen.append(argv[1])
        ok, reason = tool.check(Path(argv[1]), tmp_path / "s.log", 60)
        Path(argv[-1]).write_text(json.dumps({"ok": ok, "reason": reason}))

    r.be.default.effect = run_the_tool
    gguf_dir(r.work / "a")
    assert r.call("handin", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert Path(seen[0]).is_file() and Path(seen[0]) == e.dir / "model.gguf"  # the judge hands it the file
    assert e.valid is True, e.reason
    assert json.loads(record.read_text())["argv"][:2] == ["-m", str(e.dir / "model.gguf")]


def test_a_gguf_that_does_not_load_is_an_invalid_handin(tmp_path, monkeypatch):
    r = llama_rig(tmp_path, monkeypatch)
    r.be.default.effect = lambda argv: Path(argv[-1]).write_text(
        json.dumps({"ok": False, "reason": "llama-server exited 1 (see its log)"})
    )
    gguf_dir(r.work / "a")
    assert r.call("handin", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is False and "llama-server exited 1" in e.reason


def test_a_gguf_that_loads_is_admitted(tmp_path, monkeypatch):
    r = llama_rig(tmp_path, monkeypatch)
    r.be.default.effect = lambda argv: Path(argv[-1]).write_text(json.dumps({"ok": True, "reason": ""}))
    gguf_dir(r.work / "a")
    assert r.call("handin", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is True and e.eligible is True


def test_a_llama_cpp_baseline_gets_the_gguf_file_gate(tmp_path, monkeypatch):
    from qda_agent.judge.startup import select_baselines as select
    from qda_agent.sched import Refused

    base = gguf_dir(tmp_path / "base")
    (base / "second.gguf").write_bytes(b"GGUF fake")
    r = llama_rig(tmp_path, monkeypatch, budget=1, baselines=[base], select_candidate0=True)
    with pytest.raises(Refused, match=r"baselines\[0\].*2 \*\.gguf weights"):
        select(r.host)


def test_a_llama_cpp_baseline_keeps_the_rows_mtp_choice(tmp_path, monkeypatch):
    """The baseline that becomes candidate 0 is admitted as a candidate, MTP choice included."""
    from qda_agent.judge.startup import select_baselines as select

    base = gguf_dir(tmp_path / "base")
    write_gguf(base / "model.gguf", [("blk.0.nextn.eh_proj", (256,), Q4_K, b"\0" * 144)])
    r = llama_rig(
        tmp_path, monkeypatch, llamacpp_extra={"mtp": "forbidden"}, baselines=[base], select_candidate0=True
    )
    with pytest.raises(Refused, match=r"baselines\[0\].*mtp is forbidden"):
        select(r.host)


def test_the_vllm_path_still_checks_a_safetensors_handin(tmp_path):
    r = Rig(tmp_path)  # the default engine, and the default check command
    assert r.host.cfg.engine == "vllm"
    ckpt(r.work / "a")
    assert r.call("handin", "a").exit == 0
    r.settle()
    (e,) = r.host.entries
    assert e.valid is True and r.seen  # the vLLM check ran on the frozen copy


def test_a_gguf_is_not_a_vllm_candidate(tmp_path):
    r = Rig(tmp_path)
    gguf_dir(r.work / "g")
    rep = r.call("handin", "g")
    assert rep.exit == 2 and "weights are safetensors" in rep.stderr


def test_a_llama_cpp_run_needs_its_pinned_server(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    serverless = {k: v for k, v in pair.items() if k != "server"}
    with pytest.raises(ValueError, match=r"llamacpp\.server"):
        JudgeSettings(engine="llama.cpp", budget_bytes=1, engine_floor=0, llamacpp=serverless)
    cfg = JudgeConfig(
        contamination="none", kl_ref="refs", engine="llama.cpp", budget_bytes=1, engine_floor=0, llamacpp=pair
    )
    env = cfg.job_environ(None)
    assert env["QDA_LLAMA_SERVER"] == pair["server"]
    assert env["QDA_LLAMA_CPP"] == pair["llama_cpp"]
    assert env["QDA_LLAMA_BRIDGE"] == pair["bridge"]
    with pytest.raises(ValueError, match="judge.format_reference"):
        JudgeSettings(
            engine="llama.cpp",
            budget_bytes=1,
            engine_floor=0,
            llamacpp=pair,
            format_reference=tmp_path / "r",
        )


# ---------------------------------------------------------------- the candidate's files


def test_the_file_rules_are_one_top_level_gguf(tmp_path):
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "tokenizer.json").write_text('{"model": {}}')
    cand = gguf_dir(tmp_path / "cand")
    assert gguf_problem(cand) is None
    assert the_gguf(cand) == cand / "model.gguf"

    shutil.copy(anchor / "tokenizer.json", cand / "tokenizer.json")  # the anchor's own file
    assert gguf_problem(cand, anchor) is None
    (cand / "tokenizer.json").unlink()

    (cand / "extra.gguf").write_bytes(b"second")
    assert "one" in (gguf_problem(cand) or "")
    (cand / "extra.gguf").unlink()

    (cand / "nested").mkdir()
    (cand / "nested" / "other.gguf").write_bytes(b"not top level")
    assert ".gguf" in (gguf_problem(cand) or "")
    shutil.rmtree(cand / "nested")

    (cand / "tokenizer.json").write_text('{"agent": {}}')  # not the anchor's bytes
    assert "the anchor's own" in (gguf_problem(cand, anchor) or "")
    (cand / "tokenizer.json").unlink()

    (cand / ".hidden").write_text("x")
    assert "hidden" in (gguf_problem(cand) or "")
    (cand / ".hidden").unlink()

    gguf_dir(tmp_path / "empty", data=b"")
    (tmp_path / "empty" / "model.gguf").unlink()
    assert gguf_problem(tmp_path / "empty") == "no *.gguf weights"


# ---------------------------------------------------------------- receipts cover the .gguf bytes


def a_receipted_build(tmp_path, monkeypatch):
    """A receipted build whose output is one `.gguf`, as `qda build` would publish it."""
    original = tmp_path / "original"
    original.mkdir()
    r = Rig(
        tmp_path,
        model=str(original),
        engine="llama.cpp",
        budget_bytes=1,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
    )
    private = tmp_path / "custody"
    gguf_dir(private / "output")
    (private / "output" / "build.json").write_text("{}")
    config = tmp_path / "build.yaml"
    config.write_text("model: original\nout: candidate\n")
    spec = {
        "job_id": "1-b1",
        "destination": "candidate",
        "root": str(r.work),
        "config": str(config),
        "overrides": [],
        "source": str(original.resolve()),
    }
    build_receipt.capture(private, spec)
    build_receipt.publish(private)
    job = Job(
        id="1-b1",
        seq=1,
        name="b1",
        argv=["true"],
        cwd=str(tmp_path),
        expected_s=1,
        submitted=0.0,
        state=State.done,
        rc=0,
        build_private=str(private),
    )
    r.host.sched.jobs["b1"] = job
    return r, private / "published"


def test_a_hand_edited_gguf_matches_no_build(tmp_path, monkeypatch):
    r, published = a_receipted_build(tmp_path, monkeypatch)
    copy = tmp_path / "copy"
    shutil.copytree(published, copy)
    receipt, why = build_receipt.match(r.host, copy, digest(copy))
    assert receipt is not None and why == ""

    gguf = copy / "model.gguf"
    gguf.chmod(0o644)
    data = gguf.read_bytes()
    gguf.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    receipt, why = build_receipt.match(r.host, copy, digest(copy))
    assert receipt is None and "not eligible" in why


# ---------------------------------------------------------------- the stored types and the byte budget


def test_a_gguf_is_screened_by_its_whole_file_size(tmp_path):
    """The default basis is the file the scorer reads: its header and padding count, the payload does not."""
    cand = gguf_dir(tmp_path / "cand")
    size = gguf_sizes(cand)["file"]
    assert gguf_sizes(cand) == {"file": size, "payload": 144}
    assert size > 144 and gguf_problem(cand, budget_bytes=size) is None
    assert gguf_problem(cand, budget_bytes=size - 1) == f"file {size} B > budget {size - 1} B"
    # a payload-only budget is an explicit opt-in, and its refusal names that basis
    assert gguf_problem(cand, budget_bytes=144, budget_basis="payload") is None
    assert gguf_problem(cand, budget_bytes=143, budget_basis="payload") == "payload 144 B > budget 143 B"


def test_a_header_size_difference_counts_against_the_file_budget(tmp_path):
    """Two files hold the same 144 payload bytes; the one with the longer header is over the same budget."""
    small, large = tmp_path / "small", tmp_path / "large"
    small.mkdir()
    large.mkdir()
    write_gguf(small / "model.gguf", [("blk.0.wx", (256,), Q4_K, b"\0" * 144)])
    write_gguf(large / "model.gguf", [("blk.0.wxy", (256,), Q4_K, b"\0" * 144)])
    a, b = gguf_sizes(small), gguf_sizes(large)
    assert a["payload"] == b["payload"] == 144 and a["file"] < b["file"]
    assert gguf_problem(small, budget_bytes=a["file"]) is None
    assert gguf_problem(large, budget_bytes=a["file"]) == f"file {b['file']} B > budget {a['file']} B"
    assert gguf_problem(large, budget_bytes=144, budget_basis="payload") is None  # the payload is the same


def test_a_payload_basis_config_is_an_explicit_opt_in(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    settings = JudgeSettings(
        engine="llama.cpp",
        budget_bytes=1000,
        budget_basis="payload",
        engine_floor=0,
        llamacpp=pair,
    )
    assert settings.budget_basis == "payload"
    assert (
        JudgeSettings(engine="llama.cpp", budget_bytes=1000, engine_floor=0, llamacpp=pair).budget_basis
        == "file"
    )
    with pytest.raises(ValueError, match="judge.budget_basis needs judge.engine: llama.cpp"):
        JudgeSettings(kl_ref="refs", budget_basis="payload")


@pytest.mark.parametrize(
    ("carries", "mtp", "refused"),
    [
        (True, "required", False),
        (True, "forbidden", True),
        (True, "any", False),
        (False, "required", True),
        (False, "forbidden", False),
        (False, "any", False),
    ],
)
def test_the_mtp_choice_is_enforced_at_admission(tmp_path, carries, mtp, refused):
    """A row keeps one MTP choice: the candidate's stored tensor names decide it, per configured value."""
    cand = gguf_dir(tmp_path / "cand")  # one Q4_K tensor, no MTP head
    if carries:
        write_gguf(cand / "model.gguf", [("blk.0.nextn.eh_proj", (256,), Q4_K, b"\0" * 144)])
    why = gguf_problem(cand, menu=S5_MENU, mtp=mtp)
    if not refused:
        assert why is None
    elif carries:
        assert why == "1 MTP/NextN tensors (blk.0.nextn.eh_proj), and judge.llamacpp.mtp is forbidden"
    else:
        assert why == "no MTP/NextN tensors (nextn.*), and judge.llamacpp.mtp is required"


def test_mtp_tensors_are_the_pinned_llama_cpp_names(tmp_path):
    """The loader's own names decide, including the layer-less projections; a lookalike does not."""
    cand = gguf_dir(tmp_path / "cand")  # rewritten below with the pinned names
    write_gguf(
        cand / "model.gguf",
        [
            ("nextn.pre_projection", (256,), Q4_K, b"\0" * 144),
            ("nextn.post_projection", (256,), Q4_K, b"\0" * 144),
            ("blk.0.attn_nextn.weight", (256,), Q4_K, b"\0" * 144),  # not an MTP tensor
        ],
    )
    why = gguf_problem(cand, menu=S5_MENU, mtp="forbidden")
    names = "nextn.pre_projection, nextn.post_projection"
    assert why == f"2 MTP/NextN tensors ({names}), and judge.llamacpp.mtp is forbidden"


def test_the_mtp_choice_defaults_to_any_and_is_validated(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    common = dict(engine="llama.cpp", budget_bytes=1000, engine_floor=0, llamacpp=pair)
    assert JudgeSettings(**common).llamacpp.mtp == "any"
    assert JudgeSettings(**{**common, "llamacpp": {**pair, "mtp": "required"}}).llamacpp.mtp == "required"
    with pytest.raises(ValueError):
        JudgeSettings(**{**common, "llamacpp": {**pair, "mtp": "sometimes"}})


def test_a_handin_refuses_the_configured_mtp_choice(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=1 << 20,
        engine_floor=0,
        llamacpp={**pair, "mtp": "forbidden"},
        format_menu=["Q4_K"],
    )
    gguf_dir(r.work / "a")
    assert r.call("handin", "a").exit == 0  # no MTP head: the row's choice holds
    (r.work / "b").mkdir()
    write_gguf(r.work / "b" / "model.gguf", [("blk.0.nextn.eh_proj", (256,), Q4_K, b"\0" * 144)])
    rep = r.call("handin", "b")
    assert rep.exit == 2 and "judge.llamacpp.mtp is forbidden" in rep.stderr, rep.stderr


#: One 256-value row in each type: its stored bytes.
ROW = {Q4_K: 144, Q6_K: 210, Q8_0: 8 * 34, F32: 1024, F16: 512, BF16: 512}


def gguf_of(d: Path, tensors: dict[str, int], rows: int = 2) -> Path:
    """A candidate directory whose `.gguf` stores each named tensor (2-D unless it is a norm) as its type."""
    d.mkdir(parents=True, exist_ok=True)
    table = []
    for name, kind in tensors.items():
        n = 1 if "norm" in name else rows
        table.append((name, (256,) if n == 1 else (256, n), kind, b"\0" * ROW[kind] * n))
    write_gguf(d / "model.gguf", table)
    return d


@pytest.mark.parametrize("kind", [BF16, F16, Q8_0, F32], ids=["BF16", "F16", "Q8_0", "F32"])
def test_a_weight_stored_off_the_menu_is_refused_whatever_its_type(tmp_path, kind):
    """No type skips the menu: the 35B BF16 matrices passed because floats were exempt (closed here)."""
    name = {BF16: "BF16", F16: "F16", Q8_0: "Q8_0", F32: "F32"}[kind]
    cand0 = gguf_of(tmp_path / "cand0", {"blk.0.attn_norm.weight": F32, "blk.0.ffn_down.weight": Q4_K})
    cand = gguf_of(tmp_path / "cand", {"blk.0.attn_norm.weight": F32, "blk.0.ffn_down.weight": kind})
    why = gguf_problem(cand, menu=["Q4_K", "Q6_K"], reference=cand0)
    assert why == f"blk.0.ffn_down.weight is stored as {name}, outside judge.format_menu (Q4_K, Q6_K)"
    assert gguf_problem(cand, menu=["Q4_K", "Q6_K", name], reference=cand0) is None  # the menu names it
    if kind != F32:  # with no candidate 0 to read the template's floats from, F32 alone is exempt, as before
        assert gguf_problem(cand, menu=["Q4_K"]) == (
            f"blk.0.ffn_down.weight is stored as {name}, outside judge.format_menu (Q4_K)"
        )


@pytest.mark.parametrize("kind", [BF16, F16], ids=["BF16", "F16"])
def test_a_16_bit_output_head_is_admitted_off_the_menu(tmp_path, kind):
    """The default head rule keeps output.weight 16-bit; no menu entry has to name the type."""
    name = {BF16: "BF16", F16: "F16"}[kind]
    cand0 = gguf_of(tmp_path / "cand0", {"output.weight": Q6_K, "blk.0.attn_norm.weight": F32})
    cand = gguf_of(tmp_path / "cand", {"output.weight": kind, "blk.0.attn_norm.weight": F32})
    assert gguf_problem(cand, menu=["Q4_K", "Q6_K"], reference=cand0) is None
    assert gguf_problem(cand, menu=["Q4_K"]) is None  # no candidate 0's head to compare with either
    assert gguf_problem(cand, menu=["Q4_K"], head="16-bit") is None
    assert name in ("BF16", "F16")


def test_a_menu_typed_output_head_stays_admitted_under_the_16_bit_default(tmp_path):
    """Candidate 0 built by prepare stores the head Q6_K: the default rule does not refuse a menu type."""
    cand0 = gguf_of(tmp_path / "cand0", {"output.weight": Q6_K, "blk.0.attn_norm.weight": F32})
    assert gguf_problem(cand0, menu=["Q4_K", "Q6_K"], reference=cand0) is None
    # candidate 0's own off-menu head is kept as it stores it, the rule every other off-menu tensor has
    assert gguf_problem(cand0, menu=["Q4_K"], reference=cand0) is None


def test_an_exception_models_head_must_keep_its_release_type(tmp_path, monkeypatch):
    """An exception model's release stores the head Q8_0: its builds keep that type (`judge.head`)."""
    cand0 = gguf_of(tmp_path / "cand0", {"output.weight": Q8_0, "blk.0.attn_norm.weight": F32})
    menu = ["Q4_K", "Q8_0"]
    good = gguf_of(tmp_path / "good", {"output.weight": Q8_0, "blk.0.attn_norm.weight": F32})
    bad = gguf_of(tmp_path / "bad", {"output.weight": BF16, "blk.0.attn_norm.weight": F32})
    assert gguf_problem(good, menu=menu, reference=cand0, head="Q8_0") is None
    assert gguf_problem(bad, menu=menu, reference=cand0, head="Q8_0") == (
        "output.weight is stored as BF16; this model's release stores the output head as Q8_0 (judge.head): "
        "keep that type"
    )
    pair = llama_paths(tmp_path, monkeypatch)
    common = dict(engine="llama.cpp", budget_bytes=1000, engine_floor=0, llamacpp=pair)
    assert JudgeSettings(**{**common, "format_menu": menu, "head": "q8_0"}).head == "Q8_0"
    with pytest.raises(ValueError, match="judge.head: Q8_0 is not in judge.format_menu"):
        JudgeSettings(**{**common, "format_menu": ["Q4_K"], "head": "Q8_0"})
    with pytest.raises(ValueError, match="judge.head"):
        JudgeSettings(**{**common, "head": "Q4_K_M"})


def test_a_tied_head_has_no_output_weight_to_check(tmp_path):
    """A tied model stores no output.weight: the head rule checks nothing, as before."""
    cand = gguf_of(tmp_path / "cand", {"token_embd.weight": Q4_K, "blk.0.attn_norm.weight": F32})
    assert gguf_problem(cand, menu=["Q4_K"], head="Q8_0") is None


def test_candidate_0s_own_off_menu_types_are_admitted_on_its_tensors_and_kept(tmp_path):
    """The template's F32 norms, a 2-D F32 and a Q8_0 of candidate 0: admitted as candidate 0 stores them."""
    kept = {
        "blk.0.attn_norm.weight": F32,
        "blk.0.ssm_conv1d.weight": F32,
        "blk.0.ffn_gate_inp.weight": Q8_0,
        "blk.0.ffn_down.weight": Q4_K,
    }
    menu = ["Q4_K", "Q6_K"]
    cand0 = gguf_of(tmp_path / "cand0", kept)
    assert gguf_problem(cand0, menu=menu, reference=cand0) is None  # candidate 0 passes as-is
    assert (
        gguf_problem(
            gguf_of(tmp_path / "same", {**kept, "blk.0.ffn_down.weight": Q6_K}), menu=menu, reference=cand0
        )
        is None
    )  # a menu type on any other tensor
    for i, (name, kind, was) in enumerate(
        [
            ("blk.0.attn_norm.weight", BF16, "F32"),  # a norm cannot move to another float
            ("blk.0.ssm_conv1d.weight", Q4_K, "F32"),  # nor to a menu type: candidate 0's type is kept
            ("blk.0.ffn_gate_inp.weight", F32, "Q8_0"),
        ]
    ):
        cand = gguf_of(tmp_path / f"c{i}", {**kept, name: kind})
        assert gguf_problem(cand, menu=menu, reference=cand0) == (
            f"{name} is stored as {_ggml_name(kind)}, but candidate 0 stores it as {was} "
            "(a type off judge.format_menu): keep that type"
        )
    cand = gguf_of(tmp_path / "f32", {**kept, "blk.0.ffn_down.weight": F32})  # F32 where candidate 0 packs
    assert gguf_problem(cand, menu=menu, reference=cand0) == (
        "blk.0.ffn_down.weight is stored as F32, outside judge.format_menu (Q4_K, Q6_K)"
    )


def test_a_locked_mtp_block_keeps_candidate_0s_types(tmp_path):
    lock = [r"^blk\.64\."]
    menu = ["Q4_K", "Q6_K", "Q8_0"]
    kept = {"blk.63.ffn_down.weight": Q4_K, "blk.64.nextn.eh_proj.weight": Q8_0, "blk.64.attn_q.weight": Q6_K}
    cand0 = gguf_of(tmp_path / "cand0", kept)
    assert gguf_problem(cand0, menu=menu, reference=cand0, lock=lock) is None
    moved = gguf_of(tmp_path / "moved", {**kept, "blk.64.attn_q.weight": Q4_K})  # on the menu, but locked
    assert gguf_problem(moved, menu=menu, reference=cand0, lock=lock) == (
        "blk.64.attn_q.weight is stored as Q4_K, but candidate 0 stores it as Q6_K (judge.llamacpp.lock): "
        "keep that type"
    )
    assert gguf_problem(moved, menu=menu, reference=cand0) is None  # no lock: a menu type is free
    assert gguf_problem(moved, menu=None, reference=cand0, lock=lock) is not None  # any_format keeps the lock
    dropped = gguf_of(tmp_path / "dropped", {k: v for k, v in kept.items() if k != "blk.64.attn_q.weight"})
    assert gguf_problem(dropped, menu=menu, reference=cand0, lock=lock) == (
        "candidate 0's blk.64.attn_q.weight is locked (judge.llamacpp.lock), and this file does not store it"
    )
    other = gguf_of(tmp_path / "cand0b", {"blk.63.ffn_down.weight": Q4_K})
    assert "candidate 0 has no blk.64" in (gguf_problem(cand0, menu=menu, reference=other, lock=lock) or "")
    with pytest.raises(ValueError, match="needs candidate 0"):
        gguf_problem(cand0, menu=menu, lock=lock)


def test_a_bf16_matrix_is_admitted_by_a_menu_that_names_bf16_and_refused_by_one_that_does_not(
    tmp_path, monkeypatch
):
    """The decided GGUF menu, end to end: the 35B winner's BF16 matrices were admitted with BF16 off it."""
    pair = llama_paths(tmp_path, monkeypatch)
    kept = {"blk.0.attn_norm.weight": F32, "blk.0.ffn_down.weight": Q4_K}
    cand0 = gguf_of(tmp_path / "cand0", kept)
    decided = ["Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL", "Q3_K", "Q8_0"]
    for i, menu in enumerate([decided, [*decided, "BF16"]]):
        r = Rig(
            tmp_path / f"r{i}",
            check=False,
            baselines=(str(cand0),),
            engine="llama.cpp",
            budget_bytes=1 << 20,
            engine_floor=0,
            llamacpp=pair,
            format_menu=menu,
        )
        gguf_of(r.work / "a", {**kept, "blk.0.ffn_down.weight": BF16})
        rep = r.call("handin", "a")
        if "BF16" in menu:
            assert rep.exit == 0, rep.stderr
        else:
            assert rep.exit == 2 and "blk.0.ffn_down.weight is stored as BF16, outside judge.format_menu" in (
                rep.stderr
            ), rep.stderr


def test_a_handin_is_gated_by_candidate_0s_types_and_the_lock(tmp_path, monkeypatch):
    """The hand-in path: candidate 0 is `baselines[0]`; the lock is `judge.llamacpp.lock`."""
    pair = llama_paths(tmp_path, monkeypatch)
    kept = {"blk.0.attn_norm.weight": F32, "blk.0.ffn_down.weight": Q4_K, "blk.64.attn_q.weight": Q6_K}
    cand0 = gguf_of(tmp_path / "cand0", kept)
    r = Rig(
        tmp_path,
        check=False,
        baselines=(str(cand0),),
        engine="llama.cpp",
        budget_bytes=1 << 20,
        engine_floor=0,
        llamacpp={**pair, "lock": [r"^blk\.64\."]},
        format_menu=["Q4_K", "Q6_K"],
    )
    gguf_of(r.work / "a", {**kept, "blk.0.ffn_down.weight": Q6_K})
    assert r.call("handin", "a").exit == 0, r.call("handin", "a").stderr
    gguf_of(r.work / "b", {**kept, "blk.0.ffn_down.weight": BF16})
    rep = r.call("handin", "b")
    assert (
        rep.exit == 2 and "blk.0.ffn_down.weight is stored as BF16, outside judge.format_menu" in rep.stderr
    )
    gguf_of(r.work / "c", {**kept, "blk.64.attn_q.weight": Q4_K})
    rep = r.call("handin", "c")
    assert rep.exit == 2 and "candidate 0 stores it as Q6_K (judge.llamacpp.lock)" in rep.stderr, rep.stderr


def test_the_lock_is_a_list_of_regexes_and_needs_candidate_0(tmp_path, monkeypatch):
    pair = llama_paths(tmp_path, monkeypatch)
    common = dict(engine="llama.cpp", budget_bytes=1000, engine_floor=0, llamacpp=pair)
    assert JudgeSettings(**common).llamacpp.lock == []
    with pytest.raises(ValueError, match=r"judge.llamacpp.lock: '\[' is not a regex"):
        JudgeSettings(**{**common, "llamacpp": {**pair, "lock": ["["]}})
    with pytest.raises(ValueError, match="judge.llamacpp.lock needs candidate 0"):
        Rig(tmp_path, check=False, **{**common, "llamacpp": {**pair, "lock": [r"^blk\.64\."]}})


def test_a_handin_records_the_file_and_the_payload_figures(tmp_path, monkeypatch):
    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=1 << 20,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        format_menu=["Q4_K"],
    )
    gguf_dir(r.work / "a")
    assert r.call("handin", "a").exit == 0
    (entry,) = r.host.entries
    assert entry.sizes == {"file": entry.dir.joinpath("model.gguf").stat().st_size, "payload": 144}
    (record,) = r.host.handins()
    assert record.sizes == entry.sizes
    assert json.loads(record.model_dump_json())["sizes"] == entry.sizes


def test_a_gguf_is_admitted_by_its_stored_types_and_payload_bytes(tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    write_gguf(
        cand / "model.gguf",
        [
            ("blk.0.attn_q.weight", (256,), Q4_K, b"\0" * 144),
            ("blk.0.attn_k.weight", (256,), Q6_K, b"\0" * 210),
            ("blk.0.attn_norm.weight", (256,), F32, b"\0" * 1024),  # a float tensor is not a quantized weight
        ],
    )
    menu = ["Q4_K", "Q6_K"]
    assert gguf_problem(cand, menu=menu, budget_bytes=1378, budget_basis="payload") is None  # 144+210+1024
    why = gguf_problem(cand, menu=["Q4_K"])
    assert why is not None and "blk.0.attn_k.weight is stored as Q6_K, outside judge.format_menu" in why
    why = gguf_problem(cand, menu=menu, budget_bytes=1377, budget_basis="payload")
    assert why is not None and why == "payload 1378 B > budget 1377 B"
    assert (
        gguf_problem(cand, menu=["Q4_K"], budget_bytes=1378, budget_basis="payload", any_format=True) is None
    )  # the budget still holds
    (cand / "model.gguf").write_bytes(b"GGUF\x03")  # a header that ends before its tensor table
    why = gguf_problem(cand, menu=menu)
    assert why is not None and why.startswith("not a readable GGUF")


def test_a_llama_cpp_handin_states_its_menu_and_file_budget(tmp_path, monkeypatch):
    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=300,  # the same tensors cost a whole-file 240 B, or 416 B with one more
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        format_menu=["Q4_K"],
    )
    gguf_dir(r.work / "a")
    assert (r.work / "a" / "model.gguf").stat().st_size < 300
    assert r.call("handin", "a").exit == 0
    wrong_type = r.work / "b"
    wrong_type.mkdir()
    write_gguf(wrong_type / "model.gguf", [("blk.0.w", (256,), Q6_K, b"\0" * 210)])
    rep = r.call("handin", "b")
    assert rep.exit == 2 and "stored as Q6_K, outside judge.format_menu" in rep.stderr, rep.stderr
    over = r.work / "c"
    over.mkdir()
    gguf = over / "model.gguf"
    write_gguf(
        gguf,
        [("blk.0.w", (256,), Q4_K, b"\0" * 144), ("blk.1.w", (256,), Q4_K, b"\0" * 144)],
    )
    size = gguf.stat().st_size
    assert size > 300
    rep = r.call("handin", "c")
    assert rep.exit == 2 and f"file {size} B > budget 300 B" in rep.stderr, rep.stderr
    assert [e.src for e in r.host.entries] == ["a"]  # a refused candidate is not kept


def test_a_gguf_fingerprint_is_its_stored_tensors_not_its_file_bytes(tmp_path):
    one = gguf_dir(tmp_path / "one")
    same = gguf_dir(tmp_path / "same")
    changed = tmp_path / "changed"
    changed.mkdir()
    write_gguf(changed / "model.gguf", [("blk.0.attn_q.weight", (256,), Q4_K, b"\2" * 144)])
    assert gguf_fingerprint(one) == gguf_fingerprint(same)
    assert gguf_fingerprint(one) != gguf_fingerprint(changed)
    assert gguf_fingerprint(one) != hashlib.sha256(b"").hexdigest()  # never the empty hash
    assert gguf_fingerprint(tmp_path) is None  # no single .gguf: nothing to fingerprint


def test_a_gguf_near_copy_compares_packed_codes_and_scales(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    write_gguf(base / "model.gguf", [("blk.0.w", (256,), Q4_K, b"\0" * 16 + b"\1" * 128)])
    assert gguf_near_copy(base, base) == "a baseline checkpoint as-is, not your result"
    scales = tmp_path / "scales"
    scales.mkdir()
    write_gguf(scales / "model.gguf", [("blk.0.w", (256,), Q4_K, b"\2" * 16 + b"\1" * 128)])
    why = gguf_near_copy(scales, base)
    assert why == "a baseline's packed codes with its scales changed, not your result"
    codes = tmp_path / "codes"
    codes.mkdir()
    write_gguf(codes / "model.gguf", [("blk.0.w", (256,), Q4_K, b"\0" * 16 + b"\3" * 128)])
    assert gguf_near_copy(codes, base) is None  # a different candidate, not compared
    other = tmp_path / "other"
    other.mkdir()
    write_gguf(other / "model.gguf", [("blk.1.w", (256,), Q4_K, b"\0" * 16 + b"\1" * 128)])
    assert gguf_near_copy(other, base) is None  # a different tensor set


#: Blocks of one 5120 x 28672 weight: the shape of a real 27B linear, in 256-value blocks.
REAL_BLOCKS = 5120 * 28672 // 256
#: The S5 menu's quantized types, as a task states them.
S5_MENU = ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS")
#: (ggml type, values of one block, bytes of one block) of every type the menu admits.
MENU_TYPES = [(Q4_K, 256, 144), (Q5_K, 256, 176), (Q6_K, 256, 210), (Q8_0, 32, 34), (IQ4_XS, 256, 136)]


def test_the_admitted_types_take_the_pinned_gguf_py_names_and_block_layouts():
    """Names and block layouts are the pinned gguf-py's own, not hand-typed: a stale table fails here."""
    gguf = pytest.importorskip("gguf")
    from qda_agent.judge.checks import gguf as judge_gguf
    from qda_quant.gguf.file import row_bytes

    for kind, values, size in MENU_TYPES:
        upstream = gguf.GGMLQuantizationType(kind)
        assert gguf.GGML_QUANT_SIZES[upstream] == (values, size)
        assert judge_gguf._ggml_name(kind) == upstream.name
        assert row_bytes(kind, values) == size  # qda_quant sizes the same block the same way
        assert judge_gguf.GGML_BLOCKS[kind][0] == size


@pytest.mark.parametrize(("kind", "values", "block"), MENU_TYPES)
def test_a_menu_type_is_admitted_by_its_name_and_its_bytes(tmp_path, kind, values, block):
    """Q5_K and IQ4_XS were named by their numbers, so an exact menu refused them; now they pass the gate."""
    cand = tmp_path / "cand"
    cand.mkdir()
    write_gguf(cand / "model.gguf", [("blk.0.w", (values,), kind, b"\2" * block)])
    assert gguf_problem(cand, menu=S5_MENU, budget_bytes=block, budget_basis="payload") is None
    why = gguf_problem(cand, menu=S5_MENU, budget_bytes=block - 1, budget_basis="payload")
    assert why is not None and why == f"payload {block} B > budget {block - 1} B"
    if kind != Q4_K:
        assert "outside judge.format_menu" in (gguf_problem(cand, menu=["Q4_K"], budget_bytes=block) or "")


@pytest.mark.parametrize(("kind", "values", "block"), [(Q5_K, 256, 176), (Q8_0, 32, 34), (IQ4_XS, 256, 136)])
def test_a_real_tensor_of_a_new_type_is_hashed_across_read_spans(tmp_path, kind, values, block):
    """A 27B linear's blocks as each new type: larger than one 64 MiB read span, for all three."""
    blocks = REAL_BLOCKS * 256 // values
    size = block * blocks
    assert size > 64 << 20
    pattern = bytes((index * 37 + 11) % 256 for index in range(block))
    base = tmp_path / "base"
    base.mkdir()
    write_gguf(base / "model.gguf", [("blk.0.w", (values * blocks,), kind, pattern * blocks)])
    changed = tmp_path / "changed"
    changed.mkdir()
    shutil.copyfile(base / "model.gguf", changed / "model.gguf")
    at = read_table(changed / "model.gguf").data_start + size - 1
    with (changed / "model.gguf").open("r+b") as fh:  # the last stored byte, past every earlier read span
        fh.seek(at)
        fh.write(bytes([pattern[-1] ^ 1]))
    assert gguf_fingerprint(changed) != gguf_fingerprint(base)


@pytest.mark.parametrize(
    ("kind", "values", "block", "params"), [(Q5_K, 256, 176, 0), (Q8_0, 32, 34, 0), (IQ4_XS, 256, 136, 0)]
)
def test_a_new_types_scale_byte_changed_copy_is_refused(tmp_path, monkeypatch, kind, values, block, params):
    """One scale byte per block, in a span that is not a whole number of blocks, is still a re-scale."""
    monkeypatch.setattr(
        "qda_agent.judge.checks.gguf.READ_SPAN", 1024
    )  # 1024 % 176, % 34 and % 136 are all nonzero
    base = tmp_path / "base"
    base.mkdir()
    write_gguf(base / "model.gguf", [("blk.0.w", (values, 20), kind, bytes(block * 20))])
    cand = tmp_path / "cand"
    cand.mkdir()
    shutil.copyfile(base / "model.gguf", cand / "model.gguf")
    assert gguf_near_copy(cand, base) == "a baseline checkpoint as-is, not your result"
    at = read_table(cand / "model.gguf").data_start + block * 17 + params
    with (cand / "model.gguf").open("r+b") as fh:  # one scale byte, in a block after the first read span
        fh.seek(at)
        fh.write(b"\1")
    assert gguf_near_copy(cand, base) == "a baseline's packed codes with its scales changed, not your result"


def test_a_stored_type_the_judge_cannot_name_is_refused(tmp_path):
    """Fail closed: a ggml type outside the menu is refused by its number, never admitted as an unknown."""
    cand = tmp_path / "cand"
    cand.mkdir()
    write_gguf(cand / "model.gguf", [("blk.0.w", (256,), 10, b"\0" * 84)])  # Q2_K
    why = gguf_problem(cand, menu=S5_MENU)
    assert why is not None and "blk.0.w is stored as 10, outside judge.format_menu" in why


def test_a_quantized_type_with_no_block_layout_refuses_the_candidate(tmp_path):
    """A sized but unlisted quantized type cannot have its codes compared: refused, not skipped (fail closed)."""
    from qda_agent.judge.checks import CheckpointError

    base = tmp_path / "base"
    base.mkdir()
    write_gguf(base / "model.gguf", [("blk.0.w", (32,), Q5_0, b"\0" * 22)])
    cand = tmp_path / "cand"
    cand.mkdir()
    shutil.copyfile(base / "model.gguf", cand / "model.gguf")
    with pytest.raises(CheckpointError, match="stored as Q5_0"):
        gguf_near_copy(cand, base)


@pytest.mark.parametrize(("kind", "block", "params"), [(Q4_K, 144, 0), (Q6_K, 210, 192)])
def test_a_real_packed_tensor_is_compared_across_the_full_read_span(tmp_path, kind, block, params):
    """5120 x 28672 is 82,575,360 B as Q4_K and 120,422,400 B as Q6_K, both larger than the 64 MiB span."""
    size = block * REAL_BLOCKS
    assert size > 64 << 20
    base = tmp_path / "base"
    base.mkdir()
    write_gguf(base / "model.gguf", [("blk.0.w", (28672, 5120), kind, bytes(size))])
    cand = tmp_path / "cand"
    cand.mkdir()
    shutil.copyfile(base / "model.gguf", cand / "model.gguf")
    assert gguf_near_copy(cand, base) == "a baseline checkpoint as-is, not your result"
    at = read_table(cand / "model.gguf").data_start + block * 500_000 + params
    with (cand / "model.gguf").open("r+b") as fh:  # one scale byte, in a block past the first read span
        fh.seek(at)
        fh.write(b"\1")
    assert gguf_near_copy(cand, base) == "a baseline's packed codes with its scales changed, not your result"


def test_a_baseline_that_cannot_be_read_refuses_the_candidate(tmp_path, monkeypatch):
    """Fail closed: an unreadable baseline is named and the candidate refused, never skipped."""
    base = gguf_dir(tmp_path / "baseline")
    r = Rig(
        tmp_path,
        engine="llama.cpp",
        budget_bytes=1 << 20,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        baselines=[base],
    )
    candidate = gguf_dir(r.work / "a")
    write_gguf(candidate / "model.gguf", [("blk.0.attn_q.weight", (256,), Q4_K, b"\2" * 144)])
    (base / "model.gguf").write_bytes(b"GGUF\x03")  # the baseline stops being readable after the host started
    # candidate 0's types gate admission, so the hand-in itself is refused, naming the baseline
    rep = r.call("handin", "a")
    assert rep.exit == 2 and not r.host.entries
    assert "baselines[0] (candidate 0) cannot be read, so the stored types cannot be checked" in rep.stderr
    assert "not a little-endian GGUF" in rep.stderr


def test_a_second_baseline_that_cannot_be_read_refuses_the_candidate(tmp_path, monkeypatch):
    """Fail closed past admission too: the copy check names the unreadable baseline and the candidate is invalid."""
    base, other = gguf_dir(tmp_path / "baseline"), gguf_dir(tmp_path / "other")
    r = Rig(
        tmp_path,
        engine="llama.cpp",
        budget_bytes=1 << 20,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        baselines=[base, other],
    )
    candidate = gguf_dir(r.work / "a")
    write_gguf(candidate / "model.gguf", [("blk.0.attn_q.weight", (256,), Q4_K, b"\2" * 144)])
    (other / "model.gguf").write_bytes(
        b"GGUF\x03"
    )  # the baseline stops being readable after the host started
    assert r.call("handin", "a").exit == 0
    (e,) = r.host.entries
    assert e.valid is False
    assert "baselines[1] cannot be read, so a copy of it cannot be ruled out" in e.reason
    assert "not a little-endian GGUF" in e.reason


def test_a_gguf_copy_of_a_baseline_is_refused(tmp_path, monkeypatch):
    base = gguf_dir(tmp_path / "baseline")
    r = Rig(
        tmp_path,
        engine="llama.cpp",
        budget_bytes=1 << 20,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        baselines=[base],
    )
    shutil.copytree(base, r.work / "a")
    assert r.call("handin", "a").exit == 0
    (e,) = r.host.entries
    assert e.valid is False and e.reason == "a baseline checkpoint as-is, not your result"


def test_a_llama_cpp_build_is_admitted_by_its_plans_file_size(tmp_path, monkeypatch):
    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=100,
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        format_menu=["Q4_K"],
    )
    plan = SimpleNamespace(weight_bytes=80, output_bytes=101, output_bytes_exact=True)
    monkeypatch.setattr(build_receipt, "_build_plan", lambda raw, base, target=None: plan)
    with pytest.raises(Refused, match="file 101 B > budget 100 B"):
        build_receipt.budget_check(r.host, {}, str(tmp_path))
    plan.output_bytes = 100  # the header counts: a payload of 80 is not a file of 80
    assert build_receipt.budget_check(r.host, {}, str(tmp_path)) == 100 + build_receipt.GGUF_HEADROOM
    plan.output_bytes = 0  # UNKNOWN is not "fits"
    with pytest.raises(Refused, match="does not price"):
        build_receipt.budget_check(r.host, {}, str(tmp_path))
    plan.output_bytes, plan.output_bytes_exact = 100, False
    with pytest.raises(Refused, match="estimate, not the exporter's own"):
        build_receipt.budget_check(r.host, {}, str(tmp_path))
    monkeypatch.setattr(build_receipt, "_build_plan", lambda raw, base, target=None: None)
    with pytest.raises(Refused, match="could not be read"):
        build_receipt.budget_check(r.host, {}, str(tmp_path))
    assert build_receipt.budget_check(Rig(tmp_path / "vllm").host, {}, str(tmp_path)) is None


def test_a_payload_basis_build_screens_the_planned_weight_bytes(tmp_path, monkeypatch):
    """The payload basis is an explicit opt-in; the reservation is still the file's exact size."""
    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=100,
        budget_basis="payload",
        engine_floor=0,
        llamacpp=llama_paths(tmp_path, monkeypatch),
        format_menu=["Q4_K"],
    )
    plan = SimpleNamespace(weight_bytes=101, output_bytes=2**30, output_bytes_exact=True)
    monkeypatch.setattr(build_receipt, "_build_plan", lambda raw, base, target=None: plan)
    with pytest.raises(Refused, match="payload 101 B > budget 100 B"):
        build_receipt.budget_check(r.host, {}, str(tmp_path))
    plan.weight_bytes = 100
    assert build_receipt.budget_check(r.host, {}, str(tmp_path)) == 2**30 + build_receipt.GGUF_HEADROOM
    plan.weight_bytes = 0  # UNKNOWN is not "fits"
    with pytest.raises(Refused, match="does not price"):
        build_receipt.budget_check(r.host, {}, str(tmp_path))


#: (ggml type, values of one block, bytes of one block) of three types a file may store beside the menu's:
#: Q3_K (11), IQ4_NL (20) and IQ3_S (21). A Q6_K-layout file holds IQ4_NL; a Q4_K_M one holds all three.
VENDOR_TYPES = [(11, 256, 110), (20, 32, 18), (21, 256, 110)]


def test_the_vendor_types_take_the_pinned_gguf_py_names_and_block_layouts():
    """A vendor file's extra types are sized, named and laid out as the pinned gguf-py does, not hand-typed."""
    gguf = pytest.importorskip("gguf")
    from qda_agent.judge.checks import gguf as judge_gguf
    from qda_quant.gguf.file import row_bytes

    for kind, values, size in VENDOR_TYPES:
        upstream = gguf.GGMLQuantizationType(kind)
        assert gguf.GGML_QUANT_SIZES[upstream] == (values, size)
        assert judge_gguf._ggml_name(kind) == upstream.name
        assert row_bytes(kind, values) == size
        assert judge_gguf.GGML_BLOCKS[kind][0] == size


@pytest.mark.parametrize("kind", [Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, 11, 20, 21])
def test_every_block_layout_covers_its_block_exactly_once(kind):
    """Codes and parameters together are the whole block, with no byte in both and none in neither."""
    from qda_agent.judge.checks import gguf as judge_gguf

    size, codes, params = judge_gguf.GGML_BLOCKS[kind]
    covered = sorted(i for a, b in (*codes, *params) for i in range(a, b))
    assert covered == list(range(size))


@pytest.mark.parametrize("kind", [11, 20, 21])
def test_a_vendor_types_code_and_scale_spans_are_what_gguf_py_dequantizes(kind):
    """Zeroing a type's scale spans zeroes the dequantized block whatever its codes; a code byte moves a value."""
    gguf = pytest.importorskip("gguf")
    import numpy as np
    from gguf.quants import dequantize
    from qda_agent.judge.checks import gguf as judge_gguf

    size, codes, params = judge_gguf.GGML_BLOCKS[kind]
    rng = np.random.default_rng(0)
    block = rng.integers(0, 256, size, dtype=np.uint8)
    block[0:2] = np.frombuffer(
        np.float16(0.5).tobytes(), np.uint8
    )  # a finite fp16 d where the type keeps one
    if kind == 11:
        block[108:110] = np.frombuffer(np.float16(0.5).tobytes(), np.uint8)
    qtype = gguf.GGMLQuantizationType(kind)
    before = dequantize(block, qtype)
    assert np.isfinite(before).all() and np.abs(before).sum() > 0
    zeroed = block.copy()
    for a, b in params:
        zeroed[a:b] = 0
    after_scales = dequantize(zeroed, qtype)
    assert not np.array_equal(after_scales, before)
    moved = block.copy()
    a, _ = codes[0]
    moved[a] ^= 0xFF
    assert not np.array_equal(dequantize(moved, qtype), before)


@pytest.mark.parametrize(("kind", "values", "block"), VENDOR_TYPES)
def test_a_vendor_file_is_read_whole_and_gated_by_name(tmp_path, kind, values, block):
    """A file with one of these extra types is still read: sized, fingerprinted, refused by NAME by a menu."""
    base = tmp_path / "vendor"
    base.mkdir()
    write_gguf(
        base / "model.gguf",
        [
            ("blk.0.ffn_gate.weight", (values, 4), kind, bytes(range(block)) * 4),
            ("blk.0.w", (256,), Q6_K, b"\3" * 210),
        ],
    )
    assert gguf_problem(base) is None  # a baseline gets no menu gate
    sizes = gguf_sizes(base)
    assert sizes is not None and sizes["payload"] == block * 4 + 210
    assert gguf_fingerprint(base) is not None
    why = gguf_problem(base, menu=S5_MENU)
    name = {11: "Q3_K", 20: "IQ4_NL", 21: "IQ3_S"}[kind]
    assert (
        why == f"blk.0.ffn_gate.weight is stored as {name}, outside judge.format_menu ({', '.join(S5_MENU)})"
    )


#: Every type the library can write solved GPTQ codes in (`qda_quant.gguf.writer._SOLVED`), as a menu names
#: it: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL (20) and Q3_K (11). A search may pick any of them per tensor.
SOLVED_TYPES = [
    ("Q4_K", Q4_K, 256, 144),
    ("Q5_K", Q5_K, 256, 176),
    ("Q6_K", Q6_K, 256, 210),
    ("IQ4_XS", IQ4_XS, 256, 136),
    ("IQ4_NL", 20, 32, 18),
    ("Q3_K", 11, 256, 110),
]


def test_the_solved_types_are_the_ones_the_gguf_writer_encodes_codes_for():
    """The table above is the writer's own `_SOLVED`, so a type it gains and the judge has not is caught."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import type_block, type_name
    from qda_quant.gguf.writer import _SOLVED

    assert {name for name, *_ in SOLVED_TYPES} == {type_name(kind) for kind, _ in _SOLVED.values()}
    for name, kind, values, block in SOLVED_TYPES:
        assert type_name(kind) == name and type_block(kind) == (block, values)


@pytest.mark.parametrize(("name", "kind", "values", "block"), SOLVED_TYPES)
def test_a_candidate_storing_a_solved_type_is_admitted_by_a_menu_that_names_it(
    tmp_path, name, kind, values, block
):
    """A build's own codes in any solved type pass admission; only the menu decides, and it names them all."""
    cand = tmp_path / "cand"
    cand.mkdir()
    write_gguf(
        cand / "model.gguf",
        [
            ("blk.0.ffn_gate.weight", (values, 4), kind, bytes(range(block)) * 4),
            ("blk.0.attn_norm.weight", (8,), F32, bytes(32)),
        ],
    )
    menu = tuple(n for n, *_ in SOLVED_TYPES)
    assert gguf_problem(cand, menu=menu) is None
    why = gguf_problem(cand, menu=tuple(n for n in menu if n != name))
    assert why == (
        f"blk.0.ffn_gate.weight is stored as {name}, outside judge.format_menu "
        f"({', '.join(n for n in menu if n != name)})"
    )


@pytest.mark.parametrize(("kind", "values", "block"), VENDOR_TYPES)
def test_a_copy_of_a_vendor_baseline_is_refused_and_a_rescale_named(
    tmp_path, monkeypatch, kind, values, block
):
    """A candidate that is the vendor file, or its codes with one scale byte changed, is refused as a copy."""
    from qda_agent.judge.checks import gguf as judge_gguf

    monkeypatch.setattr("qda_agent.judge.checks.gguf.READ_SPAN", 1024)  # 1024 % 110 and 1024 % 18 are nonzero
    base = tmp_path / "base"
    base.mkdir()
    write_gguf(base / "model.gguf", [("blk.0.w", (values, 20), kind, bytes(block * 20))])
    cand = tmp_path / "cand"
    cand.mkdir()
    shutil.copyfile(base / "model.gguf", cand / "model.gguf")
    assert gguf_near_copy(cand, base) == "a baseline checkpoint as-is, not your result"
    _, _, params = judge_gguf.GGML_BLOCKS[kind]
    last = params[-1][0]  # IQ3_S: the scale bytes after its codes, not only the leading d
    at = read_table(cand / "model.gguf").data_start + block * 17 + last
    with (cand / "model.gguf").open("r+b") as fh:
        fh.seek(at)
        fh.write(b"\1")
    assert gguf_near_copy(cand, base) == "a baseline's packed codes with its scales changed, not your result"
    _, codes, _ = judge_gguf.GGML_BLOCKS[kind]
    with (cand / "model.gguf").open("r+b") as fh:  # and a changed code byte is no longer a copy at all
        fh.seek(read_table(cand / "model.gguf").data_start + block * 3 + codes[0][0])
        fh.write(b"\7")
    assert gguf_near_copy(cand, base) is None


def test_the_judges_own_gguf_lookup_takes_the_file_or_its_directory(tmp_path):
    from qda_agent.judge.checks import gguf as gguf_checks

    d = gguf_dir(tmp_path / "cand")
    assert gguf_checks.the_gguf(d) == gguf_checks.the_gguf(d / "model.gguf") == d / "model.gguf"
    (tmp_path / "alias.gguf").symlink_to(d / "model.gguf")
    assert gguf_checks.the_gguf(tmp_path / "alias.gguf") is None
    (d / "config.json").write_text("{}")
    assert gguf_checks.the_gguf(d / "config.json") is None
