"""One bf16 GGUF template per llama.cpp search: converted once, handed to every build, counted once.

The converter is a stand-in: a Python script at the checkout's `convert_hf_to_gguf.py` that appends one line per
run to a log and writes its `--outfile`. The plans are stand-ins too, as in `test_judge_gguf`: what is under
test is the judge's side (who converts, what each build is handed, and what the disk cap counts), not the
writer. The jobs run at isolation level `none`, so no test needs bwrap.
"""

from __future__ import annotations

import json
import stat
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from qda_agent import cand0
from qda_agent.isolation.levels.none import NoneIsolation
from qda_agent.judge import build_receipt, gguf_template, prune
from qda_agent.judge.config import JudgeSettings
from qda_quant.gguf.llama_cpp import PINNED
from safetensors_kit import write_safetensors
from test_judge import Rig, llama_paths

#: What the stand-in converter writes: a template's bytes (a real template is the source in bf16).
TEMPLATE_BYTES = 4096
#: The plan's exact `.gguf` size the stand-in plan reports.
PLANNED = 1000

CONVERTER = """import pathlib, sys
argv = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(" ".join(argv) + "\\n")
pathlib.Path(argv[argv.index("--outfile") + 1]).write_bytes(b"G" * {size})
"""


def source_dir(d: Path, *, width: int = 64) -> Path:
    """A local model: one bf16 matrix and one norm, so its template estimate is 2 x width^2 + 4 x width bytes."""
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 1}))
    write_safetensors(
        d / "model.safetensors",
        {
            "model.layers.0.self_attn.q_proj.weight": ("BF16", (width, width), None),
            "model.norm.weight": ("BF16", (width,), None),
        },
    )
    return d


def converter(paths: dict, log: Path) -> None:
    """Make the fake checkout's `convert_hf_to_gguf.py` the stand-in converter."""
    script = Path(paths["llama_cpp"]) / "convert_hf_to_gguf.py"
    script.write_text(CONVERTER.format(log=str(log), size=TEMPLATE_BYTES))


def rig(tmp_path: Path, monkeypatch, *, source: Path | None = None, **judge) -> tuple[Rig, Path, Path]:
    """A llama.cpp judge whose model is a local source and whose checkout converts with the stand-in."""
    paths = llama_paths(tmp_path, monkeypatch)
    log = tmp_path / "converter.log"
    converter(paths, log)
    source = source or source_dir(tmp_path / "original")
    monkeypatch.setattr("qda_quant.gguf.llama_cpp.commit_of", lambda _root: PINNED)
    r = Rig(
        tmp_path,
        check=False,
        engine="llama.cpp",
        budget_bytes=PLANNED,
        engine_floor=0,
        llamacpp={**paths, **judge.pop("llamacpp", {})},
        format_menu=["Q4_K"],
        anchor=source,
        qda_cmd=[sys.executable, "-m", "qda_quant.cli"],
        **judge,
    )
    # level `none`: the judge copies the agent's files itself (no bwrap on a CPU test host)
    r.host.sched.iso = NoneIsolation(
        SimpleNamespace(run_dir=r.run, isolation=SimpleNamespace(tamper_check="off"))
    )
    return r, source, log


def planned(monkeypatch) -> list[dict]:
    """Stand in for the build plan: an exact `.gguf` of `PLANNED` bytes; returns the recipes it was read from."""
    seen: list[dict] = []

    def plan(raw, base, target=None):  # noqa: ANN001, ANN202, ARG001 -- `_build_plan`'s own signature
        seen.append(raw)
        return SimpleNamespace(output_bytes=PLANNED, output_bytes_exact=True, weight_bytes=PLANNED - 100)

    monkeypatch.setattr(build_receipt, "_build_plan", plan)
    return seen


def recipe(r: Rig, name: str, model: Path, **gguf) -> str:
    """A GGUF recipe in the agent's workdir."""
    (r.work / f"{name}.yaml").write_text(
        yaml.safe_dump({"model": str(model), "out": name, "exporter": {"gguf": gguf}})
    )
    return f"{name}.yaml"


def calls(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.exists() else 0


def handed(job) -> dict | None:  # noqa: ANN001
    """The exporter a job's host override sets, or None when the job carries none."""
    argv = job.argv
    for i, a in enumerate(argv[:-1]):
        if a == "--set" and argv[i + 1].startswith("exporter="):
            return json.loads(argv[i + 1][len("exporter=") :])
    return None


# ------------------------------------------------------------------------------------- one template per run


def test_two_builds_share_one_conversion_and_each_is_handed_the_template(tmp_path, monkeypatch):
    r, source, log = rig(tmp_path, monkeypatch)
    seen = planned(monkeypatch)
    t = gguf_template.prepare(r.host)
    assert t is not None and t.made and t.source == str(source.resolve()) and t.commit == PINNED
    assert calls(log) == 1
    # in the run dir, sealed read-only like the frozen inputs, with its record
    assert t.path == r.run / "judge" / gguf_template.DIR / gguf_template.FILE
    assert stat.S_IMODE(t.path.stat().st_mode) == 0o444
    assert stat.S_IMODE(t.path.parent.stat().st_mode) == 0o555
    record = json.loads(gguf_template.record_path(t.path).read_text())
    assert record["source"] == str(source.resolve()) and record["llama_cpp"] == PINNED
    # the build sandboxes read it, inside the run dir, as a host-written area
    assert t.path in r.host.sched.cfg.readable and t.path in r.host.sched.cfg.host_written

    for name in ("c1", "c2"):
        reply = r.call("qda", "build", recipe(r, name, source), "--expected", "1m")
        assert reply.exit == 0, reply.stderr
    assert calls(log) == 1  # the second build converted nothing either
    jobs = r.host.sched.list()
    assert [handed(j) for j in jobs] == [{"gguf": {"template": str(t.path)}}] * 2
    assert all(f"model={json.dumps(str(source.resolve()))}" in j.argv for j in jobs)
    # the plan was read with the template set, so it priced the file exactly
    assert [raw["exporter"]["gguf"]["template"] for raw in seen] == [str(t.path)] * 2
    # and each build reserves its file only: the template is already on disk, counted there
    assert [j.expected_bytes for j in jobs] == [PLANNED + build_receipt.GGUF_HEADROOM] * 2


def test_a_trusted_build_is_handed_the_template_by_a_host_override(tmp_path, monkeypatch):
    """A receipted build's runner applies the host's `forced` overrides last: the template and the source."""
    r, source, log = rig(tmp_path, monkeypatch)
    seen = planned(monkeypatch)
    t = gguf_template.prepare(r.host)
    assert t is not None
    r.host.sched.cfg = r.host.sched.cfg.model_copy(update={"job_sandbox": "bwrap"})
    (r.run / "judge" / "builds").mkdir()
    private, spec = build_receipt.setup(r.host, r.work, [recipe(r, "c1", source)])
    assert spec["forced"] == [
        f"model={json.dumps(str(source.resolve()))}",
        f"exporter={json.dumps({'gguf': {'template': str(t.path)}})}",
    ]
    assert spec["expected_bytes"] == PLANNED + build_receipt.GGUF_HEADROOM
    assert seen[-1]["exporter"]["gguf"]["template"] == str(t.path) and calls(log) == 1
    from qda_quant.recipe import apply_overrides

    built = apply_overrides(yaml.safe_load(Path(spec["config"]).read_text()), spec["forced"])
    assert built["exporter"] == {"gguf": {"template": str(t.path)}}  # what the build itself will read


def test_the_template_is_counted_once_on_disk_and_never_in_a_reservation(tmp_path, monkeypatch):
    r, source, _ = rig(tmp_path, monkeypatch)
    planned(monkeypatch)
    before = prune.run_bytes(r.host)
    t = gguf_template.prepare(r.host)
    assert t is not None
    record = gguf_template.record_path(t.path).stat().st_size
    assert prune.run_bytes(r.host) == before + TEMPLATE_BYTES + record
    for name in ("c1", "c2"):
        assert r.call("qda", "build", recipe(r, name, source), "--expected", "1m").exit == 0
    assert prune.reserved_bytes(r.host) == 2 * (PLANNED + build_receipt.GGUF_HEADROOM)


def test_a_template_the_cap_cannot_hold_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    """The mutation of the guard: a cap below one template. The start is refused, naming the bytes."""
    source = source_dir(tmp_path / "original")
    size = gguf_template.estimate_bytes(source)
    r, _, log = rig(tmp_path, monkeypatch, source=source, max_run_bytes=size // 2)
    with pytest.raises(ValueError, match="does not fit") as e:
        gguf_template.prepare(r.host)
    assert f"about {size / 1e9:.1f} GB" in str(e.value) and "disk budget:" in str(e.value)
    assert calls(log) == 0 and not (r.run / "judge" / gguf_template.DIR).exists()
    assert r.host.gguf_template is None


def test_estimate_counts_matrices_in_bf16_and_vectors_in_f32(tmp_path):
    """A matrix in bf16 and a vector in F32, plus the metadata files and the header margin."""
    source = source_dir(tmp_path / "m", width=64)
    meta = (source / "config.json").stat().st_size
    assert gguf_template.estimate_bytes(source) == 64 * 64 * 2 + 64 * 4 + meta + gguf_template.HEADER_MARGIN


# --------------------------------------------------------------------------- builds the template does not fit


def test_a_build_of_another_source_converts_its_own_and_reserves_it_up_front(tmp_path, monkeypatch):
    r, _, _ = rig(tmp_path, monkeypatch)
    planned(monkeypatch)
    gguf_template.prepare(r.host)
    other = source_dir(tmp_path / "other", width=128)
    assert r.call("qda", "build", recipe(r, "c1", other), "--expected", "1m").exit == 0
    (job,) = r.host.sched.list()
    assert handed(job) is None  # the run's template is not this source's
    own = gguf_template.estimate_bytes(other)
    assert job.expected_bytes == PLANNED + build_receipt.GGUF_HEADROOM + own


def test_concurrent_builds_that_convert_their_own_are_refused_when_together_they_pass_the_cap(
    tmp_path, monkeypatch
):
    """Two builds that each fit alone: the second is refused, because the first's template is reserved."""
    other = source_dir(tmp_path / "other", width=128)
    one = PLANNED + build_receipt.GGUF_HEADROOM + gguf_template.estimate_bytes(other)
    r, _, _ = rig(tmp_path, monkeypatch, max_run_bytes=int(1.5 * one))
    planned(monkeypatch)
    r.host.gguf_template = None  # no run template: every build converts its own
    assert prune.run_bytes(r.host) < one // 2
    assert r.call("qda", "build", recipe(r, "c1", other), "--expected", "1m").exit == 0
    reply = r.call("qda", "build", recipe(r, "c2", other), "--expected", "1m")
    assert reply.exit == 2 and "disk budget:" in reply.stderr
    assert "reserved for queued/running jobs" in reply.stderr
    assert len(r.host.sched.list()) == 1  # refused before it was queued


def test_a_build_whose_own_converter_is_at_another_commit_converts_its_own(tmp_path, monkeypatch):
    r, source, _ = rig(tmp_path, monkeypatch)
    planned(monkeypatch)
    gguf_template.prepare(r.host)
    other = tmp_path / "other-llama.cpp"
    other.mkdir()
    monkeypatch.setattr(
        "qda_quant.gguf.llama_cpp.commit_of", lambda root: "f" * 40 if Path(root) == other else PINNED
    )
    assert r.call("qda", "build", recipe(r, "c1", source, llama_cpp=str(other)), "--expected", "1m").exit == 0
    (job,) = r.host.sched.list()
    assert handed(job) is None
    assert job.expected_bytes > PLANNED + build_receipt.GGUF_HEADROOM


def test_a_recipe_that_names_its_own_template_is_refused(tmp_path, monkeypatch):
    r, source, _ = rig(tmp_path, monkeypatch)
    planned(monkeypatch)
    t = gguf_template.prepare(r.host)
    assert t is not None
    foreign = tmp_path / "foreign.gguf"
    foreign.write_bytes(b"G" * TEMPLATE_BYTES)
    reply = r.call("qda", "build", recipe(r, "c1", source, template=str(foreign)), "--expected", "1m")
    assert reply.exit == 2 and "exporter.gguf.template" in reply.stderr
    assert not r.host.sched.list()
    # naming the run's own file is no foreign template
    assert r.call("qda", "build", recipe(r, "c2", source, template=str(t.path)), "--expected", "1m").exit == 0


def test_a_build_with_another_exporter_is_left_alone(tmp_path, monkeypatch):
    r, source, _ = rig(tmp_path, monkeypatch)
    gguf_template.prepare(r.host)
    raw = {"model": str(source), "out": "x"}
    got = gguf_template.for_build(r.host, raw, str(source.resolve()), str(source.resolve()), {})
    assert got == gguf_template.Handed(raw, [], 0)


# ------------------------------------------------------------------------------ the operator's template


def write_template(path: Path, kind: int = 30) -> Path:
    """A minimal GGUF the template reader accepts: an architecture, a block count, one tensor of `kind` (30 bf16)."""

    def text(s: str) -> bytes:
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    kv = text("general.architecture") + struct.pack("<I", 8) + text("llama")
    kv += text("llama.block_count") + struct.pack("<II", 4, 1)
    info = (
        text("blk.0.attn_q.weight")
        + struct.pack("<I", 1)
        + struct.pack("<Q", 32)
        + struct.pack("<IQ", kind, 0)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + kv + info
    path.write_bytes(header + b"\0" * (-len(header) % 32) + b"\0" * 64)
    return path


def test_the_operators_template_is_used_as_is_and_not_converted_or_counted(tmp_path, monkeypatch):
    (tmp_path / "prepared").mkdir()
    given = write_template(tmp_path / "prepared" / "model-bf16.gguf")
    r, source, log = rig(tmp_path, monkeypatch, llamacpp={"template": str(given)})
    gguf_template.record_path(given).write_text(
        json.dumps({"source": str(source.resolve()), "llama_cpp": PINNED})
    )
    planned(monkeypatch)
    before = prune.run_bytes(r.host)
    t = gguf_template.prepare(r.host)
    assert t is not None and not t.made and t.path == given.resolve()
    assert calls(log) == 0 and prune.run_bytes(r.host) == before
    assert (
        given.resolve() in r.host.sched.cfg.readable and given.resolve() not in r.host.sched.cfg.host_written
    )
    assert r.call("qda", "build", recipe(r, "c1", source), "--expected", "1m").exit == 0
    assert handed(r.host.sched.list()[0]) == {"gguf": {"template": str(given.resolve())}}


@pytest.mark.parametrize(
    "record, kind, why",
    [
        ({"source": "/elsewhere/model"}, 30, "not the run's model"),
        ({"llama_cpp": "0" * 40}, 30, "was converted by llama.cpp"),
        ({}, 1, "ggml type 1"),  # F16: an f16 conversion rounds differently
        ({}, 2, "ggml type 2"),  # Q4_0: a quantized template is not the converter's bf16 output
    ],
)
def test_an_operators_template_of_another_source_commit_or_dtype_is_refused(
    tmp_path, monkeypatch, record, kind, why
):
    (tmp_path / "prepared").mkdir()
    given = write_template(tmp_path / "prepared" / "model-bf16.gguf", kind)
    gguf_template.record_path(given).write_text(json.dumps(record))
    r, _, _ = rig(tmp_path, monkeypatch, llamacpp={"template": str(given)})
    with pytest.raises(ValueError, match=why):
        gguf_template.prepare(r.host)


# ------------------------------------------------------------------------------------- the disk lines


def test_the_disk_cap_line_counts_the_runs_template_once(tmp_path, monkeypatch):
    paths = llama_paths(tmp_path, monkeypatch)
    source = source_dir(tmp_path / "original", width=512)
    size = gguf_template.estimate_bytes(source)
    fields = {"engine": "llama.cpp", "budget_bytes": 100_000, "engine_floor": 0}
    cfg = JudgeSettings(**fields, llamacpp=paths, anchor=source, max_run_bytes=size + 350_000)
    assert gguf_template.run_template_bytes(cfg) == size
    line = cfg.disk_cap_line()
    assert line is not None and "bf16 GGUF template holds about" in line
    assert "about 3 builds of 0.0 GB fit beside the template" in line  # (cap - template) // budget
    # an operator's template is outside the run: the run holds none
    (tmp_path / "t.gguf").write_bytes(b"G")
    given = JudgeSettings(**fields, llamacpp={**paths, "template": str(tmp_path / "t.gguf")}, anchor=source)
    assert gguf_template.run_template_bytes(given) == 0


def test_doctor_refuses_a_cap_that_cannot_hold_the_template_and_one_build(tmp_path, monkeypatch):
    from qda_agent.doctor import check_disk_cap

    paths = llama_paths(tmp_path, monkeypatch)
    source = source_dir(tmp_path / "original", width=512)
    size = gguf_template.estimate_bytes(source)
    fields = {"engine": "llama.cpp", "budget_bytes": 100_000, "engine_floor": 0}

    def check(cap: int):  # noqa: ANN202
        judge = JudgeSettings(**fields, llamacpp=paths, anchor=source, max_run_bytes=cap)
        return check_disk_cap(SimpleNamespace(judge=judge, run_dir=tmp_path / "run", cards=[0]))

    short = check(size + 50_000)
    assert not short.ok and "cannot hold the run's bf16 GGUF template" in short.detail
    roomy = check(size + 500_000)
    assert roomy.ok and "the bf16 GGUF template about" in roomy.detail


# -------------------------------------------------------------------------- candidate 0 and prepare


def test_the_converter_prints_the_chat_template_into_its_own_log_not_the_terminal(tmp_path, capsys):
    """llama.cpp's converter logs each tensor and the model's whole chat template (`INFO:gguf.vocab:Setting
    chat_template to ...`): on an 8B terminal that buries the step's own lines under a 2 KB jinja string."""
    source = source_dir(tmp_path / "original")
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text(
        "import pathlib, sys\n"
        "print('INFO:gguf.vocab:Setting chat_template to {% for m in messages %}{{ m.content }}{% endfor %}')\n"
        "argv = sys.argv[1:]\n"
        f"pathlib.Path(argv[argv.index('--outfile') + 1]).write_bytes(b'G' * {TEMPLATE_BYTES})\n"
    )
    out = tmp_path / "bf16" / "model-bf16.gguf"
    lines: list[str] = []
    assert cand0.make_template(sys.executable, llama, source, out, log=lines.append) == out
    assert "chat_template" not in capsys.readouterr().out
    assert any("convert.log" in line for line in lines)  # the one line says where the converter's went
    assert "chat_template" in out.with_name(out.name + ".convert.log").read_text()


def test_candidate_0_converts_once_and_names_the_template_for_the_search(tmp_path):
    source = source_dir(tmp_path / "original")
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    log = tmp_path / "converter.log"
    converter({"llama_cpp": str(llama)}, log)
    out = tmp_path / "bf16" / "model-bf16.gguf"
    made = cand0.make_template(sys.executable, llama, source, out, log=lambda _: None)
    again = cand0.make_template(sys.executable, llama, source, out, log=lambda _: None)
    assert made == again == out and calls(log) == 1  # the second asks for the same one and converts nothing
    assert json.loads(out.with_name(out.name + ".json").read_text())["source"] == str(source.resolve())
    other = source_dir(tmp_path / "other")
    with pytest.raises(ValueError, match="was made from another source"):
        cand0.make_template(sys.executable, llama, other, out, log=lambda _: None)
    (source / "model.safetensors").write_bytes((source / "model.safetensors").read_bytes() + b"\0")
    with pytest.raises(ValueError, match="another state of the model's files"):
        cand0.make_template(sys.executable, llama, source, out, log=lambda _: None)
    cfg = cand0.Inputs(str(source), "w4a16", str(tmp_path / "calib.jsonl"), None, {})
    built = cand0.candidate0_recipe(cfg, method=cand0.GGUF, out=tmp_path / "c0", template=out)
    assert built["exporter"] == {"gguf": {"template": str(out)}}
    entry = cand0.entry(
        cand0.Inputs(
            str(source), "w4a16", None, None, {"judge": {"engine": "llama.cpp", "llamacpp": {"bridge": "b"}}}
        ),
        cand0.Candidate0(tmp_path / "c0", tmp_path / "r.yaml", 1.0, 2.0, cand0.GGUF, False, template=out),
    )
    assert entry["judge"]["llamacpp"] == {"bridge": "b", "template": str(out)}


def test_prepare_converts_once_for_candidate_0_the_floor_and_the_search(tmp_path, monkeypatch):
    from qda_agent import prepare as P
    from test_prepare import prepared

    paths = llama_paths(tmp_path, monkeypatch)
    log = tmp_path / "converter.log"
    converter(paths, log)
    monkeypatch.setattr(P, "llama_programs", lambda *_: {k: Path(v) for k, v in paths.items()})
    p = prepared(tmp_path, format="gguf-q4km")
    p.python = Path(sys.executable)
    got: dict = {}

    def build(inputs, **kw):  # noqa: ANN001, ANN003, ANN202 -- `build_candidate0`'s own signature
        got.update(kw)
        kw["out"].mkdir(parents=True)
        (kw["out"] / "model-Q4_K_M.gguf").write_bytes(b"\0" * 1234)
        return cand0.Candidate0(kw["out"], tmp_path / "r.yaml", 1.0, 2.0, kw["method"], False, kw["template"])

    monkeypatch.setattr(cand0, "build_candidate0", build)
    monkeypatch.setattr(P.Prepare, "calib_need", property(lambda self: {"samples": 1, "seqlen": 8}))
    p.imatrix.parent.mkdir(parents=True)
    p.imatrix.write_bytes(b"GGUF-imatrix")  # the imatrix step's output, which this test does not run
    step = next(s for s in P.plan(p, ("cand0",)))
    step.run()
    assert got["template"] == p.bf16 and got["imatrix"] == p.imatrix and p.bf16.is_file()
    assert calls(log) == 1
    assert p.recipe()["exporter"] == {"gguf": {"template": str(p.bf16), "imatrix": str(p.imatrix)}}
    assert p.make_bf16() == p.bf16 and calls(log) == 1  # the floor step reads the same file
    p.save(floor={"kl": 0.0003, "ref": "c"})
    raw = P.search_config(p)
    assert raw["judge"]["llamacpp"]["template"] == str(p.bf16)
    assert raw["judge"]["llamacpp"]["imatrix"] == str(p.imatrix)


def imatrix_file(path: Path, kind: str = "imatrix") -> Path:
    """A GGUF of `general.type` (a real `llama-imatrix` file states `imatrix`); no tensors needed."""
    import gguf

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_type(kind)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_the_runs_imatrix_is_handed_to_every_build_and_is_readable(tmp_path, monkeypatch):
    matrix = imatrix_file(tmp_path / "chat.imatrix.gguf")
    r, source, log = rig(tmp_path, monkeypatch, llamacpp={"imatrix": str(matrix)})
    planned(monkeypatch)
    t = gguf_template.prepare(r.host)
    assert r.host.imatrix == matrix.resolve() and matrix.resolve() in r.host.sched.cfg.readable
    for name in ("c1", "c2"):
        reply = r.call("qda", "build", recipe(r, name, source), "--expected", "1m")
        assert reply.exit == 0, reply.stderr
    for job in r.host.sched.list():
        assert handed(job)["gguf"]["imatrix"] == str(matrix.resolve())
        assert handed(job)["gguf"]["template"] == str(t.path)


def test_a_recipe_that_names_another_imatrix_is_refused(tmp_path, monkeypatch):
    matrix = imatrix_file(tmp_path / "chat.imatrix.gguf")
    other = imatrix_file(tmp_path / "other.imatrix.gguf")
    r, source, log = rig(tmp_path, monkeypatch, llamacpp={"imatrix": str(matrix)})
    planned(monkeypatch)
    gguf_template.prepare(r.host)
    reply = r.call("qda", "build", recipe(r, "c1", source, imatrix=str(other)), "--expected", "1m")
    assert reply.exit != 0 and "exporter.gguf.imatrix" in reply.stderr


def _prepared_imatrix(prepare, matrix: Path, *, old: bool = False) -> None:
    """The record `qda-agent prepare`'s imatrix step writes beside `matrix` (`Prepare.made_from`).

    `old`: the record an older prepare wrote, whose `source` was its own reuse key.
    """
    named = {"source": prepare.source_identity()} if old else prepare.made_from()
    assert named["source"] is not None
    record = {**named, "llama_cpp": PINNED, "calib_sha256": "0" * 64, "sha256": "1" * 64}
    gguf_template.record_path(matrix).write_text(json.dumps(record, indent=1) + "\n")


def _prepare_for(origin: str, tmp_path: Path):
    """A `Prepare` and its model directory: from a Hub id (downloaded into `DIR/model`) or a local directory."""
    from qda_agent import prepare as P

    out = tmp_path / "out"
    python = Path(sys.executable)
    if origin == "hub":
        model = source_dir(out / "model")  # the model step's download, so prepare asks the Hub nothing
        p = P.Prepare("org/model", out, python=python, serve_python=python)
        assert p.model_dir == model
        p.save(model={"repo": "org/model", "requested": "main", "revision": "0123abcd" * 5})
    else:
        model = source_dir(tmp_path / "local-model")
        p = P.Prepare(str(model), out, python=python, serve_python=python)
    return p, model


@pytest.mark.parametrize("origin", ["hub", "local"])
def test_the_imatrix_prepare_measured_is_accepted_by_the_runs_start(tmp_path, monkeypatch, origin):
    """Prepare's imatrix record and the judge's start check name the run's model the same way.

    From a Hub id, prepare downloads into `DIR/model`, writes that directory into the run config and records
    `repo@revision` beside the matrix; from a local directory it records that directory's identity. Either
    way the judge's start (`gguf_template.prepare`) must take the matrix as the run's own.
    """
    p, model = _prepare_for(origin, tmp_path)
    matrix = imatrix_file(tmp_path / "model.imatrix.gguf")
    _prepared_imatrix(p, matrix)
    r, _source, _log = rig(tmp_path, monkeypatch, source=model, llamacpp={"imatrix": str(matrix)})
    planned(monkeypatch)
    gguf_template.prepare(r.host)
    assert r.host.imatrix == matrix.resolve()


@pytest.mark.parametrize("origin", ["hub", "local"])
def test_an_imatrix_record_from_before_629_names_the_way_out(tmp_path, monkeypatch, origin):
    """A record whose `source` is prepare's old reuse key is refused, and the refusal says how to rewrite it."""
    p, model = _prepare_for(origin, tmp_path)
    matrix = imatrix_file(tmp_path / "model.imatrix.gguf")
    _prepared_imatrix(p, matrix, old=True)
    r, _source, _log = rig(tmp_path, monkeypatch, source=model, llamacpp={"imatrix": str(matrix)})
    with pytest.raises(ValueError, match=r"remove it and run `qda-agent prepare` again") as refused:
        gguf_template.prepare(r.host)
    assert f"was measured on {p.source_identity()}, not the run's model {model}" in str(refused.value)


def test_an_imatrix_of_another_kind_or_another_model_is_refused(tmp_path, monkeypatch):
    wrong = imatrix_file(tmp_path / "wrong.gguf", kind="model")
    r, source, log = rig(tmp_path, monkeypatch, llamacpp={"imatrix": str(wrong)})
    with pytest.raises(ValueError, match="not a llama-imatrix GGUF"):
        gguf_template.prepare(r.host)
    good = imatrix_file(tmp_path / "chat.imatrix.gguf")
    gguf_template.record_path(good).write_text(json.dumps({"source": "another model", "llama_cpp": PINNED}))
    (tmp_path / "second").mkdir()
    r2, _, _ = rig(tmp_path / "second", monkeypatch, llamacpp={"imatrix": str(good)})
    with pytest.raises(ValueError, match="was measured on another model"):
        gguf_template.prepare(r2.host)
