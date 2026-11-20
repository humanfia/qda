"""scale_kd, soup and staged builds on tiny saved models (CPU)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import transformers
from qda_quant.calib.sources import FileSource
from qda_quant.engine import packed as P
from qda_quant.engine import served_modules as S
from qda_quant.engine.build import build
from qda_quant.format import Format
from qda_quant.output.exporters import EXPORTERS, CompressedTensors
from qda_quant.recipe import Recipe
from qda_quant.solve import nvfp4
from qda_quant.solve.quant import Quantizer, fp8_token
from qda_quant.step import REGISTRY, Refused, Step
from qda_quant.target import Target
from safetensors import safe_open
from safetensors.torch import save_file

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, tiny_qwen3_moe, write_ids
from ..tiny import config

KD = {"steps": 4, "batch_tokens": 64, "lr": 1e-2, "warmup": 1, "head_chunk": 16, "attn": "eager"}


def _save(m, d: Path) -> str:
    m.save_pretrained(d)
    return save_tiny_tokenizer(d)


@pytest.fixture
def tiny(tmp_path):
    sha = _save(tiny_qwen3(layers=2, hidden=128, inter=256), tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (40,), generator=g).tolist() for _ in range(6)], sha
    )
    return tmp_path


def _recipe(tmp, *steps):
    return Recipe(steps=list(steps), calib=FileSource(path=str(tmp / "c.jsonl"), samples=4, seqlen=32))


def _gptq(tmp, fmt="w4a16", out="q", *, target=None):
    return build(str(tmp / "m"), _recipe(tmp, REGISTRY["gptq"](impl="torch")), format=fmt, out_dir=tmp / out,
                 target=target, devices=["cpu"], log=lambda _: None)  # fmt: skip


def _kd(tmp, src, out, devices=("cpu",), target=None, **kw):
    r = _recipe(tmp, REGISTRY["scale_kd"](**{**KD, **kw}))
    return build(str(src), r, target=target, out_dir=tmp / out, devices=list(devices), log=lambda _: None)


def _tensors(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            out |= {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
    return out


def _storage_bytes(m: torch.nn.Module) -> int:
    """Distinct storages a loaded model holds, in bytes: a tied head counts once, as it does on the card."""
    seen = {}
    for t in [*m.parameters(), *m.buffers()]:
        assert not t.is_meta
        storage = t.untyped_storage()
        seen[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(seen.values())


# ---------------------------------------------------------------- the operator


@pytest.mark.parametrize("fmt", ["w4a16", "w4a8", "w4a16-sym-g128"])
def test_served_linear_forward_and_grad(fmt, monkeypatch):
    """Forward = the served operator; the row-blocked backward = plain autograd through `weight()`."""
    torch.manual_seed(0)
    f = Format.parse(fmt)
    q = Quantizer.of(f, Target().plan(f, "dense"), "absmax")
    w = torch.randn(48, 256)
    s, z = q.params(w)
    codes = q.quantize(w, s, z)
    lin = S.ServedLinear("m", codes.to(torch.int8), None if z is None else z.to(torch.int8),
                         s.to(torch.bfloat16), 0.05 * torch.randn(s.shape), q)  # fmt: skip
    x = torch.randn(5, 256, dtype=torch.bfloat16, requires_grad=True)
    sc = (s.to(torch.bfloat16).float() * lin.delta.detach().exp()).to(torch.bfloat16).float()
    ref_w = q.dequant(codes, sc, z)
    ref_x = fp8_token(x) if f.act == "fp8_token" else x
    ref_y = ref_x.float() @ ref_w.t()
    assert (lin(x).float() - ref_y).norm() < 1e-2 * ref_y.norm()

    monkeypatch.setattr(S, "ROW_BLOCK_ELEMS", 256 * 7)  # several row blocks
    gy = torch.randn(5, 48, dtype=torch.bfloat16)
    lin(x).backward(gy)
    gd, gx = lin.delta.grad.clone(), x.grad.clone()
    # reference: autograd through the same weight, no custom Function
    d = lin.delta.detach().clone().requires_grad_(True)
    xs = x.detach().clone().requires_grad_(True)
    if f.act == "fp8_token":
        from qda_quant.solve.quant import fp8_token_parts

        c, sx = fp8_token_parts(xs.detach())
        xv = xs.float() + (c * sx - xs.float()).detach()
    else:
        xv = xs.float()
    wv = lin.weight(d)
    wv = wv if f.act == "fp8_token" else wv + (wv.to(torch.bfloat16).float() - wv).detach()
    (xv @ wv.t()).backward(gy.float())
    assert torch.allclose(gd, d.grad, rtol=1e-3, atol=1e-6)
    assert (gx.float() - xs.grad.float()).norm() < 1e-2 * xs.grad.float().norm()  # dx: a bf16 matmul


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
)
def test_w4a8_matmul_runs_on_the_e4m3_part_and_its_row_factor(device, monkeypatch):
    """W4A8 multiplies ``e4m3(q * f8)`` (exact in TF32) and applies ``chan`` after: the forward is the fp32
    product of the whole weight up to fp32 sums; a materialized step serves the per-call bits."""
    torch.manual_seed(0)
    f = Format.parse("w4a8")
    q = Quantizer.of(f, Target().plan(f, "dense"), "absmax")
    w = torch.randn(96, 512)
    w[0] *= 1e3  # a large scale next to small ones
    s, z = q.params(w)
    lin = S.ServedLinear("m", q.quantize(w, s, z).to(torch.int8), None, s.to(torch.bfloat16),
                         0.05 * torch.randn(s.shape), q).to(device)  # fmt: skip
    st = lin.stored
    assert st.split
    part, chan = st.weight_parts(lin.delta)
    assert torch.equal(part * chan, st.weight(lin.delta.detach()))
    assert torch.equal(part.to(torch.float8_e4m3fn).float(), part)
    x = torch.randn(7, 512, dtype=torch.bfloat16, device=device, requires_grad=True)
    y = lin(x)
    whole = st.matmul(x.detach(), st.weight(lin.delta.detach()))  # the fp32 product of the whole weight
    tol = 1e-6 if device == "cpu" else 1e-5
    assert (y.float() - whole.float()).norm() <= tol * whole.float().norm()
    lin.materialize()
    assert torch.equal(lin(x), y)
    gy = torch.randn(7, 96, dtype=torch.bfloat16, device=device)
    monkeypatch.setattr(S, "ROW_BLOCK_ELEMS", 512 * 40)  # several row blocks
    (gx,) = torch.autograd.grad(lin(x), x, gy)
    ref = (gy.float() @ st.weight(lin.delta.detach())).to(torch.bfloat16)
    # both end in bf16 (x's dtype): a sum near a rounding boundary can land one bf16 step apart
    tol = 1e-3 if device == "cpu" else 4e-3
    assert (gx.float() - ref.float()).norm() <= tol * ref.float().norm()


def _w4a8_linear(device, rows=96, cols=512):
    f = Format.parse("w4a8")
    q = Quantizer.of(f, Target().plan(f, "dense"), "absmax")
    w = torch.randn(rows, cols)
    w[0] *= 1e3
    s, z = q.params(w)
    return S.ServedLinear("m", q.quantize(w, s, z).to(torch.int8), None, s.to(torch.bfloat16),
                          0.3 * torch.randn(s.shape), q).to(device)  # fmt: skip


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
)
def test_w4a8_backward_without_materialize_decodes_each_row_block(device, monkeypatch):
    """Dense W4A8 never materializes: dx and d(delta) come from the per-call branch, which decodes each row block
    with that block's rows of the codes, scale and delta."""
    from qda_quant.solve.quant import fp8_token_parts

    torch.manual_seed(0)
    lin = _w4a8_linear(device)
    assert lin.stored.split
    assert lin._w is None  # no materialize()
    monkeypatch.setattr(S, "ROW_BLOCK_ELEMS", 512 * 40)  # 96 rows: three row blocks
    x = torch.randn(7, 512, dtype=torch.bfloat16, device=device, requires_grad=True)
    gy = torch.randn(7, 96, dtype=torch.bfloat16, device=device)
    lin(x).backward(gy)
    assert lin._w is None
    gd, gx = lin.delta.grad.clone(), x.grad.clone()
    d = lin.delta.detach().clone().requires_grad_(True)
    xs = x.detach().clone().requires_grad_(True)
    c, sx = fp8_token_parts(xs.detach())
    xv = xs.float() + (c * sx - xs.float()).detach()
    (xv @ lin.weight(d).t()).backward(gy.float())
    assert torch.allclose(gd, d.grad, rtol=1e-3, atol=1e-6)
    # dx ends in bf16: a sum near a rounding boundary can land one bf16 step apart
    tol = 1e-3 if device == "cpu" else 4e-3
    assert (gx.float() - xs.grad.float()).norm() <= tol * xs.grad.float().norm()


@pytest.mark.parametrize("fmt", ["w4a16", "w4a16-sym-g128", "w8a16", "w4a8"])
def test_fast_dequant_equals_reference_bits(fmt):
    """`weight_bf16` and `delta_grad` give the bits of the fp32 reference `weight` and autograd through it."""
    torch.manual_seed(0)
    f = Format.parse(fmt)
    q = Quantizer.of(f, Target().plan(f, "dense"), "absmax")
    w = torch.randn(48, 512)
    w[0] *= 1e3  # a large scale next to small ones
    s, z = q.params(w)
    st = S.Stored(q.quantize(w, s, z).to(torch.int8), None if z is None else z.to(torch.int8),
                  s.to(torch.bfloat16), q)  # fmt: skip
    delta = 0.05 * torch.randn(s.shape)
    rows = slice(5, 37)
    assert torch.equal(st.weight_bf16(delta[rows], rows), st.weight(delta[rows], rows).to(torch.bfloat16))
    gw = torch.randn(32, 512)
    d = delta[rows].clone().requires_grad_(True)
    assert torch.equal(
        st.delta_grad(delta[rows], gw, rows), torch.autograd.grad(st.weight(d, rows), d, gw)[0]
    )


# ---------------------------------------------------------------- scale_kd


def test_scale_kd_trains_scales_codes_identical(tiny):
    q = _gptq(tiny)
    out = _kd(tiny, q, "kd")
    a, b = _tensors(q), _tensors(out)
    assert set(a) == set(b)
    scales = [k for k in a if k.endswith(".weight_scale")]
    for k in a:
        if not k.endswith(".weight_scale"):
            assert torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype, k  # codes, zeros, norms: untouched
    assert sum(int((a[k] != b[k]).sum()) for k in scales) > 0
    rec = json.loads((out / "build.json").read_text())
    src = json.loads((q / "build.json").read_text())
    st = rec["history"][-1]
    assert rec["teacher"] == str(tiny / "m") and rec["lineage"] == [str(q)]
    assert rec["serve"] == src["serve"]
    assert rec["card_count"] == 1 and rec["devices"] == ["cpu"]
    assert "timings" not in rec and rec["source_build"]["timings"] == src["timings"]
    assert st["log"][-1]["kl"] < st["log"][0]["kl"]
    assert st["scales_changed"][0] > 0 and st["tokens_per_s"] > 0
    # the sidecar reproduces the stored scales, and vLLM / transformers never see it (not top-level)
    ck = P.Packed(out)
    sc, fl = ck.state()
    assert not fl and all(torch.equal(P.stored_scale(*sc[s]), b[s + ".weight_scale"]) for s in ck.stems)
    assert [p.name for p in (out / P.STATE.parent).iterdir()] == [
        P.STATE.name
    ]  # the one inert file hand-in allows
    assert not list(out.glob("qda_state*.safetensors"))
    assert "qda_state" not in json.dumps(json.loads((out / "model.safetensors.index.json").read_text()))


def test_checkpoint_stage_records_the_builds_cards(tiny):
    """A packed source built on one device, then a checkpoint stage on two, records the current cards; the
    source build's timings and residency are kept under ``source_build``, not read as this build's."""
    q = _gptq(tiny)
    src = json.loads((q / "build.json").read_text())
    out = _kd(tiny, q, "kd2", devices=["cpu", "cpu"])
    rec = json.loads((out / "build.json").read_text())
    assert rec["card_count"] == 2 and rec["devices"] == ["cpu", "cpu"]
    assert not {"timings", "rank_timings", "residency", "load_to_saved_s"} & set(rec)
    assert rec["source_build"] == {k: src[k] for k in ("timings", "residency", "load_to_saved_s")}
    assert [(s["kind"], s["steps"]) for s in rec["stages"]] == [("checkpoint", ["scale_kd"])]
    assert rec["history"][-1]["scales_changed"][0] > 0


@pytest.mark.parametrize("weights", [False, True])
def test_scale_kd_default_attention_supports_composite_audio(tmp_path, weights):
    """Automatic attention keeps SDPA for text without imposing it on an eager-only audio tower."""
    from qda_quant.intake.loading import load_model
    from qda_quant.steps.scale_kd import ScaleKD

    cfg = transformers.Gemma4Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 16,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention"],
        },
        vision_config=None,
        audio_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "output_proj_dims": 32,
            "subsampling_conv_channels": [8, 4],
        },
    )
    load_model(config=cfg, dtype=torch.bfloat16, attn_implementation="eager").save_pretrained(tmp_path)
    default = ScaleKD()
    model = load_model(tmp_path, weights=weights, dtype=torch.bfloat16, attn_implementation=default.attn)
    assert model.config.text_config._attn_implementation == "sdpa"
    assert model.config.audio_config._attn_implementation == "eager"
    assert default.model_dump()["attn"] is None
    eager = ScaleKD(attn="eager")
    model = load_model(tmp_path, weights=weights, dtype=torch.bfloat16, attn_implementation=eager.attn)
    assert model.config.text_config._attn_implementation == "eager"
    assert model.config.audio_config._attn_implementation == "eager"
    with pytest.raises(Refused, match="Gemma4AudioModel does not support"):
        load_model(tmp_path, weights=weights, attn_implementation=ScaleKD(attn="sdpa").attn)


def test_scale_kd_default_attention_dense_build(tiny):
    """The automatic attention setting reaches both loaders and stays recorded in a completed build."""
    q = _gptq(tiny)
    out = _kd(tiny, q, "kd-auto", attn=None)
    stage = json.loads((out / "build.json").read_text())["history"][-1]
    assert stage["settings"]["attn"] is None
    assert stage["scales_changed"][0] > 0


def test_student_buffers_match_from_pretrained(tiny):
    """The student's init-time buffers (rotary inv_freq) are the float model's, computed on the CPU."""
    import transformers

    ck = P.Packed(_gptq(tiny))
    sc, fl = ck.state(lambda _: None)
    s = P.load_student(ck, sc, fl, torch.device("cpu"), "eager")
    t = transformers.AutoModelForCausalLM.from_pretrained(tiny / "m")
    tb = dict(t.named_buffers())
    for n, b in s.named_buffers():
        if n in tb and b.is_floating_point():
            assert torch.equal(b, tb[n]), n


@pytest.mark.gpu
@pytest.mark.usefixtures("cpu_build")
def test_student_buffers_on_a_card(tmp_path):
    """Loaded onto a card, the student's rotary ``inv_freq`` is the CPU value from_pretrained computes.

    The regression the CPU test above cannot see: a buffer computed on the card gets other last bits. The
    student is built with its buffers on the CPU (`init_empty_weights`), so the values are right even with
    `engine.packed._cpu_buffers`, the belt-and-braces pass, disabled.
    """
    import transformers

    cfg = config("qwen3")
    cfg.head_dim, cfg.rope_theta = 128, 1_000_000.0  # Qwen3-8B's rotary: where the difference was seen
    cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads = 256, 2, 1
    torch.manual_seed(0)
    sha = _save(transformers.AutoModelForCausalLM.from_config(cfg), tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [list(range(40)) for _ in range(6)], sha)
    ck = P.Packed(_gptq(tmp_path))
    sc, fl = ck.state(lambda _: None)
    want = {
        n: b for n, b in transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "m").named_buffers()
    }
    dev = torch.device("cuda")

    def rotary(model):
        return {n: b.cpu() for n, b in model.named_buffers() if n.endswith("inv_freq")}

    orig = P._cpu_buffers
    try:
        P._cpu_buffers = lambda m: None
        got = rotary(P.load_student(ck, sc, fl, dev, "eager"))
    finally:
        P._cpu_buffers = orig
    assert got and all(torch.equal(b, want[n]) for n, b in got.items())


def test_scale_kd_w4a8(tiny):
    q = _gptq(tiny, "w4a8")
    out = _kd(tiny, q, "kd8")
    a, b = _tensors(q), _tensors(out)
    assert all(torch.equal(a[k], b[k]) for k in a if not k.endswith(".weight_scale"))
    assert json.loads((out / "build.json").read_text())["history"][-1]["scales_changed"][0] > 0


def test_staged_equals_two_builds(tiny):
    """[gptq, scale_kd] in one build is bit for bit a gptq build followed by a scale_kd build."""
    one = build(str(tiny / "m"), _recipe(tiny, REGISTRY["gptq"](impl="torch"), REGISTRY["scale_kd"](**KD)),
                format="w4a16", out_dir=tiny / "one", devices=["cpu"], log=lambda _: None)  # fmt: skip
    two = _kd(tiny, _gptq(tiny), "two")
    a, b = _tensors(one), _tensors(two)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert not (one / P.STAGES).exists()  # intermediates deleted
    rec = json.loads((one / "build.json").read_text())
    assert rec["teacher"] == str(tiny / "m") and [s["step"] for s in rec["history"]] == ["scale_kd"]
    assert not any(".partial-" in p for p in rec["lineage"])  # stages are recorded relative to the checkpoint
    assert [(s["kind"], s["steps"]) for s in rec["stages"]] == [
        ("layers", ["gptq"]),
        ("checkpoint", ["scale_kd"]),
    ]
    assert not (one / "recipe.yaml").exists()


class LabelledCT(CompressedTensors):
    """A third-party-style exporter: compressed-tensors plus a label file in its output."""

    label: str = "unlabelled"

    def writer(self, out_dir, source, graph, fmt, target, **kw):  # noqa: ANN001, ANN003, ANN201
        writer = super().writer(out_dir, source, graph, fmt, target, **kw)
        finish = writer.finish

        def labelled(*a, **k):  # noqa: ANN002, ANN003, ANN202
            out = finish(*a, **k)
            (Path(out) / "exporter-label.txt").write_text(self.label)
            return out

        writer.finish = labelled
        return writer


EXPORTERS.register("labelled_ct", LabelledCT)


class CopyCheckpoint(Step):
    """A checkpoint step: copies the packed checkpoint it is handed, `.stages` excluded."""

    name: ClassVar[str] = "copy_checkpoint"
    kind: ClassVar[str] = "checkpoint"

    def apply_checkpoint(self, src, out, run) -> None:  # noqa: ANN001, ARG002
        for p in Path(src).iterdir():
            if p.name == P.STAGES:
                continue
            if p.is_dir():
                shutil.copytree(p, Path(out) / p.name, dirs_exist_ok=True)
            else:
                shutil.copy2(p, Path(out) / p.name)


def test_staged_build_uses_a_third_party_exporter(tiny):
    """A staged build asks the exporter what it writes, not its name: a compressed-tensors subclass writes
    the float stage, the checkpoint step carries that checkpoint on, and the output loads."""
    import yaml
    from qda_quant.output.record import RESOLVED

    exporter = {"labelled_ct": {"label": "third-party"}}
    out = build(
        str(tiny / "m"),
        _recipe(tiny, REGISTRY["rtn"](), CopyCheckpoint()),
        format="w4a16-sym-g32",
        out_dir=tiny / "staged",
        devices=["cpu"],
        exporter=exporter,
        log=lambda _: None,
    )
    assert (out / "exporter-label.txt").read_text() == "third-party"  # the exporter wrote the float stage
    assert yaml.safe_load((out / RESOLVED).read_text())["exporter"] == exporter
    m = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    with torch.no_grad():
        logits = m(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()


def test_packed_source_refuses_an_exporter_whose_writer_never_runs(tiny):
    """A packed source runs no float stage, so no stage calls the exporter's writer: the config cannot name
    one whose settings would then be recorded without being applied."""
    packed = build(
        str(tiny / "m"),
        _recipe(tiny, REGISTRY["rtn"]()),
        format="w4a16-sym-g32",
        out_dir=tiny / "packed",
        devices=["cpu"],
        log=lambda _: None,
    )
    with pytest.raises(Refused, match="exporter 'labelled_ct': .*no float stage"):
        build(
            str(packed),
            _recipe(tiny, REGISTRY["soup"](**{"with": [str(packed)]})),
            out_dir=tiny / "souped",
            devices=["cpu"],
            exporter={"labelled_ct": {"label": "requested-final"}},
            log=lambda _: None,
        )
    assert not (tiny / "souped").exists()


def test_staged_build_still_refuses_gguf(tiny):
    """gguf writes one file, not a checkpoint a later stage reads: still refused before anything runs."""
    with pytest.raises(Refused, match="exporter 'gguf': .*only layer and model steps are written"):
        build(
            str(tiny / "m"),
            _recipe(tiny, REGISTRY["rtn"](grid="q4k"), CopyCheckpoint()),
            format="w4a16-asym-g32",
            out_dir=tiny / "staged",
            devices=["cpu"],
            exporter={"gguf": {"llama_cpp": str(tiny / "llama.cpp")}},
            log=lambda _: None,
        )
    assert not (tiny / "staged").exists()


def test_staged_rebuild_from_resolved_yaml(tiny):
    """A staged build writes ONE resolved.yaml for the whole recipe; ``qda build OUT/resolved.yaml`` repeats it.

    Bit for bit on the CPU. On a card the scale_kd backward is not deterministic, so a re-run there reproduces
    the config and the KL band, not the bytes.
    """
    import yaml
    from qda_quant.cli import main
    from qda_quant.output.record import RESOLVED

    steps = [{"gptq": {"impl": "torch"}}, {"scale_kd": KD}]
    cfg = {"model": str(tiny / "m"), "format": "w4a16", "devices": ["cpu"], "steps": steps,
           "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 32}}, "out": str(tiny / "a")}  # fmt: skip
    (tiny / "kd.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["build", str(tiny / "kd.yaml")]) == 0
    a = tiny / "a"
    res = yaml.safe_load((a / RESOLVED).read_text())
    assert [next(iter(s)) for s in res["steps"]] == ["gptq", "scale_kd"] and res["model"] == str(tiny / "m")
    rec = json.loads((a / "build.json").read_text())
    assert [(s["kind"], s["steps"]) for s in rec["stages"]] == [
        ("layers", ["gptq"]),
        ("checkpoint", ["scale_kd"]),
    ]
    assert all(s["seconds"] >= 0 for s in rec["stages"])
    assert main(["build", str(a / RESOLVED), "--set", f"out={tiny / 'b'}"]) == 0
    ta, tb = _tensors(a), _tensors(tiny / "b")
    assert ta.keys() == tb.keys() and all(torch.equal(ta[k], tb[k]) for k in ta)
    sa, sb = (P.Packed(d).state()[0] for d in (a, tiny / "b"))
    assert all(torch.equal(sa[k][1], sb[k][1]) for k in sa)
    # a checkpoint step on a packed source: its resolved.yaml has the checkpoint's format, no float stage
    assert main(["build", str(a / RESOLVED), "--set", f"model={a}", "--set", "format=null",
                 "--set", "steps=[{scale_kd: {steps: 1, batch_tokens: 64, head_chunk: 16, attn: eager}}]",
                 "--set", f"out={tiny / 'c'}"]) == 0  # fmt: skip
    rc = yaml.safe_load((tiny / "c" / RESOLVED).read_text())
    assert rc["format"] == P.Packed(a).fmt.name
    assert [(s["kind"]) for s in json.loads((tiny / "c" / "build.json").read_text())["stages"]] == [
        "checkpoint"
    ]


@pytest.fixture
def recorder(monkeypatch):
    """Records every entry into execution: the engine's run, staging, student/teacher loads, any step's apply."""
    import importlib

    import transformers

    B = importlib.import_module(
        "qda_quant.engine.build"
    )  # the module (`qda_quant.engine.build` is also the function)
    events: list[str] = []

    def rec(name, fn):
        def wrapped(*a, **k):
            events.append(name)
            return fn(*a, **k)

        return wrapped

    monkeypatch.setattr(B, "_run", rec("engine run", B._run))
    monkeypatch.setattr(P, "staged", rec("staged", P.staged))
    monkeypatch.setattr(P, "load_student", rec("student load", P.load_student))
    auto = transformers.AutoModelForCausalLM
    monkeypatch.setattr(auto, "from_pretrained", rec("weights load", auto.from_pretrained))
    for _, cls in REGISTRY.items():
        for m in ("apply_layer", "apply_model", "apply_checkpoint"):
            if m in vars(cls):
                monkeypatch.setattr(cls, m, rec(f"{cls.name}.{m}", vars(cls)[m]))
    return events


def test_plan_refuses_later_stages_before_anything_runs(tiny, recorder, capsys):
    """Every stage is checked before any runs: a bad scale_kd after gptq is refused by `qda plan` and `qda build`
    with no step applied, no weights loaded and no engine pass entered (SPEC-core: refusals before GPU work).

    Includes data-validation cases: a file whose only trace has an empty completion, a valid trace followed by a
    non-trace row, and a trace that max_len cuts below 2 scored tokens."""
    import transformers
    import yaml
    from qda_quant.cli import main

    other = tiny_qwen3(layers=1, hidden=128, inter=256)  # another vocabulary: 256 ids
    other.config.vocab_size = 256
    _save(transformers.AutoModelForCausalLM.from_config(other.config), tiny / "other")
    recorder.clear()
    base = {"model": str(tiny / "m"), "format": "w4a16", "devices": ["cpu"],
            "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 32}}, "out": str(tiny / "x")}  # fmt: skip
    bad = {
        "teacher": {"teacher": str(tiny / "nowhere")},
        "vocabulary": {"teacher": str(tiny / "other")},
        "trainable": {"trainable": ["gdn"]},
        "data": {"data": str(tiny / "none.jsonl")},
    }
    # Rows past the first, and the max_len cut, are checked at plan time by the rules training uses
    for name, rows in (
        ("only-empty-completion", [{"prompt_ids": [3], "completion_ids": []}]),
        ("valid-then-text-row", [{"prompt_ids": [3], "completion_ids": [4, 5]}, {"text": "invalid"}]),
        ("cut-by-max-len", [{"prompt_ids": [3, 4], "completion_ids": [5, 6]}]),
    ):
        path = tiny / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        bad[name] = {"data": str(path), **({"max_len": 3} if name == "cut-by-max-len" else {})}
    for what, kd in bad.items():
        cfg = {**base, "steps": [{"gptq": {}}, {"scale_kd": {**KD, **kd}}]}
        (tiny / "r.yaml").write_text(yaml.safe_dump(cfg))
        for verb in ("plan", "build"):
            assert main([verb, str(tiny / "r.yaml")]) == 2, (what, verb)
            assert "error: " in capsys.readouterr().err, (what, verb)  # the line, among the build's own
    # more cards than scale_kd takes is not a refusal any more: the build narrows the step (ORDER-1735). These
    # two refuse for reasons of their own, before anything runs.
    for what, steps, devices, code in (
        ("a card this host cannot see", [{"gptq": {}}, {"scale_kd": KD}], ["cuda:9"], 3),  # the machine
        ("order", [{"scale_kd": KD}, {"gptq": {}}], ["cpu"], 2),  # the input
    ):
        (tiny / "r.yaml").write_text(yaml.safe_dump({**base, "steps": steps, "devices": devices}))
        assert main(["build", str(tiny / "r.yaml")]) == code, what
    assert recorder == []
    # rows a generate stage still has to make are checked right after generation, by the whole recipe
    # (scale_kd included), before any layer work. Generation itself is a recorder writing four valid text rows.
    from qda_quant.calib.sources import Rows, SelfgenSource

    made = {"ready": False}

    def generate(source, model_dir):
        recorder.append("generation")
        made["ready"] = True
        return {"path": str(tiny / "generated.jsonl"), "sha256": "test"}

    with pytest.MonkeyPatch.context() as mp:  # the recorder's patches stay
        mp.setattr(SelfgenSource, "pending", lambda src, d: None if made["ready"] else "selfgen pending")
        mp.setattr(SelfgenSource, "generate", generate)
        mp.setattr(SelfgenSource, "read", lambda src, d: Rows([{"text": " ".join(["w3"] * 16)}] * 4, None))
        (tiny / "prompts.jsonl").write_text('{"id": 0, "prompt": "hi"}\n')
        gen = {**base, "calib": {"selfgen": {"prompts": str(tiny / "prompts.jsonl"), "samples": 4, "seqlen": 16}},
               "steps": [{"gptq": {"impl": "torch"}}, {"scale_kd": {**KD, "max_len": 2}}]}  # fmt: skip
        (tiny / "r.yaml").write_text(yaml.safe_dump(gen))
        assert main(["plan", str(tiny / "r.yaml")]) == 0 and recorder == []  # nothing generated at plan time
        assert main(["build", str(tiny / "r.yaml")]) == 2 and recorder == ["generation"]
        assert "max_len 2" in capsys.readouterr().err
        recorder.clear()
    # a valid staged recipe plans to both stages
    (tiny / "r.yaml").write_text(yaml.safe_dump({**base, "steps": [{"gptq": {}}, {"scale_kd": KD}]}))
    assert main(["plan", str(tiny / "r.yaml")]) == 0 and recorder == []
    out = capsys.readouterr().out
    assert "passes ['layers']" in out and "then checkpoint step scale_kd" in out


def test_plan_sizes_scale_kd_stages(tiny, capsys, monkeypatch):
    """`qda plan` and the build size a checkpoint stage, not only the float build: the teacher, the
    student as the card holds it, the block activations and the head log-probs, reported per stage; a stage whose
    floor is above the card's free memory is refused before stage 0, naming its largest term. The
    quantization-only recipe has no checkpoint stage and no stage peak."""
    import yaml
    from qda_quant.cli import main
    from qda_quant.engine.planning import compile_plan
    from qda_quant.step import StagePeak
    from qda_quant.steps.scale_kd import ScaleKD

    cpu = torch.device("cpu")
    q = _gptq(tiny, "w4a16-asym-g32")
    kd = {"seqs_per_forward": 2, "max_len": 32}
    plan = compile_plan(str(q), _recipe(tiny, REGISTRY["scale_kd"](**{**KD, **kd})), None, Target(), [cpu])
    assert plan.segments == [] and [s.name for s in plan.checkpoint_steps] == ["scale_kd"]
    assert len(plan.stage_peaks) == 1
    peak = plan.stage_peaks[0]
    terms = dict(peak.terms)
    assert peak.step == "scale_kd" and peak.device == "cpu" and not peak.over_free
    assert terms["teacher"] > terms["student"] > 0  # bf16 teacher against int8 codes plus bf16 floats
    assert terms["optimizer"] > 0 and terms["layer inputs"] > 0
    assert peak.bytes == sum(terms.values()) and peak.largest == max(peak.terms, key=lambda t: t[1])
    assert list(peak.terms) == sorted(peak.terms, key=lambda t: -t[1])
    only = compile_plan(
        str(tiny / "m"), _recipe(tiny, REGISTRY["gptq"](impl="torch")), Format.parse("w4a16"), Target(), [cpu]
    )
    assert only.stage_peaks == []

    # `qda plan` and the build print the stage line
    base = {
        "model": str(q),
        "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 32}},
        "devices": ["cpu"],
        "out": str(tiny / "x"),
    }
    (tiny / "r.yaml").write_text(yaml.safe_dump({**base, "steps": [{"scale_kd": {**KD, **kd}}]}))
    assert main(["plan", str(tiny / "r.yaml")]) == 0
    out = capsys.readouterr().out
    assert "then checkpoint step scale_kd" in out and "stage scale_kd: estimated at least" in out
    notes: list[str] = []
    build(str(q), _recipe(tiny, REGISTRY["scale_kd"](**{**KD, **kd, "steps": 1})), out_dir=tiny / "b",
          devices=["cpu"], log=notes.append)  # fmt: skip
    assert any(n.startswith("stage scale_kd: estimated at least") for n in notes)

    # a stage above the card's free memory refuses the plan, before stage 0, naming its largest term
    monkeypatch.setattr(
        ScaleKD,
        "checkpoint_peak",
        lambda self, ctx: StagePeak(self.name, "cuda:0", 2**30, (("teacher", 3 * 2**30),)),
    )
    assert main(["plan", str(tiny / "r.yaml")]) == 3  # a card too small for the stage is the machine
    err = capsys.readouterr().err
    assert "error: stage scale_kd:" in err and "largest term teacher 3.0 GiB" in err
    with pytest.raises(Refused, match="largest term teacher"):
        build(str(q), _recipe(tiny, REGISTRY["scale_kd"](**{**KD, **kd})), out_dir=tiny / "c",
              devices=["cpu"], log=lambda _: None)  # fmt: skip
    assert not (tiny / "c").exists()
    over = StagePeak("s", "cuda:0", 2 * 2**30, (("teacher", 3 * 2**30),))
    assert over.over_free and "teacher 3.0" in str(over)
    assert not StagePeak("s", "cpu", 1 << 50, (("teacher", 2**30),)).over_free


def test_scale_kd_stage_of_mistral_small_is_refused_on_a_free_80gb_card():
    """The numbers of the 24B run that ran out of memory (40 layers, hidden 5120, inter 32768, vocab
    131072, `head_chunk` 1024, 8 sequences of 2048 tokens in one forward) put the stage's floor above the
    78.7 GiB a free 80 GB card reports, with the teacher as its largest term; one sequence per forward fits.
    The forward pads the resolved `seqs_per_forward` sequences to the longest, so `batch_tokens` does not size
    it. The student is sized as the card holds it (int8 codes), about twice the 14.1 GiB written to disk."""
    from qda_quant.step import StagePeak
    from qda_quant.steps.scale_kd import kd_terms

    sizes = dict(
        layers=40,
        hidden=5120,
        inter=32768,
        vocab=131072,
        head_chunk=1024,
        teacher=int(43.47 * 2**30),  # the checkpoint's bf16 weights
        quantized=22_000_000_000,  # of 23.34B parameters: the rest stay bf16
        floats=1_340_000_000,  # the embedding and the head
        groups=173_670_000,
        symmetric=False,
    )
    free = int(78.7 * 2**30)
    peak = StagePeak("scale_kd", "cuda:0", free, kd_terms(tokens=8 * 2048, **sizes))
    assert peak.over_free and peak.largest[0] == "teacher"
    assert not StagePeak("scale_kd", "cuda:0", free, kd_terms(tokens=2048, **sizes)).over_free


def test_two_card_terms_size_the_four_moe_zoo_builds():
    """REQ-M4-2: the four MoE zoo builds, sized from their configs (one sequence of 2048 tokens per forward).
    One card holds none of them; with the teacher on a second card each side fits a free 80 GB card, the
    teacher's alone with the most room taken. Sizes are the measured terms, rounded (docs: status table)."""
    from qda_quant.step import StagePeak
    from qda_quant.steps.scale_kd import kd_teacher_terms, kd_terms

    # name: layers, hidden, inter, vocab, params, quantized, floats, groups
    zoo = {
        "qwen3-30b-a3b": (48, 2048, 768, 151936, 30_532e6, 29_890e6, 642e6, 232.6e6),
        "qwen3-coder-30b-a3b": (48, 2048, 768, 151936, 30_532e6, 29_890e6, 642e6, 232.6e6),
        "gemma-4-26b-a4b": (30, 2816, 1792, 262144, 25_820e6, 24_550e6, 1_270e6, 384.5e6),
        "glm-4.7-flash": (47, 2048, 1536, 151936, 29_940e6, 28_880e6, 1_060e6, 223.7e6),
    }
    free = int(78.7 * 2**30)  # the free memory of an 80 GB card
    for name, (layers, hidden, inter, vocab, params, quantized, floats, groups) in zoo.items():
        sizes = dict(
            layers=layers, tokens=2048, hidden=hidden, inter=inter, vocab=vocab, head_chunk=1024,
            quantized=int(quantized), floats=int(floats), groups=int(groups), symmetric=False,
        )  # fmt: skip
        one = StagePeak("scale_kd", "cuda:0", free, kd_terms(teacher=int(params) * 2, **sizes))
        assert one.over_free and one.largest[0] == "teacher", name
        student = StagePeak("scale_kd", "cuda:0", free, kd_terms(teacher=0, **sizes))
        teacher = StagePeak(
            "scale_kd", "cuda:1",
            free,
            kd_teacher_terms(teacher=int(params) * 2, tokens=2048, hidden=hidden, inter=inter, vocab=vocab,
                             head_chunk=1024),
        )  # fmt: skip
        assert not student.over_free and not teacher.over_free, name
        assert teacher.largest[0] == "teacher" and teacher.bytes > student.bytes, name


def test_the_plan_sizes_the_student_as_load_student_holds_it(tiny_moe):
    """`load_student` builds the class on `meta` and materializes only what stays float, so the load's peak on a
    card is the student's own tensors: `kd_terms`' student term sizes exactly those (int8 codes, bf16 floats,
    bf16 scales, fp32 deltas, int8 zero points, and the model's own buffers). The tiny MoE carries the same
    terms as a zoo build; sizing it from a meta model is what refuses an unloadable build before stage 0."""
    from qda_quant.engine.planning import compile_plan

    q = _gptq(tiny_moe, "w4a16-asym-g32")
    cpu = torch.device("cpu")
    plan = compile_plan(str(q), _recipe(tiny_moe, REGISTRY["scale_kd"](**KD)), None, Target(), [cpu, cpu])
    student = dict(plan.stage_peaks[0].terms)["student"]
    ck = P.Packed(q)
    scales, floats = ck.state(lambda _: None)
    m = P.load_student(ck, scales, floats, cpu, "eager")
    held = sum(p.numel() * p.element_size() for p in m.parameters())
    held += sum(b.numel() * b.element_size() for b in m.buffers())
    assert held == student
    # a bank's experts share one trained scale, so the optimizer's step scalars are per served module
    groups, _, _ = REGISTRY["scale_kd"](**KD)._params(m, P.served_modules(m))
    trained = sum(p.numel() for g in groups for p in g["params"])
    tensors = sum(1 for g in groups for _ in g["params"])
    assert dict(plan.stage_peaks[0].terms)["optimizer"] == 12 * trained + 4 * tensors


@pytest.mark.parametrize("moe", [False, True])
def test_the_plan_covers_the_tied_head_load_student_materializes(tmp_path, moe, monkeypatch):
    """With `tie_word_embeddings`, `load_student` ties the head while it is still on `meta`, so
    `_materialize` allocates it once and the load's peak is the student's own tensors, which the plan's student
    term sizes. Tying only after materializing would add vocab x hidden bf16 temporarily, above the estimate."""
    from qda_quant.engine.planning import compile_plan
    from qda_quant.steps.scale_kd import ScaleKD

    model = tiny_qwen3_moe() if moe else tiny_qwen3(layers=2, hidden=128, inter=256)
    model.config.tie_word_embeddings = True
    model.tie_weights()
    sha = _save(model, tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [[3, 4, 5, 6] * 8] * 4, sha)
    q = _gptq(tmp_path, "w4a16-sym-g32" if moe else "w4a16-sym-g128")
    cpu = torch.device("cpu")

    peak, materialize = [], P._materialize

    def capture(m, device):
        materialize(m, device)
        peak.append(_storage_bytes(m))

    monkeypatch.setattr(P, "_materialize", capture)
    ck = P.Packed(q)
    scales, floats = ck.state(lambda _: None)
    student = P.load_student(ck, scales, floats, cpu, "eager")

    assert student.get_input_embeddings().weight is student.get_output_embeddings().weight
    plan = compile_plan(str(q), _recipe(tmp_path, ScaleKD(**KD)), None, Target(), [cpu, cpu])
    assert peak[0] <= dict(plan.stage_peaks[0].terms)["student"]
    assert peak[0] == _storage_bytes(student)  # the peak is the student itself: no temporary untied head


def test_plan_checks_both_cards_of_a_two_card_scale_kd(tiny, monkeypatch):
    """A checkpoint step may spread its stage over cards and size each of them (`Step.checkpoint_peak`
    returns one `StagePeak` per card). `compile_plan` lists them all and refuses when either card is over its
    free memory, naming that card; a build on more cards than the step takes narrows it to its own subset."""
    from qda_quant.engine.planning import compile_plan
    from qda_quant.step import StagePeak
    from qda_quant.steps.scale_kd import ScaleKD

    q = _gptq(tiny, "w4a16-asym-g32")
    kd = REGISTRY["scale_kd"](**{**KD, "seqs_per_forward": 2, "max_len": 32})
    cpu = torch.device("cpu")

    def two_cards(self, ctx):
        assert len(ctx.devices) == 2
        return (
            StagePeak(self.name, "cuda:0", 2**40, (("student", 2**38),)),
            StagePeak(self.name, "cuda:1", 2**40, (("teacher", 2**39),)),
        )

    monkeypatch.setattr(ScaleKD, "checkpoint_peak", two_cards)
    plan = compile_plan(str(q), _recipe(tiny, kd), None, Target(), [cpu, cpu])
    assert [p.device for p in plan.stage_peaks] == ["cuda:0", "cuda:1"]
    assert [p.largest[0] for p in plan.stage_peaks] == ["student", "teacher"]

    def teacher_does_not_fit(self, ctx):
        return (
            StagePeak(self.name, "cuda:0", 2**40, (("student", 2**38),)),
            StagePeak(self.name, "cuda:1", 2**38, (("teacher", 2**39),)),
        )

    monkeypatch.setattr(ScaleKD, "checkpoint_peak", teacher_does_not_fit)
    with pytest.raises(Refused, match="free on cuda:1; largest term teacher"):
        compile_plan(str(q), _recipe(tiny, kd), None, Target(), [cpu, cpu])

    monkeypatch.undo()
    # ORDER-1735: a build planned on more cards than the step takes narrows the step to its own subset rather
    # than refusing, so the acceptance recipe runs with no `--set devices`. The build keeps its own list for the
    # float stage; only the checkpoint stage is narrowed, and its peaks are priced on the cards it will use.
    plan = compile_plan(str(q), _recipe(tiny, kd), None, Target(), [cpu, cpu, cpu])
    assert plan.devices == [cpu, cpu, cpu]
    assert plan.stage_devices[0] == [cpu]
    assert [p.device for p in plan.stage_peaks] == ["cpu"]  # one card holds it; the teacher shares it


def test_stage_devices_narrow_a_capped_step_to_its_largest_free_cards(monkeypatch):
    """`_stage_devices` gives a capped step its own subset of the build's cards: the fewest cards, most free
    memory first (ties in the build's own order), whose own priced peak fits; the cap is the ceiling. Within
    the cap nothing is replaced, and a step with no cap (the base's None) keeps the build's list as it was."""
    from qda_quant.engine import residency
    from qda_quant.engine.memory import _stage_devices
    from qda_quant.engine.packed import CheckpointPlan
    from qda_quant.step import StagePeak

    kd = REGISTRY["scale_kd"]
    assert kd.max_devices == 2 and Step.max_devices is None
    four = [torch.device(f"cuda:{i}") for i in range(4)]
    ctx = CheckpointPlan("", None, None, None, None, None, None, None, None, four, None)  # type: ignore[arg-type]
    monkeypatch.setattr(residency, "free_bytes", lambda d: {0: 10, 1: 30, 2: 20, 3: 30}[d.index])

    class Stub(Step):
        max_devices = 2
        need: ClassVar[int] = 2**30  # bytes this stage takes on each card it runs on

        def checkpoint_peak(self: Stub, ctx: CheckpointPlan) -> tuple[StagePeak, ...]:
            return tuple(StagePeak("stub", str(d), 2**40, (("stage", self.need),)) for d in ctx.devices)

    class Big(Stub):
        need: ClassVar[int] = 2**41  # over one card's free memory, within two

    one, peak = _stage_devices(Stub(), ctx, four)
    assert one == [torch.device("cuda:1")]  # one holds it: the freest, and its peak is the priced one
    assert [p.device for p in peak] == ["cuda:1"]
    two, peaks = _stage_devices(Big(), ctx, four)
    assert two == [torch.device("cuda:1"), torch.device("cuda:3")]
    assert [p.device for p in peaks] == ["cuda:1", "cuda:3"]
    pair = four[:2]
    assert _stage_devices(Stub(), ctx, pair)[0] is pair  # within the cap: not reordered, not replaced
    assert _stage_devices(Step(), ctx, four)[0] is four  # no cap: the build's own list


def test_optimizer_term_counts_only_the_trained_scales(tiny):
    """Only the scale deltas train (the zero points stay int8), so 16 bytes a trained
    scale, asymmetric or not, not the 24 the asymmetric case was charged. The fp32 delta is the served parameter
    `load_student` puts on the card, so the student term sizes it (4); the optimizer keeps its gradient and
    Adam's two moments (12) and one fp32 step scalar per trained tensor."""
    from qda_quant.engine.planning import compile_plan
    from qda_quant.steps.scale_kd import kd_terms

    q = _gptq(tiny, "w4a16-asym-g32")
    plan = compile_plan(
        str(q), _recipe(tiny, REGISTRY["scale_kd"](**KD)), None, Target(), [torch.device("cpu")]
    )
    ck = P.Packed(q)
    scales, floats = ck.state(lambda _: None)
    groups = sum(d.numel() for _, d in scales.values())  # one trained scale per group; zero points frozen
    m = P.load_student(ck, scales, floats, torch.device("cpu"), "eager")
    tensors = len(P.served_modules(m))  # one step scalar per trained scale tensor
    terms = dict(plan.stage_peaks[0].terms)
    assert terms["optimizer"] == 12 * groups + 4 * tensors
    # the same per scale on a symmetric format, which stores no zero point the optimizer would ever touch
    for symmetric in (True, False):
        sized = kd_terms(
            layers=1, tokens=1, hidden=1, inter=1, vocab=1, head_chunk=1, teacher=0, quantized=0, floats=0,
            groups=7, symmetric=symmetric, buffers=64, tensors=3,
        )  # fmt: skip
        assert dict(sized)["optimizer"] == 12 * 7 + 4 * 3
        # scales and fp32 deltas, the int8 zero point an asymmetric format stores, and the model's buffers
        assert dict(sized)["student"] == 7 * (2 + 4) + (0 if symmetric else 7) + 64


@pytest.mark.parametrize("head", [None, "body", "w4a16-sym-g64", "w4a16-asym-g32"])
@pytest.mark.parametrize("fmt", ["w4a16-asym-g128", "w4a16-sym-g128"])
def test_the_plan_sizes_a_head_in_its_own_format_and_the_model_buffers(tiny, fmt, head):
    """Tiny body/head combinations. A head a model step packs is sized by its own group count
    and symmetry (not the body's), and the model's own buffers (`rotary inv_freq`, 64 B here) are on the card as
    `load_student` computes them. Student and optimizer are at or above the tensors the student really holds."""
    from qda_quant.engine.planning import compile_plan

    steps = [REGISTRY["gptq"](impl="torch")]
    if head is not None:
        steps.insert(0, REGISTRY["head"](format=fmt if head == "body" else head))
    out = tiny / f"q537-{fmt}-{head}"
    build(
        str(tiny / "m"), _recipe(tiny, *steps), format=fmt, out_dir=out, devices=["cpu"], log=lambda _: None
    )
    cpu = torch.device("cpu")
    kd = REGISTRY["scale_kd"](**KD)
    plan = compile_plan(str(out), _recipe(tiny, kd), None, Target(), [cpu])
    terms = dict(plan.stage_peaks[0].terms)

    ck = P.Packed(out)
    scales, floats = ck.state(lambda _: None)
    m = P.load_student(ck, scales, floats, cpu, "eager")
    held = sum(p.numel() * p.element_size() for p in m.parameters())
    held += sum(b.numel() * b.element_size() for b in m.buffers())
    assert terms["student"] >= held

    groups, _, _ = kd._params(m, P.served_modules(m))
    opt = torch.optim.Adam(groups, lr=1e-2)
    for g in groups:
        for p in g["params"]:
            p.grad = torch.ones_like(p)
    opt.step()
    trained = sum(p.grad.numel() * p.grad.element_size() for g in groups for p in g["params"])
    trained += sum(
        v.numel() * v.element_size()
        for state in opt.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    )
    assert terms["optimizer"] >= trained


@pytest.mark.parametrize(
    "field,trainable", [("lr", ["norms"]), ("norm_lr", ["scales"]), ("gdn_lr", ["scales"])]
)
def test_plan_refuses_learning_rates_for_untrained_groups(tiny, recorder, field, trainable):
    """Unused rates refuse before execution; serialized defaults still allow training a subset."""
    from qda_quant.config import BuildConfig

    kd = REGISTRY["scale_kd"](trainable=trainable)
    base = {
        "model": str(tiny / "m"),
        "format": "w4a16",
        "devices": ["cpu"],
        "calib": {"file": {"path": str(tiny / "c.jsonl"), "samples": 4, "seqlen": 32}},
        "out": str(tiny / "unused-rate"),
    }
    for value in (0.01, 0.02):
        settings = {**kd.model_dump(), field: value}
        cfg = BuildConfig(**base, steps=[{"gptq": {}}, {"scale_kd": settings}])
        for check in (cfg.plan, cfg.build):
            with pytest.raises(Refused, match=rf"scale_kd\.{field} has no effect.*trainable"):
                check()
        assert recorder == []
    cfg = BuildConfig(**base, steps=[{"gptq": {}}, {"scale_kd": kd.model_dump()}])
    cfg.plan()
    assert recorder == []


def test_checkpoint_stages_use_default_attention_backends(tiny, monkeypatch):
    """Like test_build_uses_default_attention_backends_whatever_the_caller_set, for checkpoint stages: a staged
    build, and a scale_kd build straight from a packed source, train and load the student under torch's default
    attention backends whatever the caller set (vLLM turns cuDNN SDP off), and give the caller's back after."""
    from qda_quant.engine.execute import ATTENTION_BACKENDS
    from qda_quant.steps.scale_kd import ScaleKD

    seen: list[bool] = []
    orig_load = P.load_student

    def probe(*a, **k):
        seen.append(torch.backends.cuda.cudnn_sdp_enabled())
        return orig_load(*a, **k)

    monkeypatch.setattr(P, "load_student", probe)
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        staged = build(str(tiny / "m"), _recipe(tiny, REGISTRY["gptq"](impl="torch"), ScaleKD(**KD)),
                       format="w4a16", out_dir=tiny / "one", devices=["cpu"], log=lambda _: None)  # fmt: skip
        direct = _kd(tiny, staged, "two", steps=1)  # packed source: checkpoint stage only
        assert torch.backends.cuda.cudnn_sdp_enabled() is False  # restored
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)
    assert seen == [True, True]
    for out in (staged, direct):
        assert json.loads((out / "build.json").read_text())["attention_backends"] == list(ATTENTION_BACKENDS)


def test_failed_staged_build_leaves_no_output(tiny, monkeypatch):
    """A failure while the records are written leaves nothing at `out` (nothing a reader accepts), and what
    was written in its staging directory; a checkpoint already at `out` is refused, not replaced."""
    from qda_quant.config import BuildConfig
    from qda_quant.step import Refused

    q = _gptq(tiny)
    first = _kd(tiny, q, "kd", steps=1)
    before = _tensors(first)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(BuildConfig, "write_resolved", boom)
    with pytest.raises(OSError, match="disk full"):
        _kd(tiny, q, "fresh", steps=1)
    assert not (tiny / "fresh").exists() and not P.is_packed(tiny / "fresh")
    left = list(tiny.glob(".fresh.partial-*"))
    assert len(left) == 1 and (left[0] / "build.json").is_file()
    monkeypatch.undo()
    with pytest.raises(Refused, match="not an empty directory"):
        _kd(tiny, q, "kd", steps=2)
    after = _tensors(first)
    assert after.keys() == before.keys() and all(torch.equal(after[k], before[k]) for k in after)
    # keep_intermediate on a packed source with one checkpoint step (no intermediate to keep)
    r = _recipe(tiny, REGISTRY["scale_kd"](**{**KD, "steps": 1}))
    direct = build(
        str(q), r, out_dir=tiny / "direct", devices=["cpu"], keep_intermediate=True, log=lambda _: None
    )
    assert P.is_packed(direct) and (direct / "resolved.yaml").is_file() and not (direct / P.STAGES).exists()
    assert not list(tiny.glob(".direct.partial-*"))


def test_chain_keeps_teacher_and_state(tiny):
    """Round 2 starts from round 1's exact fp32 state and keeps the original teacher."""
    r1 = _kd(tiny, _gptq(tiny), "r1")
    r2 = _kd(tiny, r1, "r2", steps=0)  # no step: the state passes through unchanged
    s1, _ = P.Packed(r1).state()
    s2, _ = P.Packed(r2).state()
    assert all(torch.equal(s1[k][1], s2[k][1]) and torch.equal(s1[k][0], s2[k][0]) for k in s1)
    rec = json.loads((r2 / "build.json").read_text())
    assert rec["teacher"] == str(tiny / "m") and len(rec["lineage"]) == 2
    r3 = _kd(tiny, r1, "r3")
    assert json.loads((r3 / "build.json").read_text())["teacher"] == str(tiny / "m")


def test_refusals(tiny):
    q = _gptq(tiny)
    with pytest.raises(Refused, match="already packed"):
        build(
            str(q), _recipe(tiny, REGISTRY["gptq"]()), out_dir=tiny / "x", devices=["cpu"], log=lambda _: None
        )
    with pytest.raises(Refused, match="is w4a16"):
        build(str(q), _recipe(tiny, REGISTRY["scale_kd"](**KD)), format="w4a8", out_dir=tiny / "x",
              devices=["cpu"], log=lambda _: None)  # fmt: skip
    with pytest.raises(Refused, match="scale_kd.teacher"):
        _kd(tiny, q, "x", teacher=str(tiny / "nowhere"))


def test_norms_and_gdn_qwen3_5(tmp_path):
    """Gated DeltaNet hybrid: norms and gdn tensors train (re-implemented, no parity claim); codes stay."""
    import transformers

    cfg = config("qwen3_5")
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    sha = _save(m, tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (40,), generator=g).tolist() for _ in range(6)], sha
    )
    q = _gptq(tmp_path)
    out = _kd(tmp_path, q, "kd", trainable=["scales", "norms", "gdn"], norm_lr=1e-2, gdn_lr=1e-2)
    a, b = _tensors(q), _tensors(out)
    trained = json.loads((out / "build.json").read_text())["trained"]
    for part in ("A_log", "dt_bias", "conv1d.weight", "in_proj_a.weight", "input_layernorm.weight", "q_norm"):
        assert any(part in k for k in trained), part
    assert all(not torch.equal(a[k], b[k]) for k in trained if "A_log" in k or "layernorm" in k)
    assert all(torch.equal(a[k], b[k]) for k in a if k.endswith(("weight_packed", "weight_zero_point")))
    _, fl = P.Packed(out).state()
    assert set(fl) == set(trained)


@pytest.fixture
def tiny_moe(tmp_path):
    from ..engine_util import tiny_qwen3_moe

    sha = _save(tiny_qwen3_moe(), tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (40,), generator=g).tolist() for _ in range(6)], sha
    )
    return tmp_path


def test_served_experts_equal_eager_experts(tiny_moe):
    """`ServedExperts` = transformers' eager experts forward on the served weights; its gradients = autograd's."""
    ck = P.Packed(_gptq(tiny_moe, "w4a16-asym-g32"))
    scales, floats = ck.state()
    stu = P.load_student(ck, scales, floats, torch.device("cpu"), "eager")
    ex = stu.model.layers[0].mlp.experts
    assert isinstance(ex, S.ServedExperts) and ex.gate_up.delta.shape == (8, 64, 4)
    with torch.no_grad():
        for b in (ex.gate_up, ex.down):
            b.delta.normal_(0, 0.05)
    for b in (ex.gate_up, ex.down):
        b.delta.requires_grad_(True)
    # the reference: the stock module holding the served weights, as plain autograd of delta
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    cfg = stu.config
    cfg._experts_implementation = "eager"
    ref = Qwen3MoeExperts(cfg)
    d_gu = ex.gate_up.delta.detach().clone().requires_grad_(True)
    d_dn = ex.down.delta.detach().clone().requires_grad_(True)

    def bank_w(bank, d):  # bf16-rounded served weight, straight-through
        w = torch.stack([bank.expert(e).weight(d[e]) for e in range(bank.num_experts)])
        return w + (w.to(torch.bfloat16).float() - w).detach()

    torch.manual_seed(1)
    x = torch.randn(12, cfg.hidden_size, dtype=torch.bfloat16)
    idx = torch.randint(0, 8, (12, 2))
    idx[:, 1] = (idx[:, 0] + 1) % 8
    wts = torch.rand(12, 2, dtype=torch.bfloat16)
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    y = ex(x1, idx, wts)
    del ref.gate_up_proj, ref.down_proj
    ref.gate_up_proj = bank_w(ex.gate_up, d_gu).to(torch.bfloat16)
    ref.down_proj = bank_w(ex.down, d_dn).to(torch.bfloat16)
    with torch.no_grad():
        y_ref = ref(x2.detach(), idx, wts)
    assert torch.equal(y, y_ref)  # same bf16 weights, same loop
    gy = torch.randn_like(y)
    y.backward(gy)
    ref.gate_up_proj, ref.down_proj = bank_w(ex.gate_up, d_gu).float(), bank_w(ex.down, d_dn).float()
    y32 = ref(x2.float(), idx, wts.float())
    y32.backward(gy.float())
    for got, want in ((ex.gate_up.delta.grad, d_gu.grad), (ex.down.delta.grad, d_dn.grad)):
        assert (got - want).norm() < 2e-2 * want.norm()  # bf16 forward vs the fp32 reference graph
    assert (x1.grad.float() - x2.grad).norm() < 2e-2 * x2.grad.norm()


@pytest.mark.parametrize(
    "fmt, device",
    [
        ("w4a16-asym-g32", "cpu"),
        ("w4a16-sym-g32", "cpu"),
        pytest.param("w4a16-asym-g32", "cuda", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.usefixtures("cpu_build")
def test_grouped_experts_equal_loop(tiny_moe, fmt, device):
    """The grouped fast path = the per-expert loop: output, dx and d(delta) (CONTRIBUTING.md: fast path, slow reference).

    On the CPU the grouped matmuls run per expert; on a card (``gpu``) they are torch's ``grouped_mm``.
    """
    ck = P.Packed(_gptq(tiny_moe, fmt))
    scales, floats = ck.state()
    dev = torch.device(device)
    stu = P.load_student(ck, scales, floats, dev, "eager")
    ex = stu.model.layers[1].mlp.experts
    with torch.no_grad():
        for b in (ex.gate_up, ex.down):
            b.delta.normal_(0, 0.05)
    torch.manual_seed(2)
    x = torch.randn(16, stu.config.hidden_size, dtype=torch.bfloat16).to(dev)
    idx = torch.stack([torch.randperm(8)[:2] for _ in range(16)]).to(dev)  # expert 7 may get no token
    wts = torch.rand(16, 2, dtype=torch.bfloat16).to(dev)
    gy = torch.randn(16, stu.config.hidden_size, dtype=torch.bfloat16).to(dev)
    got = {}
    for grouped in (False, True):
        ex.grouped = grouped
        for b in (ex.gate_up, ex.down):
            b.delta.grad = None
            b.delta.requires_grad_(True)
        xi = x.clone().requires_grad_(True)
        y = ex(xi, idx, wts)
        y.backward(gy)
        got[grouped] = (y.detach(), xi.grad, ex.gate_up.delta.grad.clone(), ex.down.delta.grad.clone())
    for a, b in zip(got[False], got[True], strict=True):
        assert (a.float() - b.float()).norm() <= 1e-2 * a.float().norm()  # bf16 sums in another order
    # bf16 grouped_mm on a card (GH200): 211 of 2048 entries off by up to 0.5% of the largest, norm 1e-2 holds
    atol = 1e-6 if device == "cpu" else 1e-2 * float(got[False][2].abs().max())
    assert torch.allclose(got[False][2], got[True][2], rtol=1e-2, atol=atol)


def _tiny_model(kind: str) -> torch.nn.Module:
    import transformers

    from ..engine_util import tiny_qwen3_moe

    if kind == "moe":
        return tiny_qwen3_moe(hidden=64)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config(kind), dtype=torch.float32)


def _grad_mismatches(got: dict, want: dict, served: dict, tol: float) -> list[tuple[str, int, float]]:
    """(module, expert, relative error) of every unit whose d(delta) is off by more than `tol`: each Linear, and
    each expert of a bank on its own (one wrong expert must not hide in a whole-model norm)."""
    out = []
    for name, w in want.items():
        g = got[name]
        units = range(w.shape[0]) if isinstance(served[name], S.ServedBank) else [None]
        for e in units:
            a, b = (g, w) if e is None else (g[e], w[e])
            err = float((a - b).norm() / b.norm())
            if not err <= tol:
                out.append((name, -1 if e is None else e, err))
    return out


@pytest.mark.parametrize(
    "kind, attn", [("moe", "sdpa"), ("moe", "eager"), ("qwen3", "sdpa"), ("qwen3_5", "sdpa")]
)
def test_padded_forward_equals_one_per_forward(tmp_path, kind, attn):
    """Rows of different lengths in one right-padded forward = one forward each, on default rotary: dense, MoE
    and Gated DeltaNet hybrid. KL sum, and d(delta) of every Linear and of every expert of every bank.

    Tolerance: teacher and student run in bf16 and a group changes the row count of every matmul, so sums run in
    another order: KL within 1e-3, each unit's d(delta) within 2e-2 relative (measured: at most 6e-5 and 6.4e-3).
    Rows are long (90 to 160 tokens) so every expert gets enough rows that its gradient is not dominated by
    rounding. Left padding, scoring the pads, or one expert's zeroed gradient fail it.
    """
    from qda_quant.steps.scale_kd import kl_sequences
    from transformers import AutoModelForCausalLM

    sha = _save(_tiny_model(kind), tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    write_ids(
        tmp_path / "c.jsonl", [torch.randint(0, 128, (40,), generator=g).tolist() for _ in range(6)], sha
    )
    # Preserve the measured rounding fixture: its hidden width 64 needs Humming rather than Marlin.
    target = "sm90@0.33.0:humming" if kind == "moe" else None
    ck = P.Packed(_gptq(tmp_path, "w4a16-asym-g32", target=target))
    scales, floats = ck.state()
    cpu = torch.device("cpu")
    stu = P.load_student(ck, scales, floats, cpu, attn)
    tea = AutoModelForCausalLM.from_pretrained(tmp_path / "m", dtype=torch.bfloat16, attn_implementation=attn)
    served = P.served_modules(stu)
    banks = [n for n, m in served.items() if isinstance(m, S.ServedBank)]
    assert (kind == "moe") == bool(banks)
    for m in served.values():
        with torch.no_grad():
            m.delta.normal_(0, 0.05)
        m.delta.requires_grad_(True)
    g = torch.Generator().manual_seed(3)
    rows = [
        (torch.randint(0, 128, (n,), generator=g).tolist(), p) for n, p in ((160, 9), (90, 1), (120, 100))
    ]
    got = []
    for groups in ([rows], [[r] for r in rows]):
        for m in served.values():
            m.delta.grad = None
        kl = sum(kl_sequences(stu, tea, grp, 16, 100, cpu, backward=True) for grp in groups)
        got.append((kl, {n: m.delta.grad.clone() for n, m in served.items()}))
    (kl_b, gd_b), (kl_1, gd_1) = got
    assert abs(kl_b - kl_1) <= 1e-3 * kl_1
    assert _grad_mismatches(gd_b, gd_1, served, 2e-2) == []
    if banks:  # the check sees one expert's gradient gone, which a whole-model norm would not
        bad = dict(gd_b)
        bad[banks[0]] = bad[banks[0]].clone()
        bad[banks[0]][6] = 0
        assert [(n, e) for n, e, _ in _grad_mismatches(bad, gd_1, served, 2e-2)] == [(banks[0], 6)]


ROPE = {  # original context 16, extended 128: rows of 80/31/13 tokens cross it, as a padded group does
    "llama3": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16},
    "yarn": {"factor": 8.0, "original_max_position_embeddings": 16},
    "longrope": {"factor": 8.0, "original_max_position_embeddings": 16, "short_factor": [1.0] * 16,
                 "long_factor": [8.0] * 16},
}  # fmt: skip


@pytest.mark.parametrize("rope", sorted(ROPE))
def test_padded_forward_and_rotary_type(tmp_path, rope):
    """Position-fixed rotary types: a padded group = one forward per row, and an unset `seqs_per_forward` resolves to
    8. LongRoPE, which picks its frequencies from the padded length, resolves to 1 and refuses more at plan time
    (unpadded 0.00709 vs padded 0.00755 KL, d(delta) 46% off)."""
    from qda_quant.engine.planning import compile_plan
    from qda_quant.steps.scale_kd import kl_sequences
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = config("llama")
    cfg.max_position_embeddings = 128
    cfg.rope_parameters = {"rope_type": rope, "rope_theta": 10000.0, **ROPE[rope]}
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    with torch.no_grad():  # larger q/k make attention sensitive to position
        for n, p in m.named_parameters():
            if ".q_proj." in n or ".k_proj." in n:
                p.mul_(3)
    sha = _save(m, tmp_path / "m")
    g = torch.Generator().manual_seed(13)
    rows = [(torch.randint(0, 128, (n,), generator=g).tolist(), p) for n, p in ((80, 78), (31, 29), (13, 1))]
    write_ids(tmp_path / "c.jsonl", [r for r, _ in rows], sha)

    def rec(*steps):
        return Recipe(
            steps=list(steps), calib=FileSource(path=str(tmp_path / "c.jsonl"), samples=3, seqlen=80)
        )

    q = build(str(tmp_path / "m"), rec(REGISTRY["rtn"]()), format="w4a16-asym-g32", out_dir=tmp_path / "q",
              devices=["cpu"], log=lambda _: None)  # fmt: skip
    cpu = torch.device("cpu")

    def plan(spf, src=q, **kw):
        kd = REGISTRY["scale_kd"](
            steps=1, max_len=80, batch_tokens=124, seqs_per_forward=spf, attn="eager", **kw
        )
        return compile_plan(str(src), rec(kd), None, Target(), [cpu])

    # unset resolves at plan time, into the recipe (so resolved.yaml): 8 where batching is exact, else 1
    assert plan(None).checkpoint_steps[0].seqs_per_forward == (1 if rope == "longrope" else 8)
    assert plan(1).checkpoint_steps[0].seqs_per_forward == 1  # a given value wins
    if rope == "longrope":
        with pytest.raises(Refused, match="rotary type 'longrope' \\(of the student\\)"):
            plan(3)
        # the teacher runs padded groups too: a default-rotary student with this teacher (KL +36%, d(delta) +74%)
        import shutil

        shutil.copytree(q, tmp_path / "s")
        c = json.loads((tmp_path / "s" / "config.json").read_text())
        c["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000.0}
        (tmp_path / "s" / "config.json").write_text(json.dumps(c))
        mixed = {"src": tmp_path / "s", "teacher": str(tmp_path / "m")}
        assert plan(None, **mixed).checkpoint_steps[0].seqs_per_forward == 1
        with pytest.raises(Refused, match="rotary type 'longrope' \\(of the teacher\\)"):
            plan(3, **mixed)
        return
    plan(3)
    with pytest.raises(Refused, match="at most 8"):
        plan(9)
    ck = P.Packed(q)
    tea = AutoModelForCausalLM.from_pretrained(
        tmp_path / "m", dtype=torch.bfloat16, attn_implementation="eager"
    )
    got = []
    for groups in ([rows], [[r] for r in rows]):
        scales, floats = ck.state(lambda _: None)
        stu = P.load_student(ck, scales, floats, cpu, "eager")
        served = list(P.served_modules(stu).values())
        for b in served:
            b.delta.requires_grad_(True)
        kl = sum(kl_sequences(stu, tea, grp, 16, 16, cpu, backward=True) for grp in groups)
        got.append((kl, torch.cat([b.delta.grad.flatten() for b in served])))
    (kl_b, g_b), (kl_1, g_1) = got
    assert abs(kl_b - kl_1) <= 1e-3 * kl_1
    assert (g_b - g_1).norm() <= 1e-2 * g_1.norm()


def test_scale_kd_moe_trains_experts_codes_identical(tiny_moe):
    """Fused-expert MoE: expert scales train, codes and zero points stay byte-identical, the state round-trips."""
    q = _gptq(tiny_moe, "w4a16-asym-g32")
    out = _kd(tiny_moe, q, "kd")
    a, b = _tensors(q), _tensors(out)
    assert a.keys() == b.keys()
    diff = {k for k in a if not torch.equal(a[k], b[k])}
    assert diff and all(k.endswith(".weight_scale") for k in diff)
    assert any(".experts.3.up_proj." in k for k in diff) and any(".experts.5.down_proj." in k for k in diff)
    import shutil

    foreign = tiny_moe / "foreign"  # a parent not written by qda (no build.json) gets both serve plans
    shutil.copytree(q, foreign)
    (foreign / "build.json").unlink()
    f_out = _kd(tiny_moe, foreign, "kdf", steps=1, teacher=str(tiny_moe / "m"))
    serve = json.loads((f_out / "build.json").read_text())["serve"]
    assert set(serve["kernels"]) == {"dense", "moe"} and serve["kernels"]["moe"]
    s_out, _ = P.Packed(out).state()  # the state gives the stored scales (checked inside), one entry per stem
    assert set(s_out) == set(P.Packed(q).stems)
    g = _kd(tiny_moe, q, "kdg", experts="grouped")  # the fast path trains the same way
    logs = [json.loads((d / "build.json").read_text())["history"][-1]["log"] for d in (out, g)]
    assert abs(logs[0][0]["kl"] - logs[1][0]["kl"]) <= 1e-3 * logs[0][0]["kl"]
    r2 = _kd(tiny_moe, out, "r2", steps=0)
    s2, _ = P.Packed(r2).state()
    assert all(torch.equal(s_out[k][1], s2[k][1]) for k in s_out)


@pytest.mark.parametrize("stored_per_expert", [False, True])
def test_scale_kd_builds_with_a_whole_float_routed_bank(tmp_path, stored_per_expert):
    """A bank `ignore` keeps whole stays bf16 through GPTQ and a scales-only KD step.

    The student load refused such a checkpoint: a bank with no packed per-expert `down_proj` looked unsupported.
    It now keeps the class's own bank module and restores its two 3-D parameters as the checkpoint stores them,
    fused or one module per expert, while the other banks stay `ServedExperts` and train their scales. The
    restored weights are the source's, byte for byte, as stock transformers stacks them.
    """
    from qda_quant import FileSource, Recipe, build
    from transformers import AutoModelForCausalLM

    from ..engine_util import save_tiny_tokenizer, tiny_qwen3_moe, write_ids

    src = tmp_path / "src"
    tiny_qwen3_moe(layers=2).to(torch.bfloat16).save_pretrained(src, save_original_format=stored_per_expert)
    fingerprint = save_tiny_tokenizer(src)
    calib = tmp_path / "calib.jsonl"
    write_ids(calib, [[1, 2, 3, 4, 5, 6, 7, 8]], fingerprint)
    out = build(
        str(src),
        Recipe(
            steps=[REGISTRY["gptq"](impl="torch"), REGISTRY["scale_kd"](**KD)],
            calib=FileSource(path=str(calib), samples=1, seqlen=8),
        ),
        format="w4a16-asym-g32",
        out_dir=tmp_path / "out",
        devices=["cpu"],
        dtype=torch.bfloat16,
        ignore=["model.layers.0.mlp.experts"],
        log=lambda _: None,
    )
    stored = _tensors(out)
    assert not any(
        k.startswith("model.layers.0.mlp.experts.") and k.endswith(".weight_packed") for k in stored
    )
    assert any(k.startswith("model.layers.1.mlp.experts.0.") and k.endswith(".weight_packed") for k in stored)
    want = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).state_dict()
    stacked = {p: want[f"model.layers.0.mlp.experts.{p}"] for p in ("gate_up_proj", "down_proj")}
    bank = {k: v for k, v in stored.items() if k.startswith("model.layers.0.mlp.experts.")}
    # a float bank is written per expert whatever the source's layout: SGLang's loader keeps one
    # `is_fused_expert` flag for the checkpoint and skips every per-expert tensor after a fused name
    assert set(bank) == {
        f"model.layers.0.mlp.experts.{e}.{proj}.weight"
        for e in range(8)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    rows = stacked["gate_up_proj"].shape[1] // 2  # gate rows first, then up (`per_expert_layout`)
    for e in range(8):
        base = f"model.layers.0.mlp.experts.{e}"
        assert torch.equal(bank[f"{base}.gate_proj.weight"], stacked["gate_up_proj"][e][:rows])
        assert torch.equal(bank[f"{base}.up_proj.weight"], stacked["gate_up_proj"][e][rows:])
        assert torch.equal(bank[f"{base}.down_proj.weight"], stacked["down_proj"][e])  # the source's bytes
    ck = P.Packed(out)
    scales, floats = ck.state(lambda _: None)
    stu = P.load_student(ck, scales, floats, torch.device("cpu"), "eager")
    assert not isinstance(stu.model.layers[0].mlp.experts, S.ServedExperts)  # restored, not served
    assert isinstance(stu.model.layers[1].mlp.experts, S.ServedExperts)  # the other bank trains its scales
    for proj in ("gate_up_proj", "down_proj"):  # stacked as stock transformers stacks the source
        name = f"model.layers.0.mlp.experts.{proj}"
        assert torch.equal(getattr(stu.model.layers[0].mlp.experts, proj), want[name])
    rec = json.loads((out / "build.json").read_text())
    assert [s["step"] for s in rec["history"]] == ["scale_kd"]  # GPTQ's record stays in its stage
    assert rec["history"][-1]["scales_changed"][0] > 0


def test_two_scale_kd_rounds_read_each_others_state(tiny):
    """The second round loads the first round's training state; the first round's file gives its own
    stored scales back."""
    q = _gptq(tiny)
    first = _kd(tiny, q, "kd1")
    sc, fl = P.Packed(first).state(lambda _: None)  # the check the second round's load makes
    assert not fl
    for stem, (s0, delta) in sc.items():
        assert torch.equal(P.stored_scale(s0, delta), P.Packed(first).tensor(stem + ".weight_scale")), stem
    second = _kd(tiny, first, "kd2")
    rec = json.loads((second / "build.json").read_text())
    assert rec["lineage"][-1] == str(first)
    assert [h["step"] for h in rec["history"]] == ["scale_kd", "scale_kd"]
    again, _ = P.Packed(second).state(lambda _: None)
    assert again  # the second round's own state reads back too


def test_write_computes_the_stored_scale_from_the_state_it_saves(tiny):
    """A writer handed an s0 at a wider precision stores the scale its saved bf16 s0 gives back.

    Before the one-function fix, `P.write` computed `weight_scale` from the fp32 s0 it was given but saved the
    s0 as bf16, so the next load (`Packed.state`) refused the file it had just written.
    """
    q = _gptq(tiny)
    ck = P.Packed(q)
    scales, floats = ck.state(lambda _: None)
    stem = "model.layers.0.mlp.down_proj"
    s0, delta = scales[stem]
    scales[stem] = (
        s0.float() * 1.0004,
        delta + 0.25,
    )  # fp32 s0, as a caller outside this round's tests gives
    out = tiny / "wide"
    P.write(ck, out, scales, floats, {"step": "fake"}, ck.teacher(), [torch.device("cpu")])
    back = P.Packed(out)
    again, _ = back.state(lambda _: None)  # refused before the fix
    assert torch.equal(back.tensor(stem + ".weight_scale"), P.stored_scale(*again[stem]))
    assert back.tensor(stem + ".weight_scale").dtype == torch.bfloat16


def test_a_scale_that_cannot_read_back_is_refused_when_written(tiny):
    """A non-finite scale cannot be given back by the state file, so the writer refuses it by module."""
    q = _gptq(tiny)
    ck = P.Packed(q)
    scales, floats = ck.state(lambda _: None)
    stem = ck.stems[0]
    s0, delta = scales[stem]
    scales[stem] = (s0, torch.full_like(delta, float("nan")))
    with pytest.raises(Refused, match=rf"stored scale of {stem} would not read back"):
        P.write(ck, tiny / "nan", scales, floats, {"step": "fake"}, ck.teacher(), [torch.device("cpu")])


# ---------------------------------------------------------------- soup


def _fake_checkpoint(src: Path, ck: Path, delta_fn, floats=None) -> None:
    """Write checkpoint `src` to `ck` with a hand-made training state and the matching stored scales."""
    w = P.Packed(src)
    sc, _ = w.state(lambda _: None)
    new = {s: (s0, delta_fn(s0)) for s, (s0, _) in sc.items()}
    P.write(w, ck, new, floats or {}, {"step": "fake"}, w.teacher(), [torch.device("cpu")])


def test_soup_zero_group_stays_finite(tiny):
    q = _gptq(tiny)
    ck = P.Packed(q)
    scales, floats = ck.state(lambda _: None)
    stem = ck.stems[0]
    s0, delta = scales[stem]
    s0 = s0.clone()
    s0[0, 0] = 0
    scales[stem] = (s0, delta)
    zero = tiny / "zero"
    P.write(ck, zero, scales, floats, {"step": "fake"}, ck.teacher(), [torch.device("cpu")])
    _fake_checkpoint(zero, tiny / "A", lambda s0: torch.full(s0.shape, 0.30))
    _fake_checkpoint(zero, tiny / "B", lambda s0: torch.full(s0.shape, -0.10))
    soup = REGISTRY["soup"](**{"with": [str(tiny / "B")], "weights": [1, 3]})
    out = build(str(tiny / "A"), _recipe(tiny, soup), out_dir=tiny / "S", devices=["cpu"], log=lambda _: None)
    result = P.Packed(out)
    for name in result.stems:
        assert torch.isfinite(result.tensor(name + ".weight_scale")).all(), name
    state, _ = result.state()
    for name, (start, change) in state.items():
        assert torch.isfinite(change).all(), name
        assert torch.isfinite(start.float() * change.exp()).all(), name
    assert result.tensor(stem + ".weight_scale")[0, 0] == 0
    start, change = state[stem]
    assert (start.float() * change.exp())[0, 0] == 0


def test_soup_math_and_refusals(tiny):
    q = _gptq(tiny)
    import shutil

    norm = "model.layers.0.input_layernorm.weight"
    base = _tensors(q)[norm].float()
    _fake_checkpoint(q, tiny / "A", lambda s0: torch.full(s0.shape, 0.30), {norm: base + 1})
    _fake_checkpoint(q, tiny / "B", lambda s0: torch.full(s0.shape, -0.10), {norm: base + 3})
    soup = REGISTRY["soup"](**{"with": [str(tiny / "B")], "weights": [1, 3]})
    out = build(str(tiny / "A"), _recipe(tiny, soup), out_dir=tiny / "S", devices=["cpu"], log=lambda _: None)
    sc, fl = P.Packed(out).state()
    for s, (s0, d) in sc.items():
        assert torch.allclose(d, torch.full(s0.shape, 0.25 * 0.30 + 0.75 * -0.10), atol=1e-6), s
    assert torch.allclose(fl[norm], base + 2.5)
    a, b = _tensors(q), _tensors(out)
    assert all(torch.equal(a[k], b[k]) for k in a if k.endswith(("weight_packed", "weight_zero_point")))

    # an untrained tensor that differs: refused by name
    shutil.copytree(tiny / "B", tiny / "C")
    t = _tensors(tiny / "C")
    other = "model.layers.1.post_attention_layernorm.weight"
    t[other] = t[other] + 1
    for f in (tiny / "C").glob("*.safetensors"):
        f.unlink()
    save_file(t, str(tiny / "C" / "model.safetensors"))
    (tiny / "C" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(t, "model.safetensors")})
    )
    with pytest.raises(Refused, match=other):
        build(str(tiny / "A"), _recipe(tiny, REGISTRY["soup"](**{"with": [str(tiny / "C")]})), out_dir=tiny / "S2",
              devices=["cpu"], log=lambda _: None)  # fmt: skip
    # other codes: refused
    t = _tensors(tiny / "B")
    k = next(k for k in t if k.endswith("weight_packed"))
    t[k] = t[k] ^ 1
    shutil.copytree(tiny / "B", tiny / "D")
    for f in (tiny / "D").glob("*.safetensors"):
        f.unlink()
    save_file(t, str(tiny / "D" / "model.safetensors"))
    (tiny / "D" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(t, "model.safetensors")})
    )
    with pytest.raises(Refused, match="shared codes"):
        build(str(tiny / "A"), _recipe(tiny, REGISTRY["soup"](**{"with": [str(tiny / "D")]})), out_dir=tiny / "S3",
              devices=["cpu"], log=lambda _: None)  # fmt: skip


@pytest.mark.parametrize("low,high,target", [(4, 8, None), (3, 4, "sm90@0.33.0:humming")])
def test_mixed_bit_scale_kd_preserves_codes_and_module_formats(tiny, low, high, target):
    """A float mixed-bit build can distill scales without decoding overrides at the body width."""
    body = f"w{low}a16-sym-g128"
    other = f"w{high}a16-sym-g128"
    stem = "model.layers.1.self_attn.o_proj"
    rules = [(stem, other)]
    kwargs = {
        "format": body,
        "target": target,
        "devices": ["cpu"],
        "format_overrides": rules,
        "log": lambda _: None,
    }
    base = build(str(tiny / "m"), _recipe(tiny, REGISTRY["rtn"]()), out_dir=tiny / "mixed", **kwargs)
    distilled = build(
        str(tiny / "m"),
        _recipe(tiny, REGISTRY["rtn"](), REGISTRY["scale_kd"](**KD)),
        out_dir=tiny / "mixed-kd",
        **kwargs,
    )
    before, after = P.Packed(base), P.Packed(distilled)
    import yaml

    recorded = yaml.safe_load((distilled / "resolved.yaml").read_text())
    assert recorded["format_overrides"] == [{"pattern": stem, "format": other}]
    assert before.format_of(stem) == after.format_of(stem) == Format.parse(other)
    assert any(
        not torch.equal(before.tensor(f"{s}.weight_scale"), after.tensor(f"{s}.weight_scale"))
        for s in before.stems
    )
    for s in before.stems:
        assert before.format_of(s) == after.format_of(s)
        for suffix in ("weight_packed", "weight_shape", "weight_zero_point"):
            key = f"{s}.{suffix}"
            assert (key in before.where) == (key in after.where)
            if key in before.where:
                assert torch.equal(before.tensor(key), after.tensor(key))
    # Distillation of an existing mixed checkpoint uses its stored formats, without fresh overrides.
    resumed = _kd(tiny, distilled, "mixed-kd-resumed", steps=1)
    assert P.Packed(resumed).format_of(stem) == Format.parse(other)


# ---------------------------------------------------------------- nvfp4

NV = "w4a4-nvfp4"


def _same_tensors(a: Path, b: Path) -> bool:
    ta, tb = _tensors(a), _tensors(b)
    return set(ta) == set(tb) and all(ta[k].dtype == tb[k].dtype and torch.equal(ta[k], tb[k]) for k in ta)


def _same_files(a: Path, b: Path) -> bool:
    """Every top-level safetensors file and the training state, byte for byte."""
    fa = {p.name: p.read_bytes() for p in sorted(a.glob("*.safetensors"))}
    fb = {p.name: p.read_bytes() for p in sorted(b.glob("*.safetensors"))}
    return fa == fb and (a / P.STATE).read_bytes() == (b / P.STATE).read_bytes()


def test_nvfp4_read_back_and_write_round_trip(tiny):
    """A build's NVFP4 checkpoint reads back as qparams (ladder codes, stored E4M3 scales, global scales), and
    writing that state reproduces its tensors; exporting the read-back checkpoint again is byte for byte."""
    q = _gptq(tiny, NV, "nv", target="sm100")
    ck = P.Packed(q)
    assert ck.nvfp4 and ck.fmt.name == NV and ck.target.arch == "sm100"
    sc, fl = ck.state(lambda _: None)
    assert not fl and set(sc) == set(ck.stems)
    tensors = _tensors(q)
    for s in ck.stems:
        packed, scale = tensors[f"{s}.weight_packed"], tensors[f"{s}.weight_scale"]
        codes, zero = ck.codes(s)
        assert zero is None
        assert torch.equal(codes, nvfp4.positions_of_codes(nvfp4.unpack_nibbles(packed)))
        assert tuple(codes.shape) == (scale.shape[0], scale.shape[1] * nvfp4.GROUP)
        s0, delta = sc[s]
        assert torch.equal(delta, torch.zeros_like(delta))  # a fresh build starts at the stored scales
        assert torch.equal(ck.stored_scale(s, s0, delta), scale)
        g = ck.global_scale(s)
        assert g.shape == (1,) and g.item() == tensors[f"{s}.weight_global_scale"].item() > 0
        assert ck.input_global_scale(s).item() == tensors[f"{s}.input_global_scale"].item() > 0
        # the served step is the stored scale over the global scale: what vLLM decodes with
        assert torch.equal(ck.step(s, s0, delta), scale.float() / g)
    out = tiny / "rt"
    P.write(ck, out, sc, fl, {"step": "roundtrip"}, ck.teacher(), [torch.device("cpu")])
    assert _same_tensors(q, out)
    again = P.Packed(out)
    sc2, fl2 = again.state(lambda _: None)
    assert not fl2
    assert all(torch.equal(sc2[s][0], sc[s][0]) and torch.equal(sc2[s][1], sc[s][1]) for s in sc)
    out2 = tiny / "rt2"
    P.write(again, out2, sc2, fl2, {"step": "roundtrip"}, again.teacher(), [torch.device("cpu")])
    assert _same_files(out, out2)


def test_nvfp4_bank_reads_back_and_serves_the_cut(tiny_moe):
    """An NVFP4 routed-expert bank reads back under its per-expert global scales (the gate's for gate and up) and
    one input scale per projection; the loop and the grouped path serve the checkpoint's weights cut per 16."""
    q = _gptq(tiny_moe, NV, "nv", target="sm100")
    ck = P.Packed(q)
    sc, fl = ck.state(lambda _: None)
    stu = P.load_student(ck, sc, fl, torch.device("cpu"), "eager")
    ex = stu.model.layers[0].mlp.experts
    assert isinstance(ex, S.ServedExperts)
    n, root = ex.num_experts, "model.layers.0.mlp.experts"
    # a fused expert unit stores one weight_global_scale (the gate's) and gate/up read one input scale
    for e in range(n):
        assert torch.equal(ck.global_scale(f"{root}.{e}.gate_proj"), ck.global_scale(f"{root}.{e}.up_proj"))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert len({ck.input_global_scale(f"{root}.{e}.{proj}").item() for e in range(n)}) == 1
    for bank, projs in ((ex.gate_up, ("gate_proj", "up_proj")), (ex.down, ("down_proj",))):
        want = torch.stack([ck.global_scale(f"{root}.{e}.{projs[0]}") for e in range(n)]).float()
        assert torch.equal(bank.gscale[:, 0], want.view(n, 1))  # the gate's scale, for every row
        assert torch.equal(bank.igscale, ck.input_global_scale(f"{root}.0.{projs[0]}"))
        for e in range(n):  # the served weight is the stored codes under the stored step
            ref = []
            for p in projs:
                stem = f"{root}.{e}.{p}"
                codes, zero = ck.codes(stem)
                ref.append(ck.quantizer(stem, "moe").dequant(codes, ck.step(stem, *sc[stem]), zero))
            assert torch.equal(bank.expert(e).weight(bank.delta[e]), torch.cat(ref))
    # the grouped path = the loop, with the cut: the same served inputs, bf16 weights on the grouped side
    with torch.no_grad():
        for bank in (ex.gate_up, ex.down):
            bank.delta.normal_(0, 0.05)
    torch.manual_seed(2)
    x = torch.randn(16, stu.config.hidden_size, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(8)[:2] for _ in range(16)])
    wts = torch.rand(16, 2, dtype=torch.bfloat16)
    gy = torch.randn(16, stu.config.hidden_size, dtype=torch.bfloat16)
    got = {}
    for grouped in (False, True):
        ex.grouped = grouped
        for bank in (ex.gate_up, ex.down):
            bank.delta.grad = None
            bank.delta.requires_grad_(True)
        xi = x.clone().requires_grad_(True)
        y = ex(xi, idx, wts)
        y.backward(gy)
        got[grouped] = (y.detach(), xi.grad, ex.gate_up.delta.grad.clone(), ex.down.delta.grad.clone())
    for a, b in zip(got[False], got[True], strict=True):
        assert (a.float() - b.float()).norm() <= 1e-2 * a.float().norm()  # bf16 weights and sums
    assert ex.gate_up.delta.grad is not None and ex.down.delta.grad is not None


def test_nvfp4_delta_grad_is_autograd_through_the_weight():
    """The NVFP4 closed form of `delta_grad` (e2m1 values, no graph over the weight) is autograd's gradient."""
    from qda_quant.target import KernelPlan

    torch.manual_seed(0)
    plan = KernelPlan("K", scale_dtype="nvfp4", act_dtype="nvfp4", zero_point=False)
    q = Quantizer.of(Format.parse(NV), plan, "nvfp4")
    w = torch.randn(48, 256) * 0.02
    s, _ = q.params(w)
    codes = q.quantize(w, s, None)
    st = S.Stored(codes.to(torch.int8), None, s.to(torch.bfloat16), q,
                  gscale=q.grid.global_scale(q.stored_scale(s)))  # fmt: skip
    delta = 0.05 * torch.randn(s.shape)
    rows = slice(5, 37)
    d = delta[rows].clone().requires_grad_(True)
    gw = torch.randn(32, 256)
    ow = torch.autograd.grad(st.weight(d, rows), d, gw)[0]
    assert torch.allclose(st.delta_grad(delta[rows], gw, rows), ow, rtol=1e-5, atol=1e-8)


def test_nvfp4_bank_delta_grad_slices_the_row_global_scale():
    """A bank's flat view carries one weight global scale per row (its experts' own): a slice of rows divides by
    that slice's scales, not the whole bank's, in the weight and in its closed-form gradient."""
    from qda_quant.target import KernelPlan

    torch.manual_seed(0)
    plan = KernelPlan("K", scale_dtype="nvfp4", act_dtype="nvfp4", zero_point=False)
    q = Quantizer.of(Format.parse(NV), plan, "nvfp4")
    n_exp, out, inn = 4, 6, 32
    G = inn // 16
    codes = torch.randint(-7, 8, (n_exp, out, inn), dtype=torch.int8)
    s0 = (torch.rand(n_exp, out, G) + 0.5).to(torch.float8_e4m3fn).float().to(torch.bfloat16)
    delta = 0.05 * torch.randn(n_exp, out, G)
    gs = torch.rand(n_exp) + 0.5
    bank = S.ServedBank([((f"e{e}", out),) for e in range(n_exp)], codes, None, s0, delta, q,
                        gscale=gs, igscale=torch.ones(1))  # fmt: skip
    st = bank.flat()
    rows = slice(out + 2, 3 * out + 4)  # no whole expert: the broadcast that hid the bug cannot stand in
    d = delta.reshape(-1, G)[rows].clone().requires_grad_(True)
    gw = torch.randn(rows.stop - rows.start, inn)
    ow = torch.autograd.grad(st.weight(d, rows), d, gw)[0]
    assert torch.allclose(st.delta_grad(d, gw, rows), ow, rtol=1e-5, atol=1e-8)
    # the grouped forward's chunked backward is a slice of whole experts, through the bank
    ch = slice(1, 3)
    full = delta.reshape(-1, G).clone().requires_grad_(True)
    rows2 = slice(ch.start * out, ch.stop * out)
    d2 = full[rows2]
    want = torch.autograd.grad(st.weight(d2, rows2), d2, gw[: 2 * out])[0]
    got = bank.delta_grad(delta, gw[: 2 * out].view(2, out, inn), ch)
    assert torch.allclose(got.reshape(-1, G), want, rtol=1e-5, atol=1e-8)
    # a plain Linear's ``[1]`` global scale still covers every row of a slice
    lin = S.Stored(codes[0], None, s0[0], q, gscale=gs[:1])
    d1 = delta[0].clone().requires_grad_(True)
    sl = slice(3, 5)
    d1s = d1[sl]
    ow = torch.autograd.grad(lin.weight(d1s, sl), d1s, gw[:2])[0]
    assert torch.allclose(lin.delta_grad(d1[sl], gw[:2], sl), ow, rtol=1e-5, atol=1e-8)


def test_materialize_serves_the_same_bits_and_follows_the_optimizer(tiny):
    """`ServedLinear.materialize` decodes once and serves those bits until the next call; a moved `delta` is
    picked up by the next `materialize`, as a step's decode after `optimizer.step()`."""
    q = _gptq(tiny, NV, "nv", target="sm100")
    ck = P.Packed(q)
    sc, fl = ck.state(lambda _: None)
    m = P.load_student(ck, sc, fl, torch.device("cpu"), "eager")
    mod = m.get_submodule("model.layers.0.mlp.down_proj")
    x = torch.randn(3, mod.in_features, dtype=torch.bfloat16)
    before = mod(x)
    mod.materialize()
    first = mod(x)
    assert torch.equal(first, before)
    with torch.no_grad():
        mod.delta.add_(0.2)
    assert torch.equal(mod(x), first)  # the step's decode, not the new state
    mod.materialize()
    assert not torch.equal(mod(x), first)
    per_call = S.Stored(mod.codes, mod.zero, mod.s0, mod.q, mod.bias, mod.gscale, mod.igscale)
    assert torch.equal(mod(x), per_call(x, mod.delta))


def test_nvfp4_served_forward_is_the_checkpoint_decode(tiny):
    """The student's module decodes the stored E2M1 codes under the E4M3 scale and global scale, and cuts its
    input under `input_global_scale`; training the scale flows through that arithmetic."""
    q = _gptq(tiny, NV, "nv", target="sm100")
    ck = P.Packed(q)
    sc, fl = ck.state(lambda _: None)
    m = P.load_student(ck, sc, fl, torch.device("cpu"), "eager")
    stem = "model.layers.0.mlp.down_proj"
    mod = m.get_submodule(stem)
    assert mod.q.plan.scale_dtype == mod.q.plan.act_dtype == "nvfp4"
    assert torch.equal(mod.gscale, ck.global_scale(stem))
    assert torch.equal(mod.igscale, ck.input_global_scale(stem))
    x = torch.randn(3, mod.in_features, dtype=torch.bfloat16)
    mod.delta.requires_grad_(True)
    y = mod(x)
    stored = nvfp4.dequant(
        nvfp4.unpack_nibbles(ck.tensor(f"{stem}.weight_packed")),
        ck.tensor(f"{stem}.weight_scale"),
        ck.global_scale(stem),
    )
    cut = nvfp4.block_cut(x, ck.input_global_scale(stem))
    assert torch.equal(y, (cut.float() @ stored.float().t()).to(torch.bfloat16))
    y.float().sum().backward()
    assert torch.isfinite(mod.delta.grad).all() and (mod.delta.grad != 0).any()


def test_scale_kd_nvfp4_trains_the_stored_scales_codes_identical(tiny):
    """The trained delta reaches the stored E4M3 ``weight_scale`` the checkpoint holds: a step must cross a code
    to count, so this trains long enough (300 steps at lr 1e-3 on a real model) to move some."""
    q = _gptq(tiny, NV, "nv", target="sm100")
    out = _kd(tiny, q, "nv-kd", target="sm100", steps=40)
    a, b = _tensors(q), _tensors(out)
    assert set(a) == set(b)
    for k in a:
        if not k.endswith(".weight_scale"):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    scales = [k for k in a if k.endswith(".weight_scale")]
    assert all(b[k].dtype == torch.float8_e4m3fn for k in scales)
    assert sum(int((a[k] != b[k]).sum()) for k in scales) > 0
    ck = P.Packed(out)
    sc, fl = ck.state(lambda _: None)
    assert not fl and all(torch.equal(ck.stored_scale(s, *sc[s]), b[s + ".weight_scale"]) for s in ck.stems)
    stage = json.loads((out / "build.json").read_text())["history"][-1]
    assert stage["scales_changed"][0] > 0 and stage["log"][-1]["kl"] < stage["log"][0]["kl"]


def test_soup_nvfp4_averages_the_stored_scales(tiny):
    """soup works off the stored E4M3 scales: the averaged delta is written through `Packed.stored_scale`."""
    q = _gptq(tiny, NV, "nv", target="sm100")
    _fake_checkpoint(q, tiny / "A", lambda s0: torch.full(s0.shape, 0.30))
    _fake_checkpoint(q, tiny / "B", lambda s0: torch.full(s0.shape, -0.10))
    soup = REGISTRY["soup"](**{"with": [str(tiny / "B")], "weights": [1, 3]})
    out = build(str(tiny / "A"), _recipe(tiny, soup), target="sm100", out_dir=tiny / "S", devices=["cpu"],
                log=lambda _: None)  # fmt: skip
    result = P.Packed(out)
    sc, _ = result.state(lambda _: None)
    for s, (s0, d) in sc.items():
        assert torch.allclose(d, torch.full(s0.shape, 0.25 * 0.30 + 0.75 * -0.10), atol=1e-6), s
        assert torch.equal(result.stored_scale(s, s0, d), result.tensor(s + ".weight_scale"))
    a, b = _tensors(q), _tensors(out)
    assert all(torch.equal(a[k], b[k]) for k in a if k.endswith("weight_packed"))


def test_a_stage_projects_its_time_from_its_first_steps(tiny):
    """After three measured steps the stage logs its median s/step and the time the configured steps project.

    The projection is the measured rate times the configured count, in its own log line, so a long stage does
    not run blind; a stage of three steps or fewer has nothing left to project and logs only its steps.
    """
    from qda_quant.steps.scale_kd import _PROJECT_AFTER

    q = _gptq(tiny)
    notes: list[str] = []
    build(str(q), _recipe(tiny, REGISTRY["scale_kd"](**{**KD, "steps": _PROJECT_AFTER + 2})),
          out_dir=tiny / "proj", devices=["cpu"], log=notes.append)  # fmt: skip
    line = next(n for n in notes if n.startswith("scale_kd: measured"))
    assert f"median of {_PROJECT_AFTER}" in line and "s/step" in line
    assert f"configured {_PROJECT_AFTER + 2} steps" in line and "min of training" in line
    # it comes after the project's measured steps, which all still log
    logs = [i for i, n in enumerate(notes) if n.startswith("scale_kd {")]
    assert len(logs) == _PROJECT_AFTER + 2 and notes.index(line) == logs[_PROJECT_AFTER - 1] + 1

    short: list[str] = []
    build(str(q), _recipe(tiny, REGISTRY["scale_kd"](**{**KD, "steps": _PROJECT_AFTER})),
          out_dir=tiny / "short", devices=["cpu"], log=short.append)  # fmt: skip
    assert not [n for n in short if n.startswith("scale_kd: measured")]


@pytest.mark.parametrize(
    ("fmt", "once"),
    [("w4a16", False), ("w3a16", False), ("w4a8", False), ("w4a4-nvfp4", True), ("w8a8-fp8", True)],
)
def test_only_nvfp4_and_fp8_decode_their_served_weights_once(fmt, once):
    """The forward of NVFP4 and FP8 reads one decode per step; the integer formats decode per call."""
    assert S.decodes_once(Format.parse(fmt)) is once


def test_a_w4a16_plan_holds_no_decoded_weights_and_an_nvfp4_plan_does(tiny):
    """A W4A16 stage's terms are those before decoded weights were added (a 31B student would carry about
    46 GiB more); an NVFP4 stage still sizes its decode, bf16 or fp32 per weight."""
    from qda_quant.engine.planning import compile_plan

    cpu = torch.device("cpu")
    kd = _recipe(tiny, REGISTRY["scale_kd"](**KD))
    w4 = dict(
        compile_plan(str(_gptq(tiny, "w4a16-asym-g32")), kd, None, Target(), [cpu]).stage_peaks[0].terms
    )
    assert "served weights" not in w4  # a zero term is not listed, as before
    nv = dict(
        compile_plan(str(_gptq(tiny, NV, "nv", target="sm100")), kd, None, Target.parse("sm100"), [cpu])
        .stage_peaks[0]
        .terms
    )
    assert nv["served weights"] > 0


@pytest.mark.parametrize(("fmt", "target", "calls"), [("w4a16-asym-g32", None, False), (NV, "sm100", True)])
def test_scale_kd_materializes_only_the_formats_that_decode_once(tiny, monkeypatch, fmt, target, calls):
    """A W4A16 stage never materializes (the row-blocked backward recomputes the weight); NVFP4 does, each step."""
    seen = []
    real = S.ServedLinear.materialize

    def count(self):
        seen.append(self.stem)
        real(self)

    monkeypatch.setattr(S.ServedLinear, "materialize", count)
    q = _gptq(tiny, fmt, "q-" + fmt, target=target)
    _kd(tiny, q, "kd-" + fmt, target=target, steps=2)
    assert bool(seen) is calls
