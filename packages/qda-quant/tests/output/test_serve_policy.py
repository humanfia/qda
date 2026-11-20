"""serve_policy: a scorer applies only the serve settings `qda build` writes."""

import re

import pytest
from qda_quant.format import FORMATS, Format
from qda_quant.output.serve_policy import ServeNotAllowed, allowed, check, sglang_limits
from qda_quant.target import TARGETS, Target, merge_serve

GOOD = {
    "vllm": "0.33.0",
    "env": {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel,HummingLinearKernel"},
    "args": {"moe_backend": "marlin", "linear_backend": "marlin", "dtype": "bfloat16"},
    "plugins": [],
    "kernels": {"dense": "MarlinLinearKernel", "moe": "MarlinExperts"},
}


def test_the_modular_expert_kernel_a_self_routing_bank_binds_is_allowed():
    """Gemma 4's bank binds TrtLlmNvFp4ExpertsModular; a build.json naming it must pass the policy."""
    assert "TrtLlmNvFp4ExpertsModular" in allowed()["kernels"]
    check(
        {
            "vllm": "0.33.0",
            "env": {},
            "args": {},
            "plugins": [],
            "kernels": {"dense": "FlashInferCuteDslNvFp4LinearKernel", "moe": ["TrtLlmNvFp4ExpertsModular"]},
        }
    )


def test_every_section_qda_build_writes_passes():
    """merge_serve over every target, version, backend, format and layer mix: all allowed."""
    n = 0
    for arch in TARGETS:
        for ver in ("0.27.1", "0.30.0", "0.32.0", "0.33.0"):
            for backend in (None, "marlin", "machete", "humming"):
                t = Target(arch, ver, backend)
                for _, fmt in FORMATS.items():
                    for kinds in (("dense",), ("dense", "moe")):
                        try:
                            serve = merge_serve({k: t.plan(fmt, k) for k in kinds}, t.vllm)
                        except Exception:  # noqa: BLE001 (nothing serves this combination)
                            continue
                        check(serve)
                        n += 1
    assert n > 10
    check(GOOD)
    check(None)


@pytest.mark.parametrize(
    "bad",
    [
        {"env": {"PYTHONPATH": "/tmp/x"}},
        {"env": {"LD_PRELOAD": "/tmp/x.so"}},
        {"env": {"PATH": "/tmp/bin"}},
        {"env": {"VLLM_PLUGINS": "evil"}},
        {"env": {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel,/tmp/x"}},
        {"env": {"VLLM_DISABLED_KERNELS": 1}},
        {"args": {"trust_remote_code": True}},
        {"args": {"tokenizer": "/tmp/tok"}},
        {"args": {"model": "/tmp/m"}},
        {"args": {"quantization": "my_plugin"}},
        {"args": {"hf_overrides": {"x": 1}}},
        {"args": {"moe_backend": "unknown"}},  # Humming MoE is now written by target :humming.
        {"args": {"dtype": "auto"}},
        {"args": {"moe_backend": ["marlin"]}},
        {"plugins": ["evil_plugin"]},
        {"plugins": "qda_humming_moe"},
        {"kernels": {"dense": "MyKernel"}},
        {"kernels": {"other": "MarlinLinearKernel"}},
        {"vllm": "0.33.0; rm -rf /"},
        {"extra": 1},
        {"env": ["PYTHONPATH=/tmp"]},
        "not a mapping",
    ],
)
def test_anything_else_is_refused(bad):
    with pytest.raises(ServeNotAllowed, match="serve settings not allowed"):
        check(bad if isinstance(bad, str) else {**GOOD, **bad})


def test_the_allowed_values_are_names_not_paths():
    """No allowed env value or plugin is a path or free text, and no variable can steer code loading."""
    a = allowed()
    name = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
    assert set(a["env"]) <= {"VLLM_DISABLED_KERNELS"}, a["env"]
    for values in a["env"].values():
        assert all(name.fullmatch(v) for v in values), values
    assert all(name.fullmatch(p) for p in a["plugins"]), a["plugins"]
    assert set(a["args"]) <= {"dtype", "moe_backend", "linear_backend"}, a["args"]
    assert all(isinstance(v, str) and name.fullmatch(v) for vs in a["args"].values() for v in vs)


@pytest.mark.parametrize("fused", [False, True], ids=["plain-names", "fused-declaration"])
def test_sglang_fused_inputs_follow_remaining_quantized_modules(fused):
    """Two packed inputs in one fused module warn; explicit float ignores and other layers do not."""
    from qda_quant.arch import LayerGraph, LinearGroup, ModelGraph
    from qda_quant.arch.adapters import FusedRuntime
    from qda_quant.format import Format
    from qda_quant.output.serve_policy import sglang_limits

    pattern = r"re:model\.layers\.\d+\.self_attn\.(?:q_a_proj|kv_a_proj)$"
    declaration = FusedRuntime(pattern, ("q_a_proj", "kv_a_proj")) if fused else pattern

    def graph(inputs):
        layers = tuple(
            LayerGraph(
                i,
                f"model.layers.{i}",
                (LinearGroup(tuple(f"model.layers.{i}.self_attn.{name}" for name in names), "attn_in"),),
            )
            for i, names in enumerate(inputs)
        )
        return ModelGraph(
            "plugin",
            layers,
            "model.embed_tokens",
            "lm_head",
            "model.norm",
            extra={"runtime_names": {"sglang": {"model.layers.0.self_attn.wqkv_a": declaration}}},
        )

    fmt = Format.parse("w4a16-sym-g128")
    packed = graph((("q_a_proj", "kv_a_proj"),))
    assert any(
        "separately packed inputs of fused dense modules" in row
        for row in sglang_limits(fmt, "dense", graph=packed)
    )
    for inputs in (((),), (("q_a_proj",),), (("q_a_proj",), ("kv_a_proj",))):
        assert not sglang_limits(fmt, "dense", graph=graph(inputs))
    assert not sglang_limits(fmt, "dense")


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
@pytest.mark.parametrize("root", ["model", "model.language_model"])
def test_sglang_qwen35_shared_warning_follows_packed_gate_up(model_type, root):
    """Float shared experts load; packing either fused input needs a warning, even if the other stays float."""
    from types import SimpleNamespace

    from qda_quant.arch import LayerGraph, LinearGroup, ModelGraph
    from qda_quant.format import Format

    shared = f"{root}.layers.0.mlp.shared_expert"

    def warnings(projections, fmt="w4a16-sym-g128", architecture=model_type):
        graph = ModelGraph(
            "fused_experts",
            (
                LayerGraph(
                    0,
                    f"{root}.layers.0",
                    (LinearGroup(tuple(f"{shared}.{name}" for name in projections), "mlp_in"),),
                ),
            ),
            f"{root}.embed_tokens",
            "lm_head",
            f"{root}.norm",
        )
        return any(
            "shared-expert gate/up weights" in row
            for row in sglang_limits(
                Format.parse(fmt),
                "moe",
                graph=graph,
                text_config=SimpleNamespace(model_type=architecture),
            )
        )

    for projections in (("gate_proj",), ("up_proj",), ("gate_proj", "up_proj")):
        assert warnings(projections)
    assert not warnings(())
    assert not warnings(("down_proj",))
    assert not warnings(("gate_proj",), architecture="qwen3_next")
    assert not warnings(("gate_proj",), fmt="w4a8-sym-g128")


def test_sglang_qwen3_next_checks_packed_gdn_and_float_shared_experts():
    """Keeping one GDN projection float cannot make another packed projection loadable."""
    from types import SimpleNamespace

    from qda_quant.arch import LayerGraph, LinearGroup, ModelGraph
    from qda_quant.format import Format

    qkvz = "model.layers.0.linear_attn.in_proj_qkvz"
    shared = "model.layers.0.mlp.shared_expert.gate_proj"

    def warnings(kept=(), packed=()):
        graph = ModelGraph(
            "qwen3_next",
            (LayerGraph(0, "model.layers.0", (LinearGroup(tuple(packed), "attn_in"),)),),
            "model.embed_tokens",
            "lm_head",
            "model.norm",
            extra={"ignored": dict.fromkeys(kept)},
        )
        return sglang_limits(
            Format.parse("w4a16-sym-g128"),
            "moe",
            graph=graph,
            text_config=SimpleNamespace(model_type="qwen3_next"),
        )

    assert any("Qwen3-Next INT4" in row for row in warnings((shared,), (qkvz,)))
    assert any("Qwen3-Next INT4" in row for row in warnings((qkvz, shared)))
    assert any("Qwen3-Next INT4" in row for row in warnings((qkvz,), (qkvz.replace("layers.0", "layers.1"),)))
    assert warnings((qkvz,)) == ()


@pytest.mark.parametrize("field", ["hidden_act", "hidden_activation"])
@pytest.mark.parametrize("activation", ["gelu", "gelu_pytorch_tanh", "gelu_new", "silu"])
def test_sglang_gelu_limit_uses_expert_activation_only_for_w4a16(field, activation):
    """GELU variants warn for W4A16 experts; dense layers, SiLU and omitted configs keep their rows."""
    from types import SimpleNamespace

    from qda_quant.format import Format
    from qda_quant.output.serve_policy import sglang_limits

    config = SimpleNamespace(**{field: activation})
    fmt = Format.parse("w4a16-sym-g128")
    rows = sglang_limits(fmt, "moe", text_config=config)
    gelu_rows = [row for row in rows if "GELU INT4 experts" in row]
    assert len(gelu_rows) == int(activation.startswith("gelu"))
    assert all("vLLM 0.33.0" in row for row in gelu_rows)
    assert not sglang_limits(fmt, "dense", text_config=config)
    assert not sglang_limits(fmt, "moe")
    for other in ("w3a16", "w4a8"):
        assert not any(
            "GELU INT4 experts" in row
            for row in sglang_limits(Format.parse(other), "moe", text_config=config)
        )


@pytest.mark.parametrize("name", ["w4a16-asym-g128", "w4a16-asym-g64"])
def test_sglang_warns_about_asymmetric_expert_zero_points(name):
    """SGLang 0.5.23's fused-MoE loader transposes every expert tensor except names containing "zero"
    (`_weight_loader_impl`), so its `[experts, groups, out/packed]` zero-point buffer gets `[out/packed, groups]`
    and the load stops with a tensor-size error. Its dense loader does transpose zero points, and a symmetric
    format carries none, so neither warns. vLLM loads the asymmetric file, which is why this warns, not refuses.
    """
    warnings = sglang_limits(Format.parse(name), "moe")
    assert len(warnings) == 1, warnings
    assert "asymmetric per-expert INT4" in warnings[0] and "zero points" in warnings[0]
    assert sglang_limits(Format.parse(name), "dense") == ()
    assert sglang_limits(Format.parse(name.replace("asym", "sym")), "moe") == ()


@pytest.mark.parametrize("name", ["w4a16", "w4a16-g64"])
def test_sglang_does_not_warn_about_zero_points_a_bare_spelling_does_not_write(name):
    """A bare spelling stores the routed experts symmetric, so the SGLang zero-point warning would name a scheme
    the build does not write. `w4a16-g64` also proves the group-size line follows the expert format."""
    from qda_quant.format import Format

    fmt = Format.parse(name)
    assert fmt.symmetric_defaulted and not fmt.symmetric  # dense layers stay asymmetric
    assert sglang_limits(fmt, "moe") == ()


@pytest.mark.parametrize("name", ["w3a16-sym-g128", "w4a16-sym-g128", "w8a16-sym-g128", "w4a8"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("native", [True, False])
def test_sglang_kimi_wna16_limit_covers_native_and_text_configs(name, kind, native):
    """The MXFP4 suffix rewrite also breaks WNA16 dense layers, independent of a fused-input graph."""
    from types import SimpleNamespace

    configs = (
        {"source_config": SimpleNamespace(model_type="kimi_k3")}
        if native
        else {"text_config": SimpleNamespace(model_type="kimi_k3_text")}
    )
    rows = sglang_limits(Format.parse(name), kind, **configs)
    warnings = [row for row in rows if "Kimi-K3 WNA16" in row]
    assert len(warnings) == int(name != "w4a8")
    assert all("weight_packed to weight" in row and "bf16 base loads" in row for row in warnings)
    assert not any("Kimi-K3 WNA16" in row for row in sglang_limits(Format.parse(name), kind))
