"""A dense linear whose runtime fused shape the planned kernel cannot take is refused at plan time.

vLLM 0.33's ``CutlassW4A8LinearKernel.can_implement`` takes only an ``in`` and an ``out`` that are both
multiples of 128; the group check covers ``in`` but nothing covered ``out``. vLLM fuses a ``LinearGroup`` before serving it, so the checked ``out`` is the
group's summed rows (q+k+v, gate+up), not each checkpoint projection: 64-row k/v projections are fine beside
a 128-row q_proj. Where the engine does not fuse a group's modules (MLA with ``q_lora_rank`` unset serves
``q_proj`` and ``kv_a_proj_with_mqa`` separately), each is checked on its own shape.
"""

import pytest
import torch
import transformers
from qda_quant import REGISTRY, Format, Refused, Target
from qda_quant.arch import LayerGraph, LinearGroup, ModelGraph
from qda_quant.arch.adapters import DEEPSEEK_V2
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build_validation import check_dense_shapes
from qda_quant.engine.planning import compile_plan
from qda_quant.recipe import Recipe

from .engine_util import save_tiny_tokenizer, write_ids


def _tiny(tmp, kv_heads=2):
    """A one-layer model whose fused q/k/v is ``128 + 64 * kv_heads`` rows wide, every input width 128."""
    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=32,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "sdpa"
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    model.save_pretrained(tmp / "m")
    sha = save_tiny_tokenizer(tmp / "m")
    write_ids(tmp / "c.jsonl", [[3, 4, 5, 6]], sha)
    return str(tmp / "m"), FileSource(path=str(tmp / "c.jsonl"), samples=1, seqlen=4)


def _plan(tmp, fmt, kv_heads=2, target="sm90"):
    source, calib = _tiny(tmp, kv_heads)
    return compile_plan(
        source,
        Recipe(steps=[REGISTRY["rtn"]()], calib=calib),
        Format.parse(fmt),
        Target.parse(target),
        [torch.device("cpu")],
    )


def test_dense_w4a8_refuses_a_fused_linear_the_kernel_cannot_take(tmp_path):
    """One k/v head fuses q/k/v into 192 rows: not a multiple of 128, and the message names the fused linear."""
    with pytest.raises(
        Refused,
        match=r"fused linear of .*q_proj, .*k_proj, .*v_proj: CutlassW4A8LinearKernel cannot serve "
        r"w4a8-sym-g128: 192 out by 128 in",
    ):
        _plan(tmp_path, "w4a8", kv_heads=1)


@pytest.mark.parametrize(
    ("fmt", "kv_heads", "target"),
    [
        ("w4a8", 2, "sm90"),  # 64-row k/v alone, fused to 256: the shape vLLM serves, so it must plan
        ("w4a8", 4, "sm90"),  # every projection and the fused linear a multiple of 128
        ("w4a16-sym-g128", 1, "sm90"),  # Marlin and Humming have no output-width rule
        ("w3a16-sym-g128", 1, "sm90"),
    ],
)
def test_a_fused_shape_and_format_the_rule_does_not_refuse_still_plans(tmp_path, fmt, kv_heads, target):
    """The controls prove the refusal is the fused kernel width rule, not w4a8 or attention shapes in general."""
    _plan(tmp_path, fmt, kv_heads, target)


class _V2MHA(torch.nn.Module):
    """DeepSeek-V2 with ``q_lora_rank`` unset: q_proj and kv_a share one 128-wide input, served separately."""

    def __init__(self, q: int = 192, kv_a: int = 192) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        attn = torch.nn.Module()
        attn.q_proj = torch.nn.Linear(128, q, bias=False, device="meta")
        attn.kv_a_proj_with_mqa = torch.nn.Linear(128, kv_a, bias=False, device="meta")
        self.model.layers[0].self_attn = attn


def _graph(model: torch.nn.Module, declared: bool, adapter: str, names) -> ModelGraph:
    """The one-group layer graph this model derives, with the adapter's runtime names when `declared`."""
    modules = tuple(name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    extra = {"runtime_names": names(model)} if declared else {}
    return ModelGraph(
        model_type=adapter,
        layers=(LayerGraph(0, "model.layers.0", (LinearGroup(modules, "attn_in"),)),),
        embed="model.embed_tokens",
        lm_head="lm_head",
        final_norm="model.norm",
        adapter=adapter,
        extra=extra,
    )


def _check(model: torch.nn.Module, declared: bool, adapter: str, names) -> None:
    fmt = Format.parse("w4a8")
    plan = Target().plan(fmt, "dense")
    check_dense_shapes(
        model,
        _graph(model, declared, adapter, names),
        fmt,
        [REGISTRY["rtn"]()],
        plan,
        {(fmt.name, "dense"): plan},
    )


@pytest.mark.parametrize(
    ("widths", "declared", "refused"),
    [
        # q_lora_rank unset: vLLM serves q_proj and kv_a_proj_with_mqa as two 192-row linears. Their 384-row
        # sum is a multiple of 128, so only the per-linear check sees CUTLASS refusing both.
        ((192, 192), True, r"self_attn\.q_proj: .* 192 out by 128 in"),
        ((256, 256), True, None),
        # Without the declaration the check sees the group's 384 rows, which passes: the old blind spot.
        ((192, 192), False, None),
    ],
)
def test_unfused_mla_projections_are_checked_one_by_one(widths, declared, refused):
    """MLA's q_proj and kv_a are separate runtime linears, not one fused 384-row linear."""
    model = _V2MHA(*widths)
    if refused is None:
        _check(model, declared, "deepseek_v2", DEEPSEEK_V2.runtime_names)
    else:
        with pytest.raises(Refused, match=refused):
            _check(model, declared, "deepseek_v2", DEEPSEEK_V2.runtime_names)
