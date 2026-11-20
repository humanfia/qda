"""The MTP rewrite, in fp64: exact when the last norm's constant fit is, and its measured inexactness otherwise."""

import pytest
import torch
from qda_quant.arch.adapters.stream import SourceBlock
from qda_quant.step import Refused
from qda_quant.steps._source_block import rewrite

D, V, T = 64, 257, 16

BLOCK = SourceBlock(
    fc="mtp.fc",
    inputs=("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"),
    folds=(
        ("mtp.layers.0.input_layernorm", ("mtp.layers.0.self_attn.v_proj",)),
        ("mtp.layers.0.post_attention_layernorm", ("mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj")),
    ),
    writers=("mtp.layers.0.self_attn.o_proj", "mtp.layers.0.mlp.down_proj"),
    norm="mtp.norm",
    kept=(),
    offset=1.0,
)


def _weights(seed=0):
    """Source-basis block weights, stored in the (1 + w) norm convention."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    def gain():
        return r(D) * 0.2

    return {
        "mtp.fc.weight": r(D, 2 * D),
        "mtp.pre_fc_norm_embedding.weight": gain(),
        "mtp.pre_fc_norm_hidden.weight": gain(),
        "mtp.layers.0.input_layernorm.weight": gain(),
        "mtp.layers.0.post_attention_layernorm.weight": gain(),
        "mtp.layers.0.self_attn.v_proj.weight": r(D, D),
        "mtp.layers.0.self_attn.o_proj.weight": r(D, D),
        "mtp.layers.0.mlp.gate_proj.weight": r(2 * D, D),
        "mtp.layers.0.mlp.up_proj.weight": r(2 * D, D),
        "mtp.layers.0.mlp.down_proj.weight": r(D, 2 * D),
        "mtp.norm.weight": gain(),
    }


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _source_logits(w, emb, x, f_norm, head):
    """The block in the source basis; `emb` and `x` are the embedding output and the target's final-norm input."""
    gain = lambda k: 1.0 + w[f"{k}.weight"]  # noqa: E731
    e = _rms(emb) * gain("mtp.pre_fc_norm_embedding")
    hh = _rms(_rms(x) * (1.0 + f_norm)) * gain("mtp.pre_fc_norm_hidden")
    z = torch.cat([e, hh], -1) @ w["mtp.fc.weight"].T
    v = (_rms(z) * gain("mtp.layers.0.input_layernorm")) @ w["mtp.layers.0.self_attn.v_proj.weight"].T
    z = z + v @ w["mtp.layers.0.self_attn.o_proj.weight"].T
    m = _rms(z) * gain("mtp.layers.0.post_attention_layernorm")
    z = (
        z
        + (
            torch.nn.functional.silu(m @ w["mtp.layers.0.mlp.gate_proj.weight"].T)
            * (m @ w["mtp.layers.0.mlp.up_proj.weight"].T)
        )
        @ w["mtp.layers.0.mlp.down_proj.weight"].T
    )
    y = _rms(z) * gain("mtp.norm")
    return y @ head.T


def _rotated_logits(w, emb, x, q, f_norm, head, v):
    """The block after the rewrite, reading the rotated embedding and the rotated (identity final norm) hidden."""
    z = torch.cat([_rms(emb @ q), _rms(x @ q)], -1) @ w["mtp.fc.weight"].T
    z = (
        z
        + (_rms(z) @ w["mtp.layers.0.self_attn.v_proj.weight"].T)
        @ w["mtp.layers.0.self_attn.o_proj.weight"].T
    )
    z = (
        z
        + (
            torch.nn.functional.silu(_rms(z) @ w["mtp.layers.0.mlp.gate_proj.weight"].T)
            * (_rms(z) @ w["mtp.layers.0.mlp.up_proj.weight"].T)
        )
        @ w["mtp.layers.0.mlp.down_proj.weight"].T
    )
    y = _rms(z) * v
    return y @ ((head * (1.0 + f_norm)) @ q).T


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(T, D, generator=g, dtype=torch.float64),
        torch.randn(T, D, generator=g, dtype=torch.float64),
        torch.randn(V, D, generator=g, dtype=torch.float64),
    )


def _rotate(w, q, f_norm):
    g_f = 1.0 + f_norm
    changed, record = rewrite("mtp", BLOCK, q, g_f, w)
    return changed, record


def _error(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def _q():
    from qda_quant.steps.rotate import orthogonal

    return orthogonal(D, 0)


def test_the_rewrite_is_exact_when_the_norms_are_identity():
    w, (emb, x, head) = _weights(), _inputs()
    w["mtp.norm.weight"] = torch.zeros(D, dtype=torch.float64)
    w["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(D, dtype=torch.float64)
    w["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(D, dtype=torch.float64)
    f_norm = torch.zeros(D, dtype=torch.float64)
    q = _q()
    changed, record = _rotate(w, q, f_norm)
    assert record["norm_fit"]["constant"] == 1.0
    got = _rotated_logits(changed, emb, x, q, f_norm, head, 1.0)
    # the block's own norm normalizes the already-normalized hidden state, so its eps (~5e-7) is the floor
    assert _error(got, _source_logits(w, emb, x, f_norm, head)) < 1e-6


def test_only_the_last_norm_leaves_error_when_the_gains_are_random():
    w, (emb, x, head) = _weights(), _inputs()
    g = torch.Generator().manual_seed(2)
    f_norm = torch.randn(D, generator=g, dtype=torch.float64) * 0.1
    # the real model's two norms are strongly correlated (ratio std 0.043), not independent
    w["mtp.norm.weight"] = (1.0 + f_norm) * (
        1.0 + 0.02 * torch.randn(D, generator=g, dtype=torch.float64)
    ) - 1
    q = _q()
    changed, record = _rotate(w, q, f_norm)
    v = record["norm_fit"]["constant"]
    assert 0.01 < record["norm_fit"]["relative_residual"] < 0.1
    ref = _source_logits(w, emb, x, f_norm, head)
    got = _rotated_logits(changed, emb, x, q, f_norm, head, v)
    assert _error(got, ref) < 0.1
    assert (got.argmax(-1) == ref.argmax(-1)).float().mean() > 0.8


def test_the_rewrite_refuses_a_source_tensor_it_does_not_account_for():
    w = _weights()
    w["mtp.extra.weight"] = torch.zeros(D, D, dtype=torch.float64)
    with pytest.raises(Refused, match=r"mtp\.extra\.weight under the declared block mtp.*rotated basis"):
        _rotate(w, _q(), torch.zeros(D, dtype=torch.float64))


def test_the_rewrite_refuses_a_missing_declared_weight():
    w = _weights()
    del w["mtp.norm.weight"]
    with pytest.raises(Refused, match=r"source has no mtp\.norm\.weight for the declared block"):
        _rotate(w, _q(), torch.zeros(D, dtype=torch.float64))


def _real_mtp(hidden, inter, head_dim, seed=3):
    """The 15 tensors of Qwen3.5's MTP block, at tiny sizes."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * 0.1

    return {
        "mtp.fc.weight": r(hidden, 2 * hidden),
        "mtp.pre_fc_norm_embedding.weight": r(hidden),
        "mtp.pre_fc_norm_hidden.weight": r(hidden),
        "mtp.layers.0.input_layernorm.weight": r(hidden),
        "mtp.layers.0.post_attention_layernorm.weight": r(hidden),
        "mtp.layers.0.self_attn.q_proj.weight": r(hidden, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": r(hidden // 2, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": r(hidden // 2, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": r(hidden, hidden),
        "mtp.layers.0.self_attn.q_norm.weight": r(head_dim),
        "mtp.layers.0.self_attn.k_norm.weight": r(head_dim),
        "mtp.layers.0.mlp.gate_proj.weight": r(inter, hidden),
        "mtp.layers.0.mlp.up_proj.weight": r(inter, hidden),
        "mtp.layers.0.mlp.down_proj.weight": r(hidden, inter),
        "mtp.norm.weight": r(hidden),
    }


def test_a_text_class_plans_and_rotates_the_source_merger_and_mtp(tmp_path):
    from types import SimpleNamespace

    import transformers
    from qda_quant import REGISTRY
    from qda_quant.arch import describe
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.recipe import Recipe
    from qda_quant.target import Target
    from safetensors.torch import save_file

    from ..engine_util import save_tiny_tokenizer, write_ids
    from .test_rotate import _tiny_vlm

    m = _tiny_vlm(dtype=torch.float64)
    text_cfg = m.config.text_config
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    rows = [
        torch.randint(0, 128, (20,), generator=torch.Generator().manual_seed(i)).tolist() for i in range(5)
    ]
    write_ids(tmp_path / "c.jsonl", rows, sha)
    save_file(
        _real_mtp(text_cfg.hidden_size, text_cfg.intermediate_size, text_cfg.head_dim),
        tmp_path / "m" / "mtp.safetensors",
    )
    recipe = Recipe.from_dict(
        {
            "steps": [{"rotate": {}}, {"gptq": {}}],
            "calib": {"file": {"path": str(tmp_path / "c.jsonl"), "samples": 4, "seqlen": 16}},
        }
    )
    compile_plan(str(tmp_path / "m"), recipe, Format.parse("w4a16"), Target(), [torch.device("cpu")])

    text = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "m", dtype=torch.float64)
    graph = describe(text, verify=False)
    assert "mtp" in graph.extra["stream"]["blocks"]
    run = SimpleNamespace(plan=SimpleNamespace(source=str(tmp_path / "m")), tensors={}, info={})
    REGISTRY["rotate"]().apply_model(text, graph, run)
    source = _real_mtp(text_cfg.hidden_size, text_cfg.intermediate_size, text_cfg.head_dim)
    written = {k for k in run.tensors if k.startswith("mtp.")}
    assert written == set(source) - {
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",  # unchanged by the rotation, so the writer copies them
    }
    assert "model.visual.merger.linear_fc2.weight" in run.tensors
    assert run.info["source_blocks"]["mtp"]["norm_fit"]["constant"] > 0
