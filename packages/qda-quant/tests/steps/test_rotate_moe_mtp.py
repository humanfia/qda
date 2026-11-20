"""Rotate the real MoE decoder path and its source-only MTP, including both routing gates."""

import copy
from types import SimpleNamespace

import pytest
import torch
import transformers
from qda_quant import REGISTRY
from qda_quant.arch import describe
from qda_quant.step import Refused
from qda_quant.steps._source_block import rewrite
from qda_quant.steps.rotate import orthogonal
from safetensors.torch import save_file
from torch import nn

from ..tiny import config


class _MTP(nn.Module):
    def __init__(self, text):
        super().__init__()
        d = text.config.hidden_size
        self.fc = nn.Linear(2 * d, d, bias=False, dtype=torch.float64)
        self.pre_fc_norm_embedding = copy.deepcopy(text.model.norm)
        self.pre_fc_norm_hidden = copy.deepcopy(text.model.norm)
        self.layers = nn.ModuleList([copy.deepcopy(text.model.layers[-1])])
        self.norm = copy.deepcopy(text.model.norm)

    def forward(self, emb, hidden, rotary, positions):
        z = self.fc(torch.cat([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden)], -1))
        length = z.shape[1]
        mask = torch.full((length, length), float("-inf"), dtype=z.dtype).triu(1)[None, None]
        z = self.layers[0](z, position_embeddings=rotary(z, positions), attention_mask=mask)
        return self.norm(z)


def _models():
    cfg = config("qwen3_5_moe")
    cfg._attn_implementation = cfg._experts_implementation = "eager"
    torch.manual_seed(7)
    text = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float64).eval()
    mtp = _MTP(text).eval()
    with torch.no_grad():
        # All internal norms are nonconstant; only the two approximate boundaries have identity gains.
        for model in (text, mtp):
            for name, p in model.named_parameters():
                if "norm" in name:
                    p.uniform_(-0.2, 0.2)
        text.model.norm.weight.zero_()
        mtp.norm.weight.zero_()
    return text, mtp


@torch.no_grad()
def test_rotate_preserves_moe_main_and_mtp_logits_and_exports_every_expert(tmp_path):
    from qda_quant.format import Format
    from qda_quant.intake.source_names import under
    from qda_quant.output.export import Writer
    from qda_quant.recipe import Recipe
    from qda_quant.target import Target

    text, mtp = _models()
    source = tmp_path / "source"
    text.save_pretrained(source)
    original = {f"mtp.{n}": p for n, p in mtp.state_dict().items()}
    save_file(original, source / "mtp.safetensors")
    graph = describe(text, verify=False)
    block = graph.extra["stream"]["blocks"]["mtp"]
    ids = torch.randint(0, 128, (2, 16))
    positions = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
    before = text(ids, output_hidden_states=True, use_cache=False)
    main_ref = before.logits
    emb = text.get_input_embeddings()(ids)
    mtp_ref = text.lm_head(mtp(emb, before.hidden_states[-1], text.model.rotary_emb, positions))

    run = SimpleNamespace(plan=SimpleNamespace(source=str(source)), tensors={}, info={})
    REGISTRY["rotate"]().apply_model(text, graph, run)
    after = text(ids, output_hidden_states=True, use_cache=False)
    mtp.load_state_dict({n: run.tensors.get(f"mtp.{n}", original[f"mtp.{n}"]) for n in mtp.state_dict()})
    mtp_out = text.lm_head(
        mtp(text.get_input_embeddings()(ids), after.hidden_states[-1], text.model.rotary_emb, positions)
    )
    main_error = ((after.logits - main_ref).abs().max() / main_ref.abs().max()).item()
    mtp_error = ((mtp_out - mtp_ref).abs().max() / mtp_ref.abs().max()).item()
    print(f"MoE main max relative error: {main_error:.9e}; MTP: {mtp_error:.9e}")
    assert main_error < 1e-5
    assert mtp_error < 1e-5
    # Both final gains are 1 in the fixture, so the constant fit is exact and the record must say so.
    fit = run.info["source_blocks"]["mtp"]["norm_fit"]
    assert fit["constant"] == 1.0
    assert fit["gain_residue"] == 0.0 and fit["worst_channel"] == 0.0
    assert "constant" in fit["exact_when"]
    assert run.info["source_blocks"]["mtp"]["pre_fc_rms"]["exact_when"]
    assert set(run.tensors) & set(original) == set(original) - {f"{n}.weight" for n in block.kept}

    # Exercise the export channel: fused names must survive, and no unchanged expert may be copied over them.
    writer = Writer(tmp_path / "out", source, graph, Format.parse("w4a16"), Target())
    writer.add_source_tensors(run.tensors)
    recipe = Recipe.from_dict(
        {"steps": [{"rotate": {}}], "calib": {"file": {"path": "unused.jsonl", "samples": 1, "seqlen": 16}}}
    )
    writer.finish(text.config, recipe, {})
    written = under(tmp_path / "out", "mtp")
    assert set(written) == set(original)
    for n, p in run.tensors.items():
        if n.startswith("mtp."):
            torch.testing.assert_close(written[n], p, rtol=0, atol=0)


def test_moe_mtp_record_carries_the_approximation_error(tmp_path):
    """ORDER-1747: an approximate MTP must leave its error in build.json, not only in a log line."""
    text, mtp = _models()
    with torch.no_grad():
        mtp.norm.weight.uniform_(-0.2, 0.2)  # a nonconstant gain: g_m/g_f is no longer flat
    source = tmp_path / "source"
    text.save_pretrained(source)
    save_file({f"mtp.{n}": p for n, p in mtp.state_dict().items()}, source / "mtp.safetensors")
    block = describe(text, verify=False).extra["stream"]["blocks"]["mtp"]
    g_f = text.model.norm.weight.double() + 1.0
    _, record = rewrite(
        "mtp",
        block,
        orthogonal(g_f.numel(), 0),
        g_f,
        {"mtp." + k: v for k, v in {n: p for n, p in mtp.state_dict().items()}.items()},
    )
    fit = record["norm_fit"]
    assert fit["gain_residue"] > 0 and fit["worst_channel"] > 0
    assert fit["worst_channel"] >= fit["gain_residue"]
    assert "served logits unchanged" in record["draft"]


@pytest.mark.parametrize("fault", ["missing", "shape", "extra"])
def test_moe_source_declaration_refuses_incomplete_or_incompatible_weights(fault):
    text, mtp = _models()
    block = describe(text, verify=False).extra["stream"]["blocks"]["mtp"]
    tensors = {f"mtp.{n}": p for n, p in mtp.state_dict().items()}
    name = "mtp.layers.0.mlp.experts.gate_up_proj"
    if fault == "missing":
        del tensors[name]
        message = "source has no mtp.layers.0.mlp.experts.gate_up_proj"
    elif fault == "shape":
        tensors[name] = tensors[name][..., :-1]
        message = "source weight of reader.*has shape"
    else:
        tensors["mtp.layers.0.mlp.experts.bias"] = torch.zeros(8)
        message = "rewrite accounts for weights only"
    with pytest.raises(Refused, match=message):
        rewrite("mtp", block, orthogonal(128, 0), torch.ones(128), tensors)


@pytest.mark.parametrize(
    "mtp_weights,residual,indexed",
    [
        ("complete", True, False),
        ("complete", False, False),
        ("gate_up_proj", True, False),
        ("down_proj", True, False),
        ("fused_only", True, True),
        ("fused_only", False, False),
        ("absent", True, False),
    ],
)
def test_plan_accepts_moe_mtp_r1_and_warns_it_is_approximate(
    tmp_path, mtp_weights, residual, indexed, caplog
):
    import json

    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.recipe import Recipe
    from qda_quant.target import Target

    text, mtp = _models()
    text.save_pretrained(tmp_path)
    if mtp_weights != "absent":
        tensors = {f"mtp.{n}": p for n, p in mtp.state_dict().items()}
        if mtp_weights != "complete":
            tensors = {
                n: p
                for n, p in tensors.items()
                if n.startswith("mtp.layers.0.mlp.experts.")
                and (mtp_weights == "fused_only" or n.endswith(mtp_weights))
            }
        if indexed:
            # Presence comes from the native names in the index alone; the shard is still needed to price
            # the tensors, so it is written under the name the index gives it.
            save_file(tensors, tmp_path / "mtp.safetensors")
            (tmp_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(tensors, "mtp.safetensors")})
            )
        else:
            save_file(tensors, tmp_path / "mtp.safetensors")
    recipe = Recipe.from_dict(
        {
            "steps": [{"rotate": {"residual": residual}}],
            "calib": {"file": {"path": "unused.jsonl", "samples": 1, "seqlen": 16}},
        }
    )
    args = (str(tmp_path), recipe, Format.parse("w4a16-g64"), Target(), [torch.device("cpu")])
    with caplog.at_level("WARNING"):
        compile_plan(*args)  # ORDER-1747: an approximate MTP is built, not refused
    warned = [
        r for r in caplog.records if r.levelname == "WARNING" and "MoE MTP is exact only up to" in r.message
    ]
    assert bool(warned) == (mtp_weights != "absent" and residual)
