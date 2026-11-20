import pytest
import torch
from qda_quant import REGISTRY
from qda_quant.arch import describe

from ..tiny import model


@pytest.mark.parametrize("kind", ["qwen3", "llama", "qwen3_moe", "qwen3_5"])
def test_permutation_keeps_the_float_function(kind):
    m = model(kind, dtype=torch.float64)
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    before = {n: p.clone() for n, p in m.named_parameters()}
    REGISTRY["permute"]().apply_model(m, describe(m, verify=False), None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-5 * ref.abs().max()
    moved = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert any("embed_tokens" in n for n in moved) and any("input_layernorm" in n for n in moved)
    assert any(("down_proj" in n) for n in moved)


def test_permutation_sorts_reader_columns():
    m = model("qwen3", dtype=torch.float64)
    g = describe(m, verify=False)
    REGISTRY["permute"](intermediate=False).apply_model(m, g, None)
    score = torch.stack([m.get_submodule(r).weight.abs().amax(0) for k, rs in g.extra["readers"].items()
                         if k.endswith("layernorm") for r in rs] + [m.lm_head.weight.abs().amax(0)]).amax(0)  # fmt: skip
    assert torch.all(score[:-1] >= score[1:])


def test_tied_embedding_is_permuted_once():
    import transformers

    from ..tiny import config

    cfg = config("qwen3")
    cfg.tie_word_embeddings = True
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float64).eval()
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    REGISTRY["permute"]().apply_model(m, describe(m, verify=False), None)
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-6 * ref.abs().max()


def test_tied_heads_refused_by_rotate_and_head():
    from qda_quant import Format, Refused, Target

    graph = describe(model("qwen3"), verify=False)
    import dataclasses

    tied = dataclasses.replace(graph, extra={**graph.extra, "tied": True})
    f = Format.parse("w4a16")
    for step in (REGISTRY["rotate"](), REGISTRY["head"]()):
        with pytest.raises(Refused, match="tied"):
            step.check(tied, f, Target().plan(f, "dense"))
