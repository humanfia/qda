"""A variant must differ from its base in exactly one class, and in that class by exactly the decoded type."""

from __future__ import annotations

import pytest
import torch
from qda_quant.gguf import classes as sensitivity
from qda_quant.gguf import variant
from qda_quant.gguf.columns import value_head_columns
from safetensors.torch import load_file

from ..tiny import model


def _anchor(tmp_path):
    m = model("qwen3_5")
    d = tmp_path / "anchor"
    m.save_pretrained(d)
    return d, m.config


def _base(tmp_path):
    anchor, cfg = _anchor(tmp_path)
    base = tmp_path / "base"
    return anchor, cfg, base, variant.prepare_base(anchor, base)


def _tensors(root):
    import json as _json

    idx = _json.loads((root / "model.safetensors.index.json").read_text())
    got = {}
    for _name, shard in idx["weight_map"].items():
        got.setdefault(shard, {}).update(load_file(str(root / shard)))
    return got, idx


def test_each_shard_holds_one_class(tmp_path) -> None:
    anchor, _, base, man = _base(tmp_path)
    _, idx = _tensors(base)
    assert set(idx["weight_map"]) == set().union(*(load_file(str(s)) for s in variant._shards(anchor)))
    for klass, got in man["classes"].items():
        for shard in got["shards"]:
            names = load_file(str(base / shard))
            assert names and all(variant.class_of(n) == klass for n in names), shard


def test_a_variant_changes_exactly_one_class(tmp_path) -> None:
    _, _, base, _ = _base(tmp_path)
    before, _ = _tensors(base)
    out = variant.build_variant(base, tmp_path / "v", "attn_q", "Q8_0")
    after, _ = _tensors(tmp_path / "v")
    assert set(out["rewritten_shards"]) <= set(after)
    changed = set()
    for shard, tensors in before.items():
        for name, t in tensors.items():
            got = after[shard][name]
            if variant.class_of(name) == "attn_q":
                want = variant.fake_quant(t, "Q8_0")
                assert torch.equal(got, want), name
                changed.add(name)
            else:
                assert torch.equal(got, t), name
    assert changed and all("q_proj" in n for n in changed)


def test_ssm_out_is_quantized_in_the_converter_order(tmp_path) -> None:
    _, cfg, base, _ = _base(tmp_path)
    before, _ = _tensors(base)
    name = next(
        n for n in before["model-ssm_out-00000.safetensors"] if n.endswith(".linear_attn.out_proj.weight")
    )
    source = before["model-ssm_out-00000.safetensors"][name]
    variant.build_variant(base, tmp_path / "v", "ssm_out", "Q8_0")
    got = _tensors(tmp_path / "v")[0]["model-ssm_out-00000.safetensors"][name]
    perm = value_head_columns(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
    want = variant.fake_quant(source[:, perm], "Q8_0")[:, perm.argsort()]
    assert torch.equal(got, want)
    # A head that is a whole number of quantization groups moves whole groups, so every grid type is invariant
    # (32 values per group; 16 in Q6_K). The permutation is still applied -- it is load-bearing when a head is
    # narrower than a group, as for a 4-wide head under Q8_0's 32-value block.
    narrow = value_head_columns(2, 64, 4)
    t = torch.randn(2, 256)
    assert not torch.equal(variant.fake_quant(t, "Q4_K"), variant.fake_quant(t, "Q4_K", columns=narrow))
    assert not torch.equal(variant.fake_quant(t, "Q8_0"), variant.fake_quant(t, "Q8_0", columns=narrow))


def test_a_variant_is_repeatable_and_chunk_invariant(tmp_path) -> None:
    _, _, base, _ = _base(tmp_path)
    a = variant.build_variant(base, tmp_path / "a", "ssm_alpha", "Q8_0")
    b = variant.build_variant(base, tmp_path / "b", "ssm_alpha", "Q8_0")
    assert a == b
    shard = base / "model-attn_q-00000.safetensors"
    t = load_file(str(shard))[next(n for n in load_file(str(shard)) if n.endswith("q_proj.weight"))]
    assert torch.equal(variant.fake_quant(t, "Q8_0", chunk=1), variant.fake_quant(t, "Q8_0", chunk=1 << 20))


def test_refusals(tmp_path) -> None:
    _, _, base, _ = _base(tmp_path)
    with pytest.raises(ValueError, match="fresh directory"):
        variant.build_variant(base, tmp_path / "base", "attn_q", "Q8_0")
    with pytest.raises(ValueError, match="no class"):
        variant.build_variant(base, tmp_path / "x", "no_such", "Q8_0")
    with pytest.raises(ValueError, match="not empty"):
        variant.prepare_base(base, tmp_path / "base")  # a base is not overwritten


def test_a_fixed_bucket_is_not_a_class_a_read_may_replace(tmp_path) -> None:
    # the base carries mtp/mtp_eh/visual/keeps too, but the plan never allocates one of them: a read that
    # replaced a bucket would measure no allocatable cell and could hide a model's fixed weights
    _, _, base, manifest = _base(tmp_path)
    excluded = sorted(set(manifest["classes"]) - set(sensitivity.CLASSES))
    assert excluded  # at least the float keeps
    for klass in excluded:
        with pytest.raises(ValueError, match="fixed bucket, not an allocatable class"):
            variant.build_variant(base, tmp_path / f"out-{klass}", klass, "Q8_0")
        assert not (tmp_path / f"out-{klass}").exists()
