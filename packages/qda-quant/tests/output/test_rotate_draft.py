"""A draft rotated by ``qda rotate-draft`` computes what the original draft computed, in the rotated basis."""

import fnmatch
import json
import shutil

import huggingface_hub
import pytest
import torch
import torch.nn.functional as F
from qda_quant.output.rotate_draft import rotate
from qda_quant.step import Refused
from qda_quant.steps.rotate import orthogonal
from safetensors import safe_open
from safetensors.torch import save_file

D, VOCAB, TARGETS, LAYERS = 8, 16, 8, 2
KV, HD, INTER = 2, 4, 16
BLOCKS = TARGETS // LAYERS  # fc reads one hidden state per target layer
SEED, TOKENS = 3, 3


def _draft() -> dict[str, torch.Tensor]:
    """A tiny DFlash-shaped draft: fc over the target's hidden states, then attention and MLP layers."""
    gen = torch.Generator().manual_seed(0)
    t = {
        "fc.weight": torch.cat([torch.randn(D, D, generator=gen) + i for i in range(BLOCKS)], dim=1),
        "hidden_norm.weight": torch.ones(D),
        "norm.weight": torch.ones(D),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        t |= {
            p + "input_layernorm.weight": torch.ones(D),
            p + "post_attention_layernorm.weight": torch.ones(D),
            p + "self_attn.q_proj.weight": torch.randn(D, D, generator=gen),
            p + "self_attn.k_proj.weight": torch.randn(KV * HD, D, generator=gen),
            p + "self_attn.v_proj.weight": torch.randn(KV * HD, D, generator=gen),
            p + "self_attn.o_proj.weight": torch.randn(D, KV * HD, generator=gen),
            p + "self_attn.q_norm.weight": torch.ones(HD),
            p + "self_attn.k_norm.weight": torch.ones(HD),
            p + "mlp.gate_proj.weight": torch.randn(INTER, D, generator=gen),
            p + "mlp.up_proj.weight": torch.randn(INTER, D, generator=gen),
            p + "mlp.down_proj.weight": torch.randn(D, INTER, generator=gen),
        }
    return t


def _forward(t: dict[str, torch.Tensor], blocks: list[torch.Tensor]) -> torch.Tensor:
    """One pass in fp64: the draft's fc over `blocks`, its layers with causal attention, then its head."""

    def norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).sqrt() * gain

    x = norm(torch.cat(blocks, 1) @ t["fc.weight"].T, t["hidden_norm.weight"])
    mask = torch.triu(torch.ones(TOKENS, TOKENS, dtype=torch.bool), 1)
    for i in range(LAYERS):
        p = f"layers.{i}."
        h = norm(x, t[p + "input_layernorm.weight"])
        q = (h @ t[p + "self_attn.q_proj.weight"].T).view(TOKENS, KV, HD) * t[p + "self_attn.q_norm.weight"]
        k = (h @ t[p + "self_attn.k_proj.weight"].T).view(TOKENS, KV, HD) * t[p + "self_attn.k_norm.weight"]
        v = (h @ t[p + "self_attn.v_proj.weight"].T).view(TOKENS, KV, HD)
        scores = torch.einsum("thd,shd->hts", q, k).masked_fill(mask, float("-inf")) / HD**0.5
        out = torch.einsum("hts,shd->thd", scores.softmax(-1), v).reshape(TOKENS, D)
        x = x + out @ t[p + "self_attn.o_proj.weight"].T
        m = norm(x, t[p + "post_attention_layernorm.weight"])
        x = (
            x
            + (F.silu(m @ t[p + "mlp.gate_proj.weight"].T) * (m @ t[p + "mlp.up_proj.weight"].T))
            @ t[p + "mlp.down_proj.weight"].T
        )
    return norm(x, t["norm.weight"]) @ t["lm_head.weight"].T


def _rotation(seeds: list[int]) -> torch.Tensor:
    """The residual rotation a checkpoint records: its seeds' matrices, in order."""
    q = torch.eye(D, dtype=torch.float64)
    for seed in seeds:
        q = q @ orthogonal(D, seed)
    return q


def _blind_rows() -> torch.Tensor:
    """Rows ``e`` with ``e @ Q == e @ Q^T``: the null space of ``Q - Q^T``, which no orientation check sees."""
    q = _rotation([SEED])
    _, s, vh = torch.linalg.svd((q - q.T).T)
    rows = vh[s < 1e-10]
    assert 0 < len(rows) < D
    return rows


def _case(root, seed: int = SEED, transposed: bool = False) -> tuple:
    """(base dir, checkpoint dir) of a tiny rotated model whose embedding is `base embed @ rotation(seed)`."""
    base, ckpt = root / "base", root / "ckpt"
    gen = torch.Generator().manual_seed(1)
    embed = torch.randn(VOCAB, D, generator=gen)
    for d in (base, ckpt):
        d.mkdir()
    save_file(
        {"model.embed_tokens.weight": embed, "lm_head.weight": torch.randn(VOCAB, D, generator=gen)},
        str(base / "model.safetensors"),
    )
    (base / "config.json").write_text("{}")
    q = _rotation([seed])
    save_file(
        {"model.embed_tokens.weight": (embed.double() @ (q.T if transposed else q)).float()},
        str(ckpt / "model.safetensors"),
    )
    (ckpt / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": D, "num_hidden_layers": TARGETS}})
    )
    recipe = {
        "model": "tiny",
        "steps": [{"rotate": {"residual": True, "seed": SEED}}],
    }
    recipe["provenance"] = {"source_snapshot": str(base)}
    (ckpt / "resolved.yaml").write_text(json.dumps(recipe))
    return base, ckpt


def _write_draft(path, **cfg) -> dict[str, torch.Tensor]:
    """Write a tiny DFlash-shaped draft at `path`; ``cfg`` overrides its config; returns its tensors."""
    path.mkdir()
    tensors = _draft()
    save_file(tensors, str(path / "model.safetensors"))
    base = {"hidden_size": D, "vocab_size": VOCAB, "num_target_layers": TARGETS}
    (path / "config.json").write_text(json.dumps(base | cfg))
    return tensors


def _read(path) -> dict[str, torch.Tensor]:
    with safe_open(str(path / "model.safetensors"), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 (safe_open is not a mapping)


def _hub(monkeypatch, tmp_path, published) -> list:
    """Serve `published` as a Hub repo through ``snapshot_download``; return the requested filter per call."""
    calls = []

    def download(repo, revision=None, allow_patterns=None):
        assert repo == "z-lab/Fake-DFlash"
        snap = tmp_path / f"hub-{len(calls)}"
        calls.append(allow_patterns)
        for f in published.rglob("*"):
            rel = f.relative_to(published).as_posix()
            if f.is_file() and (
                allow_patterns is None or any(fnmatch.fnmatch(rel, p) for p in allow_patterns)
            ):
                (snap / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, snap / rel)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return calls


def test_rotated_draft_keeps_its_logits(tmp_path):
    base, ckpt = _case(tmp_path)
    tensors = _write_draft(tmp_path / "draft")

    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out")
    got = {k: v.double() for k, v in _read(out).items()}
    q = _rotation([SEED])
    assert torch.allclose(got["lm_head.weight"], _read(base)["lm_head.weight"].double() @ q, atol=1e-5)
    assert json.loads((out / "config.json").read_text())["has_own_lm_head"] is True

    gen = torch.Generator().manual_seed(2)  # distinct blocks: a whole-matrix rotation would not pass
    blocks = [torch.randn(TOKENS, D, generator=gen, dtype=torch.float64) for _ in range(BLOCKS)]
    head = _read(base)["lm_head.weight"].double()
    want = _forward({k: v.double() for k, v in tensors.items()} | {"lm_head.weight": head}, blocks)
    assert torch.allclose(_forward(got, [b @ q for b in blocks]), want, rtol=1e-5, atol=1e-4)


def test_finds_a_transposed_rotation(tmp_path):
    """A checkpoint built with Q's transpose is rotated with Q's transpose, and its record says so."""
    base, ckpt = _case(tmp_path, transposed=True)
    _write_draft(tmp_path / "draft")

    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out")
    assert json.loads((out / "rotate-draft.json").read_text())["q_transposed"] is True
    q = _rotation([SEED]).T
    head = _read(base)["lm_head.weight"].double()
    assert torch.allclose(_read(out)["lm_head.weight"].double(), head @ q, atol=1e-5)


def test_refuses_a_rotation_its_record_does_not_reproduce(tmp_path):
    _base, ckpt = _case(tmp_path, seed=SEED + 5)  # built with another matrix than the seed it records
    _write_draft(tmp_path / "draft")
    with pytest.raises(Refused, match="both miss the embedding"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out")


def test_rows_blind_to_the_orientation_do_not_decide_it(tmp_path):
    """The first 2,048 rows read the same under Q and Q^T; the later rows do not, and they decide."""
    base, ckpt = _case(tmp_path, transposed=True)
    blind, gen = _blind_rows(), torch.Generator().manual_seed(41)
    embed = torch.cat(
        [
            torch.randn(2048, len(blind), generator=gen, dtype=torch.float64) @ blind,
            torch.randn(16, D, generator=gen, dtype=torch.float64),
        ]
    )
    q = _rotation([SEED])
    save_file(
        {
            "model.embed_tokens.weight": embed.float(),
            "lm_head.weight": torch.randn(len(embed), D, generator=gen),
        },
        str(base / "model.safetensors"),
    )
    save_file({"model.embed_tokens.weight": (embed @ q.T).float()}, str(ckpt / "model.safetensors"))
    _write_draft(tmp_path / "draft", vocab_size=len(embed))

    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out")
    assert json.loads((out / "rotate-draft.json").read_text())["q_transposed"] is True
    head = _read(base)["lm_head.weight"].double()
    assert torch.allclose(_read(out)["lm_head.weight"].double(), head @ q.T, atol=1e-5)


def test_refuses_an_embedding_that_fits_both_orientations(tmp_path):
    """An embedding of only blind rows cannot say which orientation the checkpoint used."""
    base, ckpt = _case(tmp_path)
    blind, gen = _blind_rows(), torch.Generator().manual_seed(42)
    embed = torch.randn(VOCAB, len(blind), generator=gen, dtype=torch.float64) @ blind
    q = _rotation([SEED])
    for d, e in ((base, embed), (ckpt, embed @ q.T)):  # blind rows: either orientation reads the same
        save_file({"model.embed_tokens.weight": e.float()}, str(d / "model.safetensors"))
    _write_draft(tmp_path / "draft")
    with pytest.raises(Refused, match="basis is ambiguous"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_fresh_hub_draft_fetches_and_carries_every_file(tmp_path, monkeypatch):
    """A weight-only filter hides the sidecar and drops the files beside the weights of a fresh Hub snapshot."""
    _base, ckpt = _case(tmp_path)
    published = tmp_path / "published"
    _write_draft(published)
    torch.save({"mask_token_id": 1, "embedding": torch.zeros(D)}, published / "mask_embedding.pt")
    calls = _hub(monkeypatch, tmp_path, published)
    with pytest.raises(Refused, match="not known to read only through fc"):  # mask: original basis only
        rotate(ckpt, "z-lab/Fake-DFlash", tmp_path / "out")
    assert calls == [None]
    assert not (tmp_path / "out").exists()

    (published / "mask_embedding.pt").unlink()
    (published / "README.md").write_text("---\nlibrary_name: transformers\n---\n")
    (published / "assets").mkdir()
    (published / "assets" / "mask.png").write_bytes(b"\x89PNG\r\n")
    (published / "dflash.py").write_text("# published model\n")
    rotate(ckpt, "z-lab/Fake-DFlash", tmp_path / "out")
    assert (tmp_path / "out" / "README.md").read_text() == "---\nlibrary_name: transformers\n---\n"
    assert (tmp_path / "out" / "assets" / "mask.png").read_bytes() == b"\x89PNG\r\n"
    assert (tmp_path / "out" / "dflash.py").read_text() == "# published model\n"


def test_the_command_takes_bare_directories_and_a_new_output_parent(tmp_path, monkeypatch):
    """``--checkpoint ckpt`` names the directory, not a hub repo; ``--out`` may sit under a new directory."""
    import argparse

    from qda_quant.output.rotate_draft import add_parser

    _case(tmp_path)
    _write_draft(tmp_path / "draft")
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers(required=True))
    monkeypatch.chdir(tmp_path)
    a = ap.parse_args(["rotate-draft", "--checkpoint", "ckpt", "--draft", "draft", "--out", "new/out"])
    assert a.run(a) == 0
    assert (tmp_path / "new" / "out" / "model.safetensors").is_file()


# DFlash2 and DSpark drafts: a vLLM-shaped pass. The target's hidden states enter only through fc as context K/V;
# the draft's tokens enter through the embedding, its own or the target's, and leave through the output head,
# its own or the target's (vLLM shares the target's when the draft has none).
RANK, IDS, PREV = 3, torch.tensor([2, 5, 7]), torch.tensor([1, 4, 9])
LAYER_IDS = [1, 3, 5, 7]  # BLOCKS target layers, all below TARGETS


def _norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return x / x.pow(2).mean(-1, keepdim=True).sqrt() * gain


def _vllm_pass(t: dict, blocks: list, embed: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Logits (and, for DSpark, Markov-biased logits plus confidence) of one draft block, in fp64."""
    t = {k: v.double() if v.is_floating_point() else v for k, v in t.items()}
    embed, head = t.get("embed_tokens.weight", embed.double()), t.get("lm_head.weight", head.double())
    ctx = _norm(torch.cat(blocks, 1) @ t["fc.weight"].T, t["hidden_norm.weight"])
    x, n = embed[IDS], len(IDS)
    for i in range(LAYERS):
        p = f"layers.{i}."
        h = _norm(x, t[p + "input_layernorm.weight"])
        both = torch.cat([ctx, h])  # context K/V reads hidden_norm's output, the block's reads its own
        q = (h @ t[p + "self_attn.q_proj.weight"].T).view(n, KV, HD) * t[p + "self_attn.q_norm.weight"]
        k = (both @ t[p + "self_attn.k_proj.weight"].T).view(-1, KV, HD) * t[p + "self_attn.k_norm.weight"]
        v = (both @ t[p + "self_attn.v_proj.weight"].T).view(-1, KV, HD)
        scores = torch.einsum("thd,shd->hts", q, k) / HD**0.5
        x = (
            x
            + torch.einsum("hts,shd->thd", scores.softmax(-1), v).reshape(n, D)
            @ t[p + "self_attn.o_proj.weight"].T
        )
        if p + "mlp_conv.base_kernel" in t:  # a per-channel kernel, as DFlash2's convolutions apply
            x = x * t[p + "mlp_conv.base_kernel"][0, 0]
        m = _norm(x, t[p + "post_attention_layernorm.weight"])
        mlp = F.silu(m @ t[p + "mlp.gate_proj.weight"].T) * (m @ t[p + "mlp.up_proj.weight"].T)
        x = x + mlp @ t[p + "mlp.down_proj.weight"].T
    hidden = _norm(x, t["norm.weight"])
    logits = hidden @ head.T
    if "markov_head.markov_w1.weight" not in t:
        return logits
    e = t["markov_head.markov_w1.weight"][PREV]
    conf = torch.cat([hidden, e], -1) @ t["confidence_head.proj.weight"].T + t["confidence_head.proj.bias"]
    return torch.cat([logits + e @ t["markov_head.markov_w2.weight"].T, conf], -1)


def _variant(gains: bool = False, conv: bool = False, dspark: bool = False, reduced: int = 0) -> dict:
    """The tiny draft with trained norm gains, a per-channel kernel, or DSpark's heads (own embed, reduced head)."""
    t, gen = _draft(), torch.Generator().manual_seed(5)
    if gains:
        t |= {
            k: 1 + 0.3 * torch.randn(D, generator=gen)
            for k, v in t.items()
            if k.endswith("norm.weight") and v.shape == (D,)
        }
    for i in range(LAYERS if conv else 0):
        t[f"layers.{i}.mlp_conv.base_kernel"] = 1 + 0.3 * torch.randn(2, 2, D, generator=gen)
    if dspark:
        out = reduced or VOCAB
        t |= {
            "embed_tokens.weight": torch.randn(VOCAB, D, generator=gen),
            "markov_head.markov_w1.weight": torch.randn(VOCAB, RANK, generator=gen),
            "markov_head.markov_w2.weight": torch.randn(out, RANK, generator=gen),
            "confidence_head.proj.weight": torch.randn(1, D + RANK, generator=gen),
            "confidence_head.proj.bias": torch.randn(1, generator=gen),
        }
    if reduced:
        t |= {
            "lm_head.weight": torch.randn(reduced, D, generator=gen),
            "d2t": torch.arange(reduced),
            "t2d": torch.arange(VOCAB) < reduced,
        }
    return t


def _write(path, tensors: dict, **cfg) -> dict:
    path.mkdir()
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"hidden_size": D, "vocab_size": VOCAB} | cfg))
    return tensors


def _same_logits(base, ckpt, tensors: dict, out) -> None:
    """The rotated pair (rotated draft, checkpoint) and the original pair (draft, base) give the same output."""
    q = _rotation([SEED])
    gen = torch.Generator().manual_seed(2)  # distinct blocks: a whole-matrix rotation would not pass
    blocks = [torch.randn(len(IDS) + 2, D, generator=gen, dtype=torch.float64) for _ in range(BLOCKS)]
    b = _read(base)
    want = _vllm_pass(tensors, blocks, b["model.embed_tokens.weight"], b["lm_head.weight"])
    # the checkpoint's own head: a rotated draft carries its own and must not read this one
    rotated_head = torch.randn(VOCAB, D, generator=gen)
    got = _vllm_pass(
        _read(out), [x @ q for x in blocks], _read(ckpt)["model.embed_tokens.weight"], rotated_head
    )
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4)


DFLASH = {"architectures": ["DFlashDraftModel"], "dflash_config": {"target_layer_ids": LAYER_IDS}}


def test_flat_dflash_moves_into_the_rotated_basis(tmp_path):
    """z-lab's DFlash drafts have unit norm gains: every weight takes Q, and the target's embedding stays shared."""
    base, ckpt = _case(tmp_path)
    tensors = _write(tmp_path / "d-dflash", _draft(), **DFLASH)
    out = rotate(ckpt, tmp_path / "d-dflash", tmp_path / "out-dflash")
    record = json.loads((out / "rotate-draft.json").read_text())
    assert (record["basis"], record["target_layer_ids"], record["q_from"]) == ("rotated", LAYER_IDS, "seeds")
    assert "embed_tokens.weight" not in _read(out)
    _same_logits(base, ckpt, tensors, out)


@pytest.mark.parametrize(
    "arch, kind",
    [
        ("DFlashDraftModel", {"gains": True}),
        ("DFlash2DraftModel", {"conv": True}),
        ("Qwen3DSparkModel", {"gains": True, "dspark": True}),
        ("DSparkDraftModel", {"gains": True, "dspark": True, "reduced": 6}),
    ],
)
def test_a_draft_that_does_not_commute_keeps_its_basis(tmp_path, arch, kind):
    """Trained gains or a per-channel kernel: only fc takes Q, every other tensor is carried byte for byte."""
    base, ckpt = _case(tmp_path)
    tensors = _write(tmp_path / "draft", _variant(**kind), **DFLASH | {"architectures": [arch]})
    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out-dspark")
    got, record = _read(out), json.loads((out / "rotate-draft.json").read_text())
    assert record["basis"] == "original" and record["original_because"]
    assert all(torch.equal(got[k], v) for k, v in tensors.items() if k != "fc.weight")
    assert not torch.equal(got["fc.weight"], tensors["fc.weight"])
    b = _read(base)
    if "embed_tokens.weight" not in tensors:  # the draft reads the base's embedding, not the rotated one
        assert torch.equal(got["embed_tokens.weight"], b["model.embed_tokens.weight"])
    if "lm_head.weight" not in tensors:
        assert torch.equal(got["lm_head.weight"], b["lm_head.weight"])
    _same_logits(base, ckpt, tensors, out)


@pytest.mark.parametrize("reduced", [0, 6])
def test_a_flat_dspark_rotates_its_confidence_head(tmp_path, reduced):
    """With unit gains a DSpark draft moves into the rotated basis; the Markov head and token maps do not move."""
    base, ckpt = _case(tmp_path)
    tensors = _write(
        tmp_path / "draft", _variant(dspark=True, reduced=reduced), architectures=["Qwen3DSparkModel"]
    )
    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out-dspark")
    got = _read(out)
    assert json.loads((out / "rotate-draft.json").read_text())["basis"] == "rotated"
    assert all(
        torch.equal(got[k], tensors[k]) for k in tensors if k.startswith("markov_head") or k in ("d2t", "t2d")
    )
    _same_logits(base, ckpt, tensors, out)


def test_a_draft_of_unknown_reach_is_refused_rather_than_kept(tmp_path):
    """Without a known architecture, a tensor other than fc could read the target: nothing is written."""
    _base, ckpt = _case(tmp_path)
    _write(tmp_path / "draft", _variant(gains=True))
    with pytest.raises(Refused, match="not known to read only through fc"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_record_says_which_rotations_a_reader_must_assume(tmp_path):
    """Residual seeds come from resolved.yaml (the one owner); a value-head-only step or no record says so."""
    from qda_quant.output.rotate_draft import residual_seeds

    _base, ckpt = _case(tmp_path)  # its record: steps: [{rotate: {residual: True, seed: SEED}}]
    assert residual_seeds(ckpt) == [SEED]
    record = json.loads((ckpt / "resolved.yaml").read_text())
    record["steps"] = [{"rotate": {"residual": False, "seed": 5}}]  # value heads only: a draft needs nothing
    (ckpt / "resolved.yaml").write_text(json.dumps(record))
    assert residual_seeds(ckpt) == []
    (ckpt / "resolved.yaml").unlink()  # a published checkpoint: no record, nothing claimed
    assert residual_seeds(ckpt) is None


def test_a_checkpoint_without_a_record_is_fitted_from_its_embedding(tmp_path):
    """A published checkpoint has no resolved.yaml: Q is fitted to its embedding, against the base it names."""
    base, ckpt = _case(tmp_path)
    (ckpt / "resolved.yaml").unlink()
    tensors = _write(tmp_path / "draft", _variant(gains=True), **DFLASH)
    with pytest.raises(Refused, match="pass --base"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out")
    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out-dflash", base=base)
    record = json.loads((out / "rotate-draft.json").read_text())
    assert record["q_from"] == "embedding" and record["q_relative_error"] < 1e-5
    _same_logits(base, ckpt, tensors, out)

    other = tmp_path / "other"  # a base the checkpoint was not built from
    other.mkdir()
    gen = torch.Generator().manual_seed(9)
    save_file(
        {"model.embed_tokens.weight": torch.randn(VOCAB, D, generator=gen)}, str(other / "model.safetensors")
    )
    (other / "config.json").write_text("{}")
    with pytest.raises(Refused, match="no rotation of"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out-2", base=other)


def test_an_unrotated_checkpoint_needs_no_rotated_draft(tmp_path):
    base, ckpt = _case(tmp_path)
    (ckpt / "resolved.yaml").unlink()
    shutil.copy2(base / "model.safetensors", ckpt / "model.safetensors")
    _write(tmp_path / "draft", _draft(), **DFLASH)
    with pytest.raises(Refused, match="not rotated; use the draft as is"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out", base=base)


def test_training_state_is_left_behind_and_a_mask_is_carried(tmp_path):
    """Optimizer state no server reads is not copied; a mask embedding keeps the draft in its own basis."""
    base, ckpt = _case(tmp_path)
    tensors = _write(tmp_path / "draft", _draft(), **DFLASH)
    torch.save({"state": {}}, tmp_path / "draft" / "optimizer_state_dict.pt")
    torch.save({"mask_token_id": 1, "embedding": torch.ones(D)}, tmp_path / "draft" / "mask_embedding.pt")
    out = rotate(ckpt, tmp_path / "draft", tmp_path / "out-dflash")
    record = json.loads((out / "rotate-draft.json").read_text())
    assert (
        record["not_copied"] == ["optimizer_state_dict.pt"] and not (out / "optimizer_state_dict.pt").exists()
    )
    assert record["basis"] == "original"
    assert (out / "mask_embedding.pt").read_bytes() == (tmp_path / "draft" / "mask_embedding.pt").read_bytes()
    _same_logits(base, ckpt, tensors, out)


def test_refuses_target_layers_the_model_does_not_have_and_a_second_rotation(tmp_path):
    _base, ckpt = _case(tmp_path)
    _write(tmp_path / "far", _draft(), dflash_config={"target_layer_ids": [1, 3, 5, TARGETS]})
    with pytest.raises(Refused, match="reads target layers"):
        rotate(ckpt, tmp_path / "far", tmp_path / "out")
    _write(tmp_path / "few", _draft(), dflash_config={"target_layer_ids": [1, 3]})
    with pytest.raises(Refused, match="does not read 2 hidden states"):
        rotate(ckpt, tmp_path / "few", tmp_path / "out")
    _write(tmp_path / "draft", _draft(), **DFLASH)
    once = rotate(ckpt, tmp_path / "draft", tmp_path / "once")
    with pytest.raises(Refused, match="wrote it already"):
        rotate(ckpt, once, tmp_path / "twice")


def test_refuses_a_draft_that_ships_inside_the_target(tmp_path):
    _base, ckpt = _case(tmp_path)
    _write(tmp_path / "draft", _draft(), architectures=["DeepseekV4ForCausalLM"])
    with pytest.raises(Refused, match="ships inside the target checkpoint"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out")


def test_a_changed_but_unrotated_embedding_is_not_taken_for_a_rotation(tmp_path):
    """A 3-bit start can change the embedding itself; the best fit is then near the identity, and that is no Q."""
    base, ckpt = _case(tmp_path)
    (ckpt / "resolved.yaml").unlink()
    gen = torch.Generator().manual_seed(11)
    e = _read(base)["model.embed_tokens.weight"]
    save_file(
        {"model.embed_tokens.weight": e + 0.1 * torch.randn(e.shape, generator=gen)},
        str(ckpt / "model.safetensors"),
    )
    _write(tmp_path / "draft", _draft(), **DFLASH)
    with pytest.raises(Refused, match="but by no rotation, so it is not rotated"):
        rotate(ckpt, tmp_path / "draft", tmp_path / "out", base=base)
