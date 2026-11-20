"""A read writes the variant, the scorer's result and the row, and refuses what would mislabel them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from qda_quant.gguf import variant
from qda_quant.gguf.read import _default_scorer, read
from qda_quant.output.vllm_env import DEFAULT_TP

from ..tiny import model


def test_the_default_scorer_is_the_installed_kl_scorer():
    from qda_kl.score import score

    assert _default_scorer() is score


@dataclass
class _Scored:
    kl: float = 0.004
    vllm: str = "0.33.0"
    deterministic: bool | None = True
    validate_max_abs_diff: float = 0.0
    tokenizer_sha256: str = "tok"
    serve: dict = field(default_factory=dict)


class _ScoreModel:
    """Stands in for `qda-kl`'s `Score`: a pydantic model, so `dataclasses.asdict` cannot read it."""

    def __init__(self) -> None:
        self.kl = 0.004
        self.vllm = "0.33.0"
        self.deterministic: bool | None = True
        self.validate_max_abs_diff = 0.0
        self.tokenizer_sha256 = "tok"
        self.serve = {"args": {"dtype": "bfloat16"}}

    def model_dump(self, mode: str = "python") -> dict:
        assert mode == "json"
        return {
            "kl": self.kl,
            "vllm": self.vllm,
            "deterministic": self.deterministic,
            "validate_max_abs_diff": self.validate_max_abs_diff,
            "tokenizer_sha256": self.tokenizer_sha256,
            "serve": self.serve,
            "reference": {"content_sha256": "94a7ceb1"},
        }


def _base(tmp_path):
    anchor = tmp_path / "anchor"
    model("qwen3_5").save_pretrained(anchor)
    base = tmp_path / "base"
    variant.prepare_base(anchor, base)
    return base


def _scorer(seen):
    def score(checkpoint, ref, *, subset, tp, check):
        seen.append({"checkpoint": checkpoint, "ref": ref, "subset": subset, "tp": tp, "check": check})
        return _Scored(serve={"args": {"dtype": "bfloat16"}})

    return score


def test_a_cell_read_writes_the_variant_result_and_row(tmp_path) -> None:
    base = _base(tmp_path)
    out, seen = tmp_path / "out", []
    row = read(base, "attn_q", "Q8_0", "ref", out=out, subset="screen24", scorer=_scorer(seen))
    assert seen == [
        {"checkpoint": str(out / "ckpt"), "ref": "ref", "subset": "screen24", "tp": DEFAULT_TP, "check": True}
    ]
    assert row["class"] == "attn_q" and row["type"] == "Q8_0" and row["kl"] == 0.004
    assert row["subset"] == "screen24" and row["server"]["tp"] == DEFAULT_TP
    assert row["written"]["class"] == "attn_q" and row["written"]["rewritten_shards"]
    assert row["check"] == {"deterministic": True, "max_abs_diff": 0.0}
    assert json.loads((out / "result.json").read_text())["kl"] == 0.004
    assert json.loads((out / "row.json").read_text()) == json.loads(json.dumps(row, default=str))


def test_the_result_comes_through_the_scorers_own_serializer(tmp_path) -> None:
    # the real Score writes its own result with model_dump_json; asdict() on it raised after the GPU work
    base = _base(tmp_path)
    got, out = _ScoreModel(), tmp_path / "out"
    row = read(base, "attn_q", "Q8_0", "ref", out=out, scorer=lambda *a, **k: got)
    assert json.loads((out / "result.json").read_text()) == got.model_dump(mode="json")
    assert row["kl"] == 0.004 and row["reference"] == {"content_sha256": "94a7ceb1"}
    assert row["server"] == {"vllm": "0.33.0", "tp": DEFAULT_TP, "serve": {"args": {"dtype": "bfloat16"}}}


def test_a_baseline_read_scores_the_base_and_replaces_no_class(tmp_path) -> None:
    base = _base(tmp_path)
    out, seen = tmp_path / "out", []
    row = read(base, "bf16", "bf16", "ref", out=out, scorer=_scorer(seen))
    assert seen[0]["checkpoint"] == str(base)
    assert row["written"] == {"baseline": True, "base_anchor": str(tmp_path / "anchor")}
    assert not (out / "ckpt").exists()


def test_a_second_read_into_the_same_directory_is_refused(tmp_path) -> None:
    base = _base(tmp_path)
    out = tmp_path / "out"
    read(base, "attn_q", "Q8_0", "ref", out=out, scorer=_scorer([]))
    with pytest.raises(ValueError, match="fresh directory"):
        read(base, "attn_k", "Q8_0", "ref", out=out, scorer=_scorer([]))


def test_a_class_the_base_does_not_hold_and_a_baseline_that_replaces_one_are_refused(tmp_path) -> None:
    base = _base(tmp_path)
    with pytest.raises(ValueError, match="no class"):
        read(base, "experts", "Q8_0", "ref", out=tmp_path / "a", scorer=_scorer([]))
    with pytest.raises(ValueError, match="replaces no class"):
        read(base, "attn_q", "bf16", "ref", out=tmp_path / "b", scorer=_scorer([]))


def test_a_non_finite_reading_is_refused(tmp_path) -> None:
    base = _base(tmp_path)
    scored = _Scored(kl=float("nan"))
    with pytest.raises(ValueError, match="not finite"):
        read(base, "attn_q", "Q8_0", "ref", out=tmp_path / "out", scorer=lambda *a, **k: scored)


def test_a_non_deterministic_reading_is_refused(tmp_path) -> None:
    base = _base(tmp_path)
    scored = _Scored()
    scored.deterministic = False
    with pytest.raises(ValueError, match="non-deterministic"):
        read(base, "attn_q", "Q8_0", "ref", out=tmp_path / "out", scorer=lambda *a, **k: scored)
