"""Named subsets preserve frozen identity and select every answer draw without changing the reference."""

from types import SimpleNamespace

import numpy as np
import pytest
from qda_kl import score
from qda_kl.cli import main
from qda_kl.eval.report import _merge
from qda_kl.eval.suites import KLSuite
from qda_kl.reference_merge import merge
from qda_kl.store import Reference, _content_sha256
from test_reference_merge import shard
from test_tokenizer_guard import tiny
from transformers import AutoTokenizer


def test_subset_identity_and_actual_scoring(tmp_path, monkeypatch):
    a, b = shard(tmp_path / "a", "a"), shard(tmp_path / "b", "b")
    legacy = Reference(a).manifest
    assert "subsets" not in legacy.model_dump(exclude_defaults=True)
    assert _content_sha256(legacy.model_copy(update={"subsets": {"search": ["a"]}})) != legacy.content_sha256
    before = (a / "manifest.json").read_bytes()
    ids, out = tmp_path / "ids.txt", tmp_path / "merged"
    ids.write_text("b\n")
    assert main(["ref", "merge", str(out), str(a), str(b), "--subset", f"search={ids}"]) == 0
    ref = Reference(out, verify=True, subset="search")
    assert ref.content_sha256 == Reference(out).content_sha256
    assert [s.id for s in ref.sequences] == ["b#0", "b#1"]
    suite = KLSuite(subset="search", scorer={"topk": {"serve_args": {"dtype": "bfloat16"}}})
    assert suite._digest_fields() == KLSuite()._digest_fields()
    assert suite.to_dict()["kl"]["subset"] == "search"
    monkeypatch.setattr(KLSuite, "reference_dir", lambda *args: out)
    config = SimpleNamespace(dtype="bfloat16", get_vocab_size=lambda: 4)
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=config),
        collective_rpc=lambda _: [{}],
        get_tokenizer=tiny,
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tiny())
    monkeypatch.setattr(score, "open_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    seen = []

    def force(llm, tag, items, **kwargs):
        seen.extend(i.key for i in items)
        return {i.key: dict.fromkeys(("kl", "flip", "tgt"), np.zeros(len(i.tokens) - i.start)) for i in items}

    monkeypatch.setattr(score, "teacher_force", force)
    result = suite.run(str(tmp_path), "model", tmp_path)
    assert seen == ["b#0", "b#1"]
    assert set(result.detail["per_seq"]) == set(seen)
    assert result.detail["tokens"] == 6 and result.detail["subset"] == "search"
    assert result.detail["reference_sha256"] == ref.content_sha256
    assert (a / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="unknown reference subset"):
        KLSuite(subset="unknown").run(str(tmp_path), "model", tmp_path)
    for subset in (None, "other"):
        different = result.model_copy(update={"detail": {**result.detail, "subset": subset}})
        with pytest.raises(ValueError, match="different reference subsets"):
            _merge([result, different])


@pytest.mark.parametrize("ids", [["missing"], ["a", "a"]])
def test_subset_ids_refused_before_publication(tmp_path, ids):
    a = shard(tmp_path / "a", "a")
    with pytest.raises(ValueError, match="duplicate ids, or unknown ids"):
        merge(tmp_path / "merged", [a], subsets={"search": ids})
    assert not (tmp_path / "merged").exists()
