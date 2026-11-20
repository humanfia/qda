"""The GGUF screening job: decode the candidate, read the suite on the decode, remove the decode.

The decoder's own exactness tests live in qda-quant; here the contract is the judge's: one result or none, the
decoded checkpoint present only while the suite runs, and a file the decoder cannot invert refused by name
(never answered by a second engine).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import ClassVar

import pytest
from qda_kl.eval.suites import SUITES, Suite, SuiteResult

TOOLS = Path(__file__).parents[1] / "src" / "qda_agent" / "judge" / "tools"


class _Reading(Suite):
    """A suite that records the checkpoint it was handed and returns one metric."""

    name: ClassVar[str] = "gguf_screen_stub"
    seen: ClassVar[list[str]] = []

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        del reference, work
        self.seen.append(checkpoint)
        assert (Path(checkpoint) / "model.safetensors.index.json").is_file(), checkpoint
        return SuiteResult(suite=self.name, metrics={"kl": 0.5}, higher_is_better={"kl": False})


class _Fails(Suite):
    """A suite whose reading crashes: the job must leave no result and no decode behind."""

    name: ClassVar[str] = "gguf_screen_fails"

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        del checkpoint, reference, work
        raise RuntimeError("the reader crashed")


def _job():
    spec = importlib.util.spec_from_file_location("gguf_screen", TOOLS / "gguf_screen.py")
    assert spec is not None and spec.loader is not None
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def _gguf(tmp_path: Path, arch: str = "qwen35") -> Path:
    gguf = pytest.importorskip("gguf")
    import torch

    def vals(n: int) -> torch.Tensor:
        return (torch.arange(n) % 8).float()

    w = gguf.GGUFWriter(str(tmp_path / "candidate.gguf"), arch)
    w.add_block_count(1)
    w.add_uint32(f"{arch}.ssm.group_count", 2)
    w.add_uint32(f"{arch}.ssm.time_step_rank", 4)
    w.add_uint32(f"{arch}.ssm.inner_size", 16)
    w.add_uint32(f"{arch}.ssm.state_size", 4)
    tensors = {
        "token_embd.weight": vals(64).reshape(8, 8),
        "output_norm.weight": vals(8) + 1,
        "output.weight": vals(64).reshape(8, 8),
        "blk.0.attn_norm.weight": vals(8) + 1,
        "blk.0.ffn_up.weight": vals(64).reshape(8, 8),
        "blk.0.ssm_norm.weight": vals(4),
    }
    for name, t in tensors.items():
        w.add_tensor(
            name, t.numpy().astype("<f4"), raw_shape=tuple(t.shape), raw_dtype=gguf.GGMLQuantizationType.F32
        )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return tmp_path / "candidate.gguf"


def _snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {}}) + "\n")
    (snap / "tokenizer.json").write_text('{"model": {}}\n')
    return snap


def _spec(tmp_path: Path, gguf: Path, snapshot: Path, suite: str = _Reading.name) -> Path:
    for s in (_Reading, _Fails):
        if s.name not in SUITES:
            SUITES.register(s.name, s)
    ref = tmp_path / "ref"
    ref.mkdir(exist_ok=True)
    out = tmp_path / "results" / "k1.json"
    out.parent.mkdir(exist_ok=True)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "gguf": str(gguf),
                "snapshot": str(snapshot),
                "decoded": str(tmp_path / "decoded"),
                "suite": {suite: {}},
                "reference": str(ref),
                "out": str(out),
            }
        )
    )
    return spec


def test_the_suite_reads_the_decoded_checkpoint_and_the_decode_is_removed(tmp_path: Path) -> None:
    """The reading happens on the decoded HF checkpoint; after it, neither the result nor the decode is left behind."""
    _Reading.seen.clear()
    gguf = _gguf(tmp_path)
    spec = _spec(tmp_path, gguf, _snapshot(tmp_path))
    assert _job().main([str(spec)]) == 0
    assert len(_Reading.seen) == 1 and Path(_Reading.seen[0]).name == "decoded"
    got = json.loads((tmp_path / "results" / "k1.json").read_text())
    assert got["suite"] == _Reading.name and got["metrics"] == {"kl": 0.5}
    # the reading states the .gguf it read, so a fold can refuse a reading that came from another file
    assert got["detail"]["decode"]["gguf_sha256"] == hashlib.sha256(gguf.read_bytes()).hexdigest()
    assert got["detail"]["decode"]["gguf_bytes"] == gguf.stat().st_size
    assert not (tmp_path / "decoded").exists()


def test_a_crashed_reading_publishes_nothing_and_removes_the_decode(tmp_path: Path) -> None:
    """A suite that crashes is no reading: no result file, and the decoded checkpoint is still removed."""
    spec = _spec(tmp_path, _gguf(tmp_path), _snapshot(tmp_path), suite=_Fails.name)
    assert _job().main([str(spec)]) != 0
    assert not (tmp_path / "results" / "k1.json").exists()
    assert not (tmp_path / "decoded").exists()


def test_an_architecture_the_decoder_cannot_invert_is_refused_by_name(tmp_path: Path) -> None:
    """A `llama` candidate gets a refusal and no result: a screening reading never falls back to one engine."""
    spec = _spec(tmp_path, _gguf(tmp_path, arch="llama"), _snapshot(tmp_path))
    assert _job().main([str(spec)]) == 2
    assert not (tmp_path / "results" / "k1.json").exists()
    assert not (tmp_path / "decoded").exists()


def test_an_unusable_snapshot_is_refused(tmp_path: Path) -> None:
    """No snapshot means no config or tokenizer to serve the decode with: refuse, and leave no partial decode."""
    gguf = _gguf(tmp_path)
    spec = _spec(tmp_path, gguf, tmp_path / "absent")
    assert _job().main([str(spec)]) == 2
    assert not (tmp_path / "results" / "k1.json").exists()
    assert not (tmp_path / "decoded").exists()


def test_the_kl_suite_loads_the_decode_as_a_language_model() -> None:
    """The decode has no visual tower, so the checkpoint engine is told `language_model_only`; the suite's
    serve_args (the reference's engine) are untouched, and a bare `{kl: {}}` gets the default topk scorer."""
    suite = {
        "kl": {
            "corpus": {"file": {"path": "prompts.jsonl"}},
            "serve_args": {"enable_expert_parallel": True},
            "scorer": {"topk": {"serve_args": {"max_num_seqs": 8}}},
        }
    }
    got = _job()._language_only(suite)
    assert got["kl"]["scorer"]["topk"]["serve_args"] == {"max_num_seqs": 8, "language_model_only": True}
    assert got["kl"]["serve_args"] == {"enable_expert_parallel": True}
    assert _job()._language_only({"kl": {}}) == {
        "kl": {"scorer": {"topk": {"serve_args": {"language_model_only": True}}}}
    }


def test_another_suites_settings_are_not_rewritten() -> None:
    """A non-KL suite's settings are not this tool's to edit."""
    suite = {"sanity": {"prompt": "p"}}
    assert _job()._language_only(suite) == suite


def test_a_suite_that_names_language_model_only_is_refused() -> None:
    """The KL composition lets a suite-level serve_args win over the scorer's, and it would reach the
    reference engine too: refuse it by name instead of composing a non-language-only checkpoint engine."""
    suite = {"kl": {"serve_args": {"language_model_only": False}}}
    with pytest.raises(ValueError, match="language_model_only"):
        _job()._language_only(suite)


def test_an_existing_decoded_directory_is_refused_and_left_intact(tmp_path: Path) -> None:
    """The job removes only what it created: a supplied existing directory is refused, never deleted."""
    decoded = tmp_path / "decoded"
    decoded.mkdir()
    (decoded / "keep.txt").write_text("someone else's tree\n")
    spec = _spec(tmp_path, _gguf(tmp_path), _snapshot(tmp_path))
    assert _job().main([str(spec)]) == 2
    assert (decoded / "keep.txt").read_text() == "someone else's tree\n"
    assert not (tmp_path / "results" / "k1.json").exists()


def test_an_interrupted_decode_leaves_no_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decode that dies mid-way is cleaned up and the error still propagates: the tree is this job's."""
    import qda_quant.gguf.decode

    def boom(gguf: Path, out: Path, snapshot: Path) -> dict:
        del gguf, snapshot
        out.mkdir()
        (out / "partial.safetensors").write_text("half\n")
        raise KeyboardInterrupt

    monkeypatch.setattr(qda_quant.gguf.decode, "decode", boom)
    spec = _spec(tmp_path, _gguf(tmp_path), _snapshot(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        _job().main([str(spec)])
    assert not (tmp_path / "decoded").exists()
