"""An unexpected probe failure is one two-line refusal and exit 2, not a traceback.

The three instances the issue names: a model's forward raises, its config gives a per-layer top-k, and its
tokenizer has no chat template for messages rows. `QDA_TRACEBACK=1` re-raises the original for a bug report.
"""

from __future__ import annotations

import json

import pytest
from qda_quant.cli import main

from .engine_util import save_tiny_tokenizer, tiny_qwen3, tiny_qwen3_moe


def _config(tmp_path, src, calib):
    path = tmp_path / "plan.yaml"
    path.write_text(
        json.dumps(
            {
                "model": str(src),
                "format": "w4a16-sym-g32",
                "devices": ["cpu"],
                "steps": [{"gptq": {}}],  # needs a calibration forward, so `plan` reads the rows
                "calib": {"file": {"path": str(calib), "samples": 1, "seqlen": 16}},
                "out": str(tmp_path / "out"),
            }
        )
    )
    return path


def _messages_calib(tmp_path):
    path = tmp_path / "calib.jsonl"
    rows = [{"messages": [{"role": "user", "content": "w1 w2"}]} for _ in range(2)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _source(tmp_path, *, moe=False, plain_tokenizer=False):
    src = tmp_path / "m"
    (tiny_qwen3_moe() if moe else tiny_qwen3(layers=2)).save_pretrained(src)
    if plain_tokenizer:  # a serialized pipeline without a chat template
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        vocab = {f"w{i}": i for i in range(127)} | {"[UNK]": 127}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(src)
    else:
        save_tiny_tokenizer(src)
    return src


def test_a_forward_failure_is_a_two_line_refusal(tmp_path, monkeypatch, capsys):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    src = _source(tmp_path)
    cfg = _config(tmp_path, src, _messages_calib(tmp_path))

    def boom(_layer, *_args, **_kwargs):
        raise RuntimeError("the forward cannot run")

    monkeypatch.setattr(Qwen3DecoderLayer, "forward", boom)
    assert main(["plan", str(cfg)]) == 2
    err = capsys.readouterr().err
    # our label, and the `qda: next:` line inside the same message: the two dialects collide here
    assert "error: model " in err, err
    assert "probing its forward failed in model.layers.0" in err, err
    assert "RuntimeError: the forward cannot run" in err, err
    assert "\nqda: next: if this layout is new" in err, err

    monkeypatch.setenv("QDA_TRACEBACK", "1")
    with pytest.raises(RuntimeError, match="the forward cannot run"):
        main(["plan", str(cfg)])


def test_a_per_layer_top_k_is_a_two_line_refusal(tmp_path, monkeypatch, capsys):
    from qda_quant.arch import derive as derive_mod

    src = _source(tmp_path, moe=True)
    cfg = _config(tmp_path, src, _messages_calib(tmp_path))
    real_text_config = derive_mod.text_config

    class PerLayerTopK:  # what HunYuanMoEV1Config gives: one top-k per decoder layer
        def __init__(self, config):
            object.__setattr__(self, "_config", config)

        def __getattr__(self, name):
            return [2, 2] if name == "num_experts_per_tok" else getattr(self._config, name)

    monkeypatch.setattr(derive_mod, "text_config", lambda model: PerLayerTopK(real_text_config(model)))
    assert main(["plan", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "deriving its graph failed in model.layers.0" in err, err
    assert "TypeError: int() argument" in err, err
    assert "\nqda: next: if this layout is new" in err, err

    monkeypatch.setenv("QDA_TRACEBACK", "1")
    with pytest.raises(TypeError, match="int\\(\\) argument"):
        main(["plan", str(cfg)])


def test_a_tokenizer_without_a_chat_template_is_a_two_line_refusal(tmp_path, monkeypatch, capsys):
    src = _source(tmp_path, plain_tokenizer=True)
    cfg = _config(tmp_path, src, _messages_calib(tmp_path))
    assert main(["plan", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "tokenizing its calibration rows failed" in err, err
    assert "chat_template is not set" in err, err
    assert '\nqda: next: calibrate on text rows ({"text": ...})' in err, (
        err
    )  # not an adapter: the layout is fine

    monkeypatch.setenv("QDA_TRACEBACK", "1")
    with pytest.raises(ValueError, match="chat_template is not set"):
        main(["plan", str(cfg)])


def test_out_of_memory_is_not_turned_into_a_layout_refusal():
    """Running out of memory while probing is not a layout QDA cannot read; it propagates unchanged."""
    from qda_quant.step import probe_refusal

    class OutOfMemoryError(RuntimeError):  # the name torch.cuda gives its error
        pass

    for exc in (MemoryError("host"), OutOfMemoryError("CUDA out of memory")):
        with pytest.raises(type(exc)):
            raise probe_refusal("org/model", "probing its forward", exc)


def test_a_hub_cache_path_is_named_by_its_repo_id():
    """The user typed `org/name`; the refusal says that, not the cache directory it resolved to."""
    from qda_quant.step import probe_refusal

    cached = (
        "/cache/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
    )
    message = str(probe_refusal(cached, "probing its forward", RuntimeError("boom")))
    assert message.startswith("model state-spaces/mamba-130m-hf: probing its forward failed"), message
    assert "/cache/hub" not in message
