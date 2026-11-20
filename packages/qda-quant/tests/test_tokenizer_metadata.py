"""The export must not ship a ``tokenizer_config.json`` whose declared class rebuilds the checkpoint's pipeline.

Transformers maps some declarations (``LlamaTokenizerFast`` over the R1-Distill snapshots) to a constructor
that replaces the stored ByteLevel pipeline with Metaspace/ByteFallback, so text encodes to different ids than the
checkpoint's own ``tokenizer.json``.
"""

import json

import pytest
import torch
import transformers
from qda_quant import BuildConfig
from qda_quant.engine.calib import fingerprint
from qda_quant.step import Refused
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from .engine_util import save_tiny_tokenizer, tiny_qwen3, write_ids
from .tiny import SMALL

PROBE = "Hello world, this is a test.\nNext line."


def _bytelevel_source(tmp_path, tokenizer_class="LlamaTokenizerFast"):
    """A tiny Llama with a trained ByteLevel tokenizer whose config declares `tokenizer_class`."""
    cfg = transformers.LlamaConfig(**{**SMALL, "vocab_size": 300})
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    src = tmp_path / "m"
    transformers.AutoModelForCausalLM.from_config(cfg).eval().save_pretrained(src)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([PROBE], trainers.BpeTrainer(vocab_size=300))
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(src)
    conf = json.loads((src / "tokenizer_config.json").read_text())
    conf.update(tokenizer_class=tokenizer_class, legacy=True)
    (src / "tokenizer_config.json").write_text(json.dumps(conf))
    return src, backend


def _build(src, calib, out):
    BuildConfig.from_dict(
        {
            "model": str(src),
            "format": "w4a16-sym-g64",
            "devices": ["cpu"],
            "steps": [{"rtn": {}}],
            "calib": {"file": {"path": str(calib), "samples": 2, "seqlen": 16}},
            "out": str(out),
        }
    ).build()


def _calib(src, tmp_path):
    g = torch.Generator().manual_seed(0)
    rows = [torch.randint(0, 300, (16,), generator=g).tolist() for _ in range(2)]
    path = tmp_path / "c.jsonl"
    write_ids(path, rows, fingerprint(src))
    return path


def test_a_class_that_rebuilds_the_serialized_pipeline_is_corrected(tmp_path):
    src, _ = _bytelevel_source(tmp_path)
    out = tmp_path / "out"
    _build(src, _calib(src, tmp_path), out)
    source = json.loads((src / "tokenizer_config.json").read_text())
    written = json.loads((out / "tokenizer_config.json").read_text())
    assert written["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert {k: v for k, v in written.items() if k != "tokenizer_class"} == {
        k: v for k, v in source.items() if k != "tokenizer_class"
    }
    assert (out / "tokenizer.json").read_bytes() == (src / "tokenizer.json").read_bytes()
    assert transformers.AutoTokenizer.from_pretrained(out).encode(PROBE, add_special_tokens=False) == (
        PreTrainedTokenizerFast.from_pretrained(out).encode(PROBE, add_special_tokens=False)
    )
    assert json.loads((out / "build.json").read_text())["tokenizer_config"] == {
        "declared": "LlamaTokenizerFast",
        "serialized_ok": False,
        "written": "PreTrainedTokenizerFast",
    }


def test_a_matching_declaration_is_recorded_and_left_byte_identical(tmp_path):
    src, _ = _bytelevel_source(tmp_path, tokenizer_class="PreTrainedTokenizerFast")
    out = tmp_path / "out"
    _build(src, _calib(src, tmp_path), out)
    assert (out / "tokenizer_config.json").read_bytes() == (src / "tokenizer_config.json").read_bytes()
    assert json.loads((out / "build.json").read_text())["tokenizer_config"] == {
        "declared": "PreTrainedTokenizerFast",
        "serialized_ok": True,
        "written": "unchanged",
    }


def test_a_source_without_a_bytelevel_pipeline_is_a_recorded_no_op(tmp_path):
    tiny_qwen3(layers=2, hidden=64, inter=96).save_pretrained(tmp_path / "q")
    source = tmp_path / "q"
    sha = save_tiny_tokenizer(source)
    g = torch.Generator().manual_seed(0)
    calib = tmp_path / "q.jsonl"
    write_ids(calib, [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(2)], sha)
    out = tmp_path / "out"
    BuildConfig.from_dict(
        {
            "model": str(source),
            "format": "w4a16-sym-g32",
            "devices": ["cpu"],
            "steps": [{"rtn": {}}],
            "calib": {"file": {"path": str(calib), "samples": 2, "seqlen": 32}},
            "out": str(out),
        }
    ).build()
    assert (out / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
    assert json.loads((out / "build.json").read_text())["tokenizer_config"] == {
        "declared": "TokenizersBackend",
        "serialized_ok": None,
        "written": "unchanged",
        "reason": "not a ByteLevel pipeline",
    }


def test_a_source_without_tokenizer_json_is_recorded_in_build_json(tmp_path):
    """A slow-only tokenizer has no serialized pipeline: the check is skipped, and build.json must say so."""
    from transformers import BertTokenizer

    source = tmp_path / "q"
    tiny_qwen3(layers=2, hidden=64, inter=96).save_pretrained(source)
    (source / "vocab.txt").write_text(
        "\n".join(["[PAD]", "[UNK]", "hello", "world", *[f"w{i}" for i in range(124)]]) + "\n"
    )
    BertTokenizer(vocab_file=str(source / "vocab.txt"), do_lower_case=True).save_pretrained(source)
    (source / "tokenizer.json").unlink()
    g = torch.Generator().manual_seed(0)
    calib = tmp_path / "q.jsonl"
    write_ids(
        calib, [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(2)], fingerprint(source)
    )
    out = tmp_path / "out"
    BuildConfig.from_dict(
        {
            "model": str(source),
            "format": "w4a16-sym-g32",
            "devices": ["cpu"],
            "steps": [{"rtn": {}}],
            "calib": {"file": {"path": str(calib), "samples": 2, "seqlen": 32}},
            "out": str(out),
        }
    ).build()
    assert (out / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
    assert json.loads((out / "build.json").read_text())["tokenizer_config"] == {
        "declared": "BertTokenizer",
        "serialized_ok": None,
        "written": "unchanged",
        "reason": "no tokenizer.json",
    }


def test_a_comparison_that_cannot_run_refuses(tmp_path):
    src, _ = _bytelevel_source(tmp_path)
    pipeline = json.loads((src / "tokenizer.json").read_text())
    del pipeline["model"]  # still a ByteLevel pipeline, no longer a tokenizer the classes can build
    (src / "tokenizer.json").write_text(json.dumps(pipeline))
    with pytest.raises(Refused, match="could not be compared"):
        _build(src, _calib(src, tmp_path), tmp_path / "out")


def test_an_unparsable_tokenizer_config_is_refused_by_name(tmp_path, capsys):
    """A truncated config used to escape as a raw JSONDecodeError with a traceback."""
    src, _ = _bytelevel_source(tmp_path)
    calib = _calib(src, tmp_path)  # made before the config is broken, so only the parse can be the failure
    (src / "tokenizer_config.json").write_text('{"tokenizer_class": "LlamaTokenizerFast",')
    with pytest.raises(Refused, match="cannot read .*tokenizer_config.json"):
        _build(src, calib, tmp_path / "out")
    config = tmp_path / "build.yaml"
    config.write_text(
        json.dumps(
            {
                "model": str(src),
                "format": "w4a16-sym-g64",
                "devices": ["cpu"],
                "steps": [{"rtn": {}}],
                "calib": {"file": {"path": str(calib), "samples": 2, "seqlen": 16}},
                "out": str(tmp_path / "cli-out"),
            }
        )
    )
    from qda_quant.cli import main

    assert main(["build", str(config)]) == 2
    err = capsys.readouterr().err
    assert "error: tokenizer_config: cannot read" in err
    assert "Traceback" not in err
