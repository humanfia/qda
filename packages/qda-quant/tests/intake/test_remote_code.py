"""A model whose config or tokenizer is code from its repository is refused by name, and that code never runs."""

import json

import pytest
from qda_quant.cli import main
from qda_quant.intake.loading import load_config
from qda_quant.intake.tokenizer import load_tokenizer
from qda_quant.step import Refused, remote_code_refusal
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, PreTrainedTokenizerFast

#: The repository's module: importing it leaves a file, so a test can tell whether QDA ran it.
CODE = """from pathlib import Path
Path(__file__).with_name("RAN").touch()
from transformers import LlamaConfig, PreTrainedTokenizerFast
class RemoteConfig(LlamaConfig):
    model_type = "remote_x"
class RemoteTokenizer(PreTrainedTokenizerFast):
    pass
"""


def _repo(tmp_path, *, remote_config: bool, remote_tokenizer: bool):
    """A tiny Llama directory whose config and/or tokenizer `auto_map` names the repository's own module."""
    d = tmp_path / "repo"
    d.mkdir()
    (d / "remote_x.py").write_text(CODE)
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, vocab_size=300).to_dict()  # fmt: skip
    if remote_config:
        cfg |= {"model_type": "remote_x", "auto_map": {"AutoConfig": "remote_x.RemoteConfig"}}
    (d / "config.json").write_text(json.dumps(cfg))
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(["Hello world, this is a test."], trainers.BpeTrainer(vocab_size=300))
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(d)
    if remote_tokenizer:  # no tokenizer.json: the declared class is the only way to read it
        (d / "tokenizer.json").unlink()
        tc = json.loads((d / "tokenizer_config.json").read_text())
        tc |= {
            "tokenizer_class": "RemoteTokenizer",
            "auto_map": {"AutoTokenizer": [None, "remote_x.RemoteTokenizer"]},
        }
        (d / "tokenizer_config.json").write_text(json.dumps(tc))
    return d


def test_a_remote_code_tokenizer_is_refused_and_not_run(tmp_path):
    d = _repo(tmp_path, remote_config=False, remote_tokenizer=True)
    with pytest.raises(
        Refused, match=r"its tokenizer is code from the model repository \(trust_remote_code\)"
    ):
        load_tokenizer(d)
    assert not (d / "RAN").exists()


def test_a_remote_code_config_is_refused_and_not_run(tmp_path):
    d = _repo(tmp_path, remote_config=True, remote_tokenizer=False)
    with pytest.raises(Refused, match=r"its config is code from the model repository \(trust_remote_code\)"):
        load_config(d)
    assert not (d / "RAN").exists()


@pytest.mark.parametrize(
    ("remote_config", "remote_tokenizer", "what"), [(False, True, "tokenizer"), (True, False, "config")]
)
def test_fisher_map_refuses_remote_code_with_exit_2(tmp_path, capsys, remote_config, remote_tokenizer, what):
    d = _repo(tmp_path, remote_config=remote_config, remote_tokenizer=remote_tokenizer)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "".join(json.dumps({"text": "Hello world, this is a test. " * 8}) + "\n" for _ in range(2))
    )
    common = ["--model", str(d), "--calib", str(calib), "--samples", "1", "--seqlen", "8", "--device", "cpu"]
    argv = ["fisher", "map", *common, "--menu", "bf16", "--no-cache", "--out", str(tmp_path / "m.json")]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert f"its {what} is code from the model repository (trust_remote_code); QDA never runs it" in err
    assert "[y/N]" not in err  # transformers is told no; it does not ask
    assert not (d / "RAN").exists()


def test_only_transformers_remote_code_error_is_named_remote_code():
    asked = ValueError(
        "repo contains custom code. Please pass the argument `trust_remote_code=True` to allow it."
    )
    assert "its processor is code from the model repository" in str(
        remote_code_refusal("m", "processor", asked)
    )
    assert (
        remote_code_refusal("m", "tokenizer", ValueError("Couldn't instantiate the backend tokenizer"))
        is None
    )


def _compressed(d):
    """`d` as a compressed-tensors checkpoint: a ``quantization_config`` and one safetensors shard."""
    import torch
    from safetensors.torch import save_file

    cfg = json.loads((d / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "compressed-tensors", "format": "dense", "ignore": []}
    (d / "config.json").write_text(json.dumps(cfg))
    save_file({"model.embed_tokens.weight": torch.zeros(300, 16)}, str(d / "model.safetensors"))
    return d


def test_source_names_refuses_a_remote_code_config(tmp_path):
    from qda_quant.intake.source_names import SourceNames

    d = _compressed(_repo(tmp_path, remote_config=True, remote_tokenizer=False))
    with pytest.raises(Refused, match=r"its config is code from the model repository \(trust_remote_code\)"):
        SourceNames(d)
    assert not (d / "RAN").exists()


def test_load_bf16_refuses_a_remote_code_compressed_tensors_config(tmp_path, monkeypatch):
    from qda_quant.intake import fisher, source_names

    d = _compressed(_repo(tmp_path, remote_config=True, remote_tokenizer=False))
    # SourceNames refuses first in practice; stub it so the refusal is the loader's own (defensive) config read
    monkeypatch.setattr(source_names, "SourceNames", lambda path: None)
    with pytest.raises(Refused, match=r"its config is code from the model repository \(trust_remote_code\)"):
        fisher.load_bf16(d, "cpu")
    assert not (d / "RAN").exists()


def test_fisher_map_refuses_a_remote_code_compressed_tensors_checkpoint_with_exit_2(tmp_path, capsys):
    # end to end: the refusal comes from SourceNames, the first config read on this path
    d = _compressed(_repo(tmp_path, remote_config=True, remote_tokenizer=False))
    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "".join(json.dumps({"text": "Hello world, this is a test. " * 8}) + "\n" for _ in range(2))
    )
    argv = ["fisher", "map", "--model", str(d), "--calib", str(calib), "--samples", "1", "--seqlen", "8",
            "--device", "cpu", "--menu", "bf16", "--no-cache", "--out", str(tmp_path / "m.json")]  # fmt: skip
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "its config is code from the model repository (trust_remote_code); QDA never runs it" in err
    assert "pass the argument `trust_remote_code=True`" not in err
    assert not (d / "RAN").exists()
