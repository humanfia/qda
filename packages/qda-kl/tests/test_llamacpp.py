import ctypes as C
import types

import numpy as np
import pytest
from qda_kl import llamacpp
from qda_kl.scorers import SCORERS
from qda_quant.step import Refused


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class Tensor:
    def __init__(self, name, tensor_type=12, shape=(4, 4), n_bytes=16):
        self.name, self.tensor_type, self.shape, self.n_bytes = name, tensor_type, shape, n_bytes


class Reader:
    def __init__(self, tensors, fields):
        self.tensors, self.fields = tensors, fields

    def get_field(self, name):
        return self.fields.get(name)


def stub(monkeypatch, tensors, fields):
    reader = Reader(tensors, fields)
    monkeypatch.setattr(
        llamacpp, "_gguf_reader", lambda path: types.SimpleNamespace(GGUFReader=lambda model: reader)
    )


def line(name, tensor_type=12, shape=(4, 4, 1, 1)):
    return "QDA_TENSOR\t" + "\t".join([name, str(tensor_type), *map(str, shape)])


def test_strict_check_passes_when_every_tensor_loaded(monkeypatch):
    tensors = [Tensor("blk.0.a"), Tensor("output_norm.weight", 0, (4,))]
    stub(monkeypatch, tensors, {"general.architecture": Field("qwen3"), "qwen3.block_count": Field(1)})
    log = "\n".join([line("blk.0.a"), line("output_norm.weight", 0, (4, 1, 1, 1))])
    got = llamacpp.strict_check("m.gguf", log, "/checkout")
    assert got["verdict"] == "PASS" and got["tensors"] == 2 and got["skipped"] == {}


def test_strict_check_classifies_only_nextn_skips(monkeypatch):
    tensors = [Tensor("blk.0.a"), Tensor("blk.1.a"), Tensor("output_norm.weight", 0, (4,))]
    fields = {
        "general.architecture": Field("qwen3"),
        "qwen3.block_count": Field(2),
        "qwen3.nextn_predict_layers": Field(1),
    }
    stub(monkeypatch, tensors, fields)
    log = "\n".join(
        [
            line("blk.0.a"),
            line("output_norm.weight", 0, (4, 1, 1, 1)),
            "llama_model_loader: model has unused tensor blk.1.a -- ignoring",
        ]
    )
    got = llamacpp.strict_check("m.gguf", log, "/checkout")
    assert got["verdict"] == "PASS-WITH-MTP-OFF"
    assert list(got["skipped"]) == ["blk.1.a"]
    assert got["skipped"]["blk.1.a"]["classification"] == "MTP-off"
    assert got["ignored_log_names"] == ["blk.1.a"]


def test_strict_check_refuses_a_missing_non_mtp_tensor(monkeypatch):
    tensors = [Tensor("blk.0.a"), Tensor("output_norm.weight", 0, (4,))]
    stub(monkeypatch, tensors, {"general.architecture": Field("qwen3"), "qwen3.block_count": Field(1)})
    log = line("output_norm.weight", 0, (4, 1, 1, 1))
    # Refused (a ValueError) is a verdict on the input: the CLI exits 2 and writes no score.json.
    with pytest.raises(Refused, match="loaded inventory differs"):
        llamacpp.strict_check("m.gguf", log, "/checkout")


def test_strict_check_refuses_a_changed_tensor(monkeypatch):
    stub(
        monkeypatch,
        [Tensor("blk.0.a")],
        {"general.architecture": Field("qwen3"), "qwen3.block_count": Field(1)},
    )
    with pytest.raises(ValueError, match="loaded inventory differs"):
        llamacpp.strict_check("m.gguf", line("blk.0.a", 12, (4, 4, 1, 2)), "/checkout")


def test_strict_check_refuses_an_ignored_non_mtp_tensor(monkeypatch):
    stub(
        monkeypatch,
        [Tensor("blk.0.a")],
        {"general.architecture": Field("qwen3"), "qwen3.block_count": Field(1)},
    )
    log = "\n".join([line("blk.0.a"), "model has unused tensor blk.0.a -- ignoring"])
    with pytest.raises(ValueError, match="ignored non-MTP"):
        llamacpp.strict_check("m.gguf", log, "/checkout")


class FakeBridge:
    """Llama's side of the bridge: one logits buffer that every decode overwrites, as libllama's is."""

    def __init__(self, rows, vocab, stride=None):
        self.vocab, self.ctx, self.lib = vocab, 1, self
        self.row_stride = vocab if stride is None else stride
        self.buffer = np.zeros((rows, self.row_stride + vocab), dtype=np.float32)

    def qda_clear(self, ctx):
        pass

    def qda_decode(self, ctx, ids, n, offset, start):
        self.buffer[:] = offset  # every row now holds this batch's first position
        return 0

    def qda_logits(self, ctx, index):
        address = self.buffer.ctypes.data + index * self.row_stride * C.sizeof(C.c_float)
        return C.cast(address, C.POINTER(C.c_float))


class FakeProcess:
    def __init__(self, exitcode=0):
        self.exitcode, self.pid = exitcode, 1

    def join(self):
        pass


@pytest.mark.parametrize("key", ["llamacpp", "llama.cpp"])
def test_the_removed_llamacpp_scorer_is_refused_with_its_replacement(key, tmp_path, capsys):
    """An old score config naming the llama.cpp scorer prints one line pointing at decode + `qda-kl score`."""
    from qda_kl import cli

    assert "llamacpp" not in SCORERS
    config = tmp_path / "score.yaml"
    config.write_text(
        f"checkpoint: m.gguf\nreference: ref\nout: out\nscorer: {{'{key}': {{bridge: probe.so}}}}\n"
    )
    assert cli.main(["score", str(config)]) == 2
    (line,) = capsys.readouterr().err.strip().splitlines()
    assert f"scorer.{key}: removed" in line and "`qda decode-gguf`" in line and "`qda-kl score`" in line
