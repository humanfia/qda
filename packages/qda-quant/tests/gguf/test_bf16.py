"""The one-pass BF16 template expansion preserves every stored bit and exported byte."""

from pathlib import Path

import numpy as np
import pytest
import torch
from qda_quant import Recipe
from qda_quant.calib.sources import FileSource
from qda_quant.engine.build import build
from qda_quant.gguf import file, imatrix, rule
from qda_quant.steps.rtn import RTN

from .test_export import WIDE, _imatrix_file, _source


def _reference_values(self, name: str, start: int = 0, stop: int | None = None) -> torch.Tensor:
    """The two-pass reference: widen the BF16 bits before shifting them to FP32."""
    tensor = self.tensors[name]
    if tensor.tensor_type != rule.BF16:
        return _VALUES(self, name, start, stop)
    bits = tensor.data.view("<u2").reshape(-1, tensor.shape[0])[start:stop]
    return torch.from_numpy((bits.astype("<u4") << 16).view("<f4"))


_VALUES = file.Template.values


def test_every_bf16_pattern_and_expert_slice_keeps_its_bits(tmp_path: Path) -> None:
    """NaNs, subnormals and signed zero need bit equality, including flattened expert rows."""
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "patterns.gguf"
    bits = np.arange(65536, dtype="<u2").reshape(4, 64, 256)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(1)
    writer.add_tensor("blk.0.patterns", bits, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    template = file.Template(path)
    for start, stop in [(0, None), (0, 0), (1, 3), (63, 65), (64, 128), (255, 256)]:
        got = template.values("blk.0.patterns", start, stop).numpy().view("<u4")
        reference = _reference_values(template, "blk.0.patterns", start, stop).numpy().view("<u4")
        assert np.array_equal(got, reference)
        assert np.array_equal(got, bits.reshape(-1, 256)[start:stop].astype("<u4") << 16)


def test_one_pass_expansion_keeps_plain_and_weighted_whole_files(
    tmp_path: Path, monkeypatch, ggml_lib: Path
) -> None:  # noqa: ANN001
    """Layer packing and threaded template fill write the same files as two-pass expansion."""
    src, calib, template, checkout, _ = _source(tmp_path, WIDE)
    matrix = _imatrix_file(tmp_path / "imatrix.gguf", template, skip={"token_embd.weight"})

    def run(name: str, weighted: bool) -> tuple[str, dict]:
        out = build(
            str(src),
            Recipe(steps=[RTN(grid="q4k")], calib=FileSource(path=str(calib), samples=1, seqlen=8)),
            format="w4a16-asym-g32",
            out_dir=tmp_path / name,
            devices=["cpu"],
            dtype=torch.bfloat16,
            exporter={
                "gguf": {
                    "template": str(template),
                    "llama_cpp": str(checkout),
                    "libggml": str(ggml_lib),
                    "workers": 2,
                    "layer_pass": True,
                    **({"imatrix": str(matrix)} if weighted else {}),
                }
            },
            log=lambda _: None,
        )
        import json

        return imatrix.sha256(out / "model-Q4_K_M.gguf"), json.loads((out / "build.json").read_text())["gguf"]

    for weighted in (False, True):
        with monkeypatch.context() as slow:
            slow.setattr(file.Template, "values", _reference_values)
            reference = run(f"reference-{weighted}", weighted)
        assert run(f"one-pass-{weighted}", weighted) == reference
