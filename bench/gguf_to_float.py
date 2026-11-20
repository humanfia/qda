"""Decode a quantized GGUF into a bf16 checkpoint, so ``qda-kl score`` can read the weights the file holds.

vLLM 0.33 has no GGUF loader. This writes what the file stores, through llama.cpp's own ``dequantize_row_*``, as a
float checkpoint of the source model; every file scored this way goes through the same path. It measures the weights,
not llama.cpp's kernels, and bf16 rounds the decoded values (far below the quantization error).

    python bench/gguf_to_float.py FILE.gguf SOURCE_DIR OUT_DIR path/to/libggml-base.so

Refuses arch ``llama``, whose q/k rows the converter permutes.
"""

from __future__ import annotations

import argparse
import ctypes
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gguf
import numpy as np
import torch
from qda_quant.gguf import file, rule
from qda_quant.gguf.writer import _weight_map
from safetensors import safe_open
from safetensors.torch import save_file

_DECODE = {
    rule.Q4_K: "dequantize_row_q4_K",
    rule.Q6_K: "dequantize_row_q6_K",
    rule.Q5_0: "dequantize_row_q5_0",
    rule.Q8_0: "dequantize_row_q8_0",
}


def decode(tp: file.Template, name: str, lib: ctypes.CDLL) -> torch.Tensor:
    """A tensor of `tp` as fp32 ``[rows, width]``."""
    t = tp.tensors[name]
    if t.tensor_type not in _DECODE:
        return tp.values(name)
    out = np.empty((tp.rows(name), t.shape[0]), dtype=np.float32)
    fn = getattr(lib, _DECODE[t.tensor_type])
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    fn(t.data.ctypes.data, out.ctypes.data, out.size)
    return torch.from_numpy(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gguf", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("libggml", type=Path)
    a = p.parse_args()

    tp = file.Template(a.gguf)
    if tp.arch == "llama":
        raise SystemExit("arch llama: the converter permutes q/k rows, which this script does not undo")
    lib = ctypes.CDLL(str(a.libggml.resolve()))
    arch = next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == tp.arch)
    names = gguf.get_tensor_name_map(arch, tp.n_layer)
    where = _weight_map(a.source)

    def one(key: str) -> tuple[str, torch.Tensor]:
        name = names.get_name(key, try_suffixes=(".weight", ".bias"))
        if name is None or name not in tp.tensors:
            raise KeyError(f"{key}: no tensor {name} in {a.gguf}")
        with safe_open(a.source / where[key], "pt") as h:
            shape = h.get_slice(key).get_shape()
        return key, decode(tp, name, lib).reshape(shape).to(torch.bfloat16)

    with ThreadPoolExecutor(32) as pool:
        tensors = dict(pool.map(one, sorted(where)))
    a.out.mkdir(parents=True)
    save_file(tensors, a.out / "model.safetensors", metadata={"format": "pt"})
    for f in a.source.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy(f, a.out / f.name)
    print(f"{a.gguf}: {len(tensors)} tensors decoded into {a.out}")


if __name__ == "__main__":
    main()
