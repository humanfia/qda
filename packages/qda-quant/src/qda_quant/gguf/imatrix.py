"""llama.cpp's importance matrix: the per-column weights ``llama-quantize --imatrix`` fits with.

``llama-imatrix`` measures each linear's inputs over a calibration text and writes, per tensor, the sum of
squared inputs (``.in_sum2``) and the number of rows summed (``.counts``). A column's importance is their
quotient, the mean squared input: ``load_imatrix`` (``tools/quantize/quantize.cpp``) computes exactly that,
and this module follows it, so a weighted fit here is the one llama.cpp would make from the same file.

The measurement belongs to one template: the file names tensors the way that template does, and a column count
that does not match is a different model, so both are refused rather than skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import sha256_file as sha256  # noqa: F401 - the exporter reads this public name
from ..step import Refused

#: ``general.type`` of a file ``llama-imatrix`` wrote
KIND = "imatrix"
_SUMS, _COUNTS = ".in_sum2", ".counts"


def load(path: str | Path) -> dict[str, np.ndarray]:
    """``{tensor name: importance per column}`` from an imatrix GGUF.

    Raises:
        Refused: not an imatrix file, or an entry without the counts to normalise it.
    """
    import gguf

    reader = gguf.GGUFReader(str(path))
    kind = reader.fields.get("general.type")
    if kind is None or kind.contents() != KIND:
        raise Refused(
            f"gguf: {path} is not an importance matrix (general.type is not {KIND!r}); "
            "write one with llama-imatrix -m TEMPLATE -f TEXT -o FILE"
        )
    sums = {t.name[: -len(_SUMS)]: t for t in reader.tensors if t.name.endswith(_SUMS)}
    counts = {t.name[: -len(_COUNTS)]: t for t in reader.tensors if t.name.endswith(_COUNTS)}
    out = {}
    for name, tensor in sums.items():
        if name not in counts:
            raise Refused(f"gguf: importance matrix entry {name} has no {_COUNTS[1:]} to normalise it")
        per_expert = np.asarray(counts[name].data, dtype=np.float64)
        values = np.asarray(tensor.data, dtype=np.float64).reshape(len(per_expert), -1)
        # `load_imatrix`: a column's importance is its mean squared input; an unseen expert weighs 1.
        seen = per_expert > 0
        out[name] = (
            np.where(seen[:, None], values / np.where(seen, per_expert, 1)[:, None], 1.0)
            .astype(np.float32)
            .reshape(-1)
        )
    return out


def column_weights(table: dict[str, np.ndarray], name: str, ne: tuple[int, ...]) -> np.ndarray | None:
    """The importance of `name`'s columns as ``(banks, width)``, or None if the matrix does not cover it.

    An expert bank is one GGUF tensor holding ``ne[2]`` experts, and ``llama-imatrix`` measures each expert
    separately, so its entry holds one width-wide vector per expert; a plain tensor's entry holds a single
    vector that every row shares. Both are returned as ``(banks, width)``, so the caller indexes by expert and
    never has to know which kind it got.

    A bank entry of a single vector is refused rather than broadcast to every expert. llama.cpp requires one
    per expert, so such a file was measured for a different model; accepting it would weigh all of a bank's
    experts by one expert's measurement and say nothing.

    None means the matrix does not cover this tensor, which is ordinary: ``llama-imatrix`` measures the inputs
    of matrix multiplications, so a lookup table like ``token_embd`` has none to measure, and the output head
    is skipped unless ``--process-output``. llama.cpp fits such a tensor unweighted, and so does this; the
    build record names each one rather than leaving it unsaid.

    Raises:
        Refused: the entry's size is neither one vector nor one per expert.
    """
    if name not in table:
        return None
    weights = table[name]
    width, experts = ne[0], (ne[2] if len(ne) > 2 else 1)
    if weights.size == width * experts:
        return weights.reshape(experts, width)
    want = f"{width} x {experts} experts = {width * experts}" if experts > 1 else str(width)
    raise Refused(
        f"gguf: importance matrix entry {name} has {weights.size} values, expected {want} for a tensor of "
        f"shape {tuple(ne)}; it was measured on another model or template"
    )
