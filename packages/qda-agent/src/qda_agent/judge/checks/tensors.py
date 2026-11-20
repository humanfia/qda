"""Reading a checkpoint's tensors: its shards, index, inventory and fingerprint, and near-copies of a baseline.

Moved here from `checks.py` (pure move): the safetensors reader's own rules (`loaded`, `_storage_problem`),
the loaded inventory (`inventory`, `module_tensors`), the baseline fingerprint (`tensor_fingerprint`) and the
near-copy comparison (`near_copy`). `checks` re-exports the public names.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import struct
from collections.abc import Iterator
from pathlib import Path

from ..freeze import INDEX

#: Tensors no KL reading scores, left out of the baseline fingerprint (vision towers, the MTP head).
UNSCORED = ("model.visual.*", "model.vision_tower.*", "model.embed_vision.*", "mtp.*")
#: A quantized module's stored tensors, by the suffix of their names: its packed codes and their parameters.
#: An NVFP4 module stores no zero point but two fp32 global scales beside its E4M3 group scales.
QUANTIZED = (
    (".weight_packed", "packed"),
    (".weight_scale", "scale"),
    (".weight_zero_point", "zero_point"),
    (".weight_global_scale", "weight_global_scale"),
    (".input_global_scale", "input_global_scale"),
)
#: The stored tensors of a quantized module (and a fused bank's own shape metadata), by suffix and role.
STORED = (*QUANTIZED, (".weight_shape", "shape"))
#: A checkpoint's other packed storage (autoawq's), by suffix and role: only the modules it packs store them.
AWQ = ((".qweight", "packed"), (".qzeros", "zero_point"), (".scales", "scale"))
#: The hidden units the serving engine fuses into one projection, and the leaves its config groups name them by.
QKV = ("q_proj", "k_proj", "v_proj")
GATE_UP = ("gate_proj", "up_proj")
#: The routed-expert projections qda's writer stores as one fused tensor per bank, and their leaves.
FUSED = ((".experts.gate_up_proj", ("gate_proj", "up_proj")), (".experts.down_proj", ("down_proj",)))
#: A candidate whose quantized tensors differ from a baseline's by less than this relative difference in
#: weight space (over the packed codes and their scales) is that baseline as-is: the one-ulp trick moves a
#: checkpoint by ~1e-6, a rebuild by more than 1e-2.
NEAR_COPY = 1e-3
#: The span `near_copy` reads at a time (its values become float64, so it is smaller than `READ_SPAN`).
DIFF_SPAN = 16 << 20
#: The stored dtypes `near_copy` reads, as numpy's and as struct's: bf16 is not a struct type (read by its bits).
NUMERIC = {"F16": "<f2", "F32": "<f4", "I8": "<i1", "U8": "<u1", "I32": "<i4", "I64": "<i8"}
STRUCT = {"F16": "<e", "F32": "<f", "I8": "<b", "U8": "<B", "I32": "<i", "I64": "<q"}


def _e4m3() -> tuple[float, ...]:
    """FP8 E4M3's 256 values as `torch.float8_e4m3fn` stores them: bias 7, `0x7F`/`0xFF` NaN, subnormals 2**-9."""
    out = []
    for b in range(256):
        sign, exp, mant = (-1.0 if b & 0x80 else 1.0), (b >> 3) & 0xF, b & 0x7
        if exp == 0xF and mant == 0x7:
            out.append(float("nan"))
        elif exp == 0:
            out.append(sign * mant * 2.0**-9)
        else:
            out.append(sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7))
    return tuple(out)


#: FP8 E4M3 by its byte: NVFP4's `weight_scale` (neither numpy nor struct has the type).
E4M3 = _e4m3()
#: Bits of one stored value, by the dtype names the safetensors reader accepts (`F4` and `F6_*` pack several
#: values into a byte, so a shape's bit count must be whole bytes).
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "F64": 64,
    "I64": 64,
    "U64": 64,
    "C64": 64,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}
MAX_HEADER = 100 << 20
#: The token embedding (not an MTP head's or a vision tower's), checked against the format's grid.
EMBEDDING = re.compile(r"(^|\.)embed_tokens\.weight$")
#: The token embedding a tied model shares with its output head (Gemma's per-layer embedding is not it).
TIED_EMBEDDING = ("embed_tokens", "token_embedding", "input_embeddings", "word_embeddings", "wte", "shared")
READ_SPAN = 256 << 20


class CheckpointError(ValueError):
    """What is wrong with the checkpoint itself (a shard, a header, the index); its words are the agent's."""


def _json(p: Path) -> object:
    return json.loads(p.read_text())


def _storage_problem(dtype: str, shape: list[int], offsets: list[int]) -> str | None:
    """Why a header entry's dtype, shape and stored span do not agree, or None.

    The safetensors reader refuses a tensor that is not exactly its shape's worth of its dtype; checking the
    same rule here keeps the judge from calling a file servable that the serving engine cannot open. A shape
    with a zero dimension is valid and stores no bytes.
    """
    bits = DTYPE_BITS.get(dtype)
    if bits is None:
        return f"{dtype!r} is not a safetensors dtype"
    if any(x < 0 for x in shape):
        return f"shape {shape} has a negative dimension"
    total = math.prod(shape) * bits
    if total % 8:
        return f"shape {shape} of {dtype} is not a whole number of bytes"
    stored = offsets[1] - offsets[0]
    if stored != total // 8:
        return f"{stored} stored bytes for shape {shape} of {dtype} ({total // 8} needed)"
    return None


def loaded(d: Path) -> dict[str, tuple[Path, dict, int]]:
    """The tensors a loader takes from a checkpoint: name -> (shard, header entry, data start).

    From the shard index's `weight_map` when there is one, else every top-level `*.safetensors` file's.

    Raises:
        CheckpointError: an index entry naming no shard file of the checkpoint, a shard the index does not
            name, an index tensor no shard holds, a tensor in two shards or mapped elsewhere by the index,
            a file that is not safetensors.
    """
    files = sorted(p.name for p in d.glob("*.safetensors"))
    wm = None
    if (d / INDEX).is_file():
        try:
            doc = _json(d / INDEX)
            if not isinstance(doc, dict):
                raise TypeError(f"{INDEX} is not a JSON object")
            wm = doc["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise CheckpointError(f"{INDEX} is not a shard index") from e
        if not isinstance(wm, dict) or not all(isinstance(v, str) for v in wm.values()):
            raise CheckpointError(f"{INDEX}: weight_map is not a name -> shard map")
        named = set(wm.values())
        if bad := sorted(x for x in named if "/" in x or x.startswith(".") or x not in files):
            raise CheckpointError(f"{INDEX} names {bad[0]!r}, not a shard file of the checkpoint")
        if extra := sorted(set(files) - named):
            raise CheckpointError(f"{extra[0]}: a shard the index does not name")
    out: dict[str, tuple[Path, dict, int]] = {}
    for name in files:
        p = d / name
        with p.open("rb") as fh:
            size = p.stat().st_size
            n = int.from_bytes(fh.read(8), "little")
            if not 0 < n <= MAX_HEADER or n > size - 8:
                raise CheckpointError(f"{name}: not a safetensors file (header length {n})")
            try:
                h = json.loads(fh.read(n))
            except (ValueError, RecursionError):
                raise CheckpointError(f"{name}: not a safetensors header") from None
        if not isinstance(h, dict):
            raise CheckpointError(f"{name}: not a safetensors header")
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a = v.get("data_offsets") if isinstance(v, dict) else None
            if not (
                isinstance(a, list)
                and len(a) == 2
                and all(type(x) is int for x in a)
                and 0 <= a[0] <= a[1] <= size - 8 - n
                and isinstance(v.get("dtype"), str)
                and isinstance(v.get("shape"), list)
                and all(type(x) is int for x in v["shape"])
            ):
                raise CheckpointError(f"{name}: {k[:80]}: not a safetensors tensor entry")
            if why := _storage_problem(v["dtype"], v["shape"], a):
                raise CheckpointError(f"{name}: {k[:80]}: {why}")
            if k in out:
                raise CheckpointError(f"{k[:80]}: in {out[k][0].name} and {name}")
            if wm is not None and wm.get(k) != name:
                raise CheckpointError(f"{name}: {k[:80]}: the index maps it to {wm.get(k)!r}")
            out[k] = (p, v, 8 + n)
    if wm is not None and (missing := sorted(set(wm) - out.keys())):
        raise CheckpointError(f"{INDEX} names {missing[0][:80]!r}, but no shard holds it")
    return out


def inventory(d: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The loaded tensors: name -> (dtype, shape)."""
    return {k: (v["dtype"], tuple(v["shape"])) for k, (_, v, _) in loaded(d).items()}


def module_tensors(d: Path) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    """Each quantized module a checkpoint stores: module -> role -> (dtype, shape).

    A dense unit is named by its module. A routed expert bank's fused projections are named
    `<bank>.experts.<projection>`; a bank stored per expert by `<bank>.experts.<index>.<leaf>`. Roles are
    `packed`, `scale`, `zero_point`, an NVFP4 module's `weight_global_scale` and `input_global_scale`, and
    `shape` (a fused form's own shape metadata).

    Raises:
        CheckpointError: `loaded`'s.
    """
    return _module_tensors(inventory(d))


def _module_tensors(
    inv: dict[str, tuple[str, tuple[int, ...]]],
) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    """`module_tensors` from a checkpoint's inventory."""
    out: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for name, t in inv.items():
        for suffix, role in STORED:
            if name.endswith(suffix):
                out.setdefault(name[: -len(suffix)], {})[role] = t
                break
        else:
            # a fused bank's tensor is named `<projection>_<role>`: keep the projection, drop the role
            for suffix, _ in FUSED:
                hit = next(
                    (
                        r
                        for r in (
                            "packed",
                            "scale",
                            "zero_point",
                            "weight_global_scale",
                            "input_global_scale",
                            "shape",
                        )
                        if name.endswith(f"{suffix}_{r}")
                    ),
                    None,
                )
                if hit is not None:
                    out.setdefault(name[: -len(hit) - 1], {})[hit] = t
                    break
            else:
                for suffix, role in AWQ:
                    if name.endswith(suffix):
                        out.setdefault(name[: -len(suffix)], {})[role] = t
                        break
    return {m: r for m, r in out.items() if "packed" in r}


def _modules(inv: dict[str, tuple[str, tuple[int, ...]]]) -> list[str]:
    """Every module a format fact can be about: the stored quantized ones and every one with a 2-D weight."""
    out = set(_module_tensors(inv))
    out.update(
        n[: -len(".weight")] for n, (_, shape) in inv.items() if n.endswith(".weight") and len(shape) == 2
    )
    return sorted(out)


def tensor_fingerprint(d: Path, unscored: tuple[str, ...] | list[str] = UNSCORED) -> str:
    """sha256 over the scored loaded tensors' (name, dtype, shape, sha256 of the bytes), sorted.

    Independent of shard names, sharding and header metadata: a baseline renamed, re-sharded or re-saved
    has the same fingerprint.

    Raises:
        CheckpointError: `loaded`'s, or a shard that ends before its header says.
    """
    by: dict[Path, list[tuple[int, int, str, dict]]] = {}
    for k, (p, v, start) in loaded(d).items():
        if not any(fnmatch.fnmatchcase(k, g) for g in unscored):
            a = v["data_offsets"]
            by.setdefault(p, []).append((start + a[0], start + a[1], k, v))
    items = set()
    for p, ts in by.items():
        with p.open("rb", buffering=0) as fh:
            for b, e, k, v in sorted(ts):
                h, at = hashlib.sha256(), b
                fh.seek(b)
                while at < e:
                    chunk = fh.read(min(READ_SPAN, e - at))
                    if not chunk:
                        raise CheckpointError(f"{p.name} ends before {k[:80]}")
                    h.update(chunk)
                    at += len(chunk)
                items.add(json.dumps([k, v["dtype"], v["shape"], h.hexdigest()]))
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def _quantized(d: Path) -> dict[str, dict[str, tuple[Path, dict, int]]]:
    """The quantized modules a checkpoint stores: name -> its "packed", "scale" (and "zero_point") tensors."""
    out: dict[str, dict[str, tuple[Path, dict, int]]] = {}
    for k, t in loaded(d).items():
        for suffix, slot in QUANTIZED:
            if k.endswith(suffix):
                out.setdefault(k[: -len(suffix)], {})[slot] = t
    return {m: t for m, t in out.items() if "packed" in t and "scale" in t}


def _difference(a: tuple[Path, dict, int], b: tuple[Path, dict, int]) -> tuple[float, float]:
    """The squared magnitude of one tensor's stored values and of its difference from another's.

    Raises:
        CheckpointError: A shard ends before its header says.
    """
    pa, va, sa = a
    pb, vb, sb = b
    n = va["data_offsets"][1] - va["data_offsets"][0]
    a2 = d2 = 0.0
    with pa.open("rb", buffering=0) as fa, pb.open("rb", buffering=0) as fb:
        fa.seek(sa + va["data_offsets"][0])
        fb.seek(sb + vb["data_offsets"][0])
        done = 0
        while done < n:
            span = fa.read(min(DIFF_SPAN, n - done))
            other = fb.read(len(span))
            if not span or len(other) != len(span):
                raise CheckpointError(f"{pa.name} ends before its header says")
            x, y = _norms(span, other, va["dtype"])
            a2, d2 = a2 + x, d2 + y
            done += len(span)
    return a2, d2


def _norms(a: bytes, b: bytes, dtype: str) -> tuple[float, float]:
    """(Σa², Σ(a−b)²) over two equal-length spans of stored `dtype` values."""
    try:
        import numpy as np
    except ImportError:
        return _norms_slow(a, b, dtype)
    x, y = _float64(np, a, dtype), _float64(np, b, dtype)
    d = x - y
    return float(np.dot(x, x)), float(np.dot(d, d))


def _float64(np, raw: bytes, dtype: str):  # numpy, imported lazily by _norms
    """A span's stored values as float64: bf16 by its bits, E4M3 by its table, a code by its integer value."""
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4").astype(np.float64)
    if dtype == "F8_E4M3":
        return np.asarray(E4M3, dtype=np.float64)[np.frombuffer(raw, dtype=np.uint8)]
    if dtype not in NUMERIC:
        raise CheckpointError(f"{dtype}: not a numeric tensor dtype")
    return np.frombuffer(raw, dtype=np.dtype(NUMERIC[dtype])).astype(np.float64)


def _norms_slow(a: bytes, b: bytes, dtype: str) -> tuple[float, float]:
    """`_norms` without numpy: the same values, one at a time (a checkpoint takes minutes)."""
    a2 = d2 = 0.0
    for i in range(0, len(a), 1 << 20):  # a megabyte at a time, so no whole checkpoint is held
        x, y = _values(a[i : i + (1 << 20)], dtype), _values(b[i : i + (1 << 20)], dtype)
        for u, v in zip(x, y, strict=True):
            a2 += u * u
            d2 += (u - v) ** 2
    return a2, d2


def _values(raw: bytes, dtype: str) -> Iterator[float]:
    """A span's stored values, lazily."""
    if dtype == "BF16":
        return (struct.unpack("<f", b"\0\0" + raw[i : i + 2])[0] for i in range(0, len(raw), 2))
    if dtype == "F8_E4M3":
        return (E4M3[b] for b in raw)
    if dtype not in STRUCT:
        raise CheckpointError(f"{dtype}: not a numeric tensor dtype")
    return (v for (v,) in struct.iter_unpack(STRUCT[dtype], raw))


def near_copy(candidate: Path, baseline: Path) -> str | None:
    """Why `candidate` is `baseline` as-is (or within `NEAR_COPY` of it), or None.

    The quantized modules are paired by name and compared on their packed codes, scales, zero points and
    NVFP4 global scales:
    a candidate whose packed codes are all the baseline's is refused whatever its scales and zero points
    are (the same weights, re-scaled), and one whose every module's stored values differ by less than
    `NEAR_COPY` of their magnitude is refused as within tolerance. A checkpoint with no quantized tensor,
    or with modules, dtypes or shapes its baseline does not have, is not compared.

    Raises:
        CheckpointError: `loaded`'s, or a shard that ends before its header says.
    """
    mine, theirs = _quantized(candidate), _quantized(baseline)
    if not mine or set(mine) != set(theirs):
        return None
    codes, params, worst = True, True, 0.0
    for name, modules in mine.items():
        a2 = d2 = 0.0
        for _, slot in QUANTIZED:
            a, b = modules.get(slot), theirs[name].get(slot)
            if a is None or b is None:
                if a is not b:
                    return None
                continue
            if (a[1]["dtype"], a[1]["shape"]) != (b[1]["dtype"], b[1]["shape"]):
                return None
            x, y = _difference(a, b)
            a2, d2 = a2 + x, d2 + y
            if slot == "packed":
                codes = codes and y == 0.0
            else:
                params = params and y == 0.0
        worst = max(worst, (d2 / a2) ** 0.5 if a2 > 0 else (0.0 if d2 == 0 else float("inf")))
        if worst >= NEAR_COPY and not codes:  # this module differs too much, and the codes are not all equal
            return None
    if codes:
        return (
            "a baseline checkpoint as-is, not your result"
            if params
            else "a baseline's packed codes with its scales changed, not your result"
        )
    if worst < NEAR_COPY:
        return (
            f"within {NEAR_COPY:g} of a baseline in weight space (packed codes and scales), not your result"
        )
    return None
