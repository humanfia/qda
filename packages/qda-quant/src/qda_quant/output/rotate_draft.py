"""Rotate an external speculative draft into the residual basis of a rotated checkpoint.

A build with `rotate` runs the model in a rotated residual stream (readers ``W Q``, writers ``Q^T W``,
embedding ``E Q``), and a draft trained against the unrotated model reads the wrong basis there. The command
finds the checkpoint's ``Q`` (rebuilt from the rotate seeds its record names, or fitted to its embedding when it
has no record), checks it against the whole embedding, and writes the draft in one of two forms:

- ``rotated``: every draft weight moves into the rotated basis, as the checkpoint's own weights did. Exact only
  when the draft has nothing that fails to commute with ``Q`` (a residual-width norm gain that varies by channel,
  a per-channel convolution, a tensor sidecar), and used then because it adds nothing beyond the output head.
- ``original``: the draft keeps its own basis. Only ``fc``, which reads the target's hidden states, takes ``Q``;
  the draft gets the unrotated embedding and output head of the base model unless it ships its own. Exact for
  any draft whose only reader of the target is ``fc`` (DFlash, DFlash2, DSpark), at the cost of one more
  vocabulary-by-width matrix.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from ..exit import refuse
from ..intake.model_ref import parse
from ..intake.probe import _weight_map
from ..step import Refused
from .record import RESOLVED

READERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
WRITERS = ("o_proj", "down_proj")
RECORD = "rotate-draft.json"
Q_TOLERANCE = 1e-2  # over the whole embedding: Q reads ~2e-3 against bf16 storage, Q's transpose ~1.4
NORM_TOLERANCE = 1e-6  # a gain that varies by δ shifts the forward by about δ; the draft must match to 1e-5
TENSORS = (".pt", ".pth", ".bin", ".safetensors", ".npz", ".npy", ".ckpt")
# Training state some draft repositories ship beside the weights; no server reads it, so it is not copied.
TRAINING_STATE = frozenset({"optimizer_state_dict.pt", "scheduler_state_dict.pt"})
MASK_SIDECAR = "mask_embedding.pt"  # a residual-width vector vLLM loads for the mask token
# Drafts whose only reader of the target's hidden states is ``fc`` (vLLM 0.33: qwen3_dflash, qwen3_dflash2,
# qwen3_dspark, gemma4_dspark), so the ``original`` form is exact for every other tensor they carry.
FC_ONLY = frozenset(
    {"DFlashDraftModel", "DFlash2DraftModel", "DSparkDraftModel", "Qwen3DSparkModel", "Gemma4DSparkModel"}
)
# Drafts whose layers ship inside the target checkpoint: the build rotates them, as it does an MTP head.
IN_TARGET = frozenset({"DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"})
CARRIED = ("d2t", "t2d")  # vocabulary maps of a reduced-vocabulary draft
CHUNK = 8192
# A fitted Q this close to the identity everywhere is no rotation: a rotation build's Q has entries ~1/sqrt(d)
# and a diagonal far from 1; a checkpoint whose embedding was changed but not rotated fits one near I.
NEAR_IDENTITY = 0.5


class Model:
    """A model directory; ``where`` maps tensor names to shards, read one tensor at a time."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.where = _weight_map(self.path)
        self.config = json.loads((self.path / "config.json").read_text())

    def one(self, suffix: str) -> torch.Tensor:
        """The one stored tensor whose name ends in ``suffix``."""
        keys = [k for k in self.where if k == suffix or k.endswith("." + suffix)]
        if len(keys) != 1:
            raise Refused(f"{self.path}: expected one {suffix} tensor, found {len(keys)}")
        with safe_open(str(self.path / self.where[keys[0]]), "pt") as f:
            return f.get_tensor(keys[0])


def _error(m: torch.Tensor, got: torch.Tensor, want: torch.Tensor) -> float:
    """``|got @ m - want| / |want|`` over every row, in fp32 chunks: a sample can be blind to the difference."""
    err = ref = 0.0
    for i in range(0, len(want), CHUNK):
        g, w = got[i : i + CHUNK].float(), want[i : i + CHUNK].float()
        err += float((g @ m.float() - w).square().sum())
        ref += float(w.square().sum())
    return (err / ref) ** 0.5


def _embeddings(ckpt: Model, base: Model) -> tuple[torch.Tensor, torch.Tensor]:
    got, want = base.one("embed_tokens.weight"), ckpt.one("embed_tokens.weight")
    if got.shape != want.shape:
        raise Refused(f"{ckpt.path}: its embedding is {tuple(want.shape)} against {tuple(got.shape)}")
    return got, want


def fit_q(seeds: list[int], width: int, ckpt: Model, base: Model) -> tuple[torch.Tensor, float, bool]:
    """The checkpoint's ``Q`` (rebuilt from ``seeds``) or its transpose, whichever alone fits its embedding."""
    from ..steps.rotate import orthogonal

    q = torch.eye(width, dtype=torch.float64)
    for seed in seeds:
        q = q @ orthogonal(width, seed)
    got, want = _embeddings(ckpt, base)
    errors = {flipped: _error(m, got, want) for m, flipped in ((q, False), (q.T, True))}
    fits = [flipped for flipped, e in errors.items() if e <= Q_TOLERANCE]
    if len(fits) != 1:
        e = f"(errors {errors[False]:.2e}, {errors[True]:.2e})"
        if fits:
            raise Refused(f"{ckpt.path}: Q and Q^T both fit the embedding {e}; the basis is ambiguous")
        raise Refused(f"{ckpt.path}: Q and Q^T both miss the embedding {e}; the record is wrong")
    return (q.T if fits[0] else q), errors[fits[0]], fits[0]


def fit_embedding(ckpt: Model, base: Model) -> tuple[torch.Tensor, float]:
    """The orthogonal ``Q`` closest to mapping the base embedding onto the checkpoint's (Procrustes), checked.

    For a checkpoint without a build record, such as a published one: the embedding has far more rows than
    columns, so the fit is determined, and a base that is not this checkpoint's source misses the tolerance.
    """
    got, want = _embeddings(ckpt, base)
    if torch.equal(got, want):
        raise Refused(
            f"{ckpt.path}: its embedding is the base model's, so it is not rotated; use the draft as is"
        )
    m = torch.zeros(got.shape[1], got.shape[1], dtype=torch.float64)
    for i in range(0, len(want), CHUNK):
        m += got[i : i + CHUNK].double().T @ want[i : i + CHUNK].double()
    u, _, vh = torch.linalg.svd(m)
    q = u @ vh
    error = _error(q, got, want)
    if float((q - torch.eye(len(q), dtype=q.dtype)).abs().max()) < NEAR_IDENTITY:
        raise Refused(
            f"{ckpt.path}: its embedding differs from the base model's (error {error:.2e}) but by "
            "no rotation, so it is not rotated; use the draft as is"
        )
    if error > Q_TOLERANCE:
        raise Refused(
            f"{ckpt.path}: no rotation of {base.path}'s embedding gives its embedding (error {error:.2e}); "
            "pass the unrotated source model with --base"
        )
    return q, error


def _rotated(name: str, t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``t`` in the rotated basis, in fp64 then back to its dtype; ``t`` itself where the basis does not reach."""
    parts = name.split(".")
    module = parts[-2] if len(parts) > 1 else name
    if parts[-1] in CARRIED or parts[0] == "markov_head" or name == "confidence_head.proj.bias":
        return t  # token maps and the Markov head live in the vocabulary, not the residual stream
    w, d = t.double(), q.shape[0]
    if name == "fc.weight" and w.dim() == 2 and w.shape[0] == d and not w.shape[1] % d:
        out = torch.cat([q.T @ w[:, i : i + d] @ q for i in range(0, w.shape[1], d)], 1)
    elif name == "confidence_head.proj.weight" and w.dim() == 2 and w.shape[1] >= d:
        out = torch.cat([w[:, :d] @ q, w[:, d:]], 1)  # reads [hidden, markov embedding]
    elif (module in READERS or module in ("lm_head", "embed_tokens")) and w.dim() == 2 and w.shape[1] == d:
        out = w @ q
    elif module in WRITERS and w.dim() == 2 and w.shape[0] == d:
        out = q.T @ w
    elif module.endswith("norm") and w.dim() == 1:
        if w.shape[0] == d and float(w.max() - w.min()) > NORM_TOLERANCE * float(w.abs().max()):
            raise Refused(f"draft tensor {name}: its norm gain varies by channel, which does not commute")
        return t
    else:
        raise Refused(f"draft tensor {name} {list(t.shape)}: qda cannot place it in the residual stream")
    return out.to(t.dtype)


def _fc_input(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``fc`` reading the rotated target: each target layer's block of input columns times ``Q``."""
    d = q.shape[0]
    return torch.cat([w[:, i : i + d].double() @ q for i in range(0, w.shape[1], d)], 1).to(w.dtype)


def _view(cfg: dict) -> dict:
    """A draft config's fields, the speculators format's nested layer config included."""
    return (cfg.get("transformer_layer_config") or {}) | cfg


def _layer_ids(cfg: dict) -> list[int] | None:
    """The target layers whose hidden states ``fc`` reads, as the draft's config names them."""
    for nest, key in (
        ("dflash_config", "target_layer_ids"),
        ("eagle_config", "eagle_aux_hidden_state_layer_ids"),
        (None, "aux_hidden_state_layer_ids"),
        (None, "dspark_target_layer_ids"),
        (None, "target_layer_ids"),
    ):
        ids = (cfg.get(nest) or {}).get(key) if nest else cfg.get(key)
        if ids:
            return [int(i) for i in ids]
    return None


def _declared(draft_path: Path, cfg: dict) -> str | None:
    """The base model the draft says it was trained for: its README's ``base_model`` or its verifier."""
    page = draft_path / "README.md"
    readme = page.read_text(errors="replace") if page.is_file() else ""
    if m := re.search(r"^base_model:\s*(?:\[)?\s*-?\s*['\"]?([\w.-]+/[\w.-]+)", readme, re.M):
        return m[1]
    verifier = ((cfg.get("speculators_config") or {}).get("verifier") or {}).get("name_or_path")
    return verifier if isinstance(verifier, str) and verifier.count("/") == 1 else None


def residual_seeds(checkpoint: str | Path) -> list[int] | None:
    """The residual rotate seeds the checkpoint's record names; `None` when it has no run record at all.

    `resolved.yaml` is the one owner of which steps a build ran, so a reader that must know whether a
    checkpoint runs in a rotated basis (an end report, a delivery page) asks this, not its own parse.

    Args:
        checkpoint: A built checkpoint's directory.

    Returns:
        One seed per residual `rotate` step (`[]` when the record ran none, e.g. only value-head rotations).
    """
    resolved = Path(checkpoint) / RESOLVED
    if not resolved.is_file():
        return None
    record = yaml.safe_load(resolved.read_text()) or {}
    rotates = [s["rotate"] or {} for s in record.get("steps") or [] if "rotate" in s]
    return [r.get("seed", 0) for r in rotates if r.get("residual", True)]


def _record(ckpt_path: Path) -> tuple[list[int] | None, str | None, str]:
    """(residual rotate seeds, source model, built model id) from the checkpoint's record; Nones without one."""
    seeds = residual_seeds(ckpt_path)
    if seeds is None:
        return None, None, ""
    if not seeds:
        raise Refused(f"{ckpt_path}: no residual rotate step, so there is nothing to rotate")
    record = yaml.safe_load((ckpt_path / RESOLVED).read_text()) or {}
    source = (record.get("provenance") or {}).get("source_snapshot") or record.get("model")
    return seeds, source, parse(str(record.get("model") or "")).name


def beside(checkpoint: str | Path, draft: str | Path) -> Path:
    """Where a build writes its rotated copy of ``draft``: ``<checkpoint>-<draft name>``, next to it.

    The draft's own name is kept, so the ``dflash`` or ``dspark`` vLLM looks for in the path is kept too.
    """
    name = Path(parse(draft).name.rstrip("/")).name
    checkpoint = Path(checkpoint)
    return checkpoint.parent / f"{checkpoint.name}-{name}"


def rotate(
    checkpoint: str | Path, draft: str | Path, out: str | Path, base: str | Path | None = None
) -> Path:
    """Write ``draft`` rotated into ``checkpoint``'s residual basis at ``out``; nothing on ``Refused``."""
    from ..engine.planning import local_source

    ckpt_path = Path(local_source(checkpoint))
    seeds, recorded, built = _record(ckpt_path)
    if (out := Path(out)).exists():
        raise Refused(f"{out} exists")
    ckpt = Model(ckpt_path)
    text = ckpt.config.get("text_config") or ckpt.config
    if not isinstance(width := text.get("hidden_size"), int):
        raise Refused(f"{ckpt_path}: its config names no hidden_size")

    draft_path = Path(local_source(draft, weights=None))
    d = Model(draft_path)
    cfg = dict(d.config)
    view, arch = _view(cfg), (cfg.get("architectures") or [None])[0]
    if (draft_path / RECORD).is_file():
        raise Refused(f"{draft_path}: it carries {RECORD}, so `qda rotate-draft` wrote it already")
    if arch in IN_TARGET:
        raise Refused(
            f"{draft_path}: a {arch} draft ships inside the target checkpoint; the build rotates it"
        )
    declared = _declared(draft_path, cfg)
    if declared and built.count("/") == 1 and declared != built:
        raise Refused(f"{draft_path}: a draft for {declared}, not {built}")
    source = base or recorded or declared
    if not source:
        raise Refused("neither the checkpoint's record nor the draft names the source model; pass --base")
    base_model = Model(local_source(source))
    if seeds is None:
        q, error = fit_embedding(ckpt, base_model)
        transposed = None
    else:
        q, error, transposed = fit_q(seeds, width, ckpt, base_model)
    heads = [k for k in base_model.where if k.endswith("lm_head.weight")]
    head = base_model.one(heads[0] if heads else "embed_tokens.weight")

    skip = {d.path / shard for shard in d.where.values()}
    files = [f for f in d.path.rglob("*") if f.is_file() and f.suffix in TENSORS and f not in skip]
    if bad := [f for f in files if f.name not in TRAINING_STATE | {MASK_SIDECAR}]:
        raise Refused(f"{bad[0]}: a tensor file qda cannot carry into the rotated basis")
    dropped = sorted(f.name for f in files if f.name in TRAINING_STATE)
    layers, ids = text.get("num_hidden_layers"), _layer_ids(cfg)
    if view.get("vocab_size") not in (None, head.shape[0]) or cfg.get("num_target_layers") not in (
        None,
        layers,
    ):
        raise Refused(f"{draft_path}: not a draft for this model (vocabulary or target layers differ)")
    if ids and isinstance(layers, int) and not all(0 <= i < layers for i in ids):
        raise Refused(f"{draft_path}: it reads target layers {ids}, and the model has {layers}")

    raw = {}
    for name in sorted(d.where):
        with safe_open(str(draft_path / d.where[name]), "pt") as f:
            raw[name] = f.get_tensor(name)
    fc = raw.get("fc.weight")
    if fc is None or fc.dim() != 2 or fc.shape[1] % width or (ids and fc.shape[1] != len(ids) * width):
        shape = None if fc is None else list(fc.shape)
        raise Refused(
            f"{draft_path}: fc.weight {shape} does not read {len(ids or [])} hidden states of {width}"
        )

    why = None
    if (draft_path / MASK_SIDECAR).is_file():
        why = f"{MASK_SIDECAR} holds a vector in the draft's basis"
    elif view.get("hidden_size") not in (None, width):
        why = f"the draft is {view.get('hidden_size')} wide, the model {width}"
    else:
        try:
            tensors = {name: _rotated(name, t, q) for name, t in raw.items()}
        except Refused as e:
            why = str(e)
    if why is None:
        basis = "rotated"
        if not any("lm_head" in k for k in tensors):
            tensors["lm_head.weight"] = _rotated("lm_head.weight", head, q)
    elif arch not in FC_ONLY:
        raise Refused(
            f"{why}; and {arch or 'a draft without architectures'} is not known to read only through fc"
        )
    else:
        basis, tensors = "original", raw | {"fc.weight": _fc_input(fc, q)}
        if not any(k == "embed_tokens.weight" or k.endswith(".embed_tokens.weight") for k in raw):
            tensors["embed_tokens.weight"] = base_model.one("embed_tokens.weight")
        if not any("lm_head" in k for k in raw):
            tensors["lm_head.weight"] = head
    cfg["has_own_lm_head"] = True

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    try:
        keep = shutil.ignore_patterns(
            "*.safetensors", "config.json", "model.safetensors.index.json", *TRAINING_STATE
        )
        shutil.copytree(draft_path, tmp, ignore=keep, dirs_exist_ok=True)
        flat = {k: v.contiguous() for k, v in tensors.items()}
        save_file(flat, str(tmp / "model.safetensors"), {"format": "pt"})
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        meta = {"checkpoint": str(ckpt_path), "draft": str(draft), "base": str(base_model.path)}
        meta |= {"hidden_size": width, "seeds": seeds, "q_transposed": transposed, "q_relative_error": error}
        meta |= {"q_from": "embedding" if seeds is None else "seeds", "basis": basis}
        meta |= {"target_layer_ids": ids, "original_because": why, "not_copied": dropped}
        (tmp / RECORD).write_text(json.dumps(meta | {"tensors": len(tensors)}, indent=1) + "\n")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


EPILOG = """examples:
  qda rotate-draft --checkpoint qwen3.5-9b-w4a16 --draft z-lab/Qwen3.5-9B-DFlash --out qwen3.5-9b-dflash-rot
  qda rotate-draft --checkpoint qwen3.8-27b-w4a16 --draft RedHatAI/Qwen3.8-27B-speculator.dspark \\
      --out qwen3.8-27b-dspark-rot

vLLM detects a DFlash draft by "dflash" in its path: keep "dflash" (or "dspark") in --out.

exit status: 0 ok, 2 refused (nothing written)"""


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add ``rotate-draft`` to a subparser set."""
    p = sub.add_parser(
        "rotate-draft",
        help="rotate a speculative-decoding draft into a rotated checkpoint's residual basis",
        description=(__doc__ or "").split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="DIR|ID", help="a model built with `rotate`")
    p.add_argument("--draft", required=True, metavar="DIR|ID", help="the draft model to rotate")
    p.add_argument("--out", required=True, metavar="DIR", help="output directory (must not exist)")
    p.add_argument(
        "--base",
        metavar="DIR|ID",
        help="the unrotated model (default: the checkpoint's record, else the draft's declared base model)",
    )
    p.set_defaults(run=run)
    return p


def _dir_or_id(arg: str) -> str | Path:
    """An argument as a directory when one exists at that path (``ckpt``, not only ``./ckpt``), else a hub id."""
    return Path(arg) if Path(arg).is_dir() else arg


def run(a: argparse.Namespace) -> int:
    """Run a parsed ``rotate-draft`` command."""
    try:
        base = None if a.base is None else _dir_or_id(a.base)
        out = rotate(_dir_or_id(a.checkpoint), _dir_or_id(a.draft), a.out, base)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    record = json.loads((out / RECORD).read_text())
    print(json.dumps({"out": str(out), **record}))
    return 0
