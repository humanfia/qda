"""Plan-time helpers of ``scale_kd``: teacher and config checks, the rotary rule, the float-tensor selection.

`_float_params` selects the float tensors `trainable` names (``norms`` and ``gdn``, defined in the ``scale_kd``
module docstring); the rest read configs and tokenizers without loading weights.
"""

from __future__ import annotations

import json
from collections.abc import Container
from pathlib import Path
from typing import Any

import torch

#: rotary types whose frequencies do not depend on the input's length (transformers recomputes them only for
#: ``dynamic*`` and ``longrope``), so a row padded in a group sees the positions it sees alone
BATCHED_ROPE = frozenset({"default", "linear", "llama3", "yarn"})


def _float_params(model: torch.nn.Module, kinds: Container[str]) -> list[tuple[str, torch.nn.Parameter, str]]:
    """(name, parameter, kind) of the float tensors `kinds` select (see module docstring)."""
    from ..engine.served_modules import ServedLinear

    out = []
    seen: set[int] = set()
    lin_attn = [n for n, m in model.named_modules() if _is_linear_attention(m)]
    for mname, mod in model.named_modules():
        if isinstance(mod, ServedLinear):
            continue
        norm = "Norm" in type(mod).__name__
        gdn = any(mname == a or mname.startswith(a + ".") for a in lin_attn)
        kind = "norms" if norm else "gdn" if gdn else None
        if kind is None or kind not in kinds:
            continue
        for pn, p in mod.named_parameters(recurse=False):
            if id(p) not in seen:
                seen.add(id(p))
                out.append((f"{mname}.{pn}" if mname else pn, p, kind))
    return out


def _is_linear_attention(mod: torch.nn.Module) -> bool:
    """A Gated DeltaNet block: it owns the ``A_log`` / ``dt_bias`` decay parameters."""
    own = dict(mod.named_parameters(recurse=False))
    return "A_log" in own and "dt_bias" in own


def batched_rope(student: Any, teacher: Any) -> str | None:
    """None when every rotary type of both configs is position-fixed (`BATCHED_ROPE`), else the first other one.

    One rule for `ScaleKD.resolve` and `ScaleKD.check_checkpoint`: student and teacher both run padded groups.
    """
    other = sorted((_rope_types(student) | _rope_types(teacher)) - BATCHED_ROPE)
    return other[0] if other else None


def _rope_types(config: Any) -> set[str]:
    """The rotary types of a config's text model, ``default`` when none is named.

    Read from ``rope_parameters`` (one dict, or one per layer type) and the legacy ``rope_scaling``.
    """
    text = config.get_text_config()
    out = set()
    for params in (getattr(text, "rope_parameters", None), getattr(text, "rope_scaling", None)):
        if not isinstance(params, dict):
            continue
        for p in (
            params.values() if all(isinstance(v, dict) for v in params.values()) and params else [params]
        ):
            out.add(p.get("rope_type") or p.get("type") or "default")
    return out or {"default"}


def _same_model(teacher: Any, student: Any) -> tuple[Any, Any] | None:
    """None if the two configs describe the same model class, else (teacher's, student's) description.

    Compared: ``architectures`` as each ``config.json`` names them, and the text model's ``model_type``. Either
    side may lack ``architectures`` (a config written by hand); then the model types alone decide.
    """
    ta, sa = teacher.architectures or [], student.architectures or []
    if ta and sa and ta != sa:
        return ta, sa
    tt, st = teacher.get_text_config().model_type, student.get_text_config().model_type
    if tt != st:
        return f"{ta or tt} ({tt})", f"{sa or st} ({st})"
    return None


def _token_ids(model_dir: str | Path) -> tuple | None:
    """The token -> id map of a model's ``tokenizer.json`` (vocabulary and added tokens), None without one.

    Compared instead of the file's bytes: re-saving a tokenizer (llm-compressor does) changes the bytes, not the
    ids the logits are indexed by.
    """
    p = Path(model_dir) / "tokenizer.json"
    if not p.is_file():
        return None
    t = json.loads(p.read_text())
    added = sorted((a["id"], a["content"]) for a in t.get("added_tokens", []))
    return t.get("model", {}).get("type"), t.get("model", {}).get("vocab"), added
