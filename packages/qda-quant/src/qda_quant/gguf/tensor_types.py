"""Validated GGUF tensor type overrides, applied after the default Q4_K_M mix."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..step import Refused
from . import file, rule


class TensorTypeRule(BaseModel):
    """An exact GGUF tensor name or ``re:`` pattern, and the ggml type it stores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: str
    type: str

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        """Reject an empty name or an invalid regex before a build starts."""
        if not value:
            raise ValueError("tensor pattern must not be empty")
        if value.startswith("re:"):
            try:
                re.compile(value[3:])
            except re.error as e:
                raise ValueError(f"{value!r}: not a regex ({e})") from None
        return value

    @model_validator(mode="after")
    def valid_type(self) -> TensorTypeRule:
        """Reject types without an encoder or a known file layout."""
        try:
            file.type_id(self.type)
        except ValueError as e:
            raise ValueError(f"tensor pattern {self.pattern!r}: {e}") from None
        return self


def dropped_head_hint(
    message: str, overrides: list[TensorTypeRule], full: file.Template, view: file.Template
) -> str:
    """`message` plus a fix line when a rule names only tensors the headless view dropped.

    A recipe written for a file with the MTP head can state rules for the head's own tensors. With ``mtp:
    false`` those rules match nothing and `resolve_types` refuses -- correctly, a rule that names no tensor is a
    typo (or a leftover) -- but the fix is not to look for a misspelled tensor: the rules belong to the head the
    recipe just dropped. Only the head's tensors count as targeted; any other unmatched pattern keeps its message.
    """
    missing = set(full.tensors) - set(view.tensors)
    if not missing:
        return message
    for override in overrides:
        pattern = override.pattern
        regex = re.compile(pattern[3:]) if pattern.startswith("re:") else None
        if any(regex.match(name) if regex else name == pattern for name in view.tensors):
            continue
        if any(regex.match(name) if regex else name == pattern for name in missing):
            return (
                f"{message}\n"
                f"these rules only act on the MTP head block that `mtp: false` leaves out; delete {pattern!r}"
            )
    return message


def resolve_types(template: file.Template, overrides: list[TensorTypeRule]) -> dict[str, int]:
    """Resolve the mix, then apply overrides in order; the last matching rule wins.

    Raises:
        Refused: a pattern matches nothing, a non-F32 type is selected for a one-dimensional norm,
            or a selected type cannot hold a tensor's rows.
    """
    types = rule.q4_k_m_types(template.arch, template.n_layer, template.shapes())
    selected: dict[str, str] = {}
    for override in overrides:
        regex = re.compile(override.pattern[3:]) if override.pattern.startswith("re:") else None
        matches = [name for name in types if (regex.match(name) if regex else name == override.pattern)]
        if not matches:
            raise Refused(f"gguf: tensor pattern {override.pattern!r} matches no tensor in the template")
        kind = file.type_id(override.type)
        for name in matches:
            types[name] = kind
            selected[name] = override.type
    for name, kind in selected.items():
        shape = template.tensors[name].shape
        if len(shape) == 1 and "norm" in name and types[name] != rule.F32:
            raise Refused(
                f"gguf: tensor {name}: type {kind}: one-dimensional norms require F32 "
                f"for llama.cpp's elementwise multiplication; set {{pattern: {name}, type: F32}}"
            )
        try:
            file.row_bytes(types[name], shape[0])
        except ValueError as e:
            raise Refused(
                f"gguf: tensor {name}: type {kind}: {e}; choose a type that fits its rows"
            ) from None
    return types
