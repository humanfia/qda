"""Check the token-id meaning used to teacher-force a frozen reference."""

import hashlib
import json
from pathlib import Path
from typing import Any


def declared_stops(model: str | Path, revision: str | None = None) -> int | list[int] | None:
    """The end-of-sequence ids ``model`` declares: its ``generation_config.json``'s, else its ``config.json``'s.

    A ``generation_config.json`` that names no ``eos_token_id`` (a published checkpoint's file may carry only
    the sampling settings) declares no stops of its own, so ``config.json`` answers, as it does for a model
    with no ``generation_config.json`` at all. None when neither file can be read or names one.
    """
    from transformers import AutoConfig, GenerationConfig

    options: dict = {"revision": revision} if revision is not None else {}
    try:
        stops = GenerationConfig.from_pretrained(model, **options).eos_token_id
    except (OSError, ValueError):  # no generation_config.json: the engine takes config.json's EOS instead
        stops = None
    if stops is not None:
        return stops
    try:
        config = AutoConfig.from_pretrained(model, **options)
    except (OSError, ValueError):  # no model config either: tokenizer EOS only
        return None
    return GenerationConfig.from_model_config(config).eos_token_id


def _identity(tokenizer: Any, stop_ids=None, revision=None) -> dict:
    """Token meanings and roles, plus effective stops; additional special-token tags are irrelevant."""
    if stop_ids is None and (model := getattr(tokenizer, "name_or_path", "")):
        stop_ids = declared_stops(model, revision)
    stops = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids or [])
    if tokenizer.eos_token_id is not None:
        stops.add(tokenizer.eos_token_id)
    roles = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    return {
        "tokens": sorted((index, token) for token, index in tokenizer.get_vocab().items()),
        "stop_ids": sorted(stops),
        "roles": {key: getattr(tokenizer, key) for key in roles},
    }


def _sha(identity: dict) -> str:
    """Hash a normalized token identity."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def tokenizer_sha256(tokenizer: Any, *, stop_ids=None) -> str:
    """Hash token meanings, roles and used stops, independently of metadata tags and stop ordering."""
    return _sha(_identity(tokenizer, stop_ids))


def original_tokenizer(original: dict, copy: Path | None = None) -> Any:
    """The reference's own anchor tokenizer: its copy inside the reference, else the model it records.

    A reference built by this version holds `copy` and needs no model directory. One built before it names a
    path instead, which may since have moved or been deleted; the refusal then says what it looked for, where,
    and how to get a reference that does not need it.

    Raises:
        Refused: there is no copy and the recorded path cannot be loaded, or no identity is recorded at all.
    """
    from qda_quant.step import Refused

    from qda_kl.tokenizer_check import REBUILD, effective_tokenizer

    if copy is not None:
        return effective_tokenizer(copy)
    model = original.get("path") or original.get("name")
    if not model:
        raise Refused("the reference records no original model identity; cannot verify token ids")
    try:
        return effective_tokenizer(model, revision=original.get("revision"))
    except (OSError, ValueError) as error:
        raise Refused(
            f"the reference holds no tokenizer/ copy and its original model {str(model)!r} cannot be loaded "
            f"({error}); restore that directory's tokenizer files (tokenizer.json, tokenizer_config.json and "
            f"the chat template), or {REBUILD}"
        ) from error


def check_tokenizer(
    original: dict, candidate: Any, *, copy: Path | None = None, original_stop_ids=None, stop_ids=None
) -> str:
    """Refuse a candidate whose token meanings, roles or effective stop ids differ from the reference's anchor.

    The anchor is the reference's own tokenizer copy (`copy`), else the model it records, loaded as the
    reference's engine used it: with ``qda build``'s class correction applied when its declared class rebuilds
    its ``tokenizer.json`` (:func:`qda_kl.tokenizer_check.effective_tokenizer`).
    """
    anchor = original_tokenizer(original, copy)
    left = _identity(anchor, revision=None if copy is not None else original.get("revision"))
    if original_stop_ids:
        left["stop_ids"] = sorted(set(left["stop_ids"]) | set(original_stop_ids))
    right = _identity(candidate, stop_ids)
    expected, actual = _sha(left), _sha(right)
    if expected != actual:
        parts = ", ".join(key for key in left if left[key] != right[key])
        raise ValueError(
            f"tokenizer fingerprint mismatch in {parts}: original sha256 {expected}, candidate sha256 {actual}"
        )
    return expected
