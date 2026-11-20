"""Load the tokenization pipeline stored in a checkpoint without rebuilding it from a class name."""

import json
from pathlib import Path
from typing import Any


def load_tokenizer(source: str | Path, *, revision: str | None = None) -> Any:
    """Read a serialized fast tokenizer, or use the declared class when none is stored.

    Transformers may map an older class name to a constructor that replaces the checkpoint's
    pre-tokenizer and decoder. A ``tokenizer.json`` already specifies the complete pipeline; loading it
    through the generic fast class preserves its encoding and decoding, along with the checkpoint's
    special tokens and chat template.

    Args:
        source: Checkpoint directory or Hub id.
        revision: Hub revision, when the source is a Hub id.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.utils.hub import cached_file

    from ..step import Refused, probe_refusal, remote_code_refusal

    try:
        serialized = cached_file(
            str(source), "tokenizer.json", revision=revision, _raise_exceptions_for_missing_entries=False
        )
        cls = PreTrainedTokenizerFast if serialized is not None else AutoTokenizer
        if revision is None:
            tokenizer = cls.from_pretrained(str(source), trust_remote_code=False)
        else:
            tokenizer = cls.from_pretrained(str(source), revision=revision, trust_remote_code=False)
        try:
            # a probe refusal names what the user asked for, not the cache path
            tokenizer._qda_source = str(source)
        except Exception:  # noqa: BLE001 - a tokenizer that does not take attributes still works
            pass
        return tokenizer
    except Refused:
        raise
    except Exception as e:  # noqa: BLE001 - a tokenizer QDA cannot read; refuse it, do not traceback
        if remote := remote_code_refusal(source, "tokenizer", e):
            raise remote from e
        raise probe_refusal(
            source,
            "reading its tokenizer",
            e,
            next_step="check the model id, the network, and HF_TOKEN for a gated model",
        ) from e


#: Probes that separate a ByteLevel pipeline from a class that rebuilds one (whitespace and a newline).
TOKENIZER_PROBES = ("Hello world, this is a test.\nNext line.", "  two  spaces\n\ttab", "What is 2 + 2?")


def tokenizer_config_record(src: Path) -> dict[str, Any]:
    """Decide whether the source's declared tokenizer class preserves its serialized ``tokenizer.json``.

    Transformers maps some declarations to a constructor that replaces the stored pipeline: ``LlamaTokenizerFast``
    over the R1-Distill snapshots becomes LlamaTokenizer with Metaspace/ByteFallback, so text encodes to different
    ids than the checkpoint's own ``tokenizer.json``. The rule covers a ByteLevel pipeline, the case the
    replacement hits: the check builds the declared class (`AutoTokenizer`) and the serialized pipeline
    (`load_tokenizer`, which honours ``tokenizer.json``) and compares ids on whitespace-sensitive probes. A
    mismatch is corrected in the export by declaring ``PreTrainedTokenizerFast``; a comparison that cannot run
    is refused rather than copied unchanged. Any other pipeline is recorded as out of scope and copied unchanged.

    Returns:
        The ``build.json`` record: ``declared``, ``serialized_ok`` (True agreed, False corrected, None when no
        comparison applies), ``written`` (``"unchanged"`` or the corrected class) and, when nothing was compared,
        ``reason``. Empty when the source has no ``tokenizer_config.json``, so nothing is declared to check.
    """
    config_file = src / "tokenizer_config.json"
    if not config_file.is_file():
        return {}
    from ..step import Refused

    try:
        declared = json.loads(config_file.read_text()).get("tokenizer_class")
    except Exception as e:  # noqa: BLE001 - refused by name below, never copied unchanged silently
        raise Refused(
            f"tokenizer_config: cannot read {config_file} ({e}); it must be a JSON object with an optional "
            "tokenizer_class, or be removed from the checkpoint"
        ) from None
    if not (src / "tokenizer.json").is_file():
        return {
            "declared": declared,
            "serialized_ok": None,
            "written": "unchanged",
            "reason": "no tokenizer.json",
        }

    try:
        from transformers import AutoTokenizer

        bytelevel = _bytelevel_pipeline(json.loads((src / "tokenizer.json").read_text()))
        agrees = None
        if bytelevel:
            built, serialized = (
                AutoTokenizer.from_pretrained(str(src), trust_remote_code=False),
                load_tokenizer(src),
            )
            agrees = all(
                built.encode(p, add_special_tokens=False) == serialized.encode(p, add_special_tokens=False)
                for p in TOKENIZER_PROBES
            )
    except Exception as e:  # noqa: BLE001 - refused by name below, never copied unchanged silently
        raise Refused(
            f"tokenizer_config: the declared class {declared!r} over the source's tokenizer.json could not be "
            f"compared with the serialized pipeline ({e}); set tokenizer_class to PreTrainedTokenizerFast in "
            "tokenizer_config.json, or build without this tokenizer"
        ) from None
    if agrees is None:
        return {
            "declared": declared,
            "serialized_ok": None,
            "written": "unchanged",
            "reason": "not a ByteLevel pipeline",
        }
    if agrees:
        return {"declared": declared, "serialized_ok": True, "written": "unchanged"}
    return {"declared": declared, "serialized_ok": False, "written": "PreTrainedTokenizerFast"}


def _bytelevel_pipeline(doc: dict[str, Any]) -> bool:
    """Whether a serialized tokenizer's pre-tokenizer or decoder uses ByteLevel."""

    def walk(v: Any) -> bool:
        if isinstance(v, dict):
            return v.get("type") == "ByteLevel" or any(walk(x) for x in v.values())
        return any(walk(x) for x in v) if isinstance(v, list) else False

    return any(walk(doc.get(k)) for k in ("pre_tokenizer", "decoder"))
