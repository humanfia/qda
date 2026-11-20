"""The released cards' sampling per original model, and the ``generation_config.json`` a build writes from it.

Engines take a checkpoint's default sampling and stop ids from its ``generation_config.json``. Some originals ship
none (``Qwen/Qwen3.5-4B``); their recommended sampling is on the model card only. A build from such a source
writes one from this table, so a checkpoint built from a recipe serves with the same defaults as a released one.
``qda_kl``'s ``full`` eval preset reads its sampling from the same table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["CARD_SAMPLING", "card_generation_config", "hub_id"]

#: Sampling of the Qwen3.5-9B/4B and Qwen3.6-35B-A3B cards: temperature/top_p/top_k/min_p/presence/repetition.
SAMPLING_QWEN35: dict[str, float] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
#: Sampling of the Qwen3.8-27B cards (no presence penalty).
SAMPLING_QWEN38: dict[str, float] = {**SAMPLING_QWEN35, "presence_penalty": 0.0}
#: Sampling of the DeepSeek-R1-Distill-Llama-8B cards.
SAMPLING_R1: dict[str, float] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
#: Sampling of the Gemma-4 cards.
SAMPLING_GEMMA: dict[str, float] = {**SAMPLING_R1, "temperature": 1.0, "top_k": 64}
#: Sampling of the Nemotron-3-Nano cards.
SAMPLING_NEMOTRON: dict[str, float] = {**SAMPLING_R1, "temperature": 1.0, "top_p": 1.0, "top_k": -1}
#: Sampling of the Qwen3-Coder cards.
SAMPLING_CODER: dict[str, float] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.05,
}

#: The original model's HF id -> the sampling its released cards record.
CARD_SAMPLING: dict[str, dict[str, float]] = {
    "Qwen/Qwen3.5-4B": SAMPLING_QWEN35,
    "Qwen/Qwen3.5-9B": SAMPLING_QWEN35,
    "Qwen/Qwen3.6-35B-A3B": SAMPLING_QWEN35,
    "Qwen/Qwen3.8-27B": SAMPLING_QWEN38,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": SAMPLING_CODER,
    "google/gemma-4-26B-A4B-it": SAMPLING_GEMMA,
    "google/gemma-4-31B-it": SAMPLING_GEMMA,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": SAMPLING_NEMOTRON,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": SAMPLING_R1,
}


def hub_id(name: str | Path) -> str:
    """The HF id a name denotes: ``org/name`` for a hub id or a path inside the Hub cache, else the name."""
    text = str(name).split("@", 1)[0].rstrip("/")
    if m := re.search(r"models--([^/]+)--([^/]+?)(?:/snapshots/|$)", text):
        return f"{m.group(1)}/{m.group(2)}"
    return text


def _ids(value: object) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return [v for v in value if isinstance(v, int)] if isinstance(value, list) else []


def _tokenizer_eos(src: Path) -> list[int]:
    """The id of the token ``tokenizer_config.json`` names as eos, looked up in ``tokenizer.json``."""
    try:
        eos = json.loads((src / "tokenizer_config.json").read_text()).get("eos_token")
        tok = json.loads((src / "tokenizer.json").read_text())
    except (OSError, ValueError):
        return []
    if isinstance(eos, dict):
        eos = eos.get("content")
    if not isinstance(eos, str):
        return []
    for added in tok.get("added_tokens", []):
        if added.get("content") == eos:
            return _ids(added.get("id"))
    vocab = (tok.get("model") or {}).get("vocab")
    return _ids(vocab.get(eos)) if isinstance(vocab, dict) else []


def card_generation_config(src: Path, config: dict) -> tuple[dict, str] | None:
    """The ``generation_config.json`` to write for a source that ships none, and the HF id whose row it came from.

    ``src`` is the source directory (a Hub snapshot for a hub id) and ``config`` its ``config.json``. The file holds
    the row's sampling and, as ``eos_token_id``, the config's eos ids followed by the tokenizer's eos token (the
    end-of-turn token of a chat model). None when ``src`` is not a Hub snapshot of a model the table names.
    """
    origin = hub_id(src)
    sampling = CARD_SAMPLING.get(origin)
    if sampling is None:
        return None
    eos = _ids(config.get("eos_token_id")) or _ids((config.get("text_config") or {}).get("eos_token_id"))
    eos += [i for i in _tokenizer_eos(src) if i not in eos]
    gen: dict = {"do_sample": True, **sampling}
    if eos:
        gen["eos_token_id"] = eos if len(eos) > 1 else eos[0]
    return gen, origin
