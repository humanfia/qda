"""The `--suite` presets of `qda eval CKPT|RUN_DIR`: `kl`, `sanity`, `full` and `specdec`.

`full` is the released cards' benchmark protocol: the six `lm_eval` tasks (with MMLU-Pro on the model's own
sampling) and LiveCodeBench v6, with per-model answer caps, sampling, thinking tag, chat-template arguments,
repeats and seeds. Those values live in one table, `qda_kl.eval.full_preset`, which a config's `- full: {}`
reads too; this module only names the preset. The original model is known once the checkpoint's `build.json`
or the run's `resolved.yaml` is read, so the row is expanded there and the line naming it printed first.

A free `{suite: settings}` entry is still accepted for a one-off benchmark; the four presets are what the docs
advertise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PRESETS", "Group"]


@dataclass(frozen=True)
class Group:
    """One `qda eval` run of a preset: its label (the output directory) and its suite entries.

    Attributes:
        label: The group's name; its config and table land under `evaluate/<preset>/<label>` when it differs
            from the preset's own name (`evaluate/full`).
        suites: The `{suite: settings}` entries, each suite key at most once (`qda eval` refuses a repeat).
    """

    label: str
    suites: tuple[dict[str, Any], ...]


#: The four presets `qda eval --suite` takes. `full` is one `qda eval` run: its entry is expanded per model by
#: `qda_kl.eval.full_preset` (the one table); the LiveCodeBench suite is a plugin (qda-agent registers it).
PRESETS: dict[str, tuple[Group, ...]] = {
    "kl": (Group("kl", ({"kl": {}},)),),
    "sanity": (Group("sanity", ({"sanity": {}},)),),
    "full": (Group("full", ({"full": {}},)),),
    "specdec": (Group("specdec", ({"specdec": {}},)),),
}
