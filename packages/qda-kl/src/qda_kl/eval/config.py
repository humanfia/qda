"""Configure ``qda eval CONFIG.yaml``: checkpoints scored against one reference by named suites.

    reference: Qwen/Qwen3.5-9B           # the original model (local dir or HF id)
    checkpoints:                         # key (the name in tables) -> local dir or HF id
      ours-w4a16: path/to/q35-w4a16
      redhat: RedHatAI/Qwen3.5-9B-quantized.w4a16
    out: runs/eval-q35                   # results/<key>/<suite>.json, refs/, resolved.yaml
    suites:                              # one {key: settings} per entry (qda.eval_suites)
      - kl: {corpus: {file: {path: prompts.jsonl}}}
      # - full: {}                       # the reference model's released-card settings; expands to lm_eval
      #                                  # (MMLU-Pro included) and livecodebench (qda_kl.eval.full_preset)
    report:                              # how `qda eval table` summarizes the KL suite's sequences
      weights: {hard: 0.9, easy: 0.1}
    paired:                              # how `qda eval paired` draws its bootstrap intervals
      resamples: 1000
      seed: 17

Use ``--set a.b=c`` for dotted overrides (``--set suites.0.kl.answers=4``). The run records defaults and
provenance in ``out/resolved.yaml``; that file works as a config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from qda_quant.step import Refused

from qda_kl.config import load_raw, validate
from qda_kl.eval.report import Paired, Report
from qda_kl.eval.suites import Suite, parse_suite

#: A checkpoint key names a directory under ``out/results`` and is matched by ``--only KEY[/SUITE[.LABEL]]``.
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _is_gguf(model: str) -> bool:
    """Whether ``model`` names a GGUF file, or a directory holding one and no HF config."""
    path = Path(model).expanduser()
    if path.is_dir():
        return not (path / "config.json").is_file() and any(path.glob("*.gguf"))
    return path.suffix.lower() == ".gguf"


class EvalConfig(BaseModel):
    """Validate the reference, checkpoints, output directory and suites of one eval run.

    Attributes:
        reference: the original model (local directory or HF id, ``org/name@rev`` pins a revision).
        checkpoints: checkpoint key -> local directory or HF id.
        out: run directory.
        suites: the suites every checkpoint is scored by, each key at most once.
        report: how ``qda eval table`` summarizes the KL suite's sequences (weights, bootstrap, easy guard);
            reporting settings, so changing them never reruns a score.
        paired: how ``qda eval paired`` draws its confidence intervals (resamples, RNG seed); recorded in
            ``resolved.yaml``, so a rendering is reproducible from the run directory alone.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    reference: str
    checkpoints: dict[str, str]
    out: str
    suites: list[Suite]
    report: Report = Field(default_factory=Report)
    paired: Paired = Field(default_factory=Paired)

    @field_validator("checkpoints")
    @classmethod
    def _keys(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise Refused("give at least one checkpoint")
        bad = [k for k in v if not KEY.fullmatch(k)]
        if bad:
            raise Refused(f"keys {bad}: use letters, digits, '.', '_' and '-' (a key names a directory)")
        return v

    @field_validator("suites", mode="before")
    @classmethod
    def _named(cls, v: Any) -> Any:
        if not isinstance(v, list) or not v:
            raise Refused("give a non-empty list of {suite: settings}")
        suites = [parse_suite(d, f"suites.{i}") for i, d in enumerate(v)]
        names = [s.name for s in suites]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise Refused(f"suites {dup} given twice; one entry per suite")
        return suites

    _full_note: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def _expand_full(self) -> EvalConfig:
        """Replace a ``full`` entry with the table's ``lm_eval`` and ``livecodebench`` entries.

        The table needs the original model, which only this config knows (its reference, else a checkpoint's
        ``base_model``), so the expansion runs here and its result is what ``resolved.yaml`` records; a later
        run of that file is the same measurement without the table. The note naming the row applied is kept
        for the command to print as its first line.
        """
        from qda_kl.eval.full_preset import FullSuite

        if not any(isinstance(suite, FullSuite) for suite in self.suites):
            return self
        notes: list[str] = []
        suites: list[Suite] = []
        for suite in self.suites:
            if isinstance(suite, FullSuite):
                got = suite.expand(self.reference, list(self.checkpoints.values()))
                notes.append(got.note)
                suites.extend(parse_suite(entry, "full") for entry in got.entries)
            else:
                suites.append(suite)
        names = [s.name for s in suites]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise Refused(
                f"suites {dup} given twice: a full entry expands to lm_eval and livecodebench; "
                "remove the explicit duplicate or the full entry"
            )
        self.suites = suites
        self._full_note = "\n".join(notes)
        return self

    @property
    def full_note(self) -> str:
        """The line naming the ``full`` row applied (empty when the config has no ``full`` entry)."""
        return self._full_note

    def to_dict(self) -> dict[str, Any]:
        """Return plain configuration data with defaults filled in for ``resolved.yaml``."""
        return {
            "reference": self.reference,
            "checkpoints": dict(self.checkpoints),
            "out": self.out,
            "suites": [s.to_dict() for s in self.suites],
            "report": self.report.model_dump(mode="json"),
            "paired": self.paired.model_dump(mode="json"),
        }

    @staticmethod
    def raw(text: str, overrides: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        """Read configuration data, drop provenance and apply dotted overrides before validation.

        Raises:
            Refused: the YAML is invalid, is not a mapping, or an override cannot be applied.
        """
        return load_raw(text, overrides, what="qda eval")

    @classmethod
    def from_yaml(cls, text: str, overrides: list[str] | tuple[str, ...] = ()) -> EvalConfig:
        """Parse YAML, apply overrides and validate.

        Raises:
            Refused: the config is invalid; messages name the affected keys.
        """
        cfg = validate(cls, cls.raw(text, overrides))
        _refuse_gguf(cfg)
        return cfg


def _refuse_gguf(cfg: EvalConfig) -> None:
    """Refuse a .gguf model: every suite loads its checkpoint in vLLM, which has no GGUF loader.

    Raises:
        Refused: the reference or a checkpoint is a GGUF file or a GGUF build directory.
    """
    named = {"reference": cfg.reference}
    named.update({f"checkpoints.{key}": model for key, model in cfg.checkpoints.items()})
    files = sorted(name for name, model in named.items() if _is_gguf(model))
    if files:
        raise Refused(
            f"{', '.join(files)}: qda eval does not score a .gguf model: its suites load it in vLLM,"
            " which has no GGUF loader",
            "decode the file to a bf16 checkpoint (`qda decode-gguf`), then evaluate that directory",
        )
