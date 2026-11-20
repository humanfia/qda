"""Configure one checkpoint scoring run with ``qda-kl score CONFIG.yaml``.

    checkpoint: path/to/checkpoint      # or a .gguf file, decoded to bf16 first (qda_kl.gguf_checkpoint)
    reference: path/to/reference        # built once with `qda-kl ref`; or one qda.datasets {key: settings}
    out: path/to/run                    # gets score.json and resolved.yaml
    scorer: {topk: {repeat: 1}}         # one scorer by key (qda_kl.scorers); default: topk

Use ``--set a.b=c`` for dotted overrides, such as ``--set scorer.topk.repeat=3``. The run records defaults
and provenance in ``out/resolved.yaml``; pass that file to ``qda-kl score`` to repeat the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from qda_quant.intake.fingerprint import model_signature
from qda_quant.intake.model_ref import parse
from qda_quant.output.record import without_provenance
from qda_quant.recipe import apply_overrides, error_lines
from qda_quant.step import Refused

from qda_kl.datasets import KL_REFERENCE, FileDataset, parse_dataset, with_defaults
from qda_kl.scorers import KLScorer, parse_scorer

#: The result file in ``out``.
RESULT = "score.json"

Model = TypeVar("Model", bound=BaseModel)


def source_identity(model: str | Path) -> dict[str, str | None]:
    """One model's source facts, as a build records them for its single source.

    A run that reads several models (an eval reads a reference and every checkpoint) cannot use the three
    single-source keys of ``resolved.yaml``, so it records one of these per model. A local model directory is
    pinned by its content signature (`qda_quant.intake.fingerprint.model_signature`, one header read per
    weight file); a hub id by the revision in ``org/name@rev``; a hub id without a revision pins nothing and
    says so. ``snapshot`` (the local directory a hub id resolved to) is left null: resolving an id here would
    download it before a run that may refuse it.

    A directory that is not a model (no ``config.json``, e.g. a ``qda-kl ref`` reference) is not signed: the
    signature reads every non-weight file whole, and a large reference corpus is not a model to pin.

    Args:
        model: the model directory or hub id, exactly as the config names it.

    Returns:
        ``{"snapshot": ..., "revision": ..., "signature": ...}``.

    Raises:
        Refused: the directory is a model but its signature cannot be read (a broken link in it).
    """
    d = Path(str(model))
    if d.is_dir():
        signable = (d / "config.json").is_file()
        return {
            "snapshot": None,
            "revision": None,
            "signature": model_signature(d) if signable else None,
        }
    return {"snapshot": None, "revision": parse(str(model)).revision, "signature": None}


def check_checkpoint(checkpoint: str) -> None:
    """Refuse a checkpoint that is a missing local path, before an engine reads it as a Hub repo id.

    A build publishes a directory, so a failed build leaves nothing at the output path. vLLM then reads the
    name as a Hub id and fails with "is not a local folder and is not a valid model identifier listed on
    https://huggingface.co/models", which reads like a typo rather than a build that refused. A path that is
    absolute, starts with ``.`` or ``~``, or has no ``/`` cannot be a Hub id (those are ``org/name``), so a
    missing one is named as the missing directory it is; a ``org/name`` is left to the engine.

    Raises:
        Refused: the path is a local one and no directory is there.
    """
    path = Path(checkpoint).expanduser()
    local = path.is_absolute() or checkpoint.startswith((".", "~")) or "/" not in checkpoint
    if local and not path.is_dir():
        # a bare name is read as a local directory; a Hub model of the old one-word kind (``gpt2``) is now
        # published under its organisation, so the refusal says how to name it
        hint = "" if "/" in checkpoint else "; a Hub model is named org/name"
        raise Refused(f"no checkpoint at {checkpoint} (did the build fail?{hint})")


def load_raw(
    text: str, overrides: list[str] | tuple[str, ...] = (), *, what: str = "qda-kl"
) -> dict[str, Any]:
    """Read YAML configuration data, drop provenance and apply dotted overrides before validation.

    Args:
        text: The YAML document.
        overrides: ``dotted.key=value`` assignments, applied after provenance is dropped.
        what: The noun naming the config in the error, for example ``"qda eval"``.

    Returns:
        The config mapping, ready for validation.

    Raises:
        Refused: The YAML is invalid, is not a mapping, or an override cannot be applied.
    """
    try:
        d = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise Refused(f"not valid YAML: {e}") from None
    if not isinstance(d, dict):
        raise Refused(f"a {what} config is a YAML mapping")
    return apply_overrides(without_provenance(d), overrides)


def validate(cls: type[Model], raw: dict[str, Any]) -> Model:
    """Validate configuration data returned by :func:`load_raw`.

    Args:
        cls: The config model to validate.
        raw: The configuration mapping.

    Returns:
        The validated config.

    Raises:
        Refused: Invalid keys, reported as semicolon-separated ``key: reason`` messages.
    """
    try:
        return cls(**raw)
    except ValidationError as e:
        raise Refused("; ".join(error_lines(e))) from None


class ScoreConfig(BaseModel):
    """Validate the checkpoint, reference, output directory and scorer for one run.

    Attributes:
        checkpoint: Checkpoint directory loaded by vLLM, or a `.gguf` file, which is decoded to bf16 in
            ``out/decoded`` and scored (`qda_kl.gguf_checkpoint`).
        reference: Original model's reference directory, built with ``qda-kl ref``: a path, or one
            ``{key: settings}`` from ``qda.datasets``.
        out: Run directory for ``score.json`` and ``resolved.yaml``.
        scorer: Scorer settings; defaults to the top-k scorer.
        snapshot: The HF snapshot a `.gguf` checkpoint was converted from; default: the reference's model.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    checkpoint: str
    reference: str | dict[str, Any]
    out: str
    scorer: KLScorer = Field(default_factory=lambda: parse_scorer(None))
    snapshot: str | None = None

    @field_validator("reference")
    @classmethod
    def _reference(cls, v: str | dict[str, Any]) -> str | dict[str, Any]:
        return with_defaults(v, "reference", KL_REFERENCE)

    def reference_path(self) -> Path:
        """The local reference directory.

        A plain path (the ``file`` entry) must exist: a scorer is handed the path as given, and only the
        built-in ``topk`` would refuse a missing one, later.

        Raises:
            Refused: the dataset cannot make it available, or a local path holds nothing.
        """
        dataset = parse_dataset(self.reference, "reference")
        path = dataset.local()
        if isinstance(dataset, FileDataset) and not path.exists():
            raise Refused(
                f"reference: no reference at {path}",
                f"build it with `qda-kl ref --model ORIGINAL --prompts PROMPTS --out {path}`, or point `reference` at "
                "an existing one",
            )
        return path

    @field_validator("scorer", mode="before")
    @classmethod
    def _named(cls, v: Any) -> Any:
        return v if isinstance(v, KLScorer) else parse_scorer(v)

    def to_dict(self) -> dict[str, Any]:
        """Return plain configuration data with defaults filled in for ``resolved.yaml``."""
        return {
            "checkpoint": self.checkpoint,
            "reference": self.reference,
            "out": self.out,
            "scorer": self.scorer.to_dict(),
            **({"snapshot": self.snapshot} if self.snapshot else {}),
        }

    @staticmethod
    def raw(text: str, overrides: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        """Read configuration data and apply dotted overrides before validation.

        Drop provenance so resolved configs can be used as input.

        Raises:
            Refused: The YAML is invalid, is not a mapping, or an override cannot be applied.
        """
        return load_raw(text, overrides, what="qda-kl score")

    @classmethod
    def from_yaml(cls, text: str, overrides: list[str] | tuple[str, ...] = ()) -> ScoreConfig:
        """Parse YAML, apply overrides and validate the scoring configuration.

        Raises:
            Refused: The config is invalid; validation messages name the affected keys.
        """
        return cls.from_raw(cls.raw(text, overrides))

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ScoreConfig:
        """Validate configuration data returned by :meth:`raw`.

        Raises:
            Refused: Invalid keys, reported as semicolon-separated ``key: reason`` messages.
        """
        return validate(cls, raw)
