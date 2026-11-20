"""Select transformers model classes for planning, loading and source-name conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .model_ref import parse

if TYPE_CHECKING:
    from ..target import Target


def load_config(source: str | Path) -> Any:
    """The transformers config of `source`, read after architecture plugins register their classes.

    Raises:
        Refused: transformers cannot read the config.
    """
    from transformers import AutoConfig

    from ..arch.adapters import register_models
    from ..arch.derive_trace import _import_fla
    from ..step import Refused, probe_refusal, remote_code_refusal

    _import_fla()  # before AutoConfig: fla's import must not choose the model class
    register_models()
    try:
        name, revision = parse(source)
        return AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=False)
    except Refused:
        raise
    except (ValueError, OSError) as e:  # OSError: a missing or malformed config.json
        if remote := remote_code_refusal(source, "config", e):
            raise remote from e
        raise Refused(f"model {source}: cannot read its transformers config: {e}") from e
    except Exception as e:  # noqa: BLE001 - a config QDA cannot read; refuse it, do not traceback
        raise probe_refusal(source, "reading its transformers config", e) from e


def load_model(
    source: str | Path | None = None,
    *,
    config: Any = None,
    weights: bool = True,
    target: Target | None = None,
    images: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a checkpoint, or construct its architecture when only `config` is supplied.

    Image calibration requires transformers' image-text mapping so the vision tower reaches the language
    layers. Otherwise prefer the causal-LM mapping, falling back to image-text when it has no entry. Text
    builds keep their existing class, checkpoint bytes and memory estimate.

    Args:
        source: Checkpoint directory or Hub id; None constructs a model from `config` without weights.
        config: Transformers config; read from `source` when absent.
        weights: False constructs the float architecture from the source config without reading weights.
        target: Bind architecture numerics to this explicit build target before returning the model.
        images: Require a model that accepts image inputs, as declared by the calibration source.
        **kwargs: Passed unchanged to transformers' construction or loading method.

    Raises:
        Refused: No installed transformers mapping handles the configuration.
    """
    from ..arch.adapters import configure_model, register_models
    from ..arch.derive_trace import _import_fla

    _import_fla()  # before AutoConfig/AutoModel: fla's import must not choose the model class
    register_models()
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    )

    from ..step import Refused, probe_refusal

    if weights and source is not None and (why := _serve_only(source)):
        raise Refused(f"model {source}: this QDA build is serve-only: {why}")
    if config is None:
        if source is None:
            raise ValueError("load_model needs a source or config")
        config = load_config(source)
    if not weights and getattr(config, "quantization_config", None) is not None:
        delattr(config, "quantization_config")
    candidates = (
        (AutoModelForCausalLM, MODEL_FOR_CAUSAL_LM_MAPPING),
        (AutoModelForImageTextToText, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING),
    )
    for auto, mapping in candidates[1:] if images else candidates:
        if type(config) in mapping:
            try:
                if source is None or not weights:
                    model = auto.from_config(config, **kwargs)
                else:
                    ref = parse(source)
                    model = auto.from_pretrained(ref.name, config=config, **ref.hub(), **kwargs)
                if target is not None:
                    configure_model(model, target)
                from ..arch.derive import (
                    route_gdn,
                )  # local: importing torch is the loader's job, not the module's

                route_gdn(model)  # a CUDA forward keeps fla's GDN kernels; CPU and meta take the reference
                if source is not None:  # a probe refusal names what the user asked for, not the cache path
                    try:
                        model._qda_source = str(source)
                    except Exception:  # noqa: BLE001 - a model that does not take attributes still loads
                        pass
                return model
            except Refused:
                raise
            except ValueError as e:
                raise Refused(f"model type {config.model_type!r}: {auto.__name__} cannot load it: {e}") from e
            except Exception as e:  # noqa: BLE001 - the class cannot build this config; refuse it
                raise probe_refusal(
                    source if source is not None else config, "building its model class", e
                ) from e
    if images:
        raise Refused(
            f"model type {config.model_type!r} ({type(config).__name__}): image calibration requires an "
            "installed transformers image-text-to-text mapping"
        )
    raise Refused(
        f"model type {config.model_type!r} ({type(config).__name__}): installed transformers has no "
        "causal-LM or image-text-to-text mapping"
    )


def _serve_only(source: str | Path) -> str | None:
    """Why QDA marks `source` serve-only (its ``build.json``), if it does.

    A build keeps a routed expert bank float when ``ignore`` names the whole bank. Stored one module per expert,
    stock transformers 5 opens the checkpoint without an error and leaves every routed expert at its random
    initialization, so the export records ``build.json["serve_only"]`` and QDA's own transformers load refuses
    it. A source that is not a QDA checkpoint, or whose record cannot be read, is left to the load's own checks.
    """
    record = Path(source) / "build.json"
    if not record.is_file():
        return None
    try:
        info = json.loads(record.read_text())
    except (OSError, ValueError):
        return None
    mark = info.get("serve_only") if isinstance(info, dict) else None
    return mark.get("why") if isinstance(mark, dict) else None
