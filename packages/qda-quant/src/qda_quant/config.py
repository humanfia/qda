"""`BuildConfig`: one YAML file describes one build, and every build writes the file that repeats it.

    model: Qwen/Qwen3-8B
    format: w4a16-sym-g128
    steps: [{gptq: {actorder: static}}]
    calib: {file: {path: calib.jsonl, samples: 128, seqlen: 2048}}
    out: ckpt/

``qda build CONFIG.yaml --set out=other/`` reads it; ``--set`` overrides one key by its dotted path (the same
rule as recipe overrides, `qda_quant.recipe.apply_overrides`). The config is a thin layer over
`qda_quant.build`: `BuildConfig.build` passes its values to that function, and that function writes
``resolved.yaml`` into the output: this config with every default filled in, plus a ``provenance`` section
(the qda git commit, the local snapshot or content signature of the model, and the package and version of every
plugin the run used). Building from ``resolved.yaml`` with another ``out`` repeats the configuration. Layer and
model steps without gradients reproduce checkpoint bytes on the same GPU type, software stack and card count;
gradient methods can vary between runs. ``provenance`` is a record and is ignored when the file is read back.

Config file paths resolve beside the file (`qda_quant.recipe.resolve_paths`) and are recorded absolute.
A bare model or teacher with no local directory there stays a Hub id; output paths stay as given.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from .arch import KEEP_ROLES
from .clips import CLIPS
from .engine.layer_cache import LayerCacheSetting
from .engine.stats import DEFAULT_HESSIAN, hessian_setting
from .format import FORMATS, Format
from .intake.fingerprint import model_signature
from .output.exporters import EXPORTERS, Exporter, check_exporter, parse_exporter, record_exporter
from .output.record import plugins, provenance, without_provenance, write_resolved
from .recipe import Recipe, apply_overrides, error_fix, error_lines, expand_path, resolve_paths
from .registry import Registry
from .solve.grid import DEFAULT_GRID, GRIDS
from .step import Refused
from .target import BINDINGS, TARGETS, Target

if TYPE_CHECKING:
    import torch

    from .engine.planning import Plan
    from .engine.stats import HessianSetting


class BuildConfig(BaseModel):
    """Everything one build reads. Unknown keys are refused, never ignored.

    Attributes:
        model: local path (absolute after file resolution) or Hub id; ``org/name@rev`` pins a revision.
        format: stored format (``w4a16``, ``w4a16-sym-g128``, ...); kept in canonical form. Optional when
            `model` is a packed checkpoint (it has one; a given format must agree).
        format_overrides: rules that store chosen layer units in another format,
            ``[{pattern: ..., format: ...}]``: a pattern is an exact module name as ``qda plan`` prints it, or
            ``re:`` and a regex matched from the start (the rule of `ignore`), and names a whole solve unit (a
            dense group, or every expert bank). A pattern matching no unit or part of a fused group, a unit two
            rules store differently, and a format the target cannot serve are refused at plan time; the body
            ``format`` stores every other unit, and heads stay with `steps.head`.
        formats: the same recipe as one mapping, ``{default: w4a16-g128, lm_head: bf16,
            layers.0-3.mlp: w8a16-g32}`` (see `qda_quant.format_recipe`): sugar compiled at load into
            ``format``, `format_overrides` and `ignore`. A pattern is an exact module name, ``re:`` and a
            regex, or ``layers.<a>-<b>[.<suffix>]``; a value is a format name or ``bf16``. A ``bf16`` entry
            wins over a format entry that also names the unit (``ignore`` removes it before the overrides
            resolve), so ``layers.0-3.mlp: w8a16-g32`` with ``layers.0-3.mlp.down_proj: bf16`` keeps down_proj
            as it is. A config states its per-unit formats with ``formats`` or with the three keys it sets,
            never both.
        target: serving target (``sm90``, ``h200``, ``sm90@0.33.0``); kept in canonical ``arch@vllm`` form.
        devices: card indices, any number this process can see; null uses every visible card, or the first one
            when a step runs on one card only.
        hessian: ``{dtype, buffer_rows}`` or ``fast``; kept as the full dict.
        batch_size: calibration rows per forward.
        residency: a weight store (`qda_quant.engine.stores`: ``resident``, ``stream``, ``disk``, ...) to override
            the planner; null lets it choose.
        nccl_nvls: let the card processes join NCCL's NVLS (multicast) collectives; false by default (see
            `qda_quant.build`).
        dtype: the dtype the model is loaded in, ``bf16`` (default) or ``fp32``. ``bf16`` is the reproduction
            path: the published rotated bases are bf16 checkpoints, so a `rotate` step stores one bf16 rounding
            before `gptq` reads it, as they did. ``fp32`` keeps the rotation's full precision and lets `gptq`
            quantize from it with no bf16 rounding in between: a candidate for a later search round, whose
            checkpoint differs from the published one by design (`qda_quant.engine.build`).
        grid: the quantization grid (`qda_quant.solve.grid`) of every step that does not name its own; null names
            none, and each such step takes its own default grid, else `DEFAULT_GRID` (filled in by `plan`).
        steps: the recipe's steps, ``[{name: {settings}}, ...]``; kept with every setting filled in, except a
            ``grid`` equal to the key above.
        calib: the recipe's calibration data.
        meta: free-form notes carried with the recipe (left out of resolved.yaml when empty).
        out: output checkpoint directory (needed by ``build``, not by ``plan``).
        exporter: what the build writes (`qda_quant.output.exporters`), ``{name: settings}`` with every setting filled in;
            ``compressed_tensors`` by default.
        ignore: linears that stay bf16. Each entry is a role (`qda_quant.arch.KEEP_ROLES`: ``gdn_gates``, Gated
            DeltaNet ``in_proj_a`` / ``in_proj_b``; ``shared_experts``), an exact module name as ``qda plan``
            prints it, or ``re:`` and a regex matched from the start of the name (llm-compressor's rule). Both
            roles plus any adapter defaults when omitted; ``[]`` quantizes them too. A name or
            pattern that matches no linear the build would quantize is refused at plan time, except one that
            names the output head: it stays bf16 unless a head step stores it, and a config that asks for both
            is refused.
        keep_intermediate: keep a staged build's intermediate checkpoints in ``out/.stages``
            (`qda_quant.engine.packed`).
        drafts: external speculative drafts (DFlash, DFlash2, DSpark; a Hub id or a directory) to rotate into
            the build's residual basis once it is written (`qda_quant.output.rotate_draft`); each goes beside
            ``out`` as ``<out>-<draft name>``. Only for a build with a residual `rotate` step (a draft runs on
            any other build as it is). Left out of resolved.yaml when empty.
        layer_cache: ``{dir, match}``: reuse and keep solved layers (`qda_quant.engine.layer_cache`); null solves
            every layer. ``match: exact`` reuses a layer only when every layer up to it has the same formats;
            ``formats`` also reuses one whose own formats match, which makes the build a screening build only.
        provenance: written into resolved.yaml by a build; ignored when read.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    format: str | None
    format_overrides: list[dict[str, str]] = Field(default_factory=list)
    target: str = Field(default_factory=lambda: target_name(Target()))
    devices: list[int | str] | None = None
    hessian: dict[str, Any] = Field(default_factory=lambda: _hessian_dict(DEFAULT_HESSIAN))
    batch_size: int = Field(default=1, ge=1)
    residency: str | None = None
    nccl_nvls: bool = False
    dtype: Literal["bf16", "fp32"] = "bf16"
    grid: str | None = None
    steps: list[dict[str, dict[str, Any] | None]]
    calib: dict[str, Any]
    meta: dict[str, Any] = Field(default_factory=dict)
    out: str | None = None
    exporter: dict[str, dict[str, Any]] = Field(default_factory=lambda: parse_exporter(None).to_dict())
    keep_intermediate: bool = False
    drafts: list[str] = Field(default_factory=list)
    layer_cache: LayerCacheSetting | None = None
    ignore: list[str] = Field(default_factory=lambda: list(KEEP_ROLES))
    provenance: dict[str, Any] | None = None

    _recipe: Recipe = PrivateAttr()
    #: The file `from_file` read, when it did: a refusal names it and its `devices` line.
    _source_path: Path | None = PrivateAttr(default=None)
    #: `format` named no symmetry, so the format it canonicalized to records what `DEFAULT_SYMMETRIC` gave
    #: (`Format.symmetric_defaulted`): the planner then stores routed experts symmetric.
    _format_inherits_symmetry: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _note_format_spelling(cls, data: Any, handler: Any) -> BuildConfig:
        """Keep `Format.parse`'s inherited-symmetry mark through the field that canonicalizes the name.

        The canonical name always names a symmetry (``w4a16`` becomes ``w4a16-asym-g128``), so the spelling
        is read before validation: a config that wrote ``w4a16`` defaults its experts symmetric, one that
        wrote ``-asym`` pins them asymmetric, and a recorded ``resolved.yaml`` (canonical) pins them. A
        ``formats: {default: ...}`` spelling is the body format's spelling (`qda_quant.format_recipe`).
        """
        model = handler(data)
        if isinstance(data, BuildConfig):
            model._format_inherits_symmetry = data._format_inherits_symmetry
        else:
            raw = data.get("format") if isinstance(data, dict) else getattr(data, "format", None)
            if raw is None and isinstance(data, dict) and isinstance(data.get("formats"), dict):
                raw = data["formats"].get("default")
            model._format_inherits_symmetry = isinstance(raw, str) and Format.parse(raw).symmetric_defaulted
        return model

    @property
    def body_format(self) -> Format | None:
        """`format` as the config wrote it: a bare ``w4a16`` keeps its inherited-symmetry mark."""
        if self.format is None:
            return None
        return replace(Format.parse(self.format), symmetric_defaulted=self._format_inherits_symmetry)

    @model_validator(mode="before")
    @classmethod
    def _expand_type_map(cls, data: Any) -> Any:
        """A GGUF candidate's ``{unit: type}`` map becomes `tensor_types` plus ``format_overrides`` here.

        The expansion needs the model's solve units and the template's headers, which a search host derives
        once with ``qda sensitivity groups`` into the cache `exporter.gguf.groups` names
        (`qda_quant.gguf.groups`). Expanding before the fields validate means a bad type, a unit that names
        nothing, or a fused unit that disagrees is refused at config time, as a hand-written table would be,
        and the tables the build runs and ``resolved.yaml`` records are the expanded ones.

        Raises:
            ValueError: `type_map` is given without `groups` or with `tensor_types`, or it conflicts with a
                stated ``format_overrides`` pattern.
        """
        if not isinstance(data, dict):
            return data
        exporter = data.get("exporter")
        if not isinstance(exporter, dict) or not isinstance(exporter.get("gguf"), dict):
            return data
        settings = exporter["gguf"]
        if not settings.get("type_map"):
            return data
        where = settings.get("groups")
        if not where:
            raise ValueError(
                "exporter.gguf.type_map: needs exporter.gguf.groups, the derived-group cache for this model and "
                "template (derive it with `qda sensitivity groups CONFIG --template TEMPLATE --out FILE`)"
            )
        if settings.get("tensor_types"):
            raise ValueError(
                "exporter.gguf: type_map and tensor_types both name the stored types; a config states them one way"
            )
        from .gguf import groups as _groups

        tables = _groups.expand(_groups.load(where), settings["type_map"])
        rules = list(data.get("format_overrides") or ())
        # a rule is written {pattern, format} or [pattern, format] (`_format_overrides` accepts both); one it
        # cannot read is left to that validator to refuse with its own message
        stated = {}
        for rule in rules:
            if isinstance(rule, dict) and "pattern" in rule and "format" in rule:
                stated[rule["pattern"]] = rule["format"]
            elif isinstance(rule, (list, tuple)) and len(rule) == 2:
                stated[rule[0]] = rule[1]
        added = []
        for rule in tables["format_overrides"]:
            if rule["pattern"] in stated:
                if stated[rule["pattern"]] != rule["format"]:
                    raise ValueError(
                        f"format_overrides: {rule['pattern']!r} is {stated[rule['pattern']]} in the config "
                        f"and {rule['format']} from the type map; one format per unit"
                    )
                continue  # the recipe already states this unit's pair: one entry, not two
            added.append(rule)
        return {
            **data,
            "exporter": {**exporter, "gguf": {**settings, "tensor_types": tables["tensor_types"]}},
            "format_overrides": [*rules, *added],
        }

    @model_validator(mode="before")
    @classmethod
    def _packed_source_format(cls, data: Any) -> Any:
        """A packed checkpoint as `model` has its own format: ``format`` may be left out (then null)."""
        if isinstance(data, dict) and "format" not in data and isinstance(data.get("model"), str):
            from .engine.packed import is_packed

            if is_packed(data["model"]):
                return {**data, "format": None}
        return data

    @model_validator(mode="before")
    @classmethod
    def _formats_recipe(cls, data: Any) -> Any:
        """Compile the ``formats:`` mapping into ``format``, `format_overrides` and `ignore`.

        The mapping is sugar (`qda_quant.format_recipe`): it is compiled before the field validators run, so
        a recipe goes through exactly the checks an explicit override list does, and the record a build
        writes (``resolved.yaml``) is the compiled form. A config states its per-unit formats one way.
        """
        if not isinstance(data, dict) or "formats" not in data:
            return data
        from .format_recipe import compile_formats

        if data.get("format") is not None or "format_overrides" in data or "ignore" in data:
            raise ValueError(
                "formats: a recipe sets `format`, `format_overrides` and `ignore` itself; a config states its "
                "per-unit formats with `formats` or with those keys, never both"
            )
        if isinstance(data.get("model"), str):
            from .engine.packed import is_packed

            if is_packed(data["model"]):
                raise ValueError(
                    "formats: a packed checkpoint has one stored format; a formats recipe needs a float model"
                )
        body, rules, ignores = compile_formats(data["formats"])
        out = {k: v for k, v in data.items() if k != "formats"}
        if body is not None:
            out["format"] = body
        if rules:
            out["format_overrides"] = rules
        if ignores:  # added to the roles that always stay bf16, like the `ignore` default
            out["ignore"] = sorted(set(KEEP_ROLES) | set(ignores))
        return out

    @field_validator("ignore")
    @classmethod
    def _patterns(cls, v: list[str]) -> list[str]:
        """Each entry once, sorted (the order a build records); a ``re:`` entry must compile."""
        for p in v:
            if p.startswith("re:"):
                try:
                    re.compile(p[3:])
                except re.error as e:
                    raise ValueError(f"{p!r}: not a regex ({e})") from None
        return sorted(set(v))

    @field_validator("format_overrides", mode="before")
    @classmethod
    def _format_overrides(cls, v: Any) -> list[dict[str, str]]:
        """Each rule as ``{pattern, format}``, the format in canonical form; a ``re:`` pattern must compile."""
        out = []
        for item in v or ():
            if isinstance(item, dict) and set(item) == {"pattern", "format"}:
                pattern, name = item["pattern"], item["format"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pattern, name = item
            else:
                raise ValueError(f"{item!r}: a rule is {{pattern: ..., format: ...}}")
            if not isinstance(pattern, str) or not isinstance(name, str):
                raise ValueError(f"{item!r}: pattern and format must be strings")
            if pattern.startswith("re:"):
                try:
                    re.compile(pattern[3:])
                except re.error as e:
                    raise ValueError(f"{pattern!r}: not a regex ({e})") from None
            out.append({"pattern": pattern, "format": Format.parse(name).name})
        return out

    @property
    def format_rules(self) -> list[tuple[str, str]]:
        """`format_overrides` as the ``(pattern, format)`` pairs `qda_quant.engine.build.build` takes."""
        return [(r["pattern"], r["format"]) for r in self.format_overrides]

    @field_validator("exporter", mode="before")
    @classmethod
    def _exporter(cls, v: Any) -> dict[str, dict[str, Any]]:
        return parse_exporter(v).to_dict()

    @field_validator("residency")
    @classmethod
    def _residency(cls, v: str | None) -> str | None:
        from .engine.stores import WEIGHT_STORES

        if v is not None and v not in WEIGHT_STORES:
            raise ValueError(f"unknown weight store {v!r}; known: {sorted(WEIGHT_STORES)}")
        return v

    @field_validator("format")
    @classmethod
    def _format(cls, v: str | None) -> str | None:
        return None if v is None else Format.parse(v).name

    @field_validator("target", mode="before")
    @classmethod
    def _target(cls, v: Any) -> str:
        return target_name(Target.parse(v))

    @field_validator("hessian", mode="before")
    @classmethod
    def _hessian(cls, v: Any) -> dict[str, Any]:
        return _hessian_dict(hessian_setting(v))

    @model_validator(mode="after")
    def _resolve_recipe(self) -> BuildConfig:
        if self.format is not None and (clip := Format.parse(self.format).default_clip) is not None:
            self.steps = [_with_clip(s, clip) for s in self.steps]
        recipe = Recipe.from_dict(
            {
                "grid": self.grid,
                "steps": self.steps,
                "calib": self.calib,
                "meta": self.meta,
            }
        )
        full = recipe.to_dict(full=True)
        self.steps, self.calib = full["steps"], full["calib"]
        self._recipe = recipe
        residual = any("rotate" in s and (s["rotate"] or {}).get("residual", True) for s in self.steps)
        if self.drafts and not residual:
            raise ValueError("drafts: the build has no residual rotate step, so a draft runs on it as it is")
        return self

    @classmethod
    def from_file(cls, path: str | Path, overrides: Sequence[str] = ()) -> BuildConfig:
        """Read a YAML config file, then apply ``dotted.key=value`` overrides (see `from_dict`).

        A relative path in the file is relative to the file's directory.

        Raises:
            Refused: the file cannot be read or parsed, or does not validate.
        """
        try:
            data = yaml.safe_load(Path(path).read_text())
        except (OSError, yaml.YAMLError) as e:
            raise Refused(f"{path}: {e}; check the path and that the file is YAML") from None
        cfg = cls.from_dict(data, overrides, name=str(path), base=Path(path).resolve().parent)
        cfg._source_path = Path(path)  # a build refusal names this file and its `devices` line
        return cfg

    @classmethod
    def from_dict(cls, data: Any, overrides: Sequence[str] = (), name: str = "config",
                  base: str | Path | None = None) -> BuildConfig:  # fmt: skip
        """Validate plain data after applying ``dotted.key=value`` overrides (values parsed as YAML).

        `base` is the directory for relative paths (default: the working directory).

        Raises:
            Refused: the data is not a mapping, or a key is missing, unknown or invalid; the message
                starts with `name` and lists every bad key.
        """
        if not isinstance(data, dict):
            raise Refused(
                f"{name}: expected a mapping of keys, got {type(data).__name__}; write the config as "
                "`key: value` lines"
            )
        try:
            data = apply_overrides(data, list(overrides))
        except ValueError as e:
            raise Refused(f"{name}: {e}") from None
        base = Path.cwd() if base is None else Path(base)
        data = _groups_beside(data, base)
        try:
            return cls.model_validate(_local_model(resolve_paths(data, base), base))
        except ValidationError as e:
            lines = "\n".join(f"  {line}" for line in error_lines(e))
            raise Refused(f"{name}:\n{lines}", error_fix(e)) from None

    @property
    def recipe(self) -> Recipe:
        """The `Recipe` (steps, calib) of this config."""
        return self._recipe

    def build(self, **engine: Any) -> Path:
        """Run the build; `engine` takes the engine-only options of `qda_quant.engine.build` (``log``).

        An `Exporter` object as ``exporter``, or a `torch` dtype as ``dtype``, replaces the configured one
        (a test's writer, or an experiment's load precision); a ``dtype`` no config key names is recorded in
        ``resolved.yaml``'s ``provenance.not_recorded`` (`dtype_name`).

        Raises:
            Refused: ``out`` is not set.
        """
        from .engine.build import build
        from .output.rotate_draft import beside, rotate

        if self.out is None:
            raise Refused("config: missing required key 'out' (set it in the file or with --set out=DIR)")
        if taken := [str(p) for d in self.drafts if (p := beside(self.out, d)).exists()]:
            raise Refused(f"drafts: {taken[0]} exists")
        # before any weight is loaded: on a job that holds cards, a config that puts card steps on CPU would
        # spend hours there. `engine.build` enforces the same rule for a direct Python-API call; here the
        # refusal can name the file this config was read from and its `devices` line.
        from .checks import cpu_step_refusal

        if why := cpu_step_refusal(self.devices, self.recipe.steps, self._source_path):
            raise Refused(why)
        out = build(
            self.model,
            self.recipe,
            format=self.body_format,
            format_overrides=self.format_rules,
            # an output served by its own runtime decides the target, and refuses one a config names: pass
            # it only when this config actually named it, not when the field is sitting at its default
            target=self.target if "target" in self.model_fields_set else None,
            out_dir=self.out,
            devices=self.devices,
            batch_size=self.batch_size,
            residency=self.residency,
            nccl_nvls=self.nccl_nvls,
            hessian=self.hessian,
            keep_intermediate=self.keep_intermediate,
            layer_cache=self.layer_cache,
            ignore=self.ignore if "ignore" in self.model_fields_set else None,
            # `dtype` is a config key now; a Python-API call may still override it, as it may the exporter
            **{"exporter": self.exporter, "dtype": torch_dtype(self.dtype), **engine},
        )
        for draft in self.drafts:  # the checkpoint and its record are written: its basis can be read back
            rotate(out, draft, beside(out, draft))
        return out

    def plan(self) -> Plan:
        """Plan the build without loading weights; raises `Refused` with the reason it cannot run.

        Every stage is planned and checked, checkpoint steps included (`Plan.checkpoint_steps`), and so is
        ``out`` when it is set.
        """
        from .engine.build import _target_for
        from .engine.planning import (
            check_output,
            compile_plan,
            local_source,
            one_card_devices,
            resolve_devices,
        )

        if self.out is not None:
            check_output(self.out)
        devs = resolve_devices(self.devices)
        exporter = parse_exporter(self.exporter)
        source = local_source(self.model, weights=False)
        # the same rule `build` uses: an output served by its own runtime names the target, and an unset
        # one takes the table of the cards, so a plan never reports a target the build would not use
        tgt = _target_for(exporter, self.target if "target" in self.model_fields_set else None, devs)
        # and the same cards: a step that runs on one card only takes the first when `devices` is unset
        devs = one_card_devices(self.devices, devs, source, self.recipe, self.body_format, tgt, print)
        plan = compile_plan(
            source,
            self.recipe,
            self.body_format,
            tgt,
            devs,
            residency=self.residency,
            keep=frozenset(self.ignore) if "ignore" in self.model_fields_set else None,
            batch_size=self.batch_size,
            overrides=self.format_rules,
            dtype=torch_dtype(self.dtype),  # the dtype `build` loads in: its plan prices the same bytes
            exporter=exporter,  # an output that stores a unit in another type binds it before the solve
        )
        check_exporter(exporter, plan, self.recipe)
        record_exporter(exporter, plan, self.recipe)  # the exporter's own output size, when it knows it
        full = self.recipe.to_dict(full=True)  # planning names every grid: keep the public copy in step
        self.grid, self.steps = full["grid"], full["steps"]
        if "ignore" not in self.model_fields_set:
            self.ignore = sorted(plan.keep)
        return plan

    @classmethod
    def of_build(
        cls,
        model: str | Path,
        recipe: Recipe,
        *,
        fmt: Format,
        target: Target,
        devices: Sequence[int | str | torch.device] | None,
        hessian: HessianSetting,
        batch_size: int,
        residency: str | None,
        nccl_nvls: bool,
        out: str | Path,
        dtype: Any = None,
        keep_intermediate: bool = False,
        ignore: Sequence[str] = KEEP_ROLES,
        exporter: Exporter | None = None,
        format_overrides: Sequence[tuple[str, str]] = (),
        format_defaults: Mapping[str, str] | None = None,
        layer_cache: LayerCacheSetting | None = None,
    ) -> BuildConfig:
        """The config of a build started through the Python API, with every default filled in.

        Not validated again: `build` has checked every value, and the recipe may hold steps that are not
        registered plugins (a step class defined in a script). Such a config is a record, but reading it back
        needs those steps registered. Declared paths are recorded absolute under the working directory.

        `format_defaults` is the plan's resolved per-class scheme (`Plan.format_defaults`). When it is not
        empty, `format` is recorded as the bare spelling the build inherited its symmetry from, not its
        canonical name: reading the record back must give the experts the same symmetry this build stored.

        `dtype` is the `torch` dtype the build loaded the model in: ``bf16`` when None, the dtype a config
        names when it is one (`dtype_name`), and the bf16 default when it is not -- a dtype no config key
        names cannot be recorded, and ``not_recorded`` says so.
        """
        r = resolve_paths(recipe.to_dict(full=True), Path.cwd())
        source = str(model)
        cfg = cls.model_construct(
            model=str(expand_path(source).resolve())
            if (Path(source).is_absolute() or source.startswith((".", "~")))
            else source,
            format=fmt.spelling if format_defaults and fmt.symmetric_defaulted else fmt.name,
            target=target_name(target),
            devices=None if devices is None else [d if isinstance(d, int | str) else str(d) for d in devices],
            hessian=_hessian_dict(hessian),
            batch_size=batch_size,
            residency=residency,
            nccl_nvls=nccl_nvls,
            dtype=dtype_name(dtype) or "bf16",
            grid=r["grid"],
            steps=r["steps"],
            calib=r["calib"],
            meta=r.get("meta", {}),
            out=str(out),
            exporter=parse_exporter(exporter).to_dict(),
            keep_intermediate=keep_intermediate,
            layer_cache=layer_cache,
            ignore=list(ignore),
            format_overrides=[{"pattern": p, "format": Format.parse(f).name} for p, f in format_overrides],
            provenance=None,
        )
        cfg._recipe = recipe
        return cfg

    def resolved(
        self,
        source_snapshot: str | None = None,
        adapters: Sequence[str] = (),
        cards: int | None = None,
        not_recorded: Sequence[str] = (),
        store: str | None = None,
        format_defaults: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """This config as plain data, with the ``provenance`` of the run that used it.

        ``provenance`` is a record of things the config does not pin. ``cards`` is the card count the build
        used (it changes the bits; ``devices: null`` means every visible card). ``source_revision`` is the
        hub commit a repo id resolved to: a rebuild from a bare ``org/name`` downloads the repo again and
        gets whatever the hub holds then, so to repeat the build exactly, pin it as
        ``model: org/name@<source_revision>``. A local model is signed instead (``source_signature``,
        `qda_quant.intake.fingerprint.model_signature`), so a rebuild can tell its weights changed.
        ``not_recorded`` lists the Python-API build options that
        differed from their defaults and are not config keys (a `torch` ``dtype`` other than ``bf16`` or
        ``fp32``); a rebuild from this file uses their defaults. ``dtype`` itself is a config key, and a build
        that loaded at ``fp32`` repeats it.

        Args:
            source_snapshot: the local directory a hub repo id resolved to (None for a local model).
            adapters: names of the architecture adapters the model graph used (`qda_quant.arch.adapters.used`).
            cards: number of cards the build ran on.
            not_recorded: names of build options this file does not repeat.
            store: the weight store the build used (`residency` names it when set; else the planner chose it).
            format_defaults: the plan's resolved per-class scheme (`Plan.format_defaults`): what the body
                format's inherited symmetry gave the units it named no scheme for.
        """
        d = self.model_dump(exclude={"provenance"})
        if not d["meta"]:
            del d["meta"]
        if not d["drafts"]:  # a build without drafts writes the record it wrote before the key existed
            del d["drafts"]
        if d["layer_cache"] is None:  # keep an uncached build's resolved config unchanged
            del d["layer_cache"]
        # a local model has no hub snapshot to pin: the content of its directory stands in
        local = source_snapshot is None and Path(self.model).is_dir()
        d["provenance"] = provenance(
            self.plugins(adapters, store),
            source_snapshot,
            # the HF cache keeps a snapshot as .../snapshots/<commit>
            source_revision=Path(source_snapshot).name if source_snapshot else None,
            source_signature=model_signature(self.model) if local else None,
            cards=cards,
            not_recorded=not_recorded,
        )
        if format_defaults:
            d["provenance"]["format_defaults"] = dict(sorted(format_defaults.items()))
        return d

    def write_resolved(self, out_dir: str | Path, **prov: Any) -> Path:
        """Write `resolved` as ``OUT/resolved.yaml`` (atomically); `prov` goes to `resolved`."""
        d = self.resolved(**prov)
        return write_resolved(out_dir, without_provenance(d), d["provenance"])

    def plugins(
        self, adapters: Sequence[str] = (), store: str | None = None
    ) -> dict[str, dict[str, str | None]]:
        """``{name: {group, package, version}}`` of every plugin this build uses (:func:`qda_quant.output.record.plugins`).

        A step that is not a registered plugin is recorded with the package of its module and group null.
        The format is recorded under the key of the registered instance it matches, or under its canonical
        name when no registered format describes it (a spelling the built-in family parses). A recorded bare
        spelling parses to a marked copy of the registered instance (`Format.symmetric_defaulted`); its
        provenance still names that instance, which is the plugin that provided the format.
        """
        from .arch.adapters import ADAPTERS
        from .calib.sources import SOURCES, CorpusSource
        from .intake.corpus import CORPORA
        from .step import REGISTRY, STATISTICS, QuantStep, statistics_of
        from .steps.head import Head

        calib = self.recipe.calib
        arch = Target.parse(self.target).arch
        used: list[tuple[Registry[Any], str, Any]] = [(REGISTRY, s.name, type(s)) for s in self.recipe.steps]
        used += [(STATISTICS, n, st) for s in self.recipe.steps for n, st in statistics_of(s.needs).items()]
        used += [(SOURCES, calib.name, type(calib)), (TARGETS, arch, TARGETS[arch])]
        if any(isinstance(s, Head) for s in self.recipe.steps) and arch in BINDINGS:
            used.append((BINDINGS, arch, BINDINGS[arch]))  # the export resolves the head's kernel with it
        if isinstance(calib, CorpusSource):  # the nested corpus plugin, not only the source that reads it
            corpus = calib.parsed()
            used.append((CORPORA, corpus.name, type(corpus)))
        grids = {s.grid or DEFAULT_GRID for s in self.recipe.steps if isinstance(s, QuantStep)}
        used += [(GRIDS, name, GRIDS[name]) for name in sorted(grids)]
        if self.format is not None:
            fmt = Format.parse(self.format)
            # a recorded bare spelling parses to a marked copy, so keep the registered instance for provenance
            key = next((name for name, registered in FORMATS.items() if registered == fmt), None)
            used.append((FORMATS, key or fmt.name, FORMATS[key] if key is not None else fmt))
        # a clip-using step (rtn, gptq, ...) names one; no other kind of step has a string `clip` setting
        clips: set[str] = set()
        for s in self.recipe.steps:
            clip = getattr(s, "clip", None)
            if isinstance(s, QuantStep) and isinstance(clip, str):
                clips.add(clip)
        used += [(CLIPS, name, CLIPS[name]) for name in sorted(clips)]
        exporter = parse_exporter(self.exporter)
        used.append((EXPORTERS, exporter.name, type(exporter)))
        store = store or self.residency
        if store is not None:
            from .engine.stores import WEIGHT_STORES

            used.append((WEIGHT_STORES, store, WEIGHT_STORES[store]))
        for adapter in adapters:
            used.append((ADAPTERS, adapter, ADAPTERS[adapter]))
            dependencies = getattr(ADAPTERS[adapter], "plugins", None)
            if dependencies is not None:
                used += dependencies(Target.parse(self.target))
        out = plugins(used)
        out[exporter.name] = {**out[exporter.name], **exporter.provenance()}
        return out


def target_name(t: Target) -> str:
    """Canonical name of a target, which `Target.parse` reads back: ``sm90@0.33.0`` or ``sm90@0.33.0:marlin``."""
    return f"{t.arch}@{t.vllm}" + (f":{t.backend}" if t.backend else "")


#: The model dtypes a config names; `torch_dtype` resolves one and `dtype_name` names it back.
DTYPES = {"bf16": "bfloat16", "fp32": "float32"}

#: The record a tool that downloaded a model leaves in its directory (``{"repo": ..., "revision": ...}``).
#: A build that reads such a directory records the repo as `build.json`'s ``source_repo``, so a checkpoint
#: built from a download named ``.../model`` still serves under the original model's name.
SOURCE_RECORD = ".qda-source.json"


def torch_dtype(name: str) -> Any:
    """The `torch` dtype a config's ``dtype`` names (``bf16``, ``fp32``)."""
    import torch

    return getattr(torch, DTYPES[name])


def dtype_name(dtype: Any) -> str | None:
    """The config name of a `torch` dtype a build loaded at; None for one no config key names.

    A config records ``bf16`` or ``fp32``; any other dtype a Python-API call asks for is not a config key, so
    ``resolved.yaml`` cannot repeat it and names it in ``provenance.not_recorded`` instead.
    """
    return next((name for name, attr in DTYPES.items() if dtype == torch_dtype(name)), None)


def _hessian_dict(h: HessianSetting) -> dict[str, Any]:
    return {"dtype": h.dtype, "buffer_rows": h.buffer_rows}


def _with_clip(step: Any, clip: str) -> Any:
    """`step` running `clip` (`Format.default_clip`) when it takes a clip and names none or a range search.

    The recipe then records the clip that runs: a quickstart recipe with ``clip: hmse`` builds at ``w4a4`` as
    ``fp4hmse``. A named clip that picks no range (``minmax``, ``fp4mse``) or seeds a solve (``grid3``) is kept.
    Anything `Recipe.from_dict` would refuse is passed through for it to refuse.
    """
    from .step import REGISTRY, QuantStep

    if not (isinstance(step, dict) and len(step) == 1):
        return step
    ((name, settings),) = step.items()
    if name not in REGISTRY or not isinstance(settings, dict | None):
        return step
    cls = REGISTRY[name]
    if not (issubclass(cls, QuantStep) and "clip" in cls.model_fields):  # `guided`'s clip caps token weights
        return step
    named = (settings or {}).get("clip")
    if named is not None and not (
        named in CLIPS and CLIPS[named].picks_ranges and not CLIPS[named].needs_solve
    ):
        return step
    return {name: {**(settings or {}), "clip": clip}}


def _groups_beside(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """`exporter.gguf.groups` as an absolute path under `base`, the way the config file states it.

    The cache is written beside the config it serves, so a bare ``groups: derived-groups.json`` reads next to
    the file. An absolute path is kept as given; an exporter this config does not name is left alone.
    """
    exporter = data.get("exporter")
    if not isinstance(exporter, dict) or not isinstance(exporter.get("gguf"), dict):
        return data
    settings = exporter["gguf"]
    where = settings.get("groups")
    if not isinstance(where, str) or Path(where).is_absolute():
        return data
    return {
        **data,
        "exporter": {**exporter, "gguf": {**settings, "groups": str((base / expand_path(where)).resolve())}},
    }


def _local_model(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve existing directories and explicit relative or home paths under `base`; preserve Hub ids."""
    model = data.get("model")
    if isinstance(model, str):
        path = expand_path(model)
        if model.startswith((".", "~")) or (base / path).is_dir():
            return {**data, "model": str((base / path).resolve())}
    return data
