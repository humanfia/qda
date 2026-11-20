"""Read, transform and write packed checkpoint stages."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..format import NVFP4, W8A8_FP8, Format
from ..output.record import BUILD_RECORDS
from ..recipe import Recipe
from ..solve import nvfp4
from ..solve.grid import DEFAULT_GRID, GRIDS
from ..solve.packing import unpack
from ..solve.quant import Quantizer
from ..step import Refused, Step
from ..target import LayerKind, Target, merge_serve

STAGES = ".stages"


@dataclass
class CheckpointPlan:
    """What `compile_plan` knows when it checks checkpoint steps (`Step.check_checkpoint`): metadata only."""

    source: str
    packed: Packed | None
    teacher: str | None
    config: Any
    graph: Any
    meta: torch.nn.Module
    fmt: Format
    target: Target
    plan_moe: Any
    devices: list[torch.device]
    recipe: Recipe
    output_bytes: int = 0

    def check_banks(self) -> None:
        """Refuse routed experts the student cannot read: biased banks, or a packed source storing them fused."""
        from ..arch.adapters import fused_experts
        from ..intake.source_names import SourceNames

        banks = [b.experts for lg in self.graph.layers for b in lg.banks if b.fused]  # fused: the module name
        for name in banks:
            mod = self.meta.get_submodule(name)
            if any(
                getattr(mod, f"{p}_bias", None) is not None
                for p in (fused_experts.in_param(mod), fused_experts.DOWN)
            ):
                raise Refused(f"{name}: expert banks with biases are not supported")
        if self.packed is not None:
            names = SourceNames(self.packed.path)
            for name in banks:
                if names.tensor(f"{name}.0.down_proj.weight_packed") is None:
                    raise Refused(f"{self.packed.path}: {name} is not stored as quantized per-expert modules")


@dataclass
class CheckpointRun:
    """The build context handed to checkpoint steps (``apply_checkpoint(src, out, run)``)."""

    source: str
    teacher: str | None
    recipe: Recipe
    devices: list[torch.device]
    log: Callable[[str], None] = print
    info: dict = field(default_factory=dict)


def check_stage_order(steps: Sequence[Step], packed: bool) -> None:
    """Refuse layer/model steps after a checkpoint step, and any on a packed source (they quantize a float model)."""
    first = next((i for i, s in enumerate(steps) if s.kind == "checkpoint"), len(steps))
    late = [s.name for s in steps[first:] if s.kind != "checkpoint"]
    if late:
        raise Refused(
            f"steps {late} follow a checkpoint step; layer and model steps run first, on the float model"
        )
    if packed and first:
        raise Refused(
            f"steps {[s.name for s in steps[:first]]} quantize a float model; the source is already packed"
        )


def _staged_log(log: Callable[[str], None], staging: Path) -> Callable[[str], None]:

    def emit(message: str) -> None:
        log(message.replace(f"{staging}/", "").replace(str(staging), "."))

    return emit


def _without_completion(log: Callable[[str], None], out: Path) -> Callable[[str], None]:
    """`log` without the float stage's completion line and its data lint block: the build's own plan logged that."""
    line = f"built {Path(out).resolve()} in "

    def emit(message: str) -> None:
        if not message.startswith((line, "data lint")):
            log(message)

    return emit


def staged(
    given: str,
    plan: Any,
    recipe: Recipe,
    *,
    target: Target | str | None,
    out_dir: str | Path,
    staging: Path,
    keep_intermediate: bool = False,
    log: Callable[[str], None] = print,
    not_recorded: Sequence[str] = (),
    generated: Sequence[dict] = (),
    **build_kw: Any,
) -> None:
    """Run a planned recipe stage by stage, leaving the completed checkpoint in `staging`."""
    from ..arch.adapters import used
    from ..config import BuildConfig
    from .build import build
    from .execute import attention_backends, calibration_attention
    from .stats import hessian_setting

    src = Path(plan.source)
    packed = is_packed(src)
    teacher = Packed(src).teacher() if packed else str(src)
    log = _staged_log(log, staging)  # a stage names the output it becomes, never the staging directory
    run = CheckpointRun(str(src), teacher, recipe, plan.devices, log)
    float_steps = [s for s in recipe.steps if s.kind != "checkpoint"]
    tmp = staging / STAGES
    done: list[dict] = list(generated)
    if float_steps:
        dst = tmp / f"0-{float_steps[0].name}"
        sub = Recipe(steps=float_steps, calib=recipe.calib, meta=recipe.meta, grid=recipe.grid)
        log(f"stage 0: {[s.name for s in float_steps]} on {src}")
        build(
            str(src),
            sub,
            format=plan.fmt,
            target=target,
            out_dir=dst,
            log=_without_completion(log, dst),
            **build_kw,
        )
        done += json.loads((dst / "build.json").read_text())["stages"]
        src = dst
    first = 1 if float_steps else 0
    for j, step in enumerate(plan.checkpoint_steps):
        i = j + first
        last = step is plan.checkpoint_steps[-1]
        dst = staging if last else tmp / f"{i}-{step.name}"
        dst.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        log(f"stage {i}: {step.name} on {src}")
        # the cards the plan gave this step: the build's own, or a capped step's subset of them
        run.devices = plan.stage_devices.get(j, plan.devices)
        # the plan's attention backends, as the float stage: not what the caller or vLLM set
        with calibration_attention(plan):
            step.apply_checkpoint(src, dst, run)
        done.append(
            {
                "kind": "checkpoint",
                "steps": [step.name],
                "seconds": round(time.perf_counter() - t0, 3),
                "devices": [str(d) for d in run.devices],
            }
        )
        src = dst
    rec = json.loads((staging / "build.json").read_text())
    from ..calib.template import lint_receipt

    rec = {**rec, "stages": done, "attention_backends": list(attention_backends(plan))}
    rec["data_lint"] = lint_receipt(plan.data_lint)  # this build's data, not the source checkpoint's receipt
    (staging / "build.json").write_text(json.dumps(rec, indent=1, default=str))
    BuildConfig.of_build(
        given,
        recipe,
        fmt=plan.fmt,
        target=plan.target,
        devices=build_kw.get("devices"),
        hessian=hessian_setting(build_kw.get("hessian")),
        batch_size=build_kw.get("batch_size", 1),
        residency=build_kw.get("residency"),
        nccl_nvls=build_kw.get("nccl_nvls", False),
        out=out_dir,
        dtype=build_kw.get("dtype"),
        keep_intermediate=keep_intermediate,
        ignore=sorted(plan.keep),
        exporter=build_kw.get("exporter"),
        format_overrides=build_kw.get("format_overrides", ()),
        format_defaults=plan.format_defaults,
    ).write_resolved(
        staging,
        source_snapshot=None if plan.source == given else plan.source,
        adapters=used(plan.graph),
        store=plan.residency.weights,
        cards=len(plan.devices),
        not_recorded=not_recorded,
        format_defaults=plan.format_defaults,
    )
    if not keep_intermediate:
        shutil.rmtree(tmp, ignore_errors=True)  # inside staging; absent when no stage made an intermediate


if TYPE_CHECKING:
    from .served_modules import ServedExperts


__all__ = [
    "STAGES",
    "CheckpointPlan",
    "CheckpointRun",
    "Packed",
    "check_stage_order",
    "float_names",
    "is_packed",
    "load_float",
    "load_student",
    "served_modules",
    "staged",
    "write",
]


STATE = Path("qda_state") / "scales.safetensors"

#: NVFP4 stores its group scales as E4M3: the directory's largest magnitude
_FP8_MAX = 448.0


_MEASURED = ("timings", "rank_timings", "residency", "load_to_saved_s")


def is_packed(path: str | Path) -> bool:
    """True if `path` is a local compressed-tensors checkpoint this package reads back.

    That is ``pack-quantized`` (integer codes), ``nvfp4-pack-quantized`` (E2M1 codes, FP8 group scales under a
    per-module global scale), or ``mixed-precision`` storing those two per group (``Packed`` still refuses a mixed
    group whose layout it cannot decode, naming the group).
    """
    cfg = Path(path) / "config.json"
    if not cfg.is_file():
        return False
    q = json.loads(cfg.read_text()).get("quantization_config") or {}
    return q.get("format") in ("pack-quantized", "nvfp4-pack-quantized", "mixed-precision")


def _targets_label(targets: Sequence[str]) -> str:
    """A config group's targets abbreviated: its first two names and how many more."""
    head = ", ".join(targets[:2])
    return f"{head} and {len(targets) - 2} more" if len(targets) > 2 else head or "no targets"


def _format_of(weights: dict, acts: dict | None) -> Format:
    if weights.get("type", "int") == "float":
        return _float_format(weights, acts)
    if weights.get("type", "int") != "int":
        raise Refused(f"weights of type {weights.get('type')!r}: only integer codes are read")
    group = -1 if weights.get("strategy") == "channel" else int(weights["group_size"])
    act = "bf16"
    if acts is not None:
        if (acts.get("type"), acts.get("num_bits"), acts.get("strategy")) != ("float", 8, "token"):
            raise Refused(f"input activations {acts}: only dynamic per-token FP8 (W4A8) is read")
        act = "fp8_token"
    return Format(int(weights["num_bits"]), group, bool(weights["symmetric"]), "int", act)


def _float_format(weights: dict, acts: dict | None) -> Format:
    """The format of a float-coded config group: NVFP4, or the FP8 W8A8 group a mixed build stores.

    Raises:
        Refused: the group is another float format (MXFP4, per-tensor FP8, ...), or one of these without the
            input cut it is served with.
    """
    if (
        weights.get("num_bits"),
        weights.get("group_size"),
        bool(weights.get("symmetric")),
        weights.get("strategy"),
    ) == (8, None, True, "channel"):
        if acts is not None and (
            acts.get("type"),
            acts.get("num_bits"),
            acts.get("strategy"),
            acts.get("dynamic"),
        ) != ("float", 8, "token", True):
            raise Refused(
                f"input activations {acts}: FP8 W8A8 is read with its dynamic per-token E4M3 cut; a "
                "weights-only FP8 checkpoint has no kernel row"
            )
        return W8A8_FP8
    if (
        weights.get("num_bits"),
        weights.get("group_size"),
        bool(weights.get("symmetric")),
    ) != (4, 16, True):
        raise Refused(
            f"float weights {weights}: only NVFP4 (4-bit E2M1, groups of 16, symmetric) and FP8 W8A8 (8-bit "
            "E4M3, one scale an output channel, symmetric) are read back"
        )
    if acts is not None and (
        acts.get("type"),
        acts.get("num_bits"),
        acts.get("group_size"),
        acts.get("dynamic"),
    ) != ("float", 4, 16, "local"):
        raise Refused(
            f"input activations {acts}: NVFP4 is read with its dynamic E4M3 cut per 16 values; "
            "a weights-only FP4 checkpoint has no input_global_scale to serve it"
        )
    return NVFP4


class Packed:
    """Read-only view of a packed checkpoint directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not is_packed(self.path):
            raise Refused(f"{self.path}: not a compressed-tensors pack-quantized checkpoint")
        self.config = json.loads((self.path / "config.json").read_text())
        rec = self.path / "build.json"
        self.record: dict = json.loads(rec.read_text()) if rec.is_file() else {}
        idx = self.path / "model.safetensors.index.json"
        if idx.is_file():
            self.where = json.loads(idx.read_text())["weight_map"]
        else:
            self.where = {}
            for f in sorted(self.path.glob("*.safetensors")):
                with safe_open(f, "pt") as h:
                    self.where.update(dict.fromkeys(h.keys(), f.name))
        if any(k.endswith(".weight_g_idx") for k in self.where):
            raise Refused(f"{self.path}: activation-ordered groups (weight_g_idx) are not read in v0")
        self._groups = []
        for name, g in self.config["quantization_config"]["config_groups"].items():
            try:
                f = _format_of(g["weights"], g.get("input_activations"))
            except Refused as e:
                # a checkpoint step reads every quantized module, so one group it cannot decode refuses the
                # build: name the group and the modules it stores, not just the config dict
                raise Refused(f"{self.path}: {name} ({_targets_label(g['targets'])}): {e}") from None
            self._groups.append((g["targets"], f))
        body = [f for t, f in self._groups if "Linear" in t]
        if not body:
            raise Refused(f"{self.path}: no config group targets Linear")
        self.fmt = body[0]
        packed = {k[: -len(".weight_packed")] for k in self.where if k.endswith(".weight_packed")}
        # an FP8 W8A8 module stores `<stem>.weight` (E4M3) beside `<stem>.weight_scale` (fp32); a float Linear
        # stores no scale, so the pair is what tells the two apart
        fp8 = {
            k[: -len(".weight")]
            for k in self.where
            if k.endswith(".weight") and f"{k[: -len('.weight')]}.weight_scale" in self.where
        }
        self.stems = sorted(packed | {s for s in fp8 if self.format_of(s).weight_type == "fp8"})
        self._stems = frozenset(self.stems)
        tgt = self.record.get("target")
        if tgt is None and self.fmt.weight_type in ("e2m1", "fp8"):
            # no record (a checkpoint another tool wrote): sm100 is the only table that serves NVFP4 and FP8 W8A8
            tgt = "sm100"
        self.target = Target.stored(tgt)
        self._open: dict[str, Any] = {}

    def tensor(self, name: str) -> torch.Tensor:
        """One stored tensor (CPU)."""
        f = self.where[name]
        if f not in self._open:
            self._open[f] = safe_open(self.path / f, "pt")
        return self._open[f].get_tensor(name)

    def format_of(self, stem: str) -> Format:
        """Format of quantized module `stem`: an exact or regex name-targeted group, else the body format."""
        import re

        for targets, f in self._groups:
            if any(t == stem or (t.startswith("re:") and re.match(t[3:], stem)) for t in targets):
                return f
        return self.fmt

    def stem_of(self, key: str) -> str | None:
        """The quantized stem stored tensor `key` belongs to, or None when it is a plain float tensor.

        A packed module's codes are ``<stem>.weight_packed`` (`.weight_packed` also stands in `key`'s place for
        the logical ``<stem>.weight``); an FP8 module's are ``<stem>.weight`` itself.
        """
        for suffix in (".weight_packed", ".weight"):
            if key.endswith(suffix) and key[: -len(suffix)] in self._stems:
                return key[: -len(suffix)]
        return None

    def quantizer(self, stem: str, kind: LayerKind = "dense") -> Quantizer:
        """The module's quantizer on the checkpoint's target, with the plan of a `kind` layer (routed experts: moe).

        The grid is the one that stores the format (`nvfp4` for NVFP4: E2M1 codes are positions on its ladder),
        else `DEFAULT_GRID`, the validated integer rule.
        """
        f = self.format_of(stem)
        grid = next((n for n in sorted(GRIDS) if GRIDS[n].stores == f.name), DEFAULT_GRID)
        return Quantizer.of(f, self.target.plan(f, kind), grid)

    @property
    def nvfp4(self) -> bool:
        """True when the checkpoint's body format is NVFP4 (E2M1 codes under E4M3 group scales)."""
        return self.fmt.weight_type == "e2m1"

    def codes(self, stem: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(codes ``[out, in]``, zero points ``[out, groups]`` or None), signed as stored.

        NVFP4 codes are signed ladder positions (-7 .. 7, code 0 the value 0) and an FP8 weight's code is the
        E4M3 value itself; every other format's are its int8 codes. FP8 and NVFP4 store no ``weight_shape`` and
        no zero points: the stored weight gives the shape, and its scales cover all columns (FP8) or 16 each
        (NVFP4).
        """
        f = self.format_of(stem)
        if f.weight_type == "fp8":
            return self.tensor(f"{stem}.weight"), None
        packed = self.tensor(f"{stem}.weight_packed")
        if f.weight_type == "e2m1":
            q = nvfp4.positions_of_codes(nvfp4.unpack_nibbles(packed))
            scale = self.tensor(f"{stem}.weight_scale")
            want = (scale.shape[0], scale.shape[1] * nvfp4.GROUP)
            if tuple(q.shape) != want:
                raise Refused(
                    f"{self.path}: {stem}: {q.shape[0]}x{q.shape[1]} packed codes, but its scales describe "
                    f"{want[0]}x{want[1]}"
                )
            return q, None
        shape = torch.Size(self.tensor(f"{stem}.weight_shape").tolist())
        q = unpack(packed, f.weight_bits, shape, packed_dim=1)
        z = None
        if f"{stem}.weight_zero_point" in self.where:
            groups = self.tensor(f"{stem}.weight_scale").shape
            z = unpack(self.tensor(f"{stem}.weight_zero_point"), f.weight_bits, groups, packed_dim=0)
            z = z.to(torch.int8)
        return q.to(torch.int8), z

    def _global(self, stem: str, name: str) -> torch.Tensor:
        """One fp32 global scale ``[1]`` of an NVFP4 module, refused when absent or not positive."""
        key = f"{stem}.{name}"
        if key not in self.where:
            raise Refused(f"{self.path}: {stem}: no {name}; an NVFP4 weight cannot be decoded without it")
        g = self.tensor(key).float()
        if not bool((g > 0).all()):
            raise Refused(f"{self.path}: {stem}: {name} {g.tolist()} is not positive")
        return g

    def global_scale(self, stem: str) -> torch.Tensor:
        """The fp32 ``weight_global_scale`` of an NVFP4 module `stem`: ``weight_scale / global_scale`` is its step."""
        return self._global(stem, "weight_global_scale")

    def input_global_scale(self, stem: str) -> torch.Tensor:
        """The fp32 static ``input_global_scale`` of module `stem`: the kernel cuts its input under it."""
        return self._global(stem, "input_global_scale")

    def stored_scale(self, stem: str, s0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """The ``weight_scale`` tensor `stem` stores for state (s0, delta), in the checkpoint's dtype.

        An FP8 module stores its fp32 scale per output channel as the kernel reads it, ``s0 * exp(delta)``;
        NVFP4 stores an E4M3 value per 16 columns, ``e4m3(s0 * exp(delta))``, and the kernel divides it by the
        module's ``weight_global_scale``; every other format stores the bf16 scale itself.
        """
        f = self.format_of(stem)
        if f.weight_type == "fp8":
            return s0.float() * delta.float().exp()
        if f.weight_type != "e2m1":
            return stored_scale(s0, delta)
        x = s0.float() * delta.float().exp()
        return torch.clamp(x, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)

    def step(self, stem: str, s0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """The fp32 step `stem` serves for state (s0, delta): ``stored_scale / weight_global_scale`` for NVFP4."""
        f = self.format_of(stem)
        if f.weight_type == "fp8":
            return s0.float() * delta.float().exp()
        if f.weight_type != "e2m1":
            return stored_scale(s0, delta)
        return self.stored_scale(stem, s0, delta).float() / self.global_scale(stem)

    def state(self, log: Callable[[str], None] = print) -> tuple[dict, dict]:
        """(scales, floats): ``{stem: (s0, delta)}`` and ``{tensor: fp32 master}`` to start from."""
        side = self.path / STATE
        scales: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        floats: dict[str, torch.Tensor] = {}
        if side.is_file():
            with safe_open(side, "pt") as h:
                keys = set(h.keys())
                for s in self.stems:
                    scales[s] = (h.get_tensor(f"s0.{s}"), h.get_tensor(f"delta.{s}"))
                for k in keys:
                    if k.startswith("float."):
                        floats[k[len("float.") :]] = h.get_tensor(k)
            for s in self.stems:  # the state must describe exactly the stored scales
                if not torch.equal(self.stored_scale(s, *scales[s]), self.tensor(f"{s}.weight_scale")):
                    raise Refused(f"{side}: state of {s} does not give its stored weight_scale")
        else:
            log(f"{self.path}: no {STATE}; starting from the stored scales (delta = 0)")
            for s in self.stems:
                # an FP8 scale is held in the fp32 the checkpoint stores; an NVFP4 E4M3 one is exact in bf16,
                # and its step divides by the module's stored global scale
                w = self.tensor(f"{s}.weight_scale")
                s0 = w.float() if self.format_of(s).weight_type == "fp8" else w.to(torch.bfloat16)
                scales[s] = (s0, torch.zeros(s0.shape, dtype=torch.float32))
        return scales, floats

    def teacher(self) -> str | None:
        """The original float model this checkpoint was made from, as its record names it."""
        return self.record.get("teacher") or self.record.get("source")


def stored_scale(s0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The bf16 scale a module stores: ``bf16(bf16(s0) * exp(delta))``.

    The one owner of the rule, behind `Packed.stored_scale`, which the writer (for the ``weight_scale`` it
    writes and for its own check), `Packed.state` and the loads that dequantize from a state all go through.
    The ``s0`` cast is inside the function so the value is the same whether `s0` reaches it at the state's
    storage precision or at a wider one: a writer given an fp32 s0 computes the scale the next load computes
    from the bf16 s0 it saved, instead of one that load refuses.
    """
    return (s0.to(torch.bfloat16).float() * delta.float().exp()).to(torch.bfloat16)


def _module_stem(src: Packed, names: Any, name: str) -> str | None:
    """In-memory module `name`'s quantized stem in `src`, or None when the checkpoint keeps it a float Linear.

    The checkpoint names its packed weights after the stem; the model may name the module otherwise
    (`names`), and an FP8 module's code tensor is ``<stem>.weight`` rather than ``<stem>.weight_packed``.
    """
    for key in (names.tensor(name + ".weight_packed"), names.tensor(name + ".weight")):
        stem = None if key is None else src.stem_of(key)
        if stem is not None:
            return stem
    return None


def load_student(
    src: Packed,
    scales: dict[str, tuple[torch.Tensor, torch.Tensor]],
    floats: dict[str, torch.Tensor],
    device: torch.device,
    attn: str | None = None,
    grouped_experts: bool = False,
) -> torch.nn.Module:
    """The checkpoint as a trainable model, on `device`."""
    from accelerate import init_empty_weights

    from ..arch.adapters import fused_experts
    from ..intake.loading import load_model
    from ..intake.source_names import SourceNames
    from .served_modules import ServedLinear

    # parameters come out on `meta`, every one of them overwritten below; buffers are computed on the CPU, where
    # a constructor gives the values `from_pretrained` gives (a card would give other last bits)
    with init_empty_weights(include_buffers=False):
        m = load_model(src.path, weights=False, dtype=torch.bfloat16, attn_implementation=attn)
    # outside the context, `setattr` keeps one parameter object: tie before materializing, or the output head
    # would be allocated here and dropped again when it is tied after, a temporary as large as the head
    cast(Any, m).tie_weights()
    names = SourceNames(src.path)
    stems = set(src.stems)
    float_banks: set[str] = set()
    for name, mod in list(m.named_modules()):
        if fused_experts.is_bank(mod):
            if names.tensor(f"{name}.0.down_proj.weight_packed") is None:
                # a bank `ignore` kept whole: the class's module stays, and only its two parameters are restored
                float_banks.add(name)
                continue
            served = _served_experts(src, names, name, mod, scales, device, grouped_experts)
            stems -= {s for b in (served.gate_up, served.down) for p in b.parts for s, _ in p}
        else:
            stem = _module_stem(src, names, name) if isinstance(mod, torch.nn.Linear) else None
            if stem is None:
                continue
            stems.discard(stem)
            q, z = src.codes(stem)
            s0, delta = scales[stem]
            bias = None
            if mod.bias is not None:
                bias_name = names.tensor(name + ".bias")
                assert bias_name is not None  # a model bias comes from the checkpoint
                bias = src.tensor(bias_name).to(device)
            nv: dict[str, torch.Tensor] = {}
            if src.format_of(stem).weight_type == "e2m1":
                # the kernel decodes with weight_global_scale and cuts its input under input_global_scale
                nv = {
                    "gscale": src.global_scale(stem).to(device),
                    "igscale": src.input_global_scale(stem).to(device),
                }
            served = ServedLinear(stem, q.to(device), None if z is None else z.to(device), s0.to(device),
                                  delta.to(device), src.quantizer(stem), bias, **nv)  # fmt: skip
        parent, _, leaf = name.rpartition(".")
        setattr(m.get_submodule(parent) if parent else m, leaf, served)
    if stems:
        raise Refused(f"{src.path}: quantized tensors of no Linear in the model: {sorted(stems)[:3]}")
    _materialize(m, device)
    _cpu_buffers(m)
    _load_buffers(m, src, names)
    held = {f"{b}.{p}" for b in float_banks for p in (fused_experts.IN, fused_experts.DOWN)}
    for name, p in m.named_parameters():
        if name.endswith(".delta"):
            continue
        if name in held:
            _copy_float_bank(src, names, name, p, floats)
            continue
        key = names.tensor(name)
        if key is None:
            raise Refused(f"{src.path}: no tensor for {name}")
        t = floats[key] if key in floats else src.tensor(key)
        with torch.no_grad():
            p.copy_(t.to(p.dtype))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def load_float(src: Packed, device: torch.device, attn: str = "sdpa") -> torch.nn.Module:
    """The checkpoint as a plain bf16 model of its transformers class, every packed weight dequantized."""
    from ..arch.adapters import fused_experts
    from ..intake.loading import load_model
    from ..intake.source_names import SourceNames

    scales, floats = src.state(lambda _: None)
    with torch.device(device):  # random init: every parameter is overwritten below
        m = load_model(src.path, weights=False, dtype=torch.bfloat16, attn_implementation=attn)
    _cpu_buffers(m)
    names = SourceNames(src.path)
    _load_buffers(m, src, names)
    stems = set(src.stems)
    lin_modules: dict[str, str] = {}
    bank_modules = set()
    for name, mod in m.named_modules():
        if fused_experts.is_bank(mod):
            root, n_exp = _bank_root(src, names, name, mod)
            bank_modules.add(name)
            stems -= {f"{root}.{e}.{p}" for e in range(n_exp) for p in _BANK_PARTS}
        elif isinstance(mod, torch.nn.Linear):
            stem = _module_stem(src, names, name)
            if stem is not None:
                lin_modules[name] = stem
                stems.discard(stem)
    if stems:
        raise Refused(f"{src.path}: quantized tensors of no Linear in the model: {sorted(stems)[:3]}")
    packed_params = {f"{name}.{p}" for name in bank_modules for p in (fused_experts.IN, fused_experts.DOWN)}
    for name, p in m.named_parameters():
        parent, _, leaf = name.rpartition(".")
        if (parent in lin_modules and leaf == "weight") or name in packed_params:
            continue  # a packed weight, dequantized below
        key = names.tensor(name)
        if key is None:
            raise Refused(f"{src.path}: no tensor for {name}")
        with torch.no_grad():
            p.copy_((floats[key] if key in floats else src.tensor(key)).to(p.dtype))
    for name, stem in sorted(lin_modules.items()):
        mod = m.get_submodule(name)
        q, z = src.codes(stem)
        s0, delta = scales[stem]
        with torch.no_grad():
            mod.weight.copy_(src.quantizer(stem).dequant(q, src.step(stem, s0, delta), z))
    for name in sorted(bank_modules):
        _dequant_bank(src, names, name, m.get_submodule(name), scales)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


_BANK_PARTS = ("gate_proj", "up_proj", "down_proj")


def _bank_root(src: Packed, names: Any, name: str, mod: torch.nn.Module) -> tuple[str, int]:
    from ..arch.adapters import fused_experts

    if any(
        getattr(mod, f"{p}_bias", None) is not None for p in (fused_experts.in_param(mod), fused_experts.DOWN)
    ):
        raise Refused(f"{name}: expert banks with biases are not supported")
    if not fused_experts.has_gate(mod):
        raise Refused(
            f"{name}: a non-gated expert bank (per-expert up_proj and down_proj, no gate) is not read back "
            "from a packed checkpoint yet; serve it with a stock engine or rebuild from the float model"
        )
    first = names.tensor(f"{name}.0.down_proj.weight_packed")
    if first is None:
        raise Refused(
            f"{name}: the checkpoint stores no quantized per-expert down_proj (fused or float experts)"
        )
    root = first[: -len(".0.down_proj.weight_packed")]
    return root, fused_experts.in_weight(mod).shape[0]


def _bank_globals(src: Packed, name: str, stems: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight_global_scale, input_global_scale)`` of one expert's bank projection (NVFP4).

    The weight scale is one per expert: vLLM fuses an expert's gate and up and takes the gate's scale for both.
    The input scale is one per projection -- gate and up read the same input under the same scale.
    """
    ws = [src.global_scale(s) for s in stems]
    if any(not torch.equal(w, ws[0]) for w in ws):
        raise Refused(
            f"{src.path}: {name}: {stems} store different weight_global_scales "
            f"({[w.tolist() for w in ws]}); one bank serves an expert's gate and up under one"
        )
    igs = [src.input_global_scale(s) for s in stems]
    if any(not torch.equal(i, igs[0]) for i in igs):
        raise Refused(
            f"{src.path}: {name}: {stems} store different input_global_scales ({[i.tolist() for i in igs]})"
        )
    return ws[0], igs[0]


def _copy_float_bank(
    src: Packed,
    names: Any,
    name: str,
    p: torch.Tensor,
    floats: dict[str, torch.Tensor],
) -> None:
    key = names.tensor(name)
    if key is not None:
        t = floats[key] if key in floats else src.tensor(key)
        with torch.no_grad():
            p.copy_(t.to(p.dtype))
        return
    try:
        pieces = names.stored(name, p.shape)
    except KeyError as e:
        raise Refused(
            f"{src.path}: {name}: the checkpoint keeps neither the whole bank float nor every expert of it ({e})"
        ) from None
    whole = torch.empty(p.shape, dtype=p.dtype)
    filled = torch.zeros(p.shape, dtype=torch.bool)
    for key, sl in pieces:
        t = (floats[key] if key in floats else src.tensor(key)).to(p.dtype)
        if sl == ():
            whole.copy_(t)
            filled.fill_(True)
        else:
            expert, rows = sl
            whole[expert, rows] = t
            filled[expert, rows] = True
    if not bool(filled.all()):
        raise Refused(
            f"{src.path}: {name}: the checkpoint keeps neither the whole bank float nor every expert of it"
        )
    with torch.no_grad():
        p.copy_(whole)


def _dequant_bank(
    src: Packed,
    names: Any,
    name: str,
    mod: torch.nn.Module,
    scales: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    from ..arch.adapters import fused_experts

    root, n_exp = _bank_root(src, names, name, mod)
    with torch.no_grad():
        for e in range(n_exp):
            stems = [f"{root}.{e}.{p}" for p in _BANK_PARTS]
            if any(s not in scales for s in stems):
                raise Refused(f"{src.path}: no quantized {stems[0]} ... for expert {e} of {name}")
            w = []
            for stem in stems:
                q, z = src.codes(stem)
                s0, delta = scales[stem]
                w.append(src.quantizer(stem, "moe").dequant(q, src.step(stem, s0, delta), z))
            getattr(mod, fused_experts.IN)[e].copy_(torch.cat(w[:2]))
            getattr(mod, fused_experts.DOWN)[e].copy_(w[2])


def _served_experts(
    src: Packed,
    names: Any,
    name: str,
    mod: torch.nn.Module,
    scales: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    grouped: bool,
) -> ServedExperts:
    from ..arch.adapters import fused_experts
    from .served_modules import ServedBank, ServedExperts

    root, n_exp = _bank_root(src, names, name, mod)
    if src.format_of(f"{root}.0.down_proj").weight_type == "fp8":
        raise Refused(
            f"{src.path}: {name}: FP8 expert banks are not read back: no serving row plans FP8 experts on any "
            "card, so the stage could not be served or checked"
        )

    def bank(projs: tuple[str, ...]) -> ServedBank:
        parts, codes, zeros, s0s, deltas, gs = [], [], [], [], [], []
        nvfp4 = src.format_of(f"{root}.0.{projs[0]}").weight_type == "e2m1"
        for e in range(n_exp):
            stems = [f"{root}.{e}.{p}" for p in projs]
            if any(s not in scales for s in stems):
                raise Refused(f"{src.path}: no quantized {stems[0]} ... for expert {e} of {name}")
            cz = [src.codes(s) for s in stems]
            parts.append(tuple((s, c.shape[0]) for s, (c, _) in zip(stems, cz, strict=True)))
            codes.append(torch.cat([c for c, _ in cz]))
            zeros.append(None if cz[0][1] is None else torch.cat([cast(torch.Tensor, z) for _, z in cz]))
            s0s.append(torch.cat([scales[s][0] for s in stems]))
            deltas.append(torch.cat([scales[s][1] for s in stems]))
            if nvfp4:
                gs.append(_bank_globals(src, name, stems))
        zero = None if zeros[0] is None else torch.stack(cast(list[torch.Tensor], zeros)).to(device)
        q = src.quantizer(parts[0][0][0], "moe")
        nv: dict[str, torch.Tensor] = {}
        if nvfp4:
            # one weight_global_scale per expert (the gate's, shared by gate and up) and one input scale per
            # bank projection (`execute.input_scales` stores the same scale in every expert of a projection)
            if any(not torch.equal(g[1], gs[0][1]) for g in gs):
                raise Refused(
                    f"{src.path}: {name}: the experts of a bank projection store different input_global_scales "
                    f"({[g[1].tolist() for g in gs][:4]}); one bank serves them under one"
                )
            nv = {"gscale": torch.stack([g[0] for g in gs]).to(device), "igscale": gs[0][1].to(device)}
        return ServedBank(parts, torch.stack(codes).to(device), zero, torch.stack(s0s).to(device),
                          torch.stack(deltas).to(device), q, **nv)  # fmt: skip

    gate_up, down = bank(("gate_proj", "up_proj")), bank(("down_proj",))
    if (
        gate_up.codes.shape[1:] != getattr(mod, fused_experts.IN).shape[1:]
        or down.codes.shape[1:] != getattr(mod, fused_experts.DOWN).shape[1:]
    ):
        raise Refused(f"{name}: stored expert shapes do not match the model's banks")
    return ServedExperts(gate_up, down, cast(torch.nn.Module, mod.act_fn), grouped)


def _materialize(model: torch.nn.Module, device: torch.device) -> None:
    params: dict[torch.Tensor, torch.nn.Parameter] = {}
    buffers: dict[torch.Tensor, torch.Tensor] = {}
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            if not p.is_meta:
                continue
            fresh = params.get(p)
            if fresh is None:
                fresh = params[p] = torch.nn.Parameter(torch.empty_like(p, device=device), p.requires_grad)
            setattr(mod, name, fresh)
        for name, b in mod.named_buffers(recurse=False):
            if b.is_meta:
                setattr(mod, name, torch.empty_like(b, device=device))
            elif b.device.type == "cpu" and device.type != "cpu":
                fresh = buffers.get(b)
                if fresh is None:
                    fresh = buffers[b] = b.to(device)
                setattr(mod, name, fresh)


def _load_buffers(model: torch.nn.Module, src: Packed, names: Any) -> None:
    with torch.no_grad():
        for module_name, mod in model.named_modules():
            for name, buffer in mod.named_buffers(recurse=False):
                if name in mod._non_persistent_buffers_set:
                    continue
                key = names.tensor(f"{module_name}.{name}" if module_name else name)
                if key is not None:
                    buffer.copy_(src.tensor(key).to(buffer.dtype))


def _cpu_buffers(model: torch.nn.Module) -> None:
    for mod in model.modules():
        own = {
            n: b
            for n, b in mod.named_buffers(recurse=False)
            if n in mod._non_persistent_buffers_set and b.is_floating_point()
        }
        if not own or not hasattr(mod, "config"):
            continue
        try:
            signature(type(mod)).bind(config=mod.config)
        except TypeError:  # a required constructor argument cannot be reconstructed from config alone
            continue
        with torch.device("cpu"):
            fresh = type(mod)(config=mod.config)
        for n, b in fresh.named_buffers(recurse=False):
            if n in own:
                own[n].copy_(b)


def served_modules(model: torch.nn.Module) -> dict[str, Any]:
    """``{module name: ServedLinear | ServedBank}`` of a student: everything that holds a trainable `delta`."""
    from .served_modules import ServedBank, ServedLinear

    return {n: m for n, m in model.named_modules() if isinstance(m, ServedLinear | ServedBank)}


def float_names(model: torch.nn.Module, src: Packed) -> dict[str, str]:
    """In-memory parameter name -> checkpoint tensor name, for the student's float (non-served) parameters."""
    from ..arch.adapters import fused_experts
    from ..intake.source_names import SourceNames

    names = SourceNames(src.path)
    out: dict[str, str] = {}
    for n, _ in model.named_parameters():
        if n.endswith(".delta"):
            continue
        key = names.tensor(n)
        if key is None:
            parent, _, leaf = n.rpartition(".")
            bank = model.get_submodule(parent) if parent else model
            assert leaf in (fused_experts.IN, fused_experts.DOWN) and fused_experts.is_bank(bank)
            continue  # a float bank stored per expert: stacked on load, never a trained float
        out[n] = key
    return out


def write(
    src: Packed,
    out: str | Path,
    scales: dict[str, tuple[torch.Tensor, torch.Tensor]],
    floats: dict[str, torch.Tensor],
    stage: dict,
    teacher: str | None,
    devices: Sequence[torch.device],
    moe: bool = False,
) -> Path:
    """Write `src` with new scales and float tensors to `out`; every other tensor is copied byte for byte."""
    out = Path(out)
    if out.resolve() == src.path.resolve():
        raise Refused(f"{out}: a checkpoint step cannot write over its input")
    if set(scales) != set(src.stems):
        raise ValueError("write: scales must cover exactly the checkpoint's quantized modules")
    _clear(out, src)
    # the tensors the state file keeps, and the one function that turns them into the stored scale: the shard
    # and the state are written from the same values, so the next load cannot read the two differently
    kept = {
        s: (
            (a.float() if src.format_of(s).weight_type == "fp8" else a.to(torch.bfloat16)).contiguous(),
            d.float().contiguous(),
        )
        for s, (a, d) in scales.items()
    }
    new = {f"{s}.weight_scale": src.stored_scale(s, *kept[s]) for s in src.stems}
    for s in src.stems:  # the write's own check: what it saves must read back as what it stores
        if not torch.equal(src.stored_scale(s, *kept[s]), new[f"{s}.weight_scale"]):
            raise Refused(
                f"{out}: the stored scale of {s} would not read back from its state "
                "(a scale that is not finite cannot); not written"
            )
    for k, v in floats.items():
        if k not in src.where:
            raise ValueError(f"write: {k} is not a tensor of {src.path}")
        new[k] = v
    by_file: dict[str, list[str]] = {}
    for k, f in src.where.items():
        by_file.setdefault(f, []).append(k)
    for f, keys in sorted(by_file.items()):
        with safe_open(src.path / f, "pt") as h:
            meta = h.metadata() or {"format": "pt"}
            ts = {}
            for k in keys:
                t = h.get_tensor(k)
                ts[k] = new[k].to(t.dtype if k in floats else new[k].dtype).contiguous() if k in new else t
        save_file(ts, str(out / f), metadata=meta)
    side = {f"s0.{s}": s0 for s, (s0, _) in kept.items()}
    side |= {f"delta.{s}": d for s, (_, d) in kept.items()}
    side |= {f"float.{k}": v.float().contiguous() for k, v in floats.items()}
    (out / STATE).parent.mkdir(exist_ok=True)
    save_file(side, str(out / STATE))
    # the files that make the directory look like a checkpoint last: config.json, then build.json
    rest = [p for p in src.path.iterdir() if p.is_file() and not p.name.endswith(".safetensors")]
    for p in sorted(rest, key=lambda p: p.name == "config.json"):
        if p.name not in BUILD_RECORDS:
            shutil.copy2(p, out / p.name)
    rec = dict(src.record)
    rec["teacher"] = teacher
    rec["card_count"] = len(devices)
    rec["devices"] = [str(d) for d in devices]
    # a stage inside this build's staging directory is recorded relative to the checkpoint, whose published name
    # differs from the staging name; a source outside the build keeps its path as given
    root = out.parent.parent if out.parent.name == STAGES else out
    try:
        parent = src.path.relative_to(root).as_posix()
    except ValueError:
        parent = str(src.path)
        # the source is another build's checkpoint: its measurements say what that build did, so keep them
        # apart from this build's cards/devices rather than relabel them as this build's
        measured = {k: rec.pop(k) for k in _MEASURED if k in rec}
        if measured:
            rec["source_build"] = measured
    if not src.record:
        # a checkpoint another tool wrote: no record to copy, and without these `qda inspect` of the output
        # falls back to the default (sm90) table, which has no row for the format the stage actually planned
        rec["format"] = src.fmt.name
        rec["source"] = str(src.path)
        rec["target"] = f"{src.target.arch}@{src.target.vllm}"
    rec["lineage"] = [*rec.get("lineage", []), parent]
    rec["history"] = [*rec.get("history", []), stage]
    rec.pop("stages", None)  # the parent build's stages; `staged` writes this build's
    rec["trained"] = sorted(floats)
    if "serve" not in rec:  # a checkpoint not written by qda: its plans on the default target
        # a stored router that scales each expert itself routes outside vLLM's fused kernel
        custom = any(name.endswith(".per_expert_scale") for name in src.where)
        plans = {
            k: src.target.plan(src.fmt, k, decomposed=k == "moe" and custom)
            for k in (("dense", "moe") if moe else ("dense",))
        }
        rec["serve"] = merge_serve(plans, src.target.vllm)
    # a stage on packed weights reads no calibration; the digest is the source build's, or says it has none
    if "calib_sha256" not in rec:
        from .calib import calib_record

        rec["calib_sha256"] = calib_record(None) | {
            "sha256": "not recorded: the source checkpoint's build.json has no calib_sha256"
        }
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str))
    return out


def _clear(out: Path, src: Packed) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = {p.name for p in src.path.iterdir() if p.is_file()} | set(BUILD_RECORDS)
    for name in names:
        (out / name).unlink(missing_ok=True)
    shutil.rmtree(out / STATE.parent, ignore_errors=True)
