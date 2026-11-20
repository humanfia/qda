"""``qda`` command line: build checkpoints from a config file, and tools that work on checkpoints.

qda build   CONFIG.yaml [--set key=value ...]   # build; OUT/resolved.yaml --set out=NEW repeats it
qda plan    CONFIG.yaml [--set key=value ...]   # refuse or print the plan; reads free GPU memory, runs no GPU work
qda fisher  map|allocate ...                    # per-unit bits and predicted dKL per menu format, and an allocation
qda eval    CONFIG.yaml [--only KEY[/SUITE[.LABEL]]]  # score checkpoints against a reference (with qda-kl installed)
qda inspect PATH|HF_ID [--json]                 # what a checkpoint is, what it lacks, whether it serves as is
qda calib render --model M --names chat=3,multilingual=2 --rows N --out F   # calibration text in the model's own template
qda doctor [--search CONFIG.yaml]               # check what this machine needs; each miss names its fix
qda rotate-draft --checkpoint C --draft D --out O   # a speculative draft in a rotated checkpoint's basis
qda sensitivity plan|types|table|base|read ...  # GGUF class bytes, an allocation, and one sensitivity read
qda pipeline plan|build SPEC.yaml               # a staged spec: plan every stage, then build it, hashed and resumable

The config's keys are `qda_quant.config.BuildConfig`'s. ``--set`` overrides one key by its dotted path
(``--set out=ckpt2 --set steps.0.damp=0.003 --set devices=[0,1]``); values are parsed as YAML.
Run ``qda COMMAND --help`` for each command's options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qda_quant.exit import ENVIRONMENT, INTERRUPTED, refuse
from qda_quant.io import line_buffered_stdout, show_only
from qda_quant.output.vllm_env import DEFAULT_TP
from qda_quant.registry import Registry

if TYPE_CHECKING:
    from qda_quant.arch import ModelGraph
    from qda_quant.target import KernelPlan

#: Subcommands other packages add to ``qda`` through ``qda.plugins`` hooks: ``add_parser(sub)``
#: functions, like the built-in tools'. qda-kl adds ``eval`` this way, so qda-quant never imports qda-kl.
COMMANDS = Registry(
    "commands",
    {
        "decode-gguf": "qda_quant.gguf.decode:add_parser",
        "rotate-draft": "qda_quant.output.rotate_draft:add_parser",
    },
    group="qda.commands",
)


#: The commands `qda --help` lists and 0.1.x keeps stable: every one the user docs tell you to run.
#: The rest still run when named.
PUBLIC = (
    "build",
    "plan",
    "doctor",
    "inspect",
    "eval",
    "export",
    "decode-gguf",
    "fisher",
    "calib",
    "rotate-draft",
    "sensitivity",
    "traces",
)


def _args(argv: list[str] | None) -> argparse.Namespace:
    from qda_quant import doctor, pipeline
    from qda_quant.calib import public, render
    from qda_quant.intake import cli as intake
    from qda_quant.intake import fisher
    from qda_quant.output import release

    ap = argparse.ArgumentParser(
        prog="qda",
        description="Build checkpoints from a config file, and tools that work on checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, what in (
        ("build", "build a checkpoint"),
        ("plan", "print the plan or the refusal; no GPU work"),
    ):
        p = sub.add_parser(name, help=what)
        p.add_argument("config", help="build config YAML (a checkpoint's resolved.yaml works too)")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one key")
        p.set_defaults(run=_build)
    for tool in (doctor, fisher, intake, public, render, release, pipeline):
        tool.add_parser(sub)
    _sensitivity_parser(sub)
    for _, add_parser in COMMANDS.items():
        add_parser(sub)
    show_only(sub, PUBLIC)
    return ap.parse_args(argv)


def _sensitivity_parser(sub: argparse._SubParsersAction) -> None:
    """``qda sensitivity``: the GGUF class plan and the allocation from measured reads (CPU only)."""
    p = sub.add_parser("sensitivity", help="GGUF class weight bytes and a type allocation from reads")
    q = p.add_subparsers(dest="what", required=True)
    plan = q.add_parser("plan", help="tensor payload bytes of every class at each grid type")
    plan.add_argument("template", help="a bf16 GGUF template (its header is read, nothing else)")
    plan.add_argument("--json", action="store_true", help="one object per class instead of a table")
    types = q.add_parser("types", help="the tensor_types rules of a class allocation")
    types.add_argument("template", help="the bf16 GGUF template the build will quantize")
    types.add_argument("picks", help='an allocation as JSON: {"picks": {"class": "type"}}')
    types.add_argument("--compact", action="store_true", help="one line instead of indented JSON")
    table = q.add_parser("table", help="allocate a type per class from measured cell reads")
    table.add_argument("template", help="a bf16 GGUF template")
    table.add_argument("rows", help='JSON rows: {"class", "type", "out" | "d_kl"}')
    table.add_argument("--baseline", type=float, help="the bf16 read's kl; rows without d_kl subtract it")
    table.add_argument("--budget-bytes", type=int, metavar="BYTES", help="whole-.gguf file-size budget; default: the whole frontier")
    table.add_argument("--floor", type=float, default=0.0, help="a smaller increment counts as zero")
    table.add_argument("--quantum", type=int, default=1 << 20, help="allocation grid, in bytes")
    table.add_argument("--json", action="store_true")
    cache = q.add_parser("groups", help="write the derived-group cache a candidate type map expands through")
    cache.add_argument(
        "config", help="the build config whose recipe the candidates will run (without type_map)"
    )
    cache.add_argument("--template", required=True, help="the bf16 GGUF template the builds will quantize")
    cache.add_argument("--out", required=True, help="where to write it (typically beside the config)")
    cache.add_argument("--json", action="store_true", help="print the cache instead of a summary line")
    base = q.add_parser("base", help="lay a checkpoint out one tensor class per shard, for the reads")
    base.add_argument("anchor", help="the bf16 checkpoint the reads are variants of")
    base.add_argument("--out", required=True, help="a fresh directory for the prepared base")
    base.add_argument("--max-shard", type=int, default=4 << 30, help="shard cap, in bytes")
    base.add_argument("--json", action="store_true", help="print the base manifest")
    one = q.add_parser("read", help="fake-quant one class at one type, score it, and write the row")
    one.add_argument("base", help="a prepared base (the `base` command's directory)")
    one.add_argument("--class", dest="klass", default="bf16", help="the tensor class; bf16 is the baseline")
    one.add_argument("--type", dest="kind", default="bf16", help="Q4_K, Q5_K, Q6_K, Q8_0, or bf16")
    one.add_argument("--ref", required=True, help="the sealed non-GGUF reference the judge scores against")
    one.add_argument("--out", required=True, help="a fresh directory for this read")
    one.add_argument("--subset", help="the reference subset the judge screens on")
    one.add_argument(
        "--tp", type=int, default=DEFAULT_TP, help=f"tensor parallel size (default {DEFAULT_TP})"
    )
    one.add_argument("--libggml", help="a built libggml-base; Q5_K needs one")
    one.add_argument("--no-check", dest="check", action="store_false", help="skip the scorer's check")
    one.add_argument("--json", action="store_true", help="print the row instead of its line")
    p.set_defaults(run=_sensitivity)


def _sensitivity(a: argparse.Namespace) -> int:
    from qda_quant.gguf import allocate
    from qda_quant.gguf import classes as sensitivity
    from qda_quant.step import Refused

    try:
        if a.what == "types":
            picks = allocate.one_allocation(json.loads(Path(a.picks).read_text()))
            rules = sensitivity.expand(a.template, picks["picks"])
            print(json.dumps(rules) if a.compact else json.dumps(rules, indent=1))
            return 0
        if a.what == "base":
            from qda_quant.gguf import variant

            got = variant.prepare_base(a.anchor, a.out, max_shard=a.max_shard)
            print(json.dumps(got, indent=1) if a.json else f"{a.out}: {len(got['classes'])} classes")
            return 0
        if a.what == "groups":
            from qda_quant.config import BuildConfig
            from qda_quant.gguf import file, groups

            config = BuildConfig.from_file(a.config)
            if config.exporter.get("gguf", {}).get("type_map"):
                raise Refused(
                    "this config already carries a type_map: derive the cache from the base recipe (the same "
                    "config without type_map and groups)"
                )
            data = groups.derive(config.plan().graph, file.Template(a.template), model=config.model)
            where = groups.write(data, a.out)
            if a.json:
                print(json.dumps(data, indent=1, sort_keys=True))
            else:
                print(
                    f"{where}: {len(data['units'])} units over {len(data['tensors'])} tensors "
                    f"({data['model_type']}, template sha256 {data['template']['sha256'][:12]})"
                )
            return 0
        if a.what == "read":
            from qda_quant.gguf.read import read

            try:
                row = read(
                    a.base,
                    a.klass,
                    a.kind,
                    a.ref,
                    out=a.out,
                    subset=a.subset,
                    tp=a.tp,
                    libggml=a.libggml,
                    check=a.check,
                )
            except ImportError as e:
                return refuse(
                    f"a read needs qda-kl installed ({e})",
                    "uv pip install -c constraints.txt ./packages/qda-kl",
                    code=ENVIRONMENT,
                )
            print(json.dumps(row, indent=1) if a.json else f"{a.out}: kl {row['kl']:.6g}")
            return 0
        p = sensitivity.plan(a.template)
        if a.what == "plan":
            if a.json:
                for name, c in p.classes.items():
                    print(
                        json.dumps(
                            {"class": name, "payload_bytes": c.bytes, "bf16_payload_bytes": c.bf16_bytes}
                        )
                    )
                # the whole file under the rule and at bf16: the size a budget can be stated against
                print(
                    json.dumps(
                        {
                            "totals": {
                                "rule_file_bytes": p.file_total(),
                                "rule_payload_bytes": p.payload_total(),
                                "bf16_file_bytes": p.file_total({c: "bf16" for c in p.classes}),
                            }
                        }
                    )
                )
            else:
                head = f"{'class':<12}{'tensors':>8}{'bf16':>16}" + "".join(
                    f"{t:>14}" for t, _ in sensitivity.CELL_TYPES
                )
                print(head)
                for name, c in p.classes.items():
                    row = f"{name:<12}{c.tensors:>8}{c.bf16_bytes:>16,}"
                    print(
                        row
                        + "".join(
                            f"{c.bytes[t]:>14,}" if c.bytes[t] is not None else f"{'-':>14}"
                            for t, _ in sensitivity.CELL_TYPES
                        )
                    )
                print(
                    f"{'TOTAL':<12}{sum(c.tensors for c in p.classes.values()):>8}{p.total({c: 'bf16' for c in p.classes}):>16,}"
                )
            return 0
        cells = allocate.rows_from_manifest(a.rows, baseline=a.baseline)
        got = (
            allocate.allocate(p, cells, a.budget_bytes, floor=a.floor, quantum=a.quantum)
            if a.budget_bytes is not None
            else None
        )
        rows = (
            [got.as_dict()]
            if got
            else [x.as_dict() for x in allocate.frontier(p, cells, floor=a.floor, quantum=a.quantum)]
        )
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for x in rows:
                picks = " ".join(f"{k}={v}" for k, v in x["picks"].items() if v != allocate.BF16)
                print(
                    f"file {x['file_bytes']:>16,} B  payload {x['payload_bytes']:>16,} B"
                    f"  dKL {x['d_kl']:.6g}  {picks or 'all bf16'}"
                )
        return 0
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    except ValueError as e:  # a pydantic validation error, whose message is the key that is wrong
        return refuse(str(e))


def _kernel_line(kind: str, plan: KernelPlan) -> str:
    """One readable line for a `target.KernelPlan`: its kernel, dtypes, serve arguments and unverified note."""
    from qda_quant.target import UNVERIFIED_ON_HARDWARE

    line = f"{kind} {plan.kernel}, {plan.scale_dtype} scales, {plan.act_dtype} activations"
    if plan.serve_args:
        line += ", serve " + " ".join(f"{k}={v}" for k, v in plan.serve_args.items())
    if UNVERIFIED_ON_HARDWARE in plan.notes:
        line += f" ({UNVERIFIED_ON_HARDWARE})"
    return line


def _slow_gdn(graph: ModelGraph) -> str | None:
    """The warning the plan adds when this model's GDN layers would take transformers' slow torch path.

    transformers binds the GDN delta-rule kernels to flash-linear-attention (and its conv kernels to a
    top-level `causal_conv1d` package, or to the adapter `qda_quant.arch.derive_trace` binds from fla's own
    Triton conv); when one cannot bind it silently falls back to a reference implementation that is correct
    but much slower on every calibration forward. This is the same message a CUDA build refuses with
    (`engine.build._refuse_fast_gdn`), so the plan says in one line exactly why a build would stop.
    """
    from qda_quant.checks import gdn_fallback

    return gdn_fallback(graph)


def _default_out(config: str) -> Path:
    """Where ``qda build`` writes when the config names no ``out``: ``./<config stem>/``, absolute.

    Next to the user, not next to the config (a build must not write into a checkout's recipes): the current
    directory, or the first of ``QDA_OUTPUT_ROOTS`` when it is set, whose refusal of a path outside every
    root still holds (`qda_quant.output.roots.output_path`). The build's own checks refuse a ``./<stem>/``
    that holds anything, as for any ``out``.

    Raises:
        Refused: ``QDA_OUTPUT_ROOTS`` refuses the path.
    """
    from qda_quant.output.roots import output_path

    return output_path(Path(config).stem)


def _build(a: argparse.Namespace) -> int:
    from qda_quant.arch import KEEP_ROLES
    from qda_quant.config import BuildConfig
    from qda_quant.step import Refused

    try:
        cfg = BuildConfig.from_file(a.config, a.set)
        if a.cmd == "build" and cfg.out is None:
            cfg.out = str(_default_out(a.config))
            print(f"out {cfg.out} (no out given: ./<config stem>/ in the current directory)")
        if a.cmd == "plan":
            from qda_quant.output.exporters import parse_exporter

            note = parse_exporter(cfg.exporter).prepare(cfg.recipe)
            if note:
                print(note)
            p = cfg.plan()
            print(f"format {p.fmt.name}")
            from qda_quant.output.exporters import DEFAULT_EXPORTER, parse_exporter
            from qda_quant.output.serve_policy import sglang_limits

            if parse_exporter(cfg.exporter).name == DEFAULT_EXPORTER:
                from qda_quant.intake.loading import load_config

                source_config = load_config(p.source)
                for warning in sglang_limits(
                    p.fmt,
                    "moe" if p.plan_moe is not None else "dense",
                    biased_experts=bool(p.graph.extra.get("expert_biases")),
                    graph=p.graph,
                    text_config=source_config.get_text_config(),
                    source_config=source_config,
                    formats=p.formats,
                ):
                    print(f"qda: warning: {warning}")
            if p.generate:
                print(
                    f"stage generate: {p.generate} (calibration size and vocabulary checked after generation)"
                )
            for skip in p.steps_skipped:
                print(f"step {skip['step']}: skipped (when: accepted): {skip['reason']}")
            if p.segments:
                print(f"layers {len(p.graph.layers)}  passes {[k for k, _ in p.segments]}")
                print(_kernel_line("dense", p.plan_dense))
                if p.plan_moe is not None:
                    print(_kernel_line("moe", p.plan_moe))
                overrides = {k: f for k, f in p.formats.items() if k not in p.defaulted}
                if overrides:
                    names = sorted({f.name for f in overrides.values()})
                    print(f"format_overrides: {len(overrides)} units in {names}")
                if p.stored_formats:
                    # the output's own stored types (a GGUF's mix) solved these units where the file writes
                    # their codes; no `format_overrides` rule names them
                    names = sorted(set(p.stored_formats.values()))
                    print(f"mix formats: {len(p.stored_formats)} units in {names} (from the stored types)")
                if p.format_unmet:
                    # a unit stored two ways cannot be solved in one format: its codes are taken from the template
                    print(
                        f"mix formats: {len(p.format_unmet)} fused units stored in two types, their codes "
                        "will be dropped (the template's values are written instead)"
                    )
                for cls, name in p.format_defaults.items():
                    print(
                        f"format_defaults: {cls.replace('_', ' ')} stored {name} (the format names no symmetry)"
                    )
                for got in p.clips_by_format().values():
                    # the body format is the one the clip was admitted for; the others fall back by unit
                    named = got[p.fmt.name]
                    served = sorted(f for f, clip in got.items() if clip == named)
                    takes: dict[str, list[str]] = {}
                    for f, clip in got.items():
                        if clip != named:
                            takes.setdefault(clip, []).append(f)
                    tail = "; ".join(
                        f"{', '.join(sorted(fs))} units take {clip}" for clip, fs in sorted(takes.items())
                    )
                    print(f"clip {named}: {', '.join(served)} units; {tail}")
                if p.formats:
                    for (name, kind), plan in sorted(p.plans.items()):
                        print(_kernel_line(f"{kind} in {name}", plan))
                print(f"residency {p.residency}")
                if p.output_bytes:
                    # an exporter that sizes its own artefact gives the exact figure a budget is checked
                    # against; rounded GiB alone cannot be compared with a byte cap
                    if p.output_bytes_exact:
                        weights = f", {p.weight_bytes} in weights" if p.weight_bytes else ""
                        print(
                            f"output {p.output_bytes / 2**30:.1f} GiB to write "
                            f"({p.output_bytes} bytes{weights})"
                        )
                    else:
                        # the engine's estimate, not a size to compare with a byte cap; the build records what
                        # it wrote (`build.json` `output_bytes`)
                        print(f"checkpoint {p.output_bytes / 2**30:.1f} GiB to write (predicted)")
                for what in p.exporter_builds:
                    print(f"exporter builds its own {what} first (none was given to read)")
                for role, mods in (p.graph.extra.get("kept_bf16") or {}).items():
                    how = "ignore" if role in KEEP_ROLES else "architecture"
                    print(f"kept bf16 ({role}, {how}): {len(mods)} weights, e.g. {mods[0]}")
                by_pattern: dict[str, list[str]] = {}
                for mod, (pattern, _kind) in (p.graph.extra.get("ignored") or {}).items():
                    by_pattern.setdefault(pattern, []).append(mod)
                for pattern, mods in by_pattern.items():
                    print(f"kept bf16 ({pattern}, ignore): {len(mods)} weights, e.g. {mods[0]}")
                if line := p.graph.extra.get("kept_unservable_line"):
                    print(line)  # modules no kernel of this format serves; the build keeps them bf16
            for s in p.checkpoint_steps:
                print(f"then checkpoint step {s.name} on the packed {p.fmt.name} output")
            for peak in p.stage_peaks:  # `compile_plan` refused any stage above the card's free memory
                print(f"stage {peak.step}: {peak}")
            if slow := _slow_gdn(p.graph):
                print(f"qda: warning: {slow}", file=sys.stderr)
            return 0
        out = cfg.build()
        _print_serve(out)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    return 0


def _print_serve(out: Path) -> None:
    """How to serve the checkpoint `qda build` just wrote, from the one command qda renders.

    `build.json["serve"]` is the kernel plan; :func:`qda_quant.output.serve_command.serve_command_line` turns
    it into the ``vllm serve`` line and its environment, and every user-facing place prints that same line.
    """
    from qda_quant.output.serve_command import serve_command_line
    from qda_quant.step import Refused

    try:
        line = serve_command_line(out)
    except Refused as e:  # the plan names a setting no `vllm serve` flag carries
        print(f"qda: warning: no serve command: {e}", file=sys.stderr)
        return
    if line:
        print(f"serve it with:\n  {line}")


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``qda``.

    Ctrl-C unwinds the command like any error and leaves `INTERRUPTED` (130), with one line on stderr rather
    than the frames of a `KeyboardInterrupt`; SIGTERM and SIGHUP end the process the way the OS sends them
    (the commands that own engine processes catch those too, `qda_kl.lifetime.owned`).
    """
    line_buffered_stdout()
    a = _args(argv)
    try:
        return a.run(a)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
