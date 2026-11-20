"""The words the agent is given: the task statement, the clock, and the one `/goal` instruction.

The statement follows the paper's (App. "The task statement"); it names no method to use or to avoid.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from datetime import datetime
from importlib import resources
from pathlib import Path
from string import Template

from qda_quant.format import Format

from qda_agent.commands import NAMES
from qda_agent.host import best_words
from qda_agent.judge.checks import HEAD_16BIT

__all__ = ["candidate0_kl_statement", "clock", "goal", "task_statement"]


NUMBERS = {
    0: "No",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
}


def _utc(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def goal(fmt: str, deadline: datetime, commands: Collection[str] = NAMES) -> str:
    """The one instruction, sent at the start and again whenever the agent goes idle.

    Every send carries the resume sentence: a sign-in refresh starts a new session, a one-instruction
    session is replaced, and a CLI may compact its own context, so any turn can be the one where the agent
    has lost the thread. `commands` names what the run installs, so `kl list` is asked for only when the
    run gives `kl`.

    Args:
        fmt: The format preset.
        deadline: The GPU deadline (UTC).
        commands: The commands this run installs (`toolset.commands`).

    Returns:
        The instruction, without the `/goal ` prefix.
    """
    resume = "first read notes/"
    if "kl" in commands:
        resume += " and run `kl list`"
    return (
        f"Read TASK.md in full and follow it: build the best checkpoint in the {fmt} tier and keep the best "
        f"one submitted (handin) until {_utc(deadline)}; keep notes/ current. If you are picking up a run "
        f"already in progress (a new session, or a compacted context), {resume} to recover what you already "
        "built and measured."
    )


def candidate0_kl_statement(value: float | None) -> str:
    """The configured comparison reading, shared by the task and baseline refusals."""
    if value is None:
        return ""
    return (
        f"Candidate 0's KL is {value!r}, read by the same scorer and reference as your candidates "
        "(lower KL is better)."
    )


def _prior_line(baselines: list[Path] | list[str], starts: list[tuple[str, str, float | None]] | None) -> str:
    """The Prior section after "Original model: ...": the baselines, or each labelled start with its KL.

    Without starts the sentence keeps the shipped template's exact bytes (its line break after "read the"),
    so a run that configures none renders TASK.md byte for byte as before.
    """
    if not starts:
        return (
            f" Baselines, by fallback order: {', '.join(f'`{b}`' for b in baselines) or 'none'}. Your GPU jobs "
            "can read the\noriginal model and the baselines at these paths (read-only); your shell cannot read "
            "their weights."
        )
    rows = "\n".join(
        f"- {name}: `{where}`" + (f", KL {kl:.4g}" if kl is not None else "") for name, where, kl in starts
    )
    return (
        " Your pre-built starts, best KL first (the directory your jobs read, then its measured KL):\n"
        f"{rows}\n"
        "Your GPU jobs can read the original model and the starts (read-only); your shell cannot read their "
        "weights."
    )


def task_statement(
    *,
    model: str,
    fmt: str,
    n_cards: int,
    start: datetime,
    deadline: datetime,
    baselines: list[Path] | list[str],
    metric: str,
    final_rules: str,
    handin_rule: str = "",
    budget_bpw: float | None,
    format_menu: list[str],
    require_build_receipt: bool,
    max_kl: float,
    gen_max_tokens: int,
    max_run_bytes: int | None = None,
    min_free_bytes: int | None = None,
    candidate0_kl: float | None = None,
    eval_max_tokens: int | None = None,
    kl_noise: float | None = None,
    starts: list[tuple[str, str, float | None]] | None = None,
    calib: str | None = None,
    calib_note: str = "",
    commands: tuple[str, ...] = NAMES,
    subcommands: dict[str, list[str]] | None = None,
    tools: list[str] | None = None,
    template: Path | None = None,
    engine: str = "vllm",
    head: str = HEAD_16BIT,
    head_lease: tuple[str, ...] = (),
    budget_bytes: int | None = None,
    budget_basis: str = "file",
    engine_floor: float | None = None,
    full_card_moe: bool = False,
) -> str:
    """Render TASK.md.

    Args:
        model: The original model.
        fmt: The format preset.
        n_cards: GPUs the search holds.
        start: Start of the search (UTC).
        deadline: GPU deadline (UTC).
        baselines: Baseline checkpoints in fallback order (named by the path the agent's jobs read).
        metric: The metric the search ranks readings by (`judge.metric`).
        final_rules: What counts at the end (`judge.final_rules.statement`).
        handin_rule: The old separate Submission sentence for a `handin`-first rule
            (`judge.final_rules.final_handin_statement`), now empty: the rule states itself once, in the
            deadline sentence. Kept for templates that carry the slot.
        budget_bpw: The configured bits per quantized weight budget; None: no budget.
        format_menu: The judge's configured format menu for this tier.
        require_build_receipt: Whether recorded-build eligibility and retention are enforced.
        max_kl: The judge's threshold for immediate KL refusal.
        gen_max_tokens: The longest answer of this search's generation check, in tokens.
        max_run_bytes: The run's disk cap (`judge.max_run_bytes`), stated in GB; None: no cap is stated.
        min_free_bytes: The run's free-space floor (`judge.min_free_bytes`); 0 states no floor, None states
            the default floor.
        candidate0_kl: Candidate 0's configured KL reading; None: no comparison number is given.
        eval_max_tokens: The final evaluation's generation cap, in tokens (`judge.eval_max_tokens`); a
            template that renders the long-generation paragraph needs it.
        kl_noise: The row's relative KL tie band (`judge.kl_noise`, 0.02 = 2 %); None: the statement says no
            tie band.
        starts: The row's pre-built starts (`judge.starts`) in KL order, each as `(label, agent-visible dir,
            measured KL or None)`; when given, the Prior section lists every start with the directory its
            jobs read and its KL, and never a host path. None or empty: the baselines by fallback order.
        calib: The default calibration file, if any.
        calib_note: What that calibration's metadata says it lacks (a sentence after the file, or ""); see
            `qda_agent.prepare.calib_thinking_note`.
        commands: The commands the agent has (they are counted in the statement, and the statement names a
            command only when it is in this tuple).
        subcommands: The verbs each command offers (`Toolset.subcommands`); None: those of the toolset the
            default commands name. A clause that names a verb (`qda fisher`, `qda plan`) is kept only when it is
            here.
        tools: The Tools list items, one per toolset component (`Toolset.sections`), in order; None: those of
            the `qda` preset's four commands. The last item's closing `;` becomes a `.`.
        template: A `string.Template` file replacing the built-in statement.
        engine: The judge's serving engine.
        head: The judge's head rule (`judge.head`): `16-bit` by default; an exception model names the type or
            format its release stores the output head in.
        head_lease: The formats `judge.head_lease` offers the head (`checks.budget.lease_formats`); empty
            when the lease is off.
        budget_bytes: The GGUF byte budget, for llama.cpp.
        budget_basis: What `budget_bytes` counts: `file` (the whole `.gguf` the scorer reads, the
            default) or `payload` (its stored tensor bytes only, an explicit opt-in).
        engine_floor: The BF16 GGUF's KL (decoded, read with vLLM) against the sealed BF16 reference, for llama.cpp.
        full_card_moe: Whether this is a serialized full-card MoE search (its paragraph is rendered behind
            this flag, never appended to the finished text).

    Returns:
        The statement text.
    """
    text = template.read_text() if template else resources.files(__package__).joinpath("task.md").read_text()
    if eval_max_tokens is None and "${eval_max_tokens}" in text:
        raise ValueError(
            "the task statement states the evaluation's generation cap: set judge.eval_max_tokens (the final "
            "evaluation's max_tokens for this row)"
        )
    if template is not None and "${tools}" not in text:
        warnings.warn(
            f"task template {template} has no ${{tools}}: the statement lists none of the toolset's tools "
            "(the Tools list is assembled from the toolset's components since v0.2)",
            stacklevel=2,
        )
    gguf = engine == "llama.cpp"
    have = set(commands)
    verbs = subcommands if subcommands is not None else _default_subcommands()
    qda_verbs = set(verbs.get("qda", ())) if "qda" in have else set()
    return Template(text).substitute(
        model=model,
        format=fmt,
        n_cards=n_cards,
        start=_utc(start),
        deadline=_utc(deadline),
        baselines=", ".join(f"`{b}`" for b in baselines) or "none",
        prior_line=_prior_line(baselines, starts),
        candidate0_kl_line=candidate0_kl_statement(candidate0_kl),
        eval_max_tokens=eval_max_tokens,
        tie_line="" if kl_noise is None else f" Two readings within {100 * kl_noise:g} % are a tie.",
        budget_line=(
            (
                (
                    f"Budget: at most {budget_bytes} bytes for the whole .gguf file."
                    if budget_basis == "file"
                    else f"Budget: at most {budget_bytes} bytes of stored tensor payload."
                )
                if budget_bytes is not None
                else "No GGUF-byte budget is configured."
            )
            if gguf
            else (
                f"Budget: at most {budget_bpw} bits per quantized weight (bpw)."
                if budget_bpw is not None
                else "No bits-per-weight budget is configured."
            )
        ),
        budget_accounting=(
            (
                "The budget counts the whole .gguf file the scorer reads: its header, metadata and "
                "alignment padding count, and the stored tensor payload is recorded beside it."
                if budget_basis == "file"
                else "Count the GGUF's stored tensor payload bytes, including weights, scales and zero "
                "points; file headers and alignment padding do not count."
            )
            if gguf
            else (
                "The numerator counts stored weight bits, scales and zero points of the candidate's decoder "
                "Linears,\nincluding Linears kept in bf16 at their stored width. The denominator is fixed: "
                "the number of weights\nin the original model's decoder Linears that llm-compressor's "
                "uniform default quantizes."
            )
        ),
        engine_facts=(
            "Engine facts: the checkpoint uses the GGUF types in the menu and runs in stock llama.cpp."
            if gguf
            else (
                "Engine facts: fused q/k/v share a format; fused gate/up share a format; a routed expert "
                f"bank shares\na format; {_layer_acts_fact(format_menu)}. Small-batch generation speed is "
                f"roughly\nproportional to bits per weight. {_unservable_sentence()}"
            )
        ),
        engine_floor_line=(
            f"\nEngine floor: the BF16 GGUF, decoded and scored like every candidate, has KL {engine_floor!r} "
            "against the sealed BF16 reference. Candidate KL includes this floor."
            if gguf and engine_floor is not None
            else ""
        ),
        serving_engine="stock llama.cpp" if gguf else "stock vLLM",
        gguf_build_line=_gguf_build_line(format_menu) if gguf else "",
        head_line=_head_line(head, gguf=gguf, lease=head_lease),
        group_line="" if gguf else _group_line(format_menu),
        module_formats_line=(
            "" if gguf else _module_formats_line(fmt, format_menu, qda_verbs, bpw="bpw" in have)
        ),
        format_menu=", ".join(f"`{fmt}`" for fmt in format_menu) or "no quantized formats",
        max_kl=max_kl,
        disk_line=(
            f"Disk budget: the run may hold {max_run_bytes / 1e9:.1f} GB in all (builds, hand-in copies and "
            "your work/ together). A build, job or hand-in that would pass it is refused: prune your work/ "
            "or hand in."
            if max_run_bytes is not None
            else ""
        ),
        floor_line=(
            "Disk floor: the judge keeps a free-space buffer on the run's disks. Below it the judge "
            "refuses new builds, readings and hand-ins, stops running builds; only the judge's own "
            "commands free this run's builds: free space only under your work directory, never delete a file "
            "outside it."
            if min_free_bytes != 0
            else ""
        ),
        full_card_moe_section=_full_card_moe_section(full_card_moe, have, qda_verbs),
        commands_line=_commands_line(commands),
        receipt_policy="enabled" if require_build_receipt else "disabled",
        receipt_facts=_receipt_facts(have, qda_verbs, require_build_receipt),
        final_rules=final_rules,
        handin_rule=handin_rule,
        best_metric=best_words(metric),  # a template from before the rules were plugins
        gen_max_tokens=gen_max_tokens,
        calib_line=_calib_line(calib, calib_note),
        n_commands=NUMBERS.get(len(commands), str(len(commands))),
        tools=_items(tools if tools is not None else _default_tools()),
        qda_line="",  # a template from before the Tools list was assembled: its items are in `tools` now
    )


def _calib_line(calib: str | None, note: str) -> str:
    """The default calibration file, and the note its metadata earns."""
    line = f"the default calibration file is `{calib}`." if calib else "no default file is given."
    if note:
        line += f" {note[0].upper()}{note[1:]}."
    return line


def _per_channel(menu: list[str]) -> list[str]:
    """The menu's formats whose scales are one an output row (`w8a8-fp8`), beside a grouped format; else none.

    Such a format has no group to share with the rest of a checkpoint (`RunConfig`'s menu rule leaves it out
    of the one-group rule for the same reason), and the builder stores it beside the grouped modules.
    """
    fmts = [(name, Format.parse(name)) for name in menu if name != "bf16"]
    if not any(f.group_size != -1 for _, f in fmts):
        return []
    return [name for name, f in fmts if f.group_size == -1]


def _layer_acts_fact(menu: list[str]) -> str:
    """The engine fact about a layer's activation format, for this menu.

    One activation format per layer, except the pair the builder runs together: a per-channel format's
    per-token cut beside the body's cut over each group of values (NVFP4 with FP8 W8A8).
    """
    if not (apart := _per_channel(menu)):
        return "each layer has one activation format"
    names = ", ".join(f"`{n}`" for n in apart)
    return (
        f"each module's input is cut as its own format says, so a layer may hold {names} modules beside\n"
        "the other quantized modules"
    )


def _and(items: list[str]) -> str:
    """`a`, `a and b`, `a, b and c`: a list in running text."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _default_subcommands() -> dict[str, list[str]]:
    """The verbs of the toolset the default commands name (for a caller that gives no toolset)."""
    from qda_agent.components import Run
    from qda_agent.toolsets import Qda

    return Qda.model_validate({"components": list(NAMES)}).subcommands(Run(workdir=Path()))


def _commands_line(commands: tuple[str, ...]) -> str:
    """The sentence that counts the commands in `bin/`, or nothing when the run gives none."""
    if not commands:
        return ""
    n = len(commands)
    noun = "command" if n == 1 else "commands"
    return f'{NUMBERS.get(n, str(n))} {noun} in `bin/` reach the rest\n(`export PATH="$PWD/bin:$PATH"`; each takes `--help`):\n'


def _full_card_moe_section(enabled: bool, have: set[str], qda_verbs: set[str]) -> str:
    """The serialized full-card MoE paragraph, behind its flag and naming only the tools the run gives."""
    if not enabled:
        return ""
    build = "qda" in have and "build" in qda_verbs
    parts = ["This is a serialized full-card MoE search."]
    if build:
        step = "For each candidate, run `qda build` with no `--cards` override, wait for it to finish"
        if "kl" in have:
            step += ", run `kl` on its output, and wait for that score"
        parts.append(step + " before submitting the next build.")
    parts.append("The host refuses overlapping candidates.\n")
    return "\n" + " ".join(parts) + "\n"


def _receipt_facts(have: set[str], qda_verbs: set[str], enabled: bool) -> str:
    """The build-receipt paragraph, naming only the commands and verbs the run gives the agent."""
    if not enabled:
        return ""
    build = "qda" in have and "build" in qda_verbs
    qsched = "qsched" in have
    tools = ["`qsched` jobs"] if qsched else []
    if build:
        text = "\nWith it enabled, `qda build` runs the judge's own copy of the library"
        text += ": a change in `src/` reaches your\n`qsched` jobs only." if qsched else "."
    elif qsched:
        text = "\nWith it enabled, a change to `src/` reaches your\n`qsched` jobs only."
    else:
        text = "\nWith it enabled, only the recorded outputs of a build can win."
    if tools:
        text += (
            f"The outputs of {_and(tools)} match no build's record, "
            "and a\nbuild whose `model` is one of your checkpoints has a non-original source: they are "
            "scored but cannot win."
        )
    return text


def _solved() -> dict[str, str]:
    """ggml type name -> the solved format that stores it (the writer's own `_SOLVED` table)."""
    from qda_quant.gguf.file import type_name
    from qda_quant.gguf.writer import _SOLVED

    return {type_name(kind): fmt for fmt, (kind, _) in _SOLVED.items()}


def _unservable_sentence() -> str:
    """The rule a build applies to a shape no kernel serves, from its one owner in `qda-quant`."""
    from qda_quant.engine.build_validation import UNSERVABLE_KEPT_RULE

    return UNSERVABLE_KEPT_RULE


def _head_line(head: str, *, gguf: bool, lease: tuple[str, ...] = ()) -> str:
    """The output head's sentence, rendered from the judge's one rule (`judge.head`).

    The default keeps the head 16-bit: BF16 or F16 in a GGUF, the source's own float type on vLLM. An
    exception model, released with a quantized head, keeps the release's own type or format instead. The
    judge reads the same value (`checks.gguf._type_problem`, `checks.formats._head_format_problem`), so the
    statement cannot tell the agent to store a head the check refuses. With `judge.head_lease` (`lease`, the
    formats it offers) the head may be narrower, and what it saves is lent to the body's budget.
    """
    if head != HEAD_16BIT:
        where = "type" if gguf else "format"
        who = "`output.weight`" if gguf else "lm_head"
        return f"{who} keeps the original model's released {where}, {head}; every build stores it so."
    if gguf:
        return (
            "The output head (`output.weight`) stays 16-bit: store it BF16 or F16, whether or not the menu "
            "names that type."
        )
    if lease:
        narrower = " or ".join(f"`{name}`" for name in lease if name != "bf16")
        return (
            f"lm_head stays 16-bit or is stored {narrower} (the `head` step). Head lease: the head is charged "
            "16 bits a weight in the budget, and the bits a narrower head saves are lent to the decoder "
            "Linears: (their bits + the head's stored bits - 16 x the head's weights) / the denominator must "
            "be inside the budget."
        )
    return "lm_head stays 16-bit (bf16 or fp16)."


def _gguf_build_line(menu: list[str]) -> str:
    """The GGUF override paragraph: the menu's own type names, never one the row does not offer.

    The type -> format pairs and the facts about a type are read from the writer (`_SOLVED`), and every
    clause keeps only the types the row's `format_menu` names. A menu is a row's contract with its exporter:
    a type off the menu is never named, because it is not one this run accepts.
    """
    solved = _solved()
    types = {t: fmt for t, fmt in solved.items() if t in menu}
    unsolved = [t for t in ("Q8_0", "Q5_0", "IQ3_S") if t in menu]
    native = [t for t in ("Q5_K", "IQ4_XS", "IQ4_NL", "Q3_K", "IQ3_S") if t in menu]
    if not (types or unsolved):
        return (
            "\nThe menu names no quantized GGUF type: store every tensor as its template's type, or pick the "
            "types from `exporter.gguf.tensor_types`."
        )
    text = (
        "\nUse `exporter.gguf.tensor_types` to choose each GGUF tensor's stored type independently, "
        "including per layer and projection."
    )
    if types:
        pairs = "; ".join(f"{t} with `{fmt}`" for t, fmt in types.items())
        text += (
            f" For solved codes, add a matching `format_overrides` entry: {pairs}."
            " A unit an override puts in one of those formats is solved on the grid that stores it, so the "
            "recipe keeps one `grid` (`q4k` by default) and still stores several layouts. "
            "Solved q/k/v and gate/up groups share one format; a routed bank's down projection may differ."
        )
    if unsolved:
        text += (
            f" {_and(unsolved)} have no solved format: those slots take template values, with no "
            "`format_overrides` entry."
        )
    if native:
        text += (
            f" {_and(native)} slots, and `clip: imatrix`, need a built `libggml-base`, named by "
            "`exporter.gguf.libggml`."
        )
    if types:
        text += (
            " `clip: imatrix` fits the scales the way `llama-quantize --imatrix` does, from this build's own "
            "calibration, and works on every solved format in the menu."
        )
    text += (
        " A tensor taken from the template instead is fitted with `exporter.gguf.imatrix`: when the run names "
        "one (`judge.llamacpp.imatrix`, the matrix `qda-agent prepare` measured) it hands that file to every "
        "build, and a recipe naming another is refused"
    )
    if "Q8_0" in menu:
        text += "; Q8_0 ignores importance weights"
    text += ". The stored type determines the file's weight bytes. A format override alone does not change that type, and a type the build has no matching override for takes template values"
    if types:
        dropped = "Q4_K codes are dropped with no error" if "Q4_K" in types else ""
        refused = "any other solved format is refused" if dropped else "a solved format is refused"
        text += f": {dropped + ', ' if dropped else ''}{refused}"
    text += (
        ", and `gguf.rounded_from_template` in `build.json` lists every tensor it happened to. Inside a GPU "
        "job, `python -m qda_quant.cli sensitivity plan TEMPLATE.gguf` prices each tensor class at each type, "
        "and `sensitivity table TEMPLATE.gguf ROWS.json --budget-bytes BYTES` picks one type per class from measured "
        "reads (`sensitivity base` and `sensitivity read` make them)."
    )
    return text


def _group_line(menu: list[str]) -> str:
    """The menu's groups and symmetries, which a checkpoint fixes once; bit width is what varies per module."""
    groups: set[int] = set()
    syms: dict[str, set[str]] = {}
    apart = _per_channel(menu)
    for name in menu:
        if name == "bf16":
            continue
        f = Format.parse(name)
        if name not in apart:
            groups.add(f.group_size)
        family = f"w{f.weight_bits}a{f.act_bits}"
        syms.setdefault(family, set()).update(
            {"sym", "asym"} if f.symmetric_defaulted else {"sym" if f.symmetric else "asym"}
        )
    shown = ", ".join("channel" if g == -1 else f"g{g}" for g in sorted(groups))
    each = "; ".join(f"`{fam}` {' or '.join(sorted(s, reverse=True))}" for fam, s in syms.items())
    if apart:
        names = ", ".join(f"`{n}`" for n in apart)
        return (
            f"Groups in the menu: {shown}; {names} has one scale an output row and no group. "
            f"Symmetry: {each}.\n"
            "One checkpoint stores every grouped module at one group, and each format family at one symmetry\n"
            f"(routed and shared experts counted apart). A module stored in {names} sits beside the grouped\n"
            "modules in the same checkpoint: the format, and so the bit width, may differ per module."
        )
    return (
        f"Groups in the menu: {shown}. Symmetry: {each}.\n"
        "One checkpoint stores every quantized module at one group, and each format family at one symmetry\n"
        "(routed and shared experts counted apart); bit width is what may differ per module."
    )


def _module_formats_line(
    fmt: str, menu: list[str], qda_verbs: set[str] | None = None, *, bpw: bool = False
) -> str:
    """How a config sets per-module formats, with an example from this row's menu, and the related facts.

    The sentence about `qda plan` is kept only when the run gives the agent `qda plan`; the `bpw --plan`
    sentence only when it gives `bpw`.
    """
    plan = "`qda plan` prints the units each rule resolves to. " if "plan" in (qda_verbs or ()) else ""
    price = (
        "`bpw --plan RECIPE.yaml` prices a recipe's formats against the budget without building it. "
        if bpw
        else ""
    )
    bits = Format.parse(fmt).weight_bits
    wider = sorted(
        (m for m in menu if m != "bf16" and Format.parse(m).weight_bits > bits),
        key=lambda m: Format.parse(m).weight_bits,
    )
    # The example widens the attention output projection: it is one unit of its own, not part of the fused
    # q/k/v group, so a rule on it cannot fall back to the default scheme on layers that share k and v.
    example = (
        f"for example\n`formats: {{default: {fmt}, 're:.*self_attn\\.o_proj': {wider[0]}}}`"
        if wider
        else f"for example `formats: {{default: {fmt}}}`"
    )
    return (
        f"\nA build config's `formats:` maps `default` and module patterns to menu formats, {example};\n"
        f"{plan}{price}`scale_kd` later in the same recipe keeps each\n"
        "module's stored format; `formats:` on a build whose `model` is an already quantized checkpoint is\n"
        "refused (stored codes are not repacked).\n"
        "One way to search a mix: change one structural group at a time — the attention projections,\n"
        "the state-space projections, the shared experts, or a contiguous run of layers — to a wider\n"
        "format, build it, and compare `kl` with the previous build. A rule names a whole unit: a dense\n"
        "group, or every expert bank of a routed layer. The modules the engine serves fused (q/k/v,\n"
        "gate/up) share one format, and a rule that matches no unit or only part of a fused group is\n"
        "refused at plan time."
    )


def _items(sections: list[str]) -> str:
    """The Tools items as one block; the last one ends the sentence."""
    text = "".join(sections)
    return text[: -len(";\n")] + ".\n" if text.endswith(";\n") else text


def _default_tools() -> list[str]:
    from qda_agent.components import Run
    from qda_agent.toolsets import Qda

    # `components` takes keys, which its before-validator turns into settings models
    toolset = Qda.model_validate({"components": list(NAMES)})
    return toolset.sections(Run(workdir=Path()))


def clock(start: datetime, deadline: datetime, cards: list[int]) -> str:
    """Render CLOCK.md."""
    return f"# Clock (UTC)\nstart {_utc(start)}\nGPU deadline {_utc(deadline)}\ncards {len(cards)}\n"
