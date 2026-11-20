"""The rendered contract must match configurable judge limits and allow mixed formats."""

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qda_agent import prompt, wiki
from qda_agent.components import Run
from qda_agent.toolsets import Qda

#: A measured figure in text the agent reads: a time, a percentage, a rank correlation, a multiplier, a KL
#: reading, or a model size. Format specs (g32, w4a16, Q6_K) and configured limits (`KL > 1`) are not
#: measurements and do not match; a tool contract is stated without a figure for the same reason.
MEASURED = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds|min|mins|minutes|h|hours|%)\b"
    r"|Spearman\s*[-\d.]"
    r"|~?\d+(?:\.\d+)?\s*x\b"
    r"|KL is\s*[-\d.]"
    r"|\b\d+(?:\.\d+)?B(?:-A\d+(?:\.\d+)?B)?\b"
)

#: A vendor's name or its private file naming, which would hint at where a checkpoint came from.
VENDOR = re.compile(r"unsloth|bartowski|\bUD-", re.IGNORECASE)


def _statement(**kw) -> str:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    return prompt.task_statement(
        metric="kl",
        final_rules="Then the last submission that passed counts.",
        model="m",
        fmt="w4a16",
        n_cards=1,
        start=t,
        deadline=t,
        baselines=["baselines/0"],
        budget_bpw=kw.pop("budget_bpw", 4.4),
        format_menu=kw.pop("format_menu", ["bf16", "w8a16", "w4a16"]),
        require_build_receipt=kw.pop("require_build_receipt", True),
        max_kl=kw.pop("max_kl", 1),
        gen_max_tokens=8192,
        eval_max_tokens=kw.pop("eval_max_tokens", 32768),
        tools=[],
        **kw,
    )


@pytest.mark.parametrize("value", [None, 0.0, 0.11649179775139838])
def test_task_states_only_the_configured_candidate0_reading(value):
    text = _statement(candidate0_kl=value)
    if value is None:
        assert "Candidate 0's KL is" not in text
        assert "Your result must beat this number" not in text
    else:
        assert prompt.candidate0_kl_statement(value) in text
        assert str(value) in text and "same scorer and reference" in text
    assert "`kl` and `handin` take your candidate directories" in text


def test_the_disk_cap_is_stated_only_when_it_is_set():
    assert "Disk budget:" not in _statement()
    text = " ".join(_statement(max_run_bytes=10_000_000_000).split())
    assert "Disk budget: the run may hold 10.0 GB in all" in text
    assert "prune your work/ or hand in" in text


def test_the_free_space_floor_says_where_the_agent_may_free_space():
    """The floor is on by default, and the statement tells the agent the one place it may delete files."""
    text = " ".join(_statement().split())
    assert "Disk floor:" in text
    assert "never delete a file outside it" in text
    off = " ".join(_statement(min_free_bytes=0).split())
    assert "Disk floor:" not in off


def test_tier_contract_is_complete_and_has_no_removed_rules_or_module_advice():
    text = " ".join(_statement().split())
    assert "original model's decoder Linears that llm-compressor's uniform default quantizes" in text
    assert "fused q/k/v share a format; fused gate/up share a format" in text
    assert "a routed expert bank shares a format; each layer has one activation format" in text
    assert "lm_head stays 16-bit (bf16 or fp16)." in text
    assert "KL > 1 is refused at once: check rotation / basis" in text
    banned = (
        "every quantized module keeps this format",
        "its format is the baselines'",
        "embedding",
        "sensitive layers",
        "keep these high",
    )
    assert all(phrase not in text for phrase in banned)


def test_mixed_formats_are_stated_as_a_method_and_the_example_widens_an_unfused_unit():
    text = " ".join(_statement().split())
    assert "change one structural group at a time" in text
    assert "compare `kl` with the previous build" in text
    assert "a rule that matches no unit or only part of a fused group is refused at plan time" in text
    assert "'re:.*self_attn\\.o_proj'" in text
    assert "self_attn\\..*_proj" not in text  # the fused q/k/v group: no v_proj on k_eq_v layers
    gguf = " ".join(_statement(engine="llama.cpp", format_menu=["Q4_K"]).split())
    assert "change one structural group at a time" not in gguf  # a GGUF build has no `formats:` rules


def test_the_shipped_statement_tells_fisher_to_confirm_its_picks():
    toolset, run = Qda(), Run(workdir=Path("/nonexistent"))
    sections = " ".join(" ".join(toolset.sections(run)).split())
    assert "verify a chosen allocation with a build and `kl` against the previous one" in sections


def test_the_free_recipe_pricing_is_named_only_when_the_run_gives_bpw():
    assert "`bpw --plan RECIPE.yaml`" not in " ".join(_statement().split())
    priced = " ".join(_statement(commands=("qda", "bpw"), subcommands={"qda": ["plan"]}).split())
    assert (
        "`bpw --plan RECIPE.yaml` prices a recipe's formats against the budget without building it" in priced
    )
    assert "`qda plan` prints the units each rule resolves to" in priced


@pytest.mark.parametrize("receipts", [False, True])
def test_limits_and_receipt_policy_match_the_effective_config(receipts):
    text = _statement(budget_bpw=4.936317940046343, max_kl=0.75000049, require_build_receipt=receipts)
    assert "at most 4.936317940046343 bits" in text and "KL > 0.75000049" in text
    assert f"Build-receipt policy: {'enabled' if receipts else 'disabled'}. When enabled" in text


@pytest.mark.parametrize("budget,floor", [(17_548_181_504, 0.000273), (123_456_789, 0.0)])
def test_gguf_contract_uses_its_weight_budget_menu_and_engine_floor(budget, floor):
    text = " ".join(
        _statement(
            engine="llama.cpp",
            budget_bytes=budget,
            engine_floor=floor,
            format_menu=["Q4_K", "Q6_K"],
        ).split()
    )
    assert f"Budget: at most {budget} bytes for the whole .gguf file." in text
    assert "Format menu: `Q4_K`, `Q6_K`." in text
    assert f"has KL {floor!r} against the sealed BF16 reference" in text
    assert "Candidate KL includes this floor" in text
    assert "must load and run a short inference in stock llama.cpp" in text
    assert "header, metadata and alignment padding count" in text
    assert "`exporter.gguf.tensor_types`" in text and "matching `format_overrides` entry" in text
    assert "Q4_K with `w4a16-asym-g32`; Q6_K with `w6a16-sym-g16`" in text
    assert all(off_menu not in text for off_menu in ("Q5_K", "IQ4_XS", "IQ4_NL", "Q3_K", "Q8_0", "IQ3_S"))
    assert "stored type determines the file's weight bytes" in text
    assert "Q4_K codes are dropped with no error" in text and "`gguf.rounded_from_template`" in text
    assert "keeps every build you handed in whole" in text and "KL > 1 is refused at once" in text
    assert all(
        word not in text for word in ("bpw", "bits per", "denominator", "fused q/k/v", "llm-compressor")
    )


def test_a_gguf_menu_with_bf16_is_stated_as_configured():
    text = " ".join(_statement(engine="llama.cpp", format_menu=["Q4_K", "Q8_0", "BF16"]).split())
    assert "Format menu: `Q4_K`, `Q8_0`, `BF16`." in text
    assert "Q8_0 have no solved format" in text and "IQ3_S" not in text


def test_the_gguf_contract_names_every_solved_type_the_menu_offers_and_no_other():
    """The pairs are the writer's own `_SOLVED` table, filtered to the row's menu: an off-menu type is never named."""
    pytest.importorskip("gguf")
    from qda_quant.gguf.file import type_name
    from qda_quant.gguf.writer import _SOLVED

    names = {fmt: type_name(kind) for fmt, (kind, _) in _SOLVED.items()}
    for fmt, name in names.items():
        text = " ".join(_statement(engine="llama.cpp", format_menu=[name]).split())
        assert f"{name} with `{fmt}`" in text, (name, fmt)
        for other_fmt, other in names.items():
            if other != name:
                assert f"{other} with `{other_fmt}`" not in text, (name, other)


def test_the_gguf_contract_does_not_understate_which_grids_take_the_imatrix_clip():
    """Every grid the clip accepts is one the statement tells the agent it can solve with `clip: imatrix`."""
    pytest.importorskip("gguf")
    from qda_quant.clips import CLIPS
    from qda_quant.gguf.file import type_name
    from qda_quant.gguf.writer import _SOLVED
    from qda_quant.solve.grid import GRIDS

    grids = CLIPS["imatrix"].only_grid
    assert grids is not None
    stored = {s for name in grids if (s := GRIDS[name].stores) is not None}
    assert stored <= set(_SOLVED), "the clip accepts a grid whose format the GGUF writer cannot store"
    menu = [type_name(_SOLVED[fmt][0]) for fmt in stored]
    text = " ".join(_statement(engine="llama.cpp", format_menu=menu).split())
    assert "`clip: imatrix` fits the scales" in text
    for fmt in stored:
        assert f"{type_name(_SOLVED[fmt][0])} with `{fmt}`" in text


def test_a_payload_basis_budget_says_so_in_the_statement():
    """The default states the whole file; a payload-basis config (an explicit opt-in) is labelled as such."""
    text = " ".join(
        _statement(engine="llama.cpp", budget_bytes=123, budget_basis="payload", engine_floor=0).split()
    )
    assert "Budget: at most 123 bytes of stored tensor payload." in text
    assert "headers and alignment padding do not count" in text


def test_vllm_contract_ignores_gguf_budget_and_floor():
    assert _statement(engine="vllm", budget_bytes=123_456_789, engine_floor=0.000273) == _statement()


def test_the_menu_states_its_groups_and_symmetries_as_row_properties():
    """TASK.md names the menu's groups and each family's symmetries, fixed once per checkpoint; bits vary."""
    menu = ["bf16", "w4a16-sym-g64", "w4a16-asym-g64", "w4a16-asym-g128", "w8a16-sym-g64", "w8a16-sym-g128"]
    text = " ".join(_statement(format_menu=menu).split())
    assert "Groups in the menu: g64, g128. Symmetry: `w4a16` sym or asym; `w8a16` sym." in text
    assert (
        "One checkpoint stores every quantized module at one group, and each format family at one symmetry"
        in text
    )
    assert "bit width is what may differ per module" in text and "not a search dimension" not in text


def test_a_per_channel_format_sits_beside_the_grouped_modules():
    """The FP8 W8A8 menu takes no group of the row's and states its own activation cut, as the builder serves it."""
    text = " ".join(_statement(format_menu=["bf16", "w4a4-nvfp4", "w8a8-fp8"]).split())
    assert "Groups in the menu: g16; `w8a8-fp8` has one scale an output row and no group." in text
    assert "each module's input is cut as its own format says" in text
    assert "each layer has one activation format" not in text


def test_the_generation_cap_and_the_tie_band_are_stated_only_when_configured():
    """The long-generation paragraph states the row's eval cap; the tie sentence only with a measured band."""
    text = " ".join(_statement(eval_max_tokens=81920).split())
    assert "The evaluation generates up to 81920 tokens; a check with a smaller N is faster" in text
    assert "are a tie" not in text  # no measured band: the sentence is left out
    banded = " ".join(_statement(kl_noise=0.02).split())
    assert "Two readings within 2 % are a tie." in banded
    with pytest.raises(ValueError, match="eval_max_tokens"):
        _statement(eval_max_tokens=None)


def test_the_prior_section_lists_each_start_with_its_dir_and_kl():
    """The approved wording lists each start: method label, the directory the agent's jobs read, its KL."""
    text = _statement(starts=[("gptq", "baselines/1", 0.02), ("awq", "baselines/0", 0.04)])
    flat = " ".join(text.split())
    assert (
        "Your pre-built starts, best KL first (the directory your jobs read, then its measured KL):" in flat
    )
    assert "- gptq: `baselines/1`, KL 0.02" in flat
    assert "- awq: `baselines/0`, KL 0.04" in flat
    assert "Baselines, by fallback order" not in flat
    assert "Your GPU jobs can read the original model and the starts (read-only)" in flat
    unread = _statement(starts=[("gptq", "baselines/0", None)])
    assert "- gptq: `baselines/0`\n" in unread  # not read yet: no number is invented
    plain = " ".join(_statement().split())
    assert "Baselines, by fallback order: `baselines/0`." in plain
    assert "pre-built starts" not in plain


def test_the_prior_section_without_starts_is_the_legacy_bytes():
    """Empty (or absent) starts changes no byte: the two lines keep the break after "read the"."""
    legacy = (
        "Original model: `m`. Baselines, by fallback order: `baselines/0`. Your GPU jobs can read the\n"
        "original model and the baselines at these paths (read-only); your shell cannot read their weights.\n"
    )
    assert legacy in _statement()
    assert _statement(starts=[]) == _statement()


def _measured(text: str) -> list[str]:
    """The measured figures in `text`."""
    return [m.group(0) for m in MEASURED.finditer(text)]


def test_the_shipped_task_statement_carries_no_measured_figure():
    """P3: the default TASK.md states what the tools do, never what a past search measured with them."""
    toolset, run = Qda(), Run(workdir=Path("/nonexistent"))
    text = prompt.task_statement(
        metric="kl",
        final_rules="Then the last submission that passed counts.",
        model="tiny/model",
        fmt="w4a16",
        n_cards=1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        deadline=datetime(2026, 1, 1, tzinfo=UTC),
        baselines=["baselines/0"],
        budget_bpw=4.4,
        format_menu=["bf16", "w8a16", "w4a16"],
        require_build_receipt=True,
        max_kl=1,
        gen_max_tokens=8192,
        eval_max_tokens=32768,
        commands=toolset.commands(run),
        subcommands=toolset.subcommands(run),
        tools=toolset.sections(run),
    )
    assert not _measured(text), _measured(text)


def test_the_shipped_skill_carries_no_measured_figure():
    """The skill ships with the package; a figure in it reaches every search.

    The experience manual (`references/manual/`) is model- and result-specific by construction and is
    being removed separately; this test covers the skill pages that remain.
    """
    root = Path(__file__).parents[1] / "src" / "qda_agent" / "wiki" / "skill"
    found = {
        str(path.relative_to(root)): _measured(path.read_text())
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in (".md", ".txt")
        and "references/manual/" not in str(path.relative_to(root))
        and _measured(path.read_text())
    }
    assert not found, found


#: A hand-off to our own reproduction material: a recipe path or identifier, or a link into the recipes tree.
RECIPE = re.compile(r"\brecipes\s*/|\brecipes-\d+\b")
#: A concrete quantization recipe's name (a model part followed by a stored format), which the search must not cite.
RECIPE_NAME = re.compile(r"\b[\w.]+-w\d+a\d+(?:-[a-z0-9]+)*\b")


def test_the_search_toolset_cites_no_recipe_or_calibration_provenance():
    """Item 9: the skill and TASK.md ship mechanisms, never our recipes or where their calibration came from.

    Method cards stay: they cite their publications and code, not a build of any model here.
    """
    source = Path(__file__).parents[1] / "src" / "qda_agent"
    skill = source / "wiki" / "skill"
    texts = {
        str(path.relative_to(source)): path.read_text()
        for path in sorted(skill.rglob("*"))
        if path.is_file() and path.suffix in (".md", ".txt", ".toml")
    }
    texts["wiki/cards.toml"] = (source / "wiki" / "cards.toml").read_text()
    texts["generated capability.md"] = wiki.capability()
    texts["generated stacking.md"] = wiki.stacking()
    texts["prompt/task.md"] = (source / "prompt" / "task.md").read_text()
    toolset, run = Qda(), Run(workdir=Path("/nonexistent"))
    texts["rendered TASK.md"] = prompt.task_statement(
        metric="kl",
        final_rules="Then the last submission that passed counts.",
        model="tiny/model",
        fmt="w4a16",
        n_cards=1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        deadline=datetime(2026, 1, 1, tzinfo=UTC),
        baselines=["baselines/0"],
        budget_bpw=4.4,
        format_menu=["bf16", "w8a16", "w4a16"],
        require_build_receipt=True,
        max_kl=1,
        gen_max_tokens=8192,
        eval_max_tokens=32768,
        commands=toolset.commands(run),
        subcommands=toolset.subcommands(run),
        tools=toolset.sections(run),
    )
    found = {}
    for where, text in texts.items():
        hits = RECIPE.findall(text) + RECIPE_NAME.findall(text)
        hits += [
            link
            for link in re.findall(r"https?://[^\s)>\]]+", text)
            if "recipe" in link or "huggingface.co" in link
        ]
        if hits:
            found[where] = hits
    assert not found, found
    # The mechanisms themselves stay: the card pages are part of what the search ships.
    assert list((skill / "quant-methods" / "references" / "methods").glob("*.md"))


def test_no_vendor_name_is_in_an_agent_readable_source():
    """The package source is readable by the agent (PYTHONPATH): it names no vendor or vendor file scheme."""
    packages = Path(__file__).parents[2]
    found = {}
    for package in packages.glob("*/src"):
        for path in sorted(package.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".md", ".txt", ".yaml"):
                if hit := VENDOR.search(path.read_text()):
                    found[str(path.relative_to(packages))] = hit.group(0)
    assert not found, found


def test_the_head_line_follows_the_engine_and_the_head_rule():
    """16-bit by default, an exception model's release rule otherwise, on either engine."""
    assert "lm_head stays 16-bit (bf16 or fp16)." in _statement()
    text = " ".join(_statement(engine="llama.cpp", format_menu=["Q4_K", "Q6_K"]).split())
    assert "lm_head stays" not in text
    assert "The output head (`output.weight`) stays 16-bit: store it BF16 or F16" in text
    text = " ".join(_statement(engine="llama.cpp", format_menu=["Q4_K", "Q6_K"], head="Q8_0").split())
    assert "keeps the original model's released type, Q8_0" in text
    text = " ".join(_statement(head="w4a16-sym-g64").split())
    assert "lm_head keeps the original model's released format, w4a16-sym-g64" in text


def test_the_instruction_tells_a_session_picked_up_mid_run_to_recover_first():
    """A sign-in refresh starts a new session and a CLI may compact its context, so every send names where
    the run's state is; `kl list` is asked for only when the run gives `kl`."""
    deadline = datetime(2026, 1, 1, tzinfo=UTC)
    assert "first read notes/ and run `kl list`" in prompt.goal("w4a16", deadline)
    without = prompt.goal("w4a16", deadline, ("qda", "qsched"))
    assert "first read notes/" in without and "`kl list`" not in without
