"""Tests for the generated-docs writer."""

import pytest

from tools.gen_docs import _with_block, support_matrix


def test_block_is_replaced_and_rest_kept() -> None:
    text = "head\n<!-- BEGIN generated: m -->\nold\n<!-- END generated: m -->\ntail\n"
    out = _with_block(text, "m", "new", "f.md")
    assert out.startswith("head\n<!-- BEGIN generated: m -->\n") and out.endswith(
        "<!-- END generated: m -->\ntail\n"
    )
    assert "new" in out and "old" not in out
    assert _with_block(out, "m", "new", "f.md") == out  # idempotent: --check passes right after a write


def test_missing_block_names_the_file() -> None:
    with pytest.raises(SystemExit, match="f.md: no <!-- BEGIN generated: m -->"):
        _with_block("no markers\n", "m", "x", "f.md")


def test_matrix_rows_are_cards_and_cells_come_from_the_tables() -> None:
    from qda_quant.format import PRESETS
    from qda_quant.target import _PENDING, TARGETS, Target

    from tools.gen_docs import CARDS

    page = support_matrix()
    for label, name in CARDS:
        assert f"| {label} |" in page
        assert name in _PENDING or Target.parse(name).arch in TARGETS  # every row names a card a table knows
    assert all(name.upper() in page for name in PRESETS)
    rows = [line for line in page.splitlines() if line.startswith("| ") and not line.startswith("| GPU |")]
    assert [row.split(" | ")[0].removeprefix("| ") for row in rows] == [label for label, _ in CARDS]


def test_matrix_lists_only_rows_a_card_has_run() -> None:
    """The third mark is gone: an unobserved row stops the generator instead of printing another mark."""
    from qda_quant.format import PRESETS
    from qda_quant.step import Refused
    from qda_quant.target import _PENDING, Target

    from tools.gen_docs import CARDS, FORMAT_ORDER, KINDS

    assert "☑" not in support_matrix()
    for label, name in CARDS:
        if name in _PENDING:
            continue
        target = Target.parse(name)
        for fmt in FORMAT_ORDER:
            for kind in KINDS:
                try:
                    plan = target.plan(PRESETS[fmt], kind)
                except Refused:
                    continue
                assert not any("unverified" in n for n in plan.notes), (label, fmt, kind)


def test_a_refused_cell_quotes_the_refusal_a_build_prints() -> None:
    """The refusal quoted under the matrix is the target table's own text, not the generator's rewording of it."""
    from qda_quant.format import PRESETS
    from qda_quant.step import Refused
    from qda_quant.target import Target

    with pytest.raises(Refused) as refused:
        Target.parse("l40s").plan(PRESETS["w4a8"], "dense")
    reason = str(refused.value).split(": ", 1)[-1]
    matrix = support_matrix()
    assert f'stops with: "{reason}"' in matrix
    assert "❌ Hopper only" in matrix  # the short cell the quoted line explains
    assert matrix.count("[^") <= 2 and "v0" not in matrix


def test_cli_reference_lists_subcommands_not_option_choices() -> None:
    """Sections come from each parser's subparsers: `qda eval --format {md,html}` is not a command."""
    from tools.gen_docs import cli_reference

    page = cli_reference()
    assert "### `qda-kl ref`" in page and "### `qda doctor`" in page and "#### `qda fisher map`" in page
    assert "### `qda-kl corpus`" not in page and "### `qda-agent evaluate`" not in page  # hidden
    for fake in ("`qda eval md`", "`qda-agent prepare w4a16`", "`qda-agent cand0 gptq`"):
        assert fake not in page


def test_cli_reference_states_what_0_1_keeps_stable_before_the_first_command() -> None:
    """The stability promise heads the page, above every command's help, and points at the changelog."""
    from tools.gen_docs import _stability, cli_reference

    STABILITY = _stability()
    page = cli_reference()
    assert 0 <= page.index(STABILITY) < page.index("## `qda`")
    assert "CHANGELOG.md" in STABILITY and "0.1.x" in STABILITY
