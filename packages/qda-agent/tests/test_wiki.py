"""The quant-methods skill: complete for the installed library, and free of results (the leak lint)."""

import re
from pathlib import Path

import pytest
from qda_agent import wiki

SKILL = wiki.skill_source()
PAGES = sorted(SKILL.rglob("*.md"))

#: Card names a page may link to (`references/methods/<card>.md`).
CARD_REF = re.compile(r"\]\((?:references/)?methods/([a-z0-9-]+)\.md\)")


def test_every_builtin_step_has_a_cards_toml_row():
    """A new qda_quant step needs its row in cards.toml (a card, or [no_card])."""
    assert wiki.gaps() == ([], [])
    page = wiki.capability()
    for step in wiki.steps():
        assert f"`{step}`" in page, step


def test_a_plugin_step_without_a_row_is_listed_and_warned(caplog):
    class New:
        """Brand-new step from another package."""

        kind = "layer"

    page = wiki.capability({**wiki.steps(), "brand_new": New})
    assert "- `brand_new`: Brand-new step from another package" in page
    assert "brand_new" in caplog.text


def test_cards_toml_names_every_card_and_only_cards():
    cards = {p.stem for p in (SKILL / "references" / "methods").glob("*.md")}
    assert set(wiki.load_cards()["cards"]) == cards


def test_stacking_lists_every_step_by_kind():
    page = wiki.stacking()
    for step, cls in wiki.steps().items():
        assert f"| `{step}` | {cls.kind} |" in page


def test_install_writes_the_generated_pages(tmp_path):
    dst = wiki.install(tmp_path / "quant-methods")
    assert (dst / "SKILL.md").read_text().startswith("---\nname: quant-methods\n")
    assert (dst / "references" / "capability.md").is_file() and (dst / "references" / "stacking.md").is_file()


def test_no_page_links_a_missing_card():
    cards = {p.stem for p in (SKILL / "references" / "methods").glob("*.md")}
    for page in PAGES:
        for ref in CARD_REF.findall(page.read_text()):
            assert ref in cards, (page.name, ref)
    text = (SKILL / "SKILL.md").read_text()
    assert "references/capability.md" in text and "references/stacking.md" in text


# ------------------------------------------------------------------------------------------------ index guard
# `SKILL.md` is an index over the pages, not authored text: the research flow generated it with
# `wiki/tools/index.py --check`, which enforced its two tables, the counts in its first paragraph, a
# 150-line / 16,000-byte budget and no dangling card names. qda ships the output without that generator
# (the prose around the tables is qda's, the tables and counts are not), so the same checks live here.

SKILL_MD = SKILL / "SKILL.md"
MAX_LINES, BUDGET = 150, 16_000
SUMMARY_CHARS, CELL_CHARS = 88, 120
GROUPS = (
    ("rot", "rotation / permutation"),
    ("obj", "objective / loss"),
    ("grid", "grid, quantizer, format"),
    ("lptq", "learned / trained"),
    ("cond", "conditioner, MoE, assembly"),
)
PHENOMENA = "## Phenomenon -> how to see it on your model -> mechanisms"
CARDS = "## Cards"
#: A card name in prose (`rot-hadamard-r1`), not only in a link.
CARD_NAME = re.compile(r"(?<![\w-])((?:rot|obj|grid|lptq|cond|calib|kv)-[a-z0-9]+(?:-[a-z0-9]+)*)\b")


def _pages(kind: str) -> list[Path]:
    """The method or phenomenon pages, in the order the generator sorts them."""
    return sorted((SKILL / "references" / kind).glob("*.md"))


def _field(text: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def _first_clause(s: str, limit: int) -> str:
    """The clause before the field's first `;`/`.`, truncated as the flow's generator truncated it."""
    s = re.split(r"(?<=[a-z)])[;.](?:\s|$)", s)[0].strip().rstrip(".;,")
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "..."


def _index_tables() -> dict[str, list[str]]:
    """The two `SKILL.md` tables as the flow's generator renders them from the pages."""
    cards: dict[str, tuple[str, bool]] = {}
    for page in _pages("methods"):
        text = page.read_text()
        cards[page.stem] = (
            _first_clause(_field(text, "Addresses").split("Measured as:")[0], SUMMARY_CHARS),
            _field(text, "Inference overhead").lower().startswith("none"),
        )
    phenomena: list[str] = []
    for page in _pages("phenomena"):
        text = page.read_text()
        named = [c.strip() for c in _field(text, "Cards").split(",") if c.strip()]
        links = " ".join(f"[{c}](references/methods/{c}.md)" for c in named)
        phenomena.append(
            f"| [{page.stem}](references/phenomena/{page.stem}.md): "
            f"{_first_clause(_field(text, 'Observable'), CELL_CHARS)} | "
            f"{_first_clause(_field(text, 'How to see it on your own model'), CELL_CHARS)} | {links} |"
        )
    kind = dict(GROUPS)
    order = {prefix: n for n, (prefix, _) in enumerate(GROUPS)}
    card_rows = [
        f"| [{card}](references/methods/{card}.md) | {kind.get(card.split('-')[0], 'other')} | {addresses} | "
        f"{'' if free else '*'} |"
        for card, (addresses, free) in sorted(
            cards.items(), key=lambda item: (order.get(item[0].split("-")[0], len(order)), item[0])
        )
    ]
    return {
        PHENOMENA: ["| phenomenon | how to see it on your own model | cards |", "|---|---|---|", *phenomena],
        CARDS: ["| card | kind | addresses | |", "|---|---|---|---|", *card_rows],
    }


def _table(heading: str) -> list[str]:
    """The contiguous table lines under `heading` in `SKILL.md`."""
    text = SKILL_MD.read_text()
    lines = text[text.index(heading) + len(heading) :].splitlines()
    first = next(n for n, line in enumerate(lines) if line.startswith("|"))
    end = next(n for n in range(first, len(lines)) if not lines[n].startswith("|"))
    return lines[first:end]


def test_skill_index_matches_the_pages():
    """A card or phenomenon edit updates SKILL.md's tables and counts; nothing regenerates them."""
    for heading, expected in _index_tables().items():
        assert _table(heading) == expected, f"{heading}: re-render its table from the pages"
    counts = f"{len(_pages('methods'))} mechanism cards, {len(_pages('phenomena'))} phenomena"
    assert counts in SKILL_MD.read_text(), f"the first paragraph no longer says {counts!r}"


def test_skill_index_stays_within_its_budget():
    """An agent loads the whole skill when it applies, so its index keeps the flow's line and byte budget."""
    text = SKILL_MD.read_text()
    assert text.count("\n") <= MAX_LINES and len(text.encode()) <= BUDGET, (
        f"{text.count(chr(10))} lines / {len(text.encode())} B, budget {MAX_LINES} / {BUDGET}"
    )


def test_no_page_names_a_missing_card():
    """A page that names a card in prose, not only in a link, names one that exists."""
    have = {page.stem for page in _pages("methods")}
    bad = [
        f"{page.relative_to(SKILL)}: names {name!r}, which has no card"
        for page in sorted((SKILL / "references").rglob("*.md"))
        for name in sorted(set(CARD_NAME.findall(page.read_text())))
        if name not in have
    ]
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------------- leak lint
# A card describes a mechanism and how to measure on any model. It carries no result of a model, no score, no
# benchmark or test-set name, no model or run name, and nothing of the team that wrote it. A comparison between
# builds ("ends above its start", "beats", "better than its") is a result too. Ported from the research flow's
# wiki lint (its line rules; a `Source:` line may name a paper's title).

B, E = r"(?<![\w-])", r"(?![\w-])"
RULES = {
    "benchmark": re.compile(
        B + r"(?:aime\w*|hmmt\w*|gpqa(?:[-_ ]?diamond)?\w*|supergpqa|lcb\w*|livecodebench\w*|ifbench\w*|"
        r"ifeval\w*|mmlu(?:[-_ ]?pro)?\w*|math-?500|(?-i:MATH)|gsm8k|humaneval\w*|mbpp\w*|bbh|hellaswag|"
        r"winogrande|truthfulqa|mt-?bench|alpacaeval|scicode|olympiad\w*|amc\d*|aider|swe-?bench\w*|"
        r"codeforces|c-?eval|cmmlu|wikitext\w*|arc-?[ce])" + E,
        re.I,
    ),
    "eval-word": re.compile(
        r"\bbenchmark\w*\b|\bleaderboard\b|\bscorecard\b|\baccurac\w+\b|\bpass@\w+|\bheld[- ]?out\b|"
        r"\btest[- ](?:set|sets|items?|split)\b|\bmain[- ]table\b",
        re.I,
    ),
    "model-name": re.compile(
        B + r"(?:qwen\w*|llama(?!\.cpp)\w*|gemma\w*|deepseek\w*|mistral\w*|mixtral\w*|phi-?\d\w*|olmo\w*|"
        r"r1-distill\w*|gpt-?[0-9o]\w*|claude\w*|kimi\w*|glm-?\d\w*|internlm\w*|minimax\w*)"
        + E
        + r"|\b\d+(?:\.\d+)?B(?:-A\d+(?:\.\d+)?B)?\b",
        re.I,
    ),
    "result-number": re.compile(
        r"\d\s*%|\d\s*per\s?cent\b|\b\d+(?:\.\d+)?\s*(?:pp|percentage[- ]points?)\b|\bnats?\b", re.I
    ),
    "kl-result": re.compile(
        r"\bKL\b[^.;]{0,40}?(?<![\w.])(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)|\bperplexit\w+\b|\bppl\b", re.I
    ),
    # A comparison between builds is a result, even worded without a number. The referent keeps `its`/`their`,
    # so a mechanism sentence such as "ends below round-to-nearest" is not a hit.
    "outcome": re.compile(
        r"\b(?:ends?|lands?|finishes?)\s+(?:above|below)\s+(?:its|their)\b|"
        r"\bbeats?\b|\b(?:better|worse)\s+than\s+(?:its|their)\b",
        re.I,
    ),
    "voice": re.compile(r"\b(?:our|we|us)\b", re.I),
    "run-name": re.compile(
        r"\bround\s+[a-z]?\d+\b|\b(?:tier|holder)\s*\d\b|\bthis round\b|\bour runs?\b", re.I
    ),
    "abs-path": re.compile(r"(?<![\w.)\]}>:/~-])/[\w.-]+/|~/"),
}
URL = re.compile(r"https?://\S+")


def lint(text: str) -> list[tuple[str, str]]:
    """`(rule, match)` for every rule a page's lines break."""
    hits = []
    for line in text.splitlines():
        body = URL.sub(" ", line)
        if line.startswith("Source:"):
            body = re.sub(r'"[^"]*"', " ", body)  # a paper's title may name a model
        hits += [(rule, m.group(0)) for rule, pat in RULES.items() if (m := pat.search(body))]
    return hits


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(SKILL)))
def test_page_is_clean(page):
    assert lint(page.read_text()) == []


def test_the_shipped_skill_has_no_experience_manual():
    """The skill carries mechanism cards; no page of the removed experience manual ships with it."""
    assert not (SKILL / "references" / "manual").exists()


def test_generated_pages_are_clean():
    assert lint(wiki.capability()) == [] and lint(wiki.stacking()) == []


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("On GSM8K the method gains.", "benchmark"),
        ("Reported on MATH.", "benchmark"),
        ("The MMLU-Pro split is excluded.", "benchmark"),
        ("GPQA-Diamond is dropped.", "benchmark"),
        ("the build reaches a high accuracy", "eval-word"),
        ("Measured on Qwen3-8B.", "model-name"),
        ("It cuts the error by 12%.", "result-number"),
        ("KL drops to 0.031 after it", "kl-result"),
        ("KL = 2 after the fit", "kl-result"),
        ("The build ends above its start.", "outcome"),
        ("It beats its GPTQ start.", "outcome"),
        ("It is better than its start.", "outcome"),
        ("our measurement shows it", "voice"),
        ("in round a0 it drifts", "run-name"),
        ("see /e/scratch/x/notes", "abs-path"),
    ],
)
def test_the_lint_fires(line, rule):
    assert rule in {r for r, _ in lint(line)}


def test_a_clean_mechanism_line_passes():
    """Numeric recipe knobs and a plain comparison of the mechanism are not results."""
    assert (
        lint("Group 64, damp 0.003 and 2,048 tokens; a larger share is not better than the original.") == []
    )


def test_a_source_title_may_name_a_model():
    assert lint('Source: "Quantizing Llama Models Better", https://arxiv.org/abs/0000.00000') == []


def test_skill_dir_names(tmp_path):
    """Both CLIs' skill dirs get the same tree."""
    from qda_agent.components import SKILL_DIRS, Run, Wiki

    Wiki().install(Run(workdir=tmp_path / "work", overlay=tmp_path / "ov"))
    ov = tmp_path / "ov"
    trees = [sorted(p.relative_to(ov / d) for p in (ov / d).rglob("*")) for d in SKILL_DIRS]
    assert trees[0] == trees[1] and Path("quant-methods/SKILL.md") in trees[0]
