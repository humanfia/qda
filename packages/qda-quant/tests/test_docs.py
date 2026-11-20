"""The qda-quant pages of ``docs/`` work as written: doctests pass, configs validate, tables match the code.

Python examples are doctests. A ``yaml`` block is a build config (`BuildConfig`); an extension guide's
``toml`` block declares a ``qda.plugins`` hook. No example downloads anything or needs a GPU.
"""

from __future__ import annotations

import doctest
import re
import subprocess
import textwrap
import tomllib
from pathlib import Path

import pytest
import yaml
from qda_quant.config import BuildConfig

DOCS = Path(__file__).resolve().parents[3] / "docs"
#: Each page and whether its TOML examples declare registration hooks.
PAGES = {
    "design.md": False,
    "add-method.md": False,
    "extend.md": True,
    "add-architecture.md": False,
    "add-target.md": False,
    "recipe.md": False,
    "quickstart.md": False,
    "quantize.md": False,
}


def blocks(page: str, lang: str) -> list[str]:
    """The bodies of the fenced ``lang`` blocks of `page`."""
    found = re.findall(rf"^([ \t]*)```{lang}\n(.*?)^[ \t]*```", (DOCS / page).read_text(), re.S | re.M)
    return [textwrap.dedent(body) for _, body in found]


@pytest.mark.parametrize("page", [p for p in PAGES if ">>>" in (DOCS / p).read_text()])
def test_doctests_pass(page):
    failed, tried = doctest.testfile(
        str(DOCS / page), module_relative=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert tried > 0 and failed == 0


@pytest.mark.parametrize(
    ("page", "text"), [(p, t) for p in PAGES for t in blocks(p, "yaml")], ids=lambda x: str(x)[:20]
)
def test_yaml_blocks_are_build_configs(page, text):
    BuildConfig.from_dict(yaml.safe_load(text), name=page)


@pytest.mark.parametrize("page", [p for p, hooks in PAGES.items() if hooks])
def test_toml_blocks_declare_registration_hooks(page):
    texts = blocks(page, "toml")
    assert texts
    for text in texts:
        registries = tomllib.loads(text)["project"]["entry-points"]
        assert list(registries) == ["qda.plugins"]
        entries = registries["qda.plugins"]
        assert entries and all(
            re.fullmatch(r"\w+", name) and re.fullmatch(r"[\w.]+:\w+", value)
            for name, value in entries.items()
        )


def test_plugin_guide_links_each_external_example():
    """The extension guide names every supported example kind and links its source file."""
    text = (DOCS / "extend.md").read_text()
    assert text.startswith("# Extend QDA\n")
    assert "uv pip install -e examples/plugins" in text
    kinds = {
        "Quantization step",
        "Quantization grid",
        "Architecture adapter",
        "KL scorer",
        "Agent backend",
        "Tool component",
    }
    named = {h.removeprefix("## ").lower() for h in re.findall(r"^## .+$", text, re.M)}
    named |= {k.lower() for k in re.findall(r"^\| ([A-Za-z][A-Za-z ]+) \|", text, re.M)}  # kinds-table rows
    assert {k.lower() for k in kinds} <= named
    for module in ("steps", "grids", "architecture", "scorers", "backends", "components"):
        assert f"examples/plugins/src/qda_example_plugins/{module}.py" in text


#: The pages that teach an extension; their Python blocks (plain or doctest) are the code a reader copies.
EXTENSION_PAGES = ("extend.md", "add-method.md", "add-architecture.md", "add-target.md", "add-scorer.md")


@pytest.mark.parametrize("page", EXTENSION_PAGES)
def test_extension_examples_name_what_exists(page):
    """Each Python block parses, and every ``qda_quant`` / ``qda_kl`` name it imports or registers exists."""
    import ast
    import importlib

    parser = doctest.DocTestParser()
    sources = [
        "".join(ex.source for ex in parser.get_examples(t)) if t.lstrip().startswith(">>>") else t
        for t in blocks(page, "python")
    ]
    assert sources, f"{page} has no python block: the test would pass without checking anything"
    names = []
    for text in sources:
        names += [
            (node.module, alias.name)
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in ("qda_quant", "qda_kl")
            for alias in node.names
        ]
        names += [tuple(v.split(":")) for v in re.findall(r'"(qda_(?:quant|kl)[\w.]*:\w+)"', text)]
    assert names, f"{page}: no qda_quant / qda_kl import or registration in its python blocks"
    missing = [f"{m}:{n}" for m, n in names if not hasattr(importlib.import_module(m), n)]
    assert not missing, missing


def test_plugin_guide_shows_the_example_package():
    """extend.md shows the example's ``pyproject.toml`` as it is, and its file list names files that exist."""
    example = DOCS.parent / "examples" / "plugins"
    assert (example / "pyproject.toml").read_text() in blocks("extend.md", "toml")
    ((tree,),) = [[b for b in blocks("extend.md", "text") if b.startswith("examples/plugins/\n")]]
    files = [line.split()[0] for line in tree.splitlines()[1:]]
    assert files and all((example / f).is_file() for f in files)
    assert "extend.md#a-complete-package" in (DOCS / "add-method.md").read_text()


def test_user_pages_name_no_kernel_pin():
    """The quickstart, published and reproduce pages pin no kernel: the serve line a build or a card prints does.

    The kernel differs per card, so a page that names one goes stale.
    """
    for page in ("quickstart.md", "published.md", "reproduce.md"):
        text = (DOCS / page).read_text()
        assert not re.search(
            r"--linear-backend|linear_backend|--moe-backend|Marlin", text, re.I
        ), page


def test_quickstart_reprints_the_build_serve_line():
    """The quickstart's serve block calls the function `qda build` prints its serve line with."""
    from qda_quant.output import serve_command

    (line,) = [x for x in blocks("quickstart.md", "bash")[2].splitlines() if x.strip()]
    called = re.search(
        r"from qda_quant\.output\.serve_command import (\w+); print\(\1\(\"([^\"]+)\"\)\)", line
    )
    assert called and callable(getattr(serve_command, called[1]))
    assert called[2] == yaml.safe_load((DOCS.parent / "configs" / "quickstart.yaml").read_text())["out"]


#: The agent's methods skill (package data whose index links to pages generated at install).
SKILL = "packages/qda-agent/src/qda_agent/wiki/skill/"


def markdown_files() -> list[Path]:
    """The repository's tracked Markdown files, except the agent's methods skill.

    The skill (package data) links to two pages that are generated when it is installed.
    """
    root = DOCS.parent
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=root, capture_output=True, text=True, check=True)
    return [root / p for p in out.stdout.split() if not p.startswith(SKILL)]


def anchors(page: Path) -> set[str]:
    """GitHub's heading anchors of `page`: lower case, punctuation dropped, spaces as dashes."""
    text = re.sub(r"^```.*?^```", "", page.read_text(), flags=re.S | re.M)
    heads = re.findall(r"^#+ (.+)$", text, re.M)
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in heads}


#: A link to this repository's own files on GitHub; the docs use it for files outside docs/.
REPO_URL = re.compile(r"https://github\.com/(?:WhiteCmile|Radioheading|humanfia)/qda/(?:blob|tree)/(?:main|HEAD)/")


def test_markdown_links_resolve():
    """Every relative link, and every link to a file of this repository on GitHub, names an existing file, and an
    existing heading if any."""
    broken = []
    for page in markdown_files():
        text = re.sub(r"^```.*?^```", "", page.read_text(), flags=re.S | re.M)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            in_repo = REPO_URL.match(target)
            if not in_repo and re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = (target[in_repo.end() :] if in_repo else target).partition("#")
            dest = (DOCS.parent / path) if in_repo else (page.parent / path).resolve() if path else page
            if not dest.exists():
                broken.append(f"{page.relative_to(DOCS.parent)}: {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                broken.append(f"{page.relative_to(DOCS.parent)}: {target} (no such heading)")
    assert not broken, "\n".join(broken)


def toctree_entries() -> list[str]:
    """The pages the site's nav lists: the entries of every ``toctree`` in ``docs/index.md``."""
    text = (DOCS / "index.md").read_text()
    entries = []
    for body in re.findall(r"^```\{toctree\}\n(.*?)^```", text, re.S | re.M):
        entries += [x.strip() for x in body.splitlines() if x.strip() and not x.startswith(":")]
    return entries


def test_every_page_is_in_the_nav():
    """Each page under ``docs/`` is in the nav exactly once, and every nav entry names a page."""
    pages = {str(p.relative_to(DOCS).with_suffix("")) for p in DOCS.rglob("*.md")} - {"index"}
    entries = toctree_entries()
    assert sorted(entries) == sorted(pages), "nav and docs/ differ"


def test_the_exit_status_table_is_the_contract():
    """Troubleshooting's table is where a user reads the exit statuses; it is `qda_quant.exit`, code for code."""
    from qda_quant import exit as status

    rows = dict(re.findall(r"^\| (\d+) \| (.+) \|$", (DOCS / "troubleshooting.md").read_text(), re.M))
    assert {int(code) for code in rows} == {
        status.OK,
        status.ERROR,
        status.USAGE,
        status.ENVIRONMENT,
        status.INTERRUPTED,
    }
    assert all(text.strip() for text in rows.values())  # and each row says what its code means


def test_readme_commands_are_the_install_pages():
    """The README's command blocks are copies: each command is on the install page, the quickstart or the search
    page, unchanged. The home page has no command block; it links the install page."""

    def commands(text: str) -> list[str]:
        blocks = re.findall(r"^```bash\n(.*?)^```", text, re.S | re.M)
        return [re.sub(r"\s+#.*$", "", x) for b in blocks for x in b.splitlines() if x.strip()]

    known = {
        x for p in ("install.md", "quickstart.md", "search.md") for x in commands((DOCS / p).read_text())
    }
    lines = commands((DOCS.parent / "README.md").read_text())
    missing = [x for x in lines if x not in known]
    assert lines and not missing, missing
    assert not commands((DOCS / "index.md").read_text())


def test_quickstart_config_is_the_documented_default():
    """``configs/quickstart.yaml`` validates and is the recipe the quickstart page describes.

    The default recipe: gptq (actorder static, damp 0.01, clip hmse), ``w4a16-asym-g128``, the pinned public chat set,
    ``residency: stream``, rotate ``when: accepted`` (skipped on Qwen3-0.6B, which ties its embeddings). Planning the
    shipped keys is `test_config.test_quickstart_config_plans`; this checks the page names the file.
    """
    path = DOCS.parent / "configs" / "quickstart.yaml"
    cfg = yaml.safe_load(path.read_text())
    BuildConfig.from_file(path)
    page = " ".join((DOCS / "quickstart.md").read_text().split())
    assert "`configs/quickstart.yaml`" in page and "125 samples of 2,048 tokens" in page
    assert cfg["format"] == "w4a16-asym-g128" and cfg["residency"] == "stream"
    assert cfg["steps"] == [
        {"rotate": {"residual": True, "when": "accepted"}},
        {"gptq": {"actorder": "static", "damp": 0.01, "clip": "hmse"}},
    ]
    hf = cfg["calib"]["hf"]
    assert hf["dataset"] == "mlabonne/open-perfectblend" and re.fullmatch(r"[0-9a-f]{40}", hf["revision"])
    assert (hf["rows"], hf["samples"], hf["seqlen"]) == (1024, 125, 2048)


def test_the_w4a4_facts_are_the_formats_page_and_the_other_pages_link_it():
    """formats.md names W4A4's cards and head lease; quantize, search and their reference link there."""
    formats = (DOCS / "formats.md").read_text()
    assert "## Head lease" in formats and "B200 (sm100)" in formats and "GB300 (sm103)" in formats
    for page in ("quantize.md", "search.md", "search-config.md"):
        assert "formats.md#head-lease" in (DOCS / page).read_text(), page


def test_install_activates_a_venv_and_your_model_names_the_gated_model_token():
    """The README's install activates the venv before the first command, and the quantize page names a gated model's
    token."""
    readme = (DOCS.parent / "README.md").read_text()
    install = re.search(r"## (?:Install|Getting Started).*?^```bash\n(.*?)^```", readme, re.S | re.M)
    assert install is not None and ". .venv/bin/activate" in install.group(1)
    page = (DOCS / "quantize.md").read_text()
    assert "gated" in page and "HF_TOKEN" in page and "hf auth login" in page


def test_install_lines_include_the_gguf_extra():
    """Every install line that fetches qda-quant installs the ``gguf`` extra, so the GGUF path runs as written.

    The extra is optional (`packages/qda-quant/pyproject.toml`): without it a GGUF build refuses by name.
    """
    pyproject = tomllib.loads((DOCS.parent / "packages/qda-quant/pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    assert {"hf", "gguf"} <= set(extras)
    line = '"./packages/qda-quant[hf,gguf]"'
    readme = (DOCS.parent / "README.md").read_text()
    assert line in readme
    assert line in (DOCS / "install.md").read_text()
    assert '"./packages/qda-quant[hf,gguf]"' in blocks("formats.md", "bash")[0]


def test_install_lines_seed_the_venv_with_pip():
    """Every install line makes its venv with ``--seed``, so pip is there for ``vllm download-kernels``.

    A bare ``uv venv`` has no pip module; the prebuilt-kernel step resolves its wheels with
    ``python -m pip download`` and fails with ``No module named pip``.
    """
    lines = [
        line
        for page in (DOCS.parent / "README.md", DOCS / "install.md")
        for body in re.findall(r"^```bash\n(.*?)^```", page.read_text(), re.S | re.M)
        for line in body.splitlines()
        if line.startswith("uv venv ")
    ]
    assert lines and all("--seed" in line for line in lines), lines
