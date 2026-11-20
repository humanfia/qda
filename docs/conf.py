"""Sphinx configuration of the QDA docs: MyST pages, furo theme, API read statically from the source.

Build: uv sync --no-default-groups --group docs && uv run --no-sync sphinx-build -W docs site
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

project = "QDA"
author = "QDA contributors"
html_title = "QDA"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}  # .rst: autoapi's autosummary needs it
root_doc = "index"
exclude_patterns = ["_build"]

# MyST: plain Markdown; GitHub-style heading anchors, so `page.md#heading` links work the same on GitHub and on
# the site.
myst_enable_extensions = ["attrs_inline", "html_image"]
myst_heading_anchors = 4

# API: sphinx-autoapi reads the packages' source with astroid and imports nothing, so the docs build needs neither
# torch nor the packages installed. Only the classes api.md names are rendered.
autoapi_dirs = [
    str(ROOT / "packages" / p / "src" / m)
    for p, m in (("qda-quant", "qda_quant"), ("qda-kl", "qda_kl"), ("qda-agent", "qda_agent"))
]
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autoapi_options = ["members", "show-inheritance"]
autoapi_member_order = "bysource"
autoapi_python_class_content = "class"
add_module_names = False
toc_object_entries = False  # the right-hand list shows headings, not every member
# autoapi skips an assignment to several names at once (PEP 257; `Q4K, Q5K, Q6K = ...` in target.py,
# `Q3_K, IQ4_NL, IQ3_S = ...` in gguf/rule.py, `IN, DOWN = ...` in arch/adapters/fused_experts.py), so the modules
# importing those names warn. None of them is on the API page; a class api.md names that cannot be found still
# fails the build.
suppress_warnings = ["autoapi.python_import_resolution"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# A docstring's "Attributes:" list becomes fields of its class, so it does not describe the members `:members:`
# lists a second time; "Class attributes:" (qda_quant.step.Step) is the same section.
napoleon_use_ivar = True
napoleon_custom_sections = [("Class attributes", "params_style")]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["qda.css"]
html_js_files = ["light.js"]
html_logo = "assets/logo.png"
html_favicon = "assets/logo.png"
html_show_sphinx = False
html_copy_source = False
html_show_sourcelink = False
pygments_style = "default"
_COLORS = {
    "color-brand-primary": "#1A1A1A",
    "color-brand-content": "#A31F34",
    "color-link": "#A31F34",
    "color-link--hover": "#A31F34",
    "color-link-underline": "transparent",
    "color-link-underline--hover": "#A31F34",
    "color-sidebar-background": "#FFFFFF",
    "color-sidebar-background-border": "#E3E3E3",
    "color-sidebar-link-text--top-level": "#1A1A1A",
    "color-table-border": "#E3E3E3",
    "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif",
    "font-stack--monospace": "ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace",
}
html_theme_options = {
    "light_css_variables": _COLORS,
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/humanfia/qda/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view"],
}
