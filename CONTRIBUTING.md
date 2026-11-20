# Contributing to QDA

After this page you can set up the tree, run the checks CI runs, and send a change that passes review. Read
[design](docs/design.md) before your first change.

## Setup

The dev environment has the three packages, their test extras and the `dev` group (pyright), without vLLM:

```sh
uv sync --no-default-groups --all-packages --extra dev --group dev
uv run --no-sync pytest -q
```

Pass `--no-sync` to every `uv run`; a plain `uv run` installs the default group (vLLM) again. The default run builds
tiny random models from `transformers` configs, so it downloads nothing and needs no GPU.

## Checks

CI (`.github/workflows/ci.yml`) runs these on every push and pull request, on Ubuntu with no GPU. Run them before
you push:

| check | command |
|---|---|
| lint | `uvx ruff@0.17.3 check .` (rules E, F, I, UP, B; `E501` off) |
| types | `uv run --no-sync pyright` (basic mode) |
| package layers | `python tools/check_package_layers.py` |
| public tree | `python tools/check_public_tree.py` |
| tests | `uv run --no-sync pytest -q`, on Python 3.12; `qda-quant` and `qda-kl` also on 3.11 and 3.13; `packages/*/tests/test_frozen.py` pins cache keys, registry names and the CLI |
| docs | `uv sync --no-default-groups --group docs && uv run --no-sync sphinx-build -W docs site` (open `site/index.html`) |

Tests marked `gpu` need an NVIDIA GPU and vLLM, and do not run by default: `pytest -m gpu`. Tests on real models
take local model directories and skip without them; name the test directory before the option, or pytest rejects it:
`pytest -m gpu packages/qda-quant/tests --model-dir qwen3-8b=/models/Qwen3-8B` (the names are in
`packages/qda-quant/tests/conftest.py`). The sandbox tests start bubblewrap, which needs unprivileged user
namespaces; on Ubuntu 24.04 run `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`.
Where bwrap cannot start (user namespaces off, as on many clusters), a test that starts it and fails with
bwrap's own refusal is reported as skipped with that error; any other failure stays a failure. Run the tests with `TMPDIR=/tmp`: a long `TMPDIR` or one under a group-writable
directory skips the tests whose sockets or custody directories it breaks.

The docs are tested too (`packages/*/tests/test_docs.py`): `>>>` examples are doctests, `yaml` blocks validate as
the config the page is about, `qda-kl` and `qda-agent` lines name real commands and options, plugin tables match
the registries, every relative Markdown link resolves, and every page is in a `toctree` of `docs/index.md` (the site's nav). When
you move a page, move its entry there too.

## Rules

| rule | what it means |
|---|---|
| one source of truth | anything that changes a build, a score or a search is a config key and is recorded in `resolved.yaml`; no environment-variable switches, no hidden defaults, no second copy of a setting |
| plugins only where users extend | methods, formats, architectures, eval suites, datasets and agent backends enter through a named registry and are chosen by name in the config; everything else is plain code |
| smallest change | no speculative options, shims or drive-by refactors; delete before you add |
| layers, five concepts | [design](docs/design.md) |
| methods | a `Step` subclass with pydantic settings; it declares `needs` and never registers hooks or loops over calibration data |
| fast paths | every fast path keeps its slow reference path, and a test checks that the two agree |
| no model-name cases | outside `qda_quant/arch/adapters/` |
| tests | commit a test only when it checks behaviour a reader could get wrong and a plausible change would break; otherwise run the check once and put the command and output in the pull request |
| claims | a speed or quality claim names the command, the numbers and where the log is |

## Style

- Lines up to 110 characters.
- Docstrings on the public API only, 1 to 3 lines (Google style): what it does and why, not how.
- Comments say why. Say a thing once, where it is true.
- Index a dict this module wrote (`d[key]`, not `d.get(key, default)`); refuse unknown modes in one place.
- One helper instead of a pasted rule; no function-level re-import; no import of another module's `_private` name.
- An early return instead of an `assert` that only narrows a type.
- A cleanup PR touches one module, changes no behaviour, and removes more lines than it adds.

## Pull requests

- Small commits with Conventional Commit messages (`fix:`, `feat:`, `docs:`, `refactor:`).
- Stage files by name. Never revert, stash or check out files you did not write.
- Adding a method: [adding a method](docs/add-method.md). Adding a model family:
  [adding an architecture](docs/add-architecture.md). Adding a GPU or kernel row:
  [adding a target](docs/add-target.md). Other plugins: [extending QDA](docs/extend.md).

## Releasing

The three packages share one version and are released together from `main`, as a commit and an annotated tag.

1. `main` is green in CI, and `pytest -m gpu` passes on a GPU machine.
2. The README quick start runs from a fresh clone with empty `UV_CACHE_DIR` and `HF_HOME`, and
   `qda build OUT/resolved.yaml --set out=again/` writes the same bytes.
3. Every number in the README and `docs/` comes from a file in the repository.
4. Bump `version` in the three `pyproject.toml` files, `__version__` in each `__init__.py`, and `CITATION.cff`; then
   `uv lock`.
5. Add a `CHANGELOG.md` section on what changed for users, naming anything deprecated and the release that removes it.
6. Commit `release: vX.Y.Z`, tag `vX.Y.Z`, push both.

A schema change keeps older `resolved.yaml` files loading. A deprecated command form works for at least one more
minor release.
