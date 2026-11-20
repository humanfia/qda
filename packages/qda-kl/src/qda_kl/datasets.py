"""Datasets: the plugins behind the data the final judging reads.

A config names a dataset by key, with its settings, or gives a plain path (the ``file`` entry)::

    judge:
      kl_ref: refs/q8b                                  # a plain path
      gencheck_prompts: {file: {path: hidden.jsonl}}    # the same thing, spelled out
    suites:
      - lm_eval: {tasks: [my_task], datasets: [{lm_eval_builtin: {}}, {my_tasks: {version: 2}}]}

One registry serves the three uses because each only needs a local file or directory, which is all
:meth:`Dataset.local` gives: a KL reference directory (``qda-kl ref``), a long-generation prompt JSONL of
``{id, domain, prompt}`` rows, or a directory of lm-eval task YAMLs. A dataset that holds only one of these says so in
:attr:`Dataset.uses`, so a wrong pairing is refused when the config loads.

Rows that are split between calibration and evaluation are a corpus (:mod:`qda_quant.intake.corpus`), which the ``kl`` suite
reads; a reference directory or a task directory has no rows to split.

The built-ins below are an explicit table; a third-party package adds its own from a ``qda.plugins``
registration hook (:func:`qda_quant.registry.load_plugins`). ``qda.datasets`` stays the group name a
checkpoint's provenance record names a dataset under.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePath
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError
from qda_quant.recipe import error_lines
from qda_quant.registry import Registry
from qda_quant.step import Refused

#: Datasets by key; the provenance group name is ``qda.datasets``.
DATASETS = Registry(
    "datasets",
    {
        "file": "qda_kl.datasets:FileDataset",
        "gencheck_builtin": "qda_kl.datasets:GencheckBuiltin",
        "lm_eval_builtin": "qda_kl.datasets:LmEvalBuiltin",
    },
    group="qda.datasets",
)

#: A KL reference directory, as ``qda-kl ref`` writes it.
KL_REFERENCE = "kl_reference"
#: A long-generation check's JSONL of ``{id, domain, prompt}`` rows; the rows name the domains.
PROMPTS = "gencheck_prompts"
#: A directory of lm-eval task YAMLs.
TASKS = "lm_eval_tasks"


class Dataset(BaseModel):
    """Base class of a dataset: its settings, and where its data is on this machine.

    Attributes:
        uses: the uses this dataset can serve (:data:`KL_REFERENCE`, :data:`PROMPTS`, :data:`TASKS`); None: any.
        public: the agent may read this data, as it may read anything shipped in an installed package, which every
            agent job binds. The judge then skips its refusals of a path the agent can read. A KL reference is
            never public.
    """

    model_config = ConfigDict(extra="forbid")

    name: ClassVar[str] = ""
    uses: ClassVar[frozenset[str] | None] = None
    public: ClassVar[bool] = False

    def local(self) -> Path:
        """The local file or directory holding the data.

        It may fetch the data into a cache first, but does no GPU work and returns the same path every time: the
        judge checks the path it returns at start and reads it later.

        Raises:
            Refused: the data cannot be made available.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """``{key: settings}``, the form a config file holds (every default filled in)."""
        return {self.name: self.model_dump(mode="json")}


def parse_dataset(value: Any, where: str, use: str | None = None) -> Dataset:
    """The dataset a config value names: a plain path (the ``file`` entry) or one ``{key: settings}``.

    Args:
        value: the config value.
        where: its config key, for the messages.
        use: the use it is read for; None when that was checked as the config loaded.

    Raises:
        Refused: not a path or one key, an unknown key (the message lists the known ones), invalid settings, or
            a dataset that does not serve ``use``.
    """
    if isinstance(value, (str, PurePath)):
        value = {"file": {"path": str(value)}}
    if not isinstance(value, dict) or len(value) != 1:
        raise Refused(f"{where}: expected a path or one dataset {{key: settings}}; got {value!r}")
    ((key, cfg),) = value.items()
    if key not in DATASETS:
        raise Refused(f"{where}.{key}: unknown dataset; known: {sorted(DATASETS)}")
    if not isinstance(cfg, dict | None):
        raise Refused(f"{where}.{key}: a dataset's settings are a mapping; got {cfg!r}")
    try:
        dataset = DATASETS[key](**(cfg or {}))
    except ValidationError as e:
        raise Refused("; ".join(error_lines(e, f"{where}.{key}"))) from None
    if use is not None and dataset.uses is not None and use not in dataset.uses:
        raise Refused(f"{where}.{key}: serves {sorted(dataset.uses)}, not {use}")
    return dataset


def with_defaults(value: Any, where: str, use: str) -> Any:
    """A config's selection as its record holds it: a named dataset with every default filled in, a path as given.

    Raises:
        Refused: as :func:`parse_dataset`.
    """
    dataset = parse_dataset(value, where, use)
    return dataset.to_dict() if isinstance(value, dict) else value


def content_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, or of a directory's relative file names and bytes (compiled caches aside).

    Raises:
        OSError: the path cannot be read.
    """
    h = hashlib.sha256()
    if not path.is_dir():
        h.update(path.read_bytes())
        return h.hexdigest()
    for file in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f"{file.relative_to(path)}\0".encode())
        h.update(hashlib.sha256(file.read_bytes()).digest())
    return h.hexdigest()


class FileDataset(Dataset):
    """A file or directory on disk; a plain path in a config is this entry.

    Attributes:
        path: the file or directory.
    """

    name: ClassVar[str] = "file"
    path: str

    def local(self) -> Path:
        """The path as given."""
        return Path(self.path)


class GencheckBuiltin(Dataset):
    """The long-generation prompts shipped with qda-kl (``data/README.md`` lists their sources and licenses)."""

    name: ClassVar[str] = "gencheck_builtin"
    uses: ClassVar[frozenset[str]] = frozenset({PROMPTS})
    public: ClassVar[bool] = True

    def local(self) -> Path:
        """The shipped JSONL."""
        return Path(__file__).with_name("data") / "gencheck-prompts.jsonl"


class LmEvalBuiltin(Dataset):
    """The task YAMLs shipped with qda-kl: the paper's benchmarks that lm-evaluation-harness does not ship."""

    name: ClassVar[str] = "lm_eval_builtin"
    uses: ClassVar[frozenset[str]] = frozenset({TASKS})
    public: ClassVar[bool] = True

    def local(self) -> Path:
        """The shipped task directory."""
        return Path(__file__).with_name("eval") / "tasks"
