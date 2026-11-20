"""Calibration sources: the plugins behind a recipe's ``calib`` key.

A recipe names one source by its key, with the source's settings::

    calib: {file: {path: calib.jsonl, samples: 128, seqlen: 2048}}
    calib: {hf: {dataset: HuggingFaceH4/ultrachat_200k, revision: <commit, branch or tag>, split: train_sft}}
    calib: {vlm: {}}  # image + text rows, for a vision-language model
    calib: {slice: {pool: calib/pool, names: if-verifiable}}  # a named slice of a pool, or a weighted list
    calib: {corpus: {file: {path: prompts.jsonl}}}  # the calib side of a corpus plugin (:mod:`qda_quant.intake.corpus`)

A source only reads rows (:meth:`CalibSource.read` returns :class:`Rows`). The engine tokenizes, cuts and orders them
(:func:`qda_quant.engine.calib.load_samples`), so two sources that give the same rows give the same samples.
Every source carries the engine's settings ``samples``, ``seqlen`` and ``seed``.

``SOURCES`` lists built-ins by name. A package's ``qda.plugins`` hook adds a source class with
``SOURCES.register(name, class)``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ..exit import ENVIRONMENT
from ..intake.corpus import CORPORA, Corpus, parse_corpus, split
from ..output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION
from ..registry import Registry
from ..step import Refused
from .contamination import CHARS, KIND, WINDOW, PromptIndex, filter_rows

#: Calibration sources by key: built-ins and package registrations.
SOURCES = Registry(
    "calib",
    {
        "corpus": "qda_quant.calib.sources:CorpusSource",
        "file": "qda_quant.calib.sources:FileSource",
        "mix": "qda_quant.calib.sources:MixSource",
        "selfgen": "qda_quant.calib.sources:SelfgenSource",
        "slice": "qda_quant.calib.sources:SliceSource",
        "hf": "qda_quant.calib.sources:HFSource",
        "vlm": "qda_quant.calib.sources:VLMSource",
    },
)

#: The reference index a search host passes to every build job (an index file, see
#: :mod:`qda_quant.calib.contamination`). A source's own ``contamination_index`` setting wins over it.
INDEX_ENV = "QDA_CONTAMINATION_INDEX"
#: Set by a search host: then a build with no reference index fails to start (an infrastructure error; a
#: candidate's recipe is never the reason), so no search build can run unfiltered by omission.
REQUIRED_ENV = "QDA_CONTAMINATION_REQUIRED"


@dataclass(frozen=True)
class Rows:
    """What a source gives the engine.

    Attributes:
        rows: JSON rows: ``{"messages"}``, ``{"text"}``, ``{"input_ids"}`` or traces
            ``{"prompt_ids", "completion_ids"}`` (see :mod:`qda_quant.engine.calib`).
        tokenizer: sha256 of the ``tokenizer.json`` that made token-id rows (None when unknown).
    """

    rows: list[dict[str, Any]]
    tokenizer: str | None


class CalibSource(BaseModel):
    """Base class of a calibration source: its settings, plus the engine's cut settings.

    Attributes:
        samples: samples the build uses.
        seqlen: tokens per sample.
        seed: sample order: 0 keeps the rows' order, k > 0 shuffles the samples with ``random.Random(k)``.
        chunks: cut every row into consecutive pieces of ``seqlen`` tokens, including a shorter final piece.
            Each piece is one sample, and ``samples`` counts pieces. When false, long rows are truncated and
            short rows are concatenated (:mod:`qda_quant.engine.calib`).
        contamination_index: an index of the evaluation reference's PROMPT hashes (:mod:`qda_quant.calib.contamination`)
            to filter every row this source reads: rows that match a reference prompt by id, normalized text or a
            20-word window are dropped and recorded in ``build.json``, never a refusal. Only prompts are compared:
            answer bodies (a generated calibration answer) are never checked.
    """

    model_config = ConfigDict(extra="forbid")

    name: ClassVar[str] = ""
    #: File settings resolved beside the config file by `recipe.resolve_paths`.
    paths: ClassVar[tuple[str, ...]] = ("contamination_index",)
    _dropped: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _exempted: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _read: int = PrivateAttr(default=0)
    _index: PromptIndex | None = PrivateAttr(default=None)
    _index_path: str | None = PrivateAttr(default=None)
    _index_sha256: str | None = PrivateAttr(default=None)
    _absorbed: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _templates: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _lints: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    #: Nested settings: ``{field: ("source" or "corpus", part model)}``. ``"*"`` covers plugin keys;
    #: `part model` supplies paths for list elements that name no plugin (a mix file part).
    nested: ClassVar[dict[str, tuple[str, type[BaseModel] | None]]] = {}
    samples: int = 128
    seqlen: int = 2048
    seed: int = 0
    chunks: bool = False
    contamination_index: str | None = None

    @property
    def label(self) -> str:
        """Names the data in messages."""
        return self.name

    def index(self) -> PromptIndex | None:
        """The contamination index this source filters with (None: it has none).

        Outside a search the source's ``contamination_index`` setting wins, else the host's `INDEX_ENV` applies. In
        a search (`REQUIRED_ENV`) the host's index is the only one: a recipe cannot name its own - neither to
        replace the host's reference nor to stand in for a missing one - because the recipe is the agent's.
        `REQUIRED_ENV` makes a missing index an infrastructure failure (a search build must not run unfiltered).

        Raises:
            Refused: a search build's recipe names its own index, a search build has no host index, or the
                index file is unreadable, malformed or empty.
        """
        env = os.environ.get(INDEX_ENV)
        if os.environ.get(REQUIRED_ENV):
            nested = [src for src, _w in getattr(self, "source_parts", list)()]
            if any(getattr(src, "contamination_index", None) for src in nested):
                raise Refused(
                    "contamination-index: a search build's recipe names its own contamination index in a "
                    "nested source (calib.mix.parts), which cannot replace or stand in for the judge host's "
                    f"reference ({INDEX_ENV}); remove it (infrastructure: the host passes the search's index)"
                )
            if self.contamination_index:
                raise Refused(
                    "contamination-index: a search build's recipe names its own contamination index "
                    f"(calib.<source>.contamination_index), which cannot replace or stand in for the judge host's reference "
                    f"({INDEX_ENV}); remove it (infrastructure: the host passes the search's own index)"
                )
            if not env:
                raise Refused(
                    f"contamination-index: this search build was given no reference index ({INDEX_ENV}), so it "
                    "would run unfiltered; the judge host passes the index to every build job "
                    "(infrastructure error, not the candidate's)"
                )
        path = env or self.contamination_index
        if not path:
            return None
        if self._index is None or self._index_path != path:
            from . import sha256_file

            self._index = PromptIndex.load(path)
            self._index_path = path
            self._index_sha256 = sha256_file(path)
        return self._index

    def screen(
        self, rows: list[dict[str, Any]], tokenizer: Any = None, record: bool = True
    ) -> list[dict[str, Any]]:
        """The rows left after the contamination index, and every drop or exemption recorded for ``build.json``.

        A source without an index keeps every row. Prompts only: an answer body is never compared. `tokenizer`
        decodes token-id rows so a pre-tokenized file is screened like the text it holds. `record: false` only
        applies the filter (a second intake of the same rows, already recorded at plan time).
        """
        self._read += len(rows) if record else 0
        return self._screen_rows(rows, tokenizer, record)

    def _screen_rows(
        self, rows: list[dict[str, Any]], tokenizer: Any = None, record: bool = True
    ) -> list[dict[str, Any]]:
        """Filter rows by the prompts they carry (the rule a source's own :meth:`screen` may replace)."""
        index = self.index()
        if index is None:
            return rows
        kept, dropped, exempted = filter_rows(rows, index, tokenizer)
        if record:
            self._dropped = [*self._dropped, *dropped]
            self._exempted = [*self._exempted, *exempted]
        return kept

    def apply_index(
        self, rows: list[dict[str, Any]], tokenizer: Any = None, record: bool = True
    ) -> list[dict[str, Any]]:
        """Filter rows a step reads itself by this source's index, recording the drops.

        Unlike :meth:`screen`, this never dispatches to a source's own prompt file: the rows passed in are the
        rows the step reads (``scale_kd``'s traces), so a selfgen source's prompt list is irrelevant here.

        Prompts only: an answer body is never compared. `tokenizer` decodes token-id rows.
        """
        index = self.index()
        if record:
            self._read += len(rows)
        if index is None:
            return rows
        kept, dropped, exempted = filter_rows(rows, index, tokenizer)
        if record:
            self._dropped = [*self._dropped, *dropped]
            self._exempted = [*self._exempted, *exempted]
        return kept

    def absorb(self, part: CalibSource) -> None:
        """Add a part source's read count, drops, exemptions and record to this source's (a mix of sources)."""
        self._read += part._read
        self._dropped = [*self._dropped, *part._dropped]
        self._exempted = [*self._exempted, *part._exempted]
        if (record := part.contamination_record()) is not None:
            self._absorbed = [*self._absorbed, record]
        self._templates = [*self._templates, *part._templates]
        self._lints = {**self._lints, **part._lints}

    def note_template(self, record: dict[str, Any]) -> None:
        """Keep the chat-template check's record of the rows this source read (:mod:`qda_quant.calib.template`).

        A second read of the same source (a step reading the calibration again) replaces the record, never adds.
        """
        self._templates = [record]

    def note_lint(self, record: dict[str, Any]) -> None:
        """Keep the data lint's record of one file a build read (:func:`qda_quant.calib.template.lint_rows`).

        Keyed by the record's ``file``: a second read of the same file replaces its record, never adds.
        """
        self._lints = {**self._lints, record["file"]: record}

    def lint_records(self) -> list[dict[str, Any]]:
        """What ``build.json`` records of the data lint: one record per file read, in first-read order."""
        return list(self._lints.values())

    def note_template_reset(self) -> None:
        """Forget the chat-template records (a mix of sources reads its parts again)."""
        self._templates = []

    def template_record(self) -> dict[str, Any] | None:
        """What ``build.json`` records of the chat-template check: own / plain / foreign row counts (None: not run).

        A mix of sources sums its parts' counts and lists the families each part's model template produces.
        """
        if not self._templates:
            return None
        out = dict(self._templates[0])
        for r in self._templates[1:]:
            out["produced"] = sorted({*out["produced"], *r["produced"]})
            for k in ("own", "plain", "foreign"):
                out[k] += r[k]
        return out

    @property
    def dropped(self) -> list[dict[str, Any]]:
        """The rows this source's contamination index dropped, in read order."""
        return list(self._dropped)

    @property
    def exempted(self) -> list[dict[str, Any]]:
        """The rows a shared generic window would have dropped but for an exemption, in read order."""
        return list(self._exempted)

    def contamination_record(self) -> dict[str, Any] | None:
        """What ``build.json`` records of the contamination filter once rows were read (None: no filter ran)."""
        if self._index is not None:
            return {
                "filter": "calibration reference-prompt filter",
                "index": self._index_path or self.contamination_index,
                "sha256": self._index_sha256,
                "reference_prompts": self._index.prompts,
                "kind": KIND,
                "window": WINDOW,
                "chars": CHARS,
                "compared": "prompts only, never answer bodies (self-generated answers are allowed)",
                "rows": self._read,
                "kept": self._read - sum(int(d.get("rows", 1)) for d in self._dropped),
                "dropped": sum(int(d.get("rows", 1)) for d in self._dropped),
                "matches": self._dropped,
                "exempted": len(self._exempted),
                "exempted_rows": self._exempted,
            }
        if self._absorbed:  # a mix of sources: one record per part
            return {
                "parts": self._absorbed,
                "compared": "prompts only, never answer bodies (self-generated answers are allowed)",
                "rows": sum(p["rows"] for p in self._absorbed),
                "dropped": sum(p["dropped"] for p in self._absorbed),
                "exempted": sum(p.get("exempted", 0) for p in self._absorbed),
                "ids": [m["id"] for p in self._absorbed for m in p["matches"]],
            }
        return None

    def read(self, model_dir: str | Path) -> Rows:
        """The rows for the model at ``model_dir`` (a source may depend on the model, e.g. ``selfgen``).

        Never does GPU work: a source whose rows must be generated first reports that in :meth:`pending`.
        """
        raise NotImplementedError

    def pin(self) -> None:
        """Replace a setting that names moving data with the data it points at now (the default: nothing moves).

        Plan time calls this before anything reads the source, so the recipe a build records names the exact
        rows it read and a rebuild repeats them.
        """

    def pending(self, model_dir: str | Path) -> str | None:  # noqa: ARG002  (most sources need nothing)
        """What must be generated before :meth:`read` can give rows (None: nothing; the default).

        Plan time calls this only: it resolves the settings and the cache key and checks the cache, no more.
        """
        return None

    def generate(self, model_dir: str | Path) -> dict[str, Any]:
        """Make the rows :meth:`pending` names (the build's ``generate`` stage); returns a record of the result."""
        raise NotImplementedError(f"calib {self.name}: nothing to generate")

    def to_dict(self) -> dict[str, Any]:
        """``{key: settings}``, the form a recipe file holds (every default filled in)."""
        return {self.name: self.model_dump()}


def read_file(path: str | Path) -> Rows:
    """A calibration JSONL's rows, and its tokenizer fingerprint.

    The fingerprint is a leading ``{"tokenizer": "<sha256>"}`` header row (removed from the rows), else the
    ``<path>.tokenizer`` sidecar.

    Raises:
        Refused: the file cannot be read (missing, a directory, or not permitted).
    """
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as e:
        raise Refused(f"calib file {p}: cannot read: {e}") from e
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    tag = None
    if rows and set(rows[0]) == {"tokenizer"}:
        tag = rows.pop(0)["tokenizer"]
    side = p.with_name(p.name + ".tokenizer")
    if tag is None and side.is_file():
        tag = side.read_text().strip()
    return Rows(rows, tag)


def parse_source(d: dict[str, Any]) -> CalibSource:
    """The source a recipe's ``calib`` mapping names: exactly one key, the source's, with its settings.

    Raises:
        Refused: not one key, or an unknown key (the message lists the known ones). A setting beside the
            source's key (``calib.samples``) is named as that dotted key.
    """
    named = [k for k in d if k in SOURCES] if isinstance(d, dict) else []
    if len(named) == 1 and len(d) > 1:
        raise Refused(
            "; ".join(
                f"calib.{k}: unknown key; a source's settings go under its key (calib.{named[0]}.{k})"
                for k in sorted(set(d) - set(named))
            )
        )
    if not isinstance(d, dict) or len(d) != 1:
        raise Refused(f"calib: expected one source {{key: settings}}, known: {sorted(SOURCES)}; got {d!r}")
    ((key, cfg),) = d.items()
    if key not in SOURCES:
        raise Refused(f"calib.{key}: unknown calib source; known: {sorted(SOURCES)}")
    return SOURCES[key](**(cfg or {}))


class FileSource(CalibSource):
    """A calibration JSONL on disk.

    Attributes:
        path: the file.
    """

    name: ClassVar[str] = "file"
    paths: ClassVar[tuple[str, ...]] = ("path", "contamination_index")
    path: str

    @property
    def label(self) -> str:
        """The file's path."""
        return self.path

    def read(self, model_dir: str | Path) -> Rows:  # noqa: ARG002  (a file does not depend on the model)
        """The file's rows (:func:`read_file`)."""
        return read_file(self.path)


class CorpusSource(CalibSource):
    """The rows a corpus plugin reserves for calibration (:func:`qda_quant.intake.corpus.split`).

    Settings are the corpus, named by key, as in ``qda eval``::

        calib: {corpus: {file: {path: prompts.jsonl}}}
        calib: {corpus: {reasoning: {}}}

    The engine's own settings (``samples``, ``seqlen``, ``seed``, ``chunks``) stay beside the corpus's key.
    A ``prompt`` row is calibrated as its text: a corpus written for evaluation keeps the question in
    ``prompt``, and calibration has no model to ask.
    """

    # The corpus's name is a plugin key, not a field, so it and its settings arrive as extra keys.
    model_config = ConfigDict(extra="allow")

    name: ClassVar[str] = "corpus"
    nested: ClassVar[dict[str, tuple[str, type[BaseModel] | None]]] = {"*": ("corpus", None)}

    @model_validator(mode="after")
    def _one_corpus(self) -> CorpusSource:
        names = sorted(self.model_extra or {})
        if len(names) != 1:
            raise ValueError(
                f"corpus: expected one corpus {{name: settings}}, known: {sorted(CORPORA)}; got {names}"
            )
        parse_corpus(self.named())  # refuse at load time, not when the engine first reads
        return self

    def named(self) -> dict[str, Any]:
        """``{<corpus name>: settings}``, the one entry this source holds."""
        return dict(self.model_extra or {})

    def parsed(self) -> Corpus:
        """The corpus this source names."""
        return parse_corpus(self.named())

    @property
    def label(self) -> str:
        """``corpus <name>``: which corpus the calibration rows came from."""
        return f"corpus {self.parsed().name}"

    def read(self, model_dir: str | Path) -> Rows:  # noqa: ARG002  (a corpus does not depend on the model)
        """The corpus's calib-side rows; ``prompt`` rows gain the same text as ``text``."""
        rows = split(self.parsed(), "calib")
        return Rows([dict(r, text=r["prompt"]) if "prompt" in r else r for r in rows], None)

    def to_dict(self) -> dict[str, Any]:
        """``{corpus: {<name>: settings, <engine settings>}}``, every default filled in."""
        c = self.parsed()
        engine = {
            "samples": self.samples,
            "seqlen": self.seqlen,
            "seed": self.seed,
            "chunks": self.chunks,
            "contamination_index": self.contamination_index,
        }
        return {self.name: {c.name: c.model_dump(), **engine}}


class SelfgenSource(CalibSource):
    """The original model's own answers to a prompt file, cached per model (:func:`qda_quant.calib.selfgen.selfgen`).

    Attributes:
        prompts: JSONL of ``{"id", "prompt"}`` lines.
        n: answers per prompt.
        max_tokens: the cap of one answer.
        gen_seed: the seed of draw 0 (draw ``d`` uses ``gen_seed + d``).
        gpu_memory_utilization: vLLM's share of the card (default 0.85); use 0.2 when sharing.
        chat_template_kwargs: extra chat-template arguments (e.g. ``{"enable_thinking": false}``).
        render: ``text`` rows (the rendered prompt plus the answer, verbatim) or ``messages`` rows.
        sampling: the sampling settings; its fields win over the model's ``generation_config.json``. Required
            when the model ships no ``generation_config.json`` (the build refuses otherwise).
    """

    name: ClassVar[str] = "selfgen"
    paths: ClassVar[tuple[str, ...]] = ("prompts", "contamination_index")
    prompts: str
    n: int = 1
    max_tokens: int = 2048
    gen_seed: int = 17
    gpu_memory_utilization: float = Field(default=DEFAULT_GPU_MEMORY_UTILIZATION, gt=0, le=1)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    render: Literal["text", "messages"] = "text"
    sampling: dict[str, Any] | None = None

    def _cache(self, model_dir: str | Path) -> tuple[Path, bool]:
        from .selfgen import cache_file

        return cache_file(
            model_dir,
            self.prompts,
            self.n,
            self.max_tokens,
            self.gen_seed,
            gpu_memory_utilization=self.gpu_memory_utilization,
            chat_template_kwargs=self.chat_template_kwargs,
            render=self.render,
            sampling=self.sampling,
        )

    def pending(self, model_dir: str | Path) -> str | None:
        """The generation a cache miss needs; None on a hit."""
        path, hit = self._cache(model_dir)
        return None if hit else f"selfgen: {self.n} answer(s) per prompt of {self.prompts} into {path}"

    def generate(self, model_dir: str | Path) -> dict[str, Any]:
        """Generate the cache (vLLM on the model, in a child process); returns ``{path, sha256}`` of the file."""
        from . import sha256_file
        from .selfgen import generate_isolated

        path = generate_isolated(
            model_dir,
            self.prompts,
            self.n,
            self.max_tokens,
            self.gen_seed,
            gpu_memory_utilization=self.gpu_memory_utilization,
            chat_template_kwargs=self.chat_template_kwargs,
            render=self.render,
            sampling=self.sampling,
        )
        return {"path": str(path), "sha256": sha256_file(path)}

    def read(self, model_dir: str | Path) -> Rows:
        """The cached generation for ``model_dir``.

        Raises:
            Refused: no cache for this model and these settings (``qda build`` generates it as its first stage).
        """
        path, hit = self._cache(model_dir)
        if not hit:
            raise Refused(
                f"calibration selfgen: no cached generation at {path}; `qda build` generates it as its first stage"
            )
        return read_file(path)

    def screen(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any = None,  # noqa: ARG002  (the prompts are text; no tokenizer is needed)
        record: bool = True,
    ) -> list[dict[str, Any]]:
        """The rows left after the contamination index: a dropped PROMPT takes its answers with it.

        The answers are never compared. The cache holds ``n`` contiguous rows per prompt, so the prompt file is
        screened first and prompt ``j``'s rows go when that prompt is a hit. This is the only correct rule for
        ``render="text"``, whose row is the rendered prompt and the answer in one string (nothing could split
        them afterwards).

        Raises:
            Refused: a cache whose row count is not the prompt file's times ``n``; no row could be attributed to
                a prompt, and screening the whole rendered row would compare the answer body.
        """
        self._read += len(rows) if record else 0
        index = self.index()
        if index is None:
            return rows
        prompts = read_file(self.prompts).rows
        if self.n < 1 or len(rows) != len(prompts) * self.n:
            raise Refused(
                f"calibration selfgen: the cache has {len(rows)} rows for {len(prompts)} prompts x {self.n}; "
                "a row cannot be attributed to a prompt, and the rendered row holds the answer body; "
                "regenerate the cache"
            )
        _kept, dropped, exempted = filter_rows(prompts, index)
        for d in dropped:
            d["rows"] = self.n  # one record per prompt; the count is the answer rows it takes with it
        if record:
            self._dropped = [*self._dropped, *dropped]
            self._exempted = [*self._exempted, *exempted]
        gone = {d["row"] for d in dropped}
        return [r for i, r in enumerate(rows) if i // self.n not in gone]


#: A pinned revision: a lowercase 40-character commit hash, which `HFSource.pin` leaves as it is
_COMMIT = re.compile(r"[0-9a-f]{40}")


#: ShareGPT speaker names (a turn's ``from``) and the chat role each becomes; any other name is refused.
SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _sharegpt(dataset: str, turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    """ShareGPT turns ``{from, value}`` as chat messages ``{role, content}``.

    Raises:
        Refused: a turn's ``from`` is not in `SHAREGPT_ROLES`.
    """
    out = []
    for m in turns:
        if m["from"] not in SHAREGPT_ROLES:
            raise Refused(
                f"calib hf {dataset}: ShareGPT turn from {m['from']!r}; known speakers: {', '.join(SHAREGPT_ROLES)}"
            )
        out.append({"role": SHAREGPT_ROLES[m["from"]], "content": m["value"]})
    return out


#: The data files of a dataset repository, read with `pyarrow` (parquet) or `json` (the two JSON forms).
DATA_SUFFIXES = (".parquet", ".jsonl", ".json")

#: A file or directory name that says which split its rows are; a repo with none holds one unnamed split.
_SPLIT_NAME = re.compile(r"(train|test|validation|valid|dev)([-_.]|$)")


def _split_files(names: list[str], subset: str | None, split: str) -> list[str]:
    """The data files of one split among a dataset repository's file names, in order (`HFSource.read`)."""
    files = [
        p
        for p in sorted(PurePosixPath(n) for n in names)
        if p.suffix in DATA_SUFFIXES and not any(d.startswith(".") for d in p.parts)
    ]
    if subset is not None:
        files = [p for p in files if subset in p.parts[:-1]]
    mine = [p for p in files if p.stem == split or p.stem.startswith(f"{split}-") or split in p.parts[:-1]]
    named = any(_SPLIT_NAME.match(part) for p in files for part in (*p.parts[:-1], p.stem))
    if not mine and not named and split == "train":
        mine = files  # names no split anywhere: the repository is one split, and that split is `train`
    return [str(p) for p in mine]


class HFSource(CalibSource):
    """Rows of a Hugging Face dataset, read from its files in ``HF_HOME`` (needs the ``hf`` extra: ``pyarrow``).

    Nothing is shipped: the dataset id and revision in the recipe name the data, and both are recorded with
    the recipe in every checkpoint. A commit hash is used as given; a branch or tag is resolved to its commit
    at plan time, and the record keeps the hash, so a rebuild reads the same rows. Check the dataset's
    license before you use it.

    The rows are read from the repository's own ``.parquet``, ``.jsonl`` and ``.json`` files, so a machine with
    no network needs only ``hf download <dataset> --repo-type dataset --revision <revision>`` first; a machine
    with one downloads the files the rows come from and no others. A file belongs to ``split`` when its name is
    the split, begins with the split and a ``-`` (a shard), or sits in a directory named after the split, under
    a ``subset`` directory when the dataset has configurations; a repository whose names say no split at all
    is the ``train`` split.

    Attributes:
        dataset: the dataset id on the Hub.
        revision: the dataset version: a full 40-character commit hash, or a branch or tag (resolved to its
            commit at plan time).
        subset: the dataset's configuration name, if it has several.
        split: the split to read.
        column: the column holding each row: a list of chat messages (``{role, content}``, or ShareGPT's
            ``{from, value}`` turns, mapped by `SHAREGPT_ROLES`), or a text string.
        rows: read only the first this many rows of the split (later files of the split are not read
            and, when the dataset is not on disk already, not downloaded).
    """

    name: ClassVar[str] = "hf"
    dataset: str
    revision: str
    subset: str | None = None
    split: str = "train"
    column: str = "messages"
    rows: int | None = None

    def pin(self) -> None:
        """Resolve a branch or tag ``revision`` to the commit it points at (a commit hash stays as it is).

        Raises:
            Refused: the hub cannot resolve the revision (unknown, offline, or unreachable), or names no
                commit hash for it.
        """
        if _COMMIT.fullmatch(self.revision):
            return
        from huggingface_hub import HfApi

        try:
            sha = HfApi().dataset_info(self.dataset, revision=self.revision).sha
        except Exception as e:  # noqa: BLE001  (any hub error: say it, refuse)
            raise Refused(
                f"calib hf {self.dataset}@{self.revision}: cannot resolve the revision: "
                f"{type(e).__name__}: {e}"
            ) from e
        if sha is None or not _COMMIT.fullmatch(sha):
            raise Refused(f"calib hf {self.dataset}@{self.revision}: the hub gave no commit hash: {sha!r}")
        self.revision = sha

    def read(self, model_dir: str | Path) -> Rows:  # noqa: ARG002
        """The split's first ``rows`` rows as ``{"messages"}`` or ``{"text"}`` rows.

        Raises:
            Refused: the dataset is neither downloaded nor downloadable (offline, unreachable, unknown id or
                revision), the split has no data file, ``pyarrow`` is missing for a parquet file, or the column
                is absent or holds neither chat messages nor text.
        """
        root, names = self._repo()
        files = _split_files(names, self.subset, self.split)
        if not files:
            raise Refused(
                f"calib hf {self.label}: no data file of split {self.split!r}"
                f"{f' subset {self.subset!r}' if self.subset else ''} among {sorted(names)}",
                f"hf download {self.dataset} --repo-type dataset --revision {self.revision}",
            )
        out: list[dict[str, Any]] = []
        for name in files:
            want = None if self.rows is None else self.rows - len(out)
            out += [self._row(v) for v in islice(self._column(self._file(root, name)), want)]
            if self.rows is not None and len(out) >= self.rows:
                break
        return Rows(out, None)

    def _repo(self) -> tuple[Path | None, list[str]]:
        """The repository's files: the snapshot on this machine and its names, or None and the hub's names.

        A snapshot already downloaded is read as it is, so an offline machine needs only
        ``hf download <dataset> --repo-type dataset``; otherwise only the files the rows come from are
        downloaded, as :meth:`_file` reaches them. A snapshot with no data file of the split (``datasets``
        leaves only ``README.md`` there and keeps the rows in its own cache) is read from the hub instead
        when the hub answers; when it does not, the snapshot is returned and :meth:`read` names the download.

        Raises:
            Refused: nothing is on disk and the hub has no such dataset or revision, or cannot be reached.
        """
        from huggingface_hub import HfApi, snapshot_download

        root = None
        try:
            root = Path(
                snapshot_download(
                    self.dataset, repo_type="dataset", revision=self.revision, local_files_only=True
                )
            )
        except Exception:  # noqa: BLE001  (not on disk, whatever the hub client says: read the hub)
            pass
        else:
            names = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
            if _split_files(names, self.subset, self.split):
                return root, names
        try:
            return None, HfApi().list_repo_files(self.dataset, repo_type="dataset", revision=self.revision)
        except Exception as e:  # noqa: BLE001  (any hub error, offline or transport: say it, refuse)
            if root is not None:
                return root, names  # offline with a snapshot that lacks the split: read() names the download
            raise Refused(f"calib hf {self.label}: cannot read the dataset: {type(e).__name__}: {e}") from e

    def _file(self, root: Path | None, name: str) -> Path:
        """One data file: in the downloaded snapshot, or downloaded into ``HF_HOME`` now.

        Raises:
            Refused: the hub cannot be reached for the file.
        """
        if root is not None:
            return root / name
        from huggingface_hub import hf_hub_download

        try:
            return Path(hf_hub_download(self.dataset, name, repo_type="dataset", revision=self.revision))
        except Exception as e:  # noqa: BLE001  (any hub error, offline or transport: say it, refuse)
            raise Refused(f"calib hf {self.label}: cannot read the dataset: {type(e).__name__}: {e}") from e

    def _column(self, path: Path) -> Iterator[Any]:
        """Each row's ``column`` value in one data file, read lazily.

        Raises:
            Refused: ``pyarrow`` is missing, the file is not a list of rows, or a row has no such column.
        """
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise Refused(
                    "calib hf: needs the `pyarrow` package to read a parquet dataset: "
                    'uv pip install -c constraints.txt "./packages/qda-quant[hf]"',
                    code=ENVIRONMENT,
                ) from e
            parquet = pq.ParquetFile(path)
            self._column_of(path, parquet.schema_arrow.names)
            for batch in parquet.iter_batches(batch_size=64, columns=[self.column]):
                yield from batch.column(self.column).to_pylist()
            return
        with path.open(encoding="utf-8") as f:
            rows = (
                json.load(f) if path.suffix == ".json" else (json.loads(line) for line in f if line.strip())
            )
            if path.suffix == ".json" and not isinstance(rows, list):
                raise Refused(f"calib hf {self.label}: {path.name} holds {type(rows).__name__}, not rows")
            for r in rows:
                self._column_of(path, r)
                yield r[self.column]

    def _column_of(self, path: Path, columns: Any) -> None:
        """Refuse a file or row without this source's column, naming the ones it has."""
        if self.column not in columns:
            raise Refused(
                f"calib hf {self.label}: {path.name} has no column {self.column!r}; it has {sorted(columns)}"
            )

    def _row(self, v: Any) -> dict[str, Any]:
        """One column value as a calibration row.

        Raises:
            Refused: the value is neither chat messages, ShareGPT turns nor text.
        """
        if isinstance(v, str):
            return {"text": v}
        if isinstance(v, list) and all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in v):
            return {"messages": [{"role": m["role"], "content": m["content"]} for m in v]}
        if isinstance(v, list) and v and all(isinstance(m, dict) and {"from", "value"} <= set(m) for m in v):
            return {"messages": _sharegpt(self.dataset, v)}
        raise Refused(
            f"calib hf {self.dataset}: column {self.column!r} holds {type(v).__name__}; "
            "expected chat messages [{role, content}, ...], ShareGPT turns [{from, value}, ...] or a text "
            "string"
        )

    @property
    def label(self) -> str:
        """``dataset@revision``."""
        return f"{self.dataset}@{self.revision}"


class VLMSource(CalibSource):
    """Image-text rows of a Hugging Face dataset, for a model that reads images (``calib: {vlm: {}}``).

    Text-only calibration never shows the language model the activations image tokens make. Here the engine runs
    each row through the model's own processor (:func:`qda_quant.engine.calib.load_samples`), and the build's
    calibration forward runs the vision tower, so the language-model layers are calibrated on image embeddings.
    Each row gives one sample: the user turn (its images, then the question) and the assistant's answer.

    The default rows are the first shard of the ``localized_narratives`` subset of HuggingFaceM4/the_cauldron at
    a pinned revision: descriptions of Open Images training images (Localized Narratives, CC BY 4.0; Open Images
    lists each image's license, mostly CC BY 2.0). It is a training set, not a test benchmark. Nothing is
    shipped: the shard (490 MB) is downloaded at build time and read with ``pyarrow``; images are decoded with
    Pillow and the model's processor needs ``torchvision``. The ``vlm`` extra brings all three.

    Attributes:
        dataset: the dataset repository on the Hub.
        revision: its full 40-character commit hash.
        file: the parquet file in the repository to read rows from, in order.
        images: the column holding each row's list of images (``{bytes}`` structs, as datasets stores them).
        texts: the column holding each row's turns, ``[{user, assistant}, ...]`` (the_cauldron's layout).
        rows: read at most this many rows (default twice ``samples``); rows over ``seqlen`` tokens are skipped.
    """

    name: ClassVar[str] = "vlm"
    #: the engine processes these rows with the model's processor and keeps their pixel inputs
    images_source: ClassVar[bool] = True
    dataset: str = "HuggingFaceM4/the_cauldron"
    revision: str = Field(default="847a98a779b1652d65111daf20c972dfcd333605", pattern=r"^[0-9a-f]{40}$")
    file: str = "localized_narratives/train-00000-of-00043-537ccba7695649be.parquet"
    images: str = "images"
    texts: str = "texts"
    rows: int | None = None

    @property
    def label(self) -> str:
        """``dataset/file@revision``."""
        return f"{self.dataset}/{self.file}@{self.revision}"

    def read(self, model_dir: str | Path) -> Rows:  # noqa: ARG002
        """``{"messages", "images"}`` rows: each row's first turn with its images, as chat messages.

        Raises:
            Refused: ``pyarrow`` or Pillow is missing, the file cannot be downloaded, or a row is not images +
                turns.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise Refused(
                'calib vlm: needs the `pyarrow` package: uv pip install -c constraints.txt "./packages/qda-quant[vlm]"',
                code=ENVIRONMENT,
            ) from e
        try:
            import PIL.Image  # noqa: F401  (checked here, before the 490 MB download; `_row` decodes with it)
        except ImportError as e:
            raise Refused(
                'calib vlm: needs the `pillow` package to decode images: uv pip install -c constraints.txt "./packages/qda-quant[vlm]"',
                code=ENVIRONMENT,
            ) from e
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(self.dataset, self.file, repo_type="dataset", revision=self.revision)
        except OSError as e:  # the hub's network and not-found errors are OSErrors
            raise Refused(f"calib vlm {self.label}: cannot download: {type(e).__name__}: {e}") from e
        n = self.rows or 2 * self.samples  # room for rows skipped as too long
        out: list[dict[str, Any]] = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=[self.images, self.texts]):
            out += [self._row(r) for r in batch.to_pylist()[: n - len(out)]]
            if len(out) >= n:
                break
        return Rows(out, None)

    def _row(self, r: dict[str, Any]) -> dict[str, Any]:
        import io

        from PIL import Image

        images, turns = r.get(self.images), r.get(self.texts)
        if not images or not turns or not {"user", "assistant"} <= set(turns[0]):
            raise Refused(
                f"calib vlm {self.label}: expected images in {self.images!r} and turns "
                f"[{{user, assistant}}, ...] in {self.texts!r}"
            )
        user = [{"type": "image"} for _ in images] + [{"type": "text", "text": turns[0]["user"]}]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": turns[0]["assistant"]}]},
        ]
        return {
            "messages": messages,
            "images": [Image.open(io.BytesIO(im["bytes"])).convert("RGB") for im in images],
        }


class SliceFile(BaseModel):
    """A slice as plan time recorded it: its name and the sha256 of the file the name resolved to."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: str


class SliceSource(CalibSource):
    """Named slices of a calibration pool, selected by name or mixed by a weighted list.

    A pool is a directory of ``<name>.jsonl`` calibration files, each built by one pinned command (its
    ``.provenance.json`` says how). The recipe selects them by name, one slice or a weighted list::

        calib: {slice: {pool: calib/pool, names: if-verifiable}}
        calib: {slice: {pool: calib/pool, names: {if-verifiable: 9, chat: 1}}}

    A single slice reads the file's rows, as ``file`` does. A weighted list gives each slice its share of
    ``samples``, cut with this source's ``seqlen`` and ordered by ``mix_seed``, as a mix of sources
    (:func:`qda_quant.engine.calib.load_samples`). Plan time resolves every name and records its file's sha256 in
    ``files``, so ``resolved.yaml`` names the exact slices a build read. The agent chooses slices; the pool does
    not fix one mix.

    Attributes:
        pool: directory of ``<name>.jsonl`` slices.
        names: one slice name, or ``{name: weight}`` for a weighted list (a weight may be a fraction string).
        mix_seed: the interleaving seed of a weighted list (``seed`` orders the cut samples, as for every source).
        files: the slices and their file digests as plan time recorded them (read back from ``resolved.yaml``).
    """

    name: ClassVar[str] = "slice"
    paths: ClassVar[tuple[str, ...]] = ("pool", "contamination_index")
    pool: str
    names: str | dict[str, str | float]
    mix_seed: int = 0
    files: list[SliceFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_slice(self) -> SliceSource:
        self.selected()  # refuse a bad name, a bad weight or an empty list at load time
        return self

    def selected(self) -> list[tuple[str, Fraction]]:
        """The selected slices and their weights, normalized: one name weighs 1.

        Raises:
            ValueError: no slice, a name that is not a file name in the pool, or a weight that is not positive.
        """
        items = [(self.names, 1)] if isinstance(self.names, str) else list(self.names.items())
        if not items:
            raise ValueError("calib.slice.names: select at least one slice")
        out = []
        for name, w in items:
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError(f"calib.slice.names: {name!r} is not a slice name (a file name in the pool)")
            weight = Fraction(w) if isinstance(w, str) else Fraction(w)
            if weight <= 0:
                raise ValueError(f"calib.slice.names.{name}: weight {w} is not positive")
            out.append((name, weight))
        return out

    def _path(self, name: str) -> Path:
        """The pool file of slice ``name``, checked against the sha256 the recipe recorded for it.

        Raises:
            Refused: the pool has no such file (the message lists the slices it has), or the file changed after
                plan time pinned it.
        """
        pool, path = Path(self.pool), Path(self.pool) / f"{name}.jsonl"
        if not path.is_file():
            known = sorted(p.stem for p in pool.glob("*.jsonl")) if pool.is_dir() else []
            raise Refused(f"calib slice {name}: no {path.name} in pool {self.pool}; known: {known}")
        for recorded in self.files:
            if recorded.name == name:
                from . import sha256_file

                got = sha256_file(path)
                if got != recorded.sha256:
                    raise Refused(
                        f"calib slice {name}: {path} changed since the recipe pinned it "
                        f"(sha256 {got} != {recorded.sha256})"
                    )
        return path

    def pin(self) -> None:
        """Resolve every name and record its file's sha256 (``files``), so ``resolved.yaml`` names the data."""
        from . import sha256_file

        self.files = [SliceFile(name=n, sha256=sha256_file(self._path(n))) for n, _ in self.selected()]

    @property
    def label(self) -> str:
        """``slice`` and the selected names."""
        return "slice " + ", ".join(n for n, _ in self.selected())

    def source_parts(self) -> list[tuple[CalibSource, Fraction]]:
        """The weighted list as file parts (empty for one slice, which reads rows as one file does)."""
        parts = self.selected()
        if len(parts) < 2:
            return []
        return [
            (
                FileSource(
                    path=str(self._path(n)),
                    samples=self.samples,
                    seqlen=self.seqlen,
                    seed=self.seed,
                    chunks=self.chunks,
                    contamination_index=self.contamination_index,
                ),
                w,
            )
            for n, w in parts
        ]

    def read(self, model_dir: str | Path) -> Rows:  # noqa: ARG002
        """The one slice's rows (a weighted list gives samples, read by the engine through `source_parts`).

        Raises:
            Refused: a weighted list, or a slice whose file is missing from the pool.
        """
        parts = self.selected()
        if len(parts) > 1:
            raise Refused("calib slice: a weighted list gives samples, not rows (engine.calib.load_samples)")
        return read_file(self._path(parts[0][0]))


class MixPart(BaseModel):
    """One input of a mix: a calibration JSONL and its weight."""

    model_config = ConfigDict(extra="forbid")

    paths: ClassVar[tuple[str, ...]] = ("path",)
    path: str
    weight: str | float = 1


#: MixSource settings that act on a mix of files only (rows, the held-out guard, the cut, the written file)
_ROW_SETTINGS = ("rows", "heldout", "ngram", "out", "seqlen", "chunks", "contamination_index")


class MixSource(CalibSource):
    """A deterministic weighted mix of calibration files (:func:`qda_quant.calib.mix.mix`), or of sources.

    File parts (``{path, weight}``) are mixed as rows into one JSONL. Source parts (``{<source key>: settings,
    weight}``, e.g. ``{vlm: {}, weight: 1}``) are mixed as samples: each source gives its share of ``samples``,
    cut with its own ``seqlen``, and the engine interleaves them with ``mix_seed``
    (:func:`qda_quant.engine.calib.load_samples`). Use source parts to mix image rows with text, which a JSONL
    cannot hold::

        calib: {mix: {parts: [{vlm: {}, weight: 1}, {hf: {dataset: allenai/c4, ...}, weight: 1}], samples: 128}}

    A mix holds file parts or source parts, not both. A mix of sources takes only ``samples``, ``seed`` and
    ``mix_seed``: ``rows``, ``heldout``, ``ngram``, ``out``, ``seqlen`` and ``chunks`` act on a mix of files and are
    refused here (no held-out guard runs on source parts).

    Attributes:
        parts: the inputs and their weights (a weight may be a fraction string, e.g. ``"1/3"``).
        rows: rows in the mix (file parts; required for them).
        mix_seed: the mixing seed (``seed`` orders the cut samples, as for every source).
        heldout: drop rows that share a word n-gram with this JSONL (e.g. the KL prompts).
        ngram: n of the held-out guard.
        out: where the mix is written; default ``$QDA_CACHE/calib/mix/<key>.jsonl``.
    """

    name: ClassVar[str] = "mix"
    paths: ClassVar[tuple[str, ...]] = ("heldout", "contamination_index")
    nested: ClassVar[dict[str, tuple[str, type[BaseModel] | None]]] = {"parts": ("source", MixPart)}
    parts: list[MixPart | dict[str, Any]] = Field(min_length=1)
    rows: int | None = None
    mix_seed: int = 0
    heldout: str | None = None
    ngram: int = 13
    out: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> MixSource:
        files = [isinstance(p, MixPart) for p in self.parts]
        if any(files) and not all(files):
            raise ValueError(
                "calib.mix.parts: either files {path, weight} or sources {<key>: settings, weight}"
            )
        if all(files) and self.rows is None:
            raise ValueError("calib.mix.rows: required for a mix of files")
        if not any(files):
            # each part reads, cuts and guards its own rows: a mix-level setting here would be silently ignored
            ignored = [k for k in _ROW_SETTINGS if getattr(self, k) != type(self).model_fields[k].default]
            if ignored:
                raise ValueError(
                    f"calib.mix.{ignored[0]}: does not apply to a mix of sources (each part reads and cuts its "
                    "own rows, and no held-out guard runs on them); set it on the parts, or mix files instead"
                )
            full = []
            for src, w in self.source_parts():
                if isinstance(src, MixSource):  # a mix of mixes has no meaning: each part forces a split
                    raise ValueError("calib.mix.parts: a mix cannot be a part of a mix")
                full.append({**src.to_dict(), "weight": w.numerator if w.denominator == 1 else str(w)})
            self.parts = full  # every default filled in, so the recorded recipe names the exact data
        return self

    def source_parts(self) -> list[tuple[CalibSource, Fraction]]:
        """The source parts, each parsed, with its weight (empty for a mix of files)."""
        out = []
        for p in self.parts:
            if isinstance(p, dict):
                cfg = dict(p)
                w = cfg.pop("weight", 1)
                out.append((parse_source(cfg), Fraction(w) if isinstance(w, str) else Fraction(w)))
        return out

    @property
    def images_source(self) -> bool:
        """True if a part reads images (the engine then keeps each sample's pixel inputs)."""
        return any(getattr(src, "images_source", False) for src, _ in self.source_parts())

    def pin(self) -> None:
        """Resolve every source part's moving revision (see :meth:`CalibSource.pin`); file parts need nothing."""
        parts: list[MixPart | dict[str, Any]] = []
        for p in self.parts:
            if isinstance(p, dict):
                src = parse_source({k: v for k, v in p.items() if k != "weight"})
                src.pin()
                p = {**src.to_dict(), "weight": p["weight"]}
            parts.append(p)
        self.parts = parts

    def pending(self, model_dir: str | Path) -> str | None:
        """What the source parts must generate first, one part's message per pending part; None if none is.

        A mix of files never generates: its parts are paths, not sources.
        """
        msgs = [m for src, _ in self.source_parts() if (m := src.pending(model_dir))]
        return "; ".join(msgs) if msgs else None

    def generate(self, model_dir: str | Path) -> dict[str, Any]:
        """Generate every source part that is pending; returns ``{parts: [{source, cache}, ...]}``.

        ``cache`` is the part's own :meth:`CalibSource.generate` record (a ``path`` and a ``sha256``), so
        ``build.json`` names every cache this stage generated (a part already in its cache is read but not
        recorded, as for a plain ``selfgen`` cache hit).
        """
        parts = []
        for src, _ in self.source_parts():
            if src.pending(model_dir) is not None:
                parts.append({"source": src.name, "cache": src.generate(model_dir)})
        return {"parts": parts}

    def read(self, model_dir: str | Path) -> Rows:
        """The mixed file's rows (written, or reused from the cache).

        Every row each part file holds is linted as that file (:func:`qda_quant.calib.template.lint_rows`), not only
        the rows the mix picks: a defect in a part does not hide behind the selection.

        Raises:
            Refused: a mix of sources, which the engine reads as samples (`source_parts`), not as rows; or a part
                row in another model family's chat template.
        """
        if self.source_parts():
            raise Refused("calib mix: a mix of sources gives samples, not rows (engine.calib.load_samples)")
        from .mix import NGramGuard, mix, read_source

        guard = NGramGuard(self.heldout, self.ngram) if self.heldout else None
        assert self.rows is not None  # a mix of files requires rows (checked by _one_kind)
        parts = [
            (p.path, Fraction(p.weight) if isinstance(p.weight, str) else p.weight)
            for p in self.parts
            if isinstance(p, MixPart)  # the check above means every part is a file
        ]
        path = mix(
            parts, self.rows, self.mix_seed, guard=guard, out=self.out, inspect=self._lint_part(model_dir)
        )
        rows, tag = read_source(path)
        return Rows(rows, tag)

    def _lint_part(self, model_dir: str | Path) -> Callable[[Path, list[dict]], None]:
        """The `mix` hook linting each part file's raw rows; the model's tokenizer is loaded once, if a row needs it."""
        from .template import has_markers, lint_rows

        loaded: list[Any] = []

        def lint(path: Path, rows: list[dict]) -> None:
            ids = any(isinstance(r, dict) and ("input_ids" in r or "prompt_ids" in r) for r in rows)
            if (ids or has_markers(rows)) and not loaded:
                from ..intake.tokenizer import load_tokenizer

                try:
                    loaded.append(load_tokenizer(model_dir))
                except Exception as e:  # noqa: BLE001 - the ids still calibrate; the record says why unlinted
                    self.note_lint(
                        {"file": str(path), "check": f"not run: the model's tokenizer cannot be loaded ({e})"}
                    )
                    return
            found = lint_rows(rows, loaded[0] if loaded else None, str(path))
            if (why := found.refusal(str(path))) is not None:
                raise Refused(why)
            self.note_lint(found.lint_record())

        return lint


__all__ = [
    "CalibSource",
    "CorpusSource",
    "FileSource",
    "HFSource",
    "MixPart",
    "MixSource",
    "Rows",
    "SOURCES",
    "SelfgenSource",
    "SliceFile",
    "SliceSource",
    "VLMSource",
    "parse_source",
    "read_file",
]
