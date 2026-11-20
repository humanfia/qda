"""One bf16 GGUF template per llama.cpp search: made (or taken) once, handed to every GGUF build.

A GGUF build fills its file from a bf16 template, llama.cpp's own conversion of the source
(`convert_hf_to_gguf.py --outtype bf16`). A build that is not given one converts the whole source itself, in its
own output directory, and deletes the copy when it ends: about 54 GB written per candidate for a 27B, which the
build's reservation (the `.gguf` it writes) never counted. Within a run the source and the pinned converter are
fixed, so every candidate's template is the same bytes. So the run holds one:

* `judge.llamacpp.template`, when the operator names one, is used as is. It sits outside the run directory, so
  the run's cap does not count it. `qda-agent prepare` names the bf16 GGUF it already makes for the engine floor
  (and candidate 0 reads). Its header must be a bf16 GGUF. If the record a prepared template carries beside it
  (`<file>.json`, `RECORD`) exists, it must name this source and the run's converter commit.
* Otherwise the judge makes it once at start (`prepare`), on the host, with the run's pinned checkout
  (`judge.llamacpp.llama_cpp`) and the trusted build's interpreter, into `<run_dir>/judge/gguf-template/`.
  Its estimated size is reserved against `judge.max_run_bytes` first: when it does not fit, the start is refused
  before a byte is written. It is sealed read-only, carries the same record, and `run_bytes` counts it once, as
  it counts every file of the run.

Every GGUF build the judge admits (`qda build`, trusted or local) gets the template when it fits: the build
reads the template's source, and its own converter, if it names one, is at the template's commit. Before
the plan is read, the recipe's exporter is rewritten with `template:` set and its model pinned to the resolved
source. So the plan prices the file exactly, and the build cannot be pointed at another model afterwards.

A recipe that names its own `template` is refused, unless it names this file. A template supplies every tensor
that no step changed. A foreign template would put bytes into a receipted build that its recorded source never
held, and checking that a 54 GB file is a faithful conversion means converting again. A GGUF build the run's
template does not fit (another source, or another converter commit) still converts its own copy. Its
reservation then includes that copy's estimated size, before the build starts.

The writer reads a supplied template exactly as it reads one it made itself. `GGUFWriter` opens the same
`file.Template`; only `own_template` differs, and that decides only whether `finish` deletes the file. So a
build's file is byte-identical either way when the template bytes are, and they are: same source, same
converter commit, same interpreter.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .host import JudgeHost

__all__ = [
    "DIR",
    "FILE",
    "Handed",
    "Template",
    "estimate_bytes",
    "for_build",
    "gguf_settings",
    "prepare",
    "record_path",
    "run_template_bytes",
    "source_dir",
]

_log = logging.getLogger(__name__)

#: The run's own template directory, under the judge's directory.
DIR = "gguf-template"
#: The run's own template file in it.
FILE = "template-bf16.gguf"
#: What a template's size estimate adds to its tensors: the vocabulary and metadata, and the tensor table.
HEADER_MARGIN = 64 << 20
#: Model files that hold weights, which the estimate counts by their tensors rather than their size.
_WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


@dataclass(frozen=True)
class Template:
    """The run's one bf16 GGUF template.

    Attributes:
        path: The file.
        source: The resolved source directory it converts (a build of another source makes its own).
        commit: The llama.cpp commit of the converter that made it.
        made: True when the run made it (it is in the run directory and counts in the cap); False when the
            operator named it (`judge.llamacpp.template`), outside the run.
        size: Its bytes.
    """

    path: Path
    source: str
    commit: str | None
    made: bool
    size: int


@dataclass(frozen=True)
class Handed:
    """What admission does to one build's recipe.

    Attributes:
        raw: The recipe the plan is read from (with the template set when it is handed).
        forced: Host-authored overrides the build applies last: the exporter with the template set, when it
            is handed (the caller pins `model` to the resolved source beside it).
        extra: Bytes the build's reservation adds: its own template's estimate when it converts one.
    """

    raw: dict[str, Any]
    forced: list[str]
    extra: int


def record_path(template: Path) -> Path:
    """The record beside a template: `<file>.json`, naming its source and converter commit."""
    return template.with_name(template.name + ".json")


def source_dir(cfg: Any) -> Path | None:
    """The original model's local directory, which the run's template converts: `judge.anchor`, else `judge.model`.

    It is named as the records beside a template and an importance matrix name it (`workspace.model_name`),
    so a record and the run compare as strings.
    """
    from qda_agent.workspace import model_name

    for named in (getattr(cfg, "anchor", None), getattr(cfg, "model", None)):
        if named and Path(named).expanduser().is_dir():
            return Path(model_name(named))
    return None


def estimate_bytes(source: Path) -> int:
    """An upper bound on the bytes of `source`'s bf16 GGUF, from its safetensors headers (no tensor is read).

    The converter writes a matrix in bf16 and a vector (a norm, a bias) in F32, so each tensor counts 2 or 4 bytes
    a value. An FP8 source therefore counts twice its own size. The vocabulary and metadata count as the model's
    non-weight files plus `HEADER_MARGIN`. A source whose headers cannot be read counts twice its files.
    """
    from .checks import CheckpointError, loaded

    meta = 0
    files = 0
    try:
        for path in source.iterdir():
            if path.is_file():
                size = path.stat().st_size
                files += size
                if path.suffix not in _WEIGHTS:
                    meta += size
    except OSError:
        return 0
    try:
        values = 0
        for _shard, header, _start in loaded(source).values():
            shape = header.get("shape") or []
            values += math.prod(shape) * (4 if len(shape) <= 1 else 2)
    except (CheckpointError, OSError, TypeError, ValueError):
        return 2 * files + HEADER_MARGIN
    return values + meta + HEADER_MARGIN


def run_template_bytes(cfg: Any) -> int:
    """Bytes the run's own template adds to the run directory, once; 0 when the run makes none there.

    A llama.cpp search with a local source and no `judge.llamacpp.template` makes one at start (`prepare`). The
    disk-cap line and doctor's disk check subtract it from the cap before counting builds.
    """
    llamacpp = getattr(cfg, "llamacpp", None)
    if getattr(cfg, "engine", None) != "llama.cpp" or llamacpp is None or llamacpp.template is not None:
        return 0
    source = source_dir(cfg)
    return 0 if source is None else estimate_bytes(source)


def gguf_settings(raw: dict[str, Any]) -> dict[str, Any] | None:
    """The `gguf` exporter's settings a recipe names (`gguf`, `{gguf: null}` or `{gguf: {...}}`), else None."""
    exporter = raw.get("exporter")
    if exporter == "gguf":
        return {}
    if isinstance(exporter, dict) and len(exporter) == 1 and "gguf" in exporter:
        settings = exporter["gguf"]
        if settings is None:
            return {}
        return dict(settings) if isinstance(settings, dict) else None
    return None


def _commit(root: Path | str | None) -> str | None:
    """The llama.cpp commit of checkout `root`, or None when it is not one or cannot be read."""
    if root is None:
        return None
    from qda_quant.gguf.llama_cpp import commit_of
    from qda_quant.step import Refused

    try:
        return commit_of(root)
    except Refused:
        return None


def for_build(host: JudgeHost, raw: dict[str, Any], source: str, read: str, env: dict[str, str]) -> Handed:
    """Hand the run's template and importance matrix to one GGUF build, or reserve the template it will make.

    Args:
        host: The judge (its template `host.gguf_template` and matrix `host.imatrix`).
        raw: The build's recipe, snapshotted and overridden.
        source: The host identity of the model it reads (`build_receipt._source`).
        read: The path the build is handed to read it with.
        env: The build's environment, for the `QDA_LLAMA_CPP` its writer would convert with.

    Raises:
        Refused: The recipe names a `template` or an `imatrix` other than the run's.
    """
    from ..sched import Refused

    settings = gguf_settings(raw)
    if settings is None:
        return Handed(raw, [], 0)
    t: Template | None = getattr(host, "gguf_template", None)
    matrix: Path | None = getattr(host, "imatrix", None)
    named = settings.get("template")
    if named is not None and (t is None or str(named) != str(t.path)):
        raise Refused(
            "exporter.gguf.template: the judge hands every GGUF build the run's bf16 template, and a recipe "
            "may not name its own (a template supplies every tensor no step changed); drop the setting"
        )
    named_matrix = settings.get("imatrix")
    if named_matrix is not None and (matrix is None or str(named_matrix) != str(matrix)):
        raise Refused(
            "exporter.gguf.imatrix: the judge hands every GGUF build the run's importance matrix, and a "
            "recipe may not name its own (the matrix weighs the tensors a build takes from the template); "
            "drop the setting"
        )
    handed = {**settings, **({"imatrix": str(matrix)} if matrix is not None else {})}
    if t is not None and source and source == t.source and _converts_alike(settings, env, t):
        exporter = {"gguf": {**handed, "template": str(t.path)}}
        return Handed({**raw, "exporter": exporter}, [f"exporter={json.dumps(exporter)}"], 0)
    # The build converts the source itself: its template sits beside its output until it ends. Prefer the
    # source's own directory: behind a bind, the path the build is handed is the (near-empty) mount point.
    local = next((Path(p) for p in (source, read) if p and Path(p).is_dir()), source_dir(host.cfg))
    forced: list[str] = []
    if matrix is not None:  # the matrix does not depend on which template file the build reads
        exporter = {"gguf": handed}
        raw, forced = {**raw, "exporter": exporter}, [f"exporter={json.dumps(exporter)}"]
    return Handed(raw, forced, estimate_bytes(local) if local is not None else host.max_bytes)


def _converts_alike(settings: dict[str, Any], env: dict[str, str], t: Template) -> bool:
    """Whether the build's own converter, if it names one, is at the template's commit.

    A build that names no checkout (`exporter.gguf.llama_cpp`, `QDA_LLAMA_CPP`) could not convert at all, so the
    run's template is the only one it can have. A checkout whose commit cannot be read decides nothing either.
    """
    named = settings.get("llama_cpp") or env.get("QDA_LLAMA_CPP")
    commit = _commit(Path(str(named)).expanduser()) if named else None
    return commit is None or t.commit is None or commit == t.commit


def prepare(host: JudgeHost) -> Template | None:
    """Take or make the run's template and take its importance matrix at start, so every build may read them.

    Only a llama.cpp search has either. Without `judge.llamacpp.template` and without a local source (a Hub
    id), there is nothing to convert on the host, and each GGUF build converts (and reserves) its own; the
    importance matrix, when the config names one, is still handed to every build.

    Raises:
        ValueError: The named template is not a bf16 GGUF, or its record names another source or converter
            commit; the named matrix is not a `llama-imatrix` GGUF, or its record names another source or
            commit; the run's own template does not fit the disk cap (nothing is written), or the converter
            fails.
    """
    cfg = host.cfg
    llamacpp = cfg.llamacpp
    if cfg.engine != "llama.cpp" or llamacpp is None:
        return None
    commit = _commit(llamacpp.llama_cpp)
    source = source_dir(cfg)
    matrix = None if llamacpp.imatrix is None else _given_imatrix(Path(llamacpp.imatrix), source, commit)
    t: Template | None
    if llamacpp.template is not None:
        t = _given(Path(llamacpp.template), source, commit)
    elif source is None:
        _log.warning(
            "judge.llamacpp: the run's model is not a local directory, so no bf16 template is made at start; "
            "each GGUF build converts its own and reserves it"
        )
        t = None
    else:
        t = _make(host, source, Path(llamacpp.llama_cpp), commit)
    readable = [path for path in ((t.path if t is not None else None), matrix) if path is not None]
    if readable:
        sched = host.sched
        update: dict[str, Any] = {"readable": [*sched.cfg.readable, *readable]}
        # a bind source inside the run dir must be a host-written area
        inside = [path for path in readable if path.is_relative_to(host.run_dir)]
        if inside:
            update["host_written"] = [*sched.cfg.host_written, *inside]
        sched.cfg = sched.cfg.model_copy(update=update)
    host.gguf_template = t
    host.imatrix = matrix
    return t


def _given(path: Path, source: Path | None, commit: str | None) -> Template:
    """The operator's template, checked as a bf16 GGUF of this source at this converter commit, never rewritten."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"judge.llamacpp.template: {path} is not a file")
    try:
        from qda_quant.gguf import file, rule

        tp = file.Template(path)
        odd = next(
            ((n, t.tensor_type) for n, t in tp.tensors.items() if t.tensor_type not in (rule.BF16, rule.F32)),
            None,
        )
    except Exception as e:  # noqa: BLE001 -- any unreadable header is the same refusal
        raise ValueError(f"judge.llamacpp.template: {path} is not a readable GGUF ({e})") from None
    if odd is not None:
        name, kind = odd
        raise ValueError(
            f"judge.llamacpp.template: {path} stores {name} as ggml type {kind}; the template is the "
            "converter's bf16 output (weights bf16, vectors f32; --outtype bf16)"
        )
    record: dict[str, Any] = {}
    if record_path(path).is_file():
        try:
            record = json.loads(record_path(path).read_text())
        except (OSError, ValueError):
            raise ValueError(f"judge.llamacpp.template: {record_path(path)} is not a JSON record") from None
    named_source = record.get("source")
    if source is not None and named_source is not None and named_source != str(source):
        raise ValueError(
            f"judge.llamacpp.template: {path} converts {named_source}, not the run's model {source}"
        )
    named_commit = record.get("llama_cpp")
    if commit is not None and named_commit is not None and named_commit != commit:
        raise ValueError(
            f"judge.llamacpp.template: {path} was converted by llama.cpp {named_commit}, not the run's "
            f"{commit}"
        )
    owner = str(source) if source is not None else str(named_source or "")
    return Template(path, owner, named_commit or commit, False, path.stat().st_size)


def _given_imatrix(path: Path, source: Path | None, commit: str | None) -> Path:
    """The operator's importance matrix, checked as a `llama-imatrix` GGUF of this source at this commit.

    The file is used as is, never rewritten: it is a measurement, not something the run can make cheaper. A
    record beside it (`qda-agent prepare` writes one: the source, its digest and the llama.cpp commit) must
    name this model and commit when it exists; an unknown file is checked by its own header alone.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"judge.llamacpp.imatrix: {path} is not a file")
    try:
        import gguf

        field = gguf.GGUFReader(str(path)).fields.get("general.type")
        kind = None if field is None else field.contents()
    except Exception as e:  # noqa: BLE001 -- any unreadable header is the same refusal
        kind = f"unreadable ({e})"
    if kind != "imatrix":
        raise ValueError(
            f"judge.llamacpp.imatrix: {path} is not a llama-imatrix GGUF (general.type is {kind!r}, not "
            "'imatrix'; write one with llama-imatrix -m TEMPLATE -f TEXT -o FILE)"
        )
    record: dict[str, Any] = {}
    if record_path(path).is_file():
        try:
            record = json.loads(record_path(path).read_text())
        except (OSError, ValueError):
            raise ValueError(f"judge.llamacpp.imatrix: {record_path(path)} is not a JSON record") from None
    named_source = record.get("source")
    if source is not None and named_source is not None and named_source != str(source):
        # A record without `source_identity` is one a prepare wrote before its `source` named the model as
        # the judge does (`repo@revision`, `local <path> ...`): the matrix may be this model's own.
        again = (
            "; its record predates the model naming the run checks: remove it and run `qda-agent prepare` "
            "again with the --model and --out that made it, which measures it again and rewrites the record"
            if "source_identity" not in record
            else ""
        )
        raise ValueError(
            f"judge.llamacpp.imatrix: {path} was measured on {named_source}, not the run's model {source}{again}"
        )
    named_commit = record.get("llama_cpp")
    if commit is not None and named_commit is not None and named_commit != commit:
        raise ValueError(
            f"judge.llamacpp.imatrix: {path} was measured by llama.cpp {named_commit}, not the run's {commit}"
        )
    return path


def _make(host: JudgeHost, source: Path, root: Path, commit: str | None) -> Template:
    """Convert `source` once into the run's template directory, after reserving its size against the cap."""
    from ..sched import Refused
    from ..sched.snapshot import remove_tree
    from .build_receipt import program
    from .prune import check_room, disk_lock

    try:
        interpreter, _paths = program(host)
    except ValueError as e:
        raise ValueError(f"judge.llamacpp: the run's bf16 template needs the build's Python: {e}") from None
    script = root / "convert_hf_to_gguf.py"
    size = estimate_bytes(source)
    where = host.jdir / DIR
    path, partial = where / FILE, where / f".{FILE}.partial"
    with disk_lock(host):
        try:
            check_room(host, size)
        except Refused as e:
            raise ValueError(
                f"judge.max_run_bytes: the run's bf16 GGUF template of {source} (about {size / 1e9:.1f} GB, made "
                f"once and shared by every build) does not fit: {e}. Name a prepared one with "
                "judge.llamacpp.template (`qda-agent prepare` makes it), or raise judge.max_run_bytes"
            ) from None
        where.mkdir(mode=0o755)
        # The writer's own conversion (`gguf.writer.make_template`), run by the build's interpreter, with the
        # environment a trusted build gives it (no PYTHONPATH), so the bytes are the ones a build would make.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"CUDA_VISIBLE_DEVICES": ""}
        cmd = [interpreter, str(script), str(source), "--outtype", "bf16", "--outfile", str(partial)]
        _log.info("judge: converting %s once for every GGUF build (about %.1f GB)", source, size / 1e9)
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=where)
        if r.returncode or not partial.is_file():
            remove_tree(where)
            why = r.stderr[-2000:] if r.returncode else "it exited 0 and wrote no file"
            raise ValueError(
                f"judge.llamacpp: convert_hf_to_gguf.py could not make the run's bf16 template: {why}"
            )
        partial.rename(path)
        path.chmod(0o444)
        record = {
            "source": str(source),
            "llama_cpp": commit,
            "interpreter": interpreter,
            "bytes": path.stat().st_size,
        }
        record_path(path).write_text(json.dumps(record, indent=1) + "\n")
        record_path(path).chmod(0o444)
        where.chmod(0o555)
    return Template(path, str(source), commit, True, path.stat().st_size)
