r"""Refuse to score text when the engine's tokenizer does not tokenize the way the checkpoint says.

Under transformers 5, a repository that declares ``LlamaTokenizerFast`` over a ByteLevel ``tokenizer.json``
(DeepSeek-R1-Distill-Llama-8B and its vendor repositories) is built as a Metaspace/ByteFallback tokenizer:
``encode('Hello world, this is a test.\\nNext line.')`` gives 8 ids instead of 11, and the correct ids decode
with literal ``Ġ``/``Ċ``. Stock vLLM 0.33 inherits it, so every benchmark that grades generated
text on such a checkpoint would score the wrong text while the run looks healthy. The ids in
``tokenizer.json`` are the checkpoint's own; comparing against them catches the class the engine picked.

The check is one function, :func:`tokenizer_refusal`, shared by the qda-kl eval pipeline and the qda-agent
LiveCodeBench suite.

The fix is one rule, also kept here: the tokenizer a checkpoint actually uses. ``qda build`` decides whether the
declared class reproduces the serialized ``tokenizer.json`` and, when it does not, writes
``PreTrainedTokenizerFast`` and records the correction in ``build.json`` (``tokenizer_config``:
``declared -> written``). :func:`read_tokenizer_config` returns that same decision for any checkpoint, by calling
the build's own function; ``qda-kl ref`` and ``qda-kl score`` hand the engine a tokenizer corrected by it
(:func:`engine_tokenizer`), ``qda-kl ref`` keeps that same tokenizer inside the reference
(:func:`save_tokenizer`), and the judge's anchor check accepts exactly that correction
(:func:`tokenizer_config_difference`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, NamedTuple

#: Text with a newline and punctuation, which every tokenizer class splits differently.
PROBE = "Hello world, this is a test.\nNext line."
#: What a reference without a tokenizer copy (and without the directory it was built from) is missing, and the
#: one fix; the readers that refuse say the same sentence.
REBUILD = "rebuild the reference (`qda-kl ref` copies the anchor's tokenizer into it)"


def _tokenizer_json(model: str | Path, revision: str | None) -> tuple[Path | None, bool]:
    """The checkpoint's ``tokenizer.json``, and whether `model` is a local directory.

    A local snapshot names the file directly. A Hub id is read from the local Hugging Face cache (``HF_HOME``)
    at `revision`; nothing is downloaded, so a missing cache entry is the caller's to refuse.
    """
    local = Path(model)
    if local.is_dir():
        path = local / "tokenizer.json"
        return (path if path.is_file() else None), True
    from huggingface_hub import try_to_load_from_cache
    from qda_quant.intake.model_ref import parse

    ref = parse(model)  # ``org/name@rev`` names the same snapshot as ``org/name`` with ``revision=rev``
    hit = try_to_load_from_cache(
        repo_id=ref.name, filename="tokenizer.json", revision=revision or ref.revision
    )
    return (Path(hit) if isinstance(hit, str) else None), False


def tokenizer_refusal(
    tokenizer: Any, model_dir: str | Path, *, revision: str | None = None, probe: str = PROBE
) -> str | None:
    """A refusal naming both tokenizations when the engine's tokenizer differs from the checkpoint's.

    Args:
        tokenizer: the tokenizer the engine built (vLLM's ``get_tokenizer()`` or a transformers one).
        model_dir: the checkpoint: a local snapshot, or a Hub id resolved from the local cache.
        revision: the Hub revision to read the cached ``tokenizer.json`` at; default the cached main revision.
        probe: the fixed text encoded both ways, without special tokens.

    Returns:
        None when the ids agree, or when a local snapshot keeps its tokenizer in another format and so has
        nothing to compare; else a message with both id lists, the tokenizer classes and the remedy — or, for
        a Hub id whose ``tokenizer.json`` is not cached, a refusal to score what cannot be checked.
    """
    path, local = _tokenizer_json(model_dir, revision)
    if path is None:
        if local:
            return None
        return (
            f"cannot check {model_dir!r} (revision {revision or 'main'}): its tokenizer.json is not in the local "
            "Hugging Face cache, so the engine's tokenization cannot be compared with the checkpoint's and the "
            "benchmark's text cannot be trusted. Download the checkpoint, or pass its local snapshot as the "
            "model, and rerun."
        )
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(str(path))
    want = reference.encode(probe, add_special_tokens=False).ids
    encoded = tokenizer.encode(probe, add_special_tokens=False)
    # a transformers tokenizer returns ids; a `tokenizers` one returns an Encoding
    got = list(getattr(encoded, "ids", encoded))
    if got == want:
        return None
    return (
        f"the engine's tokenizer ({type(tokenizer).__name__}) encodes {probe!r} as {len(got)} ids {got}, but "
        f"the checkpoint's tokenizer.json gives {len(want)} {want}: the class tokenizer_config.json declares "
        "rebuilds the stored pipeline (under transformers 5, LlamaTokenizerFast over a ByteLevel tokenizer.json "
        "becomes Metaspace/ByteFallback), so text tokenized or generated with it is not the text the benchmark "
        "scores. qda-kl corrects this itself only when the engine is given no explicit tokenizer and the "
        "pipeline is ByteLevel. Otherwise give the engine a tokenizer directory whose tokenizer_config.json "
        "declares PreTrainedTokenizerFast over the same tokenizer.json, through vLLM's `tokenizer` argument: "
        f'`qda-kl ref ... --serve-args \'{{"tokenizer": "<dir>"}}\'`, or `serve_args: {{tokenizer: <dir>}}` '
        "under the scorer or suite in the config (`scorer.topk.serve_args`), and rerun."
    )


#: The class ``qda build`` writes when the declared one does not reproduce ``tokenizer.json``: it loads the
#: serialized pipeline exactly as stored (``qda_quant.intake.tokenizer.load_tokenizer``).
SERIALIZED_CLASS = "PreTrainedTokenizerFast"

#: Directory of named chat templates a checkpoint may carry beside its tokenizer files (the judge's anchor files).
_TEMPLATE_DIR = "additional_chat_templates"


class TokenizerConfig(NamedTuple):
    """A checkpoint's ``tokenizer_config.json`` and ``qda build``'s decision about its declared class.

    ``record`` is what ``qda build`` writes to ``build.json["tokenizer_config"]`` for this checkpoint as a source:
    ``declared``, ``serialized_ok`` (True: the declared class reproduces ``tokenizer.json``; False: it does not,
    and ``written`` is the corrected class; None: no comparison applies, with a ``reason``).
    """

    declared: dict
    record: dict

    @property
    def corrected(self) -> bool:
        """Whether the build would write another class than the declared one."""
        return self.record["written"] != "unchanged"

    @property
    def effective(self) -> dict:
        """The config the checkpoint is used with: the declared one, with the build's class correction applied.

        Equal, as JSON, to the ``tokenizer_config.json`` a ``qda build`` of this checkpoint writes.
        """
        if not self.corrected:
            return dict(self.declared)
        return {**self.declared, "tokenizer_class": self.record["written"]}


def read_tokenizer_config(model_dir: str | Path) -> TokenizerConfig | None:
    """`model_dir`'s tokenizer config and ``qda build``'s class decision for it; None without the file.

    The decision is the build's own function (``qda_quant.intake.tokenizer.tokenizer_config_record``), called
    unchanged, so the build, ``qda-kl`` and the judge cannot disagree about one checkpoint.

    Raises:
        Refused: the config is not a JSON object, or the declared class cannot be compared with the serialized
            pipeline (the build refuses such a source too).
    """
    from qda_quant.intake.tokenizer import tokenizer_config_record
    from qda_quant.step import Refused

    path = Path(model_dir) / "tokenizer_config.json"
    if not path.is_file():
        return None
    record = tokenizer_config_record(Path(model_dir))
    declared = json.loads(path.read_text())
    if not isinstance(declared, dict):
        raise Refused(f"{path}: not a JSON object")
    return TokenizerConfig(declared, record)


def tokenizer_dir(model: str | Path, *, revision: str | None = None, fetch: bool = False) -> Path | None:
    """The local directory holding `model`'s tokenizer files, or None when there is none to read.

    A local directory is itself. A Hub id (``org/name``, ``org/name@rev``) is its snapshot in the Hugging Face
    cache at `revision`, so a Hub id and its downloaded snapshot get the same decision. With `fetch`, files the
    cache lacks are downloaded first, as the engine about to load the model would download them: the tokenizer
    files and ``config.json`` (transformers picks a class from its ``model_type`` when the declared one does not
    fit). Without it nothing is downloaded.
    """
    local = Path(str(model)).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    from qda_quant.intake.model_ref import parse
    from qda_quant.output.export import SMALL_FILES

    ref = parse(model)
    want: dict[str, Any] = dict(
        repo_id=ref.name, revision=revision or ref.revision, allow_patterns=["config.json", *SMALL_FILES]
    )
    snapshot = None
    for local_only in (True, False) if fetch else (True,):  # the cache first: no network when it has them
        try:
            snapshot = Path(snapshot_download(**want, local_files_only=local_only))
        except Exception:  # noqa: BLE001 - not cached / offline / no such repo: whatever was found stands
            continue
        if (snapshot / "tokenizer_config.json").is_file() and (snapshot / "tokenizer.json").is_file():
            break
    return snapshot


def _read_config(
    model: str | Path, revision: str | None, fetch: bool
) -> tuple[Path | None, TokenizerConfig | None]:
    """`model`'s local tokenizer directory and ``qda build``'s class decision for it, or None for either."""
    from qda_quant.step import Refused

    src = tokenizer_dir(model, revision=revision, fetch=fetch)
    try:
        return src, (read_tokenizer_config(src) if src is not None else None)
    except Refused:
        return src, None


def _write_tokenizer(src: Path, config: TokenizerConfig, into: Path) -> None:
    """Copy `src`'s tokenizer files into `into`, with ``qda build``'s class correction applied."""
    from qda_quant.output.export import SMALL_FILES

    into.mkdir(parents=True)
    # a corrected copy keeps no config.json: transformers would pick a class for some model types (qwen2)
    # whatever the config declares, so the correction would not survive; an uncorrected copy is the directory
    # the engine read, config.json included, and must tokenize and stop exactly as it did
    names = SMALL_FILES if config.corrected else ("config.json", *SMALL_FILES)
    for name in names:
        if (src / name).is_file():
            shutil.copyfile(src / name, into / name)
    if (src / _TEMPLATE_DIR).is_dir():
        shutil.copytree(src / _TEMPLATE_DIR, into / _TEMPLATE_DIR)
    (into / "tokenizer_config.json").write_text(json.dumps(config.effective, indent=2) + "\n")


def correct_tokenizer(
    model: str | Path, into: Path, *, revision: str | None = None, fetch: bool = True
) -> dict:
    """Write `model`'s tokenizer files into `into` with ``qda build``'s class correction applied.

    The files are the ones a build copies (``qda_quant.output.export.SMALL_FILES``, plus a named-template
    directory), so the chat template and special tokens are the checkpoint's own; only ``tokenizer_class``
    changes, written as the build writes it. A Hub id's tokenizer files are fetched (`fetch`) as the engine
    would fetch them.

    Returns:
        The build's record when it corrects the class (and `into` was written), else ``{}`` and nothing is
        written: no tokenizer config, a class that already reproduces ``tokenizer.json``, a pipeline the rule
        does not cover, or one it cannot compare (the engine's own check then decides).
    """
    src, config = _read_config(model, revision, fetch)
    if src is None or config is None or not config.corrected:
        return {}
    _write_tokenizer(src, config, into)
    return dict(config.record)


def save_tokenizer(
    model: str | Path, into: Path, *, revision: str | None = None, fetch: bool = True
) -> tuple[Path | None, dict]:
    """Write the tokenizer `model` is used with into `into`, so a reference carries its own copy of it.

    The same files and the same class correction as :func:`correct_tokenizer`, written whether the class is
    corrected or not: the copy is what the engine read, so it tokenizes and stops as the engine did.

    Returns:
        The directory the files were read from (None when there is none), and ``qda build``'s record for
        `model` (``{}`` and nothing written when there is no tokenizer config to copy: a reference then records
        only the path it was built from).
    """
    src, config = _read_config(model, revision, fetch)
    if src is None or config is None:
        return src, {}
    _write_tokenizer(src, config, into)
    return src, dict(config.record)


def engine_tokenizer(model: str | Path, args: dict[str, Any], into: Path) -> tuple[dict[str, Any], dict]:
    """The ``vllm.LLM`` arguments to open `model` with, and ``qda build``'s tokenizer record when corrected.

    An explicit ``tokenizer`` argument is the caller's and is kept. Otherwise, when the build would correct the
    declared class, the corrected tokenizer is written into `into` and named by ``tokenizer``; `into` must stay
    while the engine runs. The record is ``{}`` when nothing was corrected.
    """
    if args.get("tokenizer"):
        return dict(args), {}
    revision = args.get("tokenizer_revision") or args.get("revision")
    record = correct_tokenizer(model, into, revision=revision)
    if not record:
        return dict(args), {}
    return {**args, "tokenizer": str(into)}, record


def effective_tokenizer(model: str | Path, *, revision: str | None = None) -> Any:
    """The transformers tokenizer `model` is used with: its declared class, or its serialized pipeline when
    ``qda build`` would correct the class (what the engine gets from :func:`engine_tokenizer`).

    A Hub id is read from the local cache; one that is not cached is loaded by its declared class.
    """
    from qda_quant.intake.model_ref import parse
    from qda_quant.intake.tokenizer import load_tokenizer
    from transformers import AutoTokenizer

    src, config = _read_config(model, revision, fetch=False)
    if src is not None and config is not None and config.corrected:
        return load_tokenizer(src)
    ref = parse(model)
    revision = revision or ref.revision
    return AutoTokenizer.from_pretrained(ref.name, **({"revision": revision} if revision else {}))


def preload_refusal(
    source: str | Path, *, revision: str | None = None, probes: tuple[str | None, ...] = (), **options: Any
) -> str | None:
    """:func:`tokenizer_refusal` for the tokenizer transformers builds from `source`, before any engine loads.

    The engine builds its tokenizer the same way (``AutoTokenizer`` over its ``tokenizer`` argument, else the
    model), so a class that rebuilds the pipeline is refused here in seconds, not after the weights load.
    `options` are ``from_pretrained`` keywords (``trust_remote_code``); the probes after :data:`PROBE` are
    checked in turn. A tokenizer that cannot be loaded here is left to the engine's own check.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            str(source), **({"revision": revision} if revision else {}), **options
        )
    except Exception:  # noqa: BLE001 - the engine loads it its own way, and the check after it still runs
        return None
    for probe in (PROBE, *probes):
        if probe and (refusal := tokenizer_refusal(tok, source, revision=revision, probe=probe)):
            return refusal
    return None


def tokenizer_config_difference(mine: Path, anchor: Path) -> str | None:
    """Why `mine`'s ``tokenizer_config.json`` is not the anchor's once ``qda build``'s correction is applied.

    The caller has already found both ``tokenizer.json`` files byte-identical. Two differences are accepted,
    and nothing else:

    - the JSON's layout (whitespace, key order): ``qda build`` rewrites the file when it corrects it;
    - ``tokenizer_class``, when one side declares ``PreTrainedTokenizerFast`` (the one class the build writes)
      and ``qda build``'s record over the other side compared the declared class with the serialized
      ByteLevel pipeline: it either rebuilt it, so the build corrects it to ``PreTrainedTokenizerFast``
      (``serialized_ok`` False), or reproduced it under this transformers, so the correction is a no-op here
      (``serialized_ok`` True). Either way both sides tokenize as the shared ``tokenizer.json`` does.

    Returns:
        None, or the first field that differs (with both values for ``tokenizer_class``, and the record).
    """
    from qda_quant.step import Refused

    a, b = mine / "tokenizer_config.json", anchor / "tokenizer_config.json"
    if a.is_file() != b.is_file():
        return "only on one side"
    if not a.is_file():
        return None
    try:
        left, right = json.loads(a.read_text()), json.loads(b.read_text())
    except (OSError, ValueError) as error:
        return f"not readable JSON ({error})"
    if left == right:
        return None
    if not isinstance(left, dict) or not isinstance(right, dict):
        return "not a JSON object"
    if (field := _json_difference(left, right, skip="tokenizer_class")) is not None:
        return field
    mine_class, anchor_class = left.get("tokenizer_class"), right.get("tokenizer_class")
    told = f"tokenizer_class {mine_class!r}, the anchor's {anchor_class!r}"
    if SERIALIZED_CLASS not in (mine_class, anchor_class):
        return f"{told}; only qda build's correction to {SERIALIZED_CLASS!r} is accepted"
    other = anchor if mine_class == SERIALIZED_CLASS else mine
    try:
        config = read_tokenizer_config(other)
    except Refused as why:
        return f"{told}, and qda build's check of {other.name}'s declared class cannot run ({why})"
    record = config.record if config is not None else {"serialized_ok": None}
    if record["serialized_ok"] is None:
        return (
            f"{told}, and qda build does not correct {other.name}'s declared class (record {record}); only its "
            f"correction to {SERIALIZED_CLASS!r} over a ByteLevel tokenizer.json is accepted"
        )
    return None


def _json_difference(mine: Any, theirs: Any, *, skip: str, path: str = "") -> str | None:
    """The first field two JSON documents differ in, with the top-level key `skip` dropped, or None."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        for k in sorted(set(mine) | set(theirs)):
            if not path and k == skip:
                continue
            if k not in mine or k not in theirs:
                return f"{path}{k} is only on one side"
            if (why := _json_difference(mine[k], theirs[k], skip=skip, path=f"{path}{k}.")) is not None:
                return why
        return None
    if isinstance(mine, list) and isinstance(theirs, list) and len(mine) == len(theirs):
        for i, (x, y) in enumerate(zip(mine, theirs, strict=True)):
            if (why := _json_difference(x, y, skip=skip, path=f"{path}{i}.")) is not None:
                return why
        return None
    return None if mine == theirs else path.rstrip(".")
