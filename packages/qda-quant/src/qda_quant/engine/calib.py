"""Tokenize and select calibration samples from recipe sources."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch

from ..calib.template import lint_rows
from ..step import Refused, probe_refusal, remote_code_refusal

if TYPE_CHECKING:
    from ..calib.sources import CalibSource, MixSource


class Sample(NamedTuple):
    """One calibration sample: its token ids, and where its scored part starts.

    A step that scores next-token predictions scores positions ``prompt_len - 1 .. len(ids) - 2``. An image
    sample also carries the processor's other model inputs (``pixel_values`` and the like), which the
    calibration forward passes to the model with the ids.
    """

    ids: torch.Tensor
    prompt_len: int
    inputs: dict[str, torch.Tensor] | None = None


def fingerprint(model_dir: str | Path) -> str | None:
    """sha256 hex digest of the model's ``tokenizer.json`` (None if it has none)."""
    p = Path(model_dir) / "tokenizer.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def cut(rows: Iterable[list[int]], length: int, n: int | None = None) -> list[list[int]]:
    """Samples of `length` ids: long rows truncated, short rows concatenated and cut."""
    out: list[list[int]] = []
    buf: list[int] = []
    for ids in rows:
        if len(ids) >= length:
            out.append(ids[:length])
        else:
            buf += ids
            while len(buf) >= length:
                out.append(buf[:length])
                buf = buf[length:]
        if n is not None and len(out) >= n:
            break
    return out


def pieces(rows: Iterable[list[int]], length: int, n: int | None = None) -> list[list[int]]:
    """Every row cut into consecutive pieces of `length` ids, the last one shorter; rows and pieces in order."""
    out: list[list[int]] = []
    for ids in rows:
        out += [ids[i : i + length] for i in range(0, len(ids), length)]
        if n is not None and len(out) >= n:
            break
    return out


def _render_chat(render: Callable[..., str], messages: list[dict[str, Any]], **kw: Any) -> str:
    """Render a calibration chat; a final assistant turn is training data, not a serving request.

    A tokenizer validating serving (Mistral's) refuses it and takes the continuation form; others never retry.
    """
    try:
        return render(messages, **kw)
    except Exception:
        if not messages or messages[-1].get("role") != "assistant":
            raise
        return render(messages, continue_final_message=True, **kw)


def _tokenize(r: dict[str, Any], tok: Any) -> list[int]:
    if "messages" not in r:
        return tok(r["text"], add_special_tokens=False)["input_ids"]
    try:
        text = tok.apply_chat_template(r["messages"], tokenize=False)
    except Exception:
        if not r["messages"] or r["messages"][-1].get("role") != "assistant":
            raise
        return list(
            tok.apply_chat_template(
                r["messages"], tokenize=True, continue_final_message=True, return_dict=False
            )
        )
    return tok(text, add_special_tokens=False)["input_ids"]


def _token_rows(rows: list[dict[str, Any]], seqlen: int, chunks: bool) -> list[tuple[list[int], int]]:
    """(ids, prompt_len) of token-id rows: ``input_ids`` rows or traces, one kind per file."""
    if all("input_ids" in r for r in rows):
        if chunks:
            return [(ids, 1) for ids in pieces((list(r["input_ids"]) for r in rows), seqlen)]
        return [(list(r["input_ids"][:seqlen]), 1) for r in rows]
    if not all("prompt_ids" in r and "completion_ids" in r for r in rows):
        raise Refused("mixes input_ids rows with trace rows {prompt_ids, completion_ids}")
    if chunks:
        raise Refused("chunks: true does not apply to traces (a trace's scored part would be split)")
    out = []
    for r in rows:
        p = list(r["prompt_ids"])
        ids = (p + list(r["completion_ids"]))[:seqlen]
        if 1 <= len(p) < len(ids):
            out.append((ids, len(p)))
    return out


def load_samples(
    calib: CalibSource,
    model_dir: str | Path,
    tokenizer: Any = None,
    *,
    images: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[Sample]:
    """The calibration samples of `calib` for the model at `model_dir`, in the order every build uses.

    Args:
        calib: the recipe's calibration source.
        model_dir: local model directory (its tokenizer and fingerprint).
        tokenizer: the model's tokenizer, if already loaded.
        images: the caller passes each sample's ``inputs`` to the model with its ids. Image samples are given
            only then: their ids alone would calibrate on image tokens without the image.
        log: where the contamination filter says what it dropped, or that it did not run.

    Returns:
        ``calib.samples`` samples of at most ``calib.seqlen`` ids (text and chat samples have exactly that,
        unless ``calib.chunks``).

    Raises:
        Refused: token-id rows without the model's tokenizer fingerprint, text rows written in another model's
            chat template (:mod:`qda_quant.calib.template`), mixed row kinds, or too few samples.
    """
    parts = getattr(calib, "source_parts", lambda: [])()
    if parts:
        samples = _mixed_samples(cast("MixSource", calib), parts, model_dir, tokenizer, images)
        _log_filter(calib, log)
        return samples
    got = calib.read(model_dir)
    rows, label = got.rows, calib.label
    if (
        tokenizer is None
        and calib.index() is not None
        and any("input_ids" in r or "prompt_ids" in r for r in rows)
    ):
        from ..intake.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(model_dir)  # decode token-id rows so the filter compares the prompt text
    rows = calib.screen(rows, tokenizer)  # the reference-prompt filter; its record goes to build.json
    _log_filter(calib, log)
    if not rows:
        dropped = calib.dropped
        if dropped:
            ids = ", ".join(str(d["id"]) for d in dropped[:20])
            ref = calib.contamination_index or "the host index"
            raise Refused(
                f"calibration {label}: every row matched the reference prompts of {ref} and was dropped "
                f"({len(dropped)} rows: {ids}{', ...' if len(dropped) > 20 else ''}); give a calibration "
                "source with prompts the evaluation does not use"
            )
        raise Refused(f"calibration {label}: no rows")
    if any("images" in r for r in rows):
        if not images:
            raise Refused(
                f"calibration {label}: image rows reach this step as token ids only, without their pixels; "
                "use a text calibration source for it, or set the step's `data:` to its own text rows"
            )
        return _image_samples(calib, rows, model_dir)
    tokens = ["input_ids" in r or "prompt_ids" in r for r in rows]
    if all(tokens):
        want = fingerprint(model_dir)
        if got.tokenizer is None or got.tokenizer != want:
            raise Refused(
                f"calibration {label}: token ids made by tokenizer {got.tokenizer or '(no fingerprint)'}, the "
                f"model's is {want or '(none)'}; use text or messages rows, or ids with the matching fingerprint"
            )
        try:
            samples = _token_rows(rows, calib.seqlen, calib.chunks)
        except Refused as e:
            raise Refused(f"calibration {label}: {e}") from None
        _lint_ids(calib, got.rows, model_dir, tokenizer)  # decoded: another family's chat markup refuses
    elif any(tokens):
        raise Refused(f"calibration {label}: mixes token-id rows with text rows")
    else:
        if tokenizer is None:
            from ..intake.tokenizer import load_tokenizer

            tokenizer = load_tokenizer(model_dir)
        try:
            calib.note_template(_lint(calib, got.rows, tokenizer).record())
            need = None if calib.seed else calib.samples
            split = pieces if calib.chunks else cut
            samples = [(ids, 1) for ids in split((_tokenize(r, tokenizer) for r in rows), calib.seqlen, need)]
        except Refused:
            raise
        except Exception as e:  # noqa: BLE001 - a tokenizer QDA cannot render rows with; refuse it
            raise probe_refusal(
                tokenizer,
                "tokenizing its calibration rows",
                e,
                next_step='calibrate on text rows ({"text": ...}), or use a model whose tokenizer has a chat template',
            ) from e
    if calib.seed:
        random.Random(calib.seed).shuffle(samples)
    if len(samples) < calib.samples:
        raise _Short(
            f"calibration {label}: {len(samples)} samples of {calib.seqlen}, recipe asks for {calib.samples}",
            _short_hint(calib, len(samples)),
        )
    return [Sample(torch.tensor(ids, dtype=torch.long), n) for ids, n in samples[: calib.samples]]


def _lint(calib: CalibSource, rows: list[dict[str, Any]], tokenizer: Any) -> Any:
    """Lint every row the source read (:func:`qda_quant.calib.template.lint_rows`) and keep the file's record.

    Every row is read, so a refusal names the file's own row index.

    Raises:
        Refused: a row is written in another model family's chat template (token ids decoded).
    """
    found = lint_rows(rows, tokenizer, calib.label)
    if (why := found.refusal(calib.label)) is not None:
        raise Refused(why)
    calib.note_lint(found.lint_record())
    return found


def _lint_ids(calib: CalibSource, rows: list[dict[str, Any]], model_dir: str | Path, tokenizer: Any) -> None:
    """Lint token-id rows on their ids decoded with the model's tokenizer (loaded here if not given).

    Token-id rows calibrate without the tokenizer, so a model directory whose tokenizer cannot be loaded still
    builds: the file's lint record then says the lint did not run, and why.

    Raises:
        Refused: a row decodes to another model family's chat markup.
    """
    if tokenizer is None:
        from ..intake.tokenizer import load_tokenizer

        try:
            tokenizer = load_tokenizer(model_dir)
        except Exception as e:  # noqa: BLE001 - any loader failure: the ids still calibrate, unlinted
            calib.note_lint(
                {"file": calib.label, "check": f"not run: the model's tokenizer cannot be loaded ({e})"}
            )
            return
    _lint(calib, rows, tokenizer)


class _Short(Refused):
    """A source packed fewer samples than asked for; a mix of sources re-words it for the enclosing setting."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message + hint)
        self.head = message


def _short_hint(calib: CalibSource, got: int) -> str:
    """What to change when a calibration packs fewer samples than it asks for: read more rows, or ask for fewer.

    A model's tokenizer decides how many samples the same rows pack into, so a recipe written for one model can
    come up short on another; the refusal names the setting instead of leaving the reader to find it.
    """
    key = f"calib.{calib.name}" if calib.name else "calib"
    fewer = f"lower {key}.samples to {got}" if got else "give it rows with text"
    rows = getattr(calib, "rows", None)
    if isinstance(rows, int):
        return f"; raise {key}.rows (now {rows}) or {fewer}"
    return f"; give it more rows or {fewer}"


def _mixed_hint(calib: CalibSource, part: CalibSource) -> str:
    """What to change when one part of a mix (or of a weighted slice list) packs fewer samples than its share.

    The enclosing source replaces a part's own ``samples`` with that part's share of ``calib.samples``, so
    naming the part's count would name a setting that changes nothing; lowering the enclosing count shrinks
    every share, and the part's own rows or ``seqlen`` decide what that share can pack.
    """
    key = f"calib.{calib.name}"
    hint = f"; lower {key}.samples (now {calib.samples}) so each part's share fits"
    rows = getattr(part, "rows", None)
    if isinstance(rows, int):
        hint += f", or raise the part's rows (now {rows}) in its settings"
    elif getattr(part, "images_source", False):
        hint += f", or raise the part's seqlen (now {part.seqlen}) in its settings"
    return hint


def _image_hint(calib: CalibSource, got: int, longer: int) -> str:
    """What to change when image rows give fewer samples than asked for.

    A row longer than ``seqlen`` tokens is skipped whole, so the settings that matter are the row count, that
    seqlen, and the sample count; the refusal names them instead of leaving the reader to find them.
    """
    key = f"calib.{calib.name}" if calib.name else "calib"
    rows = getattr(calib, "rows", None)
    more = f"raise {key}.rows (now {rows})" if isinstance(rows, int) else f"raise {key}.rows"
    fewer = f"lower {key}.samples to {got}" if got else "give it rows with images"
    if longer:
        return (
            f"; raise {key}.seqlen (now {calib.seqlen}) ({longer} image rows are longer) or {more} or {fewer}"
        )
    return f"; {more} or {fewer}"


def _log_filter(calib: CalibSource, log: Callable[[str], None] | None) -> None:
    """Say on the build's log what the contamination filter did, or that it did not run."""
    if log is None:
        return
    record = calib.contamination_record()
    if record is None:
        log("contamination filter: not run (no reference index)")
        return
    ids = ", ".join(str(m["id"]) for m in record.get("matches", [])[:10])
    log(
        f"contamination filter: {record['rows']} rows, {record['dropped']} dropped"
        f"{f' ({ids})' if ids else ''}, {record['exempted']} exempted; compared: prompts only, never answer bodies"
    )


def _mixed_samples(
    calib: MixSource,
    parts: list,
    model_dir: str | Path,
    tokenizer: Any,
    images: bool,
) -> list[Sample]:
    """The samples of a mix of sources, in a fixed order.

    Each part gives its share of ``calib.samples`` (largest remainder, by weight) with its own settings; the
    samples are interleaved with ``random.Random(calib.mix_seed)``, then ordered by ``calib.seed`` as any source.
    """
    from ..calib.mix import allocate

    out: list[Sample] = []
    calib.note_template_reset()  # this read's parts give the record, not an earlier read's
    for (src, _w), n in zip(parts, allocate([w for _, w in parts], calib.samples), strict=True):
        if n:
            part = src.model_copy(update={"samples": n})
            try:
                out += load_samples(part, model_dir, tokenizer, images=images)
            except _Short as error:
                # the part's own count is the share the enclosing source gave it: name the enclosing setting
                raise _Short(error.head, _mixed_hint(calib, src)) from None
            calib.absorb(part)  # a part's dropped rows are the mix's record
    random.Random(calib.mix_seed).shuffle(out)
    if calib.seed:
        random.Random(calib.seed).shuffle(out)
    return out


def _image_samples(calib: CalibSource, rows: list[dict[str, Any]], model_dir: str | Path) -> list[Sample]:
    """Samples of image rows ``{"messages", "images"}``: one per row, made by the model's own processor.

    The processor renders the chat, expands each image placeholder into the model's image tokens and makes the
    pixel inputs. A row longer than ``seqlen`` tokens is skipped (cutting it could split an image's tokens).

    Raises:
        Refused: text rows mixed in, ``chunks``, or fewer usable rows than ``samples``.
    """
    from transformers import AutoProcessor

    label = calib.label
    if not all("images" in r and "messages" in r for r in rows):
        raise Refused(f"calibration {label}: mixes image rows {{messages, images}} with other rows")
    if calib.chunks:
        raise Refused(f"calibration {label}: chunks: true does not apply to image rows")
    try:
        proc = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=False)
    except ValueError as e:
        if remote := remote_code_refusal(model_dir, "processor", e):
            raise remote from e
        raise
    except ImportError as e:  # e.g. torchvision, which some processors import; others do not
        raise Refused(
            f'calibration {label}: the model\'s processor is missing a package: uv pip install -c constraints.txt "./packages/qda-quant[vlm]"'
        ) from e
    samples, longer = [], 0
    for r in rows:
        text = _render_chat(proc.apply_chat_template, r["messages"], tokenize=False)
        enc = proc(text=text, images=r["images"], return_tensors="pt")
        ids = enc.pop("input_ids")[0]
        enc.pop("attention_mask", None)  # no padding: every position is attended
        if ids.numel() <= calib.seqlen:
            samples.append(Sample(ids.long(), 1, dict(enc)))
        else:
            longer += 1
        if not calib.seed and len(samples) == calib.samples:
            break
    if calib.seed:
        random.Random(calib.seed).shuffle(samples)
    if len(samples) < calib.samples:
        raise _Short(
            f"calibration {label}: {len(samples)} image rows of at most {calib.seqlen} tokens, "
            f"recipe asks for {calib.samples}",
            _image_hint(calib, len(samples), longer),
        )
    return samples[: calib.samples]


def load_ids(
    calib: CalibSource,
    model_dir: str | Path,
    tokenizer: Any = None,
    *,
    log: Callable[[str], None] | None = None,
) -> list[torch.Tensor]:
    """The token ids of :func:`load_samples` (for the calibration forward, which needs no ``prompt_len``)."""
    return [s.ids for s in load_samples(calib, model_dir, tokenizer, log=log)]


def load_with_inputs(
    calib: CalibSource, model_dir: str | Path, *, log: Callable[[str], None] | None = None
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]] | None]:
    """The token ids of :func:`load_samples`, and each sample's other model inputs (None for text samples)."""
    samples = load_samples(calib, model_dir, images=True, log=log)
    inputs = [s.inputs or {} for s in samples]
    return [s.ids for s in samples], inputs if any(inputs) else None


NOT_RECORDED = "not recorded"
#: What build.json's ``calib_sha256`` digest covers, and what it does not: written beside the digest, so the field
#: cannot be read as "all of this build's data is recorded".
COVERS = (
    "the token ids (and image inputs) of the samples the float stage's calibration forward read, in order"
)
NOT_COVERED = (
    "data a step loads on its own: scale_kd's teacher samples, and the data of checkpoint steps "
    "(steps on the packed output); those are not in this digest"
)


def samples_sha256(
    ids: list[torch.Tensor] | None, inputs: list[dict[str, torch.Tensor]] | None = None
) -> str:
    """sha256 of the calibration samples a build feeds its model, in order.

    Each sample's token ids, and its other model inputs (an image's ``pixel_values``) by name, dtype, shape and
    bytes. It names the calibration DATA a build used, whatever the source (a file, a generated cache, a mix): two
    builds with the same digest calibrated on the same tokens. ``"not recorded: <why>"`` when the build read no
    samples or they could not be hashed, never an empty string, so a missing digest reads as missing; a record
    never fails a build.
    """
    if not ids:
        return f"{NOT_RECORDED}: the build read no calibration samples"
    h = hashlib.sha256(b"qda calib samples v1\n")

    def put(t: torch.Tensor) -> None:
        t = t.detach().to("cpu").contiguous().reshape(-1)
        h.update(f"{t.dtype}:{t.numel()}\n".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())

    try:
        for i, x in enumerate(ids):
            h.update(f"sample {i} ids {tuple(x.shape)}\n".encode())
            put(x.to(torch.long))
            extra = (inputs[i] if inputs else None) or {}
            for k in sorted(extra):
                h.update(f"input {k} {tuple(extra[k].shape)}\n".encode())
                put(extra[k])
    except Exception as e:  # noqa: BLE001 - any failure is recorded as such, the build goes on
        return f"{NOT_RECORDED}: the samples could not be hashed ({type(e).__name__}: {e})"
    return h.hexdigest()


def calib_record(
    ids: list[torch.Tensor] | None, inputs: list[dict[str, torch.Tensor]] | None = None
) -> dict[str, str]:
    """build.json's ``calib_sha256``: the digest (:func:`samples_sha256`), what it covers and what it does not."""
    return {"sha256": samples_sha256(ids, inputs), "covers": COVERS, "not_covered": NOT_COVERED}


def batches(ids: list[torch.Tensor], batch_size: int = 1) -> list[torch.Tensor]:
    """Consecutive rows of equal length stacked into [b, T] batches (order kept; no padding)."""
    out: list[torch.Tensor] = []
    cur: list[torch.Tensor] = []
    for r in ids:
        if cur and (len(cur) == batch_size or cur[0].numel() != r.numel()):
            out.append(torch.stack(cur))
            cur = []
        cur.append(r)
    if cur:
        out.append(torch.stack(cur))
    return out
