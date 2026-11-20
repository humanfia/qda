"""The files and the stored tensors of a llama.cpp candidate: one `.gguf` (the whole model, tokenizer included).

An anchor's non-weight files are taken byte for byte. There is no shard index and no `quantization_config`:
the file itself is the checkpoint, so admission reads its header -- the stored type of every tensor (the
task's menu) and the bytes they hold (the byte budget) -- and its identity is those tensors, not the file's
padding or KV order.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from qda_quant.gguf import file as gguf_file
from qda_quant.gguf.rule import BF16, F16, F32, IQ4_XS, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0

from ..freeze import file_sha256 as _sha
from ..freeze import walk
from .formats import HEAD_16BIT, HEAD_16BIT_TYPES
from .model_files import CODE, INERT, inert_match
from .tensors import CheckpointError

#: Three GGUF types a quantized file may store besides a search menu's: a 3-bit k-quant, 4-bit nonlinear
#: codes in 32-value blocks, and 3-bit grid codes with signs (numbers as ggml's `ggml_type`). The judge sizes,
#: names and compares them so any `.gguf` can be read whole, as a baseline or a reference.
Q3_K, IQ4_NL, IQ3_S = 11, 20, 21
#: A top-level GGUF file: the checkpoint itself.
GGUF = re.compile(r"[^/]+\.gguf")
#: A stored weight file of the safetensors engine: never part of a llama.cpp candidate.
SHARD = re.compile(r"[^/]+\.safetensors")
#: The ggml types a GGUF stores that are not a quantized weight. A menu gate exempts none of them: a tensor stored
#: off the menu is admitted only as candidate 0 stores it (`gguf_problem`).
GGML_FLOAT = frozenset({F32, F16, BF16})
#: ggml type -> the name a GGUF search writes in its menu; other types are named by their number.
GGML_NAMES = {
    F32: "F32",
    F16: "F16",
    Q5_0: "Q5_0",
    Q8_0: "Q8_0",
    Q4_K: "Q4_K",
    Q5_K: "Q5_K",
    Q6_K: "Q6_K",
    IQ4_XS: "IQ4_XS",
    BF16: "BF16",
    Q3_K: "Q3_K",
    IQ4_NL: "IQ4_NL",
    IQ3_S: "IQ3_S",
}
#: ggml type -> (bytes of one block, the spans holding its packed codes, the spans holding its scales and
#: parameters). The spans are the pinned gguf-py's own splits (`gguf.quants`); together they cover each block
#: exactly once. Q5_K's block is d, dmin and 12 scale bytes, then 32 high-bit bytes and 128 low-nibble bytes of
#: codes; Q8_0's is one fp16 scale and 32 int8 codes; IQ4_XS's is d, the two scale halves and 128 code bytes.
#: The three extra types: Q3_K is hmask 32, qs 64 (codes), then 12 scale bytes and d; IQ4_NL is d and 16 code
#: bytes; IQ3_S is d, then qs 64, qh 8 and signs 32 (codes), then 4 scale bytes -- its parameters are two
#: spans. A quantized type missing here is refused by `gguf_near_copy`, never skipped (fail closed).
GGML_BLOCKS: dict[int, tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = {
    Q4_K: (144, ((16, 144),), ((0, 16),)),
    Q5_K: (176, ((16, 176),), ((0, 16),)),
    Q6_K: (210, ((0, 192),), ((192, 210),)),
    Q8_0: (34, ((2, 34),), ((0, 2),)),
    IQ4_XS: (136, ((8, 136),), ((0, 8),)),
    Q3_K: (110, ((0, 96),), ((96, 110),)),
    IQ4_NL: (18, ((2, 18),), ((0, 2),)),
    IQ3_S: (110, ((2, 106),), ((0, 2), (106, 110))),
}
#: An MTP/NextN head tensor as the pinned llama.cpp names it (`src/llama-arch.cpp:728-738`, the
#: LLM_TENSOR_NEXTN_* table): `nextn.pre_projection`, `nextn.post_projection`, `blk.N.nextn.<name>`.
#: The dot on both sides keeps a lookalike such as `blk.0.attn_nextn.weight` out.
MTP = re.compile(r"(?:^|\.)nextn\.")
#: The output head's one tensor: llama.cpp's GGUF name for lm_head (`qda_quant.gguf.classes`).
HEAD = re.compile(r"output\.weight")
#: The span `_codes_and_params_sha` reads at a time.
READ_SPAN = 64 << 20


def the_gguf(d: Path) -> Path | None:
    """The one top-level `.gguf` of `d`, or `d` itself when it is a regular `.gguf` file; None otherwise.

    None for a link, a directory holding none or several, or a file of another kind.
    """
    if d.is_symlink():
        return None
    if d.is_file():
        return d if d.suffix == ".gguf" else None
    found = sorted(p for p in d.glob("*.gguf") if p.is_file() and not p.is_symlink())
    return found[0] if len(found) == 1 else None


def _ggml_name(kind: int) -> str:
    """The name a menu spells a ggml type by; the number itself for a type upstream adds later."""
    return GGML_NAMES.get(kind, str(kind))


def _table(d: Path) -> tuple[Path, gguf_file.Table] | None:
    """`d`'s one `.gguf` and the tensor table its header states; None when `d` holds none or several.

    Raises:
        CheckpointError: the file is not a GGUF this module reads, or a tensor it cannot size.
    """
    p = the_gguf(d)
    if p is None:
        return None
    try:
        table = gguf_file.read_table(p)
    except ValueError as e:
        raise CheckpointError(str(e).replace(str(p), p.name)) from None
    return p, table


def gguf_fingerprint(d: Path) -> str | None:
    """sha256 over the stored tensors' (name, ggml type, shape, sha256 of the bytes), sorted.

    Independent of the file's padding, its KV order and its tensor order: a baseline re-converted with the
    same stored values has the same fingerprint, and a changed stored value changes it. None when `d` does
    not hold exactly one `.gguf`.

    Raises:
        CheckpointError: the file is not a GGUF this module reads, or a tensor it cannot size.
    """
    found = _table(d)
    if found is None:
        return None
    p, table = found
    items = []
    for t in table.tensors:
        try:
            size = gguf_file.tensor_bytes(t)
        except ValueError as e:
            raise CheckpointError(str(e)) from None
        sha = _span_sha(p, table.data_start + t.offset, size)
        items.append(json.dumps([t.name, _ggml_name(t.tensor_type), list(t.shape), sha]))
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def gguf_near_copy(candidate: Path, baseline: Path) -> str | None:
    """Why `candidate` is `baseline` as-is (or its packed codes re-scaled), or None.

    Quantized tensors are paired by name. Every packed code equal and every scale and parameter equal is the
    baseline as-is; codes equal with any scale or parameter changed is the baseline's codes re-scaled. Float
    tensors do not decide it (a re-saved embedding may move by a bit while the weights are copied), and a
    candidate that stores different tensor names, types or shapes, or no quantized tensor at all, is not
    compared. A quantized type whose block layout is not in `GGML_BLOCKS` is refused, never skipped: without
    the spans its codes cannot be compared, so a copy of a baseline storing it cannot be ruled out.

    Raises:
        CheckpointError: a file that is not a GGUF this module reads, a tensor it cannot size, or a quantized
            tensor whose block layout this module does not know.
    """
    mine, theirs = _table(candidate), _table(baseline)
    if mine is None or theirs is None:
        return None
    (pa, ta), (pb, tb) = mine, theirs
    a = {t.name: t for t in ta.tensors}
    b = {t.name: t for t in tb.tensors}
    if set(a) != set(b):
        return None
    codes_equal, params_equal, compared = True, True, False
    for name, t in a.items():
        u = b[name]
        if t.tensor_type not in GGML_BLOCKS:
            if u.tensor_type in GGML_BLOCKS:
                return None  # this candidate stores it packed, the baseline does not: not the same result
            if t.tensor_type not in GGML_FLOAT:  # a float tensor does not decide a re-save, a packed one does
                raise CheckpointError(
                    f"{name} is stored as {_ggml_name(t.tensor_type)}: no block layout is known for that type, "
                    "so a copy of the baseline's packed codes cannot be ruled out"
                )
            continue
        if u.tensor_type != t.tensor_type or u.shape != t.shape:
            return None
        block, codes_span, params_span = GGML_BLOCKS[t.tensor_type]
        try:
            size = gguf_file.tensor_bytes(t)
        except ValueError as e:
            raise CheckpointError(str(e)) from None
        hc_a, hp_a = _codes_and_params_sha(pa, ta.data_start + t.offset, size, block, codes_span, params_span)
        hc_b, hp_b = _codes_and_params_sha(pb, tb.data_start + u.offset, size, block, codes_span, params_span)
        compared = True
        codes_equal = codes_equal and hc_a == hc_b
        params_equal = params_equal and hp_a == hp_b
    if not compared:
        return None
    if codes_equal:
        return (
            "a baseline checkpoint as-is, not your result"
            if params_equal
            else "a baseline's packed codes with its scales changed, not your result"
        )
    return None


def _span_sha(p: Path, at: int, size: int) -> str:
    """sha256 of `size` bytes at `at`.

    Raises:
        CheckpointError: the file ends before its header says.
    """
    h = hashlib.sha256()
    with p.open("rb", buffering=0) as fh:
        fh.seek(at)
        done = 0
        while done < size:
            chunk = fh.read(min(READ_SPAN, size - done))
            if not chunk:
                raise CheckpointError(f"{p.name} ends before its header says")
            h.update(chunk)
            done += len(chunk)
    return h.hexdigest()


def _codes_and_params_sha(
    p: Path,
    at: int,
    size: int,
    block: int,
    codes: tuple[tuple[int, int], ...],
    params: tuple[tuple[int, int], ...],
) -> tuple[str, str]:
    """sha256 of a quantized tensor's packed codes and of its scales and parameters.

    A quantization block holds its own codes and parameters, so the two are separated per block; the file is
    read a span at a time, never whole. A span is a whole number of blocks: `READ_SPAN` is a whole number of
    MiB and a block is 18 to 210 bytes, so a raw span would end inside a block on a real tensor. A
    type whose codes or parameters sit in several runs of a block (IQ3_S) has each run concatenated in order.

    Raises:
        CheckpointError: the file ends before its header says, or ends inside a block.
    """
    import numpy as np

    span = max(block, READ_SPAN - READ_SPAN % block)
    hc, hp = hashlib.sha256(), hashlib.sha256()
    with p.open("rb", buffering=0) as fh:
        fh.seek(at)
        done = 0
        while done < size:
            chunk = fh.read(min(span, size - done))
            if not chunk:
                raise CheckpointError(f"{p.name} ends before its header says")
            if len(chunk) % block:
                raise CheckpointError(f"{p.name} ends inside a quantization block at {at + done}")
            blocks = np.frombuffer(chunk, np.uint8).reshape(-1, block)
            hc.update(np.concatenate([blocks[:, a:b] for a, b in codes], axis=1).tobytes())
            hp.update(np.concatenate([blocks[:, a:b] for a, b in params], axis=1).tobytes())
            done += len(chunk)
    return hc.hexdigest(), hp.hexdigest()


def gguf_sizes(d: Path) -> dict[str, int] | None:
    """The byte figures a llama.cpp candidate is measured by, or None without exactly one `.gguf`.

    `file` is the frozen `.gguf`'s `os.stat` size -- the file the scorer reads, header and padding
    included; `payload` is the bytes its tensors store, without the header.

    Raises:
        CheckpointError: the file is not a GGUF this module reads, or a tensor it cannot size.
    """
    found = _table(d)
    if found is None:
        return None
    p, table = found
    try:
        payload = sum(gguf_file.tensor_bytes(t) for t in table.tensors)
    except ValueError as e:
        raise CheckpointError(str(e)) from None
    return {"file": p.stat().st_size, "payload": payload}


def _type_problem(
    table: gguf_file.Table,
    menu: tuple[str, ...] | list[str] | None,
    reference: Path | None,
    lock: tuple[str, ...] | list[str],
    head: str = HEAD_16BIT,
) -> str | None:
    """Why a tensor of `table` is stored as a type the task does not admit, or None (`gguf_problem`'s types).

    The output head (`output.weight`) has its own rule (`judge.head`): by default it may be stored 16-bit
    (BF16 or F16) even where the menu names neither, and a menu type is admitted as before; an exception
    model's head must be stored exactly the release's type. An explicit `lock` still wins.
    """
    if menu is None and not lock:
        return None
    theirs: dict[str, int] | None = None
    if reference is not None:
        try:
            found = _table(reference)
        except CheckpointError as e:
            found, error = None, str(e)
        else:
            error = "no single top-level .gguf"
        if found is None:  # fail closed: without candidate 0's table no off-menu type or lock can be checked
            return (
                f"baselines[0] (candidate 0) cannot be read, so the stored types cannot be checked: {error}"
            )
        theirs = {t.name: t.tensor_type for t in found[1].tensors}
    allowed = set(menu) if menu is not None else None
    locks = [re.compile(r) for r in lock]

    def pinned(name: str) -> bool:
        return any(r.search(name) for r in locks)

    ours = {t.name for t in table.tensors}
    for name in theirs or {}:
        if pinned(name) and name not in ours:
            return f"candidate 0's {name} is locked (judge.llamacpp.lock), and this file does not store it"
    for t in table.tensors:
        name, kind = t.name, _ggml_name(t.tensor_type)
        want = theirs.get(name) if theirs is not None else None
        if pinned(name):
            if want is None:
                return f"{name} is locked to candidate 0's type (judge.llamacpp.lock), but candidate 0 has no {name}"
            if want != t.tensor_type:
                return (
                    f"{name} is stored as {kind}, but candidate 0 stores it as {_ggml_name(want)} "
                    "(judge.llamacpp.lock): keep that type"
                )
            continue
        if HEAD.fullmatch(name) and allowed is not None:
            if head == HEAD_16BIT:
                if kind in HEAD_16BIT_TYPES:
                    continue  # the task's head rule is the menu's own: 16-bit needs no menu entry
            elif kind == head:
                continue
            else:
                return (
                    f"output.weight is stored as {kind}; this model's release stores the output head as {head} "
                    "(judge.head): keep that type"
                )
        if want is not None and allowed is not None and _ggml_name(want) not in allowed:
            if want != t.tensor_type:
                return (
                    f"{name} is stored as {kind}, but candidate 0 stores it as {_ggml_name(want)} (a type off "
                    "judge.format_menu): keep that type"
                )
            continue
        if allowed is None or kind in allowed:
            continue
        if theirs is None and t.tensor_type == F32:
            continue  # no candidate 0 to read the template's float tensors from: F32 stays exempt, as before
        return f"{name} is stored as {kind}, outside judge.format_menu ({', '.join(menu or ())})"
    return None


def gguf_problem(
    d: Path,
    anchor: Path | None = None,
    inert: tuple[str, ...] | list[str] = INERT,
    *,
    menu: tuple[str, ...] | list[str] | None = None,
    budget_bytes: int | None = None,
    budget_basis: str = "file",
    mtp: str = "any",
    any_format: bool = False,
    reference: Path | None = None,
    lock: tuple[str, ...] | list[str] = (),
    head: str = HEAD_16BIT,
) -> str | None:
    """Why `d` is not a llama.cpp candidate of the task, or None.

    One top-level `.gguf` and nothing else, except inert files (`inert`) and the anchor's own files at the
    same relative path, byte for byte; hidden paths, links and code are refused as for safetensors.

    The header then decides what the file stores: every tensor must be stored as a type `menu` names, float
    types included, except where candidate 0 (`reference`) stores that same tensor off the menu: such a tensor
    must keep candidate 0's type (the template's F32 norms, say), and a tensor `lock` names must keep it too,
    on the menu or not. With no `reference`, F32 alone needs no menu entry, as before. The file must fit
    `budget_bytes`.
    The budget's basis is the whole `.gguf` file by default (`budget_basis: file`: `os.stat` of the file the
    scorer reads, header and padding included); `payload` is an explicit opt-in that counts only the stored
    tensor bytes. `mtp` refuses a required MTP/NextN head that is absent, or a forbidden one present.

    Args:
        d: The candidate directory.
        anchor: The original model, whose own files may sit beside the candidate.
        inert: Globs of files nothing loads, accepted as they are.
        menu: The GGUF types the task admits (`judge.format_menu`); None: no menu gate.
        budget_bytes: The most bytes the file may cost (`judge.budget_bytes`); None: no byte budget.
        budget_basis: `file` (the whole `.gguf`, `judge.budget_basis`'s default) or `payload` (its stored
            tensors only, an explicit opt-in).
        mtp: `judge.llamacpp.mtp`: `required` refuses a file with no MTP/NextN tensor, `forbidden` one that
            carries any, `any` (the default) does not decide. A row's file choice keeps one answer.
        any_format: Skip the menu gate (`judge.allow_any_format`, development only); the byte budget holds.
        reference: Candidate 0's directory (`baselines[0]`): the types it stores off the menu are the
            only off-menu types admitted, each on its own tensor. None: F32 is exempt from the menu instead.
        lock: `judge.llamacpp.lock`: regexes (`re.search` on the GGUF tensor name) of tensors that must be
            stored as candidate 0 stores them, e.g. ``["^blk\\.64\\."]`` for an MTP block; needs `reference`.
        head: The output head's rule (`judge.head`): `16-bit` admits `output.weight` stored BF16 or F16, on
            the menu or not, and still admits a menu type; an exception model names the release's type,
            which the head must keep.
    """
    if budget_basis not in ("file", "payload"):
        raise ValueError(f"budget_basis {budget_basis!r}: file or payload")
    if mtp not in ("required", "forbidden", "any"):
        raise ValueError(f"mtp {mtp!r}: required, forbidden or any")
    if not d.is_dir():
        return "not a directory"
    ggufs = 0
    for p in walk(d):
        rel = p.relative_to(d).as_posix()
        if any(part.startswith(".") for part in p.relative_to(d).parts):
            return f"{rel}: hidden files and directories are not served"
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            return f"{rel} is not a regular file"
        if CODE.search(rel):
            return f"{rel}: code or pickle files are not served"
        if p.is_dir():
            continue
        if "/" not in rel and GGUF.fullmatch(rel):
            ggufs += 1
            continue
        if GGUF.fullmatch(rel) or SHARD.fullmatch(rel):
            return f"{rel}: a llama.cpp candidate is one top-level .gguf file, not this"
        if any(inert_match(rel, g) for g in inert):
            continue
        a = anchor / rel if anchor is not None else None
        if not (a is not None and a.is_file() and not a.is_symlink() and _sha(a) == _sha(p)):
            return f"{rel}: not served (a llama.cpp candidate is one .gguf; any other file must be the anchor's own)"
    if ggufs == 0:
        return "no *.gguf weights"
    if ggufs > 1:
        return f"{ggufs} *.gguf weights; a llama.cpp candidate is one"
    try:
        found = _table(d)
    except CheckpointError as e:
        return f"not a readable GGUF ({e})"
    if found is None:  # unreadable as a single file: the walk above already said which rule it broke
        return "no single .gguf file (a llama.cpp candidate is one .gguf)"
    p, table = found
    if lock and reference is None:
        raise ValueError("judge.llamacpp.lock needs candidate 0 (baselines[0]) to lock to")
    try:
        if (why := _type_problem(table, menu if not any_format else None, reference, lock, head)) is not None:
            return why
        payload = sum(gguf_file.tensor_bytes(t) for t in table.tensors)
        mtp_names = [t.name for t in table.tensors if MTP.search(t.name)]
    except ValueError as e:
        return f"not a readable GGUF ({e})"
    if mtp == "required" and not mtp_names:
        return "no MTP/NextN tensors (nextn.*), and judge.llamacpp.mtp is required"
    if mtp == "forbidden" and mtp_names:
        return (
            f"{len(mtp_names)} MTP/NextN tensors ({', '.join(mtp_names)}), and judge.llamacpp.mtp is "
            "forbidden"
        )
    if budget_bytes is not None:
        size = p.stat().st_size if budget_basis == "file" else payload
        if size > budget_bytes:
            return f"{budget_basis} {size} B > budget {budget_bytes} B"
    return None


__all__ = [
    "GGUF",
    "SHARD",
    "gguf_fingerprint",
    "gguf_near_copy",
    "gguf_problem",
    "gguf_sizes",
    "the_gguf",
]
