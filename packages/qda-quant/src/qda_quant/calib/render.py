"""``qda calib render``: a frozen calibration text file for one model, rendered with its own chat template.

    qda calib render --model M --pool qda-lab/qda-calibration-pools --names chat=3,multilingual=2 \\
        --samples 128 --seqlen 2048 --out calib.jsonl [--thinking keep|off|drop] [--seed 0]

A ``messages`` row reaches a build un-rendered and is rendered there by whatever template is installed, and a
pre-rendered text row carries the markup of the model it was rendered for. Neither is a fixed artefact for a new
model, so this command makes one, as the release recipes ship theirs (``*_text.jsonl`` with a sha256):

1. every row of each named pool file is normalised to chat turns: ChatML text (``d1.jsonl``) is parsed back
   into turns, an empty think block is stripped, and an inline ``<think>...</think>`` at the start of an
   assistant turn moves into the turn's ``reasoning_content``; a row with markup of another family, markup
   inside a message, token ids or no assistant answer is left out and counted by reason;
2. the rows are mixed by weight (:func:`qda_quant.calib.mix.mix`: the same names, rows and seed give the same
   picks);
3. each row is rendered with the model's own chat template (below);
4. with ``--samples S --seqlen L`` (what the build that reads the file asks for), as many rows are kept as the
   build's packing (:func:`qda_quant.engine.calib.cut`) needs for S samples of L tokens, mixing more rows when
   the first ones fall short; a pool too small for it is refused here, not at the build. ``--rows N`` keeps
   N rows instead.

Reasoning lands in the model's own thinking format when its template renders ``reasoning_content`` (checked on a
sentinel), and is dropped otherwise: no marker the template does not produce is ever written (the rendered text
is checked, family by family).

``--thinking`` is the policy: ``keep`` (reasoning in the model's format where it has one), ``off`` (reasoning
removed, the template's thinking-off form, ``enable_thinking: false``), ``drop`` (reasoning removed, the
template's default form). A turn without reasoning is rendered as the template renders it: a Qwen3 template
writes an empty think block there, which is that model's own thinking-off form.

The output holds ``{"text"}`` rows; ``<out>.meta.json`` records the pool (dataset and resolved revision, or the
directory), each file's sha256, the names and weights, the rows, the seed, the template's sha256, the thinking
policy and what it did, the normalisation counts and the output's sha256. User-only prompt pools
(``reasoning.jsonl``: no answers) are not rendered; generate their answers first (:mod:`qda_quant.calib.selfgen`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

from ..exit import refuse
from ..step import Refused
from . import read_rows, sha256_file, write_jsonl
from .template import FAMILIES, THINK_CLOSE, THINK_OPEN

#: The public calibration pools on the Hub.
POOL = "qda-lab/qda-calibration-pools"
#: Thinking policies (the module docstring).
POLICIES = ("keep", "off", "drop")
Policy = Literal["keep", "off", "drop"]

_TURN = re.compile(r"<\|im_start\|>([a-z_]+)\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
_LEAD_THINK = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE) + r"\s*", re.S)
_SENTINEL_REASONING = "qda sentinel reasoning"


def _markers(text: str) -> list[str]:
    return [m for marks in FAMILIES.values() for m in marks if m in text]


def _turn(role: str, content: str, counts: Counter[str]) -> dict[str, Any] | str:
    """One normalised turn, or why the row is left out. An assistant's leading think block becomes its reasoning."""
    if _markers(content):
        return "chat markup inside a message"
    turn: dict[str, Any] = {"role": role, "content": content}
    if role != "assistant":
        return (
            turn
            if THINK_OPEN not in content and THINK_CLOSE not in content
            else "think markup outside an answer"
        )
    body = content.lstrip()
    if (lead := _LEAD_THINK.match(body)) is not None:
        reasoning, body = lead.group(1).strip(), body[lead.end() :]
        counts["inline think moved to reasoning_content" if reasoning else "empty think stripped"] += 1
        turn = {"role": role, "content": body}
        if reasoning:
            turn["reasoning_content"] = reasoning
    if THINK_OPEN in body or THINK_CLOSE in body:
        return "think markup inside an answer"
    return turn


def normalise(row: Any, counts: Counter[str]) -> dict[str, Any] | str:
    """`row` as ``{"messages"}`` (or a plain ``{"text"}`` with no markup), or the reason it is left out."""
    if not isinstance(row, dict) or "input_ids" in row or "prompt_ids" in row:
        return "token ids or no row"
    if "messages" in row:
        turns = []
        for m in row["messages"]:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                return "a message without text content"
            got = _turn(str(m.get("role")), m["content"], counts)
            if isinstance(got, str):
                return got
            if isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip():
                if "reasoning_content" in got:
                    return "reasoning given twice"
                got["reasoning_content"] = m["reasoning_content"].strip()
            turns.append(got)
    elif isinstance(row.get("text"), str):
        text = row["text"]
        found = _markers(text)
        if not found:
            if THINK_OPEN in text or THINK_CLOSE in text:
                return "think markup in plain text"
            return {"text": text}
        if any(m not in FAMILIES["chatml"] for m in found):
            return "pre-rendered in a template other than ChatML"
        turns, end = [], 0
        for match in _TURN.finditer(text):
            if text[end : match.start()].strip():
                return "text outside the ChatML turns"
            got = _turn(match.group(1), match.group(2).strip("\n"), counts)
            if isinstance(got, str):
                return got
            turns.append(got)
            end = match.end()
        if text[end:].strip():
            return "text outside the ChatML turns"
        counts["ChatML text parsed into turns"] += 1
    else:
        return "no messages or text"
    if not turns or turns[-1]["role"] != "assistant" or not turns[-1]["content"].strip():
        return "no assistant answer"
    return {"messages": turns}


def _apply(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """The engine's own rendering (`qda_quant.engine.calib`): a refused final assistant turn is continued."""
    try:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, **kwargs))
    except Exception:  # noqa: BLE001 - a template refusing a final assistant turn (Mistral's) takes the continuation
        return str(
            tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True, **kwargs)
        )


def template_facts(tokenizer: Any) -> dict[str, Any]:
    """What the model's template produces: its marker families, a thinking format, and its sha256."""
    plain = [
        {"role": "user", "content": "qda sentinel question"},
        {"role": "assistant", "content": "qda answer"},
    ]
    reasoned = [plain[0], {**plain[1], "reasoning_content": _SENTINEL_REASONING}]
    try:
        texts = [_apply(tokenizer, plain), _apply(tokenizer, reasoned)]
    except Exception as e:  # noqa: BLE001 - no chat template: there is no format of the model's own to render in
        raise Refused(
            f"the model's tokenizer renders no chat ({type(e).__name__}: {e}); give a chat model"
        ) from None
    template = getattr(tokenizer, "chat_template", None)
    raw = template if isinstance(template, str) else json.dumps(template, sort_keys=True)
    return {
        "families": sorted(n for n, marks in FAMILIES.items() if any(m in t for t in texts for m in marks)),
        "reasoning": _SENTINEL_REASONING in texts[1],
        "think_markers": any(THINK_OPEN in t or THINK_CLOSE in t for t in texts),
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def render_rows(
    rows: list[dict[str, Any]], tokenizer: Any, thinking: Policy
) -> tuple[list[dict[str, str]], dict]:
    """Render normalised rows with the model's template; return the text rows and what the policy did.

    Raises:
        Refused: a rendered row holds a marker the model's template does not produce (never written).
    """
    if thinking not in POLICIES:
        raise Refused(f"--thinking {thinking!r}: one of {', '.join(POLICIES)}")
    facts = template_facts(tokenizer)
    own = {m for name in facts["families"] for m in FAMILIES[name]}
    kwargs = {"enable_thinking": False} if thinking == "off" else {}
    keep = thinking == "keep" and facts["reasoning"]
    done: Counter[str] = Counter()
    out = []
    for i, row in enumerate(rows):
        if "text" in row:
            out.append({"text": row["text"]})
            done["plain text"] += 1
            continue
        messages = []
        for m in row["messages"]:
            if "reasoning_content" in m:
                done[
                    "turns with reasoning, rendered" if keep else "turns with reasoning, reasoning dropped"
                ] += 1
                if not keep:
                    m = {k: v for k, v in m.items() if k != "reasoning_content"}
            messages.append(m)
        text = _apply(tokenizer, messages, **kwargs)
        alien = [m for m in _markers(text) if m not in own]
        if not facts["think_markers"] and (THINK_OPEN in text or THINK_CLOSE in text):
            alien.append(THINK_OPEN)
        if alien:
            raise Refused(
                f"row {i}: the render holds {alien[0]!r}, which this model's template never produces"
            )
        out.append({"text": text})
        done["chats"] += 1
    return out, {"policy": thinking, "template": facts, **dict(sorted(done.items()))}


def parse_names(spec: str) -> dict[str, int]:
    """``chat=3,multilingual=1`` as ``{name: weight}``."""
    out: dict[str, int] = {}
    for part in spec.split(","):
        name, sep, weight = part.strip().partition("=")
        if not name or not sep or not weight.isdigit() or int(weight) < 1 or name in out:
            raise Refused(f"--names {spec!r}: give unique NAME=WEIGHT pairs with positive integer weights")
        out[name] = int(weight)
    return out


def pool_files(pool: str, names: list[str]) -> tuple[dict[str, Path], dict[str, Any]]:
    """Each named file of the pool (a directory, or a Hub dataset ``REPO[@REVISION]``) and the pool's record."""
    if Path(pool).expanduser().is_dir():
        root = Path(pool).expanduser().resolve()
        files = {n: root / f"{n}.jsonl" for n in names}
        if missing := [str(p) for p in files.values() if not p.is_file()]:
            raise Refused(f"--pool {root}: no {', '.join(missing)}")
        return files, {"dir": str(root)}
    from huggingface_hub import HfApi, hf_hub_download

    repo, _, revision = pool.partition("@")
    sha = HfApi().dataset_info(repo, revision=revision or None).sha
    files = {n: Path(hf_hub_download(repo, f"{n}.jsonl", repo_type="dataset", revision=sha)) for n in names}
    return files, {"dataset": repo, "revision": sha}


def _enough(tokenizer: Any, text: list[dict[str, str]], samples: int, seqlen: int) -> tuple[int, int | None]:
    """How many samples of `seqlen` the build cuts from `text`, and the fewest leading rows that give `samples`.

    The build tokenizes a text row as written (no special tokens added) and packs the rows with `cut`.
    """
    from ..engine.calib import cut

    ids = [tokenizer(r["text"], add_special_tokens=False)["input_ids"] for r in text]
    got = len(cut(ids, seqlen))
    if got < samples:
        return got, None
    lo, hi = 1, len(ids)  # `cut` of a longer prefix never gives fewer samples
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if len(cut(ids[:mid], seqlen)) < samples else (lo, mid)
    return got, lo


def render(
    tokenizer: Any,
    files: dict[str, Path],
    weights: dict[str, int],
    *,
    rows: int | None = None,
    samples: int | None = None,
    seqlen: int | None = None,
    seed: int,
    thinking: Policy,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Normalise, mix and render the pool files; return the text rows and their record (without the pool's).

    Give `rows`, or `samples` and `seqlen`: the rows the build needs for that many packed samples.

    Raises:
        Refused: the pools hold too few usable rows for `rows`, or for `samples` of `seqlen` tokens.
    """
    from .mix import mix

    if rows is not None:
        if samples is not None or seqlen is not None:
            raise Refused("give --rows, or --samples with --seqlen, not both")
        if rows < 1:
            raise Refused("--rows must be at least 1")
    elif samples is None or seqlen is None:
        raise Refused("give --rows, or --samples with --seqlen (what the build that reads the file asks for)")
    elif samples < 1 or seqlen < 1:
        raise Refused("--samples and --seqlen must be at least 1")

    counts: Counter[str] = Counter()
    left_out: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="qda-render-") as tmp:
        parts = []
        for name, path in files.items():
            usable, why = [], Counter()
            for row in read_rows(path):
                got = normalise(row, counts)
                if isinstance(got, str):
                    why[got] += 1
                else:
                    usable.append(got)
            left_out[name] = dict(sorted(why.items()))
            part = Path(tmp) / f"{name}.jsonl"
            part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in usable))
            parts.append((part, weights[name]))
        want, want_ok, got, packed = rows if rows is not None else cast(int, samples), 0, 0, None
        while True:
            try:
                picked = read_rows(mix(parts, want, seed, out=Path(tmp) / "mix.jsonl"))
            except Refused as exc:
                if rows is None and got:
                    reason = (
                        f"{samples} samples of {seqlen} tokens need more rows than the pools give at these weights "
                        f"({got} samples from {want_ok} rows); ask for fewer samples or name more pools"
                    )
                else:
                    reason = str(exc)
                raise Refused(f"{reason}; rows left out of each file, by reason: {left_out}") from None
            text, thought = render_rows(picked, tokenizer, thinking)
            if rows is not None:
                break
            got, keep = _enough(tokenizer, text, cast(int, samples), cast(int, seqlen))
            if keep is not None:
                text, packed = text[:keep], cast(int, samples)
                break
            want_ok = want
            want = max(
                want + 1, -(-want * cast(int, samples) * 5 // (4 * max(got, 1)))
            )  # 25 % over the shortfall
    record = {
        "files": {n: sha256_file(p) for n, p in files.items()},
        "names": weights,
        "rows": len(text),
        **({"samples": packed, "seqlen": seqlen} if packed is not None else {}),
        "seed": seed,
        "normalised": dict(sorted(counts.items())),
        "left_out": left_out,
        "thinking": thought,
    }
    if "reasoning" not in files:
        record["reasoning"] = "not in the mix: reasoning.jsonl holds prompts without answers"
    return text, record


def _run(a: argparse.Namespace) -> int:
    from ..intake.tokenizer import load_tokenizer
    from ..output.roots import output_path

    try:
        out = output_path(a.out)
        if out.exists():
            raise Refused(
                f"{out} already exists; remove it or pick another --out (a build hashes its inputs)"
            )
        weights = parse_names(a.names)
        files, source = pool_files(a.pool, list(weights))
        tokenizer = load_tokenizer(a.model)
        rows, record = render(
            tokenizer,
            files,
            weights,
            rows=a.rows,
            samples=a.samples,
            seqlen=a.seqlen,
            seed=a.seed,
            thinking=a.thinking,
        )
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    meta = {
        "kind": "qda-calib-render",
        "model": str(a.model),
        "pool": source,
        **record,
        "sha256": hashlib.sha256(data.encode()).hexdigest(),
    }
    write_jsonl(out, rows, meta)
    packed = f", {meta['samples']} samples of {meta['seqlen']}" if "samples" in meta else ""
    print(f"{meta['sha256']}  {out}  ({len(rows)} rows{packed}, thinking {a.thinking})")
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``qda calib render``."""
    p = sub.add_parser("calib", help="make calibration files for one model")
    q = p.add_subparsers(dest="what", required=True)
    one = q.add_parser("render", help="pool rows rendered with the model's own chat template, as frozen text")
    one.add_argument("--model", required=True, help="the model (its tokenizer and chat template)")
    one.add_argument("--pool", default=POOL, help=f"a directory or Hub dataset REPO[@REV] (default {POOL})")
    one.add_argument("--names", required=True, help="pool files and weights, e.g. chat=3,multilingual=2")
    one.add_argument("--rows", type=int, help="rows of the output (or --samples with --seqlen)")
    one.add_argument("--samples", type=int, help="packed samples the build reads (with --seqlen)")
    one.add_argument("--seqlen", type=int, help="tokens per sample the build reads")
    one.add_argument("--out", required=True, help="the JSONL file to write (refused if it exists)")
    one.add_argument("--thinking", choices=POLICIES, default="keep", help="reasoning policy (default keep)")
    one.add_argument("--seed", type=int, default=0, help="the mixing seed (default 0)")
    one.set_defaults(run=_run)
