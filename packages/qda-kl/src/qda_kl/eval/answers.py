r"""The paper's answer graders for AIME, HMMT and GPQA: strict ``\boxed{}`` first, the lenient patterns after.

Vendored from the code that produced the paper's numbers (protocol v2.1, commit d32dfe8e: ``qda.scoring``'s
``math_answer``, ``answer_region``, ``math_expr`` and ``cli.grade``). The extraction rules, patterns and limits are
copied unchanged, so a regrade of the paper's raw generations reproduces its per-draw verdicts. Each grader reads
an answer region, the text after the model's first closing thinking tag
(:func:`qda_kl.eval.lm_eval.answer_region`); an unclosed thought leaves an empty region, which reads as no answer.

* The strict reading is the last ``\boxed{}`` whose body the axis can read, braces counted, so an unreadable
  trailing box falls through to the one before it.
* Only when no box reads, the lenient patterns run on the region's tail: for maths a ``\boxed`` whose backslash
  came out as a backspace, "final answer: X", "the answer is X" and ``<answer>X</answer>``, the last that reads;
  for letters a bare A-D alone on the last line, else the last "Answer: X".
* AIME bodies must be one integer, GPQA bodies one option letter. HMMT bodies are closed-form numbers
  (fractions, radicals, powers, ``\pi``), read by a small LaTeX-to-SymPy translator and compared by exact
  symbolic equivalence, with a 1e-60 tolerance at magnitude-scaled precision only for irrational forms SymPy
  does not cancel. Lists, sets, intervals and free symbols are unreadable, never a lucky match.
"""

from __future__ import annotations

import re
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import sympy
from sympy.parsing.sympy_parser import parse_expr, rationalize, standard_transformations

#: Where a ``\boxed{...}`` begins; its closing brace is found by counting.
BOXED_OPEN = re.compile(r"\\boxed\s*\{")

#: Presentation-only wrappers, unwrapped by name: ``\text{A}`` means A, ``\dfrac{1}{2}`` does not mean 1.
WRAPPER = re.compile(r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|mbox|operatorname)\s*\{([^{}]*)\}")

_INT = re.compile(r"-?\d+")
#: The body must be an option, not merely contain one of its letters (the A of ``\frac``).
_LETTER = re.compile(r"\(?([ABCD])(?:[).:,]|\s|$)")

#: The lenient maths patterns, tried only when no box reads; a name breaks a tie between candidates at one position.
LENIENT_MATH = [
    ("ctrl_boxed", re.compile(r"(?:\x08|(?<![\\A-Za-z]))oxed\s*\{")),
    ("final_answer", re.compile(r"final\s+answer\s*(?:is|:)?\s*[:\-]?\s*(.+)", re.I)),
    ("answer_is", re.compile(r"the\s+answer\s+is\s*:?\s*(.+)", re.I)),
    ("answer_tag", re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)),
]
#: The lenient letter patterns: "Answer: X" anywhere in the tail, or a bare letter as the last line.
LENIENT_ANSWER = re.compile(
    r"(?i)\b(?:final\s+)?answer\s*[:\uff1a]\s*\$?\s*(?:\\(?:text|textbf|mathbf|boxed)\s*\{\s*)?\(?([A-D])\)?(?![A-Za-z0-9])"
)
LENIENT_LAST_LINE = re.compile(r"^\s*\(?\$?(?:\\boxed\{)?\(?([A-D])\)?\}?\$?\)?\.?\s*$")

#: A box longer than this is prose, not an answer.
MAX_CHARS = 400
#: The largest exponent evaluated: ``3^{2025}`` is an HMMT Feb 2026 answer, ``9^{9^{9}}`` would hold the grader.
MAX_EXPONENT = 100_000
#: The most decimal digits any power or factorial may have, judged on the unevaluated tree.
MAX_DIGITS = 100_000
#: Wall clock for reading or comparing one answer; an answer that reaches it is unreadable or unequal.
SECONDS = 20
#: Absolute tolerance, at magnitude-scaled precision, for irrational forms exact arithmetic did not cancel.
TOLERANCE = sympy.Rational(1, 10**60)

NAMES = {
    "sqrt": sympy.sqrt,
    "real_root": sympy.real_root,
    "binomial": sympy.binomial,
    "factorial": sympy.factorial,
    "log": sympy.log,
    "exp": sympy.exp,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "pi": sympy.pi,
    "E": sympy.E,
    "I": sympy.I,
}
TRANSFORMS = (*standard_transformations, rationalize)
#: The only other names ``parse_expr``'s generated code may call; a missing one is a loud NameError.
_BUILDERS = {
    name: getattr(sympy, name)
    for name in ("Add", "Mul", "Pow", "Integer", "Float", "Rational", "Symbol", "factorial2")
}
_BINARY = {
    "frac": "(({0})/({1}))",
    "dfrac": "(({0})/({1}))",
    "tfrac": "(({0})/({1}))",
    "cfrac": "(({0})/({1}))",
    "binom": "binomial({0}, {1})",
    "dbinom": "binomial({0}, {1})",
    "tbinom": "binomial({0}, {1})",
}
_WORDS = {
    "pi": " pi ",
    "cdot": "*",
    "times": "*",
    "div": "/",
    "ln": "log",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
}
_SILENT = {
    "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "displaystyle", "textstyle", "quad", "qquad", ",", ";", ":", "!", " ", "limits",
}  # fmt: skip
#: Wrappers: their argument if it holds a number, else nothing (a unit).
_WRAPPERS = {
    "text",
    "textbf",
    "textit",
    "textrm",
    "mathrm",
    "mathbf",
    "mathit",
    "mbox",
    "operatorname",
    "boxed",
}
_DEGREES = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|\\degree|°|\\text\{\s*degrees?\s*\}|\s*degrees?\b")
_THOUSANDS = re.compile(r"(?<![\d.])\d{1,3}(?:(?:\{,\}|,)\d{3})+(?![\d.])")
_UNICODE = {
    "−": "-",
    "–": "-",
    "×": "*",
    "·": "*",
    "÷": "/",
    "π": "\\pi ",
    "√": "\\sqrt",
    "²": "^2",
    "³": "^3",
}
_ALLOWED = re.compile(r"[0-9A-Za-z_+\-*/()., !]*")

T = TypeVar("T")


def aime(region: str, gold: str) -> bool:
    """Whether an AIME answer region states the integer ``gold``."""
    got = answer_of(region, as_int)
    if got is None:
        got = lenient_math(region, _read_int)
    return got is not None and got == as_int(gold)


def hmmt(region: str, gold: str) -> bool:
    """Whether an HMMT answer region states a number exactly equal to ``gold``.

    Raises:
        ValueError: ``gold`` is not a closed form this grader reads, so no answer could be scored against it.
    """
    want = parse(gold)
    if want is None:
        raise ValueError(f"gold {gold!r} does not parse")
    got = answer_of(region, parse)
    if got is None:
        got = lenient_math(region, _read_expr)
    return equivalent(got, want)


def gpqa(region: str, gold: str) -> bool:
    """Whether a GPQA answer region chooses the option letter ``gold``."""
    got = answer_of(region, as_letter)
    if got is None:
        got = lenient_letter(region)
    return got is not None and got == gold.upper()


def boxed_bodies(text: str) -> list[str]:
    r"""Every ``\boxed{}`` body in ``text``, in order, braces matched rather than excluded."""
    out = []
    for opened in BOXED_OPEN.finditer(text):
        start = opened.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start + 1 : i].strip())
                    break
    return out


def answer_of(text: str, read: Callable[[str], T | None]) -> T | None:
    r"""The last ``\boxed{}`` body that ``read`` reads, scanning from the end; None when none does."""
    for body in reversed(boxed_bodies(text)):
        got = read(body)
        if got is not None:
            return got
    return None


def unwrap(raw: str) -> str:
    r"""``\text{A}`` -> ``A``, until it stops changing (at most four layers)."""
    for _ in range(4):
        got = WRAPPER.sub(r"\1", raw).strip()
        if got == raw:
            break
        raw = got
    return raw


def as_int(raw: str) -> str | None:
    r"""A body as a bare integer string, give or take a ``label =`` prefix; ``\dfrac{1200}{7}`` is None."""
    got = unwrap(raw)
    if "=" in got:
        got = got.rsplit("=", 1)[1]
    got = got.replace(",", "").replace(" ", "").replace("\\!", "")
    return str(int(got)) if _INT.fullmatch(got) else None


def as_letter(raw: str) -> str | None:
    """A body as one of A-D, or None."""
    got = unwrap(raw)
    if "=" in got:
        got = got.rsplit("=", 1)[1]
    found = _LETTER.match(got.strip().upper())
    return found.group(1) if found else None


def lenient_math(region: str, read: Callable[[str], T | None]) -> T | None:
    """The last lenient candidate in the region's last 4,000 characters that ``read`` reads."""
    tail = region[-4000:]
    candidates = []
    for name, pattern in LENIENT_MATH:
        for found in pattern.finditer(tail):
            if name == "ctrl_boxed":
                body = _brace_body(tail, found.end() - 1)
            elif name == "answer_tag":
                body = found.group(1)
            else:
                body = found.group(1).split("\n")[0][:200]
            candidates.append((found.start(), name, body))
    for _, _, body in sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True):
        got = read(body) if body is not None else None
        if got is not None:
            return got
    return None


def lenient_letter(region: str) -> str | None:
    """A bare letter on the region's last non-empty line, else its last "Answer: X" (last 1,500 characters)."""
    text = region.replace("*", "")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if lines and (found := LENIENT_LAST_LINE.match(lines[-1])):
        return found.group(1).upper()
    said = list(LENIENT_ANSWER.finditer(text[-1500:]))
    return said[-1].group(1).upper() if said else None


def _brace_body(text: str, start: int) -> str | None:
    """The body of the brace group opening at ``text[start]``, within 400 characters, else None."""
    depth = 0
    for i in range(start, min(len(text), start + 400)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1 : i]
    return None


def _clean(raw: str) -> str:
    """A lenient candidate without emphasis, math delimiters, an outer box or a final full stop."""
    s = re.sub(r"^[*_\s]+|[*_\s]+$", "", raw.strip())
    for delimiter in ("$", "\\(", "\\)", "\\[", "\\]"):
        s = s.replace(delimiter, "")
    s = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", s)
    return s.strip().rstrip(".").strip()


def _read_int(raw: str) -> str | None:
    """A lenient AIME candidate as an integer string: degree marks dropped, an integral closed form accepted."""
    s = re.sub(r"\^\s*\{?\\circ\}?|°", "", _clean(raw))
    got = as_int(s)
    if got is None:
        value = parse(s)
        if value is not None and value.is_Integer:
            got = str(int(value))
    return got


def _read_expr(raw: str) -> sympy.Expr | None:
    """A lenient HMMT candidate as a closed-form number."""
    return parse(_clean(raw))


def parse(raw: str) -> sympy.Expr | None:
    r"""One answer body as an exact SymPy number, or None if it is not a closed-form real number read here."""
    if not raw.strip() or len(raw) > MAX_CHARS:
        return None
    try:
        with _deadline(SECONDS):
            py = _implicit(_translate(_normalise(raw)))
            names = set(re.findall(r"[A-Za-z_]+", py))
            if not py.strip() or not _ALLOWED.fullmatch(py) or not names <= set(NAMES):
                return None
            # Unevaluated first, so a power tower is refused before anything computes it.
            if not _bounded(_parse(py, evaluate=False)):
                return None
            got = _parse(py, evaluate=True)
            if (
                not isinstance(got, sympy.Expr)
                or got.free_symbols
                or not got.is_number
                or got.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan)
                or got.is_real is False
            ):
                return None
    except NameError:
        raise  # a builder missing from `_BUILDERS`: a grader bug, never an unreadable answer
    except Exception:  # noqa: BLE001 - any reader failure (TokenError, RecursionError, SymPy's own) is unreadable
        return None
    return got


def equivalent(got: sympy.Expr | None, gold: sympy.Expr | None) -> bool:
    """Whether two parsed answers are the same number: exactly, or within the tolerance for uncancelled forms."""
    if got is None or gold is None:
        return False
    try:
        with _deadline(SECONDS):
            diff = sympy.expand(got - gold)
            if diff == 0:
                return True
            if diff.is_Rational:
                return False
            if sympy.simplify(diff) == 0:
                return True
            # Precision scaled to the magnitude, so the threshold can be absolute.
            scale = max(abs(sympy.N(got, 15)), abs(sympy.N(gold, 15)), 1)
            digits = 100 + int(sympy.N(sympy.log(scale, 10), 15))
            if digits > MAX_DIGITS + 100:
                return False
            return bool(abs(sympy.N(diff, digits)) < TOLERANCE)
    except NameError:
        raise
    except Exception:  # noqa: BLE001 - a comparison that fails is unequal, never a crashed run
        return False


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    """Raise TimeoutError after ``seconds`` in the main thread; a no-op in any other thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(_signum, _frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _group(s: str, i: int) -> tuple[str, int]:
    """The argument starting at ``s[i]`` (a ``{...}`` group, a command or one character) and the index after it."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise ValueError("missing argument")
    if s[i] == "{":
        depth = 0
        for j in range(i, len(s)):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                return s[i + 1 : j], j + 1
        raise ValueError("unbalanced braces")
    if s[i] == "\\" and (name := re.match(r"\\([A-Za-z]+)", s[i:])):
        return s[i : i + name.end()], i + name.end()
    return s[i], i + 1


def _translate(s: str) -> str:
    """LaTeX to a SymPy-parsable string; raises ValueError on anything outside the closed forms read here."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            found = re.match(r"\\([A-Za-z]+|.)", s[i:])
            if found is None:
                raise ValueError("a trailing backslash")
            name = found.group(1)
            i += found.end()
            if name in _SILENT:
                continue
            if name in _BINARY:
                one, i = _group(s, i)
                two, i = _group(s, i)
                out.append(_BINARY[name].format(_translate(one), _translate(two)))
            elif name == "sqrt":
                while i < len(s) and s[i] == " ":
                    i += 1
                index = None
                if i < len(s) and s[i] == "[":
                    close = s.find("]", i)
                    if close < 0:
                        raise ValueError("unclosed root index")
                    index, i = s[i + 1 : close], close + 1
                arg, i = _group(s, i)
                out.append(
                    f"sqrt({_translate(arg)})"
                    if index is None
                    else f"real_root({_translate(arg)}, ({_translate(index)}))"
                )
            elif name in _WRAPPERS:
                arg, i = _group(s, i)
                if re.search(r"\d", arg):
                    out.append(f"({_translate(arg)})")
            elif name in _WORDS:
                out.append(_WORDS[name])
            else:
                raise ValueError(f"unread command \\{name}")  # `\{`, a set, among them
        elif c == "{":
            arg, i = _group(s, i)
            out.append(f"({_translate(arg)})")
        elif c == "^":
            arg, i = _group(s, i + 1)
            out.append(f"**({_translate(arg)})")
        elif c in "_[]<>|;&$%=":
            raise ValueError(f"unread symbol {c!r}")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _normalise(raw: str) -> str:
    """The body with presentation removed (layout, units, degrees, thousands separators), still LaTeX."""
    s = " ".join(raw.split())
    for old, new in _UNICODE.items():
        s = s.replace(old, new)
    s = s.replace("$", "").replace("\\!", "").replace("~", " ")
    # The imaginary unit, so a Cardano form with a real value is read; `parse` refuses a non-real value.
    s = re.sub(r"\\(?:mathrm|mathit|text)\s*\{\s*i\s*\}", " i ", s)
    s = re.sub(r"(?<![A-Za-z\\])i(?![A-Za-z])", " I ", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    if "\\approx" in s:
        s = s.split("\\approx", 1)[0]
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    s = _DEGREES.sub("", s)
    s = _THOUSANDS.sub(lambda m: m.group(0).replace("{,}", "").replace(",", ""), s)
    if "," in s.replace("\\,", ""):
        raise ValueError("a list")
    s = s.strip().rstrip(".").strip()
    # `\frac12` and `\sqrt2`: the one-token shorthand, given its braces.
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    return re.sub(r"\\sqrt\s*(\d+)", r"\\sqrt{\1}", s)


def _implicit(py: str) -> str:
    """Explicit ``*`` where LaTeX writes a product by juxtaposition."""
    py = re.sub(r"(?<![A-Za-z_])e(?![A-Za-z_])", "E", py)
    for _ in range(2):
        py = re.sub(r"(\d|\))\s*(?=[(A-Za-z])", r"\1*", py)
        py = re.sub(r"\b(pi|E|I)\s*(?=[(\dA-Za-z])", r"\1*", py)
        py = re.sub(r"\)\s*(?=\d)", ")*", py)
    return py


def _bounded(node: sympy.Basic) -> bool:
    """Whether no power, factorial or exponential in the unevaluated tree is too large to evaluate."""
    if not all(_bounded(arg) for arg in node.args):
        return False
    if isinstance(node, sympy.Pow):
        exponent = node.exp.doit() if node.exp.args else node.exp
        if exponent.is_number:
            size = abs(sympy.N(exponent, 15))
            if not size.is_finite or size > MAX_EXPONENT:
                return False
    if (
        isinstance(node, (sympy.factorial, sympy.factorial2, sympy.exp))
        and node.args[0].is_number
        and abs(sympy.N(node.args[0], 15)) > MAX_EXPONENT
    ):
        return False
    if isinstance(node, sympy.Pow) and node.exp.is_number:
        # A rational base is raised exactly: height x exponent bounds the result's size.
        base = node.base.doit() if node.base.args else node.base
        if base.is_Rational and base != 0:
            height = max(abs(base.p).bit_length(), base.q.bit_length()) * 0.30103  # pyright: ignore[reportAttributeAccessIssue]
            if height * abs(float(sympy.N(node.exp, 15))) > MAX_DIGITS:
                return False
    if isinstance(node, (sympy.Pow, sympy.factorial, sympy.factorial2)) and node.is_number:
        size = abs(sympy.N(node, 15))
        if (
            size.is_number
            and size != 0
            and (not size.is_finite or abs(sympy.N(sympy.log(size, 10), 15)) > MAX_DIGITS)
        ):
            return False
    return True


def _parse(py: str, *, evaluate: bool) -> sympy.Basic:
    """``parse_expr`` over an already whitelisted string, with decimals made exact."""
    got = parse_expr(
        py,
        local_dict=dict(NAMES),
        global_dict=dict(_BUILDERS),
        transformations=TRANSFORMS,
        evaluate=evaluate,
    )
    if not isinstance(got, sympy.Basic):
        raise ValueError("not an expression")
    return got
