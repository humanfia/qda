"""Exact refusal text, one golden file per module: `golden/errors_<module>.txt`, one `name || text` row each.

A test module keeps a dict of renderers (name -> callable returning the message) and one test parametrised
over it that calls `check_golden`. Behaviour (what is refused, what is left alone) stays in the module's
own tests; the golden file pins the wording, so a reword shows up as one diff in review.
Regenerate after an intended reword: rerun that test with `QDA_GOLDEN_UPDATE=1`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from qda_quant.step import Refused

GOLDEN = Path(__file__).parent / "golden"


def refusal(call: Callable[[], object], error: type[BaseException] = Refused) -> str:
    """The message `call` refuses with; fails the test when it does not refuse.

    A pydantic error is rendered as `loc: msg` lines: its own text adds the pydantic version's URL and a
    truncated repr of the input, which would make the row depend on the install and the temp path.
    """
    from pydantic import ValidationError

    with pytest.raises(error) as e:
        call()
    if isinstance(e.value, ValidationError):
        return "\n".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.value.errors())
    return str(e.value)


def check_golden(module: str, name: str, text: str, names: list[str]) -> None:
    """Compare one rendered message with its row; `names` is every row the module renders, in order."""
    path = GOLDEN / f"errors_{module}.txt"
    line = text.replace("\n", "\\n")
    rows = dict(r.split(" || ", 1) for r in path.read_text().splitlines()) if path.exists() else {}
    if os.environ.get("QDA_GOLDEN_UPDATE") == "1":
        rows[name] = line
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{n} || {rows[n]}\n" for n in names if n in rows))
        return
    assert list(rows) == names, f"{path} rows differ from the renderers: regenerate with QDA_GOLDEN_UPDATE=1"
    assert rows[name] == line, f"{path} row {name!r} is stale:\n- {rows[name]}\n+ {line}"
