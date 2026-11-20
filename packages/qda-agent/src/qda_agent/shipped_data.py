"""Whether the data pools a search hands its agent (`data` component sets) hold un-rendered text.

The agent draws from a pool and renders what it draws, so a pool row holds no chat markup of any family, the
model's own included (`qda_quant.calib.template.check_file`). Checked once before the clock starts, on every
`*.jsonl` of every dataset: a set that is a file, or the `*.jsonl` files under a set that is a directory, symbolic
links followed. Each physical file is read once, however many sets or links reach it. Text rows,
``prompt`` (+ ``completion``) rows and token-id rows (decoded with the model's tokenizer) are checked; ``messages``
rows pass. Other files are named as not checked and never read. Each file is read once, streaming. The default
calibration file keeps its own rule (`qda_agent.flow.check_calib_template`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qda_agent.toolsets import Toolset

#: How many unchecked files the log line names.
NAMED = 5


def _files(path: Path) -> list[Path]:
    """Every file of a dataset at `path`, symbolic links followed (a directory reached twice is walked once)."""
    if not path.is_dir():
        return [path]
    out, seen = [], set()
    for top, dirs, names in os.walk(path, followlinks=True):
        real = os.path.realpath(top)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        out += [Path(top, n) for n in sorted(names) if Path(top, n).is_file()]
    return out


def check(toolset: Toolset, model: str) -> None:
    """Refuse a dataset file whose rows hold pre-rendered chat markup; print one line of what was checked.

    The tokenizer of `model` (the run's `model` setting) is loaded once, only for a token-id row, from the model
    resolved as a build resolves it (`qda_quant.engine.planning.local_source`, metadata files only).

    Raises:
        CalibTemplateRefused: a row holds chat markup (decoded ids included), a token-id row is not ids of the model's
            tokenizer, a file cannot be read as text, or the model's tokenizer cannot be found.
    """
    from qda_agent.components import Data
    from qda_agent.flow import CalibTemplateRefused

    sets = {n: p for c in toolset.components if isinstance(c, Data) for n, p in sorted(c.sets.items())}
    if not sets:
        return
    from qda_quant.calib.template import check_file

    loaded: list[Any] = []

    def tokenizer() -> Any:
        if not loaded:
            from qda_quant.engine.planning import local_source
            from qda_quant.intake.tokenizer import load_tokenizer
            from qda_quant.step import Refused

            try:
                source = local_source(model, weights=False)
            except Refused as e:
                raise CalibTemplateRefused(f"data sets: cannot read the model's chat template: {e}") from e
            loaded.append(load_tokenizer(source))
        return loaded[0]

    checked, rows, skipped, read = 0, 0, [], set()
    for name, path in sets.items():
        for f in _files(path):
            if f.suffix != ".jsonl":
                skipped.append(f)
                continue
            try:
                st = f.stat()
            except OSError as e:
                raise CalibTemplateRefused(f"data set {name} {f}: cannot read as JSONL text: {e}") from e
            # one physical file under two sets, a symbolic link or a hard link: read once
            if (inode := (st.st_dev, st.st_ino)) in read:
                continue
            read.add(inode)
            try:
                found = check_file(f, tokenizer)
            except (OSError, UnicodeDecodeError) as e:
                raise CalibTemplateRefused(f"data set {name} {f}: cannot read as JSONL text: {e}") from e
            if (why := found.refusal(f"data set {name}")) is not None:
                raise CalibTemplateRefused(why)
            checked, rows = checked + 1, rows + found.rows
    line = f"data sets, pre-rendered chat markup: {checked} JSONL files, {rows} rows checked"
    if skipped:
        names = ", ".join(str(p) for p in skipped[:NAMED]) + (", ..." if len(skipped) > NAMED else "")
        line += f"; {len(skipped)} files not checked (not JSONL): {names}"
    print(line, file=sys.stderr)
