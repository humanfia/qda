"""The LiveCodeBench v6 dataset and prompts: which rows, how a problem is worded, where the answer is.

The dataset is ``livecodebench/code_generation_lite`` at a pinned revision; the v6 window is
``[2024-08-01, 2025-05-01)``, 454 of the release's 1055 problems. Test cases are downloaded into the data
directory, never committed. The ``official`` prompt is LiveCodeBench's own (a system message and the user
message of :func:`prompt_of`); the ``paper`` prompt is the paper's single user message (:func:`paper_prompt_of`).
The answer extractor follows the official runner. A cached item is rebuilt when :data:`ITEM_VERSION` changes.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: The dataset and the revision this suite pins (release_v6, 1055 problems, 2023-05-07 .. 2025-04-06).
REPO = "livecodebench/code_generation_lite"
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

#: The files that can hold a v6 problem: ``test`` .. ``test2`` end before the window, ``test3`` enters it.
FILES = ("test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")

#: The v6 window, half-open, compared on the ISO date prefix.
WINDOW = ("2024-08-01", "2025-05-01")

#: The window's problem count at :data:`REVISION`; a full run refuses another count.
PROBLEMS = 454

#: Bump when the item layout (prompt wording included) changes, so a cached selection is rebuilt.
ITEM_VERSION = 1

#: LiveCodeBench's generic prompt, as ``lcb_runner/prompts/code_generation.py`` builds it upstream.
SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification and passes all tests."
)
FORMAT_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters."
)
FORMAT_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
    "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def prompt_of(question: str, starter: str) -> str:
    """LiveCodeBench's generic question format, with the starter code or the stdin instruction."""
    text = f"### Question:\n{question}\n\n"
    if starter:
        text += f"### Format: {FORMAT_STARTER}\n```python\n{starter}\n```\n\n"
    else:
        text += f"### Format: {FORMAT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    return text + "### Answer: (use the provided format with backticks)\n\n"


#: The paper's user message (judging as described in the paper's Appendix B): no system message, the question
#: and the same format instruction, each instruction line ending in a newline.
PAPER_PROMPT = (
    "You will be given a question (problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests.\n\nQuestion: {question}\n\n{instruction}"
)


def paper_prompt_of(question: str, starter: str) -> str:
    """The paper's wording of one problem, byte for byte."""
    if starter:
        instruction = f"{FORMAT_STARTER}\n```python\n{starter}\n```\n"
    else:
        instruction = f"{FORMAT_STDIN}\n```python\n# YOUR CODE HERE\n```\n"
    return PAPER_PROMPT.format(question=question, instruction=instruction)


def code_of(generation: str | None) -> str | None:
    """The last fenced code block, as LiveCodeBench's extractor takes it; None when there is none."""
    lines = (generation or "").split("\n")
    fences = [n for n, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return None
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def in_window(row: dict) -> bool:
    """Whether a release row's contest date falls in the v6 window."""
    return WINDOW[0] <= str(row.get("contest_date", ""))[:10] < WINDOW[1]


class _PlainUnpickler(pickle.Unpickler):
    """Unpickle the JSON string LiveCodeBench encodes, and refuse to import anything the payload names."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing global {module}.{name} in private test cases")


def cases_of(row: dict) -> tuple[list[dict], str | None]:
    """The official public plus private cases of one problem, and its function name (None: stdin).

    The private block is LiveCodeBench's own encoding, ``base64 -> zlib -> pickle -> JSON string``. The
    unpickler accepts only built-in values: a tampered payload cannot make it import or call anything.
    """
    metadata = row.get("metadata")
    meta = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
    public = json.loads(row.get("public_test_cases") or "[]")
    raw = row.get("private_test_cases") or ""
    private: list[dict] = []
    if raw:
        inner = _PlainUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(raw)))).load()
        private = json.loads(inner) if isinstance(inner, str) else list(inner)
    return [*public, *private], meta.get("func_name")


def download(data: Path, name: str) -> Path:
    """Download one dataset file at the pinned revision into ``data`` and return its path.

    Raises:
        RuntimeError: the download failed; the message names the file and the reason.
    """
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=str(data)))
    except Exception as why:  # noqa: BLE001 - the hub raises its own hierarchy
        raise RuntimeError(f"livecodebench: cannot download {REPO}/{name}@{REVISION}: {why}") from why


def rows(path: Path) -> Iterator[dict]:
    """The non-empty JSON rows of one JSONL file, streamed in file order."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def item_of(row: dict, protocol: str) -> dict:
    """One problem as the selection records it, worded as ``protocol`` asks; no test cases, so a run stays small."""
    metadata = row.get("metadata")
    meta = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
    starter = (row.get("starter_code") or "").strip()
    return {
        "id": str(row["question_id"]),
        "prompt": (paper_prompt_of if protocol == "paper" else prompt_of)(row["question_content"], starter),
        "fn_name": meta.get("func_name"),
        "difficulty": row.get("difficulty"),
        "platform": row.get("platform"),
        "date": str(row.get("contest_date", ""))[:10],
    }
