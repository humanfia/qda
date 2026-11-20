"""Where a search's budget comes from: `qda-agent prepare --budget-bpw N | candidate0 | hf:REPO[:FILE]`.

- ``N``: that many bits per quantized weight (`judge.budget_bpw`); a vLLM search only.
- ``candidate0``: candidate 0's own price: its bpw by the judge's rule, or its `.gguf` size in bytes.
- ``hf:REPO[:FILE]``: a published checkpoint's price, read from the Hub without downloading its weights. A
  GGUF file's size is the budget in bytes (`judge.budget_bytes`); a compressed-tensors, GPTQ or AWQ
  checkpoint's safetensors headers (HTTP range reads) give its decoder Linears' stored bits, which the
  config step divides by the original's decoder-Linear weight count, the judge's own rule
  (`judge.checks.budget`). Any other checkpoint is refused.

Without the flag a vLLM search takes the row's default (`DEFAULT_BPW`) and a llama.cpp search candidate 0's
size.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = ["DEFAULT_BPW", "QUANT_METHODS", "BudgetRefused", "BudgetSpec", "HubPrice", "hub_price", "parse"]

#: The default bits per quantized weight of a row, by its family.
DEFAULT_BPW = {"w4a16": 4.4, "w4a8": 4.4, "w3a16": 3.4, "w4a4": 4.8}
#: The `quantization_config.quant_method`s whose safetensors the judge's rule can price.
QUANT_METHODS = ("compressed-tensors", "gptq", "awq")
#: One part of a split GGUF: `<stem>-00001-of-00003.gguf`.
SPLIT_GGUF = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<parts>\d{5})\.gguf$")


class BudgetRefused(ValueError):
    """A budget source the rule cannot price, with the change that fixes it (`fix`, the refusal's second line)."""

    def __init__(self, problem: str, fix: str) -> None:
        super().__init__(problem)
        self.fix = fix


@dataclass(frozen=True)
class BudgetSpec:
    """`--budget-bpw` as given: a number, candidate 0, or a Hub checkpoint."""

    kind: Literal["number", "candidate0", "hf"]
    bpw: float | None = None
    repo: str | None = None
    file: str | None = None

    def __str__(self) -> str:
        if self.kind == "number":
            return f"{self.bpw:g}"
        if self.kind == "candidate0":
            return "candidate0"
        return f"hf:{self.repo}" + (f":{self.file}" if self.file else "")


def parse(raw: str) -> BudgetSpec:
    """Read `--budget-bpw`.

    Raises:
        ValueError: neither a positive number, `candidate0`, nor `hf:ORG/NAME[:FILE]`.
    """
    raw = raw.strip()
    if raw == "candidate0":
        return BudgetSpec("candidate0")
    if raw.startswith("hf:"):
        repo, _, file = raw[3:].partition(":")
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise ValueError(f"--budget-bpw {raw!r}: hf: takes a Hub repository, ORG/NAME[:FILE]")
        return BudgetSpec("hf", repo=repo, file=file or None)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"--budget-bpw {raw!r}: a number of bits per weight, candidate0, or hf:ORG/NAME[:FILE]"
        ) from None
    if not value > 0 or math.isinf(value):  # NaN included
        raise ValueError(f"--budget-bpw {raw!r}: the budget is a positive number of bits per weight")
    return BudgetSpec("number", bpw=value)


@dataclass(frozen=True)
class HubPrice:
    """What a Hub checkpoint costs: a GGUF file's bytes, or the stored bits of its decoder Linears.

    Attributes:
        kind: `gguf` (`size` is the file's bytes) or the safetensors checkpoint's `quant_method`.
        repo: The repository read.
        revision: The commit read: a rerun of the same prepare reads the record, not the Hub (`record`).
        file: The GGUF file read (its first part for a split GGUF), or None.
        size: A GGUF file's bytes (every part of a split GGUF).
        bits: A safetensors checkpoint's decoder-Linear stored bits (`judge.checks.budget.stored_bits`).
        linears: The names of its decoder Linears, to tell whether it is a quantization of this model.
    """

    kind: str
    repo: str
    revision: str
    file: str | None = None
    size: int | None = None
    bits: int | None = None
    linears: tuple[str, ...] = ()

    @property
    def source(self) -> str:
        """``REPO@REVISION[:FILE]``, as the record and the config's note name it."""
        return f"{self.repo}@{self.revision}" + (f":{self.file}" if self.file else "")

    def record(self) -> dict[str, Any]:
        """What prepare.json keeps, so the same prepare run again prices the same commit without the Hub."""
        return {"kind": self.kind, "repo": self.repo, "revision": self.revision, "file": self.file,
                "size": self.size, "bits": self.bits, "linears": list(self.linears)}  # fmt: skip

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> HubPrice:
        return cls(str(r["kind"]), str(r["repo"]), str(r["revision"]), r.get("file"), r.get("size"),
                   r.get("bits"), tuple(r.get("linears") or ()))  # fmt: skip


def hub_price(spec: BudgetSpec) -> HubPrice:
    """Read `spec`'s Hub checkpoint: file names, sizes, `config.json` and safetensors headers, no weights.

    A split GGUF (`SPLIT_GGUF`) is priced as all its parts: `:FILE` names any one of them.

    Raises:
        ValueError: the Hub cannot be read, the repository holds no GGUF or several without `:FILE`, or a
            part of a split GGUF is missing.
        BudgetRefused: a safetensors checkpoint of another kind than `QUANT_METHODS`.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from qda_agent.judge.checks.budget import linear_names, stored_bits

    repo = str(spec.repo)
    api = HfApi()
    try:
        revision = api.model_info(repo).sha
        files = api.list_repo_files(repo, revision=revision)
    except Exception as e:  # noqa: BLE001 -- any Hub error (offline, gated, missing) is the refusal
        raise ValueError(f"--budget-bpw {spec}: cannot read the Hub repository ({e})") from None
    ggufs = sorted(f for f in files if f.endswith(".gguf"))
    if spec.file is not None or (ggufs and not any(f.endswith(".safetensors") for f in files)):
        if spec.file is not None and spec.file not in files:
            raise ValueError(f"--budget-bpw {spec}: {repo} has no file {spec.file}")
        name = spec.file or (ggufs[0] if len(ggufs) == 1 else None)
        if name is None:
            raise ValueError(
                f"--budget-bpw {spec}: {repo} holds {len(ggufs)} GGUF files; name one: hf:{repo}:FILE"
            )
        if not name.endswith(".gguf"):
            raise ValueError(f"--budget-bpw {spec}: {name} is not a .gguf; FILE names a GGUF file")
        names = [name]
        if split := SPLIT_GGUF.match(name):  # every part is stored: the budget is their sum
            stem, parts = split["stem"], int(split["parts"])
            names = [f"{stem}-{i:05d}-of-{parts:05d}.gguf" for i in range(1, parts + 1)]
            if missing := [n for n in names if n not in files]:
                raise ValueError(f"--budget-bpw {spec}: {repo} is missing part(s) {', '.join(missing)} of {name}")
            name = names[0]
        sizes = [getattr(info, "size", None) for info in api.get_paths_info(repo, names, revision=revision)]
        if len(sizes) != len(names) or None in sizes:  # a RepoFile has a size; a folder of that name does not
            raise ValueError(f"--budget-bpw {spec}: {name} is not a file")
        return HubPrice("gguf", repo, str(revision), name, size=sum(int(n) for n in sizes if n is not None))
    if "config.json" not in files or not any(f.endswith(".safetensors") for f in files):
        raise ValueError(
            f"--budget-bpw {spec}: {repo} holds neither a GGUF file nor safetensors with a config.json"
        )
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    q = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    method = q.get("quant_method") if isinstance(q, dict) else None
    if method not in QUANT_METHODS:
        raise BudgetRefused(
            f"--budget-bpw {spec}: {repo} is {method or 'not quantized'}, and its bits per weight cannot be read",
            "name a GGUF file (hf:ORG/NAME:FILE.gguf), a compressed-tensors, GPTQ or AWQ safetensors checkpoint, "
            "or a number (--budget-bpw 4.4)",
        )
    meta = api.get_safetensors_metadata(repo, revision=revision)
    headers = {
        name: {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": list(t.data_offsets)}
        for fm in meta.files_metadata.values()
        for name, t in fm.tensors.items()
    }
    return HubPrice(
        str(method), repo, str(revision), bits=stored_bits(headers), linears=tuple(sorted(linear_names(headers)))
    )
