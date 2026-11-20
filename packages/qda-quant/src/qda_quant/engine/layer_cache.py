"""A cache of solved layers, so a screening build re-solves only the layers whose formats changed.

A layer's solve reads the quantized stream: each solved layer is written back and run again before the next
layer gathers its statistics (`engine.runner.Pass.run`). So a layer's result is exactly determined by
everything the build shares (`solve_key`) plus the solve formats of that layer AND of every layer before it.
The exact key is therefore a chain: ``chain[i] = sha256(chain[i-1], formats[i])``.

``match: exact`` reuses a layer only under its exact chain key, so a build equals the whole build up to its
first changed layer and solves every layer from there. ``match: formats`` also reuses a layer whose own formats
match an entry solved under a different upstream: of those, the entry sharing the longest prefix of the chain
(ties: the smallest key). A changed layer is still solved, on this build's own stream, but an unchanged layer
after it keeps codes solved for a slightly different input. From a reused entry on, the chain includes that
entry, so a layer this build solves after it is never kept under a key that claims an exact upstream. Such a
build is for screening only, and ``build.json`` records, per layer, which of the three it was.

An entry holds the layer's parameters after its steps and its codes (``qparams``), on the host. Restoring one
puts both back exactly, so the writer and the next layer's forward see what a solve would have left.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict


class LayerCacheSetting(BaseModel):
    """``layer_cache`` in a config: where solved layers are kept and when one may be reused.

    Attributes:
        dir: the cache directory; one per model and recipe family is usual, on a local disk (an entry holds a
            layer's parameters and codes, about 1.5 times its bf16 size).
        match: ``exact`` reuses a layer only when every layer up to it was solved in the same formats;
            ``formats`` also reuses one whose own formats match (see the module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    dir: str
    match: Literal["exact", "formats"] = "exact"


def digest(value: Any) -> str:
    """sha256 of a JSON-able value, keys sorted."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def tensors_digest(tensors: list[torch.Tensor]) -> str:
    """sha256 over the shapes, dtypes and bytes of `tensors`, in order."""
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().contiguous().cpu()
        h.update(f"{tuple(t.shape)}{t.dtype}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def source_digest() -> str:
    """sha256 over this package's Python sources: a change to any solve code is a different key."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@dataclass
class LayerCache:
    """One build's view of a cache directory: its keys, what it reused and what it solved."""

    setting: LayerCacheSetting
    solve_key: str  #: everything a layer's result depends on besides the layers' formats
    formats: dict[str, str]  #: unit -> solve format name, for the units `plan.formats` names
    chain: list[str] = field(default_factory=list)
    record: list[dict[str, Any]] = field(default_factory=list)

    def _layer(self, module: str) -> dict[str, str]:
        return {k: f for k, f in self.formats.items() if k.startswith(module + ".")}

    def _dir(self, i: int, module: str) -> Path:
        return Path(self.setting.dir) / digest([self.solve_key, i, self._layer(module)])[:32]

    def find(self, i: int, module: str) -> Path | None:
        """The entry layer `i` reuses, or None to solve it; extends the chain either way."""
        previous = self.chain[-1] if self.chain else self.solve_key
        self.chain.append(digest([previous, module, self._layer(module)]))
        own = self._dir(i, module)
        exact = own / f"{self.chain[-1][:32]}.pt"
        if exact.is_file():
            self.record.append({"layer": i, "reused": "exact", "entry": exact.stem})
            return exact
        if self.setting.match == "formats" and own.is_dir():
            ours = [c[:32] for c in self.chain]

            def shared(meta: Path) -> int:
                theirs = json.loads(meta.read_text())["chain"]
                return next(
                    (k for k, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), len(ours)
                )

            metas = sorted(own.glob("*.json"))  # `max` keeps the first of equals: the smallest key
            if metas:
                best = max(metas, key=shared)
                self.record.append(
                    {
                        "layer": i,
                        "reused": "upstream differs",
                        "entry": best.stem,
                        "shared_layers": shared(best),
                    }
                )
                # from here the chain names what this build actually holds, so no later layer it solves is
                # kept under a key that claims an exact upstream
                self.chain[-1] = digest([self.chain[-1], "reused", best.stem])
                return best.with_suffix(".pt")
        self.record.append({"layer": i, "reused": None, "entry": self.chain[-1][:32]})
        return None

    def save(self, i: int, layer: torch.nn.Module, module: str, qparams: dict[str, dict[str, Any]]) -> None:
        """Keep layer `i`'s parameters and codes under the key `find` gave it (written atomically)."""
        own = self._dir(i, module)
        own.mkdir(parents=True, exist_ok=True)
        name = self.chain[-1][:32]
        mine = {k: _host(v) for k, v in qparams.items() if k.startswith(module + ".")}
        state = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
        tmp = own / f".{name}.{os.getpid()}.tmp"
        torch.save({"state": state, "qparams": mine}, tmp)
        tmp.replace(own / f"{name}.pt")
        (own / f"{name}.json").write_text(
            json.dumps({"chain": [c[:32] for c in self.chain], "module": module})
        )

    def restore(
        self, entry: Path, layer: torch.nn.Module, module: str, qparams: dict[str, dict[str, Any]]
    ) -> None:
        """Put an entry's parameters into `layer` and its codes into `qparams`, as the solve left them."""
        saved = torch.load(entry, map_location="cpu", weights_only=True)
        layer.load_state_dict(saved["state"])
        for k, qp in saved["qparams"].items():
            if "weight_check" in qp:  # the same bytes in this build's parameter object
                p = _parameter(layer, k[len(module) + 1 :])
                qp["weight_check"] = (id(p), qp["weight_check"][1])
            qparams[k] = qp

    def info(self) -> dict[str, Any]:
        """``build.json``'s ``layer_cache``: the setting and what each layer did."""
        reused = sum(r["reused"] is not None for r in self.record)
        return {
            **self.setting.model_dump(),
            "screening_only": any(r["reused"] == "upstream differs" for r in self.record),
            "reused_layers": reused,
            "solved_layers": len(self.record) - reused,
            "layers": self.record,
        }


def _host(qp: dict[str, Any]) -> dict[str, Any]:
    return {f: v.detach().cpu() if isinstance(v, torch.Tensor) else v for f, v in qp.items()}


def _parameter(layer: torch.nn.Module, rel: str) -> torch.nn.Parameter:
    try:
        return layer.get_parameter(rel + ".weight")
    except AttributeError:  # a fused expert parameter: the entry names the parameter itself
        return layer.get_parameter(rel)


__all__ = ["LayerCache", "LayerCacheSetting", "digest", "source_digest", "tensors_digest"]
