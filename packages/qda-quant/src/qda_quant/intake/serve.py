"""The serving check: load a checkpoint in stock vLLM and name every tensor it never read."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ..output.vllm_env import DEFAULT_GPU_MEMORY_UTILIZATION
from .probe import Source, _tied

_SERVE = "QDA_SERVE "


#: names a vLLM model class drops on purpose, so no weight loader sees them: MTP layers and rotary buffers.
#: A head tied to the embedding is dropped too, if it stores the embedding's own bytes (``_classify``)
_DROPPED_PREFIXES = ("mtp.",)


_DROPPED_SUFFIXES = (".inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached")


def _classify(unread: set[str], tied: bool, tensors: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(unconsumed, dropped) of the names vLLM never read.

    ``dropped`` are the names a model class ignores on purpose: an MTP layer, an unused rotary buffer, or a
    ``lm_head`` under a tied config that stores the embedding's own bytes. A different head under a tied config
    is a checkpoint its config describes wrongly: vLLM serves the embedding as the head, so it is reported.
    """
    import torch

    emb = next((t for n, t in tensors.items() if n.endswith("embed_tokens.weight")), None)

    def copy_of_embedding(name: str) -> bool:
        t = tensors.get(name)
        return (
            t is not None
            and emb is not None
            and t.dtype == emb.dtype
            and t.shape == emb.shape
            and torch.equal(t, emb)
        )

    dropped = {
        n
        for n in unread
        if n.startswith(_DROPPED_PREFIXES)
        or n.endswith(_DROPPED_SUFFIXES)
        or (tied and n.startswith("lm_head.") and copy_of_embedding(n))
    }
    return sorted(unread - dropped), sorted(dropped)


def _reader() -> type:
    """A ``torch.Tensor`` subclass that sets ``read`` when an operation reads its values.

    Metadata queries (``.shape``, ``.dtype``, ``.dim()``, ``len()``) do not count: vLLM asks them about tensors
    it then drops without loading (its MoE loader tests ``.dim()`` on every expert tensor).
    """
    import torch

    def getter(attr: str) -> Any:
        return getattr(torch.Tensor, attr).__get__

    metadata = frozenset({
        torch.Tensor.dim, torch.Tensor.size, torch.Tensor.numel, torch.Tensor.__len__,
        torch.Tensor.stride, torch.Tensor.is_floating_point, torch.Tensor.is_contiguous,
        torch.Tensor.is_complex, torch.Tensor.element_size, getter("shape"), getter("dtype"),
        getter("device"), getter("ndim"), getter("is_cuda"), getter("layout"), getter("requires_grad"),
    })  # fmt: skip

    class Read(torch.Tensor):
        read: bool

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func not in metadata:
                for x in (*args, *(kwargs or {}).values()):
                    cls._mark(x)
            return super().__torch_function__(func, types, args, kwargs)

        @classmethod
        def _mark(cls, x: Any) -> None:
            if isinstance(x, cls):
                x.read = True
            elif isinstance(x, (list, tuple)):
                for y in x:
                    cls._mark(y)

    return Read


def serve_check(path: str, gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION) -> dict[str, Any]:
    """Load `path` in stock vLLM and name every safetensors tensor its weight loading never read.

    vLLM's own completeness check covers the model's parameters and is off for quantized models, where an
    unknown tensor can be dropped silently: the checkpoint then loads and generates with a weight nothing
    uses. This reads the checkpoint the way vLLM does, in a child process (vLLM changes process-wide torch
    state), and compares every tensor with the ones a weight loader read.

    Returns:
        ``{"ok", "unconsumed", "dropped", "error"}``: ``ok`` is True when vLLM loaded the checkpoint and read
        every tensor it uses; ``unconsumed`` lists the names it read none of, ``dropped`` the ones its model
        class ignores on purpose (MTP layers, rotary buffers, a redundant copy of the embedding stored as a tied
        head); ``error`` is vLLM's refusal.

    Raises:
        RuntimeError: the child process died before it could report.
    """
    import subprocess

    from ..output.vllm_env import child_env

    argv = [
        sys.executable,
        "-m",
        "qda_quant.intake.serve",
        "--serve-child",
        str(path),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    done = subprocess.run(argv, capture_output=True, text=True, check=False, env=child_env())
    line = next((x for x in done.stdout.splitlines() if x.startswith(_SERVE)), None)
    if line is None:
        last = next((x.strip() for x in reversed(done.stderr.splitlines()) if x.strip()), "no output")
        raise RuntimeError(f"the serving check failed (exit {done.returncode}): {last}")
    return json.loads(line[len(_SERVE) :])


def _serve_in_vllm(path: str, gpu_memory_utilization: float) -> dict[str, Any]:
    from vllm import LLM
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from ..output.vllm_env import NO_JIT_ENV, engine_start, no_jit_args

    Read = _reader()
    src = Source.open(path)
    keys = set(src.tensors)
    cfg = src.jsons.get("config.json") or {}
    tied = _tied(cfg)
    pulled: dict[str, Any] = {}
    originals: dict[str, Any] = {}  # unmarked: comparing a stored head marks a tensor read
    chain = DefaultModelLoader.get_all_weights

    def get_all_weights(self, model_config, model):  # noqa: ANN001, ANN202  (vLLM's method)
        for name, tensor in chain(self, model_config, model):
            t = tensor.as_subclass(Read)
            t.read = False
            pulled[name] = t
            originals[name] = tensor
            yield name, t

    DefaultModelLoader.get_all_weights = get_all_weights  # type: ignore[method-assign]
    error = ""
    try:
        # the engine core must load in this process, or the hook would not see the load; no FlashInfer kernel is
        # JIT-built (no nvcc), as for qda-kl's engines
        args = no_jit_args({"gpu_memory_utilization": gpu_memory_utilization, "enforce_eager": True})
        engine_start({"VLLM_ENABLE_V1_MULTIPROCESSING": "0", **NO_JIT_ENV}, lambda: LLM(model=path, **args))
    except Exception as e:  # noqa: BLE001  (vLLM's refusal is the verdict, with its type)
        error = f"{type(e).__name__}: {str(e)[:500]}"
    read = {name for name, t in pulled.items() if t.read}
    # a failed load reached only the tensors it asked for; the later ones were never pulled
    unconsumed, dropped = _classify((set(pulled) if error else keys) - read, tied, originals)
    return {"ok": not error and not unconsumed, "unconsumed": unconsumed, "dropped": dropped, "error": error}


def _serve_main(argv: list[str] | None = None) -> int:
    """``python -m qda_quant.intake.serve --serve-child PATH``: print the serving check's verdict."""
    ap = argparse.ArgumentParser(description="load a checkpoint in stock vLLM and say what it never read")
    ap.add_argument("--serve-child", required=True, metavar="PATH")
    ap.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    a = ap.parse_args(argv)
    print(_SERVE + json.dumps(_serve_in_vllm(a.serve_child, a.gpu_memory_utilization)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_serve_main())
