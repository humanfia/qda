"""Run one speed-bench cell with QDA or llm-compressor and append one JSON line.

    python bench/bench_cell.py --library {qda,llmc} --model MODEL --method {gptq,awq-rtn,rtn} \\
        --cards 1 --calib CALIB.jsonl --out NEW_DIR [--json cells.jsonl]

Both sides read calibration through QDA's reader and quantize the same modules: the bf16-keep list is derived
from QDA's graph, not hand-written. A cell a library cannot run, or whose module coverage differs, gives an
``unsupported`` line, never a timed substitute recipe; a cell that ran out of memory says ``oom`` and any other
failure says ``failed``, so a row's status names the cause. The llmc cell needs llm-compressor; both need
qda-quant.

Every line names the interpreter that ran and the installed stack, whatever its status, so a failed cell is as
auditable as a timed one. ``fla_kernel_available`` is transformers' own check (the package and a CUDA, XPU or MLU
device); it is not proof that a GDN kernel ran, and a run with CUDA hidden reports false.
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

START = time.perf_counter()  # process wall clock, imports included; a cell's own clock excludes them
from importlib import metadata, util  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from qda_quant.step import Refused  # noqa: E402
from transformers.utils.import_utils import is_flash_linear_attention_available  # noqa: E402

SAMPLES, SEQLEN, SEED = 128, 2048, 0  # 128 x 2,048 token rows, seed 0 on both sides
#: the shared environment a cell ran in, by distribution name as it installs
STACK = (
    "torch",
    "transformers",
    "llmcompressor",
    "compressed-tensors",
    "flash-linear-attention",
    "qda-quant",
)
CELLS = {
    "gptq": ("w4a16-sym-g128", [{"gptq": {"actorder": "static", "damp": 0.01, "block": 128}}]),
    "awq-rtn": ("w4a16-asym-g128", [{"awq": {}}, {"rtn": {}}]),
    "rtn": ("w4a16-sym-g128", [{"rtn": {}}]),
}
LIBRARY_CODE = {"qda": ("qda-quant", "qda_quant"), "llmc": ("llmcompressor", "llmcompressor")}


def coverage(model_id):
    from qda_quant.arch import describe
    from qda_quant.intake.loading import load_model

    with torch.device("meta"):
        skeleton = load_model(model_id, weights=False, dtype=torch.bfloat16)
    graph = describe(skeleton, verify=False)
    quantized = {module for layer in graph.layers for group in layer.groups for module in group.modules}
    for bank in (bank for layer in graph.layers for bank in layer.banks):
        quantized.update(
            [bank.experts]
            if isinstance(bank.experts, str)
            else [n for ins, out in bank.experts for n in (*ins, out)]
        )
    linears = {name for name, module in skeleton.named_modules() if isinstance(module, torch.nn.Linear)}
    return quantized, linears, sorted(linears - quantized)


def _smi(query):
    try:
        run = subprocess.run(["nvidia-smi", *query.split()], capture_output=True, text=True, timeout=5)
        return run.stdout
    except OSError:
        return ""


def _visible_gpus():
    rows = [x.split(", ") for x in _smi("--query-gpu=index,uuid,name --format=csv,noheader").splitlines()]
    spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    by_key = {key: row for row in rows for key in row[:2]}
    chosen = rows if spec is None else [by_key[k.strip()] for k in spec.split(",") if k.strip() in by_key]
    return [(uuid, name) for _index, uuid, name in chosen]


def _tree_rss():
    children = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError, ValueError):
            stat = (entry / "stat").read_text()
            children.setdefault(int(stat[stat.rfind(")") + 2 :].split()[1]), []).append(int(entry.name))
    total, stack = 0, [os.getpid()]
    while stack:
        pid = stack.pop()
        with contextlib.suppress(OSError, ValueError):
            total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        stack += children.get(pid, [])
    return total


class Peaks:
    def __init__(self, gpus, interval=0.5):
        self.gpu_mib = dict.fromkeys((uuid for uuid, _name in gpus), 0)
        self.rss, self.interval, self._stop = 0, interval, threading.Event()

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        self.rss = max(self.rss, _tree_rss())
        lines = _smi("--query-gpu=uuid,memory.used --format=csv,noheader,nounits").splitlines()
        for uuid, mib in ((u.strip(), int(m)) for u, m in (x.split(",") for x in lines if "," in x)):
            if uuid in self.gpu_mib:
                self.gpu_mib[uuid] = max(self.gpu_mib[uuid], mib)


def _versions():
    """The installed stack the running interpreter sees; a package that is not installed reads ``absent``."""
    found = {}
    for name in STACK:
        with contextlib.suppress(metadata.PackageNotFoundError):
            found[name] = metadata.version(name)
    return {name: found.get(name, "absent") for name in STACK}


def _git_revision():
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or None


def _imported_code(dist, module):
    """The code the cell actually runs. An editable install points at another checkout, so the driver's own
    revision (`git_revision`) says nothing about the library; record where it came from and its git state."""
    spec = util.find_spec(module)
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    root = next((p for p in origin.parents if (p / ".git").exists()), None) if origin else None
    revision = dirty = None
    if origin is not None and root is not None:
        revision = (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or None
        )
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--", str(origin.parent)],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    try:
        version = metadata.version(dist)
    except metadata.PackageNotFoundError:
        version = None
    return {
        "module": module,
        "origin": str(origin) if origin else None,
        "git_revision": revision,
        "dirty": dirty,
        "version": version,
    }


def _qda(args, quantized, linears):
    from qda_quant.calib.sources import FileSource
    from qda_quant.config import BuildConfig
    from qda_quant.engine.calib import load_samples
    from qda_quant.engine.planning import local_source
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = FileSource(path=args.calib, samples=SAMPLES, seqlen=SEQLEN, seed=SEED)
    rows = [s.ids.tolist() for s in load_samples(source, local_source(args.model, weights=False), tokenizer)]
    fmt, steps = CELLS[args.method]
    config = BuildConfig(
        model=args.model,
        format=fmt,
        steps=steps,
        out=args.out,
        calib={"file": {"path": args.calib, "samples": SAMPLES, "seqlen": SEQLEN}},
        devices=list(range(args.cards)),
        residency=args.residency,
    )
    config.build(log=lambda text: print(text, flush=True) if "layer " not in text else None)
    info = json.loads((Path(args.out) / "build.json").read_text())
    return {
        "load_to_saved_s": info["load_to_saved_s"],
        "cards_used": info["card_count"],
        "versions": info["versions"],
        "quantized": sorted(quantized & linears),
        "calib_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
    }


def failure(exc: BaseException) -> tuple[str, str]:
    """The status an unfinished cell's row gets, and its message truncated for the line.

    A planner refusal that stopped the cell before it ran is ``unsupported``; running out of the cards' memory is
    ``oom``; anything else the build raised is ``failed``. The three are different facts about a cell, so a row
    never reports one as another.
    """
    if isinstance(exc, Refused):
        return "unsupported", str(exc)[:2000]
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom", str(exc)[:2000]
    return "failed", f"{type(exc).__name__}: {exc}"[:2000]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", choices=["qda", "llmc"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=sorted(CELLS), default="gptq")
    parser.add_argument("--cards", type=int, default=1)
    parser.add_argument("--residency", default=None, help="QDA weight store override, e.g. stream")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default="cells.jsonl", help="JSONL file the cell's line is appended to")
    args = parser.parse_args()

    quantized, linears, ignore = coverage(args.model)
    gpus = _visible_gpus()
    status, error, result = "ok", None, {}
    with Peaks(gpus) as peaks:
        try:
            if args.library == "qda":
                result = _qda(args, quantized, linears)
            else:
                import llmc_cell

                result = llmc_cell.run(args.model, args.calib, args.out, args.method, args.cards, ignore)
                prepared, actual = set(result["prepared"]), set(result["quantized"])
                fused = quantized - linears  # llm-compressor linearizes these before oneshot
                if actual != prepared - set(ignore) or any(
                    not any(n.startswith(f + ".") for n in prepared) for f in fused
                ):
                    raise Refused(
                        f"coverage differs: QDA {len(quantized & linears)} linears + {len(fused)} fused, "
                        f"llm-compressor {len(actual)} of {len(prepared)} linears"
                    )
        except Exception as exc:  # a cell that did not finish is data too; never time a substitute
            status, error = failure(exc)

    line = {
        **vars(args),
        "seed": SEED,
        "samples": SAMPLES,
        "seqlen": SEQLEN,
        "python": sys.executable,
        "command": " ".join([sys.executable, *sys.argv]),
        "cards_used": result.get("cards_used"),
        "calib_sha256": result.get("calib_sha256"),
        "status": status,
        "error": error,
        "load_to_saved_s": result.get("load_to_saved_s"),
        "process_s": round(time.perf_counter() - START, 3),
        "peak_host_rss_gib": round(peaks.rss / 2**30, 2),
        "peak_gpu_mib": peaks.gpu_mib or None,
        "gpu_name": sorted({name for _uuid, name in gpus}),
        "gpu_count": len(gpus),
        "git_revision": _git_revision(),
        "library_code": _imported_code(*LIBRARY_CODE[args.library]),
        "neighbours": os.environ.get("BENCH_NEIGHBOURS") or None,  # other jobs on the node, from the caller
        "versions": {**_versions(), **(result.get("versions") or {})},
        "quantized_linears": len(result["quantized"]) if result.get("quantized") else None,
        "ignored_bf16_linears": len(ignore),
        "ignore_sha256": hashlib.sha256(json.dumps(ignore).encode()).hexdigest(),
        "causal_conv1d": metadata.version("causal-conv1d") if util.find_spec("causal_conv1d") else "absent",
        "fla_kernel_available": is_flash_linear_attention_available(),
    }
    with Path(args.json).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    print(json.dumps(line, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
