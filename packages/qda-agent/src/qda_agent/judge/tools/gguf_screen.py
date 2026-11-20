"""The in-flow GGUF reading: decode a candidate `.gguf` to bf16, read the named suite, delete the decode.

    python gguf_screen.py SPEC.json

Every `.gguf` candidate a run scores, a hand-in included, comes through this path. The judge writes `SPEC` (the
candidate `.gguf`, the HF snapshot its converter read, a new node-local decoded directory, the suite JSON, the
reference and `OUT`); the decoder rebuilds a servable bf16 checkpoint from the file, and the suite runs on that
checkpoint through `qda eval`'s own runner exactly as `suite_job.py` does. No llama.cpp process runs here. The
decoded directory is removed in every case. One result is published, or none.

A file the decoder cannot invert (an architecture it does not support yet) is refused by name with no result:
an in-flow reading never falls back to another engine, or one run would rank candidates from two cohorts.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from qda_kl.cli import publish
from qda_kl.eval.config import EvalConfig
from qda_kl.eval.run import run
from qda_kl.eval.suites import SuiteResult, parse_suite


def main(argv: list[str]) -> int:
    """Decode the candidate, run the suite on the decoded checkpoint, publish, and remove the decode."""
    spec_path = Path(argv[0])
    try:
        spec = json.loads(spec_path.read_text())
        gguf, snapshot, decoded = Path(spec["gguf"]), Path(spec["snapshot"]), Path(spec["decoded"])
        suite, reference, out = json.dumps(spec["suite"]), spec["reference"], Path(spec["out"])
    except (OSError, ValueError, KeyError, TypeError) as why:
        print(f"gguf_screen: refused: {spec_path}: {why}", file=sys.stderr)
        return 2
    # This job owns only what it creates: a supplied directory that already exists is refused untouched, so a
    # refusal or a crash can never remove a tree that was there before the job ran.
    if decoded.exists():
        print(f"gguf_screen: refused: {decoded}: exists; the decoded directory must be new", file=sys.stderr)
        return 2
    try:
        try:
            from qda_quant.gguf.decode import decode

            manifest = decode(gguf, decoded, snapshot)
        except Exception as why:  # noqa: BLE001 -- any failure leaves no reading; a refused one is not a verdict
            print(f"gguf_screen: refused: {gguf}: {why}", file=sys.stderr)
            return 2
        return _suite(suite, str(decoded), reference, out, manifest)
    finally:
        # `decode` creates the directory (reached only when it did not exist), so everything under it belongs
        # to this job: remove it on success, on refusal, and on an interrupt (BaseException) alike.
        _rmtree(decoded)


def _suite(suite: str, checkpoint: str, reference: str, out: Path, manifest: dict) -> int:
    """Run the named suite on `checkpoint`; publish its result with the decode's identity recorded."""
    # One work directory per job: `run` writes a `resolved.yaml` at the root of its `out`, so jobs sharing one
    # would overwrite each other's config record.
    key, work = out.stem, out.parent / "suites" / out.stem
    try:
        # `parse_suite` is the validation `EvalConfig` applies to a raw `{key: settings}` entry; building the
        # `Suite` here keeps the constructor's declared `list[Suite]` honest instead of leaning on its
        # `mode="before"` validator with a plain dict.
        suites = [parse_suite(_language_only(json.loads(suite)), "suites.0")]
        cfg = EvalConfig(reference=reference, checkpoints={key: checkpoint}, out=str(work), suites=suites)
    except (TypeError, ValueError) as why:
        print(f"gguf_screen: refused: {why}", file=sys.stderr)
        return 2
    if rc := run(cfg):
        return rc
    result = work / "results" / key / f"{cfg.suites[0].name}.json"
    if not result.is_file():
        print(f"gguf_screen: refused: suite {cfg.suites[0].name} splits into parts", file=sys.stderr)
        return 2
    got = SuiteResult.model_validate_json(result.read_text())
    # The reading states what it read: the decoder and its pin, the .gguf's digest, and the template it was read
    # against. A fold can refuse a reading whose stated source is another file (an in-flow reading is never
    # another engine's answer).
    got.detail["decode"] = {
        "decoder": manifest["decoder"],
        "gguf_sha256": manifest["gguf"]["sha256"],
        "gguf_bytes": manifest["gguf"]["bytes"],
        "snapshot": manifest["snapshot"],
        "files": manifest["files"],
    }
    publish(out, got.model_dump_json(indent=1))
    return 0


def _language_only(suite: dict) -> dict:
    """Load the decoded checkpoint as the language model it is.

    The decoded directory copies the snapshot's `config.json`, which still declares a vision tower, but it holds
    no visual weights: vLLM refuses to serve it unless the checkpoint-side engine is told
    ``language_model_only``. The flag rides the KL scorer's ``serve_args`` (the checkpoint's engine) and never
    the suite's, so the reference is still loaded as the full model; text rows never run the tower, so the
    reference's readings are unchanged by omitting it on the decode.
    """
    if set(suite) != {"kl"}:
        return suite  # another suite's engine settings are not this tool's to rewrite
    settings = dict(suite["kl"] or {})
    # `KLSuite._scorer` merges this suite-level value over the scorer's (`{**scorer, **suite}`), so a suite value
    # would silently win over the flag below — and it would also reach the reference engine, which must stay the
    # full model. Refuse it by name instead of composing a checkpoint engine that is not language-only.
    if "language_model_only" in (settings.get("serve_args") or {}):
        raise ValueError(
            "SUITE kl.serve_args names language_model_only; the tool sets it on the checkpoint only"
        )
    scorer = settings.get("scorer") or {"topk": {}}
    if not isinstance(scorer, dict) or len(scorer) != 1:
        raise ValueError(f"SUITE kl.scorer: expected one named scorer, got {scorer!r}")
    name, own = next(iter(scorer.items()))
    own = dict(own or {})
    own["serve_args"] = {**(own.get("serve_args") or {}), "language_model_only": True}
    settings["scorer"] = {name: own}
    return {"kl": settings}


def _rmtree(p: Path) -> None:
    """Remove a decoded checkpoint, ignoring one that is already gone."""
    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
