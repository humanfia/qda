"""Build candidate 0 in one step: a one-step recipe, its build, and the run config entry.

A search compares every candidate with candidate 0 (`baselines[0]`) and the report's `time_ratio` divides
the search's wall time by candidate 0's build time. `qda-agent cand0` does both halves in one command:

1. compose a build recipe for a named method (`gptq` or `awq`, optionally refined by `gsq` after it),
   with the method's own defaults from `qda-quant` (no tuned value lives here: the search toolset ships no
   know-how), or `gguf-q4km`, docs/formats.md's measured llama.cpp Q4_K_M recipe, and write it beside the
   checkpoint so the build can be repeated;
2. build it on the run's cards with `qda build` (the public command; this package never imports
   `qda_quant`), and read the build's own measured time out of its `build.json`;
3. write the entry into a generated run config beside the user's (`<stem>.candidate0.yaml`, or the file
   itself with `--in-place`): `baselines[0]` is the new checkpoint and `judge.candidate0_build_s` is the
   measured build time, so nothing is hand-filled.

Building candidate 0 reads only `model`, `format`, `calib` and `toolset` (`Inputs`): the config's other
inputs (the KL reference, the holdout prompts, `baselines[0]` itself) may not exist yet, so the rest of the
config is validated only when the run loads the generated file. A `baselines` entry or a
`judge.format_reference` still left a `<...>` placeholder names the slot candidate 0 fills.

The user's own file is never rewritten unless `--in-place` is given, and then only when it holds no
comments (a YAML round trip would drop them) unless `--force` says so.

A `gguf-q4km` build fills its file from a bf16 GGUF template of the model. Candidate 0 reads the one the config
names (`judge.llamacpp.template`), or converts it once (`make_template`, `<out dir>/bf16/model-bf16.gguf`, with a
record of its source and converter commit beside it). The entry names it as `judge.llamacpp.template`, so the
search hands that same file to every candidate build instead of converting the model again.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qda_agent import configfile

__all__ = [
    "Candidate0",
    "Inputs",
    "candidate0_recipe",
    "build_candidate0",
    "entry",
    "make_template",
    "template_command",
    "write_config",
]

#: The build steps each method composes, by `qda-quant`'s public step names (`qda_quant.steps`). Empty
#: settings: every default is the method's own (`gptq`/`awq`/`gsq` are the `qda-quant` step classes'; a user
#: who wants another value writes the recipe with `--recipe-only` and builds it with `qda build`).
METHODS: dict[str, dict[str, Any]] = {"gptq": {}, "awq": {}}

#: The llama.cpp candidate 0: docs/formats.md's best measured Q4_K_M row (Qwen3-8B, KL 0.012751 over 3 runs;
#: `damp: 0.1` was chosen on that KL). Q4_K fixes the codes and groups, so the run's `format` does not apply.
GGUF = "gguf-q4km"
GGUF_RECIPE: dict[str, Any] = {
    "format": "w4a16-asym-g32",
    "grid": "q4k",
    "steps": [{"gptq": {"clip": "imatrix", "damp": 0.1}}],
    "exporter": {"gguf": {}},
}
CHOICES = (*METHODS, GGUF)

#: The run config keys a candidate 0 build reads (and `toolset.qda`, else `toolset.python`); nothing else of
#: the config is checked before it.
NEEDS = ("model", "format", "calib")


@dataclass(frozen=True)
class Inputs:
    """What building candidate 0 reads of a run config: the model, format, calibration file and `qda` command.

    Attributes:
        model: The original model.
        format: The run's format preset.
        calib: The calibration file, or None (refused at the recipe).
        program: The `qda` command (`toolset.qda`, else `toolset.python -m qda_quant.cli`), or None.
        raw: The effective run config (every `--set` applied, paths resolved), unvalidated.
    """

    model: str
    format: str
    calib: str | None
    program: list[str] | None
    raw: dict[str, Any]

    @classmethod
    def read(cls, path: Path, overrides: list[str] | None = None) -> Inputs:
        """Read only what the build needs from the run config at `path`.

        Raises:
            ConfigError: The file cannot be read, a key the build reads is a placeholder, or `model` or
                `format` is missing.
        """
        raw = configfile.read(path, overrides, need=NEEDS)
        for key in ("model", "format"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise configfile.ConfigError(f"{path}: {key}: candidate 0 needs it")
        toolset: dict[str, Any] = raw["toolset"] if isinstance(raw.get("toolset"), dict) else {}
        used = "qda" if toolset.get("qda") else "python"
        if configfile.placeholders({used: toolset.get(used)}):
            raise configfile.ConfigError(f"{path}: fill in toolset.{used} (the qda that builds candidate 0)")
        program = None
        if toolset.get("qda"):
            program = [str(a) for a in toolset["qda"]]
        elif toolset.get("python"):
            program = [str(toolset["python"]), "-m", "qda_quant.cli"]
        calib = raw.get("calib")
        return cls(raw["model"], raw["format"], str(calib) if calib else None, program, raw)

    @property
    def judge(self) -> dict[str, Any]:
        """The config's `judge` section as written (empty when it has none)."""
        return self.raw.get("judge") or {}


@dataclass(frozen=True)
class Candidate0:
    """One built candidate 0.

    Attributes:
        out: The checkpoint directory.
        recipe: The build recipe written beside it (outside the checkpoint: a build fills an empty out).
        build_seconds: The build's own measured time (`build.json`'s `load_to_saved_s`, else
            `timings.total`), or None when it records neither.
        wall_seconds: The whole `qda build` run, interpreter start included.
        method: `gptq` or `awq`, and whether `gsq` preprocessing ran.
        template: The bf16 GGUF template a `gguf-q4km` build read, or None.
    """

    out: Path
    recipe: Path
    build_seconds: float | None
    wall_seconds: float
    method: str
    gsq: bool
    template: Path | None = None


def candidate0_recipe(
    cfg: Inputs,
    *,
    method: str,
    gsq: bool = False,
    out: Path,
    calib_samples: int | None = None,
    calib_seqlen: int | None = None,
    template: Path | str | None = None,
    imatrix: Path | str | None = None,
    tensor_types: list[dict[str, str]] | None = None,
    libggml: Path | str | None = None,
) -> dict[str, Any]:
    """The one-step build recipe for `method` (a `qda-quant` build config, as `qda build` reads it).

    Args:
        cfg: The build's inputs: the model, format and calibration file.
        method: `gptq`, `awq` (`METHODS`) or `gguf-q4km` (`GGUF_RECIPE`, which sets its own format).
        gsq: Append `qda-quant`'s `gsq` step: it refines the codes the method wrote, so it runs after it.
        out: Where the build writes its checkpoint.
        calib_samples: Calibration rows; None: `qda-quant`'s default for the file.
        calib_seqlen: Tokens per row; None: `qda-quant`'s default.
        template: A `gguf-q4km` build's bf16 GGUF template (`exporter.gguf.template`); None: the build converts
            its own. Ignored by the other methods.
        imatrix: A `llama-imatrix` file (`exporter.gguf.imatrix`) that weights the fit of the tensors this
            build takes from the template; None: those are fitted unweighted. Ignored by the other methods.
        tensor_types: The `exporter.gguf.tensor_types` rules of a measured type map (the sensitivity step);
            None: the exporter's default Q4_K_M mix. Ignored by the other methods.
        libggml: The built `libggml-base` a measured type map's Q5_K cells quantize with
            (`exporter.gguf.libggml`); None: no key, as a run without a type map writes. Ignored without
            `tensor_types`.

    Returns:
        The recipe, ready to be written as YAML.
    """
    if method not in CHOICES:
        raise ValueError(f"--method {method!r} is not one of {', '.join(CHOICES)}")
    if not cfg.calib:
        raise ValueError(
            "the run config names no calibration file (calib:), so candidate 0 has no data to build on"
        )
    if gsq and method == "awq":
        # `qda-quant`'s own AWQ docstring: "both require a later `rtn` or `gptq` step to quantize", and its
        # gsq step refines codes a quantizing step wrote (`qda_quant/steps/gsq.py`: "gsq refines existing
        # codes; put gptq or rtn before gsq in the recipe"). So `awq -> gsq` has nothing to refine.
        raise ValueError(
            "--gsq with --method awq has nothing to refine: qda-quant's awq only folds scales and needs a "
            "later rtn or gptq step to quantize (qda_quant/steps/awq.py), and gsq refines codes a quantizing "
            "step wrote (qda_quant/steps/gsq.py); use --method gptq --gsq, or awq alone"
        )
    if gsq and method == GGUF:  # docs/formats.md, "Limits": AutoRound and GSQ are refused for GGUF
        raise ValueError(
            "--gsq with --method gguf-q4km: the GGUF exporter refuses gsq (docs/formats.md, Limits)"
        )
    calib: dict[str, Any] = {"path": str(cfg.calib)}
    if calib_samples is not None:
        calib["samples"] = calib_samples
    if calib_seqlen is not None:
        calib["seqlen"] = calib_seqlen
    if method == GGUF:
        gguf = {k: v for k, v in GGUF_RECIPE.items() if k != "exporter"}
        exporter = {"gguf": {}}
        if template is not None:
            exporter["gguf"]["template"] = str(template)
        if imatrix is not None:
            exporter["gguf"]["imatrix"] = str(imatrix)
        if tensor_types:
            exporter["gguf"]["tensor_types"] = [dict(rule) for rule in tensor_types]
            if libggml is not None:  # Q5_K needs native ggml_quantize_chunk, not the Python port
                exporter["gguf"]["libggml"] = str(libggml)
        return {
            "model": cfg.model,
            **gguf,
            "out": str(out),
            "calib": {"file": calib},
            "exporter": exporter,
        }
    steps: list[dict[str, Any]] = [{method: dict(METHODS[method])}]
    if gsq:  # quantize first, then refine the codes the method wrote
        steps.append({"gsq": {}})
    return {
        "model": cfg.model,
        "format": cfg.format,
        "out": str(out),
        "steps": steps,
        "calib": {"file": calib},
    }


def build_program(cfg: Inputs) -> list[str]:
    """The `qda` command that builds here: `toolset.qda` when named, else `toolset.python -m qda_quant.cli`.

    Raises:
        ValueError: The config names neither, so no interpreter that can build is known.
    """
    if cfg.program:
        return list(cfg.program)
    raise ValueError(
        "the run config names no build interpreter: set toolset.python (a virtualenv with torch and "
        "qda-quant) or toolset.qda (the qda command)"
    )


def build_candidate0(
    cfg: Inputs,
    *,
    method: str,
    out: Path,
    gsq: bool = False,
    calib_samples: int | None = None,
    calib_seqlen: int | None = None,
    cards: list[int] | None = None,
    log: Any = print,
    template: Path | None = None,
    imatrix: Path | None = None,
    tensor_types: list[dict[str, str]] | None = None,
    libggml: Path | None = None,
) -> Candidate0:
    """Write the recipe and build it on the run's cards, timing the build.

    The build runs through the public `qda build OUT/recipe.yaml`, with the same interpreter the run gives
    its jobs (or `toolset.qda`), so nothing here imports `qda_quant`. It sees the cards this shell sees, or
    exactly the ones `cards` names.

    Args:
        cfg: The build's inputs.
        method: `gptq`, `awq` or `gguf-q4km`.
        out: A path that does not exist yet: this command creates and fills it.
        gsq: Refine the method's codes with `qda-quant`'s `gsq` step (it runs after the method).
        calib_samples: Calibration rows (None: `qda-quant`'s default).
        calib_seqlen: Tokens per row (None: `qda-quant`'s default).
        cards: The cards to build on, as `CUDA_VISIBLE_DEVICES`; None: every card this shell can see (run
            under your allocation, and narrow it here when it holds more cards than you want used).
        log: Where the command writes its progress lines.
        template: A `gguf-q4km` build's bf16 GGUF template (`make_template`); None: the build converts its own.
        imatrix: A `gguf-q4km` build's `llama-imatrix` file; None: the tensors taken from the template are
            fitted unweighted.
        tensor_types: A `gguf-q4km` build's per-tensor type rules (a measured type map); None: the default mix.
        libggml: The built `libggml-base` those rules' Q5_K cells need; None: no `exporter.gguf.libggml` key.

    Returns:
        The built checkpoint and its two measured times.

    Raises:
        ValueError: `out` exists, the interpreter is unknown, or the build wrote no checkpoint.
        subprocess.CalledProcessError: The build failed (`qda build`'s own message is on the log).
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError(
            f"{out} already exists: candidate 0 is built into a new directory, never a copy of one"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    recipe = candidate0_recipe(
        cfg,
        method=method,
        gsq=gsq,
        out=out,
        calib_samples=calib_samples,
        calib_seqlen=calib_seqlen,
        template=template if method == GGUF else None,
        imatrix=imatrix if method == GGUF else None,
        tensor_types=tensor_types if method == GGUF else None,
        libggml=libggml if method == GGUF else None,
    )
    recipe_path = out.with_name(f"{out.name}.recipe.yaml")  # beside it: `qda build` wants an empty out
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    command = [*build_program(cfg), "build", str(recipe_path)]
    env = dict(os.environ)
    if cards is not None:  # default: every card this shell can see, as `qda build` resolves it
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
    log(f"candidate 0: {' '.join(command)}")
    started = time.monotonic()
    subprocess.run(command, check=True, env=env)  # a build's own output: the user reads its progress
    wall_seconds = time.monotonic() - started
    build_seconds = _recorded_seconds(out)
    if build_seconds is None:
        log(f"candidate 0: {out / 'build.json'} records no build time; the entry uses the wall clock")
    return Candidate0(
        out=out,
        recipe=recipe_path,
        build_seconds=build_seconds,
        wall_seconds=wall_seconds,
        method=method,
        gsq=gsq,
        template=template if method == GGUF else None,
    )


def template_command(python: str | Path, llama_cpp: str | Path, model: str | Path, out: Path) -> list[str]:
    """llama.cpp's own bf16 conversion of `model` into `out`: the command a GGUF build runs for its template."""
    script = Path(llama_cpp) / "convert_hf_to_gguf.py"
    return [str(python), str(script), str(model), "--outtype", "bf16", "--outfile", str(out)]


def _llama_commit(llama_cpp: str | Path) -> str | None:
    """The checkout's commit (`git rev-parse HEAD`), or None when git cannot read it."""
    try:
        r = subprocess.run(
            ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def _place(cached: Path, out: Path) -> None:
    """Put the cached template at `out` as a hard link (a copy across file systems): a link a build's sandbox
    binds by path would point outside the bind."""
    tmp = out.with_name(f".{out.name}.{os.getpid()}.link")
    tmp.unlink(missing_ok=True)
    try:
        os.link(cached, tmp)
    except OSError:  # another file system
        shutil.copyfile(cached, tmp)
    tmp.replace(out)


def make_template(
    python: str | Path,
    llama_cpp: str | Path,
    model: str | Path,
    out: Path,
    *,
    cache: Path | None = None,
    log: Any = print,
) -> Path:
    """The bf16 GGUF template of `model` at `out`: reused when its record names this source and converter commit.

    Every GGUF build of the model fills its file from this one conversion, so candidate 0 and every search
    candidate share it (`judge.llamacpp.template`) rather than each converting the model again. The record
    beside it (`<out>.json`: the resolved source, its digest, the converter's commit) is what the judge checks
    at start; a local source whose files changed (replaced weights in the same directory) fails that digest.
    ``cache`` is the machine-wide conversion (`prepare` computes its path, because this module never imports
    `qda_quant`); the cached file is hard-linked to ``out`` so a build's sandbox can bind it by path.

    Raises:
        ValueError: `out` exists with a record naming another source, another state of it, or another commit,
            or with none (an unknown file is never taken for a template), or the converter wrote nothing.
        subprocess.CalledProcessError: The converter failed (its output is in `<out>.convert.log`, which the
            error names: the converter prints the model's whole chat template, so it does not reach a terminal).
    """
    out = Path(out)
    record = out.with_name(out.name + ".json")
    from qda_agent.workspace import model_source

    named = model_source(model)  # the record's naming, which the judge's `source_dir` reads back
    source, digest = named["source"], named["source_sha256"]
    commit = _llama_commit(llama_cpp)
    if out.exists():
        try:
            made = json.loads(record.read_text())
        except (OSError, ValueError):
            raise ValueError(
                f"{out} exists without its record {record.name}: remove it, and it is converted again"
            ) from None
        stale = []
        if made.get("source") != source:
            stale.append(f"another source ({made.get('source')})")
        elif digest is not None and made.get("source_sha256") != digest:
            stale.append("another state of the model's files")
        if commit is not None and made.get("llama_cpp") != commit:
            stale.append(f"llama.cpp {made.get('llama_cpp')}, not {commit}")
        if stale:
            raise ValueError(f"{out} was made from {', '.join(stale)}: remove it, and it is converted again")
        log(f"bf16 template: {out} (made earlier, reused)")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cached = cache
    if cached is not None and cached.is_file():
        log(f"bf16 template: {cached} (the template cache: converted once per model and llama.cpp commit)")
    else:
        target = cached or out
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
        command = template_command(python, llama_cpp, model, partial)
        # the converter logs every tensor it reads and prints the model's whole chat template at INFO: its own
        # file, and one line here naming it (a terminal is not the place for a 2 KB jinja string)
        noise = out.with_name(out.name + ".convert.log")
        log(f"bf16 template: converting into {target} (one line per tensor goes to {noise})")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"CUDA_VISIBLE_DEVICES": ""}
        with noise.open("w") as quiet:
            try:
                subprocess.run(command, check=True, env=env, stdout=quiet, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise subprocess.CalledProcessError(
                    e.returncode, e.cmd, output=f"the converter's log: {noise}"
                ) from None
        if not partial.is_file():
            raise ValueError(f"convert_hf_to_gguf.py exited 0 and wrote no {partial}")
        if cached is not None:
            partial.chmod(0o444)  # read by every build of the model: never written in place
        partial.rename(target)
    if cached is not None:
        _place(cached, out)
    record.write_text(
        json.dumps({"source": source, "source_sha256": digest, "llama_cpp": commit}, indent=1) + "\n"
    )
    return out


def _recorded_seconds(out: Path) -> float | None:
    """The build's own measured seconds from its `build.json`, or None.

    `load_to_saved_s` is the whole build (model load to saved checkpoint), which is the time the report's
    `time_ratio` divides by; `timings.total` (the engine phases, without part of the I/O) is the fallback
    for a `build.json` that records only that.
    """
    try:
        record = json.loads((Path(out) / "build.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for value in (record.get("load_to_saved_s"), (record.get("timings") or {}).get("total")):
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return None


def entry(cfg: Inputs, built: Candidate0) -> dict[str, Any]:
    """The run config's new keys: candidate 0 first in `baselines` and its measured build time.

    `judge.select_candidate0` is set when the config names neither an explicit `candidate0_kl` nor
    `judge.starts`: the host then reads candidate 0's KL at start, so no KL is hand-filled either. A
    `baselines` entry still a placeholder is the slot candidate 0 fills, and so is a placeholder
    `judge.format_reference` (a checkpoint of the task's format; a llama.cpp row has none).

    Args:
        cfg: The run config the build was made for.
        built: The build.

    Returns:
        The `baselines` list, and the `judge` keys to set.
    """
    seconds = built.build_seconds if built.build_seconds is not None else round(built.wall_seconds, 3)
    judge: dict[str, Any] = {"candidate0_build_s": round(seconds, 3)}
    if cfg.judge.get("candidate0_kl") is None and not cfg.judge.get("starts"):
        judge["select_candidate0"] = True
    if configfile.placeholders({"r": cfg.judge.get("format_reference")}):
        judge["format_reference"] = None if cfg.judge.get("engine") == "llama.cpp" else str(built.out)
    if built.template is not None and cfg.judge.get("engine") == "llama.cpp":
        # the search hands this same file to every GGUF build instead of converting the model again
        judge["llamacpp"] = {**(cfg.judge.get("llamacpp") or {}), "template": str(built.template)}
    others = [str(p) for p in cfg.raw.get("baselines") or () if not configfile.placeholders({"b": str(p)})]
    return {"baselines": [str(built.out)] + [p for p in others if p != str(built.out)], "judge": judge}


def write_config(
    config_path: Path,
    *,
    raw: dict[str, Any],
    changes: dict[str, Any],
    in_place: bool = False,
    force: bool = False,
) -> Path:
    """Write the config with the entry set: a generated file beside the user's, or the user's own file.

    Args:
        config_path: The config `cand0` was given.
        raw: The effective run config (`Inputs.raw`): every `--set` override applied.
        changes: Keys to set (top-level, and inside `judge`).
        in_place: Rewrite `config_path` itself; a file with comments is refused (a YAML round trip drops
            them) unless `force`.
        force: Rewrite even then.

    Returns:
        The file to run (`qda-agent run <it>`).

    Raises:
        ValueError: `in_place` on a file with comments and no `force`.
    """
    config_path = Path(config_path)
    text = config_path.read_text()
    target = config_path if in_place else config_path.with_name(f"{config_path.stem}.candidate0.yaml")
    if in_place and not force and _has_comments(text):
        raise ValueError(
            f"{config_path} has comments, which a rewrite would drop: {config_path.stem}.candidate0.yaml "
            "is the default target, or pass --in-place --force"
        )
    merged = dict(raw)
    for key, value in changes.items():
        if key == "judge":
            merged["judge"] = {**merged.get("judge", {}), **value}
        else:
            merged[key] = value
    target.write_text(yaml.safe_dump(merged, sort_keys=False))
    return target


def _has_comments(text: str) -> bool:
    """Whether any line carries a comment (`#`, before a quote would make this a YAML parse)."""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return True
        if " #" in line:
            return True
    return False


def add_parser(sub: Any) -> Any:
    """Add the `cand0` subcommand to the CLI's subparsers and return it."""
    p = sub.add_parser("cand0", help="build candidate 0 in one step and write the run config entry")
    p.add_argument("config", type=Path, help="the run YAML the search will launch with")
    p.add_argument("--method", choices=CHOICES, default="gptq", help="the build method (default gptq)")
    p.add_argument(
        "--gsq", action="store_true", help="refine the method's codes with qda-quant's gsq step (after it)"
    )
    p.add_argument("--out", type=Path, help="the checkpoint directory (default: <config dir>/cand0-<method>)")
    p.add_argument(
        "--cards",
        nargs="+",
        help="cards to build on, as CUDA_VISIBLE_DEVICES: 0,1 or 0 1 (default: every visible card)",
    )
    p.add_argument("--recipe-only", action="store_true", help="print the recipe and stop (no build)")
    p.add_argument("--in-place", action="store_true", help="rewrite the config itself instead of a new file")
    p.add_argument("--force", action="store_true", help="with --in-place: rewrite even a file with comments")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override one key of the run config (the value is YAML); repeatable",
    )
    return p


def parse_cards(values: list[str]) -> list[int]:
    """Card indices given as `0,1,2`, `0 1 2` or both.

    Raises:
        ValueError: A part is not a card index.
    """
    try:
        return [int(c) for v in values for c in v.split(",") if c.strip()]
    except ValueError:
        raise ValueError(f"--cards {' '.join(values)!r}: card indices, as 0,1 or 0 1") from None


def _template_for(cfg: Inputs, out: Path) -> Path | None:
    """A `gguf-q4km` candidate 0's template: the config's `judge.llamacpp.template`, else one made beside `out`.

    None when no llama.cpp checkout is named (`judge.llamacpp.llama_cpp`, `QDA_LLAMA_CPP`): the build then
    names the missing checkout itself.
    """
    llamacpp = cfg.judge.get("llamacpp") or {}
    if llamacpp.get("template"):
        return Path(llamacpp["template"])  # the operator's: used as is, as the search uses it
    root = llamacpp.get("llama_cpp") or os.environ.get("QDA_LLAMA_CPP")
    if not root:
        return None
    raw_toolset = cfg.raw.get("toolset")
    toolset = raw_toolset if isinstance(raw_toolset, dict) else {}
    python = toolset.get("python")
    if not python and cfg.program and cfg.program[1:] == ["-m", "qda_quant.cli"]:
        python = cfg.program[0]
    return make_template(python or sys.executable, root, cfg.model, out.parent / "bf16" / "model-bf16.gguf")


def _imatrix_for(cfg: Inputs) -> Path | None:
    """A `gguf-q4km` candidate 0's importance matrix: the config's `judge.llamacpp.imatrix`, or None.

    A config that names one gets exactly that file (`qda-agent prepare` writes the run's one); a path that is
    not there is refused here, named, rather than failing inside the build's exporter.
    """
    named = (cfg.judge.get("llamacpp") or {}).get("imatrix")
    if not named:
        return None
    path = Path(named)
    if not path.is_file():
        raise ValueError(
            f"judge.llamacpp.imatrix: {path} is not a file (a llama-imatrix GGUF of this model); "
            "run `qda-agent prepare` with the imatrix step, or drop the setting"
        )
    return path


def run(args: Any) -> int:
    """The `cand0` subcommand: build candidate 0 and write the run config entry."""
    # The command's exit statuses only: the build itself runs through `qda build`, not this process.
    from qda_quant.exit import INTERRUPTED, refuse
    from qda_quant.step import Refused

    try:
        cfg = Inputs.read(args.config, args.overrides)
        cards = parse_cards(args.cards) if args.cards else None
    except (configfile.ConfigError, ValueError) as e:
        return refuse(str(e))
    if cfg.judge.get("starts"):
        return refuse(
            "the config labels its baselines with judge.starts; those labels are the run's own",
            "drop judge.starts from the config, or label candidate 0 yourself",
        )
    if cfg.judge.get("candidate0_kl") is not None and not args.force:
        return refuse(
            "the config names judge.candidate0_kl, which describes another checkpoint",
            "drop judge.candidate0_kl (the KL is then read at host start), or pass --force to keep it",
        )
    out = (
        Path(args.out).resolve() if args.out else Path(args.config).resolve().parent / f"cand0-{args.method}"
    )
    if args.recipe_only:
        named = (cfg.judge.get("llamacpp") or {}).get("template")
        try:
            recipe = candidate0_recipe(
                cfg,
                method=args.method,
                gsq=args.gsq,
                out=out,
                template=named,
                imatrix=_imatrix_for(cfg) if args.method == GGUF else None,
            )
        except ValueError as e:  # the same clean refusal the build path gives
            return refuse(str(e))
        sys.stdout.write(yaml.safe_dump(recipe, sort_keys=False))
        return 0
    try:
        imatrix = _imatrix_for(cfg) if args.method == GGUF else None
        built = build_candidate0(
            cfg,
            method=args.method,
            out=out,
            gsq=args.gsq,
            cards=cards,
            template=_template_for(cfg, out) if args.method == GGUF else None,
            imatrix=imatrix,
        )
    except subprocess.CalledProcessError as e:
        # the child is a `qda build`: its status is the answer a caller acts on, and a signal is an
        # interruption (`qda_quant.exit`); see `evaluate` for the same rule
        return refuse(str(e), code=e.returncode if e.returncode > 0 else INTERRUPTED)
    except Refused as e:
        return refuse(str(e), e.fix or None, code=e.code)
    except ValueError as e:
        return refuse(str(e))
    changes = entry(cfg, built)
    try:
        target = write_config(
            args.config, raw=cfg.raw, changes=changes, in_place=args.in_place, force=args.force
        )
    except ValueError as e:
        return refuse(str(e))
    print(f"candidate 0: {built.out} ({built.method}{' + gsq' if built.gsq else ''})")
    print(
        f"  build time: {built.build_seconds if built.build_seconds is not None else 'unrecorded'} s recorded"
        f" (build.json), {built.wall_seconds:.1f} s whole command"
    )
    print(
        f"  entry: baselines[0] = {changes['baselines'][0]}, "
        f"judge.candidate0_build_s = {changes['judge']['candidate0_build_s']}"
        + (", judge.select_candidate0 = true" if changes["judge"].get("select_candidate0") else "")
    )
    from qda_agent.config import RunConfig

    try:  # the whole config is checked here, once candidate 0 exists, not before its build
        RunConfig.load(target)
    except configfile.ConfigError as e:
        print(f"  before the run, fix: {e}")
        return 0
    print(f"  run: qda-agent run {target}")
    return 0
