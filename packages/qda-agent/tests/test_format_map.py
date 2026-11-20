"""`judge.format_map`: a fixed per-module allocation, read from a paper checkpoint and enforced at admission.

The two rows it was written for: a 27B W4A16 whose paper stores every Linear w4 asym g128 and keeps lm_head and
the linear-attention `in_proj_a`/`in_proj_b` float, and an R1-8B W3A16 whose paper stores every Linear w3 sym
g128 and lm_head w4 sym g64 in a group of its own (`re:.*lm_head$`), the embedding float.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qda_agent.judge import JudgeConfig
from qda_agent.judge.checks import format_problem, map_head_format, map_menu_problem, stored_formats
from safetensors_kit import CONFIG, write_safetensors

HIDDEN, VOCAB = 128, 16
LAYER = "model.layers.0"


def _w(bits: int, group: int, symmetric: bool) -> dict:
    return {"num_bits": bits, "group_size": group, "strategy": "group", "type": "int", "symmetric": symmetric}


def _ckpt(d: Path, modules: dict[str, tuple[int, int, bool] | None], groups: dict, ignore: list[str]) -> Path:
    """A checkpoint storing each module packed at (bits, group, symmetric), or as a bf16 weight (None)."""
    tensors: dict = {}
    for name, fmt in modules.items():
        if fmt is None:
            tensors[f"{name}.weight"] = (
                "BF16",
                (VOCAB if "lm_head" in name or "embed" in name else HIDDEN, HIDDEN),
                None,
            )
            continue
        bits, group, symmetric = fmt
        rows = VOCAB if "lm_head" in name else HIDDEN
        tensors[f"{name}.weight_packed"] = ("I32", (rows, HIDDEN * bits // 32), None)
        tensors[f"{name}.weight_scale"] = ("BF16", (rows, HIDDEN // group), None)
        if not symmetric:
            tensors[f"{name}.weight_zero_point"] = ("I32", (rows * bits // 32, HIDDEN // group), None)
    cfg = {**CONFIG, "hidden_size": HIDDEN}
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": groups,
        "ignore": ignore,
    }
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    write_safetensors(d / "model.safetensors", tensors)
    return d


W4 = (4, 128, False)
Q27 = ("self_attn.q_proj", "mlp.down_proj")


def _q27(
    d: Path, *, changes: dict | None = None, groups: dict | None = None, ignore: list[str] | None = None
):
    """The 27B shape: Linears w4 asym g128, lm_head and linear_attn.in_proj_a/b float."""
    modules: dict = {"model.embed_tokens": None, "lm_head": None}
    modules |= {f"{LAYER}.{m}": W4 for m in Q27}
    modules |= {f"{LAYER}.linear_attn.in_proj_a": None, f"{LAYER}.linear_attn.in_proj_b": None}
    modules |= changes or {}
    groups = groups or {"group_0": {"targets": ["Linear"], "weights": _w(4, 128, False)}}
    if ignore is None:
        ignore = ["lm_head", f"{LAYER}.linear_attn.in_proj_a", f"{LAYER}.linear_attn.in_proj_b"]
    return _ckpt(d, modules, groups, ignore)


R1_GROUPS = {
    "group_0": {"targets": ["Linear"], "weights": _w(3, 128, True)},
    "group_head": {"targets": ["re:.*lm_head$"], "weights": _w(4, 64, True)},
}


def _r1(
    d: Path, *, head: tuple[int, int, bool] | None = (4, 64, True), groups: dict | None = None, ignore=None
):
    """The R1 shape: Linears w3 sym g128, lm_head w4 sym g64 (its own group), the embedding float."""
    modules: dict = {"model.embed_tokens": None, "lm_head": head}
    modules |= {f"{LAYER}.{m}": (3, 128, True) for m in ("self_attn.q_proj", "mlp.down_proj")}
    groups = groups if groups is not None else dict(R1_GROUPS)
    return _ckpt(d, modules, groups, [] if ignore is None else ignore)


MENU_27 = ["bf16", "w4a16-asym-g128"]
MENU_R1 = ["bf16", "w3a16-sym-g128"]


def test_the_papers_own_checkpoint_is_admitted_for_both_rows(tmp_path):
    q27 = _q27(tmp_path / "q27")
    assert format_problem(q27, MENU_27, format_map=q27) is None
    r1 = _r1(tmp_path / "r1")
    assert format_problem(r1, MENU_R1, format_map=r1) is None
    assert map_head_format(stored_formats(r1)) == "w4a16-sym-g64"
    assert map_head_format(stored_formats(q27)) is None
    assert map_menu_problem(stored_formats(q27), MENU_27) is None
    assert map_menu_problem(stored_formats(r1), MENU_R1) is None


def test_effective_formats_are_compared_not_ignore_spellings(tmp_path):
    """27B candidate 0 lists alias spellings (and names no module stores) in `ignore`: same stored bits, admitted."""
    paper = _q27(tmp_path / "paper")
    aliases = [
        "lm_head",
        "re:.*linear_attn\\.in_proj_a$",
        "linear_attn.in_proj_b",  # a dotted-suffix spelling
        f"{LAYER}.linear_attn.in_proj_a",
        "model.layers.7.linear_attn.in_proj_a",  # no such module here
        "mtp.fc",
    ]
    cand = _q27(tmp_path / "cand", ignore=aliases)
    assert format_problem(cand, MENU_27, format_map=paper) is None
    # R1 head spelled by an explicit name instead of the paper's `re:` target, embedding listed as ignored
    groups = {
        "body": {"targets": ["Linear"], "weights": _w(3, 128, True)},
        "head": {"targets": ["lm_head"], "weights": _w(4, 64, True)},
    }
    r1 = _r1(tmp_path / "r1-cand", groups=groups, ignore=["model.embed_tokens"])
    assert format_problem(r1, MENU_R1, format_map=_r1(tmp_path / "r1-paper")) is None


def test_r1_with_a_bf16_head_is_refused(tmp_path):
    paper = _r1(tmp_path / "paper")
    groups = {"group_0": R1_GROUPS["group_0"]}
    cand = _r1(tmp_path / "cand", head=None, groups=groups, ignore=["lm_head"])
    got = format_problem(cand, MENU_R1, format_map=paper)
    assert got is not None and got.startswith(
        "format: lm_head is stored bf16; judge.format_map fixes it at w4a16"
    ), got
    # the same bf16-head checkpoint is admitted when no map is set (the search tier's bf16 head rule)
    assert format_problem(cand, MENU_R1) is None


def test_a_head_that_declares_the_map_but_stores_another_group_is_refused(tmp_path):
    """The map's quantized head is read back from the tensors: a declaration alone does not pass."""
    paper = _r1(tmp_path / "paper")
    cand = _r1(tmp_path / "cand")
    for f in cand.glob("*.safetensors"):
        f.unlink()
    tensors = {
        "model.embed_tokens.weight": ("BF16", (VOCAB, HIDDEN), None),
        "lm_head.weight_packed": ("I32", (VOCAB, HIDDEN * 4 // 32), None),
        "lm_head.weight_scale": ("BF16", (VOCAB, HIDDEN // 128), None),  # g128 scales under a g64 declaration
    }
    for m in ("self_attn.q_proj", "mlp.down_proj"):
        tensors[f"{LAYER}.{m}.weight_packed"] = ("I32", (HIDDEN, HIDDEN * 3 // 32), None)
        tensors[f"{LAYER}.{m}.weight_scale"] = ("BF16", (HIDDEN, 1), None)
    write_safetensors(cand / "model.safetensors", tensors)
    got = format_problem(cand, MENU_R1, format_map=paper)
    assert got is not None and "lm_head is stored 4-bit g128 (read from its tensors)" in got, got


def test_one_27b_linear_at_w8_is_refused_naming_it(tmp_path):
    paper = _q27(tmp_path / "paper")
    groups = {
        "group_0": {"targets": ["Linear"], "weights": _w(4, 128, False)},
        "wide": {"targets": [f"{LAYER}.mlp.down_proj"], "weights": _w(8, 128, True)},
    }
    cand = _q27(tmp_path / "cand", changes={f"{LAYER}.mlp.down_proj": (8, 128, True)}, groups=groups)
    menu = [*MENU_27, "w8a16-sym-g128"]
    assert format_problem(cand, menu) is None  # the menu alone admits the promotion
    got = format_problem(cand, menu, format_map=paper)
    assert got == (
        f"format: {LAYER}.mlp.down_proj is stored w8a16 (g128, sym); judge.format_map fixes it at "
        "w4a16 (g128, asym) (the per-module allocation is fixed: only the method may change)"
    )


def test_a_module_made_float_that_the_paper_quantizes_is_refused(tmp_path):
    paper = _q27(tmp_path / "paper")
    q = f"{LAYER}.self_attn.q_proj"
    cand = _q27(
        tmp_path / "cand",
        changes={q: None},
        ignore=["lm_head", q, f"{LAYER}.linear_attn.in_proj_a", f"{LAYER}.linear_attn.in_proj_b"],
    )
    got = format_problem(cand, MENU_27, format_map=paper)
    assert got is not None and got.startswith(
        f"format: {q} is stored bf16; judge.format_map fixes it at w4a16"
    ), got


def test_a_module_quantized_that_the_paper_keeps_float_is_refused(tmp_path):
    paper = _q27(tmp_path / "paper")
    a = f"{LAYER}.linear_attn.in_proj_a"
    cand = _q27(tmp_path / "cand", changes={a: W4}, ignore=["lm_head", f"{LAYER}.linear_attn.in_proj_b"])
    got = format_problem(cand, MENU_27, format_map=paper)
    assert got is not None and got.startswith(
        f"format: {a} is stored w4a16 (g128, asym); judge.format_map fixes it at bf16"
    ), got


def test_packed_codes_an_ignore_entry_keeps_float_are_refused(tmp_path):
    paper = _q27(tmp_path / "paper")
    q = f"{LAYER}.self_attn.q_proj"
    cand = _q27(
        tmp_path / "cand",
        ignore=["lm_head", q, f"{LAYER}.linear_attn.in_proj_a", f"{LAYER}.linear_attn.in_proj_b"],
    )
    got = format_problem(cand, MENU_27, format_map=paper)
    assert got == f"format: {q} stores packed codes, but its quantization_config keeps it float"


def test_the_menu_must_hold_exactly_the_maps_formats(tmp_path):
    q27 = stored_formats(_q27(tmp_path / "q27"))
    r1 = stored_formats(_r1(tmp_path / "r1"))
    assert map_menu_problem(q27, ["w4a16-asym-g128"]) is None  # bf16 may be left out
    got = map_menu_problem(q27, [*MENU_27, "w8a16-sym-g128"])
    assert got is not None and "offers w8a16-sym-g128, which judge.format_map stores in no module" in got
    got = map_menu_problem(r1, ["bf16", "w4a16-sym-g128"])
    assert got is not None and "outside judge.format_menu" in got  # the map's w3 body is not on it
    got = map_menu_problem(r1, [*MENU_R1, "w4a16-sym-g64"])  # the head's format is not a menu entry
    assert got is not None and "offers w4a16-sym-g64" in got


def test_format_map_config_rules(tmp_path):
    paper = _r1(tmp_path / "paper")
    cfg = JudgeConfig(contamination="none", kl_cmd=["score"], format_map=paper)
    assert cfg.format_map == paper
    with pytest.raises(ValueError, match="judge.format_map .* is not a checkpoint directory"):
        JudgeConfig(contamination="none", kl_cmd=["score"], format_map=tmp_path / "missing")
    with pytest.raises(ValueError, match="judge.allow_any_format skips the format gate"):
        JudgeConfig(contamination="none", kl_cmd=["score"], format_map=paper, allow_any_format=True)
    with pytest.raises(ValueError, match="head_format and format_map"):
        format_problem(paper, MENU_R1, format_map=paper, head_format="w4a16-sym-g64")


def _host(run: Path, **judge):
    from qda_agent.judge import JudgeHost
    from qda_agent.sched import FakeBackend, FakeClock, SchedConfig

    ref = run.parent / "calib-ref"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "prompts.jsonl").write_text(
        json.dumps({"id": "r1", "prompt": "a reference prompt used by the judge tests only"})
        + "\n"
        + json.dumps(
            {"id": "h1", "prompt": "a holdout prompt used by the judge tests only", "split": "holdout"}
        )
        + "\n"
    )
    (ref / "manifest.json").write_text(json.dumps({"build": {"prompts": str(ref / "prompts.jsonl")}}))
    clock = FakeClock()
    cfg = JudgeConfig(
        contamination="none",
        kl_cmd=["score"],
        check_cmd=None,
        kl_ref=str(ref),
        kl_holdout_ref=str(ref),
        **judge,
    )
    return JudgeHost(
        run,
        SchedConfig(cards=[4], budget_s=3600, fake_gpus=True),
        cfg,
        clock,
        FakeBackend(clock),
        autotick=False,
    )


def test_the_host_refuses_to_start_on_a_menu_that_is_not_the_maps(tmp_path, monkeypatch):
    paper = _q27(tmp_path / "paper")
    host = _host(tmp_path / "bad", format_map=paper, format_menu=[*MENU_27, "w8a16-sym-g128"])
    with pytest.raises(ValueError, match=r"judge\.format_map .*offers w8a16-sym-g128"):
        host.start()
    ok = _host(tmp_path / "ok", format_map=paper, format_menu=MENU_27)

    def evaluator():
        raise RuntimeError("passed the format_map gate")

    monkeypatch.setattr(ok, "install_evaluator", evaluator)
    with pytest.raises(RuntimeError, match="passed the format_map gate"):
        ok.start()


def test_a_module_the_map_lacks_or_holds_alone_is_refused(tmp_path):
    """Module-set equality: a stored module the map has no entry for, and a map module not stored, are refused."""
    paper = _q27(tmp_path / "paper")
    extra = _q27(tmp_path / "extra", changes={f"{LAYER}.mlp.up_proj": W4})
    got = format_problem(extra, MENU_27, format_map=paper)
    assert got == (
        f"format: {LAYER}.mlp.up_proj is stored w4a16 (g128, asym), and judge.format_map has no such module"
    )
    got = format_problem(paper, MENU_27, format_map=extra)
    assert got == f"format: {LAYER}.mlp.up_proj is not stored; judge.format_map stores it w4a16 (g128, asym)"


def _run(tmp_path: Path, paper: Path, task: Path | None, menu=MENU_R1):
    from qda_agent.config import RunConfig

    judge = {"kl_cmd": ["score"], "format_menu": menu, "format_map": paper}
    run = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
    extra = {"task": task} if task is not None else {}
    return RunConfig(
        model="tiny/model",
        format="w3a16-sym-g128",
        run_dir=run,
        budget_gpu_hours=1,
        cards=[0],
        judge=judge,
        contamination={"name": "none"},  # explicit: a test config names its check (no guard needed)
        **extra,
    )


def test_resolved_config_records_the_map_and_the_statement_must_state_it(tmp_path):
    paper = _r1(tmp_path / "paper")
    fixed = tmp_path / "task-fixed.md"
    fixed.write_text(
        "Every decoder Linear is w3a16-sym-g128 and lm_head w4a16-sym-g64. Menu: ${format_menu}\n"
    )
    cfg = _run(tmp_path, paper, fixed)
    assert cfg.resolved()["judge"]["format_map"] == str(paper)  # a replay of resolved.yaml still enforces it
    with pytest.raises(ValueError, match="no task template is named"):
        _run(tmp_path, paper, None)
    for spelling in ("${group_line}", "$module_formats_line\n"):
        offers = tmp_path / f"task-{len(list(tmp_path.iterdir()))}.md"
        offers.write_text(f"Formats: ${{format_menu}}\n{spelling}")
        with pytest.raises(ValueError, match="offers per-module formats"):
            _run(tmp_path, paper, offers)
