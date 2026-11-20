"""The Fisher map's dKL is the KL a round-to-nearest unit actually costs, to second order, per token."""

import copy
import json

import pytest
import torch
from qda_quant.arch import describe
from qda_quant.engine.format_bindings import resolve_formats
from qda_quant.format import Format
from qda_quant.intake.fisher import allocate, fisher_map
from qda_quant.solve.grid import DEFAULT_GRID
from qda_quant.solve.quant import Quantizer
from qda_quant.target import Target

from ..engine_util import tiny_qwen3, tiny_qwen3_moe

FMT = "w4a16-asym-g32"


def _kl(m, m2, ids) -> float:
    kl = t = 0.0
    with torch.no_grad():
        for x in ids:
            lp = m(input_ids=x[None]).logits[0, :-1].log_softmax(-1)
            lq = m2(input_ids=x[None]).logits[0, :-1].log_softmax(-1)
            kl += float((lp.exp() * (lp - lq)).sum())
            t += lp.shape[0]
    return kl / t


def test_dkl_predicts_the_measured_kl_of_one_rtn_unit():
    """A wrong scale (per sequence, not per token) or the corpus's labels instead of the model's breaks this."""
    m = tiny_qwen3().float()
    ids = [torch.randint(0, 128, (64,), generator=torch.Generator().manual_seed(i)) for i in range(64)]
    mp = fisher_map(m, ids, ["bf16", FMT], "sm90@0.33.0")
    fmt = Format.parse(FMT)
    q = Quantizer.of(fmt, Target.parse("sm90@0.33.0").plan(fmt, "dense"), DEFAULT_GRID)
    for u in mp["units"][:2]:
        m2 = copy.deepcopy(m)
        with torch.no_grad():
            for name in u["modules"]:
                w = m2.get_submodule(name).weight
                s, z = q.params(w.float())
                w.copy_(q.fake(w.float(), s, z))
        assert u["options"][FMT]["dkl"] == pytest.approx(_kl(m, m2, ids), rel=0.2)
        assert u["options"]["bf16"] == {"bits": 16.0 * u["weights"], "dkl": 0.0}


def test_allocate_runs_from_the_cli_without_an_out_option(tmp_path, capsys):
    """`allocate`'s subparser defines no `--out` (only `map` does), and reading it unconditionally made every
    `qda fisher allocate MAP --body F --budget-bpw B` an `AttributeError` before it allocated anything."""
    from qda_quant.cli import main

    path = tmp_path / "map.json"
    path.write_text(json.dumps(_synthetic_map(["bf16", "w8a16-sym-g32", "w4a16-sym-g32"])))
    assert main(["fisher", "allocate", str(path), "--body", "w4a16-sym-g32", "--budget-bpw", "9.0"]) == 0
    allocation = json.loads(capsys.readouterr().out)
    assert allocation["bpw"] <= 9.0 and "formats" in allocation


def test_allocate_keeps_every_bank_in_one_format_and_inside_the_budget():
    m = tiny_qwen3_moe().float()
    ids = [torch.randint(0, 128, (32,), generator=torch.Generator().manual_seed(i)) for i in range(4)]
    mp = fisher_map(m, ids, ["w8a16-sym-g32", FMT], "sm90@0.33.0")
    assert [u["kind"] for u in mp["units"]].count("bank") == len(m.model.layers)
    body = (16 * mp["kept_weights"] + sum(u["options"][FMT]["bits"] for u in mp["units"])) / mp["denominator"]
    for budget in (body, body + 0.5, 9.0):
        al = allocate(mp, FMT, budget)
        assert al["bpw"] <= budget + 1e-9
        assert set(al["moved"]) <= {u["unit"] for u in mp["units"] if u["kind"] == "dense"} | {"experts"}
    full = allocate(mp, FMT, 9.0)
    assert full["moved"]["experts"] == "w8a16-sym-g32"
    # The block is one the builder takes: every rule names whole units, the banks together.
    graph = describe(m, verify=False)
    rules = [(k, v) for k, v in full["formats"].items() if k != "default"]
    tgt, body = Target.parse("sm90@0.33.0"), Format.parse(FMT)
    chosen, _, _ = resolve_formats(m, graph, rules, body, tgt, tgt.plan(body, "dense"), tgt.plan(body, "moe"))
    assert {f.name for f in chosen.values()} == {"w8a16-sym-g32"}


def _synthetic_map(menu: list[str]) -> dict:
    """Two dense units and one bank; every wider or more precise option saves dKL, so any move is attractive."""

    def options(w: int) -> dict:
        out = {}
        for f in menu:
            b = 16 if f == "bf16" else int(f[1])
            out[f] = {"bits": float(w * (b + (0.5 if "asym" in f else 0.25))), "dkl": 1.0 / b}
        return out

    qkv = [f"model.layers.0.self_attn.{x}_proj" for x in "qkv"]
    units = [
        {
            "unit": "l0.qkv",
            "layer": 0,
            "kind": "dense",
            "modules": qkv,
            "weights": 100,
            "options": options(100),
        },
        {
            "unit": "l0.o",
            "layer": 0,
            "kind": "dense",
            "modules": ["model.layers.0.self_attn.o_proj"],
            "weights": 100,
            "options": options(100),
        },  # fmt: skip
        {
            "unit": "l0.bank",
            "layer": 0,
            "kind": "bank",
            "modules": ["model.layers.0.mlp.experts"],
            "weights": 800,
            "options": options(800),
        },  # fmt: skip
    ]
    return {"menu": menu, "units": units, "kept_weights": 0, "denominator": 1000}


def test_allocate_picks_only_formats_a_checkpoint_of_the_body_can_hold():
    """One group per checkpoint; the body's family keeps the body's symmetry; a promotion is symmetric."""
    menu = [
        "bf16",
        "w8a16-sym-g32",
        "w8a16-asym-g32",
        "w4a16-sym-g32",
        FMT,
        "w8a16-sym-g64",
        "w4a16-asym-g64",
    ]
    full = allocate(_synthetic_map(menu), FMT, 99.0)  # room for any move: only the rule limits the choice
    assert full["moved"] and set(full["moved"].values()) <= {"bf16", "w8a16-sym-g32"}
    for wrong in ("w8a16-asym-g32", "w4a16-sym-g32", "w8a16-sym-g64", "w4a16-asym-g64"):
        assert f"{wrong} (dense)" in full["excluded"] and f"{wrong} (banks)" in full["excluded"]
    assert "group 64, not the body's 32" in full["excluded"]["w8a16-sym-g64 (dense)"]
    assert "sym, not the body's asym" in full["excluded"]["w4a16-sym-g32 (dense)"]
    assert "stored asymmetric" in full["excluded"]["w8a16-asym-g32 (dense)"]
    # A body that spells no symmetry stores its banks symmetric, so its banks may take the sym twin.
    bare = allocate(_synthetic_map(["w4a16-g32", "w4a16-sym-g32"]), "w4a16-g32", 99.0)
    assert "w4a16-sym-g32 (banks)" not in bare["excluded"] and "w4a16-sym-g32 (dense)" in bare["excluded"]


def test_cards_for_matches_the_measured_maps():
    """Measured on 80 GB cards: 8B (16.38 GB of bf16 weights) ran on 2 in 73 s, 27B (55.56 GB) on 3 in
    187 s; a model inside one card's share takes one."""
    from qda_quant.intake.fisher import CARD_SHARE, cards_for

    eighty = int(80 * 2**30 * CARD_SHARE)  # what a map may use of an 80 GB card
    assert cards_for(16_381_516_776, eighty) == 2
    assert cards_for(55_563_006_776, eighty) == 3
    assert cards_for(1_000, eighty) == 1


def test_a_small_card_takes_more_cards_so_the_map_does_not_run_into_an_oom():
    """The card size is read, not assumed: 8B needed 50.7 GiB of each 80 GB card, which no 48 GB card holds,
    so two of them are not enough (the old fixed 75 GB rule said two)."""
    from qda_quant.intake.fisher import CARD_SHARE, cards_for

    forty_eight = int(48 * 2**30 * CARD_SHARE)
    assert cards_for(16_381_516_776, forty_eight) == 3
    assert cards_for(16_381_516_776, int(24 * 2**30 * CARD_SHARE)) == 5


def test_auto_devices_takes_the_cards_the_model_needs_and_refuses_fewer(tmp_path, monkeypatch):
    """A job sized for builds (one card) must be told, not run into an OOM: the refusal states both numbers."""
    import qda_quant.intake.fisher as fisher
    from qda_quant.step import Refused

    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(fisher, "weight_bytes", lambda _model: 16_381_516_776)
    monkeypatch.setattr(fisher, "card_bytes", lambda _n: int(80 * 2**30 * fisher.CARD_SHARE))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert fisher.auto_devices(model) == ["cuda:0", "cuda:1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(Refused, match="needs 2 cards of 80 GiB and this job sees 1"):
        fisher.auto_devices(model)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert fisher.visible_cards() == 0


def test_auto_devices_on_48_gb_cards_refuses_a_two_card_job(tmp_path, monkeypatch):
    """L40S-class cards: 8B wants three of them, so a two-card job is one line (not two cards and an OOM)."""
    import qda_quant.intake.fisher as fisher
    from qda_quant.step import Refused

    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(fisher, "weight_bytes", lambda _model: 16_381_516_776)
    monkeypatch.setattr(fisher, "card_bytes", lambda _n: int(48 * 2**30 * fisher.CARD_SHARE))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(Refused, match="needs 3 cards of 48 GiB and this job sees 2"):
        fisher.auto_devices(model)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert fisher.auto_devices(model) == ["cuda:0", "cuda:1", "cuda:2"]


def test_the_map_cache_hits_the_same_inputs_and_misses_a_changed_one(tmp_path, monkeypatch):
    """The key covers the model, the calibration, the menu and the sampling: each change is a new map."""
    import qda_quant.intake.fisher as fisher

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"w" * 128)
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"text": "a"}\n')
    built: list = []

    def build(dest):
        built.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"units": []}\n')

    def payload(menu=("bf16", "w4a16"), samples=128, seqlen=2048):
        return fisher.cache_payload(model, calib, list(menu), samples=samples, seqlen=seqlen)

    first, was_built = fisher.cached_map(payload(), build, log=lambda _: None)
    assert was_built and first.is_file() and len(built) == 1
    assert first.parent.with_suffix(
        ".json"
    ).is_file()  # the payload, beside the entry (refcache's convention)
    log: list = []
    again, was_built = fisher.cached_map(payload(), build, log=log.append)
    assert again == first and not was_built and len(built) == 1
    assert any("from cache" in line and str(first) in line for line in log)
    entry = first.parent
    assert fisher.cache_entry(payload(menu=("w4a16", "bf16"))) != entry  # the menu as given, not a set
    assert fisher.cache_entry(payload(samples=64)) != entry
    assert fisher.cache_entry(payload(seqlen=4096)) != entry
    calib.write_text('{"text": "b"}\n')  # a new calibration is a new map
    changed = payload()
    assert fisher.cache_entry(changed) != entry
    assert fisher.cached_map(changed, build, log=lambda _: None)[1] and len(built) == 2


def test_a_failed_build_leaves_no_entry(tmp_path, monkeypatch):
    import qda_quant.intake.fisher as fisher

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"w" * 128)
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"text": "a"}\n')
    payload = fisher.cache_payload(model, calib, ["bf16"], samples=8, seqlen=16)

    def boom(_dest):
        raise RuntimeError("an architecture without a backward")

    with pytest.raises(RuntimeError, match="without a backward"):
        fisher.cached_map(payload, boom, log=lambda _: None)
    assert (
        not fisher.cache_entry(payload).exists()
        and not fisher.cache_entry(payload).with_suffix(".json").exists()
    )


def test_the_map_prices_an_untied_head_outside_the_denominator_and_none_for_a_tied_one():
    ids = [torch.randint(0, 128, (32,), generator=torch.Generator().manual_seed(i)) for i in range(2)]
    m = tiny_qwen3().float()
    assert not m.config.tie_word_embeddings
    mp = fisher_map(m, ids, ["bf16", "w8a16-sym-g32", FMT], "sm90@0.33.0")
    head = mp["head"]
    assert head["module"] == "lm_head" and head["weights"] == 128 * 64
    assert (
        head["options"]["bf16"] == {"bits": 16.0 * 128 * 64, "dkl": 0.0} and head["options"][FMT]["dkl"] > 0
    )
    assert mp["kept_weights"] == mp["denominator"] - sum(u["weights"] for u in mp["units"])  # head in neither
    m.config.tie_word_embeddings = True
    m.lm_head.weight = m.model.embed_tokens.weight
    assert fisher_map(m, ids, ["bf16", FMT], "sm90@0.33.0")["head"] is None


def _with_head(menu: list[str], dkl: dict[str, float]) -> dict:
    """`_synthetic_map` plus a 1,000-weight head (10x the units), whose savings can pay for body upgrades."""
    m = _synthetic_map(menu)
    w = 1000
    m["head"] = {
        "module": "lm_head",
        "weights": w,
        "options": {
            f: {"bits": float(w * (16 if f == "bf16" else int(f[1]) + 0.25)), "dkl": dkl[f]} for f in menu
        },
    }
    return m


def test_head_lease_off_leaves_the_head_alone_and_on_lends_its_savings_to_the_body():
    menu = ["bf16", "w8a16-sym-g32", "w4a16-sym-g32"]
    m = _with_head(menu, {"bf16": 0.0, "w8a16-sym-g32": 1e-4, "w4a16-sym-g32": 5.0})
    body, budget = "w4a16-sym-g32", 4.5
    off = allocate(m, body, budget)
    assert "head" not in off and off == allocate({k: v for k, v in m.items() if k != "head"}, body, budget)
    on = allocate(m, body, budget, head_lease=True)
    # a 4-bit head costs 5.0 dKL; an 8-bit one 1e-4 and frees 7.75 bits a weight for the body
    assert on["head"] == "w8a16-sym-g32"
    assert on["head_step"] == {"head": {"format": "w8a16-sym-g32"}}
    assert on["bpw"] > off["bpw"] and on["leased_bpw"] <= budget + 1e-9
    assert on["dkl"] < off["dkl"] + 1e-4  # the body's upgrades bought with the head's bits
    # the lease's total: body + head bits fit budget x body weights + 16 x head weights
    head_bits = m["head"]["options"][on["head"]]["bits"]
    assert on["bpw"] * m["denominator"] + head_bits <= budget * m["denominator"] + 16 * 1000 + 1e-6


def test_head_lease_offers_each_row_only_its_heads_and_refuses_a_map_without_one():
    menu = ["bf16", "w8a16-sym-g32", "w4a16-sym-g32"]
    m = _with_head(menu, {"bf16": 0.0, "w8a16-sym-g32": 1.0, "w4a16-sym-g32": 1e-6})
    assert allocate(m, "w4a16-sym-g32", 4.5, head_lease=True)["head"] == "w4a16-sym-g32"
    with pytest.raises(ValueError, match="--head-lease needs the head in the map"):
        allocate({k: v for k, v in m.items() if k != "head"}, "w4a16-sym-g32", 4.5, head_lease=True)


def test_head_lease_judges_the_body_against_the_lease_total_not_the_body_budget_alone():
    """The body alone (4.25 bpw) is over 4.0, but with the head at 8 bits the lease's total holds it: the judge
    would take that checkpoint, so allocate must not refuse it."""
    menu = ["bf16", "w8a16-sym-g32", "w4a16-sym-g32"]
    m = _with_head(menu, {"bf16": 0.0, "w8a16-sym-g32": 1e-4, "w4a16-sym-g32": 5.0})
    with pytest.raises(ValueError, match="the body alone is 4.2500 bpw, over the budget 4.0"):
        allocate(m, "w4a16-sym-g32", 4.0)
    on = allocate(m, "w4a16-sym-g32", 4.0, head_lease=True)
    assert on["head"] == "w8a16-sym-g32" and on["head_step"] == {"head": {"format": "w8a16-sym-g32"}}
    assert on["leased_bpw"] <= 4.0 + 1e-9
    # a 10-weight head lends too little: still refused, and the message says which head it assumed
    small = m | {"head": m["head"] | {"weights": 10, "options": {
        f: {"bits": o["bits"] / 100, "dkl": o["dkl"]} for f, o in m["head"]["options"].items()}}}  # fmt: skip
    with pytest.raises(ValueError, match="4.2500 bpw with the head at w4a16-sym-g32, over the budget 4.0"):
        allocate(small, "w4a16-sym-g32", 4.0, head_lease=True)


@pytest.mark.parametrize(
    ("body", "menu", "want"),
    [
        # an asymmetric body: the head in the body's family keeps its zero points, another family is symmetric
        ("w4a16-asym-g32", ["bf16", "w8a16-sym-g32", "w8a16-asym-g32", "w4a16-sym-g32", "w4a16-asym-g32"],
         "w4a16-asym-g32"),
        # a per-channel body: the head is per-channel too, spelled as the map priced it
        ("w4a16-sym-pc", ["bf16", "w8a16-sym-pc", "w8a16-sym-g32", "w4a16-sym-pc"], "w8a16-sym-pc"),
    ],
)  # fmt: skip
def test_the_head_step_stores_the_format_the_head_was_priced_in(body, menu, want):
    dkl = {f: (1e-6 if f == want else 1.0) for f in menu} | {"bf16": 0.0}
    m = _with_head(menu, dkl)
    on = allocate(m, body, 4.4, head_lease=True)
    assert on["head"] == want and on["head_step"] == {"head": {"format": want}}
    priced = m["head"]["options"][on["head"]]["bits"]
    assert on["leased_bpw"] == pytest.approx(
        (on["bpw"] * m["denominator"] + priced - 16 * m["head"]["weights"]) / m["denominator"]
    )


def test_a_w4a8_body_never_gets_a_4_bit_head():
    menu = ["bf16", "w8a16-sym-g128", "w4a16-sym-g128", "w4a8-sym-g128"]
    m = _with_head(menu, {"bf16": 0.0, "w8a16-sym-g128": 1.0, "w4a16-sym-g128": 1e-9, "w4a8-sym-g128": 1.0})
    on = allocate(m, "w4a8-sym-g128", 4.4, head_lease=True)
    assert on["head"] in ("bf16", "w8a16-sym-g128")  # the 4-bit head is cheapest in dKL, and never offered
    only_four = _with_head(["w4a16-sym-g128", "w4a8-sym-g128"], {"w4a16-sym-g128": 0.0, "w4a8-sym-g128": 1.0})
    with pytest.raises(ValueError, match="has no head format a w4a8-sym-g128 checkpoint can store"):
        allocate(only_four, "w4a8-sym-g128", 4.4, head_lease=True)


def test_allocate_head_lease_from_the_cli_writes_the_head_step(tmp_path, capsys):
    from qda_quant.cli import main

    menu = ["bf16", "w8a16-sym-g32", "w4a16-sym-g32"]
    path = tmp_path / "map.json"
    path.write_text(json.dumps(_with_head(menu, {"bf16": 0.0, "w8a16-sym-g32": 1e-4, "w4a16-sym-g32": 5.0})))
    args = ["fisher", "allocate", str(path), "--body", "w4a16-sym-g32", "--budget-bpw", "4.0"]
    assert main([*args, "--head-lease"]) == 0
    on = json.loads(capsys.readouterr().out)
    assert on["head"] == "w8a16-sym-g32" and on["head_step"] == {"head": {"format": "w8a16-sym-g32"}}
    assert main(args) != 0  # off: the same budget is the body's alone, and it is over
    err = capsys.readouterr().err
    assert "over the budget 4.0" in err and "Traceback" not in err


def test_a_w4a4_body_offers_an_fp8_head_and_never_an_fp8_bank():
    from qda_quant.head_lease import family, lease_formats

    assert family("w4a4") == family("w4a4-nvfp4") == "w4a4" and family("w8a8-fp8") == "fp8"
    assert lease_formats("w4a4-nvfp4") == ("bf16", "fp8") and family("w4a8") == "w4a8"
    menu = ["bf16", "w8a8-fp8", "w4a4-nvfp4"]
    m = _with_head(menu, {"bf16": 0.0, "w8a8-fp8": 1e-4, "w4a4-nvfp4": 5.0})
    on = allocate(m, "w4a4-nvfp4", 4.8, head_lease=True)
    assert on["head"] == "w8a8-fp8" and on["head_step"] == {"head": {"format": "w8a8-fp8"}}
    assert on["moved"].get("experts") != "w8a8-fp8" and "w8a8-fp8 (banks)" in on["excluded"]
