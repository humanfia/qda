"""`prepare --sensitivity`: one class sweep, cached and turned into candidate 0's per-tensor type map.

CPU only: the stand-in `qda sensitivity` verbs write what each verb writes, and the reads record the KL the
allocation is derived from. What is under test is the step's wiring, its budget and the cache, not the
measurements.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from qda_agent import cli
from qda_agent import prepare as P
from test_budget_source import fake_hub
from test_prepare import _fake_llama, prepared

REF_CARD_BYTES = P.ref_card_bytes  # the real sizing, before a fixture stands in for it

PLAN = [
    {
        "class": "attn_q",
        "payload_bytes": {"Q8_0": 90, "Q6_K": 70, "Q5_K": 60, "Q4_K": 50},
        "bf16_payload_bytes": 180,
    },
    {
        "class": "ffn_down",
        "payload_bytes": {"Q8_0": 900, "Q6_K": 700, "Q5_K": 600, "Q4_K": 500},
        "bf16_payload_bytes": 1800,
    },
    {"totals": {"rule_file_bytes": 8000, "rule_payload_bytes": 7000, "bf16_file_bytes": 16000}},
]
#: What each read reports: a class's cost at Q4_K, times a factor per type; the baseline is 0.0.
BASE_KL = {"attn_q": 0.002, "ffn_down": 0.02}
FACTOR = {"Q8_0": 0.25, "Q6_K": 0.5, "Q5_K": 0.75, "Q4_K": 1.0}
CELLS = {(k, t) for k in BASE_KL for t in FACTOR}


def kl_of(klass: str, kind: str) -> float:
    return BASE_KL[klass] * FACTOR[kind]


ALLOCATION = {
    "picks": {"attn_q": "Q6_K", "ffn_down": "Q4_K"},
    "file_bytes": 7000,
    "payload_bytes": 6000,
    "d_kl": 0.004,
}
RULES = [
    {"pattern": "blk.0.attn_q.weight", "type": "Q6_K"},
    {"pattern": "blk.0.ffn_down.weight", "type": "Q4_K"},
]


class Verbs:
    """A stand-in for the `qda sensitivity` verbs prepare runs; every other command is `test_prepare`'s."""

    def __init__(self, monkeypatch):
        from test_prepare import Commands

        self.base = Commands(monkeypatch)
        self.calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", self)

    def __call__(self, argv, check=False, env=None, capture_output=False, text=None, **kw):
        argv = [str(a) for a in argv]
        if "sensitivity" not in argv:
            return self.base(argv, check=check, env=env, **kw)
        self.calls.append(argv)
        what = argv[argv.index("sensitivity") + 1]
        arg = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        out = ""
        if what == "plan":
            out = "\n".join(json.dumps(row) for row in PLAN)
        elif what == "base":
            Path(arg["--out"]).mkdir(parents=True)
        elif what == "read":
            read_out = Path(arg["--out"])
            read_out.mkdir(parents=True)
            klass, kind = arg.get("--class", "bf16"), arg.get("--type", "bf16")
            kl = 0.0 if klass == "bf16" else kl_of(klass, kind)
            (read_out / "row.json").write_text(json.dumps({"class": klass, "type": kind, "kl": kl}))
        elif what == "table":
            out = json.dumps([ALLOCATION])
        elif what == "types":
            out = json.dumps(RULES)
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    def of(self, what: str) -> list[list[str]]:
        return [c for c in self.calls if c[c.index("sensitivity") + 1] == what]


def test_the_sweep_is_measured_once_and_candidate_0_reads_its_type_map(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    llama = _fake_llama(tmp_path, monkeypatch)
    verbs = Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True)
    steps = ("model", "corpus", "ref", "calib", "imatrix", "sensitivity", "cand0")
    P.execute(p, P.plan(p, steps), log=lambda _line: None)
    reads = verbs.of("read")
    cells = {(c[c.index("--class") + 1], c[c.index("--type") + 1]) for c in reads if "--class" in c}
    assert cells == CELLS  # one read per legal cell, plus the bf16 baseline
    rows = json.loads(p.sensitivity_rows.read_text())
    assert {(r["class"], r["type"], round(r["d_kl"], 6)) for r in rows} == {
        (klass, kind, round(kl_of(klass, kind), 6)) for klass, kind in CELLS
    }
    record = json.loads(p.sensitivity_record.read_text())
    assert record["budget"] == 8000 and record["budget_spec"] == "candidate0"
    assert len(record["arms"]) == len(CELLS)
    exporter = yaml.safe_load((p.out / "cand0-gguf-q4km.recipe.yaml").read_text())["exporter"]["gguf"]
    assert exporter["tensor_types"] == RULES
    assert exporter["libggml"] == str(llama / "build" / "bin" / "libggml-base.so")

    verbs.calls.clear()  # the cache and the type map make a second prepare run nothing
    again = P.Prepare(
        p.model, p.out, format="gguf-q4km", python=p.python, serve_python=p.serve_python, sensitivity=True
    )
    P.execute(again, P.plan(again, ("sensitivity", "cand0")), log=lambda _line: None)
    assert verbs.calls == []


def test_the_allocation_budget_is_the_hf_file_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    fake_hub(monkeypatch, tmp_path, {"a-Q4_K_M.gguf": 5_030_000_000, "README.md": 10})
    _fake_llama(tmp_path, monkeypatch)
    verbs = Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True, budget_bpw="hf:org/model-GGUF")
    steps = ("model", "corpus", "ref", "calib", "imatrix", "sensitivity")
    P.execute(p, P.plan(p, steps), log=lambda _line: None)
    record = json.loads(p.sensitivity_record.read_text())
    assert record["budget"] == 5_030_000_000
    table = verbs.of("table")[0]
    assert table[table.index("--budget-bytes") + 1] == "5030000000"
    assert p.budget_value() == 5_030_000_000


def test_a_budget_change_re_allocates_from_the_cached_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    verbs = Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True)
    P.execute(
        p, P.plan(p, ("model", "corpus", "ref", "calib", "imatrix", "sensitivity")), log=lambda _l: None
    )
    verbs.calls.clear()
    fake_hub(monkeypatch, tmp_path, {"a-Q4_K_M.gguf": 4000})
    narrower = P.Prepare(
        p.model,
        p.out,
        format="gguf-q4km",
        python=p.python,
        serve_python=p.serve_python,
        sensitivity=True,
        budget_bpw="hf:org/model-GGUF",
    )
    P.execute(narrower, P.plan(narrower, ("sensitivity",)), log=lambda _l: None)
    assert not verbs.of("read")  # no new measurement
    table = verbs.of("table")[0]
    assert table[table.index("--budget-bytes") + 1] == "4000"
    assert json.loads(narrower.allocation.read_text())[0] == ALLOCATION
    assert json.loads(narrower.sensitivity_record.read_text())["budget"] == 4000


def test_the_judge_budget_is_the_sweep_budget_with_sensitivity(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True)
    steps = ("model", "corpus", "ref", "calib", "imatrix", "sensitivity", "cand0")
    P.execute(p, P.plan(p, steps), log=lambda _line: None)
    p.save(floor={"kl": 0.0003, "ref": "c", "seconds": 548.19, "cards": 1})
    # candidate 0 has no .gguf of its own here: the judge's budget came from the sweep, not from that file
    assert P.search_config(p)["judge"]["budget_bytes"] == 8000
    assert p.budget_value() == 8000


def test_a_read_takes_the_fewest_cards_that_hold_the_template(tmp_path, monkeypatch):
    """A model that fits one card is read on every card at once; a larger one on as many as it needs."""
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True, cards=[0, 1, 2, 3])
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: 1000)
    assert p.sensitivity_tp() == 4  # no template yet (a dry run on a machine with cards): the widest read
    p.bf16.parent.mkdir(parents=True, exist_ok=True)
    p.bf16.write_bytes(b"x" * 600)
    assert p.sensitivity_tp() == 1  # 600 B fills 60% of a card
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: 500)
    assert p.sensitivity_tp() == 2  # needs 800 B of cards
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: 100)
    assert p.sensitivity_tp() == 4  # more than every card holds: the widest read
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: None)  # the driver will not say
    assert p.sensitivity_tp() == 4


def _uneven_node(monkeypatch):
    """A node whose physical cards 0-3 hold 80 GiB and 4-7 hold 48 GiB; torch's first cards must not be asked."""
    from types import SimpleNamespace

    from qda_agent.isolation import devices

    gib = 2**30
    table = [
        SimpleNamespace(ordinal=i, bus=f"0000:{i + 1:02x}:00.0", minor=i, uuid=f"GPU-{i}") for i in range(8)
    ]
    sizes = [80 * gib] * 4 + [48 * gib] * 4
    monkeypatch.setattr(devices, "smi_memory", lambda proc=None: list(zip(table, sizes, strict=True)))
    monkeypatch.setattr(
        devices, "resolve_cards", lambda cards, proc=None, visible=None: [table[c] for c in cards]
    )
    monkeypatch.setattr(
        "qda_quant.intake.fisher.card_bytes", lambda count: pytest.fail("torch's first cards were asked")
    )
    return devices, gib


def test_a_read_is_sized_by_the_physical_cards_named(tmp_path, monkeypatch):
    """`--cards 4,5,6,7` on a node whose cards 4-7 are the smaller ones: a template one 80 GiB card holds but
    one 48 GiB card does not is read on two cards; with no driver table, one read takes every card."""
    from qda_quant.intake.fisher import CARD_SHARE

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    Verbs(monkeypatch)
    monkeypatch.setattr(P, "ref_card_bytes", REF_CARD_BYTES)  # the real sizing, not the fixtures' stand-in
    devices, gib = _uneven_node(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True, cards=[4, 5, 6, 7])
    # 40 GiB of template needs 53.3 GiB of cards at SENSITIVITY_CARD_FILL: one 80 GiB card, two 48 GiB ones
    size = 40 * gib
    assert size / P.SENSITIVITY_CARD_FILL <= 80 * gib * CARD_SHARE
    assert size / P.SENSITIVITY_CARD_FILL > 48 * gib * CARD_SHARE
    p.bf16.parent.mkdir(parents=True, exist_ok=True)
    with p.bf16.open("wb") as f:
        f.truncate(size)  # sparse: the size is what is read, not the bytes
    assert p.sensitivity_tp() == 2
    monkeypatch.setattr(devices, "smi_memory", lambda proc=None: None)
    assert p.sensitivity_tp() == 4, "a driver that will not say: the widest read"


def test_the_gguf_floor_is_sized_by_the_physical_cards_named(monkeypatch):
    """A 60 GB decode fits one of cards 0-3 (80 GiB) but not one of cards 4-7 (48 GiB): `--cards 4,5` is
    refused at prepare, `--cards 0,1` passes; with no driver table it passes and the reading's load decides."""
    devices, _ = _uneven_node(monkeypatch)
    size = 60 * 10**9
    with pytest.raises(P.Refusal, match="on one card"):
        P._floor_fits_one_card(size, [4, 5])
    P._floor_fits_one_card(size, [0, 1])
    monkeypatch.setattr(devices, "smi_memory", lambda proc=None: None)
    P._floor_fits_one_card(size, [4, 5])


def test_candidate_0_refuses_when_the_sweep_has_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True)
    P.execute(p, P.plan(p, ("model", "corpus", "ref", "calib", "imatrix")), log=lambda _l: None)
    with pytest.raises(P.Refusal, match="no sensitivity type map"):
        P.execute(p, P.plan(p, ("cand0",)), log=lambda _l: None)


def test_candidate_0_refuses_a_type_map_without_libggml_before_the_build(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    llama = _fake_llama(tmp_path, monkeypatch)
    verbs = Verbs(monkeypatch)
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True)
    steps = ("model", "corpus", "ref", "calib", "imatrix", "sensitivity")
    P.execute(p, P.plan(p, steps), log=lambda _line: None)
    (llama / "build" / "bin" / "libggml-base.so").unlink()
    verbs.calls.clear()
    with pytest.raises(P.Refusal, match="libggml-base.so exists"):
        P.execute(p, P.plan(p, ("cand0",)), log=lambda _line: None)
    assert not [c for c in verbs.calls if "build" in c]  # refused before `qda build` ran


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--sensitivity-tp", "2"), ("--sensitivity-budget", "4"), ("--sensitivity-floor", "0.1")],
)
def test_the_removed_sensitivity_flags_are_not_options(flag, value, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["prepare", "--model", "m", "--out", "o", flag, value, "--dry-run"])
    assert stopped.value.code == 2
    assert flag in capsys.readouterr().err


def test_sensitivity_is_refused_for_a_vllm_search(tmp_path, monkeypatch):
    from test_prepare import Commands

    Commands(monkeypatch)
    with pytest.raises(P.Refusal, match="--sensitivity is a GGUF option"):
        prepared(tmp_path, sensitivity=True)


def test_a_failed_read_keeps_its_whole_output(tmp_path):
    """An engine's last line only points above; the refusal names a file holding everything it said."""
    import sys

    from qda_agent import prepare as P

    log = tmp_path / "reads" / "attn_k-Q4_K.log"
    command = [
        sys.executable,
        "-c",
        "import sys; print('CUDA error: out of memory', file=sys.stderr); "
        "print('RuntimeError: see root cause above', file=sys.stderr); sys.exit(1)",
    ]
    with pytest.raises(P.Refusal, match=r"see root cause above \(its whole output: .*attn_k-Q4_K.log\)"):
        P._qda_output(command, log=log)
    assert "out of memory" in log.read_text()


def test_two_reads_never_share_a_card(tmp_path, monkeypatch):
    """Reads of uneven length on one-card groups: a read takes a free group, so no card hosts two engines."""
    import threading
    import time

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    _fake_llama(tmp_path, monkeypatch)
    Verbs(monkeypatch)
    monkeypatch.setattr(P, "ref_card_bytes", lambda cards: 10**12)  # every read fits one card
    p = prepared(tmp_path, format="gguf-q4km", sensitivity=True, cards=[0, 1, 2, 3])
    real, busy, clash, lock = P._sensitivity_read, set(), [], threading.Lock()

    def read(p_, base, cell, work, cards, log):
        with lock:
            clash.extend(c for c in cards if c in busy)
            busy.update(cards)
        time.sleep(
            0.05 if cell is not None and hash(cell) % 3 == 0 else 0.005
        )  # uneven: groups free out of order
        try:
            return real(p_, base, cell, work, cards, log)
        finally:
            with lock:
                busy.difference_update(cards)

    monkeypatch.setattr(P, "_sensitivity_read", read)
    steps = ("model", "corpus", "ref", "calib", "imatrix", "sensitivity")
    P.execute(p, P.plan(p, steps), log=lambda _line: None)
    assert p.sensitivity_rows.is_file() and clash == []
