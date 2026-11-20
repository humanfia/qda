"""The end-of-run summary card: what each of the three commands prints, and what it leaves out.

CPU only: the checkpoints are the judge's tiny safetensors fixtures, and the two commands are driven through
`cli.main` with their public subcommands stood in for, as `test_prepare` does.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from qda_agent import cli, summary
from qda_quant.format import Format
from safetensors_kit import CONFIG, anchor_dir, packed_tensors, quant_dir, write_safetensors
from test_prepare import Commands, prepared

FIXTURES = Path(__file__).parent / "fixtures"

RULE = "  " + "-" * 58


@pytest.fixture(autouse=True)
def cuda_cards_visible(monkeypatch):
    """The root conftest empties CUDA_VISIBLE_DEVICES; a prepare names its cards to CUDA."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def sized(path: Path, total: int) -> None:
    """Give a written shard the size of a real checkpoint's, without writing the bytes (a sparse tail)."""
    with path.open("r+b") as f:
        f.truncate(total)


def checkpoint(d: Path, format: str = "w4a16-asym-g128") -> Path:
    """A packed one-layer checkpoint with a build record, as `qda build` leaves one."""
    q = quant_dir(d)
    (q / "build.json").write_text(json.dumps({"format": format, "load_to_saved_s": 42.5}))
    return q


def test_the_card_states_each_fact_the_run_measured(tmp_path):
    """The block, line for line: the label column, the ratio the size carries, and the reading's subject."""
    anchor = anchor_dir(tmp_path / "anchor")
    q = checkpoint(tmp_path / "cand0")
    sized(anchor / "model.safetensors", 2_000_000_000)
    sized(q / "model.safetensors", 500_000_000)
    assert summary.card(q, original=anchor, kl=0.0191) == (
        f"{RULE}\n"
        "  format        w4a16-asym-g128\n"
        "  bpw           6.00\n"
        "  size          0.5 GB  (25% of the original's 2.0 GB)\n"
        "  KL            0.0191  vs the original\n"
        f"{RULE}\n"
    )


def test_a_fact_the_run_did_not_measure_is_left_out(tmp_path):
    """No KL line before a reading, no bpw or ratio without the original, and nothing for a checkpoint with
    no facts at all -- never a placeholder, a zero or a number from the config."""
    anchor = anchor_dir(tmp_path / "anchor")
    q = checkpoint(tmp_path / "cand0")
    read = summary.card(q, original=anchor)  # a built checkpoint the run read no reference for
    assert "bpw           6.00" in read and "KL " not in read and "vs the original" not in read
    alone = summary.card(q, kl=0.0191)  # no original named: the size, without a ratio to it
    assert "bpw " not in alone and "of the original's" not in alone and "size          0.0 GB\n" in alone
    assert summary.card(tmp_path / "absent") == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert summary.card(empty) == ""  # a directory storing nothing, with no reading: no card at all


def test_a_mixed_precision_checkpoint_states_every_format_it_stores(tmp_path):
    """A build that overrode some modules (`format_overrides`) stores two formats: the card names both rather
    than the build record's default alone, and modules the config keeps bf16 are not a stored format."""
    from test_format_map import _q27, _w

    mixed = {
        "group_0": {"targets": ["Linear"], "weights": _w(4, 128, False)},
        "group_out": {"targets": ["re:.*o_proj$"], "weights": _w(8, 128, True)},
    }
    q = _q27(tmp_path / "mixed", changes={"model.layers.0.mlp.o_proj": (8, 128, True)}, groups=mixed)
    (q / "build.json").write_text(json.dumps({"format": "w4a16-asym-g128", "load_to_saved_s": 42.5}))
    text = summary.card(q, kl=0.0191)
    assert "format        w4a16-asym-g128 + w8a16-sym-g128\n" in text
    assert "bf16" not in text  # the float lm_head and in_proj_a/b are the format of no stored weight


def group_weights(bits: int, *, group: int | None = None, symmetric: bool = True, type: str = "int") -> dict:
    """A config group's `weights` as compressed-tensors writes it."""
    out = {"num_bits": bits, "type": type, "symmetric": symmetric}
    if group is None:
        out["strategy"] = "channel"
    else:
        out |= {"strategy": "group", "group_size": group}
    return out


def group_acts(bits: int, type: str) -> dict:
    """A config group's `input_activations`."""
    return {"num_bits": bits, "type": type, "dynamic": True}


def config_dir(d: Path, weights_of: dict, acts: dict | None = None) -> Path:
    """A checkpoint whose config declares one format for every Linear, with packed tensors to match."""
    d.mkdir(parents=True)
    group = {"targets": ["Linear"], "weights": weights_of}
    if acts is not None:
        group["input_activations"] = acts
    cfg = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"group_0": group}}}
    (d / "config.json").write_text(json.dumps({**CONFIG, **cfg}))
    write_safetensors(d / "model.safetensors", packed_tensors())
    return d


def facts(fmt: Format) -> tuple:
    """The facts that make two spellings the same format (`Format`'s classes compare by class)."""
    return (fmt.weight_bits, fmt.group_size, fmt.symmetric, fmt.weight_type, fmt.act)


#: Every body format a checkpoint's config can declare, with the name the card must print and what that name
#: parses back to. FP8 is here although `stored_formats` reports an FP8 module as bf16 (it stores no packed
#: codes): the name the card makes for one must still be a config's.
FORMATS = [
    (group_weights(4, group=128, symmetric=False), None, "w4a16-asym-g128", Format(4, 128, False)),
    (group_weights(3, group=64), None, "w3a16-sym-g64", Format(3, 64)),
    (group_weights(8), None, "w8a16-sym-pc", Format(8, -1)),
    (group_weights(4, group=128), group_acts(8, "float"), "w4a8-sym-g128", Format(4, 128, act="fp8_token")),
    (group_weights(8), group_acts(8, "int"), "w8a8i-sym-pc", Format(8, -1, act="int8_token")),
    (
        group_weights(8, type="float"),
        group_acts(8, "float"),
        "w8a8-fp8",
        Format(8, -1, weight_type="fp8", act="fp8_token"),
    ),
    (
        group_weights(4, group=16, type="float"),
        group_acts(4, "float"),
        "w4a4-nvfp4",
        Format(4, 16, weight_type="e2m1", act="nvfp4_block"),
    ),
]


@pytest.mark.parametrize(("weights_of", "acts", "name", "fmt"), FORMATS)
def test_a_checkpoints_card_names_its_format_the_way_a_config_spells_it(
    tmp_path, weights_of, acts, name, fmt
):
    """The card's name is the stored format's own spelling: parsing it gives that format back.

    The three families with a name of their own (FP8 W8A8, INT8 W8A8, NVFP4) must print it; their integer
    spelling names no format at all (`w4a4-sym-g16`) or another family (`w8a8-sym-pc` is not INT8).
    """
    q = config_dir(tmp_path / "q", weights_of, acts)
    assert f"  format        {name}\n" in summary.card(q)
    assert facts(Format.parse(name)) == facts(fmt)
    assert Format.parse(name).name == name  # the printed name is a spelling the parser gives back as is


def test_the_card_names_an_fp8_module_the_way_a_config_spells_it():
    """An FP8 module stores no packed codes, so the card reaches its name only from a build record; the
    name it makes is `w8a8-fp8`, which a config takes, not `w8a8-fp8-sym-pc`, which none does."""
    from qda_agent.judge.checks.formats import Stored

    fp8 = Stored(8, -1, True, "float", "channel", (8, "float", True))
    assert summary._format_name(fp8) == "w8a8-fp8"


def test_an_nvfp4_checkpoints_card_names_the_format_configs_accept(tmp_path):
    """A W4A4 row's card printed `w4a4-sym-g16`, which names FP4 as a symmetric integer and no config takes."""
    q = config_dir(tmp_path / "q", group_weights(4, group=16, type="float"), group_acts(4, "float"))
    assert "format        w4a4-nvfp4\n" in summary.card(q)


def test_a_float_weight_only_card_prints_no_integer_format(tmp_path):
    """FP8 weights and MXFP4 store float codes with no activation quantization; the integer spelling they
    would get names an integer format (`w8a16-sym-pc`, `w4a16-sym-g32`). The card names none, and a build
    record's own name still prints unchanged."""
    fp8 = config_dir(tmp_path / "fp8", group_weights(8, type="float"))
    assert "format" not in summary.card(fp8) and "w8a16" not in summary.card(fp8)
    mxfp4 = config_dir(tmp_path / "mxfp4", group_weights(4, group=32, type="float"))
    assert "format" not in summary.card(mxfp4) and "w4a16" not in summary.card(mxfp4)
    (mxfp4 / "build.json").write_text(json.dumps({"format": "mxfp4"}))
    assert "  format        mxfp4\n" in summary.card(mxfp4)


@pytest.mark.parametrize("gguf_format", ["w4a16-iq4xs", "w4a16-iq4nl", "w3a16-q3k"])
def test_a_gguf_build_records_format_parses_back(tmp_path, gguf_format):
    """A `.gguf` states the name its build record carries; every GGUF format a record can name parses."""
    q = tmp_path / "cand0-gguf"
    q.mkdir()
    (q / "model.gguf").write_bytes(b"\0" * 16)
    (q / "build.json").write_text(json.dumps({"format": gguf_format}))
    assert f"  format        {gguf_format}\n" in summary.card(q)
    assert Format.parse(gguf_format).name == gguf_format


def test_a_gguf_checkpoints_card_states_its_mix_and_prices_no_bpw(tmp_path):
    """A `.gguf` stores no packed safetensors decoder Linears: its record's mix is its format, and bpw is
    left out rather than priced at zero."""
    anchor = anchor_dir(tmp_path / "anchor")
    q = tmp_path / "cand0-gguf"
    q.mkdir()
    (q / "model-Q4_K_M.gguf").write_bytes(b"\0" * 1_000_000)
    (q / "build.json").write_text(json.dumps({"gguf": {"mix": "Q4_K_M"}, "format": "w4a16-asym-g32"}))
    text = summary.card(q, original=anchor, kl=0.0127)
    assert "format        Q4_K_M\n" in text and "w4a16-asym-g32" not in text  # the mix, not the recipe
    assert "bpw " not in text  # the judge's rule prices packed decoder Linears, which a `.gguf` has none of
    assert "size          0.0 GB" in text and "of the original's" in text
    assert "KL            0.0127  vs the original" in text


def run_dir(tmp_path, *, choice, baseline_reading=None, anchor=None, export=None, head_lease=False) -> Path:
    """A finished run's directory: its config, its FINAL.json and whatever records it holds."""
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)  # a caller may have put the exported copy here already
    judge = {"kl_ref": str(tmp_path / "ref"), "anchor": str(anchor) if anchor else None}
    if head_lease:
        judge["head_lease"] = True
    (run / "resolved.yaml").write_text(yaml.safe_dump({"model": "m", "cards": [0], "judge": judge}))
    final = {"rule": "handin"} | choice
    if baseline_reading is not None:
        final["baseline_readings"] = [baseline_reading]
    if export is not None:
        final["export"] = export
    (run / "FINAL.json").write_text(json.dumps(final))
    return run


def leased_pair(root: Path) -> tuple[Path, Path]:
    """An original with a 1024-wide body Linear and a 128-row head, and a candidate with an FP8 head.

    The body stores 4.125 bits a weight. The FP8 head costs 8.03 bits a head weight against the 16 the
    lease charges it, and the head is an eighth of the body, so the lease lends the body ~1 bit a weight.
    """
    body = "model.layers.0.self_attn.q_proj"
    anchor, q = root / "anchor", root / "cand"
    anchor.mkdir(parents=True)
    (anchor / "config.json").write_text(json.dumps({**CONFIG, "hidden_size": 1024}))
    write_safetensors(
        anchor / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (128, 1024), None),
            f"{body}.weight": ("BF16", (1024, 1024), None),
            "lm_head.weight": ("BF16", (128, 1024), None),
        },
    )
    q.mkdir()
    groups = {
        "group_0": {"targets": ["Linear"], "weights": group_weights(4, group=128, symmetric=False)},
        "group_head": {
            "targets": ["re:.*lm_head$"],
            "weights": group_weights(8, type="float"),
            "input_activations": group_acts(8, "float"),
        },
    }
    cfg = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": groups}}
    (q / "config.json").write_text(json.dumps({**CONFIG, "hidden_size": 1024, **cfg}))
    write_safetensors(
        q / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (128, 1024), None),
            f"{body}.weight_packed": ("I32", (1024, 128), None),
            f"{body}.weight_scale": ("BF16", (1024, 8), None),
            f"{body}.weight_shape": ("I32", (2,), b"\x00\x04\x00\x00\x00\x04\x00\x00"),
            "lm_head.weight": ("F8_E4M3", (128, 1024), None),
            "lm_head.weight_scale": ("F32", (128, 1), None),
        },
    )
    return anchor, q


@pytest.mark.parametrize("head_lease", [False, True])
def test_the_box_reports_the_bpw_the_judge_prices(tmp_path, head_lease):
    """A run's box bpw is the number the judge's `bpw` command prints for that pick, both ways.

    With `judge.head_lease` on that is the lease's number (`leased_bpw`, the head's savings lent to the
    body); with it off, the body's own.
    """
    from qda_agent.judge.checks import budget

    anchor, q = leased_pair(tmp_path)
    priced = (
        budget.leased_bpw(q, budget.denominator(anchor), budget.head_weights(anchor))
        if head_lease
        else budget.bpw(q, anchor)
    )
    run = run_dir(
        tmp_path,
        choice={"choice": {"dir": str(q), "metrics": {"kl": 0.01}}},
        anchor=anchor,
        head_lease=head_lease,
    )
    text = summary.of_run(run)
    assert f"  bpw           {priced:.2f}\n" in text
    assert text == summary.card(q, original=anchor, kl=0.01, head_lease=head_lease)


def test_the_leases_savings_reach_the_box(tmp_path):
    """The same pick: 4.12 bpw body only, 3.13 with the FP8 head's savings lent to it."""
    anchor, q = leased_pair(tmp_path)
    assert "  bpw           4.12\n" in summary.card(q, original=anchor)
    assert "  bpw           3.13\n" in summary.card(q, original=anchor, head_lease=True)


@pytest.mark.parametrize("head_lease", [False, True])
def test_an_original_with_no_decoder_linear_states_no_bpw_either_way(tmp_path, head_lease):
    """An original of an embedding and a head has no decoder Linear to divide by: `budget.bpw` refuses it,
    and the lease divides by the same zero. The card states no bpw either way, and the lease raises nothing."""
    from qda_agent.judge.checks import budget

    anchor, q = leased_pair(tmp_path)
    write_safetensors(
        anchor / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (128, 1024), None),
            "lm_head.weight": ("BF16", (128, 1024), None),
        },
    )
    assert budget.denominator(anchor) == 0
    text = summary.card(q, original=anchor, head_lease=head_lease)
    assert "bpw" not in text and "format" in text  # the card still states what the checkpoint stores


def test_of_run_names_the_runs_pick_and_the_reading_it_was_picked_on(tmp_path):
    anchor = anchor_dir(tmp_path / "anchor")
    q = checkpoint(tmp_path / "cand0")
    chosen = {"choice": {"dir": str(q), "metrics": {"kl": 0.0191}}}
    assert summary.of_run(run_dir(tmp_path / "a", choice=chosen, anchor=anchor)) == summary.card(
        q, original=anchor, kl=0.0191
    )
    # a run that kept candidate 0: the pick is its baseline, and the reading is that baseline's own
    fallback = {"baseline": str(q)}
    kept = run_dir(
        tmp_path / "b", choice=fallback, baseline_reading={"dir": str(q), "metrics": {"kl": 0.0213}}
    )
    assert "KL            0.0213  vs the original" in summary.of_run(kept)
    # an exported run: the pick is the copy under `run/winner` (the export renamed it) and the reading stays
    # under the name the checkpoint had when it was read, which the export records as its source
    d = tmp_path / "d"
    copy = d / "run" / "winner"
    copy.parent.mkdir(parents=True)
    shutil.copytree(q, copy)
    exported = run_dir(
        d,
        choice={"baseline": str(copy)},
        baseline_reading={"dir": str(q), "metrics": {"kl": 0.0213}},
        export={"winner": str(copy), "source": str(q), "sha256": "0" * 64},
    )
    assert "KL            0.0213  vs the original" in summary.of_run(exported)
    # a pick no record of the run read, and a run that picked nothing at all
    unread = run_dir(tmp_path / "c", choice={"choice": {"dir": str(q), "metrics": {}}})
    assert "KL " not in summary.of_run(unread)
    (tmp_path / "c" / "run" / "FINAL.json").unlink()
    assert summary.of_run(tmp_path / "c" / "run") == ""


def test_evaluate_ends_with_the_card_of_the_checkpoint_its_verdict_is_about(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    anchor = anchor_dir(tmp_path / "anchor")
    winner = checkpoint(run / "winner")
    holdout = tmp_path / "data" / "corpus.holdout.jsonl"
    holdout.parent.mkdir()
    holdout.write_text("")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "manifest.json").write_text(json.dumps({"build": {"sampling": {"temperature": 1.0}}}))
    judge = {
        "kl_ref": str(ref),
        "kl_holdout_ref": str(holdout),
        "anchor": str(anchor),
        "serve_python": sys.executable,
    }
    resolved = {"model": str(anchor), "cards": [0], "baselines": [str(tmp_path / "c0")], "judge": judge}
    (run / "resolved.yaml").write_text(yaml.safe_dump(resolved))
    (run / "FINAL.json").write_text(
        json.dumps({"choice": {"dir": str(winner)}, "export": {"winner": str(winner)}})
    )

    def fake(argv, **kw):
        if "score" in argv:
            body = yaml.safe_load(Path(argv[-1]).read_text())
            kl = 0.0191 if body["checkpoint"].endswith("winner") else 0.0213
            Path(body["out"], "score.json").write_text(json.dumps({"kl": kl}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(["evaluate", str(run)]) == 0
    out = capsys.readouterr().out
    assert out.index("verdict:") < out.index(RULE)  # the card ends the command, after the verdict
    assert "format        w4a16-asym-g128" in out and "bpw           6.00" in out
    assert "KL            0.0191  vs the original" in out  # the winner's holdout reading, not candidate 0's


def test_prepare_ends_with_the_card_of_candidate_0(tmp_path, monkeypatch, capsys):
    class Built(Commands):
        """The stand-in build writes a real packed checkpoint, so the card has facts to state."""

        def __call__(self, argv, **kw):
            done = super().__call__(argv, **kw)
            if "build" in argv:
                out = Path(yaml.safe_load(Path(argv[argv.index("build") + 1]).read_text())["out"])
                write_safetensors(out / "model.safetensors", packed_tensors())
                (out / "build.json").write_text(
                    json.dumps({"format": "w4a16-asym-g128", "load_to_saved_s": 42.5})
                )
            return done

    Built(monkeypatch)
    source = anchor_dir(tmp_path / "source")
    # The shared `anchor_dir` model has 8 input columns, which is not a whole number of 128-wide groups, so
    # the group-size check refuses the default `w4a16-asym-g128` before any step runs and
    # names per-channel as the way to store it. The card's own facts come from the stand-in's build.json, so
    # the requested preset does not change what this test asserts.
    p = prepared(tmp_path, model=str(source), format="w4a16-pc")
    assert (
        cli.main(
            [
                "prepare",
                "--model",
                str(source),
                "--out",
                str(p.out),
                "--format",
                "w4a16-pc",
                "--budget-bpw",
                "candidate0",  # the stand-in's per-channel packing is 6 bpw, over the row's default 4.4
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "format        w4a16-asym-g128" in out and "bpw           6.00" in out
    assert "KL " not in out  # prepare reads no reference: no reading to state
    assert out.index(RULE) < out.index("next: qda-agent search")  # the card, then what to run


def test_a_finished_search_ends_with_the_card_of_its_pick(tmp_path, capsys):
    """`qda-agent run` prints the card of the checkpoint FINAL.json picked, after the result."""
    from testkit import scripted

    over = ["judge.final_rules=[handin]", "judge.candidate0_kl=0.08", "judge.candidate0_build_s=600"]
    assert scripted(FIXTURES / "scripted-flow.yaml", tmp_path / "run", 3, overrides=over) == 0
    out = capsys.readouterr().out
    assert out.index(RULE) > out.rindex('"ended"')  # after the result the run is read by
    assert "KL            0.0700  vs the original" in out  # the reading the pick was chosen on
