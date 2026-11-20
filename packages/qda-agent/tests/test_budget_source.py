"""`qda-agent prepare --budget-bpw N | candidate0 | hf:REPO[:FILE]` and `--head-lease` (`qda_agent.budget_source`)."""

import json
from types import SimpleNamespace

import huggingface_hub
import pytest
import yaml
from qda_agent import budget_source as B
from qda_agent import prepare as P
from test_prepare import Commands, model, prepared

#: A Hub W4A16 g128 quantization of `test_prepare.model` (its one q_proj): codes and a bf16 scale per group.
Q = "model.layers.0.self_attn.q_proj"
HUB_TENSORS = {
    f"{Q}.weight_packed": ("I32", (128, 16), 128 * 16 * 4),
    f"{Q}.weight_scale": ("BF16", (128, 1), 128 * 2),
    f"{Q}.weight_zero_point": ("I32", (16, 1), 16 * 4),
    "lm_head.weight": ("BF16", (256, 128), 256 * 128 * 2),  # outside the body
}


def fake_hub(monkeypatch, tmp_path, files: dict[str, int], config: dict | None = None):
    """A Hub whose one repository holds `files` (name -> size) and, for safetensors, `HUB_TENSORS`."""
    cfg = tmp_path / "hub-config.json"
    cfg.write_text(json.dumps(config or {}))
    start, tensors = 0, {}
    for name, (dtype, shape, n) in HUB_TENSORS.items():
        tensors[name] = SimpleNamespace(dtype=dtype, shape=list(shape), data_offsets=(start, start + n))
        start += n

    class Api:
        def model_info(self, repo):
            return SimpleNamespace(sha="0123abc")

        def list_repo_files(self, repo, revision=None):
            return list(files)

        def get_paths_info(self, repo, paths, revision=None):
            return [SimpleNamespace(path=p, size=files[p]) for p in paths]

        def get_safetensors_metadata(self, repo, revision=None):
            return SimpleNamespace(files_metadata={"model.safetensors": SimpleNamespace(tensors=tensors)})

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, revision=None: str(cfg))


def config_of(tmp_path, monkeypatch, **kw) -> dict:
    Commands(monkeypatch)
    p = prepared(tmp_path, **kw)
    P.execute(p, P.plan(p), log=lambda _: None)
    return yaml.safe_load(p.config.read_text())["judge"]


def test_a_number_is_the_budget_and_each_row_has_a_default(tmp_path, monkeypatch):
    assert B.DEFAULT_BPW == {"w4a16": 4.4, "w4a8": 4.4, "w3a16": 3.4, "w4a4": 4.8}
    assert config_of(tmp_path / "default", monkeypatch)["budget_bpw"] == 4.4
    judge = config_of(tmp_path / "n", monkeypatch, budget_bpw="4.2")
    assert judge["budget_bpw"] == 4.2 and "head_lease" not in judge  # off by default below W4A4
    for bad in ("0", "-1", "nan", "inf", "four", "hf:noslash"):
        with pytest.raises(ValueError, match="--budget-bpw"):
            B.parse(bad)


def test_a_budget_candidate_0_cannot_meet_is_refused_before_the_run(tmp_path, monkeypatch):
    """The config step runs the host's own start check: 4.125 bpw over a 4.0 budget never reaches the run."""
    with pytest.raises(
        P.Refusal, match=r"4\.125 bits per quantized weight, over the judge's budget of 4 bpw"
    ):
        config_of(tmp_path, monkeypatch, budget_bpw="4.0")


def test_candidate0_prices_candidate_0_by_the_judges_rule(tmp_path, monkeypatch):
    assert config_of(tmp_path, monkeypatch, budget_bpw="candidate0")["budget_bpw"] == 4.125


def test_hf_gguf_is_a_byte_budget_for_a_llama_cpp_search(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path, {"a-Q4_K_M.gguf": 5_030_000_000, "README.md": 10})
    spec = B.parse("hf:org/model-GGUF")
    price = B.hub_price(spec)
    assert (price.kind, price.size, price.source) == (
        "gguf",
        5_030_000_000,
        "org/model-GGUF@0123abc:a-Q4_K_M.gguf",
    )
    # a vLLM search is priced in bits, so a GGUF price is refused before any step
    with pytest.raises(P.Refusal, match="a gguf checkpoint prices a llama.cpp search"):
        prepared(tmp_path, budget_bpw="hf:org/model-GGUF")
    fake_hub(monkeypatch, tmp_path, {"a.gguf": 1, "b.gguf": 2})
    with pytest.raises(ValueError, match="holds 2 GGUF files; name one"):
        B.hub_price(spec)
    assert B.hub_price(B.parse("hf:org/model-GGUF:b.gguf")).size == 2


@pytest.mark.parametrize("method", B.QUANT_METHODS)
def test_hf_safetensors_is_its_bpw_by_the_judges_rule(tmp_path, monkeypatch, method):
    """Only the headers are read; the head is outside the body, so the number is the q_proj's alone."""
    fake_hub(
        monkeypatch,
        tmp_path,
        {"config.json": 1, "model.safetensors": 1},
        {"quantization_config": {"quant_method": method}},
    )
    judge = config_of(tmp_path, monkeypatch, budget_bpw="hf:org/model-W4A16")
    assert judge["budget_bpw"] == round((128 * 16 * 32 + 128 * 16 + 16 * 32) / (128 * 128), 4)  # 4.156


def test_any_other_hub_checkpoint_is_refused(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path, {"config.json": 1, "model.safetensors": 1}, {})
    with pytest.raises(B.BudgetRefused, match="is not quantized, and its bits per weight cannot be read") as no:
        B.hub_price(B.parse("hf:org/bf16"))
    assert "GGUF file" in no.value.fix and "compressed-tensors, GPTQ or AWQ" in no.value.fix
    fake_hub(
        monkeypatch,
        tmp_path,
        {"config.json": 1, "model.safetensors": 1},
        {"quantization_config": {"quant_method": "bitsandbytes"}},
    )
    with pytest.raises(ValueError, match="is bitsandbytes"):
        B.hub_price(B.parse("hf:org/bnb"))
    fake_hub(monkeypatch, tmp_path, {"model.onnx": 1})
    with pytest.raises(ValueError, match="neither a GGUF file nor safetensors"):
        B.hub_price(B.parse("hf:org/onnx"))
    with pytest.raises(ValueError, match="model.onnx is not a .gguf"):
        B.hub_price(B.parse("hf:org/onnx:model.onnx"))


def test_head_lease_is_written_when_asked_and_refused_for_gguf(tmp_path, monkeypatch):
    assert config_of(tmp_path, monkeypatch, budget_bpw="4.2", head_lease=True)["head_lease"] is True
    with pytest.raises(P.Refusal, match="byte budget already holds the output and token_embd"):
        P.Prepare(str(model(tmp_path / "g")), tmp_path / "g" / "out", format="gguf-q4km", head_lease=True)
    with pytest.raises(P.Refusal, match="a llama.cpp search states its budget in bytes"):
        P.Prepare(str(model(tmp_path / "h")), tmp_path / "h" / "out", format="gguf-q4km", budget_bpw="4.4")


def test_calib_file_is_candidate_0s_and_the_searchs_calibration_as_is(tmp_path, monkeypatch):
    """No render: the builder's own sample rule checks the file, and the pools still come down as data sets."""
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="abc")),
    )
    mine = tmp_path / "mine.jsonl"
    mine.write_text('{"text": "t"}\n')
    commands = Commands(monkeypatch)
    p = prepared(tmp_path, calib_file=mine)
    P.execute(p, P.plan(p), log=lambda _: None)
    raw = yaml.safe_load(p.config.read_text())
    assert raw["calib"] == str(mine) and yaml.safe_load(
        p.cand0.with_name(f"{p.cand0.name}.recipe.yaml").read_text()
    )["calib"]["file"]["path"] == str(mine)
    assert not any("render" in c for c in commands.calls) and any("-c" in c for c in commands.calls)
    assert P.plan(p, ("calib",))[0].done()  # a second prepare reuses it

    def short(argv, **kw):
        if "-c" in argv:
            return SimpleNamespace(returncode=2, stdout="", stderr="the file packs 3 of 128 samples")
        return Commands.__call__(commands, argv, **kw)

    monkeypatch.setattr(P.subprocess, "run", short)
    q = prepared(tmp_path / "again", calib_file=mine)
    with pytest.raises(P.Refusal, match=r"--calib .*mine.jsonl: the file packs 3 of 128 samples"):
        P.execute(q, P.plan(q, ("calib",)), log=lambda _: None)


def test_an_unpriceable_hub_checkpoint_is_a_two_line_refusal(tmp_path, monkeypatch, capsys):
    from qda_agent import cli

    fake_hub(
        monkeypatch,
        tmp_path,
        {"config.json": 1, "model.safetensors": 1},
        {"quantization_config": {"quant_method": "bitsandbytes"}},
    )
    m = model(tmp_path)
    argv = ["prepare", "--model", str(m), "--out", str(tmp_path / "out"), "--budget-bpw", "hf:org/bnb", "--dry-run"]
    assert cli.main(argv) != 0
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(("error: ", "fix: "))]
    assert len(lines) == 2 and lines[0].startswith("error: ") and "is bitsandbytes" in lines[0]
    assert lines[1].startswith("fix: ") and "compressed-tensors, GPTQ or AWQ" in lines[1]


def test_a_second_prepare_prices_the_recorded_hub_commit_without_the_hub(tmp_path, monkeypatch):
    fake_hub(
        monkeypatch,
        tmp_path,
        {"config.json": 1, "model.safetensors": 1},
        {"quantization_config": {"quant_method": "compressed-tensors"}},
    )
    Commands(monkeypatch)
    p = prepared(tmp_path, budget_bpw="hf:org/model-W4A16")
    P.execute(p, P.plan(p), log=lambda _: None)
    saved = json.loads((p.out / "prepare.json").read_text())["budget_source"]
    assert saved["spec"] == "hf:org/model-W4A16" and (saved["repo"], saved["revision"]) == ("org/model-W4A16", "0123abc")
    text = p.config.read_text()
    assert "  # budget: --budget-bpw hf:org/model-W4A16, read from org/model-W4A16@0123abc\n  budget_bpw:" in text

    class Offline:
        def __getattr__(self, name):
            raise AssertionError(f"the second prepare called the Hub ({name})")

    monkeypatch.setattr(huggingface_hub, "HfApi", Offline)
    again = P.Prepare(p.model, p.out, python=p.python, serve_python=p.serve_python, budget_bpw="hf:org/model-W4A16")
    assert again.hub == p.hub and again.budget_value() == p.budget_value()
    assert P.plan(again, ("config",))[0].done()  # the same config, the same budget
    # another source is read afresh
    with pytest.raises(P.Refusal, match="cannot read the Hub repository"):
        P.Prepare(p.model, p.out, python=p.python, serve_python=p.serve_python, budget_bpw="hf:org/other")


def test_a_split_gguf_is_priced_as_all_its_parts(tmp_path, monkeypatch):
    parts = {f"m-Q4_K_M-0000{i}-of-00003.gguf": 10 * i for i in (1, 2, 3)}
    fake_hub(monkeypatch, tmp_path, parts)
    price = B.hub_price(B.parse("hf:org/model-GGUF:m-Q4_K_M-00002-of-00003.gguf"))
    assert (price.size, price.file) == (60, "m-Q4_K_M-00001-of-00003.gguf")
    fake_hub(monkeypatch, tmp_path, {k: v for k, v in parts.items() if "00003-of" not in k})
    with pytest.raises(ValueError, match="missing part"):
        B.hub_price(B.parse("hf:org/model-GGUF:m-Q4_K_M-00001-of-00003.gguf"))


def test_the_config_says_where_the_budget_came_from(tmp_path, monkeypatch):
    Commands(monkeypatch)
    p = prepared(tmp_path)
    P.execute(p, P.plan(p), log=lambda _: None)
    assert "  # budget: the w4a16 row's default\n  budget_bpw: 4.4\n" in p.config.read_text()


def test_the_dry_run_shows_the_price_it_read(tmp_path, monkeypatch):
    """The number the config would carry: bpw for safetensors, bytes for a GGUF file."""
    fake_hub(
        monkeypatch,
        tmp_path,
        {"config.json": 1, "model.safetensors": 1},
        {"quantization_config": {"quant_method": "compressed-tensors"}},
    )
    p = prepared(tmp_path / "st", budget_bpw="hf:org/model-W4A16")
    assert "read from org/model-W4A16@0123abc: 4.1562; head lease off" in P.show_plan(p, P.plan(p))
    (p.model_dir / "config.json").unlink()  # not downloaded yet: the divisor is unknown
    assert "read from org/model-W4A16@0123abc: priced after the download" in P.show_plan(p, [])
    fake_hub(monkeypatch, tmp_path, {"a-Q4_K_M.gguf": 5_030_000_000})
    g = prepared(tmp_path / "gguf", format="gguf-q4km", budget_bpw="hf:org/model-GGUF")
    shown = P.show_plan(g, P.plan(g))
    assert "(bytes), read from org/model-GGUF@0123abc:a-Q4_K_M.gguf: 5030000000\n" in shown
