"""Output publication of `qda-kl score`: nonzero exit => no result file, a result is written whole or not at all."""

from pathlib import Path

import pytest
import yaml
from qda_kl import cli, score
from qda_kl.serve import ServeSettings, resolve_dtype
from qda_quant.exit import ENVIRONMENT
from qda_quant.step import Refused


def fake_result(tokens=10, kl=0.02, kl_text=0.04, math=0.01):
    from qda_kl.score import Score

    return Score(
        kl=kl,
        tokens=tokens,
        domains={"math": math, "code": 0.03},
        domain_tokens={"math": 5, "code": 5},
        kl_text=kl_text,
        text_tokens=4,
        domains_text={"web": kl_text},
        domain_text_tokens={"web": 4},
        flip_rate=0.0,
        kernels={},
        vllm="0.33.0",
        wall_s=1.0,
        dtype="bfloat16",
        dtype_from="test",
        serve=ServeSettings(),
        tp=1,
        checkpoint={},
        reference={},
        validate_max_abs_diff=0.0,
        seconds={},
        versions={},
    )


def sequence_scorer(results):
    it = iter(results)

    class Fake:
        def __init__(self, *a, **k):
            pass

        def score(self, **k):
            return next(it)

    return Fake


def scorer(result=None, error=None):
    class Fake:
        def __init__(self, *a, **k):
            if error is not None:
                raise error

        def score(self, **k):
            return result

    return Fake


def config(tmp_path, *, checkpoint=None, reference=None, out=None, **scorer_cfg):
    """A score config over ``tmp_path``; extra keywords are ``scorer.topk`` settings."""
    d = {
        "checkpoint": str(checkpoint or tmp_path),
        "reference": str(reference or tmp_path),
        "out": str(out or tmp_path),
    }
    if scorer_cfg:
        d["scorer"] = {"topk": scorer_cfg}
    p = tmp_path / "score.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_success_publishes_and_leaves_no_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "Scorer", scorer(fake_result()))
    assert cli.main(["score", str(config(tmp_path))]) == 0
    import json

    r = json.loads((tmp_path / "score.json").read_text())
    assert r["kl"] == 0.02 and r["kl_text"] == 0.04 and r["domains"] == {"math": 0.01, "code": 0.03}
    assert not list(tmp_path.glob(".score.json.*"))


def test_refusal_removes_an_old_result(tmp_path, monkeypatch):
    (tmp_path / "score.json").write_text('{"kl": "old"}')
    monkeypatch.setattr(score, "Scorer", scorer(error=Refused("no answer tokens scored")))
    assert cli.main(["score", str(config(tmp_path))]) == 2
    assert not (tmp_path / "score.json").exists()


def test_a_refusal_of_the_machine_exits_3(tmp_path, monkeypatch, capsys):
    """A scorer that refuses for the machine (its free memory) exits 3, not 2."""
    monkeypatch.setattr(
        score,
        "Scorer",
        scorer(error=Refused("card 0 has 3.1 GiB free", code=ENVIRONMENT)),
    )
    assert cli.main(["score", str(config(tmp_path))]) == 3
    assert capsys.readouterr().err.startswith("error: card 0 has 3.1 GiB free")


def test_a_missing_vllm_is_the_machine(tmp_path, monkeypatch, capsys):
    """vLLM is the `vllm` extra and is imported lazily: without it the command says so and exits 3, not 1."""
    from qda_kl import reference

    def no_vllm(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'vllm'", name="vllm")

    monkeypatch.setattr(reference, "build", no_vllm)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id": "a", "prompt": "hi"}\n')
    argv = [
        "ref",
        "--model",
        "m",
        "--prompts",
        str(prompts),
        "--out",
        str(tmp_path / "r"),
        "--sampling",
        '{"temperature": 1.0}',
    ]
    assert cli.main(argv) == 3
    err = capsys.readouterr().err
    assert err.startswith("error: this command needs vLLM (No module named 'vllm')")
    assert 'uv pip install -c constraints.txt "./packages/qda-kl[vllm]"' in err


def test_an_engine_crash_is_an_error_not_a_refusal(tmp_path, monkeypatch, capsys):
    """A vLLM start-up failure (out of memory, NCCL) is an error: exit 1, naming its type, not a verdict."""
    (tmp_path / "score.json").write_text('{"kl": "old"}')
    monkeypatch.setattr(
        score,
        "Scorer",
        scorer(error=RuntimeError("Engine core initialization failed. See root cause above.")),
    )
    assert cli.main(["score", str(config(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "error: RuntimeError: Engine core initialization failed." in err
    assert "refused" not in err and "fix:" not in err  # nothing to fix: a crash is not a refusal
    assert not (tmp_path / "score.json").exists()


def test_an_engine_crash_with_a_known_fix_keeps_vllms_words_and_adds_the_fix(tmp_path, monkeypatch, capsys):
    """No nvcc: vLLM's own line stays in the error, the fix QDA knows follows it, and the machine exits 3."""
    from qda_quant.output.vllm_env import add_fix

    error = RuntimeError("Engine core initialization failed. See root cause above.")
    add_fix(
        error, "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist\n"
    )
    monkeypatch.setattr(score, "Scorer", scorer(error=error))
    assert cli.main(["score", str(config(tmp_path))]) == 3
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("error: RuntimeError: Engine core initialization failed.")
    assert lines[0].endswith(
        "; RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"
    )
    assert lines[1].startswith("fix:   install the CUDA toolkit or put its nvcc on PATH")


#: vLLM's warning on every engine start where the kernels came from `vllm download-kernels` and nvcc is absent.
NVCC_QUERY = (
    "Could not query nvcc for compute_107a support: Could not find nvcc and default "
    "cuda_home='/usr/local/cuda' doesn't exist.\n"
)
OOM = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of "
    "94.97 GiB of which 95.50 MiB is free.\n"
)


def test_an_engine_out_of_memory_in_the_autotune_is_told_to_halve_the_batch(tmp_path, monkeypatch, capsys):
    """Qwen3-30B-A3B W4A4 on a 96 GB RTX PRO 6000: FlashInfer's MoE autotune runs vLLM's 16384-token batch after
    the KV cache is allocated and runs out of memory; 8192 starts. The nvcc warning before it is not the cause."""
    from qda_quant.output.vllm_env import add_fix

    error = RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")
    autotune = (
        '  File ".../vllm/model_executor/warmup/kernel_warmup.py", line 379, in '
        "_run_flashinfer_autotune_dummy_runs\n"
    )
    add_fix(error, NVCC_QUERY + autotune + OOM)
    monkeypatch.setattr(score, "Scorer", scorer(error=error))
    assert cli.main(["score", str(config(tmp_path))]) == 3
    lines = capsys.readouterr().err.splitlines()
    assert "torch.OutOfMemoryError: CUDA out of memory." in lines[0]
    assert "nvcc" not in lines[0]
    assert lines[1] == (
        "fix:   halve vLLM's batch: qda-kl kl: --serve-args '{\"max_num_batched_tokens\": 8192}'; qda-kl score: "
        "scorer.topk.serve_args.max_num_batched_tokens: 8192; a search: "
        "judge.kl_layout.serve_args.max_num_batched_tokens: 8192"
    )


def test_an_engine_out_of_memory_loading_weights_names_no_fix():
    from qda_quant.output.vllm_env import engine_fix

    load = '  File ".../vllm/model_executor/model_loader/default_loader.py", line 300, in load_weights\n'
    assert engine_fix(NVCC_QUERY + load + OOM) is None


def test_vllms_nvcc_query_warning_names_no_fix_and_hides_no_real_nvcc_failure():
    from qda_quant.output.vllm_env import engine_fix

    assert engine_fix(NVCC_QUERY) is None
    real = "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"
    found = engine_fix(NVCC_QUERY.strip() + "; " + real + "\n")
    assert found is not None and found[1].startswith("install the CUDA toolkit")


def test_nondeterminism_is_refused_without_a_file(tmp_path, monkeypatch):
    (tmp_path / "score.json").write_text("old")
    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result(), fake_result(kl=0.03)]))
    assert cli.main(["score", str(config(tmp_path, repeat=2))]) == 2
    assert not (tmp_path / "score.json").exists()


@pytest.mark.parametrize("change", [{"kl_text": 0.05}, {"math": 0.011}])
def test_repeat_guard_compares_every_reading(tmp_path, monkeypatch, change):
    # same kl, different text KL or one domain: still a disagreement
    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result(), fake_result(**change)]))
    assert cli.main(["score", str(config(tmp_path, repeat=2))]) == 2
    assert not (tmp_path / "score.json").exists()


def test_identical_repeats_are_deterministic(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result()] * 3))
    assert cli.main(["score", str(config(tmp_path, repeat=3, allow_nondeterministic=True))]) == 0
    r = json.loads((tmp_path / "score.json").read_text())
    assert r["deterministic"] is True and r["kl"] == 0.02
    assert r["repeats"] == {"kl": [0.02] * 3, "kl_text": [0.04] * 3} and r["spread"] == {
        "kl": 0.0,
        "kl_text": 0.0,
    }


def test_allow_nondeterministic_reports_mean_and_spread(tmp_path, monkeypatch):
    import json

    runs = [
        fake_result(kl=0.02, kl_text=0.04, math=0.01),
        fake_result(kl=0.03, kl_text=0.05, math=0.02),
        fake_result(kl=0.025, kl_text=0.06, math=0.03),
    ]
    monkeypatch.setattr(score, "Scorer", sequence_scorer(runs))
    assert cli.main(["score", str(config(tmp_path, repeat=3, allow_nondeterministic=True))]) == 0
    r = json.loads((tmp_path / "score.json").read_text())
    assert r["deterministic"] is False
    assert r["kl"] == pytest.approx(0.025) and r["kl_text"] == pytest.approx(0.05)
    assert r["domains"]["math"] == pytest.approx(0.02) and r["domains_text"]["web"] == pytest.approx(0.05)
    assert r["repeats"] == {"kl": [0.02, 0.03, 0.025], "kl_text": [0.04, 0.05, 0.06]}
    assert r["spread"]["kl"] == pytest.approx(0.01 / 0.025) and r["spread"]["kl_text"] == pytest.approx(
        0.02 / 0.05
    )


def test_single_score_has_no_repeat_fields(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result()]))
    assert cli.main(["score", str(config(tmp_path))]) == 0
    r = json.loads((tmp_path / "score.json").read_text())
    assert r["deterministic"] is None and r["repeats"] is None and r["spread"] is None


def test_failed_publish_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "Scorer", scorer(fake_result()))

    def fail(self, target):
        raise OSError("injected rename failure")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        cli.main(["score", str(config(tmp_path))])
    assert not (tmp_path / "score.json").exists()
    assert not list(tmp_path.glob(".score.json.*"))


def test_temp_names_are_unique(tmp_path):
    import tempfile

    names = {tempfile.mkstemp(dir=tmp_path, prefix=".result.json.", suffix=".tmp")[1] for _ in range(20)}
    assert len(names) == 20


def test_dtype_auto_is_refused():
    with pytest.raises(ValueError, match="auto"):
        resolve_dtype(ServeSettings(args={"dtype": "auto"}), "bfloat16")
    with pytest.raises(ValueError, match="auto"):
        resolve_dtype(ServeSettings(args={"dtype": "AUTO"}, source="build.json"), None)


def test_reference_without_answers_is_refused(tmp_path, monkeypatch):
    import numpy as np
    from test_topk import write_ref

    m, _ = write_ref(
        tmp_path / "ref", seqs=[("web", "text", 20, "easy", "eval"), ("books", "text", 12, "easy", "eval")]
    )
    sc = object.__new__(score.Scorer)
    sc.ref = score.Reference(tmp_path / "ref")
    sc.checkpoint, sc.llm = str(tmp_path), None
    per = {
        s.id: {
            "kl": np.full(s.n_scored, 0.1, np.float32),
            "flip": np.zeros(s.n_scored, bool),
            "tgt": np.zeros(s.n_scored, np.float32),
        }
        for s in m.sequences
    }
    monkeypatch.setattr(score, "teacher_force", lambda *a, **k: per)
    with pytest.raises(ValueError, match="no answer tokens"):
        sc.score()


@pytest.fixture
def no_engine(monkeypatch):
    """A vllm whose LLM constructor fails the test if it is ever called."""
    import sys
    import types

    calls = []
    fake = types.ModuleType("vllm")
    fake.LLM = lambda *a, **k: calls.append(k) or pytest.fail(f"vllm.LLM called with {k}")
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(score, "vllm_version", lambda: "0.33.0")
    return calls


@pytest.mark.parametrize(
    "key", ["data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_data_and_context_parallel_are_refused_before_any_engine(tmp_path, key, no_engine, capsys):
    from test_topk import write_ref

    write_ref(tmp_path / "ref")
    out = tmp_path / "run"
    p = config(
        tmp_path,
        reference=tmp_path / "ref",
        out=out,
        serve_args={"dtype": "bfloat16", key: 2},
    )
    rc = cli.main(["score", str(p)])
    assert rc == 2 and not no_engine and not (out / "score.json").exists()
    assert f"scoring is not supported ({key}=2)" in capsys.readouterr().err


def test_spread_is_null_when_differing_readings_average_to_zero():
    from qda_kl.score import combine

    s = combine([fake_result(kl=0.01, kl_text=0.0), fake_result(kl=-0.01, kl_text=0.0)])
    assert s.deterministic is False and s.kl == 0.0
    assert s.spread["kl"] is None and s.spread["kl_text"] == 0.0
    s = combine([fake_result(kl=-0.02), fake_result(kl=-0.04)])
    assert s.spread["kl"] == pytest.approx(0.02 / 0.03)  # |mean| in the denominator


def test_mean_mode_prints_a_labelled_aggregate(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setattr(score, "Scorer", sequence_scorer([fake_result(kl=0.02), fake_result(kl=0.04)]))
    p = config(tmp_path, repeat=2, allow_nondeterministic=True)
    assert cli.main(["score", str(p)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [x.get("aggregate") for x in lines] == [None, None, True]
    assert lines[-1]["kl"] == pytest.approx(0.03) and lines[-1]["deterministic"] is False


@pytest.mark.parametrize("make", [False, True], ids=["no-dir", "empty-dir"])
def test_missing_reference_is_a_refusal(tmp_path, capsys, make):
    ref = tmp_path / "ref"
    if make:
        ref.mkdir()
    (tmp_path / "score.json").write_text("old")
    p = config(tmp_path, reference=ref)
    assert cli.main(["score", str(p)]) == 2
    err = capsys.readouterr().err
    said = "not a qda-kl reference" if make else f"no reference at {ref}"
    assert err.startswith("error: ") and said in err and "qda-kl ref" in err
    assert "Traceback" not in err and not (tmp_path / "score.json").exists()


def test_reference_without_sampling_is_refused_before_the_engine(tmp_path, capsys):
    """A model with no generation_config.json and no --sampling: refused, naming the key."""
    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text("{}")
    (tmp_path / "p.jsonl").write_text('{"id": "a", "prompt": "hi"}\n')
    argv = [
        "ref",
        "--model",
        str(m),
        "--prompts",
        str(tmp_path / "p.jsonl"),
        "--out",
        str(tmp_path / "ref"),
    ]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and "--sampling" in err[0]  # the reason, then...
    assert err[1] == 'fix: pass --sampling \'{"temperature": 1.0, "top_p": 0.95}\''  # ...a line to copy


def test_ref_prints_the_released_cards_sampling_as_its_first_line(tmp_path, monkeypatch, capsys):
    """A model the cards name is built from that row, and the first line says so."""
    import json
    from types import SimpleNamespace

    import qda_kl.reference as reference

    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-9B"}))
    (m / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    (tmp_path / "p.jsonl").write_text('{"id": "a", "prompt": "hi"}\n')
    seen = {}

    def fake(model, prompts, out, **kw):
        seen.update(model=model, sampling=kw.get("sampling"))
        entry = {"serve_env": {}, "seconds": {}}
        return SimpleNamespace(
            sequences=[1], positions=2, tail_mass={}, model={"engine": "vllm x"}, build=entry
        )

    monkeypatch.setattr(reference, "build", fake)
    argv = ["ref", "--model", str(m), "--prompts", str(tmp_path / "p.jsonl"), "--out", str(tmp_path / "ref")]
    assert cli.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "ref: sampling of Qwen/Qwen3.5-9B's released cards"
    assert json.loads("\n".join(lines[1:]))["out"] == str(tmp_path / "ref")
    assert seen["sampling"] is None  # the build resolves the row itself; no flag was needed


def test_ref_names_the_cards_row_over_the_files_own_sampling(tmp_path, monkeypatch, capsys):
    """A model the table names is sampled as its cards were, and the line says which row supplied it."""
    import json
    from types import SimpleNamespace

    import qda_kl.reference as reference

    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.5-9B"}))
    (m / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    (tmp_path / "p.jsonl").write_text('{"id": "a", "prompt": "hi"}\n')

    def fake(model, prompts, out, **kw):
        entry = {"serve_env": {}, "seconds": {}}
        return SimpleNamespace(
            sequences=[1], positions=2, tail_mass={}, model={"engine": "vllm x"}, build=entry
        )

    monkeypatch.setattr(reference, "build", fake)
    argv = ["ref", "--model", str(m), "--prompts", str(tmp_path / "p.jsonl"), "--out", str(tmp_path / "ref")]
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.splitlines()[0] == ("ref: sampling of Qwen/Qwen3.5-9B's released cards")


def test_missing_prompt_file_is_a_refusal(tmp_path, capsys):
    argv = [
        "ref",
        "--model",
        str(tmp_path),
        "--prompts",
        str(tmp_path / "nope.jsonl"),
        "--out",
        str(tmp_path),
    ]
    assert cli.main(argv) == 2
    assert "no such prompt file" in capsys.readouterr().err


def test_ref_builds_every_group_in_one_call_and_prints_each(tmp_path, monkeypatch, capsys):
    """`qda-kl ref --group` hands every group to one build and prints one summary entry per reference."""
    import json
    from types import SimpleNamespace

    import qda_kl.reference as reference

    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\n")
    seen = {}

    def fake(model, groups, **kw):
        seen.update(model=model, groups=groups, kw=kw)
        entry = {"serve_env": {}, "seconds": {}}
        return [
            SimpleNamespace(sequences=[1], positions=2, tail_mass={}, model={"engine": "vllm x"}, build=entry)
            for _ in groups
        ]

    monkeypatch.setattr(reference, "build_many", fake)
    argv = [
        "ref",
        "--model",
        "m",
        "--prompts",
        "p.jsonl",
        "--out",
        "r1",
        "--subset",
        f"search96={ids}",
        "--group",
        json.dumps({"prompts": "p2.jsonl", "out": "r2", "subsets": {"holdout": str(ids)}}),
    ]
    assert cli.main(argv) == 0
    assert [str(g.out) for g in seen["groups"]] == ["r1", "r2"]
    assert seen["groups"][0].subsets == {"search96": ["a", "b"]}
    assert seen["groups"][1].subsets == {"holdout": ["a", "b"]}
    summary = json.loads(capsys.readouterr().out)
    assert [entry["out"] for entry in summary["refs"]] == ["r1", "r2"]


@pytest.mark.parametrize(
    "group",
    [
        "not json",
        "[]",
        '{"prompts": "p"}',
        '{"prompts": "p", "out": "o", "extra": 1}',
        '{"prompts": "p", "out": "o", "subsets": {"n": 3}}',
    ],
)
def test_a_group_that_is_not_a_prompt_file_and_a_directory_is_refused(tmp_path, capsys, group):
    ids = tmp_path / "ids.txt"
    ids.write_text("a\n")
    argv = [
        "ref",
        "--model",
        "m",
        "--prompts",
        "p.jsonl",
        "--out",
        "r1",
        "--subset",
        f"x={ids}",
        "--group",
        group,
    ]
    assert cli.main(argv) == 2
    assert capsys.readouterr().err.startswith("error: --group")  # the contract's line, not `refused:`


def test_ref_says_how_many_rows_were_over_the_context(tmp_path, monkeypatch, capsys):
    """A row the model's window cannot hold is dropped and counted in one line, not silently (#...)."""
    import json

    import qda_kl.refcache as refcache

    def fake(model, groups, settings, log=print):
        entry = tmp_path / "entry"
        entry.mkdir(exist_ok=True)
        build = {
            "prompts": str(groups[0].prompts),
            "dropped_over_context": ["x"],
            "prompt_rows": 40,
            "context": 16,
        }
        (entry / "manifest.json").write_text(
            json.dumps({"k": 128, "vocab": 10, "sequences": [], "positions": 0, "build": build})
        )
        for g in groups:
            if g.out is not None:
                refcache.link(Path(g.out), entry)
        return [entry for _ in groups], [True for _ in groups]

    monkeypatch.setattr(refcache, "cached_many", fake)
    argv = ["ref", "--model", "m", "--prompts", "p.jsonl", "--out", str(tmp_path / "r"), "--cache"]
    assert cli.main(argv) == 0
    assert "p.jsonl: 39 of 40 rows kept, 1 over the model's 16-token context" in capsys.readouterr().err


def test_ref_cache_warms_every_group_and_may_link_none(tmp_path, monkeypatch, capsys):
    """With --cache, every group goes through the reference cache, and a group may name no `out`
    (prepare warms the holdout entry that evaluate later links into the run)."""
    import json

    import qda_kl.refcache as refcache

    seen = {}

    def fake(model, groups, settings, log=print):
        seen.update(model=model, groups=groups, settings=settings)
        entry = tmp_path / "cache-entry"
        entry.mkdir(exist_ok=True)
        (entry / "manifest.json").write_text(
            json.dumps({"k": 128, "vocab": 10, "sequences": [], "positions": 0, "tail_mass": {}})
        )
        return [entry for _ in groups], [True for _ in groups]

    monkeypatch.setattr(refcache, "cached_many", fake)
    argv = [
        "ref",
        "--model",
        "m",
        "--prompts",
        "p.jsonl",
        "--out",
        "r1",
        "--top-k",
        "64",
        "--cache",
        "--group",
        json.dumps({"prompts": "p2.jsonl"}),
    ]
    assert cli.main(argv) == 0
    assert [str(g.out) for g in seen["groups"]] == ["r1", "None"]
    assert seen["groups"][1].prompts == "p2.jsonl"
    assert seen["settings"]["subsets"] is None
    assert seen["settings"]["k"] == 64
    summary = json.loads(capsys.readouterr().out)
    assert [entry["out"] for entry in summary["refs"]] == ["r1", str(tmp_path / "cache-entry")]
