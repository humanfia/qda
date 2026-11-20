import argparse
import errno
import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml
from qda_kl.serve import (
    ServeSettings,
    check_kernels,
    check_vllm,
    deterministic_args,
    engine_env,
    settings_for,
)
from qda_kl.vllm_hooks import bound_kernels
from qda_quant.exit import FIX_LINE
from qda_quant.output import vllm_env
from qda_quant.output.vllm_env import tool_path
from qda_quant.step import Refused

SERVE = {
    "vllm": "0.33.0",
    "env": {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"},
    "args": {"moe_backend": "marlin"},
    "plugins": [],
    "kernels": {"dense": "MarlinLinearKernel", "moe": "MarlinExperts"},
}
PIN = {"compilation_config": {"inductor_compile_config": {"deterministic": True}}}


def ckpt(tmp_path, serve=SERVE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "build.json").write_text(json.dumps({"serve": serve, "timings": {}}))
    return tmp_path


def test_settings_from_build_json(tmp_path):
    s = settings_for(ckpt(tmp_path))
    assert s.env == SERVE["env"] and s.args == {**SERVE["args"], **PIN}
    assert s.kernels == {
        "dense": {"MarlinLinearKernel"},
        "moe": {"MarlinExperts"},
    }  # strings: written before lists
    assert s.vllm == "0.33.0" and s.source == "build.json"


def test_no_build_json_means_only_the_reduction_pin(tmp_path):
    s = settings_for(tmp_path)
    assert s == ServeSettings(args=PIN)


def test_overrides_win(tmp_path):
    env = {"VLLM_DISABLED_KERNELS": "X", "A": "1"}
    args = {"dtype": "bfloat16", "gpu_memory_utilization": 0.7, "enforce_eager": True}
    s = settings_for(ckpt(tmp_path), env, args)
    assert s.env == {"VLLM_DISABLED_KERNELS": "X", "A": "1"}
    assert s.args == {
        "moe_backend": "marlin",
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.7,
        "enforce_eager": True,
        **PIN,
    }
    assert s.source == "build.json+override"
    assert settings_for(tmp_path / "none", env).source == "override"


def test_the_reduction_pin_leaves_compilation_level_zero_alone():
    """vLLM's ``compilation_config=0`` means no compilation, so the pin must not switch it on."""
    assert deterministic_args({"compilation_config": 0}) == {"compilation_config": 0}


def test_the_reduction_pin_maps_a_compilation_level_to_vllms_dict_form():
    """vLLM reads an integer as ``CompilationMode(n)``; the pin goes under that mode, not over it."""
    assert deterministic_args({"compilation_config": 3}) == {
        "compilation_config": {"mode": 3, "inductor_compile_config": {"deterministic": True}}
    }


def test_an_explicit_deterministic_value_wins_over_the_reduction_pin():
    """A caller that turns the pin off keeps it off; the caller's other compilation keys survive."""
    args = {"compilation_config": {"mode": 3, "inductor_compile_config": {"deterministic": False}}}
    got = deterministic_args(args)
    assert got["compilation_config"] == {
        "mode": 3,
        "inductor_compile_config": {"deterministic": False},
    }
    assert args["compilation_config"]["inductor_compile_config"]["deterministic"] is False  # not mutated


def test_kernel_mismatch_is_refused():
    check_kernels(SERVE["kernels"], {"dense": ["MarlinLinearKernel"], "moe": ["MarlinExperts"]})
    check_kernels({}, {"dense": ["MacheteLinearKernel"]})  # nothing named, nothing checked
    with pytest.raises(
        Refused, match="dense: expected \\['MarlinLinearKernel'\\], bound \\['MacheteLinearKernel'\\]"
    ):
        check_kernels(SERVE["kernels"], {"dense": ["MacheteLinearKernel"], "moe": ["MarlinExperts"]})
    with pytest.raises(Refused, match="moe: expected \\['MarlinExperts'\\], bound none"):
        check_kernels(SERVE["kernels"], {"dense": ["MarlinLinearKernel"]})
    with pytest.raises(Refused, match="dense"):  # mixed binding: not the one kernel named
        check_kernels(
            {"dense": "MarlinLinearKernel"}, {"dense": ["MacheteLinearKernel", "MarlinLinearKernel"]}
        )


@pytest.mark.parametrize(
    "body,head,shape,dense",
    [
        # vLLM 0.33's choose_mp_linear_kernel with each kernel's can_implement, Machete disabled: a W4 head beside a
        # W3 body or beside the default W4 body binds Humming, the body's own kernel
        ("w3a16", "w4a16-sym-g64", (256, 128), ["HummingLinearKernel"]),
        ("w4a16", "w4a16-sym-g64", (256, 128), ["HummingLinearKernel"]),
    ],
)
def test_a_head_in_its_own_format_binds_what_vllm_picks(body, head, shape, dense):
    """`serve.kernels` holds the head's kernel as vLLM binds it beside the body, and exactly that set must bind."""
    from qda_quant.format import Format
    from qda_quant.target import Target, merge_serve

    t = Target()
    plans = {"dense": t.plan(Format.parse(body), "dense")}
    serve = merge_serve(plans, t.vllm)
    serve = merge_serve(plans, t.vllm, {"dense": t.binds(Format.parse(head), shape, serve)})
    assert serve["kernels"] == {"dense": dense}
    check_kernels(serve["kernels"], {"dense": dense})
    with pytest.raises(Refused, match="dense"):
        check_kernels(serve["kernels"], {"dense": ["HummingLinearKernel", "MacheteLinearKernel", "X"]})


def test_vllm_mismatch():
    s = ServeSettings(vllm="0.33.0")
    assert check_vllm(s, "0.33.0", False) is None
    assert check_vllm(ServeSettings(), "0.27.1", False) is None
    with pytest.raises(Refused, match="scorer.topk.allow_vllm_mismatch"):
        check_vllm(s, "0.27.1", False)
    assert "0.27.1" in check_vllm(s, "0.27.1", True)


def test_engine_env_is_restored(monkeypatch):
    monkeypatch.setenv("KEEP", "old")
    monkeypatch.delenv("QDA_KL_TEST_NEW", raising=False)
    with engine_env(ServeSettings(env={"KEEP": "new", "QDA_KL_TEST_NEW": "1"}, plugins=["p1", "p2"])):
        assert os.environ["KEEP"] == "new" and os.environ["QDA_KL_TEST_NEW"] == "1"
        assert os.environ["VLLM_PLUGINS"] == "p1,p2"
    assert os.environ["KEEP"] == "old" and "QDA_KL_TEST_NEW" not in os.environ


def test_engine_env_sets_nvls_from_the_config_key(monkeypatch):
    """A multi-card engine dies at NCCL init where NVLS cannot bind, and CI has no cards to show it."""
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "from-the-shell")
    with engine_env(ServeSettings(nccl_nvls=False)):
        assert os.environ["NCCL_NVLS_ENABLE"] == "0"
    with engine_env(ServeSettings(nccl_nvls=True)):
        assert os.environ["NCCL_NVLS_ENABLE"] == "1"
    with engine_env(ServeSettings()):  # a caller with no such key leaves it alone
        assert os.environ["NCCL_NVLS_ENABLE"] == "from-the-shell"


class MarlinLinearKernel:
    pass


class MarlinExperts:
    pass


class CompressedTensorsLinearMethod:
    pass


class UnquantizedLinearMethod:
    pass


class CompressedTensorsKVCacheMethod:
    pass


def test_bound_kernels_walks_layers():
    m = torch.nn.Module()
    lin, head, moe, scheme = torch.nn.Module(), torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
    scheme.kernel = MarlinLinearKernel()
    lin.quant_method, lin.scheme = CompressedTensorsLinearMethod(), scheme
    head.quant_method = UnquantizedLinearMethod()
    moe.quant_method = type(
        "Method", (), {"moe_kernel": type("MK", (), {"fused_experts": MarlinExperts()})()}
    )()
    moe.w13_weight = torch.zeros(1)
    attn = torch.nn.Module()
    attn.quant_method = CompressedTensorsKVCacheMethod()  # vLLM 0.33 attention layers carry one
    m.lin, m.head, m.moe, m.attn = lin, head, moe, attn
    assert bound_kernels(m) == {"dense": ["MarlinLinearKernel"], "moe": ["MarlinExperts"]}


class CutlassFP8ScaledMMLinearKernel:
    pass


def test_an_fp8_scheme_names_its_kernel_on_fp8_linear():
    """vLLM 0.33's W8A8 FP8 scheme holds its kernel on `fp8_linear`, not `kernel`."""
    m, fp8 = torch.nn.Module(), torch.nn.Module()
    fp8.quant_method, fp8.scheme = CompressedTensorsLinearMethod(), torch.nn.Module()
    fp8.scheme.fp8_linear = CutlassFP8ScaledMMLinearKernel()
    m.q_proj = fp8
    assert bound_kernels(m) == {"dense": ["CutlassFP8ScaledMMLinearKernel"]}


def test_dtype_precedence(tmp_path):
    from qda_kl.serve import resolve_dtype

    with_dtype = ckpt(tmp_path / "a", {**SERVE, "args": {"dtype": "float16"}})
    plain = tmp_path / "b"
    plain.mkdir()
    # 1. build.json's dtype requires permission when it differs from the anchor's.
    for s in (settings_for(with_dtype), settings_for(with_dtype, args={"dtype": "float16"})):
        with pytest.raises(ValueError, match="float16.*bfloat16.*allow_dtype_mismatch"):
            resolve_dtype(s, "bfloat16")
        got, why = resolve_dtype(s, "bfloat16", allow_mismatch=True)
        assert got.args["dtype"] == "float16" and why in ("serve-settings", "build.json+override")
    # a conflicting override is refused, never dropped
    with pytest.raises(ValueError, match="conflicts"):
        settings_for(with_dtype, args={"dtype": "bfloat16"})
    # 2. scorer.topk.serve_args.dtype over the anchor's
    with pytest.raises(ValueError, match="allow_dtype_mismatch"):
        resolve_dtype(settings_for(plain, args={"dtype": "float32"}), "bfloat16")
    got, why = resolve_dtype(settings_for(plain, args={"dtype": "float32"}), "bfloat16", allow_mismatch=True)
    assert got.args["dtype"] == "float32" and why == "override"
    # 3. the anchor's dtype; never vLLM's auto
    got, why = resolve_dtype(settings_for(plain), "bfloat16")
    assert got.args["dtype"] == "bfloat16" and why == "reference anchor"
    with pytest.raises(ValueError, match="no dtype"):
        resolve_dtype(settings_for(plain), None)


def test_parallel_settings_are_checked_before_any_engine():
    from qda_kl.engine import check_parallel, open_engine

    check_parallel({"tensor_parallel_size": 2, "pipeline_parallel_size": 2})  # both are supported
    with pytest.raises(Refused, match="data-parallel scoring is not supported"):
        check_parallel({"data_parallel_size": 2})
    with pytest.raises(ValueError, match="tp must be at least 1"):
        open_engine("no-such-model", tp=0, max_model_len=16)
    with pytest.raises(ValueError, match="set tp=2 instead"):  # one card count: the config's tp
        open_engine(
            "no-such-model",
            tp=2,
            max_model_len=16,
            settings=ServeSettings(args={"tensor_parallel_size": 3}),
        )


def test_expect_kernel_reaches_the_check(tmp_path, monkeypatch):
    from qda_kl import cli, score

    seen = {}

    def fake_scorer(checkpoint, ref, settings, **kw):
        seen["kernels"] = settings.kernels
        raise Refused("stop")

    monkeypatch.setattr(score, "Scorer", fake_scorer)
    cfg = tmp_path / "score.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "checkpoint": str(tmp_path),
                "reference": str(tmp_path),
                "out": str(tmp_path / "run"),
                "scorer": {"topk": {"expect_kernels": {"moe": "MarlinExperts"}}},
            }
        )
    )
    rc = cli.main(["score", str(cfg)])
    assert rc == 2 and seen["kernels"] == {"moe": "MarlinExperts"}


def test_tool_path_puts_the_venv_bin_first_once(monkeypatch, tmp_path):
    monkeypatch.setattr(vllm_env.sys, "executable", str(tmp_path / "bin" / "python"))
    monkeypatch.setenv("PATH", "/usr/bin")
    assert tool_path() == os.pathsep.join([str(tmp_path / "bin"), "/usr/bin"])
    monkeypatch.setenv("PATH", tool_path())
    assert tool_path() == os.environ["PATH"]


def test_engine_env_has_the_venv_bin_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(vllm_env.sys, "executable", str(tmp_path / "bin" / "python"))
    monkeypatch.setenv("PATH", "/usr/bin")
    with engine_env(ServeSettings()):
        assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")
    assert os.environ["PATH"] == "/usr/bin"


@pytest.mark.parametrize(
    "bad",
    [
        {"env": {"PYTHONPATH": "/tmp/x"}},
        {"env": {"LD_PRELOAD": "/x.so"}},
        {"args": {"trust_remote_code": True}},
        {"args": {"tensor_parallel_size": 2}},  # a card count is the scorer's tp, not the checkpoint's
    ],
)
def test_build_json_serve_beyond_qdas_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="serve settings not allowed"):
        settings_for(ckpt(tmp_path, {**SERVE, **bad}))


def test_explicit_overrides_stay_the_callers(tmp_path):
    """A user scoring their own checkpoint may still set what they like on the command line."""
    s = settings_for(ckpt(tmp_path), env={"MY_FLAG": "1"}, args={"enforce_eager": True})
    assert s.env["MY_FLAG"] == "1" and s.args["enforce_eager"] is True


def _caches_under(monkeypatch, tmp_path):
    """The engine caches go under tmp_path."""
    for name in vllm_env.CACHE_ENV:
        monkeypatch.setenv(name, str(tmp_path / "caches" / name))


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("model config is invalid"),
        OSError(errno.EIO, "Input/output error", "model.safetensors"),
    ],
)
def test_engine_start_raises_a_failure_once_unchanged(tmp_path, monkeypatch, error):
    """An engine start is not retried: a failed start raises once, the same object the start raised."""
    from qda_kl.serve import start_engine

    _caches_under(monkeypatch, tmp_path)
    calls = []

    def start():
        calls.append(1)
        raise error

    with pytest.raises(type(error)) as caught:
        start_engine(ServeSettings(), start)
    assert caught.value is error and len(calls) == 1


def test_engine_start_runs_without_a_stderr_to_capture():
    """A process with fd 2 closed still starts an engine: there is simply nothing to capture."""
    from qda_kl.serve import start_engine

    def closed(fd):
        raise OSError(errno.EBADF, "Bad file descriptor")

    with mock.patch.object(vllm_env.os, "dup", closed):
        assert start_engine(ServeSettings(), lambda: "engine") == "engine"


def test_a_prompt_set_engine_serves_its_prompts_at_most_at_vllms_default():
    """The prompt count, never above vLLM's smaller default; a ``max_num_seqs`` of the caller's wins."""
    from qda_kl.serve import VLLM_MAX_NUM_SEQS, batch_args

    assert batch_args({"seed": 1}, 32) == {"seed": 1, "max_num_seqs": 32}
    assert batch_args({}, 10_000)["max_num_seqs"] == VLLM_MAX_NUM_SEQS
    assert batch_args({}, 0)["max_num_seqs"] == 1
    assert batch_args({"max_num_seqs": 2048}, 32) == {"max_num_seqs": 2048}


def test_too_few_cache_blocks_is_refused_in_one_line(tmp_path, monkeypatch):
    """A hybrid model's engine core said this in its own traceback; the parent saw only a bare ``RuntimeError``."""
    from qda_kl.serve import start_engine

    _caches_under(monkeypatch, tmp_path)

    def start():
        os.write(
            2,
            b"ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (918). Each decode sequence "
            b"requires one Mamba cache block, so CUDA graph capture cannot proceed.\n",
        )
        raise RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")

    with pytest.raises(Refused) as caught:
        start_engine(ServeSettings(), start)
    why, fix = str(caught.value), caught.value.fix
    assert "\n" not in why and "max_num_seqs=1024" in why and "918 Mamba cache blocks" in why
    assert "\n" not in fix and "qda-kl kl: add --serve-args '{\"max_num_seqs\": 512}'" in fix
    assert (
        "scorer.topk.serve_args.max_num_seqs: 512" in fix
        and "judge.kl_layout.serve_args.max_num_seqs: 512" in fix
    )
    assert "up to 918" in fix


@pytest.mark.parametrize(("blocks", "n"), [(249, 128), (256, 256), (918, 512), (1, 1)])
def test_the_fix_names_a_power_of_two_the_blocks_hold(blocks, n):
    """The ``qda-kl kl`` default 256 on a hybrid with 249 blocks: the fix line is the flag, with 128."""
    from qda_kl.serve import cache_refusal

    got = cache_refusal(f"max_num_seqs (256) exceeds available Mamba cache blocks ({blocks}).")
    assert got is not None and f"--serve-args '{{\"max_num_seqs\": {n}}}'" in got.fix
    assert cache_refusal("RuntimeError: Engine core initialization failed.") is None


def test_the_kl_command_prints_the_cache_fix_line(monkeypatch, capsys):
    """`qda-kl kl` prints the refusal and its `fix:` line, not vLLM's traceback."""
    from qda_kl import kl
    from qda_kl.serve import cache_refusal

    def refuse(*_a, **_k):
        raise cache_refusal("max_num_seqs (256) exceeds available Mamba cache blocks (249).")  # type: ignore[misc]

    monkeypatch.setattr("qda_kl.refcache.local_model", lambda m: Path(m))
    monkeypatch.setattr("qda_kl.refcache.cached", refuse)
    monkeypatch.setattr(kl, "default_prompts", lambda: Path("prompts.jsonl"))
    monkeypatch.setattr("qda_kl.reference.default_sampling", lambda _m: {})
    parser = argparse.ArgumentParser()
    kl.add_arguments(parser)
    code = kl.run(parser.parse_args(["ckpt", "--original", "orig"]))
    err = capsys.readouterr().err
    assert code != 0 and "max_num_seqs=256 exceeds the engine's 249 Mamba cache blocks" in err
    assert FIX_LINE + "qda-kl kl: add --serve-args '{\"max_num_seqs\": 128}'" in err


@pytest.mark.parametrize(
    ("said", "fix"),
    [
        (
            "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist",
            "CUDA toolkit",
        ),
        ("/bin/sh: 1: /opt/cuda/bin/nvcc: not found", "CUDA toolkit"),
        ("FileNotFoundError: [Errno 2] No such file or directory: 'ninja'", "pip install ninja"),
        ("RuntimeError: Failed to find C compiler. Please specify via CC environment variable", "set CC"),
        ("/bin/sh: 1: c++: not found", "set CXX"),
    ],
)
def test_a_known_engine_failure_keeps_its_error_and_adds_a_fix_line(tmp_path, monkeypatch, said, fix):
    """vLLM's parent sees a bare RuntimeError; the engine's own line and QDA's fix are noted after it."""
    from qda_kl.serve import start_engine

    _caches_under(monkeypatch, tmp_path)
    error = RuntimeError("Engine core initialization failed. See root cause above.")

    def start():
        os.write(2, f"{said}\n".encode())
        raise error

    with pytest.raises(RuntimeError) as caught:
        start_engine(ServeSettings(), start)
    assert caught.value is error and str(error).startswith("Engine core initialization failed")
    assert caught.value.__notes__[0] == said
    assert caught.value.__notes__[1].startswith(FIX_LINE) and fix in caught.value.__notes__[1]


def test_an_unknown_engine_failure_gets_no_fix_line(tmp_path, monkeypatch):
    from qda_kl.serve import start_engine

    _caches_under(monkeypatch, tmp_path)

    def start():
        raise RuntimeError("model config is invalid")

    with pytest.raises(RuntimeError) as caught:
        start_engine(ServeSettings(), start)
    assert not getattr(caught.value, "__notes__", None)


def test_a_revision_after_the_at_is_the_engines_revision_argument():
    """vLLM reads ``@rev`` as part of the repo id; ``pinned`` hands it over as ``revision=``."""
    from qda_kl.serve import pinned
    from qda_quant.step import Refused

    assert pinned("org/m@abc", {"seed": 1}) == ("org/m", {"seed": 1, "revision": "abc"})
    assert pinned("org/m", {"seed": 1}) == ("org/m", {"seed": 1})
    assert pinned("org/m@abc", {"revision": "abc"}) == ("org/m", {"revision": "abc"})
    with pytest.raises(Refused, match="names revision def, the model id abc"):
        pinned("org/m@abc", {"revision": "def"})
