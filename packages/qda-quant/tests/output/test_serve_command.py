"""serve_command: one rendered `vllm serve` line, from the checkpoint's own serve section."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from qda_quant.format import FORMATS
from qda_quant.output.serve_command import (
    NVIDIA_LIBS,
    binds_nvrtc_kernel,
    command_environment,
    command_line,
    serve_argv,
    serve_command_line,
    serve_section,
    served_name,
)
from qda_quant.output.serve_policy import ServeNotAllowed
from qda_quant.output.vllm_env import nvidia_lib_path
from qda_quant.step import Refused
from qda_quant.target import Target, merge_serve


def _served(tmp_path, serve, name="build.json"):
    (tmp_path / name).write_text(json.dumps({"serve": serve}))
    return tmp_path


def test_serve_argv_renders_every_setting():
    """The flags carry the settings: booleans as flags, mappings as JSON."""
    argv = serve_argv(
        "/ckpt",
        {
            "max_model_len": 114688,
            "enable_prefix_caching": False,
            "tokenizer": "/tokenizer/ckpt",
            "compilation_config": {"inductor_compile_config": {"deterministic": True}},
        },
        port=8123,
    )
    assert argv[:3] == ["vllm", "serve", "/ckpt"]
    assert argv[argv.index("--port") + 1] == "8123"
    assert "--no-enable-prefix-caching" in argv
    assert argv[argv.index("--tokenizer") + 1] == "/tokenizer/ckpt"
    assert json.loads(argv[argv.index("--compilation-config") + 1]) == {
        "inductor_compile_config": {"deterministic": True}
    }


def test_serve_argv_refuses_a_setting_it_cannot_render():
    with pytest.raises(Refused, match="no vllm serve flag"):
        serve_argv("/ckpt", {"something_new": 1})


def test_command_line_prefixes_the_environment_and_quotes():
    """The kernel pin rides in front of the command, so one pasted line starts the scored engine."""
    env = command_environment({"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}, ["mypkg"])
    line = command_line(serve_argv("/ckpt", {"linear_backend": "marlin"}), env)
    assert line.startswith("VLLM_DISABLED_KERNELS=MacheteLinearKernel VLLM_PLUGINS=mypkg vllm serve ")
    assert shlex.split(line) == [
        "VLLM_DISABLED_KERNELS=MacheteLinearKernel",
        "VLLM_PLUGINS=mypkg",
        "vllm",
        "serve",
        "/ckpt",
        "--served-model-name",
        "ckpt",  # no record: the checkpoint's own name
        "--linear-backend",
        "marlin",
    ]


def test_a_user_line_names_the_model_and_leaves_out_the_servers_flags(tmp_path):
    """A checkpoint has no interface, port or seed: `qda build` prints the model's name and the kernel's flags."""
    built = tmp_path / "ckpt"
    built.mkdir()
    line = serve_command_line(_served(built, {"args": {"linear_backend": "marlin"}}))
    argv = shlex.split(line)
    assert argv[argv.index("--served-model-name") + 1] == "ckpt"
    for flag in ("--host", "--port", "--seed", "--disable-log-requests"):
        assert flag not in argv


def test_the_served_name_is_the_originals_name_and_the_stored_format(tmp_path):
    """`build.json` names the original model and the format, so one checkpoint has one name everywhere."""
    built = tmp_path / "ckpt"
    built.mkdir()
    _served(built, {})
    (built / "build.json").write_text(
        json.dumps({"source": "/models/Qwen3-0.6B", "format": "w4a16-asym-g128", "serve": {"args": {}}})
    )
    assert served_name(built) == "Qwen3-0.6B-W4A16"
    line = serve_command_line(built)
    assert shlex.split(line)[shlex.split(line).index("--served-model-name") + 1] == "Qwen3-0.6B-W4A16"


def test_the_quickstart_at_w4a4_is_served_as_the_name_its_page_gives(tmp_path):
    """docs/quickstart.md names the w4a4 build's server `Qwen3-0.6B-W4A4-NVFP4`: the name a request must carry."""
    built = tmp_path / "ckpt"
    built.mkdir()
    _served(built, {})
    (built / "build.json").write_text(
        json.dumps({"source_repo": "Qwen/Qwen3-0.6B@" + "c" * 40, "format": "w4a4-nvfp4", "serve": {}})
    )
    assert served_name(built) == "Qwen3-0.6B-W4A4-NVFP4"
    page = (Path(__file__).parents[4] / "docs" / "quickstart.md").read_text()
    assert "`Qwen3-0.6B-W4A4-NVFP4` for `format: w4a4`" in page

    (built / "build.json").write_text(  # a hub id's snapshot: the repo name, not the commit
        json.dumps({
            "source": "/hub/models--Qwen--Qwen3.5-9B/snapshots/abc123",
            "format": "w4a16-sym-g32",
            "serve": {"args": {}},
        })
    )
    # the short spelling keeps a symmetry it does not inherit, so this one stays explicit
    assert served_name(built) == "Qwen3.5-9B-W4A16-SYM-G32"


def test_a_prepared_download_is_named_after_the_repo_that_was_downloaded(tmp_path):
    """A prepared run stores the model as `<out>/model`: the recorded repo names it, not the directory."""
    out = tmp_path / "run"
    built = out / "cand0-w4a16"
    built.mkdir(parents=True)
    (out / "model").mkdir()
    (built / "build.json").write_text(
        json.dumps({
            "source": str(out / "model"),
            "source_repo": "Qwen/Qwen3-8B",
            "format": "w4a16-asym-g128",
            "serve": {"args": {}},
        })
    )
    assert served_name(built) == "Qwen3-8B-W4A16"


def test_a_local_directory_is_named_after_the_repo_its_config_names(tmp_path):
    """A model downloaded by hand keeps its repo in `config.json`; the directory name is the last resort."""
    built = tmp_path / "ckpt"
    built.mkdir()
    (built / "build.json").write_text(
        json.dumps({"source": "/data/bf16", "format": "w4a16-asym-g128", "serve": {"args": {}}})
    )
    assert served_name(built) == "bf16-W4A16"  # nothing in the record names a repo
    (built / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3-8B"}))
    assert served_name(built) == "Qwen3-8B-W4A16"


def test_serve_argv_keeps_the_flags_the_measurement_needs():
    """The scoring engine's own server passes a name and the flags a user's line leaves out."""
    argv = serve_argv(
        "/ckpt", {}, name="qda", host="127.0.0.1", port=8123, seed=17, disable_log_requests=True
    )
    assert argv[3:5] == ["--served-model-name", "qda"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8123"
    assert argv[argv.index("--seed") + 1] == "17"
    assert "--disable-log-requests" in argv


def test_the_sm90_plan_is_the_command(tmp_path):
    """The line comes from the target table: the default disables Machete and pins nothing else; `:marlin` pins
    Marlin with the flag on 0.33 and with the env on 0.32."""
    t = Target("sm90", "0.33.0")
    plans = {"dense": t.plan(FORMATS["w4a16"], "dense"), "moe": t.plan(FORMATS["w4a16"], "moe")}
    line = serve_command_line(_served(tmp_path, merge_serve(plans, "0.33.0")))
    assert line.removeprefix(NVIDIA_LIBS + " ").startswith(
        "VLLM_DISABLED_KERNELS=MacheteLinearKernel "
    )  # Humming
    assert "--linear-backend" not in line and "--moe-backend" not in line

    marlin = merge_serve(
        {"dense": Target("sm90", "0.33.0", "marlin").plan(FORMATS["w4a16"], "dense")}, "0.33.0"
    )
    line = serve_command_line(_served(tmp_path, marlin))
    assert line.startswith("VLLM_DISABLED_KERNELS=MacheteLinearKernel ") and "--linear-backend marlin" in line

    old = merge_serve({"dense": Target("sm90", "0.32.0", "marlin").plan(FORMATS["w4a16"], "dense")}, "0.32.0")
    line = serve_command_line(_served(tmp_path, old))
    assert line.startswith("VLLM_DISABLED_KERNELS=HummingLinearKernel,MacheteLinearKernel ")
    assert "--linear-backend" not in line


def test_no_serve_section_means_no_command(tmp_path):
    """A GGUF names no vLLM server; a checkpoint qda did not build names none either."""
    assert serve_command_line(tmp_path) is None  # no build.json
    (tmp_path / "build.json").write_text(json.dumps({"gguf": {"file": "model.gguf"}}))
    assert serve_command_line(tmp_path) is None


@pytest.mark.parametrize("serve", [{"env": {"LD_PRELOAD": "/tmp/x.so"}}, [], {"kernels": {"dense": [{}]}}])
def test_a_section_qda_build_never_writes_is_not_rendered(tmp_path, serve):
    with pytest.raises(ServeNotAllowed):
        serve_command_line(_served(tmp_path, serve))


def _plan(fmt, backend=None, vllm="0.33.0"):
    """The serve section `qda build` writes for a dense `fmt` on an H100/H200."""
    return merge_serve({"dense": Target("sm90", vllm, backend).plan(FORMATS[fmt], "dense")}, vllm)


#: ``build.json`` of two qda-lab repos exactly as published, no weights: a W4A16 that pins Marlin by its args
#: and a W3A16 that records Humming.
PUBLISHED = Path(__file__).parent / "published"
SHELLS = [
    pytest.param(sh, marks=pytest.mark.skipif(shutil.which(sh) is None, reason=f"needs {sh}"))
    for sh in ("bash", "zsh")
]


def _venv_bin(tmp_path, *, python=True):
    """A venv's ``bin``: a ``vllm`` that prints what it was started with and a ``python``.

    `python`: True is the python running these tests (so the line finds this environment's ``nvidia/*/lib``);
    False has none; a string is the body of a shell script that stands in for it.
    """
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    vllm = bin_dir / "vllm"
    vllm.write_text(
        '#!/bin/sh\nprintf "%s\\n" "LD=$LD_LIBRARY_PATH" "DISABLED=$VLLM_DISABLED_KERNELS" "$@"\n'
    )
    vllm.chmod(0o755)
    if python is not False:
        shim = bin_dir / "python"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "$@"\n' if python is True else f"#!/bin/sh\n{python}\n"
        )
        shim.chmod(0o755)
    return bin_dir


def _run(shell, line, path, *, ld=None, cwd=None):
    """`line` run by `shell` with `path` in front of ``PATH`` and the caller's ``LD_LIBRARY_PATH`` `ld`."""
    env = {**os.environ, "PATH": f"{path}{os.pathsep}/usr/bin{os.pathsep}/bin"}
    env.pop("LD_LIBRARY_PATH", None)
    if ld is not None:
        env["LD_LIBRARY_PATH"] = ld
    return subprocess.run([shell, "-c", line], env=env, cwd=cwd, capture_output=True, text=True, check=False)


def test_a_marlin_line_carries_no_loader_path(tmp_path, monkeypatch):
    """A Marlin plan needs no NVRTC: the line is the kernel's flags and nothing of the machine's."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
    line = serve_command_line(_served(tmp_path, _plan("w4a16", "marlin")))
    assert "LD_LIBRARY_PATH" not in line
    assert "/venv" not in line.replace(str(tmp_path), "") and "/opt/lib" not in line
    assert line.startswith("VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve ")
    assert "--linear-backend marlin" in line  # what vLLM needs to bind the scored kernel stays


@pytest.mark.parametrize("fmt", ["w3a16", "w4a16"])
def test_a_humming_line_finds_the_loader_path_when_it_runs(tmp_path, monkeypatch, fmt):
    """Humming's NVRTC child dies without the venv's nvidia/*/lib: the line derives them where it runs.

    On an H100/H200 the default W4A16 plan lets vLLM pick, and with Machete off it picks Humming.
    The line names no directory of the machine that printed it (the venv's, nor the printing shell's own
    ``LD_LIBRARY_PATH``): the user's shell resolves both, so the line is short and runs in any activated venv.
    """
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
    ckpt = tmp_path / "Qwen3.5-4B"
    ckpt.mkdir()
    line = serve_command_line(_served(ckpt, _plan(fmt)))
    assert line.startswith(NVIDIA_LIBS + " ")
    for host in ("/venv", "site-packages", "/opt/lib"):
        assert host not in line.replace(str(ckpt), "")
    argv = shlex.split(line.removeprefix(NVIDIA_LIBS + " "))
    assert argv[:3] == ["VLLM_DISABLED_KERNELS=MacheteLinearKernel", "vllm", "serve"]  # the pin rides along


@pytest.mark.parametrize("shell", SHELLS)
def test_the_derived_loader_path_is_the_one_qda_starts_its_engines_with(tmp_path, shell):
    """The line's shell text gives `nvidia_lib_path` first and the caller's own value after it, in bash and zsh."""
    if not nvidia_lib_path():
        pytest.skip("this interpreter has no nvidia-* wheels")
    bin_dir = _venv_bin(tmp_path)
    dirs = set(nvidia_lib_path().split(os.pathsep))
    got = (
        _run(shell, f"{NVIDIA_LIBS} vllm", bin_dir, ld="/opt/lib").stdout.splitlines()[0].removeprefix("LD=")
    )
    assert got.split(os.pathsep)[-1] == "/opt/lib" and set(got.split(os.pathsep)[:-1]) == dirs
    got = _run(shell, f"set -u; {NVIDIA_LIBS} vllm", bin_dir).stdout.splitlines()[0].removeprefix("LD=")
    assert set(got.split(os.pathsep)) == dirs, (
        "no empty entry (the working directory) when the caller has no value"
    )

    other = (
        tmp_path / "conda" / "bin"
    )  # another environment's `python` first on PATH (conda base): not vLLM's
    other.mkdir(parents=True)
    (other / "python").write_text("#!/bin/sh\necho /conda/nvidia/cu13/lib\n")
    (other / "python").chmod(0o755)
    got = _run(shell, f"{NVIDIA_LIBS} vllm", f"{other}{os.pathsep}{bin_dir}").stdout.splitlines()[0]
    assert "/conda" not in got and set(got.removeprefix("LD=").split(os.pathsep)) == dirs


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("repo", ["QDA-Qwen3.5-4B-W4A16", "QDA-Qwen3.5-4B-W3A16"])
def test_a_published_records_line_runs_in_bash_and_zsh(tmp_path, shell, repo):
    """Both published records render the Humming line: vLLM picks the W4A16's kernel now, Machete off.

    The W4A16 record pins Marlin by its args and records Marlin; read as vLLM picks, it needs the loader path.
    """
    ckpt = PUBLISHED / repo
    line = serve_command_line(ckpt)
    name = {"QDA-Qwen3.5-4B-W4A16": "Qwen3.5-4B-W4A16-G64", "QDA-Qwen3.5-4B-W3A16": "Qwen3.5-4B-W3A16"}[repo]
    want = f"VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve {shlex.quote(str(ckpt))} --served-model-name {name}"
    assert line == f"{NVIDIA_LIBS} {want}"
    ran = _run(shell, line, _venv_bin(tmp_path), ld="/opt/lib")
    assert ran.returncode == 0 and ran.stderr == ""
    ld, disabled, *argv = ran.stdout.splitlines()
    assert ld.endswith(f"{os.pathsep}/opt/lib") or not nvidia_lib_path()
    assert disabled == "DISABLED=MacheteLinearKernel"
    assert argv == ["serve", str(ckpt), "--served-model-name", name]


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("python", [False, None], ids=["vllm-without-python", "no-vllm"])
def test_a_line_without_python_next_to_vllm_prints_one_line_and_starts_nothing(tmp_path, shell, python):
    """A `vllm` with no `python` in its directory (a `uv tool` or pipx shim), or no `vllm` at all: no loader path.

    A `python` in the working directory (a shell standing in a venv's ``bin``) is not the one asked.
    """
    bin_dir = _venv_bin(tmp_path, python=False) if python is False else tmp_path / "empty"
    bin_dir.mkdir(exist_ok=True)
    here = tmp_path / "here"
    here.mkdir()
    (here / "python").write_text("#!/bin/sh\necho /here/nvidia/lib\n")
    (here / "python").chmod(0o755)
    ran = _run(
        shell, serve_command_line(PUBLISHED / "QDA-Qwen3.5-4B-W3A16"), bin_dir, ld="/opt/lib", cwd=here
    )
    assert ran.returncode != 0
    assert ran.stdout == "", "vllm must not start"
    assert ran.stderr == "no python next to vllm: activate the venv that has vLLM\n"


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("ld", [None, "/opt/lib"])
def test_a_venv_without_nvidia_wheels_leaves_the_loader_path_as_it_was(tmp_path, shell, ld):
    """A conda env keeps CUDA outside ``site-packages/nvidia``: nothing to add, and no empty entry (the working
    directory) in front of the user's own value either. A script's ``set -u`` does not stop the line."""
    ran = _run(shell, f"set -u; {NVIDIA_LIBS} vllm", _venv_bin(tmp_path, python=""), ld=ld)
    assert ran.returncode == 0 and ran.stdout.splitlines()[0] == f"LD={ld or ''}"


@pytest.mark.parametrize("fmt", ["w4a16", "w3a16"])
def test_qda_build_prints_the_same_short_line(tmp_path, capsys, fmt):
    """`qda build` prints the renderer's line after `serve it with:`: no directory of the machine in it."""
    from qda_quant.cli import _print_serve

    ckpt = tmp_path / "out"
    ckpt.mkdir()
    _served(ckpt, _plan(fmt))
    _print_serve(ckpt)
    printed = capsys.readouterr().out
    assert printed == f"serve it with:\n  {serve_command_line(ckpt)}\n"
    assert sys.prefix not in printed and "site-packages/nvidia" not in printed


def test_only_the_kernels_that_compile_through_nvrtc_need_the_loader_path():
    assert binds_nvrtc_kernel(_plan("w3a16"))
    assert binds_nvrtc_kernel(_plan("w4a16"))  # sm90 default: vLLM picks, Machete off, so Humming
    assert not binds_nvrtc_kernel(_plan("w4a16", "marlin"))
    mixed = {"dense": ["MarlinLinearKernel"], "moe": ["HummingIndexedExperts"]}
    assert binds_nvrtc_kernel({"kernels": mixed})
    # an AWQ checkpoint records no kernels: vLLM picks, and may pick Humming for its experts
    assert binds_nvrtc_kernel({"kernels": {}})


@pytest.mark.parametrize(
    ("target", "args"), [("sm90@0.33.0", {}), ("sm90@0.33.0:marlin", {"linear_backend": "marlin"})]
)
def test_a_stored_sm90_record_on_the_old_marlin_default_serves_unpinned(tmp_path, target, args):
    """The published sm90 W4A16 records pin Marlin by their args; a `:marlin` build keeps its pin."""
    serve = {"vllm": "0.33.0", "env": {}, "args": {"linear_backend": "marlin"}, "plugins": [],
             "kernels": {"dense": ["MarlinLinearKernel"]}}  # fmt: skip
    (tmp_path / "build.json").write_text(json.dumps({"target": target, "serve": serve}))
    assert serve_section(tmp_path)["args"] == args
    assert ("--linear-backend" in serve_command_line(tmp_path)) == bool(args)
