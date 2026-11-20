"""Every place that tells a user how to serve prints the checkpoint's own serve section, rendered once."""

import json
from datetime import UTC, datetime

from qda_agent.host import Final, HandIn, serve_line, with_serve
from qda_quant.output.serve_command import NVIDIA_LIBS

SERVE = {
    "vllm": "0.33.0",
    "env": {},
    "args": {"linear_backend": "marlin"},
    "plugins": [],
    "kernels": {"dense": ["MarlinLinearKernel"]},
}


def _built(tmp_path, serve=SERVE):
    """A checkpoint directory with the build.json `qda build` writes."""
    (tmp_path / "build.json").write_text(json.dumps({"serve": serve}))
    return tmp_path


def test_the_line_is_the_checkpoints_serve_section(tmp_path):
    line = serve_line(_built(tmp_path))
    assert line.startswith(f"VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve {tmp_path} ")
    assert "LD_LIBRARY_PATH" not in line  # a Marlin line has no loader path
    assert "--linear-backend marlin" in line  # the kernel the search scored, not vLLM's own pick


def test_a_humming_pick_derives_its_loader_path_instead_of_printing_the_venvs(tmp_path, monkeypatch):
    """The end report's W3A16 line names no directory of the machine that ran the search."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib64")
    humming = {**SERVE, "args": {}, "kernels": {"dense": ["HummingLinearKernel"]}}
    line = serve_line(_built(tmp_path, humming))
    assert line.startswith(NVIDIA_LIBS + " ")
    for host in ("site-packages", "/opt/cuda"):
        assert host not in line
    assert f"vllm serve {tmp_path} " in line


def test_the_winners_line_names_the_original_and_the_format(tmp_path):
    """The end report's line, `qda build`'s and `qda inspect --serve`'s send one `model` name."""
    (tmp_path / "build.json").write_text(
        json.dumps({"source": "/models/Qwen3-8B", "format": "w4a16-asym-g128", "serve": SERVE})
    )
    line = serve_line(tmp_path)
    assert "--served-model-name Qwen3-8B-W4A16" in line
    assert "--host" not in line and "--port" not in line and "--seed" not in line


def test_a_prepared_runs_winner_is_named_after_the_original_repo(tmp_path):
    """A prepared run builds from `<out>/model`, so the name comes from the repo the recipe recorded."""
    run = tmp_path / "run"
    winner = run / "cand3-w4a16"
    winner.mkdir(parents=True)
    (run / "model").mkdir()
    (winner / "build.json").write_text(
        json.dumps({
            "source": str(run / "model"),
            "source_repo": "Qwen/Qwen3-8B",
            "format": "w4a16-asym-g128",
            "serve": SERVE,
        })
    )
    assert "--served-model-name Qwen3-8B-W4A16" in serve_line(winner)


def test_the_three_places_that_serve_name_the_checkpoint_alike(tmp_path):
    """One checkpoint, one name: the build's line, the search's report and `qda inspect --serve`."""
    import shlex

    from qda_quant.intake.report import inspect
    from qda_quant.output.serve_command import serve_command_line

    (tmp_path / "build.json").write_text(
        json.dumps({"source": "/models/Qwen3-8B", "format": "w4a16-asym-g128", "serve": SERVE})
    )

    def named(line: str | None) -> str:
        argv = shlex.split(line or "")
        return argv[argv.index("--served-model-name") + 1]

    report = inspect(str(tmp_path))["serve_name"]
    assert named(serve_command_line(tmp_path)) == named(serve_line(tmp_path)) == report == "Qwen3-8B-W4A16"


def test_no_planned_server_means_no_line(tmp_path):
    """A GGUF is served by llama.cpp; a checkpoint qda did not build names no vLLM server either."""
    assert serve_line(tmp_path) is None
    assert serve_line(None) is None
    (tmp_path / "build.json").write_text(json.dumps({"gguf": {"file": "model.gguf"}}))
    assert serve_line(tmp_path) is None


def test_the_pick_carries_the_line_into_final_json(tmp_path):
    """FINAL.json (and the end report reading it) names how to serve the picked checkpoint."""
    built = _built(tmp_path)
    handin = HandIn(dir=built, submitted_at=datetime.now(UTC))
    final = with_serve(Final(rule="handin", choice=handin))
    assert final.serve == serve_line(built)
    assert json.loads(final.model_dump_json())["serve"] == final.serve
    assert "serve" not in json.loads(Final(rule="none").model_dump_json())  # nothing picked, no line


def test_a_line_already_on_the_record_is_kept(tmp_path):
    handin = HandIn(dir=_built(tmp_path), submitted_at=datetime.now(UTC))
    final = Final(rule="handin", choice=handin, serve="vllm serve elsewhere")
    assert with_serve(final).serve == "vllm serve elsewhere"
