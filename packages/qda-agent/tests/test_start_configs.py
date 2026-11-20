"""The launch preflight refuses a staged start / candidate-0 config that puts a card step on CPU.

A start config handed to the agent is the one its later configs inherit; with cards available, a `devices`
naming only CPU on a step that runs on a card turns a minutes-long build into an hours-long one. `devices`
unset, null or naming a card passes, as does a CPU-only step list (RTN alone).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from qda_agent.judge import startup

if TYPE_CHECKING:
    from qda_agent.judge.host import JudgeHost


class _Host:
    """The scheduler settings `_start_configs` reads; nothing else of a judge host is needed."""

    def __init__(self, *, readable=(), binds=None, gpus=True, fake=False, cards=(0,), qda_roots=()) -> None:
        self.sched = SimpleNamespace(
            cfg=SimpleNamespace(
                readable=[Path(p) for p in readable],
                binds={Path(k): Path(v) for k, v in (binds or {}).items()},
                gpus=gpus,
                fake_gpus=fake,
                cards=list(cards),
            )
        )
        self.qda_roots = [Path(p) for p in qda_roots]


def config(tmp_path: Path, name: str, *, devices: str | None, steps: str = "  - gptq: {}") -> Path:
    body = f"model: /m\nformat: w4a16\nsteps:\n{steps}\n"
    if devices is not None:
        body += f"devices:\n{devices}\n"
    path = tmp_path / name
    path.write_text(body)
    return path


def refuse(host: _Host) -> str:
    with pytest.raises(ValueError) as refused:
        startup._start_configs(cast("JudgeHost", host))
    return str(refused.value)


def test_a_cpu_devices_start_with_a_gptq_step_is_refused(tmp_path):
    path = config(tmp_path, "cand0-G-T.yaml", devices="  - cpu")
    message = refuse(_Host(readable=[path]))
    assert f"{path}:5" in message, message
    assert "devices: cpu puts gptq on CPU" in message, message
    assert "the run has cards" in message


def test_devices_unset_or_null_or_a_card_pass(tmp_path):
    paths = [
        config(tmp_path, "unset.yaml", devices=None),
        config(tmp_path, "null.yaml", devices="  null"),
        config(tmp_path, "card.yaml", devices="  - 0"),
        config(tmp_path, "mixed.yaml", devices="  - 0\n  - cpu"),
    ]
    startup._start_configs(cast("JudgeHost", _Host(readable=paths)))


def test_a_cpu_only_step_list_passes(tmp_path):
    rtn = config(tmp_path, "rtn.yaml", devices="  - cpu", steps="  - rtn: {}")
    startup._start_configs(cast("JudgeHost", _Host(readable=[rtn])))


def test_a_directory_of_staged_configs_is_scanned(tmp_path):
    (tmp_path / "configs").mkdir()
    bad = config(tmp_path / "configs", "start.yaml", devices="  - cpu")
    message = refuse(_Host(readable=[tmp_path / "configs"]))
    assert str(bad) in message and "gptq" in message


def test_a_bound_config_file_is_scanned(tmp_path):
    path = config(tmp_path, "known-good.yaml", devices="  - cpu")
    message = refuse(_Host(binds={"/work/known-good.yaml": path}))
    assert str(path) in message


def test_a_bound_config_directory_tree_is_scanned(tmp_path):
    (tmp_path / "starts" / "nested").mkdir(parents=True)
    path = config(tmp_path / "starts" / "nested", "start.yaml", devices="  - cpu")
    message = refuse(_Host(binds={"/inputs/starts": tmp_path / "starts"}))
    assert str(path) in message, "a bound directory is scanned recursively"


def test_a_cpu_or_fake_gpu_run_is_not_refused(tmp_path):
    path = config(tmp_path, "cand0-G-T.yaml", devices="  - cpu")
    startup._start_configs(cast("JudgeHost", _Host(readable=[path], gpus=False, cards=())))
    startup._start_configs(cast("JudgeHost", _Host(readable=[path], fake=True)))


def test_a_bound_interpreter_installation_is_not_scanned(tmp_path):
    """A conda prefix is bound so jobs can import stdlib; its YAML test fixtures are not staged configs."""
    install = tmp_path / "miniconda3"
    (install / "pkgs" / "tomlkit" / "info" / "test").mkdir(parents=True)
    config(install / "pkgs" / "tomlkit" / "info" / "test", "start.yaml", devices="  - cpu")
    startup._start_configs(cast("JudgeHost", _Host(readable=[install], qda_roots=[install])))
    # the same tree without the installation mark is an operator input, so it is refused
    message = refuse(_Host(readable=[install]))
    assert "devices: cpu puts gptq on CPU" in message, message


def test_a_pathologically_nested_yaml_is_skipped_not_fatal(tmp_path):
    """RecursionError from PyYAML's composer is an unreadable file, not a dead launch."""
    staged = tmp_path / "starts"
    staged.mkdir()
    (staged / "nested.yaml").write_text("a: " + "[" * 1000 + "]" * 1000 + "\n")
    bad = config(staged, "start.yaml", devices="  - cpu")
    message = refuse(_Host(readable=[staged]))
    assert str(bad) in message  # the readable config is still found beside the pathological one
    startup._start_configs(cast("JudgeHost", _Host(readable=[staged], qda_roots=[staged])))
