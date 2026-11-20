"""The pid record a run writes is the base's: same key, same place, same value, same probe.

The pid probe left config loading (B8): the level's own record fills `provenance.pid`. This pins all six
level x sandbox cases to what the config used to record - the key order of the written resolved.yaml, the
pid value, no extra key, and the probe only where the base probed (a bwrap sandbox at sandbox/strict; a run
at none, or a sandbox that is not bwrap, probes nothing).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml
from qda_agent import backends, cli
from qda_agent.config import RunConfig
from qda_agent.isolation import bwrap
from qda_agent.isolation.api import make

BACKEND = {"name": "claude", "model": "claude-opus-5-5", "effort": "high"}

#: The provenance keys a run writes, in the base's order.
BASE_KEYS = ["commit", "plugins", "pid", "contamination_skipped"]


def _config(tmp_path, level: str, sandbox: str) -> RunConfig:
    run = tmp_path / f"{level}-{sandbox}"
    fields: dict[str, Any] = {
        "contamination": "none",
        "model": "example/model",
        "format": "w4a16",
        "budget_gpu_hours": 1,
        "cards": [0],
        "run_dir": run,
        "agent": {"backend": BACKEND},
        "isolation": {"level": level},
        "sandbox": sandbox,
    }
    return RunConfig(**cast(Any, fields))


class _Stop(BaseException):
    """Stops `cli.execute` after it writes resolved.yaml and before any run starts."""


@pytest.mark.parametrize("level", ["none", "sandbox", "strict"])
@pytest.mark.parametrize("sandbox", ["none", "bwrap"])
def test_the_written_provenance_is_the_base_record(tmp_path, monkeypatch, level, sandbox):
    """The written keys and order are the base's; pid is the level's value; nothing extra is written."""
    # its subject is a later refusal: the run dir rule (`isolation.run_dir_in_tmp`) holds wherever tmp_path is
    monkeypatch.setattr("qda_agent.isolation.run_dir_in_tmp", lambda cfg: None)
    cfg = _config(tmp_path, level, sandbox)
    expected = make(cfg).provenance()["pid_form"]

    def stop(*args: object, **kwargs: object) -> None:
        raise _Stop

    monkeypatch.setattr(backends, "make_run", stop)
    with pytest.raises(_Stop):
        cli.execute(cfg)
    data = yaml.safe_load((cfg.run_dir / "resolved.yaml").read_text())
    assert list(data["provenance"]) == BASE_KEYS
    assert data["provenance"]["pid"] == expected
    assert "isolation" not in data["provenance"]


@pytest.mark.parametrize("level", ["none", "sandbox", "strict"])
@pytest.mark.parametrize("sandbox", ["none", "bwrap"])
def test_the_level_probes_bwrap_exactly_where_the_config_did(tmp_path, monkeypatch, level, sandbox):
    """A probe only for a bwrap sandbox at sandbox/strict; every other case records None without one."""
    cfg = _config(tmp_path, level, sandbox)
    probes: list[str] = []

    def counter(requested: str = "auto") -> str:
        probes.append(requested)
        return "shared"

    monkeypatch.setattr(bwrap, "pid_form", counter)
    form = make(cfg).provenance()["pid_form"]
    wanted = level != "none" and sandbox == "bwrap"
    assert probes == (["auto"] if wanted else [])
    assert form == ("shared" if wanted else None)
