"""`isolation.level: strict`: today's wall as a level, one refusal earlier and nothing else."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from qda_agent.config import RunConfig
from qda_agent.isolation import bwrap as bwrap_mod
from qda_agent.isolation import wall
from qda_agent.isolation.api import make
from qda_agent.isolation.levels import legacy, strict
from qda_agent.isolation.levels.legacy import LegacyIsolation


def _cfg(
    tmp_path: Path, *, pid: str = "auto", sandbox: str = "bwrap", cards: tuple[int, ...] = ()
) -> RunConfig:
    """The attributes the level reads, as a run config has them (today's preflight is patched out)."""
    return cast(
        RunConfig,
        SimpleNamespace(
            sandbox=SimpleNamespace(name=sandbox, pid=pid, hide=[]),
            isolation=SimpleNamespace(level="strict"),
            scheduler=SimpleNamespace(job_sandbox="bwrap", gpus=True),
            run_dir=tmp_path,
            run=SimpleNamespace(overlay=None),
            cards=list(cards),
            read_paths=lambda: ([], {}),
        ),
    )


@pytest.fixture
def today_passes(monkeypatch):
    """Today's preflight passes: these tests are about the one rule this level adds."""
    monkeypatch.setattr(LegacyIsolation, "preflight", lambda self, cfg: None)


def test_make_builds_the_strict_level(tmp_path):
    iso = make(_cfg(tmp_path))
    assert isinstance(iso, strict.StrictIsolation)
    assert isinstance(iso, LegacyIsolation)  # strict is today's behaviour by construction
    assert iso.level == "strict"


def test_the_preflight_runs_todays_first_then_the_walls_rule(tmp_path, monkeypatch):
    """The wall's rule is asked second, so the static refusals keep today's priority and sentence."""
    asked: list[str] = []
    monkeypatch.setattr(LegacyIsolation, "preflight", lambda self, cfg: asked.append("today"))
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "shared")
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: "other running job(s) of this user: 9")
    cfg = _cfg(tmp_path)
    with pytest.raises(wall.WallRefused):
        strict.StrictIsolation.for_config(cfg).preflight(cfg)
    assert asked == ["today"]


def test_the_preflight_refuses_a_shared_pid_with_the_walls_own_sentence(tmp_path, monkeypatch, today_passes):
    """Today's wall refuses `pid: shared` on a node it does not have; the preflight must say the same."""
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "shared")
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: "other running job(s) of this user: 9")
    cfg = _cfg(tmp_path)
    with pytest.raises(wall.WallRefused) as from_preflight:
        strict.StrictIsolation.for_config(cfg).preflight(cfg)
    with pytest.raises(wall.WallRefused) as from_wall, wall.gpu_wall(cfg):
        pass
    assert str(from_preflight.value) == str(from_wall.value)
    assert "agent's process" in str(from_preflight.value)


def test_the_preflight_passes_a_private_pid_or_a_dedicated_node(tmp_path, monkeypatch, today_passes):
    cfg = _cfg(tmp_path)
    iso = strict.StrictIsolation.for_config(cfg)
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: "other running job(s) of this user: 9")
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "private")
    iso.preflight(cfg)
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "shared")
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: None)
    iso.preflight(cfg)


def test_no_wall_no_wall_rule(tmp_path, monkeypatch, today_passes):
    """A config that names no wall puts none up today, so the wall's rule is not asked (and no pid is read)."""
    cfg = _cfg(tmp_path, sandbox="none")
    monkeypatch.setattr(wall, "pid_form", lambda requested="auto": "shared")
    monkeypatch.setattr(wall, "why_not_dedicated", lambda: "other running job(s) of this user: 9")
    strict.StrictIsolation.for_config(cfg).preflight(cfg)


def test_the_provenance_carries_the_interface_keys(tmp_path, monkeypatch):
    """`isolation.api`'s callers read `pid_form` and `uuids`; the level inherits today's record."""
    monkeypatch.setattr(bwrap_mod, "pid_form", lambda requested="auto": "shared")
    iso = strict.StrictIsolation.for_config(_cfg(tmp_path, cards=(0, 1)))
    monkeypatch.setattr(legacy, "devices", SimpleNamespace(card_uuids=lambda cards: ["gpu-0", "gpu-1"]))
    assert iso.provenance() == {
        "level": "strict",
        "pid_form": "shared",
        "uuids": {0: "gpu-0", 1: "gpu-1"},
    }
