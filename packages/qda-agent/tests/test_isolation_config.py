"""`isolation.level`: the one knob, and what the legacy sandbox settings map to."""

import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from qda_agent.config import RunConfig

PACKAGE = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
#: The presets users copy, at the product default; every fixture states its level too (all at `none`).
CONFIGS = {p: "none" for p in sorted((PACKAGE / "configs").glob("*.yaml"))}
SHIPPED = CONFIGS | {p: "none" for p in sorted(FIXTURES.glob("*.yaml"))}


def build(tmp_path: Path, **fields: object) -> RunConfig:
    """A minimal run config, built directly: the legacy mapping's subject, with no level in a fixture file."""
    raw = {
        "model": "Qwen/Qwen3-8B",
        "format": "w4a16",
        "budget_gpu_hours": 0.01,
        "cards": [0],
        "run_dir": str(tmp_path / "run"),
        "contamination": "none",
        **fields,
    }
    return RunConfig.model_validate(raw)


def test_a_config_that_names_no_level_takes_the_product_default_or_the_legacy_pair(tmp_path):
    """Neither knob named is the product default (`none`); bwrap on both old sides is `strict`."""
    assert build(tmp_path).isolation.level == "none"
    assert build(tmp_path, sandbox="bwrap").isolation.level == "strict"
    assert build(tmp_path, sandbox="none").isolation.level == "none"


def test_one_bwrap_side_only_is_sandbox_and_warns(tmp_path):
    """A bwrap agent with plain jobs (`sandbox` it is), and an unconfined agent with bwrap jobs (`none`)."""
    with pytest.warns(DeprecationWarning, match="isolation.level is sandbox"):
        cfg = build(tmp_path, scheduler={"name": "qsched", "job_sandbox": "none"})
        assert cfg.isolation.level == "sandbox"
    with pytest.warns(DeprecationWarning, match="isolation.level is none"):
        cfg = build(tmp_path, sandbox="none", scheduler={"name": "qsched", "job_sandbox": "bwrap"})
        assert cfg.isolation.level == "none"


def test_an_explicit_level_wins(tmp_path):
    """A named level is never overwritten by the legacy settings, whatever they say."""
    cfg = build(
        tmp_path,
        sandbox="none",
        scheduler={"name": "qsched", "job_sandbox": "bwrap"},
        isolation={"level": "strict"},
    )
    assert cfg.isolation.level == "strict"
    assert cfg.sandbox.name == "none"  # the old settings stay as written in the record


def test_the_resolved_record_names_the_level(tmp_path):
    """`resolved.yaml` records the effective level, and loading it again gives the same one."""
    cfg = build(tmp_path, isolation={"level": "sandbox"})
    record = cfg.resolved()
    assert record["isolation"] == {"level": "sandbox", "tamper_check": "digest"}
    again = tmp_path / "resolved.yaml"
    again.write_text(yaml.safe_dump(record))
    assert RunConfig.load(again).isolation.level == "sandbox"


def test_an_unknown_level_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="level"):
        with warnings.catch_warnings():  # the mapping warning is not the subject here
            warnings.simplefilter("ignore")
            build(tmp_path, isolation={"level": "docker"})


def test_every_shipped_config_states_its_isolation_level():
    """A shipped config names its level, so the product default never decides what it runs under."""
    assert SHIPPED
    for path, level in SHIPPED.items():
        raw = yaml.safe_load(path.read_text())
        assert (raw.get("isolation") or {}).get("level") == level, path


def test_every_shipped_config_carries_the_isolation_note():
    """The note above `isolation:` is one text (`prepare.ISOLATION_NOTE`), the same in every config and in
    the one `prepare` writes: the default and how to leave it are stated once."""
    from qda_agent.prepare import ISOLATION_NOTE

    note = f"# {ISOLATION_NOTE}"
    assert CONFIGS
    for path in CONFIGS:
        lines = path.read_text().splitlines()
        (i,) = [i for i, line in enumerate(lines) if line.startswith("isolation:")]
        assert lines[i - 1] == note, path


def test_a_fixture_loads_at_the_level_it_names(tmp_path):
    for path in sorted(FIXTURES.glob("*.yaml")):
        assert RunConfig.load(path, {"run_dir": tmp_path / path.stem}).isolation.level == SHIPPED[path]


def test_a_copy_that_changes_a_legacy_knob_re_derives_the_level(tmp_path):
    """`model_copy` runs no validator, so changing `sandbox`/`scheduler` must not leave a stale level.

    A copy left naming the old level would run every check and every job under an isolation the copy no
    longer describes (`test_fenced_run_dir_not_under_tmp` found this). The config here names the legacy
    settings explicitly and leaves the level to them.
    """
    from qda_agent.sandbox_settings import NoSandbox

    cfg = build(tmp_path, sandbox="bwrap")  # bwrap on both old sides, no level: derived strict
    assert cfg.isolation.level == "strict"

    plain = cfg.model_copy(update={"sandbox": NoSandbox()})
    assert plain.isolation.level == "none"

    unconfined_jobs = cfg.model_copy(
        update={"scheduler": cfg.scheduler.model_copy(update={"job_sandbox": "none"})}
    )
    assert unconfined_jobs.isolation.level == "sandbox"


def test_a_level_chosen_by_a_copy_survives_the_next_copy(tmp_path):
    """A copy that supplies `isolation` names its level, so the next copy keeps it instead of re-deriving."""
    cfg = build(tmp_path, sandbox="bwrap")  # derived strict
    chosen = cfg.model_copy(update={"isolation": cfg.isolation.model_copy(update={"level": "sandbox"})})
    assert chosen.isolation.level == "sandbox"
    again = chosen.model_copy(update={"scheduler": chosen.scheduler.model_copy()})
    assert again.isolation.level == "sandbox"


def test_a_copy_keeps_a_level_the_run_named_itself(tmp_path):
    """Only a derived level follows the legacy knobs; a level the run named is the run's choice."""
    from qda_agent.sandbox_settings import NoSandbox

    chosen = build(tmp_path, sandbox="bwrap", isolation={"level": "sandbox"})  # the knobs would say strict
    assert chosen.model_copy(update={"sandbox": NoSandbox()}).isolation.level == "sandbox"
    same = build(tmp_path, sandbox="bwrap", isolation={"level": "strict"})  # named, and equal to the knobs
    assert same.model_copy(update={"sandbox": NoSandbox()}).isolation.level == "strict"


@pytest.mark.parametrize(
    ("level", "knob", "jobs"),
    [
        ("none", None, "none"),  # the product default: a judge build keeps the shell's PATH
        ("none", "bwrap", "none"),  # level none always runs plain jobs
        ("sandbox", None, "bwrap"),  # unset never takes a job out of the sandbox
        ("sandbox", "none", "none"),
        ("strict", None, "bwrap"),
        ("strict", "bwrap", "bwrap"),
    ],
)
def test_an_unset_job_sandbox_follows_the_level(tmp_path, level, knob, jobs):
    """One resolution for the jobs, the judge and the record: `isolation.api.job_sandbox(level, knob)`."""
    scheduler = {"name": "qsched"} | ({} if knob is None else {"job_sandbox": knob})
    cfg = build(tmp_path, isolation={"level": level}, scheduler=scheduler)
    assert cfg.job_sandbox() == jobs
    assert cfg.resolved()["scheduler"]["job_sandbox"] == jobs
