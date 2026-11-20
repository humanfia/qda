import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from qda_agent.judge import JudgeHost, make_host
from qda_agent.judge.config import JudgeSettings
from qda_agent.rpc import Request
from qda_agent.sched import FakeBackend, FakeClock, FakeRun
from qda_kl.eval.suites import SUITES, Layout, Suite, SuiteResult
from safetensors_kit import quant_dir

TOOLS = Path(__file__).parents[1] / "src" / "qda_agent" / "judge" / "tools"


def judge_ref(root: Path) -> Path:
    """An eval+holdout reference: a search host derives its calibration index at start (old one-file shape)."""
    d = root / "ref"
    d.mkdir(parents=True, exist_ok=True)
    prompts = root / "kl-prompts.jsonl"
    prompts.write_text(
        json.dumps({"id": "r1", "prompt": "Which city is the capital of France"})
        + "\n"
        + json.dumps({"id": "h1", "prompt": "Name the longest river in Egypt", "split": "holdout"})
        + "\n"
    )
    (d / "manifest.json").write_text(json.dumps({"build": {"prompts": str(prompts)}}))
    return d


def ckpt(d: Path, serve: dict | None = None) -> Path:
    quant_dir(d)  # a real (tiny) safetensors checkpoint
    if serve is not None:
        (d / "build.json").write_text(json.dumps({"serve": serve}))
    return d


def bwrap_can_start() -> bool:
    """Whether this machine's bwrap can start a sandbox, as `qda doctor`'s sandbox line asks it.

    A host that disables user namespaces answers "Creating new namespace failed"; the host reads a job's
    checkpoint through exactly such a sandbox, so every test that hands one in needs this.
    """
    from qda_agent.doctor import check_bwrap
    from qda_agent.sched import SchedConfig

    return check_bwrap(SchedConfig(cards=[0], fake_gpus=True), level="sandbox").ok


def fake_qda_kl(root: Path, tag: str) -> Path:
    pkg = root / "qda_kl"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"TAG = {tag!r}\n")
    (pkg / "cli.py").write_text(
        "import json, os, sys, qda_kl\n"
        "a = sys.argv[1:]\n"
        "assert a[0] == 'score' and a[1].endswith('.yaml') and len(a) == 2, a\n"
        "c = json.load(open(a[1]))  # the judge writes JSON, which is YAML\n"
        "os.makedirs(c['out'])\n"
        "json.dump({'kl': 0.05, 'tokens': 125, 'domains': {'math': 0.04}, 'kl_text': 0.06,\n"
        "           'per_seq': {'p#0': {'kl': 9.99, 'tokens': 125, 'domain': 'math'}},\n"
        "           'tag': qda_kl.TAG, 'ref': c['reference'], 'file': qda_kl.__file__},\n"
        "          open(os.path.join(c['out'], 'score.json'), 'w'))\n"
    )
    return pkg


def test_frozen_evaluator_shadows_the_one_installed_in_the_serving_venv(tmp_path):
    venv = tmp_path / "serve-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    fake_qda_kl(site, "installed")
    live = fake_qda_kl(tmp_path / "src", "frozen")
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "job_sandbox": "none", "tick_s": 0.05},
        {
            "contamination": "none",
            "allow_any_format": True,
            "evaluator": str(live),
            "serve_python": str(venv / "bin" / "python"),
            "kl_ref": str(judge_ref(tmp_path)),
            "kl_holdout_ref": str(judge_ref(tmp_path)),
            "check_cmd": None,
            "job_env": {"QDA_TEST_JOB_ENV": "1"},
            "qda_cmd": [sys.executable],
        },
    )
    host.start()
    try:
        (live / "__init__.py").write_text("TAG = 'edited'\n")
        ckpt(run / "work" / "a")
        assert host.handle(Request("kl", ["a"], str(run / "work"))).exit == 0
        assert host.drain(30)
        got = json.loads((run / "judge" / "results" / "k1" / "score.json").read_text())
        assert got["tag"] == "frozen" and got["ref"] == str(judge_ref(tmp_path))
        assert Path(got["file"]).is_relative_to(run / "judge" / "eval")
        out = host.handle(Request("kl", ["show", "k1"], str(run / "work"))).stdout
        assert "KL 0.05" in out and "9.99" not in out and "per_seq" not in out
        rec = json.loads((run / "judge" / "judge.json").read_text())
        assert len(rec["tools_sha256"]) == 64 and (Path(rec["tools"]) / "serve_check.py").is_file()
    finally:
        host.stop()


def test_job_env_and_failed_reading(tmp_path):
    clock = FakeClock()

    def writes_then_fails(argv):  # a score.json left behind by a scorer that exits nonzero is no result
        out = Path(json.loads(Path(argv[argv.index("score") + 1]).read_text())["out"])
        out.mkdir(parents=True)
        (out / "score.json").write_text(json.dumps({"kl": 0.01}))

    be = FakeBackend(clock, default=FakeRun(10, rc=1, effect=writes_then_fails))
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "fake_gpus": True},
        {
            "contamination": "none",
            "kl_ref": str(judge_ref(run.parent)),
            "kl_holdout_ref": str(judge_ref(run.parent)),
            "serve_python": "/venv/bin/python",
            "job_env": {"CUDA_HOME": "/cuda"},
        },
        clock,
        backend=be,
        autotick=False,
    )
    ckpt(run / "work" / "a")
    host.handle(Request("kl", ["a"], str(run / "work")))
    host.tick()
    (name,) = be.argvs
    assert (
        be.argvs[name][:3] == ["/venv/bin/python", "-m", "qda_kl.cli"]
        and be.envs[name]["CUDA_HOME"] == "/cuda"
    )
    clock.advance(10)
    host.tick()
    (e,) = host.handins()
    assert e.valid is False and not e.metrics and "failed" in e.reason


def suite_host(run: Path, results: dict[str, object]):
    """A host whose score command writes `results[<checkpoint's source name>]`, ranking by `compactness`."""
    clock = FakeClock()

    def writes(argv):
        Path(argv[-1]).write_text(json.dumps(results[(Path(argv[-2]) / "README.md").read_text()]))

    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "fake_gpus": True},
        {
            "contamination": "none",
            "allow_any_format": True,
            "metric": "compactness",
            "final_rules": ["best_metric"],  # a scored build is the pick (the default rule takes hand-ins)
            "kl_cmd": ["score", "{ckpt}", "{out}"],
            "kl_ref": str(judge_ref(run.parent)),
            "kl_holdout_ref": str(judge_ref(run.parent)),
        },
        clock,
        backend=FakeBackend(clock, default=FakeRun(10, effect=writes)),
        autotick=False,
    )

    def read(*names: str) -> None:
        for name in names:
            d = ckpt(run / "work" / name)
            (d / "README.md").write_text(name)  # an inert record the judge's copy keeps
            assert host.handle(Request("kl", [name], str(run / "work"))).exit == 0, name
            host.tick()
            clock.advance(10)
            host.tick()

    return host, read


def test_a_suite_result_is_ranked_by_the_named_metric_and_its_stated_direction(tmp_path):
    def result(value, **also):
        return {
            "suite": "compactness",
            "metrics": {"compactness": value},
            "higher_is_better": {"compactness": True},
            "detail": {"hidden": 9.99},
            **also,
        }

    run = tmp_path / "run"
    host, read = suite_host(
        run,
        {
            "low": result(2.0),
            "high": result(3.0),
            "undirected": result(9.0, higher_is_better={}),
            "malformed": result("9"),
            "nan": result(float("nan")),
            # a complete reading that does not hold the named metric
            "other": {"kl": 0.01, "tokens": 100, "per_seq": {"p#0": {"kl": 1.0, "tokens": 100}}},
        },
    )
    read("low", "high", "undirected", "malformed", "nan", "other")
    said = host.handle(Request("kl", ["list"], str(run / "work"))).stdout.splitlines()
    assert said[:2] == [
        "k1 low: valid: compactness 2 (higher is better)",
        "k2 high: valid: compactness 3 (higher is better)",
    ]
    assert said[2] == "k3 undirected: refused: the reading does not state the direction of compactness"
    assert all(line.endswith("refused: no finite overall compactness") for line in said[3:])
    final = host.final()
    assert final.rule == "best_metric" and final.choice.dir.name == "k2"
    assert final.reason.endswith("the highest compactness (3)")


def _judge_env(tmp_path: Path, job_env: dict | None = None) -> tuple[Path, dict[str, str], JudgeHost]:
    """The environment of a fake judge job's command, from a deep run dir that vLLM's IPC path outgrows."""
    clock = FakeClock()
    be = FakeBackend(clock)
    run = tmp_path / ("deep-run-directory-" * 5) / "run"
    host = make_host(
        run,
        {"cards": [0], "budget_s": 3600, "fake_gpus": True},
        {
            "contamination": "none",
            "kl_ref": str(judge_ref(tmp_path)),
            "kl_holdout_ref": str(judge_ref(tmp_path)),
            "serve_python": "/venv/bin/python",
            "job_env": job_env or {},
        },
        clock,
        backend=be,
        autotick=False,
    )
    ckpt(run / "work" / "a")
    host.handle(Request("kl", ["a"], str(run / "work")))
    host.tick()
    (name,) = be.argvs
    return run, be.envs[name], host


SYMM = f"/symm_mem-{'0' * 20}"  # torch's symmetric-memory socket name under $TMPDIR (a multi-card job)


def test_a_judge_job_binds_its_sockets_through_a_short_absolute_name(tmp_path, monkeypatch):
    """Sockets are bound under `$TMPDIR` (torch's `symm_mem-<id>`) and the vLLM IPC base; the kernel caps a
    socket path at 107 bytes, and a 113-byte temp dir refused a 2-card gencheck at engine start. The job names
    its temp dir by a short link, absolute and fixed when the job is queued."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)  # the private base is /tmp
    run, env, host = _judge_env(tmp_path)
    short = env["TMPDIR"]
    assert env["TMP"] == env["TEMP"] == env["VLLM_RPC_BASE_PATH"] == short
    real = run / "judge" / "tmp" / "k1"
    assert Path(short).is_absolute() and Path(short).is_symlink()
    assert Path(os.path.realpath(short)) == real.resolve() and real.is_dir()  # outside the agent's fence
    assert len(os.fsencode(str(real))) >= 113  # as deep as the run dir whose 2-card gencheck failed
    assert len(os.fsencode(f"{real}{SYMM}")) > 107
    assert len(os.fsencode(f"{short}{SYMM}")) <= 107
    assert len(os.fsencode(f"{short}/{'0' * 36}")) <= 107  # vLLM appends a UUID to its base
    base = Path(short).parent
    assert base.name.startswith("qda-jt-") and not base.stat().st_mode & 0o077
    host.stop()
    assert not base.exists()  # the links leave with the host; the temp dir stays in the judge's dir
    assert real.is_dir()


def test_an_ended_jobs_short_name_is_removed(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    _, env, host = _judge_env(tmp_path)
    try:
        short = Path(env["TMPDIR"])
        assert short.is_symlink()
        host.sched.cancel("judge-k1")
        host.tick()
        assert not short.exists() and not short.is_symlink()
    finally:
        host.stop()


def test_job_temps_hold_under_two_threads(tmp_path, monkeypatch):
    """The final rule's check makes and releases its link outside the host's lock while the tick thread
    folds ended jobs: neither may see the other's half-made state (a dict changed during iteration, a
    link number taken twice)."""
    import threading
    from types import SimpleNamespace

    from qda_agent.judge.jobtmp import JobTemps

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as the interpreter allows
    temps = JobTemps(tmp_path / "judge")
    made: list[str] = []
    ended = SimpleNamespace(terminal=True)
    sched = SimpleNamespace(list=lambda origin: [SimpleNamespace(name=j, state=ended) for j in list(made)])
    errors: list[BaseException] = []
    done = threading.Event()

    def maker() -> None:
        try:
            for i in range(400):
                job = f"judge-k{i}"
                made.append(job)  # "ended" at once: the folder may release it while it is being made
                temps.make(f"k{i}", job)
                if i % 3 == 0:
                    temps.release(job)
        except BaseException as e:  # noqa: BLE001 -- reported below
            errors.append(e)
        finally:
            done.set()

    def folder() -> None:
        try:
            while not done.is_set():
                temps.release_ended(sched)
        except BaseException as e:  # noqa: BLE001 -- reported below
            errors.append(e)

    threads = [threading.Thread(target=maker), threading.Thread(target=folder)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        temps.release_ended(sched)
        assert temps.count == 400 and temps.links == {}
        assert temps.base is not None and not any(temps.base.iterdir())  # every link removed, none twice
        assert all((tmp_path / "judge" / "tmp" / f"k{i}").is_dir() for i in range(400))
    finally:
        sys.setswitchinterval(interval)
        temps.close()


CHILD = """
import os, socket, sys, tempfile
os.chdir(sys.argv[1])  # as ninja does before it runs nvcc in a build dir
fd, path = tempfile.mkstemp(prefix="tmpxft_")
os.write(fd, b"x")
os.close(fd)
s = socket.socket(socket.AF_UNIX)
s.bind(os.environ["TMPDIR"] + sys.argv[2])  # torch's symmetric memory binds here (2-card gencheck)
s.close()
print(path)
"""


def test_a_child_that_changes_directory_writes_under_the_jobs_temp_dir(tmp_path, monkeypatch):
    """The job's temp dir does not depend on any process's cwd: a child that `chdir`s (ninja running nvcc for
    FlashInfer's JIT) still writes its temp files, and binds its sockets, in the job's own temp dir."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    run, env, host = _judge_env(tmp_path)
    try:
        elsewhere = tmp_path / "build"
        elsewhere.mkdir()
        (elsewhere / "tmp" / "k1").mkdir(parents=True)  # where a cwd-relative name would have led
        job = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "TMP", "TEMP")}
        out = subprocess.run(
            [sys.executable, "-c", CHILD, str(elsewhere), SYMM],
            cwd=run / "judge",  # the job's cwd
            env=job | {k: env[k] for k in ("TMPDIR", "TMP", "TEMP")},
            capture_output=True,
            text=True,
            check=True,
        )
        written = Path(os.path.realpath(out.stdout.strip()))
        real = (run / "judge" / "tmp" / "k1").resolve()
        assert written.parent == real and written.read_bytes() == b"x"
        assert (real / SYMM.lstrip("/")).is_socket()
        assert not any((elsewhere / "tmp" / "k1").iterdir())
    finally:
        host.stop()


def test_a_configured_vllm_rpc_base_path_wins(tmp_path):
    _, env, host = _judge_env(tmp_path, {"VLLM_RPC_BASE_PATH": "/tmp/short"})
    host.stop()
    assert env["VLLM_RPC_BASE_PATH"] == "/tmp/short"


def test_a_judge_job_keeps_the_hosts_cache_locations(tmp_path, monkeypatch):
    """A cache location the user exported reaches every judge job unchanged (the job inherits the host's env)."""
    from qda_quant.output.vllm_env import CACHE_ENV

    for name in CACHE_ENV:
        monkeypatch.setenv(name, f"/node-local/{name}")
    _, env, host = _judge_env(tmp_path)
    host.stop()
    assert {name: env[name] for name in CACHE_ENV} == {name: f"/node-local/{name}" for name in CACHE_ENV}


@pytest.mark.skipif(not bwrap_can_start(), reason="this machine's bwrap cannot start a sandbox")
def test_a_judge_job_caches_node_locally_never_under_home_or_the_run(tmp_path, monkeypatch):
    """With nothing exported, vLLM, Triton and FlashInfer would compile under `$HOME`, and the run dir is often
    on a network filesystem whose lock two first compiles cannot both hold: every judge job caches in the
    node-local directory instead (`cache_root`). A location `judge.job_env` names still wins."""
    from qda_quant.output.vllm_env import CACHE_ENV, cache_root

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for name in (*CACHE_ENV, "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    run, env, host = _judge_env(tmp_path, {"TRITON_CACHE_DIR": "/configured/triton"})
    try:
        assert env["TRITON_CACHE_DIR"] == "/configured/triton"
        for name in CACHE_ENV - {"TRITON_CACHE_DIR"}:
            assert Path(env[name]).parent == cache_root(), name
            assert not Path(env[name]).is_relative_to(home)
            assert not Path(env[name]).is_relative_to(run)
    finally:
        host.stop()


def test_job_env_pythonpath_comes_after_the_evaluator_copy():
    # `qda-kl score CONFIG.yaml` imports qda_quant: a serving venv without it gets qda-quant's sources from job_env
    from qda_agent.judge import JudgeConfig

    cfg = JudgeConfig(
        contamination="none",
        allow_any_format=True,
        kl_ref="ref",
        job_env={"PYTHONPATH": "/qq/src", "CUDA_HOME": "/c"},
    )
    from qda_quant.output.vllm_env import cache_env

    off = {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}  # the judge's engines have Machete off too
    assert cfg.job_environ(Path("/run/judge/eval")) == cache_env(
        {"PYTHONPATH": "/run/judge/eval:/qq/src", "CUDA_HOME": "/c", **off}
    )
    assert cfg.job_environ(None) == cache_env({"PYTHONPATH": "/qq/src", "CUDA_HOME": "/c", **off})
    plain = JudgeConfig(contamination="none", allow_any_format=True, kl_ref="ref").job_environ(Path("/e"))
    assert plain == cache_env({"PYTHONPATH": "/e", **off})


class _Size(Suite):
    name: ClassVar[str] = "judge_tools_size"

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        del checkpoint, work
        return SuiteResult(suite=self.name, metrics={"size": len(reference)}, higher_is_better={"size": True})


class _SizeParts(_Size):
    name: ClassVar[str] = "judge_tools_size_parts"

    def parts(self) -> list[Suite]:
        return [_SizePart()]


class _SizePart(_SizeParts):
    def parts(self) -> list[Suite]:
        return [self]

    def part_label(self) -> str | None:
        return "one"


class _Wide(Layout):
    name: ClassVar[str] = "judge_tools_wide"


def _registered(*suites: type[Suite]) -> None:
    for suite in suites:
        if suite.name not in SUITES:
            SUITES.register(suite.name, suite)


def test_a_suites_readings_take_the_cards_it_declares(tmp_path):
    _registered(_Wide)
    wide = {_Wide.name: {"tp": 2}}
    cmd = ["score", "{suite}", "{ckpt}", "{out}"]
    clock = FakeClock()
    backend = FakeBackend(clock, default=FakeRun(10))
    run = tmp_path / "run"
    host = make_host(
        run,
        {"cards": [0, 1, 2], "budget_s": 3600, "fake_gpus": True},
        {
            "contamination": "none",
            "allow_any_format": True,
            "suite": wide,
            "kl_cmd": cmd,
            "kl_ref": str(judge_ref(run.parent)),
            "kl_holdout_ref": str(judge_ref(run.parent)),
        },
        clock,
        backend=backend,
        autotick=False,
    )
    ckpt(run / "work" / "a")
    assert host.handle(Request("kl", ["a"], str(run / "work"))).exit == 0
    host.tick()
    (entry,) = host.rows("scored")
    assert host.sched.get(entry.job).ncards == 2
    (argv,) = backend.argvs.values()
    assert json.loads(argv[1]) == {
        _Wide.name: {"tp": 2, "serve_env": {}, "serve_args": {}, "nccl_nvls": False}
    }  # with its defaults, as recorded
    for bad, said in (
        ({"suite": wide}, "go together"),  # the default command would not run it
        ({"kl_cmd": cmd}, "go together"),  # a slot with nothing to fill it
        ({"suite": wide, "kl_cmd": cmd, "kl_layout": {"tp": 4}}, "takes the cards"),
        ({"suite": {"judge_tools_unknown": {}}, "kl_cmd": cmd}, "unknown suite"),
        ({"suite": {_Wide.name: {"tp": 0}}, "kl_cmd": cmd}, "tp"),
    ):
        with pytest.raises(ValueError, match=said):
            JudgeSettings.model_validate(bad)


def test_resolved_records_a_suite_and_its_nested_dataset_plugins(tmp_path, monkeypatch):
    """A third-party suite, and its nested dataset plugin, are recorded with their package and version."""
    from typing import Any

    from pydantic import Field, field_serializer, field_validator
    from qda_agent.config import RunConfig
    from qda_kl.datasets import DATASETS, Dataset, parse_dataset

    class Plain(Suite):
        name: ClassVar[str] = "provenance_third_party"

    class SuiteDataset(Dataset):
        name: ClassVar[str] = "provenance_dataset"
        public: ClassVar[bool] = True

        def local(self) -> Path:
            return tmp_path / "data"

    class ThirdParty(Suite):
        name: ClassVar[str] = "provenance_suite_with_data"
        datasets: list[Dataset] = Field(default_factory=list)

        @field_validator("datasets", mode="before")
        @classmethod
        def _named(cls, value: Any) -> Any:
            return [parse_dataset(d, f"datasets.{i}") for i, d in enumerate(value)]

        @field_serializer("datasets")
        def _keyed(self, datasets: list[Dataset]) -> list[dict[str, Any]]:
            return [d.to_dict() for d in datasets]

    (tmp_path / "data").mkdir()
    list(DATASETS)
    monkeypatch.setitem(DATASETS._entries, SuiteDataset.name, SuiteDataset)
    dataset_provenance = DATASETS.provenance
    monkeypatch.setattr(
        DATASETS,
        "provenance",
        lambda key: ("thirdparty-datasets", "1.4.0") if key == SuiteDataset.name else dataset_provenance(key),
    )
    _registered(Plain, ThirdParty)
    suite_provenance = SUITES.provenance
    monkeypatch.setattr(
        SUITES,
        "provenance",
        lambda key: (
            ("thirdparty-suite", "2.3.1") if key in {Plain.name, ThirdParty.name} else suite_provenance(key)
        ),
    )

    def record(suite: dict) -> dict:
        cfg = RunConfig.model_validate(
            {
                "model": "tiny/model",
                "format": "w4a16",
                "budget_gpu_hours": 1,
                "contamination": "none",
                "cards": [0, 1],
                "run_dir": str(tmp_path / "run"),
                "judge": {"suite": suite, "kl_cmd": ["score", "{suite}", "{ckpt}", "{out}"]},
            }
        )
        return cfg.resolved()

    # a suite with no datasets is recorded alone: nothing nested appears
    plain = record({Plain.name: {}})["provenance"]["plugins"]
    assert plain["judge.suite"] == {"name": Plain.name, "package": "thirdparty-suite", "version": "2.3.1"}
    assert not [key for key in plain if key.startswith("judge.suite.")]
    # a suite whose datasets are plugins of their own: both are recorded
    suite = {ThirdParty.name: {"datasets": [{SuiteDataset.name: {}}]}}
    got = record(suite)
    assert got["provenance"]["plugins"]["judge.suite"] == {
        "name": ThirdParty.name,
        "package": "thirdparty-suite",
        "version": "2.3.1",
    }
    assert got["provenance"]["plugins"]["judge.suite.datasets.0"] == {
        "name": SuiteDataset.name,
        "package": "thirdparty-datasets",
        "version": "1.4.0",
    }
    # the resolved settings still hold the nested selection, unchanged by the record
    assert got["judge"]["suite"] == suite


def test_suite_job_writes_the_named_suites_result_where_the_judge_reads_it(tmp_path):
    _registered(_Size, _SizeParts)
    spec = importlib.util.spec_from_file_location("suite_job", TOOLS / "suite_job.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    ck, ref, out = tmp_path / "ck", tmp_path / "ref", tmp_path / "results" / "k1.json"
    ck.mkdir()
    ref.mkdir()
    out.parent.mkdir()

    def argv(suite: str) -> list[str]:
        return [json.dumps({suite: {}}), str(ck), str(ref), str(out)]

    assert job.main(argv("judge_tools_unknown")) == 2 and not out.exists()
    assert job.main(argv(_SizeParts.name)) == 2 and not out.exists()
    assert job.main(argv(_Size.name)) == 0
    got = SuiteResult.model_validate_json(out.read_text())
    assert got.suite == _Size.name and got.metrics == {"size": len(str(ref))} and got.higher_is_better["size"]
