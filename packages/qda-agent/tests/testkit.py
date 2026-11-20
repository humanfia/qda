"""CPU fixtures for the search flow and its host boundary.

- `FakeHost`: an in-memory host with the real lifecycle; hand-ins and KL scores go straight into its
  ledger. Its pick follows the judge's rule; the real rule lives in `qda_agent.judge`.
- `ScriptedWorker`: a Humanize `FakeAgentDriver` reply that plays a fixed script, one step per turn,
  then goes idle (as an agent that thinks its goal is met does).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from qda_agent.agents import BACKENDS, Backend
from qda_agent.config import RunConfig
from qda_agent.host import Final, HandIn
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.final_rules import apply

__all__ = [
    "FakeHost",
    "ScriptedWorker",
    "SelectingHost",
    "fake_host",
    "sandbox_helpers",
    "scripted",
    "selecting_host",
    "shell_worker",
]


@pytest.fixture
def sandbox_helpers(request):
    """Start test commands under the turn guard and reap them at teardown, including after Ctrl+C.

    Direct ``live_state`` probes have no pid namespace to end their forked relay. The production guard
    owns those children; the fixture stops and waits for the guard even when a test never waits itself.
    """
    from qda_agent.isolation import wall

    def start(argv, **kwargs):
        process = subprocess.Popen(
            [*wall.LIVE_PYTHON, wall.TURN_SOURCE, "--", *argv], start_new_session=True, **kwargs
        )

        def stop():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

        request.addfinalizer(stop)
        return process

    return start


Step = Callable[[], Awaitable[str] | str]

#: The live fake hosts by run dir, so a scripted worker can reach the one its flow made.
HOSTS: dict[Path, FakeHost] = {}


class ScriptedWorker:
    """Play one scripted step per turn, then idle when the script ends."""

    def __init__(self, steps: Sequence[Step | BaseException] = (), idle_s: float = 0.1) -> None:
        self.steps, self.idle_s = list(steps), idle_s
        self.turns = 0

    async def __call__(self, prompt: str, **_: object) -> str:
        """Play one turn."""
        del prompt
        self.turns += 1
        if not self.steps:
            await asyncio.sleep(self.idle_s)
            return "Done: the best checkpoint is submitted."
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        said = step()
        return await said if asyncio.iscoroutine(said) else str(said)


BUILD_SH = """#!/bin/bash
# A stand-in build: writes a tiny safetensors checkpoint the judge can freeze and check.
set -e
mkdir -p "$1"
echo '{"model_type": "qwen3"}' > "$1/config.json"
python3 - "$1/model.safetensors" "$2" <<'PY'
import json, struct, sys
data = sys.argv[2].encode().ljust(8, b"\\0")[:8]
h = json.dumps({"w": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode()
h += b" " * (-len(h) % 8)
open(sys.argv[1], "wb").write(struct.pack("<Q", len(h)) + h + data)
PY
echo '{}' > "$1/tokenizer.json"  # last: the scripted worker waits on this file, then reads the weights
"""


def shell_worker(cfg: RunConfig, wait_s: float = 20.0, steps: int | None = None) -> ScriptedWorker:
    """Create a worker that exercises the installed commands through the real host.

    Its turns wait without holding the event loop, as a real agent's turn does, so a signal or the deadline
    reaches the flow while a build waits on the queue. `steps`: play only the first ones.
    """
    work = cfg.workdir

    async def sh(name: str, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            str(work / "bin" / name), *args, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        out = (await process.communicate())[0].decode()
        return f"$ {name} {' '.join(args)} -> {process.returncode}\n{out}"

    async def build(name: str) -> str:
        (work / "scripts" / "build.sh").write_text(BUILD_SH)
        (work / "scripts" / f"build-{name}.sh").write_text(f"bash scripts/build.sh out/{name} {name}\n")
        said = await sh(
            "qsched", "add", "--expected", "5s", "--name", f"build-{name}", f"scripts/build-{name}.sh"
        )
        deadline = time.monotonic() + wait_s
        while not (work / "out" / name / "tokenizer.json").exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return said + await sh("kl", f"out/{name}")

    async def handed_in(name: str) -> str:
        return await build(name) + await sh("handin", f"out/{name}")

    async def rejected_handin() -> str:
        said = await build("c")
        if not (work / "out" / "c").is_dir():
            return said
        (work / "out" / "c" / "extra").symlink_to(work / "TASK.md")
        return said + await sh("handin", "out/c")

    script: list[Step] = [
        lambda: build("a"),
        lambda: handed_in("b"),
        rejected_handin,
        lambda: sh("kl", "list"),
    ]
    return ScriptedWorker(script[:steps])


class FakeHost:
    """An in-memory host: lifecycle calls are recorded, the ledger is filled by `submit` / `score`."""

    def __init__(
        self,
        run_dir: Path,
        sched: dict[str, Any],
        judge: dict[str, Any],
        *,
        isolation: object | None = None,
        run_config: object | None = None,
    ) -> None:
        self.run_dir, self.sched, self.judge = run_dir, sched, judge
        self.iso, self.run_config = isolation, run_config
        self.calls: list[str] = []
        self.ledger: list[HandIn] = []
        self.lists = 0  # one `handin` call is one ordered list
        self.pending = 0
        self.open = False
        #: When the disk went below the run's free-space floor, as the real judge records it; a test sets it.
        self.below_floor_since: float | None = None
        self._lock = threading.Lock()
        HOSTS[run_dir] = self

    def start(self) -> None:
        """Open the ledger."""
        self.calls.append("start")
        self.open = True

    def cutoff(self) -> None:
        """Close the ledger to the agent."""
        self.calls.append("cutoff")
        self.open = False

    def drain(self, timeout_s: float) -> bool:
        """Finish pending checks (each passes)."""
        self.calls.append(f"drain({timeout_s:g})")
        with self._lock:
            self.pending = 0
            for h in self.ledger:
                if h.valid is None:  # a check that finished during the drain
                    h.valid = True
        return True

    def _add(
        self,
        d: Path,
        kind: Literal["handin", "scored"],
        valid: bool | None,
        kl: float | None,
        reason: str,
        list_id: int | None = None,
        slot: int = 0,
    ) -> str:
        with self._lock:
            if not self.open:
                return "refused: the search is over"
            read = {} if kl is None else {"kl": kl}
            self.ledger.append(
                HandIn(
                    dir=d,
                    submitted_at=datetime.now(UTC),
                    kind=kind,
                    valid=valid,
                    metrics=read,
                    higher_is_better=dict.fromkeys(read, False),
                    reason=reason,
                    handin_list=list_id,
                    slot=slot,
                )
            )
            return "ok"

    def _list(self) -> int:
        with self._lock:
            self.lists += 1  # one `handin` call is one ordered list
            return self.lists

    def submit(self, d: Path, *, valid: bool | None = True, kl: float | None = None, reason: str = "") -> str:
        """Record a one-directory hand-in (`valid=None`: its check is still running)."""
        return self._add(d, "handin", valid, kl, reason, self._list(), 0)

    def submit_list(self, entries: list[tuple[Path, bool | None, float | None, str]]) -> list[str]:
        """Record one `handin DIR [DIR...]` call: `(dir, valid, kl, reason)` per directory, best first."""
        list_id = self._list()
        return [
            self._add(d, "handin", valid, kl, reason, list_id, slot)
            for slot, (d, valid, kl, reason) in enumerate(entries)
        ]

    def score(self, d: Path, kl: float, *, valid: bool = True) -> str:
        """Record a KL reading of a checkpoint that was not handed in."""
        return self._add(d, "scored", valid, kl, "")

    def handins(self) -> list[HandIn]:
        """The ledger, oldest first."""
        return list(self.ledger)

    def final(self) -> Final:
        """The judge's rules (`judge.final_rules`), applied to the ledger; none named: the default `handin`."""
        baselines = [Path(b) for b in self.judge.get("baselines") or []]
        candidate0_kl = self.judge.get("candidate0_kl")
        settings = JudgeSettings(
            final_rules=self.judge.get("final_rules", ["handin"]),
            candidate0_kl=candidate0_kl,
        )
        metric = self.judge.get("metric", "kl")
        # the standing-in host records the two facts the real host's `final` records
        return apply(settings.final_rules, self.ledger, baselines, metric, candidate0_kl).with_delivery(
            metric
        )

    def write_final(self, path: Path, final: Final | None = None) -> None:
        """Write the selected final record, or `final()`, to `path`."""
        self.calls.append("write_final")
        final = final if final is not None else self.final()
        path.write_text(final.model_dump_json(indent=2))

    def stop(self) -> None:
        """Forget this host."""
        self.calls.append("stop")
        HOSTS.pop(self.run_dir, None)


def fake_host(
    run_dir: Path,
    sched: dict[str, Any],
    judge: dict[str, Any],
    *,
    isolation: object | None = None,
    run_config: object | None = None,
) -> FakeHost:
    """Host factory for `host: testkit:fake_host` in a run config."""
    return FakeHost(run_dir, sched, judge, isolation=isolation, run_config=run_config)


class SelectingHost(FakeHost):
    """A host that also reads candidate 0 at start (`judge.select_candidate0`).

    The flow reads its `baseline_readings` right after `start` to pick the lowest-KL baseline and rewrite
    TASK.md; this host gives one reading per configured baseline, in KL order.
    """

    def final(self) -> Final:
        got = super().final()
        readings = [
            HandIn(
                dir=Path(b),
                submitted_at=datetime.now(UTC),
                kind="scored",
                valid=True,
                metrics={"kl": 0.1 * (i + 1)},
                reason="baseline",
            )
            for i, b in enumerate(self.judge.get("baselines") or [])
        ]
        return got.model_copy(update={"baseline_readings": readings})


def selecting_host(
    run_dir: Path,
    sched: dict[str, Any],
    judge: dict[str, Any],
    *,
    isolation: object | None = None,
    run_config: object | None = None,
) -> SelectingHost:
    """Host factory for `host: testkit:selecting_host` in a run config.

    The flow calls every host factory with its isolation and the run config as keywords, as `make_host` takes
    them; this host stores both and uses neither.
    """
    return SelectingHost(run_dir, sched, judge, isolation=isolation, run_config=run_config)


def build_ckpt(workdir: Path, name: str) -> Path:
    """A stand-in checkpoint directory, as a build job would leave it."""
    d = workdir / "out" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"quantization_config": {"format": "pack-quantized"}}))
    return d


def _flow_worker(cfg: RunConfig) -> ScriptedWorker:
    """Check scoring, submission and final selection through the in-memory host."""
    run, work = cfg.run_dir, cfg.workdir

    def step(name: str, kl: float, handin: bool = False, valid: bool = True) -> Step:
        def go() -> str:
            d = build_ckpt(work, name)
            host = HOSTS[run]
            host.score(d, kl)
            if handin:
                host.submit(d, valid=valid, kl=kl, reason="" if valid else "does not load")
            return f"{name}: KL {kl}"

        return go

    return ScriptedWorker(
        [step("a", 0.09), step("b", 0.07, handin=True), step("c", 0.05, handin=True, valid=False)]
    )


class ScriptedBackend(Backend):
    """Test-only backend for exercising the CLI and real host with a scripted worker."""

    name: Literal["test_scripted"] = "test_scripted"
    cli: ClassVar[str | None] = None

    #: Play only the first steps of the command script (None: all of it).
    steps: int | None = None

    @property
    def spec(self) -> str:
        """Name the fixture backend in run records."""
        return "script/test_scripted"

    @property
    def place(self) -> str:
        """Return its script place."""
        return self.spec

    def driver(self, cfg: RunConfig) -> Any:
        """Return a Humanize driver with the command-flow script."""
        from hmz.sdk import fakes

        return fakes.FakeAgentDriver(reply=shell_worker(cfg, steps=self.steps), model=self.spec)


BACKENDS.register("test_scripted", ScriptedBackend)


def scripted(
    config: Path,
    run_dir: Path,
    seconds: float = 3.0,
    script: Callable[[RunConfig], ScriptedWorker] | None = None,
    overrides: list[str] | None = None,
) -> int:
    """Run a fixture script through the regular CLI execution path on CPU stand-ins."""
    from hmz.sdk import fakes
    from qda_agent import cli

    base = RunConfig.load(config, overrides or [])
    settings = [
        f"run_dir={run_dir.resolve()}",
        f"budget_gpu_hours={seconds / 3600 * len(base.cards)}",
        "sandbox=none",
        "scheduler.gpus=false",
    ]
    cfg = RunConfig.load(config, [*(overrides or []), *settings])
    worker = script(cfg) if script else (_flow_worker(cfg) if not cfg.commands else shell_worker(cfg))
    return cli.execute(cfg, worker=fakes.FakeAgentDriver(reply=worker, model="script/test_scripted"))
