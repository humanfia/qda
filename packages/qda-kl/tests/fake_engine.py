"""Run ``qda-kl ref`` on a fake vLLM whose engine is a process tree, for test_lifetime.py.

Usage: ``python fake_engine.py WORK MODE [main|unowned]``. The fake ``vllm.LLM`` starts what a real engine leaves
behind when it is orphaned: a forked child (vLLM forks its engine core), and a session of its own holding a
leader that ignores SIGTERM and that leader's child. It writes ``pid start`` lines to ``WORK/pids`` and touches
``WORK/ready`` once its generation runs; MODE ``hang`` then sleeps there, MODE ``raise`` raises, and MODE
``start`` touches it and sleeps in the constructor instead (a stop while the engine starts). ``unowned`` runs
the command without its owner (``cli.main.__wrapped__``): the behaviour before commands owned their engines.
"""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import time
import types
from pathlib import Path

SESSION = """
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
time.sleep(600)
"""


def started(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()[19])


def main(work: Path, mode: str, which: str) -> int:
    class LLM:
        def __init__(self, **kwargs: object) -> None:
            fork = multiprocessing.get_context("fork").Process(target=time.sleep, args=(600,), daemon=True)
            fork.start()
            leader = subprocess.Popen([sys.executable, "-c", SESSION], start_new_session=True)
            children = Path(f"/proc/{leader.pid}/task/{leader.pid}/children")
            end = time.monotonic() + 30
            while not (kids := children.read_text().split()) and time.monotonic() < end:
                time.sleep(0.05)  # the leader's own child
            pids = [fork.pid, leader.pid, *map(int, kids)]
            (work / "pids").write_text("".join(f"{p} {started(p)}\n" for p in pids))
            self.processes = (fork, leader)
            if mode == "start":  # still starting: vLLM's constructor waits for the engine core
                (work / "ready").touch()
                time.sleep(600)

        def collective_rpc(self, name: str, *args: object, **kwargs: object) -> list:
            return [None]

        def generate(self, *args: object, **kwargs: object) -> list:
            (work / "ready").touch()
            if mode == "raise":
                raise RuntimeError("the fake engine failed")
            time.sleep(600)
            return []

    sys.modules["vllm"] = types.SimpleNamespace(LLM=LLM)  # type: ignore[assignment]
    from qda_quant.output import vllm_env

    vllm_env.host_compiler_env = lambda env: {}

    from qda_kl import cli, lifetime, reference
    from qda_kl.engine import open_engine

    lifetime.DEFAULT_GRACE_S = 1.0  # the session leader ignores SIGTERM: the sweep must escalate

    def build(model: str, *args: object, **kwargs: object) -> object:
        llm = open_engine(model, max_model_len=16)  # the path every reference build starts its engine by
        llm.generate([], [])
        raise AssertionError("unreachable")

    reference.build = build  # type: ignore[assignment]
    argv = ["ref", "--model", "m", "--prompts", str(work / "p.jsonl"), "--out", str(work / "out")]
    return (cli.main if which == "main" else cli.main.__wrapped__)(argv)  # type: ignore[attr-defined]


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "main"))
