"""Importing the run config pulls neither the judge nor the scheduler's sandbox.

`qda_agent.config` is the run's front door: a command that only reads a config (`qda doctor`, a resolved
record, a test's fixture) must not import the judge's checkers or the scheduler's sandbox code. The models
are defined in `qda_agent.run_config`, which the facade imports on first use.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"

#: Modules that must not be in `sys.modules` after `import qda_agent.config`.
BANNED_PREFIXES = ("qda_agent.judge", "qda_agent.sched")


def test_importing_the_config_imports_no_judge_and_no_sandbox():
    """A fresh process imports the config and names no judge or scheduler module."""
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "import qda_agent.config; "
        f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({BANNED_PREFIXES!r}))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert json.loads(out.stdout) == []


def test_the_facade_hands_out_the_models():
    """The public names are the model classes, imported on first use, and they are imported only once."""
    import qda_agent.config as facade
    import qda_agent.run_config as models

    assert facade.RunConfig is models.RunConfig
    assert facade.IsolationSettings is models.IsolationSettings
    assert "RunConfig" in dir(facade)


def test_starting_a_run_imports_no_vllm():
    """`qda-agent run` holds no card: its process must not import vLLM, whose GPU worker loads the CUDA stack.

    Jobs run vLLM in their own processes, pinned to the run's cards. vLLM may be absent here, so the import
    is caught when it is attempted, not looked up in `sys.modules` afterwards.
    """
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "tried = []; "
        "Spy = type('Spy', (), {'find_spec': staticmethod("
        "lambda name, path=None, target=None: tried.append(name) if name.split('.')[0] == 'vllm' else None)}); "
        "sys.meta_path.insert(0, Spy()); "
        "import qda_agent.cli, qda_agent.flow, qda_agent.judge.host, qda_agent.judge.startup; "
        "print(json.dumps(tried))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert json.loads(out.stdout) == []
