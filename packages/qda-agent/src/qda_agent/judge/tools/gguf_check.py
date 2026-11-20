"""The admission check of a llama.cpp candidate: the pinned server loads it and answers a short prompt.

Run by the judge with `judge.llama_cmd` as `python gguf_check.py CKPT OUT`, where CKPT is the frozen `.gguf`
file the judge hands every llama.cpp job (or, run by hand, a directory holding exactly one). It starts `$QDA_LLAMA_SERVER`
(the pinned llama.cpp 9e2a04a build) on the candidate's one `.gguf` with the argv of one measured admission,
waits for `/health`, checks the loaded inventory through `qda_kl.llamacpp.strict_check` when the evaluator
provides it (`$QDA_LLAMA_CPP` names the checkout, `$QDA_LLAMA_BRIDGE` the probe that logs the inventory), and
makes one short native completion. It writes `{"ok": bool, "reason": str}` to OUT and the server's log beside
it (`OUT` with `.server.log`); it imports no qda_agent.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The argv of one measured admission (`qda_kl.gguf.run`'s `server()`, context 2048, parallel 1).
SETTINGS = ["-ngl", "99", "-c", "2048", "-np", "1", "-b", "128", "-ub", "128", "-t", "8", "-tb", "8"]
SETTINGS += ["--check-tensors", "--no-repack", "--fit", "off", "--no-context-shift"]
SETTINGS += ["-ctk", "bf16", "-ctv", "bf16", "-fa", "on", "--jinja", "--log-verbosity", "4"]
#: The one request: its `tokens_predicted` and `truncated` are the answer's verdict.
BODY = {
    "prompt": "Say hello.",
    "n_predict": 16,
    "temperature": 0,
    "seed": 17,
    "cache_prompt": False,
    "return_tokens": True,
}


def request(port: int, route: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    """One request to the server's native API, its JSON answer as a dict."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        got = json.load(response)
    return got if isinstance(got, dict) else {}


def free_port() -> int:
    """A port nothing is bound to right now, for this one server."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def strict(model: Path, log: Path, llama_cpp: str) -> dict:
    """The strict load check of the server's log (`qda_kl.llamacpp.strict_check`).

    ImportError when the evaluator copy has none; ValueError on a different inventory.
    """
    from qda_kl.llamacpp import strict_check

    return strict_check(model, log.read_text(), llama_cpp)


def the_model(ckpt: Path) -> Path | None:
    """The `.gguf` CKPT stands for: CKPT itself when it is a regular `.gguf` file, else the one top-level `.gguf`
    of the directory CKPT. None for a link, a directory holding none or several, or anything else."""
    if ckpt.is_symlink():
        return None
    if ckpt.is_file():
        return ckpt if ckpt.suffix == ".gguf" else None
    if not ckpt.is_dir():
        return None
    found = sorted(p for p in ckpt.glob("*.gguf") if p.is_file() and not p.is_symlink())
    return found[0] if len(found) == 1 else None  # the file rules admit exactly one; a check never guesses


def check(ckpt: Path, log: Path, timeout: float) -> tuple[bool, str]:
    """Load the candidate in the pinned server, check what it loaded, and answer one short prompt."""
    server = os.environ.get("QDA_LLAMA_SERVER", "")
    if not server:
        return False, "no pinned llama-server: set judge.llamacpp.server"
    model = the_model(ckpt)
    if model is None:
        return False, "no single top-level .gguf"
    port = free_port()
    argv = [server, "-m", str(model), "--host", "127.0.0.1", "--port", str(port), *SETTINGS]
    bridge = os.environ.get("QDA_LLAMA_BRIDGE", "")
    env = {**os.environ, "OMP_NUM_THREADS": "8", **({"LD_PRELOAD": bridge} if bridge else {})}
    with log.open("w") as out:
        # no new session: the server stays in the job's process group, so a cancelled job's group kill
        # reaches it instead of leaving it on the cards
        proc = subprocess.Popen(  # noqa: S603 -- the pinned build, a host setting
            argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                if proc.poll() is not None:
                    return False, f"llama-server exited {proc.returncode} (see its log)"
                try:
                    request(port, "health", timeout=5)
                    break
                except (urllib.error.URLError, ConnectionError, OSError):
                    if time.monotonic() > deadline:
                        return False, f"llama-server was not ready within {timeout:g}s"
                    time.sleep(0.5)
            llama_cpp = os.environ.get("QDA_LLAMA_CPP", "")
            if llama_cpp:
                if not os.environ.get("QDA_LLAMA_BRIDGE"):
                    return (
                        False,
                        "no probe.so (judge.llamacpp.bridge): the server's loaded tensors cannot be checked",
                    )
                try:
                    strict(model, log, llama_cpp)
                except ImportError:
                    pass  # an evaluator copy without qda_kl.llamacpp: load and answer are this check
                except Exception as e:  # noqa: BLE001 -- a mismatch, or a checkout/file it cannot read
                    return False, f"the strict load check refused it: {type(e).__name__}: {str(e)[:200]}"
            try:
                got = request(port, "completion", BODY, timeout=timeout)
            except (urllib.error.URLError, OSError, ValueError) as e:
                return False, f"no answer to the short prompt: {type(e).__name__}: {str(e)[:200]}"
            if got.get("tokens_predicted", 0) <= 0 or got.get("truncated", False):
                return False, "the short prompt got no token or was truncated"
            return True, ""
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main(argv: list[str]) -> int:
    """Write the verdict; exit 0 whenever OUT was written."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--timeout", type=float, default=300.0)
    a = p.parse_args(argv)
    try:
        ok, reason = check(a.ckpt, a.out.with_suffix(".server.log"), a.timeout)
    except Exception as e:  # noqa: BLE001  (any load failure is a refusal, with its type)
        ok, reason = False, f"llama-server could not serve it: {type(e).__name__}: {str(e)[:200]}"
    a.out.write_text(json.dumps({"ok": ok, "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
