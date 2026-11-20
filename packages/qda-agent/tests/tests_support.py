"""Helpers the tests import: contamination checks of their own, agent settings from a Humanize spec, and
shipped configs with their placeholders filled in."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
import yaml
from qda_agent.agents import AgentBackend, Dsh
from qda_agent.configfile import PLACEHOLDER
from qda_agent.judge.contamination import CHECKS, ContaminationCheck


class Endpoint:
    """The endpoint a `dsh` run's key goes to, standing in for one: it records the turn asked for.

    `answers` is what it gives back, as `(status, body)`: the official API's refusal to a model it does not
    serve is one of them (a 400 naming the names it does serve). A `bytes` or `str` body goes out as it is,
    which is how a test writes a body an endpoint would have escaped.
    """

    def __init__(self) -> None:
        #: What it answers with, or a queue of them for a test that asks more than once (a redirect, say).
        self.answers: tuple[int, object] | list[tuple[int, object]] = (
            200,
            {"choices": [{}], "usage": {"total_tokens": 32}},
        )
        #: A `Content-Length` to claim rather than the body's own, for a body that stops half way through.
        self.length: int | None = None
        #: A `Location` to answer with, for a redirect: the status it goes with is the test's to set.
        self.location: str | None = None
        self.asked: list[dict] = []
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- the name BaseHTTPRequestHandler calls
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                endpoint.asked.append(
                    {"path": self.path, "headers": dict(self.headers), "json": json.loads(body)}
                )
                answers = endpoint.answers
                status, said = answers.pop(0) if isinstance(answers, list) else answers
                # str and bytes go out as written: a test says a body a JSON dump cannot (an escaped key)
                if isinstance(said, str):
                    sent = said.encode()
                elif isinstance(said, bytes):
                    sent = said
                else:
                    sent = json.dumps(said).encode()
                self.send_response(status)
                if endpoint.location is not None:
                    self.send_header("Location", endpoint.location)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Content-Length", str(len(sent) if endpoint.length is None else endpoint.length)
                )
                self.end_headers()
                self.wfile.write(sent)

            def log_message(self, *args: object) -> None:  # noqa: ARG002 -- a test says nothing to stderr
                """Keep the server quiet: the test's own assertions are the record."""

        return Handler

    def __enter__(self) -> Endpoint:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def endpoint(monkeypatch):
    """A stub endpoint that answers for a `dsh` run's key, with the SDK's modules taken as installed.

    The run's own `agent turn` check asks that endpoint before its clock starts (and `qda doctor --search`
    asks the same), so a test that runs one needs somewhere for it to answer; the modules are what CI does
    not install (`--extra dev` pulls no `dsh`), and they are asked for before any request.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "a-test-key")
    monkeypatch.setattr(Dsh, "modules", ())
    with Endpoint() as endpoint:
        monkeypatch.setenv("DEEPSEEK_BASE_URL", endpoint.url)
        yield endpoint


def assert_job_process(pid: int, tmp_path: Path, script: Path | None = None) -> None:
    """Refuse to signal a process outside this test's directory and job script."""
    assert pid > 1 and pid not in {os.getpid(), os.getppid()}, pid
    proc = Path(f"/proc/{pid}")
    try:
        cwd = (proc / "cwd").readlink()
        argv = (proc / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        if (proc / "stat").exists():
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            assert state == "Z", pid
        return
    assert cwd.is_relative_to(tmp_path) or (script is not None and os.fsencode(script) in argv), (pid, cwd)


def assert_job_group(pgid: int, tmp_path: Path, script: Path | None = None) -> None:
    """Refuse special, caller-owned or foreign process groups before signalling a test job."""
    assert pgid > 1, pgid
    assert pgid != os.getpgid(0), pgid
    assert pgid != os.getpgid(os.getppid()), pgid
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            assert_job_process(int(stat.parent.name), tmp_path, script)


def backend(spec: str) -> dict:
    """`CLI[@ACCOUNT]/MODEL[:EFFORT]` as an `agent.backend` mapping."""
    cli, model = spec.split("/", 1)
    name, _, account = cli.partition("@")
    model, _, effort = model.partition(":")
    return {"name": name, "model": model, "effort": effort or None, "account": account or None}


def agent(spec: str, *fallbacks: str, **settings: object) -> dict:
    """An `agent:` section: `spec` as the backend, then `fallbacks`."""
    return {"backend": backend(spec), "fallbacks": [backend(f) for f in fallbacks], **settings}


class NoConfigCli(AgentBackend):
    """A backend plugin whose CLI is one Humanize knows, but `isolation.CLI_CONFIG` does not."""

    cli: ClassVar[str] = "qwen"
    name: Literal["qwen"] = "qwen"


def filled(config: Path, tmp: Path) -> Path:
    """A copy of the shipped `config` in `tmp` with every `<...>` placeholder replaced by a path under `tmp`.

    The paths are created, since a run config must name inputs that exist: a placeholder naming an
    interpreter (`<path/to/venv/bin/python>`) becomes a virtualenv, with its `pyvenv.cfg`; one whose last
    name has a suffix is a file, and any other is a directory.
    """

    def target(name: str) -> Path:
        path = tmp / name
        if path.name == "python":  # a virtualenv's interpreter
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            (path.parent.parent / "pyvenv.cfg").write_text("home = /usr/bin\n")
        elif "." in path.name:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def fill(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: "0" * 64 if k.endswith("_sha256") else fill(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fill(v) for v in node]
        if isinstance(node, str):
            return PLACEHOLDER.sub(lambda m: str(target(m.group(0).strip("<>"))), node)
        return node

    out = tmp / config.name
    out.write_text(yaml.safe_dump(fill(yaml.safe_load(config.read_text())), sort_keys=False))
    return out


class Dirty(ContaminationCheck):
    """Finds a test item in the first file."""

    name: Literal["dirty"] = "dirty"

    def scanner(self):
        return lambda paths: [f"{p.name}: test item 17" for p in paths[:1]]


SCANS: list[tuple[float, int]] = []


class Slow(ContaminationCheck):
    """A clean scan that takes 0.3 s and records when it ended and how many files it saw."""

    name: Literal["slow"] = "slow"

    def scanner(self):
        def scan(paths):
            time.sleep(0.3)
            SCANS.append((time.time(), len(paths)))
            return []

        return scan


CHECKS.register("dirty", Dirty)
CHECKS.register("slow", Slow)


#: The CPU test fixtures: the scripted search config and the files it names.
FIXTURES = Path(__file__).parent / "fixtures"


def search_config(tmp_path: Path, **judge: Any) -> Path:
    """The scripted search config with its paths made absolute, and `judge` overridden.

    One copy for every test file that needs it: a test module never imports another `test_*` module for a
    helper, because the packages share test basenames (`test_cli.py`) and a repository-root run resolves such
    an import to the other package's file.
    """
    raw = yaml.safe_load((FIXTURES / "scripted-search.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "run")
    raw["baselines"] = [str((FIXTURES / b).resolve()) for b in raw["baselines"]]
    raw["judge"]["evaluator"] = str((FIXTURES / raw["judge"]["evaluator"]).resolve())
    for key in ("kl_ref", "kl_holdout_ref"):
        if raw["judge"].get(key):
            raw["judge"][key] = str((FIXTURES / raw["judge"][key]).resolve())
    raw["judge"].update(judge)
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path
