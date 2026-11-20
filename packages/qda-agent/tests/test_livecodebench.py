"""The LiveCodeBench grader: the official runners in a sandbox, and a limit the sandbox enforces."""

import base64
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import types
import zlib
from functools import lru_cache
from pathlib import Path

import pytest
from qda_agent import livecodebench, livecodebench_sandbox
from qda_agent.livecodebench import LiveCodeBenchSuite
from qda_agent.livecodebench_data import cases_of
from qda_agent.livecodebench_sandbox import READ_TAIL, _remove_tree, _tail_fd, grade_code, interpreter_trees

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="the grader runs test cases in bubblewrap"
)

#: A stdin case and a functional case, each with its own expected output.
STDIN = [{"input": "21\n", "output": "42\n"}]
FUNCTIONAL = [{"input": "21", "output": "42"}]

#: A probe that prints ``yes`` only when the statement it builds holds inside the sandbox.
YES = [{"input": "", "output": "yes"}]

#: The two /proc views a suite can ask the sandbox for (the strict one is the default).
PROCS = ["private", "host-readonly"]


@pytest.fixture
def two_cores():
    """Skips on a host with one physical core: grading reserves whole cores (SMT siblings included) and keeps
    one apart for the engine, so the grader's CPU choice refuses there before any program is graded."""
    available = os.sched_getaffinity(0)
    cores = {frozenset(LiveCodeBenchSuite._siblings({cpu})) for cpu in available}
    if len(cores) < 2:
        pytest.skip(f"{len(available)} CPUs on one physical core: the grader and the engine need one each")


@lru_cache(maxsize=1)
def _private_proc_refused() -> bool:
    """Whether this container refuses a fresh proc mount, so the strict-mode cases cannot run here."""
    if shutil.which("bwrap") is None:
        return False
    with tempfile.TemporaryDirectory() as scratch:
        try:
            grade_code("print(int(input()) * 2)", None, STDIN, Path(scratch), 8.0)
        except RuntimeError as why:
            return "Can't mount proc" in str(why)
    return False


@pytest.fixture(params=PROCS)
def proc(request):
    """Each /proc view; `private` skips only where the container refuses a fresh proc mount."""
    if request.param == "private" and _private_proc_refused():
        pytest.skip("this container refuses a fresh /proc; run the suite with proc: host-readonly")
    return request.param


def _probe(body: str, tmp_path, proc: str = "private"):
    """Run ``body`` (which prints ``yes`` or ``no``) through the sandboxed grader and return its verdict."""
    return grade_code(textwrap.dedent(body), None, YES, tmp_path, 8.0, proc=proc)


@needs_bwrap
def test_the_grader_passes_correct_programs_and_fails_a_wrong_one(tmp_path, proc):
    assert grade_code("print(int(input()) * 2)", None, STDIN, tmp_path, 8.0, proc=proc) == (True, 1, 0, 0)
    assert grade_code("print(int(input()) + 2)", None, STDIN, tmp_path, 8.0, proc=proc)[0] is False
    solution = "class Solution:\n    def double(self, x: int) -> int:\n        return 2 * x\n"
    assert grade_code(solution, "double", FUNCTIONAL, tmp_path, 8.0, proc=proc) == (True, 1, 0, 0)


@needs_bwrap
def test_the_grader_kills_a_program_that_never_stops(tmp_path, proc):
    passed, ran, timeouts, errors = grade_code("while True:\n    pass", None, STDIN, tmp_path, 1.0, proc=proc)
    assert (passed, ran, timeouts, errors) == (False, 1, 1, 0)


@needs_bwrap
def test_the_paper_runner_gives_stdin_a_real_process_and_limits_cpu_time(tmp_path, proc):
    # The official runner replaces sys.stdin/sys.stdout in its own interpreter, so sys.stdout.buffer fails there.
    io = "import sys\nsys.stdout.buffer.write(str(int(open(0).read()) * 2).encode())\n"
    assert grade_code(io, None, STDIN, tmp_path, 8.0, proc=proc)[0] is False
    assert grade_code(io, None, STDIN, tmp_path, 8.0, proc=proc, protocol="paper") == (True, 1, 0, 0)
    solution = "class Solution:\n    def double(self, x: int) -> int:\n        return 2 * x\n"
    assert grade_code(solution, "double", FUNCTIONAL, tmp_path, 8.0, proc=proc, protocol="paper") == (
        True,
        1,
        0,
        0,
    )
    # A sleeping program spends no CPU time; the wall-clock backstop still ends it, as a timeout.
    asleep = grade_code("import time\ntime.sleep(60)", None, STDIN, tmp_path, 1, proc=proc, protocol="paper")
    assert asleep == (False, 1, 1, 0)


@needs_bwrap
def test_official_state_and_separate_compilation_and_case_budgets(tmp_path, proc):
    code = (
        "import time\nclass Solution:\n"
        "    def __init__(self):\n        time.sleep(0.7)\n        self.n = 0\n"
        "    def count(self, x):\n        time.sleep(0.7)\n        self.n += 1\n        return self.n\n"
    )
    cases = [{"input": "0", "output": "1"}, {"input": "0", "output": "2"}]
    assert grade_code(code, "count", cases, tmp_path, 1.0, proc=proc) == (True, 2, 0, 0)


@pytest.mark.usefixtures("two_cores")
@pytest.mark.parametrize("retry_pass", [True, False])
def test_timeout_retry_waits_for_the_parallel_chunk_and_replaces_the_score(tmp_path, monkeypatch, retry_pass):
    suite = LiveCodeBenchSuite(repeats=2)
    lock = threading.Lock()
    active = 0
    attempts = {}

    def grade(code, *args, **kwargs):
        nonlocal active
        with lock:
            active += 1
            attempts[code] = attempts.get(code, 0) + 1
            if attempts[code] == 2:
                assert active == 1
                assert attempts.get("ok") == 1
        time.sleep(0.05)
        with lock:
            active -= 1
        return (
            (False, 1, 1, 0)
            if code == "slow" and (attempts[code] == 1 or not retry_pass)
            else (True, 2, 0, 0)
        )

    monkeypatch.setattr(livecodebench, "grade_code", grade)
    chunk = [("q", ["slow", "slow"], STDIN, None), ("r", ["ok", None], STDIN, None)]
    scores = {}
    counts = {"cases_run": 0, "timeouts": 0, "timeouts_before_retry": 0, "errors": 0}
    suite._grade_chunk(chunk, tmp_path, scores, counts, sandbox="bwrap")
    assert scores == {"q": [float(retry_pass), float(retry_pass)], "r": [1.0, 0.0]}
    assert attempts == {"slow": 2, "ok": 1}
    assert counts == {
        "cases_run": 4 + int(retry_pass),
        "timeouts": int(not retry_pass),
        "timeouts_before_retry": 1,
        "errors": 0,
    }


@pytest.mark.parametrize("stall", [False, True])
def test_launcher_failure_after_reporting_a_child_is_never_a_verdict(tmp_path, monkeypatch, stall):
    launcher = tmp_path / "failed_bwrap.py"
    launcher.write_text(
        "import os, sys, time\n"
        "if '--json-status-fd' in sys.argv:\n"
        "    fd = int(sys.argv[sys.argv.index('--json-status-fd') + 1])\n"
        "    os.write(fd, b'{\"child-pid\": 123}\\n')\n"
        'sys.stderr.write("bwrap: Can\'t mount proc: Operation not permitted\\n")\n'
        "sys.stderr.flush()\n" + ("time.sleep(30)\n" if stall else "sys.exit(1)\n")
    )
    original = livecodebench_sandbox.sandbox_argv

    def failed_bwrap(where, status_fd, **kwargs):
        argv, pinned = original(where, status_fd, **kwargs)
        return [sys.executable, str(launcher), *argv[1:]], pinned

    monkeypatch.setattr(livecodebench_sandbox, "sandbox_argv", failed_bwrap)
    with pytest.raises(RuntimeError, match="refused:.*Can't mount proc.*administrator"):
        grade_code("print(int(input()) * 2)", None, STDIN, tmp_path, 0.5)


def test_prepare_preflights_even_a_cached_selection_before_downloading(tmp_path, monkeypatch):
    suite = LiveCodeBenchSuite(data=str(tmp_path / "data"), max_problems=1)
    root = tmp_path / "livecodebench"
    root.mkdir()
    (root / "items.json").write_text(
        json.dumps(
            {
                "key": {
                    "version": livecodebench.ITEM_VERSION,
                    "revision": livecodebench.REVISION,
                    "window": list(livecodebench.WINDOW),
                    "max_problems": 1,
                    "protocol": "official",
                },
                "files": [],
                "items": [],
            }
        )
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")

    def refuse(*args, **kwargs):
        raise RuntimeError("refused: private /proc unavailable; ask the administrator")

    def no_download(*args, **kwargs):
        pytest.fail("dataset selection ran before sandbox preflight")

    monkeypatch.setattr(livecodebench, "grade_code", refuse)
    monkeypatch.setattr(LiveCodeBenchSuite, "_select", no_download)
    with pytest.raises(RuntimeError, match="refused: private /proc unavailable"):
        suite.prepare("unused", tmp_path)


def _prepared(suite, tmp_path, monkeypatch, bwrap: bool):
    """Prepare `suite` on a machine with (or without) bubblewrap, recording what the grader was asked for."""
    asked: list[str] = []

    def grade(*args, **kwargs):
        asked.append(kwargs["sandbox"])
        return True, 1, 0, 0

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap" if bwrap else None)
    monkeypatch.setattr(LiveCodeBenchSuite, "_select", lambda self, data: ([], []))
    monkeypatch.setattr(livecodebench, "grade_code", grade)
    suite.prepare("unused", tmp_path)
    return asked


def test_the_default_sandbox_grades_in_bubblewrap_where_it_is_installed(tmp_path, monkeypatch):
    """`auto` is the default and asks for the sandbox on a machine that has bubblewrap."""
    suite = LiveCodeBenchSuite(data=str(tmp_path / "data"), max_problems=1)
    assert suite.sandbox == "auto"
    assert _prepared(suite, tmp_path, monkeypatch, bwrap=True) == ["bwrap"]


def test_without_bubblewrap_the_default_refuses_and_names_the_unsandboxed_option(tmp_path, monkeypatch):
    """`auto` never grades unsandboxed by itself: it stops, and says what to pass to accept that."""
    suite = LiveCodeBenchSuite(data=str(tmp_path / "data"), max_problems=1)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(LiveCodeBenchSuite, "_select", lambda self, data: pytest.fail("prepared anyway"))
    with pytest.raises(ValueError, match=r"--lcb-sandbox none"):
        suite.prepare("unused", tmp_path)
    # `bwrap` is the old hard requirement: it refuses too, and offers nothing else
    with pytest.raises(ValueError, match="needs bubblewrap"):
        LiveCodeBenchSuite(sandbox="bwrap").prepare("unused", tmp_path)


def test_sandbox_none_grades_without_bubblewrap_after_one_warning(tmp_path, monkeypatch, capsys):
    """`none` is the opt-in lm_eval's --confirm_run_unsafe_code asks for: it warns, then it grades."""
    suite = LiveCodeBenchSuite(data=str(tmp_path / "data"), max_problems=1, sandbox="none")
    assert _prepared(suite, tmp_path, monkeypatch, bwrap=False) == ["none"]
    warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("WARNING")]
    assert len(warnings) == 1 and "unsandboxed" in warnings[0]


def test_an_unsandboxed_program_is_graded_by_the_same_grader(tmp_path):
    """The `none` backend runs the real grader on this machine: a right program passes, a wrong one fails."""
    right = grade_code("print(int(input()) * 2)", None, STDIN, tmp_path, 8.0, sandbox="none")
    assert right == (True, 1, 0, 0)
    assert grade_code("print(int(input()) + 2)", None, STDIN, tmp_path, 8.0, sandbox="none")[0] is False


@needs_bwrap
def test_a_program_that_exits_after_startup_is_a_grader_error(tmp_path, proc):
    assert grade_code("import os\nos._exit(1)", None, STDIN, tmp_path, 8.0, proc=proc) == (False, 1, 0, 1)


@needs_bwrap
def test_a_program_cannot_print_bwrap_errors_and_abort_the_run(tmp_path, proc):
    code = 'import sys\nsys.stderr.write("bwrap: fake\\n")\nprint(int(input()) * 2)'
    assert grade_code(code, None, STDIN, tmp_path, 8.0, proc=proc) == (True, 1, 0, 0)


@needs_bwrap
def test_the_sandbox_has_no_run_directory(tmp_path, proc):
    passed, *_ = _probe(
        """
        import os
        print("yes" if not os.path.exists("/run/user") else "no")
        """,
        tmp_path,
        proc,
    )
    assert passed


@needs_bwrap
def test_a_program_emptying_its_scratch_cannot_abort_the_run(tmp_path, proc):
    # The readiness pipe is closed before generated code runs; emptying scratch cannot erase startup.
    passed, *_ = _probe(
        """
        import glob, os
        for path in glob.glob("/scratch/*") + glob.glob("/scratch/.*"):
            try:
                if os.path.isfile(path):
                    open(path, "w").close()
            except OSError:
                pass
        print("yes")
        """,
        tmp_path,
        proc,
    )
    assert passed


def test_the_output_is_read_by_descriptor_and_bounded(tmp_path):
    path = tmp_path / "out"
    path.write_bytes(b"x" * (READ_TAIL + 7) + b"TAIL")
    fd = os.open(path, os.O_RDONLY)
    try:
        tail = _tail_fd(fd, READ_TAIL)
    finally:
        os.close(fd)
    assert len(tail) == READ_TAIL and tail.endswith("TAIL")
    zero = os.open("/dev/zero", os.O_RDONLY)
    try:
        assert _tail_fd(zero, READ_TAIL) == ""
    finally:
        os.close(zero)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        assert _tail_fd(fd, READ_TAIL) == ""
    finally:
        os.close(fd)


def test_cleanup_removes_a_case_directory_the_program_locked(tmp_path):
    root = tmp_path / "case-x"
    locked = root / "locked" / "inner"
    locked.mkdir(parents=True)
    (locked / "answer.txt").write_text("x")
    locked.chmod(0)
    locked.parent.chmod(0)
    root.chmod(0)
    _remove_tree(root)
    assert not root.exists()


@needs_bwrap
def test_a_program_cannot_redirect_the_hosts_read_of_its_output(tmp_path, tmp_path_factory, proc):
    outside = tmp_path_factory.mktemp("host")
    host_fifo = outside / "fifo"
    os.mkfifo(host_fifo)
    opened = []

    def writer():
        with host_fifo.open("w") as stream:
            opened.append(True)
            stream.write("host data\n")

    threading.Thread(target=writer, daemon=True).start()
    plants = {
        "link to /dev/zero": "posix.symlink('/dev/zero', '/scratch/.stdout')",
        "program-made FIFO": "posix.mkfifo('/scratch/.stdout')",
        "link to a host FIFO": f"posix.symlink({str(host_fifo)!r}, '/scratch/.stdout')",
    }
    for name, plant in plants.items():
        code = f"import posix\nposix.unlink('/scratch/.stdout')\n{plant}\nprint(int(input()) * 2)"
        verdict = []
        thread = threading.Thread(
            target=lambda code=code, verdict=verdict: verdict.append(
                grade_code(code, None, STDIN, tmp_path, 3.0, proc=proc)
            ),
            daemon=True,
        )
        started = time.monotonic()
        thread.start()
        thread.join(20)
        assert verdict, f"{name}: grade_code did not return in 20 s"
        assert time.monotonic() - started < 10
        assert verdict[0] == (True, 1, 0, 0)
    time.sleep(0.3)
    assert not opened, "the host opened the FIFO the program's link names"


@needs_bwrap
def test_the_sandbox_root_is_exactly_the_allowlist(tmp_path, proc):
    # A Python installed outside /usr (uv's --managed-python, pyenv, conda) adds the top directory of its own
    # installation, and nothing else (`test_home_and_the_repository_are_absent` checks what is under it).
    want = ["deps", "dev", "grader", "lib", "lib64", "proc", "scratch", "usr"]
    want = sorted({*want, *(tree.parts[1] for tree in interpreter_trees())})
    passed, *_ = _probe(
        f"""
        import os
        print("yes" if sorted(os.listdir("/")) == {want!r} else "no")
        """,
        tmp_path,
        proc,
    )
    assert passed


@needs_bwrap
def test_only_the_scratch_directory_is_writable(tmp_path, proc):
    # `host-readonly` binds the host's /proc read-only, so /proc is no more writable than the rest.
    passed, *_ = _probe(
        """
        ok = True
        for path, allowed in (
            ("/qda-probe", False),
            ("/tmp/qda-probe", False),
            ("/proc/qda-probe", False),
            ("/proc/sys/qda-probe", False),
            ("/scratch/qda-probe", True),
        ):
            try:
                open(path, "w").write("x")
                ok = ok and allowed
            except OSError:
                ok = ok and not allowed
        print("yes" if ok else "no")
        """,
        tmp_path,
        proc,
    )
    assert passed


@needs_bwrap
def test_the_sandbox_cannot_connect_a_unix_socket(tmp_path, proc):
    passed, *_ = _probe(
        """
        import os, socket
        ok = True
        for path in (f"/run/user/{os.getuid()}/bus", "/run/dbus/system_bus_socket"):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(path)
                ok = False
            except OSError:
                pass
            finally:
                sock.close()
        print("yes" if ok else "no")
        """,
        tmp_path,
        proc,
    )
    assert passed


@needs_bwrap
def test_home_and_the_repository_are_absent(tmp_path, proc):
    # /home is absent, unless the interpreter's own installation lies under it (uv's --managed-python puts it
    # in ~/.local/share/uv/python): then the only paths there are that installation and the directories that
    # lead to it, which hold nothing else. The same holds for any top directory such an installation adds.
    trees = [str(tree) for tree in interpreter_trees()]
    leads = sorted(
        {str(parent) for tree in interpreter_trees() for parent in tree.parents if parent != Path("/")}
    )
    tops = sorted({"/home", *(f"/{tree.parts[1]}" for tree in interpreter_trees())})
    passed, *_ = _probe(
        f"""
        import os
        trees, leads = {trees!r}, set({leads!r})
        ok = not os.path.exists(os.environ.get("HOME", "/nonexistent"))
        for top in {tops!r}:
            for where, dirs, files in os.walk(top):
                if any(where == t or where.startswith(t + "/") for t in trees):
                    dirs[:] = []
                    continue
                inside = {{os.path.join(where, d) for d in dirs}}
                ok = ok and where in leads and not files and inside <= leads | set(trees)
        print("yes" if ok else "no")
        """,
        tmp_path,
        proc,
    )
    assert passed


@needs_bwrap
def test_proc_shows_only_the_sandboxs_own_processes(tmp_path):
    if _private_proc_refused():
        pytest.skip("this container refuses a fresh /proc; run the suite with proc: host-readonly")
    passed, *_ = _probe(
        """
        import os
        pids = {int(name) for name in os.listdir("/proc") if name.isdigit()}
        print("yes" if pids <= {1, os.getpid()} else "no")
        """,
        tmp_path,
        "private",
    )
    assert passed


@needs_bwrap
def test_an_outbound_tcp_connection_fails(tmp_path, proc):
    passed, *_ = _probe(
        """
        import socket
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=2)
            ok = False
        except OSError:
            ok = True
        print("yes" if ok else "no")
        """,
        tmp_path,
        proc,
    )
    assert passed


def test_private_cases_decode_without_importing_what_the_payload_names():
    cases = [{"input": "1", "output": "2"}]
    raw = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
    row = {"metadata": "{}", "public_test_cases": "[]", "private_test_cases": raw}
    assert cases_of(row) == (cases, None)

    class Payload:
        def __reduce__(self):
            return (print, ("the payload ran",))

    evil = base64.b64encode(zlib.compress(pickle.dumps(Payload()))).decode()
    with pytest.raises(pickle.UnpicklingError):
        cases_of({**row, "private_test_cases": evil})


@pytest.mark.usefixtures("two_cores")
def test_a_missing_dataset_row_refuses_instead_of_scoring_zero(tmp_path):
    suite = LiveCodeBenchSuite(data=str(tmp_path))
    items = [{"id": "absent", "prompt": "", "difficulty": "easy"}]
    with pytest.raises(RuntimeError, match="no dataset row"):
        suite._grade(items, [[None]], tmp_path / "scratch", sandbox="bwrap")


@pytest.mark.usefixtures("two_cores")
def test_repeated_programs_keep_their_draw_scores(monkeypatch, tmp_path):
    """Deduplicating grader work must retain draw order and count an absent program as a failure."""
    import qda_agent.livecodebench as lcb

    calls = []

    def grade(code, *args, **kwargs):
        calls.append(code)
        return code == "right", 1, 0, 0

    row = {"question_id": "q", "public_test_cases": "[]"}
    monkeypatch.setattr(lcb, "grade_code", grade)
    monkeypatch.setattr(LiveCodeBenchSuite, "_rows_for", lambda self, wanted: iter([row]))
    suite = LiveCodeBenchSuite(repeats=4)
    got, counts = suite._grade(
        [{"id": "q"}],
        [["```\nright\n```", "```\nwrong\n```", None, "```\nright\n```"]],
        tmp_path,
        sandbox="bwrap",
    )
    assert got == {"q": [1.0, 0.0, 0.0, 1.0]}
    assert sorted(calls) == ["right", "wrong"]
    assert counts["no_code"] == 1


@pytest.mark.usefixtures("two_cores")
def test_unfinished_thinking_is_not_graded_and_scores_zero(monkeypatch, tmp_path):
    """A code fence inside an unclosed think block is not an answer; a closed one is graded."""
    import qda_agent.livecodebench as lcb

    calls = []

    def grade(code, *args, **kwargs):
        calls.append(code)
        return True, 1, 0, 0

    row = {"question_id": "q", "public_test_cases": "[]"}
    monkeypatch.setattr(lcb, "grade_code", grade)
    monkeypatch.setattr(LiveCodeBenchSuite, "_rows_for", lambda self, wanted: iter([row]))
    suite = LiveCodeBenchSuite(repeats=2, think_end_token="</think>")
    unclosed = "<think>plan\n```python\nprint(42)\n```"
    closed = "<think>plan</think>\n```python\nprint(42)\n```"

    got, counts = suite._grade([{"id": "q"}], [[unclosed, closed]], tmp_path, sandbox="bwrap")

    assert got == {"q": [0.0, 1.0]}
    assert calls == ["print(42)"]  # only the closed answer's program reached the grader
    assert counts["no_code"] == 1


def test_generation_names_the_thinking_mode_in_the_template(monkeypatch, tmp_path):
    """Every prompt asks for the same mode as the other suites, not the tokenizer's unset switch."""
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            if kwargs.get("tokenize") is False:
                return "rendered question"
            return {"input_ids": [1]}

    class Engine:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, jobs, params, use_tqdm=False):
            outputs = [types.SimpleNamespace(text="```python\nprint(42)\n```", finish_reason="stop")]
            return [types.SimpleNamespace(outputs=outputs) for _ in jobs]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=Engine, TokensPrompt=dict, SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs)
        ),
    )
    suite = LiveCodeBenchSuite(repeats=1, sampling={"temperature": 0.0})

    answers, finishes, hashes = suite._generate(str(tmp_path), ["question"])

    assert answers == [["```python\nprint(42)\n```"]]
    assert finishes == [["stop"]]
    assert hashes == [hashlib.sha256(b"rendered question").hexdigest()]
    assert len(calls) == 2 and all(call["enable_thinking"] is True for call in calls)


def test_a_closing_tag_suite_keeps_special_tokens_in_the_stored_text(monkeypatch, tmp_path):
    """vLLM strips special tokens by default, which would delete the tag answer_region() cuts at."""
    params = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002 - the HF signature
            # HF's own contract: the rendered text when tokenize is off, ids otherwise.
            return "rendered question" if kwargs.get("tokenize") is False else {"input_ids": [1]}

    class Engine:
        def __init__(self, **kwargs):  # noqa: ARG002 - vLLM's signature
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, jobs, gen_params, use_tqdm=False):  # noqa: ARG002 - vLLM's signature
            params.extend(gen_params)
            outputs = [types.SimpleNamespace(text="```python\nprint(42)\n```", finish_reason="stop")]
            return [types.SimpleNamespace(outputs=outputs) for _ in jobs]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=Engine, TokensPrompt=dict, SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs)
        ),
    )

    LiveCodeBenchSuite(repeats=1, sampling={"temperature": 0.0}, think_end_token="<channel|>")._generate(
        str(tmp_path), ["question"]
    )
    assert params and all(param.skip_special_tokens is False for param in params)

    params.clear()
    LiveCodeBenchSuite(repeats=1, sampling={"temperature": 0.0})._generate(str(tmp_path), ["question"])
    assert params and all(param.skip_special_tokens is True for param in params)


@pytest.mark.usefixtures("two_cores")
def test_without_a_close_tag_the_whole_generation_is_an_answer(monkeypatch, tmp_path):
    """A checkpoint that does not think keeps the extraction it had before think_end_token existed."""
    import qda_agent.livecodebench as lcb

    monkeypatch.setattr(lcb, "grade_code", lambda code, *args, **kwargs: (True, 1, 0, 0))
    row = {"question_id": "q", "public_test_cases": "[]"}
    monkeypatch.setattr(LiveCodeBenchSuite, "_rows_for", lambda self, wanted: iter([row]))
    suite = LiveCodeBenchSuite(repeats=1)

    got, _ = suite._grade([{"id": "q"}], [["```python\nprint(42)\n```"]], tmp_path, sandbox="bwrap")

    assert got == {"q": [1.0]}


def test_a_tokenizer_that_changes_the_tokenization_refuses_before_generating(monkeypatch, tmp_path):
    """The engine tokenizer is checked against the checkpoint's tokenizer.json; a mismatch refuses."""
    import qda_kl.tokenizer_check as check

    generated = []

    def refusal(tokenizer, model_dir, **kwargs):  # noqa: ARG001 - the shared check's signature
        return "the engine's tokenizer (LlamaTokenizer) encodes the probe as 8 ids, not 11"

    class Engine:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return object()

        def generate(self, jobs, params, use_tqdm=False):  # noqa: ARG002 - vLLM's signature
            generated.append(jobs)
            return []

    monkeypatch.setattr(check, "tokenizer_refusal", refusal)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=Engine, TokensPrompt=dict, SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs)
        ),
    )
    suite = LiveCodeBenchSuite(repeats=1, sampling={"temperature": 0.0})

    with pytest.raises(RuntimeError, match="as 8 ids"):
        suite._generate(str(tmp_path), ["question"])
    assert generated == []


#: The repository and revision whose tokenizer transformers 5 builds wrongly; pinned as the results are.
R1_REPO = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
R1_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"


@pytest.mark.gpu
def test_a_real_engine_refuses_the_r1_tokenization_before_generating(tmp_path, monkeypatch):
    """Stock vLLM builds the mismatched tokenizer for a tiny Llama carrying the pinned R1 metadata."""
    vllm = pytest.importorskip("vllm")
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import LlamaConfig, LlamaForCausalLM

    cached = try_to_load_from_cache(repo_id=R1_REPO, filename="tokenizer.json", revision=R1_REVISION)
    if not isinstance(cached, str):  # None (not cached) or _CACHED_NO_EXIST; read the cache, never download
        pytest.skip(f"needs {R1_REPO}@{R1_REVISION[:8]} in the local Hugging Face cache")
    snapshot = Path(cached).parent
    source = json.loads((snapshot / "config.json").read_text())
    model_dir = tmp_path / "tiny-r1"
    torch.manual_seed(0)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=source["vocab_size"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=source["bos_token_id"],
            eos_token_id=source["eos_token_id"],
            head_dim=32,
        )
    ).to(torch.bfloat16).save_pretrained(model_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (snapshot / name).is_file():
            shutil.copyfile(snapshot / name, model_dir / name)

    real_llm = vllm.LLM
    engines, generated = [], []

    def tracked_llm(**kwargs):
        engine = real_llm(**kwargs)
        engines.append(engine)
        actual = engine.generate

        def generate(*args, **options):
            generated.append(True)
            return actual(*args, **options)

        engine.generate = generate
        return engine

    monkeypatch.setattr(vllm, "LLM", tracked_llm)
    suite = LiveCodeBenchSuite(
        repeats=1,
        max_tokens=8,
        sampling={"temperature": 0.0},
        serve_args={
            "gpu_memory_utilization": 0.2,
            "max_model_len": 128,
            "enforce_eager": True,
            "kv_cache_memory_bytes": 64 * 1024 * 1024,
        },
    )
    try:
        with pytest.raises(RuntimeError) as error:
            suite._generate(str(model_dir), ["Write a function returning 42."])
        message = str(error.value)
        assert "as 8 ids" in message and "gives 11" in message, message
        assert "--serve-args" in message and '"tokenizer"' in message  # the remedy
        assert engines and generated == []
    finally:
        for engine in engines:
            engine.llm_engine.engine_core.shutdown()


def test_the_guard_compares_the_tokenizer_the_engine_actually_loads(monkeypatch, tmp_path):
    """A `tokenizer=` override is what vLLM loads, so the check reads that tokenizer.json, not the checkpoint."""
    import qda_kl.tokenizer_check as check

    calls = []

    def refusal(tokenizer, model_dir, **kwargs):  # noqa: ARG001 - the shared check's signature
        calls.append((model_dir, kwargs))
        return

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002 - the HF signature
            # HF's own contract: the rendered text when tokenize is off, ids otherwise.
            return "rendered prompt" if kwargs.get("tokenize") is False else {"input_ids": [1, 2]}

    class Engine:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, jobs, params, use_tqdm=False):  # noqa: ARG002 - vLLM's signature
            outputs = [types.SimpleNamespace(text="```python\nprint(42)\n```", finish_reason="stop")]
            return [types.SimpleNamespace(outputs=outputs) for _ in jobs]

    monkeypatch.setattr(check, "tokenizer_refusal", refusal)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=Engine, TokensPrompt=dict, SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs)
        ),
    )
    suite = LiveCodeBenchSuite(
        repeats=1,
        max_tokens=8,
        sampling={"temperature": 0.0},
        serve_args={"tokenizer": "/fixed/tokenizer", "revision": "abc123"},
    )

    suite._generate(str(tmp_path), ["first question", "second question"])

    assert [source for source, _ in calls] == ["/fixed/tokenizer"] * 2
    assert calls[0] == ("/fixed/tokenizer", {"revision": "abc123"})
    assert calls[1] == ("/fixed/tokenizer", {"revision": "abc123", "probe": "first question"})


def test_the_served_path_sends_the_in_process_prompt_ids_and_sampling(monkeypatch, tmp_path):
    """A served generation asks for the ids, seeds and sampler the in-process engine is given, per draw."""
    import qda_kl.eval.served as served
    import qda_kl.tokenizer_check as check

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("tokenize") is False:
                return "rendered question"
            return {"input_ids": [1, 2, 3]}

    class Engine:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, jobs, params, use_tqdm=False):  # noqa: ARG002 - vLLM's signature
            outputs = [types.SimpleNamespace(text="in process", finish_reason="stop")]
            return [types.SimpleNamespace(outputs=outputs) for _ in jobs]

    seen: list[dict] = []
    posted: list[dict] = []

    def fake_params(**kwargs):
        seen.append(kwargs)
        return types.SimpleNamespace(**kwargs)

    def fake_post(url, payload, **kwargs):  # noqa: ARG001 - the helper's signature
        posted.append(payload)
        return {"choices": [{"text": "served", "finish_reason": "length"}]}

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=Engine, TokensPrompt=dict, SamplingParams=fake_params),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
        ),
    )
    monkeypatch.setattr(check, "tokenizer_refusal", lambda *a, **k: None)
    monkeypatch.setattr(served, "post_completions", fake_post)

    suite = LiveCodeBenchSuite(repeats=2, max_tokens=16, sampling={"temperature": 0.7, "top_p": 0.9})
    in_answers, in_finishes, in_hashes = suite._generate(str(tmp_path), ["one", "two"])
    in_params, seen[:] = list(seen), []
    served_suite = suite.model_copy(update={"server": "http://127.0.0.1:9/v1/completions"})
    out_answers, out_finishes, out_hashes = served_suite._generate(str(tmp_path), ["one", "two"])

    assert out_hashes == in_hashes
    assert out_answers == [["served", "served"], ["served", "served"]]
    assert out_finishes == [["length", "length"], ["length", "length"]]  # kept, not inferred
    assert len(posted) == 4
    assert [p["prompt"] for p in posted] == [[1, 2, 3]] * 4
    assert [p["seed"] for p in posted] == [17, 18, 17, 18]  # the in-process seeds, per draw
    assert all(p["max_tokens"] == 16 for p in posted)
    assert all(p["skip_special_tokens"] is True for p in posted)  # no closing tag: vLLM's default
    # The sampler is the in-process object's: every field the request carries equals it, and vice versa.
    for in_params, payload in zip(seen, posted, strict=True):
        for name in ("temperature", "top_p", "max_tokens", "seed", "skip_special_tokens"):
            assert payload[name] == in_params[name]
        assert set(payload) - {"model", "prompt"} <= set(in_params) | {
            "max_tokens",
            "seed",
            "skip_special_tokens",
        }
    assert in_answers == [["in process", "in process"], ["in process", "in process"]]
    assert in_finishes == [["stop", "stop"], ["stop", "stop"]]


def test_a_closing_tag_served_cell_keeps_special_tokens_too(monkeypatch, tmp_path):
    """The tag the answer cut uses is a special token: the served request must ask for it as in-process does."""
    import qda_kl.eval.served as served
    import qda_kl.tokenizer_check as check

    posted: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: types.SimpleNamespace(
                    apply_chat_template=lambda messages, **kw: (
                        "rendered" if kw.get("tokenize") is False else {"input_ids": [1]}
                    )
                )
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **k: types.SimpleNamespace(**k))
    )
    monkeypatch.setattr(check, "tokenizer_refusal", lambda *a, **k: None)
    monkeypatch.setattr(
        served,
        "post_completions",
        lambda url, payload, **k: (
            posted.append(payload) or {"choices": [{"text": "x", "finish_reason": "stop"}]}
        ),
    )
    suite = LiveCodeBenchSuite(
        server="http://x/v1/completions", repeats=1, sampling={"temperature": 0.0}, think_end_token="</think>"
    )

    suite._generate(str(tmp_path), ["q"])

    assert posted[0]["skip_special_tokens"] is False


def test_a_served_cell_records_the_engine_it_read_from_the_server(monkeypatch):
    """The record names the server's own vLLM and model, and refuses a server serving another checkpoint."""
    import qda_kl.eval.served as served

    identity = {
        "url": "http://s",
        "vllm": "0.33.0",
        "model": {"id": "qda", "root": "/ckpt"},
        "models": ["qda"],
    }
    monkeypatch.setattr(served, "server_identity", lambda url, **k: identity)
    checked: list[tuple] = []
    monkeypatch.setattr(
        served,
        "check_server_identity",
        lambda ident, checkpoint, expected: checked.append((ident, checkpoint, expected)),
    )
    monkeypatch.setitem(
        sys.modules,
        "qda_kl.serve",
        types.SimpleNamespace(settings_for=lambda *a, **k: types.SimpleNamespace(vllm="0.33.0")),
    )
    suite = LiveCodeBenchSuite(server="http://s/v1/completions")

    assert suite._served_identity("/ckpt") == identity
    assert checked == [(identity, "/ckpt", "0.33.0")]


def test_the_server_flag_reaches_a_livecodebench_suite(monkeypatch, tmp_path):
    """`qda eval CONFIG --server URL` used to refuse any config without an lm_eval suite; it must not."""
    import yaml as yaml_module
    from qda_kl.eval import run as run_module
    from qda_kl.eval.config import EvalConfig

    (tmp_path / "ck").mkdir()
    cfg = EvalConfig.from_yaml(
        yaml_module.safe_dump(
            {
                "reference": str(tmp_path / "ck"),
                "checkpoints": {"one": str(tmp_path / "ck")},
                "out": str(tmp_path / "out"),
                "suites": [{"livecodebench": {"sampling": {"temperature": 0.0}}}],
            }
        )
    )
    monkeypatch.setattr(run_module, "jobs", lambda cfg, only: [])
    url = "http://127.0.0.1:9/v1/completions"

    assert run_module.run(cfg, (), server=url) == 0

    (suite,) = cfg.suites
    assert isinstance(suite, LiveCodeBenchSuite)
    assert suite.server == url


def test_the_lcb_sandbox_flag_writes_the_suites_own_key():
    """`qda-agent evaluate --lcb-sandbox` is the one place the choice is made: it lands in the config."""
    import argparse

    from qda_agent import evaluate
    from qda_kl.eval.presets import Group
    from qda_kl.eval.rundir import with_lcb_sandbox
    from qda_kl.eval.suites import parse_suite

    parser = argparse.ArgumentParser()
    evaluate.add_parser(parser.add_subparsers())
    assert parser.parse_args(["evaluate", "run"]).lcb_sandbox == "auto"
    chosen = parser.parse_args(["evaluate", "run", "--lcb-sandbox", "none"]).lcb_sandbox

    groups = (Group("full", ({"lm_eval": {"tasks": ["ifeval"]}}, {"livecodebench": {"protocol": "paper"}})),)
    (group,) = with_lcb_sandbox(groups, chosen)

    assert group.suites[0] == {"lm_eval": {"tasks": ["ifeval"]}}  # only the one suite is given the key
    assert group.suites[1] == {"livecodebench": {"protocol": "paper", "sandbox": "none"}}
    assert parse_suite(group.suites[1], "suites.1").sandbox == "none"  # a key the suite really reads
