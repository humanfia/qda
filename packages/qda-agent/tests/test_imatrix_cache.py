"""The importance-matrix cache: one measurement per key, a hit only with its record, a placed hard link.

CPU only: the stand-in `llama-imatrix` writes a few bytes where `-o` names.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from qda_agent import imatrix_cache as C

COMMIT = "9e2a04a439edc49f08c5acdafefae19dfa25a658"


def model(root: Path) -> Path:
    m = root / "model"
    m.mkdir(parents=True, exist_ok=True)
    (m / "config.json").write_text("{}")
    (m / "model.safetensors").write_bytes(b"\0" * 64)
    return m


def calib(root: Path) -> Path:
    p = root / "data" / "calib.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"text": "one"}\n{"text": "two"}\n')
    return p


def fake_program(root: Path, *, fails: bool = False) -> Path:
    """A stand-in `llama-imatrix`: writes its `-o` file and records every call beside itself."""
    p = root / "llama-imatrix"
    body = (
        "import sys\n"
        "args = sys.argv[1:]\n"
        "out = args[args.index('-o') + 1]\n"
        "open(out, 'wb').write(b'GGUF-imatrix')\n"
        "open(__file__ + '.calls', 'a').write(' '.join(args) + '\\n')\n"
        f"sys.exit({1 if fails else 0})\n"
    )
    p.write_text(f"#!{sys.executable}\n{body}")
    p.chmod(0o755)
    return p


def measure(tmp_path: Path, program: Path, **kw):
    args = {
        "program": program,
        "template": tmp_path / "template.gguf",
        "text": C.text_file(calib(tmp_path), tmp_path / "data" / C.TEXT),
        "model": model(tmp_path / "m"),
        "calib": calib(tmp_path),
        "commit": COMMIT,
        "context": 2048,
        "chunks": 128,
        "log": lambda _line: None,
    }
    args.update(kw)
    return C.cached(**args)


def test_the_key_covers_the_model_the_calibration_the_commit_and_the_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    m, c = model(tmp_path / "a"), calib(tmp_path)
    base = C.key_payload(m, c, COMMIT, context=2048, chunks=128, process_output=True)
    assert base["llama_cpp"] == COMMIT and base["settings"]["chunks"] == 128
    assert len(base["model"]) == 64
    assert C.entry(base).parent.name == base["model"][:16]
    for other in (
        C.key_payload(m, c, COMMIT[:-1] + "0", context=2048, chunks=128, process_output=True),
        C.key_payload(m, c, COMMIT, context=4096, chunks=128, process_output=True),
        C.key_payload(m, c, COMMIT, context=2048, chunks=64, process_output=True),
        C.key_payload(m, c, COMMIT, context=2048, chunks=128, process_output=False),
    ):
        assert C.entry(other) != C.entry(base)


def test_a_second_measurement_of_one_key_is_a_hit_and_runs_the_program_once(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    program = fake_program(tmp_path)
    first, built = measure(tmp_path, program)
    assert built and first.is_file() and first.read_bytes() == b"GGUF-imatrix"
    again, built_again = measure(tmp_path, program)
    assert again == first and built_again is False
    assert len(Path(str(program) + ".calls").read_text().splitlines()) == 1
    record = json.loads(C.record_path(first.parent).read_text())
    assert record["payload"]["llama_cpp"] == COMMIT and record["sha256"] == C.sha256_file(first)
    assert "-o" in record["command"] and "--process-output" in record["command"]


def test_a_file_without_its_record_or_with_other_bytes_is_not_a_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    program = fake_program(tmp_path)
    key = C.key_payload(
        model(tmp_path / "m"), calib(tmp_path), COMMIT, context=2048, chunks=128, process_output=True
    )
    target = C.entry(key)
    target.mkdir(parents=True)
    (target / C.FILE).write_bytes(b"GGUF-imatrix")  # no record: an unfinished measurement
    assert C.lookup(key) is None
    first, _ = measure(tmp_path, program)
    first.write_bytes(b"tampered")  # the record's sha no longer matches
    assert C.lookup(key) is None


def test_a_failed_measurement_raises_and_leaves_no_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    program = fake_program(tmp_path, fails=True)
    with pytest.raises(subprocess.CalledProcessError):
        measure(tmp_path, program)
    key = C.key_payload(
        model(tmp_path / "m"), calib(tmp_path), COMMIT, context=2048, chunks=128, process_output=True
    )
    assert C.lookup(key) is None


def test_placing_the_cached_file_is_a_hard_link_not_a_link_out_of_the_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    program = fake_program(tmp_path)
    cached, _ = measure(tmp_path, program)
    placed = C.place(cached, tmp_path / "out" / "imatrix" / "model.imatrix.gguf")
    assert placed.read_bytes() == b"GGUF-imatrix"
    assert not placed.is_symlink()
    assert placed.stat().st_ino == cached.stat().st_ino


def test_the_measurement_text_is_the_rendered_rows_joined_with_a_blank_line(tmp_path):
    out = C.text_file(calib(tmp_path), tmp_path / "text.txt")
    assert out.read_text() == "one\n\ntwo\n"
    unrendered = tmp_path / "bad.jsonl"
    unrendered.write_text('{"messages": []}\n')
    with pytest.raises(Exception, match="rendered calibration row"):
        C.text_file(unrendered, tmp_path / "x.txt")
