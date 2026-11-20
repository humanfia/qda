"""Checkpoint gate and fingerprint cases on synthetic safetensors files."""

import json
import random
import shutil
import struct
from pathlib import Path

import pytest
from qda_agent.judge import checks
from qda_agent.judge.checks import (
    CheckpointError,
    files_problem,
    inventory,
    tensor_fingerprint,
)
from safetensors_kit import CONFIG, QCONFIG, write_safetensors


# ------------------------------------------------------------------ ported: test_judge.py (fingerprint, index, headers)
def read_st(p: Path) -> tuple[dict, dict]:
    b = p.read_bytes()
    n = int.from_bytes(b[:8], "little")
    h = json.loads(b[8 : 8 + n])
    h.pop("__metadata__", None)
    return h, {k: b[8 + n + v["data_offsets"][0] : 8 + n + v["data_offsets"][1]] for k, v in h.items()}


def write_st(p: Path, header: dict, data: dict, meta: dict | None) -> None:
    write_safetensors(p, {k: (v["dtype"], tuple(v["shape"]), data[k]) for k, v in header.items()}, meta)


@pytest.fixture
def base1(tmp_path):
    d = tmp_path / "base1"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({**CONFIG, "quantization_config": QCONFIG}))
    write_safetensors(
        d / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (4, 2), bytes(range(16))),
            "model.layers.0.q.weight_packed": ("I32", (2, 1), b"\x01\0\0\0\x02\0\0\0"),
            "model.layers.0.q.weight_scale": ("BF16", (2, 1), b"\x03\0\x04\0"),
            "lm_head.weight": ("BF16", (4, 2), bytes(range(16, 32))),
        },
    )
    return d


def test_the_fingerprint_is_of_what_is_loaded(tmp_path, base1):
    h, data = read_st(base1 / "model.safetensors")
    names = list(h)

    def made(name, shards, index):
        d = tmp_path / name
        shutil.copytree(base1, d)
        (d / "model.safetensors").unlink()
        for f, (part, vals) in shards.items():
            write_st(d / f, {k: h[k] for k in part}, {k: vals.get(k, data[k]) for k in part}, None)
        if index is not None:
            (d / checks.INDEX).write_text(json.dumps({"weight_map": index}))
        return d

    split = {"a.safetensors": (names[:2], {}), "b.safetensors": (names[2:], {})}
    wm = {k: "a.safetensors" if k in names[:2] else "b.safetensors" for k in names}
    good = made("good", split, wm)
    vis = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}
    h["model.visual.patch.weight"], data["model.visual.patch.weight"] = vis, b"\x01" * 8
    seen = made(
        "seen",
        {
            "a.safetensors": ([*names[:2], "model.visual.patch.weight"], {}),
            "b.safetensors": split["b.safetensors"],
        },
        {**wm, "model.visual.patch.weight": "a.safetensors"},
    )
    other_vis = made(
        "other-vis",
        {
            "a.safetensors": (
                [*names[:2], "model.visual.patch.weight"],
                {"model.visual.patch.weight": b"\x02" * 8},
            ),
            "b.safetensors": split["b.safetensors"],
        },
        {**wm, "model.visual.patch.weight": "a.safetensors"},
    )
    assert {tensor_fingerprint(seen), tensor_fingerprint(other_vis)} == {tensor_fingerprint(good)}
    for vt in ("model.vision_tower.encoder.weight", "model.embed_vision.embedding_projection.weight"):
        h[vt], data[vt] = vis, b"\x03" * 8
        g4 = made(
            vt,
            {"a.safetensors": ([*names[:2], vt], {}), "b.safetensors": split["b.safetensors"]},
            {**wm, vt: "a.safetensors"},
        )
        assert tensor_fingerprint(g4) == tensor_fingerprint(good), vt
    assert tensor_fingerprint(seen, ()) != tensor_fingerprint(other_vis, ())
    assert tensor_fingerprint(good) == tensor_fingerprint(base1)  # re-sharded: the same
    other = {names[0]: bytes(len(data[names[0]]))}
    for name, shards, index, why in (
        ("unnamed", {**split, "c.safetensors": ([names[0]], other)}, wm, "a shard the index does not name"),
        (
            "ghost",
            split,
            {**wm, "mtp.fc.weight": "a.safetensors"},
            "names 'mtp.fc.weight', but no shard holds it",
        ),
        (
            "twice",
            {**split, "c.safetensors": ([names[0]], other)},
            {**wm, names[0]: "c.safetensors"},
            "the index maps it to 'c.safetensors'",
        ),
        ("elsewhere", split, {**wm, names[0]: "b.safetensors"}, "the index maps it to 'b.safetensors'"),
        (
            "twice-no-index",
            {**split, "c.safetensors": ([names[0]], other)},
            None,
            "in a.safetensors and c.safetensors",
        ),
    ):
        d = made(name, shards, index)
        with pytest.raises(ValueError, match=why):
            tensor_fingerprint(d)
        with pytest.raises(ValueError, match=why):
            inventory(d)
        assert why in files_problem(d), name


def test_header_records_must_describe_their_stored_bytes(tmp_path):
    """A dtype, shape and byte length that do not agree is refused, as the stock reader refuses it."""
    d = tmp_path / "hdr"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "llama"}')
    path = d / "model.safetensors"

    def write(dtype, shape, raw):
        header = json.dumps({"x.weight": {"dtype": dtype, "shape": shape, "data_offsets": [0, len(raw)]}})
        header = header.encode()
        header += b" " * (-len(header) % 8)
        path.write_bytes(struct.pack("<Q", len(header)) + header + raw)

    for dtype, shape, raw in (
        ("BF16", [1, 1], b"xx"),
        ("I32", [1], bytes(4)),
        ("F32", [], bytes(4)),
        ("F32", [0, 3], b""),
        ("F8_E4M3", [2], bytes(2)),
        ("F8_E5M2", [2], bytes(2)),
    ):
        write(dtype, shape, raw)
        assert files_problem(d) is None, (dtype, shape, files_problem(d))
    for dtype, shape, raw in (("BF16", [1, 1], b""), ("BF16", [-1, 1], b"xx"), ("BOGUS", [1, 1], b"xx")):
        write(dtype, shape, raw)
        why = files_problem(d)
        assert why is not None and "x.weight" in why, (dtype, shape, why)


def test_a_bad_safetensors_header_is_a_clean_error(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    for h in (
        [],
        {"t": None},
        {"t": {"dtype": "U8", "shape": [1], "data_offsets": [10**100, 10**100 + 1]}},
        {"t": {"dtype": "U8", "shape": [1], "data_offsets": [1000, 1001]}},
        {"t": {"dtype": "U8", "shape": [1], "data_offsets": [1, 0]}},
        {"t": {"data_offsets": [0, 0]}},
    ):
        j = json.dumps(h).encode()
        (d / "model.safetensors").write_bytes(len(j).to_bytes(8, "little") + j + b"x")
        with pytest.raises(ValueError):
            tensor_fingerprint(d)
    j = b'{"t":' + b"[" * 10000 + b"]" * 10000 + b"}"
    (d / "model.safetensors").write_bytes(len(j).to_bytes(8, "little") + j)
    with pytest.raises(CheckpointError):
        tensor_fingerprint(d)


def test_fingerprint_multi_shard_many_tensors(tmp_path):
    rnd = random.Random(7)
    d = tmp_path / "multi"
    d.mkdir()
    wm, all_ = {}, {}
    for si in range(3):
        t = {}
        for i in range(200):
            n = rnd.choice((1, 2, 3, 64, 4093))
            t[f"s{si}.t{i}"] = ("U8", (n,), rnd.randbytes(n))
        t[f"s{si}.big"] = ("U8", (1 << 20,), rnd.randbytes(1 << 20))
        name = f"model-0000{si + 1}-of-00003.safetensors"
        write_safetensors(d / name, t, {"format": "pt"})
        wm.update({k: name for k in t})
        all_.update(t)
    (d / checks.INDEX).write_text(json.dumps({"weight_map": wm}))
    one = tmp_path / "one"
    one.mkdir()
    write_safetensors(one / "model.safetensors", dict(sorted(all_.items())))
    for uns in (checks.UNSCORED, ("s1.*",), ("s*.t1*",)):
        assert tensor_fingerprint(d, uns) == tensor_fingerprint(one, uns), uns


def test_qda_tool_records_are_inert(tmp_path, base1):
    d = tmp_path / "rec"
    shutil.copytree(base1, d)
    for n in ("moefast.json", "sym4-record.json", "scaletune.json", "splice-assign.json"):
        (d / n).write_text("{}")
    assert files_problem(d) is None
    (d / "other.json").write_text("{}")
    assert "other.json" in files_problem(d)


def test_files_problem_cases(tmp_path, base1):
    """The old judge's files-check cases: only `good` and `inert` pass."""
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "config.json").write_text(json.dumps(CONFIG))
    (anchor / "tokenizer.json").write_text("{}")
    (anchor / "generation_config.json").write_text('{"a": 1}')
    shutil.copy(anchor / "tokenizer.json", base1)
    shutil.copy(anchor / "generation_config.json", base1)
    cases = {
        "good": {},
        "code": {"modeling.py": "import os"},
        "pickle": {"x.bin": "x"},
        "inert": {"README.md": "x", "LICENSE.txt": "x"},
        "unknown": {"params.json": "{}"},
        "tokenizer": {"tokenizer.json": '{"vocab": "mine"}'},
        "gen": {"generation_config.json": "{}"},
        "auto_map": {"config.json": json.dumps({**CONFIG, "quantization_config": QCONFIG, "auto_map": {}})},
        "config": {"config.json": json.dumps({"model_type": "mistral"})},
        "index": {"model.safetensors.index.json": json.dumps({"weight_map": {"a": "../x.safetensors"}})},
    }
    said = {}
    for name, extra in cases.items():
        d = tmp_path / name
        shutil.copytree(base1, d)
        for f, text in extra.items():
            (d / f).write_text(text)
        said[name] = files_problem(d, anchor, base1)
    assert [n for n, x in said.items() if x is None] == ["good", "inert"], said
