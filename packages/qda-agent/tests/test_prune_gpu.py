"""On a card: a frozen copy a floor pass reclaimed is frozen again from the agent's own files, vLLM
serves the re-frozen copy, and the final rule picks it.

A hand-measured hand-off drove a Qwen3-0.6B search by hand; this is the same chain as a test.
A tiny randomly initialized Qwen3 checkpoint stands in for the build: the test asserts structure (which
directory exists, which entry is picked), never a KL value. The `kl` scorer is a stand-in writing a reading;
the serving check is the judge's real `serve_check.py`, so a real vLLM engine opens the re-frozen directory.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm", reason="the check serves the re-frozen copy with vLLM")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card"),
    pytest.mark.skipif(shutil.which("bwrap") is None, reason="the judge copies the checkpoint with bwrap"),
]

from qda_agent import rpc  # noqa: E402
from qda_agent.judge import make_host  # noqa: E402
from qda_agent.judge.freeze import walk  # noqa: E402
from qda_agent.sched import SchedConfig, SystemClock  # noqa: E402
from test_judge import KL_RESULT, _ref_dir  # noqa: E402

SERVE_CHECK = Path(__file__).parents[1] / "src" / "qda_agent" / "judge" / "tools" / "serve_check.py"
SCORE = (
    "import json, sys; from pathlib import Path; "
    "kl = json.loads((Path(sys.argv[1]) / 'build.json').read_text())['test_kl']; "
    "Path(sys.argv[2]).write_text(json.dumps({**json.loads(sys.argv[3]), 'kl': kl}))"
)
WAIT_S = 600


def tiny_model(d: Path, seed: int, kl: float) -> Path:
    """A tiny servable Qwen3 checkpoint with a word-level tokenizer; `build.json` carries its test KL."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
    )
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    d.mkdir(parents=True)
    model.save_pretrained(d, safe_serialization=True)
    vocab = {f"w{i}": i for i in range(127)} | {"[UNK]": 127}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = "{% for m in messages %}w1 {{ m['content'] }} w2 {% endfor %}"
    fast.save_pretrained(d)
    (d / "build.json").write_text(json.dumps({"serve": {}, "test_kl": kl}))
    return d


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    """A judge host on the first visible card with the real serving check and a stand-in KL scorer."""
    root = tmp_path_factory.mktemp("prune-gpu")
    run, work = root / "run", root / "run" / "work"
    score = root / "score.py"
    score.write_text(SCORE)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    card = int(visible.split(",")[0]) if visible else 0  # `cards` are physical numbers, not indices
    judge = {
        "kl_ref": str(_ref_dir(root)),
        "kl_holdout_ref": str(_ref_dir(root)),
        "contamination": "none",
        "engine": "vllm",
        "serve_python": sys.executable,
        "kl_cmd": [sys.executable, str(score), "{ckpt}", "{out}", json.dumps(KL_RESULT)],
        "kl_score_s": 10,
        "kl_load_s_per_gib": 0,
        "check_cmd": [sys.executable, str(SERVE_CHECK), "{ckpt}", "{out}"],
        "check_expected_s": 600,
        "max_kl": 1.0,
        "allow_any_format": True,
        "format_menu": ["bf16"],
        "min_free_bytes": 0,
        "qda_cmd": [sys.executable, "-m", "qda_quant.cli"],
    }
    h = make_host(
        run,
        SchedConfig(cards=[card], budget_s=7200, fake_gpus=False),
        judge,
        SystemClock(),
        autotick=False,
    )
    try:
        yield h, work
    finally:
        h.stop()


def wait_entry(host, eid: str):
    """Tick the host (which launches and reaps jobs) until entry `eid` has a verdict; return it."""
    end = time.time() + WAIT_S
    while time.time() < end:
        host.tick()
        entry = next((e for e in host.entries if e.id == eid), None)
        if entry is not None and entry.valid is not None:
            return entry
        time.sleep(1)
    entry = next((e for e in host.entries if e.id == eid), None)
    pytest.fail(f"{eid} did not end in {WAIT_S}s: {entry}")


def call(host, work: Path, verb: str, *argv: str):
    return host.handle(rpc.Request(verb, list(argv), str(work)))


def weights(tree: Path) -> list[Path]:
    return [p for p in walk(tree) if p.suffix == ".safetensors"]


def test_a_reclaimed_copy_is_frozen_again_and_vllm_serves_it(host):
    h, work = host
    tiny_model(work / "ckpt" / "A", seed=1, kl=0.2)
    tiny_model(work / "ckpt" / "B", seed=2, kl=0.1)  # B reads better and is handed in: the pick
    for name in ("A", "B"):
        reply = call(h, work, "kl", f"ckpt/{name}")
        assert reply.exit == 0, reply.stderr
    wait_entry(h, "k1")
    wait_entry(h, "k2")
    reply = call(h, work, "handin", "ckpt/B")
    assert reply.exit == 0, reply.stderr
    handed_b = wait_entry(h, "h1")
    assert handed_b.valid is True
    first = next(e.dir for e in h.entries if e.id == "k1")  # A's copy, scored but never handed in
    assert weights(first)

    # The next build would not fit above the floor: the pass frees A's copy (B is the final pick). A copy with
    # no build receipt is freed by removing its weight files and keeping its metadata (`prune`'s rule for a
    # superseded frozen copy), so "freed" is no weights left, not no directory; the re-freeze below must then
    # ignore that stripped directory (`intake.frozen_for`) and make a whole new copy.
    h.cfg.min_free_bytes = _free_bytes(work) - 1
    h.tick()
    h.tick()
    assert not weights(first), "the scored build A is not handed in and not the pick: its copy is freed"
    assert next(e.dir for e in h.entries if e.id == "k2").is_dir()
    h.cfg.min_free_bytes = 0  # room again: the re-hand-in and its check run normally

    reply = call(h, work, "handin", "ckpt/A")
    assert reply.exit == 0, reply.stderr
    again = wait_entry(h, "h2")
    assert again.valid is True, again.reason
    assert again.dir != first and again.dir.is_dir() and weights(again.dir)
    assert read_json(again.dir / "build.json")["test_kl"] == 0.2  # the agent's own files, frozen again
    assert not any(
        "Cannot find any model weights" in (h.sched.log(j.name, 200) or "") for j in h.sched.list()
    )
    frozen = {p.lstat().st_ino for p in weights(again.dir)}
    assert all(p.lstat().st_nlink == 1 for p in weights(again.dir))  # a copy, not another link
    assert frozen

    final = h.final(generation_checks=False)
    assert final.choice is not None and final.choice.id == "h2", final.reason
    assert final.choice.metrics.get("kl") == 0.2  # the reading of the same files, matched by digest


def _free_bytes(path: Path) -> int:
    import os

    info = os.statvfs(str(path))
    return info.f_bavail * info.f_frsize


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())
