"""Multi-card builds start the card processes before the plan; a refusal tears them down (CPU only)."""

from __future__ import annotations

import errno
import importlib
import json
import os
import resource
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import torch.multiprocessing as mp
from qda_quant.engine.ranks import raise_open_files
from qda_quant.exit import ENVIRONMENT
from qda_quant.output.exporters import EXPORTERS, CompressedTensors
from qda_quant.step import Refused
from qda_quant.steps.gptq import GPTQ

from ..engine_util import save_tiny_tokenizer, tiny_qwen3, tiny_qwen3_moe, write_ids

build_mod = importlib.import_module(
    "qda_quant.engine.build"
)  # the module (`engine.build` is also the function)


@pytest.fixture(autouse=True)
def _cpu_host_driver(monkeypatch):
    """These builds stand CPU tensors in for named cards: this host's CUDA driver is not theirs to judge."""
    from qda_quant import checks

    monkeypatch.setattr(checks, "cuda_driver_problem", lambda **k: None)


CPU2 = [torch.device("cpu")] * 2
RECIPE = "steps: [{rtn: {}}]\ncalib: {file: {path: rows.jsonl, samples: 2, seqlen: 8}}\n"
#: A recipe whose step asks for statistics: its calibration forward is what a multi-card build shards.
NEEDS_RECIPE = "steps: [{gptq: {}}]\ncalib: {file: {path: rows.jsonl, samples: 2, seqlen: 8}}\n"


def _record_rank(rank: int, devices: list, queues: list, init_file: str, nccl_nvls: bool) -> None:
    """Stands in for `_rank_main`: waits for the job (a directory) and writes this rank's file into it."""
    job = queues[rank].get()
    if job is None:
        return
    if job == "fail" and rank == 1:
        raise RuntimeError("rank 1 failed")
    if job != "fail":
        (Path(job) / f"rank{rank}").write_text(init_file)


def _hang_rank(rank: int, devices: list, queues: list, init_file: str, nccl_nvls: bool) -> None:
    """A rank that never reads its queue (stuck in setup)."""
    time.sleep(600)


def _die_rank(rank: int, devices: list, queues: list, init_file: str, nccl_nvls: bool) -> None:
    """A rank that fails in setup (NCCL init, OOM, an import) before it reads its job."""
    raise RuntimeError(f"rank {rank} died in setup")


def _exit_codes(ranks: build_mod._Ranks) -> list[int | None]:
    return [p.exitcode for p in ranks.procs.processes]


def test_ranks_run_hands_every_rank_the_job(tmp_path):
    ranks = build_mod._Ranks(CPU2, fn=_record_rank)
    ranks.run(str(tmp_path))
    ranks.close()  # after run: a no-op
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rank0", "rank1"]
    assert _exit_codes(ranks) == [0, 0]


def test_ranks_close_without_a_job_exits_cleanly(tmp_path):
    ranks = build_mod._Ranks(CPU2, fn=_record_rank)
    ranks.close()
    ranks.close()  # safe twice
    assert _exit_codes(ranks) == [0, 0]
    assert not list(tmp_path.iterdir())


def test_ranks_run_raises_a_rank_error():
    ranks = build_mod._Ranks(CPU2, fn=_record_rank)
    with pytest.raises(mp.ProcessRaisedException, match="rank 1 failed"):
        ranks.run("fail")
    ranks.close()
    assert not any(p.is_alive() for p in ranks.procs.processes)


def test_ranks_run_raises_when_ranks_died_before_a_large_job():
    """A job over the pipe buffer must not block on dead ranks: the rank's error is raised."""
    ranks = build_mod._Ranks(CPU2, fn=_die_rank)
    time.sleep(2.0)  # both ranks are dead before the job is sent
    got: list[BaseException] = []

    def go() -> None:
        try:
            ranks.run("x" * 1_000_000)
        except BaseException as e:  # noqa: BLE001
            got.append(e)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive(), "run() hung on ranks that died before reading the job"
    assert len(got) == 1 and isinstance(got[0], mp.ProcessRaisedException)
    assert "died in setup" in str(got[0])
    ranks.close()


def test_ranks_close_terminates_a_stuck_rank():
    ranks = build_mod._Ranks(CPU2, fn=_hang_rank)
    ranks.close(timeout=1.0)
    assert not any(p.is_alive() for p in ranks.procs.processes)


class _FakeRanks:
    """Records when `build` starts and closes the card processes."""

    events: list[str] = []

    def __init__(self, devices, **kwargs) -> None:
        self.events.append(f"start {len(devices)} nvls={kwargs.get('nccl_nvls', False)}")

    def run(self, job) -> None:
        self.events.append("run")

    def close(self) -> None:
        self.events.append("close")


@pytest.fixture
def fake_ranks(monkeypatch):
    _FakeRanks.events = []
    monkeypatch.setattr(build_mod, "_Ranks", _FakeRanks)
    return _FakeRanks.events


@pytest.mark.parametrize("nvls", [False, True])
@pytest.mark.usefixtures("python_headers")  # the build compiles Triton's driver for the faked cards
def test_build_starts_ranks_before_the_plan_and_closes_them_on_a_refusal(
    fake_ranks, monkeypatch, tmp_path, nvls
):
    def refuse(*args, **kwargs):
        fake_ranks.append("plan")
        raise Refused("not on this target")

    monkeypatch.setattr(build_mod, "compile_plan", refuse)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)  # the two named cards are visible
    tiny_qwen3(layers=1).save_pretrained(tmp_path / "m")  # a step asking for statistics keeps both cards
    with pytest.raises(Refused, match="not on this target"):
        build_mod.build(
            str(tmp_path / "m"), NEEDS_RECIPE, format="w4a16-sym-g32", out_dir=tmp_path / "o",
            devices=["cuda:0", "cuda:1"], nccl_nvls=nvls,
        )  # fmt: skip
    assert fake_ranks == [f"start 2 nvls={nvls}", "plan", "close"]


@pytest.mark.usefixtures("python_headers")  # the build compiles Triton's driver for the faked cards
def test_a_recipe_of_layer_steps_with_no_statistics_runs_on_one_card(fake_ranks, monkeypatch, tmp_path):
    """RTN on minmax has no forward pass to shard: several cards run one process on the first of them."""
    planned: list[list] = []

    def refuse(source, recipe, fmt, target, devices, *args, **kwargs):
        planned.append(devices)
        raise Refused("stop after the plan")

    monkeypatch.setattr(build_mod, "compile_plan", refuse)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)  # the two named cards are visible
    tiny_qwen3(layers=1).save_pretrained(tmp_path / "m")
    log: list[str] = []
    with pytest.raises(Refused, match="stop after the plan"):
        build_mod.build(
            str(tmp_path / "m"), RECIPE, format="w4a16-sym-g32", out_dir=tmp_path / "o",
            devices=["cuda:0", "cuda:1"], log=log.append,
        )  # fmt: skip
    assert fake_ranks == []  # no card process started: there is no sharded work
    assert [len(devices) for devices in planned] == [1]
    assert "one process on cuda:0" in "\n".join(log)


def test_a_no_statistics_build_on_two_devices_writes_the_one_device_bytes(tmp_path):
    """RTN on minmax: the second cpu "card" changes neither the checkpoint nor the recorded card count."""
    import yaml
    from qda_quant.calib.sources import FileSource
    from qda_quant.recipe import Recipe
    from qda_quant.steps.rtn import RTN
    from safetensors.torch import load_file

    m = tiny_qwen3(layers=1)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [[1, 2, 3, 4, 5, 6, 7, 8]] * 2, sha)
    calib = FileSource(path=str(tmp_path / "c.jsonl"), samples=2, seqlen=8)
    weights, cards, said = {}, {}, []
    for tag, devices in (("one", ["cpu"]), ("two", ["cpu", "cpu"])):
        out = build_mod.build(
            str(tmp_path / "m"), Recipe(steps=[RTN()], calib=calib), format="w4a16-sym-g32",
            out_dir=tmp_path / tag, devices=devices, dtype=torch.float32, log=said.append,
        )  # fmt: skip
        weights[tag] = {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}
        cards[tag] = yaml.safe_load((out / "resolved.yaml").read_text())["provenance"]["cards"]
    assert cards == {"one": 1, "two": 1}  # `resolved.yaml` names the one card the build ran on
    assert "one process on cpu" in "\n".join(said)
    a, b = weights["one"], weights["two"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_rank_main_sets_nvls_from_the_setting_before_it_joins(monkeypatch, tmp_path):
    """The card process sets NCCL_NVLS_ENABLE from the setting the build passes it (gloo, world 1 here)."""
    from qda_quant.engine.ranks import _rank_main

    monkeypatch.setenv("NCCL_NVLS_ENABLE", "x")
    for nvls, want in ((False, "0"), (True, "1")):
        q = mp.get_context("spawn").SimpleQueue()
        q.put(None)  # no job: the rank leaves the group and returns
        _rank_main(0, [torch.device("cpu")], [q], str(tmp_path / f"store-{nvls}"), nvls)
        assert os.environ["NCCL_NVLS_ENABLE"] == want


class WholeGPTQ(GPTQ):
    """GPTQ with the routed-expert split off: rank 0 solves every bank (the reference for the split's bytes).

    Module level: spawn pickles the recipe.
    """

    name = "gptq_whole_cpu_test"
    splits_experts: ClassVar[bool] = False


def _saved_moe(tmp_path: Path) -> Path:
    """A tiny fused-MoE model and 8 calibration rows (8 batches of one: enough for 8 cards)."""
    m = tiny_qwen3_moe(layers=1)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    g = torch.Generator().manual_seed(0)
    rows = [torch.randint(0, 128, (8,), generator=g).tolist() for _ in range(8)]
    write_ids(tmp_path / "c.jsonl", rows, sha)
    return tmp_path / "m"


class PreparingTensors(CompressedTensors):
    """Records the process every `prepare` runs in: the parent's call does not reach a spawned rank."""

    name = "preparing_tensors_cpu_test"
    marker: str

    def prepare(self, recipe=None) -> str | None:
        with Path(self.marker).open("a") as f:
            f.write(f"{os.getpid()}\n")
        return super().prepare()


EXPORTERS.register(PreparingTensors.name, PreparingTensors)


def test_every_card_process_prepares_the_exporter(tmp_path):
    """Each rank quantizes in its own process, so it runs the exporter's process-wide setup before the job.

    The step asks for statistics, so the build is sharded across the cards; a recipe without them runs in one
    process (`test_a_recipe_of_layer_steps_with_no_statistics_runs_on_one_card`).
    """
    from qda_quant.calib.sources import FileSource
    from qda_quant.recipe import Recipe

    m = tiny_qwen3(layers=1)
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [[1, 2, 3, 4, 5, 6, 7, 8]] * 2, sha)
    marker = tmp_path / "prepared"
    calib = FileSource(path=str(tmp_path / "c.jsonl"), samples=2, seqlen=8)
    build_mod.build(
        str(tmp_path / "m"), Recipe(steps=[GPTQ()], calib=calib), format="w4a16-sym-g32",
        out_dir=tmp_path / "out", devices=["cpu", "cpu"], dtype=torch.float32,
        exporter=PreparingTensors(marker=str(marker)), log=lambda _: None,
    )  # fmt: skip
    pids = marker.read_text().split()
    assert len(pids) == 3, pids  # the parent and both ranks
    assert len(set(pids)) == 3 and str(os.getpid()) in pids


@pytest.mark.parametrize(("cards", "blend"), [(3, 0.0), (8, 0.0), (3, 0.25)])
def test_expert_split_bytes_equal_rank0_solve_on_cpu(tmp_path, cards, blend):
    """The real `_Ranks` -> `_run` -> `Comm` pipeline on gloo, at N=3 and 8, stores the same checkpoint byte
    for byte whether the routed-expert solve is split (GPTQ) or rank 0 solves every bank (same statistics, so
    only the split differs). The tiny MoE has 8 experts per bank, so at 8 cards every rank gets one. With an
    all-token blend every rank reads the whole bank's all-token statistics.
    """
    from qda_quant.calib.sources import FileSource
    from qda_quant.recipe import Recipe
    from safetensors.torch import load_file

    src = _saved_moe(tmp_path)
    calib = FileSource(path=str(tmp_path / "c.jsonl"), samples=8, seqlen=8)
    got = {}
    for tag, step in (("split", GPTQ(all_token_blend=blend)), ("whole", WholeGPTQ(all_token_blend=blend))):
        out = build_mod.build(
            str(src), Recipe(steps=[step], calib=calib), format="w4a16-sym-g32", out_dir=tmp_path / tag,
            devices=["cpu"] * cards, dtype=torch.float32, log=lambda _: None,
        )  # fmt: skip
        phases = json.loads((out / "build.json").read_text())["timings"]["phases"]
        assert ("gather" in phases) is (tag == "split"), (
            tag,
            sorted(phases),
        )  # the split ran only where meant
        got[tag] = {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}
    a, b = got["split"], got["whole"]
    assert a.keys() == b.keys() and any("experts" in k for k in a)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _saved_dense(tmp_path: Path) -> None:
    m = tiny_qwen3(layers=1)  # 4 calibration batches: one a card at 4 cards, two at 2
    m.save_pretrained(tmp_path / "m")
    sha = save_tiny_tokenizer(tmp_path / "m")
    write_ids(tmp_path / "c.jsonl", [list(range(i, i + 8)) for i in (1, 2, 3, 4)], sha)


def _awq_rtn_bytes(
    tmp_path: Path, devices: list[str], dtype: torch.dtype, samples: int = 4
) -> dict[str, torch.Tensor]:
    """An AWQ + RTN build on `devices`; the checkpoint's tensors, by name."""
    from qda_quant.calib.sources import FileSource
    from qda_quant.recipe import Recipe
    from qda_quant.steps.awq import AWQ
    from qda_quant.steps.rtn import RTN
    from safetensors.torch import load_file

    calib = FileSource(path=str(tmp_path / "c.jsonl"), samples=samples, seqlen=8)
    out = build_mod.build(
        str(tmp_path / "m"), Recipe(steps=[AWQ(), RTN()], calib=calib), format="w4a16-sym-g32",
        out_dir=tmp_path / f"awq-{len(devices)}", devices=devices, dtype=dtype, log=lambda _: None,
    )  # fmt: skip
    return {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}


@pytest.mark.parametrize("cards", [2, 3, 4])
def test_awq_multicard_bytes_equal_on_cpu(tmp_path, cards):
    """AWQ's parent objective re-runs attention over every calibration batch; a multi-card build
    gathers the ranks' batches in rank order, so the search prices the rows a one-card build holds and
    writes the same checkpoint byte for byte. The tiny dense model has no summed cross-rank statistic.
    Three cards over four batches give a longer and a shorter shard, so the gather's pad is exercised too
    (the batches come back as views into each rank's gathered rows, not a second concatenation).
    """
    _saved_dense(tmp_path)
    a = _awq_rtn_bytes(tmp_path, ["cpu"], torch.float32)
    b = _awq_rtn_bytes(tmp_path, ["cpu"] * cards, torch.float32)
    assert a.keys() == b.keys() and any(k.endswith(".weight_scale") for k in a)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _rtn_gsq_bytes(tmp_path: Path, devices: list[str], samples: int = 4) -> dict[str, torch.Tensor]:
    """An RTN + GSQ build on `devices`; the checkpoint's tensors, by name. float32 for determinism.

    `samples` batches of one row: an epoch takes all of them as a single Lion step, so a row-shared build
    prices the same rows as a one-card build (each rank its shard, the gradients averaged) and the two can be
    compared step for step.
    """
    from qda_quant.calib.sources import FileSource
    from qda_quant.recipe import Recipe
    from qda_quant.steps.gsq import GSQ
    from qda_quant.steps.rtn import RTN
    from safetensors.torch import load_file

    m = tmp_path / "m"
    if not m.exists():
        tiny_qwen3(layers=1).save_pretrained(m)
        sha = save_tiny_tokenizer(m)
        g = torch.Generator().manual_seed(0)
        rows = [torch.randint(0, 128, (8,), generator=g).tolist() for _ in range(8)]
        write_ids(tmp_path / "c8.jsonl", rows, sha)
    calib = FileSource(path=str(tmp_path / "c8.jsonl"), samples=samples, seqlen=8)
    step = GSQ(epochs=1, batches=samples, validation=0, logits_dtype="float32")
    out = build_mod.build(
        str(m), Recipe(steps=[RTN(), step], calib=calib), format="w4a16-sym-g32",
        out_dir=tmp_path / f"gsq-{len(devices)}", devices=devices, dtype=torch.float32, log=lambda _: None,
    )  # fmt: skip
    return {k: v for f in sorted(out.glob("*.safetensors")) for k, v in load_file(f).items()}


@pytest.mark.parametrize("cards", [2, 3, 4])
def test_gsq_multicard_matches_one_card_on_cpu(tmp_path, cards):
    """Data-parallel GSQ, as upstream runs it (`src/trainer.py` 03fc16484c): every rank trains on its own
    contiguous calibration shard and the Lion step is the mean of the ranks' gradients (`average_grads`),
    each starting from rank 0's codes (`broadcast_qparams`). With one epoch over the same 8 rows the
    gradient is the one-card gradient (up to fp sum order), so the served codes agree.

    Before 2624ed37 a `layer_io` step was refused on >1 card at all; that commit gathered every rank's rows
    onto rank 0 (correct, not faster). This is the faster path: the codes are compared, not just the build.
    """
    a = _rtn_gsq_bytes(tmp_path, ["cpu"])
    b = _rtn_gsq_bytes(tmp_path, ["cpu"] * cards)
    assert a.keys() == b.keys() and any(k.endswith(".weight_scale") for k in a)
    for k in a:
        assert a[k].shape == b[k].shape
        if a[k].is_floating_point():
            # scales take the gradient through a different sum order: compare within fp32 tolerance
            assert torch.allclose(a[k], b[k], rtol=1e-2, atol=0), (
                k,
                float((a[k] - b[k]).abs().max()),
            )
        else:
            same = int((a[k] == b[k]).sum())
            # a code at Lion's sign boundary can flip when the gradient sums in another order: the fit is a
            # mean of the same rows, not the same fp addition, so a few codes differ (upstream's DP too)
            assert same >= a[k].numel() * 0.99, (k, f"{same}/{a[k].numel()} codes equal")


def test_a_row_sharing_step_runs_on_every_rank(tmp_path):
    """AWQ declares `shares_rows` on a dense layer: each rank prices its own calibration batches and the
    per-batch sums are all gathered in calibration order (`Comm.all_rows`), so rank 1 runs the steps too.

    Before, only rank 0 ran them, over every rank's gathered rows: rank 1's `steps` phase is the tell.
    """
    _saved_dense(tmp_path)
    _awq_rtn_bytes(tmp_path, ["cpu", "cpu"], torch.float32)
    rec = json.loads((tmp_path / "awq-2" / "build.json").read_text())
    ranks = rec["rank_timings"]
    assert len(ranks) == 2
    assert all(r["phases"].get("steps", 0.0) > 0 for r in ranks), ranks


def test_awq_multicard_bytes_equal_on_moe_cpu(tmp_path):
    """A fused-expert model: each bank's all-token inputs stay one tensor, the routed Hessians are summed in
    rank order, and AWQ + RTN still writes the one-card checkpoint (RTN reads no Hessian, so no order shows).
    """
    _saved_moe(tmp_path)
    a = _awq_rtn_bytes(tmp_path, ["cpu"], torch.float32, samples=8)
    b = _awq_rtn_bytes(tmp_path, ["cpu", "cpu"], torch.float32, samples=8)
    assert a.keys() == b.keys() and any("experts" in k for k in a)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.gpu
def test_awq_multicard_bytes_equal_on_two_gpus(tmp_path):
    """The same comparison on two CUDA cards (NCCL): the two-card build matches the one-card checkpoint."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible cards")  # a single-card host runs this file without -m gpu
    _saved_dense(tmp_path)
    a = _awq_rtn_bytes(tmp_path, ["cuda:0"], torch.bfloat16)
    b = _awq_rtn_bytes(tmp_path, ["cuda:0", "cuda:1"], torch.bfloat16)
    assert a.keys() == b.keys() and any(k.endswith(".weight_scale") for k in a)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_build_starts_no_ranks_on_one_card_or_on_cpus(fake_ranks, monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise Refused("stop")

    monkeypatch.setattr(build_mod, "compile_plan", refuse)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)  # the named card is visible
    tiny_qwen3(layers=1).save_pretrained(tmp_path / "m")
    for devices in (["cuda:0"], ["cpu", "cpu"]):
        with pytest.raises(Refused):
            build_mod.build(
                str(tmp_path / "m"), NEEDS_RECIPE, format="w4a16-sym-g32", out_dir=tmp_path / "o",
                devices=devices,
            )  # fmt: skip
    assert fake_ranks == []


#: The two ways a send reports the limit: the pipe's EMFILE, and torch's message when it shares a storage.
OUT_OF_FILES = [
    OSError(errno.EMFILE, "Too many open files"),
    RuntimeError(
        "unable to open shared memory object </torch_1_2_3> in read-write mode: Too many open files (24)"
    ),
]


@pytest.mark.parametrize("error", OUT_OF_FILES, ids=["oserror", "torch"])
def test_ranks_run_refuses_a_send_that_ran_out_of_files_instead_of_hanging(tmp_path, error):
    """Rank 1 is alive and waits for a job that never comes: the build stops the ranks and refuses.

    The failed send leaves a duplicate descriptor in `resource_sharer` for a rank that will never fetch it, as a
    real one does; the refusal closes it, or the build's cleanup would run out of descriptors too.
    """
    from multiprocessing import resource_sharer

    r, w = os.pipe()
    os.close(w)
    held: list = []

    def out_of_files(obj) -> None:
        held.append(resource_sharer.DupFd(r))  # registers a duplicate of `r`, open until fetched or closed
        raise error

    ranks = build_mod._Ranks(CPU2, fn=_record_rank)
    ranks.queues[1].put = out_of_files  # the parent's end only: rank 1 still waits on its queue
    got: list[BaseException] = []

    def go() -> None:
        try:
            ranks.run(str(tmp_path))  # rank 0 gets it and finishes
        except BaseException as e:  # noqa: BLE001
            got.append(e)

    t0 = time.monotonic()
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive(), "run() hung on a send that failed to a live rank"
    assert time.monotonic() - t0 < 30
    assert len(got) == 1 and isinstance(got[0], Refused), got
    assert got[0].code == ENVIRONMENT
    assert "card process 1 ran out of file descriptors" in str(got[0])
    assert "ulimit -n " in got[0].fix
    assert not any(p.is_alive() for p in ranks.procs.processes)
    sharer = resource_sharer._resource_sharer  # pyright: ignore[reportAttributeAccessIssue]
    assert held and not sharer._cache  # the duplicate was closed
    os.close(r)
    ranks.close()


class _Limits:
    """Stands in for `resource.getrlimit`/`setrlimit` on RLIMIT_NOFILE."""

    def __init__(self, soft: int, hard: int) -> None:
        self.soft, self.hard = soft, hard

    def get(self, which: int) -> tuple[int, int]:
        assert which == resource.RLIMIT_NOFILE
        return self.soft, self.hard

    def set(self, which: int, limits: tuple[int, int]) -> None:
        assert which == resource.RLIMIT_NOFILE and limits[1] == self.hard
        self.soft = limits[0]


@pytest.fixture
def limits(monkeypatch):
    def make(soft: int, hard: int) -> _Limits:
        lim = _Limits(soft, hard)
        monkeypatch.setattr(resource, "getrlimit", lim.get)
        monkeypatch.setattr(resource, "setrlimit", lim.set)
        return lim

    return make


def test_ranks_start_raises_the_soft_open_file_limit_to_the_hard_limit(limits):
    lim = limits(1024, 131072)
    raise_open_files()
    assert lim.soft == 131072


def test_ranks_start_leaves_a_low_hard_limit_to_the_send_refusal(limits):
    """A 1024 hard limit holds a 128-sample build on four cards; only a send that runs out is refused."""
    lim = limits(1024, 1024)
    raise_open_files()
    assert lim.soft == 1024
