"""The worker hook on a fake runner: rows found in chunked, batched steps equal the slow full-logits path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from qda_kl import vllm_hooks
from qda_kl.store import Reference, ReferenceWriter
from qda_kl.topk import anchor_topk, kl_topk
from qda_kl.vllm_hooks import TAG, KLWorker, unpack

V, H = 64, 8


class Model:
    def __init__(self, seed):
        self.w = torch.randn(H, V, generator=torch.Generator().manual_seed(seed)) * 2

    def compute_logits(self, h):
        return h @ self.w


def hidden(tokens):
    # a deterministic "hidden state" per (position, token): what a real model computes from the prefix
    g = torch.Generator().manual_seed(1)
    table = torch.randn(512, V, H, generator=g)
    return torch.stack([table[p, t] for p, t in enumerate(tokens)])


def full_logp(model, tokens):
    return model.compute_logits(hidden(tokens)).float().log_softmax(-1)


class FakeV2:
    """Steps of a V2 runner: each step schedules a chunk of every request (chunked prefill), batched."""

    def __init__(self, model):
        self.model = model
        self.device = "cpu"
        self.prompt_logprobs_worker = SimpleNamespace(
            add_request=lambda *a: None,
            remove_request=lambda *a: None,
            compute_prompt_logprobs=lambda *a, **k: {},
        )

    def run(self, reqs, step=7):
        w = self.prompt_logprobs_worker
        for rid, (_, sp) in reqs.items():
            w.add_request(rid, 0, sp)
        done = dict.fromkeys(reqs, 0)
        while any(done[r] < len(t) for r, (t, _) in reqs.items()):
            ids, q, comp, hs = [], [0], [], []
            for r, (t, _) in reqs.items():
                if done[r] >= len(t):
                    continue
                n = min(step, len(t) - done[r])
                ids.append(r)
                comp.append(done[r])
                hs.append(hidden(t)[done[r] : done[r] + n])
                q.append(q[-1] + n)
                done[r] += n
            batch = SimpleNamespace(
                req_ids=ids,
                query_start_loc_np=np.array(q),
                num_computed_prefill_tokens_np=np.array(comp),
                num_tokens=q[-1],
            )
            w.compute_prompt_logprobs(self.model.compute_logits, torch.cat(hs), batch, None, None, None)
        for rid in reqs:
            w.remove_request(rid)


class FakeV1:
    def __init__(self, model):
        self.model, self.device, self.requests = model, "cpu", {}
        self._get_prompt_logprobs_dict = lambda *a, **k: {}

    def run(self, reqs, step=7):
        for rid, (_, sp) in reqs.items():
            self.requests[rid] = SimpleNamespace(sampling_params=sp, num_computed_tokens=0)
        while any(self.requests[r].num_computed_tokens < len(t) for r, (t, _) in reqs.items()):
            sched, hs, q = {}, [], [0]
            self.input_batch = SimpleNamespace(req_id_to_index={})
            for r, (t, _) in reqs.items():
                c = self.requests[r].num_computed_tokens
                if c >= len(t):
                    continue
                n = min(step, len(t) - c)
                self.input_batch.req_id_to_index[r] = len(sched)
                sched[r] = n
                hs.append(hidden(t)[c : c + n])
                q.append(q[-1] + n)
            self.query_start_loc = SimpleNamespace(np=np.array(q))
            self._get_prompt_logprobs_dict(torch.cat(hs), sched)
            for r, n in sched.items():
                self.requests[r].num_computed_tokens += n


SEQS = {"a#0": (list(range(3, 40)), 6), "b#0": ([7, 1, 2] * 9, 3), "c#1": (list(range(10, 64, 2)), 20)}


class BlockShaped:
    """A 3-bit lm-head stand-in: large, coarse logits whose fp32 accumulation order follows the call's row count.

    A quantized GEMM tiles the reduction according to the row-block shape, so the same rows summed in blocks of 3
    or 4 rows round differently. Here the reduction is split at a row-count-dependent point in float32.
    """

    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(H, V, generator=g) * 3e4

    def compute_logits(self, h):
        k = min(H - 1, H // 2 + h.shape[0] % 3)  # the kernel's reduction split follows the block shape
        return h[:, :k] @ self.w[:k] + h[:, k:] @ self.w[k:]


def engine_logp(model, tokens, start):
    """vLLM's own prompt-logprob blocks over a step buffer of this sequence: `engine_chunk` rows from row 0."""
    h, out, block = hidden(tokens), {}, vllm_hooks.engine_chunk()
    for o in range(0, len(tokens), block):
        e = min(o + block, len(tokens))
        lp = model.compute_logits(h[o:e]).float().log_softmax(-1)
        for p in range(max(o, start - 1), min(e, len(tokens) - 1)):
            out[p] = lp[p - o, tokens[p + 1]]
    return out


def own_blocks_logp(model, tokens, start):
    """The blocking the hook used before this fix: CHUNK rows from the first scored row."""
    h, out = hidden(tokens), {}
    for s in range(start - 1, len(tokens) - 1, vllm_hooks.CHUNK):
        e = min(len(tokens) - 1, s + vllm_hooks.CHUNK)
        lp = model.compute_logits(h[s:e]).float().log_softmax(-1)
        for p in range(s, e):
            out[p] = lp[p - s, tokens[p + 1]]
    return out


def test_the_hook_recomputes_the_engine_row_blocks(monkeypatch):
    """A 3-bit-like lm-head rounds by row-block shape: only the engine's own blocks reproduce its log-probs."""
    monkeypatch.setattr(vllm_hooks, "engine_chunk", lambda: 4)
    monkeypatch.setattr(vllm_hooks, "CHUNK", 3)
    model = BlockShaped(0)
    tokens, start = list(range(3, 28)), 6
    w = worker(FakeV2(model), monkeypatch)
    w.qda_kl_install()
    got = run(w, "ref", seqs={"a#0": (tokens, start)})["a#0"]["tgt"]
    engine, own = engine_logp(model, tokens, start), own_blocks_logp(model, tokens, start)
    moved = max(abs(own[p] - engine[p]) for p in engine)
    assert moved > 1e-3, f"the fixture must show the old blocking moving log-probs (moved {moved:.3g})"
    for p, value in engine.items():
        assert abs(float(got[p - (start - 1)]) - value) <= 1e-6


def worker(runner, monkeypatch):
    w = KLWorker()
    w.model_runner = runner
    monkeypatch.setattr(KLWorker, "_qda_kl_reader", lambda self: True)
    return w


def run(w, mode, k=8, ref_path=None, seqs=None):
    seqs = SEQS if seqs is None else seqs
    w.qda_kl_begin(mode, {key: (s, t) for key, (t, s) in seqs.items()}, k, ref_path)
    reqs = {
        f"req-{i}": (t, SimpleNamespace(extra_args={TAG: {"key": key, "start": s}}))
        for i, (key, (t, s)) in enumerate(seqs.items())
    }
    reqs["untagged"] = (list(range(30)), SimpleNamespace(extra_args=None))
    w.model_runner.run(reqs)
    return {k: unpack(v) for k, v in w.qda_kl_end().items()}


@pytest.mark.parametrize("fake", [FakeV2, FakeV1])
def test_hook_equals_full_logits(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(vllm_hooks, "CHUNK", 4)  # several hook chunks per step
    anchor, cand = Model(0), Model(1)
    w = worker(fake(anchor), monkeypatch)
    assert w.qda_kl_install() == ("v2" if fake is FakeV2 else "v1")
    got = run(w, "ref")
    ref = ReferenceWriter(tmp_path / "ref", k=8, vocab=V)
    for key, (t, s) in SEQS.items():
        slow = anchor_topk(full_logp(anchor, t)[s - 1 : -1], 8)
        for name, x in zip(("topv", "topi", "log_rest"), slow, strict=True):
            torch.testing.assert_close(torch.from_numpy(got[key][name]), x)
        tgt = full_logp(anchor, t)[s - 1 : -1].gather(1, torch.tensor(t[s:])[:, None])[:, 0]
        torch.testing.assert_close(torch.from_numpy(got[key]["tgt"]), tgt)
        ref.add(key, "d", t, s, slow)
    ref.close()

    w = worker(fake(cand), monkeypatch)
    w.qda_kl_install()
    got = run(w, "score", ref_path=str(tmp_path / "ref"))
    stored = Reference(tmp_path / "ref")
    for s in stored.sequences:
        t = SEQS[s.id][0]
        slow = kl_topk(stored.topk(s), full_logp(cand, t)[s.start - 1 : -1])
        torch.testing.assert_close(torch.from_numpy(got[s.id]["kl"]), slow)
        assert got[s.id]["kl"].min() > 0


def test_hook_reports_unseen_positions(monkeypatch):
    w = worker(FakeV2(Model(0)), monkeypatch)
    w.qda_kl_install()
    w.qda_kl_begin("ref", {"x": (2, list(range(20)))}, 4)
    assert unpack(w.qda_kl_end()["x"])["missing"] == 18
