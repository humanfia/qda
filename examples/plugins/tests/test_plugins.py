"""Load the external package through entry points and exercise each plugin on a CPU-sized input."""

import asyncio
import importlib.metadata
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import torch
from qda_quant import registry as registration
from qda_quant.arch import LayerGraph, LinearGroup, ModelGraph
from qda_quant.format import Format
from qda_quant.registry import Registry
from qda_quant.solve.quant import Quantizer
from qda_quant.step import LayerContext, Step
from qda_quant.target import Target
from torch import nn
from transformers import AutoModelForCausalLM, Qwen3Config

ROOT = Path(__file__).parents[3]
PACKAGE = ROOT / "examples" / "plugins"


def load_plugin(group: str, name: str, monkeypatch):
    """Load the package hook, then read the requested example from its registry."""
    declared = tomllib.loads((PACKAGE / "pyproject.toml").read_text())
    value = declared["project"]["entry-points"]["qda.plugins"]["example"]
    entry_point = importlib.metadata.EntryPoint(name="example", value=value, group="qda.plugins")
    from qda_agent.agents import BACKENDS
    from qda_agent.components import COMPONENTS
    from qda_agent.judge.final_rules import FINAL_RULES
    from qda_kl.datasets import DATASETS
    from qda_kl.eval.suites import SUITES
    from qda_kl.scorers import SCORERS
    from qda_quant.arch.adapters import ADAPTERS
    from qda_quant.solve.grid import GRIDS
    from qda_quant.step import REGISTRY

    registries = {
        reg.group: reg
        for reg in (
            REGISTRY,
            GRIDS,
            ADAPTERS,
            SCORERS,
            SUITES,
            DATASETS,
            FINAL_RULES,
            BACKENDS,
            COMPONENTS,
        )
    }
    with monkeypatch.context() as scoped:
        for reg in registries.values():
            scoped.setattr(reg, "_entries", dict(reg._entries))
        scoped.syspath_prepend(str(PACKAGE / "src"))
        scoped.setattr(
            importlib.metadata,
            "entry_points",
            lambda *, group: [entry_point] if group == entry_point.group else [],
        )
        scoped.setattr(registration, "_plugins_loaded", False)
        scoped.setattr(registration, "_plugins_loading", False)
        scoped.setattr(registration, "_plugin_failure", None)
        registration.load_plugins()
        registry = registries[group]
        assert name in registry
        return registry[name]


def test_external_plugins_load_and_run_on_small_cpu_inputs(tmp_path, monkeypatch):
    """The package's entry points register each kind and each example does useful work."""
    step_type = load_plugin("qda.steps", "clip_rtn", monkeypatch)
    assert issubclass(step_type, Step)
    layer = nn.Module()
    layer.proj = nn.Linear(4, 3, bias=False)
    graph = LayerGraph(0, "block", (LinearGroup(("block.proj",), "other"),))
    fmt = Format.parse("w4a16-sym-pc")
    context = LayerContext(
        layer,
        graph,
        ModelGraph("example", (graph,), "", "", ""),
        fmt,
        Target.parse("h200").plan(fmt, "dense"),
        None,
        {},
        device=torch.device("cpu"),
    )
    step_type().apply_layer(context)
    assert context.qparams["block.proj"]["codes"].shape == layer.proj.weight.shape

    grid = load_plugin("qda.grids", "signed8_bf16", monkeypatch)
    bf16 = Quantizer(fmt, context.plan_dense, grid)
    scale, _ = bf16.params(layer.proj.weight.data)
    assert bf16.stored_scale(scale).dtype == torch.bfloat16
    assert grid.refusal(Format.parse("w4a16"), context.plan_dense)

    adapter = load_plugin("qda.arch_adapters", "tiny_projection", monkeypatch)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).eval()
    model.model.layers[0].projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    assert adapter.matches(model)
    from qda_quant.arch import adapters as arch_adapters
    from qda_quant.arch import describe

    registry = Registry("arch_adapters", group="qda.arch_adapters")
    registry.register("tiny_projection", adapter)
    monkeypatch.setattr(arch_adapters, "ADAPTERS", registry)
    adapted = describe(model, verify=False)
    assert adapted.adapter == "tiny_projection"
    assert adapted.layers[0].groups[-1].modules == ("model.layers.0.projection",)
    assert Target.parse("h200").plan(fmt, "dense").kernel

    scorer_type = load_plugin("qda_kl.scorers", "tensor_kl", monkeypatch)
    checkpoint, reference = tmp_path / "checkpoint", tmp_path / "reference"
    checkpoint.mkdir()
    reference.mkdir()
    original = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    candidate = original + torch.tensor([[0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]])
    torch.save(original, reference / "logits.pt")
    torch.save(candidate, checkpoint / "logits.pt")
    scored = scorer_type().run(str(checkpoint), reference).result
    assert scored.tokens == 2 and scored.kl > 0

    suite_type = load_plugin("qda.eval_suites", "compactness", monkeypatch)
    measured = suite_type().run(str(checkpoint), str(reference), tmp_path)
    assert measured.metrics["compactness"] > 0 and measured.higher_is_better["compactness"]

    backend_type = load_plugin("qda_agent.backends", "greedy_demo", monkeypatch)
    backend = backend_type()
    calls = []

    def request(socket, verb, argv, cwd):
        calls.append((socket, verb, argv, cwd))
        if verb == "kl":
            return {"exit": 0, "stdout": "a out/a: read: KL 0.2\nb out/b: read: KL 0.1\n"}
        return {"exit": 0, "stdout": "submitted"}

    backend_module = sys.modules["qda_example_plugins.backends"]
    monkeypatch.setattr(backend_module, "request", request)
    cfg = SimpleNamespace(host_socket="not-settled-yet")
    turn = backend.turn(cfg)
    cfg.host_socket = "test-socket"  # the flow settles the socket after the driver is made
    response = asyncio.run(turn("choose the best checkpoint"))
    assert "out/b at KL 0.1" in response
    assert calls == [
        ("test-socket", "kl", ["list"], "."),
        ("test-socket", "handin", ["out/b"], "."),
    ]
    assert backend.driver(SimpleNamespace(host_socket="test-socket")) is not None

    component_type = load_plugin("qda_agent.components", "read_only_note", monkeypatch)
    workdir, overlay = tmp_path / "work", tmp_path / "overlay"
    workdir.mkdir()
    component = component_type(text="Example plugin note")
    from qda_agent.components import Run

    run = Run(workdir=workdir, overlay=overlay)
    component.install(run)
    assert (overlay / "PLUGIN_NOTE.md").read_text() == "Example plugin note\n"
    assert component.protected(run) == ["PLUGIN_NOTE.md"]


def test_a_final_rule_is_added_by_its_entry_point_alone(tmp_path):
    """A process whose only knowledge of the rule is an installed entry point names it in a judge config."""
    declared = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]["entry-points"]
    info = tmp_path / "third_party-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: third-party\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(f"[qda.plugins]\nexample = {declared['qda.plugins']['example']}\n")
    code = """
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from qda_agent.host import HandIn
from qda_agent.judge.config import JudgeSettings
from qda_agent.judge.final_rules import apply

at = datetime(2026, 1, 1, tzinfo=UTC)
ledger = [HandIn(dir=Path(n), submitted_at=at, valid=True) for n in ("a", "b")]
rules = JudgeSettings(final_rules=["first_valid", "baseline"]).final_rules
final = apply(rules, ledger, [], "kl")
print(final.rule, final.choice.dir)
print(apply(rules, [], [], "kl").rule)
print(apply(JudgeSettings().final_rules, ledger, [], "kl").choice.dir)
try:
    JudgeSettings(final_rules=["nope"])
except ValidationError as e:
    print(e.errors()[0]["msg"])
"""
    path = os.pathsep.join([str(tmp_path), str(PACKAGE / "src"), *filter(None, sys.path)])
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": path}
    )
    assert done.returncode == 0, done.stderr
    picked, empty, default, refused = done.stdout.splitlines()
    assert picked == "first_valid a"
    assert empty == "baseline"  # the rule left the decision to the next one
    assert default == "b"  # no config names it: the built-in order
    assert "'nope'" in refused and "first_valid" in refused and "last_valid" in refused
