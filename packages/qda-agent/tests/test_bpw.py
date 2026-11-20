"""`bpw DIR` and `bpw --plan RECIPE`: the judge's own bits-per-weight rule, on a checkpoint and on a plan."""

import json
from pathlib import Path

import pytest
import torch
from qda_agent.commands import NAMES
from qda_agent.judge.checks import budget
from qda_quant.config import BuildConfig
from qda_quant.format import Format
from test_judge import Rig
from test_judge_budget import K, Q, anchor, model_dir, quantized

COMMANDS = [*NAMES, "bpw"]


def test_bpw_prices_a_checkpoint_with_the_judges_own_rule(tmp_path):
    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True, check=False, commands=COMMANDS)
    model_dir(r.work / "mixed", {**quantized(Q, 8), **quantized(K, 4)}, quantized=True)
    rep = r.call("bpw", "mixed")
    assert rep.exit == 0, rep.stderr
    value = budget.bpw(r.work / "mixed", a)  # the admission's own function, not a second reading
    assert f"mixed: {value:.3f} bits per quantized weight; budget 6.5 bpw (inside)" in rep.stdout


def test_bpw_says_over_the_budget_and_refuses_a_link_or_a_path_outside(tmp_path):
    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True, check=False, commands=COMMANDS)
    # one Linear kept bf16: 12.0625 bpw
    model_dir(r.work / "heavy", {**quantized(Q, 8), f"{K}.weight": ("BF16", (128, 128), None)}, True)
    rep = r.call("bpw", "heavy")
    assert rep.exit == 0 and "12.062 bits per quantized weight; budget 6.5 bpw (over)" in rep.stdout
    (r.work / "link").symlink_to(r.work / "heavy")
    assert r.call("bpw", "link").exit == 2
    assert "a link is not priced" in r.call("bpw", "link").stderr
    assert r.call("bpw", "../up").exit == 2
    assert "not inside the work directory" in r.call("bpw", "../up").stderr
    assert "inside the work directory" in r.call("bpw", str(tmp_path / "anchor")).stderr


def test_bpw_refuses_a_link_in_any_component_and_a_linked_shard(tmp_path):
    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True, check=False, commands=COMMANDS)
    private = model_dir(
        tmp_path / "private", {**quantized(Q, 8), f"{K}.weight": ("BF16", (128, 128), None)}, True
    )
    (r.work / "alias").symlink_to(private)  # a symlinked parent of the checkpoint
    rep = r.call("bpw", "alias")
    assert rep.exit == 2 and "a link is not priced" in rep.stderr
    assert r.call("bpw", "alias/x").exit == 2  # nor does a checkpoint under it price
    good = model_dir(r.work / "good", {**quantized(Q, 8), **quantized(K, 4)}, quantized=True)
    (good / "model.safetensors").unlink()
    (good / "model.safetensors").symlink_to(private / "model.safetensors")  # a swapped shard
    rep = r.call("bpw", "good")
    assert rep.exit == 2 and "a link is not priced" in rep.stderr


def test_bpw_prices_a_private_copy_so_a_swap_cannot_redirect_it(tmp_path, monkeypatch):
    """The host reads headers through a pinned descriptor into its own dir; the agent's path is not reopened."""
    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True, check=False, commands=COMMANDS)
    mixed = {**quantized(Q, 8), **quantized(K, 4)}
    model_dir(r.work / "mixed", mixed, quantized=True)
    value = budget.bpw(r.work / "mixed", a)
    private = model_dir(
        tmp_path / "private", {**quantized(Q, 8), f"{K}.weight": ("BF16", (128, 128), None)}, True
    )
    real = budget.bpw
    seen = []

    def swapped(candidate, anchor):
        seen.append(candidate)
        (r.work / "mixed").rename(r.work / "moved")
        (r.work / "mixed").symlink_to(private)  # what the agent would swap in mid-price
        return real(candidate, anchor)

    monkeypatch.setattr(budget, "bpw", swapped)
    rep = r.call("bpw", "mixed")
    assert rep.exit == 0, rep.stderr
    assert f"mixed: {value:.3f} bits per quantized weight" in rep.stdout  # not the private 12.062 bpw
    (only,) = seen
    assert Path(only).is_relative_to(r.host.jdir) and not Path(only).is_relative_to(r.work)
    assert not Path(only).exists()  # the staging copy is removed


def _sharded(d: Path, shards: int, per_shard: int) -> dict[str, str]:
    """`shards` sparse safetensors files of per-expert NVFP4-style tensors (headers only on disk); the weight_map."""
    d.mkdir(parents=True)
    wm: dict[str, str] = {}
    k = 0
    for s in range(shards):
        shard = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        header: dict[str, dict] = {}
        off = 0
        for _ in range(per_shard):
            layer, rest = divmod(k, 384 * 3 * 4)
            expert, rest = divmod(rest, 3 * 4)
            proj, part = divmod(rest, 4)
            name = (
                f"model.layers.{layer}.mlp.experts.{expert}.{('gate', 'up', 'down')[proj]}_proj."
                + ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")[part]
            )
            shape = ([2048, 3584], [2048, 448], [1], [1])[part]
            dtype = ("U8", "F8_E4M3", "F32", "F32")[part]
            n = shape[0] * shape[1] if len(shape) == 2 else 4
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
            wm[name] = shard
            off += n
            k += 1
        raw = json.dumps(header, separators=(",", ":")).encode()
        with (d / shard).open("wb") as fh:
            fh.write(len(raw).to_bytes(8, "little") + raw)
            fh.truncate(8 + len(raw) + off)
    return wm


def test_a_57_mb_index_over_96_shards_is_staged_and_an_index_larger_than_its_headers_is_not(tmp_path):
    """A fixed 16 MiB cap refused a real 96-shard checkpoint's 57 MB index; the cap is now the 16 MiB floor
    plus the shards' header bytes, so the index prices while one past that bound is refused.
    """
    from qda_agent.judge import commands
    from qda_agent.judge.checks.tensors import loaded
    from qda_agent.sched import Refused

    d = tmp_path / "work" / "big"
    wm = _sharded(d, 96, 5900)
    index = d / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}, indent=2))
    size = index.stat().st_size
    headers = sum(int.from_bytes((d / f).open("rb").read(8), "little") for f in set(wm.values()))
    assert 55e6 < size < 60e6 and size > commands.INDEX_FLOOR, size  # 57 MB: what 16 << 20 refused
    assert size < headers  # as in real checkpoints: an index entry is shorter than its header entry
    stage = tmp_path / "stage"
    stage.mkdir()
    commands._stage(tmp_path / "work", "big", stage, "big")
    assert (stage / index.name).read_bytes() == index.read_bytes()
    assert len(loaded(stage)) == len(wm)  # the staged index maps every tensor the shards hold

    # past the floor plus every header: refused, with its fix
    index.write_text(json.dumps({"weight_map": wm, "pad": "x" * (commands.INDEX_FLOOR + headers)}))
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    with pytest.raises(Refused, match=r"too large to price: over \d+ B.*\(fix: keep only weight_map"):
        commands._stage(tmp_path / "work", "big", stage2, "big")


def test_bpw_prices_an_index_up_to_the_floor_plus_its_headers(tmp_path, monkeypatch):
    """End to end through `bpw DIR`: with the floor at 1 B every passing index byte is paid for by headers."""
    from qda_agent.judge import commands

    a = anchor(tmp_path / "anchor")
    r = Rig(tmp_path, anchor=a, budget_bpw=6.5, allow_any_format=True, check=False, commands=COMMANDS)
    monkeypatch.setattr(commands, "INDEX_FLOOR", 1)
    tensors = {**quantized(Q, 8), **quantized(K, 4)}
    d = model_dir(r.work / "indexed", tensors, quantized=True)
    index = d / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")}, indent=2))
    header = int.from_bytes((d / "model.safetensors").read_bytes()[:8], "little")
    assert 1 < index.stat().st_size <= header
    rep = r.call("bpw", "indexed")
    assert rep.exit == 0, rep.stderr
    assert f"indexed: {budget.bpw(d, a):.3f} bits per quantized weight" in rep.stdout

    index.write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors"), "pad": "x" * header})
    )
    rep = r.call("bpw", "indexed")
    assert rep.exit == 2, rep.stdout
    assert f"over {header + 1} B" in rep.stderr and "(fix: " in rep.stderr


def test_shards_claiming_headers_that_do_not_parse_add_nothing_to_the_index_bound(tmp_path, monkeypatch):
    """50 sparse shards each claiming a MAX_HEADER header (zeros, or bytes that are not JSON) leave the bound
    at the floor: only a header that parses pays for index bytes.
    """
    from qda_agent.judge import commands
    from qda_agent.sched import Refused

    monkeypatch.setattr(commands, "MAX_HEADER", 1 << 20)  # 100 MiB in production; same rule, smaller files
    monkeypatch.setattr(commands, "INDEX_FLOOR", 1 << 20)
    d = tmp_path / "work" / "sparse"
    d.mkdir(parents=True)
    names = []
    for i in range(50):
        shard = d / f"model-{i + 1:05d}-of-00050.safetensors"
        with shard.open("wb") as fh:
            fh.write((1 << 20).to_bytes(8, "little"))
            if i % 2:
                fh.write(b"{" * 4096)  # claimed, partly written, not JSON
            fh.truncate(8 + (1 << 20) + 64)  # sparse: the claimed header reads back as zeros
        names.append(shard.name)
    index = d / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {f"t{i}": n for i, n in enumerate(names)}, "pad": "x" * (2 << 20)})
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(
        Refused, match=rf"over {1 << 20} B, the 1 MiB floor plus .*\(fix: keep only weight_map"
    ):
        commands._stage(tmp_path / "work", "sparse", stage, "sparse")


def test_an_index_over_the_absolute_cap_is_refused_whatever_its_headers_state(tmp_path, monkeypatch):
    """Headers that parse raise the bound only up to INDEX_CAP; an index past it is refused with its fix."""
    from qda_agent.judge import commands
    from qda_agent.sched import Refused

    assert commands.INDEX_CAP == 256 << 20  # about 4x the largest real index (57 MB)
    monkeypatch.setattr(commands, "INDEX_CAP", 1 << 20)
    monkeypatch.setattr(commands, "INDEX_FLOOR", 1)
    d = tmp_path / "work" / "capped"
    d.mkdir(parents=True)
    raw = json.dumps({"__metadata__": {"x": "y" * (3 << 20)}}).encode()  # a valid 3 MiB header
    for i in range(2):
        (d / f"model-{i + 1:05d}-of-00002.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
    wm = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    index = d / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": wm, "pad": "x" * (2 << 20)})
    )  # under its headers, over the cap
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(
        Refused, match=rf"over {1 << 20} B, the 1 MiB cap on any index \(fix: keep only weight_map"
    ):
        commands._stage(tmp_path / "work", "capped", stage, "capped")
    index.write_text(json.dumps({"weight_map": wm, "pad": "x" * ((1 << 20) - 200)}))  # at the cap: staged
    commands._stage(tmp_path / "work", "capped", stage, "capped")
    assert (stage / index.name).read_bytes() == index.read_bytes()


def test_an_index_naming_absent_shards_is_refused_for_them_before_its_size(tmp_path, monkeypatch):
    """A build still writing (2 of 5 shards present): its index is over the bound its present headers set, and
    the refusal names the missing shard, not the size whose fix would price a partial checkpoint.
    """
    from qda_agent.judge import commands
    from qda_agent.sched import Refused

    monkeypatch.setattr(commands, "INDEX_FLOOR", 1)
    d = tmp_path / "work" / "partial"
    wm = _sharded(d, 5, 40)
    index = d / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": wm}, indent=2))
    for s in range(3, 6):
        (d / f"model-{s:05d}-of-00005.safetensors").unlink()
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(
        Refused,
        match=r"partial: model.safetensors.index.json names 'model-00003-of-00005"
        r".safetensors', not a shard file of the checkpoint",
    ):
        commands._stage(tmp_path / "work", "partial", stage, "partial")


def test_bpw_needs_the_anchor(tmp_path):
    r = Rig(tmp_path, check=False, commands=COMMANDS)
    rep = r.call("bpw", "x")
    assert rep.exit == 2 and "bpw needs judge.anchor" in rep.stderr


def test_a_plan_that_packs_more_than_the_original_is_refused():
    with pytest.raises(ValueError, match="more than the original"):
        budget.planned_bpw([((128, 128), Format.parse("w4a16"))], 128 * 128 - 1)


def _tiny_model(d: Path) -> str:
    """A two-layer Qwen3 in `d` with a tokenizer and a four-row calibration file; returns the tokenizer digest."""
    from qda_quant.engine.calib import fingerprint
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
    )
    cfg._attn_implementation = "sdpa"
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval().save_pretrained(d)
    vocab = {f"w{i}": i for i in range(127)} | {"[UNK]": 127}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = "{% for m in messages %}w1 {{ m['content'] }} w2 {% endfor %}"
    fast.save_pretrained(d)
    sha = fingerprint(d)
    assert sha is not None
    return sha


def test_a_plans_bpw_is_the_built_checkpoints_bpw(tmp_path):
    """The plan prices each stored tensor at its format's packed bits, so it must equal the built checkpoint."""
    model = tmp_path / "tiny"
    model.mkdir()
    sha = _tiny_model(model)
    rows = [list(range(1, 9)) for _ in range(4)]
    fs = [json.dumps({"tokenizer": sha}), *[json.dumps({"input_ids": r}) for r in rows]]
    (tmp_path / "calib.jsonl").write_text("\n".join(fs))
    recipe = {
        "model": str(model),
        "format": "w4a16-sym-g32",
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp_path / "calib.jsonl"), "samples": 4, "seqlen": 8}},
        "devices": ["cpu"],
    }
    built = tmp_path / "built"
    config = BuildConfig.from_dict(recipe, base=str(tmp_path))
    config.out = str(built)
    config.build()
    import yaml

    r = Rig(
        tmp_path / "run", anchor=model, budget_bpw=16.0, allow_any_format=True, check=False, commands=COMMANDS
    )
    (r.work / "recipe.yaml").write_text(yaml.safe_dump({**recipe, "out": str(built)}))
    rep = r.call("bpw", "--plan", "recipe.yaml")
    assert rep.exit == 0, rep.stderr
    value = budget.bpw(built, model)  # the built checkpoint, read back through the judge's own rule
    total = next(line for line in rep.stdout.splitlines() if line.startswith("total:"))
    assert float(total.split()[1]) == pytest.approx(value, abs=5e-4)
    assert "w4a16-sym-g32:" in rep.stdout and "budget 16 bpw (inside)" in rep.stdout


def test_a_formats_recipe_prices_through_the_librarys_resolution(tmp_path):
    """`formats:` is compiled by the library, so the plan prices its per-module formats, not a parser here."""
    model = tmp_path / "tiny"
    model.mkdir()
    sha = _tiny_model(model)
    rows = [list(range(1, 9)) for _ in range(4)]
    fs = [json.dumps({"tokenizer": sha}), *[json.dumps({"input_ids": r}) for r in rows]]
    (tmp_path / "calib.jsonl").write_text("\n".join(fs))
    recipe = {
        "model": str(model),
        "formats": {"default": "w4a16-sym-g32", "model.layers.0.mlp.down_proj": "w8a16-sym-g32"},
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp_path / "calib.jsonl"), "samples": 4, "seqlen": 8}},
        "devices": ["cpu"],
    }
    built = tmp_path / "built"
    config = BuildConfig.from_dict(recipe, base=str(tmp_path))
    config.out = str(built)
    config.build()
    import yaml

    r = Rig(
        tmp_path / "run", anchor=model, budget_bpw=16.0, allow_any_format=True, check=False, commands=COMMANDS
    )
    (r.work / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    rep = r.call("bpw", "--plan", "recipe.yaml")
    assert rep.exit == 0, rep.stderr
    value = budget.bpw(built, model)  # the same recipe built: the compiled formats, read back
    total = next(line for line in rep.stdout.splitlines() if line.startswith("total:"))
    assert float(total.split()[1]) == pytest.approx(value, abs=5e-4)
    assert "w4a16-sym-g32:" in rep.stdout and "w8a16-sym-g32:" in rep.stdout  # one class per format


def test_a_plan_prices_kept_weights_at_the_originals_stored_width(tmp_path):
    """A whole layer kept as bf16 on an fp32 original stays fp32 in the build, and the plan must price it so."""
    model = tmp_path / "tiny"
    model.mkdir()
    sha = _tiny_model(model)  # saved in fp32: kept weights store 32 bits, not 16
    rows = [list(range(1, 9)) for _ in range(4)]
    fs = [json.dumps({"tokenizer": sha}), *[json.dumps({"input_ids": r}) for r in rows]]
    (tmp_path / "calib.jsonl").write_text("\n".join(fs))
    recipe = {
        "model": str(model),
        "formats": {"default": "w4a16-sym-g32", "re:model\\.layers\\.0\\.self_attn\\.": "bf16"},
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp_path / "calib.jsonl"), "samples": 4, "seqlen": 8}},
        "devices": ["cpu"],
    }
    built = tmp_path / "built"
    config = BuildConfig.from_dict(recipe, base=str(tmp_path))
    config.out = str(built)
    config.build()
    import yaml

    r = Rig(
        tmp_path / "run", anchor=model, budget_bpw=16.0, allow_any_format=True, check=False, commands=COMMANDS
    )
    (r.work / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    rep = r.call("bpw", "--plan", "recipe.yaml")
    assert rep.exit == 0, rep.stderr
    value = budget.bpw(built, model)  # the built checkpoint: the kept layer stores fp32
    total = next(line for line in rep.stdout.splitlines() if line.startswith("total:"))
    assert float(total.split()[1]) == pytest.approx(value, abs=5e-4)
    assert "f32: 12288 weights" in rep.stdout  # layer 0's q/k/v/o kept fp32: 4096 + 2048 + 2048 + 4096


@pytest.mark.parametrize("swap", [False, True])
def test_bpw_never_prices_host_files_through_an_agent_link(tmp_path, monkeypatch, swap):
    original = anchor(tmp_path / "anchor")
    r = Rig(tmp_path / "run", anchor=original, budget_bpw=16, check=False, commands=COMMANDS)
    outside = tmp_path / "private" / "checkpoint"
    model_dir(outside, {**quantized(Q, 8), **quantized(K, 4)}, quantized=True)
    expected = None
    if not swap:
        (r.work / "alias").symlink_to(outside.parent, target_is_directory=True)
        name = "alias/checkpoint"
    else:
        path = r.work / "checkpoint"
        model_dir(path, {**quantized(Q, 4), **quantized(K, 4)}, quantized=True)
        real = budget.bpw
        expected = real(path, original)

        def swap_then_price(candidate, anchor):
            path.rename(r.work / "old-checkpoint")
            path.symlink_to(outside, target_is_directory=True)
            return real(candidate, anchor)

        monkeypatch.setattr(budget, "bpw", swap_then_price)
        name = "checkpoint"
    reply = r.call("bpw", name)
    assert reply.exit == 2 or (
        swap and reply.exit == 0 and f"{expected:.3f} bits per quantized weight" in reply.stdout
    ), "refuse an outside path or price a safely pinned original; never read host files"


def test_a_mixed_plan_prices_kept_weights_at_their_stored_width(tmp_path):
    model = tmp_path / "tiny"
    model.mkdir()
    sha = _tiny_model(model)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "\n".join(
            [
                json.dumps({"tokenizer": sha}),
                *[json.dumps({"input_ids": list(range(1, 9))}) for _ in range(4)],
            ]
        )
    )
    from qda_quant.arch import KEEP_ROLES

    recipe = {
        "model": str(model),
        "format": "w4a16-sym-g32",
        "format_overrides": [{"pattern": r"re:model.layers.0.mlp\.", "format": "w8a16-sym-g32"}],
        "ignore": [*KEEP_ROLES, r"re:model.layers.1.self_attn\."],
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(calib), "samples": 4, "seqlen": 8}},
        "devices": ["cpu"],
        "out": str(tmp_path / "built"),
    }
    config = BuildConfig.from_dict(recipe)
    config.build()
    r = Rig(tmp_path / "run", anchor=model, budget_bpw=9, check=False, commands=COMMANDS)
    import yaml

    (r.work / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    reply = r.call("bpw", "--plan", "recipe.yaml")
    assert reply.exit == 0, reply.stderr
    value = budget.bpw(Path(recipe["out"]), model)
    total = next(line for line in reply.stdout.splitlines() if line.startswith("total:"))
    assert float(total.split()[1]) == pytest.approx(value, abs=5e-4), reply.stdout
    assert "(over)" in reply.stdout  # retained FP32 weights put this build above the 9-bpw budget


def test_a_recipe_naming_no_target_is_planned_for_the_runs_cards(tmp_path, monkeypatch):
    """`bpw --plan` on a W4A4 recipe plans the run's cards, not the default `Target()` (sm90).

    The judge holds no card, so a recipe that names no `target:` used to be planned for sm90, which serves no
    W4A4: the plan refused until the recipe said `target: sm100`. The run's cards are sm103 here; a recipe's
    own `target:` still wins.
    """
    import yaml
    from qda_agent import group_size
    from qda_quant.target import Target

    model = tmp_path / "tiny"
    model.mkdir()
    sha = _tiny_model(model)
    rows = [list(range(1, 9)) for _ in range(4)]
    fs = [json.dumps({"tokenizer": sha}), *[json.dumps({"input_ids": r}) for r in rows]]
    (tmp_path / "calib.jsonl").write_text("\n".join(fs))
    recipe = {
        "model": str(model),
        "format": "w4a4-nvfp4",
        "steps": [{"rtn": {}}],
        "calib": {"file": {"path": str(tmp_path / "calib.jsonl"), "samples": 4, "seqlen": 8}},
        "devices": ["cpu"],
    }
    monkeypatch.setattr(group_size, "run_target", lambda cards: Target.parse("sm103"))
    r = Rig(
        tmp_path / "run", anchor=model, budget_bpw=16.0, allow_any_format=True, check=False, commands=COMMANDS
    )
    (r.work / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    rep = r.call("bpw", "--plan", "recipe.yaml")
    assert rep.exit == 0, rep.stderr
    assert "w4a4-nvfp4:" in rep.stdout
    (r.work / "recipe.yaml").write_text(yaml.safe_dump({**recipe, "target": "sm90"}))
    rep = r.call("bpw", "--plan", "recipe.yaml")
    assert rep.exit == 2 and "sm90" in rep.stderr
