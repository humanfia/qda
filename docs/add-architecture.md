# Add an architecture

For a researcher whose model `qda plan` refuses, or whose build prints `norm fold abandoned`: see what QDA derives
from the model, then write a small adapter that supplies what it missed. Most architectures need no code: run
[`qda plan`](quantize.md#when-qda-plan-refuses) first.

## What qda derives from the model

`qda_quant.arch.describe(model)` returns a `ModelGraph` (`qda_quant/arch/__init__.py`). It runs one tiny forward
pass with a hook on every module (`arch/derive.py`). For each decoder layer (`LayerGraph`) the graph holds:

- `groups`: `LinearGroup`s, the linear modules that read the very same input tensor, with the norm that produced
  that input (`norm`) or the single linear before it (`prev`);
- `banks`: `ExpertBank`s, the routed experts of an MoE layer, with their router and `top_k`.

It also names the embedding, the final norm and the LM head. `describe` then checks every norm and `prev` relation
numerically in fp64 on CPU (`arch/verify.py`). A relation that fails is dropped and listed in
`graph.extra["dropped"]`, and a build whose steps read it prints `qda: warning: norm fold abandoned: <norm> ->
<group>`: the checkpoint is correct, but the fold is lost until an adapter supplies the relation.

To check a new model, build a tiny random copy from its `transformers` config and look at its graph:

```python
>>> import torch, transformers
>>> from qda_quant.arch import describe
>>> def tiny(cfg):
...     cfg._attn_implementation = "eager"
...     cfg._experts_implementation = "eager"
...     torch.manual_seed(0)
...     return transformers.AutoModelForCausalLM.from_config(cfg).eval()
>>> small = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
...              num_key_value_heads=2, head_dim=16, vocab_size=128)
>>> dense = tiny(transformers.Qwen3Config(**small))
>>> graph = describe(dense)
>>> graph.verified, graph.extra["dropped"], graph.adapter
(True, [], None)
>>> [(g.kind, g.modules) for g in graph.layers[0].groups]
[('attn_in', ('model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj', 'model.layers.0.self_attn.v_proj')),
 ('attn_out', ('model.layers.0.self_attn.o_proj',)),
 ('mlp_in', ('model.layers.0.mlp.gate_proj', 'model.layers.0.mlp.up_proj')),
 ('mlp_out', ('model.layers.0.mlp.down_proj',))]

```

## Write an adapter

An adapter has `matches(model)`, True for the models it fixes, and `adjust(model, graph)`, which returns the
corrected graph. Without one, `qda plan` refuses Gemma 4's stacked experts because their router is nested:

```text
error: fused expert bank model.layers.0.experts: no router beside it (a sibling with one weight row per expert)
```

The router is `model.layers.0.router`, and the expert branch reads `pre_feedforward_layernorm_2`. The
`gemma4_experts` adapter supplies those paths and the configured top-k (abridged from `arch/adapters/gemma4.py`):

```python
from dataclasses import replace

from qda_quant.arch import ExpertBank
from qda_quant.arch.derive import routes_itself, text_config


class Gemma4Experts:
    def matches(self, model):
        config = text_config(model)
        return config.model_type == "gemma4_text" and config.enable_moe_block

    def adjust(self, model, graph):
        readers = dict(graph.extra["readers"])
        layers = []
        for layer in graph.layers:
            prefix = layer.module
            router = model.get_submodule(f"{prefix}.router")
            bank = ExpertBank(
                f"{prefix}.experts",
                f"{prefix}.router",
                text_config(model).top_k_experts,
                fused=True,
                custom_router=routes_itself(router),
            )
            norm = f"{prefix}.pre_feedforward_layernorm_2"
            readers[norm] = (*readers.get(norm, ()), bank.experts)
            layers.append(replace(layer, banks=(*layer.banks, bank)))
        return replace(graph, layers=tuple(layers), extra={**graph.extra, "readers": readers})


GEMMA4_EXPERTS = Gemma4Experts()
```

Register the instance, not the class, from your package's [registration hook](extend.md#the-registration-hook):

```python
def register():
    from qda_quant.arch.adapters import ADAPTERS

    ADAPTERS.register("my_layout", "my_pkg.arch:MY_LAYOUT")
```

Check that it ran: `describe(model).adapter` is `"my_layout"`, and a build's `resolved.yaml` lists `my_layout`
under `provenance.plugins`. `describe` applies at most one fix-up adapter and one stream declaration; two fix-up
adapters that match one model are refused, naming both.

Test it as `packages/qda-quant/tests/test_arch_gemma4.py` does: build a tiny config (`Gemma4TextConfig`), check
`graph.adapter`, the bank's router and top-k, and that `graph.extra["dropped"]` is empty.

## Declaring a multimodal model's residual stream

For a model with a vision tower, a projector or an MTP head outside the text model, where derivation does not reach:
`rotate` with `residual: true` rotates a Linear there only if a declaration says it writes the residual stream, and
refuses, naming the module, when no adapter declares it.

Declare it with an adapter that subclasses `StreamDeclaration` and returns `stream(model)`:

```python
import torch
from qda_quant.arch.adapters import SourceBlock, StreamDeclaration


class MyVL(StreamDeclaration):
    def matches(self, model):
        return getattr(model.config, "model_type", "") == "my_vl"

    def stream(self, model):
        return {
            "writers": {
                "model.vision_tower": (),  # never writes the stream
                "model.projector": ("model.projector.linear_2",),  # its last Linear does
            },
            "blocks": {
                "mtp": SourceBlock(
                    fc="mtp.fc",
                    inputs=("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"),
                    folds=(("mtp.layers.0.input_layernorm", ("mtp.layers.0.self_attn.q_proj",)),),
                    writers=("mtp.layers.0.self_attn.o_proj",),
                    norm="mtp.norm",
                    kept=(),
                    offset=1.0,
                )
            },
        }

    def probe(self, model, root):
        if root != "model.projector":
            return None
        width = model.config.vision_config.hidden_size
        return {"image_features": torch.randn(4, width, dtype=torch.float64)}
```

- `writers`: every module outside the text model, with the Linears in it whose output is the residual stream (an
  empty tuple when it never reaches it).
- `blocks`: a head outside the text model, such as an MTP head, whose weights `rotate` rewrites from the source
  checkpoint; its last norm is stored as the best constant fit.
- `probe(model, root)`: the keyword inputs `rotate` runs each loaded module that has writers on, before and after
  rotation; it refuses unless the output comes out rotated, and refuses a module the probe returns None for
  (`packages/qda-quant/tests/steps/test_rotate_stream.py`).

A step that changes the residual basis must carry the declared modules (`rotate` does; `permute` refuses).

## Declaring serving names and layouts

For a checkpoint that a stock engine loads while leaving some weights at initialization: compare the checkpoint's
names, config and payloads with the engine's loader, then add the smallest declaration to the adapter:

| mismatch | declaration |
|---|---|
| a float module needs an engine alias in `quantization_config.ignore` | `runtime_names(model)` |
| a fused runtime regex needs the shard names the engine probes | `FusedRuntime(checkpoint, shards)` |
| stored tensors need a renamed module tree | `serve_names(model)` |
| packed fields need different names or per-expert slices | `export_layout(name, shape)` |
| saved config fields differ from the loader's schema | `export_config(config)` |
| a graph/format combination cannot load in stock vLLM | `check_served(graph, fmt)` |
| more linears should default to float when `ignore` is omitted | `default_ignore(model)` |

Prove it with a tiny model whose weights differ from the engine's initialization: every payload must be loaded (its
value changed or a loader wrote it), and a forward must run before you claim the engine serves it.

## Adding an adapter to QDA

For an adapter added to this repository rather than to your own package:

- Put it in its own module under `qda_quant/arch/adapters/` and its lazy reference in that package's `ADAPTERS`
  table. Model-name special cases belong only there; everything else reads the `ModelGraph`.
- Add a CPU test on a tiny random model (`packages/qda-quant/tests/tiny.py` builds them; `tests/test_arch.py` shows
  the checks).
