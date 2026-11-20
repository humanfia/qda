---
name: quant-methods
description: Mechanism cards for post-training weight quantization of a language model (rotations, grids and solvers, objectives, learned PTQ, conditioners, calibration sets). Use when a quantized build needs a method, a start to search from, or a measured state of weights or activations needs a mechanism that addresses it.
---

# quant-methods

26 mechanism cards, 10 phenomena. The two tables below are generated from the pages they index; read a page
under `references/` when its row applies.

Every card has the same parts: `Addresses:` a phenomenon and how to measure it on any model (`Measured as:`), `Mechanism:` the method, `Effect:` how the method moves that measured quantity, `Preconditions:`, `Inference overhead:`, `Pitfalls:`. A card says nothing about how a build of any model came out; whether a method moves the quantity on your model, and what that is worth, is your measurement. This is a catalogue of what can be tried, in no order. Where a card has `Origin:`, it is how the publication's authors got from the phenomenon to the method; `Source:` is the publication and its official code.

## Phenomenon -> how to see it on your model -> mechanisms

| phenomenon | how to see it on your own model | cards |
|---|---|---|
| [group-scale-dominated-by-few-channels](references/phenomena/group-scale-dominated-by-few-channels.md): inside a quantization group of a weight matrix's input columns, the ratio of the group's largest absolute value to its... | for one Linear, reshape its input dimension into groups of the container's group size and histogram absmax/rms per group | [rot-hadamard-r1](references/methods/rot-hadamard-r1.md) [rot-hadamard-r2](references/methods/rot-hadamard-r2.md) [rot-spinquant](references/methods/rot-spinquant.md) [rot-channel-perm](references/methods/rot-channel-perm.md) [cond-awq](references/methods/cond-awq.md) [cond-awq-moe](references/methods/cond-awq-moe.md) [grid-scheme](references/methods/grid-scheme.md) [grid-clip-search](references/methods/grid-clip-search.md) [grid-hqq-proximal](references/methods/grid-hqq-proximal.md) [grid-rtn](references/methods/grid-rtn.md) [lptq-omniquant](references/methods/lptq-omniquant.md) |
| [hessian-dominated-by-few-tokens](references/phenomena/hessian-dominated-by-few-tokens.md): at a module's input, a small fraction of the token positions holds most of the sum of squared activation norms over the... | hook one Linear's input over the calibration set you intend to build with, record ||x_t||^2 per token position, sort it... | [obj-gradient-token-weights](references/methods/obj-gradient-token-weights.md) [cond-moe-expert-hessian](references/methods/cond-moe-expert-hessian.md) |
| [ill-conditioned-calibration-hessian](references/phenomena/ill-conditioned-calibration-hessian.md): the calibration Hessians of a model's modules differ in conditioning | during one calibration pass, keep each module's accumulated H and compute the eigenvalues of the normalised, symmetric... | [grid-gptq](references/methods/grid-gptq.md) [grid-group-cd](references/methods/grid-group-cd.md) [cond-moe-expert-hessian](references/methods/cond-moe-expert-hessian.md) |
| [local-damage-is-not-repair-value](references/phenomena/local-damage-is-not-repair-value.md): the modules that carry the most local output-error energy are not the modules whose repair moves the model's output the... | take one whole group of modules from the donor into the base and read the spliced checkpoint's output distance to the... | [cond-module-mixing](references/methods/cond-module-mixing.md) [obj-gradient-token-weights](references/methods/obj-gradient-token-weights.md) |
| [local-objective-vs-model-output](references/phenomena/local-objective-vs-model-output.md): the quantity the solver minimises and the quantity you care about move independently | build two variants that differ in one knob of the local objective (damping, observer, feedback range, token weighting) | [obj-fp-target-refit](references/methods/obj-fp-target-refit.md) [obj-gradient-token-weights](references/methods/obj-gradient-token-weights.md) [lptq-e2e-scale-tune](references/methods/lptq-e2e-scale-tune.md) [lptq-block-ap](references/methods/lptq-block-ap.md) [lptq-qat-kd](references/methods/lptq-qat-kd.md) [lptq-gsq](references/methods/lptq-gsq.md) [lptq-signround](references/methods/lptq-signround.md) [grid-group-cd](references/methods/grid-group-cd.md) [obj-distillation-loss](references/methods/obj-distillation-loss.md) |
| [long-generation-drift](references/phenomena/long-generation-drift.md): the model's behaviour over a long generation moves while its agreement with the source over short contexts does not | teacher-force a fixed text and record the per-position agreement with the source | [lptq-e2e-scale-tune](references/methods/lptq-e2e-scale-tune.md) [obj-gradient-token-weights](references/methods/obj-gradient-token-weights.md) |
| [mlp-intermediate-channel-outliers](references/phenomena/mlp-intermediate-channel-outliers.md): at the input of the projection that reads the MLP's intermediate activation, a few channels are magnitude outliers | hook the intermediate activation, take the per-channel mean of |x| over a calibration batch, and sort it | [rot-channel-perm](references/methods/rot-channel-perm.md) [cond-awq](references/methods/cond-awq.md) [grid-clip-search](references/methods/grid-clip-search.md) |
| [per-module-damage-is-context-dependent](references/phenomena/per-module-damage-is-context-dependent.md): a module's damage measured alone does not predict its damage in the joint build | measure agreement with the source on a fixed set of positions for each single-module swap, then for the full build | [cond-module-mixing](references/methods/cond-module-mixing.md) [obj-fp-target-refit](references/methods/obj-fp-target-refit.md) |
| [routing-and-mixer-sensitivity](references/phenomena/routing-and-mixer-sensitivity.md): a discrete decision inside the stack flips | on identical token sequences, report per layer the fraction of positions whose router top-k set differs between source... | [cond-awq-moe](references/methods/cond-awq-moe.md) [cond-moe-expert-hessian](references/methods/cond-moe-expert-hessian.md) [rot-hadamard-r1](references/methods/rot-hadamard-r1.md) |
| [served-grid-differs-from-stored](references/phenomena/served-grid-differs-from-stored.md): the weights the kernel computes with are not the weights the checkpoint stores | reconstruct one module's weights from the checkpoint's packed codes and scales twice, with the scales as stored and... | [grid-served-scale](references/methods/grid-served-scale.md) [cond-head-embed-served-form](references/methods/cond-head-embed-served-form.md) [grid-scheme](references/methods/grid-scheme.md) |

## Cards

`*` = costs something at inference.

| card | kind | addresses | |
|---|---|---|---|
| [rot-channel-perm](references/methods/rot-channel-perm.md) | rotation / permutation | channel order is arbitrary, so one heavy channel can share a quantization group with... |  |
| [rot-hadamard-r1](references/methods/rot-hadamard-r1.md) | rotation / permutation | a few residual-stream channels with large magnitude force the group scale of every... |  |
| [rot-hadamard-r2](references/methods/rot-hadamard-r2.md) | rotation / permutation | outlier value channels inside one head give o_proj's input group (one head is typically... |  |
| [rot-spinquant](references/methods/rot-spinquant.md) | rotation / permutation | a random rotation flattens a given model's channel concentration by an amount that is... |  |
| [obj-distillation-loss](references/methods/obj-distillation-loss.md) | objective / loss | a next-token training loss uses only the label, not the source model's full output... |  |
| [obj-fp-target-refit](references/methods/obj-fp-target-refit.md) | objective / loss | quantization error accumulates block by block, and a symmetric local objective does not... |  |
| [obj-gradient-token-weights](references/methods/obj-gradient-token-weights.md) | objective / loss | the effect of a module's output error on the model's output varies per token and per... |  |
| [grid-clip-search](references/methods/grid-clip-search.md) | grid, quantizer, format | one outlier sets a group's range and leaves the other weights on a coarse grid |  |
| [grid-gptq](references/methods/grid-gptq.md) | grid, quantizer, format | correlated input channels make independent rounding errors add up in a layer's output |  |
| [grid-group-cd](references/methods/grid-group-cd.md) | grid, quantizer, format | a single forward pass compensates each column only onto later columns |  |
| [grid-hqq-proximal](references/methods/grid-hqq-proximal.md) | grid, quantizer, format | a least-squares fit of a group's scale is pulled by the few largest residuals, entries... |  |
| [grid-rtn](references/methods/grid-rtn.md) | grid, quantizer, format | a group's weights must map onto a fixed number of levels |  |
| [grid-scheme](references/methods/grid-scheme.md) | grid, quantizer, format | in-group outliers inflate a group's scale, a skewed group range wastes half of a... |  |
| [grid-served-scale](references/methods/grid-served-scale.md) | grid, quantizer, format | the kernel re-rounds the stored group scales at load (for example to an e4m3 value... |  |
| [lptq-block-ap](references/methods/lptq-block-ap.md) | learned / trained | error accumulates block by block |  |
| [lptq-e2e-scale-tune](references/methods/lptq-e2e-scale-tune.md) | learned / trained | block-local and layer-local objectives leave small per-layer errors that compound along... |  |
| [lptq-gsq](references/methods/lptq-gsq.md) | learned / trained | at three bits, code choice is a combinatorial problem that greedy column-wise... |  |
| [lptq-omniquant](references/methods/lptq-omniquant.md) | learned / trained | the most extreme weights of a group set its range, so a min/max grid spends its step on... |  |
| [lptq-qat-kd](references/methods/lptq-qat-kd.md) | learned / trained | every one-shot method optimises a local surrogate, so a model-level output gap remains |  |
| [lptq-signround](references/methods/lptq-signround.md) | learned / trained | round-to-nearest and fixed clipping do not minimise the block-output error |  |
| [cond-awq](references/methods/cond-awq.md) | conditioner, MoE, assembly | a few input channels with large activation magnitude dominate a layer's output error,... |  |
| [cond-awq-moe](references/methods/cond-awq-moe.md) | conditioner, MoE, assembly | on a mixture-of-experts layer, a dense scale-migration mapping either misses the router... |  |
| [cond-head-embed-served-form](references/methods/cond-head-embed-served-form.md) | conditioner, MoE, assembly | the output head and the input embedding are vocabulary-wide matrices whose error no... |  |
| [cond-module-mixing](references/methods/cond-module-mixing.md) | conditioner, MoE, assembly | two builds of one served scheme carry their errors on different serving units |  |
| [cond-moe-expert-hessian](references/methods/cond-moe-expert-hessian.md) | conditioner, MoE, assembly | on a mixture-of-experts layer, an all-token expert Hessian includes tokens the router... |  |
| [meas-small-difference](references/methods/meas-small-difference.md) | other | two builds of one source model can differ by less than the sampling noise of one... |  |

## Other pages

- [calibration](references/calibration.md) — the properties of a calibration set, each as phenomenon, method and effect.
- [capability](references/capability.md) and [stacking](references/stacking.md) — generated from the installed library: which `qda build` steps implement each card, and what kind of step each one is and how steps stack in a recipe.
