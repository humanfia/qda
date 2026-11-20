# cond-module-mixing
Addresses: two builds of one served scheme carry their errors on different serving units; the scheme fixes the bits, so the source build of each unit is a free variable. Measured as: per serving unit, the change of its output on its real input when taken from the other build; for the assembled checkpoint, its output distance to the source on a fixed text beside the same distance of each end.
Mechanism: Assemble a checkpoint from a base A and a donor B of the same scheme (both symmetric or both asymmetric, same bits and group size): the worker names the serving units or groups to take from B; every other tensor is A's.
Effect: taken units equal B's tensors byte for byte, all others A's; bits and serving cost are unchanged.
Preconditions: one scheme across the whole checkpoint; non-quantized tensors identical between A and B (the same transforms).
Inference overhead: none.
Pitfalls: donor modules were solved against the donor's own upstream errors, so taken units do not compose additively; measure the assembled checkpoint as a whole. Second: packed-tensor surgery touches tensors the config does not mention, so a spliced checkpoint can serve while wrong; check bytes and coverage on the assembled artifact, not a successful start.
Origin: Li et al. (QuantMoE-Bench) observed that the components of a routed model do not degrade alike at one bit width (attention, the shared expert, the earliest blocks and frequently routed experts deviate more), and moved bits toward them.
Source: engineering (module splicing within one served scheme), no publication; component-wise sensitivity of a routed model: Li et al., "QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts", arXiv 2024. https://arxiv.org/abs/2406.08155
