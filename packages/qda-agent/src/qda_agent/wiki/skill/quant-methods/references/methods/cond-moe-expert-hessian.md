# cond-moe-expert-hessian
Addresses: on a mixture-of-experts layer, an all-token expert Hessian includes tokens the router does not send to that expert, and ignores the gate weight that scales the expert's output. Measured as: per expert, the routed-token count and the relative distance between its all-token H and its routed, gate-weighted H.
Mechanism: Build each expert's GPTQ Hessian from its routed tokens, weighted by the router's gate weight for that expert (MoEQuant's affinity-guided calibration). An expert without routed tokens keeps the all-token Hessian.
Effect: each expert's H becomes a gate-weighted sum over its routed tokens; other tokens are removed from it.
Preconditions: MoE with per-expert linears and router gate weights available at calibration; dense models: not applicable.
Inference overhead: none.
Pitfalls: an expert with few routed tokens has a rank-deficient H; check the counts before using its solve. Second: routing the quantized stream at a precision other than the server's, or on the source's hidden state instead of its own, gives each expert a Hessian of tokens it will not see when served.
Origin: Hu et al. (MoEQuant) observed that the gate weight differs by orders of magnitude among one expert's tokens, and that it scales the expert's output; they put it into the second-moment accumulation.
Source: Hu et al., "MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance", ICML 2025. https://arxiv.org/abs/2505.03804 ; routed-token counts: Frantar and Alistarh, "QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models", MLSys 2024. https://arxiv.org/abs/2310.16795
