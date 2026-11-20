# calibration
Scope: every calibrated method (GPTQ and its variants, a scale-migration search, the learned PTQ cards) turns a set of texts into per-module statistics -- a Hessian H = Σ x xᵀ, per-channel activation magnitudes, a block loss. The build represents the served activations only as well as those statistics do. Each section is one property of the calibration set; none has a neutral default.

## domain and source
Addresses: a domain that served traffic never exercises still takes a share of H, and a served domain absent from the set leaves its directions unexcited. Measured as: per module, the distance between the calibration H and the H of served-style text (relative Frobenius norm, overlap of the leading eigenvectors), and each domain's share of H.
Mechanism: choose the mix of sources -- pretraining-style text, chat transcripts, reasoning traces (a long thinking segment followed by a conclusion), text the source model generated -- and fix a per-domain quota.
Effect: the expected H is each domain's second moment weighted by its token share. As the mix approaches the served mix, the expected calibration H moves toward the served-style H; a finite sample keeps its estimation error.
Pitfalls: a public reasoning trace can contain, in full or in part, the statement of a problem that is also used to measure builds. Second: a set the source model generated puts the source's output distribution into H; read a build-to-source distance on text the source did not generate. Third: change the calibration and the solver separately, or neither change is attributable.

## rendering: chat template and thinking segments
Addresses: the served model sees its inputs in a chat template, and a long-thinking model spends most generated positions inside a thinking segment. Plain text, or chat text without that segment, gives a different token and position distribution. Measured as: the fraction of calibration positions inside template markup and inside thinking segments, beside the same fractions in served generations; per module, the distance between the calibration H and the H of served generations.
Mechanism: before tokenizing, render every calibration text through the served model's own chat template in its served mode, with the thinking segment where the served model produces one and with the template's begin/end markers. Chunk the rendered stream exactly as the producer will.
Effect: the fractions of template and thinking positions equal those of the reproduced rendering, so the token and position distribution behind each module's H moves toward the served one; content and context can still differ.
Pitfalls: a thinking segment is long, so it moves the stream toward late positions and toward the thinking style; record it as a separate calibration change. Second: a template applied twice (by the data source and by the producer) doubles the markers.

## token budget
Addresses: H is a sum over tokens, so its small eigen-directions are estimated from few effective samples and move when the set is redrawn. Measured as: per module, the distance between the H of two disjoint random subsets of the same texts, and the stability of their smallest eigenvalues; the same difference between two builds of one recipe on two subsets is that recipe's calibration repeat noise.
Mechanism: set the token count (texts times sequence length); the token count, not the text count, sets how well H's small directions are estimated.
Effect: under representative sampling, the subset-to-subset distance of the sample-normalised H shrinks as the token count grows, and the build's dependence on the drawn texts falls. Build time and the host memory of any per-sample cache grow with it.
Pitfalls: a scale-migration search can run on the first N texts at a shorter length while the solver after it runs on the full set; the search statistic then has the repeat noise of the smaller budget.

## packing and position coverage
Addresses: with one text per chunk and texts shorter than the chunk, H holds no statistics for the late positions, and the first positions of every chunk (the attention sink) recur once per chunk. Measured as: the histogram of positions that contribute to H, and the share of Σ‖x_t‖² held by the first positions of each chunk.
Mechanism: pack texts back to back with an end-of-sequence token between them, or fill every chunk to the full length.
Effect: every position of the context contributes to H. The number of chunk-initial positions per token falls, which lowers the sink positions' share of H unless their activations change with it.
Pitfalls: packing also moves the sink and the first tokens of each text, so it changes the heavy-token structure; read the energy curve of hessian-dominated-by-few-tokens on both streams.

## damping
Addresses: directions the texts barely excite have near-zero eigenvalues in H, and the solver's update multiplies by H⁻¹ along them. Measured as: per module, the smallest eigenvalues of the sample-normalised H relative to mean(diag H).
Mechanism: add λ·mean(diag H) to the diagonal before the factorisation; λ decides how much the solver trusts the weakly excited directions.
Effect: every eigenvalue of the damped H is at least λ·mean(diag H), so the update along weakly excited directions is bounded. As λ grows, the update shrinks toward round-to-nearest on the same grid.
Pitfalls: λ is in the recipe, not in the checkpoint's config, so two builds with identical quantization configs can be different solves; read the recipe.
Cards: grid-gptq

## routed-expert statistics
Addresses: on a mixture-of-experts layer, each expert's H can come from every token or only from the tokens the router sends it. Measured as: per expert, the routed-token count and the distance between its all-token H and its routed H.
Mechanism: accumulate each expert's statistics on its routed tokens only.
Effect: each expert's H moves to the distribution of the tokens it processes when served; an expert with no routed token keeps the all-token H.
Cards: cond-moe-expert-hessian, cond-awq-moe
