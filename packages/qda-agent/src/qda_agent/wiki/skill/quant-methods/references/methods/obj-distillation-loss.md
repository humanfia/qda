# obj-distillation-loss
Addresses: a next-token training loss uses only the label, not the source model's full output distribution. Measured as: the student's divergence from the source's next-token distribution on training data kept out of training.
Mechanism: Minimise the per-position divergence from the source (bf16) model's next-token distribution to the student's, instead of cross-entropy on the label. Any trainable set can use it; the teacher is the unquantized source model.
Effect: the student's whole next-token distribution moves toward the source's on the training data.
Preconditions: teacher logits for the training data (online or cached) at the training sequence length.
Inference overhead: none.
Pitfalls: the training data decides the target; divergence on source-generated text is a different objective from divergence on external text. Second: a quantized teacher pulls the student toward that build's errors; check that the teacher is the unquantized source.
Source: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015. https://arxiv.org/abs/1503.02531
