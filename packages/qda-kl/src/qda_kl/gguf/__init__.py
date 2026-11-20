"""The GGUF harness shipped with qda-kl: `python -m qda_kl.gguf.run CONFIG.yaml`.

`run` starts a pinned `llama-server`, checks the tensors it loaded, and runs long-generation and throughput
checks. `probe.cpp` is the bridge the load check preloads, compiled against the same llama.cpp checkout. KL of a
`.gguf` is `qda-kl score`'s (it decodes the file and reads it with vLLM).
"""
