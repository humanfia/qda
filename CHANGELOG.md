# Changelog

## 0.1.0 — 2026-11-19

Linux, NVIDIA GPU, Python 3.12. Pinned stack: torch 2.14.0, transformers 5.20.2, vLLM 0.33.0 ([install](docs/install.md)).

### Added

- The `qda`, `qda-kl` and `qda-agent` CLIs; the [CLI reference](docs/cli.md) lists their commands and the ones 0.1.x
  keeps stable.
- 5 formats: W4A16, W3A16, W4A8, W4A4 (NVFP4), GGUF Q4_K_M ([formats](docs/formats.md)).
- 6 GPU rows in the serving matrix: H100 / H200 / H800, A100, L40S, RTX PRO 6000, B200, GB300
  ([matrix](docs/formats.md#does-my-gpu-run-it)).
- 30 published checkpoints of 10 models on [Hugging Face](https://huggingface.co/qda-lab).
- Agent search under a bits-per-weight budget, default 4.4 (W4A16, W4A8), 3.4 (W3A16), 4.8 (W4A4)
  ([budget](docs/search.md#budget)).

### Not in 0.1.0

- W4A8 off Hopper; W4A4 off B200 / GB300 / RTX PRO 6000. QDA refuses both before building.
- SGLang: W3A16, W4A8, and the W4A16 checkpoints of Nemotron-3-Nano-30B-A3B, Qwen3-Coder-30B-A3B and
  Gemma-4-26B-A4B.
