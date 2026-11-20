# Third-party notices

## llm-compressor

The `parent` objective candidate grid in `qda_quant.steps.awq.AWQ.grid` follows
`AWQModifier._get_grid_search_params` in llm-compressor 0.15.0, tag/version `0.15.0`, licensed under Apache-2.0
(see `LICENSE`). The column-selector reduction in `qda_quant.solve._triton._block` is also attributed to its GPTQ
kernel conservatively: the origin is undetermined. These files adapt the source to QDA's interfaces and tensor
layout. The inspected license and AWQ source contain no project-specific copyright notice.

The installed 0.15.0 distribution contains no `NOTICE` file. The source checkout at `de16f38` does; the relevant
project attribution is copied exactly below. Its notices for model implementations, images, audio and unrelated
dependencies do not pertain to the two attributed spans.

```text
LLM Compressor

This product includes software developed in association with the vLLM Project (https://github.com/vllm-project).
```

## GSQ

The GSQ trainer and Gumbel rounding in `qda_quant.steps.gsq` and `qda_quant.steps._gumbel` derive from
[IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ/tree/03fc16484c369e3127225615d5e03e8d3a6043e3),
commit `03fc16484c369e3127225615d5e03e8d3a6043e3`, licensed under Apache-2.0 (see `LICENSE`).

Paper: Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtić, Max Kleinegger,
and Dan Alistarh. *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax
Sampling*, [arXiv:2604.18556](https://arxiv.org/abs/2604.18556).

## IFBench

The IFBench instruction checkers in `qda_kl.third_party.ifbench` are copied from
[allenai/IFBench](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d), commit
`1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`, licensed under Apache-2.0 (see the `LICENSE` beside them). The only
edit is the import rewrite recorded in that directory's `VENDOR.json`, which also records the upstream and
vendored SHA-256 of every file. The IFBench test set (`allenai/IFBench_test`, ODC-BY-1.0) is downloaded at the
revision the `ifbench` task YAML pins and is never committed here.

## Lion

The Lion update in `qda_quant.steps._gumbel` follows
[`lion_pytorch` 0.2.5](https://pypi.org/project/lion-pytorch/0.2.5/), under this license:

```text
MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## LiveCodeBench

The LiveCodeBench grader in `qda_agent.third_party.livecodebench` is the official test runner from
[LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24),
commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, copied verbatim (no rewrites; see `VENDOR.json`).
It is licensed under MIT, and its license text is copied verbatim below.

```text
MIT License

Copyright (c) 2024 LiveCodeBench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The test cases are not vendored: they are downloaded from the
[`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite)
dataset at revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.

## llama.cpp

The Q4_K and Q6_K quantizers in `qda_quant.gguf.kernel` are a line-by-line port of `quantize_row_q4_K_ref` and
`quantize_row_q6_K_ref` (with their helpers `make_qkx2_quants` and `make_qx_quants`) from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/9e2a04a439edc49f08c5acdafefae19dfa25a658),
commit `9e2a04a439edc49f08c5acdafefae19dfa25a658`, licensed under MIT. The port reproduces the reference's fp32
arithmetic and summation order so the bytes match; the vectorized path was written for QDA.

`qda_quant/gguf/_quant.c` (compiled into linux wheels by `hatch_build.py`) carries the same functions verbatim
from that commit's `ggml/src/ggml-quants.c` and `ggml/src/ggml-impl.h`, plus the two block layouts of
`ggml/src/ggml-common.h`; only names were mapped (`GGML_RESTRICT`, the fp16 conversion names, the block structs).
It exports `quantize_row_q4_K_ref` and `quantize_row_q6_K_ref` unchanged, so its bytes are the library's.

The Hugging Face to GGUF converter is not vendored: a build runs llama.cpp's own `convert_hf_to_gguf.py` from a
checkout at the pinned commit (see [`docs/gguf.md`](docs/formats.md)), so its files stay upstream's. The
export's byte check against `llama-quantize` from that commit is what the parity claims rest on.

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
