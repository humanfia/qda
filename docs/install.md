# Install

For anyone who runs QDA on their own Linux GPU machine. Check [requirements](#requirements), run the
[steps](#steps). Read a later section only when doctor prints `MISS` or its heading names your machine.

## Requirements

| | need |
|---|---|
| OS | Linux |
| GPU | compute capability 8.0 (A100), 8.6 (A10, A40, RTX 30xx), 8.9 (L40S, RTX 4090), 9.0 (H100, H200), 10.0 (B200), 10.3 (GB300), 12.0 (RTX PRO 6000); which formats each serves: [GPU table](formats.md#does-my-gpu-run-it) |
| NVIDIA driver | 580 or newer (CUDA 13.0); 570 on A100 / H100 / L40S with [CUDA compat](#an-older-driver-cuda-12) |
| Python | 3.12; `uv venv --managed-python` downloads it |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (no root) |
| C compiler | `gcc`; and `c++` or `CXX` unless you run `vllm download-kernels` |
| disk | 9.3 GB (9.2 GB venv, 0.1 GB Python) with the uv cache on the same filesystem; 18.5 GB on two filesystems; +9.7 GB for `vllm download-kernels` |
| bubblewrap | only for a search at `isolation.level: sandbox` or `strict`, and LiveCodeBench: `sudo apt install bubblewrap` |

## Steps

```bash
git clone https://github.com/humanfia/qda.git && cd qda
uv venv -p 3.12 --managed-python --seed && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
vllm download-kernels   # only on L40S, RTX 4090, RTX PRO 6000, B200, GB300, or with no nvcc
qda doctor
```

Cold uv cache, A100 host: 226 packages, 9.6 GB download, 32 min at 5 MB/s. `constraints.txt` pins torch 2.14.0,
transformers 5.20.2 and vLLM 0.33.0.

- Run `vllm download-kernels` before `qda doctor`: its `MISS C compiler` for `c++` and `MISS FlashInfer kernels`
  clear once they are installed.
- Build here, serve elsewhere: leave out `"./packages/qda-kl[vllm]"` (5.6 GB less). `qda build` runs without it.
- No clone: install from git. The [quickstart](quickstart.md) still needs the clone for its configs.

```bash
uv venv -p 3.12 --managed-python --seed && . .venv/bin/activate
uv pip install -c https://raw.githubusercontent.com/humanfia/qda/main/constraints.txt \
  "qda-quant[hf,gguf] @ git+https://github.com/humanfia/qda.git#subdirectory=packages/qda-quant" \
  "qda-kl[vllm] @ git+https://github.com/humanfia/qda.git#subdirectory=packages/qda-kl" \
  "qda-agent @ git+https://github.com/humanfia/qda.git#subdirectory=packages/qda-agent"
```

## `qda doctor`

Run it after the install and after any refusal. It prints one line per check, then a count. A100 host with
`CXX` set:

```text
OK   disk room: QDA_CACHE, HF_HOME, VLLM_CACHE_ROOT and TMPDIR on /data: 37.5 GB free, 898,491,983 inodes
OK   GPU: 8 × NVIDIA A100-SXM4-80GB, driver 570.195.03
OK   card health: no fault on cards [0, 1, 2, 3, 4, 5, 6, 7]
OK   CUDA driver: torch 2.14.0+cu130 (CUDA 13.0) starts CUDA on this driver
OK   vLLM: 0.33.0
OK   engine cache: /data/qda-cache/triton, /data/qda-cache/vllm, /tmp/qda_<user>/flashinfer, /tmp/qda_<user>/inductor (set: TRITON_CACHE_DIR, VLLM_CACHE_ROOT): lockable
OK   C compiler: /usr/bin/gcc
---- Hub access: cannot be checked in advance; checked by the build's model download and calibration step (an offline node needs both in the Hub cache)
---- kernel JIT: cannot be checked in advance; checked by the first build or engine start, which compiles its kernels
---- free GPU memory: cannot be checked in advance; checked by each build or engine start, on its own cards
---- ports: cannot be checked in advance; checked by the engine start, which binds its port
checked 7, 0 MISS, 4 cannot be checked in advance
```

`disk room` prints one line per filesystem, so caches on two filesystems give 8 `OK` lines and `checked 8`.
Each line is tagged:

| tag | meaning |
|---|---|
| `OK` | passes |
| `MISS` | a command would refuse; the line names the fix ([troubleshooting](troubleshooting.md#install-and-qda-doctor)) |
| `skip` | does not apply here |
| `----` | checked later, by the step the line names |

Exit status 3 when any line is `MISS`, else 0. `qda doctor --search` exits 2 when the run config does not load.

- `qda doctor --search work/q8b/search.yaml` adds bubblewrap, the run directory's disk and the agent's [sign-in](search-config.md#sign-in).
- `qda doctor --report` prints versions, driver and filesystems (no paths) for a bug report.

## An older driver (CUDA 12)

For driver 570 (CUDA 12.8), where `qda doctor` prints `MISS CUDA driver`. Upgrade to 580, or, on
A100, H100 or L40S (not GeForce), load NVIDIA's CUDA 13.0 forward-compatibility libraries, no root:

```bash
mkdir -p ~/cuda-compat && cd ~/cuda-compat
curl -LO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb
dpkg-deb -x cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb .
export LD_LIBRARY_PATH=$HOME/cuda-compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python -c "import torch; torch.zeros(1).cuda(); print('ok')"
```

RHEL / Rocky: unpack the `.rpm` with `rpm2cpio <file> | cpio -idm`. Put the `export` in every shell that runs
`qda`, `qda-kl`, `qda-agent` or `vllm serve`.

Do not install another torch: `constraints.txt` pins vLLM 0.33.0, built for CUDA 13, and a cu128 torch breaks it.

## Open-file limit

A build on N cards opens about one file per calibration sample per card (512 samples, 4 cards: 2.5k). When it
stops with `error:` and `fix:` on open files, raise the hard limit (root, or ask your admin):

```bash
ulimit -n 65536
```

## Network

- Mirror: `export HF_ENDPOINT=https://<your mirror>`.
- `429 Too Many Requests` during `prepare`: rerun the same command; it skips finished steps.
- GPU node without internet: download on a machine with internet into the same `HF_HOME`, then go offline:

```bash
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
hf download mlabonne/open-perfectblend --repo-type dataset --revision af60f3c18201652a83a93f46fcfee1b646ba3df7
export HF_HUB_OFFLINE=1     # on the GPU node, before qda build / qda-kl
```

Offline, name the model as `<id>@<revision>`, with the revision you downloaded, wherever a command takes it; a
bare id stops and prints the cached revision. `qda-kl kl` reads the revision from the build's `resolved.yaml`, but
its default prompts download 11 datasets on the first call; offline, measure on the local prompt file:

```bash
qda-kl kl qwen3-0.6b-w4a16 --prompts packages/qda-kl/examples/prompts-small.jsonl
```

## Where things go

| what | default | move it |
|---|---|---|
| models, datasets | `~/.cache/huggingface` | `export HF_HOME=/big/disk/hf` |
| QDA cache (calibration, KL references) | `~/.cache/qda` | `export QDA_CACHE=/big/disk/qda` |
| engine compile caches | `/tmp/qda_<user>/{flashinfer,inductor,triton,vllm}` for engines qda starts; a plain `vllm serve` keeps vLLM's own default, `~/.cache/vllm`, which can fill a quota'd home | `export FLASHINFER_WORKSPACE_BASE=... TORCHINDUCTOR_CACHE_DIR=... TRITON_CACHE_DIR=... VLLM_CACHE_ROOT=...`; before `vllm serve`: `(umask 077; mkdir -p /tmp/qda_$USER/vllm)` and `export VLLM_CACHE_ROOT=/tmp/qda_$USER/vllm` |
| uv Python and packages | `~/.local/share/uv/python`, `~/.cache/uv` | `export UV_PYTHON_INSTALL_DIR=... UV_CACHE_DIR=...` |

Put the compile caches on a node-local disk (not NFS, GPFS or Lustre), in a directory only you can write (mode
`0700`), not a symlink. Nothing deletes them; `rm -r` them when done.
