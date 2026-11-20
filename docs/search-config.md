# Search config

When you change a key with `--set` or write a run config by hand, look it up here. `qda-agent prepare` writes a
complete one (`search.yaml`).

## Commands

```sh
qda-agent prepare --model Qwen/Qwen3-8B --out work/q8b --format w4a16 --cards 0 --dry-run
qda-agent search work/q8b --set budget_gpu_hours=8
qda eval work/q8b/run
qda doctor --search packages/qda-agent/configs/search.yaml --set run_dir=runs/q8b-w4a16
qda-agent run packages/qda-agent/configs/search.yaml --set run_dir=runs/q8b-w4a16
```

`search` runs `qda doctor --search`, then `run`. `run` takes a config written by hand. `--set KEY.PATH=VALUE`
overrides one key; the value is YAML. Write a `--set` path relative to the directory you run the command from (the
two lines above write `./runs/q8b-w4a16`), a path in the config file relative to the file's directory. `run` exits 0 at the deadline ([other statuses](troubleshooting.md#exit-status));
during the close, Ctrl-C waits for pending readings and three Ctrl-C abandon them.

## A config by hand

Copy the commented default `packages/qda-agent/configs/search.yaml` and fill in your paths:

```yaml
model: models/Qwen3-8B             # the original bf16 model
format: w4a16                      # w4a16 | w4a8 | w3a16 | w4a4
budget_gpu_hours: 4                # over all cards
cards: [0]
run_dir: runs/q8b-w4a16            # a new directory per run
calib: data/calib.jsonl            # the calibration file the agent's task names
baselines: [ckpts/q8b-gptq]        # at most two; [0] is candidate 0
agent:
  backend: {name: claude, model: claude-opus-5-5, effort: high}
scheduler: {name: qsched, readable: [models/Qwen3-8B]}   # read-only inputs of the agent's jobs
toolset: {name: qda, python: venvs/quant/bin/python}
contamination: {name: ngram, guard: guards/q8b.txt}
judge:
  kl_ref: refs/q8b                 # built once with qda-kl ref
  kl_holdout_ref: data/holdout.jsonl  # prompts no build may calibrate on
  anchor: models/Qwen3-8B
  serve_python: venvs/vllm/bin/python
```

Have ready: the original model, a calibration file, a baseline checkpoint (candidate 0), a KL reference
([build one](verify.md#2-measure-kl)), a Python with vLLM for scoring, a Python with `qda-quant` for the agent's
jobs, and the agent's CLI signed in.

Set `judge.kl_holdout_ref` (required) to a prompt JSONL (`id`, `domain`, `prompt`) outside every path the agent can
read. Every build's calibration is filtered against it and the `kl_ref` prompts.

The deadline is the budget divided by the cards:

```python
>>> from qda_agent.config import RunConfig
>>> cfg = RunConfig.model_validate(
...     {"model": "m", "format": "w4a16", "budget_gpu_hours": 4, "cards": [0], "run_dir": "runs/x",
...      "contamination": {"name": "none"}}
... )
>>> cfg.wall_s / 3600
4.0

```

## Judge keys

The judge scores and checks every checkpoint the agent hands in. Keep the run directory's path short: a Unix socket
lives inside it.

| key | default | what it does |
|---|---|---|
| `judge.budget_bpw` | none; `prepare` writes W4A16 / W4A8 4.4, W3A16 3.4, W4A4 4.8 | largest bits per weight of a checkpoint's decoder Linears, vLLM searches; needs `judge.anchor` ([bits per weight](search.md#bits-per-weight)) |
| `judge.head_lease` | off; `prepare` turns it on for W4A4 | lend what a narrower output head saves to the body ([head lease](formats.md#head-lease)); on, `judge.budget_bpw` and the end-of-run card count the leased head; needs `judge.budget_bpw`, `judge.head: 16-bit` and no `judge.format_map` |
| `judge.budget_bytes` | none; `prepare` writes candidate 0's size for `gguf-q4km` | largest `.gguf` a llama.cpp candidate may be, in bytes; replaces `judge.budget_bpw` |
| `judge.budget_basis` | `file` | which `.gguf` figure `judge.budget_bytes` screens: `file` (the whole file) or `payload` (the stored tensors) |
| `judge.format_menu` | vLLM `[bf16, w8a16, w4a16]`; llama.cpp the target's GGUF types | formats admitted; a W3 search adds `w3a16`, W4A8 adds `w4a8`; W4A4 names `[bf16, w8a8-fp8, w4a4-nvfp4]` |
| `judge.head` | `16-bit` | the output head's storage rule (BF16 or F16) |
| `judge.max_kl` | `1.0` | a reading above it is refused at once |
| `judge.final_rules` | `[handin]` | the rules that pick the result |
| `judge.starts` | `[]` | pre-built starts, `[{name, path, origin}]`; the lowest valid KL among them is the bar |
| `judge.max_run_bytes` | none | a byte cap on the run directory |
| `judge.min_free_bytes` | twice the original's bf16 weights; 10 GB without `judge.anchor` | the free-space floor; 0 turns it off |
| `judge.llamacpp.template` | written by `prepare` | a GGUF search's bf16 GGUF of the model |

**`handin`**: the first valid entry of the agent's latest hand-in list counts, else an earlier list's, else
candidate 0. `FINAL.json`'s `chosen_by` says which. A check queued at the deadline runs in the drain window; one
unfinished at its end is cut, with no verdict.

**Disk.** Below the free-space floor, the start and new builds are refused and the agent's running jobs stop, with
two `DISK FLOOR` lines; judge readings go on. After ten minutes below it, the run submits the best scored valid
entry.

**300B+ MoE on one node.** Set `judge.full_card_moe: true`. It needs eight cards, `judge.kl_layout.tp: 8`,
`judge.require_build_receipt: true` (so `isolation.level: sandbox` or `strict`), and the same local FP8 directory
for `model` and `judge.anchor`. Build `judge.kl_ref` with `qda-kl ref --tp 8`. Set `judge.max_copy_bytes` to the
largest candidate and `judge.max_run_bytes` to at least twice that. Each build then uses all eight cards, one at a
time.

## Sign-in

The agent's CLI signs in from its credential file (`~/.claude/.credentials.json`, `~/.codex/auth.json`), from an
account its backend names, or from an environment variable. Each run gives the CLI a private config directory, so
nothing it writes reaches yours. A run with no sign-in is refused before it starts (exit 3).

| variable | use |
|---|---|
| `ANTHROPIC_API_KEY` | an Anthropic API key |
| `ANTHROPIC_AUTH_TOKEN` | a bearer token for an LLM gateway; also set `ANTHROPIC_BASE_URL` to the gateway's URL in the shell that runs `qda-agent search`. A run signed in by file or account removes `ANTHROPIC_BASE_URL` |
| `CLAUDE_CODE_OAUTH_TOKEN` | a one-year token from `claude setup-token` |
| `OPENAI_API_KEY`, `CODEX_API_KEY`, `OPENAI_BASE_URL` | the `codex` CLI: the key (`OPENAI_API_KEY` when both are set), and the endpoint (default `https://api.openai.com/v1`) |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | `dsh`: the key, and an OpenAI-compatible base URL (with `/v1` if the gateway serves under it) |

At `isolation.level: sandbox` and `strict`, the levels that use bubblewrap, the run's start and
`qda doctor --search` read the expiry in the sign-in file (`~/.claude/.credentials.json`, or the one in
`agent.backend.sign_in_dir`) and refuse a run (exit 3) whose sign-in expires before the deadline plus 30 minutes. Inside bubblewrap the file is read-only, so the CLI cannot refresh it.
With `agent.backend.sign_in_dir` set and `sandbox: {name: bwrap}`, the run starts and logs a warning instead; with
`sandbox: {name: none}` and no `sign_in_dir`, nothing is checked.
At `isolation.level: none` nothing is checked: the CLI refreshes your own file, and the `OK agent sign-in` row's
`covers deadline + 30 min: no` needs no action.

**Before a run longer than your sign-in**, sign Claude Code in with a token that lasts one year and needs no refresh:

1. Run `claude setup-token` and copy the token it prints.
2. `export CLAUDE_CODE_OAUTH_TOKEN=<token>` in the shell that starts the run (and in the job script on a cluster).
   Remove `agent.backend.sign_in_dir` from the config: a token beside it is refused.
3. Run `qda doctor --search <config>`. The `agent sign-in` row reads
   `claude: setup-token OAuth token via CLAUDE_CODE_OAUTH_TOKEN`.

The run passes the token to the agent at every isolation level. Unset it to sign in from the file again.

**When `qda doctor --search` says `codex: credential file in ... expiry coverage unknown`**, codex signs in from
`~/.codex/auth.json` (a ChatGPT sign-in), which records no expiry qda reads: qda checks only that the file exists
and does not refuse the run. At `sandbox` and `strict` with `sandbox: {name: bwrap}` codex cannot refresh that
file. For such a run, `export OPENAI_API_KEY=<key>` (or `CODEX_API_KEY`), remove `agent.backend.sign_in_dir` if
set, and rerun `qda doctor --search <config>`.

`codex app-server`, which the run drives, reads no key variable and no `OPENAI_BASE_URL` by itself. When one of
the codex key variables is set, the run's codex config names a model provider that reads the key from that variable
at each request and sends it to `OPENAI_BASE_URL` (or the official API); the key itself is never written to a file.
The `agent sign-in` row then reads `codex: API key via OPENAI_API_KEY; endpoint https://api.openai.com/v1` (or
`endpoint OPENAI_BASE_URL`). You do not need `codex login --with-api-key`.
A run does not use a `model_provider` other than `openai` that the `config.toml` beside the sign-in file names (in
`agent.backend.sign_in_dir`, else `~/.codex`; the active `profile`'s `model_provider` counts first): the sign-in file
still signs the run in, against the official API. The `agent sign-in` row says so (it is no MISS, and the run start
prints it too): set `OPENAI_API_KEY` and `OPENAI_BASE_URL` to that provider's key and URL.
Under `sandbox` and `strict` the network fence lets the run reach that endpoint's host. An endpoint that answers
`function_call_output requires call_id` rejects codex's goal message: set `agent.goal: false`.

`dsh` with `agent.backend.account` uses that account's endpoint and ignores `DEEPSEEK_BASE_URL`. A `dsh` config
that names no model runs the one its endpoint serves.

## Isolation

| `isolation.level` | the agent and its jobs |
|---|---|
| `none` (default) | plain processes, as you, inheriting your `PATH` and compile caches. The run digests the judge's files at start, before every job and at the end, and refuses a reading when one changed (`run_dir/trust.jsonl`) |
| `sandbox` | the jobs run in bubblewrap: the system, the Python environment and the configured inputs read-only, the work directory writable, the run's `judge/`, the reference data and `$HOME` hidden, no network, system `PATH` only. With `sandbox: {name: bwrap}` the agent's turns run in bubblewrap too, with one way out: the model's API through a host proxy |
| `strict` | as `sandbox` |

Set `isolation.level` and `sandbox` together: `sandbox: {name: bwrap}` takes effect only at `sandbox` or `strict`.
`judge.require_build_receipt` and `judge.custody_dir` need `sandbox` or `strict`. `resolved.yaml` and `gate.json`
record how the jobs ran. Where unprivileged user namespaces are disabled, bubblewrap cannot start; `qda doctor
--search` says so.

Before you publish scores, set `contamination: {name: ngram, guard: FILE}`: it refuses a workspace that holds a
test item. `prepare` writes `contamination: {name: none}`.

On a node that forces a shared pid namespace (the `pid rule` row of `qda doctor --search`), `sandbox` and `strict`
with `sandbox: {name: bwrap}` refuse the run while you have other work on the node. To run anyway, name the
operator's acceptance and watch the run from outside:

```yaml
model: models/Qwen3-8B
format: w4a16
budget_gpu_hours: 4
cards: [0]
run_dir: runs/shared
contamination: none
isolation:
  level: strict
  shared_node:
    accepted_by: the operator who accepts it
    reason: why this run shares the node, in one sentence
```

## Parts

Every part is chosen by name in the config. New parts are [plugins](extend.md).

| part | config key | registry group | built-in |
|---|---|---|---|
| agent backend | `agent.backend` | `qda_agent.backends` | claude, codex, dsh |
| GPU scheduler | `scheduler` | `qda_agent.schedulers` | qsched |
| run launcher | `launcher` | `qda_agent.launchers` | local |
| agent toolset | `toolset` | `qda_agent.toolsets` | qda |
| toolset component | `toolset.components` | `qda_agent.components` | baselines, bpw, data, fisher, gencheck, handin, kl, probe, qda, qsched, serve, source, views, wiki |
| contamination check | `contamination` | `qda_agent.contamination_checks` | ngram, none |
| final rule | `judge.final_rules` | `qda.final_rules` | baseline, best_metric, handin, last_valid |
| agent sandbox | `sandbox` | `qda_agent.sandboxes` | bwrap, none |
| isolation level | `isolation.level` | - | none, sandbox, strict |

To run a subset of the tools, list them: `--set "toolset.components=[qda, kl, qsched, handin]"`.
