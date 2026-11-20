"""Measure a GGUF with pinned llama-server and libllama: python -m qda_kl.gguf.run CONFIG.yaml."""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import yaml
from qda_quant.gguf.llama_cpp import PINNED, commit_of
from transformers import AutoTokenizer

from qda_kl.gencheck import is_loop


def digest(path):
    """Hash file contents without copying large checkpoints into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    """Keep machine-readable evidence with strict finite JSON values."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def request(port, route, body=None):
    """Use the native completion API so raw generated token IDs are available."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as response:
        return json.load(response)


def strict(model, log, llama_cpp):
    """Compare the file inventory with every tensor observed after the actual server load."""
    sys.path.insert(0, str(Path(llama_cpp) / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(model)
    expected = {
        t.name: (int(t.tensor_type), tuple(map(int, t.shape)) + (1,) * (4 - len(t.shape)))
        for t in reader.tensors
    }
    actual = {}
    for line in Path(log).read_text().splitlines():
        if line.startswith("QDA_TENSOR\t"):
            _, name, dtype, *shape = line.split("\t")
            value = (int(dtype), tuple(map(int, shape)))
            if name in actual and actual[name] != value:
                raise ValueError(f"conflicting loaded copies: {name}")
            actual[name] = value

    def field(key):
        found = reader.get_field(key)
        if found is None:
            raise ValueError(f"{model}: no {key} in the GGUF header")
        return found.contents()

    arch = str(field("general.architecture"))
    nextn = reader.get_field(f"{arch}.nextn_predict_layers")
    n_nextn = 0 if nextn is None else int(nextn.contents())
    n_layers = int(field(f"{arch}.block_count"))
    skipped = {}
    for name in expected.keys() - actual.keys():
        match = re.fullmatch(r"blk\.(\d+)\..+", name)
        if match and n_layers - n_nextn <= int(match[1]) < n_layers:
            skipped[name] = {"classification": "MTP-off", "file_type_shape": expected[name]}
    active = {name: value for name, value in expected.items() if name not in skipped}
    if actual != active:
        raise ValueError(
            f"loaded inventory differs: missing={set(expected) - set(actual)}, "
            f"extra={set(actual) - set(expected)}, "
            f"changed={[k for k in expected.keys() & actual.keys() if expected[k] != actual[k]]}"
        )
    ignored = re.findall(r"model has unused tensor ([^ ]+) .*-- ignoring", Path(log).read_text())
    if set(ignored) - skipped.keys():
        raise ValueError(f"server ignored non-MTP tensors: {set(ignored) - skipped.keys()}")
    return {
        "verdict": "PASS-WITH-MTP-OFF" if skipped else "PASS",
        "tensors": len(expected),
        "types": dict(Counter(v[0] for v in expected.values())),
        "tensor_payload_bytes": sum(t.n_bytes for t in reader.tensors),
        "inventory": actual,
        "skipped": skipped,
        "ignored_log_names": ignored,
    }


def state(config, label):
    """Record server lifecycle under the team's per-card start lock."""
    coordination = config["coordination"]
    if coordination is not None:
        with Path(coordination["status"]).open("a") as stream:
            stream.write(f"{label} {coordination['label']} {datetime.now(UTC).strftime('%H:%M:%SZ')}\n")
            stream.flush()


def server(config, out):
    """Start one server; release the start lock only after readiness or failed startup."""
    runtime = config["runtime"]
    command = [
        str(Path(config["build"]) / "bin/llama-server"),
        "-m",
        config["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(runtime["port"]),
        "-ngl",
        str(runtime["gpu_layers"]),
        "-c",
        str(runtime["context"] * runtime["parallel"]),
        "-np",
        str(runtime["parallel"]),
        "-b",
        str(runtime["batch_tokens"]),
        "-ub",
        str(runtime["batch_tokens"]),
        "-t",
        str(runtime["threads"]),
        "-tb",
        str(runtime["threads"]),
        "--check-tensors",
        "--no-repack",
        "--fit",
        "off",
        "--no-context-shift",
        "-ctk",
        runtime["kv_dtype"],
        "-ctv",
        runtime["kv_dtype"],
        "-fa",
        runtime["flash_attention"],
        "--jinja",
        "--log-verbosity",
        "4",
    ]
    write(out / "server-command.json", command)
    env = {**os.environ, "LD_PRELOAD": config["bridge"]}
    lock = None
    if config["coordination"] is not None:
        lock = Path(config["coordination"]["lock"]).open("a")  # noqa: SIM115 - released after startup below
        fcntl.flock(lock, fcntl.LOCK_EX)
    state(config, "STARTING")
    log = (out / "server.log").open("w")
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + runtime["startup_timeout"]
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ValueError(f"server exited {proc.returncode}; see {out / 'server.log'}")
            try:
                request(runtime["port"], "health")
                write(out / "strict.json", strict(config["model"], out / "server.log", config["llama_cpp"]))
                state(config, "READY")
                return proc, log
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        raise ValueError("server readiness timed out")
    except BaseException:
        proc.terminate()
        proc.wait(timeout=60)
        log.close()
        state(config, "DONE startup-failed")
        raise
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def gencheck(config, out):
    """Preserve raw requests and responses and count cuts and periodic suffixes."""
    spec = config["gencheck"]
    prompts = [json.loads(line) for line in Path(spec["prompts"]).read_text().splitlines() if line.strip()]
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("prompt set is empty or contains duplicate IDs")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    counts = {}
    with (out / "generations.jsonl").open("w") as stream:
        for prompt in prompts:
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **spec["chat_template_kwargs"],
            )
            rendered = tokenizer.decode(tokens, skip_special_tokens=False)
            got = request(config["runtime"]["port"], "tokenize", {"content": rendered, "add_special": False})
            if got["tokens"] != tokens:
                raise ValueError(f"{prompt['id']}: HF and GGUF prompt token IDs differ")
            body = {
                "prompt": tokens,
                "return_tokens": True,
                "cache_prompt": False,
                **spec["sampling"],
            }
            response = request(config["runtime"]["port"], "completion", body)
            if response.get("truncated") or response["tokens_predicted"] != len(response["tokens"]):
                raise ValueError(f"{prompt['id']}: truncated context or incomplete output IDs")
            actual = response["generation_settings"]
            for key, value in spec["sampling"].items():
                requested = (
                    struct.unpack("f", struct.pack("f", value))[0] if isinstance(value, float) else value
                )
                if actual[key] != requested:
                    raise ValueError(f"{prompt['id']}: actual {key}={actual[key]!r}, requested {value!r}")
            record = {"id": prompt["id"], "domain": prompt["domain"], "request": body, "response": response}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            domain = counts.setdefault(prompt["domain"], {"prompts": 0, "tokens": 0, "cuts": 0, "loops": 0})
            domain["prompts"] += 1
            domain["tokens"] += len(response["tokens"])
            domain["cuts"] += response["stop_type"] == "limit"
            domain["loops"] += is_loop(response["tokens"], **spec["loop"])
    write(
        out / "gencheck.json",
        {"domains": counts, "prompts_sha256": digest(spec["prompts"]), "differences": spec["differences"]},
    )


def speed(config, out):
    """Report decode-only server time at b1 and aggregate concurrent throughput separately."""
    cards = os.environ.get("GPU_RUN_LEASE", "").split(",")
    if len(cards) != 1 or not cards[0]:
        raise ValueError("throughput requires an exclusive GPU lease")
    holders = Path(config["coordination"]["holders"])
    holder = (holders / f"gpu{cards[0]}.holder").read_text().split(" | ")[0]
    if holder != config["coordination"]["who"]:
        raise ValueError("exclusive lease holder differs from this measurement")
    spec = config["speed"]
    rows = []
    for batch in spec["batches"]:
        if batch > config["runtime"]["parallel"]:
            raise ValueError("speed batch exceeds server parallel slots")
        body = {
            "prompt": spec["prompt_tokens"],
            "n_predict": spec["new_tokens"],
            "temperature": 0,
            "ignore_eos": True,
            "cache_prompt": False,
            "return_tokens": True,
            "seed": spec["seed"],
        }
        for repeat in range(-spec["warmups"], spec["repeats"]):
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
                jobs = [
                    pool.submit(partial(request, config["runtime"]["port"], "completion", body))
                    for _ in range(batch)
                ]
                answers = [job.result() for job in jobs]
            elapsed = time.perf_counter() - start
            if any(len(a["tokens"]) != spec["new_tokens"] or a.get("truncated") for a in answers):
                raise ValueError("speed request produced unexpected length or truncated prompt")
            if repeat >= 0:
                rows.append(
                    {
                        "batch": batch,
                        "repeat": repeat,
                        "wall_seconds": elapsed,
                        "aggregate_tokens_per_second_including_prefill": batch * spec["new_tokens"] / elapsed,
                        "decode_tokens_per_second_per_request": [
                            a["timings"]["predicted_per_second"] for a in answers
                        ],
                        "request": body,
                        "responses": answers,
                    }
                )
    summary = []
    for batch in spec["batches"]:
        subset = [r for r in rows if r["batch"] == batch]
        rates = [r["aggregate_tokens_per_second_including_prefill"] for r in subset]
        decode = [v for r in subset for v in r["decode_tokens_per_second_per_request"]]
        summary.append(
            {
                "batch": batch,
                "aggregate_median": statistics.median(rates),
                "aggregate_range": [min(rates), max(rates)],
                "decode_per_request_median": statistics.median(decode),
            }
        )
    write(out / "speed.json", {"summary": summary, "raw": rows})


def built_at_pin(build_info):
    """Whether the build stamped the pinned commit: git abbreviates it to 7 or more digits, by repository size."""
    found = re.search(r'LLAMA_COMMIT = "([0-9a-f]+)"', Path(build_info).read_text())
    return found is not None and len(found[1]) >= 7 and PINNED.startswith(found[1])


#: The one line an old config's `references:` (the removed llama.cpp KL) gets.
NO_KL = (
    "references: removed; KL of a .gguf is `qda-kl score` with the .gguf as `checkpoint` (it runs `qda decode-gguf` "
    'and reads the decode with vLLM; docs/verify.md, "A GGUF file"): delete `references`'
)


def main():
    """Run only explicitly configured measurements; refuse an existing output directory."""
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    if isinstance(config, dict) and "references" in config:
        sys.exit(f"Error: {sys.argv[1]}: {NO_KL}")
    required = {
        "llama_cpp",
        "build",
        "bridge",
        "model",
        "tokenizer",
        "out",
        "runtime",
        "coordination",
        "gencheck",
        "speed",
    }
    if set(config) != required:
        raise ValueError(f"config keys differ: {set(config) ^ required}")
    if commit_of(config["llama_cpp"]) != PINNED:
        raise ValueError("llama.cpp checkout differs from the QDA pin")
    if not built_at_pin(Path(config["build"]) / "common/build-info.cpp"):
        raise ValueError("llama.cpp build differs from the checkout pin")
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write(
        out / "identity.json",
        {
            "llama_cpp": PINNED,
            "model_sha256": digest(config["model"]),
            "model_file_bytes": Path(config["model"]).stat().st_size,
            "bridge_sha256": digest(config["bridge"]),
            "server_sha256": digest(Path(config["build"]) / "bin/llama-server"),
            "sources": {name: digest(Path(__file__).parent / name) for name in ("run.py", "probe.cpp")},
            "libraries": {
                name: digest(Path(config["build"]) / "bin" / name)
                for name in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cuda.so")
            },
            "utc": datetime.now(UTC).isoformat(),
            "lease": os.environ.get("GPU_RUN_LEASE"),
        },
    )
    proc, log = server(config, out)
    try:
        if config["gencheck"] is not None:
            gencheck(config, out)
        if config["speed"] is not None:
            speed(config, out)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        state(config, "DONE")


if __name__ == "__main__":
    main()
