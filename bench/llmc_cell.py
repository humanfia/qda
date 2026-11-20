"""One llm-compressor 0.15 cell on QDA's calibration rows and derived bf16-keep list.

bench_cell.py calls run(); imports happen at module load, so the wall clock excludes them.
"""

import hashlib
import json
import time
from importlib import metadata

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from qda_quant.calib.sources import FileSource
from qda_quant.engine.calib import load_samples
from qda_quant.engine.planning import local_source
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def _rows(model, tokenizer, calib, samples, seqlen, seed):
    source = FileSource(path=str(calib), samples=samples, seqlen=seqlen, seed=seed)
    return [s.ids.tolist() for s in load_samples(source, local_source(model, weights=False), tokenizer)]


def _recipe(method, ignore):
    """llm-compressor's counterpart of QDA's steps (W4A16 g128; the rest are llm-compressor defaults)."""
    if method == "gptq":
        # actorder static, damp 0.01, block 128 are GPTQModifier's defaults; they are the recipe's settings
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    if method == "rtn":
        return QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    if method == "awq-rtn":
        return [
            AWQModifier(duo_scaling=True, n_grid=20),
            QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=ignore),
        ]
    raise ValueError(f"unknown method {method!r}")


def run(model_id, calib, out, method, cards, ignore, samples=128, seqlen=2048, seed=0):
    model_dir = local_source(model_id, weights=False)  # the reader checks this snapshot's fingerprint
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # oneshot wants a processor whenever a dataset is given. The text-only repos have none (`Qwen3-8B` is a
    # plain tokenizer); the conditional-generation repos need torchvision for their image processor, which the
    # shared venv does not have. The rows are already token ids, so llm-compressor never uses it; a tokenizer
    # stands in and is recorded as such.
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        processor_kind = type(processor).__name__
    except Exception as error:  # noqa: BLE001 - any processor-init failure takes the same fallback
        processor = tokenizer
        processor_kind = f"{type(tokenizer).__name__} (processor unavailable: {type(error).__name__})"
    placement = "cuda" if cards == 1 else "auto"  # llm-compressor uses `cards` as device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=placement)
    rows = _rows(model_dir, tokenizer, calib, samples, seqlen, seed)
    data = Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]})
    oneshot(
        model=model,
        dataset=data,
        processor=processor,
        recipe=_recipe(method, ignore),
        max_seq_length=seqlen,
        num_calibration_samples=len(rows),
        shuffle_calibration_samples=False,
        batch_size=1,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)
    linears = {name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    quantized = sorted(n for n in linears if getattr(model.get_submodule(n), "quantization_scheme", None))
    return {
        "load_to_saved_s": round(time.perf_counter() - t0, 3),
        "cards_used": len({p.device for p in model.parameters() if p.device.type == "cuda"}),
        "device_map": placement,
        "model_class": type(model).__name__,
        "processor": processor_kind,
        "prepared": sorted(linears),
        "quantized": quantized,
        "calib_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
        "versions": {
            "torch": torch.__version__,
            **{p: metadata.version(p) for p in ("transformers", "llmcompressor", "compressed-tensors")},
        },
    }
