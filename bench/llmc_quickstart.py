"""Build the quickstart checkpoint with llm-compressor: GPTQ, W4A16 symmetric g128, 128 x 2,048 tokens of C4.

python bench/llmc_quickstart.py Qwen/Qwen3-0.6B qwen3-0.6b-llmc
"""

import sys
import time

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from qda_quant.config import BuildConfig
from qda_quant.engine.calib import load_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, out = sys.argv[1], sys.argv[2]
start = time.time()
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

# QDA's own loader on the quickstart's config, so both libraries calibrate on the same token ids
calib = BuildConfig.from_file("configs/compare-gptq.yaml").recipe.calib
samples = [ids.tolist() for ids in load_ids(calib, model_id, tok)]
data = Dataset.from_dict({"input_ids": samples, "attention_mask": [[1] * len(ids) for ids in samples]})

oneshot(
    model=model,
    dataset=data,
    recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    max_seq_length=2048,
    num_calibration_samples=len(samples),
)
model.save_pretrained(out, save_compressed=True)
tok.save_pretrained(out)
print(f"built {out} in {time.time() - start:.2f} s")
