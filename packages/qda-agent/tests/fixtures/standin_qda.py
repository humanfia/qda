"""A stand-in `qda` for CPU tests: `qda build CONFIG.yaml` writes a tiny valid checkpoint.

The checkpoint's config.json carries `standin_kl`, a fixed function of the build's steps, which the stand-in
KL command (`standin_kl.py`) reports as the reading. Any other subcommand prints its argv.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import yaml

if sys.argv[1:2] != ["build"]:
    print("stand-in qda:", *sys.argv[1:])
    sys.exit(0)
build = yaml.safe_load(Path(sys.argv[2]).read_text())
digest = hashlib.sha256(json.dumps(build["steps"], sort_keys=True).encode()).digest()
out = Path(build["out"])
out.mkdir(parents=True, exist_ok=True)
kl = 0.01 + int.from_bytes(digest[:4]) % 1000 / 10000
(out / "config.json").write_text(json.dumps({"model_type": "qwen3", "standin_kl": kl}))
(out / "tokenizer.json").write_text("{}")
h = json.dumps({"w": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode()
h += b" " * (-len(h) % 8)
(out / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + digest[:8])
print(out)
