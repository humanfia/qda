"""A stand-in KL command (`python standin_kl.py CKPT OUT`): the reading is the checkpoint's `standin_kl`."""

import json
import sys
from pathlib import Path

kl = json.loads((Path(sys.argv[1]) / "config.json").read_text())["standin_kl"]
reading = {
    "kl": kl,
    "domains": {},
    "kl_text": kl,
    "domains_text": {},
    "tokens": 1,
    "per_seq": {"p#0": {"kl": kl, "tokens": 1}},  # a complete reading: every sequence it scores is reported
}
Path(sys.argv[2]).write_text(json.dumps(reading))
