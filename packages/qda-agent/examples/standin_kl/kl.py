"""A stand-in KL command: write a fixed reading (`python kl.py CKPT OUT`)."""

import json
import sys
from pathlib import Path

reading = {
    "kl": 0.07,
    "domains": {"math": 0.06, "code": 0.08},
    "kl_text": 0.069,
    "domains_text": {},
    "tokens": 2,
    "per_seq": {
        "p#0": {"kl": 0.06, "tokens": 1, "domain": "math"},
        "p#1": {"kl": 0.08, "tokens": 1, "domain": "code"},
    },
}
Path(sys.argv[2]).write_text(json.dumps(reading))
