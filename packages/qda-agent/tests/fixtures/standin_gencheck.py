"""A stand-in for `qda-kl gencheck CONFIG.yaml`: writes a contract-shaped `OUT/gencheck.json` and the texts.

The original model's answers are "generated" once per `original_cache` (a marker file counts how often), and
the result carries what the agent must never see: a prompt, an answer, a host path and a domain name that is
not a plain name. A config with `fail: true` exits 1 and writes nothing.
"""

import json
import sys
from pathlib import Path

SECRET_PROMPT = "PROMPT-SECRET: prove that the sum of two odd numbers is even"
SECRET_ANSWER = "ANSWER-SECRET: let a = 2m + 1 and b = 2n + 1"


def main() -> int:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    if cfg.get("fail"):
        return 1
    out = Path(cfg["out"])
    cache = Path(cfg["original_cache"])
    marker = cache / "original.json"
    runs = json.loads(marker.read_text())["runs"] if marker.exists() else 0
    if not marker.exists():
        marker.write_text(json.dumps({"runs": runs + 1}))
    n = cfg["per_domain"]
    mine = {
        "math": {"n": n, "cut": 3, "loops": 1, "mean_tokens": 5123.4, "prompt": SECRET_PROMPT},
        "code": {"n": n, "cut": 0, "loops": 0, "mean_tokens": 812.0},
        "../etc/passwd": {"n": n, "cut": 9, "loops": 9, "mean_tokens": 1.0},
        "a" * 40: {"n": n, "cut": 9, "loops": 9, "mean_tokens": 1.0},
    }
    orig = {
        "math": {"n": n, "cut": 1, "loops": 0, "mean_tokens": 4870.1},
        "code": {"n": n, "cut": 0, "loops": 0, "mean_tokens": 790.0},
    }
    doc = {
        "max_tokens": cfg["max_tokens"],
        "per_domain": n,
        "checkpoint": mine,
        "original": orig,
        "delta": {"math": {"cut": 2, "loops": 1, "mean_tokens": 253.3}},
        "total": {
            "checkpoint": {"n": 2 * n, "cut": 3, "loops": 1},
            "original": {"n": 2 * n, "cut": 1, "loops": 0},
        },
        "texts": str(out / "texts"),
        "sample": SECRET_ANSWER,
    }
    (out / "texts").mkdir(parents=True, exist_ok=True)
    for side in ("checkpoint", "original"):
        row = {"id": "m1", "domain": "math", "prompt": SECRET_PROMPT, "answer": SECRET_ANSWER}
        (out / "texts" / f"{side}.jsonl").write_text(json.dumps(row) + "\n")
    (out / "gencheck.json").write_text(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
