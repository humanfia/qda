"""Bench a real build from a build config (`qda_quant.config.BuildConfig`); KL-ready checkpoint.

python bench/engine_build.py CONFIG.yaml [--set out=DIR] [--set devices=[0,1]]
"""

import argparse
import json
import logging
import sys
import time

T0 = time.perf_counter()
import torch  # noqa: E402, F401  (timed from T0, as before)
from qda_quant.config import BuildConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = BuildConfig.from_file(a.config, a.set)
    gsq_log = logging.getLogger("qda_quant.steps.gsq")
    gsq_log.setLevel(logging.INFO)
    gsq_log.addHandler(logging.StreamHandler(sys.stdout))
    out = cfg.build(
        log=lambda s: print(s, flush=True) if "layer " not in s else None,
    )
    t_built = time.perf_counter() - T0
    info = json.loads((out / "build.json").read_text())
    print("TIMINGS", json.dumps(info["timings"]))
    print("LOAD_TO_SAVED", info["load_to_saved_s"], "PROCESS_TO_BUILT", round(t_built, 1))


if __name__ == "__main__":  # multi-card builds spawn processes that re-import this file
    main()
