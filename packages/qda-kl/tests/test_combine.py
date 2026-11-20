"""Engine repeat conventions remain identical for stored floating-point readings."""

import json
from pathlib import Path

import numpy as np
import pytest
from qda_kl import score, tally

CASES = [
    ([0.0, 0.0], [None, None], [1.0, 1.0]),
    ([0.125, 0.5, 0.25], [0.25, 0.5, 0.125], [0.5, 1.0, 1.5]),
    ([-0.25, 0.25], [0.0, 0.0], [1.0, 3.0]),
    ([0.5, 0.5], [None, 0.25], [1.0, 3.0]),
    ([0.5, 0.5], [0.25, 0.25], [1.0, 3.0]),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_combine_preserves_stored_readings(case, dtype):
    native = False
    readings = [
        score.Score.model_construct(
            kl=dtype(k),
            kl_text=None if t is None else dtype(t),
            domains={"hard": dtype(d)},
            domains_text={"easy": dtype(d)},
            seconds={"decode": 3.0},
        )
        for k, t, d in zip(*CASES[case], strict=True)
    ]
    expected = json.loads(Path(__file__).with_suffix(".json").read_text())
    got = tally.combine(readings, native=native)
    assert got.model_dump_json(exclude_unset=True) == expected[f"{native}-{case}-{dtype.__name__}"]
