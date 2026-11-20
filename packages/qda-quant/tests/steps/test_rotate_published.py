"""The residual rotation `orthogonal` builds is the one the published checkpoints carry.

A rotation is exact for any orthogonal matrix, but GPTQ on the rotated weights is not invariant to which one, so a
rebuild of a published checkpoint needs its exact matrix. For seed 35 these were read back from the shipped 9B (4096)
and 27B (5120) W4A16 embeddings to the bf16 floor: diag(s) Sylvester and
diag(s) (Paley-I(19) (x) Sylvester(256)), with ``s`` the seed's sign draw. The sign-pattern hashes pin that result.
"""

import hashlib

import pytest
import torch
from qda_quant.steps.rotate import hadamard, orthogonal

PUBLISHED = {
    4096: "c74925bd12ac8ebaaf87e74267d4d797cbb93a318c1bbf6290c29d546c8e7a29",
    5120: "c6d6296f62ab70e2d98ea72ce45ac275cd7eb1ece0056e70b65ae9a74a41499a",
}


@pytest.mark.parametrize("d", sorted(PUBLISHED))
def test_seed_35_is_the_published_rotation(d):
    q = orthogonal(d, 35)
    assert torch.allclose(q.T @ q, torch.eye(d, dtype=q.dtype), atol=1e-12)
    assert torch.allclose(q.abs(), torch.full_like(q, d**-0.5))  # a Hadamard matrix, every entry +-1/sqrt(d)
    assert hashlib.sha256((q > 0).numpy().tobytes()).hexdigest() == PUBLISHED[d]


def test_signs_are_per_row_and_unverified_widths_keep_the_random_matrix():
    h = hadamard(64)
    assert h is not None
    q = orthogonal(64, 3) * 64**0.5
    assert torch.equal(q, torch.sign(q[:, :1]) * h)  # diag(s) H: row i is +-H's row i
    assert hadamard(2688) is None and hadamard(3072) is None  # no published Paley order verifies these
    q = orthogonal(2688, 0)
    assert torch.allclose(q.T @ q, torch.eye(2688, dtype=q.dtype), atol=1e-10)
