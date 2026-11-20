"""A tensor-backed KL scorer showing the ``qda_kl.scorers`` interface without vLLM."""

from pathlib import Path
from typing import ClassVar

import torch
from pydantic import BaseModel
from qda_kl.scorers import KLScorer, Outcome


class TensorReading(BaseModel):
    """The mean token KL and tensor size returned by this example scorer."""

    kl: float
    tokens: int


class TensorKLScorer(KLScorer):
    """Compare saved reference and candidate logits using mean KL in nats."""

    name: ClassVar[str] = "tensor_kl"

    def run(self, checkpoint: str, reference: Path) -> Outcome:
        """Read ``logits.pt`` from both directories and compare their next-token distributions."""
        candidate = torch.load(Path(checkpoint) / "logits.pt", map_location="cpu", weights_only=True)
        original = torch.load(reference / "logits.pt", map_location="cpu", weights_only=True)
        if candidate.shape != original.shape or candidate.ndim < 2:
            raise ValueError("candidate and reference logits must have the same shape and a vocabulary axis")
        log_original = original.float().log_softmax(dim=-1)
        log_candidate = candidate.float().log_softmax(dim=-1)
        probabilities = log_original.exp()
        values = (probabilities * (log_original - log_candidate)).sum(dim=-1)
        reading = TensorReading(kl=float(values.mean()), tokens=values.numel())
        return Outcome([reading], reading)
