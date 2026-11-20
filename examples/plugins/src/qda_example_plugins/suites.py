"""An eval suite showing the ``qda.eval_suites`` interface, and a metric the search can rank by, without vLLM."""

from pathlib import Path
from typing import ClassVar

from qda_kl.eval.suites import Suite, SuiteResult


def _bytes(directory: str) -> int:
    """The size of every file under ``directory``."""
    return sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())


class CompactnessSuite(Suite):
    """How many times smaller a checkpoint is than the reference: higher is better."""

    name: ClassVar[str] = "compactness"

    def run(self, checkpoint: str, reference: str, work: Path) -> SuiteResult:
        """Compare the two directories' sizes."""
        del work  # no reference to build
        ratio = _bytes(reference) / _bytes(checkpoint)
        return SuiteResult(
            suite=self.name, metrics={"compactness": ratio}, higher_is_better={"compactness": True}
        )
