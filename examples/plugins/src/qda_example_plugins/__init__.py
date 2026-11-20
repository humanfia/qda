"""External plugin examples for QDA."""


def register() -> None:
    """Register examples through the same hook an external package uses."""
    from qda_agent.agents import BACKENDS
    from qda_agent.components import COMPONENTS
    from qda_agent.judge.final_rules import FINAL_RULES
    from qda_kl.datasets import DATASETS
    from qda_kl.eval.suites import SUITES
    from qda_kl.scorers import SCORERS
    from qda_quant.arch.adapters import ADAPTERS
    from qda_quant.solve.grid import GRIDS
    from qda_quant.step import REGISTRY

    from .architecture import TINY_PROJECTION
    from .backends import GreedyBackend
    from .components import ReadOnlyNote
    from .datasets import VersionedPrompts
    from .final_rules import FirstValid
    from .grids import SIGNED8_BF16
    from .scorers import TensorKLScorer
    from .steps import ClipRTN
    from .suites import CompactnessSuite

    REGISTRY.register("clip_rtn", ClipRTN)
    GRIDS.register("signed8_bf16", SIGNED8_BF16)
    ADAPTERS.register("tiny_projection", TINY_PROJECTION)
    SCORERS.register("tensor_kl", TensorKLScorer)
    SUITES.register("compactness", CompactnessSuite)
    DATASETS.register("versioned_prompts", VersionedPrompts)
    FINAL_RULES.register("first_valid", FirstValid)
    BACKENDS.register("greedy_demo", GreedyBackend)
    COMPONENTS.register("read_only_note", ReadOnlyNote)
