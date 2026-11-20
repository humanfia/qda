"""``qda eval``: score a list of checkpoints against one reference, one JSON per (checkpoint, suite part)."""

#: The install line for the ``eval`` extra (lm-evaluation-harness and the IFBench checkers), as README spells it.
INSTALL_EVAL = 'uv pip install -c constraints.txt "./packages/qda-kl[eval]"'
