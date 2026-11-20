# Add a target

For a researcher or maintainer who builds for a GPU that QDA has no table for, or who adds a kernel row: add a table
that `qda build` plans with and the support matrix shows.

A target is a `Table` registered in `TARGETS` under its architecture name (`sm89`). A build calls
`Target.plan(fmt, kind)`; the table returns the `KernelPlan` vLLM serves that layer kind with (kernel, scale and
activation dtypes, zero point, serve settings) or raises `Refused` with the reason. Quantizers match the plan's
numerics, so a wrong row is a wrong checkpoint.

## A card whose kernels match an existing table

Give the card its own `Table` over the existing rows function, with its own `aliases` and `capabilities`. A Jetson
Orin (compute capability 8.7) serves the A100 rows:

```python
>>> from qda_quant.format import Format
>>> from qda_quant.target import TARGETS, Table, Target
>>> orin = Table(TARGETS["sm80"].rows, aliases=("jetson_orin",), capabilities=((8, 7),))
>>> plan = orin(Target("orin"), Format.parse("w4a16"), "dense")
>>> plan.kernel, plan.notes[-1]
('MarlinLinearKernel', "unverified on real hardware: no card has run this row, it comes from vLLM's source")

```

From your own package, register the table and the A100 binding rules in its
[registration hook](extend.md#the-registration-hook):

```python
def register():
    from qda_quant.target import BINDINGS, TARGETS

    TARGETS.register("orin", "my_pkg.target:orin")
    BINDINGS.register("orin", "qda_quant.target:sm80_binds")
    BINDINGS.register("orin_moe", "qda_quant.target:sm80_moe_binds")
```

Check it with `qda plan CONFIG --set target=jetson_orin`: it prints each layer kind's kernel and, until the row is
[observed](#observed-rows), the `unverified on real hardware` note. A format the rows refuse prints the refusal,
e.g. `w4a8-sym-g128 dense on orin vLLM 0.33.0: FP8 activations need Hopper: ...`.

## A card with its own kernels

Write a rows function `rows(target, fmt, kind)`: it takes a `Target` (`arch`, `vllm`, `backend`), a `Format` and
`"dense"` or `"moe"`, and returns a `KernelPlan`. `sm80` (`_sm80_rows` in `qda_quant/target.py`) is the small one
to copy. L40S (`sm89`) uses `_ada_rows` (abridged); B200's `_sm100_rows` adds its NVFP4 and Blackwell rows in
front of it, and RTX PRO 6000's `_sm120_rows` its NVFP4 rows:

```python
def _ada_refusal(target: Target, fmt: Format, kind: LayerKind) -> str | None:
    """Why no row serves `fmt` on Ada or on an RTX PRO 6000, or None."""
    if target.vllm != "0.33.0":
        return f"the {target.arch} rows are read from vLLM 0.33.0; which kernel binds on another version is not known"
    if fmt.act == "fp8_token":
        return "FP8 activations are not served here: vLLM's CUTLASS W4A8 kernel needs compute capability 90 ...; use w4a16"
    ...
    return None


def _ada_rows(target: Target, fmt: Format, kind: LayerKind) -> KernelPlan:
    if why := _ada_refusal(target, fmt, kind):
        raise Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {why}")
    if fmt.weight_bits == 3:
        return KernelPlan(
            "HummingLinearKernel" if kind == "dense" else "HummingIndexedExperts",
            zero_point=not fmt.symmetric,
            notes=("Marlin has no int3; Humming serves W3 here", ...),
        )
    kernel = "MarlinLinearKernel" if kind == "dense" else "MarlinExperts"
    args = {"linear_backend": "marlin"} if kind == "dense" else {"moe_backend": "marlin"}
    return KernelPlan(kernel, zero_point=not fmt.symmetric, serve_args=args, notes=(...,))


sm89 = Table(_ada_rows, aliases=("l40s", "l4", "4090", "ada"), observed=_ADA_OBSERVED, capabilities=((8, 9),))
```

- Keep the refusal in its own function that returns a sentence or `None`; the rows function raises
  `Refused(f"{fmt.name} {kind} on {target.arch} vLLM {target.vllm}: {why}")`.
- The text after that prefix is what a build prints. Say what the
  card cannot do and what to build instead (`use w4a16`).
- Refuse a vLLM version whose kernel choice you have not read (`target.vllm != "0.33.0"`).
- Set `serve_args` / `serve_env` when the kernel binds only with them; they are written into the checkpoint's
  `build.json` `serve` section.

## Table fields

| field | what it says |
|---|---|
| `rows` | the function above |
| `aliases` | GPU names `--set target=` accepts (`l40s` -> `sm89`); two tables cannot claim one name |
| `capabilities` | `(major, minor)` of the cards the rows plan for. A build that names no target takes the one table covering its cards (`Target.on_cards`); `capabilities_serving` and `serving_refusal` read it too |
| `backends` | backend pins it accepts (`h200:marlin`); default `(None,)` |
| `versions` | vLLM versions whose rows differ, beside `Target.vllm` |
| `observed` | `(vllm, kernel, kind, symmetric)` rows seen bound on a real card; a fifth item, the weight bits, limits a row to that width |
| `verified_vllm` | vLLM versions all of whose rows count as observed |
| `unverified` | the note on every other row; default `UNVERIFIED_ON_HARDWARE` |

## Observed rows

A row carries the `unverified` note, which `qda plan` prints, unless `target.vllm` is in `verified_vllm` or
`(target.vllm, plan.kernel, kind, fmt.symmetric)` is in `observed`, e.g.
`("0.33.0", "MarlinLinearKernel", "dense", False)` for an asymmetric W4A16 dense row.

Add a row only after a card has served it: vLLM's log names the kernel `build.json` planned, and the prompt logprobs
match a dequantized reference (`tests/test_vllm_gpu.py`). Name the card in a comment above the set.

## Binding rules

`BINDINGS` holds, per architecture, the rule for which kernel vLLM binds for one module of a given shape under the
checkpoint's serve settings: `<arch>` for dense modules, `<arch>_moe` for experts (shape
`(hidden, unpadded intermediate)`). A rule is called as `rule(target, fmt, shape, args, env)` and returns the kernel
class name or raises `Refused`. Reuse the rule of the table whose rows you reuse (`sm80_binds` above; `sm89_binds` for the Ada rows): it
refuses a shape the row's kernel cannot take (a Marlin group straddle).

Without a rule, a build with `format_overrides` (or `formats:`) refuses with
`no binding rule says which kernel serves ... beside the body`.

## Adding a table to QDA

For a table added to this repository rather than to your own package:

- Put it in both registries in `target.py`:

  ```python
  TARGETS = Registry("targets", {"sm90": ..., "sm89": "qda_quant.target:sm89"})
  BINDINGS = Registry(
      "bindings",
      {"sm90": ..., "sm89": "qda_quant.target:sm89_binds", "sm89_moe": "qda_quant.target:sm89_moe_binds"},
  )
  ```

- Remove the card's name from `target._PENDING` (names a build refuses with a pointer to the tracking issue).
- Add the card's label and one name its table claims to `CARDS` in `tools/gen_docs.py`, then regenerate the support
  matrix in `docs/formats.md` (never edit the block by hand):

  ```bash
  python tools/gen_docs.py            # rewrite the generated blocks
  python tools/gen_docs.py --check    # what CI runs: exit 1 if a block differs from the code
  ```

  A cell is ✅ for an observed row and ❌ (add the format to `NEEDS` in the generator to name what it requires); dense and MoE that differ get both
  (`✅ dense · ❌ moe`). A planned row that still carries the `unverified` note stops the generator: observe the row
  or make it refuse first.
- Tests in `packages/qda-quant/tests/test_target_quant.py`: copy `test_ada_and_rtx_pro_6000_plan_their_rows` (each
  alias parses to the architecture; each format's kernel, zero point, serve args and unverified note; each refusal)
  and `test_ada_bindings_are_marlin_and_refuse_a_group_straddle`; add the capability to
  `test_a_target_on_cards_resolves_when_one_table_covers_them` and update
  `test_an_unset_target_refuses_when_no_table_serves_the_cards`, which quotes the supported capabilities.
- Extend `packages/qda-quant/tests/test_target_serving.py` when the table changes which capabilities serve a format
  (`capabilities_serving`, `serving_refusal`), e.g. a new card that plans W4A8.
