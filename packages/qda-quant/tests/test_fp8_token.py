"""The fused FP8 activation round trip is the reference's bits.

`fp8_token` simulates vLLM's CUTLASS W4A8 activation quant-dequant on the calibration forwards. The reference
is a chain of elementwise passes over fp32 intermediates; the fused Triton kernel reads each row twice (its
amax, then its quant-dequant) and writes it once. On CUDA with bf16 or fp16 inputs the two must agree bit for
bit, since a build whose activations differ in the last bit writes different GPTQ codes. Without CUDA, ``auto``
must stay on the reference.
"""

from __future__ import annotations

import pytest
import torch
from qda_quant.solve.quant import FP8_MIN_SCALE, fp8_token, fp8_token_reference, triton_fp8


@pytest.fixture
def fp8_card() -> None:
    """Skips below compute capability 8.9, where Triton has no E4M3 cast and ``auto`` is the reference.

    Read in the test, not at collection: asking for the capability initializes CUDA in the pytest process.
    """
    if not triton_fp8(torch.device("cuda")):
        pytest.skip("Triton's E4M3 cast needs compute capability 8.9 or newer")


#: odd widths exercise the kernel's column mask; the large ones are the shapes the issue measured
SHAPES = [(1, 1), (2, 3, 5), (4, 1000), (2, 4096), (2, 12288)]
#: a zero-size last dimension has no channel to scale; ``(0, 5)`` has no token either
EMPTY_SHAPES = [(0,), (2, 0), (2, 3, 0), (0, 5)]


def _values(shape: tuple[int, ...], dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    """Random rows plus the edges: an all-zero row (the scale floor), a row below it, and values past ±448."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device) * 7.5
    if shape[0] > 1:
        x[1] = 0
    if shape[0] > 2:
        x[2] = x[2] * (FP8_MIN_SCALE * 1e-3)  # amax under the kernel's FP8 scale floor
    if shape[0] > 3:
        x[3, 0] = 1000
        x[3, 1] = -1000  # clamps to ±448 before the fp8 cast
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_auto_is_the_reference_without_cuda(dtype):
    """``auto`` is the elementwise reference on the CPU and for an fp32 input, not an approximate path."""
    for shape in SHAPES:
        x = _values(shape, dtype)
        assert torch.equal(fp8_token(x), fp8_token_reference(x))


def test_an_input_the_kernel_does_not_take_is_refused():
    x = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="CUDA bf16 or fp16"):
        fp8_token(x, impl="triton")
    with pytest.raises(ValueError, match="is not 'auto'"):
        fp8_token(x, impl="numpy")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_an_empty_last_dimension_is_empty_on_every_cpu_path(shape, dtype):
    """No channel to scale: the reference, ``torch`` and ``auto`` return the input's shape and dtype."""
    x = torch.empty(shape, dtype=dtype)
    for out in (fp8_token_reference(x), fp8_token(x, impl="torch"), fp8_token(x)):
        assert out.shape == x.shape
        assert out.dtype == dtype


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_an_empty_last_dimension_is_empty_on_every_cuda_path(shape, dtype):
    """As on the CPU, and the fused kernel launches no program: empty shape and dtype on every path."""
    x = torch.empty(shape, dtype=dtype, device="cuda")
    fused = [fp8_token(x, impl="triton")] if triton_fp8(x.device) else []
    for out in (fp8_token_reference(x), fp8_token(x, impl="torch"), fp8_token(x), *fused):
        assert out.shape == x.shape
        assert out.dtype == dtype


@pytest.mark.gpu
@pytest.mark.usefixtures("fp8_card")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_the_fused_kernel_is_bit_identical(shape, dtype):
    x = _values(shape, dtype, "cuda")
    assert torch.equal(fp8_token(x, impl="triton"), fp8_token_reference(x))
    assert torch.equal(fp8_token(x), fp8_token_reference(x))  # auto takes the fused path on CUDA


@pytest.mark.gpu
@pytest.mark.usefixtures("fp8_card")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scalar_scale_rounding_matches_the_reference(dtype):
    """Torch divides by scalar 448 via a rounded reciprocal, which differs from IEEE division here."""
    x = torch.tensor([[27.125, -10.6875]], dtype=dtype, device="cuda")
    reference = fp8_token_reference(x)
    assert torch.equal(fp8_token(x, impl="triton"), reference)
    assert torch.equal(fp8_token(x), reference)


@pytest.mark.gpu
@pytest.mark.usefixtures("fp8_card")
def test_the_probe_shape_and_rows_are_bit_identical():
    """The 4,096-wide real shape, with a zero row, a row under the scale floor and the ±1000 clamps."""
    torch.manual_seed(123)
    x = torch.randn((2048, 4096), dtype=torch.bfloat16, device="cuda") * 7.5
    x[1] = 0
    x[2] *= FP8_MIN_SCALE * 1e-3
    x[3, :2] = torch.tensor([1000.0, -1000.0], device="cuda", dtype=torch.bfloat16)
    assert torch.equal(fp8_token(x), fp8_token_reference(x))


@pytest.mark.gpu
@pytest.mark.usefixtures("fp8_card")
def test_rows_past_2_to_the_31_elements_are_bit_identical():
    """Row 2^19 starts at element 2^31; rows are independent of each other."""
    torch.manual_seed(0)
    x = torch.randn(((1 << 19) + 4, 4096), dtype=torch.bfloat16, device="cuda") * 7.5
    assert torch.equal(fp8_token(x, impl="triton")[-4:], fp8_token_reference(x[-4:]))


@pytest.mark.gpu
@pytest.mark.usefixtures("fp8_card")
def test_a_non_contiguous_input_is_flattened_first():
    """The reference takes any strides; the kernel gets a contiguous ``[rows, C]`` view of the same values."""
    torch.manual_seed(0)
    x = (torch.randn(4096, 8, device="cuda", dtype=torch.bfloat16) * 3).t()
    assert not x.is_contiguous()
    assert torch.equal(fp8_token(x), fp8_token_reference(x))


@pytest.mark.gpu
def test_auto_is_the_reference_below_compute_capability_8_9():
    """An A100 cannot compile Triton's E4M3 cast: ``auto`` is the reference there and ``triton`` is refused."""
    if triton_fp8(torch.device("cuda")):
        pytest.skip("needs a card below compute capability 8.9 (A100)")
    x = _values((4, 1000), torch.bfloat16, "cuda")
    assert torch.equal(fp8_token(x), fp8_token_reference(x))
    with pytest.raises(ValueError, match="compute capability 8.9"):
        fp8_token(x, impl="triton")
