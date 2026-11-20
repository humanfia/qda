"""Dense int32 packing of signed low-bit codes, the layout of compressed-tensors ``pack-quantized``.

Each row (``packed_dim=1``) or column (``packed_dim=0``) is packed on its own: codes are shifted to unsigned
by ``1 << (bits - 1)`` and laid LSB first into a little-endian stream of 32-bit words, element ``i`` at bits
``i * bits`` onward, crossing a word boundary when needed, with no padding between elements. A row of
``n`` codes takes ``ceil(n * bits / 32)`` words, stored as two's complement int32. Tensors with more than
two dimensions (fused experts) are packed per slice of dim 0.

The work is vectorized. At 1, 2, 4 and 8 bits a word holds whole codes, so a word is one shifted sum. Other
widths use that 32 codes always fill exactly ``bits`` words: rows are padded to whole groups of 32 codes and one
loop over the 32 positions of a group packs every group at once.
"""

from __future__ import annotations

import torch

__all__ = ["pack", "unpack"]

_GROUP = 32  # codes per group; a group fills exactly `bits` words
_MASK32 = (1 << 32) - 1


def _check_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError(f"bits must be an int in 1..8, got {bits!r}")


def _check_dim(packed_dim: int) -> None:
    if packed_dim not in (0, 1):
        raise ValueError(f"packed_dim must be 0 or 1, got {packed_dim!r}")


def _positions(bits: int) -> list[tuple[int, int, int, int]]:
    """For each code of a group: (code index, word index, bit offset, bits spilled into the next word)."""
    out = []
    for j in range(_GROUP):
        word, off = divmod(j * bits, 32)
        out.append((j, word, off, max(0, off + bits - 32)))
    return out


def _pack_rows(u: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack each row of unsigned codes `u` ``[rows, cols]`` (int64) into int32 words ``[rows, words]``."""
    rows, cols = u.shape
    if 32 % bits == 0:  # 1, 2, 4, 8 bits: whole codes per word, one shifted sum
        k = 32 // bits
        words = -(-cols // k)
        u = torch.nn.functional.pad(u, (0, words * k - cols)).reshape(rows, words, k)
        w = (u << (torch.arange(k, device=u.device) * bits)).sum(-1)
        return torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32)
    groups = -(-cols // _GROUP)
    u = torch.nn.functional.pad(u, (0, groups * _GROUP - cols)).view(rows, groups, _GROUP)
    w = torch.zeros(rows, groups, bits, dtype=torch.int64, device=u.device)
    for j, word, off, spill in _positions(bits):
        c = u[:, :, j]
        w[:, :, word] |= (c << off) & _MASK32
        if spill:
            w[:, :, word + 1] |= c >> (bits - spill)
    w = w.view(rows, groups * bits)[:, : -(-cols * bits // 32)]
    return torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32)


def _unpack_rows(p: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Inverse of `_pack_rows`: unsigned codes ``[rows, cols]`` (int64) from int32 words ``[rows, words]``."""
    rows, words = p.shape
    w = p.to(torch.int64) & _MASK32
    lo = (1 << bits) - 1
    if 32 % bits == 0:
        k = 32 // bits
        u = (w.unsqueeze(-1) >> (torch.arange(k, device=p.device) * bits)) & lo
        return u.reshape(rows, words * k)[:, :cols]
    groups = -(-cols // _GROUP)
    w = torch.nn.functional.pad(w, (0, groups * bits - words)).view(rows, groups, bits)
    u = torch.empty(rows, groups, _GROUP, dtype=torch.int64, device=p.device)
    for j, word, off, spill in _positions(bits):
        c = w[:, :, word] >> off
        if spill:
            c = c | (w[:, :, word + 1] << (bits - spill))
        u[:, :, j] = c & lo
    return u.view(rows, groups * _GROUP)[:, :cols]


def pack(value: torch.Tensor, bits: int, packed_dim: int = 1) -> torch.Tensor:
    """Pack signed int8 codes into int32 words along `packed_dim`.

    Args:
        value: int8 codes in ``-2^(bits-1) .. 2^(bits-1) - 1``; 2D, or more dimensions packed per slice of dim 0.
        bits: code width, 1..8.
        packed_dim: 1 packs each row, 0 packs each column.

    Returns:
        Contiguous int32 tensor on `value`'s device; the packed dimension has ``ceil(n * bits / 32)`` words.

    Raises:
        ValueError: `value` is not int8, or `bits` / `packed_dim` is out of range.
    """
    if value.dtype != torch.int8:
        raise ValueError(f"pack needs an int8 tensor, got {value.dtype}")
    _check_bits(bits)
    _check_dim(packed_dim)
    if value.dim() > 2:
        return torch.stack([pack(v, bits, packed_dim) for v in value]).contiguous()
    if value.dim() != 2:
        raise ValueError(f"pack needs at least 2 dimensions, got shape {tuple(value.shape)}")
    u = value.to(torch.int64) + (1 << (bits - 1))
    if packed_dim == 0:
        return _pack_rows(u.t(), bits).t().contiguous()
    return _pack_rows(u, bits).contiguous()


def unpack(
    value: torch.Tensor, bits: int, shape: torch.Size | tuple[int, ...], packed_dim: int = 1
) -> torch.Tensor:
    """Unpack int32 words into signed int8 codes: the exact inverse of `pack`.

    Args:
        value: int32 words as `pack` wrote them.
        bits: code width, 1..8.
        shape: the codes' shape before packing.
        packed_dim: the dimension `pack` packed along.

    Returns:
        int8 tensor of `shape` on `value`'s device.

    Raises:
        ValueError: `value` is not int32, `bits` / `packed_dim` is out of range, or `shape` does not fit.
    """
    if value.dtype != torch.int32:
        raise ValueError(f"unpack needs an int32 tensor, got {value.dtype}")
    _check_bits(bits)
    _check_dim(packed_dim)
    shape = tuple(shape)
    if len(shape) != value.dim():
        raise ValueError(f"shape {shape} does not match packed tensor of shape {tuple(value.shape)}")
    if value.dim() > 2:
        return torch.stack([unpack(v, bits, shape[1:], packed_dim) for v in value])
    if value.dim() != 2:
        raise ValueError(f"unpack needs at least 2 dimensions, got shape {tuple(value.shape)}")
    p = value.t() if packed_dim == 0 else value
    rows, cols = (shape[1], shape[0]) if packed_dim == 0 else shape
    if p.shape != (rows, -(-cols * bits // 32)):
        raise ValueError(
            f"packed shape {tuple(value.shape)} does not fit codes of shape {shape} at {bits} bits"
        )
    q = (_unpack_rows(p, bits, cols) - (1 << (bits - 1))).to(torch.int8)
    return q.t().contiguous() if packed_dim == 0 else q
