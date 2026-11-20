"""Group-wise block-scaled quantize/dequantize for the sub-8-bit extension.

Every group of `group_size` consecutive elements along the reduction axis
shares one fp32 scale, computed from that group's own max magnitude
(symmetric, zero-point-free quantization, the standard scheme for weight
and activation quantization at this precision). This is deliberately the
same scheme for FP8 and INT4 so the two arms differ only in format, not in
quantization methodology:

- FP8 (E4M3/E5M2): scale = group_amax / format_max, cast to the format's
  real torch dtype (torch.float8_e4m3fn / torch.float8_e5m2), which this
  machine's GPU (Ada, sm_89) stores natively even though its tensor cores
  do not execute fp8 matmul in hardware (that is Hopper-and-later); see the
  README for what "FP8 Triton kernel" means on this GPU as a result.
- INT4: scale = group_amax / 7, rounded to the nearest integer in [-8, 7]
  and stored in a plain int8 (one value per byte, not bit-packed two per
  byte). Bit-packing is a storage-density optimization; it is not exercised
  here because nothing this project measures depends on it, and packing
  would add a real bug surface (shift/mask logic) with no numerics payoff
  for an equivalence harness. This is disclosed, not quietly assumed.
"""
import torch

FP8_FORMAT_MAX = {
    "fp8_e4m3": 448.0,
    "fp8_e5m2": 57344.0,
}
FP8_MANTISSA_BITS = {
    "fp8_e4m3": 3,
    "fp8_e5m2": 2,
}
FP8_TORCH_DTYPE = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

INT4_MAX_LEVEL = 7  # symmetric range [-8, 7]; scale is derived from 7 so the
                    # positive side never needs to clip a representable value


def _group_view(x: torch.Tensor, group_size: int):
    """Reshapes (..., K) -> (..., K // group_size, group_size). K must be a
    multiple of group_size, which every shape in this sweep's grid is (the
    K sizes are 32/64/128/256/512, group_size is 32)."""
    k = x.shape[-1]
    assert k % group_size == 0, f"K={k} not a multiple of group_size={group_size}"
    return x.reshape(*x.shape[:-1], k // group_size, group_size)


def quantize_fp8_groupwise(x: torch.Tensor, group_size: int, fmt: str):
    """x: (..., K) fp32. Returns (q, scale): q is x's shape cast to the fp8
    torch dtype, scale is (..., K // group_size) fp32."""
    xg = _group_view(x, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_FORMAT_MAX[fmt]
    q = (xg / scale).to(FP8_TORCH_DTYPE[fmt])
    return q.reshape(x.shape), scale.squeeze(-1)


def dequantize_fp8_groupwise(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    qg = _group_view(q.to(torch.float32), group_size)
    deq = qg * scale.unsqueeze(-1)
    return deq.reshape(q.shape)


def quantize_int4_groupwise(x: torch.Tensor, group_size: int):
    """x: (..., K) fp32. Returns (q, scale): q is int8 holding values in
    [-8, 7], scale is (..., K // group_size) fp32."""
    xg = _group_view(x, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / INT4_MAX_LEVEL
    q = torch.round(xg / scale).clamp(-8, 7).to(torch.int8)
    return q.reshape(x.shape), scale.squeeze(-1)


def dequantize_int4_groupwise(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    qg = _group_view(q.to(torch.float32), group_size)
    deq = qg * scale.unsqueeze(-1)
    return deq.reshape(q.shape)


def fp8_quant_step(scale: torch.Tensor, fmt: str) -> torch.Tensor:
    """Per-group absolute quantization step: half the format's own ULP at
    this group's scale, i.e. the largest rounding error a single element
    in this group could have carried. mantissa_bits+1 because a float
    format's relative step between adjacent representable values near 1.0
    is 2^-mantissa_bits, and rounding error is at most half of that step."""
    return scale * (2.0 ** -(FP8_MANTISSA_BITS[fmt] + 1)) * FP8_FORMAT_MAX[fmt]


def int4_quant_step(scale: torch.Tensor) -> torch.Tensor:
    """INT4's levels are integer-spaced, so the quantization step is
    exactly `scale` (one level), and the max rounding error is half that."""
    return scale * 0.5
