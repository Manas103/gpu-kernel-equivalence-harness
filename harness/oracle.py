"""A deliberately slow, obviously correct reference: convert inputs to
numpy float64 on the CPU and matmul there. This is independent of every
implementation under test (the hand-written CUDA kernel, the Triton
kernel, and cuBLAS all run in reduced precision on the GPU; this runs in
full double precision on the CPU, sharing no code path with any of them),
which is what makes it usable as a correctness oracle rather than just
another reduced-precision implementation to diff against."""
import torch


def oracle_matmul_f64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a64 = a.detach().to("cpu", dtype=torch.float64)
    b64 = b.detach().to("cpu", dtype=torch.float64)
    return (a64 @ b64).to(torch.float32)
