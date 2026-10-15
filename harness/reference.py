"""The cuBLAS reference: ordinary torch.matmul on CUDA tensors, which
dispatches to cuBLAS for fp16, bf16 and fp32 GEMM. For "tf32" the comparison
is against cuBLAS run in its own real TF32 tensor-core mode
(torch.backends.cuda.matmul.allow_tf32 = True), not the hand-written
kernel's bit-truncation emulation of the same arithmetic; that is the whole
point of comparing the two."""
import torch


def cublas_matmul(a: torch.Tensor, b: torch.Tensor, dtype_kind: str) -> torch.Tensor:
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = (dtype_kind == "tf32")
        return torch.matmul(a, b).to(torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
