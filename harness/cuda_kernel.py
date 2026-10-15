"""Loads the hand-written CUDA kernel in kernels/matmul.cu as a plain
shared library via ctypes, operating directly on a torch CUDA tensor's
device pointer (no host copy). See kernels/matmul.cu for why this is
ctypes rather than torch.utils.cpp_extension.load."""
import ctypes
import os

import torch

_SO_PATH = os.path.join(os.path.dirname(__file__), "..", "kernels", "libmatmul.so")
_lib = None

_LAUNCHERS = {
    "fp16": "launch_matmul_fp16",
    "bf16": "launch_matmul_bf16",
    "tf32": "launch_matmul_tf32",
}

_TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "tf32": torch.float32,  # TF32 truncation happens inside the kernel; storage is fp32
}


def _get_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_SO_PATH):
            raise RuntimeError(f"{_SO_PATH} not built; run kernels/build.sh first")
        lib = ctypes.CDLL(_SO_PATH)
        for name in _LAUNCHERS.values():
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype = None
        lib.matmul_sync_and_get_last_error.argtypes = []
        lib.matmul_sync_and_get_last_error.restype = ctypes.c_int
        _lib = lib
    return _lib


def cuda_matmul(a: torch.Tensor, b: torch.Tensor, dtype_kind: str) -> torch.Tensor:
    """dtype_kind is one of "fp16", "bf16", "tf32". For "tf32", a and b must
    already be torch.float32 tensors; the kernel truncates each operand's
    mantissa to TF32's 10 bits itself."""
    lib = _get_lib()
    assert a.is_cuda and b.is_cuda
    assert a.dtype == _TORCH_DTYPE[dtype_kind] and b.dtype == _TORCH_DTYPE[dtype_kind]
    a = a.contiguous()
    b = b.contiguous()
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)

    launcher = getattr(lib, _LAUNCHERS[dtype_kind])
    launcher(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
    )
    err = lib.matmul_sync_and_get_last_error()
    if err != 0:
        raise RuntimeError(f"CUDA kernel launch failed, cudaError_t={err}")
    return c
