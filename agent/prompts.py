"""Builds the prompt sent to the model for each task/attempt. The model
never sees the oracle, the tolerance budget, or any other task's result;
it sees only its own task description and, on a retry, the deterministic
control plane's structured feedback on its own previous attempt."""

DESCRIPTIONS = {
    "vector_add": "elementwise c = a + b on two 1D tensors",
    "vector_mul": "elementwise c = a * b on two 1D tensors",
    "elementwise_sub": "elementwise c = a - b on two 1D tensors",
    "elementwise_div": "elementwise c = a / (|b| + 0.5) on two 1D tensors",
    "relu": "elementwise ReLU: max(a, 0) on a 1D tensor",
    "leaky_relu": "elementwise leaky ReLU with negative slope 0.01: a if a>0 else 0.01*a",
    "sigmoid": "elementwise sigmoid: 1 / (1 + exp(-a))",
    "tanh_act": "elementwise tanh(a)",
    "gelu_tanh_approx": (
        "elementwise GELU, tanh approximation: "
        "0.5*a*(1 + tanh(0.7978845608*(a + 0.044715*a^3)))"
    ),
    "softplus": "elementwise softplus: log(1 + exp(a)), computed in a numerically stable way",
    "clamp": "elementwise clamp(a, -1.0, 1.0)",
    "elementwise_fma": "elementwise fused multiply-add: d = a*b + c on three 1D tensors",
    "sqrt_positive": "elementwise sqrt(a); input is already positive",
    "exp_elementwise": "elementwise exp(a)",
    "log_elementwise": "elementwise log(a); input is already positive",
    "cumsum_row": "row-wise cumulative sum along the last dimension of a 2D tensor, accumulated in fp32",
    "row_sum": "row-wise sum reduction along the last dimension of a 2D tensor, accumulated in fp32, output is 1D (one value per row)",
    "row_max": "row-wise max reduction along the last dimension of a 2D tensor, output is 1D (one value per row)",
    "softmax_row": "row-wise softmax along the last dimension of a 2D tensor, numerically stable (subtract the row max before exponentiating), accumulated in fp32",
    "layernorm_row": "row-wise layer normalization along the last dimension of a 2D tensor, no learned scale/bias, eps=1e-5, accumulated in fp32",
    "rmsnorm_row": "row-wise RMSNorm along the last dimension of a 2D tensor, no learned scale, eps=1e-5: a * rsqrt(mean(a^2) + eps)",
    "l2_normalize_row": "row-wise L2 normalization along the last dimension of a 2D tensor: a / ||a||_2 (p=2 norm over the row)",
    "matmul_small": "matrix multiplication C = A @ B, A is (128,128), B is (128,128), fp32 accumulation",
    "matmul_rect": "matrix multiplication C = A @ B, A is (384,256), B is (256,192), fp32 accumulation",
}

_N = 262144
_ROW, _COL = 512, 1024

SHAPES = {
    "vector_add": f"a: ({_N},), b: ({_N},)",
    "vector_mul": f"a: ({_N},), b: ({_N},)",
    "elementwise_sub": f"a: ({_N},), b: ({_N},)",
    "elementwise_div": f"a: ({_N},), b: ({_N},)",
    "relu": f"a: ({_N},)",
    "leaky_relu": f"a: ({_N},)",
    "sigmoid": f"a: ({_N},)",
    "tanh_act": f"a: ({_N},)",
    "gelu_tanh_approx": f"a: ({_N},)",
    "softplus": f"a: ({_N},)",
    "clamp": f"a: ({_N},)",
    "elementwise_fma": f"a: ({_N},), b: ({_N},), c: ({_N},)",
    "sqrt_positive": f"a: ({_N},)",
    "exp_elementwise": f"a: ({_N},)",
    "log_elementwise": f"a: ({_N},)",
    "cumsum_row": f"a: ({_ROW}, {_COL}), output same shape",
    "row_sum": f"a: ({_ROW}, {_COL}), output ({_ROW},)",
    "row_max": f"a: ({_ROW}, {_COL}), output ({_ROW},)",
    "softmax_row": f"a: ({_ROW}, {_COL}), output same shape",
    "layernorm_row": f"a: ({_ROW}, {_COL}), output same shape",
    "rmsnorm_row": f"a: ({_ROW}, {_COL}), output same shape",
    "l2_normalize_row": f"a: ({_ROW}, {_COL}), output same shape",
    "matmul_small": "a: (128,128), b: (128,128), output (128,128)",
    "matmul_rect": "a: (384,256), b: (256,192), output (384,192)",
}

SYSTEM_PREAMBLE = """You write a single OpenAI Triton GPU kernel file. Output ONLY one Python code block, no prose before or after it.

Hard requirements for the code block:
1. It must define one or more @triton.jit kernel functions, and a module-level
   `TRITON_KERNELS = [<kernel_fn>, ...]` list naming every @triton.jit kernel you defined
   (not the launch wrapper, the raw jit function objects).
2. It must define `def run(*tensors: torch.Tensor) -> torch.Tensor:` that launches your
   kernel(s) on CUDA tensors and returns the result as a torch.Tensor on the same device
   and same dtype as the inputs, with the shape stated in the task.
3. Only `import torch`, `import triton`, `import triton.language as tl`. Do not call any
   torch tensor-math op (+, *, torch.add, torch.matmul, etc.) to compute the actual result;
   torch may only be used for simple shape/empty-tensor bookkeeping (torch.empty,
   .shape, .stride, .device, .dtype).
4. Do not write a __main__ block, a test, or example usage. Just the kernel(s) and `run`.
5. Internal accumulation in fp32 where the task says so; store the final result back in
   the input dtype.
"""

RETRY_PREFIX = """Your previous attempt at this task did not pass. Here is the deterministic
control plane's structured feedback on that attempt (compile/runtime errors, the max
absolute difference against an fp64 oracle if it ran, and a profiling summary if it ran).
Fix the kernel. Output another complete, standalone code block in the same format.

--- feedback on previous attempt ---
{feedback}
--- end feedback ---
"""


def build_prompt(task_name, dtype_name, feedback=None):
    desc = DESCRIPTIONS[task_name]
    body = (
        f"Task: {task_name}\n"
        f"Operation: {desc}\n"
        f"Input dtype: torch.{dtype_name}, device: cuda\n"
        f"Shapes: {SHAPES[task_name]}\n"
    )
    if feedback:
        return SYSTEM_PREAMBLE + "\n" + body + "\n" + RETRY_PREFIX.format(feedback=feedback)
    return SYSTEM_PREAMBLE + "\n" + body
