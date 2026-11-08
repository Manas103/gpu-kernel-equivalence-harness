"""The 24 Triton-kernel-writing tasks the agent loop attempts.

Each task supplies a plain PyTorch eager reference (`ref`), used two ways:
cast to float64 it is the correctness oracle; run in the input dtype it is
the speedup baseline. `shapes` returns freshly seeded input tensors. `k`
is the reduction length used by the tolerance budget (see tolerance.py);
`k=1` for a task with no reduction (a pointwise map).
"""
import torch

TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _seeded(seed, shape, dtype, device, lo=-1.0, hi=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = (torch.rand(shape, generator=g, dtype=torch.float32) * (hi - lo) + lo)
    return t.to(device=device, dtype=dtype)


class Task:
    def __init__(self, name, category, k, make_inputs, ref, out_dtype_matches_input=True):
        self.name = name
        self.category = category  # "elementwise" | "reduction" | "norm" | "matmul"
        self.k = k  # reduction length for the tolerance budget; 1 for pointwise
        self.make_inputs = make_inputs
        self.ref = ref
        self.out_dtype_matches_input = out_dtype_matches_input

    def inputs(self, dtype, device, seed):
        return self.make_inputs(dtype, device, seed)

    def oracle(self, inputs_fp64):
        return self.ref(*inputs_fp64)

    def eager(self, *inputs):
        return self.ref(*inputs)


def _mk(shape):
    def make(dtype, device, seed):
        a = _seeded(seed, shape, dtype, device)
        b = _seeded(seed + 1, shape, dtype, device)
        return (a, b)
    return make


def _mk1(shape, lo=-1.0, hi=1.0):
    def make(dtype, device, seed):
        return (_seeded(seed, shape, dtype, device, lo, hi),)
    return make


N = 1 << 18  # 262144 elements for 1D pointwise tasks
ROW, COL = 512, 1024  # row-wise reduction tasks: 512 rows x 1024 cols


TASKS = [
    Task("vector_add", "elementwise", 1, _mk((N,)), lambda a, b: a + b),
    Task("vector_mul", "elementwise", 1, _mk((N,)), lambda a, b: a * b),
    Task("elementwise_sub", "elementwise", 1, _mk((N,)), lambda a, b: a - b),
    Task("elementwise_div", "elementwise", 1, _mk((N,)),
         lambda a, b: a / (b.abs() + 0.5)),
    Task("relu", "elementwise", 1, _mk1((N,)), lambda a: torch.relu(a)),
    Task("leaky_relu", "elementwise", 1, _mk1((N,)),
         lambda a: torch.where(a > 0, a, 0.01 * a)),
    Task("sigmoid", "elementwise", 1, _mk1((N,), -4.0, 4.0), lambda a: torch.sigmoid(a)),
    Task("tanh_act", "elementwise", 1, _mk1((N,), -3.0, 3.0), lambda a: torch.tanh(a)),
    Task("gelu_tanh_approx", "elementwise", 1, _mk1((N,), -3.0, 3.0),
         lambda a: 0.5 * a * (1.0 + torch.tanh(0.7978845608 * (a + 0.044715 * a ** 3)))),
    Task("softplus", "elementwise", 1, _mk1((N,), -4.0, 4.0),
         lambda a: torch.log1p(torch.exp(-a.abs())) + torch.relu(a)),
    Task("clamp", "elementwise", 1, _mk1((N,), -3.0, 3.0),
         lambda a: torch.clamp(a, -1.0, 1.0)),
    Task("elementwise_fma", "elementwise", 1,
         lambda dtype, device, seed: (
             _seeded(seed, (N,), dtype, device), _seeded(seed + 1, (N,), dtype, device),
             _seeded(seed + 2, (N,), dtype, device)),
         lambda a, b, c: a * b + c),
    Task("sqrt_positive", "elementwise", 1, _mk1((N,), 0.01, 4.0), lambda a: torch.sqrt(a)),
    Task("exp_elementwise", "elementwise", 1, _mk1((N,), -2.0, 2.0), lambda a: torch.exp(a)),
    Task("log_elementwise", "elementwise", 1, _mk1((N,), 0.01, 4.0), lambda a: torch.log(a)),
    Task("cumsum_row", "reduction", COL, _mk1((ROW, COL), -1.0, 1.0),
         lambda a: torch.cumsum(a.float(), dim=-1).to(a.dtype)),
    Task("row_sum", "reduction", COL, _mk1((ROW, COL), -1.0, 1.0),
         lambda a: a.float().sum(dim=-1).to(a.dtype)),
    Task("row_max", "reduction", COL, _mk1((ROW, COL), -1.0, 1.0),
         lambda a: a.float().amax(dim=-1).to(a.dtype)),
    Task("softmax_row", "norm", COL, _mk1((ROW, COL), -2.0, 2.0),
         lambda a: torch.softmax(a.float(), dim=-1).to(a.dtype)),
    Task("layernorm_row", "norm", COL, _mk1((ROW, COL), -2.0, 2.0),
         lambda a: torch.nn.functional.layer_norm(a.float(), (a.shape[-1],), eps=1e-5).to(a.dtype)),
    Task("rmsnorm_row", "norm", COL, _mk1((ROW, COL), -2.0, 2.0),
         lambda a: (a.float() * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(a.dtype)),
    Task("l2_normalize_row", "norm", COL, _mk1((ROW, COL), -2.0, 2.0),
         lambda a: torch.nn.functional.normalize(a.float(), p=2.0, dim=-1).to(a.dtype)),
    Task("matmul_small", "matmul", 128, _mk((256, 128)),
         lambda a, b: None,  # placeholder, replaced below with real shapes
         ),
    Task("matmul_rect", "matmul", 256, _mk((384, 256)), lambda a, b: None),
]


def _matmul_inputs(M, K, Ndim):
    def make(dtype, device, seed):
        a = _seeded(seed, (M, K), dtype, device)
        b = _seeded(seed + 1, (K, Ndim), dtype, device)
        return (a, b)
    return make


# Replace the two matmul placeholders with real (M, K, N) shapes and refs.
TASKS[-2] = Task("matmul_small", "matmul", 128, _matmul_inputs(128, 128, 128),
                  lambda a, b: a @ b)
TASKS[-1] = Task("matmul_rect", "matmul", 256, _matmul_inputs(384, 256, 192),
                  lambda a, b: a @ b)

assert len(TASKS) == 24, f"expected 24 tasks, got {len(TASKS)}"
TASK_NAMES = [t.name for t in TASKS]
TASKS_BY_NAME = {t.name: t for t in TASKS}
