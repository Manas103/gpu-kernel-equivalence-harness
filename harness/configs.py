"""The sweep grid: 80 shapes x 3 dtypes x 10 repeats = 2,400 configurations."""

M_SIZES = [64, 128, 256, 512]
N_SIZES = [64, 128, 256, 512]
K_SIZES = [32, 64, 128, 256, 512]
DTYPES = ["fp16", "bf16", "tf32"]
REPEATS = 10

SHAPES = [(m, n, k) for m in M_SIZES for n in N_SIZES for k in K_SIZES]

assert len(SHAPES) == 80
assert len(SHAPES) * len(DTYPES) * REPEATS == 2400
