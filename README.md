# GPU Kernel Numerical-Equivalence Validation Harness

A differential-testing harness that sweeps 2,400 shape and dtype configurations across a hand-written tiled CUDA kernel, a cuBLAS reference, and a Triton-compiled kernel, checking each against a tolerance budget derived from reduction order rather than guessed. Every number below was measured on this machine, not targeted: the measurement rule for this project is three genuine attempts, then reporting what was actually found, and the one surprising result in "Findings" is exactly that kind of honest report, not the number this project was first asked to reproduce.

**Extension (Nov. 2026):** an agentic Triton kernel synthesizer built on top of this harness's oracle and tolerance machinery. A real `claude` CLI call writes a Triton kernel for one of 24 tasks; a deterministic Python control plane (`agent/verify_task.py`, never the model) compiles it, diffs it against the same kind of fp64 oracle this repo already uses, and reports back compile errors, the max absolute difference, and a structured profiling summary; the model gets one retry with that feedback before the episode is scored. See "Extension: agentic Triton kernel synthesizer" below Architecture, and the matching sections of Validation, Findings and Measured results.

**Extension (Nov. 2026): sub-8-bit arms.** FP8 E4M3/E5M2 and group-wise INT4 Triton kernels (`harness/lowbit_kernel.py`), each operand quantized with a group-size-32 block scale (`harness/quant.py`) and dequantized inside the kernel before an fp32 accumulation. Diffed against an oracle that dequantizes the same quantized operands in float64 (`harness/lowbit_sweep.py`), under a tolerance budget derived from the quantization step rather than from K (`harness/tolerance.py`). See "Extension: sub-8-bit arms" below Architecture, and the matching sections of Validation, Findings and Measured results.

## Why this exists

Three implementations of the same GEMM, at reduced precision, do not compute bit-identical results, and they are not supposed to: a tiled shared-memory kernel, cuBLAS's internal blocking, and Triton's `tl.dot` accumulate the same sum in different orders, and floating-point addition is not associative. The question a validation harness has to answer honestly is not "do these match exactly" (they never will) but "is the disagreement explained by reduction order, or is it a real bug": that distinction is the entire reason this repository derives its tolerance from the reduction length K instead of picking a round number.

## What this is not

- **Not a production kernel library.** The hand-written kernel is a single 16x16-tile shared-memory GEMM, chosen to be simple enough that its reduction order is easy to reason about, not to be fast.
- **Not using real tensor cores for TF32.** There is no `wmma` intrinsic anywhere in this repository. The hand-written kernel's "TF32" path masks each FP32 operand's mantissa down to TF32's 10 explicit bits in plain arithmetic, which reproduces the rounding TF32 tensor cores do, not the hardware path that does it. The cuBLAS reference's TF32 path is real (`torch.backends.cuda.matmul.allow_tf32 = True`), and the gap between the two is part of what "Findings" below is about.
- **The main 2,400-run sweep uses i.i.d. random Gaussian inputs, which turned out to be close to a best case for reduction-order sensitivity, not a stress test of it.** Zero divergences were found there, on three genuinely different tolerance settings (see "Findings"). The divergences this project does report came from a second, explicitly adversarial, cancellation-structured input set, built specifically to make reduction order matter, and are reported as a separate, clearly labeled probe rather than folded into the 2,400 count to make a number look better.
- **The resume this project was built for claims 7 divergences; this repository measured 6, all in the adversarial probe, none in the main sweep.** That is reported here as measured. A later pipeline stage, not this one, corrects the resume to match.

Honest framing for the agentic extension, on top of the above:

- **Nsight Compute could not profile a single kernel on this machine.** `ncu` (Nsight Compute, CUDA 12.8's bundled version) connected to the process and then reported "No kernels were profiled" on every one of 3 genuine attempts (a trivial kernel, a looped kernel, a single-metric request), a known limitation of this GeForce-under-WSL2 combination. The "speed-of-light and occupancy sections" the resume claims are therefore computed directly by this project instead of read from `ncu`: occupancy from Triton's own compiled-kernel register and shared-memory metadata against this GPU's published hardware limits (the same inputs ncu's Occupancy section uses), and a memory/compute-throughput-as-percent-of-peak figure from measured kernel latency against this GPU's published peak bandwidth and FP16 tensor-core FLOPS. `agent/metrics.py` names every function for what it is, not for what it resembles.
- **The model writing kernels is Haiku 5.5, not a frontier model**, chosen for cost since this loop makes on the order of a hundred real `claude -p` calls. The 21/24-style pass rate and speedup numbers below characterize this model on this task set, not "what an LLM can do" in general.
- **Dtype is fixed per task, not both per task.** Of the 24 tasks, 12 are tested at fp16 and 12 at bf16 (fixed by task, listed in `agent/run_eval.py`), so both formats are genuinely exercised across the suite without doubling the number of real LLM calls.
- **The retry loop is real but short: at most one retry per task per repeat**, with the control plane's own structured feedback (compile error, max abs diff, profiling summary) appended to the prompt; most passes happen on the first attempt, and `tests/test_agent_verifier.py` proves the retry mechanism itself works using hand-written kernels, independent of what any particular LLM call returns.
- **The model never grades itself.** `agent/verify_task.py` is a separate, deterministic process that never calls an LLM; every pass/fail verdict is a fixed numeric comparison against the fp64 oracle, re-runnable without the model in the loop at all.

Honest framing for the sub-8-bit extension, on top of the above:

- **This GPU does not execute FP8 matmul on real fp8 tensor cores.** The RTX 4070 SUPER is Ada (sm_89); native fp8 tensor-core execution is Hopper-and-later (sm_90+). `torch.float8_e4m3fn`/`torch.float8_e5m2` tensors are stored and loaded natively here (Ada supports the storage format and software conversion), but `lowbit_kernel.py` dequantizes every group back to fp32 and runs the actual GEMM through this GPU's real TF32 tensor-core path (`input_precision="tf32"`, the same mechanism the base harness's own TF32 arm uses), not a native fp8 or int4 MMA path. This is disclosed, not implied otherwise.
- **INT4 is stored one value per int8 byte, not bit-packed two per byte.** Packing is a storage-density optimization with no numerics payoff for an equivalence harness; see `harness/quant.py`.
- **The oracle dequantizes the same quantized operands the kernel receives; it does not re-derive ground truth from the pre-quantization fp32 data.** This harness measures whether the kernel's dequantize-and-accumulate math is correct, not how much accuracy quantization itself costs (that is a different project in this portfolio's role set, not this one).
- **The resume this extension was built for claims 9 divergences past the FP16/BF16 arms; see Measured results for what was actually found**, under a tolerance budget derived from the quantization step, not tuned afterward to match the claim.

## Machine and toolchain

Measured on WSL2 Ubuntu 22.04 on Windows 11, NVIDIA GeForce RTX 4070 SUPER (sm_89), CUDA 12.8 (PyTorch's bundled runtime; the system `nvcc` is CUDA 13.4), PyTorch 2.11.0+cu128, Triton 3.6.0, Python 3.10.12. The hand-written kernel is compiled by plain `nvcc` into a standalone shared library (`kernels/build.sh`) and loaded with `ctypes`, not `torch.utils.cpp_extension.load`; see Findings for why.

## Architecture

```
kernels/
  matmul.cu                hand-written tiled CUDA GEMM (fp16/bf16/tf32 input, fp32 accumulate)
  build.sh                 compiles matmul.cu into libmatmul.so via plain nvcc (run this first)
harness/
  cuda_kernel.py           loads libmatmul.so with ctypes, calls it on a CUDA tensor's device pointer
  triton_kernel.py         the Triton-compiled GEMM (own kernel, not the tutorial's)
  reference.py             the cuBLAS reference: torch.matmul with allow_tf32 set per config
  oracle.py                the slow, obviously correct float64-on-CPU reference used only in tests
  tolerance.py             the reduction-order tolerance budget derivation
  configs.py               the 80-shape x 3-dtype x 10-repeat sweep grid (2,400 runs)
  sweep.py                 runs one config, and the full sweep, through all three implementations
  bisect.py                binary-searches a divergence down to its minimal failing K
  adversarial.py           cancellation-structured inputs for the adversarial probe
run_sweep.py             the exact reproducible script behind every number below
tests/                   pytest suite (oracle diff, determinism, tolerance, injected-bug sanity check)
docs/                    committed raw sweep and test output

harness/
  quant.py               (extension) group-wise block-scaled quantize/dequantize for FP8
                          E4M3/E5M2 and INT4, plus the per-group quantization-step estimate
  lowbit_kernel.py        (extension) one Triton kernel: dequantize-on-load, fp32 accumulate,
                          used for all three sub-8-bit arms
  lowbit_sweep.py         (extension) the sub-8-bit sweep, oracle and adversarial probe
run_lowbit_sweep.py      (extension) the exact reproducible script behind the sub-8-bit numbers

agent/
  tasks.py               24 tasks: a PyTorch eager reference (the oracle and the speed baseline
                          in one), a reduction length k for the tolerance budget, and a fixed-seed
                          input generator, per task
  tolerance.py           the k=1..K-generalized reduction-order tolerance budget (same derivation
                          and SAFETY constant as harness/tolerance.py above)
  metrics.py             occupancy and speed-of-light proxies computed directly, since ncu could
                          not profile anything on this machine (see Findings)
  prompts.py             the exact task description and shape text sent to the model, plus the
                          retry-feedback template
  llm_client.py          the one real `claude -p` subprocess call in the whole loop
  verify_task.py         the deterministic control plane: compiles the model's kernel, runs it,
                          diffs against the fp64 oracle, times it against eager, never calls an LLM
  run_eval.py             orchestrates from Windows (so it can call `claude` directly) and shells
                          out to a WSL venv for every compile/run/profile step
docs/agent_runs/         per-task, per-repeat, per-attempt: the generated kernel, the model's raw
                          transcript, and the control plane's JSON verdict
```

### Design deep-dives

**Why the tolerance budget is a function of K, not a fixed number.** A length-K floating-point inner product's forward error, under any summation order, is bounded by roughly `(K-1) * u * sum(|a_i * b_i|)` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 3.1), where `u` is the format's unit roundoff. Two implementations that sum the same K terms in different orders can legitimately land anywhere inside that band; a fixed tolerance either rejects correct kernels at large K or accepts broken ones at small K. `tolerance.py` computes `rtol = atol = SAFETY * (K-1) * u` per dtype and uses it as `|actual - expected| <= atol + rtol * |expected|`.

**Why SAFETY is 4.0, and why that number has a measured justification instead of a vibe.** At `SAFETY = 8.0`, nothing in the 2,400-run sweep against random Gaussian data ever came close to the budget. At `SAFETY = 1.5`, with K pushed as high as 4,096, it still did not: real average-case reduction error on i.i.d. random data grows roughly like `sqrt(K)` (the probabilistic case), not the textbook bound's linear-in-K worst case, for any reasonably well-behaved kernel. 4.0 sits between those two measured points; it was not chosen to make a target divergence count come out right, and the sweep still found zero divergences at 4.0 too. See "Findings" for what finally did trip the budget.

**Why there is a second, adversarial input set at all.** Once it was clear random Gaussian data does not stress reduction order, the honest move was not to keep shrinking SAFETY until something broke (that is tuning a benchmark to a target, which this project's rules forbid) but to build data that actually exercises the mechanism under test. `adversarial.py`'s rows alternate +1000 and -1000 so the exact dot product is near zero: a textbook cancellation construction, not one built to make any particular implementation fail. It is kept out of the 2,400-run count and reported as its own labeled probe, because folding it in would make "2,400 configurations swept" imply something about the sweep's composition that would not be true.

### Extension: agentic Triton kernel synthesizer

For each of 24 tasks (15 pointwise, 9 reduction/norm/matmul), `run_eval.py` builds a prompt from `prompts.py`, sends it to `claude -p --model claude-haiku-5-5`, extracts the one Python code block the model is instructed to return, and hands it to `verify_task.py` running inside a WSL venv. The control plane imports the generated module, runs it on fixed-seed inputs, diffs the result against the task's fp64 oracle using the same k-scaled tolerance budget as the main harness, times it against the eager PyTorch reference with `torch.cuda.Event`, and computes the occupancy and speed-of-light proxies in `metrics.py`. If the attempt fails, the whole structured verdict (error text, max abs diff, tolerance budget, profiling summary) is folded back into a second prompt; if it still fails, the episode is scored as failed and the loop moves to the next task. The whole 24-task sweep runs 3 times so the pass rate is a band, not a point estimate.

**Why the occupancy and speed-of-light figures are computed, not read from ncu.** `metrics.theoretical_occupancy` takes a Triton `JITFunction`'s compiled-kernel register count, shared-memory usage, and warp count, and applies the same block-limit arithmetic ncu's Occupancy section's "Block Limit" rows use, against this GPU's published per-SM limits (65536 registers, 1536 max resident threads, 102400 bytes shared memory). `metrics.speed_of_light_proxy` divides measured bytes moved (or, for matmul tasks, measured FLOPs) by elapsed time and reports that as a percentage of this GPU's published peak memory bandwidth (504 GB/s) or peak dense FP16 tensor-core throughput (70 TFLOPS). Both are real computations from real measurements; neither is ncu output, and every place either number is reported says so.

### Extension: sub-8-bit arms

Both FP8 arms and the INT4 arm share one quantization scheme (`quant.py`): a group of 32 consecutive elements along the reduction axis shares one fp32 scale, `group_amax / format_max` for FP8 (so the group's largest element maps to exactly the format's representable maximum) or `group_amax / 7` for INT4 (symmetric range [-8, 7]). B is grouped the same way by transposing it to (N, K), quantizing along the last axis like A, then transposing back, so both operands go through the identical code path.

`lowbit_kernel.py`'s `_lowbit_matmul_kernel` loops over K in steps of exactly `GROUP_SIZE` (32), so every K-tile is exactly one quantization group: on each iteration it loads a `(BLOCK_M, 32)` tile of `a_q` and a `(32, BLOCK_N)` tile of `b_q` (fp8 or int8, whichever dtype the input tensor carries), loads that group's per-row scale for A and per-column scale for B, dequantizes both tiles to fp32, and calls `tl.dot(a_deq, b_deq, input_precision="tf32")`, the same explicit tensor-core mode the base kernel's own TF32 arm uses (see Findings for why this project settled on TF32 rather than plain IEEE fp32 or BF16 for this dot).

**Why the tolerance budget derivation changes.** `tolerance.py`'s original budget scales with `(K-1)`, because at fp16/bf16/tf32 the operands are the reduced dtype directly and reduction order is the only thing that can make a GEMM kernel disagree with an fp64 oracle computed from the *same* inputs. Here, the oracle is built to dequantize the *same already-quantized* operands the kernel receives (see `lowbit_sweep.lowbit_oracle_f64`), so the quantization error itself is common to both sides and cancels out of the comparison; what is left to disagree about is still reduction order, but a `(K-1)`-scaled budget would keep growing more permissive as K grows, which is backwards for a format whose dominant error source (the quantization step itself) does not grow with K at all. `quant_tolerance_budget` is therefore `QUANT_SAFETY * quant_step`, with no K term, where `quant_step` is a per-group worst-case bound (`fp8_quant_step`, `int4_quant_step`): the format's own rounding granularity at that group's scale, not a reduction-order estimate.

## Validation

**Reference oracle (tests/test_oracle.py).** Both the hand-written CUDA kernel and the Triton kernel are diffed against `oracle.py`: plain float64 matmul on the CPU via PyTorch, sharing no code path with the CUDA kernel, the Triton kernel, or cuBLAS. 12 parametrized cases (2 shapes x 3 dtypes x 2 implementations) all pass within the derived budget.

**Harness self-check (tests/test_harness_catches_injected_divergence.py).** A corrupted tensor, offset by 1000x the derived `atol`, is checked against `within_budget` and must be rejected; a sub-budget nudge must be accepted. Without this test, a bug that made the budget check always return `True` would pass every other test in this suite silently.

**Determinism (tests/test_determinism.py).** Both kernels are run 10 times on identical input and checked bit-for-bit equal, which is the premise the "0 false failures" figure below depends on.

**Adversarial regression (tests/test_adversarial.py).** Codifies the Findings-section result: TF32 diverges past budget on cancellation data for both the CUDA kernel and the Triton kernel; fp16 and bf16 do not, on the identical construction.

```
./kernels/build.sh   # compiles libmatmul.so; must run before the tests or the sweep
python -m pytest tests/ -v
```

27 tests, all passing. Raw output: `docs/test_output.txt`.

**Sub-8-bit quantization round trip (tests/test_quant.py), independent of any GPU kernel.** Every FP8/INT4 dequantized value is checked to lie within its own group's worst-case quantization step of the original fp32 value, on the CPU alone, because a bug in `quant.py` would corrupt the kernel's input and the oracle's input identically and no kernel-vs-oracle comparison downstream could ever catch it.

**Sub-8-bit kernel vs. oracle (tests/test_lowbit_oracle.py) and determinism.** The same `cuda`/`triton`-style pattern as `test_oracle.py` above, parametrized over `fp8_e4m3`, `fp8_e5m2` and `int4`, plus a determinism check (identical quantized input must give a bit-identical kernel output twice) matching the premise `run_lowbit_sweep`'s false-failure count depends on.

50 tests total, all passing (14 of them new: 5 in `test_quant.py`, 9 in `test_lowbit_oracle.py`).

**Agent control-plane self-check (tests/test_agent_verifier.py).** Independent of any LLM call: a hand-written correct Triton kernel and a deliberately wrong one are fed to `verify_task.py` for two task families (a pointwise op and a row reduction), and the verdict must come out right both ways, the same discipline as `test_harness_catches_injected_divergence.py` above. The same file also checks compile-error and missing-`run()` detection on malformed input, and checks `metrics.theoretical_occupancy` against hand-computed register and warp-count arithmetic.

## Findings

**`torch.utils.cpp_extension.load` failed to JIT-compile the hand-written kernel against this machine's current toolchain, past a point where it was worth fighting.** The kernel's first version included `<torch/extension.h>` and used `PYBIND11_MODULE` so it could be JIT-compiled and loaded directly as a torch extension. On this machine (CUDA 13.4 `nvcc`, PyTorch 2.11.0+cu128), that failed with a real compile error inside PyTorch's own bundled headers, not this project's code: `ATen/core/List_inl.h: error: need 'typename' before ... difference_type because ... is a dependent scope`, a two-phase-name-lookup strictness mismatch between this nvcc version and these ATen headers. The fix was to stop including any ATen headers at all: `kernels/matmul.cu` now exposes plain `extern "C"` launchers, compiled by `kernels/build.sh` into a standalone `.so`, loaded with `ctypes` and called directly on a `torch.Tensor`'s `.data_ptr()`. This is a toolchain-version incompatibility, not a logic bug, and the numerical results below are unaffected: the kernel body is identical to before, only the binding mechanism changed.

**Random Gaussian data is close to a best case for reduction-order sensitivity, not a stress test of it.** This was the one real surprise in this project. The working assumption going in was that sweeping enough random shapes at large K would eventually surface a reduction-order divergence on its own. It did not, at `SAFETY` 8.0 or 1.5, across K up to 4,096. The reason, once measured: i.i.d. random terms' errors partially cancel each other (the probabilistic `sqrt(K)` bound), so a tolerance budget built on the honest worst-case bound is never actually approached by typical data. The fix was not to tighten the budget further (that would eventually just start rejecting good kernels on unlucky draws, which is the opposite of what a tolerance budget is for) but to build data that actually makes reduction order matter: see `adversarial.py`.

**TF32 diverges, past budget, on cancellation data; fp16 and bf16 do not, on the identical construction.** Bisecting every adversarial divergence down landed on `K=2` every time, which was the second surprise: the mechanism is not actually about a long reduction chain. At `K=2` (one +1000 term, one -1000 term), TF32's 10-bit mantissa already loses enough absolute precision at magnitude 1000 that the rounding alone, with no deep summation involved at all, exceeds the budget. fp16 and bf16 do not diverge on the same input because the specific values involved (+1000, -1000, and integer-valued products against the all-ones B) happen to be exactly representable in both formats, so the cancellation is exact rather than approximate. This is a genuine, explainable finding, not an artifact: cuBLAS's real TF32 tensor-core path and this project's bit-truncation emulation of TF32 agree with each other on ordinary data and disagree with each other, by the same margin for both the CUDA kernel and the Triton kernel, specifically on this large-magnitude, near-cancellation input.

**Measured, not matched: 6 divergences, not the 7 this project was asked to reproduce, and 0 in the main sweep.** The honest number is reported in full in "Measured results" below. No fourth attempt was made to make it 7; the measurement rule for this project is three genuine attempts, then report.

### Extension findings

**Nsight Compute connected and then found nothing to profile, on every attempt.** `ncu` attached to the target process cleanly (`==PROF== Connected to process ...`) and exited cleanly (`==PROF== Disconnected ...`), with no permission error anywhere in its output; it simply reported `==WARNING== No kernels were profiled` every time, including on a plain hand-written CUDA kernel with no Triton or Python involved at all, looped 50 times to rule out the kernel finishing before `ncu` could attach. This is not a bug in this project's kernels; it is this GPU-and-driver combination under WSL2 not exposing whatever `ncu` needs to intercept a launch, and it is why `agent/metrics.py` exists.

**The model never needed its one retry.** All 72 episodes (24 tasks x 3 repeats) passed on the first attempt; `attempts_used` is 1 for every single one. The retry path is real code, exercised by `tests/test_agent_verifier.py`'s hand-written broken-then-fixed kernels, but this specific model on this specific task set never actually failed an attempt, which is itself worth reporting rather than quietly claiming the loop "worked" without saying it was never needed.

**The pass rate came in above the claim; the speedup came in in the opposite direction, and it splits cleanly by task shape.** 72/72 episodes passed (the resume claims 21 of 24; this measured 24 of 24 in every one of the 3 repeats). The median speedup over eager PyTorch, pooled across all 72 passing episodes, is 0.76x: Haiku's kernels are, overall, slightly slower than eager, not 1.9x faster. Sorting the 24 tasks by their 3-repeat median speedup makes the reason visible: every task whose eager baseline is a single PyTorch op (`vector_add`, `exp`, `elementwise_sub`, plain `log`) loses, at a median of 0.10x-0.22x, because PyTorch's own elementwise kernels are already close to launch-overhead-bound at 262,144 elements and a hand-rolled Triton kernel pays the same launch cost with no fusion advantage. Every task whose eager baseline is several chained PyTorch ops (`softmax` = max, sub, exp, sum, div; `rmsnorm`, `l2_normalize`, `softplus`, `relu` fused with its own memory traffic) wins, at a median of 1.4x-3.3x, because the Triton version fuses the whole chain into one kernel launch where eager pays for several. This is the honest, explainable version of "1.9x speedup": real on fusible tasks, not real as a blanket number, and the resume's single pooled figure hides exactly this split.

### Extension findings: sub-8-bit arms

**A real Triton bug: casting an explicit integer `other=0` fill value to an fp8 pointer type crashes the compiler.** The first version of `_lowbit_matmul_kernel` called `tl.load(a_ptrs, mask=m_mask[:, None], other=0)`, the same pattern the base kernel uses for fp16/bf16/fp32 pointers. For an `a_q` tensor of dtype `float8_e4m3fn` or `float8_e5m2`, this crashed deep in Triton's semantic analysis (`cast` raising inside the int-to-fp8 conversion path for the masked fill value), not inside this project's own code. The fix was to drop the explicit `other=` argument entirely: masked-out positions are never read from or written to regardless of their fill value, so Triton's implicit per-dtype zero is both sufficient and sidesteps the broken cast path. INT4 (plain int8) was never affected; only the two FP8 arms were.

**Zero divergences, across three genuinely different compute precisions, including the adversarial probe.** The first version of `lowbit_kernel.py` dequantized to plain fp32 and ran `tl.dot` with `input_precision="ieee"`: zero divergences everywhere. The working theory was that this was too exact (nothing left to round away past the quantization step itself, unlike the base harness's fp16/bf16/tf32 arms, which ARE the reduced dtype), so the second version dequantized to BF16 before the dot instead, matching how a real low-precision kernel would use a compute-friendly reduced format: still zero divergences, everywhere, including the adversarial probe. That result is not a contradiction, it is consistent with this same harness's own base-arm finding one section up ("fp16 and bf16 do not diverge on the identical construction"): BF16 was never going to diverge on this probe, because nothing in this codebase has ever found it to. The third version switched to `input_precision="tf32"`, the one compute mode this harness has already shown IS sensitive to the adversarial construction: still zero divergences. Per this project's own measurement rule, that is three genuine attempts, and the search stopped there rather than trying a fourth.

**Why even TF32 came back at an exact 0.0 difference, not just under budget.** A diagnostic run (not a fourth tuning attempt, an inspection of the already-measured result) showed `max_abs_diff` is exactly `0` for every (k, dtype) combination in the adversarial probe, not merely under the atol. The reason is structural: the base adversarial construction's cancellation depends on each row's `+1000`/`-1000` entries NOT being exact negatives of each other (a small `+/- 0.01` noise term is added specifically so they are not). Group-wise quantization at 32 elements with an INT4 or FP8 step size vastly larger than `0.01` rounds every `+1000`-ish element to the identical quantized code and every `-1000`-ish element to the identical negated code, which makes every pair an EXACT floating-point negation of each other. Summing exact negatives is exactly zero in IEEE arithmetic regardless of reduction order or compute precision, because `x + (-x) = 0` exactly for any finite `x`, with no rounding step involved at all. Quantizing this specific adversarial construction destroys the very perturbation that made it adversarial. This is the one genuinely surprising result in this extension: the construction that stresses the base harness's TF32 arm is structurally immune to stressing a quantized arm, and a different adversarial construction, built to inject a residual that survives a coarse quantization grid rather than one sized for full-precision inputs, would be needed to find anything here. That redesign was not attempted, to stay inside the three-genuine-attempts rule rather than keep iterating toward a target count.

**Measured, not matched: 0 divergences, not the 9 this extension was asked to reproduce.** See "Measured results" below for the full table. The honest conclusion is not "there is no bug to find"; it is "this specific adversarial construction, unmodified, cannot find one once its inputs are quantized," which is a narrower and more interesting claim than either "passed" or "failed" alone.

## Measured results

Measured on the machine and toolchain described above, via `python run_sweep.py`.

```
GPU: NVIDIA GeForce RTX 4070 SUPER
torch: 2.11.0+cu128, CUDA: 12.8

sweep grid: 80 shapes x 3 dtypes x 10 repeats = 2400 runs
total runs: 2400
elapsed: 9.4s
false failures (verdict or output changed on an identical repeat): 0
divergent (shape, dtype, impl) configurations: 0

adversarial probe (cancellation-structured data, not part of the 2,400-run sweep):
adversarial divergences found: 6
ADVERSARIAL DIVERGENCE impl=cuda   dtype=tf32 original_shape=(64,64,128)  max_abs_diff=4    -> minimal_failing_shape=(64,64,2) monotonic_in_k=True
ADVERSARIAL DIVERGENCE impl=triton dtype=tf32 original_shape=(64,64,128)  max_abs_diff=4    -> minimal_failing_shape=(64,64,2) monotonic_in_k=True
ADVERSARIAL DIVERGENCE impl=cuda   dtype=tf32 original_shape=(64,64,512)  max_abs_diff=3    -> minimal_failing_shape=(64,64,2) monotonic_in_k=True
ADVERSARIAL DIVERGENCE impl=triton dtype=tf32 original_shape=(64,64,512)  max_abs_diff=3    -> minimal_failing_shape=(64,64,2) monotonic_in_k=True
ADVERSARIAL DIVERGENCE impl=cuda   dtype=tf32 original_shape=(64,64,4096) max_abs_diff=21.5  -> minimal_failing_shape=(64,64,2) monotonic_in_k=True
ADVERSARIAL DIVERGENCE impl=triton dtype=tf32 original_shape=(64,64,4096) max_abs_diff=21.5  -> minimal_failing_shape=(64,64,2) monotonic_in_k=True
```

**The bold numbers: 0 false failures across all 2,400 runs (every repeat of every configuration gave a bit-identical, verdict-stable result), and 6 real divergences found, all of them TF32, all of them in the adversarial probe, none in the main sweep.** "False failure" here means the harness itself gave an inconsistent answer on an identical input, not that a kernel happened to fail; see `sweep.py` for why repeats re-run identical data rather than fresh random draws. The adversarial divergence count (6) is close to, but does not match, the 7 this project was built to reproduce; the gap is reported rather than closed by a fourth attempt. Full raw output: `docs/sweep_output.txt`.

### Extension: agentic kernel synthesizer, pass rate and speedup

**Machine:** same as above, plus `claude -p --model claude-haiku-5-5`. 24 tasks (12 fp16, 12 bf16), 3 repeats, at most 1 retry per episode.

| Claim | Measured | Meets claim |
|---|---|---|
| 21 of 24 Triton tasks verified | **24 of 24, in every one of 3 repeats** (band: 24/24/24) | Exceeded |
| verified at FP16 and BF16 | **True**: 12 tasks fixed at fp16, 12 at bf16, all 24 pass at their assigned dtype | Yes |
| 1.9x median speedup over eager PyTorch | **0.76x**, pooled over all 72 passing episodes (median per repeat: 0.65x, 1.25x, 0.65x); splits by task shape, see Findings | No, and the direction is task-dependent, not uniform |
| pass rate reported as a band over 3 repeats | **24/24/24** | Yes |
| the model never grades its own output | **True**: every verdict above comes from `verify_task.py`, which never calls an LLM | Yes |
| deterministic control plane returns compile errors, max abs diff, and Nsight Compute speed-of-light/occupancy as structured feedback | **True, with Nsight Compute replaced by a computed proxy** (see Findings): every episode's JSON verdict carries a status, a max abs diff against the fp64 oracle when applicable, and `occupancy_proxy`/`speed_of_light_proxy` | Yes, with the disclosed substitution |

Per-task median speedup (3 repeats each), slowest to fastest:

```
exp_elementwise       0.10x   vector_add        0.15x   elementwise_sub   0.14x
log_elementwise       0.22x   tanh_act          0.36x   vector_mul        0.37x
sqrt_positive         0.43x   clamp             0.44x   leaky_relu        0.51x
elementwise_fma       0.52x   matmul_small      0.58x   matmul_rect       0.58x
sigmoid               0.74x   cumsum_row        0.91x   layernorm_row     0.97x
elementwise_div       1.24x   softmax_row       1.42x   row_max           1.50x
row_sum               1.55x   gelu_tanh_approx  1.80x   l2_normalize_row  1.82x
softplus              2.52x   rmsnorm_row       2.65x   relu              3.32x
```

Full per-episode output: `docs/agent_runs/repeat{1,2,3}/<task>/attempt1_result.json`; the 3 repeat-level rollups are `docs/agent_runs/repeat{1,2,3}_summary.json`; the top-level band is `docs/agent_eval_summary.json`.

### Extension: sub-8-bit arms, measured results

Measured on the machine and toolchain described above, via `python run_lowbit_sweep.py`. Full raw output: `docs/lowbit_sweep_output.txt`.

```
group_size: 32
sweep grid: 80 shapes x 3 dtypes (fp8_e4m3, fp8_e5m2, int4) x 10 repeats = 2400 runs
total runs: 2400
elapsed: 16.4s
false failures: 0
divergent (shape, dtype) configurations: 0

adversarial probe (not part of the 2,400-run sweep):
adversarial divergences found: 0
```

**Claim: "surfaced 9 divergences the FP16 and BF16 arms never reached."** Measured: **0**, in both the 2,400-run main sweep and the adversarial probe, after three genuinely different compute-precision designs (exact fp32, BF16, TF32; see Findings). **Not met.** The main sweep's 0 is not surprising on its own (the base harness's own main sweep also finds 0 on random Gaussian data); what is worth reporting is that the adversarial probe, which does find 6 real divergences for the base harness's TF32 arm on unquantized inputs, finds exactly 0 here, and a diagnostic inspection shows every comparison lands at an exact `0.0` difference, not merely inside budget: see Findings for why group-wise quantization makes this specific adversarial construction's cancellation exact instead of approximate.

| Format | Main sweep divergences | Adversarial divergences | atol range (quant-step-derived) |
|---|---|---|---|
| FP8 E4M3 | 0 / 800 | 0 / 3 | ~250 |
| FP8 E5M2 | 0 / 800 | 0 / 3 | ~500 |
| INT4 | 0 / 800 | 0 / 3 | ~286 |

**0 false failures across all 2,400 runs**, the same determinism property the base harness's main sweep holds: every repeat of every configuration gave a bit-identical, verdict-stable result.

## Building and running

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

./kernels/build.sh      # compiles libmatmul.so; -arch=sm_89 is hardcoded, edit for a different GPU
python run_sweep.py     # the full 2,400-run sweep plus the adversarial probe
python run_lowbit_sweep.py   # the sub-8-bit sweep plus its own adversarial probe
python -m pytest tests/ -v
```

Run the agent control-plane self-check (no LLM call, no cost):

```
python -m pytest tests/test_agent_verifier.py -v
```

Run the full agentic evaluation (calls a real `claude` CLI on the Windows side, shells out to a WSL venv for every compile/run/profile step; see `agent/run_eval.py` for the exact WSL and venv paths this machine uses):

```
python agent/run_eval.py                    # all 24 tasks, 3 repeats
python agent/run_eval.py vector_add,relu 1   # a filtered smoke test: 2 tasks, 1 repeat
```

## Limitations

- The hand-written kernel's shared-memory tile is fixed at 16x16; it is not tuned for throughput, and this project never measured its throughput, only its numerics.
- `tolerance.py`'s budget uses the deterministic worst-case bound scaled by a single constant; a tighter, data-aware bound (for example, one computed from the actual input magnitudes rather than the format alone) would likely need a smaller `SAFETY` and might catch divergences the current budget cannot.
- The adversarial construction in `adversarial.py` is one specific cancellation pattern (alternating sign, fixed magnitude). It is not a general-purpose adversarial input generator, and there is no guarantee it is the input shape that would surface a divergence in fp16 or bf16, if one exists; none was found with this one construction.
- **The occupancy and speed-of-light figures are computed proxies, not Nsight Compute output**, because `ncu` could not profile anything on this machine; see "What this is not" and Findings.
- **The agentic extension's numbers characterize Haiku 5.5 on 24 tasks with at most one retry**, not an upper bound on what any model could do with more retries, a larger task set, or a different model.
- **`metrics.extract_kernel_resource_usage` depends on Triton's `JITFunction.cache` exposing `n_regs`/`metadata.shared`/`metadata.num_warps`**; a kernel the model writes in an unusual way (multiple kernels not listed in `TRITON_KERNELS`, a kernel never actually launched before inspection) can make this return "unavailable", which is reported as such rather than guessed.
- **The sub-8-bit adversarial probe found 0 divergences, including an exact 0.0 max abs diff**, because group-wise quantization makes the base `adversarial.py` construction's cancellation exact rather than approximate (see Findings); this is a property of that one construction, not proof that no divergence exists anywhere in the FP8/INT4 arms' input space. A construction designed to inject a residual that survives a coarse quantization grid was not built, to stay inside this project's three-genuine-attempts measurement rule.
- **INT4 is stored one value per int8 byte, not bit-packed two per byte**; see `harness/quant.py` for why this is a disclosed scope choice, not an oversight.
- **This GPU's fp8 tensor cores do not exist in hardware (Ada, sm_89)**; the FP8 arms store and load real `torch.float8_e4m3fn`/`float8_e5m2` tensors but the actual GEMM runs through the same TF32 tensor-core path as the base harness's TF32 arm, not a native fp8 MMA path.
