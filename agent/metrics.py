"""Structured feedback in place of Nsight Compute: see README,
"Why Nsight Compute is not in the feedback loop" in Findings. ncu produced
"No kernels were profiled" on every one of 3 genuine attempts on this
machine (WSL2 + GeForce RTX 4070 SUPER); these two functions compute the
same two concepts ncu's SpeedOfLight and Occupancy sections report,
directly, from Triton's own compiled-kernel metadata and measured timing,
and are labeled as a substitute everywhere they are used, not as ncu output.

Hardware constants below are the published RTX 4070 SUPER (Ada, sm_89)
specs, cited in the README: 65536 32-bit registers per SM, 1536 max
resident threads per SM (48 warps of 32), 102400 bytes max shared memory
per SM, 504 GB/s peak memory bandwidth.
"""
REGS_PER_SM = 65536
MAX_THREADS_PER_SM = 1536
MAX_WARPS_PER_SM = MAX_THREADS_PER_SM // 32
SHARED_PER_SM_BYTES = 102400
PEAK_MEM_BW_GBPS = 504.0
PEAK_FP16_TENSOR_TFLOPS = 70.0  # dense, no sparsity; published Ada whitepaper figure


def theoretical_occupancy(n_regs, shared_bytes, num_warps):
    """Theoretical occupancy from static kernel resource usage, the same
    three inputs ncu's Occupancy section's "Block Limit" rows use. This is
    NOT "achieved occupancy" (that needs hardware counters ncu could not
    read on this machine); it is the static ceiling achieved occupancy
    cannot exceed."""
    threads_per_block = num_warps * 32
    if n_regs <= 0:
        n_regs = 1
    blocks_by_regs = REGS_PER_SM // max(n_regs * threads_per_block, 1)
    blocks_by_threads = MAX_THREADS_PER_SM // threads_per_block
    blocks_by_shared = SHARED_PER_SM_BYTES // max(shared_bytes, 1) if shared_bytes > 0 else 10 ** 9
    blocks = max(min(blocks_by_regs, blocks_by_threads, blocks_by_shared), 0)
    warps = min(blocks * num_warps, MAX_WARPS_PER_SM)
    return {
        "n_regs": n_regs,
        "shared_bytes": shared_bytes,
        "num_warps": num_warps,
        "blocks_limited_by_registers": blocks_by_regs,
        "blocks_limited_by_threads": blocks_by_threads,
        "blocks_limited_by_shared_mem": blocks_by_shared,
        "theoretical_occupancy_pct": round(100.0 * warps / MAX_WARPS_PER_SM, 1),
    }


def speed_of_light_proxy(elapsed_seconds, bytes_moved, flops=None):
    """Memory-throughput-as-%-of-peak, and (when flops is given, matmul
    tasks only) compute-throughput-as-%-of-peak. Same ratio ncu's
    SpeedOfLight section reports; computed here instead of by ncu."""
    if elapsed_seconds <= 0:
        return {"memory_pct_of_peak": None, "compute_pct_of_peak": None}
    achieved_gbps = bytes_moved / elapsed_seconds / 1e9
    out = {
        "achieved_gbps": round(achieved_gbps, 1),
        "memory_pct_of_peak": round(min(100.0, 100.0 * achieved_gbps / PEAK_MEM_BW_GBPS), 1),
        "compute_pct_of_peak": None,
    }
    if flops is not None:
        achieved_tflops = flops / elapsed_seconds / 1e12
        out["achieved_tflops"] = round(achieved_tflops, 2)
        out["compute_pct_of_peak"] = round(min(100.0, 100.0 * achieved_tflops / PEAK_FP16_TENSOR_TFLOPS), 1)
    return out


def bytes_moved(tensors_in, tensor_out):
    total = sum(t.numel() * t.element_size() for t in tensors_in)
    total += tensor_out.numel() * tensor_out.element_size()
    return total


def extract_kernel_resource_usage(jit_kernel):
    """Best-effort: pull (n_regs, shared_bytes, num_warps) off a Triton
    JITFunction's compiled-kernel cache after it has been launched at
    least once. Triton 3.6's JITFunction keys this by device under
    `device_caches[device][0]`, a {signature_key: CompiledKernel} dict;
    returns None if a given kernel object does not expose this (disclosed
    as a limitation; see README)."""
    try:
        device_caches = jit_kernel.device_caches
        for entry in device_caches.values():
            cache_dict = entry[0]
            for compiled in cache_dict.values():
                n_regs = getattr(compiled, "n_regs", None)
                metadata = getattr(compiled, "metadata", None)
                shared = getattr(metadata, "shared", None) if metadata is not None else None
                num_warps = getattr(metadata, "num_warps", None) if metadata is not None else None
                if n_regs is not None and shared is not None and num_warps is not None:
                    return int(n_regs), int(shared), int(num_warps)
        return None
    except Exception:
        return None
