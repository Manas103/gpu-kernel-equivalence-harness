"""Bisects a divergence down to a minimal failing shape.

M and N are held at the original failing configuration's values; only K,
the reduction length, is binary-searched, because the budget in
tolerance.py is a function of K and K alone, which is the whole premise of
this project: the divergence under investigation is a reduction-order
effect, not an M/N tiling effect, so the dimension that actually explains
it is the one bisection shrinks.

Binary search assumes divergence is monotonic in K, which is the
tolerance budget's own assumption (it grows linearly in K) but is not
guaranteed for any single random draw of data. The result is reported as
measured, including the one case where that assumption did not hold (see
docs/sweep_output.txt): bisection returned a K, but K-1 also independently
diverged, which is noted rather than silently re-run until it looks clean.
"""
from .sweep import make_inputs, run_one


def bisect_minimal_k(m, n, k_start, dtype_kind, impl, input_fn=None):
    input_fn = input_fn or make_inputs

    def diverges_at(k):
        a, b = input_fn(m, n, k, dtype_kind)
        r = run_one(m, n, k, dtype_kind, a, b)
        return not r[f"{impl}_pass"], r[f"{impl}_abs"], r[f"{impl}_rel"]

    lo, hi = 1, k_start
    best_k, best_abs, best_rel = k_start, None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        bad, abs_diff, rel_diff = diverges_at(mid)
        if bad:
            best_k, best_abs, best_rel = mid, abs_diff, rel_diff
            hi = mid - 1
        else:
            lo = mid + 1

    # Sanity check the monotonicity assumption rather than trust it blindly.
    monotonic_ok = True
    if best_k > 1:
        bad_below, _, _ = diverges_at(best_k - 1)
        monotonic_ok = not bad_below

    return {
        "m": m, "n": n, "minimal_k": best_k, "dtype": dtype_kind, "impl": impl,
        "max_abs_diff": best_abs, "max_rel_diff": best_rel,
        "monotonic_in_k": monotonic_ok,
    }
