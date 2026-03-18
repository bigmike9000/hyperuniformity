"""
Numerically measure the hyperuniformity exponent alpha for URL patterns
by fitting S(k) ~ B * |k|^alpha in the small-k regime.

Strategy:
  - Stay well below the first Bragg peak (k << 2*pi)
  - Log-bin S(k) before fitting to reduce noise
  - Average over many realizations
  - Use large N so the small-k signal is resolved
"""
import numpy as np
from scipy.stats import linregress
from disordered_patterns import generate_url
from two_phase_media import compute_structure_factor

N = 1_000_000
N_REALIZATIONS = 20
A_VALUES = [0.1, 0.5, 1.0]

# Fit range: well below first Bragg peak at k=2*pi.
# Small a -> S(k) very small at low k, so we need a wider range to get signal.
FIT_RANGES = {
    0.1: (0.05, 1.5),   # S is tiny; use wider range but still << 2*pi
    0.5: (0.02, 0.8),
    1.0: (0.02, 0.5),
}
N_BINS = 40  # log-spaced bins in fit range

rng = np.random.default_rng(42)

print(f"URL alpha measurement: N={N:,}, {N_REALIZATIONS} realizations per a")
print(f"Log-binned S(k) fit, {N_BINS} bins\n")

for a in A_VALUES:
    k_min, k_max = FIT_RANGES[a]

    # Accumulate S(k) across realizations for ensemble average
    k_ref, Sk_sum = None, None

    for i in range(N_REALIZATIONS):
        points, L = generate_url(N, a, rng=rng)
        k, Sk = compute_structure_factor(points, L)
        if k_ref is None:
            k_ref = k
            Sk_sum = Sk.copy()
        else:
            Sk_sum += Sk

    Sk_mean = Sk_sum / N_REALIZATIONS

    # Log-bin the ensemble-averaged S(k) in the fit range
    mask = (k_ref > k_min) & (k_ref < k_max) & (Sk_mean > 0)
    k_fit  = k_ref[mask]
    Sk_fit = Sk_mean[mask]

    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), N_BINS + 1)
    k_binned, S_binned = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (k_fit >= lo) & (k_fit < hi)
        if in_bin.sum() > 5:
            k_binned.append(np.mean(k_fit[in_bin]))
            S_binned.append(np.mean(Sk_fit[in_bin]))

    k_binned = np.array(k_binned)
    S_binned = np.array(S_binned)

    slope, intercept, r, _, se = linregress(np.log(k_binned), np.log(S_binned))

    print(f"  a = {a}:  fit range k in [{k_min}, {k_max}]")
    print(f"          alpha = {slope:.4f} +/- {se:.4f}  (R^2 = {r**2:.5f})")
    print(f"          theory: 2.000")
    print()
