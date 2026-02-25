"""
Phase 1: Poisson Variance Benchmark
Validates the number-variance computation against the exact result
sigma^2(R) = 2*rho*R for a 1D Poisson point pattern.
Averages over multiple independent realizations for statistical accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def generate_poisson_1d(N, L, rng):
    """Generate N uniformly distributed points on [0, L)."""
    return rng.uniform(0, L, N)


def compute_number_variance_1d(points, L, R_array, num_windows=10000, rng=None):
    """
    Compute the number variance for a series of window radii R.
    Uses periodic boundary conditions and sorted-array binary search
    for O(num_windows * log N) performance per radius.
    """
    if rng is None:
        rng = np.random.default_rng()

    sorted_pts = np.sort(points)
    N = len(sorted_pts)
    variances = np.empty(len(R_array))
    mean_counts = np.empty(len(R_array))

    for j, R in enumerate(R_array):
        centers = rng.uniform(0, L, num_windows)
        counts = np.empty(num_windows)

        for i, c in enumerate(centers):
            lo = c - R
            hi = c + R

            if lo >= 0 and hi <= L:
                # Window doesn't wrap around
                counts[i] = (np.searchsorted(sorted_pts, hi, side='right')
                             - np.searchsorted(sorted_pts, lo, side='left'))
            elif lo < 0:
                # Window wraps past the left boundary
                counts[i] = (np.searchsorted(sorted_pts, hi, side='right')
                             + N - np.searchsorted(sorted_pts, L + lo, side='left'))
            else:
                # Window wraps past the right boundary
                counts[i] = (N - np.searchsorted(sorted_pts, lo, side='left')
                             + np.searchsorted(sorted_pts, hi - L, side='right'))

        variances[j] = np.var(counts)
        mean_counts[j] = np.mean(counts)

    return variances, mean_counts


# ==========================================
# 1. Setup Parameters
# ==========================================
rng = np.random.default_rng(seed=42)

N = 10000              # Points per realization
L = 10000.0            # Domain length
rho = N / L            # Number density
num_realizations = 40  # Independent Poisson patterns to average over
num_windows = 10000    # Window samples per realization per R

R_array = np.linspace(1, 50, 30)

# ==========================================
# 2. Run the Benchmark (multi-realization)
# ==========================================
print(f"Running Poisson benchmark: {num_realizations} realizations, "
      f"{num_windows} windows each")
t0 = time.perf_counter()

all_variances = np.zeros((num_realizations, len(R_array)))

for n in range(num_realizations):
    points = generate_poisson_1d(N, L, rng)
    v, _ = compute_number_variance_1d(points, L, R_array,
                                      num_windows=num_windows, rng=rng)
    all_variances[n] = v
    print(f"  Realization {n+1}/{num_realizations} done")

# Ensemble-averaged variance and standard error
variances = np.mean(all_variances, axis=0)
variance_stderr = np.std(all_variances, axis=0) / np.sqrt(num_realizations)

elapsed = time.perf_counter() - t0
print(f"Total time: {elapsed:.1f}s")

# Theoretical prediction: sigma^2(R) = 2 * rho * R
theoretical_variance = 2 * rho * R_array

# ==========================================
# 3. Quantitative Validation
# ==========================================
relative_errors = np.abs(variances - theoretical_variance) / theoretical_variance
print(f"\nRelative error vs theory (2*rho*R):")
print(f"  Mean:  {np.mean(relative_errors):.4f}")
print(f"  Max:   {np.max(relative_errors):.4f}")

passed = np.mean(relative_errors) < 0.05
print(f"\nBenchmark {'PASSED' if passed else 'FAILED'} "
      f"(mean relative error {'<' if passed else '>'} 5%)")

# ==========================================
# 4. Plot the Results
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: variance vs R
ax1.errorbar(R_array, variances, yerr=2*variance_stderr, fmt='ko', ms=4,
             capsize=3, label=r'Simulated $\sigma^2(R)$ ($\pm 2\sigma$)')
ax1.plot(R_array, theoretical_variance, 'r--', lw=2,
         label=r'Theory: $\sigma^2 = 2\rho R$')
ax1.set_xlabel(r'Window Radius $R$', fontsize=13)
ax1.set_ylabel(r'Number Variance $\sigma^2(R)$', fontsize=13)
ax1.set_title('1D Poisson: Number Variance', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, ls=':', alpha=0.6)

# Right panel: relative error vs R
ax2.bar(R_array, relative_errors * 100, width=1.5, color='steelblue', alpha=0.8)
ax2.axhline(5, color='red', ls='--', lw=1, label='5% threshold')
ax2.set_xlabel(r'Window Radius $R$', fontsize=13)
ax2.set_ylabel('Relative Error (%)', fontsize=13)
ax2.set_title('Benchmark Accuracy', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, ls=':', alpha=0.6)

plt.tight_layout()
plt.savefig('poisson_variance_benchmark.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to poisson_variance_benchmark.png")
plt.show()
