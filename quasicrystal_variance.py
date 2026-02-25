"""
Phase 3: Number Variance Analysis of 1D Quasiperiodic Chains
Computes sigma^2(R) for Fibonacci, silver, and bronze ratio chains,
verifies Class I hyperuniform behavior (bounded variance), and
extracts the surface-area coefficient Lambda_bar.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points


def compute_number_variance_1d(points, L, R_array, num_windows=20000, rng=None,
                               periodic=True):
    """
    Compute sigma^2(R) using sorted-array binary search.
    O(num_windows * log N) per radius value.

    Parameters
    ----------
    points : ndarray
        1D point positions.
    L : float
        Domain length.
    R_array : ndarray
        Array of window half-widths.
    num_windows : int
        Number of random window centers per R value.
    rng : numpy.random.Generator, optional
        Random number generator.
    periodic : bool
        If True, use periodic boundary conditions (wrap-around counting).
        If False, restrict window centers to [R, L-R] so windows stay within domain.
    """
    if rng is None:
        rng = np.random.default_rng()

    sorted_pts = np.sort(points)
    N = len(sorted_pts)
    variances = np.empty(len(R_array))
    mean_counts = np.empty(len(R_array))

    for j, R in enumerate(R_array):
        if periodic:
            centers = rng.uniform(0, L, num_windows)
        else:
            centers = rng.uniform(R, L - R, num_windows)

        lo = centers - R
        hi = centers + R

        # Vectorized binary search for all windows at once
        idx_hi = np.searchsorted(sorted_pts, hi, side='right')
        idx_lo = np.searchsorted(sorted_pts, lo, side='left')
        counts = (idx_hi - idx_lo).astype(np.float64)

        if periodic:
            # Fix wrap-left: window extends past x=0
            wrap_left = lo < 0
            if np.any(wrap_left):
                wrapped_lo = L + lo[wrap_left]
                counts[wrap_left] = (
                    np.searchsorted(sorted_pts, hi[wrap_left], side='right')
                    + N - np.searchsorted(sorted_pts, wrapped_lo, side='left')
                ).astype(np.float64)

            # Fix wrap-right: window extends past x=L
            wrap_right = hi > L
            if np.any(wrap_right):
                wrapped_hi = hi[wrap_right] - L
                counts[wrap_right] = (
                    N - np.searchsorted(sorted_pts, lo[wrap_right], side='left')
                    + np.searchsorted(sorted_pts, wrapped_hi, side='right')
                ).astype(np.float64)

        variances[j] = np.var(counts)
        mean_counts[j] = np.mean(counts)

    return variances, mean_counts


def lattice_variance_exact(R_array):
    """
    Exact analytic number variance for the 1D integer lattice (spacing = 1).

    For half-width R, the window [x-R, x+R] has length 2R.
    sigma^2(R) = 2f(1-2f)   for f = {R} < 0.5
    sigma^2(R) = 2(1-f)(2f-1)  for f = {R} >= 0.5

    Period-average: Lambda_bar = 1/6 exactly.
    """
    R = np.asarray(R_array, dtype=np.float64)
    f = R % 1.0
    var = np.where(f < 0.5, 2 * f * (1 - 2 * f), 2 * (1 - f) * (2 * f - 1))
    return var


def compute_second_derivative(R_array, variances):
    """
    Compute the numerical second derivative d^2 sigma^2 / dR^2.

    For piecewise quadratic sigma^2(R), this should be piecewise constant
    (a step function), providing verification of the analytic structure.

    Uses central finite differences with the given R sampling.
    """
    R = np.asarray(R_array)
    var = np.asarray(variances)
    # Use numpy gradient twice for second derivative
    dvar = np.gradient(var, R)
    d2var = np.gradient(dvar, R)
    return d2var


def compute_lambda_bar(R_array, variances):
    """
    Compute Lambda_bar = <sigma^2(R)> averaged over the asymptotic regime.

    For 1D Class I hyperuniform systems, sigma^2(R) is bounded and oscillating.
    Lambda_bar is the long-R average of this bounded function.

    Reference: 1D integer lattice has Lambda_bar = 1/6 (exact).
    The analytic sigma^2(R) for Z is 2*frac(R)*(1-2*frac(R)) for frac(R)<0.5,
    whose period-average is exactly 1/6.

    We use the upper portion of the R range where oscillations are well-established,
    and ensure enough oscillation periods are sampled for accurate averaging.
    """
    # Use the last 2/3 of the R range for averaging
    start = len(R_array) // 3
    return np.mean(variances[start:])


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    rng = np.random.default_rng(seed=123)

    print("=" * 70)
    print("  Phase 3: Number Variance of Quasiperiodic Chains")
    print("=" * 70)

    # -------------------------------------------------------
    # Reference: Integer lattice (perfect crystal, Lambda_bar = 1/6)
    # -------------------------------------------------------
    N_lattice = 100000
    lattice_points = np.arange(N_lattice, dtype=np.float64)
    L_lattice = float(N_lattice)

    # Dense R sampling: ~20 samples per lattice spacing over 100 spacings
    R_lattice = np.linspace(0.05, 100, 2000)

    print("\nComputing integer lattice reference...")
    t0 = time.perf_counter()
    var_lattice, mean_lattice = compute_number_variance_1d(
        lattice_points, L_lattice, R_lattice, num_windows=30000, rng=rng)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    lambda_bar_lattice = compute_lambda_bar(R_lattice, var_lattice)
    print(f"  Lambda_bar (integer lattice) = {lambda_bar_lattice:.6f}  "
          f"(expected 1/6 = {1/6:.6f})")

    # -------------------------------------------------------
    # Quasiperiodic chains
    # -------------------------------------------------------
    chain_results = {}

    for name in ['fibonacci', 'silver', 'bronze']:
        info = CHAINS[name]
        print(f"\n--- {info['name']} ---")

        # Generate chain (target ~500k tiles)
        for iters in range(5, 40):
            M = info['matrix']
            vec = np.array([0, 1], dtype=np.int64)
            Mn = np.linalg.matrix_power(M, iters)
            n_pred = int(np.sum(Mn @ vec))
            if n_pred > 500_000:
                break

        print(f"  Generating ({iters} iterations)...")
        seq = generate_substitution_sequence(name, iters)
        points, L_domain = sequence_to_points(seq, name)
        rho = len(points) / L_domain
        print(f"  N = {len(points):,}, L = {L_domain:.1f}, rho = {rho:.6f}")

        # Dense R sampling: cover many oscillation periods
        # Mean spacing determines the fundamental oscillation scale
        mean_spacing = 1.0 / rho
        R_max = min(300 * mean_spacing, L_domain / 4)
        R_array = np.linspace(mean_spacing, R_max, 1000)

        print(f"  Computing variance (R from {R_array[0]:.2f} to {R_array[-1]:.1f}, "
              f"1000 R values)...")
        t0 = time.perf_counter()
        variances, mean_counts = compute_number_variance_1d(
            points, L_domain, R_array, num_windows=30000, rng=rng)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s")

        lambda_bar = compute_lambda_bar(R_array, variances)
        print(f"  Lambda_bar = {lambda_bar:.6f}")
        print(f"  Variance range: [{np.min(variances):.4f}, {np.max(variances):.4f}]")

        chain_results[name] = {
            'R_array': R_array,
            'variances': variances,
            'mean_counts': mean_counts,
            'rho': rho,
            'lambda_bar': lambda_bar,
            'L_domain': L_domain,
            'N': len(points),
        }

    # -------------------------------------------------------
    # Summary table
    # -------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  Summary: Lambda_bar Values")
    print("=" * 70)
    print(f"  {'Pattern':30s}  {'Lambda_bar':>12s}  {'vs Lattice':>12s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    print(f"  {'Integer Lattice':30s}  {lambda_bar_lattice:12.6f}  "
          f"{'(reference)':>12s}")
    for name in ['fibonacci', 'silver', 'bronze']:
        r = chain_results[name]
        ratio = r['lambda_bar'] / lambda_bar_lattice
        print(f"  {CHAINS[name]['name']:30s}  {r['lambda_bar']:12.6f}  "
              f"{ratio:11.2f}x")

    # -------------------------------------------------------
    # Plot 1: Variance curves (4-panel)
    # -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Integer lattice
    ax = axes[0, 0]
    ax.plot(R_lattice, var_lattice, 'b-', lw=0.5, alpha=0.8)
    ax.axhline(lambda_bar_lattice, color='red', ls='--', lw=1.5,
               label=rf'$\bar{{\Lambda}} = {lambda_bar_lattice:.4f}$ (theory: 1/6)')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$\sigma^2(R)$')
    ax.set_title('Integer Lattice (Reference)')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)

    plot_positions = [(0, 1), (1, 0), (1, 1)]
    colors = ['#2ca02c', '#9467bd', '#d62728']
    for idx, name in enumerate(['fibonacci', 'silver', 'bronze']):
        r = chain_results[name]
        ax = axes[plot_positions[idx]]
        ax.plot(r['R_array'], r['variances'], '-', color=colors[idx], lw=0.5,
                alpha=0.8)
        ax.axhline(r['lambda_bar'], color='red', ls='--', lw=1.5,
                   label=rf"$\bar{{\Lambda}} = {r['lambda_bar']:.4f}$")
        ax.set_xlabel(r'$R$')
        ax.set_ylabel(r'$\sigma^2(R)$')
        ax.set_title(f"{CHAINS[name]['name']} (N={r['N']:,})")
        ax.legend()
        ax.grid(True, ls=':', alpha=0.5)

    plt.suptitle('Class I Hyperuniform: Bounded Number Variance', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('quasicrystal_variance.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to quasicrystal_variance.png")

    # -------------------------------------------------------
    # Plot 2: sigma^2/R vs R (hyperuniformity test)
    # -------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(['fibonacci', 'silver', 'bronze']):
        r = chain_results[name]
        ax2.plot(r['R_array'], r['variances'] / r['R_array'],
                 '-', color=colors[idx], lw=1,
                 label=f"{CHAINS[name]['name']}")
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_xlabel(r'$R$', fontsize=13)
    ax2.set_ylabel(r'$\sigma^2(R) / R$', fontsize=13)
    ax2.set_title(r'Hyperuniformity Test: $\sigma^2(R)/R \to 0$ for Class I',
                  fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, ls=':', alpha=0.5)
    ax2.set_ylim(bottom=-0.005)
    plt.tight_layout()
    plt.savefig('hyperuniformity_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved to hyperuniformity_test.png")
    plt.show()
