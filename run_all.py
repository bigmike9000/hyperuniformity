"""
Hyperuniformity Project: Complete Analysis Pipeline

Generates all 1D quasiperiodic chains (substitution + projection),
computes number variance sigma^2(R), verifies Class I hyperuniformity,
and produces publication-quality figures.

Figures produced:
  1. Poisson benchmark (validates variance computation)
  2. Bounded variance 4-panel (lattice + 3 substitution chains)
  3. sigma^2/R -> 0 hyperuniformity test
  4. Projection Class I vs Class II comparison
  5. Piecewise quadratic second-derivative verification

All outputs saved to results/ directory.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    predict_chain_length, verify_eigenvalue_prediction,
)
from quasicrystal_variance import (
    compute_number_variance_1d, compute_lambda_bar,
    lattice_variance_exact, compute_second_derivative,
)
from projection_method import cut_and_project, validate_projection_spacings

# ============================================================
# Configuration
# ============================================================
SEED = 42
TARGET_N = 10_000_000   # Target chain size for substitution
PROJECTION_N = 200_000  # Target for projection method
NUM_WINDOWS = 30_000    # Windows per R value for variance
NUM_R_POINTS = 1000     # R values for standard variance curves
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Figure 1: Poisson benchmark
# ============================================================
def run_poisson_benchmark(rng):
    """Reproduce Phase 1 Poisson benchmark."""
    print("\n" + "=" * 70)
    print("  Figure 1: Poisson Variance Benchmark")
    print("=" * 70)

    N = 10_000
    L = 10_000.0
    rho = N / L
    num_realizations = 40
    num_windows = 10_000
    R_array = np.linspace(1, 50, 30)

    print(f"  {num_realizations} realizations, {num_windows} windows each...")
    t0 = time.perf_counter()

    all_variances = np.zeros((num_realizations, len(R_array)))
    for n in range(num_realizations):
        points = rng.uniform(0, L, N)
        v, _ = compute_number_variance_1d(
            points, L, R_array, num_windows=num_windows, rng=rng)
        all_variances[n] = v

    variances = np.mean(all_variances, axis=0)
    variance_stderr = np.std(all_variances, axis=0) / np.sqrt(num_realizations)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    theoretical = 2 * rho * R_array
    rel_errors = np.abs(variances - theoretical) / theoretical
    mean_err = np.mean(rel_errors)
    print(f"  Mean relative error: {mean_err:.4f}")
    print(f"  Benchmark {'PASSED' if mean_err < 0.05 else 'FAILED'}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(R_array, variances, yerr=2*variance_stderr, fmt='ko', ms=4,
                 capsize=3, label=r'Simulated $\sigma^2(R)$ ($\pm 2\sigma$)')
    ax1.plot(R_array, theoretical, 'r--', lw=2,
             label=r'Theory: $\sigma^2 = 2\rho R$')
    ax1.set_xlabel(r'Window Radius $R$', fontsize=13)
    ax1.set_ylabel(r'Number Variance $\sigma^2(R)$', fontsize=13)
    ax1.set_title('1D Poisson: Number Variance', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, ls=':', alpha=0.6)

    ax2.bar(R_array, rel_errors * 100, width=1.5, color='steelblue', alpha=0.8)
    ax2.axhline(5, color='red', ls='--', lw=1, label='5% threshold')
    ax2.set_xlabel(r'Window Radius $R$', fontsize=13)
    ax2.set_ylabel('Relative Error (%)', fontsize=13)
    ax2.set_title('Benchmark Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig1_poisson_benchmark.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return mean_err


# ============================================================
# Figure 2 & 3: Substitution chains variance analysis
# ============================================================
def run_substitution_analysis(rng):
    """Generate large substitution chains and compute variance."""
    print("\n" + "=" * 70)
    print("  Figures 2-3: Substitution Chain Variance Analysis")
    print("=" * 70)

    # --- Integer lattice reference ---
    N_lattice = 100_000
    lattice_points = np.arange(N_lattice, dtype=np.float64)
    L_lattice = float(N_lattice)
    R_lattice = np.linspace(0.05, 100, 2000)

    print("\n  Computing integer lattice reference...")
    t0 = time.perf_counter()
    var_lattice, _ = compute_number_variance_1d(
        lattice_points, L_lattice, R_lattice, num_windows=NUM_WINDOWS, rng=rng)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    lambda_bar_lattice = compute_lambda_bar(R_lattice, var_lattice)
    print(f"  Lambda_bar (lattice) = {lambda_bar_lattice:.6f} (exact: {1/6:.6f})")

    # --- Generate and analyze all three chains ---
    chain_results = {}

    for name in ['fibonacci', 'silver', 'bronze']:
        info = CHAINS[name]
        print(f"\n  --- {info['name']} ---")

        # Find iterations to reach TARGET_N
        for iters in range(5, 60):
            n_pred = predict_chain_length(name, iters)
            if n_pred > TARGET_N:
                break

        print(f"  Generating ({iters} iterations, predicted {n_pred:,} tiles)...")
        t0 = time.perf_counter()
        seq = generate_substitution_sequence(name, iters)
        points, L_domain = sequence_to_points(seq, name)
        del seq  # Free memory
        elapsed = time.perf_counter() - t0
        rho = len(points) / L_domain
        print(f"  N = {len(points):,}, L = {L_domain:.1f}, rho = {rho:.6f}, "
              f"time = {elapsed:.1f}s")

        # Variance computation
        mean_spacing = 1.0 / rho
        R_max = min(300 * mean_spacing, L_domain / 4)
        R_array = np.linspace(mean_spacing, R_max, NUM_R_POINTS)

        print(f"  Computing variance ({NUM_R_POINTS} R values, "
              f"{NUM_WINDOWS} windows)...")
        t0 = time.perf_counter()
        variances, mean_counts = compute_number_variance_1d(
            points, L_domain, R_array, num_windows=NUM_WINDOWS, rng=rng)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s")

        lambda_bar = compute_lambda_bar(R_array, variances)
        print(f"  Lambda_bar = {lambda_bar:.6f}")
        print(f"  Variance range: [{np.min(variances):.4f}, {np.max(variances):.4f}]")

        chain_results[name] = {
            'R_array': R_array,
            'variances': variances,
            'rho': rho,
            'lambda_bar': lambda_bar,
            'L_domain': L_domain,
            'N': len(points),
        }
        del points  # Free memory

    # --- Figure 2: 4-panel bounded variance ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
    for idx, cname in enumerate(['fibonacci', 'silver', 'bronze']):
        r = chain_results[cname]
        ax = axes[plot_positions[idx]]
        ax.plot(r['R_array'], r['variances'], '-', color=colors[idx], lw=0.5,
                alpha=0.8)
        ax.axhline(r['lambda_bar'], color='red', ls='--', lw=1.5,
                   label=rf"$\bar{{\Lambda}} = {r['lambda_bar']:.4f}$")
        ax.set_xlabel(r'$R$')
        ax.set_ylabel(r'$\sigma^2(R)$')
        ax.set_title(f"{CHAINS[cname]['name']} (N={r['N']:,})")
        ax.legend()
        ax.grid(True, ls=':', alpha=0.5)

    plt.suptitle('Class I Hyperuniform: Bounded Number Variance', fontsize=15,
                 y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig2_bounded_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # --- Figure 3: sigma^2/R -> 0 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, cname in enumerate(['fibonacci', 'silver', 'bronze']):
        r = chain_results[cname]
        ax.plot(r['R_array'], r['variances'] / r['R_array'],
                '-', color=colors[idx], lw=1,
                label=f"{CHAINS[cname]['name']}")
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel(r'$R$', fontsize=13)
    ax.set_ylabel(r'$\sigma^2(R) / R$', fontsize=13)
    ax.set_title(r'Hyperuniformity Test: $\sigma^2(R)/R \to 0$ for Class I',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_ylim(bottom=-0.005)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig3_hyperuniformity_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return lambda_bar_lattice, chain_results


# ============================================================
# Figure 4: Projection Class I vs Class II
# ============================================================
def run_projection_comparison(rng):
    """Demonstrate Class I vs Class II via projection method."""
    print("\n" + "=" * 70)
    print("  Figure 4: Projection Method — Class I vs Class II")
    print("=" * 70)

    mu = CHAINS['fibonacci']['metallic_mean']

    # Class I: ideal strip width
    print("\n  Generating Class I (omega = tau)...")
    t0 = time.perf_counter()
    pts_I, L_I, info_I = cut_and_project('fibonacci', PROJECTION_N, omega=mu)
    rho_I = len(pts_I) / L_I
    print(f"  N = {len(pts_I):,}, L = {L_I:.1f}, rho = {rho_I:.6f}, "
          f"time = {time.perf_counter() - t0:.1f}s")

    # Validate spacings
    unique_sp, ratio, expected = validate_projection_spacings(pts_I, 'fibonacci')
    print(f"  Spacings: {unique_sp}, ratio = {ratio:.6f} (expected {expected:.6f})")

    # Class II: non-ideal strip width
    omega_II = 0.9 * mu
    print(f"\n  Generating Class II (omega = 0.9*tau = {omega_II:.4f})...")
    t0 = time.perf_counter()
    pts_II, L_II, info_II = cut_and_project('fibonacci', PROJECTION_N, omega=omega_II)
    rho_II = len(pts_II) / L_II
    print(f"  N = {len(pts_II):,}, L = {L_II:.1f}, rho = {rho_II:.6f}, "
          f"time = {time.perf_counter() - t0:.1f}s")

    # Compute variance for both (non-periodic BCs)
    mean_sp_I = 1.0 / rho_I
    R_max_I = min(200 * mean_sp_I, L_I / 4)
    R_I = np.linspace(mean_sp_I, R_max_I, 800)

    mean_sp_II = 1.0 / rho_II
    R_max_II = min(200 * mean_sp_II, L_II / 4)
    R_II = np.linspace(mean_sp_II, R_max_II, 800)

    print("\n  Computing Class I variance (non-periodic BCs)...")
    t0 = time.perf_counter()
    var_I, _ = compute_number_variance_1d(
        pts_I, L_I, R_I, num_windows=20_000, rng=rng, periodic=False)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    print("  Computing Class II variance (non-periodic BCs)...")
    t0 = time.perf_counter()
    var_II, _ = compute_number_variance_1d(
        pts_II, L_II, R_II, num_windows=20_000, rng=rng, periodic=False)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    lambda_bar_proj = compute_lambda_bar(R_I, var_I)
    print(f"\n  Class I Lambda_bar (projection) = {lambda_bar_proj:.6f}")

    # Class II: check envelope growth
    third = len(var_II) // 3
    avg_early = np.mean(var_II[:third])
    avg_late = np.mean(var_II[-third:])
    print(f"  Class II early avg = {avg_early:.4f}, late avg = {avg_late:.4f}")
    growing = avg_late > avg_early * 1.2
    print(f"  Envelope {'GROWING' if growing else 'not clearly growing'} "
          f"(ratio = {avg_late/avg_early:.2f}x)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(R_I, var_I, 'b-', lw=0.5, alpha=0.8)
    ax1.axhline(lambda_bar_proj, color='red', ls='--', lw=1.5,
                label=rf'$\bar{{\Lambda}} = {lambda_bar_proj:.4f}$')
    ax1.set_xlabel(r'$R$')
    ax1.set_ylabel(r'$\sigma^2(R)$')
    ax1.set_title(rf'Class I: Ideal width $\omega = \tau$ (N={len(pts_I):,})')
    ax1.legend()
    ax1.grid(True, ls=':', alpha=0.5)

    ax2 = axes[1]
    ax2.plot(R_II, var_II, 'r-', lw=0.5, alpha=0.8)
    # Logarithmic envelope fit
    try:
        from scipy.optimize import curve_fit

        def log_model(R, C, b):
            return C * np.log(R) + b

        popt, _ = curve_fit(log_model, R_II, var_II, p0=[0.1, 0])
        R_fit = np.linspace(R_II[0], R_II[-1], 500)
        ax2.plot(R_fit, log_model(R_fit, *popt), 'k--', lw=1.5,
                 label=rf'$C \ln R + b$ fit ($C={popt[0]:.4f}$)')
    except Exception as e:
        print(f"  Warning: log fit failed ({e})")
    ax2.set_xlabel(r'$R$')
    ax2.set_ylabel(r'$\sigma^2(R)$')
    ax2.set_title(rf'Class II: Non-ideal $\omega = 0.9\tau$ (N={len(pts_II):,})')
    ax2.legend()
    ax2.grid(True, ls=':', alpha=0.5)

    plt.suptitle('Projection Method: Class I vs Class II Hyperuniformity',
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig4_projection_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    return lambda_bar_proj


# ============================================================
# Figure 5: Piecewise quadratic verification
# ============================================================
def run_piecewise_quadratic_verification(rng):
    """Verify piecewise quadratic structure via second derivative."""
    print("\n" + "=" * 70)
    print("  Figure 5: Piecewise Quadratic Verification")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top left: Integer lattice analytic vs numerical ---
    N_lattice = 100_000
    lattice_points = np.arange(N_lattice, dtype=np.float64)
    L_lattice = float(N_lattice)

    # High-res R over a few lattice spacings
    R_hires = np.linspace(0.01, 5, 5000)
    print("\n  Computing high-res lattice variance...")
    t0 = time.perf_counter()
    var_hires, _ = compute_number_variance_1d(
        lattice_points, L_lattice, R_hires, num_windows=30_000, rng=rng)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    var_exact = lattice_variance_exact(R_hires)

    ax = axes[0, 0]
    ax.plot(R_hires, var_hires, 'b-', lw=0.8, alpha=0.7, label='Numerical')
    ax.plot(R_hires, var_exact, 'r--', lw=1.2, label='Exact analytic')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$\sigma^2(R)$')
    ax.set_title('Integer Lattice: Numerical vs Exact')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)

    # --- Top right: Lattice second derivative ---
    d2_exact = compute_second_derivative(R_hires, var_exact)
    d2_numerical = compute_second_derivative(R_hires, var_hires)

    ax = axes[0, 1]
    ax.plot(R_hires, d2_exact, 'r-', lw=1, label='Exact (analytic)')
    ax.plot(R_hires, d2_numerical, 'b-', lw=0.5, alpha=0.6, label='Numerical')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r"$d^2\sigma^2/dR^2$")
    ax.set_title('Integer Lattice: Second Derivative (Piecewise Constant)')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_ylim(-15, 15)

    # --- Bottom: Fibonacci second derivative ---
    print("  Generating Fibonacci for high-res analysis...")
    for iters in range(5, 40):
        n_pred = predict_chain_length('fibonacci', iters)
        if n_pred > 500_000:
            break
    seq = generate_substitution_sequence('fibonacci', iters)
    fib_pts, fib_L = sequence_to_points(seq, 'fibonacci')
    fib_rho = len(fib_pts) / fib_L
    fib_spacing = 1.0 / fib_rho
    del seq

    # High-res R over ~10 mean spacings
    R_fib_hires = np.linspace(fib_spacing * 0.5, fib_spacing * 10, 5000)

    print(f"  Computing Fibonacci high-res variance (5k R points, 30k windows)...")
    t0 = time.perf_counter()
    var_fib_hires, _ = compute_number_variance_1d(
        fib_pts, fib_L, R_fib_hires, num_windows=NUM_WINDOWS, rng=rng)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    d2_fib = compute_second_derivative(R_fib_hires, var_fib_hires)

    # Bottom left: Fibonacci variance (zoomed)
    ax = axes[1, 0]
    ax.plot(R_fib_hires / fib_spacing, var_fib_hires, 'g-', lw=0.5, alpha=0.8)
    ax.set_xlabel(r'$R$ / mean spacing')
    ax.set_ylabel(r'$\sigma^2(R)$')
    ax.set_title(f'Fibonacci: High-Resolution Variance (N={len(fib_pts):,})')
    ax.grid(True, ls=':', alpha=0.5)

    # Bottom right: Fibonacci second derivative
    ax = axes[1, 1]
    ax.plot(R_fib_hires / fib_spacing, d2_fib, 'g-', lw=0.3, alpha=0.7)
    ax.set_xlabel(r'$R$ / mean spacing')
    ax.set_ylabel(r"$d^2\sigma^2/dR^2$")
    ax.set_title('Fibonacci: Second Derivative')
    ax.grid(True, ls=':', alpha=0.5)

    plt.suptitle(r'Piecewise Quadratic Verification: $d^2\sigma^2/dR^2$ '
                 r'$\approx$ Step Function', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig5_piecewise_quadratic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Summary table
# ============================================================
def print_summary(lambda_bar_lattice, chain_results, lambda_bar_proj):
    print("\n\n" + "=" * 70)
    print("  Summary: Lambda_bar Values")
    print("=" * 70)
    print(f"  {'Pattern':30s}  {'N':>12s}  {'Lambda_bar':>12s}  {'vs Lattice':>12s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"  {'Integer Lattice':30s}  {'100,000':>12s}  {lambda_bar_lattice:12.6f}  "
          f"{'(reference)':>12s}")

    for name in ['fibonacci', 'silver', 'bronze']:
        r = chain_results[name]
        ratio = r['lambda_bar'] / lambda_bar_lattice
        print(f"  {CHAINS[name]['name']:30s}  {r['N']:>12,}  {r['lambda_bar']:12.6f}  "
              f"{ratio:11.2f}x")

    print(f"\n  {'Fibonacci (projection)':30s}  {'~200k':>12s}  {lambda_bar_proj:12.6f}  "
          f"{lambda_bar_proj / lambda_bar_lattice:11.2f}x")

    # Convergence check
    if 'fibonacci' in chain_results:
        fib_sub = chain_results['fibonacci']['lambda_bar']
        agreement = abs(fib_sub - lambda_bar_proj) / fib_sub * 100
        print(f"\n  Projection vs Substitution (Fibonacci): "
              f"{agreement:.1f}% difference")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    total_t0 = time.perf_counter()
    rng = np.random.default_rng(seed=SEED)

    print("=" * 70)
    print("  Hyperuniformity Project: Complete Analysis Pipeline")
    print("=" * 70)

    ensure_results_dir()

    # Run all analyses
    poisson_err = run_poisson_benchmark(rng)
    lambda_bar_lattice, chain_results = run_substitution_analysis(rng)
    lambda_bar_proj = run_projection_comparison(rng)
    run_piecewise_quadratic_verification(rng)

    # Summary
    print_summary(lambda_bar_lattice, chain_results, lambda_bar_proj)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n  Total runtime: {total_elapsed:.1f}s")
    print(f"  All figures saved to: {RESULTS_DIR}")
    print("\nDone.")
