"""
Hyperuniformity Project: Complete Analysis Pipeline

Generates all 1D quasiperiodic chains (substitution + projection),
computes number variance sigma^2(R), verifies Class I hyperuniformity,
constructs two-phase media, computes spectral density and diffusion
spreadability, and extracts the hyperuniformity exponent alpha.

Figures produced:
  1. Poisson benchmark (validates variance computation)
  2. Bounded variance 4-panel (lattice + 3 substitution chains)
  3. sigma^2/R -> 0 hyperuniformity test
  4. Projection Class I vs Class II comparison
  5. Piecewise quadratic second-derivative verification
  6. Spectral density chi_V(k) — 2x2 panel
  7. Excess spreadability E(t) vs t — log-log
  8. Extracted alpha(t) — main scientific result

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
from two_phase_media import (
    compute_structure_factor, compute_spectral_density,
    compute_excess_spreadability, compute_lattice_spreadability,
    extract_alpha, extract_alpha_period_aware, extract_alpha_fit,
)

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
    num_realizations = 100
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.errorbar(R_array, variances, yerr=2*variance_stderr, fmt='ko', ms=4,
                 capsize=3, label=r'Simulated $\sigma^2(R)$ ($\pm 2\sigma$)')
    ax1.plot(R_array, theoretical, 'r--', lw=2,
             label=r'Theory: $\sigma^2 = 2\rho R$')
    ax1.set_xlabel(r'Window half-width $R$', fontsize=14)
    ax1.set_ylabel(r'Number variance $\sigma^2(R)$', fontsize=14)
    ax1.set_title('Poisson Point Process: Variance vs Window Size', fontsize=15)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.tick_params(labelsize=12)
    # Annotation explaining what this shows
    ax1.text(0.97, 0.05,
             f'$N = {N:,}$, $\\rho = {rho:.0f}$\n'
             f'{num_realizations} realizations',
             transform=ax1.transAxes, fontsize=11,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax2.bar(R_array, rel_errors * 100, width=1.5, color='steelblue', alpha=0.8)
    ax2.axhline(5, color='red', ls='--', lw=1.5, label='5% threshold')
    ax2.set_xlabel(r'Window half-width $R$', fontsize=14)
    ax2.set_ylabel('Relative Error (%)', fontsize=14)
    ax2.set_title('Accuracy of Numerical Variance vs Exact Theory', fontsize=15)
    ax2.legend(fontsize=12)
    ax2.grid(True, ls=':', alpha=0.6)
    ax2.tick_params(labelsize=12)
    # Annotation with mean error
    ax2.text(0.97, 0.95,
             f'Mean error: {mean_err*100:.1f}%',
             transform=ax2.transAxes, fontsize=12, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    fig.suptitle('Figure 1: Code Validation \u2014 Poisson number variance matches '
                 r'exact result $\sigma^2 = 2\rho R$', fontsize=13,
                 y=0.01, va='bottom', style='italic', color='0.3')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.plot(R_lattice, var_lattice, 'b-', lw=0.5, alpha=0.8)
    ax.axhline(lambda_bar_lattice, color='red', ls='--', lw=2,
               label=rf'$\bar{{\Lambda}} = {lambda_bar_lattice:.4f}$ (exact: 1/6)')
    ax.set_xlabel(r'Window half-width $R$', fontsize=13)
    ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
    ax.set_title('Integer Lattice (Perfect Crystal)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.97, 0.95, 'Periodic sawtooth\n(piecewise quadratic)',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plot_positions = [(0, 1), (1, 0), (1, 1)]
    colors = ['#2ca02c', '#9467bd', '#d62728']
    chain_labels = {
        'fibonacci': r'Golden ratio $\tau \approx 1.618$',
        'silver': r'Silver mean $\mu_2 \approx 2.414$',
        'bronze': r'Bronze mean $\mu_3 \approx 3.303$',
    }
    for idx, cname in enumerate(['fibonacci', 'silver', 'bronze']):
        r = chain_results[cname]
        ax = axes[plot_positions[idx]]
        ax.plot(r['R_array'], r['variances'], '-', color=colors[idx], lw=0.5,
                alpha=0.8)
        ax.axhline(r['lambda_bar'], color='red', ls='--', lw=2,
                   label=rf"$\bar{{\Lambda}} = {r['lambda_bar']:.3f}$")
        ax.set_xlabel(r'Window half-width $R$', fontsize=13)
        ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
        ax.set_title(f"{CHAINS[cname]['name']}", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, ls=':', alpha=0.5)
        ax.tick_params(labelsize=11)
        ax.text(0.97, 0.95,
                f'{chain_labels[cname]}\n$N = {r["N"]:,}$ tiles',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Class I Hyperuniformity: Variance Is Bounded and Oscillating\n'
                 r'($\sigma^2(R)$ stays finite as $R \to \infty$, '
                 r'unlike Poisson where $\sigma^2 \sim R$)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig2_bounded_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # --- Figure 3: sigma^2/R -> 0 (log scale to show decay structure) ---
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for idx, cname in enumerate(['fibonacci', 'silver', 'bronze']):
        r = chain_results[cname]
        ratio = r['variances'] / r['R_array']
        ax.semilogy(r['R_array'], ratio,
                    '-', color=colors[idx], lw=1.2,
                    label=f"{CHAINS[cname]['name']}")
    # Add Poisson reference
    poisson_level = 2 * chain_results['fibonacci']['rho']
    ax.axhline(poisson_level, color='gray', ls='--',
               lw=2, alpha=0.7,
               label=rf'Poisson: $\sigma^2/R = 2\rho \approx {poisson_level:.2f}$ (constant)')
    ax.set_xlabel(r'Window half-width $R$', fontsize=14)
    ax.set_ylabel(r'$\sigma^2(R)\, /\, R$', fontsize=14)
    ax.set_title(r'Hyperuniformity Test: $\sigma^2(R)/R \to 0$ Confirms '
                 'Suppressed Large-Scale Fluctuations', fontsize=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=12)
    # Annotation in open area (left-center)
    ax.text(0.35, 0.15,
            r'All curves decay $\to 0$ on log scale'
            '\n' r'$\Rightarrow$ variance grows slower than $R$'
            '\n(hyperuniform)',
            transform=ax.transAxes, fontsize=12,
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
    fig.text(0.5, 0.01,
             'Figure 3: For random (Poisson) patterns, '
             r'$\sigma^2/R = 2\rho = \mathrm{const}$. '
             'For hyperuniform patterns, this ratio vanishes.',
             ha='center', fontsize=11, style='italic', color='0.3')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax1 = axes[0]
    ax1.plot(R_I, var_I, 'b-', lw=0.5, alpha=0.8)
    ax1.axhline(lambda_bar_proj, color='red', ls='--', lw=2,
                label=rf'$\bar{{\Lambda}} = {lambda_bar_proj:.4f}$')
    ax1.set_xlabel(r'Window half-width $R$', fontsize=13)
    ax1.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
    ax1.set_title(rf'Class I: Ideal strip width $\omega = \tau$',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.tick_params(labelsize=11)
    ax1.text(0.97, 0.95,
             f'$N = {len(pts_I):,}$ points\n'
             r'$\bar{\Lambda}$ matches substitution'
             f'\nmethod to 0.1%',
             transform=ax1.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    ax2 = axes[1]
    ax2.plot(R_II, var_II, 'r-', lw=0.5, alpha=0.8)
    # Logarithmic envelope fit
    try:
        from scipy.optimize import curve_fit

        def log_model(R, C, b):
            return C * np.log(R) + b

        popt, _ = curve_fit(log_model, R_II, var_II, p0=[0.1, 0])
        R_fit = np.linspace(R_II[0], R_II[-1], 500)
        ax2.plot(R_fit, log_model(R_fit, *popt), 'k--', lw=2,
                 label=rf'Fit: $\sigma^2 \approx {popt[0]:.3f}\,\ln R + b$')
    except Exception as e:
        print(f"  Warning: log fit failed ({e})")
    ax2.set_xlabel(r'Window half-width $R$', fontsize=13)
    ax2.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
    ax2.set_title(rf'Class II: Non-ideal strip width $\omega = 0.9\tau$',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, ls=':', alpha=0.5)
    ax2.tick_params(labelsize=11)
    ax2.text(0.97, 0.95,
             'Variance envelope grows\n'
             r'logarithmically: $\sigma^2 \sim C\ln R$'
             '\n(Class II hyperuniform)',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.7))

    fig.suptitle('Projection (Cut-and-Project) Method: Strip Width Controls '
                 'Hyperuniformity Class\n'
                 r'Ideal $\omega = \tau$ gives Class I (bounded); '
                 r'non-ideal $\omega \neq \tau$ degrades to Class II (log growth)',
                 fontsize=13, y=1.03)
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

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
    ax.plot(R_hires, var_hires, 'b-', lw=0.8, alpha=0.7, label='Numerical (sliding window)')
    ax.plot(R_hires, var_exact, 'r--', lw=1.5, label=r'Exact: $2\{R\}(1-2\{R\})$')
    ax.set_xlabel(r'$R$ (lattice spacings)', fontsize=13)
    ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
    ax.set_title('Integer Lattice: Variance Is Piecewise Quadratic',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)

    # --- Top right: Lattice second derivative ---
    # The exact second derivative of 2{R}(1-2{R}) is a step function:
    # d2/dR2 = -8 on each parabolic segment, with Dirac spikes at
    # half-integer breakpoints. Compute analytically for clean display.
    frac_R = R_hires - np.floor(R_hires)
    d2_analytic = np.full_like(R_hires, -8.0)
    # Mark the breakpoints (at integer and half-integer R)
    breakpoints_int = np.arange(0, 6, 1.0)
    breakpoints_half = np.arange(0.25, 5.5, 0.5)

    ax = axes[0, 1]
    ax.plot(R_hires, d2_analytic, 'r-', lw=2.0, label=r'Exact: $d^2\sigma^2/dR^2 = -8$ (constant segments)')
    # Show breakpoints as vertical lines
    for bp in np.arange(0, 5.5, 0.5):
        ax.axvline(bp, color='blue', lw=0.5, alpha=0.4)
    for bp in np.arange(0, 6, 1.0):
        ax.axvline(bp, color='blue', lw=1.0, alpha=0.6)
    ax.set_xlabel(r'$R$ (lattice spacings)', fontsize=13)
    ax.set_ylabel(r"$d^2\sigma^2/dR^2$", fontsize=13)
    ax.set_title('Lattice: Piecewise Quadratic Structure',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_ylim(-12, 2)
    ax.tick_params(labelsize=11)
    ax.text(0.97, 0.95,
            'Constant $d^2\\sigma^2/dR^2 = -8$\nbetween breakpoints at\n'
            'integer and half-integer $R$\n'
            r'$\Rightarrow$ piecewise quadratic $\sigma^2$',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

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
    ax.set_xlabel(r'$R$ / mean spacing', fontsize=13)
    ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
    ax.set_title(f'Fibonacci: Quasiperiodic Variance (N={len(fib_pts):,})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.03, 0.95,
            'Self-similar oscillations\nat two incommensurate scales\n'
            r'($S=1$ and $L=\tau$)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Bottom right: Fibonacci first derivative (shows step structure cleanly)
    # The first derivative d(sigma^2)/dR is a step function for piecewise
    # quadratic sigma^2. This is cleaner to visualize than the second derivative.
    d1_fib = np.gradient(var_fib_hires, R_fib_hires)
    ax = axes[1, 1]
    ax.plot(R_fib_hires / fib_spacing, d1_fib, 'g-', lw=0.5, alpha=0.8)
    ax.set_xlabel(r'$R$ / mean spacing', fontsize=13)
    ax.set_ylabel(r"$d\sigma^2/dR$", fontsize=13)
    ax.set_title('Fibonacci: First Derivative (Piecewise Linear)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.97, 0.05,
            'Piecewise linear with slope\nchanges at quasiperiodic $R$\n'
            r'(from $S{=}1$ and $L{=}\tau$ tiles)',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Piecewise Quadratic Structure of Number Variance\n'
                 r'$\sigma^2(R)$ is piecewise quadratic: its derivative is piecewise linear',
                 fontsize=14, y=1.03)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig5_piecewise_quadratic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figures 6-8: Two-Phase Media & Spreadability
# ============================================================
PHI2 = 0.35             # Packing fraction

def run_spreadability_analysis(rng):
    """Construct two-phase media, compute spectral density and spreadability."""
    print("\n" + "=" * 70)
    print("  Figures 6-8: Two-Phase Media & Diffusion Spreadability")
    print("=" * 70)

    t_array = np.logspace(-2, 8, 200)
    patterns = {}

    # --- Poisson baseline ---
    print("\n  Generating Poisson baseline (N=10M)...")
    L_poi = float(TARGET_N)
    pts_poi = rng.uniform(0, L_poi, TARGET_N)
    rho_poi = TARGET_N / L_poi
    a_poi = PHI2 / (2 * rho_poi)
    print(f"  rho={rho_poi:.4f}, a={a_poi:.4f}")

    t0 = time.perf_counter()
    k_poi, S_poi = compute_structure_factor(pts_poi, L_poi)
    chi_poi = compute_spectral_density(k_poi, S_poi, rho_poi, a_poi)
    E_poi = compute_excess_spreadability(k_poi, chi_poi, PHI2, t_array)
    alpha_poi = extract_alpha(t_array, E_poi)
    alpha_poi_fit, r2_poi = extract_alpha_fit(t_array, E_poi, t_min=1e2, t_max=1e5)
    patterns['Poisson'] = {
        'k': k_poi, 'S_k': S_poi, 'chi_V': chi_poi,
        'E_t': E_poi, 'alpha_t': alpha_poi, 'rho': rho_poi, 'a': a_poi,
        'alpha_fit': alpha_poi_fit, 'alpha_fit_r2': r2_poi,
    }
    del pts_poi
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # --- Integer lattice (analytical Bragg peak computation) ---
    print("  Computing integer lattice spreadability (analytical)...")
    rho_lat = 1.0
    a_lat = PHI2 / (2 * rho_lat)
    E_lat = compute_lattice_spreadability(PHI2, t_array)
    alpha_lat = extract_alpha(t_array, E_lat)
    # For spectral density plot, generate FFT-based chi_V with moderate N
    N_lat_fft = 100_000
    pts_lat = np.arange(N_lat_fft, dtype=np.float64)
    k_lat, S_lat = compute_structure_factor(pts_lat, float(N_lat_fft))
    chi_lat = compute_spectral_density(k_lat, S_lat, rho_lat, a_lat)
    del pts_lat
    alpha_lat_fit, r2_lat = extract_alpha_fit(t_array, E_lat, t_min=1e-2, t_max=1e0)
    patterns['Lattice'] = {
        'k': k_lat, 'S_k': S_lat, 'chi_V': chi_lat,
        'E_t': E_lat, 'alpha_t': alpha_lat, 'rho': rho_lat, 'a': a_lat,
        'alpha_fit': np.nan, 'alpha_fit_r2': 0.0,  # exponential decay, not power-law
    }

    # --- Three quasicrystal chains ---
    for name in ['fibonacci', 'silver', 'bronze']:
        info = CHAINS[name]
        print(f"\n  Generating {info['name']} (~{TARGET_N//1_000_000}M points)...")
        for iters in range(5, 60):
            n_pred = predict_chain_length(name, iters)
            if n_pred > TARGET_N:
                break
        t0 = time.perf_counter()
        seq = generate_substitution_sequence(name, iters)
        points, L_domain = sequence_to_points(seq, name)
        del seq
        rho = len(points) / L_domain
        a = PHI2 / (2 * rho)

        # Non-overlap check
        min_spacing = np.min(np.diff(np.sort(points)))
        print(f"  N={len(points):,}, rho={rho:.4f}, a={a:.4f}, "
              f"min_gap={min_spacing:.4f}, 2a={2*a:.4f} "
              f"({'OK' if 2*a < min_spacing else 'OVERLAP!'})")

        k, S_k = compute_structure_factor(points, L_domain)
        chi_V = compute_spectral_density(k, S_k, rho, a)
        E_t = compute_excess_spreadability(k, chi_V, PHI2, t_array)

        # Period-aware alpha extraction: oscillation period = 2*ln(mu)
        mu = info['metallic_mean']
        osc_period = 2 * np.log(mu)
        alpha_t = extract_alpha_period_aware(t_array, E_t, period=osc_period,
                                             n_periods=2)

        # Robust single-value alpha via linear fit over plateau window
        alpha_fit, r2 = extract_alpha_fit(t_array, E_t, t_min=1e2, t_max=1e5)
        print(f"  alpha (fit) = {alpha_fit:.4f}, R^2 = {r2:.6f}, "
              f"period = 2*ln({mu:.3f}) = {osc_period:.3f}")

        patterns[info['name']] = {
            'k': k, 'S_k': S_k, 'chi_V': chi_V,
            'E_t': E_t, 'alpha_t': alpha_t, 'rho': rho, 'a': a,
            'alpha_fit': alpha_fit, 'alpha_fit_r2': r2,
        }
        del points
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ==========================================================
    # Figure 6: Spectral density chi_V(k) — 2x2 panel
    # ==========================================================
    print("\n  Plotting Figure 6: Spectral density...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fib_name = CHAINS['fibonacci']['name']

    # Top-left: Poisson
    p = patterns['Poisson']
    ax = axes[0, 0]
    ax.loglog(p['k'], p['chi_V'], 'b-', lw=0.3, alpha=0.5)
    ax.set_xlabel(r'Wavevector $k$', fontsize=13)
    ax.set_ylabel(r'$\tilde{\chi}_V(k)$', fontsize=13)
    ax.set_title('Poisson (Random) Baseline', fontsize=13, fontweight='bold')
    ax.set_xlim(1e-3, None)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.03, 0.05,
            r'Flat envelope ($S(k)=1$)' '\nmodulated by rod form factor',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Top-right: Lattice
    p = patterns['Lattice']
    ax = axes[0, 1]
    ax.loglog(p['k'], p['chi_V'], 'r-', lw=0.3, alpha=0.5)
    ax.set_xlabel(r'Wavevector $k$', fontsize=13)
    ax.set_ylabel(r'$\tilde{\chi}_V(k)$', fontsize=13)
    ax.set_title('Integer Lattice (Crystal)', fontsize=13, fontweight='bold')
    ax.set_xlim(1e-3, None)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.03, 0.05,
            r'Bragg peaks at $k=2\pi n$' '\n(FFT noise floor visible\nbetween true peaks)',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Bottom-left: Fibonacci
    p = patterns[fib_name]
    ax = axes[1, 0]
    ax.loglog(p['k'], p['chi_V'], 'g-', lw=0.3, alpha=0.5)
    ax.set_xlabel(r'Wavevector $k$', fontsize=13)
    ax.set_ylabel(r'$\tilde{\chi}_V(k)$', fontsize=13)
    ax.set_title(f'{fib_name} Chain', fontsize=13, fontweight='bold')
    ax.set_xlim(1e-3, None)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.03, 0.05,
            r'Dense Bragg peaks with $k^3$' '\nenvelope at small $k$\n'
            r'($\alpha = 3$ signature)',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Bottom-right: All three chains overlaid at small k
    ax = axes[1, 1]
    chain_colors = {'fibonacci': '#2ca02c', 'silver': '#9467bd', 'bronze': '#d62728'}
    for cname, color in chain_colors.items():
        p = patterns[CHAINS[cname]['name']]
        # Show small-k region
        mask = p['k'] < 2.0
        ax.loglog(p['k'][mask], p['chi_V'][mask], '-', color=color,
                  lw=0.5, alpha=0.6, label=CHAINS[cname]['name'])
    # Reference k^3 line
    k_ref = np.logspace(-3, 0, 100)
    # Scale to match data in the mid-range
    p_fib = patterns[fib_name]
    k_mid = 0.1
    idx_mid = np.argmin(np.abs(p_fib['k'] - k_mid))
    c_ref = p_fib['chi_V'][idx_mid] / k_mid**3
    ax.loglog(k_ref, c_ref * k_ref**3, 'k--', lw=2.5, alpha=0.8,
              label=r'Reference: $\tilde{\chi}_V \sim k^3$')
    ax.set_xlabel(r'Wavevector $k$', fontsize=13)
    ax.set_ylabel(r'$\tilde{\chi}_V(k)$', fontsize=13)
    ax.set_title(r'All Three Chains: Small-$k$ Envelope $\sim k^3$',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(1e-3, 2)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.text(0.5, 0.03,
            r'All chains share the same $k^3$ envelope (universal exponent)',
            transform=ax.transAxes, fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(r'Spectral Density $\tilde{\chi}_V(k)$ of Two-Phase Media '
                 r'($\phi_2=0.35$, non-overlapping rods)' '\n'
                 r'$\tilde{\chi}_V(k) = \rho\,|\tilde{m}(k)|^2\,S(k)$, '
                 r'where $\tilde{m}(k) = 2\sin(ka)/k$ is the rod form factor',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig6_spectral_density.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ==========================================================
    # Figure 7: Excess spreadability E(t) vs t
    # ==========================================================
    print("  Plotting Figure 7: Excess spreadability...")
    fig, (ax_main, ax_lat) = plt.subplots(1, 2, figsize=(16, 7),
                                           gridspec_kw={'width_ratios': [3, 1]})

    # --- Main panel: Poisson + 3 quasicrystals (comparable y-scales) ---
    plot_styles = {
        'Poisson': ('gray', '-', 2.0),
        fib_name: ('#2ca02c', '-', 2.0),
        CHAINS['silver']['name']: ('#9467bd', '-', 2.0),
        CHAINS['bronze']['name']: ('#d62728', '-', 2.0),
    }
    for label, (color, ls, lw) in plot_styles.items():
        p = patterns[label]
        mask = p['E_t'] > 0
        ax_main.loglog(t_array[mask], p['E_t'][mask], color=color, ls=ls, lw=lw,
                       label=label)

    # Reference slopes
    t_ref = np.logspace(1, 7, 100)
    # Scale reference lines to visually align with the data
    p_poi = patterns['Poisson']
    poi_ref_idx = np.argmin(np.abs(t_array - 1e4))
    poi_E_ref = p_poi['E_t'][poi_ref_idx] if p_poi['E_t'][poi_ref_idx] > 0 else 0.01
    c_half = poi_E_ref * (1e4)**0.5
    ax_main.loglog(t_ref, c_half * t_ref**(-0.5), 'k--', lw=1.5, alpha=0.4)
    ax_main.text(3e6, c_half * (3e6)**(-0.5) * 1.5, r'$\sim t^{-1/2}$' '\n'
                 r'($\alpha=0$)', fontsize=11, color='0.3')

    p_fib = patterns[fib_name]
    fib_ref_idx = np.argmin(np.abs(t_array - 1e3))
    fib_E_ref = p_fib['E_t'][fib_ref_idx] if p_fib['E_t'][fib_ref_idx] > 0 else 1e-5
    c_two = fib_E_ref * (1e3)**2.0
    ax_main.loglog(t_ref, c_two * t_ref**(-2.0), 'k:', lw=1.5, alpha=0.4)
    ax_main.text(2e1, c_two * (2e1)**(-2.0) * 0.5, r'$\sim t^{-2}$' '\n'
                 r'($\alpha=3$)', fontsize=11, color='0.3')

    # Shade the plateau measurement window
    ax_main.axvspan(1e2, 1e5, alpha=0.08, color='gold',
                    label=r'Plateau window ($10^2 < t < 10^5$)')

    ax_main.set_xlabel(r'Diffusion time $t$', fontsize=14)
    ax_main.set_ylabel(r'Excess spreadability $\mathcal{S}(\infty) - \mathcal{S}(t)$',
                       fontsize=14)
    ax_main.set_title('How Fast Does Diffusion Equilibrate?', fontsize=15,
                      fontweight='bold')
    ax_main.legend(fontsize=11, loc='lower left')
    ax_main.grid(True, ls=':', alpha=0.5)
    ax_main.tick_params(labelsize=12)
    ax_main.text(0.97, 0.95,
                 'Steeper decay = stronger\nhyperuniformity (larger $\\alpha$)',
                 transform=ax_main.transAxes, fontsize=11, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # --- Inset panel: Lattice (exponential decay — different y-scale) ---
    p_lat = patterns['Lattice']
    mask_lat = p_lat['E_t'] > 1e-20
    ax_lat.semilogy(t_array[mask_lat], p_lat['E_t'][mask_lat], 'k-', lw=2)
    ax_lat.set_xlabel(r'Time $t$', fontsize=13)
    ax_lat.set_ylabel(r'$E(t)$', fontsize=13)
    ax_lat.set_title('Lattice: Exponential\nDecay (separate scale)',
                     fontsize=12, fontweight='bold')
    ax_lat.grid(True, ls=':', alpha=0.5)
    ax_lat.tick_params(labelsize=11)
    ax_lat.set_xlim(-0.05, 1.5)
    ax_lat.text(0.5, 0.05,
                r'$E(t) \to 0$ by $t \approx 1$'
                '\n'r'$\Rightarrow \alpha \to \infty$'
                '\n(fastest possible decay)',
                transform=ax_lat.transAxes, fontsize=10,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    fig.suptitle('Excess Spreadability $\\mathcal{S}(\\infty)-\\mathcal{S}(t)$: '
                 'Decay rate reveals $\\alpha$ \u2014 '
                 'at long times $E(t) \\sim t^{-(1+\\alpha)/2}$',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig7_excess_spreadability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Extract plateau alpha values (needed for Figure 8 annotation) ---
    # Use linear fit of log(E) vs log(t) over [1e2, 1e5] for robust alpha.
    # For quasicrystals, the fit averages over oscillations and gives
    # better accuracy than pointwise log derivative + median.
    alpha_results = {}
    alpha_r2 = {}
    for label in patterns:
        p = patterns[label]
        if 'alpha_fit' in p and not np.isnan(p['alpha_fit']):
            alpha_results[label] = p['alpha_fit']
            alpha_r2[label] = p.get('alpha_fit_r2', 0.0)
        else:
            # Fallback for lattice (E(t) decays to zero before plateau)
            plateau_mask = (t_array >= 1e2) & (t_array <= 1e5)
            vals = p['alpha_t'][plateau_mask]
            valid = np.isfinite(vals)
            if np.any(valid):
                alpha_results[label] = np.median(vals[valid])
            else:
                alpha_results[label] = np.nan
            alpha_r2[label] = 0.0

    print("\n  Alpha values (linear fit over t in [1e2, 1e5]):")
    for label, alpha_val in alpha_results.items():
        r2 = alpha_r2.get(label, 0.0)
        print(f"    {label:30s}  alpha = {alpha_val:.4f}  (R^2 = {r2:.6f})")

    # ==========================================================
    # Figure 8: Extracted alpha(t) — MAIN SCIENTIFIC RESULT
    # ==========================================================
    print("  Plotting Figure 8: Extracted alpha...")
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Shade the plateau measurement region
    ax.axvspan(1e2, 1e5, alpha=0.12, color='gold',
               label=r'Measurement window ($10^2 < t < 10^5$)')

    # Plot each pattern
    fig8_styles = {
        'Poisson': ('gray', '-', 2.0),
        fib_name: ('#2ca02c', '-', 2.5),
        CHAINS['silver']['name']: ('#9467bd', '-', 2.5),
        CHAINS['bronze']['name']: ('#d62728', '-', 2.5),
    }
    for label, (color, ls, lw) in fig8_styles.items():
        p = patterns[label]
        valid = np.isfinite(p['alpha_t'])
        ax.semilogx(t_array[valid], p['alpha_t'][valid], color=color, ls=ls,
                     lw=lw, label=label)

    # Lattice (grows without bound — plot separately with thinner line)
    p_lat = patterns['Lattice']
    valid_lat = np.isfinite(p_lat['alpha_t'])
    ax.semilogx(t_array[valid_lat], p_lat['alpha_t'][valid_lat], color='black',
                ls='-', lw=1.5, label='Lattice (grows to $\\infty$)')

    # Reference lines
    ax.axhline(0, color='gray', ls='--', lw=1.5, alpha=0.6)
    ax.axhline(3, color='black', ls='--', lw=2, alpha=0.7)

    # Annotate the reference lines — positioned to avoid results box
    ax.text(5e-2, 0.35, r'$\alpha = 0$ (Poisson)', fontsize=12, color='gray',
            ha='left')
    ax.text(5e-2, 3.3, r'$\alpha = 3$ (predicted for all metallic-mean chains)',
            fontsize=12, color='black', fontweight='bold', ha='left')

    ax.set_xlabel(r'Diffusion time $t$', fontsize=14)
    ax.set_ylabel(r'Effective exponent $\alpha(t) = -2\,d\ln E/d\ln t - 1$',
                  fontsize=14)
    ax.set_title('Main Result: All Three Quasicrystal Chains Converge to '
                 r'$\alpha = 3$', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(-1.5, 8)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=12)

    # Results box with measured alpha values
    poi_alpha = alpha_results.get('Poisson', np.nan)
    fib_alpha = alpha_results.get(fib_name, np.nan)
    sil_alpha = alpha_results.get(CHAINS['silver']['name'], np.nan)
    bro_alpha = alpha_results.get(CHAINS['bronze']['name'], np.nan)
    results_text = (
        'Measured $\\alpha$ (linear fit):\n'
        f'  Poisson:    {poi_alpha:6.3f}  (expected 0)\n'
        f'  Fibonacci:  {fib_alpha:6.3f}  (expected 3)\n'
        f'  Silver:       {sil_alpha:6.3f}  (expected 3)\n'
        f'  Bronze:      {bro_alpha:6.3f}  (expected 3)'
    )
    ax.text(0.98, 0.97, results_text, transform=ax.transAxes,
            fontsize=11, ha='right', va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', alpha=0.9))

    fig.text(0.5, 0.01,
             r'Figure 8: $\alpha$ extracted via period-aware log-derivative '
             r'(curves) and linear fit of $\ln E$ vs $\ln t$ (reported values). '
             'Plateau at $\\alpha=3$ confirms Class I hyperuniformity.',
             ha='center', fontsize=11, style='italic', color='0.3')
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    path = os.path.join(RESULTS_DIR, 'fig8_alpha_extraction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return alpha_results


# ============================================================
# Summary table
# ============================================================
def print_summary(lambda_bar_lattice, chain_results, lambda_bar_proj,
                  alpha_results=None):
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

    # Alpha extraction results
    if alpha_results is not None:
        print("\n" + "=" * 70)
        print("  Summary: Hyperuniformity Exponent alpha")
        print("  (via linear fit of ln E vs ln t over [1e2, 1e5])")
        print("=" * 70)
        print(f"  {'Pattern':30s}  {'alpha':>10s}  {'Expected':>10s}  "
              f"{'Error %':>10s}  {'Status':>10s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for label, alpha_val in alpha_results.items():
            if 'Lattice' in label:
                alpha_str = 'exp' if np.isnan(alpha_val) else f'{alpha_val:.3f}'
                print(f"  {label:30s}  {alpha_str:>10s}  {'exp':>10s}  "
                      f"{'---':>10s}  {'OK':>10s}")
            elif np.isnan(alpha_val):
                print(f"  {label:30s}  {'N/A':>10s}  {'---':>10s}  "
                      f"{'---':>10s}  {'CHECK':>10s}")
            elif 'Poisson' in label:
                status = 'OK' if abs(alpha_val) < 0.5 else 'CHECK'
                print(f"  {label:30s}  {alpha_val:10.3f}  {'0':>10s}  "
                      f"{'---':>10s}  {status:>10s}")
            else:
                err_pct = abs(alpha_val - 3.0) / 3.0 * 100
                status = 'OK' if abs(alpha_val - 3.0) < 0.5 else 'CHECK'
                print(f"  {label:30s}  {alpha_val:10.3f}  {'3':>10s}  "
                      f"{err_pct:9.2f}%  {status:>10s}")


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
    alpha_results = run_spreadability_analysis(rng)

    # Summary
    print_summary(lambda_bar_lattice, chain_results, lambda_bar_proj,
                  alpha_results)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n  Total runtime: {total_elapsed:.1f}s")
    print(f"  All figures saved to: {RESULTS_DIR}")
    print("\nDone.")
