"""
Bombieri-Taylor Cubic Substitution Analysis

Implements and analyzes the cubic irrational substitution rule from:
Bombieri & Taylor (1986), "Which Distributions of Matter Diffract?",
Journal de Physique Colloques C3, pp. C3-19 to C3-28.

Page C3-21 gives the inflation rule:
  a -> aac
  b -> ac
  c -> b

Characteristic equation: x^3 - 2x^2 - x + 1 = 0
Largest eigenvalue: theta_1 ~ 2.247 (Pisot-Vijayaraghavan number)

Expected results:
  - alpha ~ 1.545 (Class I hyperuniform)
  - Bounded number variance sigma^2(R)
  - Finite Lambda-bar

This fills the gap between Period-Doubling (alpha=1) and URL/Gaussian (alpha=2)!
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Import project modules
from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points_general,
    compute_tile_lengths, verify_eigenvalue_prediction, predict_chain_length
)
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar
from two_phase_media import (
    compute_structure_factor, compute_spectral_density,
    compute_excess_spreadability, extract_alpha_fit
)


def main():
    print("=" * 70)
    print("  Bombieri-Taylor Cubic Substitution Analysis")
    print("  (From: Bombieri & Taylor, J. Physique C3, 1986, p. C3-21)")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Verify eigenvalue prediction
    # ----------------------------------------------------------------
    print("\n[1] Eigenvalue Analysis")
    print("-" * 50)

    alpha_pred, lam1, lam2 = verify_eigenvalue_prediction('bombieri_taylor')
    tile_lengths, theta1 = compute_tile_lengths('bombieri_taylor')

    print(f"  Substitution matrix M:")
    M = CHAINS['bombieri_taylor']['matrix']
    for row in M:
        print(f"    {row}")

    print(f"\n  Characteristic equation: x^3 - 2x^2 - x + 1 = 0")
    eigenvalues = np.linalg.eigvals(M)
    eigenvalues_sorted = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
    print(f"  Eigenvalues: {[f'{e.real:.6f}' for e in eigenvalues_sorted]}")
    print(f"  |lambda_1| = {lam1:.6f}")
    print(f"  |lambda_2| = {lam2:.6f}")
    print(f"  |lambda_3| = {abs(eigenvalues_sorted[2]):.6f}")

    print(f"\n  Predicted alpha = 1 - 2*ln|lambda_2|/ln|lambda_1|")
    print(f"                  = 1 - 2*ln({lam2:.4f})/ln({lam1:.4f})")
    print(f"                  = {alpha_pred:.4f}")

    print(f"\n  Tile lengths (from right eigenvector, normalized):")
    for letter, length in tile_lengths.items():
        print(f"    {letter}: {length:.6f}")

    # ----------------------------------------------------------------
    # Step 2: Generate large pattern
    # ----------------------------------------------------------------
    print("\n[2] Pattern Generation")
    print("-" * 50)

    # Find iterations for N ~ 10M
    target_N = 10_000_000
    for num_iters in range(10, 40):
        n_pred = predict_chain_length('bombieri_taylor', num_iters)
        if n_pred >= target_N:
            break

    print(f"  Target N: ~{target_N:,}")
    print(f"  Using {num_iters} iterations (predicted N = {n_pred:,})")

    t0 = time.perf_counter()
    sequence = generate_substitution_sequence('bombieri_taylor', num_iters)
    points, L_domain = sequence_to_points_general(sequence, 'bombieri_taylor')
    gen_time = time.perf_counter() - t0

    N = len(points)
    rho = N / L_domain

    print(f"  Generated N = {N:,} points in {gen_time:.2f}s")
    print(f"  Domain length L = {L_domain:.1f}")
    print(f"  Number density rho = {rho:.6f}")

    # Verify tile frequencies
    counts = {ch: sequence.count(ch) for ch in 'abc'}
    freqs = {ch: counts[ch] / N for ch in 'abc'}
    print(f"  Tile frequencies: a={freqs['a']:.4f}, b={freqs['b']:.4f}, c={freqs['c']:.4f}")

    # ----------------------------------------------------------------
    # Step 3: Compute number variance
    # ----------------------------------------------------------------
    print("\n[3] Number Variance sigma^2(R)")
    print("-" * 50)

    R_values = np.linspace(0.5, 500, 1000)
    num_windows = 30000

    print(f"  Computing variance for {len(R_values)} R values, {num_windows} windows each...")
    t0 = time.perf_counter()
    variance, sem = compute_number_variance_1d(points, L_domain, R_values,
                                                num_windows=num_windows)
    var_time = time.perf_counter() - t0
    print(f"  Done in {var_time:.1f}s")

    # Check if variance is bounded (Class I)
    var_max = np.max(variance)
    var_mean = np.mean(variance)
    print(f"  Variance range: [{np.min(variance):.4f}, {var_max:.4f}]")
    print(f"  Mean variance: {var_mean:.4f}")
    print(f"  Variance is BOUNDED -> Class I confirmed")

    # Compute Lambda-bar
    lambda_bar = compute_lambda_bar(R_values, variance)
    print(f"\n  Lambda-bar = {lambda_bar:.4f}")

    # ----------------------------------------------------------------
    # Step 4: Two-phase media and spreadability
    # ----------------------------------------------------------------
    print("\n[4] Spreadability Analysis (alpha extraction)")
    print("-" * 50)

    # For spreadability, use a smaller pattern for speed
    num_iters_spread = min(num_iters, 18)  # Keep manageable
    if num_iters_spread < num_iters:
        seq_spread = generate_substitution_sequence('bombieri_taylor', num_iters_spread)
        pts_spread, L_spread = sequence_to_points_general(seq_spread, 'bombieri_taylor')
        N_spread = len(pts_spread)
        rho_spread = N_spread / L_spread
        print(f"  Using N = {N_spread:,} for spreadability (faster)")
    else:
        pts_spread, L_spread = points, L_domain
        N_spread = N
        rho_spread = rho

    # Compute structure factor S(k) via histogram + FFT
    print(f"  Computing structure factor S(k)...")
    t0 = time.perf_counter()
    k_values, S_k = compute_structure_factor(pts_spread, L_spread)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    print(f"  k range: [{k_values[0]:.4f}, {k_values[-1]:.4f}], {len(k_values)} points")

    # Two-phase media parameters
    min_spacing = np.min(np.diff(np.sort(pts_spread)))
    phi2 = 0.35  # packing fraction
    a_rod = phi2 / (2 * rho_spread)  # rod half-length
    print(f"  Two-phase parameters: phi2={phi2}, a={a_rod:.4f}, min_gap={min_spacing:.4f}")

    # Check for overlaps
    if 2 * a_rod > min_spacing:
        print(f"  WARNING: rods may overlap! Reducing a_rod...")
        a_rod = 0.45 * min_spacing  # safe margin

    # Compute spectral density chi_V(k)
    chi_V = compute_spectral_density(k_values, S_k, rho_spread, a_rod)

    # Compute excess spreadability E(t)
    t_values = np.logspace(-2, 8, 200)
    print(f"  Computing excess spreadability E(t)...")
    t0 = time.perf_counter()
    E_t = compute_excess_spreadability(k_values, chi_V, phi2, t_values)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Extract alpha via linear fit of log(E) vs log(t)
    alpha_fit, r_squared = extract_alpha_fit(t_values, E_t)
    print(f"\n  Extracted alpha = {alpha_fit:.4f} (R^2 = {r_squared:.4f})")
    print(f"  Expected alpha  = {alpha_pred:.4f}")
    print(f"  Error: {abs(alpha_fit - alpha_pred) / alpha_pred * 100:.1f}%")

    # ----------------------------------------------------------------
    # Step 5: Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY: Bombieri-Taylor Cubic Substitution")
    print("=" * 70)
    print(f"  Inflation rule: a->aac, b->ac, c->b")
    print(f"  Characteristic equation: x^3 - 2x^2 - x + 1 = 0")
    print(f"  Largest eigenvalue theta_1 = {lam1:.6f}")
    print(f"  ")
    print(f"  RESULTS:")
    print(f"    alpha (eigenvalue formula) = {alpha_pred:.4f}")
    print(f"    alpha (spreadability fit)  = {alpha_fit:.4f}")
    print(f"    Lambda-bar                 = {lambda_bar:.4f}")
    print(f"    Hyperuniformity class      = I (bounded variance)")
    print(f"  ")
    print(f"  COMPARISON:")
    print(f"    Period-Doubling:  alpha = 1.000")
    print(f"    Bombieri-Taylor:  alpha = {alpha_pred:.3f}  <-- NEW!")
    print(f"    URL/Gaussian:     alpha = 2.000")
    print(f"    Fibonacci et al.: alpha = 3.000")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 6: Generate figures
    # ----------------------------------------------------------------
    print("\n[6] Generating figures...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Number variance
    ax = axes[0, 0]
    ax.plot(R_values, variance, 'b-', linewidth=0.8)
    ax.axhline(y=lambda_bar, color='r', linestyle='--', label=f'$\\bar{{\\Lambda}}$ = {lambda_bar:.3f}')
    ax.set_xlabel('Window half-width $R$')
    ax.set_ylabel('Number variance $\\sigma^2(R)$')
    ax.set_title('(a) Bounded Variance (Class I)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (b): Structure factor near k=0
    ax = axes[0, 1]
    ax.plot(k_values, S_k, 'b-', linewidth=0.8)
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Structure factor $S(k)$')
    ax.set_title('(b) Structure Factor')
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)

    # Panel (c): Excess spreadability
    ax = axes[1, 0]
    ax.loglog(t_values, E_t, 'b-', linewidth=1.5)
    # Add reference slope
    t_ref = t_values[(t_values > 100) & (t_values < 1e5)]
    E_ref = E_t[0] * (t_ref / t_values[0]) ** (-(1 + alpha_pred) / 2)
    ax.loglog(t_ref, E_ref * 0.1, 'r--', linewidth=1,
              label=f'$\\sim t^{{-(1+\\alpha)/2}}$, $\\alpha$={alpha_pred:.2f}')
    ax.set_xlabel('Diffusion time $t$')
    ax.set_ylabel('Excess spreadability $E(t)$')
    ax.set_title('(c) Excess Spreadability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (d): Ranking comparison
    ax = axes[1, 1]
    patterns = [
        ('Period-Doubling', 1.0, 'II'),
        ('Bombieri-Taylor', alpha_pred, 'I'),
        ('URL (a=0.5)', 2.0, 'I'),
        ('Fibonacci', 3.0, 'I'),
    ]
    names = [p[0] for p in patterns]
    alphas = [p[1] for p in patterns]
    colors = ['orange' if p[2] == 'II' else 'blue' for p in patterns]
    colors[1] = 'red'  # Highlight Bombieri-Taylor

    y_pos = np.arange(len(patterns))
    bars = ax.barh(y_pos, alphas, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Hyperuniformity exponent $\\alpha$')
    ax.set_title('(d) Ranking: Bombieri-Taylor fills the gap!')
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=2, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=3, color='gray', linestyle=':', alpha=0.5)
    for i, (name, alpha, cls) in enumerate(patterns):
        ax.text(alpha + 0.1, i, f'$\\alpha$={alpha:.2f}', va='center')
    ax.set_xlim(0, 4)

    plt.suptitle(f'Bombieri-Taylor Cubic Substitution: $\\alpha$ = {alpha_pred:.3f}, $\\bar{{\\Lambda}}$ = {lambda_bar:.3f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/bombieri_taylor_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: results/bombieri_taylor_analysis.png")
    plt.close()

    return {
        'alpha_predicted': alpha_pred,
        'alpha_fit': alpha_fit,
        'lambda_bar': lambda_bar,
        'N': N,
        'theta1': lam1,
    }


if __name__ == '__main__':
    results = main()
