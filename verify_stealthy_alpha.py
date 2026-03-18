"""
Verify alpha = 2 for stealthy hyperuniform patterns via spreadability analysis.

Uses spreadability decay E(t) ~ t^{-(1+alpha)/2} to extract alpha numerically
and compare with the expected theoretical value alpha = 2.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from two_phase_media import (
    compute_structure_factor, compute_spectral_density,
    compute_excess_spreadability, extract_alpha_fit
)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stealthy_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CHI_VALUES = [0.1, 0.2, 0.3]
NUM_CONFIGS = 50  # Number of configs to analyze per chi (for speed)


def read_cco(filepath):
    """Read a CCO configuration file. Returns (points_1d, L)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    d = int(lines[0].strip())
    assert d == 1, f"Expected d=1, got d={d}"

    lattice_line = lines[1].strip().split()
    L = float(lattice_line[0])

    points = []
    for line in lines[2:]:
        parts = line.strip().split()
        if parts:
            points.append(float(parts[0]))

    return np.array(points), L


def analyze_chi_spreadability(chi, num_configs=NUM_CONFIGS):
    """Compute spreadability and extract alpha for a given chi."""
    folder = os.path.join(DATA_DIR, f'{chi}_patterns')
    if not os.path.isdir(folder):
        print(f"  WARNING: folder not found: {folder}")
        return None

    config_files = sorted([
        f for f in os.listdir(folder)
        if f.endswith('.txt') and not f.startswith('sf_')
    ])[:num_configs]

    n_configs = len(config_files)
    print(f"\n  --- chi = {chi} ---")
    print(f"  Analyzing {n_configs} configurations")

    # Read first file to get parameters
    pts0, L0 = read_cco(os.path.join(folder, config_files[0]))
    N = len(pts0)
    rho = N / L0
    print(f"  N={N}, L={L0:.1f}, rho={rho:.4f}")

    # Two-phase media parameters
    min_spacing = 0.5  # Conservative estimate for stealthy
    a_rod = min_spacing / 4
    phi2 = 2 * a_rod * rho

    # Time array for spreadability
    t_values = np.logspace(-2, 6, 150)

    # Accumulate E(t) across configurations
    E_t_sum = np.zeros_like(t_values)
    t0 = time.perf_counter()

    for i, fname in enumerate(config_files):
        pts, L = read_cco(os.path.join(folder, fname))
        pts = np.sort(pts % L)

        # Compute structure factor
        k_values, S_k = compute_structure_factor(pts, L)

        # Compute spectral density
        chi_V = compute_spectral_density(k_values, S_k, rho, a_rod)

        # Compute excess spreadability
        E_t = compute_excess_spreadability(k_values, chi_V, phi2, t_values)
        E_t_sum += E_t

        if (i + 1) % 10 == 0:
            print(f"    [{i+1:3d}/{n_configs}] processed")

    # Average over configurations
    E_t_avg = E_t_sum / n_configs
    elapsed = time.perf_counter() - t0
    print(f"  Computed in {elapsed:.1f}s")

    # Extract alpha via linear fit
    alpha_fit, r_squared = extract_alpha_fit(t_values, E_t_avg, t_min=1e2, t_max=1e5)

    result = {
        'chi': chi,
        'N': N,
        'L': L0,
        'rho': rho,
        'n_configs': n_configs,
        't_values': t_values,
        'E_t': E_t_avg,
        'alpha_fit': alpha_fit,
        'r_squared': r_squared,
        'alpha_expected': 2.0,
    }

    print(f"\n  Results for chi={chi}:")
    print(f"    alpha (spreadability fit) = {alpha_fit:.4f}")
    print(f"    alpha (expected)          = 2.0000")
    print(f"    R^2 of fit                = {r_squared:.4f}")
    print(f"    Error                     = {abs(alpha_fit - 2.0) / 2.0 * 100:.1f}%")

    return result


def plot_results(results):
    """Plot spreadability curves and alpha comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'0.1': '#1f77b4', '0.2': '#ff7f0e', '0.3': '#2ca02c'}

    # Panel (a): E(t) curves
    ax = axes[0]
    for chi, res in sorted(results.items()):
        ax.loglog(res['t_values'], res['E_t'], '-', color=colors[str(chi)],
                  linewidth=1.5, label=rf'$\chi={chi}$, $\alpha$={res["alpha_fit"]:.2f}')

    # Add reference slope for alpha=2
    t_ref = np.logspace(2, 5, 50)
    E_ref = 0.1 * t_ref ** (-1.5)  # E ~ t^{-(1+2)/2} = t^{-1.5}
    ax.loglog(t_ref, E_ref, 'k--', linewidth=2, alpha=0.5, label=r'$\sim t^{-3/2}$ ($\alpha=2$)')

    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel('Excess spreadability $E(t)$', fontsize=12)
    ax.set_title('(a) Spreadability Decay', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-2, 1e6)

    # Panel (b): Alpha comparison
    ax = axes[1]
    chi_vals = sorted(results.keys())
    alphas = [results[chi]['alpha_fit'] for chi in chi_vals]
    x_pos = np.arange(len(chi_vals))

    bars = ax.bar(x_pos, alphas, color=[colors[str(chi)] for chi in chi_vals],
                  alpha=0.7, edgecolor='black')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Expected $\\alpha = 2$')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([rf'$\chi={chi}$' for chi in chi_vals], fontsize=11)
    ax.set_ylabel(r'Hyperuniformity exponent $\alpha$', fontsize=12)
    ax.set_title('(b) Alpha Verification', fontsize=12)
    ax.set_ylim(0, 3)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, alpha in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{alpha:.2f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'stealthy_alpha_verification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    print("=" * 70)
    print("  Verifying alpha = 2 for Stealthy Patterns via Spreadability")
    print("=" * 70)

    results = {}
    for chi in CHI_VALUES:
        res = analyze_chi_spreadability(chi)
        if res is not None:
            results[chi] = res

    if results:
        plot_results(results)

        # Summary table
        print("\n" + "=" * 70)
        print("  SUMMARY: Alpha Verification for Stealthy Patterns")
        print("=" * 70)
        print(f"  {'chi':>5s}  {'alpha (fit)':>12s}  {'alpha (exp)':>12s}  {'error':>8s}  {'R^2':>8s}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")
        for chi in sorted(results.keys()):
            r = results[chi]
            err = abs(r['alpha_fit'] - 2.0) / 2.0 * 100
            print(f"  {chi:5.2f}  {r['alpha_fit']:12.4f}  {2.0:12.4f}  {err:7.1f}%  {r['r_squared']:8.4f}")

        # Overall assessment
        avg_alpha = np.mean([r['alpha_fit'] for r in results.values()])
        avg_err = abs(avg_alpha - 2.0) / 2.0 * 100
        print(f"\n  Average alpha = {avg_alpha:.4f} (error: {avg_err:.1f}%)")

        if avg_err < 5:
            print("  VERDICT: alpha = 2 CONFIRMED within 5% error")
        elif avg_err < 10:
            print("  VERDICT: alpha = 2 approximately confirmed (5-10% error)")
        else:
            print("  VERDICT: Significant deviation from alpha = 2")

    print("\nDone.")
