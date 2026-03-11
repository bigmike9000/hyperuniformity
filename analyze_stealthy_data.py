"""
Analyze stealthy hyperuniform configurations from grad student data.

Reads ~4000 configurations per chi value (chi=0.1, 0.2, 0.3) with
N=2000 particles and density=1. Computes Lambda_bar for each configuration
and reports ensemble statistics.

Data format (CCO):
  Line 1: dimension d
  Next d lines: primitive lattice vectors (d elements + padding)
  Remaining lines: particle positions (d elements + padding)
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

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stealthy_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CHI_VALUES = [0.1, 0.2, 0.3]
NUM_R_POINTS = 200
NUM_WINDOWS = 20000
MAX_CONFIGS = None   # Set to e.g. 100 for quick test; None for all


def read_cco(filepath):
    """Read a CCO configuration file. Returns (points_1d, L)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    d = int(lines[0].strip())
    assert d == 1, f"Expected d=1, got d={d}"

    # Lattice vector (just one line for d=1)
    lattice_line = lines[1].strip().split()
    L = float(lattice_line[0])

    # Particle positions
    points = []
    for line in lines[2:]:
        parts = line.strip().split()
        if parts:
            points.append(float(parts[0]))

    points = np.array(points)
    return points, L


def analyze_chi(chi, max_configs=None):
    """Analyze all configurations for a given chi value."""
    folder = os.path.join(DATA_DIR, f'{chi}_patterns')
    if not os.path.isdir(folder):
        print(f"  WARNING: folder not found: {folder}")
        return None

    # Find configuration files (exclude sf_*.txt)
    config_files = sorted([
        f for f in os.listdir(folder)
        if f.endswith('.txt') and not f.startswith('sf_')
    ])

    if max_configs is not None:
        config_files = config_files[:max_configs]

    n_configs = len(config_files)
    print(f"\n  --- chi = {chi} ---")
    print(f"  Found {n_configs} configurations")

    # Read first file to get parameters
    pts0, L0 = read_cco(os.path.join(folder, config_files[0]))
    N = len(pts0)
    rho = N / L0
    print(f"  N={N}, L={L0:.1f}, rho={rho:.4f}")

    # Set R range
    R_max = min(300, L0 / 4)
    R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)

    # Compute Lambda_bar for each configuration
    rng = np.random.default_rng(seed=42)
    lambda_bars = []
    t0 = time.perf_counter()

    for i, fname in enumerate(config_files):
        pts, L = read_cco(os.path.join(folder, fname))
        pts = np.sort(pts % L)  # Ensure sorted and in [0, L)

        var, _ = compute_number_variance_1d(pts, L, R_arr,
                                            num_windows=NUM_WINDOWS, rng=rng)
        lb = compute_lambda_bar(R_arr, var)
        lambda_bars.append(lb)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_configs - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1:4d}/{n_configs}] "
                  f"Lambda_bar={lb:.6f}  "
                  f"running_mean={np.mean(lambda_bars):.6f}  "
                  f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.perf_counter() - t0
    lb_arr = np.array(lambda_bars)

    result = {
        'chi': chi,
        'N': N,
        'L': L0,
        'rho': rho,
        'n_configs': n_configs,
        'lambda_bars': lb_arr,
        'mean': np.mean(lb_arr),
        'std': np.std(lb_arr),
        'sem': np.std(lb_arr) / np.sqrt(n_configs),
        'median': np.median(lb_arr),
        'time': elapsed,
    }

    print(f"\n  Results for chi={chi}:")
    print(f"    Lambda_bar = {result['mean']:.6f} +/- {result['sem']:.6f} "
          f"(std={result['std']:.6f})")
    print(f"    median = {result['median']:.6f}")
    print(f"    {n_configs} configs analyzed in {elapsed:.1f}s")

    return result


def plot_results(results):
    """Plot Lambda_bar distributions and summary."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (chi, res) in enumerate(sorted(results.items())):
        ax = axes[idx]
        ax.hist(res['lambda_bars'], bins=50, color=colors[idx], alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.axvline(res['mean'], color='red', ls='--', lw=2,
                   label=rf"$\bar{{\Lambda}} = {res['mean']:.4f} \pm {res['sem']:.4f}$")
        ax.axvline(res['median'], color='green', ls=':', lw=2,
                   label=f"median = {res['median']:.4f}")
        ax.set_xlabel(r'$\bar{\Lambda}$', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(rf'Stealthy $\chi={chi}$ (N={res["N"]}, '
                     f'{res["n_configs"]} configs)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, ls=':', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig13_stealthy_ensemble.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    print("=" * 70)
    print("  Analyzing Stealthy Hyperuniform Data (Grad Student Configurations)")
    print("=" * 70)

    max_configs = MAX_CONFIGS
    if '--quick' in sys.argv:
        max_configs = 50
        print("  Quick mode: analyzing 50 configs per chi")
    elif '--medium' in sys.argv:
        max_configs = 500
        print("  Medium mode: analyzing 500 configs per chi")

    results = {}
    for chi in CHI_VALUES:
        res = analyze_chi(chi, max_configs=max_configs)
        if res is not None:
            results[chi] = res

    if results:
        plot_results(results)

        # Summary table
        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)
        print(f"  {'chi':>5s}  {'N':>5s}  {'configs':>8s}  "
              f"{'Lambda_bar':>12s}  {'SEM':>10s}  {'std':>10s}")
        print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")
        for chi in sorted(results.keys()):
            r = results[chi]
            print(f"  {chi:5.2f}  {r['N']:5d}  {r['n_configs']:8d}  "
                  f"{r['mean']:12.6f}  {r['sem']:10.6f}  {r['std']:10.6f}")

    print("\nDone.")
