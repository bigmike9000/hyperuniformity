"""
URL parameter sweep: Lambda_bar(a) for a ∈ [0, 2] — Week 4, Task 3.

Computes Lambda_bar both from exact formula and simulation, generating a figure
comparing the two. Marks cloaking events at integer a.

Output:
  results/figures/fig_url_sweep.png
  results/url_sweep.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from disordered_patterns import generate_url, lambda_bar_url_exact
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIG_PATH = os.path.join(OUTPUT_DIR, 'fig_url_sweep.png')
JSON_PATH = os.path.join(SCRIPT_DIR, 'results', 'url_sweep.json')

# Dense grid for exact formula
A_FINE = np.linspace(0.001, 2.0, 200)

# Simulation points (sparser)
A_SIM = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.1, 1.2, 1.3, 1.5, 1.75, 2.0]

N_POINTS = 50_000
NUM_WINDOWS = 10_000
NUM_R = 300
SEED = 42


def compute_lb_sim(a, rng):
    """Simulate URL and compute Lambda_bar."""
    points, L = generate_url(N_POINTS, a, rng=rng)
    rho = N_POINTS / L
    mean_spacing = 1.0 / rho
    R_max = min(200 * mean_spacing, L / 4)
    R_array = np.linspace(mean_spacing, R_max, NUM_R)

    variances, _ = compute_number_variance_1d(
        points, L, R_array, num_windows=NUM_WINDOWS, rng=rng, periodic=True)

    lb = compute_lambda_bar(R_array, variances)

    # Bootstrap error
    tail = variances[len(variances) // 3:]
    splits = np.array_split(tail, 4)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    lb_err = np.std(boots) / np.sqrt(len(boots)) if len(boots) > 1 else 0.0

    return lb, lb_err


def main():
    rng = np.random.default_rng(SEED)

    # Exact formula
    lb_exact = np.array([lambda_bar_url_exact(a) for a in A_FINE])

    # Simulations
    print("Computing simulated Lambda_bar for URL at various a values...")
    sim_results = []
    for a in A_SIM:
        lb, err = compute_lb_sim(a, rng)
        exact = lambda_bar_url_exact(a)
        pct_err = abs(lb - exact) / exact * 100
        print(f"  a={a:.2f}: Lambda_bar_sim={lb:.5f}±{err:.5f}, "
              f"exact={exact:.5f}, diff={pct_err:.2f}%")
        sim_results.append({
            'a': float(a),
            'lambda_bar_sim': float(lb),
            'err': float(err),
            'lambda_bar_exact': float(exact),
        })

    # Save JSON
    with open(JSON_PATH, 'w') as f:
        json.dump({'simulations': sim_results, 'N': N_POINTS,
                   'NUM_WINDOWS': NUM_WINDOWS}, f, indent=2)
    print(f"\nSaved: {JSON_PATH}")

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: Lambda_bar(a)
    ax1.plot(A_FINE, lb_exact, 'b-', lw=2,
             label=r'Exact: $\bar\Lambda = (1+a^2)/6$')

    a_sim = [r['a'] for r in sim_results]
    lb_sim = [r['lambda_bar_sim'] for r in sim_results]
    err_sim = [r['err'] for r in sim_results]
    ax1.errorbar(a_sim, lb_sim, yerr=err_sim, fmt='ro', ms=5, capsize=3,
                 label='Simulation', zorder=5)

    # Mark special points
    ax1.axhline(y=1/6, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax1.text(0.05, 1/6 + 0.005, r'$1/6$ (lattice)', fontsize=8, color='gray')
    ax1.axhline(y=1/3, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax1.text(0.05, 1/3 + 0.005, r'$1/3$ (cloaked)', fontsize=8, color='gray')

    # Mark cloaking at a=1
    ax1.axvline(x=1.0, color='green', ls=':', lw=1.2, alpha=0.7)
    ax1.text(1.02, 0.20, 'cloaking\n($a=1$)', fontsize=8, color='green')

    ax1.set_ylabel(r'$\bar\Lambda$', fontsize=12)
    ax1.set_title(r'URL: $\bar\Lambda(a)$ for displacement amplitude $a$',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 2.05)

    # Bottom panel: relative error
    rel_err = [(r['lambda_bar_sim'] - r['lambda_bar_exact']) / r['lambda_bar_exact'] * 100
               for r in sim_results]
    ax2.bar(a_sim, rel_err, width=0.05, color='steelblue', alpha=0.7)
    ax2.axhline(y=0, color='black', lw=0.5)
    ax2.set_xlabel(r'Displacement amplitude $a$', fontsize=12)
    ax2.set_ylabel('Relative error (%)', fontsize=10)
    ax2.set_xlim(0, 2.05)
    ax2.set_ylim(-3, 3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")


if __name__ == '__main__':
    main()
