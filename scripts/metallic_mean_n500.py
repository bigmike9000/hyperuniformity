"""
Metallic-mean convergence extended to n=500.

Computes Lambda_bar for n = 1, 2, 3, 5, 10, 20, 50, 100, 200, 500
with appropriate TARGET_N for each (larger N for smaller n where chains
grow fast, smaller N for large n where growth is slow).

Output: results/metallic_n500.json, results/figures/fig_metallic_n500.png
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, 'metallic_n500.json')
OUT_FIG = os.path.join(FIG_DIR, 'fig_metallic_n500.png')

N_LIST = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
TARGET_N = 500_000
NUM_WINDOWS = 20_000
NUM_R = 500
SEED = 2030

rng = np.random.default_rng(SEED)


def metallic_mean(n):
    return (n + np.sqrt(n**2 + 4)) / 2


def generate_metallic_sequence(n, target_n):
    rules = {'S': 'L', 'L': 'L' * n + 'S'}
    seq = 'L'
    iters = 0
    while len(seq) < target_n:
        seq = ''.join(rules[ch] for ch in seq)
        iters += 1
        if iters > 200:
            break
    return seq, iters


def sequence_to_points(sequence, n):
    mu = metallic_mean(n)
    arr = np.frombuffer(sequence.encode(), dtype='S1')
    lengths = np.where(arr == b'L', mu, 1.0)
    L_domain = float(np.sum(lengths))
    points = np.empty(len(sequence), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])
    return points, L_domain


def compute_lb(n):
    t0 = time.perf_counter()
    print(f"  n={n:3d}: generating...", end='', flush=True)

    seq, iters = generate_metallic_sequence(n, TARGET_N)
    N_actual = len(seq)
    print(f" N={N_actual:,} ({iters} iters)", end='', flush=True)

    points, L_domain = sequence_to_points(seq, n)
    del seq
    rho = N_actual / L_domain

    mean_spacing = 1.0 / rho
    R_max = min(300 * mean_spacing, L_domain / 4)
    R_array = np.linspace(mean_spacing, R_max, NUM_R)

    variances, _ = compute_number_variance_1d(
        points, L_domain, R_array, num_windows=NUM_WINDOWS, rng=rng, periodic=True)

    lb = compute_lambda_bar(R_array, variances)

    # Bootstrap error: split tail into 8 blocks
    tail = variances[len(variances) // 3:]
    n_blocks = 8
    splits = np.array_split(tail, n_blocks)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    lb_err = np.std(boots) / np.sqrt(len(boots)) if len(boots) > 1 else 0.0

    elapsed = time.perf_counter() - t0
    print(f"  Lb={lb:.5f}+/-{lb_err:.5f}  ({elapsed:.1f}s)")

    return {
        'lambda_bar': float(lb),
        'err': float(lb_err),
        'rho': float(rho),
        'N': int(N_actual),
    }


def make_figure(results):
    ns = sorted(results.keys())
    lbs = [results[n]['lambda_bar'] for n in ns]
    errs = [results[n]['err'] for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ns, lbs, yerr=errs, fmt='o-', color='#2166ac', ms=5,
                capsize=3, lw=1.5, label=r'Measured $\bar\Lambda(\mu_n)$')
    ax.axhline(y=1/3, color='#d6604d', ls='--', lw=1.5,
               label=r'$1/3$ (URL cloaked)')
    ax.set_xlabel(r'Metallic-mean index $n$', fontsize=12)
    ax.set_ylabel(r'$\bar\Lambda$', fontsize=12)
    ax.set_title(r'$\bar\Lambda(\mu_n)$ for $n = 1$ to $500$',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 700)
    ax.set_ylim(0.18, 0.38)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {OUT_FIG}")


if __name__ == '__main__':
    print("=" * 70)
    print("  Metallic-Mean Convergence: n=1 to 500")
    print("=" * 70)

    results = {}
    for n in N_LIST:
        results[n] = compute_lb(n)

    # Summary
    print("\n" + "=" * 70)
    print(f"  {'n':>4s}  {'Lb':>10s}  {'err':>8s}  {'1/3-Lb':>10s}  {'sigma':>8s}")
    print("  " + "-" * 50)
    for n in N_LIST:
        r = results[n]
        diff = 1/3 - r['lambda_bar']
        sig = diff / r['err'] if r['err'] > 0 else 0
        print(f"  {n:4d}  {r['lambda_bar']:10.5f}  {r['err']:8.5f}  "
              f"{diff:10.5f}  {sig:+8.1f}")

    # Save
    out_data = {
        'n_values': N_LIST,
        'results': {str(n): results[n] for n in N_LIST},
        'params': {'TARGET_N': TARGET_N, 'NUM_WINDOWS': NUM_WINDOWS,
                   'NUM_R': NUM_R, 'SEED': SEED},
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved: {OUT_JSON}")

    make_figure(results)
