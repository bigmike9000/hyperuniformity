"""
Extended Metallic-Mean Study: Lambda_bar for n up to 50.

Computes Lambda_bar(mu_n) for n = 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50
with reduced parameters (TARGET_N=300k, NUM_WINDOWS=15k, NUM_R=400) for speed.

Key question: does Lambda_bar stay strictly below 1/3, or could it overshoot?
"""

import os
import sys
import json
import time
import numpy as np
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------
N_LIST = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]
TARGET_N = 300_000
NUM_WINDOWS = 15_000
NUM_R = 400
SEED = 2026

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, 'metallic_extended.json')

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------
# Helper functions (from metallic_mean_convergence.py)
# ----------------------------------------------------------------

def metallic_mean(n):
    """mu_n = (n + sqrt(n^2 + 4)) / 2"""
    return (n + np.sqrt(n**2 + 4)) / 2


def generate_metallic_sequence(n, target_n):
    """
    Generate the metallic-mean substitution chain of index n.
    Rules: S -> L,  L -> L^n S
    """
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
    """Convert tile sequence to 1D point positions."""
    mu = metallic_mean(n)
    arr = np.frombuffer(sequence.encode(), dtype='S1')
    lengths = np.where(arr == b'L', mu, 1.0)
    L_domain = float(np.sum(lengths))
    points = np.empty(len(sequence), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])
    return points, L_domain


def compute_lb(n, verbose=True):
    """Compute Lambda_bar for metallic-mean chain of index n."""
    t0 = time.perf_counter()
    if verbose:
        print(f"  n={n:3d}: generating chain (target N={TARGET_N:,})...", end='', flush=True)

    seq, iters = generate_metallic_sequence(n, TARGET_N)
    N_actual = len(seq)
    if verbose:
        print(f" N={N_actual:,} ({iters} iters, {time.perf_counter()-t0:.1f}s)", flush=True)

    points, L_domain = sequence_to_points(seq, n)
    del seq
    rho = N_actual / L_domain

    mean_spacing = 1.0 / rho
    R_max = min(300 * mean_spacing, L_domain / 4)
    R_array = np.linspace(mean_spacing, R_max, NUM_R)

    if verbose:
        t_var = time.perf_counter()
        print(f"  n={n:3d}: computing variance ({NUM_WINDOWS} windows, {NUM_R} R-pts)...",
              end='', flush=True)

    variances, _ = compute_number_variance_1d(
        points, L_domain, R_array, num_windows=NUM_WINDOWS, rng=rng, periodic=True)

    if verbose:
        print(f" done ({time.perf_counter()-t_var:.1f}s)", flush=True)

    lb = compute_lambda_bar(R_array, variances)

    # Bootstrap error
    tail = variances[len(variances) // 3:]
    splits = np.array_split(tail, 4)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    lb_err = np.std(boots) / np.sqrt(len(boots)) if len(boots) > 1 else 0.0

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  n={n:3d}: Lambda_bar = {lb:.5f} +/- {lb_err:.5f}  "
              f"(rho={rho:.5f}, {elapsed:.1f}s)\n")

    return lb, lb_err, rho, N_actual


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("  Extended Metallic-Mean Study: Lambda_bar up to n=50")
    print("=" * 70)

    results = {}
    for n in N_LIST:
        lb, lb_err, rho, N_act = compute_lb(n, verbose=True)
        results[n] = {
            'lambda_bar': float(lb),
            'err': float(lb_err),
            'rho': float(rho),
            'N': int(N_act),
        }

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY: Lambda_bar for metallic-mean chains (extended)")
    print("=" * 70)
    print(f"  {'n':>4s}  {'mu_n':>10s}  {'rho':>8s}  {'Lambda_bar':>12s}  "
          f"{'err':>8s}  {'1/3 - Lb':>10s}  {'N':>10s}")
    print("  " + "-" * 72)
    for n in N_LIST:
        r = results[n]
        mu = metallic_mean(n)
        diff = 1/3 - r['lambda_bar']
        print(f"  {n:4d}  {mu:10.4f}  {r['rho']:8.5f}  {r['lambda_bar']:12.5f}  "
              f"{r['err']:8.5f}  {diff:10.5f}  {r['N']:10,d}")
    print(f"  {'URL':>4s}  {'a=1':>10s}  {'---':>8s}  {1/3:12.5f}  "
          f"{'(exact)':>8s}  {0.0:10.5f}  {'---':>10s}")

    # ----------------------------------------------------------------
    # Check: does any n overshoot 1/3?
    # ----------------------------------------------------------------
    overshoots = [n for n in N_LIST if results[n]['lambda_bar'] > 1/3]
    print(f"\n  Values above 1/3: {overshoots if overshoots else 'NONE'}")

    # Monotonicity check
    lbs_ordered = [results[n]['lambda_bar'] for n in N_LIST]
    is_monotone = all(lbs_ordered[i] <= lbs_ordered[i+1] for i in range(len(lbs_ordered)-1))
    print(f"  Monotonically increasing: {is_monotone}")

    # ----------------------------------------------------------------
    # Save to JSON
    # ----------------------------------------------------------------
    out_data = {
        'n_values': N_LIST,
        'lambda_bar': [results[n]['lambda_bar'] for n in N_LIST],
        'err': [results[n]['err'] for n in N_LIST],
        'rho': [results[n]['rho'] for n in N_LIST],
        'N': [results[n]['N'] for n in N_LIST],
        'params': {
            'TARGET_N': TARGET_N,
            'NUM_WINDOWS': NUM_WINDOWS,
            'NUM_R': NUM_R,
            'SEED': SEED,
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Results saved to {OUT_JSON}")
    print("=" * 70)
