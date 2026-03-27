"""
Metallic-mean convergence extended to n=100.

Extends the n=1..50 computation to n=100 to verify Lb → 1/3.
Uses same infrastructure as metallic_mean_extended.py.

Output: results/metallic_n100.json
"""

import os
import sys
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

# n values: fill in gaps + extend to 100
N_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

TARGET_N = 300_000
NUM_WINDOWS = 15_000
NUM_R = 400
SEED = 2027

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, 'metallic_n100.json')

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

    # Bootstrap error
    tail = variances[len(variances) // 3:]
    splits = np.array_split(tail, 4)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    lb_err = np.std(boots) / np.sqrt(len(boots)) if len(boots) > 1 else 0.0

    elapsed = time.perf_counter() - t0
    print(f"  Lb={lb:.5f}±{lb_err:.5f}  ({elapsed:.1f}s)")

    return {
        'lambda_bar': float(lb),
        'err': float(lb_err),
        'rho': float(rho),
        'N': int(N_actual),
    }


if __name__ == '__main__':
    print("=" * 70)
    print("  Metallic-Mean Convergence: n=1 to 100")
    print("=" * 70)

    results = {}
    for n in N_LIST:
        results[n] = compute_lb(n)

    # Summary
    print("\n" + "=" * 70)
    print(f"  {'n':>4s}  {'mu_n':>10s}  {'Lambda_bar':>12s}  "
          f"{'err':>8s}  {'1/3 - Lb':>10s}")
    print("  " + "-" * 50)
    for n in N_LIST:
        r = results[n]
        mu = metallic_mean(n)
        diff = 1/3 - r['lambda_bar']
        print(f"  {n:4d}  {mu:10.4f}  {r['lambda_bar']:12.5f}  "
              f"{r['err']:8.5f}  {diff:10.5f}")
    print(f"  {'inf':>4s}  {'inf':>10s}  {1/3:12.5f}  "
          f"{'exact':>8s}  {0.0:10.5f}")

    # Checks
    overshoots = [n for n in N_LIST if results[n]['lambda_bar'] > 1/3 + results[n]['err']]
    print(f"\n  Overshoot 1/3 (beyond error): {overshoots if overshoots else 'NONE'}")

    lbs = [results[n]['lambda_bar'] for n in N_LIST]
    monotone = all(lbs[i] <= lbs[i+1] + 0.005 for i in range(len(lbs)-1))
    print(f"  Approximately monotone: {monotone}")

    # Save
    out_data = {
        'n_values': N_LIST,
        'results': {str(n): results[n] for n in N_LIST},
        'params': {'TARGET_N': TARGET_N, 'NUM_WINDOWS': NUM_WINDOWS,
                   'NUM_R': NUM_R, 'SEED': SEED},
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Saved: {OUT_JSON}")
