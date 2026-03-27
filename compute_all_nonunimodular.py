"""
Compute Lambda_bar for all 28 non-unimodular 2x2 Pisot matrices with alpha in (2,3).

These are 2x2 non-negative integer matrices M with:
  - |det M| = 2
  - |lambda_2| < 1  (Pisot property)
  - alpha = 3 - 2*ln|det M|/ln(lambda_1)  in (2, 3)

Row sums up to 10 are searched.

Output: results/nonunimodular_all.json
"""

import os
import sys
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, 'nonunimodular_all.json')

TARGET_N = 500_000
NUM_WINDOWS = 15_000
NUM_R = 400
SEED = 2028


def find_all_nonunimodular_2x2(max_row_sum=10):
    """Find all 2x2 non-negative integer matrices with |det|=2, Pisot, alpha in (2,3)."""
    results = []
    for a in range(0, max_row_sum + 1):
        for b in range(0, max_row_sum + 1 - a):
            for c in range(0, max_row_sum + 1):
                for d in range(0, max_row_sum + 1 - c):
                    det = a * d - b * c
                    if abs(det) != 2:
                        continue
                    # Eigenvalues
                    tr = a + d
                    disc = tr**2 - 4 * det
                    if disc < 0:
                        continue  # complex eigenvalues
                    sqrt_disc = np.sqrt(disc)
                    lam1 = (tr + sqrt_disc) / 2
                    lam2 = (tr - sqrt_disc) / 2
                    if lam1 <= 1:
                        continue  # not a valid PF eigenvalue
                    if abs(lam2) >= 1:
                        continue  # not Pisot
                    if lam1 <= 0:
                        continue
                    alpha = 1.0 - 2.0 * np.log(abs(lam2)) / np.log(lam1)
                    if alpha <= 2.0 or alpha >= 3.0:
                        continue
                    results.append({
                        'M': [[a, b], [c, d]],
                        'det': det,
                        'lambda1': float(lam1),
                        'lambda2': float(lam2),
                        'alpha': float(alpha),
                        'row_sums': [a + b, c + d],
                    })
    return results


def generate_chain_2x2(M, target_n):
    """Generate substitution chain from 2x2 matrix.

    M = [[a,b],[c,d]] means:
      tile 'a' -> 'a'^a 'b'^b  (a copies of tile a, then b copies of tile b)
      tile 'b' -> 'a'^c 'b'^d
    """
    a_row = M[0]  # [num_a, num_b] produced by tile 'a'
    b_row = M[1]  # [num_a, num_b] produced by tile 'b'

    rule_a = 'a' * a_row[0] + 'b' * a_row[1]
    rule_b = 'a' * b_row[0] + 'b' * b_row[1]
    rules = {'a': rule_a, 'b': rule_b}

    seq = 'a'
    for _ in range(200):
        if len(seq) >= target_n:
            break
        seq = ''.join(rules[ch] for ch in seq)

    return seq


def sequence_to_points_2x2(seq, lam1):
    """Convert sequence to points. Tile 'a' has length 1, tile 'b' has length lambda1."""
    arr = np.frombuffer(seq.encode(), dtype='S1')
    lengths = np.where(arr == b'b', lam1, 1.0)
    L_domain = float(np.sum(lengths))
    points = np.empty(len(seq), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])
    return points, L_domain


def compute_lb_for_matrix(info, rng):
    """Compute Lambda_bar for one non-unimodular matrix."""
    M = info['M']
    lam1 = info['lambda1']
    alpha = info['alpha']

    t0 = time.perf_counter()
    seq = generate_chain_2x2(M, TARGET_N)
    N = len(seq)

    if N < 1000:
        return None  # too few points

    points, L_domain = sequence_to_points_2x2(seq, lam1)
    del seq
    rho = N / L_domain

    mean_spacing = 1.0 / rho
    R_max = min(200 * mean_spacing, L_domain / 4)
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
    return {
        'lambda_bar': float(lb),
        'err': float(lb_err),
        'rho': float(rho),
        'N': int(N),
        'elapsed': float(elapsed),
    }


def main():
    print("=" * 70)
    print("  Finding all 2x2 non-unimodular Pisot matrices with alpha in (2,3)")
    print("=" * 70)

    matrices = find_all_nonunimodular_2x2(max_row_sum=10)
    print(f"\n  Found {len(matrices)} matrices with |det|=2, alpha in (2,3)")

    # Sort by alpha
    matrices.sort(key=lambda x: x['alpha'])

    # Print summary
    print(f"\n  {'#':>3s}  {'M':>20s}  {'det':>4s}  {'lam1':>8s}  {'lam2':>8s}  "
          f"{'alpha':>8s}  {'rows':>8s}")
    print("  " + "-" * 70)
    for i, m in enumerate(matrices):
        M = m['M']
        print(f"  {i+1:3d}  [{M[0][0]},{M[0][1]};{M[1][0]},{M[1][1]}]"
              f"{'':>12s}{m['det']:4d}  {m['lambda1']:8.4f}  {m['lambda2']:8.4f}  "
              f"{m['alpha']:8.4f}  ({m['row_sums'][0]},{m['row_sums'][1]})")

    # Compute Lambda_bar for each
    print(f"\n\nComputing Lambda_bar for each matrix (TARGET_N={TARGET_N:,})...")
    rng = np.random.default_rng(SEED)

    for i, m in enumerate(matrices):
        M = m['M']
        print(f"\n  [{i+1}/{len(matrices)}] M=[{M[0]},{M[1]}], alpha={m['alpha']:.4f}...",
              end='', flush=True)
        result = compute_lb_for_matrix(m, rng)
        if result is None:
            print(" SKIPPED (too few points)")
            m['result'] = None
        else:
            m['result'] = result
            print(f" Lb={result['lambda_bar']:.4f}+/-{result['err']:.4f} "
                  f"(N={result['N']:,}, {result['elapsed']:.1f}s)")

    # Summary table
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Lambda_bar for non-unimodular chains")
    print("=" * 70)
    print(f"  {'#':>3s}  {'M':>18s}  {'alpha':>7s}  {'Lambda_bar':>12s}  "
          f"{'err':>8s}  {'N':>10s}")
    print("  " + "-" * 65)
    for i, m in enumerate(matrices):
        M = m['M']
        r = m.get('result')
        if r:
            print(f"  {i+1:3d}  [{M[0][0]},{M[0][1]};{M[1][0]},{M[1][1]}]"
                  f"{'':>8s}{m['alpha']:7.4f}  {r['lambda_bar']:12.5f}  "
                  f"{r['err']:8.5f}  {r['N']:10,d}")

    # Save
    out_data = {
        'matrices': matrices,
        'params': {'TARGET_N': TARGET_N, 'NUM_WINDOWS': NUM_WINDOWS,
                   'NUM_R': NUM_R, 'SEED': SEED},
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_JSON}")


if __name__ == '__main__':
    main()
