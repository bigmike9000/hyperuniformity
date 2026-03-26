"""
Compute Lambda_bar for all 18 distinct lambda_1 values among the 504 primitive
3x3 unimodular substitution matrices with alpha=2 (complex conjugate eigenvalue pairs).

For 3x3 unimodular (|det M|=1) Pisot matrices with a complex conjugate pair of
subdominant eigenvalues, the constraint |lambda_1| * |lambda_2|^2 = 1 forces
|lambda_2| = 1/sqrt(lambda_1), giving alpha = 1 - 2*ln(1/sqrt(lam1))/ln(lam1)
                                             = 1 - 2*(-0.5) = 2 exactly.

Steps:
  1. Enumerate all 3x3 non-neg integer matrices with row sums in [1,4]
  2. Filter: Pisot, primitive, |det|=1, complex conjugate pair, alpha~2
  3. Group by distinct lambda_1 (rounded to 6 decimals)
  4. For each group, pick one representative and compute Lambda_bar
  5. Save results to results/alpha2_lambda_bar_table.json
"""

import os
import sys
import time
import json
import itertools
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from substitution_tilings import CHAINS
from quasicrystal_variance import compute_number_variance_1d

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHABET = 'abc'
rng = np.random.default_rng(42)


# ================================================================
# Step 1: Find all 504 matrices
# ================================================================

def enum_rows(max_sum=4):
    rows = []
    for s in range(1, max_sum + 1):
        for a in range(s + 1):
            for b in range(s - a + 1):
                c = s - a - b
                rows.append((a, b, c))
    return rows


def is_primitive(M, max_power=10):
    Mk = M.copy().astype(float)
    for _ in range(max_power):
        if np.all(Mk > 0):
            return True
        Mk = Mk @ M
    return False


def has_complex_conjugate_pair(eigs, tol=1e-8):
    """Check if the subdominant eigenvalues form a complex conjugate pair."""
    # Sort by |eigenvalue| descending
    abs_eigs = np.abs(eigs)
    idx = np.argsort(abs_eigs)[::-1]
    sorted_eigs = eigs[idx]
    # Check if eigenvalues 2 and 3 (0-indexed: 1 and 2) are complex conjugates
    e2, e3 = sorted_eigs[1], sorted_eigs[2]
    if abs(e2.imag) < tol:
        return False  # both real
    # Check conjugate: e2 = conj(e3)
    return abs(e2 - np.conj(e3)) < tol


print("=" * 70)
print("  Alpha=2 Lambda_bar Survey: 3x3 Unimodular Pisot Matrices")
print("=" * 70)

rows = enum_rows(max_sum=4)
print(f"\n  Row options: {len(rows)}, total matrices: {len(rows)**3:,}")

t0 = time.perf_counter()
alpha2_matrices = []

for r0, r1, r2 in itertools.product(rows, repeat=3):
    M = np.array([r0, r1, r2], dtype=float)

    # Quick checks
    det_val = np.linalg.det(M)
    if abs(abs(det_val) - 1.0) > 0.1:
        continue

    eigenvalues = np.linalg.eigvals(M)
    abs_eigs = np.abs(eigenvalues)
    idx = np.argsort(abs_eigs)[::-1]
    eigs_sorted = eigenvalues[idx]
    abs_sorted = abs_eigs[idx]

    lam1 = abs_sorted[0]
    lam2 = abs_sorted[1]

    # Pisot check
    if lam1 <= 1.0 or lam2 >= 1.0:
        continue

    # Complex conjugate pair check
    if not has_complex_conjugate_pair(eigenvalues):
        continue

    # Alpha should be ~2
    alpha = 1.0 - 2.0 * np.log(lam2) / np.log(lam1)
    if abs(alpha - 2.0) > 0.01:
        continue

    # Primitivity check
    if not is_primitive(M):
        continue

    alpha2_matrices.append({
        'M': [list(map(int, r)) for r in [r0, r1, r2]],
        'lam1': float(lam1),
        'lam2': float(lam2),
        'det': float(det_val),
        'eigenvalues': eigs_sorted.tolist(),
    })

elapsed = time.perf_counter() - t0
print(f"  Search done in {elapsed:.1f}s")
print(f"  Found {len(alpha2_matrices)} primitive unimodular matrices with alpha=2")

# ================================================================
# Step 2: Group by distinct lambda_1
# ================================================================

# Round lambda_1 to 6 decimals for grouping
groups = {}
for p in alpha2_matrices:
    key = round(p['lam1'], 6)
    if key not in groups:
        groups[key] = []
    groups[key].append(p)

sorted_lam1s = sorted(groups.keys())
print(f"\n  Distinct lambda_1 values: {len(sorted_lam1s)}")
for lam1_val in sorted_lam1s:
    print(f"    lambda_1 = {lam1_val:.6f}  ({len(groups[lam1_val])} matrices)")


# ================================================================
# Step 3: Build substitution rules and generate chains
# ================================================================

def matrix_to_rules(M_list):
    """Convert 3x3 matrix rows to substitution rules."""
    rules = {}
    for i, (letter, row) in enumerate(zip(ALPHABET, M_list)):
        rule_str = ''
        for j, ch in enumerate(ALPHABET):
            rule_str += ch * row[j]
        rules[letter] = rule_str
    return rules


def gen_chain(M_list, target_n=500_000):
    """Generate a substitution chain and return points, L_domain, N."""
    rules = matrix_to_rules(M_list)
    M_arr = np.array(M_list, dtype=float)

    # Compute tile lengths from eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(M_arr)
    idx = np.argmax(np.abs(eigenvalues))
    theta1 = eigenvalues[idx].real
    right_eigenvec = np.abs(eigenvectors[:, idx].real)
    right_eigenvec = right_eigenvec / np.min(right_eigenvec)
    tile_lengths = {ch: right_eigenvec[i] for i, ch in enumerate(ALPHABET)}

    # Iterate until we reach target_n
    seq = 'a'
    for _ in range(50):
        if len(seq) >= target_n:
            break
        seq = ''.join(rules[ch] for ch in seq)

    if len(seq) < 1000:
        raise ValueError(f"Chain too short after 50 iterations: {len(seq)}")

    # Convert to points
    lengths = np.array([tile_lengths[ch] for ch in seq])
    L_domain = float(np.sum(lengths))
    points = np.empty(len(seq), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])

    return points, L_domain, len(seq), tile_lengths, theta1


# ================================================================
# Step 4: Compute Lambda_bar for each representative
# ================================================================

NUM_WINDOWS = 25_000
N_R_VALUES  = 1000

print(f"\n  Computing Lambda_bar for {len(sorted_lam1s)} representatives ...")
print(f"  (N ~ 500k, {NUM_WINDOWS} windows, {N_R_VALUES} R values)")
print()

results_table = []

for i, lam1_val in enumerate(sorted_lam1s):
    group = groups[lam1_val]
    # Pick the first matrix as representative
    rep = group[0]
    M_list = rep['M']
    rules = matrix_to_rules(M_list)

    rules_str = ', '.join(f"{k} -> {v}" for k, v in rules.items())
    print(f"  [{i+1}/{len(sorted_lam1s)}] lambda_1 = {lam1_val:.6f}  "
          f"({len(group)} matrices)")
    print(f"    Matrix: {M_list}")
    print(f"    Rules:  {rules_str}")

    try:
        t0 = time.perf_counter()
        points, L_domain, N, tile_lengths, theta1 = gen_chain(M_list, target_n=500_000)
        rho = N / L_domain
        gen_time = time.perf_counter() - t0
        print(f"    N = {N:,}, L = {L_domain:.1f}, rho = {rho:.6f}  [{gen_time:.1f}s gen]")

        # R range: from 1 mean spacing to L/4, using periodic BCs
        mean_spacing = 1.0 / rho
        R_max = min(300 * mean_spacing, L_domain / 4)
        R_array = np.linspace(mean_spacing, R_max, N_R_VALUES)

        t1 = time.perf_counter()
        variances, mean_counts = compute_number_variance_1d(
            points, L_domain, R_array, num_windows=NUM_WINDOWS, rng=rng,
            periodic=True)
        var_time = time.perf_counter() - t1
        print(f"    Variance computed in {var_time:.1f}s")

        # Lambda_bar: mean over last 2/3 of R range
        start = len(R_array) // 3
        lambda_bar = float(np.mean(variances[start:]))
        lambda_bar_err = float(np.std(variances[start:]))
        var_max = float(np.max(variances))

        print(f"    Lambda_bar = {lambda_bar:.4f} +/- {lambda_bar_err:.4f}")
        print(f"    sigma^2 range: [{np.min(variances):.4f}, {var_max:.4f}]")

        result = {
            'lam1': lam1_val,
            'lam2': float(rep['lam2']),
            'det': float(rep['det']),
            'n_matrices': len(group),
            'representative_M': M_list,
            'rules': {k: v for k, v in rules.items()},
            'N': N,
            'rho': float(rho),
            'lambda_bar': lambda_bar,
            'lambda_bar_err': lambda_bar_err,
            'var_max': var_max,
            'tile_lengths': {k: float(v) for k, v in tile_lengths.items()},
        }
        results_table.append(result)

        # Free memory
        del points, variances, mean_counts

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        results_table.append({
            'lam1': lam1_val,
            'n_matrices': len(group),
            'representative_M': M_list,
            'rules': {k: v for k, v in rules.items()},
            'error': str(e),
        })

    print()


# ================================================================
# Step 5: Summary table
# ================================================================
print("\n" + "=" * 90)
print("  RESULTS: Lambda_bar for alpha=2 unimodular 3x3 Pisot matrices")
print("=" * 90)
print(f"  {'lambda_1':>10s}  {'|lambda_2|':>10s}  {'N_mat':>5s}  "
      f"{'Lambda_bar':>10s}  {'err':>8s}  {'Rules'}")
print("  " + "-" * 85)

for r in sorted(results_table, key=lambda x: x['lam1']):
    if 'error' in r:
        print(f"  {r['lam1']:10.6f}  {'---':>10s}  {r['n_matrices']:5d}  "
              f"{'ERROR':>10s}  {'':>8s}  {r.get('error', '')[:40]}")
    else:
        rules_str = ', '.join(f"{k}->{v}" for k, v in r['rules'].items())
        print(f"  {r['lam1']:10.6f}  {r['lam2']:10.6f}  {r['n_matrices']:5d}  "
              f"{r['lambda_bar']:10.4f}  {r['lambda_bar_err']:8.4f}  {rules_str}")

print("=" * 90)

# Also show the known cubic_alpha2 for reference
print("\n  Reference: cubic_alpha2 in CHAINS has lambda_1 ~ 2.1479, Lambda_bar ~ 0.275")
print(f"  Integer lattice: Lambda_bar = 1/6 = {1/6:.6f}")
print(f"  URL (a=1, cloaked): Lambda_bar = 1/3 = {1/3:.6f}")

# ================================================================
# Step 6: Save to JSON
# ================================================================
save_path = os.path.join(RESULTS_DIR, 'alpha2_lambda_bar_table.json')

save_data = {
    'description': 'Lambda_bar for 3x3 unimodular Pisot matrices with alpha=2 exactly '
                   '(complex conjugate subdominant eigenvalue pair)',
    'total_alpha2_matrices': len(alpha2_matrices),
    'distinct_lam1_values': len(sorted_lam1s),
    'computation_params': {
        'target_N': 500_000,
        'num_windows': NUM_WINDOWS,
        'n_R_values': N_R_VALUES,
        'averaging': 'mean over last 2/3 of R range',
    },
    'results': sorted(results_table, key=lambda x: x['lam1']),
}

# Custom serializer for complex/numpy types
def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=serialize)
print(f"\n  Saved: {save_path}")
print("  Done.")
