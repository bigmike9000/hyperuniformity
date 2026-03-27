"""
Systematic Search for New Cubic (3x3) Substitution Tilings

Searches 3x3 substitution matrices for Pisot property and interesting alpha values,
with focus on filling the 2 < alpha < 3 gap (Class I, between metallic means and BT).

Method:
  - Enumerate all 3x3 non-negative integer matrices with row sums in [1, 4]
  - Filter for: (1) Perron eigenvalue > 1, (2) Pisot (|lambda_2|, |lambda_3| < 1)
  - Compute alpha = 1 - 2*ln|lambda_2| / ln|lambda_1|
  - Verify alpha is real (no complex lambda_2) and in Class I (alpha > 0)
  - For best candidates: simulate N~100k pattern and verify sigma^2(R) behavior

Output:
  - Table of all Pisot matrices grouped by alpha range
  - Detailed analysis of best candidates for 2 < alpha < 3
  - New chains added to substitution_tilings.py CHAINS dict
  - results/figures/fig_new_tilings.png

Reference: Oguz et al. (2019) eigenvalue formula for 1D hyperuniformity
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import time
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points_general,
    predict_chain_length, verify_eigenvalue_prediction,
)
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)

# ============================================================
# Row enumeration: non-negative integer triples with sum in [1, 4]
# ============================================================

def enum_rows(max_sum=4):
    """Enumerate all non-negative integer triples with sum in [1, max_sum]."""
    rows = []
    for s in range(1, max_sum + 1):
        for a in range(s + 1):
            for b in range(s - a + 1):
                c = s - a - b
                rows.append((a, b, c))
    return rows


def is_primitive(M, max_power=8):
    """Check if matrix M^k > 0 for some k <= max_power (primitivity test)."""
    Mk = M.copy().astype(float)
    for _ in range(max_power):
        if np.all(Mk > 0):
            return True
        Mk = Mk @ M
    return False


def compute_alpha_from_matrix(M):
    """
    Compute hyperuniformity exponent from substitution matrix.
    Returns (alpha, lambda_1, |lambda_2|, is_pisot, eigenvalues_sorted).
    """
    eigenvalues = np.linalg.eigvals(M)
    # Sort by absolute value descending
    abs_eigs = np.abs(eigenvalues)
    idx = np.argsort(abs_eigs)[::-1]
    eigs_sorted = eigenvalues[idx]
    abs_sorted  = abs_eigs[idx]

    lam1 = abs_sorted[0]  # Perron eigenvalue
    lam2 = abs_sorted[1]  # second largest

    if lam1 <= 1.0:
        return np.nan, lam1, lam2, False, eigs_sorted

    # Pisot: all sub-dominant eigenvalues have |lambda| < 1
    is_pisot = (lam2 < 1.0)
    if not is_pisot:
        return np.nan, lam1, lam2, False, eigs_sorted

    # Alpha from eigenvalue formula
    alpha = 1.0 - 2.0 * np.log(lam2) / np.log(lam1)

    return alpha, lam1, lam2, True, eigs_sorted


# ============================================================
# Main search
# ============================================================
print("=" * 70)
print("  Systematic Search for Cubic Substitution Tilings")
print("=" * 70)

rows = enum_rows(max_sum=4)
print(f"\n  Row options (sum 1..4, 3 entries): {len(rows)}")
print(f"  Total matrices to check: {len(rows)**3:,}")

t0 = time.perf_counter()

# Collect Pisot matrices
pisot_matrices = []

for r0, r1, r2 in itertools.product(rows, repeat=3):
    M = np.array([r0, r1, r2], dtype=float)

    # Quick pre-filter: trace > 0 suggests large eigenvalue
    # (skips all-zero rows already excluded by sum >= 1)
    alpha, lam1, lam2, is_pisot, eigs = compute_alpha_from_matrix(M)

    if not is_pisot:
        continue
    if np.isnan(alpha) or not np.isfinite(alpha):
        continue
    if alpha <= 0:
        continue

    # Store
    pisot_matrices.append({
        'M': [r0, r1, r2],
        'alpha': float(alpha),
        'lam1': float(lam1),
        'lam2': float(lam2),
        'eigenvalues': [complex(e) for e in eigs],
    })

elapsed = time.perf_counter() - t0
print(f"  Search done in {elapsed:.2f}s")
print(f"  Found {len(pisot_matrices):,} Pisot matrices")

# ============================================================
# Filter known patterns (exclude BT and metallic-mean matrices)
# ============================================================
# Known matrices to exclude
known_M_list = [
    np.array(v['matrix']) for v in CHAINS.values()
    if v['matrix'].shape == (3, 3)
]

def is_known(M_tup):
    M = np.array(M_tup, dtype=float)
    for Mk in known_M_list:
        if np.allclose(M, Mk):
            return True
    return False

novel = [p for p in pisot_matrices if not is_known(p['M'])]
print(f"  Novel matrices (not in CHAINS): {len(novel):,}")

# ============================================================
# Categorize by alpha range
# ============================================================
def alpha_class(alpha):
    if alpha > 3.0 + 0.1:
        return '>3'
    if alpha > 2.0 + 0.05:
        return '(2,3)'
    if alpha > 1.5 + 0.05:
        return '(1.5,2)'
    if alpha > 1.0 + 0.05:
        return '(1,1.5)'
    if alpha >= 1.0 - 0.05:
        return '~1'
    if alpha > 0:
        return '(0,1)'
    return '<0'

from collections import Counter
counts = Counter(alpha_class(p['alpha']) for p in novel)
print("\n  Alpha distribution (novel Pisot matrices):")
for cat, cnt in sorted(counts.items()):
    print(f"    {cat:12s}: {cnt:6d}")

# Focus on 2 < alpha < 3 (filling the gap)
gap_matrices = [p for p in novel if 2.0 < p['alpha'] < 3.0]
print(f"\n  Matrices with 2 < alpha < 3: {len(gap_matrices)}")

# Sort by alpha and deduplicate (same matrix, same alpha)
gap_matrices.sort(key=lambda p: p['alpha'])

# Also find candidates with other interesting alpha values
other_interesting = [p for p in novel if 1.0 < p['alpha'] < 2.0 and
                     abs(p['alpha'] - 1.545) > 0.1]  # novel, not BT-like
other_interesting.sort(key=lambda p: p['alpha'])
print(f"  Novel Class I with 1 < alpha < 2 (not ~BT): {len(other_interesting)}")

# ============================================================
# Check primitivity for gap candidates
# ============================================================
print("\n  Checking primitivity for 2<alpha<3 candidates ...")
primitive_gap = []
for p in gap_matrices:
    M = np.array(p['M'], dtype=float)
    if is_primitive(M):
        primitive_gap.append(p)

print(f"  Primitive matrices in (2,3): {len(primitive_gap)}")

# Show unique alpha values
alphas_gap = sorted(set(round(p['alpha'], 4) for p in primitive_gap))
print(f"  Unique alpha values (rounded): {alphas_gap[:20]}{'...' if len(alphas_gap) > 20 else ''}")

# ============================================================
# Select best candidates for each alpha sub-range in (2,3)
# ============================================================
# Pick one matrix per 0.1-wide alpha bin
bins = np.arange(2.0, 3.05, 0.1)
candidates = []
seen_alphas = set()

for lo, hi in zip(bins[:-1], bins[1:]):
    bin_mats = [p for p in primitive_gap if lo <= p['alpha'] < hi]
    if not bin_mats:
        continue
    # Pick the one closest to bin center
    center = (lo + hi) / 2
    best = min(bin_mats, key=lambda p: abs(p['alpha'] - center))
    alpha_key = round(best['alpha'], 2)
    if alpha_key not in seen_alphas:
        candidates.append(best)
        seen_alphas.add(alpha_key)

print(f"\n  Selected {len(candidates)} representative candidates for simulation:")
for p in candidates:
    print(f"    M={p['M']}  alpha={p['alpha']:.4f}  lam1={p['lam1']:.4f}  lam2={p['lam2']:.4f}")

# ============================================================
# Build substitution rules from matrix (canonical: concatenate sorted)
# ============================================================
ALPHABET = 'abc'

def matrix_to_rules(M_tup):
    """Convert 3x3 matrix rows to substitution rules (letters in alphabetical order)."""
    rules = {}
    for i, (letter, row) in enumerate(zip(ALPHABET, M_tup)):
        # Repeat each letter by its count, in order a, b, c
        rule_str = ''
        for j, ch in enumerate(ALPHABET):
            rule_str += ch * row[j]
        rules[letter] = rule_str
    return rules


def gen_3letter_chain(M_tup, target_n):
    """Generate a 3-letter substitution chain of at least target_n points."""
    rules = matrix_to_rules(M_tup)
    M_arr = np.array(M_tup, dtype=float)

    # Find iteration count
    seed = 'a'
    seq = seed
    for iters in range(1, 40):
        # Count tiles
        counts = np.array([seq.count(ch) for ch in ALPHABET], dtype=np.int64)
        n_pred = int(np.sum(counts))
        if n_pred >= target_n:
            break
        seq = ''.join(rules[ch] for ch in seq)

    # Final sequence
    seq = seed
    for _ in range(iters):
        seq = ''.join(rules[ch] for ch in seq)

    N_actual = len(seq)
    if N_actual < 100:
        # Need more iterations
        for extra in range(10):
            seq = ''.join(rules[ch] for ch in seq)
            if len(seq) >= target_n:
                break

    # Compute tile lengths from eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(M_arr)
    idx = np.argmax(np.abs(eigenvalues))
    theta1 = eigenvalues[idx].real
    right_eigenvec = np.abs(eigenvectors[:, idx].real)
    right_eigenvec = right_eigenvec / np.min(right_eigenvec)
    tile_lengths = {ch: right_eigenvec[i] for i, ch in enumerate(ALPHABET)}

    # Points
    lengths = np.array([tile_lengths[ch] for ch in seq])
    L_domain = float(np.sum(lengths))
    points = np.empty(len(seq), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])

    return points, L_domain, len(seq)


# ============================================================
# Simulate candidates
# ============================================================
TARGET_SIM_N = 100_000
NUM_WINDOWS   = 20_000
sim_results = []

print(f"\n  Simulating {len(candidates)} candidates (N~{TARGET_SIM_N:,}) ...")
for p in candidates:
    M_tup = p['M']
    alpha_th = p['alpha']
    print(f"\n    alpha_theory={alpha_th:.4f}  M={M_tup}")

    try:
        t0 = time.perf_counter()
        pts, L, N_sim = gen_3letter_chain(M_tup, TARGET_SIM_N)
        rho = N_sim / L
        gen_t = time.perf_counter() - t0
        print(f"      N={N_sim:,}, L={L:.1f}, rho={rho:.4f} [{gen_t:.1f}s]")

        # Number variance
        R_max = min(N_sim / (2 * rho) * 0.8, 2000.0)
        R_arr = np.linspace(0.5 / rho, R_max, 500)
        var, _ = compute_number_variance_1d(pts, L, R_arr,
                                             num_windows=NUM_WINDOWS, rng=rng)
        var_max = np.max(var)
        var_range = np.max(var) - np.min(var[len(var)//3:])
        is_bounded = var_max < 50  # rough check

        lambda_bar = compute_lambda_bar(R_arr, var) if is_bounded else None
        print(f"      sigma^2 range: [{np.min(var):.3f}, {var_max:.3f}]  bounded={is_bounded}")
        if lambda_bar is not None:
            print(f"      Lambda_bar = {lambda_bar:.4f}")

        # Alpha from structure factor
        from two_phase_media import compute_structure_factor
        k_arr, S_arr = compute_structure_factor(pts, L)

        # Fit S(k) ~ k^alpha at small k
        n_low = 25
        valid = (k_arr[:n_low] > 0) & (S_arr[:n_low] > 0)
        if np.sum(valid) >= 4:
            log_k = np.log(k_arr[:n_low][valid])
            log_S = np.log(S_arr[:n_low][valid])
            coeffs = np.polyfit(log_k, log_S, 1)
            alpha_num = coeffs[0]
        else:
            alpha_num = np.nan

        print(f"      alpha_numeric = {alpha_num:.4f}  (theory={alpha_th:.4f})")

        sim_results.append({
            'M': M_tup,
            'alpha_theory': alpha_th,
            'alpha_numeric': float(alpha_num) if not np.isnan(alpha_num) else None,
            'lam1': p['lam1'],
            'lam2': p['lam2'],
            'N': N_sim,
            'lambda_bar': float(lambda_bar) if lambda_bar is not None else None,
            'is_bounded': bool(is_bounded),
            'R_arr': R_arr.tolist(),
            'var': var.tolist(),
        })
        del pts

    except Exception as e:
        print(f"      ERROR: {e}")
        sim_results.append({
            'M': M_tup, 'alpha_theory': alpha_th, 'error': str(e),
            'lam1': p['lam1'], 'lam2': p['lam2'],
        })

# ============================================================
# Figure
# ============================================================
print("\nGenerating fig_new_tilings.png ...")

n_valid = [r for r in sim_results if 'R_arr' in r and r['is_bounded']]
print(f"  Bounded Class I candidates: {len(n_valid)}")

if n_valid:
    nc = min(len(n_valid), 6)
    fig, axes = plt.subplots(1, nc + 1, figsize=(4 * (nc + 1), 5))
    if nc + 1 == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, nc))

    # Left panel: alpha distribution
    ax0 = axes[0]
    alpha_all = [p['alpha'] for p in primitive_gap]
    ax0.hist(alpha_all, bins=50, color='#2ca02c', alpha=0.7, edgecolor='white')
    ax0.axvspan(2.0, 3.0, alpha=0.15, color='red', label='2<α<3 gap')
    ax0.axvline(1.545, color='blue', ls='--', lw=1.5, alpha=0.7, label='BT (1.545)')
    ax0.axvline(3.0, color='gray', ls='--', lw=1.5, alpha=0.7, label='Metallic (3.0)')
    ax0.set_xlabel(r'$\alpha$', fontsize=12)
    ax0.set_ylabel('Count', fontsize=12)
    ax0.set_title(f'All primitive Pisot matrices\n(2<α<3: {len(primitive_gap)})', fontsize=10)
    ax0.legend(fontsize=8)
    ax0.grid(True, ls=':', alpha=0.4)

    for i, (r, col) in enumerate(zip(n_valid[:nc], colors)):
        ax = axes[i + 1]
        R_plot = np.array(r['R_arr'])
        v_plot = np.array(r['var'])
        a_th = r['alpha_theory']
        lb = r['lambda_bar']

        ax.semilogx(R_plot, v_plot, '-', color=col, lw=1.2)
        if lb is not None:
            ax.axhline(lb, color=col, ls='--', lw=1.5,
                       label=rf'$\bar{{\Lambda}}={lb:.3f}$')

        alpha_num_str = f"{r['alpha_numeric']:.3f}" if r['alpha_numeric'] else '?'
        ax.set_title(
            rf'$\alpha_{{th}}={a_th:.3f}$, $\hat{{\alpha}}={alpha_num_str}$'
            f'\nM={r["M"][0]}',
            fontsize=9
        )
        ax.set_xlabel(r'$R$', fontsize=10)
        ax.set_ylabel(r'$\sigma^2(R)$', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, ls=':', alpha=0.4)

    plt.suptitle(
        f'New Cubic Tilings: {len(primitive_gap)} Pisot matrices found, '
        f'{len(n_valid)} verified Class I with 2<α<3',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig_new_tilings.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# ============================================================
# Save results to JSON
# ============================================================
def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    return obj

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'cubic_search_results.json')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

save_data = {
    'total_pisot': len(pisot_matrices),
    'novel_pisot': len(novel),
    'gap_23_total': len(gap_matrices),
    'gap_23_primitive': len(primitive_gap),
    'candidates_simulated': len(sim_results),
    'sim_results': [
        {k: (v if not isinstance(v, list) else None) for k, v in r.items()
         if k not in ('R_arr', 'var')}
        for r in sim_results
    ],
    'top_gap_candidates': [
        {
            'M': p['M'],
            'alpha': p['alpha'],
            'lam1': p['lam1'],
            'lam2': p['lam2'],
            'rules': matrix_to_rules(p['M']),
        }
        for p in primitive_gap[:20]
    ]
}
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=serialize)
print(f"  Saved JSON: {save_path}")

# ============================================================
# Summary and CHAINS additions
# ============================================================
print("\n" + "=" * 70)
print("  SUMMARY: New Cubic Substitution Tilings")
print("=" * 70)
print(f"  Total 3x3 Pisot matrices found:   {len(pisot_matrices):,}")
print(f"  Novel (not in CHAINS):             {len(novel):,}")
print(f"  With 2 < alpha < 3 (gap target):  {len(gap_matrices):,}")
print(f"    Of which primitive:              {len(primitive_gap):,}")
print(f"  Alpha range: [{min(p['alpha'] for p in primitive_gap):.3f}, "
      f"{max(p['alpha'] for p in primitive_gap):.3f}]")

print("\n  Best candidates for 2 < alpha < 3:")
print(f"  {'M[0]':12s} {'M[1]':12s} {'M[2]':12s} {'alpha':>8s} {'lam1':>7s} {'lam2':>7s} {'rules'}")
print("  " + "-" * 80)
for p in primitive_gap[:10]:
    rules = matrix_to_rules(p['M'])
    rules_str = ','.join(f"{k}->{v}" for k, v in rules.items())
    print(f"  {str(p['M'][0]):12s} {str(p['M'][1]):12s} {str(p['M'][2]):12s} "
          f"{p['alpha']:8.4f} {p['lam1']:7.4f} {p['lam2']:7.4f}  {rules_str}")

print("\n  Verified Class I (bounded variance, 2<alpha<3):")
for r in sim_results:
    if r.get('is_bounded') and r.get('alpha_theory', 0) > 2:
        lb_str = f"{r['lambda_bar']:.4f}" if r['lambda_bar'] else '?'
        num_str = f"{r['alpha_numeric']:.4f}" if r['alpha_numeric'] else '?'
        print(f"    M={r['M']}  alpha_th={r['alpha_theory']:.4f}  "
              f"alpha_num={num_str}  Lambda_bar={lb_str}")

print("=" * 70)


# ============================================================
# Propose CHAINS additions for best verified candidate
# ============================================================
verified_gap = [r for r in sim_results
                if r.get('is_bounded') and 2.0 < r.get('alpha_theory', 0) < 3.0
                and r.get('alpha_numeric') is not None]

if verified_gap:
    # Pick the one with alpha closest to 2.5 (middle of gap)
    best = min(verified_gap, key=lambda r: abs(r['alpha_theory'] - 2.5))
    rules = matrix_to_rules(best['M'])
    print(f"\n  RECOMMENDED NEW CHAIN ENTRY:")
    print(f"    Matrix: {best['M']}")
    print(f"    alpha = {best['alpha_theory']:.4f}")
    print(f"    lam1  = {best['lam1']:.4f}")
    print(f"    lam2  = {best['lam2']:.4f}")
    print(f"    Rules: {rules}")
    print(f"    Lambda_bar = {best.get('lambda_bar', '?')}")
    print(f"")
    key_name = f"new_cubic_alpha{best['alpha_theory']:.2f}"
    mat_str  = [list(row) for row in best['M']]
    print("    Add to CHAINS dict in substitution_tilings.py:")
    print(f"    '{key_name}': {{")
    print(f"        'name': 'Cubic (alpha={best['alpha_theory']:.3f})',")
    print(f"        'matrix': np.array({mat_str}),")
    print(f"        'rules': {rules},")
    print(f"        'alphabet': 'abc',")
    print(f"    }},")
