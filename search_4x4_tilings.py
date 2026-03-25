"""
Search for 4x4 Non-Negative Integer Substitution Matrices with alpha in (2, 3)

Background:
  For 3x3 unimodular matrices, complex conjugate eigenvalue pairs force alpha=2
  exactly (|lambda_1||lambda_2|^2 = 1 => |lambda_2| = 1/sqrt(lambda_1) => alpha=2).
  An exhaustive search of 3x3 matrices with row sums <= 4 found NO alpha in (2,3).

  For 4x4 matrices, there can be:
    - Two real sub-dominant eigenvalues (distinct)
    - One real + one complex conjugate pair
    - Two complex conjugate pairs
  Only the first two cases can give alpha != 2 or 3, potentially filling (2,3).

Method:
  - Enumerate all 4x4 non-negative integer matrices with row sums <= max_row_sum
  - For each matrix: compute eigenvalues, check Pisot property, compute alpha
  - Record all Pisot matrices; highlight any with alpha in (2, 3)

Output:
  - results/4x4_search_results.json
  - results/figures/fig_4x4_alpha_dist.png

Reference: Oguz et al. (2019), eigenvalue formula for 1D hyperuniformity
"""

import os
import sys
import time
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'figures')
DATA_DIR    = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Row enumeration: non-negative integer 4-tuples with sum in [1, max_sum]
# ----------------------------------------------------------------

def enum_rows_4(max_sum=2):
    """
    Enumerate all non-negative integer 4-tuples (a,b,c,d) with
    1 <= a+b+c+d <= max_sum.
    """
    rows = []
    for s in range(1, max_sum + 1):
        for a in range(s + 1):
            for b in range(s - a + 1):
                for c in range(s - a - b + 1):
                    d = s - a - b - c
                    rows.append((a, b, c, d))
    return rows


# ----------------------------------------------------------------
# Core eigenvalue analysis
# ----------------------------------------------------------------

def compute_alpha_from_matrix_4x4(M):
    """
    Given a 4x4 matrix M (numpy array), compute:
      - eigenvalues (sorted by |.| descending)
      - Perron-Frobenius eigenvalue lambda_1 = largest |eigenvalue|
      - |lambda_2| = second-largest magnitude
      - is_pisot: True iff lambda_1 > 1 and all |lambda_j| < 1 for j >= 2
      - alpha = 1 - 2*ln|lambda_2| / ln|lambda_1|  (if Pisot)

    Returns dict with keys: alpha, lam1, lam2, is_pisot, eigenvalues (sorted).
    """
    try:
        eigenvalues = np.linalg.eigvals(M)
    except np.linalg.LinAlgError:
        return {'alpha': np.nan, 'lam1': 0.0, 'lam2': 0.0,
                'is_pisot': False, 'eigenvalues': []}

    abs_eigs = np.abs(eigenvalues)
    idx = np.argsort(abs_eigs)[::-1]
    eigs_sorted = eigenvalues[idx]
    abs_sorted  = abs_eigs[idx]

    lam1 = abs_sorted[0]
    lam2 = abs_sorted[1]

    if lam1 <= 1.0:
        return {'alpha': np.nan, 'lam1': float(lam1), 'lam2': float(lam2),
                'is_pisot': False, 'eigenvalues': eigs_sorted.tolist()}

    # Pisot: all sub-dominant eigenvalues have modulus < 1
    is_pisot = bool(lam2 < 1.0)
    if not is_pisot:
        return {'alpha': np.nan, 'lam1': float(lam1), 'lam2': float(lam2),
                'is_pisot': False, 'eigenvalues': eigs_sorted.tolist()}

    if lam2 == 0.0:
        # lambda_2 = 0 => alpha = +inf (lattice-like)
        alpha = np.inf
    else:
        alpha = 1.0 - 2.0 * np.log(lam2) / np.log(lam1)

    return {
        'alpha':      float(alpha),
        'lam1':       float(lam1),
        'lam2':       float(lam2),
        'is_pisot':   True,
        'eigenvalues': eigs_sorted.tolist(),
    }


def is_primitive_4x4(M, max_power=10):
    """Check M^k > 0 for some k <= max_power (primitivity = irreducibility + aperiodicity)."""
    Mk = M.copy().astype(float)
    for _ in range(max_power):
        if np.all(Mk > 0):
            return True
        Mk = Mk @ M
    return False


def eigenvalue_structure(eigs_sorted):
    """
    Classify the eigenvalue structure for a 4x4 matrix.
    eigs_sorted: eigenvalues sorted by |.| descending (numpy array or list of complex).

    Returns a string like:
      '4real'          — all 4 real
      '2real+1cc'      — 2 real + 1 complex conjugate pair
      '2cc'            — 2 complex conjugate pairs
      'mixed'          — other
    """
    eigs = np.array(eigs_sorted)
    # Count complex eigenvalues (non-negligible imaginary part)
    tol = 1e-8
    n_complex = int(np.sum(np.abs(eigs.imag) > tol))

    if n_complex == 0:
        return '4real'
    elif n_complex == 2:
        return '2real+1cc'
    elif n_complex == 4:
        return '2cc'
    else:
        return 'mixed'


# ----------------------------------------------------------------
# Main search
# ----------------------------------------------------------------

def run_search(max_row_sum=2, verbose=True):
    rows = enum_rows_4(max_sum=max_row_sum)
    total_candidates = len(rows) ** 4

    if verbose:
        print(f"\n  Row options (sum 1..{max_row_sum}, 4 entries): {len(rows)}")
        print(f"  Total 4x4 matrices to check:            {total_candidates:,}")

    t0 = time.perf_counter()

    pisot_matrices  = []   # All Pisot matrices
    gap_23_matrices = []   # Pisot with alpha in (2, 3)
    alpha_values    = []   # Alpha for all Pisot matrices (for histogram)

    count_total   = 0
    count_pisot   = 0

    REPORT_EVERY = 500_000

    for r0, r1, r2, r3 in itertools.product(rows, repeat=4):
        M = np.array([r0, r1, r2, r3], dtype=float)
        count_total += 1

        if verbose and count_total % REPORT_EVERY == 0:
            elapsed = time.perf_counter() - t0
            frac = count_total / total_candidates
            eta  = elapsed / frac * (1 - frac) if frac > 0 else 0
            print(f"    Progress: {count_total:,}/{total_candidates:,} "
                  f"({100*frac:.1f}%)  Pisot so far: {count_pisot:,}  "
                  f"ETA: {eta:.0f}s", flush=True)

        res = compute_alpha_from_matrix_4x4(M)
        if not res['is_pisot']:
            continue

        alpha = res['alpha']
        if not np.isfinite(alpha) or alpha <= 0:
            continue

        count_pisot += 1
        alpha_values.append(alpha)

        entry = {
            'M':          [list(r) for r in [r0, r1, r2, r3]],
            'alpha':      alpha,
            'lam1':       res['lam1'],
            'lam2':       res['lam2'],
            'eigenvalues': res['eigenvalues'],
            'eig_structure': eigenvalue_structure(res['eigenvalues']),
        }
        pisot_matrices.append(entry)

        # Key test: is alpha strictly in (2, 3)?
        if 2.0 < alpha < 3.0:
            gap_23_matrices.append(entry)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  Search complete in {elapsed:.1f}s")
        print(f"  Total matrices checked:  {count_total:,}")
        print(f"  Pisot matrices found:    {count_pisot:,}")
        print(f"  With alpha in (2,3):     {len(gap_23_matrices):,}")

    return pisot_matrices, gap_23_matrices, alpha_values, count_total, elapsed


# ----------------------------------------------------------------
# Alpha classification (same bins as 3x3 script)
# ----------------------------------------------------------------

def alpha_class(alpha):
    if alpha > 3.0 + 0.1:      return '>3'
    if alpha > 2.0 + 0.01:     return '(2,3)'
    if abs(alpha - 3.0) < 0.01: return '~3'
    if abs(alpha - 2.0) < 0.01: return '~2'
    if alpha > 1.5 + 0.05:     return '(1.5,2)'
    if alpha > 1.0 + 0.05:     return '(1,1.5)'
    if alpha >= 1.0 - 0.05:    return '~1'
    if alpha > 0:               return '(0,1)'
    return '<0'


# ----------------------------------------------------------------
# Serialize helpers
# ----------------------------------------------------------------

def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    return obj


# ----------------------------------------------------------------
# Figure: histogram of alpha values
# ----------------------------------------------------------------

def make_figure(pisot_matrices, gap_23_matrices, max_row_sum, out_path):
    alpha_vals = np.array([p['alpha'] for p in pisot_matrices
                           if np.isfinite(p['alpha']) and 0 < p['alpha'] < 6])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: full distribution
    ax = axes[0]
    ax.hist(alpha_vals, bins=120, color='#1f77b4', alpha=0.75, edgecolor='white', lw=0.3)
    ax.axvspan(2.0, 3.0, alpha=0.18, color='red', label='Target gap (2,3)')
    ax.axvline(2.0, color='red',  ls='--', lw=1.5, alpha=0.8)
    ax.axvline(3.0, color='red',  ls='--', lw=1.5, alpha=0.8)
    ax.axvline(1.545, color='blue', ls=':', lw=1.5, alpha=0.8, label='BT (1.545)')
    ax.set_xlabel(r'Hyperuniformity exponent $\alpha$', fontsize=12)
    ax.set_ylabel('Count (Pisot 4x4 matrices)', fontsize=11)
    ax.set_title(
        f'Alpha distribution: 4x4 Pisot matrices\n'
        f'(row sums <= {max_row_sum}, N_Pisot={len(pisot_matrices):,})',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)

    # Right panel: zoom into (1.5, 3.5) to inspect the gap region
    ax2 = axes[1]
    zoom_vals = alpha_vals[(alpha_vals >= 1.5) & (alpha_vals <= 3.5)]
    if len(zoom_vals) > 0:
        ax2.hist(zoom_vals, bins=100, color='#ff7f0e', alpha=0.75,
                 edgecolor='white', lw=0.3)
    ax2.axvspan(2.0, 3.0, alpha=0.18, color='red', label=f'Gap region\n({len(gap_23_matrices)} hits)')
    ax2.axvline(2.0, color='red',  ls='--', lw=1.5, alpha=0.8)
    ax2.axvline(3.0, color='red',  ls='--', lw=1.5, alpha=0.8)
    ax2.axvline(1.545, color='blue', ls=':', lw=1.5, alpha=0.8, label='BT (1.545)')
    ax2.set_xlabel(r'$\alpha$', fontsize=12)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Zoom: alpha in [1.5, 3.5]', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, ls=':', alpha=0.4)

    # Annotate gap hits
    if gap_23_matrices:
        gap_alphas = [p['alpha'] for p in gap_23_matrices]
        for ga in gap_alphas[:50]:   # annotate up to 50
            ax2.axvline(ga, color='green', lw=0.6, alpha=0.5)
        ax2.text(0.5, 0.92,
                 f'{len(gap_23_matrices)} matrices with alpha in (2,3)!',
                 transform=ax2.transAxes, ha='center', fontsize=10,
                 color='green', fontweight='bold')
    else:
        ax2.text(0.5, 0.92,
                 'No matrices with alpha strictly in (2,3)',
                 transform=ax2.transAxes, ha='center', fontsize=10,
                 color='darkred', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}")


# ================================================================
# MAIN
# ================================================================

print("=" * 70)
print("  4x4 Substitution Matrix Search: Looking for alpha in (2, 3)")
print("=" * 70)

# ----- Phase 1: max_row_sum = 2 -----
MAX_ROW_SUM = 2
print(f"\n[Phase 1]  max_row_sum = {MAX_ROW_SUM}")

pisot_list, gap23_list, alpha_vals, n_total, t_elapsed = run_search(
    max_row_sum=MAX_ROW_SUM, verbose=True
)

# Alpha distribution
print("\n  Alpha distribution (all Pisot 4x4 matrices):")
counts = Counter(alpha_class(p['alpha']) for p in pisot_list)
for cat in ['<0', '(0,1)', '~1', '(1,1.5)', '(1.5,2)', '~2', '(2,3)', '~3', '>3']:
    print(f"    {cat:12s}: {counts.get(cat, 0):6,}")

# Eigenvalue structure among Pisot matrices
print("\n  Eigenvalue structure among Pisot matrices:")
struct_counts = Counter(p['eig_structure'] for p in pisot_list)
for st, cnt in sorted(struct_counts.items(), key=lambda x: -x[1]):
    print(f"    {st:15s}: {cnt:,}")

# Gap candidates
print(f"\n  === RESULT: {len(gap23_list)} matrix/matrices found with alpha in (2, 3) ===")
if gap23_list:
    gap23_list.sort(key=lambda p: p['alpha'])
    print("\n  Candidates in gap (2, 3):")
    for p in gap23_list[:30]:
        print(f"    alpha={p['alpha']:.6f}  lam1={p['lam1']:.6f}  "
              f"lam2={p['lam2']:.6f}  struct={p['eig_structure']}")
        print(f"      M = {p['M']}")

    # Primitivity check
    print("\n  Checking primitivity of gap candidates...")
    primitive_gap = []
    for p in gap23_list:
        M = np.array(p['M'], dtype=float)
        if is_primitive_4x4(M):
            primitive_gap.append(p)
    print(f"  Primitive gap candidates: {len(primitive_gap)}")
    for p in primitive_gap[:10]:
        print(f"    alpha={p['alpha']:.6f}  M={p['M']}")
else:
    primitive_gap = []
    print("  (Searching the (2,3) gap...)")

# ----- Phase 2: max_row_sum = 3 -----
print(f"\n[Phase 2]  max_row_sum = 3  (larger search)")
MAX_ROW_SUM_2 = 3

pisot_list2, gap23_list2, alpha_vals2, n_total2, t_elapsed2 = run_search(
    max_row_sum=MAX_ROW_SUM_2, verbose=True
)

print("\n  Alpha distribution (row sums <= 3):")
counts2 = Counter(alpha_class(p['alpha']) for p in pisot_list2)
for cat in ['<0', '(0,1)', '~1', '(1,1.5)', '(1.5,2)', '~2', '(2,3)', '~3', '>3']:
    print(f"    {cat:12s}: {counts2.get(cat, 0):6,}")

print(f"\n  === RESULT (row sums <=3): {len(gap23_list2)} matrix/matrices found with alpha in (2, 3) ===")
if gap23_list2:
    gap23_list2.sort(key=lambda p: p['alpha'])
    print("\n  Candidates in gap (2, 3):")
    for p in gap23_list2[:30]:
        print(f"    alpha={p['alpha']:.6f}  lam1={p['lam1']:.6f}  "
              f"lam2={p['lam2']:.6f}  struct={p['eig_structure']}")
        print(f"      M = {p['M']}")

    print("\n  Checking primitivity...")
    primitive_gap2 = []
    for p in gap23_list2:
        M = np.array(p['M'], dtype=float)
        if is_primitive_4x4(M):
            primitive_gap2.append(p)
    print(f"  Primitive gap candidates (row sums <=3): {len(primitive_gap2)}")
    for p in primitive_gap2[:10]:
        print(f"    alpha={p['alpha']:.6f}  M={p['M']}")
else:
    primitive_gap2 = []

# Use the larger set for the figure and JSON
all_pisot   = pisot_list2
all_gap23   = gap23_list2
all_prim_gap = primitive_gap2

# Detailed alpha analysis for ALL Pisot matrices: check if gap values are exact 2 or 3
# (numerical noise can make alpha=2 appear as 2.000001 etc.)
print("\n  Checking alpha values near 2 and 3 (numerical precision):")
near_2 = [p for p in all_gap23 if abs(p['alpha'] - 2.0) < 0.005]
near_3 = [p for p in all_gap23 if abs(p['alpha'] - 3.0) < 0.005]
truly_interior = [p for p in all_gap23
                  if abs(p['alpha'] - 2.0) >= 0.005 and abs(p['alpha'] - 3.0) >= 0.005]
print(f"    Numerically near alpha=2 (within 0.005): {len(near_2)}")
print(f"    Numerically near alpha=3 (within 0.005): {len(near_3)}")
print(f"    Strictly interior to (2,3) [tol=0.005]:  {len(truly_interior)}")

if truly_interior:
    print("\n  TRULY INTERIOR gap candidates (alpha not near 2 or 3):")
    for p in sorted(truly_interior, key=lambda x: x['alpha'])[:20]:
        print(f"    alpha={p['alpha']:.6f}  lam1={p['lam1']:.6f}  "
              f"lam2={p['lam2']:.6f}  struct={p['eig_structure']}")
        print(f"      M = {p['M']}")
        print(f"      eigenvalues = {[f'{complex(e):.6f}' for e in p['eigenvalues']]}")

# Eigenvalue structure of gap matrices
if all_gap23:
    print("\n  Eigenvalue structures in gap (2,3):")
    struct_gap = Counter(p['eig_structure'] for p in all_gap23)
    for st, cnt in sorted(struct_gap.items(), key=lambda x: -x[1]):
        print(f"    {st:15s}: {cnt:,}")

# ----------------------------------------------------------------
# Figure
# ----------------------------------------------------------------
fig_path = os.path.join(RESULTS_DIR, 'fig_4x4_alpha_dist.png')
print("\nGenerating figure ...")
make_figure(all_pisot, all_gap23, 3, fig_path)

# ----------------------------------------------------------------
# Save JSON
# ----------------------------------------------------------------
json_path = os.path.join(DATA_DIR, '4x4_search_results.json')

# Compute unique alpha values (rounded to 4 decimal places) in gap
gap_unique_alphas = sorted(set(round(p['alpha'], 4) for p in all_gap23))

save_data = {
    'description': '4x4 substitution matrix Pisot search for alpha in (2,3)',
    'phase1': {
        'max_row_sum':     MAX_ROW_SUM,
        'total_matrices':  n_total,
        'elapsed_seconds': round(t_elapsed, 2),
        'pisot_count':     len(pisot_list),
        'gap_23_count':    len(gap23_list),
        'alpha_distribution': {
            cat: counts.get(cat, 0)
            for cat in ['<0','(0,1)','~1','(1,1.5)','(1.5,2)','~2','(2,3)','~3','>3']
        },
    },
    'phase2': {
        'max_row_sum':     MAX_ROW_SUM_2,
        'total_matrices':  n_total2,
        'elapsed_seconds': round(t_elapsed2, 2),
        'pisot_count':     len(pisot_list2),
        'gap_23_count':    len(gap23_list2),
        'alpha_distribution': {
            cat: counts2.get(cat, 0)
            for cat in ['<0','(0,1)','~1','(1,1.5)','(1.5,2)','~2','(2,3)','~3','>3']
        },
    },
    'gap_analysis': {
        'near_alpha_2':        len(near_2),
        'near_alpha_3':        len(near_3),
        'truly_interior':      len(truly_interior),
        'primitive_gap_count': len(all_prim_gap),
        'unique_gap_alphas_rounded': gap_unique_alphas[:50],
    },
    'gap_candidates': [
        {
            'M':             p['M'],
            'alpha':         round(p['alpha'], 6),
            'lam1':          round(p['lam1'], 6),
            'lam2':          round(p['lam2'], 6),
            'eig_structure': p['eig_structure'],
            'eigenvalues':   [
                {'real': round(complex(e).real, 6), 'imag': round(complex(e).imag, 6)}
                for e in p['eigenvalues']
            ],
        }
        for p in all_gap23[:100]   # save up to 100 gap candidates
    ],
    'truly_interior_candidates': [
        {
            'M':             p['M'],
            'alpha':         round(p['alpha'], 6),
            'lam1':          round(p['lam1'], 6),
            'lam2':          round(p['lam2'], 6),
            'eig_structure': p['eig_structure'],
            'eigenvalues':   [
                {'real': round(complex(e).real, 6), 'imag': round(complex(e).imag, 6)}
                for e in p['eigenvalues']
            ],
        }
        for p in sorted(truly_interior, key=lambda x: x['alpha'])[:50]
    ],
}

with open(json_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=serialize)
print(f"  JSON saved: {json_path}")

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print("\n" + "=" * 70)
print("  FINAL SUMMARY: 4x4 Substitution Matrix Search")
print("=" * 70)
print(f"  Phase 1 (row sums <= {MAX_ROW_SUM}):")
print(f"    Total matrices:      {n_total:,}")
print(f"    Pisot matrices:      {len(pisot_list):,}")
print(f"    Alpha in (2,3):      {len(gap23_list):,}")
print(f"  Phase 2 (row sums <= {MAX_ROW_SUM_2}):")
print(f"    Total matrices:      {n_total2:,}")
print(f"    Pisot matrices:      {len(pisot_list2):,}")
print(f"    Alpha in (2,3):      {len(gap23_list2):,}")
print()
print(f"  Gap candidates (row sums <=3):")
print(f"    Near alpha=2 (noise): {len(near_2)}")
print(f"    Near alpha=3 (noise): {len(near_3)}")
print(f"    Truly interior:       {len(truly_interior)}")
print(f"    Of which primitive:   {len(all_prim_gap)}")
print()
if truly_interior:
    print("  ** GAP FILLED: strictly interior (2,3) candidates exist! **")
    best = min(truly_interior, key=lambda p: abs(p['alpha'] - 2.5))
    print(f"  Best candidate (closest to alpha=2.5):")
    print(f"    alpha = {best['alpha']:.6f}")
    print(f"    lam1  = {best['lam1']:.6f}")
    print(f"    lam2  = {best['lam2']:.6f}")
    print(f"    struct = {best['eig_structure']}")
    print(f"    M = {best['M']}")
else:
    print("  ** GAP NOT FILLED: no 4x4 matrices (row sums<=3) give alpha strictly in (2,3) **")
    print("  (all apparent (2,3) hits are numerically at alpha=2.000 or alpha=3.000)")
print("=" * 70)
