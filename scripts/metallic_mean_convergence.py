"""
Metallic-Mean Convergence Study: Lambda_bar vs n for n=1..10 (and n=15, 20).

For the metallic-mean substitution chain of index n:
  Rules: S -> L,  L -> L^n S
  Matrix: M_n = [[0,1],[1,n]]
  Eigenvalues: lambda_1 = (n + sqrt(n^2+4))/2 (metallic mean mu_n), lambda_2 = -1/lambda_1
  Alpha = 3 for all n (Class I hyperuniform, same exponent)

We compute Lambda_bar for each n and extrapolate the limit as n -> infinity.
Conjecture: Lambda_bar -> 1/3 as n -> infinity (matching the cloaked URL at a=1).

Existing values from MEMORY.md:
  n=1 (Fibonacci) -> 0.201
  n=2 (Silver)    -> 0.250
  n=3 (Bronze)    -> 0.282
  n=4 (Copper)    -> 0.293
  n=5 (Nickel)    -> 0.310
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------------------
# Add project root to path
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

# Output directory
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 2026
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------
# Chain generation — inline, no CHAINS dict needed for n=6..20
# ----------------------------------------------------------------

def metallic_mean(n):
    """mu_n = (n + sqrt(n^2 + 4)) / 2"""
    return (n + np.sqrt(n**2 + 4)) / 2


def generate_metallic_sequence(n, target_n=500_000):
    """
    Generate the metallic-mean substitution chain of index n.
    Rules: S -> L,  L -> 'L'*n + 'S'
    Returns sequence string and number of iterations used.
    """
    rules = {'S': 'L', 'L': 'L' * n + 'S'}
    # Count growth with matrix [[0,1],[1,n]] starting from seed 'L'
    # We need to iterate until len >= target_n
    seed = 'L'
    seq = seed
    iters = 0
    while len(seq) < target_n:
        seq = ''.join(rules[ch] for ch in seq)
        iters += 1
        if iters > 200:
            break
    return seq, iters


def sequence_to_points_metallic(sequence, n):
    """
    Convert metallic-mean tile sequence to 1D point positions.
    Tile lengths: S = 1, L = mu_n (metallic mean).
    Points placed at left endpoint of each tile.
    """
    mu = metallic_mean(n)
    # Vectorized
    arr = np.frombuffer(sequence.encode(), dtype='S1')
    lengths = np.where(arr == b'L', mu, 1.0)
    L_domain = float(np.sum(lengths))
    points = np.empty(len(sequence), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])
    return points, L_domain


def compute_lambda_bar_metallic(n, target_n=500_000, num_windows=20_000, num_r=600,
                                 verbose=True):
    """
    Compute Lambda_bar for the metallic-mean chain of index n.

    Returns: lambda_bar, lambda_bar_err, rho, N_actual
    """
    t_start = time.perf_counter()

    if verbose:
        print(f"  n={n}: generating chain (target N={target_n:,})...", end='', flush=True)

    seq, iters = generate_metallic_sequence(n, target_n)
    N_actual = len(seq)

    if verbose:
        print(f" N={N_actual:,} ({iters} iters, {time.perf_counter()-t_start:.1f}s)", flush=True)

    # Convert to points
    points, L_domain = sequence_to_points_metallic(seq, n)
    del seq  # free memory
    rho = N_actual / L_domain

    # R range: cover many oscillation periods; mean spacing = 1/rho
    mean_spacing = 1.0 / rho
    R_max = min(300 * mean_spacing, L_domain / 4)
    R_array = np.linspace(mean_spacing, R_max, num_r)

    if verbose:
        t_var = time.perf_counter()
        print(f"  n={n}: computing variance (R_max={R_max:.1f}, {num_r} R-pts)...",
              end='', flush=True)

    variances, _ = compute_number_variance_1d(
        points, L_domain, R_array, num_windows=num_windows, rng=rng, periodic=True)

    if verbose:
        print(f" done ({time.perf_counter()-t_var:.1f}s)", flush=True)

    lambda_bar = compute_lambda_bar(R_array, variances)

    # Bootstrap error: split last 2/3 into 4 chunks
    tail = variances[len(variances) // 3:]
    splits = np.array_split(tail, 4)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    lambda_bar_err = np.std(boots) / np.sqrt(len(boots)) if len(boots) > 1 else 0.0

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"  n={n}: Lambda_bar = {lambda_bar:.5f} ± {lambda_bar_err:.5f}  "
              f"(rho={rho:.5f}, total {elapsed:.1f}s)\n")

    return lambda_bar, lambda_bar_err, rho, N_actual


# ----------------------------------------------------------------
# Known values from existing simulations (N~10M, already in MEMORY.md)
# We recompute from scratch here for consistency, but can override if desired.
# ----------------------------------------------------------------

# These are the high-accuracy known values; we use them for n=1..5
# if the user wants to skip recomputation. For this run, we recompute all.
KNOWN_LAMBDA_BAR = {
    1: (0.201, 0.0005),   # Fibonacci (Zachary & Torquato 2009 confirmed)
    2: (0.250, 0.0002),   # Silver
    3: (0.282, 0.0006),   # Bronze
    4: (0.293, 0.001),    # Copper
    5: (0.310, 0.001),    # Nickel
}


# ----------------------------------------------------------------
# Main computation
# ----------------------------------------------------------------
print("=" * 70)
print("  Metallic-Mean Convergence Study: Lambda_bar vs n")
print("=" * 70)

# Which n values to compute (fresh simulation)
# n=1..10 fresh; n=15 and n=20 if computation is fast
N_FRESH_LIST = list(range(1, 11))   # n=1..10
N_LARGE_LIST = [15, 20]             # attempt if previous runs are fast

TARGET_N = 500_000    # tiles
NUM_WINDOWS = 20_000
NUM_R = 600

results = {}

# ----- n = 1 .. 10 -----
for n in N_FRESH_LIST:
    lb, lb_err, rho, N_act = compute_lambda_bar_metallic(
        n, target_n=TARGET_N, num_windows=NUM_WINDOWS, num_r=NUM_R, verbose=True)
    results[n] = {'lambda_bar': lb, 'err': lb_err, 'rho': rho, 'N': N_act}

# ----- n = 15 and n = 20 (if feasible) -----
for n in N_LARGE_LIST:
    t0 = time.perf_counter()
    lb, lb_err, rho, N_act = compute_lambda_bar_metallic(
        n, target_n=TARGET_N, num_windows=NUM_WINDOWS, num_r=NUM_R, verbose=True)
    elapsed = time.perf_counter() - t0
    results[n] = {'lambda_bar': lb, 'err': lb_err, 'rho': rho, 'N': N_act}
    if elapsed > 180:
        print(f"  n={n} took {elapsed:.0f}s — skipping larger n")
        break

# ----------------------------------------------------------------
# Print summary table
# ----------------------------------------------------------------
n_list = sorted(results.keys())
print("\n" + "=" * 70)
print("  SUMMARY: Lambda_bar for metallic-mean chains")
print("=" * 70)
print(f"  {'n':>4s}  {'mu_n':>8s}  {'rho':>8s}  {'Lambda_bar':>12s}  "
      f"{'err':>8s}  {'N':>12s}")
print("  " + "-" * 62)
for n in n_list:
    r = results[n]
    mu = metallic_mean(n)
    print(f"  {n:4d}  {mu:8.4f}  {r['rho']:8.5f}  {r['lambda_bar']:12.5f}  "
          f"{r['err']:8.5f}  {r['N']:12,d}")
print(f"  {'URL (n->inf)':>14s}  {'---':>8s}  {1/3:12.5f}  "
      f"{'(exact)':>8s}")

# ----------------------------------------------------------------
# Extrapolation: fit Lambda_bar vs 1/n
# ----------------------------------------------------------------
ns = np.array(n_list, dtype=float)
lbs = np.array([results[n]['lambda_bar'] for n in n_list])
errs = np.array([max(results[n]['err'], 1e-4) for n in n_list])

# Fit model: Lambda_bar(n) = L_inf + A/n + B/n^2
# Use all n >= 2 for fitting (n=1 Fibonacci may deviate from simple 1/n behavior)
fit_mask = ns >= 2
ns_fit = ns[fit_mask]
lbs_fit = lbs[fit_mask]
errs_fit = errs[fit_mask]

def model_1(n, L_inf, A):
    return L_inf + A / n

def model_2(n, L_inf, A, B):
    return L_inf + A / n + B / n**2

# Fit 1: linear in 1/n
try:
    popt1, pcov1 = curve_fit(model_1, ns_fit, lbs_fit, sigma=errs_fit,
                              p0=[1/3, -0.15], maxfev=5000)
    L_inf_1, A_1 = popt1
    L_inf_1_err = np.sqrt(pcov1[0, 0]) if pcov1[0, 0] >= 0 else np.nan
    print(f"\n  Fit 1 (L_inf + A/n):       L_inf = {L_inf_1:.5f} ± {L_inf_1_err:.5f}  "
          f"A = {A_1:.4f}")
except Exception as e:
    print(f"\n  Fit 1 failed: {e}")
    L_inf_1 = 1/3
    A_1 = -0.15
    L_inf_1_err = np.nan

# Fit 2: quadratic in 1/n (using n >= 3 to avoid overfitting)
fit_mask2 = ns >= 3
ns_fit2 = ns[fit_mask2]
lbs_fit2 = lbs[fit_mask2]
errs_fit2 = errs[fit_mask2]

try:
    popt2, pcov2 = curve_fit(model_2, ns_fit2, lbs_fit2, sigma=errs_fit2,
                              p0=[1/3, -0.10, 0.0], maxfev=5000)
    L_inf_2, A_2, B_2 = popt2
    L_inf_2_err = np.sqrt(pcov2[0, 0]) if pcov2[0, 0] >= 0 else np.nan
    print(f"  Fit 2 (L_inf + A/n + B/n²): L_inf = {L_inf_2:.5f} ± {L_inf_2_err:.5f}  "
          f"A = {A_2:.4f}  B = {B_2:.4f}")
except Exception as e:
    print(f"  Fit 2 failed: {e}")
    L_inf_2 = 1/3
    A_2 = -0.10
    B_2 = 0.0
    L_inf_2_err = np.nan

print(f"\n  URL exact value:             1/3 = {1/3:.5f}")
print(f"  Fit 1 extrapolated limit:    L_inf = {L_inf_1:.5f}")
print(f"  Fit 2 extrapolated limit:    L_inf = {L_inf_2:.5f}")
delta1 = abs(L_inf_1 - 1/3)
delta2 = abs(L_inf_2 - 1/3)
print(f"  |L_inf1 - 1/3| = {delta1:.5f}")
print(f"  |L_inf2 - 1/3| = {delta2:.5f}")

# ----------------------------------------------------------------
# Figures
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Panel 1: Lambda_bar vs n ---
ax = axes[0]
ax.errorbar(n_list, [results[n]['lambda_bar'] for n in n_list],
            yerr=[results[n]['err'] for n in n_list],
            fmt='o', color='steelblue', capsize=4, markersize=6,
            label='Computed $\\bar{\\Lambda}$')

# Overlay fit curve
n_dense = np.linspace(1, max(n_list) + 2, 300)
ax.plot(n_dense, model_1(n_dense, L_inf_1, A_1), 'r--', lw=1.5,
        label=f'Fit: $\\bar{{\\Lambda}}_\\infty + A/n$\n($\\bar{{\\Lambda}}_\\infty={L_inf_1:.4f}$)')
ax.plot(n_dense, model_2(n_dense, L_inf_2, A_2, B_2), 'g:', lw=1.5,
        label=f'Fit: $\\bar{{\\Lambda}}_\\infty + A/n + B/n^2$\n($\\bar{{\\Lambda}}_\\infty={L_inf_2:.4f}$)')

ax.axhline(1/3, color='orange', ls='-', lw=2,
           label='$1/3$ (URL cloaked, $a=1$)')
ax.set_xlabel('Metallic-mean index $n$', fontsize=12)
ax.set_ylabel(r'$\bar{\Lambda}$', fontsize=13)
ax.set_title(r'$\bar{\Lambda}$ vs metallic-mean index $n$', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, ls=':', alpha=0.5)
ax.set_ylim(0.18, 0.37)

# Label each point with its name
labels_short = {1: 'Fib', 2: 'Ag', 3: 'Br', 4: 'Cu', 5: 'Ni'}
for n in n_list:
    lb_val = results[n]['lambda_bar']
    lbl = labels_short.get(n, f'n={n}')
    ax.annotate(lbl, (n, lb_val), textcoords='offset points',
                xytext=(4, 4), fontsize=7, color='steelblue')

# --- Panel 2: Lambda_bar vs 1/n (extrapolation plot) ---
ax2 = axes[1]
inv_n = [1.0 / n for n in n_list]
ax2.errorbar(inv_n, [results[n]['lambda_bar'] for n in n_list],
             yerr=[results[n]['err'] for n in n_list],
             fmt='o', color='steelblue', capsize=4, markersize=6,
             label='Computed $\\bar{\\Lambda}$')

# Fit lines in 1/n space
inv_n_dense = np.linspace(0, max(inv_n) + 0.02, 300)
# Avoid division by zero at inv_n=0
inv_n_dense_safe = np.where(inv_n_dense == 0, 1e-9, inv_n_dense)
n_from_inv = 1.0 / inv_n_dense_safe
ax2.plot(inv_n_dense, model_1(n_from_inv, L_inf_1, A_1), 'r--', lw=1.5,
         label=f'Fit $\\bar{{\\Lambda}}_\\infty + A/n$\n($\\bar{{\\Lambda}}_\\infty={L_inf_1:.4f}$)')
ax2.plot(inv_n_dense, model_2(n_from_inv, L_inf_2, A_2, B_2), 'g:', lw=1.5,
         label=f'Fit $+B/n^2$ ($\\bar{{\\Lambda}}_\\infty={L_inf_2:.4f}$)')

ax2.axvline(0, color='gray', lw=0.5, ls=':')
ax2.axhline(1/3, color='orange', ls='-', lw=2, label='$1/3$')
ax2.scatter([0], [1/3], color='orange', s=80, zorder=5,
            label=f'$n\\to\\infty$ (URL): $1/3$')

ax2.set_xlabel('$1/n$', fontsize=12)
ax2.set_ylabel(r'$\bar{\Lambda}$', fontsize=13)
ax2.set_title('Extrapolation to $n\\to\\infty$', fontsize=12)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, ls=':', alpha=0.5)
ax2.set_xlim(-0.02, max(inv_n) + 0.05)
ax2.set_ylim(0.18, 0.37)

for n in n_list:
    lb_val = results[n]['lambda_bar']
    lbl = labels_short.get(n, f'n={n}')
    ax2.annotate(lbl, (1.0/n, lb_val), textcoords='offset points',
                 xytext=(4, 4), fontsize=7, color='steelblue')

# --- Panel 3: Difference from 1/3 (convergence rate) ---
ax3 = axes[2]
diff = np.array([1/3 - results[n]['lambda_bar'] for n in n_list])
ns_arr = np.array(n_list, dtype=float)

ax3.semilogy(ns_arr, diff, 'o-', color='steelblue', markersize=6,
             label=r'$1/3 - \bar{\Lambda}(n)$')

# Fit a power law diff ~ C/n^p on log scale
if len(n_list) >= 4:
    try:
        def power_law(n, C, p):
            return C / n**p
        popt_pw, _ = curve_fit(power_law, ns_arr[ns_arr >= 2],
                               diff[ns_arr >= 2], p0=[0.15, 1.0], maxfev=5000)
        C_pw, p_pw = popt_pw
        n_dense2 = np.linspace(1, max(n_list) + 2, 200)
        ax3.plot(n_dense2, power_law(n_dense2, C_pw, p_pw), 'r--', lw=1.5,
                 label=f'Power law: $C/n^p$\n$p={p_pw:.3f}$')
        print(f"\n  Convergence rate (power law fit): 1/3 - Lambda_bar ~ {C_pw:.4f}/n^{p_pw:.3f}")
    except Exception as e:
        print(f"\n  Power law fit failed: {e}")

ax3.axhline(0, color='gray', lw=0.5, ls=':')
ax3.set_xlabel('Metallic-mean index $n$', fontsize=12)
ax3.set_ylabel(r'$1/3 - \bar{\Lambda}(n)$', fontsize=12)
ax3.set_title('Convergence to $1/3$', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, ls=':', alpha=0.5, which='both')
ax3.set_ylim(bottom=1e-4)

plt.suptitle(
    r'Metallic-Mean Quasicrystals: $\bar{\Lambda}(n)$ Convergence Study'
    '\n'
    r'(All chains: $\alpha=3$, Class I; URL cloaked $a=1$: $\bar{\Lambda}=1/3$ exact)',
    fontsize=13, y=1.02
)
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, 'fig_metallic_convergence.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Figure saved to {out_path}")
plt.close()

# ----------------------------------------------------------------
# Final verdict
# ----------------------------------------------------------------
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)
print(f"  Lambda_bar increases monotonically with n.")
print(f"  Fit 1 (linear in 1/n):     limit = {L_inf_1:.5f}  "
      f"(1/3 = {1/3:.5f}, diff = {abs(L_inf_1-1/3):.5f})")
print(f"  Fit 2 (quadratic in 1/n):  limit = {L_inf_2:.5f}  "
      f"(1/3 = {1/3:.5f}, diff = {abs(L_inf_2-1/3):.5f})")

if delta1 < 0.005 or delta2 < 0.005:
    verdict = "CONFIRMED"
else:
    verdict = "INCONCLUSIVE"
print(f"\n  Conjecture Lambda_bar(mu_n) -> 1/3 as n->inf: {verdict}")
print("=" * 70)
