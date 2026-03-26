"""
Generalized Scaling Metrics for Class II and III

Defines and computes generalizations of Lambda_bar for non-Class-I patterns.

Theory:
  Class I  (alpha > 1):  sigma^2(R) bounded     -> Lambda_bar = lim sigma^2(R)
  Class II (alpha = 1):  sigma^2(R) ~ C * ln(R) -> Lambda_II = lim sigma^2(R) / ln(R)
  Class III (0 < alpha < 1): sigma^2(R) ~ A * R^{1-alpha} -> Lambda_III = lim sigma^2(R)/R^{1-alpha}

Output:
  - Normalized variance plots for Period-Doubling and 0222
  - results/figures/fig_generalized_metrics.png
  - Updated quasicrystal_variance.py with compute_generalized_metric()
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)
NUM_WINDOWS = 25_000
TARGET_N    = 500_000


def compute_generalized_metric(R_array, var, class_type, alpha=None):
    """
    Compute the generalized hyperuniformity metric for any class.

    Parameters
    ----------
    R_array : ndarray
        Window half-widths.
    var : ndarray
        Number variance sigma^2(R).
    class_type : str
        'I', 'II', or 'III'.
    alpha : float, optional
        Hyperuniformity exponent. Required for class 'III'.

    Returns
    -------
    metric : float
        The generalized metric value.
    metric_err : float
        Estimated uncertainty (std over last 1/3 of R range).
    normalized_var : ndarray
        The normalized quantity sigma^2(R) / normalizer(R).
        For Class I: sigma^2(R) (no change)
        For Class II: sigma^2(R) / ln(R)
        For Class III: sigma^2(R) / R^{1-alpha}
    """
    R = np.asarray(R_array, dtype=float)
    v = np.asarray(var, dtype=float)

    if class_type == 'I':
        normalized = v.copy()
        start = len(R) // 3
        metric = float(np.mean(normalized[start:]))
        metric_err = float(np.std(normalized[start:]))

    elif class_type == 'II':
        log_R = np.log(np.maximum(R, 1e-10))
        normalized = v / log_R
        start = len(R) // 3
        metric = float(np.mean(normalized[start:]))
        metric_err = float(np.std(normalized[start:]))

    elif class_type == 'III':
        if alpha is None:
            raise ValueError("alpha required for Class III metric")
        beta = 1.0 - alpha  # exponent of R in sigma^2(R) ~ A * R^beta
        R_pow = np.power(np.maximum(R, 1e-10), beta)
        normalized = v / R_pow
        start = len(R) // 3
        metric = float(np.mean(normalized[start:]))
        metric_err = float(np.std(normalized[start:]))

    else:
        raise ValueError(f"Unknown class_type '{class_type}'. Use 'I', 'II', or 'III'.")

    return metric, metric_err, normalized


# ============================================================
# Section 1: Period-Doubling (Class II, alpha=1)
# ============================================================
print("=" * 65)
print("  Generalized Scaling Metrics")
print("=" * 65)

print("\n[1] Period-Doubling (Class II, alpha=1) ...")
alpha_pd, lam1_pd, lam2_pd = verify_eigenvalue_prediction('period_doubling')
print(f"    alpha_eigenvalue = {alpha_pd:.6f}")

for iters in range(5, 50):
    if predict_chain_length('period_doubling', iters) >= TARGET_N:
        break
t0 = time.perf_counter()
seq_pd = generate_substitution_sequence('period_doubling', iters)
pts_pd, L_pd = sequence_to_points(seq_pd, 'period_doubling')
del seq_pd
N_pd  = len(pts_pd)
rho_pd = N_pd / L_pd
mean_sp_pd = 1.0 / rho_pd
print(f"    N={N_pd:,}, rho={rho_pd:.6f}  [{time.perf_counter()-t0:.1f}s]")

R_pd = np.linspace(mean_sp_pd, min(2000 * mean_sp_pd, L_pd / 4), 1000)
t0 = time.perf_counter()
var_pd, _ = compute_number_variance_1d(pts_pd, L_pd, R_pd,
                                        num_windows=NUM_WINDOWS, rng=rng)
print(f"    Variance computed  [{time.perf_counter()-t0:.1f}s]")
del pts_pd

# Fit C*ln(R) + b using large-R data
mask_fit = R_pd > 20 * mean_sp_pd
try:
    popt, pcov = curve_fit(lambda R, C, b: C * np.log(R) + b,
                           R_pd[mask_fit], var_pd[mask_fit])
    C_pd_fit, b_pd_fit = float(popt[0]), float(popt[1])
    C_pd_err_fit = float(np.sqrt(pcov[0, 0]))
    print(f"    Curve fit: sigma^2 ~ {C_pd_fit:.5f}*ln(R) + {b_pd_fit:.4f}  "
          f"(err C: {C_pd_err_fit:.5f})")
except Exception as e:
    C_pd_fit = b_pd_fit = C_pd_err_fit = np.nan
    print(f"    Curve fit failed: {e}")

# Generalized metric
Lambda_II, Lambda_II_err, norm_pd = compute_generalized_metric(R_pd, var_pd, 'II')
print(f"    Lambda_II (PRIMARY: curve fit)           = {C_pd_fit:.5f} +/- {C_pd_err_fit:.5f}")
print(f"    Lambda_II (cross-check: plateau)         = {Lambda_II:.5f} +/- {Lambda_II_err:.5f}  [biased high at finite R]")


# ============================================================
# Section 2: 0222 Chain (Class III, alpha~0.639)
# ============================================================
print("\n[2] 0222 Chain (Class III, alpha~0.639) ...")
alpha_0222, lam1_0222, lam2_0222 = verify_eigenvalue_prediction('chain_0222')
print(f"    alpha_eigenvalue = {alpha_0222:.6f}")

for iters in range(5, 50):
    if predict_chain_length('chain_0222', iters) >= TARGET_N:
        break
t0 = time.perf_counter()
seq_0222 = generate_substitution_sequence('chain_0222', iters)
pts_0222, L_0222 = sequence_to_points(seq_0222, 'chain_0222')
del seq_0222
N_0222 = len(pts_0222)
rho_0222 = N_0222 / L_0222
mean_sp_0222 = 1.0 / rho_0222
print(f"    N={N_0222:,}, rho={rho_0222:.6f}  [{time.perf_counter()-t0:.1f}s]")

R_0222 = np.linspace(mean_sp_0222, min(2000 * mean_sp_0222, L_0222 / 4), 1000)
t0 = time.perf_counter()
var_0222, _ = compute_number_variance_1d(pts_0222, L_0222, R_0222,
                                          num_windows=NUM_WINDOWS, rng=rng)
print(f"    Variance computed  [{time.perf_counter()-t0:.1f}s]")
del pts_0222

# Fit A * R^beta
mask_fit_0222 = R_0222 > 20 * mean_sp_0222
try:
    popt2, pcov2 = curve_fit(lambda R, A, beta: A * R ** beta,
                              R_0222[mask_fit_0222], var_0222[mask_fit_0222],
                              p0=[0.5, 0.36], maxfev=5000)
    A_0222_fit, beta_0222_fit = float(popt2[0]), float(popt2[1])
    A_err_fit = float(np.sqrt(pcov2[0, 0]))
    beta_err = float(np.sqrt(pcov2[1, 1]))
    alpha_0222_num = 1.0 - beta_0222_fit
    print(f"    Curve fit: sigma^2 ~ {A_0222_fit:.5f}*R^{beta_0222_fit:.4f}  "
          f"-> alpha_num={alpha_0222_num:.4f}  (theory={alpha_0222:.4f})")
except Exception as e:
    A_0222_fit = beta_0222_fit = alpha_0222_num = A_err_fit = np.nan
    print(f"    Curve fit failed: {e}")

# Generalized metric using theoretical alpha (cross-check; plateau biased low at finite R)
Lambda_III, Lambda_III_err, norm_0222 = compute_generalized_metric(
    R_0222, var_0222, 'III', alpha=alpha_0222)
print(f"    Lambda_III (PRIMARY: curve fit)                     = {A_0222_fit:.5f} +/- {A_err_fit:.5f}")
print(f"    Lambda_III (cross-check: plateau, th. alpha={alpha_0222:.4f}) = {Lambda_III:.5f} +/- {Lambda_III_err:.5f}  [biased low at finite R]")
# Self-consistency: plateau with fitted alpha
if not np.isnan(alpha_0222_num):
    Lambda_III_fa, Lambda_III_fa_err, _ = compute_generalized_metric(
        R_0222, var_0222, 'III', alpha=alpha_0222_num)
    print(f"    Lambda_III (plateau, fitted alpha={alpha_0222_num:.4f})   = {Lambda_III_fa:.5f} +/- {Lambda_III_fa_err:.5f}")
else:
    Lambda_III_fa, Lambda_III_fa_err = np.nan, np.nan


# ============================================================
# Reference: Fibonacci Class I for comparison
# ============================================================
print("\n[3] Fibonacci (Class I, reference) ...")
for iters in range(5, 50):
    if predict_chain_length('fibonacci', iters) >= TARGET_N:
        break
t0 = time.perf_counter()
seq_fib = generate_substitution_sequence('fibonacci', iters)
from substitution_tilings import sequence_to_points as seq2pts
pts_fib, L_fib = seq2pts(seq_fib, 'fibonacci')
del seq_fib
N_fib = len(pts_fib)
rho_fib = N_fib / L_fib
mean_sp_fib = 1.0 / rho_fib
print(f"    N={N_fib:,}, rho={rho_fib:.6f}  [{time.perf_counter()-t0:.1f}s]")

R_fib = np.linspace(mean_sp_fib, min(2000 * mean_sp_fib, L_fib / 4), 1000)
var_fib, _ = compute_number_variance_1d(pts_fib, L_fib, R_fib,
                                         num_windows=NUM_WINDOWS, rng=rng)
del pts_fib
lb_fib, lb_fib_err, norm_fib = compute_generalized_metric(R_fib, var_fib, 'I')
print(f"    Lambda_bar = {lb_fib:.5f} ± {lb_fib_err:.5f}")


# ============================================================
# Figure: Normalized variance curves showing flat plateau
# ============================================================
print("\nGenerating fig_generalized_metrics.png ...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(r'Generalized Hyperuniformity Metrics by Class',
             fontsize=13, fontweight='bold')

# Panel (a): Class I — sigma^2(R) (Fibonacci)
ax = axes[0]
ax.semilogx(R_fib[R_fib > mean_sp_fib], norm_fib[R_fib > mean_sp_fib],
            '#2ca02c', lw=1.2, alpha=0.85)
ax.axhline(lb_fib, color='r', ls='--', lw=2,
           label=rf'$\bar{{\Lambda}} = {lb_fib:.4f}$')
ax.fill_between([R_fib[0], R_fib[-1]],
                lb_fib - lb_fib_err, lb_fib + lb_fib_err,
                alpha=0.15, color='r')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.set_title(r'Class I: Fibonacci ($\alpha=3$)' '\n'
             r'Metric: $\bar{\Lambda} = \lim \sigma^2(R)$', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.95,
        rf'$\bar{{\Lambda}} = {lb_fib:.4f} \pm {lb_fib_err:.4f}$',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# Panel (b): Class II — sigma^2(R) / ln(R) (Period-Doubling)
ax = axes[1]
use_pd = R_pd > 10 * mean_sp_pd
ln_R_pd = np.log(R_pd[use_pd])
ax.semilogx(R_pd[use_pd], norm_pd[use_pd], '#ff7f0e', lw=1.0, alpha=0.85)
ax.axhline(C_pd_fit, color='r', ls='--', lw=2,
           label=rf'$\Lambda_{{II}} = {C_pd_fit:.4f}$ (fit, primary)')
ax.fill_between([R_pd[use_pd][0], R_pd[use_pd][-1]],
                C_pd_fit - C_pd_err_fit, C_pd_fit + C_pd_err_fit, alpha=0.15, color='r')
ax.axhline(Lambda_II, color='k', ls=':', lw=1.5, alpha=0.7,
           label=rf'Plateau $\Lambda_{{II}} = {Lambda_II:.4f}$ (biased high)')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R) / \ln R$', fontsize=12)
ax.set_title(r'Class II: Period-Doubling ($\alpha=1$)' '\n'
             r'Metric: $\Lambda_{II} = \lim \sigma^2(R)/\ln R$', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.95,
        rf'Fit: $\Lambda_{{II}} = {C_pd_fit:.4f} \pm {C_pd_err_fit:.4f}$'
        '\n' rf'Plateau: $\Lambda_{{II}} = {Lambda_II:.4f}$ (biased high)',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# Panel (c): Class III — sigma^2(R) / R^{1-alpha} (0222)
ax = axes[2]
use_0222 = R_0222 > 10 * mean_sp_0222
ax.semilogx(R_0222[use_0222], norm_0222[use_0222], '#1f77b4', lw=1.0, alpha=0.85)
if not np.isnan(A_0222_fit):
    ax.axhline(A_0222_fit, color='r', ls='--', lw=2,
               label=rf'$\Lambda_{{III}} = {A_0222_fit:.4f}$ (fit, primary)')
    ax.fill_between([R_0222[use_0222][0], R_0222[use_0222][-1]],
                    A_0222_fit - A_err_fit, A_0222_fit + A_err_fit, alpha=0.15, color='r')
ax.axhline(Lambda_III, color='k', ls=':', lw=1.5, alpha=0.7,
           label=rf'Plateau $\Lambda_{{III}} = {Lambda_III:.4f}$ (biased low)')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(rf'$\sigma^2(R) / R^{{1-\alpha}}$', fontsize=12)
ax.set_title(rf'Class III: 0222 Chain ($\alpha={alpha_0222:.3f}$)' '\n'
             rf'Metric: $\Lambda_{{III}} = \lim \sigma^2(R)/R^{{1-\alpha}}$', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.95,
        rf'$\alpha_{{num}} = {alpha_0222_num:.4f}$, $\alpha_{{th}} = {alpha_0222:.4f}$'
        '\n' rf'Fit: $\Lambda_{{III}} = {A_0222_fit:.4f}$ (primary)'
        '\n' rf'Plateau: $\Lambda_{{III}} = {Lambda_III:.4f}$ (biased low)',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'fig_generalized_metrics.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("  SUMMARY: Generalized Hyperuniformity Metrics")
print("=" * 65)
print(f"  Class I  — Fibonacci (alpha=3):")
print(f"    Lambda_bar = {lb_fib:.5f} ± {lb_fib_err:.5f}  [lim sigma^2(R)]")
print()
print(f"  Class II — Period-Doubling (alpha=1):")
print(f"    Lambda_II  = {C_pd_fit:.5f} +/- {C_pd_err_fit:.5f}  [PRIMARY: curve fit]")
print(f"    C_II  = {Lambda_II:.5f} +/- {Lambda_II_err:.5f}  [cross-check: plateau; biased high at finite R]")
print()
print(f"  Class III — 0222 Chain (alpha={alpha_0222:.4f}):")
print(f"    Lambda_III = {A_0222_fit:.5f}  [PRIMARY: curve fit, alpha_num={alpha_0222_num:.4f}]")
if not np.isnan(Lambda_III_fa):
    print(f"    A_III = {Lambda_III_fa:.5f} +/- {Lambda_III_fa_err:.5f}  [plateau, fitted alpha={alpha_0222_num:.4f} (self-consistency)]")
print(f"    A_III = {Lambda_III:.5f} +/- {Lambda_III_err:.5f}  [cross-check: plateau, theory alpha; biased low]")
print(f"    alpha_numeric = {alpha_0222_num:.4f}  (theory = {alpha_0222:.4f})")
print("=" * 65)


# ============================================================
# Patch quasicrystal_variance.py with compute_generalized_metric
# ============================================================
print("\nPatching quasicrystal_variance.py with compute_generalized_metric() ...")

PATCH = '''

def compute_generalized_metric(R_array, var, class_type, alpha=None):
    """
    Compute the generalized hyperuniformity metric for any class.

    For Class I  (alpha > 1): sigma^2(R) bounded -> metric = Lambda_bar = lim sigma^2(R)
    For Class II (alpha = 1): sigma^2(R) ~ C*ln R -> metric = Lambda_II = lim sigma^2(R)/ln(R)
    For Class III (0<alpha<1): sigma^2(R) ~ A*R^{1-alpha} -> metric = Lambda_III = lim sigma^2/R^{1-alpha}

    Parameters
    ----------
    R_array : ndarray
        Window half-widths.
    var : ndarray
        Number variance sigma^2(R).
    class_type : str
        One of 'I', 'II', 'III'.
    alpha : float, optional
        Hyperuniformity exponent. Required for class 'III'.

    Returns
    -------
    metric : float
    metric_err : float
        Std over last 1/3 of R range.
    normalized_var : ndarray
        sigma^2(R) divided by the appropriate normalizer.
    """
    R = np.asarray(R_array, dtype=float)
    v = np.asarray(var, dtype=float)
    start = len(R) // 3

    if class_type == 'I':
        normalized = v.copy()
    elif class_type == 'II':
        normalized = v / np.log(np.maximum(R, 1e-10))
    elif class_type == 'III':
        if alpha is None:
            raise ValueError("alpha is required for Class III metric")
        normalized = v / np.power(np.maximum(R, 1e-10), 1.0 - alpha)
    else:
        raise ValueError(f"Unknown class_type '{class_type}'. Use 'I', 'II', or 'III'.")

    metric     = float(np.mean(normalized[start:]))
    metric_err = float(np.std(normalized[start:]))
    return metric, metric_err, normalized
'''

qv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'quasicrystal_variance.py')
with open(qv_path, 'r') as f:
    content = f.read()

if 'compute_generalized_metric' not in content:
    # Insert before the if __name__ == '__main__' block
    insert_at = content.rfind('\n# ====')
    if insert_at == -1:
        insert_at = content.rfind('\nif __name__')
    if insert_at > 0:
        content = content[:insert_at] + PATCH + content[insert_at:]
        with open(qv_path, 'w') as f:
            f.write(content)
        print("  Patched successfully.")
    else:
        print("  Could not find insertion point; appending to file.")
        with open(qv_path, 'a') as f:
            f.write('\n' + PATCH)
else:
    print("  compute_generalized_metric already present; skipping patch.")
