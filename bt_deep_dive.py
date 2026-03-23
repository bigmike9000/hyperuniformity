"""
Investigation 1: Bombieri-Taylor Deep Dive

Detailed analysis of BT oscillation structure and spreadability diagnosis.
- Fine-resolution sigma^2(R) with oscillation periods labeled
- S(k) wide k range vs Fibonacci comparison
- E(t) for large-N BT (N~1M) with standard and period-aware fits
- Rod overlap diagnostic

Output: results/figures/fig_bt_deep.png

Theoretical values:
  theta_1 ~ 2.247, |lambda_2| ~ 0.802
  alpha = 1.545 (Class I)
  Oscillation period in log(R): 2*ln(theta_1) ~ 1.618
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    sequence_to_points_general, verify_eigenvalue_prediction,
    predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar
from two_phase_media import (
    compute_structure_factor, compute_spectral_density,
    compute_excess_spreadability, extract_alpha_fit,
    extract_alpha_period_aware, extract_alpha,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)

# ============================================================
# Theoretical parameters
# ============================================================
alpha_pred, lam1, lam2 = verify_eigenvalue_prediction('bombieri_taylor')
theta1 = lam1
theta2 = lam2
period_log = 2.0 * np.log(theta1)   # oscillation period in log(R) or log(t)

print("=" * 65)
print("  Investigation 1: Bombieri-Taylor Deep Dive")
print("=" * 65)
print(f"  theta_1  = {theta1:.6f}")
print(f"  |lambda_2| = {theta2:.6f}")
print(f"  alpha (theory) = {alpha_pred:.6f}")
print(f"  Oscillation period in log(R): 2*ln(theta_1) = {period_log:.6f}")

# Fibonacci baseline
alpha_fib, lam1_fib, lam2_fib = verify_eigenvalue_prediction('fibonacci')
period_fib = 2.0 * np.log(lam1_fib)
print(f"\n  Fibonacci: theta_1={lam1_fib:.4f}, |lambda_2|={lam2_fib:.4f}")
print(f"    period = {period_fib:.4f}, alpha = {alpha_fib:.4f}")

# ============================================================
# Generate BT at N ~ 1M
# ============================================================
print("\n[1] Generating BT at N ~ 1M ...")
target_N = 1_000_000
for num_iters in range(10, 40):
    if predict_chain_length('bombieri_taylor', num_iters) >= target_N:
        break
n_pred = predict_chain_length('bombieri_taylor', num_iters)
print(f"    {num_iters} iterations -> predicted N = {n_pred:,}")

t0 = time.perf_counter()
seq_bt = generate_substitution_sequence('bombieri_taylor', num_iters)
pts_bt, L_bt = sequence_to_points_general(seq_bt, 'bombieri_taylor')
del seq_bt
N_bt = len(pts_bt)
rho_bt = N_bt / L_bt
print(f"    Actual N = {N_bt:,}, L = {L_bt:.1f}, rho = {rho_bt:.6f}")
print(f"    Generation time: {time.perf_counter()-t0:.2f}s")

# Generate Fibonacci at similar N for comparison
print("\n    Generating Fibonacci at N ~ 1M for comparison ...")
for num_iters_fib in range(5, 40):
    if predict_chain_length('fibonacci', num_iters_fib) >= target_N:
        break
t0 = time.perf_counter()
seq_fib = generate_substitution_sequence('fibonacci', num_iters_fib)
pts_fib, L_fib = sequence_to_points(seq_fib, 'fibonacci')
del seq_fib
N_fib = len(pts_fib)
rho_fib = N_fib / L_fib
print(f"    N_fib = {N_fib:,}, rho_fib = {rho_fib:.6f}, time={time.perf_counter()-t0:.2f}s")

# ============================================================
# Panel (a): sigma^2(R) fine resolution
# ============================================================
print("\n[2] Computing sigma^2(R) (fine resolution) ...")
R_max_sens = N_bt / (2.0 * rho_bt)
R_fine = np.linspace(0.3 / rho_bt, min(R_max_sens * 0.8, 3000.0), 2000)
t0 = time.perf_counter()
var_bt, _ = compute_number_variance_1d(pts_bt, L_bt, R_fine,
                                        num_windows=20000, rng=rng)
print(f"    Done in {time.perf_counter()-t0:.1f}s")
lambda_bar_bt = compute_lambda_bar(R_fine, var_bt)
print(f"    Lambda_bar = {lambda_bar_bt:.4f}")
print(f"    Variance range: [{np.min(var_bt):.4f}, {np.max(var_bt):.4f}]")

# Also compute Fibonacci variance for comparison
R_fib = np.linspace(0.3 / rho_fib, min(N_fib / (2.0 * rho_fib) * 0.8, 3000.0), 2000)
var_fib, _ = compute_number_variance_1d(pts_fib, L_fib, R_fib,
                                         num_windows=20000, rng=rng)
lambda_bar_fib = compute_lambda_bar(R_fib, var_fib)
print(f"    Fibonacci Lambda_bar = {lambda_bar_fib:.4f}")

# ============================================================
# Panel (b): S(k) wide range
# ============================================================
print("\n[3] Computing S(k) (wide k range) ...")

t0 = time.perf_counter()
k_bt, S_bt = compute_structure_factor(pts_bt, L_bt)
print(f"    BT S(k): {len(k_bt):,} k-points, k_max={k_bt[-1]:.3f}, time={time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
k_fib, S_fib = compute_structure_factor(pts_fib, L_fib)
print(f"    Fibonacci S(k): {len(k_fib):,} k-points, time={time.perf_counter()-t0:.1f}s")

# Count Bragg peaks above threshold in fixed k range
k_win_max = 50.0
thresh = 1.0  # peaks with S(k) > thresh
n_peaks_bt  = np.sum((k_bt < k_win_max) & (S_bt > thresh))
n_peaks_fib = np.sum((k_fib < k_win_max) & (S_fib > thresh))
print(f"    Bragg peaks > {thresh} in k < {k_win_max}: BT={n_peaks_bt}, Fib={n_peaks_fib}")

# ============================================================
# Panel (c): Spreadability E(t) with fits
# ============================================================
print("\n[4] Computing spreadability E(t) ...")

# Check rod overlap
min_sp_bt = np.min(np.diff(np.sort(pts_bt)))
phi2 = 0.35
a_rod_bt = phi2 / (2.0 * rho_bt)
overlap_bt = 2.0 * a_rod_bt > min_sp_bt
print(f"    Rod check: 2a = {2*a_rod_bt:.4f}, min_spacing = {min_sp_bt:.4f}")
if overlap_bt:
    a_rod_bt = 0.45 * min_sp_bt
    print(f"    OVERLAP detected: adjusted a_rod = {a_rod_bt:.4f}")
else:
    print(f"    No overlap (OK)")

chi_V_bt = compute_spectral_density(k_bt, S_bt, rho_bt, a_rod_bt)

# Fibonacci rod
min_sp_fib = np.min(np.diff(np.sort(pts_fib)))
a_rod_fib = phi2 / (2.0 * rho_fib)
if 2.0 * a_rod_fib > min_sp_fib:
    a_rod_fib = 0.45 * min_sp_fib
chi_V_fib = compute_spectral_density(k_fib, S_fib, rho_fib, a_rod_fib)

t_values = np.logspace(0, 10, 400)
print(f"    Computing E(t) for BT ...")
t0 = time.perf_counter()
E_bt = compute_excess_spreadability(k_bt, chi_V_bt, phi2, t_values)
print(f"    BT E(t) done in {time.perf_counter()-t0:.1f}s")

print(f"    Computing E(t) for Fibonacci ...")
t0 = time.perf_counter()
E_fib = compute_excess_spreadability(k_fib, chi_V_fib, phi2, t_values)
print(f"    Fibonacci E(t) done in {time.perf_counter()-t0:.1f}s")

# Standard fit
alpha_std_bt, r2_std_bt = extract_alpha_fit(t_values, E_bt, t_min=1e3, t_max=1e8)
alpha_std_fib, r2_std_fib = extract_alpha_fit(t_values, E_fib, t_min=1e3, t_max=1e8)
print(f"\n    BT standard fit:  alpha = {alpha_std_bt:.4f}  (R^2 = {r2_std_bt:.4f})")
print(f"    Fib standard fit: alpha = {alpha_std_fib:.4f}  (R^2 = {r2_std_fib:.4f})")

# Period-aware fit
alpha_pa_arr_bt = extract_alpha_period_aware(t_values, E_bt,
                                              period=period_log, n_periods=2)
valid_bt = ~np.isnan(alpha_pa_arr_bt)
if np.any(valid_bt):
    va = alpha_pa_arr_bt[valid_bt]
    # Use middle third for stable estimate
    alpha_pa_bt = float(np.median(va[len(va)//3 : 2*len(va)//3]))
    print(f"    BT period-aware:  alpha = {alpha_pa_bt:.4f}")
else:
    alpha_pa_bt = np.nan
    print(f"    BT period-aware:  no valid points")

alpha_pa_arr_fib = extract_alpha_period_aware(t_values, E_fib,
                                               period=period_fib, n_periods=2)
valid_fib = ~np.isnan(alpha_pa_arr_fib)
if np.any(valid_fib):
    va_f = alpha_pa_arr_fib[valid_fib]
    alpha_pa_fib = float(np.median(va_f[len(va_f)//3 : 2*len(va_f)//3]))
    print(f"    Fib period-aware: alpha = {alpha_pa_fib:.4f}")
else:
    alpha_pa_fib = np.nan

# Local derivative alpha(t)
alpha_local_bt  = extract_alpha(t_values, E_bt,  window=12)
alpha_local_fib = extract_alpha(t_values, E_fib, window=12)

# ============================================================
# Wider t-range fit attempts for BT
# ============================================================
fit_ranges = [
    (1e2, 1e5),
    (1e3, 1e7),
    (1e4, 1e8),
    (1e5, 1e9),
]
print("\n    BT fit quality at different t ranges:")
for t_lo, t_hi in fit_ranges:
    a_fit, r2 = extract_alpha_fit(t_values, E_bt, t_min=t_lo, t_max=t_hi)
    print(f"      t in [{t_lo:.0e}, {t_hi:.0e}]: alpha={a_fit:.4f}  R^2={r2:.4f}")

# ============================================================
# Generate 4-panel figure
# ============================================================
print("\n[5] Generating fig_bt_deep.png ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ---- Panel (a): sigma^2(R) fine resolution ----
ax = axes[0, 0]
ax.semilogx(R_fine, var_bt, 'b-', lw=0.7, alpha=0.85, label='Bombieri-Taylor')
ax.semilogx(R_fib, var_fib, 'g-', lw=0.7, alpha=0.6, label='Fibonacci')
ax.axhline(lambda_bar_bt, color='b', ls='--', lw=1.5, alpha=0.7,
           label=rf'BT $\bar{{\Lambda}}={lambda_bar_bt:.3f}$')
ax.axhline(lambda_bar_fib, color='g', ls='--', lw=1.5, alpha=0.7,
           label=rf'Fib $\bar{{\Lambda}}={lambda_bar_fib:.3f}$')

# Mark oscillation period: theta_1^n multiples starting from some reference R
R_ref0 = 10.0
for i in range(6):
    R_m = R_ref0 * theta1**i
    if R_m < R_fine[-1]:
        ax.axvline(R_m, color='orange', ls=':', lw=1.0, alpha=0.5)

ax.text(0.02, 0.97,
        rf'BT oscillation period $= \theta_1^n$ steps ($\theta_1={theta1:.3f}$)' '\n'
        rf'$\ln(\theta_1) = {np.log(theta1):.3f}$, period in $\ln R$: ${period_log:.3f}$',
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
ax.set_xlabel(r'$R$', fontsize=11)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=11)
ax.set_title(r'(a) Number Variance $\sigma^2(R)$', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, ls=':', alpha=0.4)

# ---- Panel (b): S(k) Bragg peak density ----
ax = axes[0, 1]
k_max_plot = 30.0
m_bt_k  = (k_bt  > 0) & (k_bt  < k_max_plot)
m_fib_k = (k_fib > 0) & (k_fib < k_max_plot)

# Plot S(k) — log scale for clarity
S_min_plot = 1e-4
ax.semilogy(k_bt[m_bt_k],  np.maximum(S_bt[m_bt_k],  S_min_plot), 'b-',
            lw=0.5, alpha=0.7, label=f'BT  (N={N_bt:,})')
ax.semilogy(k_fib[m_fib_k], np.maximum(S_fib[m_fib_k], S_min_plot), 'g-',
            lw=0.5, alpha=0.7, label=f'Fib (N={N_fib:,})')
ax.text(0.55, 0.97,
        f'Bragg peaks > {thresh:.0f} in $k<{k_win_max:.0f}$:\n'
        f'  BT: {n_peaks_bt}\n  Fib: {n_peaks_fib}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
ax.set_xlabel(r'$k$', fontsize=11)
ax.set_ylabel(r'$S(k)$', fontsize=11)
ax.set_title(r'(b) Structure Factor: Bragg Peak Density', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.set_xlim(0, k_max_plot)

# ---- Panel (c): E(t) log-log ----
ax = axes[1, 0]
mask_bt  = (E_bt  > 0) & (t_values > 0.5)
mask_fib = (E_fib > 0) & (t_values > 0.5)
ax.loglog(t_values[mask_bt],  E_bt[mask_bt],  'b-', lw=1.5, label='BT E(t)')
ax.loglog(t_values[mask_fib], E_fib[mask_fib], 'g-', lw=1.5, alpha=0.7, label='Fibonacci E(t)')

# Reference slopes
t_ref = t_values[(t_values > 3e2) & (t_values < 3e7)]
if len(t_ref) > 2:
    i_start = np.argmin(np.abs(t_values - t_ref[0]))
    if mask_bt[i_start]:
        E0 = E_bt[i_start]
        E_ref_bt = E0 * (t_ref / t_ref[0]) ** (-(1 + alpha_pred) / 2)
        ax.loglog(t_ref, E_ref_bt, 'b--', lw=2, alpha=0.6,
                  label=rf'Theory $\alpha={alpha_pred:.3f}$: $t^{{-{(1+alpha_pred)/2:.3f}}}$')
    if mask_fib[i_start]:
        E0f = E_fib[i_start]
        E_ref_fib = E0f * (t_ref / t_ref[0]) ** (-2.0)
        ax.loglog(t_ref, E_ref_fib, 'g--', lw=2, alpha=0.6, label=r'Theory $\alpha=3$: $t^{-2}$')

diag_str = (
    f'Rod overlap: {overlap_bt}\n'
    f'BT std fit:  $\\alpha=${alpha_std_bt:.3f}  ($R^2$={r2_std_bt:.3f})\n'
    f'BT period:   $\\alpha=${alpha_pa_bt:.3f}\n'
    f'Fib std fit: $\\alpha=${alpha_std_fib:.3f}  ($R^2$={r2_std_fib:.3f})'
)
ax.text(0.02, 0.05, diag_str,
        transform=ax.transAxes, va='bottom', fontsize=8,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
ax.set_xlabel(r'$t$', fontsize=11)
ax.set_ylabel(r'$\mathcal{E}(t)$', fontsize=11)
ax.set_title(r'(c) Excess Spreadability $\mathcal{E}(t)$', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, ls=':', alpha=0.4)

# ---- Panel (d): local alpha(t) ----
ax = axes[1, 1]
v_bt_loc  = ~np.isnan(alpha_local_bt)
v_fib_loc = ~np.isnan(alpha_local_fib)
ax.semilogx(t_values[v_bt_loc],  alpha_local_bt[v_bt_loc],  'b-', lw=1.2, alpha=0.8,
            label=r'BT $\alpha(t)$ (standard window)')
ax.semilogx(t_values[v_fib_loc], alpha_local_fib[v_fib_loc], 'g-', lw=1.2, alpha=0.7,
            label=r'Fib $\alpha(t)$ (standard)')

v_pa_bt = ~np.isnan(alpha_pa_arr_bt)
if np.any(v_pa_bt):
    ax.semilogx(t_values[v_pa_bt], alpha_pa_arr_bt[v_pa_bt], 'r-', lw=1.5,
                label=r'BT $\alpha(t)$ (period-aware)')

ax.axhline(alpha_pred, color='b', ls='--', lw=1.5, alpha=0.6,
           label=rf'Theory BT: $\alpha={alpha_pred:.3f}$')
ax.axhline(3.0, color='g', ls='--', lw=1.5, alpha=0.6, label=r'Theory Fib: $\alpha=3$')

ax.set_xlabel(r'$t$', fontsize=11)
ax.set_ylabel(r'Local $\hat{\alpha}(t)$', fontsize=11)
ax.set_title(r'(d) Local Exponent: Standard vs Period-Aware', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, ls=':', alpha=0.4)
ax.set_ylim(-1, 7)

fig.suptitle(
    rf'BT Deep Dive: $\alpha_\mathrm{{theory}}={alpha_pred:.3f}$, '
    rf'$\theta_1={theta1:.3f}$, $|\lambda_2|={theta2:.3f}$, '
    rf'period $= 2\ln\theta_1 = {period_log:.3f}$'
    f'\n(N={N_bt:,}, rod_overlap={overlap_bt})',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'fig_bt_deep.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {out_path}")

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 65)
print("  SUMMARY: BT Spreadability Diagnosis")
print("=" * 65)
print(f"  Theory alpha:            {alpha_pred:.4f}")
print(f"  Standard fit alpha:      {alpha_std_bt:.4f}  (R^2 = {r2_std_bt:.4f})")
print(f"  Period-aware fit alpha:  {alpha_pa_bt:.4f}")
print(f"  Fibonacci baseline:      {alpha_std_fib:.4f}  (R^2 = {r2_std_fib:.4f})")
print(f"  |lambda_2| = {theta2:.4f}  (Fibonacci |lambda_2| = {lam2_fib:.4f})")
print()
if r2_std_bt < 0.9:
    print("  DIAGNOSIS: Standard fit FAILS (R^2 < 0.9).")
    print(f"  CAUSE: Large |lambda_2| = {theta2:.3f} creates strong oscillations in E(t).")
    print(f"  Period = {period_log:.3f} in ln(t) -- poor cancellation with fixed window.")
if not np.isnan(alpha_pa_bt):
    err_std = abs(alpha_std_bt - alpha_pred) / alpha_pred * 100
    err_pa  = abs(alpha_pa_bt  - alpha_pred) / alpha_pred * 100
    print(f"  Standard error: {err_std:.1f}%  |  Period-aware error: {err_pa:.1f}%")
    if err_pa < err_std:
        print("  Period-aware fit IMPROVES accuracy.")
    else:
        print("  Period-aware fit does NOT clearly improve accuracy at this N.")
print("=" * 65)
