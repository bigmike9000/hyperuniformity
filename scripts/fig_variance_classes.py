"""
Generate variance figures for the JP, consistent with Methods §3.

Figure 1: Class II (Period-Doubling) and Class III (0222) variance with fits.
  - Class II: sigma^2 = Lambda_II * ln(R) + b
  - Class III: sigma^2 = A * R^beta (both free)

Figure 2: Three-class overview (Class I bounded, II log, III power law)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

SEED = 2026
rng = np.random.default_rng(SEED)
NUM_WINDOWS = 25_000
TARGET_N = 500_000
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'jp', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)


def gen_chain(name, target_n):
    for iters in range(5, 70):
        if predict_chain_length(name, iters) > target_n:
            break
    seq = generate_substitution_sequence(name, iters)
    pts, L = sequence_to_points(seq, name)
    del seq
    return pts, L


# ============================================================
# Generate Period-Doubling (Class II, alpha=1)
# ============================================================
print("Generating Period-Doubling...")
alpha_pd, _, _ = verify_eigenvalue_prediction('period_doubling')
pts_pd, L_pd = gen_chain('period_doubling', TARGET_N)
rho_pd = len(pts_pd) / L_pd
mean_sp_pd = 1.0 / rho_pd
print(f"  N={len(pts_pd):,}, rho={rho_pd:.5f}, alpha={alpha_pd:.4f}")

R_pd = np.linspace(mean_sp_pd, min(2000 * mean_sp_pd, L_pd / 4), 1000)
var_pd, _ = compute_number_variance_1d(pts_pd, L_pd, R_pd, num_windows=NUM_WINDOWS, rng=rng)
del pts_pd

# Class II fit: sigma^2 = Lambda_II * ln(R) + b
mask_pd = R_pd > 20 * mean_sp_pd
popt_pd, pcov_pd = curve_fit(lambda R, C, b: C * np.log(R) + b,
                              R_pd[mask_pd], var_pd[mask_pd])
Lambda_II, b_pd = popt_pd
Lambda_II_err = np.sqrt(pcov_pd[0, 0])
print(f"  Class II fit: Lambda_II = {Lambda_II:.4f} +/- {Lambda_II_err:.4f}, b = {b_pd:.4f}")


# ============================================================
# Generate 0222 chain (Class III, alpha~0.639)
# ============================================================
print("Generating 0222 chain...")
alpha_0222, _, _ = verify_eigenvalue_prediction('chain_0222')
pts_0222, L_0222 = gen_chain('chain_0222', TARGET_N)
rho_0222 = len(pts_0222) / L_0222
mean_sp_0222 = 1.0 / rho_0222
print(f"  N={len(pts_0222):,}, rho={rho_0222:.5f}, alpha={alpha_0222:.4f}")

R_0222 = np.linspace(mean_sp_0222, min(2000 * mean_sp_0222, L_0222 / 4), 1000)
var_0222, _ = compute_number_variance_1d(pts_0222, L_0222, R_0222, num_windows=NUM_WINDOWS, rng=rng)
del pts_0222

# Class III fit: sigma^2 = A * R^beta (both free)
mask_0222 = R_0222 > 20 * mean_sp_0222
popt_0222, pcov_0222 = curve_fit(lambda R, A, beta: A * R**beta,
                                  R_0222[mask_0222], var_0222[mask_0222],
                                  p0=[0.5, 0.36], maxfev=5000)
A_0222, beta_0222 = popt_0222
alpha_0222_fit = 1.0 - beta_0222
A_err = np.sqrt(pcov_0222[0, 0])
beta_err = np.sqrt(pcov_0222[1, 1])
print(f"  Class III fit: A = {A_0222:.4f} +/- {A_err:.4f}, beta = {beta_0222:.4f} +/- {beta_err:.4f}")
print(f"  alpha from fit = {alpha_0222_fit:.4f}, alpha from eigenvalue = {alpha_0222:.4f}")


# ============================================================
# Generate Fibonacci (Class I, alpha=3) for overview
# ============================================================
print("Generating Fibonacci...")
pts_fib, L_fib = gen_chain('fibonacci', TARGET_N)
rho_fib = len(pts_fib) / L_fib
mean_sp_fib = 1.0 / rho_fib

R_fib = np.linspace(mean_sp_fib * 0.5, min(300 * mean_sp_fib, L_fib / 4), 800)
# Fibonacci uses catalog parameters (different from Class II/III which need longer R range)
var_fib, _ = compute_number_variance_1d(pts_fib, L_fib, R_fib, num_windows=NUM_WINDOWS, rng=rng)
lb_fib = compute_lambda_bar(R_fib, var_fib)
del pts_fib
print(f"  Lambda_bar = {lb_fib:.4f}")


# ============================================================
# Figure: Three-class overview
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: Class I
ax = axes[0]
ax.plot(R_fib, var_fib, color='#2ca02c', lw=0.5, alpha=0.8)
ax.axhline(lb_fib, color='red', ls='--', lw=1.5, label=rf'$\bar\Lambda = {lb_fib:.3f}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.set_title('Class I: Bounded', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05, f'Fibonacci, $\\alpha = 3$', transform=ax.transAxes,
        fontsize=9, va='bottom')

# Panel 2: Class II
ax = axes[1]
ax.plot(R_pd, var_pd, color='#ff7f0e', lw=0.5, alpha=0.8)
R_fit_pd = R_pd[mask_pd]
ax.plot(R_fit_pd, Lambda_II * np.log(R_fit_pd) + b_pd, 'k--', lw=1.5, alpha=0.7,
        label=rf'$\Lambda_{{II}}\ln R + b$, $\Lambda_{{II}} = {Lambda_II:.3f}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.set_title('Class II: Logarithmic Growth', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xscale('log')
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05, f'Period-Doubling, $\\alpha = 1$', transform=ax.transAxes,
        fontsize=9, va='bottom')

# Panel 3: Class III
ax = axes[2]
ax.plot(R_0222, var_0222, color='#17becf', lw=0.5, alpha=0.8)
R_fit_0222 = R_0222[mask_0222]
ax.plot(R_fit_0222, A_0222 * R_fit_0222**beta_0222, 'k--', lw=1.5, alpha=0.7,
        label=rf'$A R^\beta$, $\Lambda_{{III}} = {A_0222:.3f}$, $\alpha = {alpha_0222_fit:.2f}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.set_title('Class III: Power-Law Growth', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05, f'0222 chain, $\\alpha = {alpha_0222:.3f}$', transform=ax.transAxes,
        fontsize=9, va='bottom')

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'fig_variance_by_class.png')
fig.savefig(path, dpi=200, bbox_inches='tight')
fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {path}")

print("\nDone.")
