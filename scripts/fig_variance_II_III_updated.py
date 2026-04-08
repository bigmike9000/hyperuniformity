"""
Generate updated 2-panel Class II/III variance figure for the JP.
Uses the same fitting approach as fig_variance_classes.py (R > 20*mean_spacing)
but the cleaner 2-panel layout from the week 2 presentation figure.
Saves to jp/figures/fig_variance_II_III.png
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
from quasicrystal_variance import compute_number_variance_1d

SEED = 2026
rng = np.random.default_rng(SEED)
NUM_WINDOWS = 25_000
TARGET_N = 500_000
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'jp', 'figures')

def gen_chain(name, target_n):
    for iters in range(5, 70):
        if predict_chain_length(name, iters) > target_n:
            break
    seq = generate_substitution_sequence(name, iters)
    pts, L = sequence_to_points(seq, name)
    del seq
    return pts, L

# ---- Period-Doubling (Class II) ----
print("Generating Period-Doubling...")
alpha_pd, _, _ = verify_eigenvalue_prediction('period_doubling')
pts_pd, L_pd = gen_chain('period_doubling', TARGET_N)
rho_pd = len(pts_pd) / L_pd
mean_sp_pd = 1.0 / rho_pd
print(f"  N={len(pts_pd):,}, rho={rho_pd:.5f}")

R_pd = np.linspace(mean_sp_pd, min(2000 * mean_sp_pd, L_pd / 4), 1000)
var_pd, _ = compute_number_variance_1d(pts_pd, L_pd, R_pd, num_windows=NUM_WINDOWS, rng=rng)
del pts_pd

mask_pd = R_pd > 20 * mean_sp_pd
popt_pd, _ = curve_fit(lambda R, C, b: C * np.log(R) + b,
                        R_pd[mask_pd], var_pd[mask_pd])
Lambda_II, b_pd = popt_pd
print(f"  Lambda_II = {Lambda_II:.4f}, b = {b_pd:.4f}")

# ---- 0222 Chain (Class III) ----
print("Generating 0222 chain...")
alpha_0222, _, _ = verify_eigenvalue_prediction('chain_0222')
pts_0222, L_0222 = gen_chain('chain_0222', TARGET_N)
rho_0222 = len(pts_0222) / L_0222
mean_sp_0222 = 1.0 / rho_0222
print(f"  N={len(pts_0222):,}, rho={rho_0222:.5f}")

R_0222 = np.linspace(mean_sp_0222, min(2000 * mean_sp_0222, L_0222 / 4), 1000)
var_0222, _ = compute_number_variance_1d(pts_0222, L_0222, R_0222, num_windows=NUM_WINDOWS, rng=rng)
del pts_0222

mask_0222 = R_0222 > 20 * mean_sp_0222
popt_0222, _ = curve_fit(lambda R, A, beta: A * R**beta,
                          R_0222[mask_0222], var_0222[mask_0222],
                          p0=[0.5, 0.36], maxfev=5000)
A_0222, beta_0222 = popt_0222
print(f"  A = {A_0222:.4f}, beta = {beta_0222:.4f}, alpha_fit = {1-beta_0222:.4f}")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(r'Number Variance $\sigma^2(R)$ — Classes II and III', fontsize=13)

# Left: Class II
ax = axes[0]
ax.set_title('Class II: Logarithmic Growth', fontsize=12, fontweight='bold')
ax.loglog(R_pd, var_pd, color='#ff7f0e', lw=1.8, label='Period-Doubling')
R_fit = R_pd[mask_pd]
ax.loglog(R_fit, Lambda_II * np.log(R_fit) + b_pd, 'k--', lw=1.8, alpha=0.7,
          label=rf'$C\ln R + b,\ C={Lambda_II:.3f}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05,
        r'Period-Doubling, $\alpha = 1$' '\n' r'$\sigma^2 \sim C\ln R$',
        transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Right: Class III
ax = axes[1]
ax.set_title('Class III: Power-Law Growth', fontsize=12, fontweight='bold')
ax.loglog(R_0222, var_0222, color='#17becf', lw=1.8, label='0222 Chain')
R_fit3 = R_0222[mask_0222]
ax.loglog(R_fit3, A_0222 * R_fit3**beta_0222, 'k--', lw=1.8, alpha=0.7,
          label=rf'$\sim R^{{{beta_0222:.3f}}}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05,
        rf'0222 Chain, $\alpha = {alpha_0222:.3f}$' '\n' r'$\sigma^2 \sim R^{1-\alpha}$',
        transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_variance_II_III.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
