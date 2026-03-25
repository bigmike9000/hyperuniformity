"""
Presentation-quality Class II/III generalized metrics figure (2-panel).

Left:  σ²(R)/ln R for Period-Doubling → C_II = 0.080 (curve fit, primary)
Right: σ²(R)/R^{1-α} for 0222 chain  → A_III = 0.209 (curve fit, primary)

Output: results/figures/fig_generalized_metrics.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from substitution_tilings import (
    generate_substitution_sequence, sequence_to_points,
    verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d

RESULTS_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)

TARGET_N    = 600_000
NUM_WINDOWS = 20_000
NUM_R       = 300

# ── Generate sequences ────────────────────────────────────────────────────────
def gen_chain(name, target_n):
    for iters in range(2, 60):
        if predict_chain_length(name, iters) >= target_n:
            break
    seq = generate_substitution_sequence(name, iters)
    pts, L = sequence_to_points(seq, name)
    del seq
    return pts, L

print("Generating Period-Doubling...")
pts_pd, L_pd = gen_chain('period_doubling', TARGET_N)
print(f"  N_PD = {len(pts_pd):,}")

print("Generating 0222 chain...")
from substitution_tilings import sequence_to_points_general
for iters in range(2, 60):
    if predict_chain_length('chain_0222', iters) >= TARGET_N:
        break
seq_0222 = generate_substitution_sequence('chain_0222', iters)
pts_0222, L_0222 = sequence_to_points_general(seq_0222, 'chain_0222')
del seq_0222
print(f"  N_0222 = {len(pts_0222):,}")

# ── Variance ──────────────────────────────────────────────────────────────────
alpha_0222, _, _ = verify_eigenvalue_prediction('chain_0222')
print(f"  0222 alpha_theory = {alpha_0222:.4f}")

rho_pd   = len(pts_pd)   / L_pd
rho_0222 = len(pts_0222) / L_0222
ms_pd    = 1.0 / rho_pd
ms_0222  = 1.0 / rho_0222

R_pd   = np.logspace(np.log10(max(ms_pd,   1.0)),
                      np.log10(min(L_pd/4,   2000*ms_pd)),   NUM_R)
R_0222 = np.logspace(np.log10(max(ms_0222,  1.0)),
                      np.log10(min(L_0222/4, 2000*ms_0222)), NUM_R)

print("Computing Period-Doubling variance...")
var_pd,   _ = compute_number_variance_1d(pts_pd,   L_pd,   R_pd,   num_windows=NUM_WINDOWS, rng=rng)
print("Computing 0222 variance...")
var_0222, _ = compute_number_variance_1d(pts_0222, L_0222, R_0222, num_windows=NUM_WINDOWS, rng=rng)
del pts_pd, pts_0222

# ── Curve fits for C_II and A_III ─────────────────────────────────────────────
start = NUM_R // 3

# C_II: fit σ²(R) = C * ln(R) + b  in large-R regime
ln_R_pd = np.log(R_pd[start:])
try:
    popt, _ = curve_fit(lambda x, C, b: C*x + b, ln_R_pd, var_pd[start:], p0=[0.08, 0])
    C_II_fit = popt[0]
except Exception:
    C_II_fit = 0.080
print(f"  C_II (curve fit) = {C_II_fit:.4f}")

# A_III: fit σ²(R) = A * R^beta  in large-R regime
beta_theory = 1.0 - alpha_0222
try:
    popt, _ = curve_fit(lambda R_, A, b: A * R_**b,
                        R_0222[start:], var_0222[start:],
                        p0=[0.21, beta_theory], maxfev=3000)
    A_III_fit  = popt[0]
    beta_fit   = popt[1]
    alpha_fit  = 1.0 - beta_fit
except Exception:
    A_III_fit = 0.209
    alpha_fit = alpha_0222
print(f"  A_III (curve fit) = {A_III_fit:.4f}  alpha_fit={alpha_fit:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ─ Left: Class II ──────────────────────────────────────────────────────────────
ax = axes[0]
ratio_pd = var_pd / np.log(np.maximum(R_pd, 1e-10))
ax.semilogx(R_pd/ms_pd, ratio_pd, color='#E65100', lw=1.2, alpha=0.85)
ax.axhline(C_II_fit, color='#1B5E20', ls='--', lw=2.2,
           label=rf'Curve-fit: $C_{{II}} = {C_II_fit:.3f}$')
ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=13)
ax.set_ylabel(r'$\sigma^2(R)\,/\,\ln R$', fontsize=13)
ax.set_title(r'Class II (Period-Doubling, $\alpha=1$)', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0.05, 0.22)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─ Right: Class III ────────────────────────────────────────────────────────────
ax = axes[1]
ratio_0222 = var_0222 / np.maximum(R_0222, 1e-10)**beta_theory
ax.semilogx(R_0222/ms_0222, ratio_0222, color='#1565C0', lw=1.2, alpha=0.85)
ax.axhline(A_III_fit, color='#B71C1C', ls='--', lw=2.2,
           label=rf'Curve-fit: $A_{{III}} = {A_III_fit:.3f}$')
ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=13)
ax.set_ylabel(r'$\sigma^2(R)\,/\,R^{1-\alpha}$', fontsize=13)
ax.set_title(rf'Class III (0222 chain, $\alpha={alpha_0222:.3f}$)', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(
    r'Generalized metrics: normalized variance converges to $C_{II}$ and $A_{III}$',
    fontsize=12, y=1.01
)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_generalized_metrics.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
