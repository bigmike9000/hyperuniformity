"""
Single-N convergence plot for BT Type I fitting.
Shows error vs fitting order for N = 6.8M.
"""
import os, sys
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from substitution_tilings import generate_substitution_sequence, sequence_to_points_general, predict_chain_length
from two_phase_media import (compute_structure_factor, compute_spectral_density,
                              compute_excess_spreadability)

PHI2 = 0.35
N_TARGET = 5_000_000
OUT = os.path.join(SCRIPT_DIR, 'results', 'figures', 'fig_yuan_improved.png')
alpha_true = 1.545

def type_I_fit_func(t, alpha_hat, C_coeffs):
    gamma = (1 + alpha_hat) / 2.0
    correction = np.zeros_like(t)
    for i, c in enumerate(C_coeffs):
        correction += c * t ** (-i / 2.0)
    correction = np.maximum(correction, 1e-30)
    return -gamma * np.log(t) + np.log(correction)

def fit_type_I(t, ln_sex, n_order):
    def residuals(params):
        return type_I_fit_func(t, params[0], params[1:]) - ln_sex
    x0 = np.zeros(n_order + 2)
    x0[0] = 2.0; x0[1] = 1.0
    result = least_squares(residuals, x0, method='lm', max_nfev=10000)
    return float(result.x[0])

# Generate BT chain
for n_iter in range(5, 70):
    if predict_chain_length('bombieri_taylor', n_iter) > N_TARGET:
        break
seq = generate_substitution_sequence('bombieri_taylor', n_iter)
points, L = sequence_to_points_general(seq, 'bombieri_taylor')
N = len(points)
rho = N / L
print(f"BT: N={N:,}")

a_rod = PHI2 / (2 * rho)
k_arr, S_k = compute_structure_factor(points, L)
chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
t_arr = np.logspace(0, 6, 500)
E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_arr, D=1.0)

mask = (E_t > 0) & (t_arr > 10)
t = t_arr[mask]
ln_sex = np.log(E_t[mask])

orders = list(range(7))
errs = []
alphas = []
for n in orders:
    alpha_hat = fit_type_I(t, ln_sex, n)
    err = abs(alpha_hat - alpha_true) / alpha_true * 100
    errs.append(err)
    alphas.append(alpha_hat)
    print(f"  order {n}: alpha = {alpha_hat:.3f} ({err:.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

# Left panel: error vs order
ax1.plot(orders, errs, 'o-', color='#084594', ms=8, lw=2, zorder=3)
ax1.axhline(47, color='#d62728', ls='--', lw=1.5, alpha=0.5,
           label='Simple log-slope (47%)')
ax1.axhline(10, color='gray', ls=':', lw=1, alpha=0.4)
ax1.text(-0.3, 10.5, '10%', fontsize=9, color='gray', va='bottom')

ax1.set_xlabel('Type I fitting order $n$', fontsize=13)
ax1.set_ylabel('Relative error (%)', fontsize=13)
ax1.set_title('(a) Error vs. fitting order', fontsize=13)
ax1.set_xticks(orders)
ax1.set_ylim(0, 55)
ax1.legend(fontsize=10, loc='upper right')
ax1.tick_params(labelsize=11)

# Right panel: extracted alpha vs order
ax2.plot(orders, alphas, 's-', color='#084594', ms=8, lw=2, zorder=3)
ax2.axhline(alpha_true, color='#2ca02c', ls='-', lw=2, alpha=0.7,
            label=f'True $\\alpha = {alpha_true}$')
ax2.axhline(alphas[0], color='#d62728', ls='--', lw=1.5, alpha=0.5,
            label=f'Simple log-slope ($\\hat\\alpha = {alphas[0]:.2f}$)')

ax2.set_xlabel('Type I fitting order $n$', fontsize=13)
ax2.set_ylabel('Extracted $\\hat\\alpha$', fontsize=13)
ax2.set_title('(b) Extracted exponent', fontsize=13)
ax2.set_xticks(orders)
ax2.set_ylim(1.0, 2.5)
ax2.legend(fontsize=10, loc='upper right')
ax2.tick_params(labelsize=11)

# no suptitle — info goes in the LaTeX caption
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"Saved: {OUT}")
