"""
Two-panel Bombieri-Taylor figure for JP §4.2:
  (a) Bounded variance σ²(R) with Λ̄ = 0.377
  (b) Spreadability decay with theoretical power law
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from substitution_tilings import generate_substitution_sequence, sequence_to_points, sequence_to_points_general, compute_tile_lengths, predict_chain_length
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar
from two_phase_media import compute_structure_factor, compute_spectral_density, compute_excess_spreadability

PHI2 = 0.35
N_TARGET = 1_000_000
OUT = os.path.join(SCRIPT_DIR, 'results', 'figures', 'fig_bt_2panel.png')

# Generate chain
for n_iter in range(5, 70):
    if predict_chain_length('bombieri_taylor', n_iter) > N_TARGET:
        break
seq = generate_substitution_sequence('bombieri_taylor', n_iter)
points, L = sequence_to_points_general(seq, 'bombieri_taylor')
N = len(points)
rho = N / L
print(f"BT: N={N:,}, rho={rho:.4f}")

# (a) Variance
R_array = np.linspace(0.5, L/4, 4000)
var, _ = compute_number_variance_1d(points, L, R_array, num_windows=30000)
lb = compute_lambda_bar(R_array, var)
print(f"  Lambda_bar = {lb:.4f}")

# (b) Spreadability
a_rod = PHI2 / (2 * rho)
k_arr, S_k = compute_structure_factor(points, L)
chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
t_arr = np.logspace(-2, 8, 500)
E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_arr, D=1.0)

alpha_true = 1.545
gamma = (1 + alpha_true) / 2

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): Variance
ax1.plot(R_array, var, color='#1f77b4', lw=0.3, alpha=0.8)
ax1.axhline(lb, color='#d62728', ls='--', lw=2, label=f'$\\bar\\Lambda = {lb:.3f}$')
ax1.set_xlabel('Window half-width $R$', fontsize=12)
ax1.set_ylabel('Number variance $\\sigma^2(R)$', fontsize=12)
ax1.set_title('(a) Bounded variance (Class I)', fontsize=13)
ax1.legend(fontsize=11, loc='lower right')
ax1.set_ylim(0, max(var)*1.15)
ax1.tick_params(labelsize=10)

# Panel (b): Spreadability
mask = E_t > 0
ax2.loglog(t_arr[mask], E_t[mask], color='#1f77b4', lw=2, label='Bombieri-Taylor')
# Theory line
t_theory = np.logspace(2, 7, 100)
# Fit amplitude from data at t=1e4
idx = np.argmin(np.abs(t_arr - 1e4))
if E_t[idx] > 0:
    C = E_t[idx] * (t_arr[idx])**gamma
    ax2.loglog(t_theory, C * t_theory**(-gamma), color='#d62728', ls='--', lw=2,
               label=f'$t^{{-{gamma:.2f}}}$ ($\\alpha = {alpha_true}$)')
ax2.set_xlabel('Diffusion time $t$', fontsize=12)
ax2.set_ylabel('Excess spreadability $E(t)$', fontsize=12)
ax2.set_title('(b) Spreadability decay', fontsize=13)
ax2.legend(fontsize=11, loc='lower left')
ax2.set_ylim(1e-14, 1)
ax2.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"Saved: {OUT}")
