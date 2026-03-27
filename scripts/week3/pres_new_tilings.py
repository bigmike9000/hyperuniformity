"""
Presentation figure for the cubic alpha=2 substitution tiling.

Left:  sigma^2(R) with Lambda_bar = 0.275 reference line
Right: Running Lambda_bar(R) converging to 0.275, with reference
       values for Silver (1/4), URL a=1 (1/3), Fibonacci (0.201)

Output: results/figures/fig_new_tilings.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points_general,
    verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d

RESULTS_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)

TARGET_N    = 2_000_000
NUM_WINDOWS = 30_000
NUM_R       = 800

# -- Generate cubic_alpha2 chain -----------------------------------------------
alpha_th, lam1, lam2 = verify_eigenvalue_prediction('cubic_alpha2')
print(f"cubic_alpha2: alpha={alpha_th:.4f}, lam1={lam1:.4f}, |lam2|={lam2:.4f}")

print(f"Generating cubic_alpha2 chain (~{TARGET_N:,} pts)...")
for iters in range(2, 60):
    if predict_chain_length('cubic_alpha2', iters) >= TARGET_N:
        break
seq = generate_substitution_sequence('cubic_alpha2', iters)
pts, L = sequence_to_points_general(seq, 'cubic_alpha2')
del seq
N = len(pts)
rho = N / L
ms = 1.0 / rho
print(f"  N={N:,}, rho={rho:.4f}, mean_spacing={ms:.4f}")

# -- Number variance -----------------------------------------------------------
R_arr = np.linspace(ms, min(L / 4, 2000 * ms), NUM_R)
print(f"Computing variance ({NUM_WINDOWS} windows, {NUM_R} R values)...")
var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
del pts

# -- Running Lambda_bar --------------------------------------------------------
running = np.zeros(NUM_R)
running[0] = var[0]
for i in range(1, NUM_R):
    dx = np.diff(R_arr[:i+1])
    integrand = 0.5 * (var[:i] + var[1:i+1])
    running[i] = np.dot(integrand, dx) / R_arr[i]

lb_mean = float(np.mean(running[NUM_R//3:]))
lb_err  = float(np.std(running[NUM_R//3:]))
print(f"  Lambda_bar = {lb_mean:.4f} +/- {lb_err:.4f}")

# -- Figure (single panel: sigma^2) --------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogx(R_arr / ms, var, color='#1565C0', lw=0.8, alpha=0.85,
            label=r'$\sigma^2(R)$')
ax.axhline(lb_mean, color='#B71C1C', ls='--', lw=2.0,
           label=rf'$\bar\Lambda = {lb_mean:.3f}$')
ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=14)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=14)
ax.set_title(rf'Cubic $\alpha=2$ chain: $a\!\to\!bc,\; b\!\to\!c,\; c\!\to\!abc$  ($N={N:,}$)',
             fontsize=13)
ax.legend(fontsize=12)
ax.set_xlim(1, R_arr[-1] / ms)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_new_tilings.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
