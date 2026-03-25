"""
Presentation-quality Silver Λ̄ = 1/4 conjecture figure (2-panel).

Left:  σ²(R) vs R/mean-spacing — oscillates around 1/4
Right: Running Λ̄(R) − 1/4 — converges to ≈ 0

Uses N ~ 1M for speed (sufficient precision for the figure).
Output: results/figures/fig_silver_lambda_quarter.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from substitution_tilings import (
    generate_substitution_sequence, sequence_to_points,
    predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d

RESULTS_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)

TARGET_N    = 5_000_000
NUM_WINDOWS = 50_000
NUM_R       = 1000

# ── Generate Silver chain ──────────────────────────────────────────────────────
print(f"Generating Silver chain (~{TARGET_N:,} pts)...")
for iters in range(2, 60):
    if predict_chain_length('silver', iters) >= TARGET_N:
        break
seq = generate_substitution_sequence('silver', iters)
pts, L = sequence_to_points(seq, 'silver')
del seq
N = len(pts)
rho = N / L
ms  = 1.0 / rho   # mean spacing (= 2 for Silver)
print(f"  N={N:,}  rho={rho:.4f}  mean-spacing={ms:.4f}")

# ── Compute variance ──────────────────────────────────────────────────────────
R_max = min(L / 5, 3000 * ms)
R_arr = np.linspace(ms, R_max, NUM_R)

print(f"Computing variance ({NUM_WINDOWS} windows, {NUM_R} R values)...")
var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
del pts

# Running Λ̄(R)
running = np.zeros(NUM_R)
running[0] = var[0]
for i in range(1, NUM_R):
    dx = np.diff(R_arr[:i+1])
    integrand = 0.5*(var[:i] + var[1:i+1])
    running[i] = np.dot(integrand, dx) / R_arr[i]

lb_mean = float(np.mean(running[NUM_R//3:]))
lb_err  = float(np.std(running[NUM_R//3:]))
print(f"  Lbar = {lb_mean:.5f} +/- {lb_err:.5f}  (deviation from 1/4: {lb_mean-0.25:.5f})")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ─ Left: σ²(R) ────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(R_arr/ms, var, color='#388E3C', lw=0.8, alpha=0.85, label=r'$\sigma^2(R)$')
ax.axhline(0.25, color='#B71C1C', ls='--', lw=2.0, label=r'$1/4 = 0.2500$')
ax.axhline(lb_mean, color='#1565C0', ls='-', lw=1.5, alpha=0.6,
           label=rf'Mean $={lb_mean:.4f}$')
ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=13)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
ax.set_title(f'Silver number variance ($N={N:,}$)', fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(0, R_arr[-1]/ms)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─ Right: Running Λ̄(R) − 1/4 ─────────────────────────────────────────────────
ax = axes[1]
ax.plot(R_arr/ms, running - 0.25, color='#388E3C', lw=1.2,
        label=r'Running $\bar\Lambda(R) - 1/4$')
ax.axhline(0, color='#B71C1C', ls='--', lw=2.0, label=r'Exact $1/4$')
ax.fill_between(R_arr/ms, -lb_err, +lb_err,
                color='#B71C1C', alpha=0.12,
                label=rf'$\pm{lb_err:.4f}$ ($1\sigma$)')
ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=13)
ax.set_ylabel(r'Running $\bar\Lambda(R) - 1/4$', fontsize=13)
nsigma = abs(lb_mean - 0.25) / lb_err if lb_err > 0 else 0
ax.set_title(
    rf'Running $\bar\Lambda$ converges toward $1/4$ (see slide text for full result)',
    fontsize=11
)
ax.legend(fontsize=11)
ax.set_xlim(0, R_arr[-1]/ms)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(
    r'Conjecture: $\bar\Lambda(\mathrm{Silver}) = 1/4$ exactly',
    fontsize=13, y=1.01
)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_silver_lambda_quarter.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
