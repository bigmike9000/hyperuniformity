"""
Generate figure for the non-unimodular chain M=[[0,1],[2,4]].
Rules: a -> b,  b -> aabbbb
alpha = 2.071  (first known 1D Class I quasicrystal with 2 < alpha < 3)
|det M| = 2,  lambda1 = 2+sqrt(6) ~ 4.449,  lambda2 = 2-sqrt(6) ~ -0.449

Two panels:
  (a) sigma^2(R) vs R (semilog x) — shows large log-periodic oscillations
  (b) running Lambda_bar(R) converging toward 0.650
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from quasicrystal_variance import compute_number_variance_1d

# ── Parameters ──────────────────────────────────────────────────────────────
LAM1 = 2.0 + np.sqrt(6.0)          # ~ 4.449   PF eigenvalue
LAM2 = abs(2.0 - np.sqrt(6.0))     # ~ 0.449   sub-leading |eigenvalue|
LOG_PERIOD = np.log(LAM1)          # ~ 1.493   period in log(R)
ALPHA_THEORY = 1.0 - 2.0*np.log(LAM2)/np.log(LAM1)  # = 2.071
LAMBDA_BAR   = 0.650
LAMBDA_ERR   = 0.015
N_ITER       = 10      # gives ~3M points from seed 'a'

# ── Generate sequence ────────────────────────────────────────────────────────
def generate_chain(n_iter):
    rules = {'a': 'b', 'b': 'aabbbb'}
    seq = 'a'
    for _ in range(n_iter):
        seq = ''.join(rules[c] for c in seq)
    return seq

def to_points(seq):
    """Tile a->1,  b->lambda1.  Points at left endpoints."""
    lengths = {'a': 1.0, 'b': LAM1}
    pos = []
    x = 0.0
    for c in seq:
        pos.append(x)
        x += lengths[c]
    return np.array(pos), x

print(f"Generating chain (iter={N_ITER})...")
seq = generate_chain(N_ITER)
pts, L = to_points(seq)
N = len(pts)
rho = N / L
print(f"  N={N:,}   L={L:.1f}   rho={rho:.4f}")

# ── Compute variance ─────────────────────────────────────────────────────────
R_min = 2.0
R_max = L / 4.0
n_R   = 400
R_arr = np.logspace(np.log10(R_min), np.log10(R_max), n_R)

print("Computing sigma^2(R)  (this may take ~30s)...")
rng = np.random.default_rng(42)
sig2, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=12000, rng=rng)
print("Done.")

# ── Running Lambda_bar ───────────────────────────────────────────────────────
running = np.zeros(n_R)
running[0] = sig2[0]
for i in range(1, n_R):
    dx = np.diff(R_arr[:i+1])
    integrand = 0.5*(sig2[:i] + sig2[1:i+1])
    running[i] = np.dot(integrand, dx) / R_arr[i]

# Period-aware running average: average running[] over complete log-periods
log_R = np.log(R_arr)
period_avgs = []
R_centers   = []
n_periods = int((log_R[-1] - log_R[0]) / LOG_PERIOD)
for k in range(n_periods):
    lo = log_R[0] + k * LOG_PERIOD
    hi = lo + LOG_PERIOD
    mask = (log_R >= lo) & (log_R < hi)
    if mask.sum() > 3:
        period_avgs.append(running[mask].mean())
        R_centers.append(np.exp(0.5*(lo+hi)))

period_mean = np.mean(period_avgs) if period_avgs else LAMBDA_BAR

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ─ Panel (a): sigma^2(R) ────────────────────────────────────────────────────
ax = axes[0]
ax.semilogx(R_arr, sig2, color='steelblue', lw=1.0, alpha=0.9)
ax.axhline(LAMBDA_BAR, color='crimson', ls='--', lw=1.5,
           label=rf'$\bar\Lambda \approx {LAMBDA_BAR}$ (rough est.)')

# Mark log-period boundaries
lR0 = log_R[0] + 0.5
for k in range(20):
    xv = np.exp(lR0 + k * LOG_PERIOD)
    if R_min < xv < R_max:
        ax.axvline(xv, color='gray', ls=':', lw=0.7, alpha=0.45)

# Annotate oscillation period
ax.text(0.97, 0.96,
        f"Period in $\\ln R$: $\\ln\\lambda_1 = {LOG_PERIOD:.3f}$\n"
        f"Amplitude: [{sig2.min():.3f}, {sig2.max():.3f}]",
        transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='goldenrod', alpha=0.9))

ax.set_xlabel('$R$', fontsize=13)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
ax.set_title(r'(a) Log-periodic oscillations in $\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim(bottom=-0.02)

# ─ Panel (b): running Lambda_bar ────────────────────────────────────────────
ax = axes[1]
ax.semilogx(R_arr, running, color='steelblue', lw=1.2,
            label=r'Running $\bar\Lambda(R)$')
if R_centers:
    ax.scatter(R_centers, period_avgs, color='darkorange', s=40, zorder=5,
               label=f'Period-avg ({len(period_avgs)} periods)')
ax.axhline(LAMBDA_BAR, color='crimson', ls='--', lw=1.5,
           label=rf'$\bar\Lambda \approx {LAMBDA_BAR}$ (rough est.)')
ax.fill_between([R_min, R_max],
                LAMBDA_BAR - LAMBDA_ERR, LAMBDA_BAR + LAMBDA_ERR,
                color='crimson', alpha=0.12)
ax.set_xlabel('$R$', fontsize=13)
ax.set_ylabel(r'Running $\bar\Lambda(R)$', fontsize=13)
ax.set_title(r'(b) Running average converges slowly (large oscillation amplitude)', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim([0.0, 1.2])

fig.suptitle(
    r'Non-unimodular chain $a\!\to\!b,\;b\!\to\!aabbbb$: '
    rf'$|\det M|=2$, $\alpha=2.071$, $N\approx{N:,}$',
    fontsize=12, y=1.01
)
plt.tight_layout()

out = os.path.join(BASE, 'results', 'figures', 'fig_nonunimodular.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
