"""
Presentation-quality Bombieri-Taylor figure (2-panel, clean).

Left:  σ²(R) for BT vs Fibonacci — large oscillations vs small,
       both with Λ̄ running-average lines.
Right: S(k) for BT vs Fibonacci — shows 12× more Bragg peaks in BT.

Output: results/figures/fig_bt_deep.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    sequence_to_points_general, verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d
from two_phase_media import compute_structure_factor

RESULTS_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)

# ── Parameters ────────────────────────────────────────────────────────────────
TARGET_N = 600_000
NUM_WINDOWS = 20_000
NUM_R = 300

alpha_bt,  lam1_bt,  lam2_bt  = verify_eigenvalue_prediction('bombieri_taylor')
alpha_fib, lam1_fib, lam2_fib = verify_eigenvalue_prediction('fibonacci')
period_log_bt = 2.0 * np.log(lam1_bt)

print(f"BT:  alpha={alpha_bt:.3f}, |lam2|={lam2_bt:.3f}, log-period={period_log_bt:.3f}")
print(f"Fib: alpha={alpha_fib:.3f}, |lam2|={lam2_fib:.3f}")

# ── Generate points ───────────────────────────────────────────────────────────
def gen_chain(name, target_n):
    chain = CHAINS[name]
    is_3letter = 'alphabet' in chain
    for iters in range(2, 60):
        if predict_chain_length(name, iters) >= target_n:
            break
    seq = generate_substitution_sequence(name, iters)
    if is_3letter:
        pts, L = sequence_to_points_general(seq, name)
    else:
        pts, L = sequence_to_points(seq, name)
    del seq
    return pts, L

print(f"Generating BT (~{TARGET_N:,} pts)...")
pts_bt, L_bt = gen_chain('bombieri_taylor', TARGET_N)
print(f"  N_BT={len(pts_bt):,}")

print(f"Generating Fibonacci (~{TARGET_N:,} pts)...")
pts_fib, L_fib = gen_chain('fibonacci', TARGET_N)
print(f"  N_Fib={len(pts_fib):,}")

# ── Number variance ───────────────────────────────────────────────────────────
rho_bt  = len(pts_bt)  / L_bt
rho_fib = len(pts_fib) / L_fib
ms_bt   = 1.0 / rho_bt
ms_fib  = 1.0 / rho_fib

R_bt  = np.logspace(np.log10(ms_bt),  np.log10(min(L_bt/4,  1500*ms_bt)),  NUM_R)
R_fib = np.logspace(np.log10(ms_fib), np.log10(min(L_fib/4, 1500*ms_fib)), NUM_R)

print("Computing BT variance...")
var_bt,  _ = compute_number_variance_1d(pts_bt,  L_bt,  R_bt,  num_windows=NUM_WINDOWS, rng=rng)
print("Computing Fibonacci variance...")
var_fib, _ = compute_number_variance_1d(pts_fib, L_fib, R_fib, num_windows=NUM_WINDOWS, rng=rng)

lb_bt  = float(np.mean(var_bt [len(var_bt)//3:]))
lb_fib = float(np.mean(var_fib[len(var_fib)//3:]))
print(f"  Lbar_BT={lb_bt:.4f}  Lbar_Fib={lb_fib:.4f}")

# ── Structure factor: count Bragg peaks ──────────────────────────────────────
K_MAX = 50
print("Computing structure factors...")
k_bt,  Sk_bt  = compute_structure_factor(pts_bt,  L_bt)
k_fib, Sk_fib = compute_structure_factor(pts_fib, L_fib)

# Restrict to k < K_MAX
mask_bt  = k_bt  < K_MAX
mask_fib = k_fib < K_MAX
k_bt,  Sk_bt  = k_bt[mask_bt],   Sk_bt[mask_bt]
k_fib, Sk_fib = k_fib[mask_fib], Sk_fib[mask_fib]

# Count peaks > threshold
threshold = 1.0
n_peaks_bt  = int(np.sum(Sk_bt  > threshold))
n_peaks_fib = int(np.sum(Sk_fib > threshold))
print(f"  BT peaks: {n_peaks_bt}  Fib peaks: {n_peaks_fib}")

del pts_bt, pts_fib

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ─ Left: σ²(R) ────────────────────────────────────────────────────────────────
ax = axes[0]

# Normalize R by mean spacing for readability
ax.semilogx(R_bt/ms_bt,  var_bt,  color='#1565C0', lw=0.9, alpha=0.85,
            label=f'Bombieri-Taylor  ($|\\lambda_2|={lam2_bt:.3f}$)')
ax.semilogx(R_fib/ms_fib, var_fib, color='#2E7D32', lw=0.9, alpha=0.85,
            label=f'Fibonacci  ($|\\lambda_2|={lam2_fib:.3f}$)')
ax.axhline(lb_bt,  color='#1565C0', ls='--', lw=1.8,
           label=rf'$\bar\Lambda_{{BT}}={lb_bt:.3f}$')
ax.axhline(lb_fib, color='#2E7D32', ls='--', lw=1.8,
           label=rf'$\bar\Lambda_{{Fib}}={lb_fib:.3f}$')

# Mark one log-period for BT
x_period = 10**(np.log10(R_bt[0]/ms_bt) + period_log_bt/2)
ax.annotate('', xy=(x_period * np.exp(period_log_bt/2), lb_bt*1.1),
            xytext=(x_period * np.exp(-period_log_bt/2), lb_bt*1.1),
            arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=1.2))
ax.text(x_period, lb_bt*1.12, f'period $={period_log_bt:.2f}$ in $\\ln R$',
        ha='center', va='bottom', fontsize=9, color='#1565C0')

ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=13)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
ax.set_title(r'Number variance: BT oscillates 12$\times$ more than Fibonacci', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(bottom=-0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─ Right: S(k) ────────────────────────────────────────────────────────────────
ax = axes[1]
ax.semilogy(k_bt,  Sk_bt,  color='#1565C0', lw=0.6, alpha=0.7,
            label=f'BT ({n_peaks_bt:,} peaks, $k<{K_MAX}$)')
ax.semilogy(k_fib, Sk_fib, color='#2E7D32', lw=0.6, alpha=0.7,
            label=f'Fibonacci ({n_peaks_fib:,} peaks, $k<{K_MAX}$)')
ax.axhline(threshold, color='gray', ls=':', lw=1.0, alpha=0.6,
           label='threshold = 1')

ax.set_xlabel(r'$k$', fontsize=13)
ax.set_ylabel(r'$S(k)$', fontsize=13)
ax.set_title(f'Dense Bragg spectrum: BT has {n_peaks_bt//n_peaks_fib}$\\times$ more peaks\n'
             r'$\Rightarrow$ spreadability $\mathcal{S}(t)$ cannot recover $\alpha$', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, K_MAX)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(
    r'Why $\bar\Lambda$ works for BT but spreadability does not',
    fontsize=13, y=1.01
)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_bt_deep.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
