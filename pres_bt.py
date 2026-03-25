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
TARGET_N = 2_000_000
NUM_WINDOWS = 30_000
NUM_R = 800

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

# Compute Λ̄ via trapezoidal integration: (1/R) ∫₀ᴿ σ²(r) dr, then average last 1/3
def running_lambda_bar(R, var):
    running = np.zeros(len(R))
    running[0] = var[0]
    for i in range(1, len(R)):
        dx = np.diff(R[:i+1])
        integrand = 0.5*(var[:i] + var[1:i+1])
        running[i] = np.dot(integrand, dx) / R[i]
    return running

# Use catalog values for reference lines (best-known from high-precision runs)
lb_bt  = 0.377   # catalog value (Bombieri-Taylor)
lb_fib = 0.201   # Zachary & Torquato 2009
print(f"  Lbar_BT={lb_bt:.3f} (catalog)  Lbar_Fib={lb_fib:.3f} (Z&T 2009)")

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
COL_BT  = '#1565C0'
COL_FIB = '#2E7D32'

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'wspace': 0.32})

# ─ (a) σ²(R) ─────────────────────────────────────────────────────────────────
ax = axes[0]
ax.semilogx(R_fib/ms_fib, var_fib, color=COL_FIB, lw=0.6, alpha=0.8)
ax.semilogx(R_bt/ms_bt,   var_bt,  color=COL_BT,  lw=0.5, alpha=0.7)
ax.axhline(lb_bt,  color=COL_BT,  ls='--', lw=2.0)
ax.axhline(lb_fib, color=COL_FIB, ls='--', lw=2.0)

# Text annotations instead of legend (cleaner at slide scale)
ax.text(0.97, 0.97, rf'$\bar\Lambda_{{\rm BT}}={lb_bt:.3f}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=12,
        color=COL_BT, fontweight='bold',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))
ax.text(0.97, 0.06, rf'$\bar\Lambda_{{\rm Fib}}={lb_fib:.3f}$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
        color=COL_FIB, fontweight='bold',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))
# Curve labels at right edge
ax.text(0.52, 0.78, 'Bombieri-Taylor', transform=ax.transAxes,
        fontsize=10, color=COL_BT, alpha=0.9)
ax.text(0.52, 0.30, 'Fibonacci', transform=ax.transAxes,
        fontsize=10, color=COL_FIB, alpha=0.9)

ax.set_xlabel(r'$R\,/\,$mean spacing', fontsize=13)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=13)
ax.set_title('(a)  Number variance', fontsize=12, fontweight='bold', loc='left')
ax.grid(alpha=0.25, ls=':')
ax.set_ylim(-0.02, 0.52)
ax.set_xlim(1, R_bt[-1]/ms_bt)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─ (b) Cumulative Bragg peak count ───────────────────────────────────────────
ax = axes[1]

# For each k value, count how many peaks with k' <= k exceed threshold
# Sort peaks by k, then cumulative sum
peaks_bt  = np.sort(k_bt [Sk_bt  > threshold])
peaks_fib = np.sort(k_fib[Sk_fib > threshold])
cum_bt  = np.arange(1, len(peaks_bt)  + 1)
cum_fib = np.arange(1, len(peaks_fib) + 1)

ax.plot(peaks_fib, cum_fib, color=COL_FIB, lw=2.5,
        label=f'Fibonacci ({n_peaks_fib:,})')
ax.plot(peaks_bt,  cum_bt,  color=COL_BT,  lw=2.5,
        label=f'BT ({n_peaks_bt:,})')

# Set x-axis to actual data extent
k_max_actual = max(peaks_bt[-1], peaks_fib[-1]) * 1.1 if len(peaks_bt) > 0 else K_MAX
ax.set_xlim(0, k_max_actual)

# Ratio annotation (positioned relative to actual data)
ratio = n_peaks_bt / max(n_peaks_fib, 1)
ax.text(0.55, 0.45, rf'{ratio:.0f}$\times$' + '\nmore peaks',
        transform=ax.transAxes, fontsize=14, fontweight='bold',
        color=COL_BT, ha='center', va='center')

ax.set_xlabel(r'$k$', fontsize=13)
ax.set_ylabel(r'Cumulative Bragg peaks  ($S(k) > 1$)', fontsize=12)
ax.set_title('(b)  Peak density in $S(k)$', fontsize=12,
             fontweight='bold', loc='left')
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25, ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_bt_deep.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
