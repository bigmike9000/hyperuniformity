"""
Finite-size convergence figure for Week 3 presentation.

Panel (a): Lbar(N) for BT and Fibonacci — flat from N~1000
Panel (b): alpha_hat vs fitting window for BT (large N)
           Shows BT slowly converges to true alpha=1.545 at large t.

Output: results/figures/fig_finite_size_alpha.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    sequence_to_points_general, verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d
from two_phase_media import (
    compute_structure_factor, compute_spectral_density,
    compute_excess_spreadability, extract_alpha_fit,
)

RESULTS_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)

LB_BT_REF  = 0.377
LB_FIB_REF = 0.201
ALPHA_BT_REF  = 1.545
PHI2 = 0.35

N_TARGETS = [300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]


def generate_chain(chain_name, target_n):
    chain = CHAINS[chain_name]
    is_3letter = 'alphabet' in chain
    for iters in range(2, 60):
        if predict_chain_length(chain_name, iters) >= target_n:
            break
    seq = generate_substitution_sequence(chain_name, iters)
    if is_3letter:
        pts, L = sequence_to_points_general(seq, chain_name)
    else:
        pts, L = sequence_to_points(seq, chain_name)
    del seq
    return pts, L


def compute_lambda_bar_at_N(chain_name, target_n, num_windows=None, num_R=300):
    pts, L = generate_chain(chain_name, target_n)
    N = len(pts)
    rho = N / L
    ms = 1.0 / rho
    if num_windows is None:
        num_windows = min(max(500, N // 5), 20_000)
    R_max = min(L / 4, 500 * ms)
    R_arr = np.linspace(ms, R_max, num_R)
    var, _ = compute_number_variance_1d(pts, L, R_arr,
                                        num_windows=num_windows, rng=rng)
    del pts
    running = np.zeros(num_R)
    running[0] = var[0]
    for i in range(1, num_R):
        dx = np.diff(R_arr[:i+1])
        integrand = 0.5 * (var[:i] + var[1:i+1])
        running[i] = np.dot(integrand, dx) / R_arr[i]
    start = num_R // 3
    lb = float(np.mean(running[start:]))
    lb_err = float(np.std(running[start:]))
    return N, lb, lb_err


def compute_bt_alpha_vs_window(target_n, window_width=3.0):
    """Compute BT spreadability alpha_hat as a function of fitting window."""
    pts, L = generate_chain('bombieri_taylor', target_n)
    N = len(pts)
    rho = N / L
    a = PHI2 / (2 * rho)
    pts_sorted = np.sort(pts)
    min_gap = np.min(np.diff(pts_sorted))
    if 2 * a >= min_gap:
        a = min_gap * 0.45

    k_arr, S_k = compute_structure_factor(pts, L)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a)
    del pts

    t_arr = np.logspace(0, 12, 1000)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_arr)

    results = []
    for log_center in np.arange(1.5 + window_width/2, 12.0 - window_width/2 + 0.1, 0.5):
        t_min = 10 ** (log_center - window_width / 2)
        t_max = 10 ** (log_center + window_width / 2)
        alpha_hat, r2 = extract_alpha_fit(t_arr, E_t, t_min=t_min, t_max=t_max)
        if not np.isnan(alpha_hat) and r2 > 0.95:
            results.append((log_center, alpha_hat, r2))

    return N, results


# ── Compute ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Finite-Size Convergence Analysis")
print("=" * 60)

# Panel (a): Lbar(N)
bt_lb = []
fib_lb = []
for tgt in N_TARGETS:
    print(f"\n  BT Lbar at N~{tgt:,}...", end=' ', flush=True)
    t0 = time.perf_counter()
    N, lb, err = compute_lambda_bar_at_N('bombieri_taylor', tgt)
    bt_lb.append((N, lb, err))
    print(f"N={N:,}, Lbar={lb:.4f}+/-{err:.4f} [{time.perf_counter()-t0:.1f}s]")

    print(f"  Fib Lbar at N~{tgt:,}...", end=' ', flush=True)
    t0 = time.perf_counter()
    N, lb, err = compute_lambda_bar_at_N('fibonacci', tgt)
    fib_lb.append((N, lb, err))
    print(f"N={N:,}, Lbar={lb:.4f}+/-{err:.4f} [{time.perf_counter()-t0:.1f}s]")

bt_lb  = np.array(bt_lb)
fib_lb = np.array(fib_lb)

# Panel (b): BT alpha vs fitting window
print("\n  BT spreadability vs window (N~500k)...")
t0 = time.perf_counter()
N_bt, bt_windows = compute_bt_alpha_vs_window(500_000)
print(f"  N={N_bt:,}, {len(bt_windows)} windows [{time.perf_counter()-t0:.1f}s]")
for lc, ah, r2 in bt_windows:
    print(f"    center={lc:.1f}  alpha={ah:.3f}  R2={r2:.4f}")

# ── Figure ───────────────────────────────────────────────────────────────────
COL_BT  = '#D32F2F'
COL_FIB = '#2E7D32'

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'wspace': 0.30})

# ── (a) Lbar(N) ──────────────────────────────────────────────────────────────
ax = axes[0]
ax.errorbar(bt_lb[:, 0], bt_lb[:, 1], yerr=bt_lb[:, 2],
            fmt='o-', color=COL_BT, lw=1.8, ms=7, capsize=4,
            label=rf'BT ($\alpha=1.545$)')
ax.errorbar(fib_lb[:, 0], fib_lb[:, 1], yerr=fib_lb[:, 2],
            fmt='s-', color=COL_FIB, lw=1.8, ms=7, capsize=4,
            label=rf'Fibonacci ($\alpha=3$)')
ax.axhline(LB_BT_REF,  color=COL_BT,  ls='--', lw=1.5, alpha=0.5)
ax.axhline(LB_FIB_REF, color=COL_FIB, ls='--', lw=1.5, alpha=0.5)
for ref, col in [(LB_BT_REF, COL_BT), (LB_FIB_REF, COL_FIB)]:
    ax.axhspan(ref * 0.99, ref * 1.01, color=col, alpha=0.08)
ax.text(2e6, LB_BT_REF + 0.005, rf'$\bar\Lambda_{{\rm BT}}={LB_BT_REF}$',
        fontsize=10, color=COL_BT, va='bottom', ha='right')
ax.text(2e6, LB_FIB_REF - 0.005, rf'$\bar\Lambda_{{\rm Fib}}={LB_FIB_REF}$',
        fontsize=10, color=COL_FIB, va='top', ha='right')
ax.axvspan(1e3, bt_lb[-1, 0] * 2, color='green', alpha=0.04)
ax.text(3e4, 0.42, 'stable from $N{\\sim}10^3$',
        fontsize=11, color='#2E7D32', ha='center', style='italic')
ax.set_xscale('log')
ax.set_xlabel(r'$N$ (chain length)', fontsize=13)
ax.set_ylabel(r'$\bar\Lambda(N)$', fontsize=13)
ax.set_title(r'(a)  $\bar\Lambda$ from $\sigma^2(R)$: converges rapidly',
             fontsize=12, fontweight='bold', loc='left')
ax.legend(fontsize=11, loc='center left')
ax.set_xlim(150, bt_lb[-1, 0] * 3)
ax.set_ylim(0.14, 0.46)
ax.grid(alpha=0.25, ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (b) BT alpha vs fitting window ──────────────────────────────────────────
ax = axes[1]
bt_w = np.array(bt_windows)

# Trim noise-floor points: stop at the minimum alpha_hat
min_idx = np.argmin(bt_w[:, 1])
bt_w = bt_w[:min_idx + 1]

ax.plot(bt_w[:, 0], bt_w[:, 1], 'o-', color=COL_BT, lw=1.8, ms=7,
        label=rf'BT ($|\lambda_2|=0.802$, $N={N_bt:,}$)')
ax.axhline(ALPHA_BT_REF, color=COL_BT, ls='--', lw=1.5, alpha=0.5,
           label=rf'True $\alpha={ALPHA_BT_REF}$')

# Annotate convergence
ax.annotate(rf'$\hat\alpha \to {ALPHA_BT_REF}$',
            xy=(bt_w[-1, 0], bt_w[-1, 1]),
            xytext=(bt_w[-1, 0] - 1.5, bt_w[-1, 1] + 0.3),
            fontsize=11, color=COL_BT,
            arrowprops=dict(arrowstyle='->', color=COL_BT, lw=1.2))

ax.set_xlabel(r'Fitting window center $\log_{10} t$', fontsize=13)
ax.set_ylabel(r'$\hat\alpha$ from spreadability', fontsize=13)
ax.set_title(r'(b)  BT: $\hat\alpha$ converges slowly in $t$',
             fontsize=12, fontweight='bold', loc='left')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.25, ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_finite_size_alpha.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'\nSaved {out}')
