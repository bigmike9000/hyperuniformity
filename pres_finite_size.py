"""
Finite-size convergence figure for Week 3 presentation.

Panel (a): Lbar(N) for BT and Fibonacci — flat from N~1000 (success)
Panel (b): alpha_hat(N) for 0222 chain — oscillates wildly (failure)

Demonstrates: use Lbar (not S(k) fitting) for Class I; Class III
convergence is inherently slow because |λ₂|>1.

Output: results/figures/fig_finite_size_alpha.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    sequence_to_points_general, verify_eigenvalue_prediction, predict_chain_length,
)
from quasicrystal_variance import compute_number_variance_1d

RESULTS_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# Catalog reference values
LB_BT_REF  = 0.377
LB_FIB_REF = 0.201
ALPHA_0222_REF = 0.6391

# Target N values (log-spaced from ~300 to ~1M)
N_TARGETS = [300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]


def compute_lambda_bar_at_N(chain_name, target_n, num_windows=None, num_R=300):
    """Generate chain at given N and compute Lbar."""
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

    # Running Lbar via trapezoidal rule
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


def compute_alpha_hat_Sk(chain_name, target_n):
    """Generate chain at given N and estimate alpha from S(k) log-log slope.
    This is the method that FAILS for Bragg-peak-dominated tilings."""
    from two_phase_media import compute_structure_factor

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
    N = len(pts)

    k_arr, S_arr = compute_structure_factor(pts, L)
    del pts

    # Fit S(k) ~ k^alpha at small k (n_low lowest k-values)
    n_low = min(30, len(k_arr) // 10)
    valid = (k_arr[:n_low] > 0) & (S_arr[:n_low] > 0)
    if np.sum(valid) >= 4:
        log_k = np.log(k_arr[:n_low][valid])
        log_S = np.log(S_arr[:n_low][valid])
        coeffs = np.polyfit(log_k, log_S, 1)
        alpha_hat = float(coeffs[0])
    else:
        alpha_hat = np.nan

    return N, alpha_hat


# ── Compute ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Finite-Size Convergence Analysis")
print("=" * 60)

# Panel (a): Lbar(N) for Class I chains
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

# Panel (b): alpha_hat from S(k) log-log slope for ALL chains
# This method fails because S(k) is Bragg-peak dominated — no smooth k^alpha
ALPHA_REFS = {
    'bombieri_taylor': 1.545,
    'fibonacci': 3.0,
    'chain_0222': 0.6391,
}
sk_data = {}
for chain_name, alpha_ref in ALPHA_REFS.items():
    sk_data[chain_name] = []
    short = chain_name.replace('bombieri_taylor', 'BT').replace(
        'fibonacci', 'Fib').replace('chain_0222', '0222')
    for tgt in N_TARGETS:
        print(f"\n  {short} S(k) slope at N~{tgt:,}...", end=' ', flush=True)
        t0 = time.perf_counter()
        N, alpha_hat = compute_alpha_hat_Sk(chain_name, tgt)
        sk_data[chain_name].append((N, alpha_hat))
        print(f"N={N:,}, alpha_hat={alpha_hat:.3f} "
              f"(theory={alpha_ref:.3f}) [{time.perf_counter()-t0:.1f}s]")
    sk_data[chain_name] = np.array(sk_data[chain_name])

# ── Figure ───────────────────────────────────────────────────────────────────
COL_BT  = '#D32F2F'
COL_FIB = '#2E7D32'
COL_0222 = '#1565C0'

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'wspace': 0.30})

# ── (a) Lbar(N): stable from N~10^3 ─────────────────────────────────────────
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

# ── (b) S(k) slope: unreliable for all chains ───────────────────────────────
ax = axes[1]

labels_cols = [
    ('bombieri_taylor', 'BT', COL_BT, 'o', 1.545),
    ('fibonacci', 'Fibonacci', COL_FIB, 's', 3.0),
    ('chain_0222', '0222', COL_0222, 'D', 0.6391),
]
for chain_name, label, col, marker, alpha_ref in labels_cols:
    d = sk_data[chain_name]
    ax.plot(d[:, 0], d[:, 1], f'{marker}-', color=col, lw=1.8, ms=7,
            label=rf'{label} (theory $\alpha={alpha_ref:.1f}$)')
    # Reference line
    ax.axhline(alpha_ref, color=col, ls=':', lw=1.2, alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel(r'$N$ (chain length)', fontsize=13)
ax.set_ylabel(r'$\hat\alpha$ from $S(k)\!\sim\!k^{\alpha}$ fit', fontsize=13)
ax.set_title(r'(b)  $\hat\alpha$ from $S(k)$: unreliable',
             fontsize=12, fontweight='bold', loc='left')
ax.legend(fontsize=9.5, loc='upper right')
ax.set_xlim(150, bt_lb[-1, 0] * 3)
ax.grid(alpha=0.25, ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.05, 0.05, 'Bragg peaks $\\Rightarrow$ no smooth\n$S(k)\\sim k^\\alpha$ background',
        transform=ax.transAxes, fontsize=10, va='bottom',
        style='italic', color='#555555',
        bbox=dict(fc='white', ec='#cccccc', alpha=0.9, pad=3))

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_finite_size_alpha.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'\nSaved {out}')
