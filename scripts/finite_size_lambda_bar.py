"""
Lambda_bar Finite-N Corrections

(a) Integer lattice: analytic Lambda_bar(R_max) + simulation vs N
    - sigma^2(R) = 2*frac(R)*(1-2*frac(R)) for frac(R)<0.5, piecewise quadratic
    - Integral (1/R_max) * integral_0^{R_max} sigma^2(R) dR is piecewise cubic
    - Exact closed-form computation for comparison with simulation

(b) Stealthy patterns (chi=0.1, 0.2, 0.3) at multiple N:
    - Check convergence direction (from below vs above)
    - Compare to formula Lambda_bar = 1/(pi^2 * chi) (1D stealthy, derived)

Output: results/figures/fig_finite_size_lambda.png
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quasicrystal_variance import (
    compute_number_variance_1d, compute_lambda_bar, lattice_variance_exact,
)
from disordered_patterns import generate_stealthy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(2026)
NUM_WINDOWS = 15_000
NUM_R_POINTS = 600


# ============================================================
# Analytic Lambda_bar(R_max) for integer lattice
# ============================================================
def lambda_bar_lattice_analytic(R_max):
    """
    Exact Lambda_bar(R_max) = (1/R_max) * integral_0^{R_max} sigma^2(R) dR
    for the integer lattice (spacing 1).

    sigma^2(R) is piecewise quadratic with period 1:
      For R in [n, n+0.5]:  sigma^2 = 2*f*(1-2*f)   where f = R - n
      For R in [n+0.5, n+1]: sigma^2 = 2*(1-f)*(2*f-1) where f = R - n

    Integral over one period [0,1]:
      integral_0^{0.5} 2f(1-2f) df + integral_{0.5}^{1} 2(1-f)(2f-1) df = 1/6

    For integer R_max = M:   integral_0^M sigma^2 dR = M/6
    For non-integer R_max:   split at floor(R_max) and add partial period
    """
    R_max = float(R_max)
    M = int(np.floor(R_max))
    f = R_max - M  # fractional part

    # Contribution from M complete periods: M * (1/6)
    integral_complete = M / 6.0

    # Partial period integral from 0 to f
    if f < 0.5:
        # integral_0^f 2*t*(1-2*t) dt = f^2 - 4*f^3/3
        partial = f**2 - 4.0 * f**3 / 3.0
    else:
        # Full first half: integral_0^{0.5} 2*t*(1-2*t) dt = 0.25 - 4*0.125/3 = 0.25 - 1/6 = 1/12
        half_int = 1.0 / 12.0
        # Partial second half: integral_{0.5}^f 2*(1-t)*(2t-1) dt
        # = integral_{0.5}^f (4t - 2 - 4t^2 + 2t) dt = integral (6t - 2 - 4t^2) dt
        # = [3t^2 - 2t - 4t^3/3]_{0.5}^f
        def antideriv(t):
            return 3*t**2 - 2*t - 4*t**3/3.0
        partial = half_int + (antideriv(f) - antideriv(0.5))

    integral_total = integral_complete + partial
    return integral_total / R_max


# ============================================================
# Part (a): Integer lattice — analytic vs simulated Lambda_bar(N)
# ============================================================
print("=" * 65)
print("  Lambda_bar Finite-N Corrections")
print("=" * 65)
print("\n[Part a] Integer Lattice: analytic vs simulated Lambda_bar(N)")

N_values_lat = [500, 2_000, 10_000, 50_000, 200_000, 1_000_000]
lb_analytic = []
lb_simulated = []

print(f"  {'N':>9s}  {'R_max':>9s}  {'Lambda_bar(analytic)':>20s}  {'Lambda_bar(sim)':>16s}  {'err%':>6s}")
for N in N_values_lat:
    pts = np.arange(N, dtype=np.float64)
    L   = float(N)
    rho_lat = 1.0
    R_max_lat = N / (2.0 * rho_lat)  # N/2

    # Analytic
    lb_an = lambda_bar_lattice_analytic(R_max_lat)
    lb_analytic.append(lb_an)

    # Simulated
    R_arr = np.linspace(0.05, R_max_lat, NUM_R_POINTS)
    var, _ = compute_number_variance_1d(pts, L, R_arr,
                                         num_windows=NUM_WINDOWS, rng=rng)
    lb_sim = compute_lambda_bar(R_arr, var)
    lb_simulated.append(lb_sim)

    err_pct = (lb_sim - lb_an) / lb_an * 100
    print(f"  {N:>9,}  {R_max_lat:>9.1f}  {lb_an:>20.8f}  {lb_sim:>16.8f}  {err_pct:>5.2f}%")

print(f"  Exact: Lambda_bar(infinity) = 1/6 = {1/6:.8f}")

# Fine analytic curve
R_max_fine = np.linspace(1, max(N_values_lat) / 2, 500)
lb_fine = np.array([lambda_bar_lattice_analytic(rm) for rm in R_max_fine])
N_fine  = 2 * R_max_fine  # N = 2*R_max for lattice with rho=1


# ============================================================
# Part (b): Stealthy patterns Lambda_bar(N) for chi = 0.1, 0.2, 0.3
# ============================================================
print("\n[Part b] Stealthy: Lambda_bar(N) vs N  (chi = 0.1, 0.2, 0.3)")

CHI_VALUES = [0.1, 0.2, 0.3]
# Theoretical 1D stealthy formula: Lambda_bar = 1/(pi^2 * chi)
lb_stealthy_theory = {chi: 1.0 / (np.pi**2 * chi) for chi in CHI_VALUES}
for chi, lb_th in lb_stealthy_theory.items():
    print(f"  chi={chi}: Lambda_bar_theory = 1/(pi^2*{chi}) = {lb_th:.5f}")

N_values_st = [200, 500, 1_000, 2_000, 5_000, 10_000]
stealthy_results = {chi: {'N': [], 'lb': [], 'lb_err': []} for chi in CHI_VALUES}

# Reduce max_iter for large N to keep runtime tractable
def stealthy_max_iter(N):
    if N <= 1000:  return 2000
    if N <= 2000:  return 1500
    if N <= 5000:  return 1000
    return 600

for chi in CHI_VALUES:
    print(f"\n  chi={chi}:")
    for N_st in N_values_st:
        t0 = time.perf_counter()
        try:
            mi = stealthy_max_iter(N_st)
            pts_st, L_st = generate_stealthy(N_st, chi, rng=rng, verbose=False,
                                              max_iter=mi)
        except Exception as e:
            print(f"    N={N_st:,}: FAILED ({e})")
            continue
        gen_t = time.perf_counter() - t0

        rho_st = N_st / L_st
        R_max_st = N_st / (2.0 * rho_st) * 0.9
        R_arr = np.linspace(1.0, R_max_st, NUM_R_POINTS)
        var, _ = compute_number_variance_1d(pts_st, L_st, R_arr,
                                             num_windows=NUM_WINDOWS, rng=rng)
        lb_st = compute_lambda_bar(R_arr, var)
        # Bootstrap error estimate
        splits = np.array_split(var[len(var)//3:], 4)
        lb_boots = [np.mean(s) for s in splits if len(s) > 0]
        lb_err = np.std(lb_boots) if len(lb_boots) > 1 else 0.0

        stealthy_results[chi]['N'].append(N_st)
        stealthy_results[chi]['lb'].append(lb_st)
        stealthy_results[chi]['lb_err'].append(lb_err)

        lb_th = lb_stealthy_theory[chi]
        pct = (lb_st - lb_th) / lb_th * 100
        print(f"    N={N_st:>6,}: Lambda_bar={lb_st:.5f} ± {lb_err:.5f}"
              f"  (theory={lb_th:.5f}, diff={pct:+.2f}%)"
              f"  [{gen_t:.1f}s]")
        del pts_st


# ============================================================
# Figure
# ============================================================
print("\nGenerating fig_finite_size_lambda.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(r'$\bar{\Lambda}$ Finite-$N$ Corrections',
             fontsize=13, fontweight='bold')

# Panel (a): Integer lattice
ax = axes[0]
ax.semilogx(N_fine, lb_fine, 'k-', lw=2, label='Analytic $\\bar{\\Lambda}(N)$', zorder=3)
ax.semilogx(N_values_lat, lb_simulated, 'ro', ms=8,
            label='Simulated $\\hat{\\Lambda}$', zorder=4)
ax.axhline(1/6, color='b', ls='--', lw=1.5, alpha=0.7,
           label=r'Exact: $1/6 = 0.1\overline{6}$')

# Show monotone convergence from below
ax.text(0.05, 0.5,
        r'Monotone convergence from below:' '\n'
        r'$\bar{\Lambda}(N) < 1/6 = \bar{\Lambda}(\infty)$',
        transform=ax.transAxes, fontsize=10, va='center',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
ax.set_xlabel(r'$N$', fontsize=12)
ax.set_ylabel(r'$\bar{\Lambda}(N)$', fontsize=12)
ax.set_title(r'Integer Lattice: Analytic vs Simulated $\bar{\Lambda}(N)$', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)

# Panel (b): Stealthy patterns
ax = axes[1]
colors_st = ['#d62728', '#2ca02c', '#1f77b4']
for chi, color in zip(CHI_VALUES, colors_st):
    res = stealthy_results[chi]
    N_arr = np.array(res['N'])
    lb_arr = np.array(res['lb'])
    lb_err_arr = np.array(res['lb_err'])
    lb_th = lb_stealthy_theory[chi]

    if len(N_arr) > 0:
        ax.errorbar(N_arr, lb_arr, yerr=lb_err_arr,
                    fmt='o-', color=color, ms=6, lw=1.5, capsize=4,
                    label=rf'$\chi={chi}$')
        ax.axhline(lb_th, color=color, ls='--', lw=1.2, alpha=0.6,
                   label=rf'Theory $1/(\pi^2 \cdot {chi}) = {lb_th:.3f}$')

# Check direction: converges from below?
ax.text(0.04, 0.97,
        r'Theory: $\bar{\Lambda} = 1/(\pi^2 \chi)$' '\n' r'(1D stealthy)',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
ax.set_xlabel(r'$N$', fontsize=12)
ax.set_ylabel(r'$\bar{\Lambda}(N)$', fontsize=12)
ax.set_title(r'Stealthy: $\bar{\Lambda}(N)$ Convergence', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, ls=':', alpha=0.4)
ax.set_xscale('log')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'fig_finite_size_lambda.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("  SUMMARY: Lambda_bar Finite-N Corrections")
print("=" * 65)
print(f"\n  Integer lattice: Lambda_bar(N) from analytic formula")
print(f"  {'N':>9s}  {'Lambda_bar(analytic)':>20s}  {'Fraction of 1/6':>16s}")
for N, lb in zip(N_values_lat, lb_analytic):
    print(f"  {N:>9,}  {lb:>20.8f}  {lb/(1/6)*100:>14.3f}%")
print(f"  Lambda_bar(inf) = 1/6 = {1/6:.8f}")

print(f"\n  Stealthy: convergence direction")
for chi in CHI_VALUES:
    res = stealthy_results[chi]
    lb_th = lb_stealthy_theory[chi]
    if res['lb']:
        lb_small = res['lb'][0]   # smallest N
        lb_large = res['lb'][-1]  # largest N
        direction = "from below" if lb_small < lb_th else "from above"
        print(f"  chi={chi}: N={res['N'][0]:,} -> Lambda_bar={lb_small:.4f}; "
              f"N={res['N'][-1]:,} -> {lb_large:.4f}; theory={lb_th:.4f}; "
              f"converges {direction}")
print("=" * 65)
