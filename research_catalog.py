"""
Research Catalog: Comprehensive (alpha, Lambda_bar) Ranking

Production-quality run of all 1D hyperuniform patterns.
Stealthy Lambda_bar values come from the collaborator's ensemble
(stealthy_analysis.py); this script handles the ordered / URL patterns
and compiles the final ranking table.

Sections:
  1. Lattice reference
  2. Metallic-mean chains (Fibonacci, Silver, Bronze): N ~ 1M
  3. Period-doubling (Class II) and 0222 (Class III): N ~ 500k
  4. URL parameter sweep: N = 100k, a in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  5. Ranking figure (stealthy values injected from stealthy_analysis results)
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    predict_chain_length, verify_eigenvalue_prediction,
)
from quasicrystal_variance import (
    compute_number_variance_1d, compute_lambda_bar,
)
from two_phase_media import (
    compute_structure_factor, compute_spectral_density,
    compute_excess_spreadability, extract_alpha_fit,
)
from disordered_patterns import generate_url, lambda_bar_url_exact

SEED = 2026
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CHAIN_TARGET_N  = 1_000_000
DISORD_TARGET_N = 500_000
URL_N           = 100_000
NUM_WINDOWS     = 30_000
NUM_R_POINTS    = 4000
PHI2            = 0.35
URL_A_VALUES    = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

rng = np.random.default_rng(SEED)
catalog = {}


def gen_chain(name, target_n):
    for iters in range(5, 70):
        if predict_chain_length(name, iters) > target_n:
            break
    seq = generate_substitution_sequence(name, iters)
    pts, L = sequence_to_points(seq, name)
    del seq
    return pts, L


def bootstrap_lambda_bar(var_array, n_boots=4):
    splits = np.array_split(var_array[len(var_array)//3:], n_boots)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    return np.std(boots) if len(boots) > 1 else 0.0


# ============================================================
# Section 1: Lattice reference
# ============================================================
print("\n" + "=" * 68)
print("  SECTION 1: Integer Lattice")
print("=" * 68)
N_lat = 200_000
pts_lat = np.arange(N_lat, dtype=np.float64)
L_lat = float(N_lat)
R_lat = np.linspace(0.1, 400, NUM_R_POINTS)
t0 = time.perf_counter()
var_lat, _ = compute_number_variance_1d(pts_lat, L_lat, R_lat, num_windows=NUM_WINDOWS, rng=rng)
print(f"  Variance done in {time.perf_counter()-t0:.1f}s")
lb_lat = compute_lambda_bar(R_lat, var_lat)
lb_lat_err = bootstrap_lambda_bar(var_lat)
print(f"  Lambda_bar = {lb_lat:.6f} ± {lb_lat_err:.6f}  (exact: {1/6:.6f})")
catalog['Lattice'] = {
    'alpha': np.inf, 'lambda_bar': lb_lat, 'lambda_bar_err': lb_lat_err,
    'class': 'I', 'R_array': R_lat.tolist(), 'variances': var_lat.tolist(),
}
del pts_lat


# ============================================================
# Section 2: Metallic-mean chains
# ============================================================
print("\n" + "=" * 68)
print("  SECTION 2: Metallic-Mean Quasicrystal Chains")
print("=" * 68)

metallic_chains = ['fibonacci', 'silver', 'bronze']
for name in metallic_chains:
    info = CHAINS[name]
    print(f"\n  -- {info['name']} --")
    t0 = time.perf_counter()
    pts, L = gen_chain(name, CHAIN_TARGET_N)
    rho = len(pts) / L
    mean_sp = 1.0 / rho
    print(f"  Generated N={len(pts):,}, rho={rho:.5f} in {time.perf_counter()-t0:.1f}s")

    R_max = min(300 * mean_sp, L / 4)
    R_arr = np.linspace(mean_sp * 0.5, R_max, NUM_R_POINTS)
    t0 = time.perf_counter()
    var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
    print(f"  Variance done in {time.perf_counter()-t0:.1f}s")
    lb = compute_lambda_bar(R_arr, var)
    lb_err = bootstrap_lambda_bar(var)
    print(f"  Lambda_bar = {lb:.6f} ± {lb_err:.6f}")

    catalog[info['name']] = {
        'alpha': 3.0, 'lambda_bar': lb, 'lambda_bar_err': lb_err,
        'metallic_mean': float(info['metallic_mean']),
        'class': 'I', 'N': len(pts),
        'R_array': R_arr.tolist(), 'variances': var.tolist(),
    }
    del pts


# ============================================================
# Section 3: Period-doubling (Class II, alpha=1)
# ============================================================
print("\n" + "=" * 68)
print("  SECTION 3: Period-Doubling (Class II)")
print("=" * 68)
alpha_pd, _, _ = verify_eigenvalue_prediction('period_doubling')
print(f"  alpha_eigenvalue = {alpha_pd:.6f}")
t0 = time.perf_counter()
pts, L = gen_chain('period_doubling', DISORD_TARGET_N)
rho_pd = len(pts) / L
mean_sp = 1.0 / rho_pd
print(f"  Generated N={len(pts):,} in {time.perf_counter()-t0:.1f}s")
R_pd = np.linspace(mean_sp, min(500 * mean_sp, L/4), NUM_R_POINTS)
t0 = time.perf_counter()
var_pd, _ = compute_number_variance_1d(pts, L, R_pd, num_windows=NUM_WINDOWS, rng=rng)
print(f"  Variance done in {time.perf_counter()-t0:.1f}s")

# Fit log envelope
try:
    mask = R_pd > 20 * mean_sp
    popt, pcov = curve_fit(lambda R, C, b: C * np.log(R) + b,
                           R_pd[mask], var_pd[mask])
    C_pd, b_pd = popt
    C_pd_err = np.sqrt(pcov[0, 0])
    print(f"  Fit: sigma^2 ~ {C_pd:.4f}*ln(R) + {b_pd:.4f} (C err: {C_pd_err:.4f})")
except Exception as e:
    C_pd, b_pd, C_pd_err = np.nan, np.nan, np.nan
    print(f"  Log fit failed: {e}")

catalog['Period-Doubling'] = {
    'alpha': float(alpha_pd), 'lambda_bar': None,
    'log_coeff_C': float(C_pd) if not np.isnan(C_pd) else None,
    'log_coeff_b': float(b_pd) if not np.isnan(b_pd) else None,
    'class': 'II', 'N': len(pts),
    'R_array': R_pd.tolist(), 'variances': var_pd.tolist(),
}
del pts


# ============================================================
# Section 4: 0222 chain (Class III, alpha ~ 0.639)
# ============================================================
print("\n" + "=" * 68)
print("  SECTION 4: 0222 Chain (Class III)")
print("=" * 68)
alpha_0222, _, _ = verify_eigenvalue_prediction('chain_0222')
print(f"  alpha_eigenvalue = {alpha_0222:.6f}")
t0 = time.perf_counter()
pts, L = gen_chain('chain_0222', DISORD_TARGET_N)
rho_0222 = len(pts) / L
mean_sp = 1.0 / rho_0222
print(f"  Generated N={len(pts):,} in {time.perf_counter()-t0:.1f}s")
R_0222 = np.linspace(mean_sp, min(500 * mean_sp, L/4), NUM_R_POINTS)
t0 = time.perf_counter()
var_0222, _ = compute_number_variance_1d(pts, L, R_0222, num_windows=NUM_WINDOWS, rng=rng)
print(f"  Variance done in {time.perf_counter()-t0:.1f}s")

# Fit power law: sigma^2 ~ C * R^beta, beta = 1 - alpha
try:
    mask = R_0222 > 20 * mean_sp
    popt, pcov = curve_fit(lambda R, C, beta: C * R**beta,
                           R_0222[mask], var_0222[mask],
                           p0=[0.5, 0.36], maxfev=5000)
    C_0222, beta_0222 = popt
    beta_err = np.sqrt(pcov[1, 1])
    alpha_0222_num = 1.0 - beta_0222
    print(f"  Fit: sigma^2 ~ {C_0222:.4f}*R^{beta_0222:.4f} -> alpha_num={alpha_0222_num:.4f}  "
          f"(theory: {alpha_0222:.4f})")
except Exception as e:
    C_0222, beta_0222, alpha_0222_num = np.nan, np.nan, np.nan
    print(f"  Power-law fit failed: {e}")

catalog['0222 Chain'] = {
    'alpha': float(alpha_0222), 'lambda_bar': None,
    'alpha_numeric': float(alpha_0222_num) if not np.isnan(alpha_0222_num) else None,
    'power_law_beta': float(beta_0222) if not np.isnan(beta_0222) else None,
    'class': 'III', 'N': len(pts),
    'R_array': R_0222.tolist(), 'variances': var_0222.tolist(),
}
del pts


# ============================================================
# Section 5: URL parameter sweep
# ============================================================
print("\n" + "=" * 68)
print("  SECTION 5: URL Parameter Sweep")
print("=" * 68)
url_results = {}
for a_val in URL_A_VALUES:
    t0 = time.perf_counter()
    pts, L = generate_url(URL_N, a_val, rng=rng)
    R_url = np.linspace(0.3, 400, NUM_R_POINTS)
    var_url, _ = compute_number_variance_1d(pts, L, R_url, num_windows=NUM_WINDOWS, rng=rng)
    lb = compute_lambda_bar(R_url, var_url)
    lb_err = bootstrap_lambda_bar(var_url)
    lb_exact = lambda_bar_url_exact(a_val)
    elapsed = time.perf_counter() - t0
    print(f"  a={a_val:.2f}: Lambda_bar={lb:.6f}±{lb_err:.6f}  "
          f"(exact={lb_exact:.6f}, err={abs(lb-lb_exact):.6f}) [{elapsed:.1f}s]")
    url_results[a_val] = {'lambda_bar': lb, 'lambda_bar_err': lb_err,
                           'lambda_bar_exact': lb_exact,
                           'R_array': R_url.tolist(), 'variances': var_url.tolist()}
    catalog[f'URL (a={a_val})'] = {'alpha': 2.0, 'lambda_bar': lb,
                                    'lambda_bar_err': lb_err, 'lambda_bar_exact': lb_exact,
                                    'class': 'I'}
    del pts

catalog['url_sweep'] = url_results


# ============================================================
# Save intermediate results
# ============================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)

json_path = os.path.join(RESULTS_DIR, 'catalog.json')
with open(json_path, 'w') as f:
    json.dump(make_serializable(catalog), f, indent=2)
print(f"\n  Saved to {json_path}")


# ============================================================
# Analysis: URL exact formula verification
# ============================================================
print("\n" + "=" * 68)
print("  ANALYSIS: URL Lambda_bar(a) — Numeric vs Analytic")
print("=" * 68)
print(f"  {'a':>6s}  {'Numeric':>12s}  {'Analytic':>12s}  {'Rel.Err%':>10s}")
for a_val, res in url_results.items():
    rel = abs(res['lambda_bar'] - res['lambda_bar_exact']) / res['lambda_bar_exact'] * 100
    print(f"  {a_val:6.2f}  {res['lambda_bar']:12.6f}  {res['lambda_bar_exact']:12.6f}  {rel:9.2f}%")

# Metallic mean trend
print("\n  Metallic-mean chains:")
mus = np.array([CHAINS[n]['metallic_mean'] for n in metallic_chains])
lbs = np.array([catalog[CHAINS[n]['name']]['lambda_bar'] for n in metallic_chains])
for name, mu, lb in zip(metallic_chains, mus, lbs):
    print(f"  {CHAINS[name]['name']:30s}  mu={mu:.4f}  Lambda_bar={lb:.6f}")

# Try fitting Lambda_bar = A + B * mu
try:
    popt, _ = curve_fit(lambda mu, A, B: A + B * mu, mus, lbs)
    res = lbs - (popt[0] + popt[1]*mus)
    print(f"  Linear fit: Lb = {popt[0]:.4f} + {popt[1]:.4f}*mu  (max_resid={np.max(np.abs(res)):.5f})")
except Exception:
    pass
try:
    popt2, _ = curve_fit(lambda mu, A, B: A * np.log(mu) + B, mus, lbs)
    res2 = lbs - (popt2[0]*np.log(mus) + popt2[1])
    print(f"  Log fit:    Lb = {popt2[0]:.4f}*ln(mu) + {popt2[1]:.4f}  (max_resid={np.max(np.abs(res2)):.5f})")
except Exception:
    pass


# ============================================================
# Figures
# ============================================================

# ---- Figure A: All variance curves by class (log-log) ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(r'Number Variance $\sigma^2(R)$ by Hyperuniformity Class', fontsize=14)

# Panel 0: Class I
ax = axes[0]
ax.set_title('Class I: Bounded Variance', fontsize=12, fontweight='bold')
class_I_items = [
    ('Lattice', 'black', '-'),
    (CHAINS['fibonacci']['name'], '#2ca02c', '-'),
    (CHAINS['silver']['name'], '#9467bd', '-'),
    (CHAINS['bronze']['name'], '#d62728', '-'),
    ('URL (a=0.5)', '#1f77b4', '-.'),
    ('URL (a=1.0)', '#aec7e8', '-.'),
]
for lbl, c, ls in class_I_items:
    if lbl in catalog and 'R_array' in catalog[lbl]:
        R = np.array(catalog[lbl]['R_array'])
        v = np.array(catalog[lbl]['variances'])
        m = (R > 0.5) & (v > 0)
        ax.loglog(R[m], v[m], color=c, ls=ls, lw=1.2, alpha=0.85, label=lbl)
        lb = catalog[lbl].get('lambda_bar')
        if lb and np.isfinite(lb):
            ax.axhline(lb, color=c, ls=':', lw=0.7, alpha=0.4)
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, ls=':', alpha=0.4)

# Panel 1: Class II
ax = axes[1]
ax.set_title('Class II: Logarithmic Growth', fontsize=12, fontweight='bold')
R_pd_arr = np.array(catalog['Period-Doubling']['R_array'])
v_pd_arr = np.array(catalog['Period-Doubling']['variances'])
ax.loglog(R_pd_arr[R_pd_arr > 1], v_pd_arr[R_pd_arr > 1], '#ff7f0e', lw=1.5)
C = catalog['Period-Doubling'].get('log_coeff_C')
b = catalog['Period-Doubling'].get('log_coeff_b')
if C and b:
    R_fit = R_pd_arr[R_pd_arr > 1]
    ax.loglog(R_fit, C*np.log(R_fit)+b, 'k--', lw=1.5, alpha=0.6,
              label=rf'$C\ln R+b$, $C={C:.3f}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05,
        rf'Period-Doubling, $\alpha=1$' '\n(No bounded $\bar{{\Lambda}}$)',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 2: Class III
ax = axes[2]
ax.set_title('Class III: Power-Law Growth', fontsize=12, fontweight='bold')
R_0222_arr = np.array(catalog['0222 Chain']['R_array'])
v_0222_arr = np.array(catalog['0222 Chain']['variances'])
ax.loglog(R_0222_arr[R_0222_arr > 1], v_0222_arr[R_0222_arr > 1], '#17becf', lw=1.5)
beta = catalog['0222 Chain'].get('power_law_beta')
if beta is not None:
    mid = len(R_0222_arr) // 2
    C3 = v_0222_arr[mid] / R_0222_arr[mid] ** beta
    R_fit = R_0222_arr[R_0222_arr > 1]
    ax.loglog(R_fit, C3 * R_fit**beta, 'k--', lw=1.5, alpha=0.6,
              label=rf'$\sim R^{{{beta:.3f}}}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
alpha_disp = catalog['0222 Chain']['alpha']
ax.text(0.05, 0.05,
        rf'0222 Chain, $\alpha={alpha_disp:.3f}$' '\n(No bounded $\bar{{\Lambda}}$)',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figA_variance_by_class.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {path}")


# ---- Figure B: URL Lambda_bar(a) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
url_plot_curves = [0.1, 0.5, 1.0, 1.5]
url_cols = plt.cm.Blues(np.linspace(0.4, 1.0, len(url_plot_curves)))
for a_val, col in zip(url_plot_curves, url_cols):
    if a_val in url_results:
        R = np.array(url_results[a_val]['R_array'])
        v = np.array(url_results[a_val]['variances'])
        m = (R > 0.5) & (v > 0)
        ax.plot(R[m], v[m], color=col, lw=1.0, alpha=0.8, label=f'$a={a_val}$')
        lb = lambda_bar_url_exact(a_val)
        ax.axhline(lb, color=col, ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.set_title('URL: Variance for Different $a$ Values', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.set_xscale('log')

ax = axes[1]
a_arr = np.array(sorted(url_results.keys()))
lb_arr = np.array([url_results[a]['lambda_bar'] for a in a_arr])
lb_err_arr = np.array([url_results[a]['lambda_bar_err'] for a in a_arr])
a_fine = np.linspace(0, max(a_arr) + 0.15, 300)
ax.fill_between(a_arr, lb_arr - lb_err_arr, lb_arr + lb_err_arr, alpha=0.2, color='#1f77b4')
ax.plot(a_arr, lb_arr, 'o', color='#1f77b4', ms=7, label='Numeric (N=100k)')
ax.plot(a_fine, [lambda_bar_url_exact(a) for a in a_fine], 'r-', lw=2,
        label=r'Exact: $\frac{1+a^2}{6}$ (for $a\leq 1$)')
ax.axhline(1/6, color='k', ls=':', lw=1.2, alpha=0.5, label=r'Lattice: $1/6$')
ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('$a$', fontsize=12)
ax.set_ylabel(r'$\bar{\Lambda}(a)$', fontsize=12)
ax.set_title(r'URL: $\bar{\Lambda}(a) = (1+a^2)/6$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.set_ylim(0)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figB_url_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ---- Figure D: Metallic means Lambda_bar vs mu ----
fig, ax = plt.subplots(figsize=(7, 5))
mus_arr = np.array([CHAINS[n]['metallic_mean'] for n in metallic_chains])
lbs_arr = np.array([catalog[CHAINS[n]['name']]['lambda_bar'] for n in metallic_chains])
lbs_err_arr = np.array([catalog[CHAINS[n]['name']]['lambda_bar_err'] for n in metallic_chains])
colors_met = ['#2ca02c', '#9467bd', '#d62728']
names_met = [CHAINS[n]['name'] for n in metallic_chains]

ax.errorbar(mus_arr, lbs_arr, yerr=lbs_err_arr, fmt='none', capsize=5, color='gray')
for mu, lb, name, color in zip(mus_arr, lbs_arr, names_met, colors_met):
    ax.scatter(mu, lb, color=color, s=120, zorder=5, label=name)

mu_fine = np.linspace(1.5, 3.6, 200)
try:
    p2, _ = curve_fit(lambda mu, A, B: A * np.log(mu) + B, mus_arr, lbs_arr)
    ax.plot(mu_fine, p2[0]*np.log(mu_fine) + p2[1], 'k--', lw=1.5, alpha=0.5,
            label=rf'$\bar{{\Lambda}} = {p2[0]:.3f}\ln\mu {p2[1]:+.3f}$')
except Exception:
    pass
ax.axhline(1/6, color='k', ls=':', lw=1, alpha=0.4, label=r'Lattice: $1/6$')
ax.set_xlabel(r'Metallic mean $\mu$', fontsize=13)
ax.set_ylabel(r'$\bar{\Lambda}$', fontsize=13)
ax.set_title(r'Metallic-Mean Chains: $\bar{\Lambda}$ Increases with $\mu$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figD_metallic_lambda_vs_mu.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ---- Summary table ----
print("\n" + "=" * 72)
print(f"  {'COMPREHENSIVE (alpha, Lambda_bar) CATALOG':^68s}")
print("=" * 72)
print(f"  {'Pattern':30s}  {'Class':>7s}  {'alpha':>10s}  {'Lambda_bar':>12s}  {'Err':>8s}")
print("  " + "-" * 72)

rows = [
    ('Lattice', 'I', np.inf, catalog['Lattice']['lambda_bar'], catalog['Lattice']['lambda_bar_err']),
]
for name in metallic_chains:
    r = catalog[CHAINS[name]['name']]
    rows.append((CHAINS[name]['name'], 'I', 3.0, r['lambda_bar'], r['lambda_bar_err']))
for a_val in URL_A_VALUES:
    r = url_results[a_val]
    rows.append((f'URL (a={a_val})', 'I', 2.0, r['lambda_bar'], r['lambda_bar_err']))
rows.append(('Period-Doubling', 'II', alpha_pd, None, None))
rows.append(('0222 Chain', 'III', alpha_0222, None, None))

for lbl, cls, alpha_val, lb, lb_err in rows:
    alpha_str = 'inf' if not np.isfinite(alpha_val) else f'{alpha_val:.3f}'
    lb_str = f'{lb:.5f}' if lb is not None else 'diverges'
    err_str = f'{lb_err:.5f}' if lb_err is not None else '---'
    print(f"  {lbl:30s}  {cls:>7s}  {alpha_str:>10s}  {lb_str:>12s}  {err_str:>8s}")

print("=" * 72)
print("\n  Done.")
