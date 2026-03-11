"""
Generate all research figures from saved catalog.json and stealthy_collab_results.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from substitution_tilings import CHAINS, verify_eigenvalue_prediction
from disordered_patterns import lambda_bar_url_exact

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# ---------- Load data ----------
with open(os.path.join(RESULTS_DIR, 'catalog.json')) as f:
    catalog = json.load(f)

with open(os.path.join(RESULTS_DIR, 'stealthy_collab_results.json')) as f:
    stealthy_raw = json.load(f)

url_results = catalog['url_sweep']
# stealthy_raw is keyed by chi as string
stealthy_results = {float(k): v for k, v in stealthy_raw.items()}

alpha_0222 = verify_eigenvalue_prediction('chain_0222')[0]

METALLIC = ['fibonacci', 'silver', 'bronze']
MET_COLORS = ['#2ca02c', '#9467bd', '#d62728']


# ============================================================
# Figure A: Variance by class
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(r'Number Variance $\sigma^2(R)$ by Hyperuniformity Class', fontsize=14)

ax = axes[0]
ax.set_title('Class I: Bounded', fontsize=12, fontweight='bold')
class_I_items = [
    ('Lattice', 'black', '-'),
    (CHAINS['fibonacci']['name'], '#2ca02c', '-'),
    (CHAINS['silver']['name'], '#9467bd', '-'),
    (CHAINS['bronze']['name'], '#d62728', '-'),
    ('URL (a=0.5)', '#1f77b4', '-.'),
    ('URL (a=1.0)', '#aec7e8', '-.'),
]
for lbl, c, ls in class_I_items:
    key = lbl
    if lbl == 'URL (a=0.5)':
        key = 'url_sweep'
        R = np.array(url_results['0.5']['R_array'])
        v = np.array(url_results['0.5']['variances'])
    elif lbl == 'URL (a=1.0)':
        R = np.array(url_results['1.0']['R_array'])
        v = np.array(url_results['1.0']['variances'])
    elif lbl in catalog and 'R_array' in catalog[lbl]:
        R = np.array(catalog[lbl]['R_array'])
        v = np.array(catalog[lbl]['variances'])
    else:
        continue
    m = (R > 0.5) & (v > 0)
    ax.loglog(R[m], v[m], color=c, ls=ls, lw=1.2, alpha=0.85, label=lbl)
    lb_val = (catalog[lbl].get('lambda_bar') if lbl in catalog else
              (url_results['0.5']['lambda_bar'] if '0.5' in lbl else
               url_results['1.0']['lambda_bar']))
    if lb_val and np.isfinite(lb_val):
        ax.axhline(lb_val, color=c, ls=':', lw=0.7, alpha=0.4)
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, ls=':', alpha=0.4)

ax = axes[1]
ax.set_title('Class II: Logarithmic Growth', fontsize=12, fontweight='bold')
R_pd = np.array(catalog['Period-Doubling']['R_array'])
v_pd = np.array(catalog['Period-Doubling']['variances'])
ax.loglog(R_pd[R_pd > 1], v_pd[R_pd > 1], color='#ff7f0e', lw=1.5, label='Period-Doubling')
C = catalog['Period-Doubling'].get('log_coeff_C')
b = catalog['Period-Doubling'].get('log_coeff_b')
if C and b:
    R_fit = R_pd[R_pd > 1]
    ax.loglog(R_fit, C*np.log(R_fit)+b, 'k--', lw=1.5, alpha=0.6,
              label=rf'$C\ln R+b$, $C={C:.3f}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05, r'Period-Doubling, $\alpha=1$' '\n(No bounded $\\bar{\\Lambda}$)',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax = axes[2]
ax.set_title('Class III: Power-Law Growth', fontsize=12, fontweight='bold')
R_0222 = np.array(catalog['0222 Chain']['R_array'])
v_0222 = np.array(catalog['0222 Chain']['variances'])
ax.loglog(R_0222[R_0222 > 1], v_0222[R_0222 > 1], color='#17becf', lw=1.5, label='0222 Chain')
beta = catalog['0222 Chain'].get('power_law_beta')
if beta:
    mid = len(R_0222) // 2
    C3 = v_0222[mid] / R_0222[mid]**beta
    ax.loglog(R_0222[R_0222>1], C3*R_0222[R_0222>1]**beta, 'k--', lw=1.5, alpha=0.6,
              label=rf'$\sim R^{{{beta:.3f}}}$')
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
alpha_disp = catalog['0222 Chain']['alpha']
ax.text(0.05, 0.05, rf'0222 Chain, $\alpha={alpha_disp:.3f}$' '\n(No bounded $\\bar{\\Lambda}$)',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figA_variance_by_class.png')
plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {path}")


# ============================================================
# Figure B: URL Lambda_bar(a)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
url_plot_a = [0.1, 0.5, 1.0, 1.5]
url_cols = plt.cm.Blues(np.linspace(0.4, 1.0, len(url_plot_a)))
for a_val, col in zip(url_plot_a, url_cols):
    key = str(a_val) if str(a_val) in url_results else str(float(a_val))
    if key not in url_results:
        continue
    R = np.array(url_results[key]['R_array'])
    v = np.array(url_results[key]['variances'])
    m = (R > 0.5) & (v > 0)
    ax.plot(R[m], v[m], color=col, lw=1.0, alpha=0.8, label=f'$a={a_val}$')
    ax.axhline(lambda_bar_url_exact(a_val), color=col, ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.set_title('URL: Variance Curves', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.set_xscale('log')

ax = axes[1]
a_keys = sorted([float(k) for k in url_results.keys()])
lb_num = [url_results[str(a)]['lambda_bar'] if str(a) in url_results
          else url_results[f'{a}']['lambda_bar']
          for a in a_keys]
# Safer key access:
def url_key(a):
    if str(a) in url_results:
        return str(a)
    if str(float(a)) in url_results:
        return str(float(a))
    return None

a_valid = [a for a in a_keys if url_key(a) is not None]
lb_num_v = [url_results[url_key(a)]['lambda_bar'] for a in a_valid]
lb_err_v = [url_results[url_key(a)]['lambda_bar_err'] for a in a_valid]
lb_exact_v = [lambda_bar_url_exact(a) for a in a_valid]

a_fine = np.linspace(0, max(a_valid)+0.15, 300)
ax.fill_between(a_valid, np.array(lb_num_v)-np.array(lb_err_v),
                np.array(lb_num_v)+np.array(lb_err_v), alpha=0.2, color='#1f77b4')
ax.plot(a_valid, lb_num_v, 'o', color='#1f77b4', ms=7, zorder=5, label='Numeric (N=100k)')
ax.plot(a_fine, [lambda_bar_url_exact(a) for a in a_fine], 'r-', lw=2,
        label=r'Exact: $(1+a^2)/6$')
ax.axhline(1/6, color='k', ls=':', lw=1.2, alpha=0.5, label=r'Lattice: $1/6$')
ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.4)
ax.set_xlabel('$a$', fontsize=12)
ax.set_ylabel(r'$\bar{\Lambda}(a)$', fontsize=12)
ax.set_title(r'URL: $\bar{\Lambda}(a) = (1+a^2)/6$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.set_ylim(0)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figB_url_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {path}")


# ============================================================
# Figure C: Stealthy Lambda_bar(chi) with theory
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

chi_vals = sorted(stealthy_results.keys())
lb_st = [stealthy_results[c]['lambda_bar'] for c in chi_vals]
lb_st_err = [stealthy_results[c]['lambda_bar_err'] for c in chi_vals]
lb_theory = [1.0/(np.pi**2*c) for c in chi_vals]

ax = axes[0]
chi_fine = np.linspace(0.05, 0.5, 200)
ax.plot(chi_fine, 1.0/(np.pi**2*chi_fine), 'r-', lw=2.5,
        label=r'Theory: $\bar{\Lambda} = 1/(\pi^2\chi)$')
ax.fill_between(chi_vals, np.array(lb_st)-np.array(lb_st_err),
                np.array(lb_st)+np.array(lb_st_err), alpha=0.3, color='#e377c2')
ax.plot(chi_vals, lb_st, 'o', color='#9467bd', ms=9, zorder=5,
        label='Collab. ensemble (N=2000, ~4314 configs)')
ax.axhline(1/6, color='black', ls='--', lw=1.5, alpha=0.6, label=r'Lattice: $1/6$')
ax.set_xlabel(r'Stealthiness $\chi$', fontsize=13)
ax.set_ylabel(r'$\bar{\Lambda}$', fontsize=13)
ax.set_title(r'Stealthy: $\bar{\Lambda}(\chi) \approx 1/(\pi^2\chi)$', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 3.5)

ax = axes[1]  # Ratio: measured/theory
ratios = [lb_st[i]/lb_theory[i] for i in range(len(chi_vals))]
ax.plot(chi_vals, ratios, 'o-', color='#9467bd', ms=8, lw=1.5,
        label=r'Ratio: measured / $[1/(\pi^2\chi)]$')
ax.axhline(1.0, color='red', ls='--', lw=1.5, alpha=0.7, label='Exact agreement')
ax.set_xlabel(r'$\chi$', fontsize=12)
ax.set_ylabel(r'$\bar{\Lambda}_{\rm meas} / \bar{\Lambda}_{\rm theory}$', fontsize=12)
ax.set_title('S(k) Overshoot Correction to Theory', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.4)
ax.set_ylim(0.95, 1.10)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figC_stealthy_lambda_chi.png')
plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {path}")


# ============================================================
# Figure D: Metallic means Lambda_bar vs mu
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
mus = np.array([CHAINS[n]['metallic_mean'] for n in METALLIC])
lbs = np.array([catalog[CHAINS[n]['name']]['lambda_bar'] for n in METALLIC])
errs = np.array([catalog[CHAINS[n]['name']]['lambda_bar_err'] for n in METALLIC])

ax.errorbar(mus, lbs, yerr=errs, fmt='none', capsize=5, color='gray', zorder=3)
for mu, lb, name, color in zip(mus, lbs, [CHAINS[n]['name'] for n in METALLIC], MET_COLORS):
    ax.scatter(mu, lb, color=color, s=120, zorder=5, label=name)

mu_fine = np.linspace(1.5, 3.6, 200)
try:
    p, _ = curve_fit(lambda mu, A, B: A * np.log(mu) + B, mus, lbs)
    ax.plot(mu_fine, p[0]*np.log(mu_fine)+p[1], 'k--', lw=1.5, alpha=0.5,
            label=rf'$\bar{{\Lambda}} = {p[0]:.3f}\ln\mu {p[1]:+.3f}$')
except Exception:
    pass

# Mark possibly-exact fractions
for lb_exact, label_str in [(1/5, r'$1/5$'), (1/4, r'$1/4$')]:
    ax.axhline(lb_exact, color='lightgray', ls=':', lw=0.8)
    ax.text(1.52, lb_exact+0.002, label_str, fontsize=8, color='gray')

ax.axhline(1/6, color='k', ls=':', lw=1, alpha=0.4, label=r'Lattice: $1/6$')
ax.set_xlabel(r'Metallic mean $\mu$', fontsize=13)
ax.set_ylabel(r'$\bar{\Lambda}$', fontsize=13)
ax.set_title(r'Metallic-Mean Chains: $\bar{\Lambda}$ vs $\mu$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.set_xlim(1.5, 3.7); ax.set_ylim(0.16, 0.33)
plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figD_metallic_lambda_vs_mu.png')
plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {path}")


# ============================================================
# Figure E: Comprehensive (alpha, Lambda_bar) ranking
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7.5))

# --- All Class I data points ---
all_classI = []

# Lattice
all_classI.append(('Lattice', np.inf, catalog['Lattice']['lambda_bar'],
                   catalog['Lattice']['lambda_bar_err'], 'black', '*'))

# Metallic means
for name, color in zip(METALLIC, MET_COLORS):
    r = catalog[CHAINS[name]['name']]
    all_classI.append((CHAINS[name]['name'], 3.0, r['lambda_bar'],
                       r['lambda_bar_err'], color, 'o'))

# URL
url_colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(a_valid)))
for a_val, col in zip(a_valid, url_colors):
    key = url_key(a_val)
    if key is None:
        continue
    res = url_results[key]
    lb_e = lambda_bar_url_exact(a_val)  # use exact value
    all_classI.append((f'URL ($a={a_val}$)', 2.0, lb_e, 0.003, col, 's'))

# Stealthy (use measured values)
st_colors = plt.cm.RdPu(np.linspace(0.4, 1.0, len(chi_vals)))
for chi_val, col in zip(chi_vals, st_colors):
    res = stealthy_results[chi_val]
    all_classI.append((rf'Stealthy ($\chi={chi_val}$)', 2.0, res['lambda_bar'],
                       res['lambda_bar_err'], col, 'D'))

alpha_clip = 4.8
for lbl, alpha_val, lb, lb_err, color, marker in all_classI:
    x = min(alpha_val, alpha_clip) if np.isfinite(alpha_val) else alpha_clip
    ms = 200 if marker == '*' else 80
    ax.scatter(x, lb, color=color, marker=marker, s=ms, zorder=5)
    ax.errorbar(x, lb, yerr=lb_err, fmt='none', capsize=3, color=color, alpha=0.6, zorder=4)
    ax.annotate(lbl, (x, lb), xytext=(5, 3), textcoords='offset points',
                fontsize=7.5, ha='left', va='bottom')

# Lattice arrow
ax.annotate('', xy=(alpha_clip+0.2, catalog['Lattice']['lambda_bar']),
            xytext=(alpha_clip, catalog['Lattice']['lambda_bar']),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Class II & III vertical lines
ax.axvline(1.0, color='#ff7f0e', ls='--', lw=2, alpha=0.7, zorder=2,
           label=r'Period-Doubling ($\alpha=1$, Class II)')
ax.text(1.02, 0.05, 'Period-\nDoubling', fontsize=8, color='#ff7f0e', va='bottom',
        transform=ax.get_xaxis_transform())

ax.axvline(alpha_0222, color='#17becf', ls='-.', lw=2, alpha=0.7, zorder=2,
           label=rf'0222 Chain ($\alpha={alpha_0222:.3f}$, Class III)')
ax.text(alpha_0222+0.02, 0.05, '0222\nChain', fontsize=8, color='#17becf', va='bottom',
        transform=ax.get_xaxis_transform())

# Reference dashed lines at alpha=2 and alpha=3
ax.axvline(2.0, color='gray', ls=':', lw=0.8, alpha=0.4, zorder=1)
ax.axvline(3.0, color='gray', ls=':', lw=0.8, alpha=0.4, zorder=1)
ax.text(2.02, 1.5, r'$\alpha=2$', fontsize=9, color='gray')
ax.text(3.02, 1.5, r'$\alpha=3$', fontsize=9, color='gray')

# Stealthy theory curve
chi_th = np.linspace(0.05, 0.5, 200)
ax.plot([2.0]*len(chi_th), 1.0/(np.pi**2*chi_th), color='#e377c2', ls='--',
        lw=1.5, alpha=0.5, label=r'Stealthy theory: $1/(\pi^2\chi)$', zorder=2)

ax.set_xlabel(r'Hyperuniformity exponent $\alpha$', fontsize=13)
ax.set_ylabel(r'Surface-area coefficient $\bar{\Lambda}$', fontsize=13)
ax.set_title(r'Comprehensive $(\alpha,\,\bar{\Lambda})$ Ranking of 1D Hyperuniform Patterns',
             fontsize=13, fontweight='bold')
ax.set_xlim(0.4, alpha_clip+0.5)
ax.set_ylim(0, 1.6)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, ls=':', alpha=0.3)
plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figE_comprehensive_ranking.png')
plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {path}")


# ============================================================
# Print comprehensive table
# ============================================================
print()
print("=" * 78)
print(f"  {'COMPREHENSIVE (alpha, Lambda_bar) CATALOG':^72}")
print("=" * 78)
print(f"  {'Pattern':32s}  {'Class':>5}  {'alpha':>9}  {'Lambda_bar':>12}  {'Source':>16}")
print("  " + "-" * 78)

table = [
    ('Lattice', 'I', np.inf, catalog['Lattice']['lambda_bar'], '1/6 (exact)'),
    (CHAINS['fibonacci']['name'], 'I', 3.0, catalog[CHAINS['fibonacci']['name']]['lambda_bar'], '~1/5 (numeric)'),
    (CHAINS['silver']['name'], 'I', 3.0, catalog[CHAINS['silver']['name']]['lambda_bar'], '1/4 (exact?)'),
    (CHAINS['bronze']['name'], 'I', 3.0, catalog[CHAINS['bronze']['name']]['lambda_bar'], 'numeric'),
]

for a_val in sorted([float(k) for k in url_results.keys()]):
    key = url_key(a_val)
    if key is None: continue
    table.append((f'URL (a={a_val})', 'I', 2.0, lambda_bar_url_exact(a_val),
                  f'(1+{a_val}^2)/6 (exact)'))

for chi_val in chi_vals:
    table.append((f'Stealthy (chi={chi_val})', 'I', 2.0, stealthy_results[chi_val]['lambda_bar'],
                  f'~1/(pi^2*{chi_val})'))

table.append(('Period-Doubling', 'II', 1.0, None, 'diverges (log)'))
table.append((f'0222 Chain', 'III', alpha_0222, None, 'diverges (power)'))

for lbl, cls, alpha_val, lb, source in table:
    alpha_str = 'inf' if not np.isfinite(alpha_val) else f'{alpha_val:.3f}'
    lb_str = f'{lb:.5f}' if lb is not None else 'diverges'
    print(f"  {lbl:32s}  {cls:>5}  {alpha_str:>9}  {lb_str:>12}  {source:>16}")

print("=" * 78)
print(f"\nAll figures saved to: {RESULTS_DIR}")
