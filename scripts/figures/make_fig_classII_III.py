"""
Generate a clean 2-panel variance figure for the presentation:
  Left:  Period-Doubling (Class II, logarithmic growth)
  Right: 0222 Chain     (Class III, power-law growth)
Saves to results/figures/fig_variance_II_III.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, 'results', 'data')
OUT_DIR  = os.path.join(BASE, 'results', 'figures')

with open(os.path.join(DATA_DIR, 'catalog.json')) as f:
    catalog = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(r'Number Variance $\sigma^2(R)$ — Classes II and III', fontsize=13)

# ---- Class II: Period-Doubling ----
ax = axes[0]
ax.set_title('Class II: Logarithmic Growth', fontsize=12, fontweight='bold')

R_pd = np.array(catalog['Period-Doubling']['R_array'])
v_pd = np.array(catalog['Period-Doubling']['variances'])
mask = R_pd > 1
ax.loglog(R_pd[mask], v_pd[mask], color='#ff7f0e', lw=1.8, label='Period-Doubling')

C = catalog['Period-Doubling'].get('log_coeff_C')
b = catalog['Period-Doubling'].get('log_coeff_b')
if C and b:
    R_fit = R_pd[mask]
    ax.loglog(R_fit, C * np.log(R_fit) + b, 'k--', lw=1.8, alpha=0.7,
              label=rf'$C\ln R + b,\ C={C:.3f}$')

ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05,
        r'Period-Doubling, $\alpha = 1$' '\n' r'$\sigma^2 \sim C\ln R$',
        transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ---- Class III: 0222 Chain ----
ax = axes[1]
ax.set_title('Class III: Power-Law Growth', fontsize=12, fontweight='bold')

R_0222 = np.array(catalog['0222 Chain']['R_array'])
v_0222 = np.array(catalog['0222 Chain']['variances'])
mask3 = R_0222 > 1
ax.loglog(R_0222[mask3], v_0222[mask3], color='#17becf', lw=1.8, label='0222 Chain')

beta = catalog['0222 Chain'].get('power_law_beta')
if beta:
    mid = len(R_0222[mask3]) // 2
    C3 = v_0222[mask3][mid] / R_0222[mask3][mid] ** beta
    ax.loglog(R_0222[mask3], C3 * R_0222[mask3] ** beta, 'k--', lw=1.8, alpha=0.7,
              label=rf'$\sim R^{{{beta:.3f}}}$')

alpha_disp = catalog['0222 Chain']['alpha']
ax.set_xlabel(r'$R$', fontsize=12)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)
ax.text(0.05, 0.05,
        rf'0222 Chain, $\alpha = {alpha_disp:.3f}$' '\n' r'$\sigma^2 \sim R^{1-\alpha}$',
        transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_variance_II_III.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
