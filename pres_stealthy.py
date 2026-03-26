"""
Single-panel stealthy variance figure from cached results.
Output: results/figures/figH_stealthy_variance.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results', 'figures')

# Load cached results
with open(os.path.join(BASE, 'results', 'data', 'stealthy_collab_results.json')) as f:
    all_results = json.load(f)

chi_colors = {0.1: '#E91E63', 0.2: '#9C27B0', 0.3: '#3F51B5'}

fig, ax = plt.subplots(figsize=(8, 5.5))

for chi_str in ['0.1', '0.2', '0.3']:
    r = all_results[chi_str]
    chi_val = r['chi']
    R = np.array(r['R_array'])
    var = np.array(r['var_mean'])
    var_std = np.array(r['var_std'])
    lb = r['lambda_bar']
    lb_err = r['lambda_bar_err']
    color = chi_colors[chi_val]

    mask = R > 0.5
    ax.fill_between(R[mask],
                    (var - var_std)[mask], (var + var_std)[mask],
                    alpha=0.15, color=color)
    ax.plot(R[mask], var[mask], '-', color=color, lw=1.5,
            label=rf'$\chi={chi_val}$,  $\bar{{\Lambda}}={lb:.3f}\pm{lb_err:.3f}$')
    ax.axhline(lb, color=color, ls='--', lw=0.8, alpha=0.7)

ax.axhline(1/6, color='black', ls=':', lw=1.2, alpha=0.5,
           label=r'Lattice: $\bar\Lambda = 1/6$')

ax.set_xlabel(r'$R$', fontsize=14)
ax.set_ylabel(r'$\sigma^2(R)$', fontsize=14)
ax.set_title('Ensemble-Averaged Number Variance (Stealthy, $N=2000$, $\\rho=1$)',
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, ls=':', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'figH_stealthy_variance.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
