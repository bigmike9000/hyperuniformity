"""Replot URL sweep from saved JSON data (no recomputation)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SCRIPT_DIR)
from disordered_patterns import lambda_bar_url_exact

JSON_PATH = os.path.join(SCRIPT_DIR, 'results', 'url_sweep.json')
FIG_PATH = os.path.join(SCRIPT_DIR, 'results', 'figures', 'fig_url_sweep.png')

with open(JSON_PATH) as f:
    data = json.load(f)

sim_results = data['simulations']

# Dense grid for exact formula
A_FINE = np.linspace(0.001, 2.0, 200)
lb_exact = np.array([lambda_bar_url_exact(a) for a in A_FINE])

a_sim = [r['a'] for r in sim_results]
lb_sim = [r['lambda_bar_sim'] for r in sim_results]
err_sim = [r['err'] for r in sim_results]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                gridspec_kw={'height_ratios': [3, 1]})

# Top panel
ax1.plot(A_FINE, lb_exact, 'b-', lw=2, label=r'Exact formula')
ax1.errorbar(a_sim, lb_sim, yerr=err_sim, fmt='ro', ms=5, capsize=3,
             label='Simulation', zorder=5)

ax1.axhline(y=1/6, color='gray', ls='--', lw=0.8, alpha=0.7)
ax1.text(0.05, 1/6 + 0.005, r'$1/6$ (lattice)', fontsize=8, color='gray')
ax1.axhline(y=1/3, color='gray', ls='--', lw=0.8, alpha=0.7)
ax1.text(0.05, 1/3 + 0.005, r'$1/3$ (cloaked)', fontsize=8, color='gray')

ax1.axvline(x=1.0, color='green', ls=':', lw=1.2, alpha=0.7)
ax1.text(1.02, 0.20, 'cloaking\n($a=1$)', fontsize=8, color='green')

ax1.set_ylabel(r'$\bar\Lambda$', fontsize=12)
# no title — info goes in the LaTeX caption
ax1.legend(fontsize=10)
ax1.set_xlim(0, 2.05)

# Bottom panel: relative error
rel_err = [(r['lambda_bar_sim'] - r['lambda_bar_exact']) / r['lambda_bar_exact'] * 100
           for r in sim_results]
ax2.bar(a_sim, rel_err, width=0.05, color='steelblue', alpha=0.7)
ax2.axhline(y=0, color='black', lw=0.5)
ax2.set_xlabel(r'Displacement amplitude $a$', fontsize=12)
ax2.set_ylabel('Relative error (%)', fontsize=10)
ax2.set_xlim(0, 2.05)
ax2.set_ylim(-3, 3)

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {FIG_PATH}")
