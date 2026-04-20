"""Replot fig_metallic_n500.png from saved data with style fixes."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(SCRIPT_DIR, 'results', 'metallic_n500.json')
OUT_FIG = os.path.join(SCRIPT_DIR, 'results', 'figures', 'fig_metallic_n500.png')

with open(JSON_PATH) as f:
    data = json.load(f)

ns = [int(x) for x in data['n_values']]
results = data['results']
lbs = [results[str(n)]['lambda_bar'] for n in ns]
errs = [results[str(n)]['err'] for n in ns]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ns, lbs, yerr=errs, fmt='o-', color='#2166ac', ms=5,
            capsize=3, lw=1.5, label=r'Measured $\bar\Lambda(\mu_n)$')
ax.axhline(y=1/3, color='#d6604d', ls='--', lw=1.5,
           label=r'$1/3$')
ax.set_xlabel(r'Metallic-mean index $n$', fontsize=12)
ax.set_ylabel(r'$\bar\Lambda$', fontsize=12)
# no title — info goes in the LaTeX caption
ax.legend(fontsize=10)
ax.set_xscale('log')
ax.set_xlim(0.8, 700)
ax.set_ylim(0.18, 0.38)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {OUT_FIG}")
