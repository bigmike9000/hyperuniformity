"""
Generate stealthy S(k) overlay figure.
Three chi values plotted on a single axis with exclusion zone annotation.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'data', 'stealthy_collab_results.json')
OUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'figures', 'stealthy_Sk_overlay.png')

with open(DATA_PATH) as f:
    data = json.load(f)

CHI_COLORS = {
    '0.1': '#e8205a',   # red-pink
    '0.2': '#9c27b0',   # purple
    '0.3': '#3f51b5',   # blue
}

fig, ax = plt.subplots(figsize=(9, 5.5))

for chi_str, color in CHI_COLORS.items():
    r = data[chi_str]
    k = np.array(r['k_bin'])
    S = np.array(r['S_bin'])
    K = r['K_theory']
    chi_val = float(chi_str)

    ax.plot(k, S, '-', color=color, lw=1.8,
            label=rf'$\chi = {chi_val}$, $K = {K:.2f}$')
    ax.axvline(K, color=color, ls='--', lw=1.0, alpha=0.55)

# Reference line S(k) = 1
ax.axhline(1.0, color='gray', ls=':', lw=1.0, alpha=0.5)

# Axes limits — y_max chosen to fully enclose the chi=0.3 spike (max ~1.658)
ax.set_xlim(0, 4.0)
ax.set_ylim(-0.02, 1.75)

ax.set_xlabel('Wavenumber $k$', fontsize=13)
ax.set_ylabel('Structure factor $S(k)$', fontsize=13)
ax.set_title(r'Stealthy Patterns: $S(k) = 0$ for $k < K = 2\pi\chi$', fontsize=13)


ax.legend(fontsize=10, loc='lower right')
ax.grid(True, ls=':', alpha=0.35)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_PATH}")
