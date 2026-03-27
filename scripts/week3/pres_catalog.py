"""
Presentation-quality catalog figure: full Λ̄ ranking of all 1D hyperuniform patterns.
Output: results/figures/fig_catalog_updated.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

# All Class-I patterns with exact/measured Λ̄
# (label, lambda_bar, alpha_str, color, hatch)
patterns = [
    ('Integer lattice',    1/6,    r'$\alpha=\infty$',  '#2196F3', ''),
    ('Fibonacci',          0.2011, r'$\alpha=3$',        '#43A047', ''),
    ('Silver',             0.2500, r'$\alpha=3$ (1/4?)',  '#66BB6A', ''),
    ('Cubic $\\alpha=2$',  0.275,  r'$\alpha=2$',        '#FFA726', ''),
    ('Bronze',             0.2821, r'$\alpha=3$',        '#D4E157', ''),
    ('Copper',             0.2930, r'$\alpha=3$',        '#FFCA28', ''),
    ('Nickel',             0.3100, r'$\alpha=3$',        '#FFB300', ''),
    ('URL ($a{=}1$)',       1/3,   r'$\alpha=2$',        '#AB47BC', ''),
    ('Stealthy $\\chi{=}0.3$', 0.357, r'no $\alpha$',   '#FF7043', '///'),
    ('Bombieri-Taylor',    0.377,  r'$\alpha=1.545$',    '#EF5350', ''),
    ('Stealthy $\\chi{=}0.2$', 0.526, r'no $\alpha$',   '#FF7043', '///'),
    ('$\\alpha=2.071$ chain',   0.650, r'$\alpha=2.071$ (rough)',  '#E91E63', ''),
    ('Stealthy $\\chi{=}0.1$', 1.021, r'no $\alpha$',   '#FF7043', '///'),
]

labels   = [p[0] for p in patterns]
values   = [p[1] for p in patterns]
alphas   = [p[2] for p in patterns]
colors   = [p[3] for p in patterns]
hatches  = [p[4] for p in patterns]

fig, ax = plt.subplots(figsize=(9, 6))

y = np.arange(len(patterns))
bars = ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.8, height=0.7)

# Apply hatching to stealthy bars
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Reference lines
for xv, lbl, ls in [(1/6, '1/6', '--'), (1/4, '1/4', ':'), (1/3, '1/3', '--')]:
    ax.axvline(xv, color='gray', ls=ls, lw=1.0, alpha=0.6)
    ax.text(xv, len(patterns) - 0.2, lbl, ha='center', va='bottom',
            fontsize=9, color='gray')

# Value + alpha labels on bars
for i, (val, alpha_lbl) in enumerate(zip(values, alphas)):
    ax.text(val + 0.01, i, f'{val:.3f}  {alpha_lbl}',
            va='center', ha='left', fontsize=11)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel(r'Surface-area coefficient $\bar\Lambda$', fontsize=13)
ax.set_xlim(0, 1.18)
ax.set_title(r'Ranking of 1D Hyperuniform Patterns by $\bar\Lambda$'
             '\n(hatched = stealthy, no power-law $\\alpha$)', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Note for Classes II/III
ax.text(0.98, 0.02,
        'Period-Doubling (Class II) and 0222 chain (Class III)\n'
        r'have $\bar\Lambda=\infty$; described by $C_{II}$, $A_{III}$ instead.',
        transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#cccccc', alpha=0.9))

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'fig_catalog_updated.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
