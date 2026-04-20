"""
Extended metallic-mean tile visualization for Week 4.

Shows n=1,2,3,4,5,6,8,10 with 25 tiles each,
illustrating how the chain structure changes with n.

Output: results/figures/fig_metallic_tiles_extended.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'fig_metallic_tiles_extended.png')

CHAINS = [
    (1,  'Fibonacci'),
    (2,  'Silver'),
    (3,  'Bronze'),
    (4,  'Copper'),
    (5,  'Nickel'),
    (6,  '$n=6$'),
    (8,  '$n=8$'),
    (10, '$n=10$'),
]

TARGET_TILES = 25
TARGET_WIDTH = 50.0
TILE_HEIGHT = 0.50
ROW_SPACING = 1.05

COLOR_L = '#2166ac'
COLOR_S = '#d6604d'


def metallic_mean(n):
    return (n + np.sqrt(n**2 + 4)) / 2


def generate_sequence(n, target_tiles=TARGET_TILES):
    rules = {'S': 'L', 'L': 'L' * n + 'S'}
    seq = 'L'
    for _ in range(50):
        if len(seq) >= target_tiles:
            break
        seq = ''.join(rules[ch] for ch in seq)
    return seq[:target_tiles]


def make_figure():
    n_rows = len(CHAINS)
    fig_height = 0.8 + n_rows * ROW_SPACING
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for row_idx, (n, name) in enumerate(CHAINS):
        mu = metallic_mean(n)
        seq = generate_sequence(n)

        raw_width = sum(mu if ch == 'L' else 1.0 for ch in seq)
        scale = TARGET_WIDTH / raw_width

        y_center = (n_rows - 1 - row_idx) * ROW_SPACING

        x = 0.0
        for ch in seq:
            tile_len = (mu if ch == 'L' else 1.0) * scale
            color = COLOR_L if ch == 'L' else COLOR_S
            rect = Rectangle(
                (x, y_center - TILE_HEIGHT / 2),
                tile_len, TILE_HEIGHT,
                facecolor=color, edgecolor='white', linewidth=1.0, zorder=2,
            )
            ax.add_patch(rect)
            if tile_len > 0.5:
                ax.text(x + tile_len / 2, y_center, ch,
                        ha='center', va='center',
                        fontsize=6 if tile_len < 0.8 else 7,
                        color='white', fontweight='bold', zorder=3)
            x += tile_len

        # Count tiles
        n_L = seq.count('L')
        n_S = seq.count('S')
        freq_S = n_S / len(seq)

        label = f'{name} ($n={n}$)' if not name.startswith('$') else name
        ax.text(-0.5, y_center, label,
                ha='right', va='center', fontsize=10, fontweight='bold')
        ax.text(TARGET_WIDTH + 0.5, y_center,
                f'{n_S}S/{n_L}L = {freq_S:.0%} short',
                ha='left', va='center', fontsize=8, color='#666666')

    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_L, edgecolor='white',
                  linewidth=0.8, label=r'L (long, length $\mu_n$)'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_S, edgecolor='white',
                  linewidth=0.8, label=r'S (short, length 1)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    ax.set_xlim(-0.3, TARGET_WIDTH + 8)
    ax.set_ylim(-ROW_SPACING * 0.6,
                (n_rows - 1) * ROW_SPACING + ROW_SPACING * 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # no overall title — info goes in the LaTeX caption

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    for n, name in CHAINS:
        seq = generate_sequence(n)
        n_L = seq.count('L')
        n_S = seq.count('S')
        print(f"  {name:12s} (n={n}): {len(seq)} tiles ({n_L}L + {n_S}S), "
              f"S fraction = {n_S/len(seq):.3f}")


if __name__ == '__main__':
    make_figure()
