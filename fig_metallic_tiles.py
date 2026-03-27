"""
Generate a visual figure showing metallic-mean substitution tile sequences.

Displays horizontal rows of L (long) and S (short) tiles for
n=1 (Fibonacci), n=2 (Silver), n=3 (Bronze), and n=5 (Nickel),
colored distinctly to show the quasiperiodic structure.

Output: results/figures/fig_metallic_tiles.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'fig_metallic_tiles.png')

# Chains to display
CHAINS_TO_SHOW = [
    (1, 'Fibonacci', r'$n=1$'),
    (2, 'Silver',    r'$n=2$'),
    (3, 'Bronze',    r'$n=3$'),
    (5, 'Nickel',    r'$n=5$'),
]

TARGET_TILES = 25   # approximate number of tiles per row
TARGET_WIDTH = 45.0 # normalize all rows to roughly this total width
TILE_HEIGHT = 0.55  # height of each tile rectangle
ROW_SPACING = 1.2   # vertical spacing between rows

# Colors
COLOR_L = '#2166ac'   # blue for Long tiles
COLOR_S = '#d6604d'   # red-orange for Short tiles
EDGE_COLOR = 'white'
EDGE_WIDTH = 1.2

# ----------------------------------------------------------------
# Chain generation (inline, same logic as metallic_mean_convergence.py)
# ----------------------------------------------------------------

def metallic_mean(n):
    """mu_n = (n + sqrt(n^2 + 4)) / 2"""
    return (n + np.sqrt(n**2 + 4)) / 2


def generate_sequence(n, target_tiles=TARGET_TILES):
    """
    Generate metallic-mean substitution chain of index n.
    Rules: S -> L, L -> L^n S
    Iterate until we have at least target_tiles tiles, then truncate.
    """
    rules = {'S': 'L', 'L': 'L' * n + 'S'}
    seq = 'L'
    for _ in range(50):
        if len(seq) >= target_tiles:
            break
        seq = ''.join(rules[ch] for ch in seq)
    return seq[:target_tiles]


# ----------------------------------------------------------------
# Figure
# ----------------------------------------------------------------

def make_figure():
    n_rows = len(CHAINS_TO_SHOW)
    fig_height = 1.0 + n_rows * ROW_SPACING
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for row_idx, (n, name, label_tex) in enumerate(CHAINS_TO_SHOW):
        mu = metallic_mean(n)
        seq = generate_sequence(n)

        # Compute raw total width, then scale factor to normalize
        raw_width = sum(mu if ch == 'L' else 1.0 for ch in seq)
        scale = TARGET_WIDTH / raw_width

        y_center = (n_rows - 1 - row_idx) * ROW_SPACING

        # Build tile rectangles
        x = 0.0
        for ch in seq:
            tile_len = (mu if ch == 'L' else 1.0) * scale
            color = COLOR_L if ch == 'L' else COLOR_S
            rect = Rectangle(
                (x, y_center - TILE_HEIGHT / 2),
                tile_len, TILE_HEIGHT,
                facecolor=color,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_WIDTH,
                zorder=2,
            )
            ax.add_patch(rect)
            # Letter label inside tile (only if tile wide enough)
            if tile_len > 0.6:
                ax.text(
                    x + tile_len / 2, y_center,
                    ch, ha='center', va='center',
                    fontsize=7 if tile_len < 1.0 else 8,
                    color='white', fontweight='bold', zorder=3,
                )
            x += tile_len

        # Row label on the left
        ax.text(
            -0.5, y_center,
            f'{name} ({label_tex})',
            ha='right', va='center',
            fontsize=11, fontweight='bold',
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_L, edgecolor='white',
                  linewidth=0.8, label=r'L (long, length $\mu_n$)'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_S, edgecolor='white',
                  linewidth=0.8, label=r'S (short, length 1)'),
    ]
    ax.legend(
        handles=legend_elements, loc='upper right',
        fontsize=9, framealpha=0.9, edgecolor='#cccccc',
    )

    # Axis formatting
    ax.set_xlim(-0.3, TARGET_WIDTH + 0.5)
    ax.set_ylim(-ROW_SPACING * 0.6, (n_rows - 1) * ROW_SPACING + ROW_SPACING * 0.6)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.set_title(
        'Metallic-Mean Substitution Chains',
        fontsize=14, fontweight='bold', pad=12,
    )
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Tile counts: ", end='')
    for n, name, _ in CHAINS_TO_SHOW:
        seq = generate_sequence(n)
        n_L = seq.count('L')
        n_S = seq.count('S')
        print(f"{name}: {len(seq)} tiles ({n_L}L + {n_S}S)  ", end='')
    print()


if __name__ == '__main__':
    make_figure()
