"""
Figure for JP introduction: two Class-I chains with alpha=3 but different Lambda-bar.
Shows Fibonacci and Copper as colored tile segments on separate rows.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── substitution rules ──────────────────────────────────────────────────────
RULES = {
    'fibonacci': {'S': 'L',      'L': 'LS'},
    'copper':    {'S': 'L',      'L': 'LLLLS'},
}
METALLIC = {
    'fibonacci': (1 + np.sqrt(5)) / 2,   # ~1.618
    'copper':    2 + np.sqrt(5),          # ~4.236
}

def generate_sequence(name, n_iter=12):
    rules = RULES[name]
    seq = 'L'
    for _ in range(n_iter):
        seq = ''.join(rules[c] for c in seq)
    return seq

def tile_positions(seq, mu):
    """Return left-edge positions and lengths; S=1, L=mu."""
    lengths = [mu if c == 'L' else 1.0 for c in seq]
    positions = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    return positions, lengths

# ── build sequences ─────────────────────────────────────────────────────────
seq_fib = generate_sequence('fibonacci', n_iter=14)
seq_cop = generate_sequence('copper',    n_iter=6)

pos_fib, len_fib = tile_positions(seq_fib, METALLIC['fibonacci'])
pos_cop, len_cop = tile_positions(seq_cop, METALLIC['copper'])

# Window: show the first W units of each chain
W = 60.0

mask_fib = pos_fib < W
mask_cop = pos_cop < W

pos_fib, len_fib = pos_fib[mask_fib], np.array(len_fib)[mask_fib]
pos_cop, len_cop = pos_cop[mask_cop], np.array(len_cop)[mask_cop]

# Clip last tile to window edge
len_fib[-1] = min(len_fib[-1], W - pos_fib[-1])
len_cop[-1] = min(len_cop[-1], W - pos_cop[-1])

# ── colors ───────────────────────────────────────────────────────────────────
COL_S = '#4C72B0'   # blue  for short tile
COL_L = '#DD8452'   # orange for long tile

def seq_mask(seq, mask):
    return [c for c, m in zip(seq, mask) if m]

tiles_fib = seq_mask(seq_fib, np.where(mask_fib)[0] < len(seq_fib))
tiles_cop = seq_mask(seq_cop, np.where(mask_cop)[0] < len(seq_cop))

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(7, 2.2),
                         gridspec_kw={'hspace': 0.55})

TILE_H   = 0.55   # tile height
POINT_Y  = 0.0    # y-level for point markers

def draw_chain(ax, positions, lengths, tiles, title, lam_bar):
    for x, w, t in zip(positions, lengths, tiles):
        col = COL_L if t == 'L' else COL_S
        rect = mpatches.FancyBboxPatch(
            (x, -TILE_H/2), w, TILE_H,
            boxstyle="square,pad=0",
            linewidth=0.4, edgecolor='white', facecolor=col, alpha=0.85)
        ax.add_patch(rect)
    # tick marks at tile boundaries
    boundaries = list(positions) + [positions[-1] + lengths[-1]]
    ax.vlines(boundaries, -TILE_H/2, TILE_H/2,
              color='white', linewidth=0.6, alpha=0.5)
    ax.set_xlim(0, W)
    ax.set_ylim(-0.7, 0.9)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(W/2, 0.72,
            title,
            ha='center', va='bottom', fontsize=9.5,
            fontfamily='serif')

draw_chain(axes[0], pos_fib, len_fib, tiles_fib,
           'Fibonacci chain', r'0.201')
draw_chain(axes[1], pos_cop, len_cop, tiles_cop,
           'Copper chain', r'0.293')

# shared legend
patch_L = mpatches.Patch(color=COL_L, label='Long tile ($L$)')
patch_S = mpatches.Patch(color=COL_S, label='Short tile ($S$)')
fig.legend(handles=[patch_L, patch_S], loc='lower center',
           ncol=2, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, -0.08))

for d in ['results/figures', 'jp/figures']:
    fig.savefig(f'{d}/fig_intro_chains.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{d}/fig_intro_chains.png', bbox_inches='tight', dpi=300)
print("Saved fig_intro_chains to results/figures/ and jp/figures/")
