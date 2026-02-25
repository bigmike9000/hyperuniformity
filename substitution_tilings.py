"""
Phase 2: 1D Substitution Tiling Generator
Generates quasiperiodic 1D point patterns via iterative substitution rules.

Supported chains:
  - Fibonacci (golden ratio, tau = (1+sqrt(5))/2)
  - Silver ratio (1+sqrt(2))
  - Bronze ratio ((3+sqrt(13))/2)

Each chain is defined by a substitution matrix M acting on tile types [S, L]:
  Fibonacci: S->L, L->LS       M = [[0,1],[1,1]]
  Silver:    S->L, L->LSS      M = [[0,1],[1,2]]
  Bronze:    S->L, L->LSSS     M = [[0,1],[1,3]]

The tile lengths are set so that L/S equals the corresponding metallic mean,
ensuring the pattern is self-similar and Class I hyperuniform.
"""

import numpy as np
import time


# ============================================================
# Substitution rules
# ============================================================

# Each rule maps a tile character to its replacement sequence.
# Tile lengths: L = metallic_mean * S_length, with S_length = 1.

CHAINS = {
    'fibonacci': {
        'name': 'Fibonacci (Golden Ratio)',
        'matrix': np.array([[0, 1], [1, 1]]),
        'rules': {'S': 'L', 'L': 'LS'},
        'metallic_mean': (1 + np.sqrt(5)) / 2,   # tau ~ 1.618
    },
    'silver': {
        'name': 'Silver Ratio',
        'matrix': np.array([[0, 1], [1, 2]]),
        'rules': {'S': 'L', 'L': 'LLS'},
        'metallic_mean': 1 + np.sqrt(2),          # ~ 2.414
    },
    'bronze': {
        'name': 'Bronze Ratio',
        'matrix': np.array([[0, 1], [1, 3]]),
        'rules': {'S': 'L', 'L': 'LLLS'},
        'metallic_mean': (3 + np.sqrt(13)) / 2,   # ~ 3.303
    },
}


def generate_substitution_sequence(chain_name, num_iterations, seed='L'):
    """
    Generate a 1D substitution tiling sequence by iteratively applying
    the substitution rules.

    Parameters
    ----------
    chain_name : str
        One of 'fibonacci', 'silver', 'bronze'.
    num_iterations : int
        Number of substitution iterations. The chain length grows as
        ~(metallic_mean)^num_iterations.
    seed : str
        Starting tile sequence (default 'L').

    Returns
    -------
    sequence : str
        The tile sequence (e.g., 'LSLLSLSL...').
    """
    chain = CHAINS[chain_name]
    rules = chain['rules']
    seq = seed
    for _ in range(num_iterations):
        seq = ''.join(rules[ch] for ch in seq)
    return seq


def sequence_to_points(sequence, chain_name):
    """
    Convert a tile sequence to a 1D point pattern (vectorized).
    Points are placed at the left endpoint of each tile.
    Tile lengths: S = 1, L = metallic_mean.

    Parameters
    ----------
    sequence : str
        Tile sequence from generate_substitution_sequence.
    chain_name : str
        Chain name (for looking up tile lengths).

    Returns
    -------
    points : ndarray
        1D array of point positions.
    L_domain : float
        Total domain length.
    """
    metal = CHAINS[chain_name]['metallic_mean']
    seq_arr = np.frombuffer(sequence.encode(), dtype='S1')
    lengths = np.where(seq_arr == b'L', metal, 1.0)
    L_domain = float(np.sum(lengths))
    points = np.empty(len(sequence), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])
    return points, L_domain


def predict_chain_length(chain_name, num_iterations, seed='L'):
    """
    Predict the number of tiles after num_iterations substitutions
    without generating the full sequence, using the substitution matrix.
    """
    chain = CHAINS[chain_name]
    M = chain['matrix']
    # Count seed tiles: [num_S, num_L]
    vec = np.array([seed.count('S'), seed.count('L')], dtype=np.int64)
    Mn = np.linalg.matrix_power(M, num_iterations)
    result = Mn @ vec
    return int(np.sum(result))


def verify_eigenvalue_prediction(chain_name):
    """
    Verify the hyperuniformity exponent prediction alpha = 1 - 2*ln|lambda2|/ln|lambda1|
    for a given chain. Should return alpha = 3 for all metallic-mean chains.
    """
    M = CHAINS[chain_name]['matrix']
    eigenvalues = np.linalg.eigvals(M)
    eigenvalues = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
    lam1, lam2 = abs(eigenvalues[0]), abs(eigenvalues[1])
    alpha = 1 - 2 * np.log(lam2) / np.log(lam1)
    return alpha, lam1, lam2


# ============================================================
# Main: generate all three chains and report
# ============================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  1D Substitution Tiling Generator")
    print("=" * 65)

    # Verify eigenvalue predictions
    print("\nEigenvalue predictions for alpha:")
    for name in CHAINS:
        alpha, lam1, lam2 = verify_eigenvalue_prediction(name)
        print(f"  {CHAINS[name]['name']:30s}  lambda1={lam1:.4f}  "
              f"|lambda2|={lam2:.4f}  alpha={alpha:.4f}")

    # Generate chains at increasing sizes
    print("\nGenerating chains:")
    print(f"  {'Chain':15s} {'Iters':>6s} {'N (tiles)':>12s} "
          f"{'Domain L':>12s} {'Density rho':>12s} {'Time':>8s}")
    print("  " + "-" * 70)

    results = {}
    for name in CHAINS:
        # Find the right number of iterations to get N ~ 10^5 - 10^6
        # (keep moderate for initial testing; can increase later)
        for iters in range(5, 40):
            n_pred = predict_chain_length(name, iters)
            if n_pred > 500_000:
                break

        t0 = time.perf_counter()
        seq = generate_substitution_sequence(name, iters)
        points, L_domain = sequence_to_points(seq, name)
        elapsed = time.perf_counter() - t0

        rho = len(points) / L_domain
        results[name] = {
            'points': points,
            'L_domain': L_domain,
            'sequence': seq,
            'num_iterations': iters,
        }

        print(f"  {CHAINS[name]['name']:15s} {iters:6d} {len(points):12,d} "
              f"{L_domain:12.1f} {rho:12.6f} {elapsed:7.2f}s")

    # Verify tile ratio converges to metallic mean
    print("\nTile ratio verification (L_count / S_count -> metallic mean):")
    for name in CHAINS:
        seq = results[name]['sequence']
        n_L = seq.count('L')
        n_S = seq.count('S')
        ratio = n_L / n_S if n_S > 0 else float('inf')
        expected = CHAINS[name]['metallic_mean']
        print(f"  {CHAINS[name]['name']:30s}  L/S = {ratio:.6f}  "
              f"(expected {expected:.6f}, err = {abs(ratio - expected):.2e})")

    # Quick sanity check: spacing histogram
    print("\nSpacing statistics (should show exactly 2 distinct spacings):")
    for name in CHAINS:
        pts = results[name]['points']
        spacings = np.diff(pts)
        unique_spacings = np.unique(np.round(spacings, 8))
        print(f"  {CHAINS[name]['name']:30s}  unique spacings: {unique_spacings}")

    print("\nDone. Point arrays stored in `results` dict.")
