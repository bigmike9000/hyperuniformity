"""
Phase 2: 1D Substitution Tiling Generator
Generates quasiperiodic 1D point patterns via iterative substitution rules.

Supported chains:
  - Fibonacci (golden ratio, tau = (1+sqrt(5))/2)
  - Silver ratio (1+sqrt(2))
  - Bronze ratio ((3+sqrt(13))/2)
  - Copper ratio (2+sqrt(5))
  - Nickel ratio ((5+sqrt(29))/2)

Each chain is defined by a substitution matrix M acting on tile types [S, L]:
  Fibonacci: S->L, L->LS       M = [[0,1],[1,1]]
  Silver:    S->L, L->LLS      M = [[0,1],[1,2]]
  Bronze:    S->L, L->LLLS     M = [[0,1],[1,3]]
  Copper:    S->L, L->LLLLS    M = [[0,1],[1,4]]
  Nickel:    S->L, L->LLLLLS   M = [[0,1],[1,5]]

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
    'copper': {
        'name': 'Copper Ratio',
        'matrix': np.array([[0, 1], [1, 4]]),
        'rules': {'S': 'L', 'L': 'LLLLS'},
        'metallic_mean': (4 + np.sqrt(20)) / 2,   # = 2 + sqrt(5) ~ 4.236
    },
    'nickel': {
        'name': 'Nickel Ratio',
        'matrix': np.array([[0, 1], [1, 5]]),
        'rules': {'S': 'L', 'L': 'LLLLLS'},
        'metallic_mean': (5 + np.sqrt(29)) / 2,   # ~ 5.193
    },
    'period_doubling': {
        'name': 'Period-Doubling',
        'matrix': np.array([[0, 2], [1, 1]]),
        'rules': {'S': 'L', 'L': 'LSS'},
        'metallic_mean': 2.0,   # tile ratio L/S = 2; eigenvalues 2 and -1 -> alpha=1 (Class II)
    },
    'chain_0222': {
        'name': '0222 Chain',
        'matrix': np.array([[0, 2], [2, 2]]),
        'rules': {'S': 'LL', 'L': 'SSLL'},
        'metallic_mean': (1 + np.sqrt(5)) / 2,   # eigenvalues 1+sqrt(5), 1-sqrt(5) -> alpha~0.639 (Class III)
    },
    # ----------------------------------------------------------------
    # Cubic irrational substitution from Bombieri & Taylor (1986)
    # "Which Distributions of Matter Diffract?", J. Physique C3-21
    # Characteristic equation: x^3 - 2x^2 - x + 1 = 0
    # Largest eigenvalue theta_1 ~ 2.247 is a Pisot-Vijayaraghavan number
    # ----------------------------------------------------------------
    'bombieri_taylor': {
        'name': 'Bombieri-Taylor Cubic',
        'matrix': np.array([[2, 0, 1], [1, 0, 1], [0, 1, 0]]),
        'rules': {'a': 'aac', 'b': 'ac', 'c': 'b'},
        'alphabet': 'abc',  # 3-letter alphabet (cubic irrational)
    },
}


def compute_tile_lengths(chain_name):
    """
    Compute tile lengths from the right eigenvector of the substitution matrix.

    For N-letter alphabets, the right eigenvector corresponding to the largest
    eigenvalue gives the tile length ratios preserved under inflation.

    Parameters
    ----------
    chain_name : str
        Chain name in CHAINS dict.

    Returns
    -------
    tile_lengths : dict
        Maps each letter to its tile length (normalized so shortest = 1).
    theta1 : float
        The largest eigenvalue (inflation factor).
    """
    chain = CHAINS[chain_name]
    M = chain['matrix']

    # For 2-letter systems, use metallic_mean directly
    if 'metallic_mean' in chain:
        metal = chain['metallic_mean']
        return {'S': 1.0, 'L': metal}, metal

    # For n-letter systems, compute from eigenvector
    alphabet = chain.get('alphabet', 'SL')
    eigenvalues, eigenvectors = np.linalg.eig(M)
    idx = np.argmax(np.abs(eigenvalues))
    theta1 = eigenvalues[idx].real
    right_eigenvec = eigenvectors[:, idx].real

    # Normalize so shortest tile = 1
    right_eigenvec = np.abs(right_eigenvec)
    right_eigenvec = right_eigenvec / np.min(right_eigenvec)

    tile_lengths = {letter: right_eigenvec[i] for i, letter in enumerate(alphabet)}
    return tile_lengths, theta1


def sequence_to_points_general(sequence, chain_name):
    """
    Convert a tile sequence to 1D point positions (generalized for any alphabet).

    Points are placed at the left endpoint of each tile.

    Parameters
    ----------
    sequence : str
        Tile sequence from generate_substitution_sequence.
    chain_name : str
        Chain name for looking up tile configuration.

    Returns
    -------
    points : ndarray
        1D array of point positions.
    L_domain : float
        Total domain length.
    """
    tile_lengths, _ = compute_tile_lengths(chain_name)

    # Vectorized length lookup
    lengths = np.array([tile_lengths[ch] for ch in sequence])
    L_domain = float(np.sum(lengths))

    # Points at left endpoint of each tile
    points = np.empty(len(sequence), dtype=np.float64)
    points[0] = 0.0
    np.cumsum(lengths[:-1], out=points[1:])

    return points, L_domain


def generate_substitution_sequence(chain_name, num_iterations, seed=None):
    """
    Generate a 1D substitution tiling sequence by iteratively applying
    the substitution rules.

    Parameters
    ----------
    chain_name : str
        One of 'fibonacci', 'silver', 'bronze', 'bombieri_taylor', etc.
    num_iterations : int
        Number of substitution iterations. The chain length grows as
        ~(largest_eigenvalue)^num_iterations.
    seed : str or None
        Starting tile sequence. If None, uses 'L' for 2-letter systems
        or 'a' for 3-letter systems.

    Returns
    -------
    sequence : str
        The tile sequence (e.g., 'LSLLSLSL...' or 'aacaacb...').
    """
    chain = CHAINS[chain_name]
    rules = chain['rules']

    # Default seed based on alphabet
    if seed is None:
        alphabet = chain.get('alphabet', 'SL')
        seed = alphabet[-1] if len(alphabet) == 2 else alphabet[0]  # 'L' for SL, 'a' for abc

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


def predict_chain_length(chain_name, num_iterations, seed=None):
    """
    Predict the number of tiles after num_iterations substitutions
    without generating the full sequence, using the substitution matrix.

    Generalized for any alphabet size.
    """
    chain = CHAINS[chain_name]
    M = chain['matrix']
    alphabet = chain.get('alphabet', 'SL')

    # Default seed based on alphabet
    if seed is None:
        seed = alphabet[-1] if len(alphabet) == 2 else alphabet[0]

    # Count seed tiles for each letter in alphabet
    vec = np.array([seed.count(ch) for ch in alphabet], dtype=np.int64)
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
        # Use generalized function for all chains (works for both 2 and 3 letter)
        points, L_domain = sequence_to_points_general(seq, name)
        elapsed = time.perf_counter() - t0

        rho = len(points) / L_domain
        results[name] = {
            'points': points,
            'L_domain': L_domain,
            'sequence': seq,
            'num_iterations': iters,
        }

        print(f"  {CHAINS[name]['name']:25s} {iters:6d} {len(points):12,d} "
              f"{L_domain:12.1f} {rho:12.6f} {elapsed:7.2f}s")

    # Verify tile ratio converges to expected value
    print("\nTile ratio verification:")
    for name in CHAINS:
        seq = results[name]['sequence']
        chain = CHAINS[name]
        alphabet = chain.get('alphabet', 'SL')

        if 'metallic_mean' in chain:
            # 2-letter: L/S ratio
            n_L = seq.count('L')
            n_S = seq.count('S')
            ratio = n_L / n_S if n_S > 0 else float('inf')
            expected = chain['metallic_mean']
            print(f"  {chain['name']:30s}  L/S = {ratio:.6f}  "
                  f"(expected {expected:.6f}, err = {abs(ratio - expected):.2e})")
        else:
            # n-letter: show tile counts
            tile_lengths, theta1 = compute_tile_lengths(name)
            counts = {ch: seq.count(ch) for ch in alphabet}
            total = sum(counts.values())
            freqs = {ch: counts[ch] / total for ch in alphabet}
            print(f"  {chain['name']:30s}  theta1={theta1:.4f}  "
                  f"freqs: {', '.join(f'{ch}:{freqs[ch]:.3f}' for ch in alphabet)}")

    # Quick sanity check: spacing histogram
    print("\nSpacing statistics (# distinct spacings = alphabet size):")
    for name in CHAINS:
        chain = CHAINS[name]
        alphabet = chain.get('alphabet', 'SL')
        pts = results[name]['points']
        spacings = np.diff(pts)
        unique_spacings = np.unique(np.round(spacings, 6))
        expected_n = len(alphabet)
        status = "OK" if len(unique_spacings) == expected_n else "CHECK"
        print(f"  {chain['name']:30s}  {len(unique_spacings)} spacings (expected {expected_n}): "
              f"{unique_spacings} [{status}]")

    print("\nDone. Point arrays stored in `results` dict.")
