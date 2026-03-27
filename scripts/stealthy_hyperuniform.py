"""
Stealthy Hyperuniform Point Patterns in 1D

Generates 1D stealthy hyperuniform point configurations via collective-
coordinate optimization. A point pattern is stealthy hyperuniform if its
structure factor vanishes over a finite range of wavevectors:

    S(k) = 0   for all 0 < |k| <= K

This is the strongest form of hyperuniformity. The constrained fraction
chi = M / (N - 1) controls the degree of stealthiness:
  - chi < 0.5: disordered ground states (no long-range order)
  - chi -> 0.5: approaches crystalline order

The optimization minimizes the collective energy:
    Phi = sum_{n=1}^{M} |n_tilde(k_n)|^2
where n_tilde(k) = sum_j exp(-i*k*r_j) and k_n = 2*pi*n/L.

References:
  Torquato, Zhang, Stillinger (2015) Phys. Rev. X 5, 021020
  Batten, Stillinger, Torquato (2008) J. Appl. Phys. 104, 033504
  Zhang, Stillinger, Torquato (2015) Phys. Rev. E 92, 022119
"""

import numpy as np
from scipy.optimize import minimize
import time


# ============================================================
# Core generation
# ============================================================

def _objective_and_gradient(positions, L, M):
    """
    Compute the stealthy energy Phi and its gradient.

    Phi = sum_{n=1}^{M} |n_tilde(k_n)|^2

    dPhi/dr_j = -2 * sum_{n=1}^{M} k_n * Im[ n_tilde(k_n) * exp(i*k_n*r_j) ]

    Parameters
    ----------
    positions : ndarray, shape (N,)
        Particle positions in [0, L).
    L : float
        Box length (periodic).
    M : int
        Number of constrained wavevectors.

    Returns
    -------
    Phi : float
        Total energy.
    grad : ndarray, shape (N,)
        Gradient of Phi with respect to each particle position.
    """
    N = len(positions)
    k_n = 2 * np.pi * np.arange(1, M + 1) / L  # shape (M,)

    # n_tilde(k_n) = sum_j exp(-i * k_n * r_j)
    # Build phase matrix: shape (M, N)
    phases = np.outer(k_n, positions)  # (M, N)
    exp_neg = np.exp(-1j * phases)     # (M, N)
    n_tilde = np.sum(exp_neg, axis=1)  # (M,)

    # Energy
    intensity = np.abs(n_tilde) ** 2   # (M,)
    Phi = np.sum(intensity)

    # Gradient: dPhi/dr_j = -2 * sum_n k_n * Im[ n_tilde(k_n) * exp(+i*k_n*r_j) ]
    exp_pos = np.exp(1j * phases)      # (M, N)
    # n_tilde(k_n) * exp(+i*k_n*r_j): broadcast (M,1) * (M,N) -> (M,N)
    product = n_tilde[:, np.newaxis] * exp_pos  # (M, N)
    # Multiply by k_n and sum over n
    grad = -2.0 * np.sum(k_n[:, np.newaxis] * np.imag(product), axis=0)  # (N,)

    return Phi, grad


def _objective(x, L, M):
    """Objective function wrapper for scipy (returns scalar)."""
    Phi, _ = _objective_and_gradient(x, L, M)
    return Phi


def _objective_grad(x, L, M):
    """Gradient wrapper for scipy."""
    _, grad = _objective_and_gradient(x, L, M)
    return grad


def generate_stealthy(chi, M=50, K=1.0, rng=None, max_restarts=10, tol=1e-20):
    """
    Generate a 1D stealthy hyperuniform point pattern.

    Uses collective-coordinate optimization (L-BFGS-B) to find a
    configuration with S(k) = 0 for all 0 < |k| <= K.

    Parameters
    ----------
    chi : float
        Constrained fraction (0 < chi < 0.5 for disordered).
    M : int
        Number of constrained k-vectors (determines system size).
    K : float
        Stealthiness cutoff wavevector.
    rng : numpy.random.Generator, optional
        Random number generator.
    max_restarts : int
        Maximum number of random restarts to attempt.
    tol : float
        Energy tolerance for declaring convergence.

    Returns
    -------
    points : ndarray
        Sorted particle positions in [0, L).
    L : float
        Box length.
    metadata : dict
        Contains chi, K, N, rho, M, final_energy, n_restarts, converged.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine N and L from chi and M
    # chi = M / (N - 1), so N = M/chi + 1
    N = int(np.round(M / chi)) + 1
    # L = 2*pi*M / K (from k_M = 2*pi*M/L = K)
    L = 2 * np.pi * M / K
    rho = N / L

    best_energy = np.inf
    best_positions = None
    n_restarts = 0

    for restart in range(max_restarts):
        # Random initial positions
        x0 = rng.uniform(0, L, N)

        # Pin first particle at 0 to remove translational degeneracy
        x0[0] = 0.0

        result = minimize(
            _objective,
            x0,
            args=(L, M),
            jac=_objective_grad,
            method="L-BFGS-B",
            options={"maxiter": 20000, "ftol": 1e-30, "gtol": 1e-20},
        )

        energy = result.fun
        n_restarts = restart + 1

        if energy < best_energy:
            best_energy = energy
            best_positions = result.x.copy()

        if energy < tol:
            break

    # Wrap into [0, L) and sort
    points = best_positions % L
    points.sort()

    metadata = {
        "chi": chi,
        "K": K,
        "N": N,
        "rho": rho,
        "M": M,
        "L": L,
        "final_energy": best_energy,
        "n_restarts": n_restarts,
        "converged": best_energy < tol,
    }

    return points, L, metadata


def generate_stealthy_ensemble(chi, n_realizations=20, M=50, K=1.0, rng=None,
                               max_restarts=10, tol=1e-20, verbose=False):
    """
    Generate an ensemble of independent stealthy hyperuniform configurations.

    Parameters
    ----------
    chi : float
        Constrained fraction.
    n_realizations : int
        Number of independent realizations to generate.
    M : int
        Number of constrained k-vectors per realization.
    K : float
        Stealthiness cutoff wavevector.
    rng : numpy.random.Generator, optional
        Random number generator.
    max_restarts : int
        Maximum restarts per realization.
    tol : float
        Energy tolerance.
    verbose : bool
        If True, print progress.

    Returns
    -------
    ensemble : list of (points, L) tuples
        Each element is a converged stealthy configuration.
    """
    if rng is None:
        rng = np.random.default_rng()

    ensemble = []
    n_converged = 0

    for i in range(n_realizations):
        points, L, meta = generate_stealthy(
            chi, M=M, K=K, rng=rng, max_restarts=max_restarts, tol=tol,
        )
        ensemble.append((points, L))

        if meta["converged"]:
            n_converged += 1

        if verbose:
            status = "converged" if meta["converged"] else "NOT converged"
            e_val = meta["final_energy"]
            print(f"  Realization {i+1}/{n_realizations}: "
                  f"energy={e_val:.2e}  ({status})")

    if verbose:
        print(f"  Converged: {n_converged}/{n_realizations}")

    return ensemble


# ============================================================
# Structure factor
# ============================================================

def stealthy_structure_factor(points, L, k_array):
    """
    Compute S(k) via the direct formula (no FFT).

    S(k) = |n_tilde(k)|^2 / N
    where n_tilde(k) = sum_j exp(-i*k*r_j).

    Parameters
    ----------
    points : ndarray
        1D particle positions.
    L : float
        Box length (periodic boundary conditions).
    k_array : ndarray
        Wavevectors at which to evaluate S(k).

    Returns
    -------
    S_k : ndarray
        Structure factor at each wavevector.
    """
    N = len(points)
    # phases: shape (len(k_array), N)
    phases = np.outer(k_array, points)
    n_tilde = np.sum(np.exp(-1j * phases), axis=1)
    S_k = np.abs(n_tilde) ** 2 / N
    return S_k


# ============================================================
# Diagnostics
# ============================================================

def verify_stealthiness(points, L, K, M=None):
    """
    Verify that S(k) ~ 0 for all constrained wavevectors k_n <= K.

    Parameters
    ----------
    points : ndarray
        Particle positions.
    L : float
        Box length.
    K : float
        Stealthiness cutoff.
    M : int, optional
        Number of constrained modes. If None, computed from K and L.

    Returns
    -------
    max_S : float
        Maximum S(k) among constrained wavevectors.
    mean_S : float
        Mean S(k) among constrained wavevectors.
    k_constrained : ndarray
        The constrained wavevectors.
    S_constrained : ndarray
        S(k) at the constrained wavevectors.
    """
    if M is None:
        M = int(np.floor(K * L / (2 * np.pi)))

    k_constrained = 2 * np.pi * np.arange(1, M + 1) / L
    S_constrained = stealthy_structure_factor(points, L, k_constrained)

    return np.max(S_constrained), np.mean(S_constrained), k_constrained, S_constrained


# ============================================================
# Main demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Stealthy Hyperuniform Point Pattern Generation (1D)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    for chi in [0.10, 0.25]:
        print()
        print("-" * 70)
        print(f"chi = {chi:.2f}")
        print("-" * 70)

        M = 50
        K = 1.0

        t0 = time.time()
        points, L, meta = generate_stealthy(
            chi, M=M, K=K, rng=rng, max_restarts=15, tol=1e-15,
        )
        elapsed = time.time() - t0

        n_pts = meta["N"]
        rho_val = meta["rho"]
        m_val = meta["M"]
        e_val = meta["final_energy"]
        conv = meta["converged"]
        nre = meta["n_restarts"]

        print(f"  N = {n_pts},  L = {L:.2f},  rho = {rho_val:.4f}")
        print(f"  M = {m_val} constrained wavevectors")
        print(f"  Final energy:  {e_val:.2e}")
        print(f"  Converged: {conv}  (restarts: {nre})")
        print(f"  Time: {elapsed:.1f}s")

        # Verify stealthiness
        max_S, mean_S, k_con, S_con = verify_stealthiness(points, L, K, M=M)
        print()
        print(f"  Stealthiness verification (k <= K = {K}):")
        print(f"    max  S(k) = {max_S:.2e}")
        print(f"    mean S(k) = {mean_S:.2e}")

        # Show S(k) at first few constrained wavevectors
        print()
        print("  S(k) at first 10 constrained wavevectors:")
        for i in range(min(10, M)):
            print(f"    k = {k_con[i]:.4f}:  S(k) = {S_con[i]:.2e}")

        # Show S(k) just beyond the cutoff
        k_beyond = 2 * np.pi * np.arange(M + 1, M + 6) / L
        S_beyond = stealthy_structure_factor(points, L, k_beyond)
        print()
        print("  S(k) just beyond K (unconstrained, should be nonzero):")
        for i in range(len(k_beyond)):
            print(f"    k = {k_beyond[i]:.4f}:  S(k) = {S_beyond[i]:.4f}")

    print()
    print("=" * 70)
    print("Done.")
