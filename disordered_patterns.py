"""
Phase 5: Disordered Hyperuniform Point Patterns

Implements:
  - Uniformly randomized lattice (URL): j + Uniform(-a/2, a/2)
  - Stealthy hyperuniform: L-BFGS-B minimization of structure factor at
    constrained k-modes (chi = M/N fraction of modes zeroed).

References:
  Klatt et al. (2020) Physical Review Research 2, 013190
  Torquato & Stillinger (2003) Physical Review E 68, 041113
"""

import numpy as np
from scipy.optimize import minimize


# ============================================================
# Uniformly Randomized Lattice (URL)
# ============================================================

def generate_url(N, a, rng=None):
    """
    Generate a uniformly randomized lattice (URL) point pattern.

    Each point is placed at j + U(-a/2, a/2) for j = 0, ..., N-1.
    For a < 1 (no overlap of displacement intervals), the pattern is
    Class I hyperuniform with alpha = 2.

    Parameters
    ----------
    N : int
        Number of points.
    a : float
        Displacement amplitude (uniform in [-a/2, a/2]).
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    points : ndarray
        Sorted 1D point positions.
    L : float
        Domain length (= N, unit density).
    """
    if rng is None:
        rng = np.random.default_rng()
    base = np.arange(N, dtype=np.float64)
    displacements = rng.uniform(-a / 2, a / 2, N)
    points = base + displacements
    # Wrap into [0, N) for periodic boundary conditions
    L = float(N)
    points = points % L
    return np.sort(points), L


def lambda_bar_url_exact(a):
    """
    Exact Lambda_bar for the uniformly randomized lattice (URL).

    From Klatt et al. (2020), eq. 12 (1D case).
    For 0 < a <= 1 (no overlap between adjacent displacement intervals):
        Lambda_bar = (1 + a^2) / 6

    Validation:
      a -> 0: Lambda_bar -> 1/6 (recovers perfect lattice)
      a = 1:  Lambda_bar = 2/6 = 1/3

    Parameters
    ----------
    a : float
        Displacement amplitude.

    Returns
    -------
    lambda_bar : float
        Exact surface-area coefficient.
    """
    a = float(a)
    if a <= 0:
        return 1.0 / 6.0
    if a <= 1.0:
        return (1.0 + a ** 2) / 6.0
    # For a > 1, use general formula from Klatt et al.
    # Λ̄(a) = a/3 + frac(a)^2*(1-frac(a))^2 / (6*a^2)  [general integer part]
    # This handles non-integer a by summing contributions from each period.
    frac_a = a - np.floor(a)
    return a / 3.0 + frac_a ** 2 * (1 - frac_a) ** 2 / (6 * a ** 2)


# ============================================================
# Stealthy hyperuniform point patterns
# ============================================================

def generate_stealthy(N, chi, rng=None, tol=1e-8, max_iter=2000, verbose=False):
    """
    Generate a stealthy hyperuniform point pattern via L-BFGS-B optimization.

    Minimizes E = sum_{n=1}^{M} S(k_n) where k_n = 2*pi*n/L and M = int(chi*N).
    Starting from a slightly perturbed lattice for fast convergence.

    Parameters
    ----------
    N : int
        Number of points.
    chi : float
        Stealthiness fraction (proportion of k-modes constrained to zero).
        chi in (0, 0.5) for typical use.
    rng : numpy.random.Generator, optional
        Random number generator.
    tol : float
        Convergence tolerance: stop when E < tol * N.
    max_iter : int
        Maximum L-BFGS-B iterations.
    verbose : bool
        Print convergence info.

    Returns
    -------
    points : ndarray
        Sorted 1D point positions in [0, L).
    L : float
        Domain length (= N, unit density).
    """
    if rng is None:
        rng = np.random.default_rng()

    L = float(N)
    M = max(1, int(chi * N))

    # Constrained k-modes: k_n = 2*pi*n/L for n = 1, ..., M
    n_modes = np.arange(1, M + 1, dtype=np.float64)
    k_modes = 2 * np.pi * n_modes / L

    # Start from slightly perturbed lattice
    x0 = np.arange(N, dtype=np.float64) + rng.uniform(-0.01, 0.01, N)
    x0 = x0 % L

    def cost_and_grad(x):
        # Structure factor contributions: F[n] = sum_j exp(-i*k_n*x_j)
        # E = (1/N) * sum_n |F[n]|^2
        phases = np.outer(k_modes, x)   # shape (M, N)
        cos_p = np.cos(phases)
        sin_p = np.sin(phases)
        F_re = cos_p.sum(axis=1)         # shape (M,)
        F_im = -sin_p.sum(axis=1)        # shape (M,)
        S_k = (F_re ** 2 + F_im ** 2) / N

        E = S_k.sum()

        # Gradient: dE/dx_j = -(2/N) * sum_n k_n * Im[F_n * exp(i*k_n*x_j)]
        # where Im[F_n * exp(i*k_n*x_j)] = F_re_n*sin(k_n*x_j) + F_im_n*cos(k_n*x_j)
        # (F_n = F_re + i*F_im, so (F_re+iF_im)(cos+isin) -> Im part = F_re*sin + F_im*cos)
        grad = -(2.0 / N) * (k_modes[:, None] * (
            F_re[:, None] * sin_p + F_im[:, None] * cos_p
        )).sum(axis=0)

        return float(E), grad

    if verbose:
        print(f"    Stealthy chi={chi}, N={N}, M={M} modes, tol*N={tol*N:.2e}")

    result = minimize(
        cost_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': tol / N, 'gtol': 1e-10},
    )

    x_opt = result.x % L
    final_E = result.fun

    if verbose:
        converged = final_E < tol * N
        print(f"    Final E={final_E:.3e}, E/N={final_E/N:.3e}, "
              f"{'CONVERGED' if converged else 'not converged'} "
              f"after {result.nit} iterations")

    return np.sort(x_opt), L


def generate_stealthy_ensemble(N, chi, rng, n_realizations=5, **kwargs):
    """
    Generate an ensemble of stealthy configurations for ensemble-averaging Lambda_bar.

    Parameters
    ----------
    N : int
        Number of points per realization.
    chi : float
        Stealthiness fraction.
    rng : numpy.random.Generator
        Random number generator (seeded for reproducibility).
    n_realizations : int
        Number of independent configurations.

    Returns
    -------
    configs : list of (points, L) tuples
    """
    configs = []
    for i in range(n_realizations):
        pts, L = generate_stealthy(N, chi, rng=rng, **kwargs)
        configs.append((pts, L))
    return configs
