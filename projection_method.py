"""
Phase 2b: Cut-and-Project (Projection) Method for 1D Quasicrystals

Generates 1D quasiperiodic point patterns by projecting Z^2 lattice points
through a strip onto a 1D "physical space" line.

For metallic-mean quasicrystals:
  - Physical-space line slope: 1/mu_n where mu_n is the metallic mean
  - Ideal strip width: omega = mu_n  (gives Class I hyperuniformity)
  - Non-ideal strip width: omega != mu_n  (degrades to Class II)

Supports Fibonacci (golden), silver ratio, and bronze ratio chains.
"""

import numpy as np
from substitution_tilings import CHAINS


def cut_and_project(chain_name, N_target, omega=None, rng=None):
    """
    Generate a 1D quasiperiodic point pattern via the cut-and-project method.

    Projects Z^2 lattice points within a strip onto a 1D physical-space line.
    The physical line has slope 1/mu (where mu is the metallic mean) and
    the perpendicular direction defines the "internal space".

    Parameters
    ----------
    chain_name : str
        One of 'fibonacci', 'silver', 'bronze'.
    N_target : int
        Approximate number of points desired.
    omega : float or None
        Strip half-width in internal space. If None, uses ideal width mu
        (which gives Class I hyperuniformity).
    rng : numpy.random.Generator, optional
        Not used (deterministic), kept for API consistency.

    Returns
    -------
    points : ndarray
        Sorted 1D point positions in physical space.
    L_domain : float
        Total domain length (max - min point position).
    info : dict
        Metadata: omega used, number of lattice points scanned, etc.
    """
    mu = CHAINS[chain_name]['metallic_mean']

    if omega is None:
        omega = mu  # Ideal width for Class I

    # Physical-space direction: e_par = (mu, 1) / ||(mu, 1)||
    # Internal-space direction: e_perp = (-1, mu) / ||(-1, mu)||
    norm = np.sqrt(mu**2 + 1)
    # For a lattice point (m, n), its projections are:
    #   x_parallel = (m * mu + n) / norm    (physical space coordinate)
    #   x_perp     = (-m + n * mu) / norm   (internal space coordinate)
    # A point is selected if |x_perp| < omega / (2 * norm)
    # Equivalently: | -m + n*mu | < omega / 2

    # Each row n contributes ~omega selected lattice points (the strip selects
    # about omega integers m per row). Total points ~ omega * (2*grid_half).
    # So grid_half ~ N_target / (2*omega), with 10% margin.
    grid_half = int(N_target / (2 * omega) * 1.1) + 50

    # Process row by row to keep memory manageable
    phys_coords = []
    n_scanned = 0

    for n in range(-grid_half, grid_half + 1):
        # For row n, find m values where | -m + n*mu | < omega/2
        # i.e., n*mu - omega/2 < m < n*mu + omega/2
        m_center = n * mu
        m_lo = int(np.floor(m_center - omega / 2))
        m_hi = int(np.ceil(m_center + omega / 2))

        m_vals = np.arange(m_lo, m_hi + 1)
        n_scanned += len(m_vals)

        # Filter: | -m + n*mu | < omega/2
        x_perp_unnorm = -m_vals + n * mu
        mask = np.abs(x_perp_unnorm) < omega / 2

        if np.any(mask):
            m_sel = m_vals[mask]
            # Physical space coordinate (unnormalized is fine, just a linear scaling)
            x_par = (m_sel * mu + n) / norm
            phys_coords.append(x_par)

    if not phys_coords:
        raise ValueError(f"No points found. Try increasing N_target or omega.")

    points = np.concatenate(phys_coords)
    points = np.sort(points)

    L_domain = points[-1] - points[0]

    # Shift so points start at 0
    points = points - points[0]

    info = {
        'omega': omega,
        'ideal_omega': mu,
        'mu': mu,
        'grid_half': grid_half,
        'n_scanned': n_scanned,
        'is_ideal': np.isclose(omega, mu),
    }

    return points, L_domain, info


def validate_projection_spacings(points, chain_name, tol=1e-6):
    """
    Validate that the projected pattern has exactly 2 distinct spacings.

    For Z^2 projection with slope 1/mu, the spacing ratio is (mu+1)/mu,
    which equals mu only for the Fibonacci case (where tau = 1 + 1/tau).

    Returns
    -------
    spacings_unique : ndarray
        Unique spacings found (rounded to tol precision).
    ratio : float
        Ratio of largest to smallest spacing.
    expected_ratio : float
        Expected ratio: (mu+1)/mu for Z^2 projection.
    """
    mu = CHAINS[chain_name]['metallic_mean']
    expected_ratio = (mu + 1) / mu
    spacings = np.diff(points)
    spacings_unique = np.unique(np.round(spacings, int(-np.log10(tol))))
    if len(spacings_unique) >= 2:
        ratio = spacings_unique[-1] / spacings_unique[0]
    else:
        ratio = 1.0
    return spacings_unique, ratio, expected_ratio


# ============================================================
# Main: demonstrate projection method and Class I vs Class II
# ============================================================

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

    rng = np.random.default_rng(seed=456)

    print("=" * 70)
    print("  Cut-and-Project Method for 1D Quasicrystals")
    print("=" * 70)

    # ----------------------------------------------------------
    # Part 1: Generate all three chains with ideal strip width
    # ----------------------------------------------------------
    print("\n--- Ideal Strip Width (Class I) ---")
    N_target = 200_000

    for name in ['fibonacci', 'silver', 'bronze']:
        mu = CHAINS[name]['metallic_mean']
        print(f"\n  {CHAINS[name]['name']}:")

        t0 = time.perf_counter()
        points, L_domain, info = cut_and_project(name, N_target)
        elapsed = time.perf_counter() - t0

        rho = len(points) / L_domain
        print(f"    N = {len(points):,}, L = {L_domain:.1f}, "
              f"rho = {rho:.6f}, time = {elapsed:.2f}s")
        print(f"    omega = {info['omega']:.6f} (ideal = {info['ideal_omega']:.6f})")

        # Validate spacings
        unique_sp, ratio, expected = validate_projection_spacings(points, name)
        print(f"    Unique spacings: {unique_sp}")
        print(f"    Spacing ratio: {ratio:.6f} (expected mu = {expected:.6f}, "
              f"err = {abs(ratio - expected):.2e})")

    # ----------------------------------------------------------
    # Part 2: Fibonacci Class I vs Class II comparison
    # ----------------------------------------------------------
    print("\n\n--- Class I vs Class II Comparison (Fibonacci) ---")
    mu_fib = CHAINS['fibonacci']['metallic_mean']

    # Class I: ideal width
    print("\n  Generating Class I (omega = tau)...")
    pts_I, L_I, info_I = cut_and_project('fibonacci', N_target, omega=mu_fib)
    rho_I = len(pts_I) / L_I
    print(f"    N = {len(pts_I):,}, L = {L_I:.1f}, rho = {rho_I:.6f}")

    # Class II: non-ideal width
    omega_II = 0.9 * mu_fib
    print(f"\n  Generating Class II (omega = 0.9*tau = {omega_II:.6f})...")
    pts_II, L_II, info_II = cut_and_project('fibonacci', N_target, omega=omega_II)
    rho_II = len(pts_II) / L_II
    print(f"    N = {len(pts_II):,}, L = {L_II:.1f}, rho = {rho_II:.6f}")

    # Compute variance for both
    mean_sp_I = 1.0 / rho_I
    R_max_I = min(200 * mean_sp_I, L_I / 4)
    R_I = np.linspace(mean_sp_I, R_max_I, 800)

    mean_sp_II = 1.0 / rho_II
    R_max_II = min(200 * mean_sp_II, L_II / 4)
    R_II = np.linspace(mean_sp_II, R_max_II, 800)

    print("\n  Computing variance (Class I, non-periodic BCs)...")
    t0 = time.perf_counter()
    var_I, _ = compute_number_variance_1d(
        pts_I, L_I, R_I, num_windows=20000, rng=rng, periodic=False)
    print(f"    Done in {time.perf_counter() - t0:.1f}s")

    print("  Computing variance (Class II, non-periodic BCs)...")
    t0 = time.perf_counter()
    var_II, _ = compute_number_variance_1d(
        pts_II, L_II, R_II, num_windows=20000, rng=rng, periodic=False)
    print(f"    Done in {time.perf_counter() - t0:.1f}s")

    lambda_bar_I = compute_lambda_bar(R_I, var_I)
    print(f"\n  Class I  Lambda_bar = {lambda_bar_I:.6f}")
    print(f"  Class II variance at max R = {var_II[-1]:.4f} "
          f"(should grow with R if Class II)")

    # Check if Class II variance envelope grows
    # Compare first third average to last third average
    third = len(var_II) // 3
    avg_early = np.mean(var_II[:third])
    avg_late = np.mean(var_II[-third:])
    print(f"  Class II early avg = {avg_early:.4f}, late avg = {avg_late:.4f}")
    if avg_late > avg_early * 1.2:
        print("  --> Variance envelope GROWING (Class II confirmed)")
    else:
        print("  --> Warning: growth not clearly detected; try larger N or wider R range")

    # ----------------------------------------------------------
    # Plot: Class I vs Class II
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(R_I, var_I, 'b-', lw=0.5, alpha=0.8)
    ax1.axhline(lambda_bar_I, color='red', ls='--', lw=1.5,
                label=rf'$\bar{{\Lambda}} = {lambda_bar_I:.4f}$')
    ax1.set_xlabel(r'$R$')
    ax1.set_ylabel(r'$\sigma^2(R)$')
    ax1.set_title(rf'Class I: Ideal width $\omega = \tau$ (N={len(pts_I):,})')
    ax1.legend()
    ax1.grid(True, ls=':', alpha=0.5)

    ax2 = axes[1]
    ax2.plot(R_II, var_II, 'r-', lw=0.5, alpha=0.8)
    # Overlay logarithmic fit
    from scipy.optimize import curve_fit
    def log_model(R, C, b):
        return C * np.log(R) + b
    try:
        popt, _ = curve_fit(log_model, R_II, var_II, p0=[0.1, 0])
        R_fit = np.linspace(R_II[0], R_II[-1], 500)
        ax2.plot(R_fit, log_model(R_fit, *popt), 'k--', lw=1.5,
                 label=rf'$C \ln R + b$ fit (C={popt[0]:.4f})')
    except Exception:
        pass
    ax2.set_xlabel(r'$R$')
    ax2.set_ylabel(r'$\sigma^2(R)$')
    ax2.set_title(rf'Class II: Non-ideal $\omega = 0.9\tau$ (N={len(pts_II):,})')
    ax2.legend()
    ax2.grid(True, ls=':', alpha=0.5)

    plt.suptitle('Projection Method: Class I vs Class II Hyperuniformity',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('projection_class_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to projection_class_comparison.png")
    plt.show()
