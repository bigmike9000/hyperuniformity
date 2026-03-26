# Context for Analytical Derivation of Lambda-bar for Metallic-Mean Chains

## Goal

We want to derive analytically why the Silver-mean quasicrystal chain has Lambda-bar = 1/4 exactly, and potentially explain the Fibonacci value Lambda-bar = 0.20110. This is a pure math derivation task.

---

## What is Lambda-bar?

For a 1D point pattern with density rho, the **number variance** sigma^2(R) counts how much the number of points in a random window of half-width R fluctuates around its mean (2*rho*R).

For **Class I hyperuniform** systems, sigma^2(R) is bounded as R -> infinity. It oscillates quasi-periodically. **Lambda-bar** is the long-R running average of this bounded variance:

    Lambda-bar = lim_{R->inf} (1/R) * integral_0^R sigma^2(R') dR'

Lambda-bar is a universal "surface-area coefficient" that ranks how ordered a Class I system is. Lower = more ordered:

- **Integer lattice** (perfectly periodic): Lambda-bar = 1/6 (global minimum for any 1D point process)
- **Fibonacci chain** (n=1 metallic mean): Lambda-bar = 0.20110 (Zachary & Torquato 2009)
- **Silver chain** (n=2): Lambda-bar = 1/4 exactly (our numerical finding, confirmed to 0.2501 +/- 0.0004)
- **Bronze** (n=3): 0.282, **Copper** (n=4): 0.293, **Nickel** (n=5): 0.310
- n=6 through n=20: monotonically increasing toward 1/3
- **Limit as n -> infinity: Lambda-bar -> 1/3** (confirmed numerically, matches the "cloaked URL" exactly)

---

## The Metallic-Mean Substitution Chains

The n-th metallic-mean chain is defined by:

- **Substitution rules:** S -> L, L -> L^n S (i.e., L maps to n copies of L followed by one S)
- **Substitution matrix:** M_n = [[0, 1], [1, n]]
- **Eigenvalues:**
  - lambda_1 = (n + sqrt(n^2 + 4)) / 2 (the n-th metallic mean, a Pisot number)
  - lambda_2 = (n - sqrt(n^2 + 4)) / 2 = -1/lambda_1 (since det M = -1)
- **Tile lengths:** S has length 1, L has length lambda_1
- **Tile frequencies:**
  - f_S = 1/(1 + lambda_1)
  - f_L = lambda_1/(1 + lambda_1)
- **Mean spacing:** ell-bar = (2 + n*lambda_1) / (1 + lambda_1)
- **Density:** rho = 1/ell-bar

### Key special cases

| n | Name | lambda_1 | rho | Lambda-bar |
|---|------|----------|-----|------------|
| 1 | Fibonacci | (1+sqrt(5))/2 = 1.618... | 0.7236 | 0.20110 |
| 2 | Silver | 1+sqrt(2) = 2.414... | 1/2 exactly | **1/4 exactly** |
| 3 | Bronze | (3+sqrt(13))/2 = 3.303... | 0.3613 | 0.282 |

**Note:** Silver is special because rho = 1/2 exactly (mean spacing = 2). This happens because ell-bar = (2 + 2*lambda_1)/(1 + lambda_1) = 2(1 + lambda_1)/(1 + lambda_1) = 2. This exact simplification may be key to the derivation.

### Hyperuniformity exponent

All metallic-mean chains have alpha = 3 (Class I), by the Oguz et al. eigenvalue formula:
    alpha = 1 - 2*ln|lambda_2| / ln|lambda_1|
Since |lambda_2| = 1/lambda_1 (unimodular determinant), this gives alpha = 1 + 2 = 3.

---

## Bragg Peak / Structure Factor Approach to Lambda-bar

For a system with a pure Bragg (discrete) diffraction spectrum, Lambda-bar can be computed from the Bragg peak intensities:

    Lambda-bar = (2*rho / pi) * sum_{G > 0} I_G / G^2

where:

- G are the Bragg peak positions (wavevectors)
- I_G are the peak intensities (as appearing in the structure factor S(k) = sum_G I_G * delta(k - G))

This follows from plugging the discrete S(k) into the variance formula:
    sigma^2(R) = (rho / pi) *integral S(k)* (2*sin(kR)/k)^2 dk

and then taking the running average over R.

### Bragg peak positions for metallic-mean chains

The peaks lie at wavevectors:
    G_{p,q} = 2*pi * (p + q*lambda_1) / ell-bar,  for (p,q) in Z^2, not both zero

### Bragg peak intensities (cut-and-project)

The intensities involve the Fourier transform of the acceptance window in the "perpendicular space" of the cut-and-project construction:

    I_{p,q} ~ |sinc(q_perp * W / 2)|^2

where q_perp is the perpendicular-space component of G and W is the window width. The perpendicular tile lengths are **swapped** relative to physical lengths (ell_S^perp = lambda_1, ell_L^perp = |lambda_2|).

This is where the derivation gets involved. We attempted a numerical Bragg summation (compute_lambda_bar_analytical.py) but encountered normalization issues that gave ~0.017 instead of 0.201 for Fibonacci. The formula structure is correct but getting the prefactors right requires care.

---

## The Zachary-Torquato Theta-Series Method

Zachary & Torquato (2009) computed Lambda-bar = 0.20110 for Fibonacci using a theta-function approach (their eq. 73):

    B_N = lim_{beta -> 0+} [ phi / (2*beta) - (1/2) * sum_k Z_k * r_k * exp(-beta * r_k^2) ]

where:

- phi is the packing fraction
- r_k are pair distances
- Z_k are coordination numbers
- beta is the "inverse temperature" regularization parameter

Then Lambda-bar = 2*phi*B_N (in d=1). They used N ~ 10^6 points and careful beta -> 0 extrapolation.

---

## What We Know / Conjectures

1. **Silver Lambda-bar = 1/4 is numerically confirmed** to high precision (N=3.88M points, 30k windows, 1500 R values; result 0.2501 +/- 0.0004, within 0.3 sigma of 1/4).

2. **The rho = 1/2 simplification is unique to Silver.** For the n-th metallic mean, rho = (1 + lambda_1) / (2 + n*lambda_1). Setting this equal to 1/2 gives 2(1+lambda_1) = 2 + n*lambda_1, i.e., lambda_1*(2-n) = 0, so n=2 is the unique solution. This exact density may enable closed-form evaluation of the Bragg sum.

3. **Fibonacci Lambda-bar = 0.20110 does NOT equal 1/5.** It's close but clearly distinct (0.20110 vs 0.20000). Whether it has a closed form in terms of the golden ratio is unknown.

4. **Limit Lambda-bar(n) -> 1/3 as n -> infinity.** The convergence rate is approximately 0.22/n^1.43. The value 1/3 matches the "cloaked" uncorrelated random lattice (URL with parameter a=1, which has Lambda-bar = (1+a^2)/6 = 1/3). The URL formula is from Klatt et al. (2020).

5. **Integer lattice Lambda-bar = 1/6 can be derived exactly.** The variance sigma^2(R) = 2*frac(R)*(1 - 2*frac(R)) for frac(R) < 1/2, which has period average exactly 1/6. This is the simplest reference derivation.

---

## Promising Derivation Strategies

1. **Exploit rho = 1/2 in the Bragg sum formula.** With rho = 1/2 and ell-bar = 2, the Bragg peak positions simplify to G_{p,q} = pi*(p + q*(1+sqrt(2))). The prefactor becomes (2*rho/pi) = 1/pi. So Lambda-bar = (1/pi)* sum_{G>0} I_G/G^2. Can this sum be evaluated in closed form?

2. **Lattice theta-function approach.** The cut-and-project lattice for Silver is Z^2 with slope 1/(1+sqrt(2)). The perpendicular-space window has width related to |lambda_2| = sqrt(2)-1. The theta function of this lattice restricted to the acceptance strip might simplify at n=2.

3. **Direct integration of sigma^2(R) for Silver.** If sigma^2(R) has a simpler quasi-periodic structure when rho = 1/2, its average might be computable by summing a geometric series.

4. **Relate to known lattice results.** Lambda-bar = 1/6 for the integer lattice. Lambda-bar = 1/4 for Silver = (1/6) * (3/2). Is there a structural explanation for this 3/2 factor?

---

## Key References

- **Torquato & Stillinger (2003):** Original definition of hyperuniformity; Lambda-bar = 1/6 for the integer lattice.
- **Zachary & Torquato (2009) J. Stat. Mech. P12015:** Theta-function method; Fibonacci Lambda-bar = 0.20110 (Table 1). The key paper for computational method.
- **Oguz et al. (2019):** Eigenvalue formula alpha = 1 - 2*ln|lambda_2|/ln|lambda_1| for substitution tilings.
- **Bombieri & Taylor (1986):** Cubic substitution; structure factor formulas for substitution chains.
- **Klatt et al. (2020):** URL formula Lambda-bar = (1+a^2)/6; cloaking at a=1 gives Lambda-bar = 1/3.
- **Torquato (2018) Physics Reports:** Comprehensive hyperuniformity review; three-class hierarchy.
- **Senechal, "Quasicrystals and Geometry":** Cut-and-project construction; Fourier analysis of quasicrystals.

---

## Summary of the Challenge

The core mathematical question: **evaluate the infinite Bragg sum sum_{G>0} I_G/G^2 in closed form for the Silver-mean tiling** (and ideally for the Fibonacci tiling too). The Bragg peaks form a dense set on the real line (indexed by Z^2 via the cut-and-project construction), and their intensities involve sinc^2 functions of irrational arguments. The difficulty is handling this doubly-indexed sum over a quasiperiodic lattice. The Silver case may be tractable because rho = 1/2 exactly and the silver mean 1+sqrt(2) has particularly nice algebraic properties.
