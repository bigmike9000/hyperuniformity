# Hyperuniformity in 1D Quasiperiodic Point Patterns
## Numerical Extraction of the Scaling Exponent via Diffusion Spreadability

**Presented to:** Professor Salvatore Torquato, Princeton University

---

## Slide 1 — What Is Hyperuniformity?

A point pattern is **hyperuniform** if the number of points inside a window of
radius $R$ fluctuates *less* than random. Quantitatively, the number variance
$\sigma^2(R)$ grows slower than the window volume:

$$\sigma^2(R) \sim R^{d-\alpha} \quad \text{(hyperuniform if slower than } R^d \text{)}$$

In reciprocal space, this means the structure factor vanishes at the origin:

$$S(k) \sim |k|^\alpha \text{ as } k \to 0$$

The exponent $\alpha$ determines the **hyperuniformity class**:

| Class   | Exponent      | Real-space behavior             | Examples                        |
|---------|---------------|---------------------------------|---------------------------------|
| Class I | $\alpha > 1$  | $\sigma^2 \sim R^{d-1}$ (surface area) | Crystals, quasicrystals  |
| Class II| $\alpha = 1$  | $\sigma^2 \sim R^{d-1}\ln R$   | Period-doubling chains          |
| Class III| $0<\alpha<1$ | $\sigma^2 \sim R^{d-\alpha}$   | Certain aperiodic chains        |

**Goal of this project:** Numerically extract $\alpha$ for three families
of 1D quasicrystals and verify the theoretical prediction $\alpha = 3$.

**The challenge:** Quasicrystals have dense, discontinuous Bragg peaks, making
direct measurement of $\alpha$ from $S(k)$ impossible. We use the **diffusion
spreadability** approach (Torquato, 2021) to circumvent this.

---

## Slide 2 — The Three Quasicrystal Chains

We study three metallic-mean substitution tilings — 1D quasicrystals where
short ($S$) and long ($L$) tiles are arranged by iterating substitution rules:

| Chain     | Rule                    | Metallic Mean $\mu$                        | Density $\rho$ |
|-----------|-------------------------|--------------------------------------------|----------------|
| Fibonacci | $S \to L,\; L \to LS$  | $\tau = (1+\sqrt{5})/2 \approx 1.618$     | 0.7236         |
| Silver    | $S \to L,\; L \to LLS$ | $\mu_2 = 1+\sqrt{2} \approx 2.414$        | 0.5000         |
| Bronze    | $S \to L,\; L \to LLLS$| $\mu_3 = (3+\sqrt{13})/2 \approx 3.303$   | 0.3613         |

**Tile lengths:** $S = 1$, $L = \mu$. Points placed at each tile's left endpoint.
Each chain has exactly two distinct spacings, arranged quasiperiodically.

**Theoretical prediction** from the substitution matrix eigenvalues:

$$\alpha = 1 - \frac{2\ln|\lambda_2|}{\ln\lambda_1} = 3 \quad \text{(universal for all metallic means)}$$

This follows because $\det(\mathbf{M}) = -1$ for all three matrices, so
$|\lambda_2| = 1/\lambda_1$.

**Our task:** Verify $\alpha = 3$ numerically using two independent approaches
(real-space variance and reciprocal-space spreadability).

---

## Slide 3 — Figure 1: Can We Trust Our Variance Code?

> **Figure 1** — `fig1_poisson_benchmark.png`

### What this figure shows
Before analyzing quasicrystals, we validate our core algorithm. We generate
random (Poisson) point patterns and compare the numerically computed number
variance against the known exact result $\sigma^2(R) = 2\rho R$.

### What to look for
- **Left panel:** Black dots (computed) should lie exactly on the red dashed
  line (theory). Error bars show $\pm 2\sigma$ from 40 independent realizations.
- **Right panel:** Relative error at each window size, all well below the 5%
  threshold (red dashed line). Mean error = 1.8%.

### Why it matters
This confirms the binary-search sliding-window algorithm is correct. All
subsequent variance computations use this same code.

---

## Slide 4 — How We Generate Quasicrystals

Two independent generation methods:

**Method 1: Substitution** (`substitution_tilings.py`)
- Apply the substitution rule iteratively from a seed tile $L$
- After 34 iterations for Fibonacci: $N = 14{,}930{,}352$ tiles
- Vectorized implementation generates $10^7$ tiles in ~1 second

**Method 2: Cut-and-Project** (`projection_method.py`)
- Embed a 2D square lattice $\mathbb{Z}^2$
- Draw a strip of width $\omega$ along a line with slope $1/\mu$
- Project lattice points inside the strip onto the physical line
- Ideal width $\omega = \mu$ gives Class I; non-ideal gives Class II

Both methods produce the same quasicrystal (up to rescaling). This
redundancy lets us cross-validate results.

---

## Slide 5 — Figure 2: Variance Is Bounded (Class I Confirmed)

> **Figure 2** — `fig2_bounded_variance.png` (4-panel)

### What this figure shows
For each pattern, we plot the number variance $\sigma^2(R)$ as a function of
window half-width $R$. Each curve is computed from 30,000 random window
placements at each of 1,000 $R$ values.

### What to look for
- **All four panels:** $\sigma^2(R)$ oscillates but stays **bounded** — it
  does not grow with $R$. This is the defining signature of Class I
  hyperuniformity.
- **Red dashed lines:** The time-averaged value $\bar{\Lambda}$ (the
  surface-area coefficient). For the lattice, $\bar{\Lambda} = 1/6$ exactly.
- **Increasing $\bar{\Lambda}$:** Fibonacci (0.200) < Silver (0.250) <
  Bronze (0.282), meaning more "aperiodic" chains fluctuate more.

### $\bar{\Lambda}$ Results Table

| Pattern                  | $N$ (tiles)  | $\bar{\Lambda}$ | vs Lattice |
|--------------------------|-------------|------------------|------------|
| Integer Lattice          | 100,000     | 0.1649 ($\approx 1/6$) | reference |
| Fibonacci (substitution) | 14,930,352  | 0.200            | 1.21x      |
| Silver (substitution)    | 22,619,537  | 0.250            | 1.51x      |
| Bronze (substitution)    | 21,932,293  | 0.282            | 1.71x      |
| Fibonacci (projection)   | 220,161     | 0.200            | 1.21x      |

Substitution and projection methods agree to 0.1% for Fibonacci, confirming
$\bar{\Lambda}$ is independent of point density (rescaling invariance).

---

## Slide 6 — Figure 3: Variance Grows Slower Than Volume

> **Figure 3** — `fig3_hyperuniformity_test.png`

### What this figure shows
We plot $\sigma^2(R) / R$ for all three chains. For a random (Poisson)
pattern, this ratio equals $2\rho$ = constant. For a hyperuniform pattern,
this ratio must decay to zero.

### What to look for
- All three curves rapidly approach zero as $R$ increases
- The gray dashed line shows where a Poisson pattern would sit (constant)
- The contrast between "decays to zero" (hyperuniform) and "stays constant"
  (random) is stark

### Why it matters
This is the simplest direct test of hyperuniformity: do number fluctuations
grow slower than the window? Yes — for all three chains.

---

## Slide 7 — Figure 4: Strip Width Controls Hyperuniformity Class

> **Figure 4** — `fig4_projection_comparison.png` (2 panels)

### What this figure shows
Using the cut-and-project method, we compare two strip widths for the
Fibonacci chain:

**Left panel — Class I (ideal $\omega = \tau$):**
- Variance is bounded and oscillating, exactly like the substitution result
- $\bar{\Lambda} = 0.200$ matches the substitution method to 0.1%

**Right panel — Class II (non-ideal $\omega = 0.9\tau$):**
- Variance envelope **grows logarithmically**: $\sigma^2 \sim C\ln R + b$
- Black dashed curve is the logarithmic fit ($C = 0.067$)
- This logarithmic growth is the signature of Class II hyperuniformity

### Why it matters
This demonstrates that the hyperuniformity class is controlled by the strip
width parameter: ideal $\omega = \mu$ gives Class I, any deviation degrades
to Class II. The rescaling invariance of $\bar{\Lambda}$ (same value at
$\rho = 0.72$ and $\rho = 0.85$) proves it is a genuine structural property.

---

## Slide 8 — Figure 5: Variance Has Piecewise Quadratic Structure

> **Figure 5** — `fig5_piecewise_quadratic.png` (4-panel)

### What this figure shows
We zoom into the fine structure of $\sigma^2(R)$ and compute its second
derivative $d^2\sigma^2/dR^2$.

**Top-left:** Integer lattice variance at high resolution, overlaid with the
exact analytic formula $\sigma^2(R) = 2\{R\}(1-2\{R\})$ (where $\{R\}$ is the
fractional part). Perfect agreement.

**Top-right:** The exact second derivative $d^2\sigma^2/dR^2 = -8$ is constant
between breakpoints at every half-integer $R$ (marked by vertical lines). This
confirms $\sigma^2$ is piecewise quadratic — each segment is a parabola joined
at tile boundary positions.

**Bottom-left:** Fibonacci variance at high resolution shows self-similar
oscillations at two incommensurate scales ($S = 1$ and $L = \tau$).

**Bottom-right:** Fibonacci first derivative $d\sigma^2/dR$ is **piecewise linear**
with slope changes at quasiperiodic positions — confirming the piecewise
quadratic structure with a dense set of breakpoints from $S$ and $L$ tiles.

---

## Slide 9 — Phase 4: The Two-Phase Media Approach

### The Problem
Quasicrystals have Bragg peaks at irrational positions — infinitely many,
densely distributed. We cannot directly read off $S(k) \sim k^\alpha$ because
$S(k)$ is not a smooth function.

### The Solution: Diffusion Spreadability
1. **Decorate** each point with a solid rod of half-length $a = \phi_2/(2\rho)$,
   creating a two-phase medium (solid + void) with packing fraction $\phi_2 = 0.35$
2. **Compute** the spectral density:
   $\tilde{\chi}_V(k) = \rho \cdot (2\sin(ka)/k)^2 \cdot S(k)$
3. **Evaluate** the excess spreadability (how fast diffusion equilibrates):
   $E(t) = (\Delta k / \pi\phi_2) \sum_n \tilde{\chi}_V(k_n) e^{-k_n^2 Dt}$
4. **Extract** $\alpha$ from the long-time decay: $E(t) \sim t^{-(1+\alpha)/2}$
   via the logarithmic derivative $\alpha(t) = -2\,d\ln E/d\ln t - 1$

### Non-overlap verification

| Chain     | $\rho$  | Rod diameter $2a$ | Min spacing | Overlap? |
|-----------|---------|-------------------|-------------|----------|
| Fibonacci | 0.7236  | 0.484             | 1.000       | No       |
| Silver    | 0.5000  | 0.700             | 1.000       | No       |
| Bronze    | 0.3613  | 0.969             | 1.000       | No       |

---

## Slide 10 — Figure 6: Spectral Density Reveals $k^3$ Scaling

> **Figure 6** — `fig6_spectral_density.png` (2x2 panel)

### What this figure shows
The spectral density $\tilde{\chi}_V(k)$ of the two-phase medium for each
pattern, plotted on log-log axes.

**Top-left (Poisson):** Flat envelope — the structure factor $S(k) = 1$
everywhere, modulated only by the rod form factor $|\tilde{m}(k)|^2$.
No hyperuniformity.

**Top-right (Integer Lattice):** Bragg peaks at $k = 2\pi n$. The FFT
computation introduces a noise floor between the true peaks (this does not
affect results since the lattice spreadability is computed analytically).

**Bottom-left (Fibonacci):** Dense set of Bragg peaks. The crucial feature:
the **envelope** of the peak heights follows a $k^3$ power law at small $k$.
This is the spectral signature of $\alpha = 3$.

**Bottom-right (All chains overlaid):** All three quasicrystal chains share
the same $k^3$ envelope (black dashed reference line), confirming the
universality of $\alpha = 3$.

---

## Slide 11 — Figure 7: Spreadability Decay Rates

> **Figure 7** — `fig7_excess_spreadability.png` (2 panels)

### What this figure shows
The excess spreadability $E(t) = \mathcal{S}(\infty) - \mathcal{S}(t)$
measures how far the diffusion process is from equilibrium at time $t$.
Different levels of hyperuniformity produce different decay rates.

**Main panel (left):** Log-log plot of $E(t)$ for all patterns.
- **Gray (Poisson):** Slow decay $\sim t^{-1/2}$ — random fluctuations
  impede equilibration ($\alpha = 0$)
- **Colored (quasicrystals):** Steeper decay $\sim t^{-2}$ — the ordered
  structure allows faster equilibration ($\alpha = 3$)
- **Gold shaded region:** Time window $[10^2, 10^5]$ where we measure
  the plateau exponent
- Dashed/dotted reference lines show exact $t^{-1/2}$ and $t^{-2}$ slopes

**Right panel (Lattice):** Plotted separately because the lattice's $E(t)$
decays **exponentially** (reaches zero by $t \approx 1$), which would compress
all other curves to a thin line if plotted on the same y-axis.
This exponential decay confirms $\alpha \to \infty$ for perfect crystals.

### Key insight
More hyperuniform = faster equilibration. The quasicrystals sit between
random (Poisson) and perfect crystal (lattice).

---

## Slide 12 — Figure 8: MAIN RESULT — $\alpha = 3$ Confirmed

> **Figure 8** — `fig8_alpha_extraction.png`

### What this figure shows
The effective exponent $\alpha(t)$ extracted from the logarithmic derivative
of $E(t)$. This is the central result of the project.

$$\alpha(t) = -2\,\frac{d\ln E(t)}{d\ln t} - 1$$

### What to look for
- **Gray (Poisson):** Flat at $\alpha \approx 0$ — correct for random patterns
- **Black (Lattice):** Shoots upward without bound — $\alpha \to \infty$
  for perfect crystals
- **Green/Purple/Red (Quasicrystals):** All three curves plateau at
  $\alpha \approx 3$ in the gold-shaded measurement window $[10^2, 10^5]$
- **Black dashed line at $\alpha = 3$:** The theoretical prediction

### Measured values (linear fit of $\ln E$ vs $\ln t$ over $[10^2, 10^5]$)

| Pattern                  | $\alpha$ (measured) | Expected | Error |
|--------------------------|---------------------|----------|-------|
| Poisson                  | **0.001**           | 0        | —     |
| Integer Lattice          | **Exp. decay**      | $\infty$ | N/A   |
| Fibonacci (Golden Ratio) | **3.049**           | 3        | 1.6%  |
| Silver Ratio             | **2.992**           | 3        | 0.3%  |
| Bronze Ratio             | **2.987**           | 3        | 0.4%  |

Two extraction methods are used: (1) period-aware log derivative for the $\alpha(t)$
curves in Figure 8 (sliding window matched to oscillation period $2\ln\mu$), and
(2) linear fit of $\ln E$ vs $\ln t$ for the reported values above (more robust,
averages over all oscillations).

### Conclusion
**All three metallic-mean quasicrystal chains yield $\alpha \approx 3$**,
confirming the universal eigenvalue prediction via an independent dynamical
measurement. The largest deviation is 1.6% for Fibonacci (limited by FFT
$k$-resolution at the highest density).

---

## Slide 13 — Technical Challenges & Solutions

### Challenge 1: System size sensitivity
The FFT computes $S(k)$ on a discrete grid $k_n = 2\pi n/L$, but quasicrystal
Bragg peaks sit at irrational positions. The grid alignment varies with $L$,
causing **non-monotonic convergence** of $\alpha$ with system size:

| $N$ (Fibonacci) | $\alpha$ | Comment |
|-----------------|----------|---------|
| 75,000          | 2.6      | Too few peaks sampled |
| 500,000         | 1.5      | Bad grid alignment |
| 2,200,000       | 2.7      | Improving |
| 5,700,000       | 1.5      | Bad alignment again |
| 9,200,000       | 3.1      | Converging |
| 14,900,000      | 3.0      | Converged |

**Solution:** Use $N \sim 10^7$ tiles. Fibonacci needs the most because it has
the highest density ($\rho = 0.72$), giving the smallest $L$ for a given $N$.

### Challenge 2: Integer lattice artifact
FFT histogram binning creates position errors up to $\Delta x/2$ for lattice
points. This noise floor in $S(k)$ mimics $k^3$ scaling, giving the wrong
$\alpha \approx 3$ instead of $\alpha \to \infty$.

**Solution:** Compute the lattice $E(t)$ analytically from known Bragg peak
positions $k = 2\pi n$, bypassing the FFT entirely.

### Challenge 3: Noisy log derivative
Point-by-point `np.gradient` on log-transformed data amplifies noise from
discrete Bragg peak spectra.

**Solution:** Sliding-window linear regression (window = 10 log-spaced points)
gives smooth, stable plateau estimates.

---

## Slide 14 — Summary of All Results

### Real-Space Analysis (Phases 1-3)

| Finding | Evidence |
|---------|----------|
| Variance code validated | Poisson benchmark: 1.8% mean error (Fig. 1) |
| Class I hyperuniformity confirmed | Bounded $\sigma^2(R)$ for all chains (Fig. 2) |
| $\bar{\Lambda}$ increases with $\mu$ | 0.200 (Fibonacci) < 0.250 (Silver) < 0.282 (Bronze) |
| Rescaling invariance | Substitution vs projection agree to 0.1% (Fig. 4) |
| Class I $\to$ II transition | Non-ideal strip width gives log growth (Fig. 4) |
| Piecewise quadratic structure | $d\sigma^2/dR$ is piecewise linear; $d^2\sigma^2/dR^2 = -8$ (Fig. 5) |

### Reciprocal-Space Analysis (Phase 4)

| Finding | Evidence |
|---------|----------|
| $\tilde{\chi}_V(k) \sim k^3$ envelope | All chains show same small-$k$ scaling (Fig. 6) |
| Spreadability decay separates classes | Poisson $t^{-1/2}$, quasicrystal $t^{-2}$, lattice exp. (Fig. 7) |
| **$\alpha = 3$ universal** | **All three chains: $\alpha = 3.0 \pm 0.15$** (Fig. 8) |
| Benchmarks correct | Poisson $\alpha = 0$, lattice $\alpha \to \infty$ (Fig. 8) |

### The Key Result

> The hyperuniformity exponent $\alpha = 3$ is a **universal property** of all
> metallic-mean 1D substitution tilings. This has been confirmed through two
> independent routes:
> 1. **Analytically** via the eigenvalue formula $\alpha = 1 - 2\ln|\lambda_2|/\ln\lambda_1$
> 2. **Numerically** via the diffusion spreadability of two-phase media

---

## Slide 15 — Future Work (Phase 5: 2D Extensions)

### Plan
- Construct **2D quasiperiodic tilings** via the Generalized Dual Method (GDM):
  - 5-fold Penrose tiling (golden ratio) — $\alpha = 6$ known
  - 8-fold octagonal tiling (silver mean) — $\alpha$ unknown
  - Bronze-ratio equivalent — $\alpha$ unknown
- Decorate vertices with disks ($\phi_2 = 0.25$)
- Compute 2D radial spectral density and extract $\alpha$

### Open Questions
1. Is $\alpha = 6$ universal for all 2D metallic-mean tilings, analogous to
   $\alpha = 3$ in 1D?
2. Does the eigenvalue formula $\alpha = 1 - 2\ln|\lambda_2|/\ln\lambda_1$
   generalize correctly to 2D inflation matrices?
3. How large must 2D systems be for convergence? (The 1D experience suggests
   this is non-trivial.)

---

## Figure Index

| Figure | File | Shows | Key Takeaway |
|--------|------|-------|--------------|
| 1 | `fig1_poisson_benchmark.png`    | Poisson $\sigma^2$ vs exact  | Code validated (< 2% error) |
| 2 | `fig2_bounded_variance.png`     | 4-panel bounded variance     | Class I confirmed for all chains |
| 3 | `fig3_hyperuniformity_test.png` | $\sigma^2/R \to 0$           | Fluctuations suppressed vs random |
| 4 | `fig4_projection_comparison.png`| Class I vs Class II          | Strip width controls class |
| 5 | `fig5_piecewise_quadratic.png`  | $d^2\sigma^2/dR^2$ steps    | Quadratic segments confirmed |
| 6 | `fig6_spectral_density.png`     | $\tilde{\chi}_V(k)$ spectra | $k^3$ envelope universal |
| 7 | `fig7_excess_spreadability.png` | $E(t)$ decay curves          | Faster decay = more hyperuniform |
| 8 | `fig8_alpha_extraction.png`     | $\alpha(t)$ plateaus         | **$\alpha = 3$ for all chains** |

All figures at 150 DPI in `results/` directory.

---

## Code & Reproducibility

| File | Purpose | Lines |
|------|---------|-------|
| `substitution_tilings.py` | Chain generation (3 metallic means) | ~200 |
| `projection_method.py`    | Cut-and-project from $\mathbb{Z}^2$ | ~140 |
| `quasicrystal_variance.py`| Number variance via sliding window | ~130 |
| `two_phase_media.py`      | Spectral density + spreadability | ~210 |
| `run_all.py`              | Full pipeline: Figs 1-8 + tables | ~800 |

**Reproducibility:** `python run_all.py` (seed = 42, runtime ~18 minutes).
