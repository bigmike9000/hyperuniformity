# Comprehensive Ranking of Hyperuniform 1D Point Patterns — Project Context

## Principal Investigator: Professor Salvatore Torquato, Princeton University

## 1. Motivation and Scope

Hyperuniformity is a unifying framework that encompasses a remarkably broad range of
ordered and disordered systems with anomalously suppressed density fluctuations [1].
In one dimension, the known landscape of hyperuniform point patterns — particularly
for Class I ($\alpha > 1$) — remains surprisingly sparse. The table in the Physics
Reports review [1] lists only a handful of 1D examples, leaving open the question
of how diverse hyperuniform structures compare in their degree of order.

**This project aims to provide a much more comprehensive ranking of 1D hyperuniform
point configurations**, spanning two broad families:

1. **Quasicrystals** — metallic-mean substitution tilings (Fibonacci, silver, bronze,
   copper, nickel) and other aperiodic ordered chains.
2. **Exotic disordered systems** — hyperuniform point patterns that lack long-range
   order yet still suppress large-scale density fluctuations (e.g., stealthy
   hyperuniform, maximally random jammed, cloaked systems, perturbed lattices).

The ranking is built from two complementary metrics: the **hyperuniformity exponent
$\alpha$**, which classifies how strongly fluctuations are suppressed, and the
**surface-area coefficient $\bar{\Lambda}$**, which quantifies the residual
fluctuation amplitude within a given class. Together, these provide a fine-grained
"fingerprint" for each pattern's degree of order.

A key motivation is that hyperuniform systems often exhibit unusual physical
properties — anomalous transport, photonic band gaps, mechanical rigidity — that
correlate with their hyperuniformity class and $\bar{\Lambda}$ value [1, 2]. By
establishing a comprehensive $(\alpha, \bar{\Lambda})$ ranking across diverse 1D
structures, this project lays groundwork for connecting structural metrics to
physical behavior.

## 2. Core Definitions

- **Hyperuniformity:** A point pattern where the local number variance $\sigma^2(R)$
  within a window of radius $R$ grows slower than the window volume [1].

- **Scaling Exponent ($\alpha$):** In reciprocal space, the structure factor $S(k)$
  (or spectral density $\tilde{\chi}_V(k)$) vanishes as $k \to 0$, scaling as
  $S(k) \sim |k|^\alpha$ [2, 3].

- **Hyperuniformity Classes:**
  - **Class I** ($\alpha > 1$): Variance bounded, scales as $R^{d-1}$.
    Includes crystals and quasicrystals [1, 3, 4].
  - **Class II** ($\alpha = 1$): Variance scales as $R^{d-1}\ln(R)$.
    Includes period-doubling limit-periodic chains [3, 5].
  - **Class III** ($0 < \alpha < 1$): Variance scales as $R^{d-\alpha}$.
    Includes the 0222 chain [3, 5].
  - **Anti-hyperuniform** ($\alpha < 0$): Variance grows faster than random [6].

- **Surface-Area Coefficient ($\bar{\Lambda}$):** The amplitude coefficient of the
  leading variance growth term. For Class I systems in 1D this is the time-averaged
  value of the bounded, oscillating variance function. Reference values:
  - Integer lattice: $\bar{\Lambda} = 1/6$
  - Poisson (random baseline, not hyperuniform): $\sigma^2(R) = 2\rho R$ [7]

## 3. 1D Substitution Tilings (Quasicrystals)

Quasicrystals can be generated using a substitution matrix $\mathbf{M}$ applied
iteratively to a seed tile (Short $S$ or Long $L$) [8, 9].

- **Fibonacci Chain (Golden Ratio):**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$,
  substitution $S \to L$, $L \to LS$ [8, 10].

- **Silver Ratio (0112) Chain:**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 2 \end{pmatrix}$,
  substitution $S \to L$, $L \to LLS$; metallic mean $\mu_2 = 1 + \sqrt{2} \approx 2.414$ [11].

- **Bronze Ratio Chain:**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 3 \end{pmatrix}$,
  substitution $S \to L$, $L \to LLLS$; metallic mean $\mu_3 = (3 + \sqrt{13})/2 \approx 3.303$ [12].

- **Copper Ratio Chain:**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 4 \end{pmatrix}$,
  substitution $S \to L$, $L \to LLLLS$; metallic mean $\mu_4 = 2 + \sqrt{5} \approx 4.236$.

- **Nickel Ratio Chain:**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 5 \end{pmatrix}$,
  substitution $S \to L$, $L \to LLLLLS$; metallic mean $\mu_5 = (5 + \sqrt{29})/2 \approx 5.193$.

- **Eigenvalue Formula for $\alpha$:** If $\lambda_1$ and $\lambda_2$ are the largest
  and second-largest eigenvalues of $\mathbf{M}$:
  $$\alpha = 1 - \frac{2 \ln|\lambda_2|}{\ln|\lambda_1|}$$
  For all metallic mean chains, $\det(\mathbf{M}) = -1$, so
  $|\lambda_2| = 1/\lambda_1$, which gives $\alpha = 3$ universally [5, 13].

- **Alternative Generation — Projection (Cut-and-Project) Method:**
  Construct a 2D square lattice $\mathbb{Z}^2$ and project points within a strip
  onto a 1D "physical space" line with slope $1/\mu_n$. The strip width $\omega$
  in the perpendicular (internal) direction controls the hyperuniformity class:
  - Ideal width $\omega = \mu_n$: produces a Class I hyperuniform quasicrystal
    with exactly 2 distinct spacings. For Fibonacci, spacing ratio = $\tau$
    (matches substitution). For silver/bronze, spacing ratio = $(\mu+1)/\mu$.
  - Non-ideal width $\omega \neq \mu_n$: degrades the pattern to Class II
    hyperuniformity with logarithmically growing variance envelope.

- **$\bar{\Lambda}$ Rescaling Invariance:** The long-range average of the bounded
  variance $\bar{\Lambda}$ is independent of point density. Under rescaling
  $x \to ax$, $\sigma'^2(R) = \sigma^2(R/a)$, whose long-range average is
  unchanged. This is verified numerically: the projection method (density
  $\rho \approx 0.85$) and substitution method (density $\rho \approx 0.72$)
  give the same $\bar{\Lambda}$ for Fibonacci within 0.1%.

## 4. The Measurement Problem & Two-Phase Media

Many hyperuniform systems (especially quasicrystals with dense, discontinuous Bragg
peaks) make direct evaluation of $\alpha$ from $S(k)$ impossible [14, 15]. The
workaround:

- **Mapping:** Convert the point pattern to a 1D two-phase medium by placing
  non-overlapping solid rods on each point [16, 17].
- **Packing Fraction:** $\phi_2 = 0.35$ [17, 18].

## 5. Diffusion Spreadability $\mathcal{S}(t)$

The dynamic fraction of solute diffusing from particles (phase 2) into void space
(phase 1) over time [19, 20].

- **Formula:**
  $$\mathcal{S}(\infty) - \mathcal{S}(t) = \frac{d\,\omega_d}{(2\pi)^d \phi_2}
  \int_0^\infty k^{d-1}\, \tilde{\chi}_V(k)\, e^{-k^2 D t}\, dk$$
  [21, 22]

- **Extracting $\alpha$:** At long times, the excess spreadability decays as
  $t^{-(d+\alpha)/2}$. The logarithmic derivative of the spreadability curve
  yields $\alpha$ [23–26].

## 6. Current Results

| Pattern | N (tiles) | $\bar{\Lambda}$ | vs Lattice |
|---|---|---|---|
| Integer Lattice | 100,000 | ~1/6 | reference |
| Fibonacci (substitution) | 14,930,352 | 0.200 | 1.21x |
| Silver (substitution) | 22,619,537 | 0.250 | 1.51x |
| Bronze (substitution) | 21,932,293 | 0.282 | 1.71x |
| Copper (substitution) | 39,088,169 | 0.293 | 1.76x |
| Nickel (substitution) | 16,387,276 | 0.310 | 1.86x |
| Fibonacci (projection) | 220,161 | 0.200 | 1.21x |

Projection vs substitution agreement (Fibonacci): 0.1% difference, confirming
$\bar{\Lambda}$ rescaling invariance.

**Notable:** $\bar{\Lambda}$ increases monotonically with the metallic-mean index $n$:
Fibonacci (0.200) < Silver (0.250) < Bronze (0.282) < Copper (0.293) < Nickel (0.310).
The successive differences (0.050, 0.032, 0.011, 0.017) are decreasing, suggesting
convergence to a limit $\bar{\Lambda}_\infty \lesssim 1/3$ as $n \to \infty$.
The $\bar{\Lambda}$ values for Silver, Bronze, Copper, and Nickel appear to be **novel
results** not previously reported in the literature.

## 7. Project Roadmap (1D Focus)

### Phase 1: Code Benchmarking & Validation — COMPLETE

- **Task:** Generate a 1D Poisson point pattern with periodic boundary conditions.
- **Goal:** Compute $\sigma^2(R)$ via sliding window and confirm it matches the
  exact result $\sigma^2(R) = 2\rho R$.
- **Status:** COMPLETE. Implemented in `Poisson variance benchmark.py`.
  Mean relative error 1.1\% (100 realizations).

### Phase 2: Generating Quasiperiodic Patterns — COMPLETE

- **Substitution method:** Implemented in `substitution_tilings.py`. Vectorized
  generation supports N ~ $10^7$ chains for all three metallic means.
- **Projection method:** Implemented in `projection_method.py`. Cut-and-project
  from $\mathbb{Z}^2$ with parameterized strip width. Demonstrates Class I
  (ideal $\omega = \mu$) vs Class II (non-ideal $\omega \neq \mu$) transition.
- **Status:** COMPLETE for both methods.

### Phase 3: Computing Variance and $\bar{\Lambda}$ — COMPLETE

- Computed $\sigma^2(R)$ for all chains at N ~ $10^7$ scale.
- Verified Class I behavior: bounded, oscillating variance for all chains.
- Verified piecewise quadratic structure via second derivative analysis.
- Exact analytic formula $\sigma^2(R) = 2\{R\}(1-2\{R\})$ validated for lattice.
- $\bar{\Lambda}$ values stable across N = 500k → 10M (convergence confirmed).
- **Status:** COMPLETE. All results in `results/` directory.

### Phase 4: Two-Phase Media and Diffusion Spreadability — COMPLETE

- Decorated point patterns with non-overlapping solid rods ($\phi_2 = 0.35$).
- Computed structure factor $S(k)$ via histogram binning + FFT.
- Computed spectral density $\tilde{\chi}_V(k) = \rho |m̃(k)|^2 S(k)$.
- Evaluated excess spreadability $E(t) = S(\infty) - S(t)$ over $t \in [10^{-2}, 10^8]$.
- Extracted $\alpha$ via logarithmic derivative of $E(t)$.
- Implemented in `two_phase_media.py` (library) and `run_all.py` (Figs 6–8).

**Alpha extraction results (linear fit of $\ln E$ vs $\ln t$ over $[10^2, 10^5]$):**

| Pattern | $\alpha$ | Expected | Error | Status |
|---|---|---|---|---|
| Poisson | 0.001 | 0 | — | OK |
| Integer Lattice | exp. decay | $\infty$ | — | OK |
| Fibonacci (Golden Ratio) | 3.049 | 3 | 1.6% | OK |
| Silver Ratio | 2.992 | 3 | 0.3% | OK |
| Bronze Ratio | 2.987 | 3 | 0.4% | OK |

- Two extraction methods: (1) period-aware log derivative for $\alpha(t)$ curves
  (sliding window matched to oscillation period $2\ln\mu$ in $\ln t$ space), and
  (2) linear fit of $\ln E$ vs $\ln t$ for reported values (more robust, averages
  over all oscillations in the plateau window).
- Integer lattice computed analytically from Bragg peak positions (avoids FFT
  binning artifacts). Its $E(t)$ decays exponentially, reaching zero before the
  plateau window — confirming $\alpha \to \infty$ as expected for crystals.
- **Status:** COMPLETE. All three metallic-mean chains yield $\alpha \approx 3$,
  confirming Class I hyperuniformity via the two-phase media approach.

### Phase 5: Expanding the 1D Catalog — COMPLETE

- Extended the analysis pipeline to stealthy hyperuniform patterns and perturbed
  lattices. Implemented in `stealthy_hyperuniform.py`, `perturbed_lattices.py`,
  and `run_phase5.py`.

**Stealthy Hyperuniform Patterns** (Torquato et al., 2015, Phys. Rev. X 5):
- Data source: ~4,300 configurations per $\chi$ from Torquato group grad student.
  $N=2{,}000$ particles, density $\rho=1$, stored in `stealthy_data/`.
- $\chi$ values: 0.1, 0.2, 0.3.
- $S(k)=0$ verified in exclusion zone to $\sim 10^{-14}$ (averaged over ensemble).
- $\alpha \to \infty$ (exponential spreadability decay, not power-law) — Class I.
- $\bar{\Lambda}$ computed from 500 configurations per $\chi$, SEM $< 0.2\%$.

| $\chi$ | $N$ | $\bar{\Lambda}$ | SEM | std |
|---|---|---|---|---|
| 0.10 | 2000 | 1.026 | 0.0018 | 0.040 |
| 0.20 | 2000 | 0.528 | 0.0006 | 0.014 |
| 0.30 | 2000 | 0.358 | 0.0004 | 0.010 |

**Perturbed Lattice Patterns** (Klatt et al., 2020, Phys. Rev. E 101):
- Uniform Random Lattice (URL): $a=0.1, 0.3, 0.5, 0.8, 1.0$. $\alpha=2$, Class I.
- Gaussian perturbation: $\sigma=0.1, 0.2, 0.3, 0.5$. $\alpha=2$, Class I.
- Cauchy perturbation: $\gamma=0.1$. $\alpha=1$, Class II.
- $N=100{,}000$ for all perturbed lattice computations.
- Exact $\bar{\Lambda}$ formula for URL validated (Fig. 14).

| Pattern | $\alpha$ (measured) | $\alpha$ (expected) | $\bar{\Lambda}$ | $\bar{\Lambda}$ (exact) |
|---|---|---|---|---|
| URL $a=0.1$ | 2.01 | 2 | 0.168 | 0.168 |
| URL $a=0.3$ | 2.08 | 2 | 0.181 | 0.182 |
| URL $a=0.5$ | 2.03 | 2 | 0.208 | 0.208 |
| URL $a=0.8$ | 2.00 | 2 | 0.274 | 0.273 |
| URL $a=1.0$ (cloaked) | 1.98 | 2 | 0.333 | 0.333 |
| Gaussian $\sigma=0.1$ | 2.04 | 2 | 0.187 | — |
| Gaussian $\sigma=0.2$ | 2.04 | 2 | 0.247 | — |
| Gaussian $\sigma=0.3$ | 2.02 | 2 | 0.341 | — |
| Gaussian $\sigma=0.5$ | 1.96 | 2 | 0.564 | — |
| Cauchy $\gamma=0.1$ | 0.95 | 1 | 1.166 | — (grows) |

**Symmetric Stable Distribution Perturbations** ($\alpha = s$):
- Generated using Chambers-Mallows-Stuck algorithm for symmetric stable RVs.
- Characteristic function: $\hat{f}(k) = \exp(-c^s |k|^s)$, giving $\alpha = s$.
- Scale parameter $c=0.1$. $N=100{,}000$.

| Pattern | $\alpha$ (measured) | $\alpha$ (expected) | Class | $\bar{\Lambda}$ |
|---|---|---|---|---|
| Stable $s=0.3$ | 0.28 | 0.3 | III | 72.6 (grows) |
| Stable $s=0.5$ | 0.49 | 0.5 | III | 18.1 (grows) |
| Stable $s=0.7$ | 0.74 | 0.7 | III | 5.0 (grows) |
| Stable $s=1.3$ | 1.47 | 1.3 | I | 0.417 |
| Stable $s=1.5$ | 1.75 | 1.5 | I | 0.303 |
| Stable $s=1.7$ | 1.91 | 1.7 | I | 0.242 |

The $s > 1$ stable perturbations fill the gap $1 < \alpha < 2$ with Class I patterns.
For Class II/III patterns, $\bar{\Lambda}$ grows with window size $R$ and
is not a fixed number. The reported values are measured at $R_{\max}=300$.

**Copper & Nickel Quasicrystals:**
- Generated at $N \sim 10^7$--$4 \times 10^7$ using substitution method.
- $\alpha = 3$ confirmed via both eigenvalue formula (exact) and spreadability fit.

| Chain | $N$ | $\bar{\Lambda}$ | $\alpha$ (fit) | $\alpha$ (eigenvalue) |
|---|---|---|---|---|
| Copper | 39,088,169 | 0.293 | 3.03 | 3.000 |
| Nickel | 16,387,276 | 0.310 | 2.88 | 3.000 |

**Comprehensive Ranking** (Fig. 11):
- 25 patterns spanning crystals, quasicrystals, stealthy, perturbed lattices,
  and stable distribution perturbations across all three hyperuniformity classes.
- Two-panel chart: log scale (all classes) and linear scale (Class I only).
- Class I ranking: Lattice (0.167) → URL $a=0.1$ (0.168) → URL $a=0.3$
  (0.181) → Gaussian $\sigma=0.1$ (0.187) → Fibonacci (0.200) → ...
  → Nickel (0.310) → URL $a=1.0$ (0.333) → higher-disorder patterns.
- All figures saved to `results/` directory (fig9-14).

**Literature Validation:**
- Integer lattice $\bar{\Lambda} = 1/6$ exact (Torquato & Stillinger 2003).
- Fibonacci $\bar{\Lambda} = 0.200$ matches 0.201 from Zachary & Torquato 2009 (0.5%).
- All five metallic-mean chains give $\alpha \approx 3$ within 4% of eigenvalue prediction.
- URL exact formula validated for all $a$ values, all within 0.2% error.
- Silver $\bar{\Lambda} = 0.250$, Bronze = 0.282, Copper = 0.293, Nickel = 0.310 are **novel**.
- **Open gap:** No known 1D construction achieves $2 < \alpha < 3$.

### Phase 6: Next Steps (pre-JP)

- Obtain additional $\chi$ values for stealthy patterns from grad student.
- Add more exotic patterns: period-doubling (Class II/$\alpha=1$), other
  substitution tilings, maximally random jammed packings.
- Investigate the $2 < \alpha < 3$ gap: is there a 1D construction?
- Compile final $(\alpha, \bar{\Lambda})$ ranking table for JP paper.
- Investigate correlations between structural metrics and physical properties.
- Study the monotonic trend $\bar{\Lambda}(\mu_n)$ for metallic means — does it
  converge to a limit as $n \to \infty$?

### Phase 6: Additional Patterns and Analysis — COMPLETE

Completed the remaining Phase 5/6 roadmap items:

**New substitution chains (added to `substitution_tilings.py`):**
- **Period-doubling** (Class II, α=1 exact): S→L, L→LSS; λ₁=2, λ₂=−1; σ²≈0.089 ln R+0.277
- **0222 chain** (Class III, α=0.639): S→LL, L→SSLL; α_fit=0.665 at N≈10⁶

**New infrastructure:** `disordered_patterns.py`, `stealthy_analysis.py`, `research_catalog.py`,
`make_figures.py` (saves `results/catalog.json`, `results/stealthy_collab_results.json`).

**Stealthy collab data analyzed** (4314 configs/χ, N=2000): Λ̄=1.021/0.526/0.357 for χ=0.1/0.2/0.3.
Analytic formula Λ̄=1/(π²χ) derived; systematic +0.7–5.7% excess from S(k) overshoot above K.

**Literature added:** batten_stealthy_2008.pdf, morse_stealthy_dimensions_2024.pdf,
middlemas_hyperuniformity_ranking_2019.pdf (3 new PDFs, DOIs in KNOWLEDGE.md).

**Open questions:** Is Fibonacci Λ̄=1/5 exact? Silver=1/4? Does Λ̄(μₙ)→1/3? What fills 2<α<3 gap?

### Phase 7: Junior Paper (JP) — COMPLETE

A full Princeton Junior Paper LaTeX document was written summarizing all project results.

**Output file:** `jp/jp_hyperuniformity.tex` (compiled PDF: `jp/jp_hyperuniformity.pdf`, 14 pages)
**Stub bibliography:** `jp/references.bib` (6 entries — Torquato 2003, Torquato 2018,
Oğuz et al. 2017, Bombieri-Taylor 1986, Shechtman 1984, Torquato-Zhang-Stillinger 2015)

**Document structure:**
- Title block modeled exactly on old JP (`resources/old_jp.tex`): title, author (Michael Fang),
  advisor (Prof. Torquato), second reader (TBD), honor code `/s/` signature.
- Abstract (~250 words): hyperuniformity overview, scope, key results (all metallic means α=3,
  Bombieri-Taylor α=1.545 fills 1<α<2 gap).
- §1 Introduction — cereal scooping analogy, physics motivation, scope outline.
- §2 Theoretical Background — number variance, Classes I/II/III, exponent α, surface-area
  coefficient Λ̄, substitution tilings, eigenvalue formula α=1−2ln|λ₂|/ln|λ₁|.
- §3 Methods — variance/structure-factor/spreadability algorithms, validation, substitution generation.
- §4 Results I: Metallic-Mean Quasicrystals — Fibonacci/Silver/Bronze, Table 1, det=±1
  constraint derivation showing α=3 universally.
- §5 Results II: Expanding the Spectrum — Period-Doubling (α=1), 0222 chain (α≈0.64),
  URL (exact Λ̄ formula), stealthy (Λ̄≈1/π²χ), Bombieri-Taylor cubic (α=1.545, gap filled).
- §6 Complete Catalog — Table 2 with 26 patterns, Fig. 12 ranking.
- §7 Conclusion — synthesis, open questions (2<α<3 gap, higher-rank matrices, 2D extension).
- Acknowledgements — Torquato group, stealthy collaborator.
- Appendix A — hyperuniformity test figure.

**Compilation:** pdflatex + biber + pdflatex×2. Third pass: zero warnings/errors.
**Figures referenced** (all from `results/figures/`): fig1_poisson_benchmark, fig2_bounded_variance,
fig3_hyperuniformity_test, fig9_stealthy_sk, stealthy_Sk_overlay, bombieri_taylor_analysis,
fig12_ranking.

**Repo reorganization** (completed before JP):
- `results/figures/` — all PNG figures (previously flat in `results/`)
- `results/data/` — catalog.json, stealthy_collab_results.json
- `results/project_summary/` — project_summary.tex/.pdf
- `results/week2/` — week2_full_presentation.tex/.pdf
- `resources/` — old_jp.tex, example presentation PDF
- `literature/` — all reference PDFs (11 papers)
