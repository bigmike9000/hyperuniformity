# Week 3 Meeting Cheat Sheet

Anticipated questions and answers, organized by slide.

---

## General / Big Picture

**Q: What's the main new result this week?**
The alpha in (2,3) gap: no substitution matrix with |det M|=1 can produce alpha between 2 and 3, but |det M|=2 can. First known 1D example at alpha=2.071.

**Q: How does this fit into the JP?**
This would go in a results section on the substitution matrix parameter space. The gap theorem + non-unimodular construction is probably the most novel contribution.

**Q: What's the most important open question?**
Whether Silver lambda-bar = 1/4 exactly. If true, there's a clean rational sequence: lattice=1/6, Silver=1/4, URL=1/3. Proving it would require applying the Zachary-Torquato theta-function method to the Silver chain.

---

## Slide 3: Why spreadability failed for BT

**Q: Does spreadability ever work for BT?**
Yes, but it needs t > 10^8. At t in [10^2, 10^5] you get alpha-hat = 2.26. At [10^8, 10^11] you finally get ~1.51. It converges, just very slowly.

**Q: Why does lambda-bar from variance work but spreadability doesn't?**
Lambda-bar is a running average of sigma^2(R) -- it doesn't need a fitting window or slope extraction. Spreadability requires fitting a power law E(t) ~ t^{-(1+alpha)/2}, and the correction terms from large |lambda_2| = 0.802 contaminate the fit at moderate t.

**Q: Could you fix spreadability by using a larger fitting window?**
You'd need t > 10^8, which requires computing the spectral density and spreadability to extremely high precision. Not practical compared to just using variance.

---

## Slide 4: Convergence with N

**Q: How many points do you need for a reliable lambda-bar?**
About N ~ 1000 for substitution chains. The running average stabilizes quickly. This is shown in panel (a) -- both BT and Fibonacci are flat by N = 10^3.

**Q: What about for the stealthy patterns?**
Stealthy data came from Sam (collaborator) -- ~4316 configurations per chi value, N=2000, rho=1. The ensemble averaging substitutes for the long-chain averaging we do for substitutions.

---

## Slide 5: Exploring cubic substitution matrices

**Q: Why row sums <= 4?**
Computational tractability. Row sums = number of tiles produced by each substitution rule. Beyond 4, the number of matrices grows very fast. The search space is (number of rows)^3; with row sums up to 4, there are 34 possible rows, giving 34^3 = 39,304 matrices.

**Q: Why does |det M|=1 matter?**
det M = product of all eigenvalues. When |det M|=1, the eigenvalues are tightly constrained relative to each other. Specifically, if lambda_2 and lambda_3 are a complex conjugate pair: |det M| = lambda_1 * |lambda_2|^2 = 1, which forces |lambda_2| = 1/sqrt(lambda_1), which gives alpha = 2 exactly.

**Q: What's special about complex conjugate eigenvalue pairs?**
For a 3x3 matrix, the characteristic polynomial is cubic with real coefficients. If it has one real root (lambda_1) and two complex roots, those complex roots must be conjugates. This is the generic case -- having all three eigenvalues real requires special structure.

**Q: Could there be alpha in (2,3) with larger row sums?**
Not with |det M|=1. The proof is algebraic and doesn't depend on row sums. For any |det M|=1 matrix at any rank: complex pairs give alpha=2 (rank 3) or alpha<2 (rank >= 4). All-real cases give alpha=3 in every example we've checked.

**Q: What are the 4,577 remaining matrices?**
They have all-real eigenvalues and all give alpha=3. These are like higher-dimensional versions of the metallic-mean chains (Fibonacci, Silver, Bronze...). No analytical proof for why alpha=3, just numerical confirmation.

---

## Slide 6: Substitution tilings at alpha=2

**Q: How different are the 504 matrices from each other?**
They have 18 distinct lambda_1 values (range 1.32 to 3.85). Matrices with the same lambda_1 share the same characteristic polynomial but have different substitution rules. Lambda-bar varies widely (0.26 to 1.7) across different matrices, even those with the same lambda_1.

**Q: Why only show one example?**
Lambda-bar was only computed for one representative per lambda_1 group. Computing all 504 is a TODO.

**Q: Is alpha=2 interesting physically?**
It's the boundary between Class I behavior where lambda-bar is finite and well-behaved, and the regime where additional eigenvalues start degrading the hyperuniformity quality. URL with a=1 also has alpha=2.

---

## Slide 7: Metrics for Classes II and III

**Q: Why not just use lambda-bar for all classes?**
Lambda-bar = lim sigma^2(R) as R -> infinity. For Class II, sigma^2 ~ Lambda_II * ln(R), which grows without bound. For Class III, sigma^2 ~ Lambda_III * R^{1-alpha}. So the limit doesn't exist -- lambda-bar is undefined (not infinite, just undefined).

**Q: Why use curve fitting instead of just taking the ratio sigma^2/ln(R)?**
The ratio sigma^2/ln(R) = Lambda_II + b/ln(R) converges to Lambda_II from above at finite R. This plateau method is biased. The curve fit sigma^2 = Lambda_II * ln(R) + b directly extracts Lambda_II without this bias.

**Q: Are Lambda_II and Lambda_III standard notation?**
No, we defined these ourselves. Torquato's papers write out the asymptotic forms but don't name the coefficients. We chose Lambda with subscripts to be consistent with lambda-bar for Class I.

**Q: What's the physical meaning of Lambda_II = 0.082?**
It's the rate of logarithmic growth of density fluctuations. A smaller Lambda_II means fluctuations grow more slowly -- closer to Class I behavior.

---

## Slide 8: Updated catalog

**Q: Why is alpha "---" for lattice and stealthy?**
Their S(k) doesn't follow a power law S(k) ~ k^alpha near k=0. The lattice has Bragg peaks (S(k) is delta functions). Stealthy has S(k)=0 by construction for |k| < K. Neither fits the power-law framework used to define alpha.

**Q: Can you compare lambda-bar across different densities?**
Yes. Lambda-bar is invariant under rescaling x_i -> x_i/rho. Verified: Fibonacci at rho=0.724 vs rescaled to rho=1, difference < 0.001.

**Q: Why is stealthy lambda-bar so high?**
Stealthy with chi=0.1 has lambda-bar = 1.021 (not shown in table). The formula lambda-bar ~ 1/(pi^2 * chi) fits well. Small chi means the S(k)=0 constraint is weak, allowing large fluctuations.

**Q: Where does the stealthy data come from?**
Sam's data. About 4316 configurations per chi value, N=2000, rho=1. These are disordered ground states found by optimization from random initial positions.

---

## Slides 9-10: Matrices with 2 < alpha < 3

**Q: Is the |det M|=1 result a proof or just numerical evidence?**
For the complex-pair case: it's a proof. The algebra is exact -- |det M| = lambda_1 * |lambda_2|^2 = 1 forces alpha=2. For the all-real case: it's numerical evidence (all examples with row sums <= 4 give alpha=3), not a proof.

**Q: Why can't you get alpha > 2.2 with |det M|=2?**
The formula is alpha = 3 - 2*ln|det M|/ln(lambda_1). To get alpha in (2,3) with |det M|=2, you need 1 < 2 < sqrt(lambda_1), i.e., lambda_1 > 4. As lambda_1 grows, alpha = 3 - 2*ln(2)/ln(lambda_1) approaches 3, but slowly. With lambda_1 around 4-10 (practical range), alpha stays in about 2.0-2.2.

**Q: Could you use |det M|=3 to get higher alpha?**
You'd need sqrt(lambda_1) > 3, i.e., lambda_1 > 9. That means very large substitution rules. With |det M|=3: alpha = 3 - 2*ln(3)/ln(lambda_1). At lambda_1=10: alpha = 2.05. At lambda_1=100: alpha = 2.52. Possible in principle but requires enormous matrices.

**Q: Has anyone else found chains with alpha in (2,3)?**
Oguz et al. (2019) have a matrix family that *implicitly* fills this range (their b=2 family), but they never noted that |det M|=1 matrices can't reach it, never generated any of these tilings, and never measured lambda-bar. The gap observation and the explicit construction are new.

**Q: Is the alpha=2.071 chain still hyperuniform?**
Yes. It's Class I (bounded sigma^2). The condition for hyperuniformity is |lambda_i| < 1 for all i >= 2 (Pisot property), which this chain satisfies (|lambda_2| = 0.449).

---

## Slide 12: Silver lambda-bar = 1/4

**Q: How confident are you that it's exactly 1/4?**
The measurement is 0.2502 +/- 0.0006, which is 0.3 sigma from 1/4. Statistically consistent, but we can't distinguish "exactly 1/4" from "very close to 1/4" numerically. An analytical proof would be needed.

**Q: How would you prove it?**
Zachary & Torquato (2009) computed Fibonacci lambda-bar = 0.20110 using a theta-function expansion of sigma^2(R). The same method applied to Silver (using the Silver-ratio acceptance window in the cut-and-project construction) could give an exact result.

**Q: Why would Silver be 1/4?**
Silver has rho = 1/2 exactly (mean spacing = 2), which is unusually clean. If Silver = 1/4, then lattice = 1/6, Silver = 1/4, URL = 1/3 forms a sequence of simple rationals. Could be coincidence, could be meaningful.

**Q: Does Fibonacci have a known exact lambda-bar?**
Zachary & Torquato (2009) report 0.20110 via theta-series, which is higher precision than our 0.201 but still a numerical result. No known closed form for Fibonacci.

---

## Slide 13: Metallic-mean convergence to 1/3

**Q: Why does the limit equal 1/3?**
As n -> infinity, the metallic-mean ratio mu_n -> infinity, so the tile ratio L/S -> 1. The chain approaches a lattice where each point is displaced quasi-randomly within its unit cell -- matching a URL with cloaking parameter a=1, which has lambda-bar = (1+a^2)/6 = 1/3 exactly (Klatt et al. 2020).

**Q: Is the convergence to 1/3 proven?**
No, it's a numerical observation. We computed lambda-bar for n=1 through 20. The convergence is monotone from below and fits 1/3 - lambda-bar ~ 0.22/n^1.43. But there's no analytical proof.

**Q: Is there a closed-form formula for lambda-bar(mu_n)?**
Open question. We have 20 data points. Silver might be 1/4 exactly. The limit is 1/3. But we don't have a formula connecting them.

**Q: How is this related to URL?**
The URL (Uniform Random Lattice) with parameter a has lambda-bar = (1+a^2)/6. At a=0 it's a perfect lattice (lambda-bar = 1/6). At a=1 it's maximally disordered (lambda-bar = 1/3). The metallic-mean chains interpolate between these as n increases.

---

## General Technical Questions

**Q: What's the eigenvalue formula for alpha?**
alpha = 1 - 2*ln|lambda_2| / ln(lambda_1), from Oguz et al. (2019). lambda_1 is the largest eigenvalue of the substitution matrix, lambda_2 is the second largest (by absolute value).

**Q: What's the difference between alpha from S(k) and alpha from spreadability?**
Same quantity in theory. S(k) ~ k^alpha as k -> 0 defines alpha. Spreadability E(t) ~ t^{-(1+alpha)/2} gives the same alpha. In practice, extracting alpha from S(k) is hard for substitution tilings because S(k) is all Bragg peaks (no smooth background to fit). Spreadability works but can be slow to converge. Variance-based lambda-bar avoids needing alpha entirely.

**Q: How do you compute sigma^2(R)?**
Place random windows of half-width R along the chain, count points in each window, compute the variance of those counts. We use 20,000-50,000 windows depending on chain length.

**Q: What system sizes are you using?**
Typically N = 500k to 1M for routine computations. High-precision runs: N = 3.88M (Silver confirmation), N = 9.37M (Silver figure), N = 17.8M (alpha=2.071 chain). BT spreadability needed N = 597k but t up to 10^11.
