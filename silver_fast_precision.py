"""
Fast high-precision Lambda-bar for Silver and Bronze chains.
Uses Monte Carlo variance with 50k windows, 2000 R values.
Runs in ~10 minutes total.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, sys, os

sys.path.insert(0, '/c/Users/minec/OneDrive/Desktop/Hyperuniformity')
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points, predict_chain_length
from quasicrystal_variance import compute_number_variance_1d

FIGURES_DIR = '/c/Users/minec/OneDrive/Desktop/Hyperuniformity/results/figures'

def run_precision(chain_name, target_N=9_000_000, num_R=2000, num_windows=50000, seed=42):
    chain = CHAINS[chain_name]
    print(f"\n{'='*60}")
    print(f"  Precision: {chain['name']}")
    print(f"{'='*60}")
    for iters in range(5, 50):
        n = predict_chain_length(chain_name, iters)
        if n >= target_N:
            break
    print(f"  {iters} iters -> N_pred={n:,}")
    t0 = time.perf_counter()
    seq = generate_substitution_sequence(chain_name, iters)
    pts, L = sequence_to_points(seq, chain_name)
    N = len(pts)
    rho = N / L
    ms = 1.0/rho  # mean spacing
    print(f"  N={N:,}, L={L:.1f}, rho={rho:.8f}, ms={ms:.8f}  [{time.perf_counter()-t0:.1f}s]")

    # R range: many oscillation periods
    R_min = ms * 0.5
    R_max = min(L / 5.0, 10000 * ms)
    R_arr = np.linspace(R_min, R_max, num_R)
    n_osc = (R_max - R_min) / ms
    print(f"  R: [{R_min:.2f}, {R_max:.1f}] ({n_osc:.0f} mean-spacing periods)")
    print(f"  Computing {num_R} R-values x {num_windows:,} windows ...")

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    variances, means = compute_number_variance_1d(pts, L, R_arr, num_windows=num_windows,
                                                   rng=rng, periodic=True)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Multiple estimators
    # 1) Simple mean over last 2/3
    s = num_R // 3
    lb_mean = np.mean(variances[s:])
    lb_std  = np.std(variances[s:])
    n_eff   = num_R - s
    lb_sem  = lb_std / np.sqrt(n_eff)

    # 2) Block averages (20 blocks) for robust error
    n_blk = 20
    bsz = n_eff // n_blk
    blk = [np.mean(variances[s + b*bsz : s + (b+1)*bsz]) for b in range(n_blk)]
    blk = np.array(blk)
    lb_blk = np.mean(blk)
    lb_blk_err = np.std(blk) / np.sqrt(n_blk)

    # 3) Running average
    running = np.cumsum(variances) / np.arange(1, num_R+1)

    # 4) Trapz
    try:
        lb_tr = np.trapezoid(variances[s:], R_arr[s:]) / (R_arr[-1] - R_arr[s])
    except AttributeError:
        lb_tr = np.trapz(variances[s:], R_arr[s:]) / (R_arr[-1] - R_arr[s])

    # 5) Also average over the LAST 1/3 only for the sharpest window
    s2 = 2 * num_R // 3
    lb_late = np.mean(variances[s2:])
    lb_late_err = np.std(variances[s2:]) / np.sqrt(num_R - s2)

    print(f"\n  Lambda-bar estimates:")
    print(f"    Simple mean (last 2/3):  {lb_mean:.7f} +/- {lb_sem:.2e}")
    print(f"    Block avg  (20 blocks):  {lb_blk:.7f}  +/- {lb_blk_err:.2e}")
    print(f"    Running avg (final):     {running[-1]:.7f}")
    print(f"    Trapz (last 2/3):        {lb_tr:.7f}")
    print(f"    Late mean (last 1/3):    {lb_late:.7f} +/- {lb_late_err:.2e}")
    print(f"    Variance range: [{variances[s:].min():.5f}, {variances[s:].max():.5f}]")
    print(f"    osc amplitude / mean = {lb_std / lb_mean:.4f}")

    return dict(N=N, rho=rho, L=L, ms=ms,
                R_arr=R_arr, variances=variances, running=running,
                lb_mean=lb_mean, lb_sem=lb_sem,
                lb_blk=lb_blk, lb_blk_err=lb_blk_err,
                lb_tr=lb_tr, lb_late=lb_late, lb_late_err=lb_late_err,
                blk=blk)


def check_candidates(val, name, tol=2e-3):
    sqrt2 = np.sqrt(2); sqrt3=np.sqrt(3); sqrt5=np.sqrt(5); sqrt13=np.sqrt(13)
    phi = (1+sqrt5)/2
    mu_s = 1 + sqrt2  # silver metallic mean
    mu_b = (3+sqrt13)/2  # bronze metallic mean
    candidates = [
        ('1/4',              1/4),
        ('1/3',              1/3),
        ('1/6',              1/6),
        ('1/5',              1/5),
        ('sqrt(2)/5',        sqrt2/5),
        ('9/32',             9/32),
        ('17/60',            17/60),
        ('11/39',            11/39),
        ('5/18',             5/18),
        ('7/25',             7/25),
        ('(sqrt(13)-1)/8',   (sqrt13-1)/8),
        ('sqrt(13)/12',      sqrt13/12),
        ('(3+sqrt(13))/24',  (3+sqrt13)/24),
        ('(sqrt(13)+1)/24',  (sqrt13+1)/24),
        ('1/(2*mu_s)',       1/(2*mu_s)),
        ('1/(2+mu_s)',       1/(2+mu_s)),
        ('mu_s/(4*(mu_s**2+1))', mu_s/(4*(mu_s**2+1))),
        ('1/(4*(mu_s+1/mu_s))', 1/(4*(mu_s+1/mu_s))),
        ('(mu_s-1)/(2*(mu_s+1)**2)', (mu_s-1)/(2*(mu_s+1)**2)),
        ('(mu_s-1)/(4*mu_s)', (mu_s-1)/(4*mu_s)),
        ('1/(6*(sqrt2-1))',  1/(6*(sqrt2-1))),
        ('(sqrt2-1)/2',     (sqrt2-1)/2),
        ('(3-sqrt2)/8',     (3-sqrt2)/8),
        ('(2-sqrt2)/4',     (2-sqrt2)/4),
        ('1/(4*sqrt2)',     1/(4*sqrt2)),
        ('mu_b/(4*(mu_b**2+1))', mu_b/(4*(mu_b**2+1))),
        ('1/(4*(mu_b+1/mu_b))', 1/(4*(mu_b+1/mu_b))),
        ('(mu_b-1)/(4*mu_b)', (mu_b-1)/(4*mu_b)),
        ('3/(4*sqrt2+4)',   3/(4*sqrt2+4)),
        ('(sqrt13-2)/4',   (sqrt13-2)/4),
    ]
    print(f"\n  Checking {name} = {val:.7f}:")
    matches = []
    for expr, cval in candidates:
        if cval is None or not np.isfinite(cval) or cval <= 0:
            continue
        diff = abs(val - cval)
        if diff < tol / 2:
            matches.append((expr, cval, diff))
            print(f"    MATCH:  {expr:35s} = {cval:.7f}  (diff={diff:.2e})")
        elif diff < tol:
            print(f"    close:  {expr:35s} = {cval:.7f}  (diff={diff:.2e})")
    if not matches:
        print(f"    (no match within {tol})")
    return matches


if __name__ == '__main__':
    print("="*70)
    print("  Fast Precision Lambda-bar: Silver and Bronze")
    print("="*70)

    # Silver: N=9.4M, 2000 R values
    silver = run_precision('silver', target_N=9_000_000, num_R=2000, num_windows=50000)

    # Fibonacci: N=5.7M, 2000 R values (cross-check vs Zachary 0.20110)
    fib    = run_precision('fibonacci', target_N=5_000_000, num_R=2000, num_windows=50000)

    # Bronze: N=6.6M, 2000 R values
    bronze = run_precision('bronze', target_N=5_000_000, num_R=2000, num_windows=50000)

    # ---- Summary ----
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    results = {'fibonacci': fib, 'silver': silver, 'bronze': bronze}
    for name in ['fibonacci', 'silver', 'bronze']:
        r = results[name]
        print(f"\n  {CHAINS[name]['name']} (N={r['N']:,}):")
        print(f"    lb_mean  = {r['lb_mean']:.7f} +/- {r['lb_sem']:.2e}")
        print(f"    lb_block = {r['lb_blk']:.7f} +/- {r['lb_blk_err']:.2e}")
        print(f"    lb_late  = {r['lb_late']:.7f} +/- {r['lb_late_err']:.2e}")

    print(f"\n  Zachary 2009: Fibonacci = 0.2011000")
    print(f"  Our Fibonacci: {fib['lb_mean']:.7f} (diff = {fib['lb_mean']-0.20110:.2e})")

    # Deviation tests
    print(f"\n  Silver deviation from 1/4: {silver['lb_mean'] - 0.25:+.5f}")
    n_sig = abs(silver['lb_mean'] - 0.25) / silver['lb_blk_err']
    print(f"  = {n_sig:.1f} sigma -> {'CONSISTENT with 1/4' if n_sig < 2 else 'NOT 1/4'}")

    # Candidate expressions
    check_candidates(silver['lb_mean'], 'Silver Lambda-bar', tol=2e-3)
    check_candidates(bronze['lb_mean'], 'Bronze Lambda-bar', tol=2e-3)
    check_candidates(fib['lb_mean'], 'Fibonacci Lambda-bar', tol=2e-3)

    # ---- Pattern analysis ----
    print("\n\n  Pattern analysis: does Lambda-bar follow f(n) for metallic index n?")
    data = [
        (1, (1+np.sqrt(5))/2,    fib['lb_mean'],    'Fibonacci'),
        (2, 1+np.sqrt(2),        silver['lb_mean'],  'Silver'),
        (3, (3+np.sqrt(13))/2,  bronze['lb_mean'],  'Bronze'),
    ]
    print(f"  {'n':>3}  {'mu_n':>10}  {'lb':>12}  {'phi*BN':>10}  {'lb*mu':>10}")
    for n, mu, lb, cname in data:
        phiBN = lb/2
        print(f"  {n:>3}  {mu:10.6f}  {lb:12.7f}  {phiBN:10.7f}  {lb*mu:10.6f}")

    # Test formula: lb = 1/6 * (something)
    print("\n  Trying lb = A + B/mu_n:")
    lbs = np.array([d[2] for d in data[:3]])
    mus = np.array([d[1] for d in data[:3]])
    A, B = np.polyfit(1/mus, lbs, 1)  # lbs = A*(1/mu) + B
    print(f"    Fit: lb = {B:.6f} + {A:.6f}/mu")
    print(f"    Predicted Fibonacci: {B + A/(1+np.sqrt(5))*2:.6f} (actual {data[0][2]:.6f})")
    print(f"    Predicted Silver:    {B + A/(1+np.sqrt(2)):.6f} (actual {data[1][2]:.6f})")
    print(f"    Predicted Bronze:    {B + A/(3+np.sqrt(13))*2:.6f} (actual {data[2][2]:.6f})")

    # Check if lb follows lb = 1/(4+2/mu) or similar
    for fn_name, fn in [
        ('1/(4+2/mu)', lambda mu: 1/(4+2/mu)),
        ('1/(4+1/mu)', lambda mu: 1/(4+1/mu)),
        ('1/6+1/(6*mu)', lambda mu: 1/6 + 1/(6*mu)),
        ('(mu+1)/(6*(mu+2))', lambda mu: (mu+1)/(6*(mu+2))),
        ('mu/(6*(mu+1))', lambda mu: mu/(6*(mu+1))),
        ('(1/6)*(1+1/mu)', lambda mu: (1/6)*(1+1/mu)),
        ('(1/6)*(1+2/mu)', lambda mu: (1/6)*(1+2/mu)),
        ('(1/4)*(1-1/(2*mu**2))', lambda mu: (1/4)*(1-1/(2*mu**2))),
        ('(1/4)*(1-1/mu**2)', lambda mu: (1/4)*(1-1/mu**2)),
        ('1/6 + (mu-1)/(6*(mu+1))', lambda mu: 1/6 + (mu-1)/(6*(mu+1))),
    ]:
        preds = [fn(d[1]) for d in data]
        residuals = [abs(preds[i]-data[i][2]) for i in range(3)]
        if max(residuals) < 5e-3:
            print(f"  FORMULA CANDIDATE: lb = {fn_name}")
            for i, d in enumerate(data):
                print(f"    {d[3]:10}: pred={preds[i]:.6f}, actual={d[2]:.6f}, diff={residuals[i]:.2e}")

    # ---- Figure ----
    print("\nMaking figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    clrs = {'fibonacci': '#1f77b4', 'silver': '#2ca02c', 'bronze': '#d62728'}
    lbls = {'fibonacci': 'Fibonacci', 'silver': 'Silver', 'bronze': 'Bronze'}

    # Panel (0,0): Silver variance curve
    ax = axes[0, 0]
    r = silver
    mask = r['R_arr'] <= 300 * r['ms']
    ax.plot(r['R_arr'][mask]/r['ms'], r['variances'][mask],
            '-', color='#2ca02c', lw=0.4, alpha=0.8)
    ax.axhline(0.25, color='red', ls='--', lw=2, label='$1/4$')
    ax.axhline(r['lb_mean'], color='orange', ls='-', lw=1.5,
               label=f"mean={r['lb_mean']:.5f}")
    ax.set_xlabel('$R$ / mean spacing'); ax.set_ylabel('$\\sigma^2(R)$')
    ax.set_title(f"Silver (N={r['N']:,})")
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)

    # Panel (0,1): Running average all 3 chains
    ax = axes[0, 1]
    for name in ['fibonacci', 'silver', 'bronze']:
        r2 = results[name]
        ax.plot(r2['R_arr']/r2['ms'], r2['running'],
                '-', color=clrs[name], lw=1.5,
                label=f"{lbls[name]}: {r2['lb_mean']:.5f}")
    ax.axhline(0.25, color='k', ls='--', lw=1, label='1/4')
    ax.axhline(0.20110, color='gray', ls=':', lw=1, label='Fib 0.20110')
    ax.set_xlabel('$R$ / mean spacing'); ax.set_ylabel('Running $\\bar{\\Lambda}$')
    ax.set_title('Running Average')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
    ax.set_ylim([0.18, 0.31])

    # Panel (0,2): Silver deviation from 1/4
    ax = axes[0, 2]
    r = silver
    start = len(r['R_arr']) // 3
    ax.plot(r['R_arr'][start:]/r['ms'], r['running'][start:] - 0.25,
            '-', color='#2ca02c', lw=1)
    ax.axhline(0, color='red', ls='--', lw=2, label='Exact 1/4')
    err = r['lb_blk_err']
    ax.fill_between([0, r['R_arr'][-1]/r['ms']], [-err, -err], [err, err],
                    alpha=0.3, color='orange', label=f'+/- {err:.4f} (1 sigma)')
    ax.set_xlabel('$R$ / mean spacing'); ax.set_ylabel('Running $\\bar{\\Lambda}$ - 1/4')
    ax.set_title('Silver: Deviation from 1/4')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)

    # Panel (1,0): Block means for Silver
    ax = axes[1, 0]
    blks = silver['blk']
    ax.bar(np.arange(1, len(blks)+1), blks, color='#2ca02c', alpha=0.7)
    ax.axhline(0.25, color='red', ls='--', lw=2, label='1/4')
    ax.axhline(silver['lb_blk'], color='orange', lw=2,
               label=f"mean={silver['lb_blk']:.5f} +/- {silver['lb_blk_err']:.4f}")
    ax.set_xlabel('Block'); ax.set_ylabel('Block mean $\\sigma^2$')
    ax.set_title('Silver: Block Means')
    ax.legend(fontsize=9); ax.set_ylim([0.22, 0.28])
    ax.grid(True, ls=':', alpha=0.4)

    # Panel (1,1): Variance distribution
    ax = axes[1, 1]
    for name in ['fibonacci', 'silver', 'bronze']:
        r2 = results[name]; s = len(r2['R_arr'])//3
        ax.hist(r2['variances'][s:], bins=60, alpha=0.4, color=clrs[name],
                label=f"{lbls[name]}: {np.mean(r2['variances'][s:]):.4f}", density=True)
    ax.set_xlabel('$\\sigma^2(R)$'); ax.set_ylabel('Density')
    ax.set_title('Distribution of $\\sigma^2$')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)

    # Panel (1,2): Summary table
    ax = axes[1, 2]; ax.axis('off')
    lines = [
        "PRECISION RESULTS (N~9M each)",
        "="*43,
        "",
        f"Fibonacci: {fib['lb_mean']:.7f} +/- {fib['lb_blk_err']:.1e}",
        f"  Zachary 2009 ref: 0.2011000",
        f"  diff = {fib['lb_mean']-0.20110:+.2e}",
        "",
        f"Silver:    {silver['lb_mean']:.7f} +/- {silver['lb_blk_err']:.1e}",
        f"  1/4    = 0.2500000",
        f"  diff   = {silver['lb_mean']-0.25:+.2e}",
        f"  sigma  = {abs(silver['lb_mean']-0.25)/silver['lb_blk_err']:.1f} sigma from 1/4",
        "",
        f"Bronze:    {bronze['lb_mean']:.7f} +/- {bronze['lb_blk_err']:.1e}",
        "",
        "Candidates for Bronze:",
        f"  9/32     = {9/32:.7f} (diff={abs(bronze['lb_mean']-9/32):.2e})",
        f"  sqrt2/5  = {np.sqrt(2)/5:.7f} (diff={abs(bronze['lb_mean']-np.sqrt(2)/5):.2e})",
        f"  17/60    = {17/60:.7f} (diff={abs(bronze['lb_mean']-17/60):.2e})",
        f"  11/39    = {11/39:.7f} (diff={abs(bronze['lb_mean']-11/39):.2e})",
        "",
        "phi*B_N (= lb/2):",
        f"  Lattice:   1/12 = {1/12:.7f}",
        f"  Fibonacci: {fib['lb_mean']/2:.7f} (Zachary: 0.100550)",
        f"  Silver:    {silver['lb_mean']/2:.7f} (=1/8? {1/8:.7f})",
        f"  Bronze:    {bronze['lb_mean']/2:.7f}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax.set_title('Precision Summary')

    plt.suptitle('High-Precision Lambda-bar: Is Silver = 1/4?', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig_silver_precision.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out}")
    plt.close()

    print("\nDone.")
