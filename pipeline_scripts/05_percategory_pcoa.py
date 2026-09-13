#!/usr/bin/env python
"""
Step 5. Per-category PERMANOVA + PERMDISP, classical PCoA, BioSample types.

The round-1 per-category script was not among the recovered files, so the method
is re-implemented from Methods 2.7 as written and validated against the published
v1.1 values in Supplementary Table S3. Run v11 first; if the control column
matches, the v12 column can be used.

usage: python 05_percategory_pcoa.py v11|v12
"""
import sys, json, numpy as np, pandas as pd
from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp
from scipy.spatial.distance import pdist, squareform

which = sys.argv[1]
SEED = 42
d = pd.read_csv(f'corpus_{which}.tsv', sep='\t', dtype=str, low_memory=False)


def permdisp_anderson(D, groups, n_perm=999, seed=42):
    """PERMDISP (Anderson 2006): distance of each point to its group centroid in
    PCoA space, then a one-way ANOVA F on those distances with a permutation
    p-value. Implemented directly because skbio's centroid routine raises
    ZeroDivisionError on groups whose members are mutually fully dissimilar."""
    g = np.asarray(groups)
    n = len(g)
    # PCoA embedding of the distance matrix (real axes only)
    C = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * C @ (np.asarray(D, dtype=float) ** 2) @ C
    w, V = np.linalg.eigh(G)
    keep = w > 1e-10
    X = V[:, keep] * np.sqrt(w[keep])

    def stat(lab):
        dist = np.empty(n)
        for c in np.unique(lab):
            i = lab == c
            dist[i] = np.linalg.norm(X[i] - X[i].mean(axis=0), axis=1)
        grand = dist.mean()
        ks = np.unique(lab)
        ssb = sum((lab == c).sum() * (dist[lab == c].mean() - grand) ** 2 for c in ks)
        ssw = sum(((dist[lab == c] - dist[lab == c].mean()) ** 2).sum() for c in ks)
        dfb, dfw = len(ks) - 1, n - len(ks)
        if ssw <= 0 or dfw <= 0 or dfb <= 0:
            return np.nan
        return (ssb / dfb) / (ssw / dfw)

    obs = stat(g)
    if not np.isfinite(obs):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    ge = sum(1 for _ in range(n_perm) if (lambda s: np.isfinite(s) and s >= obs)(stat(rng.permutation(g))))
    return float(obs), (ge + 1) / (n_perm + 1)

# ---------------------------------------------------------------- per-category
CATS = ['Human', 'Environment', 'Animal', 'Plant', 'Food', 'Clinical', 'Fungal']
# 9,999 permutations for the per-category family. At 999 the Monte Carlo standard
# error on a p-value near 0.05 is about 0.007, which is large enough to move a
# borderline q across the threshold between runs; at 9,999 it is about 0.002.
# The all-category test in rigorous_analysis.py retains 999 because its p-value is
# at the attainable floor at either count and its stability is characterised
# instead by the 20-subsample re-runs (Results 3.12).
CAP_CAT, N_PERM = 400, 9999
out = {}
print(f"{'category':12s} {'nBP':>5} {'nC':>4} {'F':>6} {'R2%':>7} {'E[R2]%':>7} {'adj%':>6} "
      f"{'p':>6} {'DISP F':>7} {'DISP p':>7}")
for cat in CATS:
    sub = d[d.broad_category == cat]
    # BioProjects with >=2 runs, countries with >=3 such BioProjects
    bp = sub.groupby('bioproject').agg(n=('run_accession', 'size'),
                                       country=('country', lambda s: s.mode().iloc[0]))
    bp = bp[bp.n >= 2]
    keep = bp.country.value_counts(); keep = keep[keep >= 3].index
    bp = bp[bp.country.isin(keep)]
    if bp.country.nunique() < 2 or len(bp) < 3:
        print(f"{cat:12s} {len(bp):5d} {bp.country.nunique():4d}   -- not testable --")
        out[cat] = dict(n_bioprojects=int(len(bp)), n_countries=int(bp.country.nunique()),
                        testable=False)
        continue
    rng = np.random.default_rng(SEED)
    if len(bp) > CAP_CAT:
        bp = bp.loc[rng.choice(bp.index, CAP_CAT, replace=False)]
    M = (sub[sub.bioproject.isin(bp.index)]
         .groupby(['bioproject', 'scientific_name']).size().unstack(fill_value=0)
         .reindex(bp.index).fillna(0))
    D = squareform(pdist(M.values.astype(float), metric='braycurtis'))
    np.nan_to_num(D, copy=False)
    dm = DistanceMatrix(D, ids=[str(i) for i in range(len(M))])
    grp = bp.country.astype(str).tolist()
    # scikit-bio's permanova and permdisp draw permutations from the global NumPy
    # RNG and accept no seed argument, so the global seed is set immediately before
    # each call. Without this the permutation p-values are re-drawn on every run
    # and the reported values are not reproducible (Methods 2.11).
    np.random.seed(SEED)
    r = permanova(dm, grp, permutations=N_PERM)
    F, p = float(r['test statistic']), float(r['p-value'])
    a, b_ = len(set(grp)), len(grp)
    R2 = (F * (a - 1)) / (F * (a - 1) + (b_ - a))       # from pseudo-F
    E = (a - 1) / (b_ - 1)                              # null expectation
    adj = (R2 - E) / (1 - E)
    # scikit-bio PERMDISP in its default form, test='median' (the geometric-median
    # variant), which is what Methods 2.7 describes. It raises ZeroDivisionError on
    # groups whose members are mutually fully dissimilar, in which case the direct
    # Anderson (2006) form is reported and flagged, so the two are never silently
    # mixed. Whether the error fires depends on the permutation draw, which is why
    # the seed below matters for the routine actually used as well as for the p-value.
    disp_src = 'skbio-median (default)'
    try:
        np.random.seed(SEED)
        rd = permdisp(dm, grp, permutations=N_PERM)
        dF, dp = float(rd['test statistic']), float(rd['p-value'])
    except Exception as e:
        dF, dp = permdisp_anderson(D, grp, n_perm=N_PERM, seed=SEED)
        disp_src = f'anderson-fallback ({type(e).__name__})'
        print(f"   [{cat}: skbio PERMDISP raised {type(e).__name__}; "
              f"reporting direct Anderson (2006) form instead]")
    print(f"{cat:12s} {b_:5d} {a:4d} {F:6.2f} {100*R2:7.2f} {100*E:7.2f} {100*adj:6.2f} "
          f"{p:6.3f} {dF:7.2f} {dp:7.3f}")
    out[cat] = dict(n_bioprojects=b_, n_countries=a, F=round(F, 3),
                    R2_pct=round(100*R2, 2), E_R2_pct=round(100*E, 2),
                    R2_adj_pct=round(100*adj, 2), p=p,
                    permdisp_F=round(dF, 3), permdisp_p=dp, permdisp_src=disp_src, testable=True)

# Benjamini-Hochberg across the testable family
tst = {k: v for k, v in out.items() if v.get('testable')}
order = sorted(tst, key=lambda k: tst[k]['p'])
m, prev = len(order), 1.0
for i, k in enumerate(reversed(order)):
    rank = m - i
    q = min(prev, tst[k]['p'] * m / rank)
    out[k]['q_BH'] = round(q, 4); prev = q
print("\nBH-adjusted q (family of %d):" % m,
      {k: out[k]['q_BH'] for k in order})

# ---------------------------------------------------------------- classical PCoA
cm = d.groupby(['country', 'scientific_name']).size().unstack(fill_value=0)
cnt = d.country.value_counts()
types = (d.groupby('country').scientific_name.nunique())
keep = [c for c in cm.index if cnt[c] >= 30 and types[c] >= 10]
P = (cm.loc[keep] > 0).astype(int)
J = squareform(pdist(P.values, metric='jaccard'))
n = len(J)
Jc = -0.5 * (np.eye(n) - np.ones((n, n))/n) @ (J**2) @ (np.eye(n) - np.ones((n, n))/n)
ev = np.linalg.eigvalsh(Jc)[::-1]
pos = ev[ev > 0]
print(f"\nPCoA: {n} countries | PC1 {100*pos[0]/pos.sum():.1f}%  PC2 {100*pos[1]/pos.sum():.1f}%  "
      f"sum {100*(pos[0]+pos[1])/pos.sum():.1f}%")
print(f"Jaccard mean {J[np.triu_indices(n,1)].mean():.4f} "
      f"(range {J[np.triu_indices(n,1)].min():.4f}-{J[np.triu_indices(n,1)].max():.4f})")
out['_pcoa'] = dict(n_countries=n, pc1=round(100*pos[0]/pos.sum(), 1),
                    pc2=round(100*pos[1]/pos.sum(), 1),
                    jaccard_mean=round(float(J[np.triu_indices(n,1)].mean()), 4))
out['_metagenome_types'] = int(d.scientific_name.nunique())
print(f"metagenome types: {out['_metagenome_types']}")

json.dump(out, open(f'percategory_{which}.json', 'w'), indent=1)
print(f"\nwrote percategory_{which}.json")
