"""Recompute the four v1.2 quantities the round-2 package is missing.

Gated: every routine is run on v1.1 first and must reproduce the published v1.1
value before its v1.2 output is trusted.

  1. all-category PERMDISP (manuscript claims F=2.63, p=0.013, n=1,145)
  2. 20-subsample PERMANOVA stability range (claims 4.2-5.3%, mean 4.6%)
  3. bias-corrected Chao1 per country (promised in S3, absent)
  4. Figure 4 human body-site crosswalk against the 73-rule labels
"""
import json
import numpy as np
import pandas as pd

BASE = '.'
SEED = 42


def bray_curtis(M):
    M = np.asarray(M, dtype=float)
    s = M.sum(axis=1)
    n = M.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        num = np.abs(M[i] - M[i + 1:]).sum(axis=1)
        den = s[i] + s[i + 1:]
        with np.errstate(invalid='ignore', divide='ignore'):
            v = np.where(den > 0, num / den, 0.0)
        D[i, i + 1:] = v
        D[i + 1:, i] = v
    return D


def permanova(D, groups, n_perm=999, seed=42):
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups)
    n = len(groups)
    ug = np.unique(groups)
    a = len(ug)
    SST = (D ** 2).sum() / (2 * n)

    def ssw_of(lab):
        s = 0.0
        for g in ug:
            idx = np.where(lab == g)[0]
            if len(idx) < 2:
                continue
            sub = D[np.ix_(idx, idx)]
            s += (sub ** 2).sum() / (2 * len(idx))
        return s

    SSW = ssw_of(groups)
    SSA = SST - SSW
    F_obs = (SSA / (a - 1)) / (SSW / (n - a))
    R2 = SSA / SST if SST > 0 else 0
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(groups)
        SSW_p = ssw_of(perm)
        F_p = ((SST - SSW_p) / (a - 1)) / (SSW_p / (n - a)) if SSW_p > 0 else 0
        cnt += (F_p >= F_obs)
    return float(F_obs), float(R2), (cnt + 1) / (n_perm + 1)


def permdisp_centroid(D, groups, n_perm=999, seed=42):
    """Anderson (2006) PERMDISP, test='centroid', in PCoA space.

    Same routine 05_percategory_pcoa.py uses for the per-category family, so the
    all-category value is produced by the identical method.
    """
    g = np.asarray(groups)
    n = len(g)
    C = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * C @ (np.asarray(D, dtype=float) ** 2) @ C
    w, V = np.linalg.eigh(G)
    keep = w > 1e-10
    X = V[:, keep] * np.sqrt(w[keep])
    ks = np.unique(g)

    def stat(lab):
        dist = np.empty(n)
        for c in ks:
            i = lab == c
            dist[i] = np.linalg.norm(X[i] - X[i].mean(axis=0), axis=1)
        grand = dist.mean()
        ssb = sum((lab == c).sum() * (dist[lab == c].mean() - grand) ** 2 for c in ks)
        ssw = sum(((dist[lab == c] - dist[lab == c].mean()) ** 2).sum() for c in ks)
        dfb, dfw = len(ks) - 1, n - len(ks)
        if ssw <= 0 or dfw <= 0 or dfb <= 0:
            return np.nan
        return (ssb / dfb) / (ssw / dfw)

    obs = stat(g)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = stat(rng.permutation(g))
        if np.isfinite(s) and s >= obs:
            ge += 1
    return float(obs), (ge + 1) / (n_perm + 1)


def build_bp(df):
    """Reproduce the pipeline's BioProject matrix and country vector."""
    bp_mat = df.groupby(['bioproject', 'scientific_name']).size().unstack(fill_value=0)
    bp_country = df.groupby('bioproject')['country'].agg(lambda x: x.mode().iloc[0])
    bp_runs = df.groupby('bioproject').size()
    keep_bp = bp_runs[bp_runs >= 3].index
    bp_mat = bp_mat.loc[bp_mat.index.isin(keep_bp)]
    bp_country = bp_country.loc[bp_country.index.isin(bp_mat.index)]
    ck = bp_country.value_counts()
    keep_c = ck[ck >= 3].index.tolist()
    bp_mat = bp_mat.loc[bp_country.isin(keep_c)]
    bp_country = bp_country.loc[bp_country.isin(keep_c)]
    return bp_mat, bp_country


def chao1_classic(x):
    obs = (x > 0).sum()
    f1 = (x == 1).sum()
    f2 = (x == 2).sum()
    if f2 == 0:
        return float(obs + f1 * (f1 - 1) / 2)
    return float(obs + f1 ** 2 / (2 * f2))


def chao1_bc(x):
    obs = (x > 0).sum()
    f1 = (x == 1).sum()
    f2 = (x == 2).sum()
    return float(obs + f1 * (f1 - 1) / (2 * (f2 + 1)))


RESULT = {}

for which in ['v11', 'v12']:
    print(f"\n{'='*62}\n{which}\n{'='*62}", flush=True)
    df = pd.read_csv(f'{BASE}/corpus_{which}.tsv', sep='\t', low_memory=False)
    df = df[df.scientific_name.notna()]
    bp_mat, bp_country = build_bp(df)
    print(f"qualifying BioProjects (>=3 runs, countries with >=3 such BPs): {len(bp_mat)}", flush=True)

    # ---- 1. all-category PERMDISP, uncapped ----
    D_full = bray_curtis(bp_mat.values)
    Fd, pd_ = permdisp_centroid(D_full, bp_country.values, n_perm=999, seed=SEED)
    print(f"ALL-CATEGORY PERMDISP (uncapped, n={len(bp_mat)}): F={Fd:.2f}, p={pd_:.3f}", flush=True)

    # ---- 2. 20-subsample PERMANOVA stability, 199 perms as stated in 3.12 ----
    r2s, ps = [], []
    for k in range(20):
        rng = np.random.RandomState(1000 + k)
        idx = rng.choice(len(bp_mat), min(500, len(bp_mat)), replace=False)
        sub = bp_mat.iloc[idx]
        subc = bp_country.iloc[idx]
        Ds = bray_curtis(sub.values)
        F, R2, p = permanova(Ds, subc.values, n_perm=199, seed=SEED)
        r2s.append(R2 * 100)
        ps.append(p)
    print(f"STABILITY over 20 subsamples: R2 {min(r2s):.1f}% to {max(r2s):.1f}%, "
          f"mean {np.mean(r2s):.1f}%, max p {max(ps):.4f}", flush=True)

    # ---- 3. Chao1 bias-corrected per country ----
    ct = df.groupby(['country', 'scientific_name']).size().unstack(fill_value=0)
    div = pd.DataFrame({
        'country': ct.index,
        'richness': [(r > 0).sum() for r in ct.values],
        'chao1': [chao1_classic(r) for r in ct.values],
        'chao1_bias_corrected': [chao1_bc(r) for r in ct.values],
        'f1_singletons': [(r == 1).sum() for r in ct.values],
        'f2_doubletons': [(r == 2).sum() for r in ct.values],
    }).sort_values('chao1', ascending=False)
    print("Chao1 (top 5 by classical):", flush=True)
    print(div.head(5).to_string(index=False), flush=True)
    for c in ['Iraq', 'Saudi Arabia', 'Iran']:
        row = div[div.country == c]
        if len(row):
            r = row.iloc[0]
            print(f"   {c}: obs={r.richness} classic={r.chao1:.0f} bc={r.chao1_bias_corrected:.0f} "
                  f"f1={r.f1_singletons} f2={r.f2_doubletons}", flush=True)

    RESULT[which] = {
        'n_bp': int(len(bp_mat)),
        'permdisp_F': round(Fd, 2), 'permdisp_p': round(pd_, 3),
        'stab_min': round(min(r2s), 1), 'stab_max': round(max(r2s), 1),
        'stab_mean': round(float(np.mean(r2s)), 1), 'stab_pmax': round(max(ps), 4),
        'chao1': div.to_dict('records'),
    }

# ---- 4. Figure 4 human crosswalk (v1.2 only) ----
print(f"\n{'='*62}\nFigure 4 human body-site crosswalk, v1.2\n{'='*62}", flush=True)
d12 = pd.read_csv(f'{BASE}/corpus_v12.tsv', sep='\t', low_memory=False)
bt = pd.read_csv(f'{BASE}/run_v12/data/mena_biosample_types.tsv', sep='\t', low_memory=False)
h = d12[d12.broad_category == 'Human'][['run_accession', 'specific_category']].merge(
    bt[['run_accession', 'biosample_type']], on='run_accession', how='left')
print(f"Human runs: {len(h)}")
cross = pd.crosstab(h.specific_category, h.biosample_type)
print(cross.to_string())
RESULT['human_crosswalk'] = cross.to_dict()

with open(f'{BASE}/RECOMPUTED_missing_values.json', 'w') as f:
    json.dump(RESULT, f, indent=1, default=str)
print("\nwritten -> RECOMPUTED_missing_values.json", flush=True)
