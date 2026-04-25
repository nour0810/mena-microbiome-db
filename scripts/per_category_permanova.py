"""
Per-category PERMANOVA — confounding by sample type removed.
Runs PERMANOVA within each broad_category to test whether country effect
on metagenome-type composition is stronger when sample-type confounding
is eliminated.
"""
import os, json, math
import pandas as pd, numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

DATA = "/home/claude/out/data/mena_metagenomics_clean.tsv"
DATADIR = "/home/claude/out/data"
TBLDIR = "/home/claude/out/tables"
FIGDIR = "/home/claude/out/figures"

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight",
})
PAL = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442","#999999",
       "#0b2545","#c9a227","#13315c","#8d6a2a"]

def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/{name}.svg", bbox_inches="tight")
    plt.close(fig)

print("Loading...")
df = pd.read_csv(DATA, sep="\t", low_memory=False)
print(f"  {len(df):,} runs")

def bray_curtis(M):
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            num = np.abs(M[i] - M[j]).sum()
            den = (M[i] + M[j]).sum()
            D[i,j] = D[j,i] = num/den if den>0 else 0
    return D

def permanova(D, groups, n_perm=999, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    groups = np.asarray(groups)
    n = len(groups)
    unique_groups = np.unique(groups)
    a = len(unique_groups)
    if a < 2 or n < a + 2:
        return None
    SST = (D**2).sum() / (2*n)
    def ssw(g):
        s = 0
        for u in unique_groups:
            idx = np.where(g==u)[0]
            nb = len(idx)
            if nb < 2: continue
            sub = D[np.ix_(idx, idx)]
            s += (sub**2).sum() / (2*nb)
        return s
    SSW = ssw(groups)
    SSA = SST - SSW
    if SSW <= 0 or SSA <= 0:
        return None
    F_obs = (SSA/(a-1)) / (SSW/(n-a))
    R2 = SSA / SST if SST > 0 else 0
    F_perm = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(groups)
        SSW_p = ssw(perm)
        SSA_p = SST - SSW_p
        F_perm[i] = (SSA_p/(a-1)) / (SSW_p/(n-a)) if SSW_p > 0 else 0
    p = (np.sum(F_perm >= F_obs) + 1) / (n_perm + 1)
    return {
        "F": float(F_obs), "R2": float(R2), "p": float(p),
        "n_samples": int(n), "n_groups": int(a),
        "groups": [str(g) for g in unique_groups],
    }

# ════════════════════════════════════════════════════════════════════
# Per-category PERMANOVA at BioProject level
# ════════════════════════════════════════════════════════════════════
print("\nPer-category PERMANOVA (BioProject level)")
print("Stratification: within each broad_category, test country effect on Bray-Curtis distance")

CATEGORIES = ["Human","Environment","Animal","Plant","Food","Clinical","Fungal"]
results = {}

for cat in CATEGORIES:
    sub = df[df["broad_category"]==cat].copy()
    if len(sub) < 30:
        print(f"  {cat:<12} skipped (n={len(sub)})")
        continue
    # Aggregate to BioProject level
    bp_mat = sub.groupby(["bioproject","scientific_name"]).size().unstack(fill_value=0)
    bp_country = sub.groupby("bioproject")["country"].agg(lambda x: x.mode().iloc[0])
    # keep BPs with ≥2 runs
    bp_runs = sub.groupby("bioproject").size()
    keep_bp = bp_runs[bp_runs >= 2].index
    bp_mat = bp_mat.loc[bp_mat.index.isin(keep_bp)]
    bp_country = bp_country.loc[bp_country.index.isin(bp_mat.index)]
    # keep countries with ≥3 BPs
    ck = bp_country.value_counts()
    keep_countries = ck[ck >= 3].index.tolist()
    if len(keep_countries) < 2:
        print(f"  {cat:<12} skipped (only {len(keep_countries)} countries with ≥3 BPs)")
        continue
    bp_mat = bp_mat.loc[bp_country.isin(keep_countries)]
    bp_country = bp_country.loc[bp_country.isin(keep_countries)]
    # cap at 400 BPs for compute
    if len(bp_mat) > 400:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(bp_mat), 400, replace=False)
        bp_mat = bp_mat.iloc[idx]
        bp_country = bp_country.iloc[idx]
    if len(bp_mat) < 10:
        continue

    print(f"  {cat:<12} {len(bp_mat):>4} BPs · {bp_country.nunique():>3} countries", end="  ")
    D = bray_curtis(bp_mat.values)
    res = permanova(D, bp_country.values, n_perm=999)
    if res is None:
        print("PERMANOVA failed")
        continue
    res["category"] = cat
    res["n_bps_per_country"] = bp_country.value_counts().to_dict()
    print(f"F={res['F']:.2f}  R²={res['R2']:.3f}  p={res['p']:.4f}")
    results[cat] = res

# Save
with open(f"{DATADIR}/mena_per_category_permanova.json","w") as f:
    json.dump(results, f, indent=2)

# Summary table
summary = pd.DataFrame([
    {
        "category": r["category"],
        "n_bioprojects": r["n_samples"],
        "n_countries": r["n_groups"],
        "F_statistic": round(r["F"],3),
        "R_squared": round(r["R2"],4),
        "R_squared_pct": round(r["R2"]*100,2),
        "p_value": round(r["p"],5),
        "significant": "yes" if r["p"]<0.05 else "no",
    } for r in results.values()
]).sort_values("R_squared", ascending=False)
summary.to_excel(f"{TBLDIR}/T3b_per_category_permanova.xlsx", index=False)

# Comparison plot: R² and -log10(p) per category
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cats_sorted = summary["category"].tolist()
r2_vals = summary["R_squared_pct"].values
pvals = summary["p_value"].values
neg_log_p = -np.log10(pvals.clip(min=1e-5))

# Add reference line: original ALL-CATEGORY result was R²=4.96%, p=0.001
ref_r2 = 4.96
ref_logp = -np.log10(0.001)

bars1 = axes[0].barh(cats_sorted[::-1], r2_vals[::-1],
                     color=[PAL[2] if r > ref_r2 else PAL[5] for r in r2_vals[::-1]])
axes[0].axvline(ref_r2, color="black", linestyle="--", lw=1.5, alpha=0.6,
                label=f"all-category baseline (R²={ref_r2}%)")
axes[0].set_xlabel("R² × 100 (% variance explained by country)")
axes[0].set_title("PERMANOVA effect size — within each category")
axes[0].legend(frameon=False, fontsize=9)
for i, v in enumerate(r2_vals[::-1]):
    axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

bars2 = axes[1].barh(cats_sorted[::-1], neg_log_p[::-1],
                     color=[PAL[2] if -np.log10(p) > -np.log10(0.05) else PAL[7] for p in pvals[::-1]])
axes[1].axvline(-np.log10(0.05), color="red", linestyle="--", lw=1.5, alpha=0.6, label="α=0.05")
axes[1].axvline(-np.log10(0.001), color="black", linestyle="--", lw=1.5, alpha=0.4, label="α=0.001")
axes[1].set_xlabel("−log₁₀(p)")
axes[1].set_title("PERMANOVA significance — within each category")
axes[1].legend(frameon=False, fontsize=9)
for i, p in enumerate(pvals[::-1]):
    label = f"p={p:.3f}" if p>=0.001 else f"p<0.001"
    axes[1].text(-np.log10(max(p,1e-5)) + 0.05, i, label, va="center", fontsize=9)

plt.tight_layout()
save(fig, "I02_permanova_per_category")

print(f"\n  saved: {DATADIR}/mena_per_category_permanova.json")
print(f"  saved: T3b_per_category_permanova.xlsx, I02_permanova_per_category.png/svg")

# Print summary
print("\n=== SUMMARY ===")
print(summary.to_string(index=False))
