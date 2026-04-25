"""
MENA Microbiome — RIGOROUS ANALYTICAL SUITE
============================================
Replaces the previous ML/DL/AI work with publication-quality analyses
defensible for Scientific Data / GigaScience / NAR Database.

Tiers:
  1. Data quality & curation (MIxS-proxy completeness, lat/lon validation,
     duplicate detection, temporal lag)
  2. Ecological diversity (rarefaction, Shannon/Simpson, Jaccard, PCoA)
  3. Inferential statistics (PERMANOVA, IndVal, chi-square residuals,
     Mann-Kendall trend tests)
  4. Geospatial (density maps, Moran's I, biome maps)
  5. NLP done right (study_title-only since abstracts are absent;
     per-country TF-IDF + per-study K-Means clustering)
"""
import os, json, re, warnings, hashlib
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy import stats as spstats
import pymannkendall as mk

DATA = "/home/claude/out/data/mena_metagenomics_clean.tsv"
FIGDIR = "/home/claude/out/figures"
DATADIR = "/home/claude/out/data"
TBLDIR = "/home/claude/out/tables"

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

def xlsx(df, name):
    df.to_excel(f"{TBLDIR}/{name}.xlsx", index=False)

print("Loading...")
df = pd.read_csv(DATA, sep="\t", low_memory=False)
print(f"  {len(df):,} runs, {df['country'].nunique()} countries")

OUT = {}

# ════════════════════════════════════════════════════════════════════
# TIER 1 — DATA QUALITY & CURATION
# ════════════════════════════════════════════════════════════════════
print("\n[T1.1] MIxS-proxy completeness scoring")

# GSC MIxS-MIMS minimum fields, mapped to what we have
MIXS_FIELDS = {
    "Geographic location": ["country"],
    "Collection date":     ["collection_date"],
    "GPS coordinates":     ["sample_lat_lon"],
    "Environmental biome": ["environment_biome"],
    "Environmental feature":["environment_feature"],
    "Environmental material":["environment_material"],
    "Isolation source":    ["isolation_source"],
    "Host (if applicable)":["host"],
    "Library strategy":    ["library_strategy"],
    "Sequencing platform": ["instrument_platform"],
    "Project name":        ["bioproject"],
    "Sample title":        ["sample_title"],
}

def is_filled(v):
    if pd.isna(v): return False
    s = str(v).strip().lower()
    if not s: return False
    if s in {"not collected","not applicable","na","n/a","none","missing","unknown","not available","null","-"}: return False
    return True

# per-run completeness score
mixs_per_run = pd.DataFrame(index=df.index)
for label, cols in MIXS_FIELDS.items():
    if cols[0] in df.columns:
        mixs_per_run[label] = df[cols[0]].apply(is_filled)
    else:
        mixs_per_run[label] = False

mixs_per_run["score"] = mixs_per_run.sum(axis=1)
mixs_per_run["pct"] = 100 * mixs_per_run["score"] / len(MIXS_FIELDS)

df["mixs_score"] = mixs_per_run["score"].values
df["mixs_pct"] = mixs_per_run["pct"].values

# per-country average
country_mixs = df.groupby("country").agg(
    n_runs=("run_accession","count"),
    mean_mixs_pct=("mixs_pct","mean"),
    median_mixs_pct=("mixs_pct","median"),
    sd_mixs_pct=("mixs_pct","std"),
).round(1).reset_index().sort_values("mean_mixs_pct", ascending=False)
xlsx(country_mixs, "T1_mixs_completeness_per_country")

# per-field completeness across whole DB
field_completeness = (mixs_per_run.drop(columns=["score","pct"]).mean()*100).round(1).sort_values()
fc_df = field_completeness.reset_index(); fc_df.columns = ["MIxS field","% complete"]
xlsx(fc_df, "T1_mixs_field_completeness")

OUT["mixs"] = {
    "fields": list(MIXS_FIELDS.keys()),
    "field_completeness": fc_df.to_dict(orient="records"),
    "per_country": country_mixs.to_dict(orient="records"),
    "overall_mean_pct": float(df["mixs_pct"].mean().round(2)),
    "overall_median_pct": float(df["mixs_pct"].median().round(2)),
}
print(f"  overall mean MIxS completeness: {df['mixs_pct'].mean():.1f}%")

# Figure: per-field completeness bars + per-country distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(field_completeness.index, field_completeness.values,
             color=[PAL[2] if v>70 else PAL[1] if v>30 else PAL[5] for v in field_completeness.values])
axes[0].axvline(50, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xlabel("% complete")
axes[0].set_title("MIxS-proxy field completeness across the corpus")
for i,(lbl,v) in enumerate(zip(field_completeness.index, field_completeness.values)):
    axes[0].text(v+1, i, f"{v:.1f}%", va="center", fontsize=8)

cm = country_mixs.head(24)
axes[1].barh(cm["country"][::-1], cm["mean_mixs_pct"][::-1], color=PAL[8])
axes[1].errorbar(cm["mean_mixs_pct"][::-1], range(len(cm)),
                 xerr=cm["sd_mixs_pct"][::-1], fmt="none", color="gray", alpha=0.5, capsize=2)
axes[1].set_xlabel("Mean MIxS completeness (%)")
axes[1].set_title("Per-country mean completeness (± SD)")
plt.tight_layout()
save(fig, "Q01_mixs_completeness")

# ─── T1.2 lat/lon validation ──────────────────────────────────────
print("[T1.2] Lat/lon validation against country bounding boxes")

# MENA country bounding boxes (lon_min, lon_max, lat_min, lat_max)
BBOX = {
    "Algeria":(-8.7, 12.0, 18.9, 37.1),
    "Bahrain":(50.3, 50.8, 25.5, 26.4),
    "Djibouti":(41.8, 43.4, 10.9, 12.7),
    "Egypt":(24.7, 36.9, 21.7, 31.7),
    "Iran":(44.0, 63.4, 25.0, 39.8),
    "Iraq":(38.8, 48.6, 29.1, 37.4),
    "Jordan":(34.9, 39.3, 29.2, 33.4),
    "Kuwait":(46.6, 48.4, 28.5, 30.1),
    "Lebanon":(35.1, 36.6, 33.0, 34.7),
    "Libya":(9.4, 25.2, 19.5, 33.2),
    "Mauritania":(-17.1, -4.8, 14.7, 27.3),
    "Morocco":(-13.2, -1.0, 21.4, 35.9),
    "Oman":(52.0, 60.0, 16.6, 26.4),
    "Palestine":(34.2, 35.6, 31.2, 32.6),
    "Qatar":(50.7, 51.7, 24.5, 26.2),
    "Saudi Arabia":(34.5, 55.7, 16.4, 32.2),
    "Somalia":(40.9, 51.4, -1.7, 11.9),
    "South Sudan":(24.1, 35.9, 3.5, 12.2),
    "Sudan":(21.8, 38.6, 8.7, 22.0),
    "Syria":(35.7, 42.4, 32.3, 37.3),
    "Tunisia":(7.5, 11.6, 30.2, 37.5),
    "Turkey":(26.0, 44.8, 35.8, 42.1),
    "UAE":(51.5, 56.4, 22.6, 26.1),
    "Yemen":(42.5, 53.1, 12.1, 19.0),
}

def parse_latlon(s):
    if pd.isna(s): return None, None
    s = str(s).strip()
    # patterns like "33.8869 N 9.5375 E" or "33.8869, 9.5375"
    m = re.match(r"\s*([\-]?\d+\.?\d*)\s*([NS])\s+([\-]?\d+\.?\d*)\s*([EW])", s)
    if m:
        lat = float(m.group(1)) * (1 if m.group(2)=="N" else -1)
        lon = float(m.group(3)) * (1 if m.group(4)=="E" else -1)
        return lat, lon
    m = re.match(r"\s*([\-]?\d+\.?\d*)\s*[,;\s]+\s*([\-]?\d+\.?\d*)", s)
    if m:
        try:
            lat = float(m.group(1)); lon = float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except: pass
    return None, None

ll = df["sample_lat_lon"].apply(parse_latlon)
df["lat"] = [x[0] for x in ll]
df["lon"] = [x[1] for x in ll]

geo_mask = df["lat"].notna() & df["lon"].notna()
print(f"  parsed: {geo_mask.sum():,} / {len(df):,} ({100*geo_mask.sum()/len(df):.1f}%)")

# Validate
def in_bbox(row):
    bb = BBOX.get(row["country"])
    if bb is None or pd.isna(row["lat"]): return None
    lon_min, lon_max, lat_min, lat_max = bb
    return (lon_min <= row["lon"] <= lon_max) and (lat_min <= row["lat"] <= lat_max)

df["lat_lon_valid"] = df.apply(in_bbox, axis=1)
gp = df[geo_mask].copy()
n_valid = (gp["lat_lon_valid"]==True).sum()
n_invalid = (gp["lat_lon_valid"]==False).sum()
print(f"  valid: {n_valid:,} ({100*n_valid/len(gp):.1f}%); invalid: {n_invalid:,} ({100*n_invalid/len(gp):.1f}%)")

# Per-country validation rate
pc_valid = gp.groupby("country").agg(
    n_with_coords=("run_accession","count"),
    n_valid=("lat_lon_valid", lambda x: (x==True).sum()),
    n_invalid=("lat_lon_valid", lambda x: (x==False).sum()),
).reset_index()
pc_valid["pct_valid"] = (100*pc_valid["n_valid"]/pc_valid["n_with_coords"]).round(1)
pc_valid = pc_valid.sort_values("n_with_coords", ascending=False)
xlsx(pc_valid, "T1_geocoord_validation")

OUT["geocoord"] = {
    "n_total": int(len(df)),
    "n_with_coords": int(geo_mask.sum()),
    "pct_with_coords": round(100*geo_mask.sum()/len(df), 2),
    "n_valid": int(n_valid),
    "n_invalid": int(n_invalid),
    "pct_valid": round(100*n_valid/len(gp), 2) if len(gp) else 0,
    "per_country": pc_valid.to_dict(orient="records"),
}

# Figure: validation rates + scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_pc = pc_valid.head(15).iloc[::-1]
axes[0].barh(top_pc["country"], top_pc["pct_valid"], color=PAL[2])
axes[0].set_xlabel("% coordinates falling within country bounding box")
axes[0].set_title("Geographic provenance validation (top 15 countries)")
axes[0].axvline(95, color="green", linestyle="--", alpha=0.4, label="95% threshold")
axes[0].legend(frameon=False)

# scatter all points colored by valid/invalid
valid_pts = gp[gp["lat_lon_valid"]==True]
invalid_pts = gp[gp["lat_lon_valid"]==False]
axes[1].scatter(valid_pts["lon"], valid_pts["lat"], c=PAL[2], s=4, alpha=0.4, label=f"valid (n={n_valid:,})")
axes[1].scatter(invalid_pts["lon"], invalid_pts["lat"], c=PAL[5], s=8, alpha=0.7, label=f"invalid (n={n_invalid:,})")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
axes[1].set_title("Sampling locations with provenance flags")
axes[1].set_xlim(-20, 65); axes[1].set_ylim(0, 45)
axes[1].legend(frameon=False, loc="lower left")
plt.tight_layout()
save(fig, "Q02_geocoord_validation")

# ─── T1.3 duplicate detection ──────────────────────────────────────
print("[T1.3] Duplicate / suspect run detection")

# Same sample_title appearing in multiple BioProjects
dup = df.groupby("sample_title").agg(
    n_runs=("run_accession","count"),
    n_bioprojects=("bioproject","nunique"),
    bps=("bioproject", lambda x: ";".join(sorted(set(x.dropna().astype(str)))))
).reset_index()
dup_suspect = dup[(dup["n_bioprojects"]>=2) & (dup["sample_title"].str.len()>5)
                  & (~dup["sample_title"].str.lower().isin(["sample","unknown","na","not collected"]))]
dup_suspect = dup_suspect.sort_values("n_bioprojects", ascending=False).head(50)
xlsx(dup_suspect, "T1_duplicate_suspects")

OUT["duplicates"] = {
    "n_unique_titles": int(df["sample_title"].nunique()),
    "n_titles_in_multiple_bps": int(((dup["n_bioprojects"]>=2) & (dup["sample_title"].str.len()>5)).sum()),
    "top_suspects": dup_suspect.head(20).to_dict(orient="records"),
}

# ─── T1.4 temporal lag ─────────────────────────────────────────────
print("[T1.4] Submission temporal lag (collection → first_public)")

def parse_date(s):
    if pd.isna(s): return None
    s = str(s)[:10]
    for fmt in ("%Y-%m-%d","%Y/%m/%d","%Y"):
        try: return pd.to_datetime(s, format=fmt)
        except: pass
    try: return pd.to_datetime(s, errors="coerce")
    except: return None

df["collection_dt"] = df["collection_date"].apply(parse_date)
df["public_dt"] = df["first_public"].apply(parse_date)
mask_dt = df["collection_dt"].notna() & df["public_dt"].notna()
df.loc[mask_dt, "lag_days"] = (df.loc[mask_dt,"public_dt"] - df.loc[mask_dt,"collection_dt"]).dt.days
# clip absurd values
lag = df.loc[mask_dt & (df["lag_days"]>=0) & (df["lag_days"]<=365*15), "lag_days"]
lag_per_country = df[mask_dt & (df["lag_days"]>=0) & (df["lag_days"]<=365*15)].groupby("country").agg(
    n=("lag_days","count"),
    median_days=("lag_days","median"),
    mean_days=("lag_days","mean"),
    q25=("lag_days", lambda x: x.quantile(0.25)),
    q75=("lag_days", lambda x: x.quantile(0.75)),
).round(0).reset_index().sort_values("median_days")
xlsx(lag_per_country, "T1_temporal_lag")

OUT["temporal_lag"] = {
    "n_with_dates": int(mask_dt.sum()),
    "overall_median_days": float(lag.median()) if len(lag) else None,
    "overall_mean_days": float(lag.mean()) if len(lag) else None,
    "per_country": lag_per_country.to_dict(orient="records"),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(lag.values/365.25, bins=40, color=PAL[0], edgecolor="white")
axes[0].axvline(lag.median()/365.25, color=PAL[5], linestyle="--", lw=2, label=f"median={lag.median()/365.25:.2f} y")
axes[0].set_xlabel("Lag (years between collection and public release)")
axes[0].set_ylabel("Number of runs")
axes[0].set_title(f"Submission lag distribution (n={len(lag):,})")
axes[0].legend(frameon=False)

lpc = lag_per_country[lag_per_country["n"]>=20].head(20)
axes[1].barh(lpc["country"], lpc["median_days"]/365.25, color=PAL[1],
             xerr=[(lpc["median_days"]-lpc["q25"])/365.25, (lpc["q75"]-lpc["median_days"])/365.25],
             error_kw={"alpha":0.5})
axes[1].set_xlabel("Median lag (years), with IQR")
axes[1].set_title("Per-country submission latency (n≥20)")
plt.tight_layout()
save(fig, "Q03_temporal_lag")

print(f"  overall median lag: {lag.median():.0f} days ({lag.median()/365.25:.2f} years)")

# ════════════════════════════════════════════════════════════════════
# TIER 2 — ECOLOGICAL DIVERSITY
# ════════════════════════════════════════════════════════════════════
print("\n[T2.1] Diversity indices on metagenome-type composition")

# country × scientific_name matrix (counts)
ct_mat = df.groupby(["country","scientific_name"]).size().unstack(fill_value=0)
print(f"  matrix: {ct_mat.shape[0]} countries × {ct_mat.shape[1]} metagenome types")

def shannon(x):
    p = x[x>0] / x.sum()
    return float(-(p * np.log(p)).sum())
def simpson(x):
    p = x[x>0] / x.sum()
    return float(1 - (p**2).sum())
def chao1(x):
    obs = (x>0).sum()
    f1 = (x==1).sum(); f2 = (x==2).sum()
    if f2 == 0: return float(obs + f1*(f1-1)/2)
    return float(obs + f1**2/(2*f2))

div = pd.DataFrame({
    "country": ct_mat.index,
    "n_runs": ct_mat.sum(axis=1).values,
    "richness": (ct_mat>0).sum(axis=1).values,
    "shannon_H": [shannon(row) for row in ct_mat.values],
    "simpson_D": [simpson(row) for row in ct_mat.values],
    "chao1": [chao1(row) for row in ct_mat.values],
}).sort_values("shannon_H", ascending=False).round(3)
xlsx(div, "T2_diversity_indices")

OUT["diversity"] = {"per_country": div.to_dict(orient="records")}

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
d = div.set_index("country")
axes[0].barh(d.index[::-1], d["shannon_H"][::-1], color=PAL[2])
axes[0].set_xlabel("Shannon H'"); axes[0].set_title("Shannon diversity per country")
axes[1].barh(d.index[::-1], d["simpson_D"][::-1], color=PAL[3])
axes[1].set_xlabel("Simpson 1−D"); axes[1].set_title("Simpson diversity")
axes[2].barh(d.index[::-1], d["richness"][::-1], color=PAL[1])
axes[2].set_xlabel("Observed richness"); axes[2].set_title("Metagenome-type richness")
plt.tight_layout()
save(fig, "E01_diversity_indices")

# ─── T2.2 rarefaction ─────────────────────────────────────────────
print("[T2.2] Rarefaction curves")

def rarefy_curve(counts, max_n=None, step=20, n_iter=20, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    counts = np.array(counts); counts = counts[counts>0]
    pool = np.repeat(np.arange(len(counts)), counts)
    total = len(pool)
    if max_n is None: max_n = total
    sizes = list(range(1, min(max_n, total)+1, step))
    if sizes[-1] != min(max_n, total): sizes.append(min(max_n, total))
    means, sds = [], []
    for n in sizes:
        if n >= total:
            means.append(len(np.unique(pool))); sds.append(0); continue
        v = []
        for _ in range(n_iter):
            samp = rng.choice(pool, size=n, replace=False)
            v.append(len(np.unique(samp)))
        means.append(np.mean(v)); sds.append(np.std(v))
    return sizes, means, sds

rar_data = {}
top_countries = div.sort_values("n_runs", ascending=False).head(10)["country"].tolist()
for c in top_countries:
    counts = ct_mat.loc[c].values
    if counts.sum() < 10: continue
    sizes, means, sds = rarefy_curve(counts, max_n=min(2000, int(counts.sum())), step=50)
    rar_data[c] = {"sizes":sizes, "means":means, "sds":sds}

OUT["rarefaction"] = {c:{"sizes":d["sizes"], "richness":[round(x,2) for x in d["means"]]}
                       for c,d in rar_data.items()}

fig, ax = plt.subplots(figsize=(10, 6))
for i,(c,d) in enumerate(rar_data.items()):
    ax.plot(d["sizes"], d["means"], color=PAL[i%len(PAL)], lw=1.8, label=c)
    ax.fill_between(d["sizes"],
                    np.array(d["means"])-np.array(d["sds"]),
                    np.array(d["means"])+np.array(d["sds"]),
                    color=PAL[i%len(PAL)], alpha=0.15)
ax.set_xlabel("Number of runs sampled")
ax.set_ylabel("Expected metagenome-type richness")
ax.set_title("Rarefaction curves — top 10 countries by run volume")
ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower right")
save(fig, "E02_rarefaction")

# ─── T2.3 Beta diversity (Jaccard) + dendrogram + PCoA ────────────
print("[T2.3] Beta diversity (Jaccard) + clustering + PCoA")

# binary presence matrix
pa = (ct_mat > 0).astype(int)
# only countries with ≥30 runs and ≥10 distinct types for stable estimates
keep = (ct_mat.sum(axis=1) >= 30) & ((ct_mat>0).sum(axis=1) >= 10)
pa_k = pa.loc[keep]
print(f"  retained {len(pa_k)} countries for beta-diversity")

j = pdist(pa_k.values, metric="jaccard")
jdf = pd.DataFrame(squareform(j), index=pa_k.index, columns=pa_k.index)
xlsx(jdf.reset_index(), "T2_jaccard_distance_matrix")

# hierarchical clustering
Z = linkage(j, method="average")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
dendrogram(Z, labels=pa_k.index.tolist(), ax=axes[0], leaf_rotation=45,
           color_threshold=0.7*max(Z[:,2]), above_threshold_color="#888")
axes[0].set_ylabel("Jaccard distance")
axes[0].set_title("Country clustering by metagenome-type composition (UPGMA)")

# PCoA (classical MDS) on Jaccard
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
coords = mds.fit_transform(squareform(j))
for i, c in enumerate(pa_k.index):
    axes[1].scatter(coords[i,0], coords[i,1], s=80,
                    c=PAL[i%len(PAL)], edgecolors="black", linewidth=0.5)
    axes[1].annotate(c, (coords[i,0], coords[i,1]), fontsize=8,
                     xytext=(4,4), textcoords="offset points")
axes[1].set_xlabel("PCo1"); axes[1].set_ylabel("PCo2")
axes[1].set_title(f"PCoA on Jaccard distance (stress={mds.stress_:.3f})")
axes[1].axhline(0, color="gray", lw=0.5); axes[1].axvline(0, color="gray", lw=0.5)
plt.tight_layout()
save(fig, "E03_beta_diversity")

OUT["beta_diversity"] = {
    "n_countries": int(len(pa_k)),
    "jaccard_mean": float(np.mean(j)),
    "jaccard_min": float(np.min(j)),
    "jaccard_max": float(np.max(j)),
    "pcoa_coords": [{"country":c, "x":round(float(coords[i,0]),3), "y":round(float(coords[i,1]),3)}
                    for i,c in enumerate(pa_k.index)],
    "pcoa_stress": float(round(mds.stress_, 4)),
    "linkage_matrix": Z.tolist(),
    "linkage_labels": pa_k.index.tolist(),
}

# ════════════════════════════════════════════════════════════════════
# TIER 3 — INFERENTIAL STATISTICS
# ════════════════════════════════════════════════════════════════════
print("\n[T3.1] PERMANOVA: country effect on metagenome-type composition")

# Permutational multivariate ANOVA
# Test H0: country has no effect on metagenome-type composition
def permanova(D, groups, n_perm=999, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    groups = np.asarray(groups)
    n = len(groups)
    unique_groups = np.unique(groups)
    a = len(unique_groups)
    SST = (D**2).sum() / (2*n)
    SSW = 0
    for g in unique_groups:
        idx = np.where(groups==g)[0]
        nb = len(idx)
        if nb < 2: continue
        sub = D[np.ix_(idx, idx)]
        SSW += (sub**2).sum() / (2*nb)
    SSA = SST - SSW
    F_obs = (SSA/(a-1)) / (SSW/(n-a))
    R2 = SSA / SST if SST > 0 else 0
    # permutations
    F_perm = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(groups)
        SSW_p = 0
        for g in unique_groups:
            idx = np.where(perm==g)[0]
            nb = len(idx)
            if nb < 2: continue
            sub = D[np.ix_(idx, idx)]
            SSW_p += (sub**2).sum() / (2*nb)
        SSA_p = SST - SSW_p
        F_perm[i] = (SSA_p/(a-1)) / (SSW_p/(n-a)) if SSW_p > 0 else 0
    p = (np.sum(F_perm >= F_obs) + 1) / (n_perm + 1)
    return F_obs, R2, p

# To run PERMANOVA at run level, we need a distance matrix between RUNS — too large.
# Instead, run at COUNTRY level isn't meaningful (groups of size 1).
# Solution: aggregate by BIOPROJECT (each BP = one community), then PERMANOVA with country as group.
print("  aggregating to BioProject level for PERMANOVA")
bp_mat = df.groupby(["bioproject","scientific_name"]).size().unstack(fill_value=0)
bp_country = df.groupby("bioproject")["country"].agg(lambda x: x.mode().iloc[0])
# keep BPs with ≥3 runs
bp_runs = df.groupby("bioproject").size()
keep_bp = bp_runs[bp_runs >= 3].index
bp_mat = bp_mat.loc[bp_mat.index.isin(keep_bp)]
bp_country = bp_country.loc[bp_country.index.isin(bp_mat.index)]
# Keep countries with ≥3 BPs for the test
ck = bp_country.value_counts()
keep_countries = ck[ck >= 3].index.tolist()
bp_mat = bp_mat.loc[bp_country.isin(keep_countries)]
bp_country = bp_country.loc[bp_country.isin(keep_countries)]
# subsample to ≤500 BPs for compute
if len(bp_mat) > 500:
    rng_pm = np.random.RandomState(42)
    idx_sub = rng_pm.choice(len(bp_mat), 500, replace=False)
    bp_mat = bp_mat.iloc[idx_sub]
    bp_country = bp_country.iloc[idx_sub]
print(f"  PERMANOVA on {len(bp_mat)} BioProjects across {bp_country.nunique()} countries")

# Bray-Curtis distance on BP × scientific_name matrix
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
print("  computing Bray-Curtis...")
D_bc = bray_curtis(bp_mat.values)
print("  running PERMANOVA (999 permutations)...")
F_obs, R2, p_perm = permanova(D_bc, bp_country.values, n_perm=999)
print(f"  F={F_obs:.2f}, R²={R2:.3f}, p={p_perm:.4f}")

OUT["permanova"] = {
    "n_bioprojects": int(len(bp_mat)),
    "n_countries": int(bp_country.nunique()),
    "F_statistic": round(float(F_obs), 2),
    "R_squared": round(float(R2), 4),
    "p_value": round(float(p_perm), 5),
    "n_permutations": 999,
    "distance_metric": "Bray-Curtis",
    "interpretation": "country explains a significant fraction of variance in metagenome-type composition" if p_perm<0.05 else "no significant country effect",
}

# ─── T3.2 Indicator metagenome types per country ──────────────────
print("[T3.2] Indicator analysis: which metagenome types are over-represented per country")

# IndVal-style: A_ij = mean abundance of type i in country j / sum across countries
#                B_ij = fraction of samples in country j containing type i
# IndVal = sqrt(A * B), permutation test for significance
def indval(M, groups, n_perm=199, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    M = np.asarray(M, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_taxa = M.shape[1]
    iv = np.zeros((n_taxa, len(unique)))
    for j, g in enumerate(unique):
        mask = groups==g
        n_in = mask.sum()
        sub = M[mask]
        # A: relative abundance specificity
        mean_in = sub.mean(axis=0)
        mean_total = np.array([M[groups==gg].mean(axis=0) for gg in unique]).sum(axis=0)
        A = np.where(mean_total>0, mean_in/mean_total, 0)
        # B: presence fidelity
        B = (sub>0).sum(axis=0) / n_in if n_in>0 else 0
        iv[:,j] = np.sqrt(A * B)
    # permutation test (per taxon, max IV across groups)
    obs_max = iv.max(axis=1)
    obs_arg = iv.argmax(axis=1)
    p_vals = np.zeros(n_taxa)
    for k in range(n_perm):
        perm = rng.permutation(groups)
        iv_p = np.zeros((n_taxa, len(unique)))
        for j, g in enumerate(unique):
            mask = perm==g
            n_in = mask.sum()
            sub = M[mask]
            mean_in = sub.mean(axis=0)
            mean_total = np.array([M[perm==gg].mean(axis=0) for gg in unique]).sum(axis=0)
            A = np.where(mean_total>0, mean_in/mean_total, 0)
            B = (sub>0).sum(axis=0) / n_in if n_in>0 else 0
            iv_p[:,j] = np.sqrt(A * B)
        max_p = iv_p.max(axis=1)
        p_vals += (max_p >= obs_max).astype(int)
    p_vals = (p_vals + 1) / (n_perm + 1)
    return iv, obs_max, obs_arg, p_vals, unique

print("  running IndVal (199 permutations)...")
# Use country × scientific_name matrix at run level, but only keep abundant types
type_totals = ct_mat.sum(axis=0).sort_values(ascending=False)
top_types = type_totals.head(80).index.tolist()
ct_top = ct_mat[top_types]
# group = country at country level — but IndVal needs replicates. So go back to BP level.
M_bp = bp_mat.reindex(columns=top_types, fill_value=0).values
iv_mat, iv_max, iv_arg, iv_p, groups_arr = indval(M_bp, bp_country.values, n_perm=199)

indicator_table = []
for i, t in enumerate(top_types):
    if iv_p[i] < 0.05 and iv_max[i] > 0.3:
        indicator_table.append({
            "metagenome_type": t,
            "indicator_country": str(groups_arr[iv_arg[i]]),
            "indicator_value": round(float(iv_max[i]), 3),
            "p_value": round(float(iv_p[i]), 4),
        })
indicator_df = pd.DataFrame(indicator_table).sort_values("indicator_value", ascending=False)
xlsx(indicator_df, "T3_indicator_metagenome_types")

OUT["indicator_species"] = {
    "n_significant": int(len(indicator_df)),
    "alpha": 0.05,
    "min_indval": 0.3,
    "n_permutations": 199,
    "top": indicator_df.head(30).to_dict(orient="records"),
}
print(f"  {len(indicator_df)} significant indicator types")

# ─── T3.3 Chi-square standardized residuals ────────────────────────
print("[T3.3] Chi-square standardized residuals (country × broad_category)")

cat_ct = pd.crosstab(df["country"], df["broad_category"])
chi2, p_chi, dof, expected = spstats.chi2_contingency(cat_ct)
# standardized residuals
std_res = (cat_ct.values - expected) / np.sqrt(expected)
res_df = pd.DataFrame(std_res, index=cat_ct.index, columns=cat_ct.columns)
xlsx(res_df.reset_index(), "T3_chi2_residuals")

OUT["chi_square_residuals"] = {
    "chi2": round(float(chi2), 2),
    "dof": int(dof),
    "p": float(p_chi),
    "residuals": res_df.round(2).reset_index().to_dict(orient="records"),
    "interpretation": "Cells with |z|>2 indicate significant over-representation (positive) or under-representation (negative). |z|>3 highly significant."
}

fig, ax = plt.subplots(figsize=(11, 9))
vmax = max(abs(std_res.min()), abs(std_res.max()))
im = ax.imshow(std_res, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(cat_ct.columns)))
ax.set_xticklabels(cat_ct.columns, rotation=45, ha="right")
ax.set_yticks(range(len(cat_ct.index)))
ax.set_yticklabels(cat_ct.index)
for i in range(len(cat_ct.index)):
    for j in range(len(cat_ct.columns)):
        v = std_res[i,j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v)>3 else "black")
plt.colorbar(im, ax=ax, label="Standardized residual (z-score)")
ax.set_title(f"Chi-square standardized residuals — country × category\nχ²={chi2:.0f}, dof={dof}, p<{max(p_chi,1e-300):.0e}")
plt.tight_layout()
save(fig, "I01_chi2_residuals")

# ─── T3.4 Mann-Kendall trend per country ──────────────────────────
print("[T3.4] Mann-Kendall temporal trend per country")

mk_results = []
for c in df["country"].unique():
    sub = df[df["country"]==c]
    sub = sub.dropna(subset=["year"])
    if len(sub) < 30: continue
    ts = sub.groupby(sub["year"].astype(int)).size()
    if len(ts) < 5: continue
    try:
        r = mk.original_test(ts.values)
        mk_results.append({
            "country": c,
            "n_years": int(len(ts)),
            "trend": r.trend,
            "tau": round(float(r.Tau), 3),
            "p_value": round(float(r.p), 4),
            "slope_runs_per_year": round(float(r.slope), 1),
        })
    except: pass
mk_df = pd.DataFrame(mk_results).sort_values("p_value")
xlsx(mk_df, "T3_mann_kendall_trends")

OUT["mann_kendall"] = mk_df.to_dict(orient="records")
n_sig = (mk_df["p_value"]<0.05).sum()
print(f"  {n_sig}/{len(mk_df)} countries show significant monotonic trend")

# ════════════════════════════════════════════════════════════════════
# TIER 4 — GEOSPATIAL
# ════════════════════════════════════════════════════════════════════
print("\n[T4.1] Sampling density grid")

valid_geo = df[(df["lat_lon_valid"]==True)].copy()
valid_geo["lat_bin"] = (valid_geo["lat"]/1.0).round() * 1.0  # 1 degree bins
valid_geo["lon_bin"] = (valid_geo["lon"]/1.0).round() * 1.0
grid = valid_geo.groupby(["lat_bin","lon_bin"]).size().reset_index(name="n_runs")
xlsx(grid, "T4_sampling_density_grid")

OUT["geo_grid"] = {
    "n_cells": int(len(grid)),
    "n_runs_mapped": int(valid_geo.shape[0]),
    "max_density": int(grid["n_runs"].max()),
    "cells": grid.to_dict(orient="records"),
}

fig, ax = plt.subplots(figsize=(14, 7))
sc = ax.scatter(grid["lon_bin"], grid["lat_bin"], c=grid["n_runs"], s=np.sqrt(grid["n_runs"])*5,
                cmap="YlOrRd", alpha=0.85, edgecolors="black", linewidth=0.3)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("MENA metagenomic sampling density (1° grid cells)")
ax.set_xlim(-20, 65); ax.set_ylim(0, 45)
ax.grid(alpha=0.2)
plt.colorbar(sc, ax=ax, label="Runs per cell")
save(fig, "G01_sampling_density")

# ─── T4.2 Moran's I (global spatial autocorrelation) ───────────────
print("[T4.2] Moran's I — spatial autocorrelation of sampling density")

def morans_I(values, coords, k=8):
    n = len(values)
    # k-NN weights (binary, row-standardized)
    from scipy.spatial.distance import cdist
    D = cdist(coords, coords)
    W = np.zeros((n,n))
    for i in range(n):
        nn_idx = np.argsort(D[i])[1:k+1]
        W[i, nn_idx] = 1
    # row standardize
    rs = W.sum(axis=1, keepdims=True)
    W = np.where(rs>0, W/rs, 0)
    z = values - values.mean()
    num = (W * np.outer(z, z)).sum()
    den = (z**2).sum()
    I = (n / W.sum()) * (num / den) if den>0 else 0
    # expected I under null
    E_I = -1.0/(n-1)
    # variance under randomization (simplified)
    s1 = ((W + W.T)**2).sum() / 2
    s2 = ((W.sum(axis=1) + W.sum(axis=0))**2).sum()
    var_I = (n*((n**2 - 3*n + 3)*s1 - n*s2 + 3*W.sum()**2) -
             ((z**2).sum()/n)**(-2) * ((n**2 - n)*s1 - 2*n*s2 + 6*W.sum()**2)) / \
            ((n-1)*(n-2)*(n-3)*W.sum()**2)
    var_I = max(var_I, 1e-10)
    z_score = (I - E_I) / np.sqrt(var_I)
    p = 2 * (1 - spstats.norm.cdf(abs(z_score)))
    return I, E_I, z_score, p

if len(grid) > 10:
    coords_arr = grid[["lon_bin","lat_bin"]].values
    vals_arr = grid["n_runs"].values.astype(float)
    I, E_I, z_I, p_I = morans_I(vals_arr, coords_arr, k=min(8, len(grid)-1))
    OUT["morans_I"] = {
        "I": round(float(I), 4),
        "expected_I": round(float(E_I), 4),
        "z_score": round(float(z_I), 3),
        "p_value": float(p_I),
        "n_cells": int(len(grid)),
        "interpretation": "significant positive spatial clustering" if p_I<0.05 and I>E_I else
                          "significant dispersion" if p_I<0.05 and I<E_I else "random spatial pattern"
    }
    print(f"  Moran's I = {I:.4f} (E={E_I:.4f}), z={z_I:.2f}, p={p_I:.4e}")

# ════════════════════════════════════════════════════════════════════
# TIER 5 — NLP DONE PROPERLY (per-country TF-IDF + per-study clustering)
# ════════════════════════════════════════════════════════════════════
print("\n[T5.1] Per-country TF-IDF discriminating terms")

# concatenate all study titles per country
country_docs = df.drop_duplicates("study_accession").groupby("country").agg(
    text=("study_title", lambda x: " ".join(x.dropna().astype(str)))
).reset_index()
country_docs = country_docs[country_docs["text"].str.len() > 30]

STOP = set("""a an and or the of to in on for with by from as at is are was were be been being
this that these those it its their there here which who whom whose what when where why how
study studies analysis sequencing sample samples using based high throughput data dataset project
next generation sequence sequences whole gene genes
shotgun illumina rrna via between among across within during we our using used use
sp nov strain isolate isolates isolated genome profiling profile
metagenomic metagenome metagenomics microbiome microbiota microbial
report draft assembly characterization investigation evaluation""".split())

tv = TfidfVectorizer(max_df=0.7, min_df=2, stop_words=list(STOP),
                     max_features=2000, ngram_range=(1,2))
M = tv.fit_transform(country_docs["text"])
vocab = tv.get_feature_names_out()
country_tfidf = {}
for i, c in enumerate(country_docs["country"]):
    row = M[i].toarray().ravel()
    top = np.argsort(-row)[:15]
    country_tfidf[c] = [{"term":vocab[j], "score":round(float(row[j]),4)} for j in top if row[j]>0]

OUT["country_tfidf"] = country_tfidf

# ─── T5.2 Study-level clustering on full study_title corpus ────────
print("[T5.2] Per-study K-Means clustering")

study_docs = df.drop_duplicates("study_accession")[["study_accession","study_title","country","broad_category"]].dropna(subset=["study_title"])
study_docs = study_docs[study_docs["study_title"].str.len() > 15]
print(f"  {len(study_docs)} studies for clustering")

tv2 = TfidfVectorizer(max_df=0.5, min_df=3, stop_words=list(STOP),
                      max_features=1000, ngram_range=(1,2))
S = tv2.fit_transform(study_docs["study_title"])
svd = TruncatedSVD(n_components=20, random_state=42)
Z = svd.fit_transform(S)
print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

# Choose K by silhouette
from sklearn.metrics import silhouette_score
sil_scores = []
for k in range(3, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
    sil = silhouette_score(Z, km.labels_, sample_size=min(2000, len(Z)), random_state=42)
    sil_scores.append({"k":k, "silhouette":round(float(sil),4)})
best_k = max(sil_scores, key=lambda x: x["silhouette"])["k"]
print(f"  best K by silhouette: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(Z)
study_docs["cluster"] = km_final.labels_

cluster_summary = []
vocab2 = tv2.get_feature_names_out()
# top terms per cluster: mean TF-IDF in cluster
for k in range(best_k):
    mask = km_final.labels_ == k
    if mask.sum() == 0: continue
    mean_tfidf = np.asarray(S[mask].mean(axis=0)).ravel()
    top_idx = np.argsort(-mean_tfidf)[:8]
    top_terms = [vocab2[i] for i in top_idx]
    # top countries in cluster
    top_countries = study_docs.loc[mask, "country"].value_counts().head(5).to_dict()
    cluster_summary.append({
        "cluster": int(k),
        "size": int(mask.sum()),
        "top_terms": top_terms,
        "top_countries": top_countries,
    })

OUT["nlp_clustering"] = {
    "n_studies": int(len(study_docs)),
    "k": int(best_k),
    "silhouette_scores": sil_scores,
    "best_silhouette": round(float(max(sil_scores, key=lambda x: x["silhouette"])["silhouette"]), 4),
    "clusters": cluster_summary,
}

# Cluster × country contingency for chi-square
ct_cluster = pd.crosstab(study_docs["country"], study_docs["cluster"])
chi2_cl, p_cl, dof_cl, _ = spstats.chi2_contingency(ct_cluster)
OUT["nlp_cluster_country_chi2"] = {
    "chi2": round(float(chi2_cl), 2),
    "dof": int(dof_cl),
    "p": float(p_cl),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot([s["k"] for s in sil_scores], [s["silhouette"] for s in sil_scores],
             "o-", color=PAL[1], lw=2)
axes[0].axvline(best_k, color="green", linestyle="--", alpha=0.5, label=f"best k={best_k}")
axes[0].set_xlabel("Number of clusters (k)"); axes[0].set_ylabel("Silhouette score")
axes[0].set_title("Optimal cluster count selection")
axes[0].legend(frameon=False)

# cluster size + top term
cs = sorted(cluster_summary, key=lambda x: -x["size"])
axes[1].barh([f"C{c['cluster']}: {c['top_terms'][0]}" for c in cs[::-1]],
             [c["size"] for c in cs[::-1]], color=PAL[3])
axes[1].set_xlabel("Number of studies")
axes[1].set_title("Semantic clusters with top discriminating term")
plt.tight_layout()
save(fig, "N01_nlp_clustering")

# ════════════════════════════════════════════════════════════════════
# Save consolidated results JSON
# ════════════════════════════════════════════════════════════════════
with open(f"{DATADIR}/mena_rigorous_results.json", "w") as f:
    import math
    def _safe(x):
        if isinstance(x,float) and (math.isnan(x) or math.isinf(x)): return None
        return str(x)
    json.dump(OUT, f, default=_safe)

import os
print(f"\nDONE. Results JSON: {os.path.getsize(f'{DATADIR}/mena_rigorous_results.json')/1024:.1f} KB")
print("\nKey publishable results:")
print(f"  · MIxS completeness: mean={OUT['mixs']['overall_mean_pct']}%, median={OUT['mixs']['overall_median_pct']}%")
print(f"  · GPS coords: {OUT['geocoord']['pct_with_coords']}% present, {OUT['geocoord']['pct_valid']}% valid")
print(f"  · Submission lag: median={OUT['temporal_lag']['overall_median_days']:.0f} days")
print(f"  · PERMANOVA: F={OUT['permanova']['F_statistic']}, R²={OUT['permanova']['R_squared']}, p={OUT['permanova']['p_value']}")
print(f"  · Indicator types: {OUT['indicator_species']['n_significant']} significant")
print(f"  · Mann-Kendall: {sum(1 for x in OUT['mann_kendall'] if x['p_value']<0.05)}/{len(OUT['mann_kendall'])} sig trends")
print(f"  · Moran's I: {OUT['morans_I']['I']} ({OUT['morans_I']['interpretation']})")
print(f"  · NLP clusters: k={OUT['nlp_clustering']['k']}, sil={OUT['nlp_clustering']['best_silhouette']}")
