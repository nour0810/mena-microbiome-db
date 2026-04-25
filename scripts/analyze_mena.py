"""
MENA Microbiome Database — downstream analysis
Produces publication-quality figures (PNG+SVG) and tables (xlsx) from mena_metagenomics.tsv
"""
import os, json, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter

DATA = "/home/claude/mena microbiome database/mena_metagenomics.tsv"
FIGDIR = "/home/claude/out/figures"
TBLDIR = "/home/claude/out/tables"
DATADIR = "/home/claude/out/data"

# ── publication style ────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# color-blind-friendly palette (Okabe-Ito)
PAL = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442","#999999","#000000"]

MENA_CANON = {
    "Saudi Arabia","Turkey","Iran","UAE","Egypt","Morocco","Tunisia","Qatar","Jordan",
    "Lebanon","Oman","Sudan","Syria","Algeria","Mauritania","Iraq","Kuwait","Djibouti",
    "Libya","Yemen","Bahrain","Palestine","Somalia","South Sudan",
}

def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/{name}.svg", bbox_inches="tight")
    plt.close(fig)

def xlsx(df, name):
    df.to_excel(f"{TBLDIR}/{name}.xlsx", index=False)

# ── LOAD & CLEAN ─────────────────────────────────────────────────────────────
print("Loading...")
df = pd.read_csv(DATA, sep="\t", low_memory=False, on_bad_lines="skip")
print(f"  raw: {len(df)} runs × {df.shape[1]} cols")

# Canonicalize country: keep only true MENA, collapse messy strings
def clean_country(c):
    if pd.isna(c) or not c: return None
    c = str(c).strip()
    # handle multi-country joined with ;
    parts = re.split(r"[;/]", c)
    for p in parts:
        p = p.strip()
        # exact match
        if p in MENA_CANON: return p
        # contains a MENA country name
        for m in MENA_CANON:
            if m.lower() in p.lower():
                return m
        # uppercase TUNISIA etc.
        if p.upper() in {m.upper() for m in MENA_CANON}:
            return next(m for m in MENA_CANON if m.upper()==p.upper())
    return None

df["country"] = df["country_clean"].apply(clean_country)
df_mena = df[df["country"].notna()].copy()
print(f"  MENA-only: {len(df_mena)} runs across {df_mena['country'].nunique()} countries")

# year
def extract_year(row):
    for col in ["first_public","collection_date","last_updated"]:
        v = row.get(col)
        if pd.isna(v): continue
        m = re.search(r"(19|20)\d{2}", str(v))
        if m:
            y = int(m.group())
            if 1990 <= y <= 2026: return y
    return None
df_mena["year"] = df_mena.apply(extract_year, axis=1)

# save cleaned MENA-only set
df_mena.to_csv(f"{DATADIR}/mena_metagenomics_clean.tsv", sep="\t", index=False)

# ── 1. GEOGRAPHIC DISTRIBUTION ───────────────────────────────────────────────
print("\n[1] Geographic distribution")
geo = df_mena["country"].value_counts().reset_index()
geo.columns = ["country","n_runs"]
geo["n_bioprojects"] = df_mena.groupby("country")["bioproject"].nunique().reindex(geo["country"]).values
geo["n_samples"] = df_mena.groupby("country")["sample_accession"].nunique().reindex(geo["country"]).values
xlsx(geo, "01_geographic_distribution")

fig, ax = plt.subplots(figsize=(9,7))
geo_s = geo.sort_values("n_runs")
bars = ax.barh(geo_s["country"], geo_s["n_runs"], color=PAL[0], edgecolor="white")
ax.set_xlabel("Number of metagenomic runs")
ax.set_title("Metagenomic runs per MENA country")
for b, n in zip(bars, geo_s["n_runs"]):
    ax.text(b.get_width()+max(geo_s["n_runs"])*0.008, b.get_y()+b.get_height()/2,
            f"{n:,}", va="center", fontsize=8)
ax.set_xlim(0, geo_s["n_runs"].max()*1.12)
save(fig, "01_geographic_distribution")

# ── 2. TEMPORAL TRENDS ───────────────────────────────────────────────────────
print("[2] Temporal trends")
yr = df_mena["year"].dropna().astype(int)
yr_counts = yr.value_counts().sort_index()
yr_counts = yr_counts[yr_counts.index >= 2008]
xlsx(yr_counts.reset_index().rename(columns={"index":"year","year":"year","count":"n_runs"}), "02_temporal_trends")

fig, ax = plt.subplots(figsize=(9,5))
ax.bar(yr_counts.index, yr_counts.values, color=PAL[2], edgecolor="white")
ax.plot(yr_counts.index, yr_counts.cumsum().values * (yr_counts.max()/yr_counts.cumsum().max()),
        color=PAL[5], lw=2, label="Cumulative (rescaled)")
ax.set_xlabel("Year of public release")
ax.set_ylabel("Number of runs")
ax.set_title("Temporal growth of MENA metagenomic submissions")
ax.legend(loc="upper left", frameon=False)
save(fig, "02_temporal_trends")

# stacked by country over time
yr_country = df_mena.dropna(subset=["year"]).groupby(["year","country"]).size().unstack(fill_value=0)
yr_country = yr_country[yr_country.index >= 2010]
top6 = df_mena["country"].value_counts().head(6).index.tolist()
yr_country_plot = yr_country[top6].copy()
yr_country_plot["Other MENA"] = yr_country.drop(columns=top6, errors="ignore").sum(axis=1)

fig, ax = plt.subplots(figsize=(10,5.5))
yr_country_plot.plot(kind="bar", stacked=True, ax=ax, color=PAL[:len(yr_country_plot.columns)], width=0.85)
ax.set_xlabel("Year"); ax.set_ylabel("Number of runs")
ax.set_title("Temporal trends by country (top 6 contributors)")
ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
plt.xticks(rotation=45)
save(fig, "02b_temporal_by_country")

# ── 3. SAMPLE TYPE / ENVIRONMENT ─────────────────────────────────────────────
print("[3] Sample type distribution")
cat = df_mena["broad_category"].value_counts()
spec = df_mena["specific_category"].value_counts()
xlsx(cat.reset_index(), "03a_broad_categories")
xlsx(spec.reset_index().head(30), "03b_specific_categories")

fig, axes = plt.subplots(1,2, figsize=(13,6))
axes[0].barh(cat.index[::-1], cat.values[::-1], color=PAL[:len(cat)])
axes[0].set_xlabel("Number of runs"); axes[0].set_title("Broad sample categories")
for i,(l,v) in enumerate(zip(cat.index[::-1], cat.values[::-1])):
    axes[0].text(v+max(cat.values)*0.01, i, f"{v:,}", va="center", fontsize=9)

top_spec = spec.head(15)
axes[1].barh(top_spec.index[::-1], top_spec.values[::-1], color=PAL[1])
axes[1].set_xlabel("Number of runs"); axes[1].set_title("Top 15 specific sample types")
for i,(l,v) in enumerate(zip(top_spec.index[::-1], top_spec.values[::-1])):
    axes[1].text(v+max(top_spec.values)*0.01, i, f"{v:,}", va="center", fontsize=8)
plt.tight_layout()
save(fig, "03_sample_types")

# ── 4. HOST DISTRIBUTION ─────────────────────────────────────────────────────
print("[4] Host distribution")
host_mask = df_mena["broad_category"].isin(["Human","Animal"])
host_cat = df_mena.loc[host_mask, ["broad_category","specific_category"]].value_counts().reset_index()
host_cat.columns = ["broad","specific","n_runs"]
xlsx(host_cat, "04_host_distribution")

# human body sites
hb = df_mena[df_mena["broad_category"]=="Human"]["specific_category"].value_counts()
an = df_mena[df_mena["broad_category"]=="Animal"]["specific_category"].value_counts()

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
axes[0].bar(hb.index, hb.values, color=PAL[3])
axes[0].set_title(f"Human body sites (n={hb.sum():,})")
axes[0].set_ylabel("Runs")
plt.setp(axes[0].get_xticklabels(), rotation=35, ha="right")
for i,v in enumerate(hb.values):
    axes[0].text(i, v+max(hb.values)*0.01, f"{v:,}", ha="center", fontsize=8)

axes[1].bar(an.index, an.values, color=PAL[2])
axes[1].set_title(f"Animal hosts (n={an.sum():,})")
axes[1].set_ylabel("Runs")
plt.setp(axes[1].get_xticklabels(), rotation=35, ha="right")
for i,v in enumerate(an.values):
    axes[1].text(i, v+max(an.values)*0.01, f"{v:,}", ha="center", fontsize=8)
plt.tight_layout()
save(fig, "04_host_distribution")

# ── 5. SEQUENCING PLATFORM ───────────────────────────────────────────────────
print("[5] Platform distribution")
plat = df_mena["instrument_platform"].value_counts()
xlsx(plat.reset_index(), "05_sequencing_platforms")

inst = df_mena["instrument_model"].value_counts().head(15)
xlsx(inst.reset_index(), "05b_instrument_models")

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
axes[0].pie(plat.values, labels=plat.index, colors=PAL, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize":9})
axes[0].set_title(f"Sequencing platforms (n={plat.sum():,})")

axes[1].barh(inst.index[::-1], inst.values[::-1], color=PAL[4])
axes[1].set_xlabel("Runs"); axes[1].set_title("Top 15 instrument models")
for i,v in enumerate(inst.values[::-1]):
    axes[1].text(v+max(inst.values)*0.01, i, f"{v:,}", va="center", fontsize=8)
plt.tight_layout()
save(fig, "05_platforms")

# ── 6. STUDY DESIGN ──────────────────────────────────────────────────────────
print("[6] Study design")
lib_strat = df_mena["library_strategy"].value_counts().head(12)
lib_src = df_mena["library_source"].value_counts().head(8)
lib_layout = df_mena["library_layout"].value_counts()
lib_sel = df_mena["library_selection"].value_counts().head(10)
xlsx(lib_strat.reset_index(), "06a_library_strategy")
xlsx(lib_src.reset_index(), "06b_library_source")

fig, axes = plt.subplots(2,2, figsize=(13,9))
axes[0,0].barh(lib_strat.index[::-1], lib_strat.values[::-1], color=PAL[0])
axes[0,0].set_title("Library strategy"); axes[0,0].set_xlabel("Runs")
axes[0,1].barh(lib_src.index[::-1], lib_src.values[::-1], color=PAL[1])
axes[0,1].set_title("Library source"); axes[0,1].set_xlabel("Runs")
axes[1,0].pie(lib_layout.values, labels=lib_layout.index, colors=PAL, autopct="%1.1f%%",
              startangle=90, textprops={"fontsize":9})
axes[1,0].set_title("Library layout")
axes[1,1].barh(lib_sel.index[::-1], lib_sel.values[::-1], color=PAL[2])
axes[1,1].set_title("Library selection"); axes[1,1].set_xlabel("Runs")
plt.tight_layout()
save(fig, "06_study_design")

# data subtype
sub = df_mena["data_subtype"].value_counts()
fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(sub.index, sub.values, color=PAL[:len(sub)])
ax.set_ylabel("Runs")
ax.set_title("Metagenomics data subtypes")
for i,v in enumerate(sub.values):
    ax.text(i, v+max(sub.values)*0.01, f"{v:,}", ha="center", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
save(fig, "06b_data_subtypes")

# ── 7. METADATA COMPLETENESS ─────────────────────────────────────────────────
print("[7] Metadata completeness")
key_fields = ["country","year","scientific_name","library_strategy","library_source",
              "instrument_platform","collection_date","host","host_sex","host_body_site",
              "environment_biome","isolation_source","read_count","base_count",
              "sample_host_disease","sample_host_age","sample_lat_lon","sample_dna_extraction"]
avail = [f for f in key_fields if f in df_mena.columns]

completeness = {}
for c in sorted(df_mena["country"].unique()):
    sub = df_mena[df_mena["country"]==c]
    row = {}
    for f in avail:
        valid = sub[f].notna() & (sub[f].astype(str).str.strip() != "")
        row[f] = 100 * valid.sum() / len(sub)
    completeness[c] = row
comp_df = pd.DataFrame(completeness).T
xlsx(comp_df.reset_index().rename(columns={"index":"country"}), "07_metadata_completeness")

fig, ax = plt.subplots(figsize=(13, 9))
im = ax.imshow(comp_df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(comp_df.columns)))
ax.set_xticklabels(comp_df.columns, rotation=45, ha="right")
ax.set_yticks(range(len(comp_df.index)))
ax.set_yticklabels(comp_df.index)
for i in range(len(comp_df.index)):
    for j in range(len(comp_df.columns)):
        v = comp_df.values[i,j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                color="black" if 30<v<80 else "white", fontsize=7)
plt.colorbar(im, ax=ax, label="% complete")
ax.set_title("Metadata field completeness per MENA country (%)")
plt.tight_layout()
save(fig, "07_metadata_completeness_heatmap")

# ── 8. KEYWORD / TOPIC CLUSTERING ────────────────────────────────────────────
print("[8] Keyword extraction from study titles")
titles = df_mena["study_title"].dropna().astype(str)
titles_uniq = df_mena.drop_duplicates("study_accession")["study_title"].dropna().astype(str)

# simple keyword extraction
STOP = set("""a an and or the of to in on for with by from as at is are was were be been being
this that these those it its their there here which who whom whose what when where why how
study studies analysis sequencing sample samples using based high throughput data dataset project
next generation rna dna genomic genomics sequence sequences whole gene genes metagenomic
shotgun illumina 16s rrna via between among across within during
sp nov strain isolate isolates isolated genome profiling profile""".split())

def tokens(s):
    return [w.lower() for w in re.findall(r"[A-Za-z]{4,}", s) if w.lower() not in STOP]

# unigrams
all_tok = []
for t in titles_uniq: all_tok.extend(tokens(t))
uni = Counter(all_tok).most_common(30)

# bigrams
def bigrams(s):
    ws = tokens(s)
    return [f"{ws[i]} {ws[i+1]}" for i in range(len(ws)-1)]
all_bi = []
for t in titles_uniq: all_bi.extend(bigrams(t))
bi = Counter(all_bi).most_common(25)

kw_df = pd.DataFrame(uni, columns=["term","count"])
bi_df = pd.DataFrame(bi, columns=["bigram","count"])
xlsx(kw_df, "08a_top_keywords")
xlsx(bi_df, "08b_top_bigrams")

fig, axes = plt.subplots(1,2, figsize=(14,7))
ku = kw_df.head(20)
axes[0].barh(ku["term"][::-1], ku["count"][::-1], color=PAL[5])
axes[0].set_title("Top 20 keywords in study titles")
axes[0].set_xlabel("Occurrences")
kb = bi_df.head(20)
axes[1].barh(kb["bigram"][::-1], kb["count"][::-1], color=PAL[6])
axes[1].set_title("Top 20 bigrams in study titles")
axes[1].set_xlabel("Occurrences")
plt.tight_layout()
save(fig, "08_keyword_clustering")

# ── 9. CROSS-TABS ────────────────────────────────────────────────────────────
print("[9] Cross-tabulations")
# country x broad_category
ct1 = pd.crosstab(df_mena["country"], df_mena["broad_category"])
ct1 = ct1.loc[ct1.sum(axis=1).sort_values(ascending=False).index]
xlsx(ct1.reset_index(), "09a_country_x_category")

fig, ax = plt.subplots(figsize=(11,8))
ct1_pct = ct1.div(ct1.sum(axis=1), axis=0) * 100
ct1_pct.plot(kind="barh", stacked=True, ax=ax, color=PAL[:ct1_pct.shape[1]], width=0.85)
ax.set_xlabel("% of runs"); ax.set_title("Sample category composition per country")
ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
ax.invert_yaxis()
save(fig, "09a_country_x_category")

# year x platform
ct2 = pd.crosstab(df_mena["year"].dropna().astype(int), df_mena["instrument_platform"])
ct2 = ct2[ct2.index >= 2010]
top_plats = df_mena["instrument_platform"].value_counts().head(5).index.tolist()
ct2_top = ct2[top_plats].copy()
ct2_top["Other"] = ct2.drop(columns=top_plats, errors="ignore").sum(axis=1)
xlsx(ct2.reset_index(), "09b_year_x_platform")

fig, ax = plt.subplots(figsize=(11,5.5))
ct2_top.plot(kind="bar", stacked=True, ax=ax, color=PAL[:ct2_top.shape[1]], width=0.85)
ax.set_xlabel("Year"); ax.set_ylabel("Runs")
ax.set_title("Sequencing platform adoption over time")
ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
plt.xticks(rotation=45)
save(fig, "09b_year_x_platform")

# country x data_subtype
ct3 = pd.crosstab(df_mena["country"], df_mena["data_subtype"])
ct3 = ct3.loc[ct3.sum(axis=1).sort_values(ascending=False).index]
xlsx(ct3.reset_index(), "09c_country_x_subtype")

fig, ax = plt.subplots(figsize=(10, 8))
ct3.plot(kind="barh", stacked=True, ax=ax, color=PAL[:3], width=0.85)
ax.set_xlabel("Runs"); ax.set_title("Metagenomics subtype per country")
ax.legend(frameon=False, loc="lower right")
ax.invert_yaxis()
save(fig, "09c_country_x_subtype")

# ── 10. ADDITIONAL: bioproject concentration, read depth ─────────────────────
print("[10] Additional analyses")

# runs per bioproject (study size distribution)
rpb = df_mena.groupby("bioproject").size()
fig, ax = plt.subplots(figsize=(9,5))
bins = [1,2,5,10,25,50,100,500,10000]
labels = ["1","2-4","5-9","10-24","25-49","50-99","100-499","500+"]
cat_rpb = pd.cut(rpb, bins=bins, labels=labels, right=False)
cat_counts = cat_rpb.value_counts().reindex(labels)
ax.bar(labels, cat_counts.values, color=PAL[7])
ax.set_xlabel("Runs per BioProject")
ax.set_ylabel("Number of BioProjects")
ax.set_title(f"BioProject size distribution (n={len(rpb):,} projects)")
for i,v in enumerate(cat_counts.values):
    if pd.notna(v):
        ax.text(i, v+max(cat_counts.dropna())*0.01, f"{int(v):,}", ha="center", fontsize=9)
save(fig, "10a_bioproject_size_distribution")
xlsx(cat_counts.reset_index(), "10a_bioproject_size_distribution")

# read count distribution
rc = pd.to_numeric(df_mena["read_count"], errors="coerce").dropna()
rc = rc[rc > 0]
fig, ax = plt.subplots(figsize=(9,5))
ax.hist(np.log10(rc), bins=50, color=PAL[0], edgecolor="white")
ax.set_xlabel("log10(read count)")
ax.set_ylabel("Number of runs")
ax.set_title(f"Read count distribution (median = {int(rc.median()):,})")
save(fig, "10b_read_count_distribution")

# top 20 bioprojects
top_bp = df_mena.groupby("bioproject").agg(
    n_runs=("run_accession","count"),
    country=("country", lambda x: x.mode().iloc[0] if len(x.mode())>0 else ""),
    category=("broad_category", lambda x: x.mode().iloc[0] if len(x.mode())>0 else ""),
    study_title=("study_title", lambda x: x.dropna().iloc[0] if x.notna().any() else ""),
).sort_values("n_runs", ascending=False).head(30).reset_index()
xlsx(top_bp, "10c_top_bioprojects")

# summary stats table
summary = {
    "Total metagenomic runs": len(df_mena),
    "Unique samples": df_mena["sample_accession"].nunique(),
    "Unique BioProjects": df_mena["bioproject"].nunique(),
    "MENA countries represented": df_mena["country"].nunique(),
    "Year range": f"{int(yr.min())}–{int(yr.max())}",
    "Shotgun metagenomics runs": int((df_mena["data_subtype"]=="shotgun_metagenomics").sum()),
    "Amplicon metagenomics runs": int((df_mena["data_subtype"]=="amplicon_metagenomics").sum()),
    "Metatranscriptomics runs": int((df_mena["data_subtype"]=="metatranscriptomics").sum()),
    "Human-associated runs": int((df_mena["broad_category"]=="Human").sum()),
    "Environmental runs": int((df_mena["broad_category"]=="Environment").sum()),
    "Animal runs": int((df_mena["broad_category"]=="Animal").sum()),
    "Plant runs": int((df_mena["broad_category"]=="Plant").sum()),
}
pd.DataFrame(list(summary.items()), columns=["Metric","Value"]).to_excel(f"{TBLDIR}/00_headline_stats.xlsx", index=False)

# save json summary for platform
stats_json = {
    "total_runs": len(df_mena),
    "total_samples": int(df_mena["sample_accession"].nunique()),
    "total_bioprojects": int(df_mena["bioproject"].nunique()),
    "total_countries": int(df_mena["country"].nunique()),
    "year_min": int(yr.min()),
    "year_max": int(yr.max()),
    "per_country": geo.to_dict(orient="records"),
    "per_year": [{"year":int(k),"n":int(v)} for k,v in yr_counts.items()],
    "categories": cat.to_dict(),
    "platforms": plat.to_dict(),
    "subtypes": sub.to_dict(),
    "top_bioprojects": top_bp.to_dict(orient="records"),
    "keywords": uni,
    "library_strategy": lib_strat.to_dict(),
    "host_body_sites": hb.to_dict(),
    "animal_hosts": an.to_dict(),
}
with open(f"{DATADIR}/mena_stats.json","w") as f:
    json.dump(stats_json, f, indent=2, default=lambda x: None if (isinstance(x,float) and (x!=x or x in (float("inf"),float("-inf")))) else str(x))

print(f"\nDONE. Figures: {FIGDIR}, Tables: {TBLDIR}, Data: {DATADIR}")
print(f"Headline: {len(df_mena):,} runs | {df_mena['sample_accession'].nunique():,} samples | {df_mena['bioproject'].nunique():,} BPs | {df_mena['country'].nunique()} countries")
