"""
BioSample-type descriptive analysis.
Harmonizes isolation_source + environment_material + specific_category into a
clean BioSample-type taxonomy, then produces per-category descriptive stats.
"""
import os, json, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter, defaultdict

DATA = "/home/claude/out/data/mena_metagenomics_clean.tsv"
DATADIR = "/home/claude/out/data"
TBLDIR = "/home/claude/out/tables"
FIGDIR = "/home/claude/out/figures"

mpl.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight",
})
PAL = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442","#999999",
       "#0b2545","#c9a227","#13315c","#8d6a2a","#a0a0a0"]
CAT_PAL = {"Human":"#CC79A7","Environment":"#009E73","Animal":"#E69F00","Plant":"#56B4E9",
           "Food":"#D55E00","Clinical":"#0072B2","Other":"#999999","Fungal":"#F0E442","Viral":"#8d6a2a"}

def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/{name}.svg", bbox_inches="tight")
    plt.close(fig)

print("Loading...")
df = pd.read_csv(DATA, sep="\t", low_memory=False)
print(f"  {len(df):,} runs")

# ════════════════════════════════════════════════════════════════════
# BIOSAMPLE TYPE HARMONIZATION
# ════════════════════════════════════════════════════════════════════
# Hierarchical rules: examine isolation_source, environment_material, specific_category,
# host_body_site, scientific_name → assign a clean biosample_type label.
# Priority: isolation_source > environment_material > specific_category > scientific_name.

# Curated keyword → biosample_type mappings, organized by broad_category
RULES = {
    # Human-associated body sites and biofluids
    "Human": [
        # (keyword pattern, biosample_type)
        (r"\bstool\b|\bfeces\b|\bfaecal\b|\bfecal\b|\bgut\b|\bcolon\b|\bintestin\b|\brectal\b|\bduoden\b", "Stool / Gut"),
        (r"\boral\b|\bsaliva\b|\bbuccal\b|\btongue\b|\bdental\b|\bplaque\b|\bmouth\b|\bgingiv\b", "Oral / Saliva"),
        (r"\bnasal\b|\bnasopharyng\b|\bnose\b|\bsinus\b|\bnaris\b", "Nasal / Nasopharyngeal"),
        (r"\brespir\b|\blung\b|\bbronch\b|\bsputum\b|\btrache\b|\bairway\b|\bpharyng\b", "Respiratory tract"),
        (r"\bskin\b|\bdermis\b|\bepiderm\b|\bcutaneous\b|\bforearm\b|\bscalp\b", "Skin"),
        (r"\bvagin\b|\bcervi(c|x)\b|\bvulv\b", "Vaginal"),
        (r"\burin\b|\bbladder\b|\bkidney\b|\burethr\b", "Urinary"),
        (r"\bblood\b|\bserum\b|\bplasma\b|\bbuffy\s*coat\b", "Blood / Serum"),
        (r"\bmilk\b|\bbreast\b|\bcolostrum\b", "Breast milk"),
        (r"\bwound\b|\bulcer\b|\bsurgical\b", "Wound / Surgical"),
        (r"\bplacenta|\bamnio|\bfetal", "Placenta / Fetal"),
        (r"\btissue\b|\bbiops\b", "Tissue / Biopsy"),
    ],
    "Animal": [
        (r"\bcoral\s*tissue|\bcoral\s*mucus|\bcoral\s*reef|\bcoral\b|\bsymbiodinium", "Coral tissue / mucus"),
        (r"\bgut\b|\bfeces\b|\bfecal\b|\bfaecal\b|\bstool\b|\bcaecum\b|\bcecum\b|\bcecal\b|\brumen\b|\bdigesta\b|\bcolon\b|\bintestin\b|\bgut\s*content", "Gut / Rumen / Feces"),
        (r"\bskin\b|\bmucus\b|\bepidermis\b", "Skin / Mucus"),
        (r"\boral\b|\bsaliva\b|\bcrop\b", "Oral / Saliva"),
        (r"\bnasal\b|\bnasopharyng\b", "Nasal"),
        (r"\bmilk\b|\budder\b|\bmastiti", "Milk / Udder"),
        (r"\bblood\b|\bserum\b|\bplasma\b", "Blood"),
        (r"\bsponge\b|\bjellyfish|\banemone", "Sponge / Cnidarian"),
        (r"\bgill\b|\bswim\s*bladder|\bfish\s*skin|\bfish\s*gut|\baquaculture", "Fish / Aquaculture"),
        (r"\binsect\b|\bbee\b|\bgut\b.*\bbee\b|\bhoney\b|\bhive|\bant\b|\bbeetle|\bphlebotomus|\bsandfly|\bmosquito|\btick|\blocust", "Insect / Arthropod"),
        (r"\begg\b|\boviduct|\bcloac", "Egg / Reproductive"),
        (r"\btissue\b|\borgan\b|\bbiops", "Tissue / Organ"),
        (r"\binvertebrate", "Invertebrate (general)"),
        (r"\brotifer\b|\bzooplankton|\bcrustacean|\bshrimp|\bcrab\b|\boyster|\bmussel|\bclam\b|\bbivalv", "Aquatic invertebrate"),
        (r"\baquatic|\bwater\b|\baeration\s*tank", "Aquatic habitat (animal-associated)"),
        (r"\bhabitat\b|\benvironment", "Habitat (general)"),
    ],
    "Plant": [
        (r"\broot\s*nodule\b|\bnodule\b", "Root nodule"),
        (r"\brhizosphere|\brhizoplane\b", "Rhizosphere"),
        (r"\broot\b|\broots\b", "Root"),
        (r"\bphyllosphere|\bleaf\b|\bleaves\b|\bfoliar\b", "Phyllosphere / Leaf"),
        (r"\bstem\b|\bstems\b|\btrunk\b|\bbark\b|\bwood\b", "Stem / Bark"),
        (r"\bseed\b|\bgrain\b|\bkernel\b", "Seed"),
        (r"\bfruit\b|\bberr(y|ies)\b|\bdate\b", "Fruit"),
        (r"\bflower\b|\bpetal\b|\bnectar\b", "Flower / Nectar"),
        (r"\bendophyt", "Endophyte"),
        (r"\bcrown\b|\bbulb\b|\btuber\b", "Crown / Tuber"),
        (r"\bmycorrhiza|\bsymbio", "Mycorrhiza / Symbiont"),
    ],
    "Environment": [
        (r"\bsoil\b|\barid\b|\bdesert\b|\bsand\b|\btopsoil|\bsubsoil|\bdust\b", "Soil / Desert"),
        (r"\bsediment\b", "Sediment"),
        (r"\bseawater|\bsea water|\bocean\b|\bmarine\b|\bsalt\s*water|\bbrine\b", "Seawater / Marine"),
        (r"\bfreshwater|\briver\b|\blake\b|\bspring\b|\bstream\b|\bpond\b|\bgroundwater\b|\bwater\s*samples?\b|\bdeep\s*well\b|\bsubsurface", "Freshwater / Groundwater"),
        (r"\bwastewater|\bsewage\b|\bsludge\b|\beffluent\b|\bactivated", "Wastewater / Sludge"),
        (r"\bair\b|\bindoor\b|\baerosol|\bbioaerosol|\batmospher", "Air / Bioaerosol"),
        (r"\bbiofilm\b", "Biofilm"),
        (r"\bhydrothermal|\bhotspring|\bhot\s*spring|\bvolcanic|\bgeotherm", "Hydrothermal / Volcanic"),
        (r"\boil\b|\bpetroleum|\btar\s*sand|\bcrude", "Petroleum / Hydrocarbon"),
        (r"\bsalt\s*marsh|\bestuar(y|ine)|\bmangrove", "Estuary / Mangrove"),
        (r"\brock\b|\bcave\b|\bsubsurf", "Rock / Cave / Subsurface"),
    ],
    "Food": [
        (r"\bcheese\b|\bdair(y|ies)\b|\bmilk\s*product|\byogurt|\blaban\b|\bkefir\b", "Dairy / Cheese / Yogurt"),
        (r"\bmeat\b|\bsausage|\bbeef\b|\bpoultry\s*meat", "Meat / Sausage"),
        (r"\bferment\b|\bsourdough|\bkimchi|\bsaurkraut|\bkaak\b|\bdosa\b", "Fermented food"),
        (r"\bolive\b|\bbrine\b", "Olive / Brine"),
        (r"\bbread\b|\bdough\b|\bflour\b", "Bread / Dough"),
        (r"\bbeverage|\bjuice|\bwine|\bbeer\b|\btea\b|\bcoffee\b", "Beverage"),
        (r"\bspice|\bherb\b|\bcondiment", "Spice / Condiment"),
    ],
    "Clinical": [
        (r"\bicu\b|\bhospital\s*environment|\bhospital\s*surface", "Hospital environment / Surface"),
        (r"\bnasopharyng|\boropharyng|\bswab\b|\bsputum\b", "Clinical swab / Sputum"),
        (r"\bblood\s*culture|\bblood\b|\bbacterem", "Blood culture"),
        (r"\burine\b|\burinary", "Urine"),
        (r"\bwound\b|\bsurgical|\bulcer\b|\babscess|\bpus\b", "Wound / Surgical site"),
        (r"\bcatheter|\bdevice\b|\bimplant", "Medical device"),
        (r"\bcsf\b|\bcerebrospinal", "CSF"),
        (r"\bbiopsy\b|\btissue\b", "Tissue / Biopsy"),
    ],
    "Fungal": [
        (r"\blichen\b", "Lichen"),
        (r"\bmushroom|\bbasidiocarp|\bfruiting\s*body", "Mushroom / Fruiting body"),
        (r"\bsoil\b", "Fungal soil"),
        (r"\bplant\b|\broot\b", "Plant-associated fungi"),
        (r"\bits\b|\bmyco", "ITS amplicon (general fungi)"),
    ],
    "Viral": [
        (r"\bvirome\b|\bviral\s*metagen", "Virome"),
        (r"\bphage\b|\bbacteriophage", "Bacteriophage"),
    ],
    "Other": [
        (r".*", "Unspecified"),
    ],
}

# Helper: build composite text per row to scan
def composite_source(row):
    parts = []
    for c in ["isolation_source","environment_material","environment_feature","host_body_site","sample_title","scientific_name","specific_category"]:
        v = row.get(c)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).lower())
    return " ; ".join(parts)

print("\nBuilding composite source strings...")
df["_src"] = df.apply(composite_source, axis=1)

def assign_biosample_type(row):
    cat = row["broad_category"]
    src = row["_src"]
    if not src:
        return "Unspecified"
    rules = RULES.get(cat, RULES["Other"])
    for pat, label in rules:
        if re.search(pat, src):
            return label
    return "Other (within category)"

print("Assigning biosample types...")
df["biosample_type"] = df.apply(assign_biosample_type, axis=1)

# ════════════════════════════════════════════════════════════════════
# DESCRIPTIVE ANALYSIS
# ════════════════════════════════════════════════════════════════════
# 1. Counts per (broad_category, biosample_type)
# 2. Per-country biosample type distribution
# 3. Top biosample types overall
# 4. Coverage: % of runs successfully assigned to a specific type

assigned = df["biosample_type"].notna() & (~df["biosample_type"].isin(["Unspecified","Other (within category)","Habitat (general)"]))
coverage = 100 * assigned.sum() / len(df)
print(f"  Assignment coverage: {coverage:.1f}% ({assigned.sum():,} of {len(df):,} runs assigned a specific biosample type)")

# Hierarchical table: category × biosample_type
hier = df.groupby(["broad_category","biosample_type"]).size().reset_index(name="n_runs")
hier = hier.sort_values(["broad_category","n_runs"], ascending=[True, False])
hier.to_excel(f"{TBLDIR}/T_biosample_types_per_category.xlsx", index=False)

# Save full per-run assignments for the database
df[["run_accession","sample_accession","bioproject","country","broad_category","specific_category","biosample_type","isolation_source","environment_material"]].to_csv(
    f"{DATADIR}/mena_biosample_types.tsv", sep="\t", index=False
)

# Build JSON for the platform
biosample_data = {
    "coverage_pct": round(float(coverage), 2),
    "n_assigned": int(assigned.sum()),
    "n_total": int(len(df)),
    "by_category": {},
    "top_overall": [],
    "by_country": {},
    "rules_count": int(sum(len(v) for v in RULES.values())),
}

for cat in sorted(df["broad_category"].unique()):
    sub = df[df["broad_category"]==cat]
    types = sub["biosample_type"].value_counts().reset_index()
    types.columns = ["biosample_type","n_runs"]
    types["pct_of_category"] = (100 * types["n_runs"] / len(sub)).round(2)
    biosample_data["by_category"][cat] = {
        "total_runs": int(len(sub)),
        "n_distinct_types": int(types.shape[0]),
        "types": types.head(15).to_dict(orient="records"),
    }

# Top overall (excluding generic catch-all categories)
top = df["biosample_type"].value_counts().head(25)
biosample_data["top_overall"] = [{"biosample_type":k, "n_runs":int(v)} for k,v in top.items()]

# Per-country: top 5 biosample types per country
for c in sorted(df["country"].unique()):
    sub = df[df["country"]==c]
    top5 = sub["biosample_type"].value_counts().head(5)
    biosample_data["by_country"][c] = [{"biosample_type":k,"n_runs":int(v)} for k,v in top5.items()]

with open(f"{DATADIR}/mena_biosample_descriptive.json","w") as f:
    json.dump(biosample_data, f, indent=2)
print(f"  saved: mena_biosample_descriptive.json")

# ════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════
print("\n[Figures]")

# B01: Heatmap of biosample types per category (top 12 types per category)
top_per_cat = {}
for cat, grp in df.groupby("broad_category"):
    top_per_cat[cat] = grp["biosample_type"].value_counts().head(8).index.tolist()

# Stacked bar: top biosample types within each broad category, normalized
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
cats_plot = ["Human","Animal","Plant","Environment","Food","Clinical","Fungal","Viral","Other"]
for idx, cat in enumerate(cats_plot):
    ax = axes.flat[idx]
    sub = df[df["broad_category"]==cat]
    if len(sub) == 0:
        ax.axis("off"); continue
    counts = sub["biosample_type"].value_counts().head(10)
    colors = [CAT_PAL.get(cat, "#999")]*len(counts)
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors, alpha=0.85)
    ax.set_title(f"{cat} (n={len(sub):,})", fontsize=11, fontweight="bold", color=CAT_PAL.get(cat,"#0b2545"))
    ax.set_xlabel("Runs", fontsize=9)
    ax.tick_params(labelsize=8)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width()+max(counts.values)*0.01, bar.get_y()+bar.get_height()/2,
                f"{val:,}", va="center", fontsize=7)
plt.suptitle("BioSample-type composition within each broad ecological category",
             fontsize=13, fontweight="bold", y=1.001)
plt.tight_layout()
save(fig, "B01_biosample_types_per_category")

# B02: Top 25 biosample types overall
fig, ax = plt.subplots(figsize=(11, 9))
top25 = df["biosample_type"].value_counts().head(25)
# color each bar by its broad_category
cat_lookup = df.groupby("biosample_type")["broad_category"].agg(lambda x: x.mode().iloc[0])
colors = [CAT_PAL.get(cat_lookup.get(t, "Other"), "#999") for t in top25.index]
bars = ax.barh(top25.index[::-1], top25.values[::-1], color=colors[::-1], alpha=0.85)
for bar, val in zip(bars, top25.values[::-1]):
    ax.text(bar.get_width()+max(top25.values)*0.005, bar.get_y()+bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
ax.set_xlabel("Number of runs")
ax.set_title("Top 25 BioSample types across the MENA Microbiome Database\n(bar color = broad ecological category)")
# legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, label=k) for k,c in CAT_PAL.items() if k in cat_lookup.values]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
plt.tight_layout()
save(fig, "B02_top_biosample_types_overall")

# B03: country × top biosample type heatmap
top10_types = df["biosample_type"].value_counts().head(15).index.tolist()
ct_bs = df[df["biosample_type"].isin(top10_types)].groupby(["country","biosample_type"]).size().unstack(fill_value=0)
# row-normalize to %
ct_bs_pct = ct_bs.div(ct_bs.sum(axis=1), axis=0) * 100
# sort countries by total runs
order = df["country"].value_counts().index.tolist()
ct_bs_pct = ct_bs_pct.reindex([c for c in order if c in ct_bs_pct.index])

fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(ct_bs_pct.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=ct_bs_pct.values.max())
ax.set_xticks(range(len(ct_bs_pct.columns)))
ax.set_xticklabels(ct_bs_pct.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(ct_bs_pct.index)))
ax.set_yticklabels(ct_bs_pct.index, fontsize=9)
for i in range(len(ct_bs_pct.index)):
    for j in range(len(ct_bs_pct.columns)):
        v = ct_bs_pct.values[i,j]
        if v >= 1:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > ct_bs_pct.values.max()*0.5 else "black")
plt.colorbar(im, ax=ax, label="% of country's runs", shrink=0.8)
ax.set_title("Per-country BioSample-type composition (% within country, top 15 types)")
plt.tight_layout()
save(fig, "B03_country_biosample_heatmap")

print(f"  Figures: B01, B02, B03 (PNG+SVG)")
print(f"  Tables: T_biosample_types_per_category.xlsx")
print(f"  Data: mena_biosample_types.tsv, mena_biosample_descriptive.json")

# ════════════════════════════════════════════════════════════════════
# Summary print
# ════════════════════════════════════════════════════════════════════
print("\n=== SUMMARY ===")
print(f"  {biosample_data['rules_count']} curated keyword rules across {len(RULES)} broad categories")
print(f"  {assigned.sum():,} of {len(df):,} runs ({coverage:.1f}%) assigned a specific BioSample type")
print(f"\n  Top 10 BioSample types overall:")
for k, v in df["biosample_type"].value_counts().head(10).items():
    cat = cat_lookup.get(k, "?")
    print(f"    {k:<32} {v:>6,}  ({cat})")
