#!/usr/bin/env python
"""Step 6c. Figures 14-26 from the repaired v1.2 corpus."""
import json, os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
OK = ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#999999','#8d6a2a','#13315c']
OUT = 'figures_v12'; os.makedirs(OUT, exist_ok=True)
d = pd.read_csv('corpus_v12.tsv', sep='\t', dtype=str, low_memory=False)
d['rc'] = pd.to_numeric(d.read_count, errors='coerce'); d['yr'] = pd.to_numeric(d.year, errors='coerce')
N = len(d)
R = json.load(open('run_v12/data/mena_rigorous_results.json'))
PC = json.load(open('percategory_v12.json'))
BS = json.load(open('biosample_both.json'))['v12']
# The catch-all BioSample type is written Other/Unclassified throughout the paper
# (Section 3.4, Supplementary Table S1, all captions). Relabel it here so the
# figures use the same name as the text and tables.
def _relabel(o):
    if isinstance(o, dict):
        return {('Other/Unclassified' if k == 'Unspecified' else k): _relabel(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_relabel(v) for v in o]
    return 'Other/Unclassified' if o == 'Unspecified' else o
BS = _relabel(BS)
bst = pd.read_csv('run_v12/tables/T_biosample_types_per_category.xlsx'.replace('.xlsx','.xlsx'),
                  sep=None, engine='python') if False else None
CATORD = ['Environment','Human','Other','Plant','Animal','Food','Clinical','Fungal','Viral']
CATLAB = {'Other':'Other/Unclassified'}
def save(fig, n):
    fig.savefig(f'{OUT}/{n}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/{n}.svg', bbox_inches='tight'); plt.close(fig); print('  ', n)
JD = lambda o, f: json.dump(o, open(f, 'w'), indent=1, default=float)

# ------------------------------------------------------------- F14 GPS
gc = R['geocoord']; per = pd.DataFrame(gc['per_country'])
per = per.sort_values('n_with_coords', ascending=False).head(15)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
ax = axes[0]
pct = 100*per.n_valid/per.n_with_coords
b = ax.bar(range(len(per)), pct, color=OK[2])
ax.axhline(95, ls='--', color=OK[5], lw=1.2, label='95% threshold')
ax.set_xticks(range(len(per))); ax.set_xticklabels(per.country, rotation=35, ha='right', fontsize=8.5)
ax.set_ylabel('% coordinates within declared country'); ax.set_ylim(0, 105); ax.legend(frameon=False)
ax.set_title('a  GPS provenance validation (top 15 by coordinate count)', loc='left', fontweight='bold')
ax = axes[1]
ax.bar(['valid within country','provenance conflict'], [gc['n_valid'], gc['n_invalid']],
       color=[OK[2], OK[5]], width=.55)
for i, v in enumerate([gc['n_valid'], gc['n_invalid']]):
    ax.text(i, v + gc['n_valid']*.02, f'{v:,}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('runs'); ax.set_ylim(0, gc['n_valid']*1.15)
ax.set_title(f"b  Of {gc['n_with_coords']:,} parseable coordinates, {gc['pct_valid']}% validate",
             loc='left', fontweight='bold')
save(fig, 'Figure14_gps_validation')

# ------------------------------------------------------------- F15 lag
lagpc = pd.DataFrame(R['temporal_lag']['per_country'])
lagpc = lagpc[lagpc.n >= 20].sort_values('median_days', ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
# reproduce rigorous_analysis.py T1.4 exactly, so n, median and mean match Section 3.6
def _parse_date(s):
    if pd.isna(s): return None
    s = str(s)[:10]
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y'):
        try: return pd.to_datetime(s, format=fmt)
        except Exception: pass
    try: return pd.to_datetime(s, errors='coerce')
    except Exception: return None
_fpc = 'first_public_meta' if 'first_public_meta' in d.columns else 'first_public'
_cdt = d['collection_date'].apply(_parse_date); _pdt = d[_fpc].apply(_parse_date)
_mask = _cdt.notna() & _pdt.notna()
_ld = (_pdt[_mask] - _cdt[_mask]).dt.days
lag = _ld[(_ld >= 0) & (_ld <= 365*15)]
assert int(_mask.sum()) == R['temporal_lag']['n_with_dates']
assert lag.median() == R['temporal_lag']['overall_median_days']
ax = axes[0]
ax.hist(lag/365.25, bins=45, color=OK[0])
ax.axvline(lag.median()/365.25, color=OK[5], ls='--', lw=2,
           label=f'median {lag.median():,.0f} d ({lag.median()/365.25:.2f} y)')
ax.set_xlabel('years from collection to public release'); ax.set_ylabel('runs'); ax.legend(frameon=False)
ax.set_title(f'a  Submission lag (n = {len(lag):,})', loc='left', fontweight='bold')
ax = axes[1]
_y = np.arange(len(lagpc))[::-1]
ax.barh(_y, lagpc.median_days.values/365.25, color=OK[4], zorder=2)
ax.errorbar(lagpc.median_days.values/365.25, _y,
            xerr=np.vstack([(lagpc.median_days - lagpc.q25).values/365.25,
                            (lagpc.q75 - lagpc.median_days).values/365.25]),
            fmt='none', ecolor='0.35', elinewidth=1.1, capsize=2.5, zorder=3)
ax.set_yticks(_y); ax.set_yticklabels(lagpc.country, fontsize=8.5)
ax.set_xlabel('median lag (years), bars show the interquartile range')
ax.set_title('b  Median lag per country (n \u2265 20)', loc='left', fontweight='bold')
save(fig, 'Figure15_submission_lag')

# ------------------------------------------- F16 biosample types per category
cats = [c for c in CATORD if c in BS['by_category']]
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, cat in zip(axes.ravel(), cats):
    c = BS['by_category'][cat]
    t = c['types'][:10]
    lbl = [x['biosample_type'] for x in t]; val = [x['n_runs'] for x in t]
    ax.barh(range(len(t))[::-1], val, color=OK[cats.index(cat) % len(OK)])
    ax.set_yticks(range(len(t))[::-1]); ax.set_yticklabels(lbl, fontsize=7.5)
    for i, v in enumerate(val):
        ax.text(v + max(val)*.02, len(t)-1-i, f'{v:,}', va='center', fontsize=7.5)
    ax.set_xlim(0, max(val)*1.32)
    ax.set_title(f"{CATLAB.get(cat,cat)} (n = {c['total_runs']:,})", loc='left',
                 fontsize=10, fontweight='bold')
for ax in axes.ravel()[len(cats):]: ax.axis('off')
fig.subplots_adjust(wspace=0.55, hspace=0.38)
save(fig, 'Figure16_biosample_per_category')

# ----------------------------------------------- F17 top 25 biosample types
top = BS['top_overall'][:25]
cat_of = {}
for cat in cats:
    for t in BS['by_category'][cat]['types']:
        cat_of.setdefault(t['biosample_type'], cat)
fig, ax = plt.subplots(figsize=(9, 8))
lbl = [x['biosample_type'] for x in top]; val = [x['n_runs'] for x in top]
cols = [OK[cats.index(cat_of.get(l, 'Other')) % len(OK)] if cat_of.get(l) in cats else OK[7] for l in lbl]
b = ax.barh(range(len(top))[::-1], val, color=cols)
ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(lbl, fontsize=8.5)
for bb, v in zip(b, val):
    ax.text(v + max(val)*.012, bb.get_y()+bb.get_height()/2, f'{v:,}', va='center', fontsize=8)
ax.set_xlabel('runs'); ax.set_xlim(0, max(val)*1.14)
ax.set_title('Top 25 BioSample types, coloured by broad category', loc='left', fontweight='bold')
save(fig, 'Figure17_top_biosample_types')

# ------------------------------- F18 country x biosample type heatmap
bt = pd.read_csv('run_v12/data/mena_metagenomics_clean.tsv', sep='\t', dtype=str,
                 usecols=['run_accession','country'], low_memory=False)
types15 = [x['biosample_type'] for x in BS['top_overall'][:15]]
bycountry = BS['by_country']
countries = list(d.country.value_counts().index)
H = np.zeros((len(countries), len(types15)))
tot = d.country.value_counts()
for i, c in enumerate(countries):
    lookup = {t['biosample_type']: t['n_runs'] for t in bycountry.get(c, [])}
    for j, t in enumerate(types15):
        H[i, j] = 100*lookup.get(t, 0)/tot[c]
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(H, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(types15))); ax.set_xticklabels(types15, rotation=40, ha='right', fontsize=8.5)
ax.set_yticks(range(len(countries))); ax.set_yticklabels(countries, fontsize=8.5)
for i in range(len(countries)):
    for j in range(len(types15)):
        if H[i, j] >= 3:
            ax.text(j, i, f'{H[i,j]:.0f}', ha='center', va='center', fontsize=7,
                    color='white' if H[i, j] > 55 else 'black')
plt.colorbar(im, ax=ax, label='% within country', fraction=.025)
ax.set_title('Country × top-15 BioSample type (% within country)', loc='left', fontweight='bold')
save(fig, 'Figure18_country_biosample')

# --------------------------------------------- F19 a,b,c cross-tabulations
cc = pd.crosstab(d.country, d.broad_category).reindex(columns=[c for c in CATORD if c in
      pd.crosstab(d.country, d.broad_category).columns]).fillna(0)
cc = cc.loc[d.country.value_counts().index]
pct = cc.div(cc.sum(axis=1), axis=0)*100
fig, ax = plt.subplots(figsize=(11, 6))
bot = np.zeros(len(pct))
for i, c in enumerate(pct.columns):
    ax.barh(range(len(pct))[::-1], pct[c].values, left=bot, color=OK[i % len(OK)],
            label=CATLAB.get(c, c)); bot += pct[c].values
ax.set_yticks(range(len(pct))[::-1]); ax.set_yticklabels(pct.index, fontsize=8.5)
ax.set_xlabel('% of country runs'); ax.set_xlim(0, 100)
ax.legend(frameon=False, fontsize=8, ncol=5, loc='lower center', bbox_to_anchor=(.5, -.2))
ax.set_title('a  Country × broad category composition', loc='left', fontweight='bold')
save(fig, 'Figure19a_country_category')

PLATLAB = {'ILLUMINA':'Illumina','LS454':'Roche 454','OXFORD_NANOPORE':'Oxford Nanopore',
           'PACBIO_SMRT':'PacBio SMRT','ION_TORRENT':'Ion Torrent','BGISEQ':'BGI/DNBSEQ','DNBSEQ':'BGI/DNBSEQ'}
w = d[d.yr.between(2013, 2024)].copy(); w['P'] = w.instrument_platform.map(lambda x: PLATLAB.get(x, 'Other'))
pv = w.pivot_table(index='yr', columns='P', values='run_accession', aggfunc='size').fillna(0)
fig, ax = plt.subplots(figsize=(10, 4.6)); bot = np.zeros(len(pv))
for i, c in enumerate(pv.columns):
    ax.bar(pv.index, pv[c].values, bottom=bot, color=OK[i % len(OK)], label=c); bot += pv[c].values
ax.set_ylabel('runs'); ax.set_xlabel('year of first public release')
ax.legend(frameon=False, fontsize=8, ncol=4)
ax.set_title('b  Year × platform', loc='left', fontweight='bold')
save(fig, 'Figure19b_year_platform')

st = pd.crosstab(d.country, d.data_subtype.fillna('unclassified'))
st = st.loc[d.country.value_counts().index]
stp = st.div(st.sum(axis=1), axis=0)*100
fig, ax = plt.subplots(figsize=(10, 6)); bot = np.zeros(len(stp))
for i, c in enumerate(stp.columns):
    ax.barh(range(len(stp))[::-1], stp[c].values, left=bot, color=OK[i % len(OK)], label=c)
    bot += stp[c].values
ax.set_yticks(range(len(stp))[::-1]); ax.set_yticklabels(stp.index, fontsize=8.5)
ax.set_xlabel('% of country runs'); ax.set_xlim(0, 100); ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title('c  Country × data subtype', loc='left', fontweight='bold')
save(fig, 'Figure19c_country_subtype')

# ---------------------------------------- F20 BioProject size + read counts
bp = d.bioproject.value_counts()
bins = {'1':0,'2-4':0,'5-9':0,'10-24':0,'25-49':0,'50-99':0,'100-499':0,'≥500':0}
for v in bp.values:
    k = ('1' if v==1 else '2-4' if v<=4 else '5-9' if v<=9 else '10-24' if v<=24 else
         '25-49' if v<=49 else '50-99' if v<=99 else '100-499' if v<=499 else '≥500')
    bins[k] += 1
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
b = ax.bar(list(bins), list(bins.values()), color=OK[0])
for bb, v in zip(b, bins.values()):
    ax.text(bb.get_x()+bb.get_width()/2, v+max(bins.values())*.02, f'{v}', ha='center', fontsize=9)
ax.set_xlabel('runs per BioProject'); ax.set_ylabel('number of BioProjects')
ax.set_ylim(0, max(bins.values())*1.14)
ax.set_title(f'a  BioProject size distribution (n = {len(bp):,} projects)', loc='left', fontweight='bold')
ax = axes[1]
rcp = d.rc.dropna(); rcp = rcp[rcp > 1]
ax.hist(np.log10(rcp), bins=45, color=OK[0])
ax.axvline(np.log10(rcp.median()), color=OK[5], ls='--', lw=2,
           label=f'median = {rcp.median():,.0f}')
ax.set_xlabel('log$_{10}$(read count)'); ax.set_ylabel('runs'); ax.legend(frameon=False)
ax.set_title(f'b  Read count distribution (n = {len(rcp):,} runs, read_count > 1)',
             loc='left', fontweight='bold')
save(fig, 'Figure20_bioproject_readcount')
JD({'bins': bins, 'n_projects': int(len(bp)), 'median_reads': int(rcp.median()),
    'n_reads': int(len(rcp))}, 'fig20_values.json')

# ------------------------------------------------------- F21 diversity
dv = pd.DataFrame(R['diversity']['per_country']).sort_values('shannon_H', ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, col, ttl, c in [(axes[0], 'shannon_H', "a  Shannon H′", OK[0]),
                        (axes[1], 'simpson_D', 'b  Simpson 1−D', OK[2]),
                        (axes[2], 'richness', 'c  Observed richness', OK[3])]:
    s = dv.sort_values(col, ascending=False)
    ax.barh(range(len(s))[::-1], s[col].values, color=c)
    ax.set_yticks(range(len(s))[::-1]); ax.set_yticklabels(s.country, fontsize=8)
    for i, v in enumerate(s[col].values):
        vv = 0.0 if abs(v) < 5e-3 else v          # avoid printing -0.00
        ax.text(max(v, 0) + s[col].max()*.015, len(s)-1-i,
                f'{vv:.2f}' if col != 'richness' else f'{int(vv)}', va='center', fontsize=7.5)
    ax.set_xlim(0, s[col].max()*1.16)
    ax.set_title(ttl, loc='left', fontweight='bold')
save(fig, 'Figure21_diversity')

# ----------------------------------------------------- F22 rarefaction
rf = R['rarefaction']
fig, ax = plt.subplots(figsize=(9, 5.4))
for i, (c, v) in enumerate(rf.items()):
    ax.plot(v['sizes'], v['richness'], lw=1.8, color=OK[i % len(OK)], label=c)
ax.set_xlabel('runs sampled'); ax.set_ylabel('expected metagenome-type richness')
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title('Rarefaction curves, top 10 countries by run volume', loc='left', fontweight='bold')
save(fig, 'Figure22_rarefaction')

# ------------------------------------------------ F23 UPGMA + PCoA
cm = d.groupby(['country','scientific_name']).size().unstack(fill_value=0)
cnt = d.country.value_counts(); typ = d.groupby('country').scientific_name.nunique()
keep = [c for c in cm.index if cnt[c] >= 30 and typ[c] >= 10]
P = (cm.loc[keep] > 0).astype(int)
J = squareform(pdist(P.values, metric='jaccard')); n = len(J)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
Z = linkage(squareform(J, checks=False), method='average')
dendrogram(Z, labels=keep, ax=axes[0], color_threshold=.75*Z[:, 2].max(), leaf_font_size=9)
axes[0].set_ylabel('Jaccard distance')
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
axes[0].set_title('a  UPGMA dendrogram', loc='left', fontweight='bold')
C = np.eye(n) - np.ones((n, n))/n
G = -.5 * C @ (J**2) @ C
w_, V_ = np.linalg.eigh(G); idx = np.argsort(w_)[::-1]
w_, V_ = w_[idx], V_[:, idx]; pos = w_[w_ > 0]
X = V_[:, :2] * np.sqrt(np.abs(w_[:2]))
axes[1].scatter(X[:, 0], X[:, 1], s=70, color=OK[0], zorder=3)
for i, c in enumerate(keep):
    axes[1].annotate(c, (X[i, 0], X[i, 1]), fontsize=8, xytext=(4, 4), textcoords='offset points')
axes[1].set_xlabel(f'PCo1 ({100*pos[0]/pos.sum():.1f}%)')
axes[1].set_ylabel(f'PCo2 ({100*pos[1]/pos.sum():.1f}%)')
axes[1].axhline(0, color='0.85', lw=.8, zorder=0); axes[1].axvline(0, color='0.85', lw=.8, zorder=0)
axes[1].set_title('b  Classical PCoA', loc='left', fontweight='bold')
save(fig, 'Figure23_upgma_pcoa')

# --------------------------------------------- F24 per-category PERMANOVA
tst = {k: v for k, v in PC.items() if isinstance(v, dict) and v.get('testable')}
o = sorted(tst, key=lambda k: -tst[k]['R2_pct'])
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
cols = [OK[5] if tst[k]['permdisp_p'] < .05 else OK[2] for k in o]
b = ax.barh(range(len(o))[::-1], [tst[k]['R2_pct'] for k in o], color=cols)
ax.set_yticks(range(len(o))[::-1]); ax.set_yticklabels(o)
for i, k in enumerate(o):
    ax.text(tst[k]['R2_pct']+.35, len(o)-1-i, f"{tst[k]['R2_pct']:.1f}%", va='center', fontsize=9)
    ax.plot(tst[k]['E_R2_pct'], len(o)-1-i, 'o', mfc='none', mec='black', ms=7)
allc = json.load(open('run_v12/data/mena_rigorous_results.json'))['permanova']
ax.axvline(100*allc['R_squared'], ls='--', color='0.4', lw=1.2,
           label=f"all-category baseline ({100*allc['R_squared']:.2f}%)")
ax.set_xlabel('R² × 100 (% variance explained by country)')
ax.set_xlim(0, max(tst[k]['R2_pct'] for k in o) * 1.72)
from matplotlib.patches import Patch
_h = [Patch(facecolor=OK[2], label='PERMDISP p > 0.05 (dispersion homogeneous)'),
      Patch(facecolor=OK[5], label='PERMDISP p < 0.05 (dispersion heterogeneous)')]
ax.legend(handles=_h + ax.get_legend_handles_labels()[0],
          frameon=False, fontsize=7.5, loc='lower right')
ax.set_title('a  Per-category R² (open circle = null expectation)', loc='left', fontweight='bold')
ax = axes[1]
ax.barh(range(len(o))[::-1], [-np.log10(tst[k]['p']) for k in o], color=OK[0])
ax.set_yticks(range(len(o))[::-1]); ax.set_yticklabels(o)
ax.axvline(-np.log10(.05), ls='--', color=OK[5], lw=1.2, label='α = 0.05')
_xm = max(-np.log10(tst[k]['p']) for k in o)
for i, k in enumerate(o):
    _q = tst[k].get('q_BH')
    # 4 decimals below 0.001 so that q = 0.0003 does not print as 0.000
    _qs = 'n/a' if _q is None else (f'{_q:.4f}' if _q < .001 else f'{_q:.3f}')
    ax.text(-np.log10(tst[k]['p']) + _xm*.02, len(o)-1-i, f'q = {_qs}', va='center',
            fontsize=8, color=('black' if _q is not None and _q < .05 else '0.35'))
ax.set_xlim(0, _xm*1.32)
ax.set_xlabel('−log$_{10}$(PERMANOVA p)'); ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.set_title('b  Significance (q = Benjamini-Hochberg across the six tests)',
             loc='left', fontweight='bold')
save(fig, 'Figure24_percategory_permanova')

# --------------------------------------------------- F25 sampling density
cells = pd.DataFrame(R['geo_grid']['cells'])
fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(cells.lon_bin, cells.lat_bin, s=np.sqrt(cells.n_runs)*3.2, c=cells.n_runs,
                cmap='YlOrRd', alpha=.85, edgecolor='0.35', linewidth=.35)
plt.colorbar(sc, ax=ax, label='runs per 1° cell', fraction=.03)
ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
mi = R['morans_I']
ax.set_title(f"1° sampling-density grid: {R['geo_grid']['n_cells']} cells, "
             f"max {R['geo_grid']['max_density']:,} runs\n"
             f"Moran's I = {mi['I']} (E = {mi['expected_I']}), z = {mi['z_score']}, p = {mi['p_value']:.4f}",
             loc='left', fontweight='bold', fontsize=10)
save(fig, 'Figure25_sampling_density')

# --------------------------------------------------------- F26 NLP
nl = R['nlp_clustering']; sil = nl['silhouette_scores']
ks = list(range(3, 3+len(sil)))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(ks, sil, 'o-', color=OK[0], lw=2)
bi = int(np.argmax(sil)); ax.plot(ks[bi], sil[bi], 'o', ms=12, color=OK[5], zorder=3)
ax.annotate(f'optimal k = {ks[bi]}\nsilhouette = {sil[bi]:.3f}', (ks[bi], sil[bi]),
            xytext=(12, -6), textcoords='offset points', fontsize=9)
ax.set_xlabel('number of clusters (k)'); ax.set_ylabel('silhouette score')
ax.set_title('a  Cluster-count selection', loc='left', fontweight='bold')
ax = axes[1]
cl = nl['clusters']
sizes = [c.get('n', c.get('size', 0)) for c in cl]
labs = [f"C{c.get('cluster', i)}  " + ', '.join(c.get('top_terms', [])[:2]) for i, c in enumerate(cl)]
ordr = np.argsort(sizes)[::-1]
b = ax.barh(range(len(cl))[::-1], [sizes[i] for i in ordr],
            color=[OK[i % len(OK)] for i in range(len(cl))])
ax.set_yticks(range(len(cl))[::-1]); ax.set_yticklabels([labs[i] for i in ordr], fontsize=8.5)
for bb, v in zip(b, [sizes[i] for i in ordr]):
    ax.text(v + max(sizes)*.015, bb.get_y()+bb.get_height()/2, f'n = {v}', va='center', fontsize=8.5)
ax.set_xlabel('studies per cluster'); ax.set_xlim(0, max(sizes)*1.2)
x2 = R['nlp_cluster_country_chi2']
ax.set_title(f"b  Clusters (χ² = {x2['chi2']:,.1f}, df = {x2['dof']}, p < 0.001)",
             loc='left', fontweight='bold')
save(fig, 'Figure26_nlp_clustering')
print('part 2b complete (F14-F26)')
