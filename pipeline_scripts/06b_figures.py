#!/usr/bin/env python
"""Step 6b. Figures 9-26 from the repaired v1.2 corpus."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats as sps

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
OK = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#999999',
      '#8d6a2a', '#13315c']
OUT = 'figures_v12'; os.makedirs(OUT, exist_ok=True)
d = pd.read_csv('corpus_v12.tsv', sep='\t', dtype=str, low_memory=False)
d['rc'] = pd.to_numeric(d.read_count, errors='coerce')
d['bc'] = pd.to_numeric(d.base_count, errors='coerce')
d['yr'] = pd.to_numeric(d.year, errors='coerce')
N = len(d)
R = json.load(open('run_v12/data/mena_rigorous_results.json'))
PC = json.load(open('percategory_v12.json'))
BS = json.load(open('biosample_both.json'))['v12']
PH = {'not collected','not applicable','not available','unknown','missing','none','null','na','n/a','-',''}
def filled(s): return ~s.fillna('').astype(str).str.strip().str.lower().isin(PH)
PLATLAB = {'ILLUMINA':'Illumina','LS454':'Roche 454','OXFORD_NANOPORE':'Oxford Nanopore',
           'PACBIO_SMRT':'PacBio SMRT','ION_TORRENT':'Ion Torrent','BGISEQ':'BGI/DNBSEQ','DNBSEQ':'BGI/DNBSEQ'}
CATORD = ['Environment','Human','Other','Plant','Animal','Food','Clinical','Fungal','Viral']
CATLAB = {'Other':'Other/Unclassified'}
def save(fig, n):
    fig.savefig(f'{OUT}/{n}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/{n}.svg', bbox_inches='tight'); plt.close(fig); print('  ', n)

d['PLAT'] = d.instrument_platform.map(lambda x: PLATLAB.get(x, 'Other'))

# --------------------------------------------------------- F9 read counts
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
order = [p for p in ['Illumina','BGI/DNBSEQ','Ion Torrent','Oxford Nanopore','PacBio SMRT','Roche 454']
         if (d.PLAT == p).sum() > 5]
data = [np.log10(d[(d.PLAT == p) & (d.rc > 1)].rc.values) for p in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
for p_, c in zip(bp['boxes'], OK): p_.set_facecolor(c); p_.set_alpha(.75)
for m in bp['medians']: m.set_color('black')
for i, p in enumerate(order):
    med = d[(d.PLAT == p) & (d.rc > 1)].rc.median()
    ax.text(i+1, np.log10(med)+.16, f'{med:,.0f}', ha='center', fontsize=8)
ax.set_ylabel('log$_{10}$(read count)')
ax.set_xticklabels(order, rotation=25, ha='right', fontsize=8.5)
ax.set_title('a  Read count per platform', loc='left', fontweight='bold')
ax = axes[1]
cats = [c for c in CATORD if (d.broad_category == c).sum() >= 100]
data = [np.log10(d[(d.broad_category == c) & (d.rc > 1)].rc.values) for c in cats]
bp = ax.boxplot(data, tick_labels=[CATLAB.get(c, c) for c in cats], patch_artist=True, showfliers=False)
for p_, c in zip(bp['boxes'], OK*2): p_.set_facecolor(c); p_.set_alpha(.75)
for m in bp['medians']: m.set_color('black')
ax.set_ylabel('log$_{10}$(read count)')
ax.set_xticklabels([CATLAB.get(c, c) for c in cats], rotation=30, ha='right', fontsize=8.5)
ax.set_title('b  Read count per broad category', loc='left', fontweight='bold')
save(fig, 'Figure09_readcount')

# -------------------------------------------------- F10 depth thresholds
top10 = list(d.country.value_counts().head(10).index)
amp_p, sh_p = [], []
for c in top10:
    a = d[(d.country == c) & (d.data_subtype == 'amplicon_metagenomics') & (d.rc > 0)]
    s = d[(d.country == c) & (d.data_subtype == 'shotgun_metagenomics') & (d.bc > 0)]
    amp_p.append(100*(a.rc >= 10000).mean() if len(a) else 0)
    sh_p.append(100*(s.bc >= 1e9).mean() if len(s) else 0)
x = np.arange(len(top10)); fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(x-.2, amp_p, .4, color=OK[0], label='amplicon ≥ 10,000 reads')
ax.bar(x+.2, sh_p, .4, color=OK[4], label='shotgun ≥ 1 Gbp')
ax.set_xticks(x); ax.set_xticklabels(top10, rotation=30, ha='right')
ax.set_ylabel('% of runs at or above reference value'); ax.legend(frameon=False)
ax.set_title('Fraction of runs meeting the descriptive depth reference values', loc='left', fontweight='bold')
save(fig, 'Figure10_depth_thresholds')
json.dump({'top10': top10, 'amplicon': [round(v,1) for v in amp_p],
           'shotgun': [round(v,1) for v in sh_p]}, open('fig10_values.json','w'), indent=1, default=float)

# ------------------------------------- F11 platform x category residuals
sub = d[~d.instrument_platform.isin(['CAPILLARY','ABI_SOLID','ELEMENT'])]
ct = pd.crosstab(sub.PLAT, sub.broad_category).reindex(columns=[c for c in CATORD if c in
      pd.crosstab(sub.PLAT, sub.broad_category).columns]).fillna(0)
chi2, p, dof, exp = sps.chi2_contingency(ct.values)
res = (ct.values - exp)/np.sqrt(exp)
V = np.sqrt(chi2/(ct.values.sum()*min(ct.shape[0]-1, ct.shape[1]-1)))
fig, ax = plt.subplots(figsize=(11, 4.6))
lim = np.abs(res).max()
im = ax.imshow(res, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels([CATLAB.get(c, c) for c in ct.columns],
                                                      rotation=30, ha='right')
ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, f'{res[i,j]:.1f}', ha='center', va='center', fontsize=8,
                color='white' if abs(res[i, j]) > lim*.55 else 'black')
plt.colorbar(im, ax=ax, label='standardised residual', fraction=.03)
ax.set_title('Platform × broad category: standardised χ² residuals', loc='left', fontweight='bold')
fig.text(.5, -.06, f"χ² = {chi2:,.1f}   df = {dof}   Cramér's V = {V:.3f}   "
         f"n = {int(ct.values.sum()):,} runs (CAPILLARY and ELEMENT excluded)",
         ha='center', fontsize=9, style='italic')
save(fig, 'Figure11_platform_category_residuals')
json.dump({'chi2': round(float(chi2),1), 'dof': int(dof), 'V': round(float(V),3),
           'n': int(ct.values.sum()),
           'residuals': {r_: {c_: round(float(res[i,j]),1) for j, c_ in enumerate(ct.columns)}
                         for i, r_ in enumerate(ct.index)}}, open('fig11_values.json','w'), indent=1)

# ---------------------------------------------------- F12 tech adoption
import pymannkendall as mk
def shannon(v):
    p_ = np.array(v, float); p_ = p_[p_ > 0]; p_ = p_/p_.sum(); return float(-(p_*np.log(p_)).sum())
w = d[d.yr.between(2013, 2024)]
yrs = sorted(w.yr.dropna().unique().astype(int))
lr = [100*w[(w.yr == y) & w.instrument_platform.isin(['OXFORD_NANOPORE','PACBIO_SMRT'])].bioproject.nunique()
      / max(w[w.yr == y].bioproject.nunique(), 1) for y in yrs]
sh = [shannon(w[w.yr == y].instrument_model.value_counts().values) for y in yrs]
pe = [100*(w[w.yr == y].library_layout.fillna('').str.upper() == 'PAIRED').mean() for y in yrs]
rd = [100*(w[w.yr == y].library_selection.fillna('') == 'RANDOM').mean() for y in yrs]
fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), constrained_layout=True)
for ax, ser, ttl, col in [(axes[0,0], lr, 'a  Long-read penetration (% of BioProjects)', OK[0]),
                          (axes[0,1], sh, "b  Technology diversity (Shannon H′, instrument model)", OK[2]),
                          (axes[1,0], pe, 'c  Paired-end adoption (% of runs)', OK[3]),
                          (axes[1,1], rd, 'd  RANDOM (PCR-free) selection (% of runs)', OK[5])]:
    r_ = mk.original_test(ser)
    ax.plot(yrs, ser, 'o-', color=col, lw=2, ms=4)
    ax.set_title(f'{ttl}\nτ = {r_.Tau:.2f}, p = {r_.p:.4f}', loc='left', fontsize=10, fontweight='bold')
    ax.set_xlabel('year of first public release')
save(fig, 'Figure12_technology_adoption')
json.dump({'years': [int(y) for y in yrs], 'longread_bp_pct': [round(v,1) for v in lr],
           'shannon_model': [round(v,2) for v in sh], 'paired_pct': [round(v,1) for v in pe],
           'random_pct': [round(v,1) for v in rd],
           'mk': {k: {'tau': round(mk.original_test(v).Tau,3), 'p': round(mk.original_test(v).p,4)}
                  for k, v in [('longread',lr),('shannon',sh),('paired',pe),('random',rd)]}},
          open('fig12_values.json','w'), indent=1, default=float)

# ------------------------------------------- F13 metadata completeness
FLDS = [('country','country'),('year','year'),('scientific_name','scientific_name'),
        ('library_strategy','library_strategy'),('library_source','library_source'),
        ('instrument_platform','instrument_platform'),('collection_date','collection_date'),
        ('host','host_meta'),('host_sex','host_sex'),('host_body_site','host_body_site_meta'),
        ('environment_biome','environment_biome'),('isolation_source','isolation_source_meta'),
        ('read_count','read_count'),('base_count','base_count'),
        ('sample_host_disease','sample_host_disease'),('sample_host_age','sample_host_age'),
        ('sample_lat_lon','sample_lat_lon'),('sample_dna_extraction','sample_dna_extraction')]
countries = list(d.country.value_counts().index)
M = np.zeros((len(countries), len(FLDS)))
for i, c in enumerate(countries):
    s = d[d.country == c]
    for j, (_, col) in enumerate(FLDS):
        cc = col if col in d.columns else col.replace('_meta','')
        M[i, j] = 100*filled(s[cc]).mean()
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(M, cmap='cividis', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(FLDS))); ax.set_xticklabels([f[0] for f in FLDS], rotation=45, ha='right', fontsize=8.5)
ax.set_yticks(range(len(countries))); ax.set_yticklabels(countries, fontsize=8.5)
plt.colorbar(im, ax=ax, label='% complete', fraction=.025)
pcm = M.mean(axis=1)
ax.set_title(f'Metadata completeness per country and field  '
             f'(per-country mean {pcm.min():.1f}% to {pcm.max():.1f}%)', loc='left', fontweight='bold')
save(fig, 'Figure13_metadata_completeness')
json.dump({'per_country_mean': {c: round(v,1) for c, v in zip(countries, pcm)}},
          open('fig13_values.json','w'), indent=1, default=float)

print('part 2a complete (F9-F13)')
