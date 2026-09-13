#!/usr/bin/env python
"""
Step 6. Regenerate every manuscript figure from the repaired v1.2 corpus.

All 30 panels are rebuilt. The 10 panels that v12.docx had already replaced were
re-styled but still plotted v1.1 data, so none of the existing artwork can be kept.

Style follows the published figures: Okabe-Ito palette, DejaVu Sans, no top/right
spines, 300 dpi PNG + SVG.
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
OK = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#999999',
      '#8d6a2a', '#13315c']
OUT = 'figures_v12'
import os; os.makedirs(OUT, exist_ok=True)

d = pd.read_csv('corpus_v12.tsv', sep='\t', dtype=str, low_memory=False)
d['rc'] = pd.to_numeric(d.read_count, errors='coerce')
d['bc'] = pd.to_numeric(d.base_count, errors='coerce')
d['yr'] = pd.to_numeric(d.year, errors='coerce')
N = len(d)
R = json.load(open('run_v12/data/mena_rigorous_results.json'))
PC = json.load(open('percategory_v12.json'))
PH = {'not collected', 'not applicable', 'not available', 'unknown', 'missing',
      'none', 'null', 'na', 'n/a', '-', ''}
def filled(s): return ~s.fillna('').astype(str).str.strip().str.lower().isin(PH)

CATORD = ['Environment', 'Human', 'Other', 'Plant', 'Animal', 'Food', 'Clinical', 'Fungal', 'Viral']
CATLAB = {'Other': 'Other/Unclassified'}
PLATLAB = {'ILLUMINA': 'Illumina', 'LS454': 'Roche 454', 'OXFORD_NANOPORE': 'Oxford Nanopore',
           'PACBIO_SMRT': 'PacBio SMRT', 'ION_TORRENT': 'Ion Torrent',
           'BGISEQ': 'BGI/DNBSEQ', 'DNBSEQ': 'BGI/DNBSEQ'}

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight')
    plt.close(fig); print('  ', name)

def bar_labels(ax, bars, vals, fmt='{:,}', dx=0.01):
    m = max(vals) if len(vals) else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + m*dx, b.get_y() + b.get_height()/2,
                fmt.format(v), va='center', fontsize=8.5)

# ---------------------------------------------------------------- F1 countries
cc = d.country.value_counts()
fig, ax = plt.subplots(figsize=(8, 7))
b = ax.barh(range(len(cc))[::-1], cc.values, color=OK[0])
ax.set_yticks(range(len(cc))[::-1]); ax.set_yticklabels(cc.index)
bar_labels(ax, b, cc.values)
ax.set_xlabel('Number of metagenomic runs'); ax.set_title(f'Metagenomic runs per MENA country (n = {N:,})')
ax.set_xlim(0, cc.max()*1.13); save(fig, 'Figure01_country_runs')

# ------------------------------------------------------------- F2 temporal a,b
yr = d.yr.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(yr.index, yr.values, color=OK[0], label='runs released')
ax2 = ax.twinx(); ax2.plot(yr.index, yr.values.cumsum(), color=OK[1], lw=2, label='cumulative')
ax2.set_ylabel('cumulative runs', color=OK[1]); ax2.spines['right'].set_visible(True)
for y in (2024, 2025, 2026):
    if y in yr.index: ax.axvspan(y-0.5, y+0.5, color='0.85', alpha=.55, zorder=0)
ax.set_xlabel('year of first public release'); ax.set_ylabel('runs')
ax.set_title('a  Annual deposition (2024–2026 shaded: latency-censored)', loc='left')
save(fig, 'Figure02a_temporal')

top6 = list(cc.head(6).index)
w = d[d.yr.between(2013, 2026)]
piv = (w[w.country.isin(top6)].pivot_table(index='yr', columns='country',
       values='run_accession', aggfunc='size').fillna(0).reindex(columns=top6))
fig, ax = plt.subplots(figsize=(9, 4.4))
bot = np.zeros(len(piv))
for i, c in enumerate(top6):
    ax.bar(piv.index, piv[c].values, bottom=bot, color=OK[i], label=c); bot += piv[c].values
ax.set_xlabel('year of first public release'); ax.set_ylabel('runs')
ax.legend(frameon=False, fontsize=8.5, ncol=3)
ax.set_title('b  Top six contributing countries by year', loc='left')
save(fig, 'Figure02b_temporal_by_country')

# --------------------------------------------------------------- F3 categories
bcv = d.broad_category.value_counts().reindex(CATORD).dropna()
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
b = ax.barh(range(len(bcv))[::-1], bcv.values, color=OK[0])
ax.set_yticks(range(len(bcv))[::-1]); ax.set_yticklabels([CATLAB.get(x, x) for x in bcv.index])
for bb, v in zip(b, bcv.values):
    pct = 100*v/N
    ax.text(bb.get_width()+bcv.max()*.012, bb.get_y()+bb.get_height()/2,
            '<0.1%' if pct < 0.1 else f'{pct:.1f}%', va='center', fontsize=9)
ax.set_xlabel('runs'); ax.set_xlim(0, bcv.max()*1.14)
ax.set_title(f'a  Broad category (n = {N:,})', loc='left', fontweight='bold')

# the 73-rule BioSample types, with the catch-all written Other/Unclassified as it is
# in Section 3.4, Supplementary Table S1 and Figures 16 and 17
_BS = json.load(open('biosample_both.json'))['v12']
top15 = _BS['top_overall'][:15]
lbl = [('Other/Unclassified' if x['biosample_type'] == 'Unspecified' else x['biosample_type'])
       for x in top15]
val = [x['n_runs'] for x in top15]
ax = axes[1]
b = ax.barh(range(len(top15))[::-1], val, color=OK[2])
ax.set_yticks(range(len(top15))[::-1]); ax.set_yticklabels(lbl, fontsize=8.5)
for bb, v in zip(b, val):
    ax.text(v + max(val)*.012, bb.get_y()+bb.get_height()/2, f'{v:,}', va='center', fontsize=8)
ax.set_xlabel('runs'); ax.set_xlim(0, max(val)*1.16)
ax.set_title('b  Top 15 BioSample types (73-rule harmonization)', loc='left', fontweight='bold')
save(fig, 'Figure03_categories')

# ------------------------------------------------------------- F4 hosts/sites
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, cat, col, ttl in [(axes[0], 'Human', OK[3], 'a  Human body sites'),
                          (axes[1], 'Animal', OK[2], 'b  Animal hosts')]:
    s = d[d.broad_category == cat].specific_category.value_counts().head(9)
    b = ax.bar(range(len(s)), s.values, color=col)
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=35, ha='right', fontsize=8.5)
    for bb, v in zip(b, s.values):
        ax.text(bb.get_x()+bb.get_width()/2, v + s.max()*.02, f'{v:,}', ha='center', fontsize=8.5)
    ax.set_ylabel('Runs'); ax.set_ylim(0, s.max()*1.14)
    ax.set_title(f'{ttl} (n = {int(s.sum()):,})', loc='left', fontweight='bold')
save(fig, 'Figure04_host_bodysite')

# ---------------------------------------------------------------- F5 platforms
pl = d.instrument_platform.map(lambda x: PLATLAB.get(x, 'Other/Unclassified')).value_counts()
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
PCOL = {name: OK[i % len(OK)] for i, name in enumerate(pl.index)}   # shared by both panels
b = ax.barh(range(len(pl))[::-1], pl.values, color=[PCOL[n] for n in pl.index])
ax.set_yticks(range(len(pl))[::-1]); ax.set_yticklabels(pl.index)
for bb, v in zip(b, pl.values):
    ax.text(bb.get_width()+pl.max()*.012, bb.get_y()+bb.get_height()/2,
            f'{v:,} ({100*v/N:.2f}%)', va='center', fontsize=8.5, fontweight='bold')
ax.set_xscale('log'); ax.set_xlabel('runs (log scale)')
ax.set_title(f'a  Sequencing platform share (n = {N:,})', loc='left', fontweight='bold')
ax = axes[1]
wy = d[d.yr.between(2013, 2024)].copy()
wy['P'] = wy.instrument_platform.map(lambda x: PLATLAB.get(x, 'Other/Unclassified'))
pv = wy.pivot_table(index='yr', columns='P', values='run_accession', aggfunc='size').fillna(0)
pv = pv.div(pv.sum(axis=1), axis=0)*100
bot = np.zeros(len(pv))
for c in [x for x in pl.index if x in pv.columns]:      # same order and colour as panel a
    ax.bar(pv.index, pv[c].values, bottom=bot, color=PCOL[c], label=c); bot += pv[c].values
ax.set_ylabel('% of runs in year'); ax.set_xlabel('year of first public release')
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='lower center', bbox_to_anchor=(.5, -.42))
ax.set_title('b  Platform composition by year', loc='left', fontweight='bold')
save(fig, 'Figure05_platforms')

# -------------------------------------------------------------- F6 instruments
im = d.instrument_model.value_counts().head(15)
fig, axes = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw={'height_ratios': [1, 1.25]})
ax = axes[0]
b = ax.barh(range(len(im))[::-1], im.values, color=OK[0])
ax.set_yticks(range(len(im))[::-1]); ax.set_yticklabels(im.index)
bar_labels(ax, b, im.values)
ax.set_xlabel('runs (exact counts)'); ax.set_xlim(0, im.max()*1.13)
ax.set_title(f'a  Top 15 instrument models — exact run counts (n = {N:,})', loc='left', fontweight='bold')
ax = axes[1]
top15c = list(cc.head(15).index); top10m = list(im.head(10).index)
H = np.zeros((len(top15c), len(top10m)))
for i, c in enumerate(top15c):
    sub = d[d.country == c]
    for j, m in enumerate(top10m):
        H[i, j] = 100*(sub.instrument_model == m).sum()/len(sub)
im_ = ax.imshow(H, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(top10m))); ax.set_xticklabels(top10m, rotation=35, ha='right', fontsize=8.5)
ax.set_yticks(range(len(top15c))); ax.set_yticklabels(top15c, fontsize=9)
for i in range(len(top15c)):
    for j in range(len(top10m)):
        if H[i, j] >= 1:
            ax.text(j, i, f'{H[i,j]:.0f}', ha='center', va='center', fontsize=7.5,
                    color='white' if H[i, j] > 55 else 'black')
plt.colorbar(im_, ax=ax, label='% within country', fraction=.03)
ax.set_title('b  Country × top-10 instrument model (% within country)', loc='left', fontweight='bold')
save(fig, 'Figure06_instruments')

# ------------------------------------------------------------- F7 library prep
lay = d.library_layout.fillna('unspecified').str.upper().value_counts()
sel = d.library_selection.fillna('unspecified').value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, s, ttl in [(axes[0], lay, 'a  Library layout'), (axes[1], sel.head(6), 'b  Library selection')]:
    ax.pie(s.values, labels=[f'{i}\n{v:,} ({100*v/N:.1f}%)' for i, v in s.items()],
           colors=OK[:len(s)], textprops={'fontsize': 8.5}, startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax.set_title(ttl, loc='left', fontweight='bold')
save(fig, 'Figure07_library_prep')

# ------------------------------------------------- F8 marker-gene ANNOTATION
amp = d[d.data_subtype == 'amplicon_metagenomics']
tg = amp.sample_target_gene.fillna('').astype(str).str.strip()
ok = filled(amp.sample_target_gene)
def norm(v):
    u = v.upper()
    if '16S' in u: return '16S rRNA'
    if 'ITS' in u: return 'ITS (fungal)'
    if '18S' in u: return '18S rRNA'
    return 'other marker'
mk = tg[ok].map(norm).value_counts()
cov = pd.Series({'target_gene': filled(amp.sample_target_gene).sum(),
                 'target_subfragment': filled(amp.sample_target_subfragment).sum(),
                 'pcr_primers': filled(amp.sample_pcr_primers).sum()})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
b = ax.barh(range(len(cov))[::-1], 100*cov.values/len(amp), color=OK[5])
ax.set_yticks(range(len(cov))[::-1]); ax.set_yticklabels(cov.index)
for bb, v, n_ in zip(b, 100*cov.values/len(amp), cov.values):
    ax.text(bb.get_width()+.15, bb.get_y()+bb.get_height()/2, f'{v:.1f}%  (n = {n_:,})',
            va='center', fontsize=9)
ax.set_xlabel(f'% of the {len(amp):,} amplicon runs'); ax.set_xlim(0, max(100*cov.values/len(amp))*1.6)
ax.set_title('a  Structured marker annotation is sparse', loc='left', fontweight='bold')
ax = axes[1]
b = ax.barh(range(len(mk))[::-1], mk.values, color=OK[0])
ax.set_yticks(range(len(mk))[::-1]); ax.set_yticklabels(mk.index)
for bb, v in zip(b, mk.values):
    ax.text(bb.get_width()+mk.max()*.02, bb.get_y()+bb.get_height()/2,
            f'{v:,} ({100*v/mk.sum():.1f}%)', va='center', fontsize=9)
ax.set_xlabel(f'runs (of the {int(ok.sum()):,} with an explicit target_gene)')
ax.set_xlim(0, mk.max()*1.3)
ax.set_title('b  Marker composition of the annotated subset', loc='left', fontweight='bold')
save(fig, 'Figure08_marker_annotation')

print('\npart 1 complete (F1-F8)')
