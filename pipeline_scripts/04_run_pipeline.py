#!/usr/bin/env python
"""
Step 4. Re-run the ORIGINAL analysis pipeline (original_pipeline/rigorous_analysis.py)
unchanged, on both corpora.

Only the three module-level path constants are redirected. No analysis code is
modified, so the v1.1 run is a true reproduction check: if it returns the
published values, the v1.2 run can be trusted for every quantity the script
produces (MIxS, GPS, latency, duplicates, diversity, rarefaction, Jaccard/PCoA,
PERMANOVA, IndVal, chi-square residuals, Mann-Kendall, Moran's I, NLP).

usage:  python 04_run_pipeline.py v11 | v12
"""
import sys, os, re, shutil, pandas as pd

which = sys.argv[1]
from pathlib import Path
SRC = str(Path(__file__).resolve().parent / 'rigorous_analysis.py')
BASE = os.path.abspath(f'run_{which}')
for sub in ('data', 'figures', 'tables'):
    os.makedirs(f'{BASE}/{sub}', exist_ok=True)

# ---- build the input file in the layout the script expects ----------------
d = pd.read_csv(f'corpus_{which}.tsv', sep='\t', dtype=str, low_memory=False)
ren = {}
for c in list(d.columns):
    if c.endswith('_meta'):
        base = c[:-5]
        if base not in d.columns:
            ren[c] = base
        else:                       # prefer the pre-filter metadata copy
            d = d.drop(columns=[base]); ren[c] = base
d = d.rename(columns=ren)
need = ['run_accession', 'sample_accession', 'bioproject', 'country', 'year',
        'scientific_name', 'library_strategy', 'library_source', 'library_selection',
        'library_layout', 'data_subtype', 'instrument_platform', 'instrument_model',
        'broad_category', 'specific_category', 'host', 'host_body_site',
        'isolation_source', 'study_title', 'sample_title', 'read_count', 'base_count',
        'first_public', 'collection_date', 'study_accession', 'environment_biome', 'environment_feature',
        'environment_material', 'sample_lat_lon', 'project_name']
missing = [c for c in need if c not in d.columns]
assert not missing, f'missing columns: {missing}'
inp = f'{BASE}/data/mena_metagenomics_clean.tsv'
d[need].to_csv(inp, sep='\t', index=False)
print(f'input: {len(d):,} runs -> {inp}')

# ---- redirect paths, run the original source verbatim ---------------------
code = open(SRC).read()
code = code.replace('DATA = "data/mena_metagenomics_clean.tsv"', f'DATA = {inp!r}')
code = code.replace('FIGDIR = "figures"', f'FIGDIR = {BASE + "/figures"!r}')
code = code.replace('DATADIR = "data"', f'DATADIR = {BASE + "/data"!r}')
code = code.replace('TBLDIR = "tables"', f'TBLDIR = {BASE + "/tables"!r}')
assert inp in code and BASE in code, 'path redirection failed'

import matplotlib
matplotlib.use('Agg')
g = {'__name__': '__main__', '__file__': SRC}
exec(compile(code, SRC, 'exec'), g)
print(f'\ndone -> {BASE}/data/mena_rigorous_results.json')
