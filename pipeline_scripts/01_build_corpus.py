#!/usr/bin/env python
"""
Step 1. Build the repaired v1.2 corpus and join the full pre-filter metadata.

Two defects in recomputed/mena_corpus_v1.2.tsv are repaired here:
  (a) the 791 runs recovered from the rejected arm carry no `year`
  (b) the same 791 runs carry no `data_subtype`
Both are mechanically derivable from fields the runs do carry, using the rules
the manuscript already states (year = year of first_public, Methods 2.4;
subtype = amplicon_metagenomics for AMPLICON libraries, Methods 2.3 step 3).
The repair is scoped to those 791 rows only, so no pre-existing label changes.

Output: corpus_v12.tsv  (56,494 rows, full metadata joined)
"""
import pandas as pd

import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
# Inputs are resolved relative to this script, or from MENA_DATA if the large
# files are kept elsewhere:  MENA_DATA=/path/to/data python 01_build_corpus.py
DATA_DIR = Path(os.environ.get('MENA_DATA', HERE / 'inputs'))
REC = str(DATA_DIR) + os.sep
SRC = str(DATA_DIR) + os.sep

v12 = pd.read_csv(REC + 'mena_corpus_v1.2.tsv', sep='\t', dtype=str, low_memory=False)
v11 = pd.read_csv(REC + 'mena_database_recategorized.tsv', sep='\t', dtype=str, low_memory=False)

added_mask = ~v12.run_accession.isin(set(v11.run_accession))
print(f"v1.1 {len(v11)} runs -> v1.2 {len(v12)} runs "
      f"({(~v11.run_accession.isin(set(v12.run_accession))).sum()} removed, {added_mask.sum()} added)")

# (a) year from first_public, added rows only
need = added_mask & v12.year.isna()
v12.loc[need, 'year'] = (v12.loc[need, 'first_public'].astype(str)
                         .str.extract(r'((?:19|20)\d{2})')[0].values)
print(f"  year filled on {need.sum()} added rows; still missing: {v12.year.isna().sum()}")

# (b) data_subtype, added rows only (all 791 are AMPLICON)
need = added_mask & v12.data_subtype.isna() & (v12.library_strategy == 'AMPLICON')
v12.loc[need, 'data_subtype'] = 'amplicon_metagenomics'
print(f"  data_subtype filled on {need.sum()} added rows; "
      f"still blank: {v12.data_subtype.isna().sum()} (all pre-existing, unchanged from v1.1)")

# join the full 91-column pre-filter metadata
keep = ['run_accession', 'library_selection', 'library_layout', 'nominal_length',
        'collection_date', 'first_public', 'sample_title', 'project_name',
        'environment_biome', 'environment_feature', 'environment_material',
        'isolation_source', 'host', 'host_sex', 'host_body_site',
        'sample_host_disease', 'sample_host_age', 'sample_dna_extraction',
        'sample_pcr_primers', 'sample_target_gene', 'sample_target_subfragment',
        'sample_lat_lon', 'exp_library_construction_protocol']
meta = pd.read_csv(SRC + 'mena_all_runs_pre_filter.csv', usecols=keep,
                   dtype=str, low_memory=False).drop_duplicates('run_accession')
print(f"  pre-filter metadata: {len(meta)} runs, {len(keep)} fields")

df = v12.merge(meta, on='run_accession', how='left', suffixes=('', '_meta'))
assert len(df) == len(v12), "join changed row count"
matched = df.first_public_meta.notna().sum()
print(f"  metadata matched for {matched}/{len(df)} runs ({100*matched/len(df):.2f}%)")

df.to_csv('corpus_v12.tsv', sep='\t', index=False)

# same join for v1.1, used as the reproduction control
ctl = v11.merge(meta, on='run_accession', how='left', suffixes=('', '_meta'))
ctl.to_csv('corpus_v11.tsv', sep='\t', index=False)
print(f"\nwrote corpus_v12.tsv ({len(df)}) and corpus_v11.tsv ({len(ctl)})")
