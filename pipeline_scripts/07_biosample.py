import sys, os, json, pandas as pd
which = sys.argv[1]
from pathlib import Path
SRC=str(Path(__file__).resolve().parent / 'biosample_descriptive.py')
BASE=os.path.abspath(f'run_{which}')
for s in ('data','figures','tables'): os.makedirs(f'{BASE}/{s}', exist_ok=True)
inp=f'{BASE}/data/mena_metagenomics_clean.tsv'
code=open(SRC).read()
code=code.replace('DATA = "data/mena_metagenomics_clean.tsv"', f'DATA = {inp!r}')
code=code.replace('DATADIR = "data"', f'DATADIR = {BASE+"/data"!r}')
code=code.replace('TBLDIR = "tables"', f'TBLDIR = {BASE+"/tables"!r}')
code=code.replace('FIGDIR = "figures"', f'FIGDIR = {BASE+"/figures"!r}')
assert inp in code and BASE in code, 'path redirection failed' 
import matplotlib; matplotlib.use('Agg')
exec(compile(code, SRC, 'exec'), {'__name__':'__main__','__file__':SRC})
