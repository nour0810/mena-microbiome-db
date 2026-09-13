# MENA Microbiome Database, v1.2 analysis pipeline

These scripts build the v1.2 corpus from the v1.1 release and reproduce every number,
figure and supplementary table reported in the manuscript. Every path is resolved from
the directory the script lives in, so the bundle runs as downloaded.

`rigorous_analysis.py` is the analysis pipeline of the first submission, carrying two
corrections made in response to review. The ordination, which the first submission
computed with `sklearn.manifold.MDS`, is now the classical principal coordinates
analysis the manuscript describes: eigendecomposition of the double-centred Jaccard
matrix, which yields the per-axis variance explained that MDS cannot provide. The
all-category PERMANOVA is now computed with scikit-bio and cross-checked against the
original Anderson (2001) pseudo-F routine, and is accompanied by a PERMDISP dispersion
test and a 20-subsample stability check. Everything else in the file is unchanged apart
from its four module-level path constants (`DATA`, `FIGDIR`, `DATADIR`, `TBLDIR`), which
carry relative defaults here and are in any case redirected at run time by
`04_run_pipeline.py`.

That file is what makes the v1.2 values checkable: run it on the v1.1 corpus and it
reproduces the published v1.1 values exactly (MIxS completeness 73.97%, coordinates
42,714 parsed / 42,137 valid / 577 conflicts, submission lag 1,171 days, Jaccard mean
0.8944, classical PCoA 12.7% and 10.3%, all-category PERMANOVA F = 1.58 and R2 = 4.96%,
all-category PERMDISP F = 2.63, country-by-category chi-square 38,205.62). The v1.2
values therefore come from the same code path, not from a re-implementation.
`biosample_descriptive.py` is released on the same terms and is used the same way by
`07_biosample.py`.

## Order

| script | what it does |
|---|---|
| `01_build_corpus.py` | builds `corpus_v12.tsv` and `corpus_v11.tsv` from the deposited inputs: removes the 4,423 runs the BioProject-level census judged not to be community metagenomes, adds the 791 marker-gene runs recovered from the rejected arm, repairs `year` and `data_subtype` on the recovered records, and joins the pre-filter metadata |
| `04_run_pipeline.py v11\|v12` | runs `rigorous_analysis.py` on either corpus, redirecting only its four path constants. Always run `v11` first as a control |
| `05_percategory_pcoa.py` | per-category PERMANOVA and PERMDISP, classical PCoA, BioSample-type assignment |
| `06_figures.py`, `06b_figures.py`, `06c_figures.py` | all 30 figure panels |
| `07_biosample.py v11\|v12` | BioSample-level counts, via `biosample_descriptive.py` |
| `12_supplementary.py` | Supplementary Tables S1 to S3 |
| `14_permdisp_chao1_stability.py` | all-category PERMDISP, the 20-subsample PERMANOVA stability range, and bias-corrected Chao1. Each routine is run on v1.1 first and must reproduce the published v1.1 value before its v1.2 output is used. The PERMDISP control requires scikit-bio 0.6.x; see Requirements |

## Inputs

Put these three files, all of which are in the Zenodo deposit alongside the released
catalog, in an `inputs/` directory beside the scripts (or point `MENA_DATA` at wherever
you keep them):

| file | what it is | read by |
|---|---|---|
| `mena_database_recategorized.tsv` | the v1.1 release, 60,126 runs | `01_build_corpus.py` |
| `mena_corpus_v1.2.tsv` | the same records carrying the adjudicated census and rejected-arm labels | `01_build_corpus.py` |
| `mena_all_runs_pre_filter.csv` | the 91-column pre-filter metadata joined onto both corpora | `01_build_corpus.py` |

```
MENA_DATA=/path/to/inputs python 01_build_corpus.py     # writes corpus_v11.tsv, corpus_v12.tsv
python 04_run_pipeline.py v11                           # control: must reproduce the v1.1 values above
python 04_run_pipeline.py v12
python 05_percategory_pcoa.py && python 07_biosample.py v12
python 06_figures.py && python 06b_figures.py && python 06c_figures.py
python 14_permdisp_chao1_stability.py
MENA_OUT=/path/to/out python 12_supplementary.py        # defaults to ./supplementary_out
```

`01_build_corpus.py` writes `corpus_v11.tsv` and `corpus_v12.tsv` into the working
directory; every later script reads them from there, so run the steps in one place.

## Requirements

Python 3.10+, with pandas 2.0, numpy >= 1.23, scipy 1.11, scikit-learn 1.3, scikit-bio 0.6 or
later, pymannkendall 1.4, matplotlib 3.7 and openpyxl. The bundle has been run end to end on
scikit-bio 0.7.3 as well as on 0.6.

**PERMDISP: which routine produces which number.** Two different dispersion routines appear in
this bundle and they do not agree, so it is worth being explicit about which one each reported
value comes from.

- `rigorous_analysis.py` and `05_percategory_pcoa.py` call `skbio.stats.distance.permdisp`.
  This is the source of every PERMDISP value reported in the manuscript. On scikit-bio 0.7.3 it
  reproduces the published F statistics exactly: F = 2.63 on the 1,145 qualifying BioProjects of
  v1.1 and F = 2.13 on the 1,124 of v1.2. There is no version incompatibility here and no pin is
  required.
- `14_permdisp_chao1_stability.py` does **not** use scikit-bio. It contains a local
  reimplementation, `permdisp_centroid()`, following Anderson (2006) in PCoA space, which returns
  F = 5.70 on the same v1.1 data. That disagreement is between the reimplementation and
  scikit-bio, not between scikit-bio versions. The manuscript reports the scikit-bio value; the
  local routine is retained only because the same script computes the Chao1 estimators and the
  20-subsample stability range, both of which reproduce exactly on either version.

**Seeding.** `skbio`'s `permanova` and `permdisp` draw their permutations from the global NumPy
RNG and accept no seed argument, so both scripts call `np.random.seed(42)` immediately before each
of them. Without that the permutation p-values are re-drawn on every run. Every other stochastic
step is seeded at 42, except the broad-category validation sample, which uses seed 20260808.

**Permutation counts.** The per-category family in `05_percategory_pcoa.py` uses 9,999
permutations, because at 999 the Monte Carlo standard error on a p-value near 0.05 is about 0.007,
enough to move a borderline q across the threshold between runs. The all-category test retains 999,
since its p-value is at the attainable floor at either count and its stability is characterised
instead by the 20-subsample re-runs.
