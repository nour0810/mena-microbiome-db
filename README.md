# MENA Microbiome Database

> **The first region-scale, harmonized metagenomic catalog for the Middle East and North Africa.**

[![GitHub Pages](https://img.shields.io/badge/Web%20Platform-Live-brightgreen)](https://nour0810.github.io/mena-microbiome-db/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

The **MENA Microbiome Database** consolidates all publicly available metagenomic records from **24 Middle Eastern and North African countries** into a single, harmonized, and fully searchable resource.

| Metric | Value |
|--------|-------|
| Metagenomic runs | **60,126** |
| Unique BioSamples | **51,365** |
| BioProjects | **2,373** |
| Countries covered | **24** |
| Data span | **2008 – 2026** |
| Primary sequencing platform | Illumina (92.7%) |

Data were retrieved from **ENA**, **NCBI SRA**, and **PubMed**-linked BioProjects, harmonized to a common schema, and enriched with BioSample / experiment / study XML attributes. Community metagenomics runs are algorithmically separated from single-organism genomics using a transparent, rule-based classifier.

---

## 🌐 Interactive Web Platform

The platform is deployed via **GitHub Pages** and requires no installation.

**→ [Open the MENA Microbiome Platform](https://nour0810.github.io/mena-microbiome-db/)**

Features:
- Full-text & multi-filter search across all 60,126 runs
- Per-country, per-category, and per-biome breakdown
- Ecology & diversity analytics (Shannon, Simpson, PCoA, PERMANOVA)
- Geospatial sampling map with Moran's I spatial clustering
- NLP-derived thematic cluster explorer (9 clusters, silhouette = 0.59)
- Export to TSV / CSV / XLSX / JSON

> **GitHub Pages setup:** In your repository settings go to **Settings → Pages**, set the source branch to `main` and the folder to `/ (root)`. The `index.html` file is the platform entry point.

---

## Repository Structure

```
mena-microbiome-db/
├── index.html                        # Interactive web platform (GitHub Pages entry point)
├── README.md
├── requirements.txt                  # Python dependencies for all scripts
├── .gitignore
└── scripts/
    ├── 01_fetch_mena_v5_patched.py   # Step 1 — Data acquisition from ENA & NCBI SRA
    ├── split_genomics_metagenomics.py # Step 2 — Classify metagenomics vs single-organism
    ├── analyze_mena.py               # Step 3 — Core downstream analysis & figures
    ├── biosample_descriptive.py      # Step 4 — BioSample-type descriptive statistics
    ├── per_category_permanova.py     # Step 5 — Per-category PERMANOVA (country effect)
    ├── rigorous_analysis.py          # Step 6 — Full publication-quality analytical suite
    └── ml_analysis.py               # Step 7 — ML / NLP supplementary analyses
```

---

## Analysis Pipeline

The scripts are designed to be run **sequentially**. Each step depends on the output of the previous one.

```
ENA / NCBI SRA / PubMed
        │
        ▼
01_fetch_mena_v5_patched.py
        │  mena_all_runs.csv  (raw, unclassified)
        ▼
split_genomics_metagenomics.py
        │  mena_metagenomics.tsv        ← main dataset
        │  mena_single_organism.tsv
        │  mena_ambiguous.tsv
        ▼
analyze_mena.py  ──────────────────────  out/figures/*.png|svg
        │                                out/tables/*.xlsx
        ▼
biosample_descriptive.py  ─────────────  BioSample taxonomy + per-category stats
        │
        ▼
per_category_permanova.py  ────────────  Country effect within each sample category
        │
        ▼
rigorous_analysis.py  ─────────────────  mena_rigorous_results.json
        │                                TF-IDF signatures, Mann-Kendall trends
        │                                Moran's I, diversity rarefaction curves
        ▼
ml_analysis.py  ───────────────────────  ML JSON blobs consumed by index.html
                                         Supplementary PCA / NMF / LDA figures
```

---

## Scripts — Quick Reference

### `01_fetch_mena_v5_patched.py` — Data Acquisition
Queries ENA Portal API and NCBI SRA (via Biopython Entrez) for all metagenomic records from 24 MENA countries. Supports checkpoint/resume for long runs.

```bash
# Full acquisition (resumes automatically from checkpoint)
python scripts/01_fetch_mena_v5_patched.py

# Test API connectivity only
python scripts/01_fetch_mena_v5_patched.py --test

# BioProject discovery only (no run-level fetch)
python scripts/01_fetch_mena_v5_patched.py --discovery-only
```

**Output:** `mena_all_runs.csv`

---

### `split_genomics_metagenomics.py` — Classification
Separates true community metagenomics from single-organism isolate sequencing using a rule-based algorithm with audit-friendly output.

```bash
python scripts/split_genomics_metagenomics.py
```

**Output:** `mena_metagenomics.tsv`, `mena_single_organism.tsv`, `mena_ambiguous.tsv`

---

### `analyze_mena.py` — Core Analysis
Produces publication-quality figures (PNG + SVG) and tables (XLSX) from `mena_metagenomics.tsv`. Covers run counts, country distributions, sequencing platform breakdown, and ecological compartment composition.

```bash
python scripts/analyze_mena.py
```

**Output:** `out/figures/`, `out/tables/`

---

### `biosample_descriptive.py` — BioSample Statistics
Harmonizes `isolation_source`, `environment_material`, and `specific_category` into a clean BioSample taxonomy, then generates per-category descriptive statistics.

```bash
python scripts/biosample_descriptive.py
```

---

### `per_category_permanova.py` — Stratified PERMANOVA
Tests whether the country effect on metagenome-type composition persists within each broad ecological category (Human, Animal, Environment, Plant, Food), removing sample-type confounding.

```bash
python scripts/per_category_permanova.py
```

Key result: country explains significant variance in Plant (R²=9.4%), Animal (R²=7.9%), and Environment (R²=7.0%) — but not Human (p=0.225).

---

### `rigorous_analysis.py` — Publication-Quality Analytical Suite
The main analysis engine, covering:
- **Data quality:** MIxS-MIMS proxy completeness, GPS coordinate validation, duplicate detection, temporal submission lag
- **Ecological diversity:** rarefaction, Shannon/Simpson indices, Jaccard distances, PCoA
- **Inferential statistics:** PERMANOVA, IndVal species indicator values, chi-square residuals, Mann-Kendall temporal trends
- **Geospatial:** sampling density maps, Moran's I spatial autocorrelation
- **NLP:** per-country TF-IDF signatures, K-Means thematic clustering of study titles (9 clusters, silhouette=0.59)

```bash
python scripts/rigorous_analysis.py
```

**Output:** `mena_rigorous_results.json`, `out/figures/`, `out/tables/`

---

### `ml_analysis.py` — Supplementary ML / NLP
Generates JSON blobs consumed by the interactive platform and supplementary figures for:
- PCA, NMF, LDA topic modelling
- Random Forest & MLP classification of sample categories
- t-SNE embedding visualizations

```bash
python scripts/ml_analysis.py
```

---

## Installation

**Python 3.8 or higher is required.**

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mena-microbiome-db.git
cd mena-microbiome-db

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The raw data files (`mena_metagenomics.tsv`, etc.) are not included in this repository due to their size (~several hundred MB). Run `01_fetch_mena_v5_patched.py` to reproduce them from scratch, or contact the authors for a hosted download link.

---

## Data Sources

| Source | Access method |
|--------|---------------|
| [ENA Portal](https://www.ebi.ac.uk/ena/portal/) | REST API |
| [NCBI SRA](https://www.ncbi.nlm.nih.gov/sra) | Biopython Entrez |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Affiliation-linked BioProjects |

All records are publicly available under their original repository licenses. This database is a derivative metadata catalog and does not redistribute raw sequencing reads.

---

## Citation

If you use the MENA Microbiome Database or this platform in your research, please cite:

> *[Citation will be added upon publication]*

---

## License

This project is released under the [MIT License](LICENSE).  
Raw sequencing data remain under the original repository terms (ENA / NCBI SRA open-access data policy).

---

## Contact

For questions, data updates, or collaboration inquiries, please open an [issue](https://github.com/YOUR_USERNAME/mena-microbiome-db/issues) on GitHub.
