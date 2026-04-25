#!/usr/bin/env python3
"""
MicrobiomeAtlas-MENA — Script 01 v5.2 (patched)
================================================
Patches vs v5_fixed:
  1. Phase E: splits semicolon-joined sample accessions before fetching
  2. Phase F: splits semicolon-joined experiment accessions before fetching
  3. safe_cache: hashes overly long filenames to stay under FS limits

Run:
    python 01_fetch_mena_v5_patched.py           # resume from checkpoint
    python 01_fetch_mena_v5_patched.py --test    # API connectivity test
    python 01_fetch_mena_v5_patched.py --discovery-only
"""

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    import requests
    from Bio import Entrez
except ImportError:
    os.system("pip install biopython pandas requests --quiet")
    import pandas as pd
    import requests
    from Bio import Entrez

# ── CONFIG ────────────────────────────────────────────────────────────────────
Entrez.email   = "nourhouda0810@gmail.com"
Entrez.api_key = ""  # add your NCBI API key here for 3× speed

OUT_DIR   = Path(".")
CACHE_DIR = OUT_DIR / "_cache"
for sub in ["", "/sample", "/experiment", "/study", "/pubmed"]:
    (CACHE_DIR / sub.lstrip("/")).mkdir(exist_ok=True, parents=True)

OUTPUT_RUNS         = OUT_DIR / "mena_all_runs.csv"
OUTPUT_SAMPLES      = OUT_DIR / "mena_all_samples.tsv"
OUTPUT_EXPERIMENTS  = OUT_DIR / "mena_all_experiments.tsv"
OUTPUT_STUDIES      = OUT_DIR / "mena_all_studies.tsv"
OUTPUT_PUBLICATIONS = OUT_DIR / "mena_all_publications.tsv"
OUTPUT_BPS          = OUT_DIR / "mena_all_bioprojects.txt"
OUTPUT_FASTQ        = OUT_DIR / "mena_fastq_urls.txt"
OUTPUT_SRR          = OUT_DIR / "mena_srr_accessions.txt"
OUTPUT_JSON         = OUT_DIR / "mena_summary.json"
OUTPUT_REPORT       = OUT_DIR / "mena_discovery_report.txt"
CHECKPOINT          = CACHE_DIR / "discovery_checkpoint.pkl"

MENA_COUNTRIES = [
    "Tunisia", "Egypt", "Morocco", "Algeria", "Libya", "Sudan", "Mauritania",
    "Saudi Arabia", "Jordan", "Lebanon", "Iraq", "Iran",
    "Palestine", "State of Palestine",
    "Syria", "Syrian Arab Republic",
    "Yemen", "Kuwait", "United Arab Emirates", "Qatar",
    "Bahrain", "Oman", "Turkey", "Djibouti", "Somalia",
]

COUNTRY_CANONICAL = {
    "state of palestine":    "Palestine",
    "palestine":             "Palestine",
    "syrian arab republic":  "Syria",
    "syria":                 "Syria",
    "united arab emirates":  "UAE",
    "uae":                   "UAE",
}
EXCLUDED_LOWER = {"israel"}
MENA_LOWER = {c.lower() for c in MENA_COUNTRIES} | {"uae", "palestine", "syria"}

SEED_BIOPROJECTS = ["PRJNA905672"]   # Mathlouthi et al. Tunisia CRC / archaeome

# ── ENA API ───────────────────────────────────────────────────────────────────
ENA_PORTAL  = "https://www.ebi.ac.uk/ena/portal/api/search"
ENA_REPORT  = "https://www.ebi.ac.uk/ena/portal/api/filereport"
HEADERS     = {"User-Agent": "MicrobiomeAtlas-MENA/5.2 (nourhouda0810@gmail.com)"}
TIMEOUT     = 120

ENA_FIELDS = [
    "run_accession","sample_accession",
    "experiment_accession","study_accession","secondary_study_accession",
    "submission_accession","scientific_name","tax_id",
    "library_strategy","library_source","library_selection",
    "library_layout","library_name","nominal_length","nominal_sdev",
    "instrument_platform","instrument_model",
    "country","location","collection_date","first_public","last_updated",
    "sample_title","experiment_title","study_title",
    "sample_description","study_name",
    "fastq_ftp","fastq_md5","fastq_bytes",
    "submitted_ftp","sra_ftp",
    "read_count","base_count",
    "host","host_tax_id","host_status","host_sex","host_body_site","host_phenotype",
    "environment_biome","environment_feature","environment_material","environmental_sample",
    "isolation_source","collected_by",
    "depth","altitude","temperature","ph","salinity",
    "center_name","broker_name","project_name",
]

# STRICT microbiome filter — no bare WGS (would pull in single-organism isolates)
ENA_MG_QUERY = (
    '(library_source="METAGENOMIC" OR library_source="METATRANSCRIPTOMIC" '
    'OR library_strategy="AMPLICON")'
)
# Broader safety net — anything under the metagenomes taxon
ENA_BROAD_QUERY = '(tax_tree(408169))'


def s(v):
    if v is None: return ""
    if isinstance(v, bytes): return v.decode("utf-8", errors="replace")
    return str(v)


def canon_country(raw):
    if not raw: return ""
    c = s(raw).split(":")[0].strip()
    key = c.lower()
    if key in EXCLUDED_LOWER: return "__EXCLUDED__"
    return COUNTRY_CANONICAL.get(key, c)


# ── PATCH 3: defensive filename sanitization ─────────────────────────────────
def safe_cache(sub, key, fn, max_age=30):
    """Read from disk cache, or call fn() and cache the result.
    Sanitizes keys to produce filesystem-safe filenames under 255 bytes."""
    safe = key.replace("/", "_").replace("\\", "_").replace(";", "_")
    # Filesystem max filename length is 255 bytes — cap well under that
    if len(safe) > 200:
        safe = safe[:150] + "_" + hashlib.md5(safe.encode()).hexdigest()[:12]
    path = CACHE_DIR / sub / f"{safe}.xml"
    if path.exists() and (time.time() - path.stat().st_mtime) / 86400 < max_age:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        content = fn()
        if content:
            path.write_text(content, encoding="utf-8")
        return content or ""
    except Exception as e:
        print(f"    cache {sub}/{key[:60]}: {e}")
        return ""


# ── ENA SEARCH ───────────────────────────────────────────────────────────────
def ena_country(country, verbose=True):
    all_rows = []
    for mg_query in [ENA_MG_QUERY, ENA_BROAD_QUERY]:
        query  = f'country="{country}" AND {mg_query}'
        offset = 0
        while True:
            params = {
                "result":  "read_run",
                "query":   query,
                "fields":  ",".join(ENA_FIELDS),
                "format":  "tsv",
                "limit":   100000,
                "offset":  offset,
            }
            try:
                r = requests.get(ENA_PORTAL, params=params,
                                 headers=HEADERS, timeout=TIMEOUT)
                if r.status_code == 400:
                    if verbose:
                        print(f"    ENA {country} (query={mg_query[:40]}): "
                              f"HTTP 400 — {r.text[:120]}")
                    break
                if r.status_code != 200:
                    if verbose:
                        print(f"    ENA {country}: HTTP {r.status_code}")
                    break
                text = r.text.strip()
                if not text:
                    break
                reader = csv.DictReader(io.StringIO(text), delimiter="\t")
                rows = list(reader)
                all_rows.extend(rows)
                if len(rows) < 100000:
                    break
                offset += 100000
                time.sleep(0.3)
            except requests.ConnectionError as e:
                if verbose:
                    print(f"    ENA {country}: connection error — {e}")
                break
            except Exception as e:
                if verbose:
                    print(f"    ENA {country}: {e}")
                break

    seen = set()
    unique = []
    for row in all_rows:
        ra = row.get("run_accession","")
        if ra and ra not in seen:
            seen.add(ra)
            unique.append(row)

    if verbose:
        print(f"    ENA  {country:<22} {len(unique):>7} runs")
    return unique


def ena_bioproject(bp, verbose=False):
    params = {
        "accession": bp,
        "result":    "read_run",
        "fields":    ",".join(ENA_FIELDS),
        "format":    "tsv",
        "limit":     100000,
    }
    try:
        r = requests.get(ENA_REPORT, params=params,
                         headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        text = r.text.strip()
        if not text:
            return []
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        rows = list(reader)
        if verbose and rows:
            print(f"    ENA BP {bp}: {len(rows)} runs")
        return rows
    except Exception as e:
        if verbose:
            print(f"    ENA BP {bp}: {e}")
        return []


def test_ena():
    print("=" * 55)
    print("ENA API connectivity test")
    print("=" * 55)
    print("\nTest 1 — ENA Portal /search  (Tunisia AMPLICON)")
    try:
        r = requests.get(ENA_PORTAL, params={
            "result": "read_run",
            "query":  'country="Tunisia" AND library_strategy="AMPLICON"',
            "fields": "run_accession,country,scientific_name,library_strategy",
            "format": "tsv", "limit": 3,
        }, headers=HEADERS, timeout=20)
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            print(f"  Rows: {len(lines)-1}")
            if len(lines) > 1: print(f"  Sample: {lines[1][:100]}")
        else:
            print(f"  Body: {r.text[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print("\nTest 2 — ENA filereport  (PRJNA728736)")
    try:
        r = requests.get(ENA_REPORT, params={
            "accession": "PRJNA728736",
            "result":    "read_run",
            "fields":    "run_accession,country,scientific_name",
            "format":    "tsv", "limit": 3,
        }, headers=HEADERS, timeout=20)
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            print(f"  Rows: {len(lines)-1}")
            if len(lines) > 1: print(f"  Sample: {lines[1][:100]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print("\nTest 3 — NCBI SRA Entrez  (Tunisia)")
    try:
        h = Entrez.esearch(db="sra",
                           term='"Tunisia"[geo_loc_name] AND "metagenome"[Organism]',
                           retmax=1)
        rec = Entrez.read(h); h.close()
        print(f"  Count: {rec['Count']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print("\n" + "=" * 55)


# ── NCBI SRA ─────────────────────────────────────────────────────────────────
def ncbi_country(country, verbose=True):
    query = (
        f'("{country}"[geo_loc_name]) AND '
        '("metagenome"[Organism] OR "microbiome"[All Fields] '
        'OR "metagenomic"[All Fields] OR "metagenomics"[All Fields] '
        'OR "microbiota"[All Fields] OR "16S rRNA"[All Fields] '
        'OR "amplicon"[All Fields] OR "ITS"[All Fields])'
    )
    try:
        h = Entrez.esearch(db="sra", term=query, retmax=10000, usehistory="y")
        rec = Entrez.read(h); h.close()
        time.sleep(0.35)
        ids = rec.get("IdList", [])
        if verbose:
            print(f"    NCBI {country:<22} {len(ids):>7} IDs")
        return set(ids)
    except Exception as e:
        if verbose: print(f"    NCBI {country}: {e}")
        return set()


def ncbi_runinfo(sra_ids, batch=100):
    records = []
    ids = list(sra_ids)
    n_batches = (len(ids)+batch-1)//batch
    for i in range(0, len(ids), batch):
        bn = i//batch+1
        if bn % 10 == 0 or bn == 1:
            print(f"    NCBI runinfo {bn}/{n_batches}  ({len(records)} rows)")
        try:
            h = Entrez.efetch(db="sra", id=",".join(ids[i:i+batch]),
                              rettype="runinfo", retmode="text")
            raw = h.read(); h.close()
            time.sleep(0.35)
            content = raw.decode("utf-8",errors="replace") if isinstance(raw,bytes) else raw
            content = content.strip()
            if not content or content.startswith("<"): continue
            for row in csv.DictReader(io.StringIO(content)):
                run = s(row.get("Run","")).strip()
                if run and run[:3] in ("SRR","ERR","DRR"):
                    records.append(row)
        except Exception as e:
            print(f"    NCBI batch {bn} error: {e}")
            time.sleep(2)
    return records


# ── PUBMED BACKFILL ───────────────────────────────────────────────────────────
def pubmed_backfill(country, verbose=True):
    terms = (
        '("microbiome"[Title/Abstract] OR "microbiota"[Title/Abstract] '
        'OR "metagenom*"[Title/Abstract] OR "16S rRNA"[Title/Abstract])'
    )
    query = f'{terms} AND "{country}"[Affiliation]'
    try:
        h = Entrez.esearch(db="pubmed", term=query, retmax=2000)
        rec = Entrez.read(h); h.close()
        time.sleep(0.35)
        pmids = rec.get("IdList",[])
        if not pmids: return set()
        if verbose: print(f"    PubMed {country:<19} {len(pmids):>5} papers")

        bps = set()
        for i in range(0, len(pmids), 200):
            try:
                h = Entrez.elink(dbfrom="pubmed", db="bioproject",
                                 id=",".join(pmids[i:i+200]))
                links = Entrez.read(h); h.close()
                time.sleep(0.35)
                for ls in links:
                    for ldb in ls.get("LinkSetDb",[]):
                        for lk in ldb.get("Link",[]): bps.add(lk["Id"])
            except Exception as e:
                print(f"    PubMed elink: {e}"); time.sleep(2)

        accs = set()
        bps_list = list(bps)
        for i in range(0, len(bps_list), 200):
            try:
                h = Entrez.esummary(db="bioproject", id=",".join(bps_list[i:i+200]))
                summ = Entrez.read(h); h.close()
                time.sleep(0.35)
                for doc in summ.get("DocumentSummarySet",{}).get("DocumentSummary",[]):
                    acc = s(doc.get("Project_Acc",""))
                    if acc.startswith("PRJ"): accs.add(acc)
            except Exception as e:
                print(f"    PubMed esummary: {e}"); time.sleep(2)
        return accs
    except Exception as e:
        if verbose: print(f"    PubMed {country}: {e}")
        return set()


# ── UNIFY ─────────────────────────────────────────────────────────────────────
def unify_ena(row):
    out = {f: s(row.get(f,"")) for f in ENA_FIELDS}
    out["bioproject"]    = out.get("study_accession") or out.get("secondary_study_accession") or ""
    out["country_clean"] = canon_country(out.get("country",""))
    out["source"]        = "ENA"
    return out


def unify_ncbi(row):
    country_raw = row.get("geo_loc_name_country") or row.get("geo_loc_name") or ""
    out = {f: "" for f in ENA_FIELDS}
    out.update({
        "run_accession":      s(row.get("Run","")),
        "sample_accession":   s(row.get("BioSample") or row.get("Sample","")),
        "experiment_accession": s(row.get("Experiment","")),
        "study_accession":    s(row.get("BioProject","")),
        "scientific_name":    s(row.get("ScientificName","")),
        "tax_id":             s(row.get("TaxID","")),
        "library_strategy":   s(row.get("LibraryStrategy","")),
        "library_source":     s(row.get("LibrarySource","")),
        "library_selection":  s(row.get("LibrarySelection","")),
        "library_layout":     s(row.get("LibraryLayout","")),
        "instrument_platform": s(row.get("Platform","")),
        "instrument_model":   s(row.get("Model","")),
        "country":            s(country_raw),
        "collection_date":    s(row.get("CollectionDate","")),
        "sample_title":       s(row.get("SampleName","")),
        "host":               s(row.get("Host","")),
        "host_body_site":     s(row.get("body_site","")),
        "isolation_source":   s(row.get("isolation_source","")),
        "read_count":         s(row.get("spots","")),
        "base_count":         s(row.get("bases","")),
        "center_name":        s(row.get("CenterName","")),
    })
    out["bioproject"]    = out["study_accession"]
    out["country_clean"] = canon_country(country_raw)
    out["source"]        = "NCBI"
    return out


# ── CATEGORIZE ────────────────────────────────────────────────────────────────
def categorize(rec):
    blob = " ".join([s(rec.get(k,"")) for k in [
        "scientific_name","host","host_body_site","host_phenotype",
        "environment_biome","environment_feature","environment_material",
        "isolation_source","sample_title","study_title","library_strategy",
    ]]).lower()
    def has(*ks): return any(k in blob for k in ks)
    if has("human","homo sapiens"):
        if has("gut","fecal","faecal","stool","colon","intestin","rectal"): return "Human","Gut"
        if has("oral","saliva","mouth","dental","plaque","tongue"):          return "Human","Oral"
        if has("skin","wound","dermis"):                                    return "Human","Skin"
        if has("vagin","cervic"):                                           return "Human","Vaginal"
        if has("lung","bronch","respir","nasal","trachea","sputum"):        return "Human","Respiratory"
        if has("urin","bladder","kidney"):                                  return "Human","Urinary"
        if has("blood","serum","plasma"):                                   return "Human","Blood"
        if has("milk","breast"):                                            return "Human","Breast milk"
        return "Human","Other"
    if has("camel","dromedary"):                                            return "Animal","Camel"
    if has("bovine","cattle","cow","bos taurus"):                          return "Animal","Cattle"
    if has("sheep","ovine","ovis"):                                        return "Animal","Sheep"
    if has("goat","caprine","capra"):                                      return "Animal","Goat"
    if has("chicken","poultry","gallus","broiler"):                        return "Animal","Poultry"
    if has("fish","tilapia","salmon","aqua"):                              return "Animal","Fish"
    if has("bee","honeybee","insect","locust"):                            return "Animal","Insect"
    if has("mouse","rat","murine","mus musculus"):                         return "Animal","Rodent"
    if has("rhizosphere","phyllosphere","endophyte","root ","leaf ",
           "date palm","olive","wheat","tomato"):                           return "Plant","Plant-associated"
    if has("soil","sediment","desert","sand"):                             return "Environment","Soil/Sediment"
    if has("marine","ocean","sea ","seawater"):                            return "Environment","Marine"
    if has("freshwater","river","lake","stream","spring"):                 return "Environment","Freshwater"
    if has("wastewater","sewage","effluent","sludge","activated"):         return "Environment","Wastewater"
    if has("ferment","dairy","cheese","yogurt","kefir","laban","food"):    return "Food","Food/Fermented"
    if has("hospital","clinical","biofilm","icu"):                         return "Clinical","Hospital"
    if has("air","indoor","dust"):                                         return "Environment","Air/Indoor"
    if has("virome","viral metagenome","phage"):                           return "Viral","Virome"
    if has("myco","fung","its"):                                           return "Fungal","Mycobiome"
    return "Other","Unclassified"


# ── XML ENRICHMENT ────────────────────────────────────────────────────────────
PROMOTE_KEYS = {
    "host_disease":     ["host disease","disease","disease_status","health state","phenotype"],
    "host_age":         ["host age","age"],
    "host_sex":         ["host sex","sex","gender"],
    "host_bmi":         ["bmi","host bmi"],
    "host_diet":        ["diet","host diet"],
    "host_antibiotics": ["antibiotics","antibiotic use","antibiotic history"],
    "dna_extraction":   ["dna extraction","extraction method","extraction kit"],
    "pcr_primers":      ["pcr primers","primers","target_primer","fwd_primer"],
    "target_gene":      ["target gene","amplicon"],
    "target_subfragment":["target subfragment","16s region","variable region"],
    "lat_lon":          ["lat_lon","lat lon"],
}

def _xml(url, cache_sub, acc):
    def _do():
        try:
            r = requests.get(f"https://www.ebi.ac.uk/ena/browser/api/xml/{acc}",
                             headers=HEADERS, timeout=TIMEOUT)
            time.sleep(0.05)
            return r.text if r.status_code==200 else ""
        except: return ""
    return safe_cache(cache_sub, acc, _do)


def parse_sample(xml_text):
    if not xml_text: return {}
    try: root = ET.fromstring(xml_text)
    except: return {}
    out = {}
    attrs = {}
    for attr in root.iter("SAMPLE_ATTRIBUTE"):
        tag = (attr.findtext("TAG") or "").strip()
        val = (attr.findtext("VALUE") or "").strip()
        unit = (attr.findtext("UNITS") or "").strip()
        if tag: attrs[tag] = f"{val} {unit}".strip() if unit else val
    out["sample_attributes_json"] = json.dumps(attrs, ensure_ascii=False)
    al = {k.lower(): v for k,v in attrs.items()}
    for col, aliases in PROMOTE_KEYS.items():
        out[col] = next((al[a] for a in aliases if a in al), "")
    for s_el in root.iter("SAMPLE"):
        out["xml_center"] = s_el.get("center_name","")
    for sc in root.iter("SCIENTIFIC_NAME"):
        out["xml_scientific_name"] = (sc.text or "").strip(); break
    return out


def parse_experiment(xml_text):
    if not xml_text: return {}
    try: root = ET.fromstring(xml_text)
    except: return {}
    out = {}
    for ld in root.iter("LIBRARY_DESCRIPTOR"):
        out["library_construction_protocol"] = (ld.findtext("LIBRARY_CONSTRUCTION_PROTOCOL") or "").strip()
        out["library_strategy_xml"]  = (ld.findtext("LIBRARY_STRATEGY") or "").strip()
        out["library_source_xml"]    = (ld.findtext("LIBRARY_SOURCE") or "").strip()
    for plat in ["ILLUMINA","OXFORD_NANOPORE","ION_TORRENT","PACBIO_SMRT","LS454"]:
        for p in root.iter(plat):
            inst = p.findtext("INSTRUMENT_MODEL")
            if inst: out["instrument_xml"] = inst.strip(); out["platform_xml"] = plat
    return out


def parse_study(xml_text):
    if not xml_text: return {}
    try: root = ET.fromstring(xml_text)
    except: return {}
    out = {}
    for desc in root.iter("DESCRIPTOR"):
        out["study_title_xml"]   = (desc.findtext("STUDY_TITLE") or "").strip()
        out["study_abstract"]    = (desc.findtext("STUDY_ABSTRACT") or "").strip()
        out["study_description"] = (desc.findtext("STUDY_DESCRIPTION") or "").strip()
        st = desc.find("STUDY_TYPE")
        out["study_type"] = st.get("existing_study_type","") if st is not None else ""
    pmids = []
    for xr in root.iter("XREF_LINK"):
        db  = (xr.findtext("DB") or "").lower()
        val = (xr.findtext("ID") or "").strip()
        if db in ("pubmed","pmid") and val: pmids.append(val)
    out["linked_pmids"] = ",".join(pmids)
    return out


def parse_pubmed(xml_text):
    if not xml_text: return {}
    try: root = ET.fromstring(xml_text)
    except: return {}
    out = {}
    for art in root.iter("PubmedArticle"):
        out["pmid"]    = (art.findtext(".//PMID") or "").strip()
        out["title"]   = (art.findtext(".//ArticleTitle") or "").strip()
        out["journal"] = (art.findtext(".//Journal/Title") or "").strip()
        out["year"]    = ((art.findtext(".//PubDate/Year") or
                           art.findtext(".//PubDate/MedlineDate") or ""))[:4]
        parts = []
        for ab in art.iter("AbstractText"):
            label = ab.get("Label"); txt = "".join(ab.itertext()).strip()
            parts.append(f"{label}: {txt}" if label else txt)
        out["abstract"] = " ".join(parts)
        authors = []
        for au in art.iter("Author"):
            last = au.findtext("LastName") or ""; init = au.findtext("Initials") or ""
            if last: authors.append(f"{last} {init}".strip())
        out["authors"] = "; ".join(authors)
        for aid in art.iter("ArticleId"):
            if aid.get("IdType")=="doi": out["doi"]=(aid.text or "").strip(); break
        break
    return out


# ── DISCOVERY ─────────────────────────────────────────────────────────────────
def discovery():
    if CHECKPOINT.exists():
        print(f"[Checkpoint] loading from {CHECKPOINT}")
        print(f"  (delete it to force re-discovery)")
        with open(CHECKPOINT,"rb") as f:
            unified, all_bps = pickle.load(f)
        print(f"  {len(unified)} runs, {len(all_bps)} BioProjects")
        return unified, all_bps

    unified = {}
    all_bps = set()

    print("\n[A] ENA Portal API — per country")
    for country in MENA_COUNTRIES:
        for row in ena_country(country):
            rec = unify_ena(row)
            run = rec["run_accession"]
            if not run or rec["country_clean"] == "__EXCLUDED__": continue
            unified[run] = rec
            if rec["bioproject"].startswith("PRJ"): all_bps.add(rec["bioproject"])
        time.sleep(0.3)
    print(f"  ENA subtotal: {len(unified)} unique runs")

    print("\n[B] NCBI SRA — per country")
    ncbi_ids = set()
    for country in MENA_COUNTRIES:
        ncbi_ids |= ncbi_country(country)
        time.sleep(0.2)
    print(f"  NCBI IDs: {len(ncbi_ids)}  (fetching runinfo...)")
    added = 0
    for row in ncbi_runinfo(ncbi_ids):
        rec = unify_ncbi(row)
        run = rec["run_accession"]
        if not run or rec["country_clean"] == "__EXCLUDED__": continue
        if run not in unified:
            unified[run] = rec
            added += 1
        if rec["bioproject"].startswith("PRJ"): all_bps.add(rec["bioproject"])
    print(f"  NCBI added {added} new runs")

    print("\n[C] PubMed affiliation backfill")
    backfill = set()
    for country in MENA_COUNTRIES:
        backfill |= pubmed_backfill(country)
        time.sleep(0.2)
    print(f"  {len(backfill)} BioProjects from PubMed")
    new_pm = 0
    for bp in backfill:
        if bp in all_bps: continue
        for row in ena_bioproject(bp):
            rec = unify_ena(row)
            run = rec["run_accession"]
            if not run or rec["country_clean"] == "__EXCLUDED__": continue
            if rec["country_clean"] and rec["country_clean"].lower() not in MENA_LOWER: continue
            if run not in unified:
                unified[run] = rec
                new_pm += 1
            all_bps.add(bp)
        time.sleep(0.2)
    print(f"  PubMed backfill added {new_pm} new runs")

    print("\n[D] Manual seed BioProjects")
    for bp in SEED_BIOPROJECTS:
        rows = ena_bioproject(bp, verbose=True)
        added_seed = 0
        for row in rows:
            rec = unify_ena(row)
            run = rec["run_accession"]
            if not run: continue
            if run not in unified:
                unified[run] = rec
                added_seed += 1
            all_bps.add(bp)
        print(f"  {bp}: {len(rows)} runs ({added_seed} new)")
        time.sleep(0.3)

    with open(CHECKPOINT,"wb") as f:
        pickle.dump((unified, all_bps), f)
    print(f"\n  Checkpoint saved.")
    return unified, all_bps


# ── ENRICHMENT (PATCHED) ──────────────────────────────────────────────────────
def enrichment(unified, all_bps):
    records = [r for r in unified.values()
               if r["country_clean"] not in ("__EXCLUDED__","")
               and r["country_clean"].lower() not in EXCLUDED_LOWER]
    for r in records:
        b, sp = categorize(r)
        r["broad_category"] = b; r["specific_category"] = sp

    uniq_samples = {r["sample_accession"] for r in records if r["sample_accession"]}
    uniq_exps    = {r["experiment_accession"] for r in records if r["experiment_accession"]}

    # ── PATCH 1: split semicolon-joined sample accessions ───────────────────
    print(f"\n[E] BioSample enrichment: {len(uniq_samples)} raw sample fields")
    expanded_samples = set()
    for acc in uniq_samples:
        for a in acc.split(";"):
            a = a.strip()
            if a:
                expanded_samples.add(a)
    print(f"  ({len(expanded_samples)} unique after expanding multi-accession fields)")

    sdata = {}
    for i, acc in enumerate(sorted(expanded_samples), 1):
        if i % 1000 == 0:
            print(f"    {i}/{len(expanded_samples)}")
        sdata[acc] = parse_sample(_xml("", "sample", acc))
    print("    done.")

    # ── PATCH 2: split semicolon-joined experiment accessions ───────────────
    print(f"\n[F] Experiment enrichment: {len(uniq_exps)} raw experiment fields")
    expanded_exps = set()
    for acc in uniq_exps:
        for a in acc.split(";"):
            a = a.strip()
            if a:
                expanded_exps.add(a)
    print(f"  ({len(expanded_exps)} unique after expanding multi-accession fields)")

    edata = {}
    for i, acc in enumerate(sorted(expanded_exps), 1):
        if i % 1000 == 0:
            print(f"    {i}/{len(expanded_exps)}")
        edata[acc] = parse_experiment(_xml("", "experiment", acc))
    print("    done.")

    print(f"\n[G] Study enrichment: {len(all_bps)} BioProjects")
    studata = {}
    all_pmids = set()
    for i,acc in enumerate(sorted(all_bps),1):
        if i % 200 == 0: print(f"    {i}/{len(all_bps)}")
        parsed = parse_study(_xml("","study",acc))
        studata[acc] = parsed
        if parsed.get("linked_pmids"):
            for p in parsed["linked_pmids"].split(","):
                p = p.strip()
                if p: all_pmids.add(p)
    print(f"    done. {len(all_pmids)} linked PMIDs.")

    print(f"\n[H] PubMed enrichment: {len(all_pmids)} articles")
    pubdata = {}
    for i,pmid in enumerate(sorted(all_pmids),1):
        if i % 100 == 0: print(f"    {i}/{len(all_pmids)}")
        def _pm(pmid=pmid):
            try:
                h = Entrez.efetch(db="pubmed",id=pmid,rettype="xml",retmode="xml")
                raw = h.read(); h.close(); time.sleep(0.35)
                return raw.decode("utf-8",errors="replace") if isinstance(raw,bytes) else raw
            except: return ""
        pubdata[pmid] = parse_pubmed(safe_cache("pubmed", pmid, _pm))
    print("    done.")

    # Merge enrichment into records — handle multi-accession by merging first hit
    for r in records:
        s_acc = r.get("sample_accession","")
        if s_acc:
            # use first sub-accession if multi
            first = s_acc.split(";")[0].strip()
            if first in sdata:
                for k,v in sdata[first].items():
                    r[f"sample_{k}"] = v
        e_acc = r.get("experiment_accession","")
        if e_acc:
            first = e_acc.split(";")[0].strip()
            if first in edata:
                for k,v in edata[first].items():
                    r[f"exp_{k}"] = v

    return records, sdata, edata, studata, pubdata


# ── OUTPUT ────────────────────────────────────────────────────────────────────
def write(records, sdata, edata, studata, pubdata, all_bps, t0):
    print("\n[Writing outputs]")
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_RUNS, index=False)
    print(f"  {OUTPUT_RUNS}  ({len(df)} rows × {len(df.columns)} cols)")

    def tsv(rows, path):
        if rows: pd.DataFrame(rows).to_csv(path, sep="\t", index=False); print(f"  {path}  ({len(rows)} rows)")

    tsv([{"sample_accession":k,**v} for k,v in sdata.items()], OUTPUT_SAMPLES)
    tsv([{"experiment_accession":k,**v} for k,v in edata.items()], OUTPUT_EXPERIMENTS)
    tsv([{"study_accession":k,**v} for k,v in studata.items()], OUTPUT_STUDIES)
    tsv([{"pmid_key":k,**v} for k,v in pubdata.items() if v], OUTPUT_PUBLICATIONS)

    with open(OUTPUT_BPS,"w") as f:
        f.write("\n".join(sorted(all_bps))+"\n")
    print(f"  {OUTPUT_BPS}  ({len(all_bps)} BioProjects)")

    urls = []
    for r in records:
        ftp = s(r.get("fastq_ftp",""))
        for u in ftp.split(";"):
            u=u.strip()
            if u:
                if u.startswith("ftp."): u="https://"+u
                elif u.startswith("ftp://"): u="https://"+u[6:]
                urls.append(u)
    with open(OUTPUT_FASTQ,"w") as f: f.write("\n".join(urls)+"\n")
    print(f"  {OUTPUT_FASTQ}  ({len(urls)} FASTQ URLs)")

    srr = sorted({r["run_accession"] for r in records if r["run_accession"]})
    with open(OUTPUT_SRR,"w") as f: f.write("\n".join(srr)+"\n")
    print(f"  {OUTPUT_SRR}  ({len(srr)} accessions)")

    by_c = defaultdict(lambda:{"n_runs":0,"bioprojects":set(),"cats":defaultdict(int)})
    for r in records:
        c = r.get("country_clean","Unknown")
        by_c[c]["n_runs"] += 1
        if s(r.get("bioproject","")).startswith("PRJ"): by_c[c]["bioprojects"].add(r["bioproject"])
        by_c[c]["cats"][r.get("broad_category","Other")] += 1

    summary = {
        "generated": datetime.now().isoformat(),
        "runtime_seconds": round(time.time()-t0,1),
        "israel_excluded": True,
        "total_runs": len(records),
        "total_samples": len(sdata),
        "total_bioprojects": len(all_bps),
        "total_publications": len([p for p in pubdata.values() if p]),
        "per_country": {
            c: {"n_runs":v["n_runs"],"n_bioprojects":len(v["bioprojects"]),"categories":dict(v["cats"])}
            for c,v in sorted(by_c.items())
        },
        "broad_categories": df["broad_category"].value_counts().to_dict() if len(df) else {},
    }
    with open(OUTPUT_JSON,"w") as f: json.dump(summary,f,indent=2)
    print(f"  {OUTPUT_JSON}")

    lines = [
        "MicrobiomeAtlas-MENA — DISCOVERY + ENRICHMENT REPORT (v5.2)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "="*65,
        f"Total runs:         {len(records)}",
        f"Total samples:      {len(sdata)}",
        f"Total BioProjects:  {len(all_bps)}",
        f"Publications linked:{len([p for p in pubdata.values() if p])}",
        f"Israel excluded:    YES",
        "",
        "PER-COUNTRY","-"*65,
    ]
    for c,v in sorted(by_c.items(),key=lambda x:-x[1]["n_runs"]):
        cats = ", ".join(f"{k}:{n}" for k,n in list(v["cats"].items())[:4])
        lines.append(f"  {c:<22} {v['n_runs']:>7} runs  {len(v['bioprojects']):>4} BPs  [{cats}]")
    with open(OUTPUT_REPORT,"w") as f: f.write("\n".join(lines))
    print(f"  {OUTPUT_REPORT}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run API connectivity test and exit")
    parser.add_argument("--discovery-only", action="store_true",
                        help="Run discovery phase only, skip enrichment")
    args = parser.parse_args()

    if args.test:
        test_ena()
        sys.exit(0)

    t0 = time.time()
    print("="*65)
    print("MicrobiomeAtlas-MENA — DISCOVERY + ENRICHMENT v5.2 (patched)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Patches: multi-accession splitting + filename length cap")
    print("="*65)

    unified, all_bps = discovery()

    if args.discovery_only:
        print("\n[discovery-only mode] skipping enrichment")
        records = [r for r in unified.values()
                   if r["country_clean"] not in ("__EXCLUDED__","")]
        for r in records:
            b,sp = categorize(r)
            r["broad_category"]=b; r["specific_category"]=sp
        pd.DataFrame(records).to_csv(OUTPUT_RUNS,index=False)
        with open(OUTPUT_BPS,"w") as f: f.write("\n".join(sorted(all_bps))+"\n")
        srr = sorted({r["run_accession"] for r in records if r["run_accession"]})
        with open(OUTPUT_SRR,"w") as f: f.write("\n".join(srr)+"\n")
        print(f"  {len(records)} runs saved to {OUTPUT_RUNS}")
        print(f"  {len(all_bps)} BioProjects saved to {OUTPUT_BPS}")
        print(f"  {len(srr)} SRR accessions saved to {OUTPUT_SRR}")
        return

    records, sdata, edata, studata, pubdata = enrichment(unified, all_bps)
    write(records, sdata, edata, studata, pubdata, all_bps, t0)

    print(f"\n{'='*65}")
    print(f"DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  {len(records)} enriched runs")
    print(f"  {len(sdata)} samples with full BioSample attributes")
    print(f"  {len([p for p in pubdata.values() if p])} linked publications")
    print("="*65)


if __name__ == "__main__":
    main()
