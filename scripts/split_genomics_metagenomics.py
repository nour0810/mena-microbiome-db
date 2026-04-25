"""
split_genomics_metagenomics.py
==============================
Splits mena_all_runs.csv into two clean TSV files:

  1. mena_metagenomics.tsv   — microbiome / community / environmental metagenomics
  2. mena_single_organism.tsv — pure single-organism genomics / transcriptomics
  3. mena_ambiguous.tsv      — rows that couldn't be cleanly assigned (for review)

Classification logic (in priority order)
-----------------------------------------
STEP 1 — library_source signal (most reliable field):
  METAGENOMIC / METATRANSCRIPTOMIC              → metagenomics
  GENOMIC / GENOMIC SINGLE CELL / TRANSCRIPTOMIC
  / TRANSCRIPTOMIC SINGLE CELL                  → single_organism
  VIRAL RNA                                     → single_organism
                                                  (virus isolates, not viromic communities)
  SYNTHETIC                                     → single_organism
  OTHER                                         → needs further checks (STEP 2)

STEP 2 — scientific_name contains "metagenome" / "microbiome" / "virome" etc.
  (catches AMPLICON+GENOMIC rows that are really community surveys)
  Any name containing these keywords             → metagenomics
  Otherwise                                      → single_organism

STEP 3 — remaining OTHER sources that survived STEP 2
  library_strategy == AMPLICON                   → metagenomics
                                                  (16S/ITS community profiling)
  else                                           → ambiguous (write to review file)

Output also adds a new column `data_type` with values:
  metagenomics | single_organism | ambiguous
and `data_subtype` with finer detail:
  shotgun_metagenomics | amplicon_metagenomics | metatranscriptomics |
  wgs_genomics | amplicon_genomics | rna_seq | targeted_genomics |
  viral_genomics | other_genomics | ambiguous
"""

import csv
import sys
import os

csv.field_size_limit(10_000_000)

INPUT_FILE = "mena_all_runs.csv"
OUT_META   = "mena_metagenomics.tsv"
OUT_SINGLE = "mena_single_organism.tsv"
OUT_AMBIG  = "mena_ambiguous.tsv"

# Keywords in scientific_name that indicate a community / environmental sample
METAGENOME_KEYWORDS = {
    "metagenome", "microbiome", "virome", "metatranscriptome",
    "microbiota", "community", "microorganism", "microbial mat",
    "biofilm", "consortium"
}

# library_source → primary bucket
GENOMIC_SOURCES  = {"GENOMIC", "GENOMIC SINGLE CELL", "TRANSCRIPTOMIC",
                    "TRANSCRIPTOMIC SINGLE CELL", "VIRAL RNA", "SYNTHETIC"}
META_SOURCES     = {"METAGENOMIC", "METATRANSCRIPTOMIC"}


def get_subtype(strategy: str, source: str, bucket: str) -> str:
    s, src = strategy.upper(), source.upper()
    if bucket == "metagenomics":
        if src == "METATRANSCRIPTOMIC":
            return "metatranscriptomics"
        if s == "AMPLICON":
            return "amplicon_metagenomics"
        if s in ("WGS", "WGA", "WCS", "SYNTHETIC-LONG-READ"):
            return "shotgun_metagenomics"
        if s in ("RAD-SEQ",):
            return "amplicon_metagenomics"
        if s == "RNA-SEQ":
            return "metatranscriptomics"
        return "shotgun_metagenomics"  # default for metagenomics
    else:  # single_organism or ambiguous
        if s == "AMPLICON":
            return "amplicon_genomics"
        if s in ("WGS", "WGA", "WCS", "SYNTHETIC-LONG-READ", "TN-SEQ"):
            return "wgs_genomics"
        if s in ("RNA-SEQ", "MIRNA-SEQ", "FAIRE-SEQ"):
            return "rna_seq"
        if s in ("TARGETED-CAPTURE", "WXS"):
            return "targeted_genomics"
        if src == "VIRAL RNA":
            return "viral_genomics"
        return "other_genomics"


def classify(row: dict) -> tuple[str, str]:
    """Returns (data_type, data_subtype)."""
    source   = row.get("library_source", "").strip().upper()
    strategy = row.get("library_strategy", "").strip().upper()
    sci_name = row.get("scientific_name", "").strip().lower()

    # STEP 1 — source-based
    if source in META_SOURCES:
        bucket = "metagenomics"
        return bucket, get_subtype(strategy, source, bucket)

    if source in GENOMIC_SOURCES:
        # STEP 2 — scientific_name override: catch community samples mislabeled as GENOMIC
        if any(kw in sci_name for kw in METAGENOME_KEYWORDS):
            bucket = "metagenomics"
            return bucket, get_subtype(strategy, source, bucket)
        bucket = "single_organism"
        return bucket, get_subtype(strategy, source, bucket)

    # source == OTHER (or empty)
    # STEP 2 — scientific_name check
    if any(kw in sci_name for kw in METAGENOME_KEYWORDS):
        bucket = "metagenomics"
        return bucket, get_subtype(strategy, source, bucket)

    # STEP 3 — strategy fallback for OTHER source
    if strategy == "AMPLICON":
        bucket = "metagenomics"
        return bucket, "amplicon_metagenomics"

    # STEP 4 — WGS/WGA + OTHER source + named organism (not a metagenome keyword)
    # e.g. Mycobacterium tuberculosis, Lactobacillus sp., wheat species, E. coli
    if strategy in ("WGS", "WGA", "WCS", "SYNTHETIC-LONG-READ", "TN-SEQ",
                    "TARGETED-CAPTURE", "WXS"):
        bucket = "single_organism"
        return bucket, get_subtype(strategy, source, bucket)

    if strategy in ("RNA-SEQ", "MIRNA-SEQ", "FAIRE-SEQ"):
        bucket = "single_organism"
        return bucket, "rna_seq"

    return "ambiguous", "ambiguous"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_path    = os.path.join(script_dir, INPUT_FILE)

    if not os.path.exists(in_path):
        sys.exit(f"ERROR: Cannot find {in_path}\n"
                 f"Place {INPUT_FILE} in the same folder as this script.")

    counts = {"metagenomics": 0, "single_organism": 0, "ambiguous": 0}

    with open(in_path, newline="") as fh:
        reader   = csv.DictReader(fh)
        fieldnames = reader.fieldnames + ["data_type", "data_subtype"]

        with (
            open(os.path.join(script_dir, OUT_META),   "w", newline="") as fh_meta,
            open(os.path.join(script_dir, OUT_SINGLE),  "w", newline="") as fh_single,
            open(os.path.join(script_dir, OUT_AMBIG),   "w", newline="") as fh_ambig,
        ):
            writers = {
                "metagenomics":   csv.DictWriter(fh_meta,   fieldnames=fieldnames, delimiter="\t"),
                "single_organism": csv.DictWriter(fh_single, fieldnames=fieldnames, delimiter="\t"),
                "ambiguous":      csv.DictWriter(fh_ambig,  fieldnames=fieldnames, delimiter="\t"),
            }
            for w in writers.values():
                w.writeheader()

            for row in reader:
                bucket, subtype = classify(row)
                row["data_type"]    = bucket
                row["data_subtype"] = subtype
                writers[bucket].writerow(row)
                counts[bucket] += 1

    total = sum(counts.values())
    print("=" * 55)
    print("  Split complete")
    print("=" * 55)
    print(f"  {'metagenomics':<20} {counts['metagenomics']:>7,}  →  {OUT_META}")
    print(f"  {'single_organism':<20} {counts['single_organism']:>7,}  →  {OUT_SINGLE}")
    print(f"  {'ambiguous':<20} {counts['ambiguous']:>7,}  →  {OUT_AMBIG}")
    print(f"  {'TOTAL':<20} {total:>7,}")
    print("=" * 55)
    print(f"\nOutput files written to: {script_dir}")


if __name__ == "__main__":
    main()
