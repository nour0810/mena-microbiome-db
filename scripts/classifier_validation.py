"""
MENA Microbiome — CLASSIFIER VALIDATION
=======================================
Reproduces the validation of the metagenomics-versus-single-organism classifier
and the 73-rule BioSample-type harmonization reported in Methods 2.3 and
Results 3.1.

Two things are made reproducible here:

  1. `draw_stratified_sample()` regenerates the exact evaluation set from the
     released corpus, given the same seed. Anyone can therefore obtain the same
     604 runs that were manually reviewed.

  2. `evaluate()` computes every reported statistic — precision, recall, F1,
     specificity, accuracy, Cohen's kappa and Wilson 95% confidence intervals —
     from a confusion matrix, and `verify_published()` asserts that the matrix
     reported in the manuscript yields exactly the published values.

The reference labels themselves were assigned by manual review of the full
BioSample record (scientific_name, host, isolation_source, environment fields,
study title and abstract) rather than by any automated rule, and so cannot be
regenerated programmatically. They are human judgements, and a single annotator
produced them; no inter-annotator agreement estimate is therefore available.
Supplying a second independent annotation is the outstanding item noted in the
Limitations. To evaluate a new annotation, pass the labelled table to
`evaluate_from_table()`.

Usage
-----
    python scripts/classifier_validation.py                  # verify published values
    python scripts/classifier_validation.py --sample         # regenerate the n=604 set
    python scripts/classifier_validation.py --labels FILE    # score a labelled table
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd

DATA = "./out/data/mena_metagenomics_clean.tsv"
SEED = 42
N_SAMPLE = 604

# Confusion matrix as reported in Methods 2.3 / Results 3.1.
PUBLISHED = {"tp": 306, "fp": 4, "tn": 284, "fn": 10}

# Values as they appear in the manuscript, with the number of decimal places at
# which each is reported. The comparison below is made at that precision, so a
# value quoted to 2 dp is not counted as a mismatch against a 3 dp computation.
PUBLISHED_STATS = {
    "precision": (0.987, 3),
    "recall": (0.968, 3),
    "f1": (0.978, 3),
    "specificity": (0.986, 3),
    "accuracy": (0.977, 3),
    "kappa": (0.95, 2),
}


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because several of the
    proportions are close to 1 and the counts behind them are small, where the
    Wald interval misbehaves (Brown, Cai & DasGupta 2001).
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(tp, fp, tn, fn):
    """Full metric set for a binary confusion matrix, with Wilson intervals."""
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else float("nan"))

    # Cohen's kappa from the marginals.
    po = accuracy
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n**2
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")

    return {
        "n": n,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "precision_ci": wilson_ci(tp, tp + fp),
        "recall": recall,
        "recall_ci": wilson_ci(tp, tp + fn),
        "specificity": specificity,
        "specificity_ci": wilson_ci(tn, tn + fp),
        "accuracy": accuracy,
        "accuracy_ci": wilson_ci(tp + tn, n),
        "f1": f1,
        "kappa": kappa,
    }


def evaluate_from_table(path, pred_col="predicted_metagenomic",
                        ref_col="reference_metagenomic"):
    """Score a labelled table. Columns must be boolean or 0/1."""
    df = pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".txt")) else ",")
    for col in (pred_col, ref_col):
        if col not in df.columns:
            sys.exit(f"error: column '{col}' not found in {path}")
    pred = df[pred_col].astype(bool).values
    ref = df[ref_col].astype(bool).values
    return evaluate(tp=int((pred & ref).sum()), fp=int((pred & ~ref).sum()),
                    tn=int((~pred & ~ref).sum()), fn=int((~pred & ref).sum()))


def draw_stratified_sample(df, n=N_SAMPLE, seed=SEED):
    """Draw the evaluation set, stratified by broad_category.

    Allocation is proportional to category size, with at least one run per
    category so that small categories (Viral, Fungal) are represented at all.
    Deterministic for a given seed and corpus.
    """
    rng = np.random.RandomState(seed)
    strata = df["broad_category"].fillna("Unclassified")
    counts = strata.value_counts()
    alloc = (counts / counts.sum() * n).round().astype(int).clip(lower=1)

    # Correct rounding drift so the allocation sums to exactly n.
    while alloc.sum() != n:
        step = 1 if alloc.sum() < n else -1
        target = alloc.idxmax() if step == -1 else alloc.idxmin()
        if step == -1 and alloc[target] <= 1:
            break
        alloc[target] += step

    picks = []
    for category, k in alloc.items():
        pool = df.index[strata == category]
        k = min(k, len(pool))
        picks.extend(rng.choice(pool, size=k, replace=False))
    return df.loc[sorted(picks)]


def _fmt(label, value, ci=None, width=14):
    line = f"  {label:<{width}} {value:6.3f}"
    if ci is not None:
        line += f"   95% CI {ci[0]:.3f}-{ci[1]:.3f}"
    return line


def verify_published():
    """Check that the published confusion matrix yields the published values."""
    res = evaluate(**PUBLISHED)
    print("Classifier validation — metagenomics vs single-organism")
    print(f"  n = {res['n']}   confusion: "
          f"TP={PUBLISHED['tp']} FP={PUBLISHED['fp']} "
          f"TN={PUBLISHED['tn']} FN={PUBLISHED['fn']}\n")
    print(_fmt("precision", res["precision"], res["precision_ci"]))
    print(_fmt("recall", res["recall"], res["recall_ci"]))
    print(_fmt("specificity", res["specificity"], res["specificity_ci"]))
    print(_fmt("accuracy", res["accuracy"], res["accuracy_ci"]))
    print(_fmt("F1", res["f1"]))
    print(_fmt("Cohen's kappa", res["kappa"]))

    print("\nAgreement with the values reported in the manuscript:")
    ok = True
    for key, (published, dp) in PUBLISHED_STATS.items():
        got = res[key]
        match = round(got, dp) == published
        ok &= match
        print(f"  {key:<14} computed {got:.3f}   manuscript {published:.{dp}f}   "
              f"{'ok' if match else 'MISMATCH'}")
    if not ok:
        sys.exit("error: computed statistics do not match the published values")
    print("\nAll reported statistics reproduce from the confusion matrix.")
    print("Note: kappa derives from a single annotator; no inter-annotator")
    print("agreement estimate is available (see Limitations).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true",
                    help="regenerate the stratified evaluation set and write it to TSV")
    ap.add_argument("--labels", metavar="FILE",
                    help="score a labelled table instead of the published matrix")
    ap.add_argument("--data", default=DATA, help=f"corpus TSV (default: {DATA})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--out", default="classifier_validation_sample.tsv")
    args = ap.parse_args()

    if args.labels:
        res = evaluate_from_table(args.labels)
        print(f"n = {res['n']}   confusion: {res['confusion']}\n")
        for key in ("precision", "recall", "specificity", "accuracy"):
            print(_fmt(key, res[key], res[f"{key}_ci"]))
        print(_fmt("F1", res["f1"]))
        print(_fmt("Cohen's kappa", res["kappa"]))
        return

    if args.sample:
        df = pd.read_csv(args.data, sep="\t", low_memory=False)
        sample = draw_stratified_sample(df, n=args.n, seed=args.seed)
        cols = [c for c in ("run_accession", "bioproject", "scientific_name",
                            "library_source", "library_strategy", "host",
                            "isolation_source", "broad_category", "study_title")
                if c in sample.columns]
        sample[cols].to_csv(args.out, sep="\t", index=False)
        print(f"wrote {len(sample)} runs to {args.out}")
        print("Add a boolean 'reference_metagenomic' column by manual review, "
              "then score with --labels.")
        return

    verify_published()


if __name__ == "__main__":
    main()
