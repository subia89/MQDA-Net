#!/usr/bin/env python
"""Paired significance testing between two methods (paper Sec. 5.1).

python scripts/significance.py --ours results/mqda/per_case.csv \
    --baseline results/baseline/per_case.csv --metrics dice_mean iou_mean hd95_mean

Both CSVs need a ``case_id`` column; rows are matched on it. For accuracy,
pass ``--metrics correct`` after adding a 0/1 ``correct`` column, or use the
``cls_true``/``cls_pred`` columns written by evaluate.py (handled automatically).
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqda.eval.stats import bonferroni, paired_test  # noqa: E402


def read(path):
    with open(path) as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f)}
    for r in rows.values():
        if "cls_true" in r and "cls_pred" in r and "correct" not in r:
            r["correct"] = float(r["cls_true"] == r["cls_pred"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    a, b = read(args.ours), read(args.baseline)
    ids = sorted(set(a) & set(b))
    results = {}
    for m in args.metrics:
        pairs = [(float(a[i][m]), float(b[i][m])) for i in ids
                 if a[i].get(m) not in (None, "") and b[i].get(m) not in (None, "")]
        results[m] = paired_test([p[0] for p in pairs], [p[1] for p in pairs], args.resamples)
    corr = bonferroni({m: r["p_value"] for m, r in results.items()}, args.alpha)
    for m in results:
        results[m]["bonferroni_significant"] = corr[m]["significant"]
    print(f"{'metric':<16}{'n':>6}{'delta':>10}{'95% CI':>22}{'p':>10}{'r_rb':>8}  Bonf.")
    for m, r in results.items():
        ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"{m:<16}{r['n']:>6}{r['delta_mean']:>10.4f}{ci:>22}{r['p_value']:>10.4g}"
              f"{r['rank_biserial']:>8.2f}  {'yes' if r['bonferroni_significant'] else 'no'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
