#!/usr/bin/env python
"""Write reproducible train / test split files.

BraTS 2023:  python scripts/make_splits.py --root <BraTS dir> --n-test 250 --prefix splits/brats2023
BraTS 2024:  python scripts/make_splits.py --root <dir> --n-test 54 --prefix splits/brats2024
Report set:  python scripts/make_splits.py --ids splits/brats2023_test.txt --n-test 150 \
                 --prefix splits/brats2023_report --only-test
FigShare:    python scripts/make_splits.py --root data/figshare --glob "*.mat" --test-fraction 0.2 \
                 --prefix splits/figshare

Image-level stratified split of an image-folder dataset (the 5,618 / 1,405 split
of the Nickparvar composite reported in the manuscript):

    python scripts/make_splits.py --stratified-folder data/nickparvar \
        --test-fraction 0.2 --prefix splits/nickparvar

It pools the dataset's Training and Testing folders (or a flat <class>/ layout),
draws a per-class 20 % test sample and writes paths relative to the root.
"""
import argparse
import glob
import os
import random


EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def stratified_folder_split(args):
    """Image-level stratified split that preserves the class proportions."""
    root = args.stratified_folder
    per_class = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not path.lower().endswith(EXTS):
            continue
        cls = os.path.basename(os.path.dirname(path))
        per_class.setdefault(cls, []).append(os.path.relpath(path, root))
    if not per_class:
        raise SystemExit(f"no images found under {root}")
    rng = random.Random(args.seed)
    train, test = [], []
    frac = args.test_fraction if args.test_fraction is not None else 0.2
    for cls, files in sorted(per_class.items()):
        files = sorted(files)
        rng.shuffle(files)
        n_test = int(round(len(files) * frac))
        test += files[:n_test]
        train += files[n_test:]
        print(f"{cls:<14}{len(files):>6}  -> train {len(files) - n_test}, test {n_test}")
    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    with open(f"{args.prefix}_train.txt", "w") as f:
        f.write("\n".join(sorted(train)) + "\n")
    with open(f"{args.prefix}_test.txt", "w") as f:
        f.write("\n".join(sorted(test)) + "\n")
    print(f"total train {len(train)}  test {len(test)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--ids", help="read candidate ids from a file instead of --root")
    ap.add_argument("--glob", default=None, help="split files matching a pattern instead of case folders")
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--test-fraction", type=float)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--only-test", action="store_true")
    ap.add_argument("--stratified-folder", help="root of an image-folder dataset to split per class")
    args = ap.parse_args()
    if args.stratified_folder:
        return stratified_folder_split(args)
    if args.ids:
        ids = [l.strip() for l in open(args.ids) if l.strip()]
    elif args.glob:
        ids = sorted(os.path.relpath(p, args.root)
                     for p in glob.glob(os.path.join(args.root, "**", args.glob), recursive=True))
    else:
        ids = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    random.Random(args.seed).shuffle(ids)
    n_test = args.n_test if args.n_test is not None else int(round(len(ids) * args.test_fraction))
    test, train = sorted(ids[:n_test]), sorted(ids[n_test:])
    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    with open(f"{args.prefix}_test.txt", "w") as f:
        f.write("\n".join(test) + "\n")
    if not args.only_test:
        with open(f"{args.prefix}_train.txt", "w") as f:
            f.write("\n".join(train) + "\n")
    print(f"train {0 if args.only_test else len(train)}  test {len(test)}")


if __name__ == "__main__":
    main()
