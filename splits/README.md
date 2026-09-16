# Split files

One case id (BraTS folder name) or relative file path (FigShare `.mat`) per line.
Generate them reproducibly with `scripts/make_splits.py` (seed 42):

```bash
python scripts/make_splits.py --root <BraTS2023-GLI training dir> --n-test 250 --prefix splits/brats2023
python scripts/make_splits.py --ids splits/brats2023_test.txt --n-test 150 --prefix splits/brats2023_report --only-test
python scripts/make_splits.py --root <BraTS2024 post-treatment dir> --n-test 54 --prefix splits/brats2024
```

For BraTS 2024 the paper follows the 217 / 54 split of Vox-MMSD; if you have
that exact list, save it here as `brats2024_train.txt` / `brats2024_test.txt`
instead of generating a random one.
