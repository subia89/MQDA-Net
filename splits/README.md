# Split files

One case id (BraTS folder name) or relative file path (FigShare `.mat`) per line.
Generate them reproducibly with `scripts/make_splits.py` (seed 42):

```bash
python scripts/make_splits.py --root <BraTS2023-GLI training dir> --n-test 250 --prefix splits/brats2023
python scripts/make_splits.py --ids splits/brats2023_test.txt --n-test 150 --prefix splits/brats2023_report --only-test
python scripts/make_splits.py --root <BraTS2024 post-treatment dir> --n-test 54 --prefix splits/brats2024
```

For the classification benchmark, the manuscript reports an image-level
stratified 80/20 split of all 7,023 images of the Nickparvar composite
(5,618 train / 1,405 test), not the Training / Testing folders that ship with
the dataset (5,712 / 1,311):

```bash
python scripts/make_splits.py --stratified-folder data/nickparvar \
    --test-fraction 0.2 --prefix splits/nickparvar
```

`configs/nickparvar.yaml` reads `splits/nickparvar_train.txt` and
`splits/nickparvar_test.txt`. These two files are a reference split generated
by that command with seed 42: the test size is ceil(0.2 N) = 1,405, allocated
per class by largest remainder (324 glioma, 329 meningioma, 352 pituitary,
400 no tumor), the sizes of the split reported in the manuscript. The lists
used for the reported results were not retained, so image membership is not
necessarily the same.

For BraTS 2024 the paper follows the 217 / 54 split of Vox-MMSD; if you have
that exact list, save it here as `brats2024_train.txt` / `brats2024_test.txt`
instead of generating a random one.
