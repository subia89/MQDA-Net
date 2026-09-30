#!/usr/bin/env python3
"""Duplicate and source-overlap analysis for the classification cohort.

Reproduces every value in Table 3b of the manuscript from the public dataset.
Table 3b is stated on the manuscript's own split protocol (image-level stratified
80/20 of the 7,023 pooled images: 5,618 / 1,405); because the overlap depends on
the particular random draw, the table reports expectations over 200 draws.  The
partition as distributed with the dataset (5,712 / 1,311) is printed for reference.

Usage:
    python overlap_analysis.py --root /path/to/dataset
        (root contains Training/<class>/ and Testing/<class>/)

The expected-contamination row uses the same split rule as scripts/make_splits.py
--stratify and as the manuscript: n_test = ceil(0.2 N) = 1,405 for the 7,023-image
cohort, allocated per class by largest remainder (324 / 329 / 352 / 400).  Pass
--test-list <file> to report, in addition, the overlap on the test partition that
was actually used.

Detection levels
    file      SHA-256 over raw file bytes
    pixel     SHA-256 over the 256x256 grayscale array (catches re-encoding)
    dhash     64-bit difference hash, reported only at Hamming distance 0;
              looser thresholds are validated and rejected (see --validate)
"""
import argparse, collections, hashlib, json, os, sys
import numpy as np
from PIL import Image

POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def walk(root):
    out = []
    for split in ('Training', 'Testing'):
        sd = os.path.join(root, split)
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                out.append((split, cls, os.path.join(cd, f)))
    return out


def hashes(path):
    raw = open(path, 'rb').read()
    im = Image.open(path).convert('L')
    big = np.asarray(im.resize((256, 256), Image.BILINEAR), dtype=np.uint8)
    a9 = np.asarray(im.resize((9, 8), Image.BILINEAR), dtype=np.float32)
    d = a9[:, 1:] > a9[:, :-1]
    v = 0
    for b in d.flatten():
        v = (v << 1) | int(b)
    return (hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(big.tobytes()).hexdigest(),
            np.uint64(v))


def ham(a, B):
    return POP[np.bitwise_xor(B, a).view(np.uint8).reshape(-1, 8)].sum(1)


def clusters(values):
    g = collections.defaultdict(list)
    for i, v in enumerate(values):
        g[v].append(i)
    return [v for v in g.values() if len(v) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--resamples', type=int, default=200,
                    help='random stratified 80/20 draws used for the expected-contamination row')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='overlap_manifest.json')
    ap.add_argument('--cache', default=None,
                    help='JSONL file of per-image hashes; written incrementally and reused on re-runs')
    ap.add_argument('--test-list', default=None,
                    help='file with one relative image path per line: the test partition actually '
                         'used. If given, the overlap on that partition is reported in addition to '
                         'the expected value under random draws')
    a = ap.parse_args()

    files = walk(a.root)
    if not files:
        sys.exit('no images found under %s' % a.root)
    print('images: %d' % len(files))

    cache = {}
    if a.cache and os.path.exists(a.cache):
        for line in open(a.cache):
            rec = json.loads(line)
            cache[rec['path']] = (rec['file'], rec['pixel'], int(rec['dhash']))
    cf_out = open(a.cache, 'a') if a.cache else None
    fh, ph, dh = [], [], []
    for n, (_, _, p) in enumerate(files, 1):
        rel = os.path.relpath(p, a.root)
        if rel in cache:
            f, q, d = cache[rel]
        else:
            f, q, d = hashes(p)
            if cf_out:
                cf_out.write(json.dumps({'path': rel, 'file': f, 'pixel': q, 'dhash': int(d)}) + '\n')
                cf_out.flush()
        fh.append(f); ph.append(q); dh.append(np.uint64(d))
        if n % 1000 == 0:
            print('  hashed %d/%d' % (n, len(files)))
    dh = np.array(dh, dtype=np.uint64)
    split = np.array([s for s, _, _ in files])
    cls = np.array([c for _, c, _ in files])
    tr = np.where(split == 'Training')[0]
    te = np.where(split == 'Testing')[0]

    pix = clusters(ph)
    conflict = sum(1 for v in pix if len(set(cls[i] for i in v)) > 1)
    involved = sum(len(v) for v in pix)

    def cross(values):
        g = collections.defaultdict(list)
        for i, v in enumerate(values):
            g[v].append(i)
        hit = set()
        for v in g.values():
            if len(v) > 1 and len({split[i] for i in v}) > 1:
                hit |= {i for i in v if split[i] == 'Testing'}
        return sorted(hit)

    cf, cp = cross(fh), cross(ph)
    near = [i for i in te if (ham(dh[i], dh[tr]) == 0).any()]

    # threshold validation: a match to a different class is a false positive
    fp = {}
    for thr in (0, 2, 4):
        tot = wrong = 0
        for i in te:
            m = tr[ham(dh[i], dh[tr]) <= thr]
            if len(m) == 0:
                continue
            tot += 1
            if cls[i] not in set(cls[m]) or len(set(cls[m])) > 1:
                wrong += 1
        fp[thr] = 100.0 * wrong / max(tot, 1)

    # expected overlap under a random image-level stratified 80/20 split (the split rule of the
    # manuscript and of scripts/make_splits.py --stratify): n_test = ceil(0.2 N), per-class test
    # counts by largest remainder.  Three levels: byte-identical file, pixel-identical array,
    # zero-distance perceptual hash.  Each is "a test image whose key also occurs in training".
    import math
    n_test_sim = int(math.ceil(len(files) * 0.2))
    names = sorted(set(cls))
    exact = {c: n_test_sim * int((cls == c).sum()) / len(files) for c in names}
    alloc = {c: int(math.floor(exact[c])) for c in names}
    for c in sorted(names, key=lambda c: exact[c] - alloc[c], reverse=True)[: n_test_sim - sum(alloc.values())]:
        alloc[c] += 1

    keys = {'byte': fh, 'pixel': ph, 'dhash0': [int(v) for v in dh]}

    def overlap_count(test_mask, key):
        train_keys = {key[i] for i in np.where(~test_mask)[0]}
        return int(sum(1 for i in np.where(test_mask)[0] if key[i] in train_keys))

    rng = np.random.default_rng(a.seed)
    sim = {k: [] for k in keys}
    for _ in range(a.resamples):
        test = np.zeros(len(files), bool)
        for c in names:
            idx = np.where(cls == c)[0]
            rng.shuffle(idx)
            test[idx[:alloc[c]]] = True
        for k, key in keys.items():
            sim[k].append(overlap_count(test, key))
    sim = {k: np.array(v) for k, v in sim.items()}
    exp = sim['pixel']

    # cross-class false-match rate of the perceptual hash over the pooled cohort (partition-free):
    # among images that have another image within the threshold, the share whose matches include
    # a different class
    fp_pooled = {}
    for thr in (0, 2, 4):
        tot = wrong = 0
        for i in range(len(files)):
            d = ham(dh[i], dh)
            d[i] = 255
            m = np.where(d <= thr)[0]
            if len(m) == 0:
                continue
            tot += 1
            if any(cls[j] != cls[i] for j in m):
                wrong += 1
        fp_pooled[thr] = 100.0 * wrong / max(tot, 1)

    actual = None
    if a.test_list:
        wanted = {l.strip().replace('\\', '/') for l in open(a.test_list) if l.strip()}
        rel = [os.path.relpath(p, a.root).replace('\\', '/') for _, _, p in files]
        test = np.array([r in wanted for r in rel])
        missing = wanted - set(rel)
        if missing:
            sys.exit('%d entries of %s were not found under %s' % (len(missing), a.test_list, a.root))
        actual = {'n_test': int(test.sum()),
                  'byte_identical_to_train': overlap_count(test, keys['byte']),
                  'pixel_identical_to_train': overlap_count(test, keys['pixel']),
                  'dhash0_match_in_train': overlap_count(test, keys['dhash0'])}

    r = {
        'n_images': len(files), 'n_train': len(tr), 'n_test': len(te),
        'pixel_duplicate_clusters': len(pix),
        'images_in_a_duplicate_cluster': involved,
        'clusters_with_conflicting_labels': conflict,
        'test_byte_identical_to_train': len(cf),
        'test_pixel_identical_to_train': len(cp),
        'test_dhash0_match_in_train': len(near),
        'expected_under_manuscript_split': {
            'n_test': n_test_sim, 'per_class_test': alloc, 'resamples': a.resamples,
            **{k: {'mean': float(v.mean()), 'min': int(v.min()), 'max': int(v.max())}
               for k, v in sim.items()}},
        'actual_split': actual,
        'cross_class_false_match_pct_distributed_partition': fp,
        'cross_class_false_match_pct_pooled': fp_pooled,
    }
    json.dump(r, open(a.out, 'w'), indent=1)

    pct = lambda k: 100.0 * k / len(te)
    print('\nTable 3b (manuscript split: expected values, n = %d, per class %s)' % (
        n_test_sim, ', '.join('%s %d' % kv for kv in sorted(alloc.items()))))
    print('  exact-pixel duplicate clusters, pooled cohort            %d' % len(pix))
    print('  images belonging to a duplicate cluster                  %d' % involved)
    print('  duplicate clusters with conflicting class labels         %d' % conflict)
    for k, label in (('byte', 'byte-identical to a training image'),
                     ('pixel', 'pixel-identical to a training image'),
                     ('dhash0', 'zero-distance perceptual-hash match in training')):
        v = sim[k]
        print('  expected test images %-46s %.0f (%.2f %%)  [mean of %d draws, range %d-%d]' % (
            label, v.mean(), 100.0 * round(v.mean()) / n_test_sim, a.resamples, v.min(), v.max()))
    print('  cross-class false-match rate, pooled cohort, d = 0 / 2 / 4   %.1f %% / %.1f %% / %.1f %%' % (
        fp_pooled[0], fp_pooled[2], fp_pooled[4]))
    if actual:
        print('  ACTUAL on the supplied test list (n=%d): byte %d, pixel %d, dhash0 %d' % (
            actual['n_test'], actual['byte_identical_to_train'], actual['pixel_identical_to_train'],
            actual['dhash0_match_in_train']))
    print('\nFor reference: partition as distributed (%d / %d)' % (len(tr), len(te)))
    print('  test byte-identical / pixel-identical / dhash0 in training   %d (%.2f %%) / %d (%.2f %%) / %d (%.2f %%)' % (
        len(cf), pct(len(cf)), len(cp), pct(len(cp)), len(near), pct(len(near))))
    print('  cross-class false-match rate, test vs train, d = 0 / 2 / 4   %.1f %% / %.1f %% / %.1f %%' % (
        fp[0], fp[2], fp[4]))
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
