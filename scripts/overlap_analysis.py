#!/usr/bin/env python3
"""Duplicate and source-overlap analysis for the classification cohort.

Reproduces every value in Table 3b of the manuscript from the public dataset.

Usage:
    python overlap_analysis.py --root /path/to/dataset
        (root contains Training/<class>/ and Testing/<class>/)

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
    a = ap.parse_args()

    files = walk(a.root)
    if not files:
        sys.exit('no images found under %s' % a.root)
    print('images: %d' % len(files))

    fh, ph, dh = [], [], []
    for n, (_, _, p) in enumerate(files, 1):
        f, q, d = hashes(p)
        fh.append(f); ph.append(q); dh.append(d)
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

    # expected contamination under a random image-level stratified 80/20 split
    memb = {}
    for gi, v in enumerate(pix):
        for i in v:
            memb[i] = gi
    rng = np.random.default_rng(a.seed)
    exp = []
    for _ in range(a.resamples):
        test = np.zeros(len(files), bool)
        for c in sorted(set(cls)):
            idx = np.where(cls == c)[0]
            rng.shuffle(idx)
            test[idx[:int(round(len(idx) * 0.2))]] = True
        trg = {memb[i] for i in np.where(~test)[0] if i in memb}
        exp.append(sum(1 for i in np.where(test)[0] if i in memb and memb[i] in trg))
    exp = np.array(exp)
    n_test_sim = int(round(len(files) * 0.2))

    r = {
        'n_images': len(files), 'n_train': len(tr), 'n_test': len(te),
        'pixel_duplicate_clusters': len(pix),
        'images_in_a_duplicate_cluster': involved,
        'clusters_with_conflicting_labels': conflict,
        'test_byte_identical_to_train': len(cf),
        'test_pixel_identical_to_train': len(cp),
        'test_dhash0_match_in_train': len(near),
        'expected_pixel_identical_random_8020': {
            'mean': float(exp.mean()), 'min': int(exp.min()), 'max': int(exp.max()),
            'n_test': n_test_sim, 'resamples': a.resamples},
        'cross_class_false_match_pct': fp,
    }
    json.dump(r, open(a.out, 'w'), indent=1)

    pct = lambda k: 100.0 * k / len(te)
    print('\nTable 3b')
    print('  exact-pixel duplicate clusters, pooled cohort            %d' % len(pix))
    print('  images belonging to a duplicate cluster                  %d' % involved)
    print('  duplicate clusters with conflicting class labels         %d' % conflict)
    print('  test byte-identical to a training image                  %d (%.2f %%)' % (len(cf), pct(len(cf))))
    print('  test pixel-identical to a training image                 %d (%.2f %%)' % (len(cp), pct(len(cp))))
    print('  test with zero-distance perceptual match in training     %d (%.2f %%)' % (len(near), pct(len(near))))
    print('  expected pixel-identical under random 80/20              %.0f (%.2f %%)' % (
        exp.mean(), 100.0 * exp.mean() / n_test_sim))
    print('  cross-class false-match rate at distance 0 / 2 / 4       %.1f %% / %.1f %% / %.1f %%' % (
        fp[0], fp[2], fp[4]))
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
