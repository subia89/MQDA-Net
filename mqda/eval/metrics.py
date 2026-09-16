"""Evaluation metrics (paper Sec. 4).

Segmentation: Dice, IoU, HD95, ASSD on the BraTS regions ET / TC / WT.
Classification: accuracy and per-class precision / recall / F1.
Report generation: BLEU, ROUGE-1 / ROUGE-L, METEOR, BERTScore, RadGraph-F1.
The text metrics use optional packages and are skipped when unavailable.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# BraTS evaluation regions from the label convention 1 NCR, 2 ED, 3 ET
REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}


def region_mask(labels: np.ndarray, region: str) -> np.ndarray:
    return np.isin(labels, REGIONS[region])


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    ps, gs = p.sum(), g.sum()
    if ps + gs == 0:
        return 1.0, 1.0
    dice = 2 * inter / (ps + gs)
    iou = inter / (np.logical_or(p, g).sum())
    return float(dice), float(iou)


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return mask ^ ndimage.binary_erosion(mask, iterations=1, border_value=0)


def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing=None):
    from scipy import ndimage

    p, g = pred.astype(bool), gt.astype(bool)
    spacing = spacing or [1.0] * p.ndim
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return dt_g[sp], dt_p[sg]


def hd95_assd(pred, gt, spacing=None, empty_value=373.13):
    """Returns (HD95, ASSD). If exactly one mask is empty, returns
    ``empty_value`` (BraTS convention: the image diagonal) for both."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0, 0.0
    if not p.any() or not g.any():
        return empty_value, empty_value
    d_pg, d_gp = surface_distances(p, g, spacing)
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assd = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
    return float(hd95), float(assd)


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, spacing=None, distances=True):
    out = {}
    for r in REGIONS:
        pm, gm = region_mask(pred, r), region_mask(gt, r)
        d, j = dice_iou(pm, gm)
        out[f"dice_{r}"], out[f"iou_{r}"] = d, j
        if distances:
            h, a = hd95_assd(pm, gm, spacing)
            out[f"hd95_{r}"], out[f"assd_{r}"] = h, a
    for k in ("dice", "iou", "hd95", "assd"):
        vals = [out[f"{k}_{r}"] for r in REGIONS if f"{k}_{r}" in out]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
    return out


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = y_true >= 0
    y_true, y_pred = y_true[keep], y_pred[keep]
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                    average="macro", zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(pm),
           "recall_macro": float(rm), "f1_macro": float(fm),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    for i, name in enumerate(class_names):
        out[f"precision_{name}"] = float(p[i])
        out[f"recall_{name}"] = float(r[i])
        out[f"f1_{name}"] = float(f[i])
        out[f"support_{name}"] = int(s[i])
    return out


def _tok(text: str):
    return text.lower().replace("\n", " ").split()


def text_metrics(predictions: Sequence[str], references: Sequence[str],
                 use_bertscore=True, use_radgraph=True, per_sample=False):
    out, per = {}, {}
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

        sm = SmoothingFunction().method1
        out["bleu"] = corpus_bleu([[_tok(r)] for r in references],
                                  [_tok(p) for p in predictions], smoothing_function=sm)
        per["bleu"] = [sentence_bleu([_tok(r)], _tok(p), smoothing_function=sm)
                       for p, r in zip(predictions, references)]
    except ImportError:
        log.warning("nltk not installed - BLEU skipped")
    try:
        from rouge_score import rouge_scorer

        sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = [sc.score(r, p) for p, r in zip(predictions, references)]
        per["rouge1_f1"] = [s["rouge1"].fmeasure for s in scores]
        per["rougeL_f1"] = [s["rougeL"].fmeasure for s in scores]
        out["rouge1_f1"] = float(np.mean(per["rouge1_f1"]))
        out["rougeL_f1"] = float(np.mean(per["rougeL_f1"]))
    except ImportError:
        log.warning("rouge-score not installed - ROUGE skipped")
    try:
        from nltk.translate.meteor_score import meteor_score

        per["meteor"] = [meteor_score([_tok(r)], _tok(p)) for p, r in zip(predictions, references)]
        out["meteor"] = float(np.mean(per["meteor"]))
    except (ImportError, LookupError) as e:
        log.warning("METEOR skipped (%s); run nltk.download('wordnet')", type(e).__name__)
    if use_bertscore:
        try:
            from bert_score import score as bscore

            _, _, f1 = bscore(list(predictions), list(references), lang="en", verbose=False)
            per["bertscore_f1"] = f1.tolist()
            out["bertscore_f1"] = float(f1.mean())
        except ImportError:
            log.warning("bert-score not installed - BERTScore skipped")
    if use_radgraph:
        try:
            from radgraph import F1RadGraph

            f1r = F1RadGraph(reward_level="partial")
            mean, reward_list, _, _ = f1r(hyps=list(predictions), refs=list(references))
            out["radgraph_f1"] = float(mean)
            per["radgraph_f1"] = list(reward_list)
        except ImportError:
            log.warning("radgraph not installed - RadGraph-F1 skipped")
    if per_sample:
        out["per_sample"] = per
    return out
