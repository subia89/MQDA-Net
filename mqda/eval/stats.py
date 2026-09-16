"""Statistical testing (paper Sec. 5.1).

Two-tailed Wilcoxon signed-rank test on paired per-sample metrics, BCa
bootstrap 95 % confidence intervals (10,000 resamples), matched-pairs
rank-biserial correlation and Bonferroni correction.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((pos - neg) / ranks.sum())


def paired_test(ours, baseline, n_resamples=10_000, confidence=0.95, seed=0):
    ours, baseline = np.asarray(ours, float), np.asarray(baseline, float)
    diff = ours - baseline
    res = {"delta_mean": float(diff.mean()), "n": int(len(diff))}
    if np.allclose(diff, 0):
        res.update(p_value=1.0, ci_low=0.0, ci_high=0.0, rank_biserial=0.0)
        return res
    w = stats.wilcoxon(ours, baseline, alternative="two-sided")
    bs = stats.bootstrap((diff,), np.mean, n_resamples=n_resamples, confidence_level=confidence,
                         method="BCa", random_state=np.random.default_rng(seed))
    res.update(p_value=float(w.pvalue), statistic=float(w.statistic),
               ci_low=float(bs.confidence_interval.low), ci_high=float(bs.confidence_interval.high),
               rank_biserial=rank_biserial(diff))
    return res


def bonferroni(p_values: dict, alpha=0.05):
    m = len(p_values)
    thr = alpha / max(m, 1)
    return {k: {"p": p, "significant": p < thr, "alpha_corrected": thr} for k, p in p_values.items()}
