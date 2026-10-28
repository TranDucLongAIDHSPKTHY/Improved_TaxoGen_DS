"""Statistics shared by the human evaluation and the recommendation experiments."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def _two_sided_p(boot: np.ndarray) -> float:
    b = len(boot)
    lower = (np.sum(boot <= 0) + 1) / (b + 1)
    upper = (np.sum(boot >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05,
                     chunk: int = 500) -> dict:
    """Bootstrap over paired units (e.g. test users) for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        boot[start:start + m] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(diff.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": _two_sided_p(boot),
            "n": n, "n_boot": n_boot}


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def fleiss_kappa(counts) -> float:
    """counts: (n_items, n_categories), the same number of raters per item."""
    counts = np.asarray(counts, dtype=np.float64)
    n_raters = counts.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa needs the same number of raters per item")
    n = n_raters[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = (np.square(counts).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), np.square(p_j).sum()
    if p_e == 1:
        return 1.0
    return float((p_bar - p_e) / (1 - p_e))


def kappa_label(kappa: float) -> str:
    """Landis & Koch (1977) bands; fixed before any human judgment was collected."""
    if not np.isfinite(kappa):
        return "undefined"
    if kappa < 0:
        return "poor"
    for upper, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")):
        if kappa <= upper:
            return name
    return "almost perfect"


def alpha_label(alpha: float) -> str:
    """Krippendorff (2004): >= 0.800 reliable; >= 0.667 only tentative
    conclusions; below that, low reliability."""
    if not np.isfinite(alpha):
        return "undefined"
    return "reliable" if alpha >= 0.800 else "tentative" if alpha >= 0.667 else "low"


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test; ties are left out by the caller."""
    from scipy.stats import binomtest
    return float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0


def krippendorff_alpha_nominal(data) -> float:
    """data: (n_raters, n_items) array of category codes, np.nan for missing."""
    data = np.asarray(data, dtype=np.float64)
    values = np.unique(data[~np.isnan(data)])
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for unit in data.T:
        present = unit[~np.isnan(unit)]
        m = len(present)
        if m < 2:
            continue
        counts = np.zeros(len(values))
        for v in present:
            counts[index[v]] += 1
        coincidence += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    disagree_obs = coincidence.sum() - np.trace(coincidence)
    disagree_exp = n_c.sum() ** 2 - np.square(n_c).sum()
    if disagree_exp == 0:
        return 1.0
    return float(1 - (n - 1) * disagree_obs / disagree_exp)


def two_way_bootstrap_diff(mat_a, mat_b, n_boot: int = 5000, seed: int = 0,
                           alpha: float = 0.05) -> dict:
    """Difference in pooled accuracy between two methods judged by the same
    annotators on disjoint item sets. mat_x: (items, annotators) with 1/0 and
    np.nan for missing. Items (within each method) and annotators (shared)
    are resampled independently, which respects both sources of dependence;
    treating every vote as independent would be pseudoreplication."""
    mat_a = np.asarray(mat_a, dtype=np.float64)
    mat_b = np.asarray(mat_b, dtype=np.float64)
    if mat_a.shape[1] != mat_b.shape[1]:
        raise ValueError("both methods must share the annotator columns")
    rng = np.random.default_rng(seed)
    n_ann = mat_a.shape[1]
    boot = np.empty(n_boot)
    for t in range(n_boot):
        ann = rng.integers(0, n_ann, n_ann)
        ia = rng.integers(0, mat_a.shape[0], mat_a.shape[0])
        ib = rng.integers(0, mat_b.shape[0], mat_b.shape[0])
        boot[t] = np.nanmean(mat_a[np.ix_(ia, ann)]) - np.nanmean(mat_b[np.ix_(ib, ann)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"acc_a": float(np.nanmean(mat_a)), "acc_b": float(np.nanmean(mat_b)),
            "diff": float(np.nanmean(mat_a) - np.nanmean(mat_b)),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": _two_sided_p(boot),
            "n_items_a": mat_a.shape[0], "n_items_b": mat_b.shape[0], "n_annotators": n_ann}
