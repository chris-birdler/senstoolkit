from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def correlation_vector(X, y, method="spearman"):
    df = pd.DataFrame(X)
    return df.corrwith(pd.Series(y), method=method).values


def bootstrap_correlation_ci(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 0,
    ci_method: str = "percentile",  # robust; BCa can be problematic with degenerate data
    eps: float = 1e-12,
    degenerate_policy: str = "zero",  # "zero" -> CI=[0,0]; "nan" -> keep NaN
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap CIs for each column of X against y (paired resamples).
    Degenerate resamples (std=0) are set to 0.0 in the stat() callback
    to avoid NaNs. Fully degenerate columns are not bootstrapped at all.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    # BCa fallback check
    if ci_method == "BCa":
        try:
            _ = stats.bootstrap((np.array([0.0]), np.array([0.0])),
                                 lambda a, b: 0.0, paired=True, n_resamples=10, method="BCa")
        except Exception:
            ci_method = "percentile"

    # valid rows only
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv, yv = X[mask], y[mask]
    n, p = Xv.shape

    lo = np.full(p, np.nan)
    hi = np.full(p, np.nan)

    # stat function with degeneracy guard (per resample)
    if method.lower().startswith("p"):
        def stat(xx, yy):
            if np.std(xx) <= eps or np.std(yy) <= eps:
                return 0.0
            return stats.pearsonr(xx, yy)[0]
    elif method.lower().startswith("s"):
        def stat(xx, yy):
            if np.std(xx) <= eps or np.std(yy) <= eps:
                return 0.0
            return stats.spearmanr(xx, yy).correlation
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    for j in range(p):
        xj = Xv[:, j]
        # fully degenerate column? -> skip bootstrap, set directly
        if xj.size < 3 or np.std(xj) <= eps or np.std(yv) <= eps:
            if degenerate_policy == "zero":
                lo[j] = 0.0; hi[j] = 0.0
            else:
                lo[j] = np.nan; hi[j] = np.nan
            continue

        res = stats.bootstrap(
            data=(xj, yv),
            statistic=stat,
            paired=True,
            vectorized=False,
            n_resamples=n_boot,
            confidence_level=1.0 - alpha,
            method=ci_method,
            random_state=rng,
        )
        lo[j] = res.confidence_interval.low
        hi[j] = res.confidence_interval.high

    return lo, hi


def build_corr_groups(X, names, threshold=0.9):
    """Group highly correlated features using hierarchical clustering."""
    C = np.corrcoef(X, rowvar=False)

    # Convert correlation to distance: d = 1 - |corr|
    dist = 1.0 - np.abs(C)
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry and non-negative values for numerical stability
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)

    Z = linkage(squareform(dist), method='complete')
    labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    # Build groups: only keep clusters with >= 2 members
    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    groups = [sorted(members) for members in cluster_map.values() if len(members) >= 2]
    group_names = [[names[i] for i in g] for g in groups]
    return groups, group_names, C
