import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Matplotlib for plots
import matplotlib.pyplot as plt

# scikit-learn for modeling and permutation importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# Try XGBoost; fall back if unavailable
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

# Try SHAP
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# Try SALib for Sobol analysis
try:
    from SALib.sample import saltelli as salib_saltelli
    from SALib.analyze import sobol as salib_sobol
    SALIB_OK = True
except ImportError:
    SALIB_OK = False

from typing import Tuple


def ensure_outdir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_doe_csv(path, names, X, response_cols=None):
    if response_cols is None:
        response_cols = ["permeability", "loss"]
    ensure_outdir(path)
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "id", range(len(df)))
    for col in response_cols:
        df[col] = ""
    df.to_csv(path, index=False)


def parse_params_json(path):
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    dims = []
    names = []
    scales = []
    for name, spec in params.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Parameter '{name}' must be an object with keys min,max,scale.")
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Parameter '{name}' must have 'min' and 'max'.")
        vmin = float(spec["min"])
        vmax = float(spec["max"])
        if vmax <= vmin:
            raise ValueError(f"Parameter '{name}': max must be > min.")
        scale = spec.get("scale", "linear").lower()
        if scale not in ("linear", "log"):
            raise ValueError(f"Parameter '{name}': scale must be 'linear' or 'log'.")
        if scale == "log" and vmin <= 0:
            raise ValueError(f"Parameter '{name}': log scale requires min > 0.")
        dims.append((vmin, vmax))
        names.append(name)
        scales.append(scale)
    return names, dims, scales


def sobol_sample(dims, n, seed=None, prefer_power_of_two=False):
    d = len(dims)
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    if prefer_power_of_two:
        m = int(math.ceil(math.log2(max(2, n))))
        u = engine.random_base2(m=m)
        u = u[:n, :]
    else:
        u = engine.random(n=n)
    arr = np.empty_like(u)
    for j, (vmin, vmax) in enumerate(dims):
        arr[:, j] = vmin + (vmax - vmin) * u[:, j]
    return arr


def apply_scaling(arr, dims, scales):
    n, d = arr.shape
    u = np.empty_like(arr)
    for j, (vmin, vmax) in enumerate(dims):
        u[:, j] = (arr[:, j] - vmin) / (vmax - vmin)
    out = np.empty_like(arr)
    for j, ((vmin, vmax), scale) in enumerate(zip(dims, scales)):
        if scale == "log":
            logmin = math.log(vmin)
            logmax = math.log(vmax)
            out[:, j] = np.exp(logmin + u[:, j] * (logmax - logmin))
        else:
            out[:, j] = vmin + u[:, j] * (vmax - vmin)
    return out


def correlation_vector(X, y, method="spearman"):
    df = pd.DataFrame(X)
    return df.corrwith(pd.Series(y), method=method).values


def plot_bar_sorted(names, values, title, out_path, xlabel="Value", invert=False):
    idx = np.argsort(-np.abs(values))
    names_sorted = [names[i] for i in idx]
    values_sorted = [values[i] for i in idx]

    plt.figure()
    y_pos = np.arange(len(names_sorted))
    plt.barh(y_pos, values_sorted)
    plt.yticks(y_pos, names_sorted)
    plt.xlabel(xlabel)
    plt.title(title)
    if invert:
        plt.gca().invert_yaxis()
    ensure_outdir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def fit_model(X, y, random_state=0):
    if XGB_OK:
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_depth=None, max_iter=400, learning_rate=0.05, random_state=random_state
        )
    model.fit(X, y)
    return model


def cv_r2(model, X, y, splits=5, seed=0):
    n = X.shape[0]
    k = min(splits, max(2, n // 5))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=None)
    return scores


def model_importance(model, X, y, n_repeats=10, seed=0):
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, scoring="r2")
    importances = result.importances_mean
    return importances


# ----- Advanced utilities -----

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
    n = C.shape[0]

    # Convert correlation to distance: d = 1 - |corr|
    dist = 1.0 - np.abs(C)
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry and non-negative values for numerical stability
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)

    Z = linkage(squareform(dist), method='complete')
    labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    # Build groups: only keep clusters with >= 2 members
    from collections import defaultdict
    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    groups = [sorted(members) for members in cluster_map.values() if len(members) >= 2]
    group_names = [[names[i] for i in g] for g in groups]
    return groups, group_names, C


def make_model_ctor(random_state):
    def ctor():
        if XGB_OK:
            return XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                objective="reg:squarederror", random_state=random_state, n_jobs=1
            )
        else:
            return HistGradientBoostingRegressor(max_depth=None, max_iter=400,
                                                  learning_rate=0.05, random_state=random_state)
    return ctor


def cv_permutation_importance(model_ctor, X, y, n_repeats=20, cv_splits=5, seed=0,
                               n_jobs=None):
    """
    KFold-CV permutation importance: fit per fold, then compute
    sklearn.inspection.permutation_importance on the validation set.
    Aggregates mean/std of R² drops across folds.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    drops = []

    base_scores = []
    for tr, va in kf.split(X):
        est = model_ctor()
        est.fit(X[tr], y[tr])
        yhat = est.predict(X[va])
        base = r2_score(y[va], yhat)
        base_scores.append(base)
        res = permutation_importance(est, X[va], y[va],
                                     scoring="r2", n_repeats=n_repeats,
                                     random_state=seed, n_jobs=n_jobs)
        # In sklearn, res.importances_mean is the delta score (R2_after_perm -
        # R2_base). We negate it to get the R² drop.
        drop = -res.importances_mean
        drops.append(drop)

    drops = np.vstack(drops)
    means = drops.mean(axis=0)
    stds = drops.std(axis=0)
    base_r2_mean = float(np.mean(base_scores))
    base_r2_std = float(np.std(base_scores))

    return means, stds, base_r2_mean, base_r2_std


def grouped_permutation_importance(model_ctor, X, y, groups, n_repeats=20, cv_splits
                                    =5, seed=0):
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=min(cv_splits, max(2, len(y)//5)), shuffle=True,
               random_state=seed)
    g_means = []
    for train_idx, val_idx in kf.split(X):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        m = model_ctor()
        m.fit(Xtr, ytr)
        base = r2_score(yval, m.predict(Xval))
        g_drops = np.zeros((n_repeats, len(groups)))
        for gi, gcols in enumerate(groups):
            for r in range(n_repeats):
                Xperm = Xval.copy()
                idx = rng.permutation(Xval.shape[0])
                Xperm[:, gcols] = Xperm[idx][:, gcols]
                s = r2_score(yval, m.predict(Xperm))
                g_drops[r, gi] = base - s
        g_means.append(np.mean(g_drops, axis=0))
    return np.mean(np.stack(g_means, axis=0), axis=0)


def partial_dependence_plots(model, X, names, top_idx, out_dir, label):
    for j in top_idx:
        fname = names[j]
        plt.figure()
        try:
            PartialDependenceDisplay.from_estimator(model, X, [j], kind="both",
                                                     grid_resolution=25)
        except Exception:
            xj = X[:, j]
            grid = np.linspace(np.nanmin(xj), np.nanmax(xj), 25)
            preds = []
            for v in grid:
                Xtmp = X.copy()
                Xtmp[:, j] = v
                preds.append(model.predict(Xtmp))
            plt.plot(grid, np.mean(np.stack(preds, axis=0), axis=1))
            plt.xlabel(fname)
            plt.ylabel("Partial dependence")
        plt.title(f"PDP/ICE — {label} — {fname}")
        ensure_outdir(os.path.join(out_dir, "pdp"))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pdp", f"pdp_{label}_{fname}.png"), dpi=150)
        plt.close()


def shap_global_and_dependence(sigma_y, model, X, names, out_dir, label, top_k=10):
    if not (SHAP_OK and XGB_OK):
        return None
    try:
        explainer = shap.TreeExplainer(model)
        X_sample = X if X.shape[0] <= 5000 else X[np.random.RandomState(0).choice(X.shape[0], size=5000, replace=False)]
        sv = explainer.shap_values(X_sample)
        sv = np.array(sv)
        if sv.ndim == 3:
            sv = np.sum(np.abs(sv), axis=0)
        mean_abs = np.mean(np.abs(sv), axis=0)
        plot_bar_sorted(names, mean_abs, f"SHAP mean |value| — {label}", os.path.join(out_dir, f"shap_bar_{label}.png"), xlabel="mean |SHAP|", invert=True)
        idx = np.argsort(-mean_abs)[:top_k]
        for j in idx:
            plt.figure()
            plt.scatter(X_sample[:, j], sv[:, j], s=8)
            plt.axhline(0.5*sigma_y, color="red", linestyle="--")
            plt.axhline(-0.5*sigma_y, color="red", linestyle="--")
            plt.xlabel(names[j]); plt.ylabel("SHAP value")
            plt.title(f"SHAP dependence — {label} — {names[j]}")
            plt.tight_layout()
            ensure_outdir(os.path.join(out_dir, "shap"))
            plt.savefig(os.path.join(out_dir, "shap", f"shap_dep_{label}_{names[j]}.png"), dpi=150)
            plt.close()
        return mean_abs
    except Exception:
        return None


def _sobol_on_surrogate_salib(model, names, dims, scales, N=8192, seed=0):
    """Sobol analysis using SALib library."""
    d = len(names)

    # For log-scale parameters, SALib samples uniformly; we transform after
    bounds = []
    for (vmin, vmax), sc in zip(dims, scales):
        if sc == "log":
            bounds.append([np.log(vmin), np.log(vmax)])
        else:
            bounds.append([vmin, vmax])

    problem = {
        'num_vars': d,
        'names': list(names),
        'bounds': bounds,
    }

    X_samples = salib_saltelli.sample(problem, N, seed=seed)

    # Transform log-scale columns back to physical space
    for j, ((vmin, vmax), sc) in enumerate(zip(dims, scales)):
        if sc == "log":
            X_samples[:, j] = np.exp(X_samples[:, j])

    Y = np.asarray(model.predict(X_samples), dtype=np.float64).reshape(-1)
    Si = salib_sobol.analyze(problem, Y, print_to_console=False)

    S1 = np.array(Si['S1'])
    ST = np.array(Si['ST'])
    V = float(np.var(Y, ddof=1))

    # Diagnostics
    s1_sum = float(np.nansum(S1))
    if (ST + 1e-12 < S1).any() or s1_sum > 1.05:
        print(f"[WARN] Sobol diagnostics: sum(S1)={s1_sum:.3f}, "
              f"S1 range [{np.nanmin(S1):.3f},{np.nanmax(S1):.3f}], at least one ST<S1. "
              f"Check N, bounds/scales, CV-R².")

    print("[Sobol] S1 vs ST:")
    for j in range(d):
        print(f"  {names[j]}: S1={S1[j]:.3f} | ST={ST[j]:.3f}")

    return S1, ST, V


def _sobol_on_surrogate_manual(model, names, dims, scales, N=8192, seed=0):
    """
    Sobol analysis on the surrogate model (manual implementation, fallback when SALib unavailable):
    First-Order S1: Saltelli (2002/2010)  S1_i = E[f(B) * (f(A_B^i) - f(A))] / Var(Y)
    Total ST     : Jansen (1999)          ST_i = E[(f(A) - f(A_B^i))^2] / (2 * Var(Y))
    """
    d = len(names)

    # 1) Two independently scrambled Sobol sequences
    m = int(math.ceil(math.log2(max(2, N))))
    engA = qmc.Sobol(d=d, scramble=True, seed=seed)
    engB = qmc.Sobol(d=d, scramble=True, seed=(seed or 0) + 12345)
    A_u = engA.random_base2(m=m)[:N, :]
    B_u = engB.random_base2(m=m)[:N, :]

    # 2) Map to physical ranges (linear / log-uniform)
    def unit_to_range(U):
        X = np.empty_like(U, dtype=np.float64)
        for j, ((vmin, vmax), sc) in enumerate(zip(dims, scales)):
            if str(sc).lower() == "log":
                lo, hi = np.log(vmin), np.log(vmax)
                X[:, j] = np.exp(lo + U[:, j] * (hi - lo))
            else:
                X[:, j] = vmin + U[:, j] * (vmax - vmin)
        return X

    A = unit_to_range(A_u)
    B = unit_to_range(B_u)

    # 3) Evaluate surrogate model
    fA = np.asarray(model.predict(A), dtype=np.float64).reshape(-1)
    fB = np.asarray(model.predict(B), dtype=np.float64).reshape(-1)

    # Variance over combined A and B (more robust)
    V = float(np.var(np.r_[fA, fB], ddof=1))
    if not np.isfinite(V) or V <= 0:
        return np.zeros(d), np.zeros(d), (V if np.isfinite(V) else 0.0)

    S1 = np.zeros(d, dtype=np.float64)
    ST = np.zeros(d, dtype=np.float64)

    # 4) For each feature i: build AB_i and compute estimators
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]                  # A_B^i
        fAB = np.asarray(model.predict(AB), dtype=np.float64).reshape(-1)

        # Saltelli First-Order
        S1[i] = np.mean(fB * (fAB - fA)) / V

        # Jansen Total-Order
        ST[i] = np.mean((fA - fAB)**2) / (2.0 * V)

    # 5) Plausibility checks
    s1_sum = float(np.nansum(S1))
    if (ST + 1e-12 < S1).any() or s1_sum > 1.05:
        print(f"[WARN] Sobol diagnostics: sum(S1)={s1_sum:.3f}, "
              f"S1 range [{np.nanmin(S1):.3f},{np.nanmax(S1):.3f}], at least one ST<S1. "
              f"Check N, bounds/scales, CV-R².")

    # Saltelli-2010 Cross-Covariance diagnostic: S1 ~ Cov(fA, fABj)/V
    S1_cov = np.zeros_like(S1)
    for j in range(d):
        ABj = A.copy(); ABj[:, j] = B[:, j]
        fABj = np.asarray(model.predict(ABj), dtype=np.float64).reshape(-1)
        S1_cov[j] = (np.mean((fA - fA.mean()) * (fABj - fABj.mean()))) / V
    print("[Sobol] S1 vs ST:")
    for j in range(d):
        print(f"  {names[j]}: S1={S1[j]:.3f} | ST={ST[j]:.3f}")

    return S1, ST, V


def sobol_on_surrogate(model, names, dims, scales, N=8192, seed=0):
    """Sobol analysis: uses SALib if available, falls back to manual implementation."""
    if SALIB_OK:
        return _sobol_on_surrogate_salib(model, names, dims, scales, N=N, seed=seed)
    else:
        return _sobol_on_surrogate_manual(model, names, dims, scales, N=N, seed=seed)


def xgb_gain_importance(model, names):
    """
    Returns the 'gain' importance per feature as np.array.
    Only for XGBoost models; returns None otherwise.
    """
    if isinstance(model, XGBRegressor):
        try:
            booster = model.get_booster()
            gain_map = booster.get_score(importance_type='gain')
        except Exception:
            arr = getattr(model, "feature_importances_", None)
            return np.array(arr, dtype=float) if arr is not None else None

        arr = np.zeros(len(names), dtype=float)
        for j in range(len(names)):
            arr[j] = float(gain_map.get(f"f{j}", 0.0))
        return arr
    else:
        return None


def build_bounds_scales_by_name(param_cols_csv, params_obj):
    """
    Returns (param_cols, dims, scales) in CSV column order,
    restricted to names present in the parameter file. Raises if mismatched.
    params_obj: dict{name:{min,max,scale}} or list[{"name":...,"min":...,"max":...,"scale":...}]
    """
    if isinstance(params_obj, list):
        mapping = {it["name"]: it for it in params_obj}
    elif isinstance(params_obj, dict):
        mapping = params_obj
    else:
        raise ValueError("Parameter JSON must be a dict or list of objects.")

    unexpected = [n for n in param_cols_csv if n not in mapping]
    if unexpected:
        raise ValueError(f"CSV contains columns not in parameter file "
                         f"(not accepted as features): {unexpected}")

    missing = [n for n in mapping.keys() if n not in param_cols_csv]
    if missing:
        raise ValueError(f"Parameter file contains parameters missing from CSV: "
                         f"{missing}")

    # Assemble in CSV column order
    dims, scales, param_cols = [], [], []
    for name in param_cols_csv:
        v = mapping[name]
        vmin, vmax = float(v["min"]), float(v["max"])
        sc = str(v.get("scale", "linear")).lower()
        if sc == "log" and vmin <= 0:
            raise ValueError(f"'{name}': log scale requires min > 0 (min={vmin}).")
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
            raise ValueError(f"Invalid bounds for '{name}': min={vmin}, max={vmax}")
        dims.append((vmin, vmax))
        scales.append(sc)
        param_cols.append(name)
    return param_cols, dims, scales


def design(params_json, n_samples, out_csv, seed=None, prefer_power_of_two=False):
    names, dims, scales = parse_params_json(params_json)
    p = len(names)
    if n_samples is None or n_samples <= 0:
        n_samples = max(10 * p, 50)
    raw = sobol_sample(dims, n_samples, seed=seed, prefer_power_of_two=
                       prefer_power_of_two)
    X = apply_scaling(raw, dims, scales)
    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"DOE_{ts}.csv"
    write_doe_csv(out_csv, names, X)
    sidecar = os.path.splitext(out_csv)[0] + ".params.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({n: {"min": mn, "max": mx, "scale": sc} for n, (mn, mx), sc in zip(
            names, dims, scales)}, f, indent=2)
    print(f"Design written to: {out_csv}")
    print(f"Wrote sidecar parameter spec: {sidecar}")
    print("Fill the response columns with your simulation results and run 'analyze'.")


def analyze(design_csv, response_cols=None, out_dir="outputs",
            seed=0,
            cv_folds=5, perm_repeats=20, bootstrap_corr=1000, group_corr_threshold=
            0.9,
            top_k_pdp=6, do_pdp=True, do_shap=True, do_group_perm=True,
            do_sobol=True, sobol_samples=2048):
    if response_cols is None:
        response_cols = ["permeability", "loss"]
    df_input = pd.read_csv(design_csv)

    # Exclude 'id' and all response columns from features
    exclude = {"id"} | set(response_cols)
    raw_param_cols = [h for h in df_input.columns if h not in exclude]
    if not raw_param_cols:
        raise ValueError("No parameter columns found.")

    # Warn about response columns missing from the CSV
    present_responses = [c for c in response_cols if c in df_input.columns]
    missing_responses = [c for c in response_cols if c not in df_input.columns]
    if missing_responses:
        print(f"[WARN] Response columns not found in CSV (skipped): {missing_responses}")
    if not present_responses:
        raise ValueError(f"None of the response columns {response_cols} found in CSV.")

    sidecar_json = os.path.splitext(design_csv)[0] + ".params.json"
    param_cols = raw_param_cols
    dims = []
    scales = []
    if os.path.exists(sidecar_json):
        with open(sidecar_json, "r", encoding="utf-8") as f:
            pj = json.load(f)
        if pj is not None:
            param_cols, dims, scales = build_bounds_scales_by_name(raw_param_cols, pj)
    else:
        print("[INFO] No sidecar parameter file found; Sobol analysis will be skipped.")

    print("[INFO] Using features (in order):", param_cols)
    X = df_input[param_cols].apply(pd.to_numeric, errors='coerce').values

    ensure_outdir(os.path.join(out_dir, "_"))
    results_summary = []

    for label in present_responses:
        y = pd.to_numeric(df_input[label], errors='coerce').values
        mask = np.isfinite(y)
        if np.sum(mask) < 5:
            n_filled = int(np.sum(mask))
            print(f"[WARN] Response '{label}': not enough filled rows ({n_filled}). Skipping.")
            continue
        Xy = X[mask, :]
        yy = y[mask]
        sigma_y = np.nanstd(yy)

        # Correlations + Bootstrap CIs
        spearman = correlation_vector(Xy, yy, method="spearman")
        pearson = correlation_vector(Xy, yy, method="pearson")
        sp_lo, sp_hi = bootstrap_correlation_ci(Xy, yy, method="spearman", n_boot=
                                                 bootstrap_corr, alpha=0.05, random_state=seed, ci_method="BCa",
                                                 degenerate_policy="zero")
        pe_lo, pe_hi = bootstrap_correlation_ci(Xy, yy, method="pearson", n_boot=
                                                 bootstrap_corr, alpha=0.05, random_state=seed, ci_method="BCa",
                                                 degenerate_policy="zero")

        pd.DataFrame({
            "name": param_cols, "spearman": spearman, "sp_ci_lo": sp_lo, "sp_ci_hi": sp_hi,
            "pearson": pearson, "pe_ci_lo": pe_lo, "pe_ci_hi": pe_hi,
        }).to_csv(os.path.join(out_dir, f"correlations_{label}.csv"), index=False)

        plot_bar_sorted(param_cols, spearman, f"Spearman correlations — {label}",
                        os.path.join(out_dir, f"corr_spearman_{label}.png"), xlabel="Spearman p", invert=True)
        plot_bar_sorted(param_cols, pearson, f"Pearson correlations — {label}",
                        os.path.join(out_dir, f"corr_pearson_{label}.png"), xlabel="Pearson r", invert=True)

        # Model & CV
        model = fit_model(Xy, yy, random_state=seed)
        cv_scores = cv_r2(model, Xy, yy, splits=cv_folds, seed=seed)
        y_hat = model.predict(Xy)
        r2_train = r2_score(yy, y_hat)

        # CV-based permutation importance
        ctor = make_model_ctor(random_state=seed)
        cv_means, cv_stds, base_mean, base_std = cv_permutation_importance(ctor, Xy,
                                                                             yy, n_repeats=perm_repeats, cv_splits=cv_folds, seed=seed)
        pd.DataFrame({
            "name": param_cols, "perm_mean": cv_means, "perm_std": cv_stds,
        }).to_csv(os.path.join(out_dir, f"perm_cv_{label}.csv"), index=False)

        plot_bar_sorted(param_cols, -cv_means, f"Permutation importance (CV mean R² drop) — {label}",
                        os.path.join(out_dir, f"pareto_perm_cv_{label}.png"), xlabel="Mean R² drop", invert=True)

        # ----------- XGBoost 'gain' Importance (optional) -----------
        gain = xgb_gain_importance(model, param_cols)
        if gain is not None:
            pd.DataFrame({
                "name": param_cols, "gain": gain,
            }).to_csv(os.path.join(out_dir, f"xgb_gain_{label}.csv"), index=False)

            plot_bar_sorted(
                param_cols, gain,
                f"XGBoost gain importance — {label}",
                os.path.join(out_dir, f"pareto_gain_{label}.png"),
                xlabel="Gain", invert=True
            )
            print(f"[GAIN] Wrote xgb_gain_{label}.csv and pareto_gain_{label}.png")
        else:
            print(f"[GAIN] skipped: model is not XGBoost or gain not available for {label}.")

        # Grouped permutation importance
        if do_group_perm:
            groups, group_names, Corr = build_corr_groups(Xy, param_cols, threshold=group_corr_threshold)
            if groups:
                g_means = grouped_permutation_importance(ctor, Xy, yy, groups,
                                                          n_repeats=perm_repeats, cv_splits=cv_folds, seed=seed)
                pd.DataFrame({
                    "group": [f"G{gi+1}" for gi in range(len(groups))],
                    "features": [";".join(nl) for nl in group_names],
                    "perm_mean": g_means,
                }).to_csv(os.path.join(out_dir, f"perm_groups_{label}.csv"), index=False)

                plot_bar_sorted([f"G{gi+1}" for gi in range(len(groups))], g_means,
                                f"Grouped permutation importance — {label}",
                                os.path.join(out_dir, f"pareto_perm_groups_{label}.png"),
                                xlabel="Mean R² drop", invert=True)
                np.savetxt(os.path.join(out_dir, f"feature_corr_{label}.csv"), Corr,
                           delimiter=",", fmt="%.6g")
            else:
                print(f"[INFO] No correlated feature groups found at threshold {group_corr_threshold}.")

        # PDP/ICE
        if do_pdp:
            top_idx = np.argsort(-cv_means)[:max(1, min(top_k_pdp, len(param_cols)))]
            partial_dependence_plots(model, Xy, param_cols, top_idx, out_dir, label)

        # SHAP (if available)
        if do_shap:
            shap_global_and_dependence(sigma_y, model, Xy, param_cols, out_dir,
                                       label, top_k=min(10, len(param_cols)))

        # Sobol on surrogate (if sidecar bounds available)
        if do_sobol and dims and scales:
            S1, ST, varY = sobol_on_surrogate(model, param_cols, dims, scales, N=sobol_samples, seed=seed)
            pd.DataFrame({
                "name": param_cols, "S1": S1, "ST": ST,
            }).to_csv(os.path.join(out_dir, f"sobol_{label}.csv"), index=False)

            plot_bar_sorted(param_cols, S1, f"Sobol first-order — {label}",
                            os.path.join(out_dir, f"sobol_S1_{label}.png"), xlabel="S1", invert=True)
            plot_bar_sorted(param_cols, ST, f"Sobol total-order — {label}",
                            os.path.join(out_dir, f"sobol_ST_{label}.png"), xlabel="ST", invert=True)

        results_summary.append({
            "response": label,
            "n_used_rows": int(np.sum(mask)),
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores, ddof=1)),
            "train_r2": float(r2_train),
            "perm_cv_base_r2_mean": float(base_mean),
            "perm_cv_base_r2_std": float(base_std),
        })

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print("Advanced analysis complete. See outputs/ for results.")


def write_params_template(path="parameters_template.json", example=None):
    if example is None:
        example = {
            "Run": {"min": 1, "max": 103, "scale": "linear"},
            "LD50": {"min": 10., "max": 25., "scale": "linear"},
            "LSigma": {"min": 0.1, "max": 0.6, "scale": "linear"},
            "LDmin": {"min": 1., "max": 10., "scale": "linear"},
            "LDmax": {"min": 30., "max": 70., "scale": "linear"},
            "LMF": {"min": 50., "max": 100., "scale": "linear"},
            "SD50": {"min": 0.5, "max": 0.9, "scale": "linear"},
            "SSigma": {"min": 0.0, "max": 0.25, "scale": "linear"},
            "SDmin": {"min": 0.5, "max": 0.9, "scale": "linear"},
            "SDmax": {"min": 1., "max": 30., "scale": "linear"},
            "SMF": {"min": 0., "max": 40., "scale": "linear"},
            "MD50": {"min": 2., "max": 4., "scale": "linear"},
            "MSigma": {"min": 0.3, "max": 0.7, "scale": "linear"},
            "MDmin": {"min": 0.9, "max": 1., "scale": "linear"},
            "MDmax": {"min": 15., "max": 30., "scale": "linear"},
            "MMF": {"min": 10., "max": 30., "scale": "linear"}
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"Wrote parameter template to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis toolkit: DOE generation and multi-method analysis."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- template ---
    p_tpl = sub.add_parser("template", help="Write a parameter template JSON file.")
    p_tpl.add_argument("--output", default="parameters_template.json",
                       help="Output path (default: parameters_template.json)")

    # --- design ---
    p_doe = sub.add_parser("design", help="Generate a Sobol DOE from a parameter JSON file.")
    p_doe.add_argument("--params", required=True, help="Path to parameters JSON file.")
    p_doe.add_argument("--n-samples", type=int, default=0,
                       help="Number of samples (default: 10*n_params or 50).")
    p_doe.add_argument("--output", default=None, help="Output CSV path (default: DOE_<timestamp>.csv).")
    p_doe.add_argument("--seed", type=int, default=None, help="Random seed.")

    # --- analyze ---
    p_ana = sub.add_parser("analyze", help="Run sensitivity analysis on a filled DOE CSV.")
    p_ana.add_argument("--csv", required=True, help="Path to the filled DOE CSV.")
    p_ana.add_argument("--response-cols", nargs="+", default=None,
                       help="Response column name(s) to analyze (default: permeability loss).")
    p_ana.add_argument("--out-dir", default="outputs", help="Output directory (default: outputs).")
    p_ana.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p_ana.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5).")
    p_ana.add_argument("--no-pdp", action="store_true", help="Skip partial dependence plots.")
    p_ana.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis.")
    p_ana.add_argument("--no-group-perm", action="store_true", help="Skip grouped permutation importance.")
    p_ana.add_argument("--no-sobol", action="store_true", help="Skip Sobol analysis.")

    args = parser.parse_args()

    if args.command == "template":
        write_params_template(path=args.output)
    elif args.command == "design":
        design(args.params, args.n_samples, args.output, seed=args.seed)
    elif args.command == "analyze":
        analyze(
            args.csv,
            response_cols=args.response_cols,
            out_dir=args.out_dir,
            seed=args.seed,
            cv_folds=args.cv_folds,
            do_pdp=not args.no_pdp,
            do_shap=not args.no_shap,
            do_group_perm=not args.no_group_perm,
            do_sobol=not args.no_sobol,
        )
