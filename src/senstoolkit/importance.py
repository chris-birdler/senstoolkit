import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# Try XGBoost; fall back if unavailable
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False


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
    return result.importances_mean


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


def grouped_permutation_importance(model_ctor, X, y, groups, n_repeats=20, cv_splits=5, seed=0):
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


def xgb_gain_importance(model, names):
    """
    Returns the 'gain' importance per feature as np.array.
    Only for XGBoost models; returns None otherwise.
    """
    if XGB_OK and isinstance(model, XGBRegressor):
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
