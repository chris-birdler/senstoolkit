import math

import numpy as np
from scipy.stats import qmc

# Try SALib for Sobol analysis
try:
    from SALib.sample import sobol as salib_sobol_sample
    from SALib.analyze import sobol as salib_sobol
    SALIB_OK = True
except ImportError:
    SALIB_OK = False


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

    X_samples = salib_sobol_sample.sample(problem, N, seed=seed)

    # Transform log-scale columns back to physical space
    for j, ((vmin, vmax), sc) in enumerate(zip(dims, scales)):
        if sc == "log":
            X_samples[:, j] = np.exp(X_samples[:, j])

    Y = np.asarray(model.predict(X_samples), dtype=np.float64).reshape(-1)
    Si = salib_sobol.analyze(problem, Y, print_to_console=False)

    S1 = np.array(Si['S1'])
    ST = np.array(Si['ST'])
    S2 = np.array(Si.get('S2', np.zeros((d, d))))
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

    return S1, ST, S2, V


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

    return S1, ST, np.zeros((d, d)), V


def sobol_on_surrogate(model, names, dims, scales, N=8192, seed=0):
    """Sobol analysis: uses SALib if available, falls back to manual implementation.
    Returns (S1, ST, S2, V) where S2 is a (d x d) matrix of second-order indices."""
    if SALIB_OK:
        return _sobol_on_surrogate_salib(model, names, dims, scales, N=N, seed=seed)
    else:
        return _sobol_on_surrogate_manual(model, names, dims, scales, N=N, seed=seed)
