"""Morris Elementary Effects screening method."""

import numpy as np

# Try SALib for Morris
try:
    from SALib.sample import morris as salib_morris_sample
    from SALib.analyze import morris as salib_morris_analyze
    SALIB_OK = True
except ImportError:
    SALIB_OK = False


def morris_screening(param_cols, dims, scales, model, N=100, num_levels=4, seed=0):
    """
    Morris Elementary Effects screening on the surrogate model.

    Returns (mu_star, sigma) arrays of shape (d,), or (None, None) if SALib
    is not available.

    Parameters
    ----------
    param_cols : list[str]
        Parameter names.
    dims : list[tuple[float, float]]
        (min, max) bounds per parameter.
    scales : list[str]
        'linear' or 'log' per parameter.
    model : fitted sklearn/xgboost estimator
        Surrogate model with a .predict(X) method.
    N : int
        Number of trajectories.
    num_levels : int
        Number of grid levels for the Morris design.
    seed : int
        Random seed.
    """
    if not SALIB_OK:
        print("[WARN] SALib not installed — Morris screening skipped.")
        return None, None

    d = len(param_cols)

    # Build bounds in transformed space (log for log-scale params)
    bounds = []
    for (vmin, vmax), sc in zip(dims, scales):
        if sc == "log":
            bounds.append([np.log(vmin), np.log(vmax)])
        else:
            bounds.append([vmin, vmax])

    problem = {
        'num_vars': d,
        'names': list(param_cols),
        'bounds': bounds,
    }

    X_samples = salib_morris_sample.sample(problem, N=N, num_levels=num_levels,
                                           seed=seed)

    # Transform log-scale columns back to physical space
    for j, ((vmin, vmax), sc) in enumerate(zip(dims, scales)):
        if sc == "log":
            X_samples[:, j] = np.exp(X_samples[:, j])

    Y = np.asarray(model.predict(X_samples), dtype=np.float64).reshape(-1)
    Si = salib_morris_analyze.analyze(problem, X_samples, Y, print_to_console=False)

    mu_star = np.array(Si['mu_star'])
    sigma = np.array(Si['sigma'])

    print("[Morris] mu* and sigma:")
    for j in range(d):
        print(f"  {param_cols[j]}: mu*={mu_star[j]:.4f} | sigma={sigma[j]:.4f}")

    return mu_star, sigma
