import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .utils import ensure_outdir


def parse_params_json(path):
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    dims = []
    names = []
    scales = []
    for name, spec in params.items():
        if name == "_meta":
            continue
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


def write_doe_csv(path, names, X, response_cols=()):
    response_cols = list(response_cols)
    ensure_outdir(path)
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "id", range(len(df)))
    for col in response_cols:
        df[col] = ""
    df.to_csv(path, index=False)


def design(params_json, n_samples, out_csv, seed=None, prefer_power_of_two=False, response_cols=()):
    names, dims, scales = parse_params_json(params_json)
    p = len(names)
    if n_samples is None or n_samples <= 0:
        n_samples = max(10 * p, 50)
    raw = sobol_sample(dims, n_samples, seed=seed, prefer_power_of_two=prefer_power_of_two)
    X = apply_scaling(raw, dims, scales)
    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"DOE_{ts}.csv"
    write_doe_csv(out_csv, names, X, response_cols=response_cols)
    sidecar = os.path.splitext(out_csv)[0] + ".params.json"
    sidecar_data = {"_meta": {"seed": seed, "n_samples": n_samples}}
    sidecar_data.update({n: {"min": mn, "max": mx, "scale": sc} for n, (mn, mx), sc in zip(
        names, dims, scales)})
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)
    print(f"Design written to: {out_csv}")
    print(f"Wrote sidecar parameter spec: {sidecar}")
    print("Fill the response columns with your simulation results and run 'analyze'.")


def extend_design(existing_csv, n_new, out_csv=None, seed=None):
    """Extend an existing Sobol DOE by appending new samples.

    Uses Sobol sequence fast-forward to skip past existing points and
    generate the next batch, preserving the quasi-random properties.
    """
    # 1. Read existing DOE
    df_existing = pd.read_csv(existing_csv)
    n_existing = len(df_existing)

    # 2. Read sidecar for parameter specs and original seed
    sidecar_path = os.path.splitext(existing_csv)[0] + ".params.json"
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            f"Sidecar file not found: {sidecar_path}. "
            f"Cannot extend without parameter bounds and seed.")

    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar_data = json.load(f)

    meta = sidecar_data.get("_meta", {})
    original_seed = meta.get("seed")
    if seed is None:
        seed = original_seed
    if seed is None:
        raise ValueError(
            "No seed found in sidecar _meta and no --seed provided. "
            "Cannot reproduce the Sobol sequence without the original seed.")

    # 3. Parse parameter specs (parse_params_json skips _meta)
    names, dims, scales = [], [], []
    for name, spec in sidecar_data.items():
        if name == "_meta":
            continue
        vmin, vmax = float(spec["min"]), float(spec["max"])
        sc = str(spec.get("scale", "linear")).lower()
        names.append(name)
        dims.append((vmin, vmax))
        scales.append(sc)

    # 4. Detect response columns (in CSV but not in params, excluding 'id')
    param_set = set(names) | {"id"}
    response_cols = [c for c in df_existing.columns if c not in param_set]

    # 5. Recreate Sobol engine, skip past existing points
    d = len(names)
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    engine.fast_forward(n_existing)

    # 6. Generate new points in [0,1]^d
    u_new = engine.random(n=n_new)

    # 7. Scale to physical space
    X_new = np.empty_like(u_new)
    for j, ((vmin, vmax), sc) in enumerate(zip(dims, scales)):
        if sc == "log":
            lo, hi = math.log(vmin), math.log(vmax)
            X_new[:, j] = np.exp(lo + u_new[:, j] * (hi - lo))
        else:
            X_new[:, j] = vmin + u_new[:, j] * (vmax - vmin)

    # 8. Build new rows with continuing IDs
    df_new = pd.DataFrame(X_new, columns=names)
    df_new.insert(0, "id", range(n_existing, n_existing + n_new))
    for col in response_cols:
        df_new[col] = ""

    # 9. Combine and write
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    if out_csv is None:
        out_csv = existing_csv
    ensure_outdir(out_csv)
    df_combined.to_csv(out_csv, index=False)

    # 10. Update sidecar _meta
    meta["n_samples"] = n_existing + n_new
    meta["seed"] = seed
    sidecar_data["_meta"] = meta
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    print(f"Extended DOE: {n_existing} -> {n_existing + n_new} samples.")
    print(f"Written to: {out_csv}")
    return out_csv


def suggest_points(existing_csv, response_col, n_suggest, out_csv=None,
                   target="high", quantile=0.8, n_candidates=10000, seed=None,
                   strategy="random"):
    """Suggest new simulation points targeting a specific response region.

    Uses the existing data to train a surrogate model, generates a large
    pool of Sobol candidates, predicts their responses, and selects the
    ones most likely to fall in the target region (e.g. high response values
    where data is sparse).

    Parameters
    ----------
    existing_csv : str
        Path to the existing DOE CSV with filled response values.
    response_col : str
        Name of the response column to target.
    n_suggest : int
        Number of points to suggest.
    out_csv : str or None
        Output CSV path. If None, writes to suggest_<response_col>.csv.
    target : str
        "high" to target high response values, "low" for low values.
    quantile : float
        Quantile threshold (0-1). For "high", keeps candidates above this
        quantile of predicted values. For "low", keeps below (1-quantile).
    n_candidates : int
        Size of the Sobol candidate pool to screen.
    seed : int or None
        Random seed.
    strategy : str
        "random" to sample randomly from the filtered region, "top" to
        pick the N most extreme predicted values.

    Returns
    -------
    str
        Path to the written CSV.
    """
    from .importance import fit_model

    # 1. Read existing data and sidecar
    df = pd.read_csv(existing_csv)
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in CSV.")

    sidecar_path = os.path.splitext(existing_csv)[0] + ".params.json"
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            f"Sidecar file not found: {sidecar_path}. "
            f"Parameter bounds are required for candidate generation.")

    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar_data = json.load(f)

    names, dims, scales = [], [], []
    for name, spec in sidecar_data.items():
        if name == "_meta":
            continue
        vmin, vmax = float(spec["min"]), float(spec["max"])
        sc = str(spec.get("scale", "linear")).lower()
        names.append(name)
        dims.append((vmin, vmax))
        scales.append(sc)

    # 2. Prepare training data
    y = pd.to_numeric(df[response_col], errors="coerce").values
    X = df[names].apply(pd.to_numeric, errors="coerce").values
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X_train, y_train = X[mask], y[mask]
    if len(y_train) < 10:
        raise ValueError(f"Only {len(y_train)} valid rows — too few to train a surrogate.")

    # 3. Train surrogate
    rs = seed if seed is not None else 0
    model = fit_model(X_train, y_train, random_state=rs)
    print(f"[SUGGEST] Trained surrogate on {len(y_train)} rows.")

    # 4. Generate large candidate pool
    cand_seed = rs + 9999
    d = len(names)
    engine = qmc.Sobol(d=d, scramble=True, seed=cand_seed)
    u = engine.random(n=n_candidates)
    X_cand = np.empty_like(u)
    for j, ((vmin, vmax), sc) in enumerate(zip(dims, scales)):
        if sc == "log":
            lo, hi = math.log(vmin), math.log(vmax)
            X_cand[:, j] = np.exp(lo + u[:, j] * (hi - lo))
        else:
            X_cand[:, j] = vmin + u[:, j] * (vmax - vmin)

    # 5. Predict and filter
    y_pred = model.predict(X_cand)

    if target == "high":
        threshold = np.quantile(y_pred, quantile)
        mask_target = y_pred >= threshold
    elif target == "low":
        threshold = np.quantile(y_pred, 1.0 - quantile)
        mask_target = y_pred <= threshold
    else:
        raise ValueError(f"target must be 'high' or 'low', got '{target}'.")

    X_filtered = X_cand[mask_target]
    y_filtered = y_pred[mask_target]

    if len(X_filtered) == 0:
        raise RuntimeError("No candidates passed the filter — try lowering the quantile.")

    # 6. Select N points from the filtered region
    if n_suggest >= len(X_filtered):
        sel = np.arange(len(X_filtered))
    elif strategy == "top":
        if target == "high":
            order = np.argsort(-y_filtered)
        else:
            order = np.argsort(y_filtered)
        sel = order[:n_suggest]
    else:
        rng = np.random.default_rng(rs)
        sel = rng.choice(len(X_filtered), size=n_suggest, replace=False)

    X_out = X_filtered[sel]
    y_out = y_filtered[sel]

    # 7. Write output
    if out_csv is None:
        out_csv = f"suggest_{response_col}.csv"
    ensure_outdir(out_csv)

    df_out = pd.DataFrame(X_out, columns=names)
    df_out.insert(0, "id", range(len(df_out)))
    df_out[f"{response_col}_predicted"] = y_out
    df_out[response_col] = ""  # empty column for actual simulation results
    df_out.to_csv(out_csv, index=False)

    print(f"[SUGGEST] Target: {target} {response_col} (threshold: {threshold:.4f})")
    print(f"[SUGGEST] Predicted range of suggestions: "
          f"{y_out.min():.4f} — {y_out.max():.4f}")
    print(f"[SUGGEST] Wrote {len(X_out)} suggested points to: {out_csv}")
    return out_csv
