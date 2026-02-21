import json
import os

import numpy as np
import pandas as pd

from .utils import ensure_outdir
from .correlation import correlation_vector, bootstrap_correlation_ci, build_corr_groups
from .importance import (
    fit_model, cv_r2, make_model_ctor, cv_permutation_importance,
    grouped_permutation_importance, xgb_gain_importance,
)
from .sobol import sobol_on_surrogate
from .morris import morris_screening
from .plotting import (
    plot_bar_sorted, partial_dependence_plots, shap_global_and_dependence,
    scatter_grid, plot_sobol_s2_heatmap, shap_interaction_analysis,
)
from sklearn.metrics import r2_score


def build_bounds_scales_by_name(param_cols_csv, params_obj):
    """
    Returns (param_cols, dims, scales) in CSV column order,
    restricted to names present in the parameter file. Raises if mismatched.
    params_obj: dict{name:{min,max,scale}} or list[{"name":...,"min":...,"max":...,"scale":...}]
    """
    if isinstance(params_obj, list):
        mapping = {it["name"]: it for it in params_obj if it.get("name") != "_meta"}
    elif isinstance(params_obj, dict):
        mapping = {k: v for k, v in params_obj.items() if k != "_meta"}
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


def analyze(design_csv, response_cols, out_dir="outputs",
            seed=0,
            cv_folds=5, perm_repeats=20, bootstrap_corr=1000, group_corr_threshold=0.9,
            top_k_pdp=6, do_pdp=True, do_shap=True, do_group_perm=True,
            do_sobol=True, sobol_samples=2048,
            do_morris=True, do_scatter=True, r2_threshold=0.5):
    if not response_cols:
        raise ValueError("response_cols must be provided and non-empty.")
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
        print("[INFO] No sidecar parameter file found; Sobol/Morris analysis will be skipped.")

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

        # 1. Scatter grid (no model needed)
        if do_scatter:
            scatter_grid(Xy, yy, param_cols, label, out_dir)
            print(f"[SCATTER] Wrote scatter_grid_{label}.png")

        # 2. Correlations + Bootstrap CIs
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

        # 3. Model & CV
        model = fit_model(Xy, yy, random_state=seed)
        cv_scores = cv_r2(model, Xy, yy, splits=cv_folds, seed=seed)
        y_hat = model.predict(Xy)
        r2_train = r2_score(yy, y_hat)
        cv_r2_mean = float(np.mean(cv_scores))

        # 4. Surrogate quality gate
        surrogate_ok = True
        if cv_r2_mean < r2_threshold:
            print(f"[WARN] Response '{label}': CV R² = {cv_r2_mean:.3f} < {r2_threshold}. "
                  f"Surrogate is unreliable — skipping PDP, SHAP, Sobol, Morris for this response.")
            surrogate_ok = False

        # 5. CV-based permutation importance
        ctor = make_model_ctor(random_state=seed)
        cv_means, cv_stds, base_mean, base_std = cv_permutation_importance(ctor, Xy,
                                                                             yy, n_repeats=perm_repeats, cv_splits=cv_folds, seed=seed)
        pd.DataFrame({
            "name": param_cols, "perm_mean": cv_means, "perm_std": cv_stds,
        }).to_csv(os.path.join(out_dir, f"perm_cv_{label}.csv"), index=False)

        plot_bar_sorted(param_cols, -cv_means, f"Permutation importance (CV mean R² drop) — {label}",
                        os.path.join(out_dir, f"pareto_perm_cv_{label}.png"), xlabel="Mean R² drop", invert=True)

        # 6. XGBoost 'gain' Importance (optional)
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

        # 7. Grouped permutation importance
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

        # 8. Morris screening (requires sidecar bounds + surrogate quality)
        if do_morris and dims and scales and surrogate_ok:
            mu_star, sigma = morris_screening(param_cols, dims, scales, model, seed=seed)
            if mu_star is not None:
                pd.DataFrame({
                    "name": param_cols, "mu_star": mu_star, "sigma": sigma,
                }).to_csv(os.path.join(out_dir, f"morris_{label}.csv"), index=False)
                plot_bar_sorted(param_cols, mu_star, f"Morris mu* — {label}",
                                os.path.join(out_dir, f"morris_mustar_{label}.png"), xlabel="mu*", invert=True)
                plot_bar_sorted(param_cols, sigma, f"Morris sigma — {label}",
                                os.path.join(out_dir, f"morris_sigma_{label}.png"), xlabel="sigma", invert=True)
                print(f"[MORRIS] Wrote morris_{label}.csv")

        # 9. PDP/ICE (requires surrogate quality)
        if do_pdp and surrogate_ok:
            top_idx = np.argsort(-cv_means)[:max(1, min(top_k_pdp, len(param_cols)))]
            partial_dependence_plots(model, Xy, param_cols, top_idx, out_dir, label)

        # 10. SHAP + SHAP interactions (requires surrogate quality)
        if do_shap and surrogate_ok:
            shap_global_and_dependence(sigma_y, model, Xy, param_cols, out_dir,
                                       label, top_k=min(10, len(param_cols)))
            shap_interaction_analysis(model, Xy, param_cols, out_dir, label,
                                      top_k=min(10, len(param_cols)))

        # 11. Sobol S1/ST/S2 (requires sidecar bounds + surrogate quality)
        if do_sobol and dims and scales and surrogate_ok:
            S1, ST, S2, varY = sobol_on_surrogate(model, param_cols, dims, scales, N=sobol_samples, seed=seed)
            pd.DataFrame({
                "name": param_cols, "S1": S1, "ST": ST,
            }).to_csv(os.path.join(out_dir, f"sobol_{label}.csv"), index=False)

            plot_bar_sorted(param_cols, S1, f"Sobol first-order — {label}",
                            os.path.join(out_dir, f"sobol_S1_{label}.png"), xlabel="S1", invert=True)
            plot_bar_sorted(param_cols, ST, f"Sobol total-order — {label}",
                            os.path.join(out_dir, f"sobol_ST_{label}.png"), xlabel="ST", invert=True)

            # S2 interaction heatmap
            if S2 is not None and np.any(S2 != 0):
                pd.DataFrame(S2, index=param_cols, columns=param_cols).to_csv(
                    os.path.join(out_dir, f"sobol_S2_{label}.csv"))
                plot_sobol_s2_heatmap(param_cols, S2, label,
                                      os.path.join(out_dir, f"sobol_S2_{label}.png"))
                print(f"[SOBOL] Wrote sobol_S2_{label}.csv and sobol_S2_{label}.png")

        results_summary.append({
            "response": label,
            "n_used_rows": int(np.sum(mask)),
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": float(np.std(cv_scores, ddof=1)),
            "train_r2": float(r2_train),
            "surrogate_ok": surrogate_ok,
            "perm_cv_base_r2_mean": float(base_mean),
            "perm_cv_base_r2_std": float(base_std),
        })

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print("Analysis complete. See outputs/ for results.")
