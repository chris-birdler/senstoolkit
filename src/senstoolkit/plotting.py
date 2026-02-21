import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from .utils import ensure_outdir

# Try SHAP
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# Need XGB_OK to gate SHAP TreeExplainer
try:
    from xgboost import XGBRegressor  # noqa: F401
    XGB_OK = True
except ImportError:
    XGB_OK = False


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
        plot_bar_sorted(names, mean_abs, f"SHAP mean |value| — {label}",
                        os.path.join(out_dir, f"shap_bar_{label}.png"), xlabel="mean |SHAP|", invert=True)
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


def scatter_grid(X, y, names, label, out_dir):
    """Scatter plot grid: one subplot per parameter vs response."""
    p = len(names)
    ncols = min(3, p)
    nrows = math.ceil(p / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for j in range(p):
        ax = axes[j // ncols][j % ncols]
        ax.scatter(X[:, j], y, s=10, alpha=0.6)
        ax.set_xlabel(names[j])
        ax.set_ylabel(label)
    for j in range(p, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(f"Scatter grid — {label}", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"scatter_grid_{label}.png")
    ensure_outdir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sobol_s2_heatmap(names, S2, label, out_path):
    """Heatmap of Sobol second-order interaction indices."""
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), max(5, len(names) * 0.5)))
    im = ax.imshow(S2, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, label="S2")
    ax.set_title(f"Sobol S2 interactions — {label}")
    fig.tight_layout()
    ensure_outdir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def shap_interaction_analysis(model, X, names, out_dir, label, top_k=10):
    """SHAP interaction value heatmap."""
    if not (SHAP_OK and XGB_OK):
        return None
    try:
        explainer = shap.TreeExplainer(model)
        X_sample = X if X.shape[0] <= 2000 else X[np.random.RandomState(0).choice(X.shape[0], size=2000, replace=False)]
        sv_inter = explainer.shap_interaction_values(X_sample)
        sv_inter = np.array(sv_inter)
        mean_abs_inter = np.mean(np.abs(sv_inter), axis=0)

        pd.DataFrame(mean_abs_inter, index=names, columns=names).to_csv(
            os.path.join(out_dir, f"shap_interactions_{label}.csv"))

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), max(5, len(names) * 0.5)))
        im = ax.imshow(mean_abs_inter, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        fig.colorbar(im, ax=ax, label="mean |SHAP interaction|")
        ax.set_title(f"SHAP interactions — {label}")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"shap_interactions_{label}.png")
        ensure_outdir(out_path)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return mean_abs_inter
    except Exception:
        return None
