# senstoolkit

A Python toolkit for **sensitivity analysis** of simulation or experimental data. It handles the full workflow: generate a space-filling Design of Experiments (DOE), run your simulations, then perform multi-method sensitivity analysis with publication-ready plots.

## Features

- **Sobol DOE generation** — quasi-random, extensible sample designs with support for linear and log-scaled parameters
- **DOE extension** — add more samples to an existing design without re-running previous simulations
- **12 sensitivity analysis methods** run in a single call:
  - Spearman & Pearson correlation with bootstrap confidence intervals
  - CV permutation importance
  - Grouped permutation importance (for correlated parameters)
  - XGBoost gain importance
  - Partial Dependence / ICE plots
  - SHAP global importance & dependence plots
  - SHAP interaction analysis
  - Sobol indices (first-order S1, total-order ST, second-order S2)
  - Morris Elementary Effects screening
  - Scatter plot grid
- **Surrogate quality gate** — automatically skips surrogate-dependent methods when the model fit is poor (CV R² below threshold)
- **CLI and Python API** — use from the command line or import into your own scripts

## Installation

```bash
pip install -e .
```

For all optional methods (XGBoost, SHAP, Sobol/Morris via SALib):

```bash
pip install -e ".[full]"
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy, pandas, scipy | Yes | Core computation |
| scikit-learn | Yes | Surrogate model, permutation importance, PDP |
| matplotlib | Yes | Plotting |
| xgboost | No | Better surrogate model (falls back to HistGradientBoosting) |
| shap | No | SHAP analysis |
| SALib | No | Sobol indices and Morris screening |

## Quick Start

### 1. Create a parameter template

```bash
senstoolkit template --output params.json
```

Edit `params.json` to define your parameters with name, min, max, and scale (linear or log):

```json
{
    "velocity":    {"min": 10.0, "max": 100.0, "scale": "linear"},
    "viscosity":   {"min": 1e-6, "max": 1e-3, "scale": "log"},
    "temperature": {"min": 200.0, "max": 800.0, "scale": "linear"}
}
```

### 2. Generate a DOE

```bash
senstoolkit design --params params.json --n-samples 256 --seed 42 --response-cols pressure drag
```

This creates a CSV with 256 Sobol-sampled parameter combinations and empty `pressure`/`drag` columns for your simulation results.

### 3. Run your simulations

Fill in the response columns in the CSV with your simulation outputs.

### 4. Run the analysis

```bash
senstoolkit analyze --csv DOE.csv --response-cols pressure drag
```

All 12 methods run by default. Results (CSVs, plots, summary JSON) are written to `outputs/`.

### 5. Extend the DOE (optional)

Need more samples? Extend the existing design without re-running previous simulations:

```bash
senstoolkit extend --csv DOE.csv --n-new 512
```

The new samples continue the same Sobol sequence, so the combined design remains space-filling.

## CLI Reference

```
senstoolkit template  --output PATH
senstoolkit design    --params PATH --n-samples N [--seed S] [--output PATH] [--response-cols COL...]
senstoolkit analyze   --csv PATH --response-cols COL... [--out-dir DIR] [--seed S]
                      [--cv-folds K] [--r2-threshold T]
                      [--no-pdp] [--no-shap] [--no-sobol] [--no-morris]
                      [--no-scatter] [--no-group-perm]
senstoolkit extend    --csv PATH --n-new N [--output PATH] [--seed S]
```

## Python API

```python
from senstoolkit import design, extend_design, analyze

# Generate DOE
design("params.json", n_samples=256, seed=42, response_cols=["pressure"])

# Run analysis
analyze("DOE.csv", response_cols=["pressure", "drag"], out_dir="results")

# Extend existing DOE
extend_design("DOE.csv", n_new=512)
```

## Output Files

For each response column, the analysis produces:

| File | Content |
|------|---------|
| `correlations_*.csv` | Spearman & Pearson correlations with bootstrap CIs |
| `perm_cv_*.csv` | CV permutation importance |
| `perm_groups_*.csv` | Grouped permutation importance |
| `xgb_gain_*.csv` | XGBoost gain importance |
| `sobol_*.csv` | Sobol S1 and ST indices |
| `sobol_S2_*.csv` | Sobol second-order interaction indices |
| `morris_*.csv` | Morris mu* and sigma |
| `corr_spearman_*.png` | Spearman correlation bar chart |
| `corr_pearson_*.png` | Pearson correlation bar chart |
| `pareto_perm_cv_*.png` | Permutation importance bar chart |
| `pareto_gain_*.png` | XGBoost gain bar chart |
| `sobol_S1_*.png` / `sobol_ST_*.png` | Sobol index bar charts |
| `sobol_S2_*.png` | Sobol interaction heatmap |
| `morris_mustar_*.png` / `morris_sigma_*.png` | Morris screening bar charts |
| `scatter_grid_*.png` | Parameter vs response scatter plots |
| `pdp_*.png` | Partial dependence / ICE plots |
| `shap_summary_*.png` | SHAP beeswarm plot |
| `shap_interactions_*.png` | SHAP interaction heatmap |
| `summary.json` | Model fit statistics and quality gate results |

## Documentation

A detailed manual with method backgrounds and interpretation guides is available at [doc/senstoolkit_manual.pdf](doc/senstoolkit_manual.pdf).

## License

MIT
