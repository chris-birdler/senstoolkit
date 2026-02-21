import argparse

from .doe import design, extend_design
from .analyze import analyze
from .utils import write_params_template


def main():
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
    p_doe.add_argument("--response-cols", nargs="+", default=[],
                       help="Response column name(s) to add as empty columns in the DOE CSV.")

    # --- analyze ---
    p_ana = sub.add_parser("analyze", help="Run sensitivity analysis on a filled DOE CSV.")
    p_ana.add_argument("--csv", required=True, help="Path to the filled DOE CSV.")
    p_ana.add_argument("--response-cols", nargs="+", required=True,
                       help="Response column name(s) to analyze.")
    p_ana.add_argument("--out-dir", default="outputs", help="Output directory (default: outputs).")
    p_ana.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p_ana.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5).")
    p_ana.add_argument("--no-pdp", action="store_true", help="Skip partial dependence plots.")
    p_ana.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis.")
    p_ana.add_argument("--no-group-perm", action="store_true", help="Skip grouped permutation importance.")
    p_ana.add_argument("--no-sobol", action="store_true", help="Skip Sobol analysis.")
    p_ana.add_argument("--no-morris", action="store_true", help="Skip Morris screening.")
    p_ana.add_argument("--no-scatter", action="store_true", help="Skip scatter plot grid.")
    p_ana.add_argument("--r2-threshold", type=float, default=0.5,
                       help="Minimum CV R² to proceed with surrogate methods (default: 0.5).")

    # --- extend ---
    p_ext = sub.add_parser("extend", help="Extend an existing Sobol DOE with additional samples.")
    p_ext.add_argument("--csv", required=True, help="Path to the existing DOE CSV.")
    p_ext.add_argument("--n-new", type=int, required=True, help="Number of new samples to add.")
    p_ext.add_argument("--output", default=None,
                       help="Output CSV path (default: overwrite the existing CSV).")
    p_ext.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: read from sidecar _meta).")

    args = parser.parse_args()

    if args.command == "template":
        write_params_template(path=args.output)
    elif args.command == "design":
        design(args.params, args.n_samples, args.output, seed=args.seed,
               response_cols=args.response_cols)
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
            do_morris=not args.no_morris,
            do_scatter=not args.no_scatter,
            r2_threshold=args.r2_threshold,
        )
    elif args.command == "extend":
        extend_design(args.csv, args.n_new, out_csv=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
