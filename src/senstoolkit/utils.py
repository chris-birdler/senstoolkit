import json
from pathlib import Path


def ensure_outdir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_params_template(path="parameters_template.json", example=None):
    if example is None:
        example = {
            "param_1": {"min": 0.0, "max": 1.0, "scale": "linear"},
            "param_2": {"min": 1.0, "max": 100.0, "scale": "log"},
            "param_3": {"min": -10.0, "max": 10.0, "scale": "linear"},
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"Wrote parameter template to: {path}")
