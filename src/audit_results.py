"""
Script: audit_results.py

Description:
    Audits the local ORBITA result artifacts: datasets, trained models,
    cross-validation metrics, benchmark CSVs, and final figures.
"""

from pathlib import Path
import re

from config import MIN_SAFE_PERIGEE
from physics.oracle import R_EQ


ARCHITECTURES = ("resnet", "mlp", "lstm", "linear", "tree")
ALTITUDE_BINS_KM = (
    (300, 640),
    (640, 980),
    (980, 1320),
    (1320, 1660),
    (1660, 2000),
)
ECCENTRICITY_BINS = (
    (0.0, 0.025),
    (0.025, 0.05),
    (0.05, 0.075),
    (0.075, 0.1),
)
INCLINATION_BINS_DEG = ((0, 45), (45, 90))

DATASET_RE = re.compile(
    r"orbita_dataset_(\d+-\d+)_(\d+\.\d+-\d+\.\d+)_(\d+-\d+)\.csv$"
)
MODEL_RE = re.compile(
    r"orbita_predictor_(resnet|mlp|lstm|linear|tree)_"
    r"(\d+-\d+)_(\d+\.\d+-\d+\.\d+)_(\d+-\d+)"
    r"(_finetuned)?\.(pth|joblib)$"
)


def _range_label(values, precision=None):
    """Formats a numeric range using ORBITA filename conventions."""
    left, right = values
    if precision is None:
        return f"{int(left)}-{int(right)}"
    return f"{left:.{precision}f}-{right:.{precision}f}"


def _cell_is_physically_valid(altitude_bounds_km, ecc_bounds):
    """Returns True if the grid cell can contain at least one safe orbit."""
    max_sma = R_EQ + altitude_bounds_km[1] * 1000.0
    min_ecc = ecc_bounds[0]
    return max_sma * (1.0 - min_ecc) >= MIN_SAFE_PERIGEE


def _expected_valid_grid_cells():
    """Builds the expected valid 5 x 4 x 2 LEO grid."""
    valid = []
    invalid = []
    for altitude in ALTITUDE_BINS_KM:
        for eccentricity in ECCENTRICITY_BINS:
            for inclination in INCLINATION_BINS_DEG:
                key = (
                    _range_label(altitude),
                    _range_label(eccentricity, precision=4),
                    _range_label(inclination),
                )
                if _cell_is_physically_valid(altitude, eccentricity):
                    valid.append(key)
                else:
                    invalid.append(key)
    return valid, invalid


def _discover_grid_datasets(root):
    """Returns discovered grid dataset keys under data/."""
    keys = set()
    for path in (root / "data").glob("orbita_dataset_*.csv"):
        match = DATASET_RE.match(path.name)
        if not match:
            continue
        key = match.groups()
        if key != ("300-2000", "0.0000-0.1000", "0-90"):
            keys.add(key)
    return keys


def _discover_models(root):
    """Returns discovered model and fine-tuned model keys."""
    base = set()
    fine_tuned = set()
    for path in (root / "models").glob("**/*"):
        if not path.is_file():
            continue
        match = MODEL_RE.match(path.name)
        if not match:
            continue
        arch, altitude, eccentricity, inclination, suffix, _ = match.groups()
        key = (arch, (altitude, eccentricity, inclination))
        if suffix:
            fine_tuned.add(key)
        else:
            base.add(key)
    return base, fine_tuned


def _exists(path):
    """Formats a path existence check for console output."""
    return "OK" if path.exists() else "MISSING"


def main():
    """Runs the artifact audit and exits non-zero when required results miss."""
    root = Path(__file__).resolve().parents[1]
    valid_cells, invalid_cells = _expected_valid_grid_cells()
    valid_cell_set = set(valid_cells)
    datasets = _discover_grid_datasets(root)
    base_models, fine_tuned_models = _discover_models(root)

    failures = []

    print("ORBITA Result Audit")
    print("=" * 80)

    print("\nGlobal models and CV:")
    global_dataset = root / "data/orbita_dataset_300-2000_0.0000-0.1000_0-90.csv"
    print(f"  global dataset: {_exists(global_dataset)}")
    for architecture in ARCHITECTURES:
        extension = "joblib" if architecture == "tree" else "pth"
        model_path = (
            root
            / "models"
            / (
                f"orbita_predictor_{architecture}_"
                f"300-2000_0.0000-0.1000_0-90.{extension}"
            )
        )
        print(f"  {architecture:>6} model: {_exists(model_path)}")
        if not model_path.exists():
            failures.append(str(model_path.relative_to(root)))

    cv_path = root / "data/metrics_cv.csv"
    print(f"  metrics_cv.csv: {_exists(cv_path)}")
    if not cv_path.exists():
        failures.append("data/metrics_cv.csv")

    print("\nResNet MoE grid:")
    missing_datasets = sorted(valid_cell_set - datasets)
    resnet_base = {
        key for architecture, key in base_models if architecture == "resnet"
    }
    resnet_fine_tuned = {
        key
        for architecture, key in fine_tuned_models
        if architecture == "resnet"
    }
    missing_base = sorted(valid_cell_set - resnet_base)
    missing_fine_tuned = sorted(valid_cell_set - resnet_fine_tuned)
    print(f"  nominal cells: {len(valid_cells) + len(invalid_cells)}")
    print(f"  physically invalid cells skipped: {len(invalid_cells)}")
    print(f"  valid cells expected: {len(valid_cells)}")
    print(f"  datasets: {len(datasets & valid_cell_set)}/{len(valid_cells)}")
    print(f"  base models: {len(resnet_base & valid_cell_set)}/{len(valid_cells)}")
    print(
        "  fine-tuned models: "
        f"{len(resnet_fine_tuned & valid_cell_set)}/{len(valid_cells)}"
    )
    if invalid_cells:
        print("  invalid cells:")
        for key in invalid_cells:
            print(f"    - {'_'.join(key)}")
    if missing_datasets:
        failures.extend(f"dataset:{'_'.join(key)}" for key in missing_datasets)
    if missing_base:
        failures.extend(f"base_model:{'_'.join(key)}" for key in missing_base)
    if missing_fine_tuned:
        failures.extend(
            f"finetuned_model:{'_'.join(key)}" for key in missing_fine_tuned
        )

    print("\nBenchmarks:")
    required_benchmarks = (
        "data/benchmark_time_domain_resnet.csv",
        "data/benchmark_space_domain_resnet.csv",
        "data/metrics_benchmark.csv",
    )
    for relative in required_benchmarks:
        path = root / relative
        print(f"  {relative}: {_exists(path)}")
        if not path.exists():
            failures.append(relative)

    print("\nFigures:")
    figure_files = sorted((root / "figures").glob("*.*"))
    print(f"  total files: {len(figure_files)}")
    required_prefixes = (
        "benchmark_time_domain_resnet_envelope",
        "benchmark_space_domain_resnet_cdf",
        "benchmark_space_domain_resnet_heatmap",
        "benchmark_space_domain_resnet_histograms",
        "benchmark_space_domain_resnet_scatter",
        "benchmark_space_domain_resnet_violin",
        "ablation_time_domain_comparison",
        "ablation_space_domain_cdf",
        "metrics_training_time",
        "metrics_model_size",
        "metrics_inference_time_domain",
        "metrics_inference_space_domain",
        "metrics_energy_kwh",
        "metrics_co2_emissions",
        "metrics_cross_validation",
    )
    for prefix in required_prefixes:
        png = root / "figures" / f"{prefix}.png"
        svg = root / "figures" / f"{prefix}.svg"
        print(f"  {prefix}: PNG={_exists(png)} SVG={_exists(svg)}")
        if not png.exists():
            failures.append(str(png.relative_to(root)))
        if not svg.exists():
            failures.append(str(svg.relative_to(root)))

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nPASS")


if __name__ == "__main__":
    main()
