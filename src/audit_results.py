"""
Script: audit_results.py

Description:
    Audits the local ORBITA result artifacts: datasets, trained models,
    cross-validation metrics, benchmark CSVs, and final figures.
"""

from pathlib import Path
import csv
import re

from config import MIN_SAFE_PERIGEE
from physics.oracle import R_EQ


ARCHITECTURES = ("resnet", "mlp", "lstm", "linear", "tree")
TIME_DOMAIN_CASES = 10000
TIME_DOMAIN_STEPS_PER_CASE = 16
SPACE_DOMAIN_SAMPLES = 100000
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
    """Returns discovered grid dataset keys under data/ datasets folders."""
    keys = set()
    for search_dir in [
        root / "data" / "datasets" / "training",
        root / "data" / "datasets",
        root / "data",
    ]:
        if not search_dir.exists():
            continue
        for path in search_dir.glob("orbita_dataset_*.csv"):
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


def _resolve_file(root, relative):
    """Resolves relative file path checking new folder structure first, then root data/."""
    p = root / relative
    if p.exists():
        return p
    # Check alternate subfolder mappings
    rel_str = str(relative)
    if rel_str.startswith("data/metrics_") or rel_str.startswith("data/emissions"):
        filename = p.name
        alt = root / "data" / "metrics" / filename
        if alt.exists():
            return alt
    elif rel_str.startswith("data/benchmark_"):
        filename = p.name
        alt_global = root / "data" / "benchmarks" / "ablation_global" / filename
        alt_resnet = root / "data" / "benchmarks" / "ablation_resnet" / filename
        if alt_global.exists():
            return alt_global
        if alt_resnet.exists():
            return alt_resnet
    elif (
        rel_str.startswith("data/orbita_dataset")
        or rel_str.startswith("data/datasets/orbita_dataset")
    ):
        filename = p.name
        alt_tr = root / "data" / "datasets" / "training" / filename
        alt_ds = root / "data" / "datasets" / filename
        if alt_tr.exists():
            return alt_tr
        if alt_ds.exists():
            return alt_ds
    return p


def _exists(path):
    """Formats a path existence check for console output."""
    return "OK" if path.exists() else "MISSING"


def _count_csv_rows(path):
    """Returns the number of data rows in a CSV file."""
    with path.open(newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _count_time_domain_cases(path):
    """Returns the number of unique benchmark cases in a time-domain CSV."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return len({row["Case_ID"] for row in reader})


def _audit_benchmark_csv(root, relative, expected_rows, failures):
    """Audits benchmark CSV existence and row count."""
    path = _resolve_file(root, relative)
    if not path.exists():
        print(f"  {relative}: MISSING")
        failures.append(relative)
        return

    row_count = _count_csv_rows(path)
    status = "OK" if row_count == expected_rows else "BAD"
    print(f"  {relative}: {status} ({row_count}/{expected_rows} rows)")
    if row_count != expected_rows:
        failures.append(f"{relative}:rows={row_count}/{expected_rows}")


def _resolve_figure(root, prefix, ext):
    """Resolves figure path checking subdirectories benchmarks/<arch>, ablation_global, etc."""
    filename = f"{prefix}.{ext}"
    candidates = []
    if prefix.startswith("benchmark_"):
        arch = "resnet"
        for a in ARCHITECTURES:
            if f"_{a}_" in prefix or prefix.endswith(f"_{a}"):
                arch = a
                break
        candidates = [
            root / "figures" / "benchmarks" / arch / filename,
            root / "figures" / "benchmarks" / filename,
            root / "figures" / filename,
        ]
    elif prefix == "orbita_expert_mesh":
        candidates = [
            root / "figures" / filename,
            root / "figures" / "ablation_resnet" / filename,
        ]
    elif prefix.startswith("ablation_") and ("resnet" not in prefix):
        candidates = [
            root / "figures" / "ablation_global" / filename,
            root / "figures" / filename,
        ]
    elif prefix.startswith("metrics_"):
        candidates = [
            root / "figures" / "metrics" / filename,
            root / "figures" / filename,
        ]
    else:
        candidates = [
            root / "figures" / "ablation_resnet" / filename,
            root / "figures" / filename,
        ]

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


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
    global_dataset = _resolve_file(root, "data/orbita_dataset_300-2000_0.0000-0.1000_0-90.csv")
    print(f"  global dataset: {_exists(global_dataset)}")
    for architecture in ARCHITECTURES:
        extension = "joblib" if architecture == "tree" else "pth"
        model_filename = (
            f"orbita_predictor_{architecture}_"
            f"300-2000_0.0000-0.1000_0-90.{extension}"
        )
        model_candidates = [
            root / "models" / architecture / "base" / model_filename,
            root / "models" / architecture / model_filename,
            root / "models" / model_filename,
        ]
        model_path = model_candidates[-1]
        for mc in model_candidates:
            if mc.exists():
                model_path = mc
                break

        print(f"  {architecture:>6} model: {_exists(model_path)}")
        if not model_path.exists():
            failures.append(str(model_path.relative_to(root)))

    cv_path = _resolve_file(root, "data/metrics/metrics_cv.csv")
    print(f"  metrics_cv.csv: {_exists(cv_path)}")
    if not cv_path.exists():
        failures.append("data/metrics/metrics_cv.csv")
    else:
        with cv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            cv_architectures = {row["architecture"] for row in reader}
        missing_cv = sorted(set(ARCHITECTURES) - cv_architectures)
        if missing_cv:
            failures.append(f"metrics_cv:missing={','.join(missing_cv)}")

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
    expected_time_rows = TIME_DOMAIN_CASES * TIME_DOMAIN_STEPS_PER_CASE
    for architecture in ARCHITECTURES:
        relative = f"data/benchmark_time_domain_{architecture}.csv"
        _audit_benchmark_csv(root, relative, expected_time_rows, failures)
        path = _resolve_file(root, relative)
        if path.exists():
            case_count = _count_time_domain_cases(path)
            case_status = "OK" if case_count == TIME_DOMAIN_CASES else "BAD"
            print(
                f"    cases: {case_status} "
                f"({case_count}/{TIME_DOMAIN_CASES})"
            )
            if case_count != TIME_DOMAIN_CASES:
                failures.append(
                    f"{relative}:cases={case_count}/{TIME_DOMAIN_CASES}"
                )

        _audit_benchmark_csv(
            root,
            f"data/benchmark_space_domain_{architecture}.csv",
            SPACE_DOMAIN_SAMPLES,
            failures,
        )

    metrics_benchmark = _resolve_file(root, "data/metrics_benchmark.csv")
    print(f"  data/metrics_benchmark.csv: {_exists(metrics_benchmark)}")
    if not metrics_benchmark.exists():
        failures.append("data/metrics_benchmark.csv")
    else:
        with metrics_benchmark.open(newline="") as f:
            reader = csv.DictReader(f)
            metric_pairs = {
                (row["architecture"], row["mode"]) for row in reader
            }
        required_pairs = {
            (architecture, mode)
            for architecture in ARCHITECTURES
            for mode in ("time_domain", "space_domain")
        }
        missing_pairs = sorted(required_pairs - metric_pairs)
        if missing_pairs:
            print("    missing metrics:")
            for architecture, mode in missing_pairs:
                print(f"      - {architecture}:{mode}")
            failures.extend(
                f"metrics_benchmark:{architecture}:{mode}"
                for architecture, mode in missing_pairs
            )

    print("\nFigures:")
    figure_files = [
        f for f in sorted((root / "figures").glob("**/*.*"))
        if f.is_file() and not f.name.startswith(".")
    ]
    print(f"  total files: {len(figure_files)}")
    required_prefixes = [
        "ablation_time_domain_comparison",
        "ablation_space_domain_cdf",
        "metrics_training_time",
        "metrics_model_size",
        "metrics_inference_time_domain",
        "metrics_inference_space_domain",
        "metrics_energy_kwh",
        "metrics_co2_emissions",
        "metrics_cross_validation",
        "orbita_expert_mesh",
    ]
    for architecture in ARCHITECTURES:
        required_prefixes.extend([
            f"benchmark_time_domain_{architecture}_envelope",
            f"benchmark_space_domain_{architecture}_cdf",
            f"benchmark_space_domain_{architecture}_heatmap",
            f"benchmark_space_domain_{architecture}_histograms",
            f"benchmark_space_domain_{architecture}_scatter",
            f"benchmark_space_domain_{architecture}_violin",
        ])
        if architecture == "resnet":
            required_prefixes.extend([
                f"benchmark_space_domain_{architecture}_heatmap_radial",
                f"benchmark_space_domain_{architecture}_heatmap_cross_track",
            ])
    for prefix in required_prefixes:
        png = _resolve_figure(root, prefix, "png")
        svg = _resolve_figure(root, prefix, "svg")
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
