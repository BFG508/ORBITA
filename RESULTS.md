# ORBITA Results Manifest

This manifest records the expected local result artifacts for the current TFM technical closure.

## Global Model & Dataset Structure

- Base Dataset: `data/datasets/training/orbita_dataset_300-2000_0.0000-0.1000_0.00-90.00.csv`.
- Base Models: ResNet, MLP, LSTM, Linear, and Decision Tree under `models/{architecture}/base/`.
- Fine-Tuned Models: ResNet, MLP, LSTM, and Linear under `models/{architecture}/finetuned/`.
- Cross-Validation Summary: `data/metrics/metrics_cv.csv`.
- Training Metrics: `data/metrics/metrics_train.csv`.
- Benchmark Metrics: `data/metrics/metrics_benchmark.csv`.
- CodeCarbon Output: `data/metrics/emissions.csv`. The emissions visualizer uses direct CodeCarbon measurements. The final energy and carbon-footprint figures stack measured training, time-domain benchmark, and space-domain benchmark phases.
- Benchmark Datasets:
  - `data/benchmarks/ablation_global/benchmark_time_domain_{architecture}.csv`: 10,000 cases and 160,000 propagated time-step rows per architecture.
  - `data/benchmarks/ablation_global/benchmark_space_domain_{architecture}.csv`: 100,000 Monte Carlo samples per architecture.
  - `data/benchmarks/ablation_resnet/`: ResNet temporal paired and spatial case benchmarks.

## ResNet MoE Grid

The ResNet Mixture-of-Experts fleet uses a nominal `5 x 4 x 2` grid (38 valid cells):

- SMA/Altitude Bins: `300-640`, `640-980`, `980-1320`, `1320-1660`, `1660-2000` km.
- Eccentricity Bins: `0.0000-0.0250`, `0.0250-0.0500`, `0.0500-0.0750`, `0.0750-0.1000`.
- Inclination Bins: `0.00-45.00`, `45.00-90.00` deg.

Physically invalid cells (perigee < 300 km):
- `300-640_0.0750-0.1000_0.00-45.00`
- `300-640_0.0750-0.1000_45.00-90.00`

The 38 valid expert weights reside under `models/resnet/base/` and `models/resnet/finetuned/`, and active-learning datasets reside under `data/datasets/finetuning/resnet/`.

## Final Figures

Final figures are exported in both PNG (high-DPI) and SVG (vectorial) under subdirectories of `figures/`:

- Individual Architecture Reports (`figures/benchmarks/{architecture}/`):
  - `benchmark_time_domain_{architecture}_envelope`
  - `benchmark_space_domain_{architecture}_cdf`
  - `benchmark_space_domain_{architecture}_heatmap`
  - `benchmark_space_domain_{architecture}_heatmap_combined`
  - `benchmark_space_domain_{architecture}_histograms`
  - `benchmark_space_domain_{architecture}_scatter`
  - `benchmark_space_domain_{architecture}_violin`

- Global Architecture Comparison (`figures/ablation_global/`):
  - `ablation_time_domain_comparison`
  - `ablation_space_domain_cdf`

- ResNet MoE Ablation (`figures/ablation_resnet/`):
  - `resnet_ablation_time_domain`
  - `resnet_ablation_regional_improvement`

- Expert Mesh (`figures/`):
  - `orbita_expert_mesh`

- Metrics Figures (`figures/metrics/`):
  - `metrics_training_time`
  - `metrics_model_size`
  - `metrics_inference_time_domain`
  - `metrics_inference_space_domain`
  - `metrics_energy_kwh`
  - `metrics_co2_emissions`
  - `metrics_cross_validation`

## Audit & Verification Command

Run this command from the repository root:

```bash
python src/audit_results.py
```

The audit verifies global models, CV metrics, physical ResNet grid cells, benchmark CSVs, row/case counts, and required PNG/SVG figures.
