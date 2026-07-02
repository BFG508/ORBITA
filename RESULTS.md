# ORBITA Results Manifest

This manifest records the expected local result artifacts for the current TFM technical closure.

## Global Model Comparison

- Dataset: `data/orbita_dataset_300-2000_0.0000-0.1000_0-90.csv`.
- Trained global models: ResNet, MLP, LSTM, Linear, and Decision Tree under `models/`.
- Fine-tuned global neural models: ResNet, MLP, LSTM, and Linear under `models/`.
- Cross-validation summary: `data/metrics_cv.csv`.
- Training metrics: `data/metrics_train.csv`.
- Benchmark metrics: `data/metrics_benchmark.csv`.
- CodeCarbon output: `data/emissions.csv`.

## ResNet MoE Grid

The ResNet Mixture-of-Experts fleet uses a nominal `5 x 4 x 2` grid:

- SMA/altitude bins: `300-640`, `640-980`, `980-1320`, `1320-1660`, `1660-2000` km.
- Eccentricity bins: `0.0000-0.0250`, `0.0250-0.0500`, `0.0500-0.0750`, `0.0750-0.1000`.
- Inclination bins: `0-45`, `45-90` deg.

Two nominal cells are physically invalid because even their best-case corner violates the minimum safe perigee:

- `300-640_0.0750-0.1000_0-45`
- `300-640_0.0750-0.1000_45-90`

The valid grid therefore contains 38 cells. Base and fine-tuned ResNet expert weights live under `models/resnet/`, and their active-learning datasets live under `data/resnet/`.

## Final Figures

Final figures are exported in both PNG and SVG under `figures/`.

ResNet performance figures:

- `benchmark_time_domain_resnet_envelope`
- `benchmark_space_domain_resnet_cdf`
- `benchmark_space_domain_resnet_heatmap`
- `benchmark_space_domain_resnet_histograms`
- `benchmark_space_domain_resnet_scatter`
- `benchmark_space_domain_resnet_violin`

Ablation and comparison figures:

- `ablation_time_domain_comparison`
- `ablation_space_domain_cdf`

Metrics figures:

- `metrics_training_time`
- `metrics_model_size`
- `metrics_inference_time_domain`
- `metrics_inference_space_domain`
- `metrics_energy_kwh`
- `metrics_co2_emissions`
- `metrics_cross_validation`

## Audit Command

Run this command from the repository root:

```bash
python src/audit_results.py
```

The audit verifies global models, CV metrics, the physically valid ResNet grid, ResNet benchmark CSVs, and all required final PNG/SVG figures.
