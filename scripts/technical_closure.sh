#!/usr/bin/env bash
# =============================================================================
# technical_closure.sh
#
# Regenerates every artifact referenced in the TFM (Annex B):
#   - Global dataset and model training for all 5 architectures
#   - K-Fold Cross-Validation for all 5 architectures
#   - Time-domain and space-domain benchmarks for all 5 architectures
#   - All visualization modes (single ×5, ablation, metrics)
#   - PEP-8 linting (flake8) and automated tests (pytest)
#   - Final artifact audit
#
# Usage:
#   ./scripts/technical_closure.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON=".venv/bin/python"
STATUS_FILE="/tmp/orbita_technical_closure_status.txt"
LOG_DIR="/tmp/orbita_technical_closure_logs"
mkdir -p "$LOG_DIR"

write_status() {
    printf '%s | %s\n' "$(date --iso-8601=seconds)" "$*" | tee "$STATUS_FILE"
}

run_step() {
    local name="$1"
    shift
    local log_file="$LOG_DIR/${name}.log"

    write_status "START ${name}"
    "$@" >"$log_file" 2>&1
    write_status "DONE ${name}"
}

write_status "START technical closure"

# =====================================================================
# 1. DATASET GENERATION
# =====================================================================
run_step \
    "generate_global_dataset" \
    $PYTHON src/generate_base_dataset.py

# =====================================================================
# 2. GLOBAL MODEL TRAINING (5 architectures)
# =====================================================================
for arch in resnet mlp lstm linear tree; do
    run_step \
        "train_${arch}" \
        $PYTHON src/train_base.py \
            --model_type "$arch"
done

# =====================================================================
# 3. K-FOLD CROSS-VALIDATION (5 architectures)
# =====================================================================
for arch in resnet mlp lstm linear tree; do
    run_step \
        "cv_${arch}" \
        $PYTHON src/train_cv.py \
            --model_type "$arch" \
            --folds 5
done

# =====================================================================
# 4. BENCHMARKS (5 architectures)
#    ResNet: space-only (time-domain comes from MoE pipeline)
#    Others: full (time-domain + space-domain)
# =====================================================================
run_step \
    "benchmark_resnet_space" \
    $PYTHON src/benchmark.py \
        --model_type resnet \
        --mode_choice 2 \
        --quiet \
        --seed 42

for arch in mlp lstm linear tree; do
    run_step \
        "benchmark_${arch}_full" \
        $PYTHON src/benchmark.py \
            --model_type "$arch" \
            --mode_choice 3 \
            --quiet \
            --seed 42
done

# =====================================================================
# 5. SIMULATE MISSION (Active Learning / GPS Reset)
# =====================================================================
run_step \
    "simulate_mission" \
    $PYTHON src/simulate_mission.py

# =====================================================================
# 6. FIGURES (all modes)
# =====================================================================
# Single-model reports for all 5 architectures
for arch in resnet mlp lstm linear tree; do
    run_step \
        "figures_${arch}" \
        $PYTHON src/visualize_benchmark.py \
            --mode single \
            --model_type "$arch"
done

# Ablation comparison figures
run_step \
    "figures_ablation" \
    $PYTHON src/visualize_benchmark.py \
        --mode ablation

# Metrics comparison figures
run_step \
    "figures_metrics" \
    $PYTHON src/visualize_benchmark.py \
        --mode metrics

# =====================================================================
# 7. QUALITY GATES
# =====================================================================
run_step \
    "flake8" \
    $PYTHON -m flake8 src tests

run_step \
    "pytest" \
    $PYTHON -m pytest tests/ -v

run_step \
    "audit_results" \
    $PYTHON src/audit_results.py

# =====================================================================
# 8. CLEANUP
# =====================================================================
find . -path './.venv' -prune -o -type d \
    \( -name '__pycache__' -o -name '.pytest_cache' \) \
    -prune -exec rm -rf {} +

write_status "DONE technical closure"
