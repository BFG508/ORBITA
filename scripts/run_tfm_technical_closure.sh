#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATUS_FILE="/tmp/orbita_tfm_technical_status.txt"
LOG_DIR="/tmp/orbita_tfm_technical_logs"
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

run_step \
    "benchmark_resnet_space" \
    .venv/bin/python src/benchmark.py \
        --model_type resnet \
        --mode_choice 2 \
        --quiet \
        --seed 42

run_step \
    "benchmark_lstm_full" \
    .venv/bin/python src/benchmark.py \
        --model_type lstm \
        --mode_choice 3 \
        --quiet \
        --seed 42

run_step \
    "figures_resnet" \
    .venv/bin/python src/visualize_benchmark.py \
        --mode single \
        --model_type resnet

run_step \
    "figures_lstm" \
    .venv/bin/python src/visualize_benchmark.py \
        --mode single \
        --model_type lstm

run_step \
    "figures_ablation" \
    .venv/bin/python src/visualize_benchmark.py \
        --mode ablation

run_step \
    "figures_metrics" \
    .venv/bin/python src/visualize_benchmark.py \
        --mode metrics

run_step \
    "flake8" \
    .venv/bin/python -m flake8 src tests

run_step \
    "pytest" \
    .venv/bin/python -m pytest tests/ -v

run_step \
    "audit_results" \
    .venv/bin/python src/audit_results.py

find . -path './.venv' -prune -o -type d \
    \( -name '__pycache__' -o -name '.pytest_cache' \) \
    -prune -exec rm -rf {} +

write_status "DONE technical closure"
